#version 450
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_GOOGLE_include_directive : enable
#include "stencil_constants.glsl"

// ═══════════════════════════════════════════════════════════════
//  PFSF PCG Update Kernel (v3: SSOR Polynomial Preconditioned)
//
//  v3 升級：Jacobi 改為 2-step Neumann SSOR 多項式近似
//    z = D⁻¹r + D⁻¹ · Σ_{j∈face} σ_ij · r[j] / diag6[j]
//  其中 diag6[j] = Σ_{d=0}^5 σ[d·N+j]（面鄰居的 6-face 對角線近似）。
//  此公式等同於一步 Gauss-Seidel / Neumann 多項式，對不均勻材料收斂
//  顯著優於純 Jacobi，且完全平行無資料相依。
//
//  兩種模式：
//    isInit == 1: 初始化模式
//      r[i] = source[i] - Ap[i]
//      z[i] = Jacobi（init 時鄰居 r 可能為 0，跳過修正）
//      p[i] = z[i]
//      計算 r·z 局部和 → partialSums[]
//
//    isInit == 0: 正常迭代模式
//      alpha = rTz_old / pAp
//      phi[i] += alpha * p[i]
//      r[i]   -= alpha * Ap[i]
//      z[i]   = SSOR(r)      ← v3: 含面鄰居修正
//      計算新的 r·z 局部和 → partialSums[]
//
//  Workgroup: 256 threads (1D flat)
// ═══════════════════════════════════════════════════════════════

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint  Lx, Ly, Lz;
    float alpha;
    uint  isInit;
    uint  padding;
} pc;

layout(set = 0, binding = 0) buffer Phi         { float phi[];         };
layout(set = 0, binding = 1) buffer Residual    { float r[];           };
layout(set = 0, binding = 2) buffer Direction   { float p[];           };
layout(set = 0, binding = 3) readonly buffer Ap { float ap[];          };
layout(set = 0, binding = 4) readonly buffer Source { float source[];  };
layout(set = 0, binding = 5) readonly buffer Type   { uint  vtype[];   };
layout(set = 0, binding = 6) buffer PartialSums { float partialSums[]; };
layout(set = 0, binding = 7) readonly buffer Reduction { float reductionBuf[]; };
layout(set = 0, binding = 8) readonly buffer Cond { float sigma[]; };

shared float sdata[256 + 32];

// ─── 26 連通 Laplacian 對角線 D_ii = Σ_j sigma_ij（與 rbgs_smooth sumSigma 一致）───
// v3: per-edge/corner directional conductivity（各向異性正確版）
float computeDiag(uint i, uint N, int igx, int igy, int igz, bool valid[6]) {
    float diag = 0.0;

    float sx_neg = sigma[0 * N + i]; float sx_pos = sigma[1 * N + i];
    float sy_neg = sigma[2 * N + i]; float sy_pos = sigma[3 * N + i];
    float sz_neg = sigma[4 * N + i]; float sz_pos = sigma[5 * N + i];

    // 6 face contributions
    if (valid[0] && sx_neg > 0.0) diag += sx_neg;
    if (valid[1] && sx_pos > 0.0) diag += sx_pos;
    if (valid[2] && sy_neg > 0.0) diag += sy_neg;
    if (valid[3] && sy_pos > 0.0) diag += sy_pos;
    if (valid[4] && sz_neg > 0.0) diag += sz_neg;
    if (valid[5] && sz_pos > 0.0) diag += sz_pos;


    // 12 edge contributions (per-edge directional)
    if (valid[0]&&valid[2]) diag += sqrt(max(sx_neg*sy_neg,0.0))*EDGE_P;
    if (valid[1]&&valid[3]) diag += sqrt(max(sx_pos*sy_pos,0.0))*EDGE_P;
    if (valid[0]&&valid[3]) diag += sqrt(max(sx_neg*sy_pos,0.0))*EDGE_P;
    if (valid[1]&&valid[2]) diag += sqrt(max(sx_pos*sy_neg,0.0))*EDGE_P;
    if (valid[0]&&valid[4]) diag += sqrt(max(sx_neg*sz_neg,0.0))*EDGE_P;
    if (valid[1]&&valid[5]) diag += sqrt(max(sx_pos*sz_pos,0.0))*EDGE_P;
    if (valid[0]&&valid[5]) diag += sqrt(max(sx_neg*sz_pos,0.0))*EDGE_P;
    if (valid[1]&&valid[4]) diag += sqrt(max(sx_pos*sz_neg,0.0))*EDGE_P;
    if (valid[2]&&valid[4]) diag += sqrt(max(sy_neg*sz_neg,0.0))*EDGE_P;
    if (valid[3]&&valid[5]) diag += sqrt(max(sy_pos*sz_pos,0.0))*EDGE_P;
    if (valid[2]&&valid[5]) diag += sqrt(max(sy_neg*sz_pos,0.0))*EDGE_P;
    if (valid[3]&&valid[4]) diag += sqrt(max(sy_pos*sz_neg,0.0))*EDGE_P;

    // 8 corner contributions (per-corner directional cbrt)
    int dxc[2] = int[2](-1, 1); int dyc[2] = int[2](-1, 1); int dzc[2] = int[2](-1, 1);
    for (int ci = 0; ci < 8; ci++) {
        int cx = dxc[ci & 1], cy = dyc[(ci>>1)&1], cz = dzc[(ci>>2)&1];
        int nx = igx+cx, ny = igy+cy, nz = igz+cz;
        if (nx<0||nx>=int(pc.Lx)||ny<0||ny>=int(pc.Ly)||nz<0||nz>=int(pc.Lz)) continue;
        float sxc=(cx<0)?sx_neg:sx_pos; float syc=(cy<0)?sy_neg:sy_pos; float szc=(cz<0)?sz_neg:sz_pos;
        diag += pow(max(sxc*syc*szc, 0.0), 1.0/3.0) * CORNER_P;
    }

    return diag;
}

// ─── SSOR 多項式修正項（2-step Neumann，僅面鄰居 6 連通）───
// 計算 Σ_{j∈face} σ_{ij} · r[j] / diag6[j]
// 呼叫時 r[j] 為本 dispatch 前的舊值（GPU 記憶體模型保證）。
float computeSSORCorrection(uint gid, uint N, bool valid[6], uint Lx, uint Ly) {
    uint LxLy = Lx * Ly;
    float corr = 0.0;

    // -x 面鄰居 (direction 0 of current voxel)
    if (valid[0]) {
        uint j = gid - 1u;
        float rj = r[j];
        float d6j = sigma[0u*N+j] + sigma[1u*N+j] + sigma[2u*N+j]
                  + sigma[3u*N+j] + sigma[4u*N+j] + sigma[5u*N+j];
        corr += sigma[0u*N+gid] * ((d6j > 1e-20) ? rj / d6j : 0.0);
    }
    // +x 面鄰居 (direction 1)
    if (valid[1]) {
        uint j = gid + 1u;
        float rj = r[j];
        float d6j = sigma[0u*N+j] + sigma[1u*N+j] + sigma[2u*N+j]
                  + sigma[3u*N+j] + sigma[4u*N+j] + sigma[5u*N+j];
        corr += sigma[1u*N+gid] * ((d6j > 1e-20) ? rj / d6j : 0.0);
    }
    // -y 面鄰居 (direction 2)
    if (valid[2]) {
        uint j = gid - Lx;
        float rj = r[j];
        float d6j = sigma[0u*N+j] + sigma[1u*N+j] + sigma[2u*N+j]
                  + sigma[3u*N+j] + sigma[4u*N+j] + sigma[5u*N+j];
        corr += sigma[2u*N+gid] * ((d6j > 1e-20) ? rj / d6j : 0.0);
    }
    // +y 面鄰居 (direction 3)
    if (valid[3]) {
        uint j = gid + Lx;
        float rj = r[j];
        float d6j = sigma[0u*N+j] + sigma[1u*N+j] + sigma[2u*N+j]
                  + sigma[3u*N+j] + sigma[4u*N+j] + sigma[5u*N+j];
        corr += sigma[3u*N+gid] * ((d6j > 1e-20) ? rj / d6j : 0.0);
    }
    // -z 面鄰居 (direction 4)
    if (valid[4]) {
        uint j = gid - LxLy;
        float rj = r[j];
        float d6j = sigma[0u*N+j] + sigma[1u*N+j] + sigma[2u*N+j]
                  + sigma[3u*N+j] + sigma[4u*N+j] + sigma[5u*N+j];
        corr += sigma[4u*N+gid] * ((d6j > 1e-20) ? rj / d6j : 0.0);
    }
    // +z 面鄰居 (direction 5)
    if (valid[5]) {
        uint j = gid + LxLy;
        float rj = r[j];
        float d6j = sigma[0u*N+j] + sigma[1u*N+j] + sigma[2u*N+j]
                  + sigma[3u*N+j] + sigma[4u*N+j] + sigma[5u*N+j];
        corr += sigma[5u*N+gid] * ((d6j > 1e-20) ? rj / d6j : 0.0);
    }

    return corr;
}

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;
    uint N = pc.Lx * pc.Ly * pc.Lz;

    float localRTz = 0.0;

    if (gid < N) {
        uint gx = gid % pc.Lx;
        uint rem = gid / pc.Lx;
        uint gy = rem % pc.Ly;
        uint gz = rem / pc.Ly;
        int igx = int(gx), igy = int(gy), igz = int(gz);

        bool valid[6] = bool[6](
            gx > 0u, gx + 1u < pc.Lx,
            gy > 0u, gy + 1u < pc.Ly,
            gz > 0u, gz + 1u < pc.Lz
        );

        if (pc.isInit == 1u) {
            if (vtype[gid] == 1u) {
                float ri = source[gid] - ap[gid];
                if (isnan(ri) || isinf(ri)) ri = 0.0;
                r[gid] = ri;

                // Init: plain Jacobi (neighbor r values are stale/zero)
                float diag = computeDiag(gid, N, igx, igy, igz, valid);
                float zi = (diag > 1e-20) ? ri / diag : ri;
                if (isnan(zi) || isinf(zi)) zi = 0.0;

                p[gid] = zi;
                localRTz = ri * zi;
            } else {
                r[gid] = 0.0;
                p[gid] = 0.0;
            }
        } else {
            if (vtype[gid] == 1u) {
                float alpha_val = pc.alpha;
                {
                    float rTz_old = reductionBuf[0];
                    float pAp = reductionBuf[1];
                    alpha_val = (pAp > 1e-30) ? rTz_old / pAp : 0.0;
                }

                // SSOR 修正項：在更新 r[gid] 之前讀取舊值（GPU 記憶體模型確保）
                float corr = computeSSORCorrection(gid, N, valid, pc.Lx, pc.Ly);

                phi[gid] += alpha_val * p[gid];
                r[gid]   -= alpha_val * ap[gid];

                phi[gid] = clamp(phi[gid], 0.0, 1e7);
                if (isnan(phi[gid])) phi[gid] = 0.0;

                float ri = r[gid];
                if (isnan(ri) || isinf(ri)) {
                    ri = 0.0;
                    r[gid] = 0.0;
                }

                // SSOR z: Jacobi（新 r） + 面鄰居修正（舊 r）
                float diag = computeDiag(gid, N, igx, igy, igz, valid);
                float zi_j = (diag > 1e-20) ? ri / diag : ri;
                float zi   = zi_j + ((diag > 1e-20) ? corr / diag : 0.0);
                if (isnan(zi) || isinf(zi)) zi = zi_j;

                localRTz = ri * zi;
            }
        }
    }

    // ─── Workgroup reduction for r·z partial sum ───
    localRTz = subgroupAdd(localRTz);

    uint sgSize = gl_SubgroupSize;
    uint sgId = gl_SubgroupInvocationID;
    if (sgId == 0u) {
        sdata[tid / sgSize] = localRTz;
    }
    barrier();

    uint numSubgroups = (gl_WorkGroupSize.x + sgSize - 1u) / sgSize;
    if (tid < numSubgroups) {
        localRTz = sdata[tid];
    } else {
        localRTz = 0.0;
    }
    if (tid < numSubgroups) {
        localRTz = subgroupAdd(localRTz);
    }

    if (tid == 0u) {
        partialSums[gl_WorkGroupID.x] = localRTz;
    }
}
