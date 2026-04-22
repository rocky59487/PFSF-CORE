#version 450
#extension GL_GOOGLE_include_directive : enable
#include "stencil_constants.glsl"

// ═══════════════════════════════════════════════════════════════
//  PFSF PCG Direction Update (v3: SSOR Polynomial Preconditioned)
//
//  v3: p = z_ssor + beta * p
//    z_ssor = D⁻¹r + D⁻¹ · Σ_{j∈face} σ_ij · r[j] / diag6[j]
//
//  beta = rTz_new / rTz_old
//    reductionBuf[0] = rTz_old
//    reductionBuf[2] = rTz_new
//
//  r 為 readonly，本 dispatch 中所有鄰居讀取一致（新 r）。
//  Workgroup: 256 threads (1D flat)
// ═══════════════════════════════════════════════════════════════

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint Lx, Ly, Lz;
} pc;

layout(set = 0, binding = 0) readonly  buffer Residual  { float r[];            };
layout(set = 0, binding = 1)           buffer Direction { float p[];            };
layout(set = 0, binding = 2) readonly  buffer Type      { uint  vtype[];        };
layout(set = 0, binding = 3) readonly  buffer Reduction { float reductionBuf[]; };
layout(set = 0, binding = 4) readonly  buffer Cond      { float sigma[];        };

// ─── 26 連通 Laplacian 對角線（與 pcg_update.computeDiag 完全一致）───
// v3: per-edge/corner directional conductivity（各向異性正確版）
float computeDiag(uint i, uint N, int igx, int igy, int igz, bool valid[6]) {
    float diag = 0.0;

    float sx_neg = sigma[0 * N + i]; float sx_pos = sigma[1 * N + i];
    float sy_neg = sigma[2 * N + i]; float sy_pos = sigma[3 * N + i];
    float sz_neg = sigma[4 * N + i]; float sz_pos = sigma[5 * N + i];

    if (valid[0] && sx_neg > 0.0) diag += sx_neg;
    if (valid[1] && sx_pos > 0.0) diag += sx_pos;
    if (valid[2] && sy_neg > 0.0) diag += sy_neg;
    if (valid[3] && sy_pos > 0.0) diag += sy_pos;
    if (valid[4] && sz_neg > 0.0) diag += sz_neg;
    if (valid[5] && sz_pos > 0.0) diag += sz_pos;


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
// r 為 readonly，所有鄰居讀取當前 dispatch 前的 r（更新 shader 寫入後）。
float computeSSORCorrection(uint gid, uint N, bool valid[6], uint Lx, uint Ly) {
    uint LxLy = Lx * Ly;
    float corr = 0.0;

    if (valid[0]) {
        uint j = gid - 1u;
        float rj = r[j];
        float d6j = sigma[0u*N+j] + sigma[1u*N+j] + sigma[2u*N+j]
                  + sigma[3u*N+j] + sigma[4u*N+j] + sigma[5u*N+j];
        corr += sigma[0u*N+gid] * ((d6j > 1e-20) ? rj / d6j : 0.0);
    }
    if (valid[1]) {
        uint j = gid + 1u;
        float rj = r[j];
        float d6j = sigma[0u*N+j] + sigma[1u*N+j] + sigma[2u*N+j]
                  + sigma[3u*N+j] + sigma[4u*N+j] + sigma[5u*N+j];
        corr += sigma[1u*N+gid] * ((d6j > 1e-20) ? rj / d6j : 0.0);
    }
    if (valid[2]) {
        uint j = gid - Lx;
        float rj = r[j];
        float d6j = sigma[0u*N+j] + sigma[1u*N+j] + sigma[2u*N+j]
                  + sigma[3u*N+j] + sigma[4u*N+j] + sigma[5u*N+j];
        corr += sigma[2u*N+gid] * ((d6j > 1e-20) ? rj / d6j : 0.0);
    }
    if (valid[3]) {
        uint j = gid + Lx;
        float rj = r[j];
        float d6j = sigma[0u*N+j] + sigma[1u*N+j] + sigma[2u*N+j]
                  + sigma[3u*N+j] + sigma[4u*N+j] + sigma[5u*N+j];
        corr += sigma[3u*N+gid] * ((d6j > 1e-20) ? rj / d6j : 0.0);
    }
    if (valid[4]) {
        uint j = gid - LxLy;
        float rj = r[j];
        float d6j = sigma[0u*N+j] + sigma[1u*N+j] + sigma[2u*N+j]
                  + sigma[3u*N+j] + sigma[4u*N+j] + sigma[5u*N+j];
        corr += sigma[4u*N+gid] * ((d6j > 1e-20) ? rj / d6j : 0.0);
    }
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
    uint gid = gl_GlobalInvocationID.x;
    uint N = pc.Lx * pc.Ly * pc.Lz;
    if (gid >= N) return;

    if (vtype[gid] != 1u) {
        p[gid] = 0.0;
        return;
    }

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

    float rTz_old = reductionBuf[0];
    float rTz_new = reductionBuf[2];
    float beta = 0.0;
    if (rTz_old > 1e-30) {
        beta = rTz_new / rTz_old;
    }
    beta = clamp(beta, 0.0, 10.0);

    float ri = r[gid];

    // SSOR z: Jacobi + 面鄰居多項式修正
    float diag = computeDiag(gid, N, igx, igy, igz, valid);
    float corr = computeSSORCorrection(gid, N, valid, pc.Lx, pc.Ly);
    float zi_j = (diag > 1e-20) ? ri / diag : ri;
    float zi   = zi_j + ((diag > 1e-20) ? corr / diag : 0.0);
    if (isnan(zi) || isinf(zi)) zi = zi_j;

    float newP = zi + beta * p[gid];
    if (isnan(newP) || isinf(newP)) newP = zi;

    p[gid] = newP;
}
