#version 450
#extension GL_GOOGLE_include_directive : enable
#include "stencil_constants.glsl"

// ═══════════════════════════════════════════════════════════════
//  PFSF Multigrid Restriction — 殘差降採樣（細 → 粗）
//
//  v2: 導率加權（conductivity-weighted）restriction
//  - 殘差和 phi 的粗網格值按面導率權重平均
//  - 高導率（高剛度）體素對粗網格貢獻更大 → 保留物理特性
//  - 避免低剛度/空氣體素稀釋高剛度區域的殘差信號
//
//  參考：PFSF 手冊 §4.4, McAdams et al. 2010
// ═══════════════════════════════════════════════════════════════

layout(local_size_x = 256) in;

layout(push_constant) uniform PushConstants {
    uint Lx_fine,  Ly_fine,  Lz_fine;
    uint Lx_coarse, Ly_coarse, Lz_coarse;
} pc;

layout(set = 0, binding = 0) readonly buffer PhiFine      { float phi_fine[];   };
layout(set = 0, binding = 1) readonly buffer SourceFine    { float rho_fine[];   };
layout(set = 0, binding = 2) readonly buffer CondFine      { float sigma_fine[]; };
layout(set = 0, binding = 3) readonly buffer TypeFine      { uint  vtype_fine[]; };
layout(set = 0, binding = 4) buffer PhiCoarse              { float phi_coarse[]; };
layout(set = 0, binding = 5) buffer SourceCoarse           { float rho_coarse[]; };

uint idxFine(uint x, uint y, uint z) {
    return x + pc.Lx_fine * (y + pc.Ly_fine * z);
}

uint idxCoarse(uint x, uint y, uint z) {
    return x + pc.Lx_coarse * (y + pc.Ly_coarse * z);
}

// 計算細網格上體素 i 的面導率總和（Laplacian 對角線的 6 面分量）
float faceCondSum(uint fx, uint fy, uint fz) {
    uint i = idxFine(fx, fy, fz);
    uint nFine = pc.Lx_fine * pc.Ly_fine * pc.Lz_fine;
    float w = 0.0;
    for (int d = 0; d < 6; d++) {
        float s = sigma_fine[d * nFine + i];
        if (s > 0.0) w += s;
    }
    return w;
}

// 計算細網格上體素 i 的殘差 r_i = rho_i + (L₂₆·phi)_i
// v3: 26 連通殘差，與 rbgs_smooth/pcg_matvec 完全一致的算子。
// 舊版（v2）只用 6 面連通，導致 V-cycle 修正錯誤方程，降低多網格效果。
float residual(uint fx, uint fy, uint fz) {
    uint i = idxFine(fx, fy, fz);

    // Air or anchor: residual = 0
    if (vtype_fine[i] != 1u) return 0.0;

    uint nFine = pc.Lx_fine * pc.Ly_fine * pc.Lz_fine;
    int ifx = int(fx), ify = int(fy), ifz = int(fz);
    int Lxf = int(pc.Lx_fine), Lyf = int(pc.Ly_fine), Lzf = int(pc.Lz_fine);

    bool valid[6] = bool[6](
        fx > 0u, fx + 1u < pc.Lx_fine,
        fy > 0u, fy + 1u < pc.Ly_fine,
        fz > 0u, fz + 1u < pc.Lz_fine
    );

    float phi_i = phi_fine[i];
    float Lphi  = 0.0;

    // ─── 6 face neighbors ───
    float sx_neg = sigma_fine[0u * nFine + i]; float sx_pos = sigma_fine[1u * nFine + i];
    float sy_neg = sigma_fine[2u * nFine + i]; float sy_pos = sigma_fine[3u * nFine + i];
    float sz_neg = sigma_fine[4u * nFine + i]; float sz_pos = sigma_fine[5u * nFine + i];

    if (valid[0] && sx_neg > 0.0) Lphi += sx_neg * (phi_fine[idxFine(fx-1u, fy,   fz  )] - phi_i);
    if (valid[1] && sx_pos > 0.0) Lphi += sx_pos * (phi_fine[idxFine(fx+1u, fy,   fz  )] - phi_i);
    if (valid[2] && sy_neg > 0.0) Lphi += sy_neg * (phi_fine[idxFine(fx,   fy-1u, fz  )] - phi_i);
    if (valid[3] && sy_pos > 0.0) Lphi += sy_pos * (phi_fine[idxFine(fx,   fy+1u, fz  )] - phi_i);
    if (valid[4] && sz_neg > 0.0) Lphi += sz_neg * (phi_fine[idxFine(fx,   fy,   fz-1u)] - phi_i);
    if (valid[5] && sz_pos > 0.0) Lphi += sz_pos * (phi_fine[idxFine(fx,   fy,   fz+1u)] - phi_i);

    // ─── 12 edge neighbors (anisotropic directional coupling) ───

    // XY plane (4 edges)
    if (valid[0] && valid[2]) { float es = sqrt(max(sx_neg*sy_neg,0.0))*EDGE_P; if(es>0.0) Lphi+=es*(phi_fine[idxFine(fx-1u,fy-1u,fz)]-phi_i); }
    if (valid[1] && valid[3]) { float es = sqrt(max(sx_pos*sy_pos,0.0))*EDGE_P; if(es>0.0) Lphi+=es*(phi_fine[idxFine(fx+1u,fy+1u,fz)]-phi_i); }
    if (valid[0] && valid[3]) { float es = sqrt(max(sx_neg*sy_pos,0.0))*EDGE_P; if(es>0.0) Lphi+=es*(phi_fine[idxFine(fx-1u,fy+1u,fz)]-phi_i); }
    if (valid[1] && valid[2]) { float es = sqrt(max(sx_pos*sy_neg,0.0))*EDGE_P; if(es>0.0) Lphi+=es*(phi_fine[idxFine(fx+1u,fy-1u,fz)]-phi_i); }
    // XZ plane (4 edges)
    if (valid[0] && valid[4]) { float es = sqrt(max(sx_neg*sz_neg,0.0))*EDGE_P; if(es>0.0) Lphi+=es*(phi_fine[idxFine(fx-1u,fy,fz-1u)]-phi_i); }
    if (valid[1] && valid[5]) { float es = sqrt(max(sx_pos*sz_pos,0.0))*EDGE_P; if(es>0.0) Lphi+=es*(phi_fine[idxFine(fx+1u,fy,fz+1u)]-phi_i); }
    if (valid[0] && valid[5]) { float es = sqrt(max(sx_neg*sz_pos,0.0))*EDGE_P; if(es>0.0) Lphi+=es*(phi_fine[idxFine(fx-1u,fy,fz+1u)]-phi_i); }
    if (valid[1] && valid[4]) { float es = sqrt(max(sx_pos*sz_neg,0.0))*EDGE_P; if(es>0.0) Lphi+=es*(phi_fine[idxFine(fx+1u,fy,fz-1u)]-phi_i); }
    // YZ plane (4 edges)
    if (valid[2] && valid[4]) { float es = sqrt(max(sy_neg*sz_neg,0.0))*EDGE_P; if(es>0.0) Lphi+=es*(phi_fine[idxFine(fx,fy-1u,fz-1u)]-phi_i); }
    if (valid[3] && valid[5]) { float es = sqrt(max(sy_pos*sz_pos,0.0))*EDGE_P; if(es>0.0) Lphi+=es*(phi_fine[idxFine(fx,fy+1u,fz+1u)]-phi_i); }
    if (valid[2] && valid[5]) { float es = sqrt(max(sy_neg*sz_pos,0.0))*EDGE_P; if(es>0.0) Lphi+=es*(phi_fine[idxFine(fx,fy-1u,fz+1u)]-phi_i); }
    if (valid[3] && valid[4]) { float es = sqrt(max(sy_pos*sz_neg,0.0))*EDGE_P; if(es>0.0) Lphi+=es*(phi_fine[idxFine(fx,fy+1u,fz-1u)]-phi_i); }

    // ─── 8 corner neighbors (per-corner directional cbrt) ───
    int dxc[2] = int[2](-1, 1); int dyc[2] = int[2](-1, 1); int dzc[2] = int[2](-1, 1);
    for (int ci = 0; ci < 8; ci++) {
        int cx = dxc[ci & 1], cy = dyc[(ci>>1)&1], cz = dzc[(ci>>2)&1];
        int nx = ifx+cx, ny = ify+cy, nz = ifz+cz;
        if (nx<0||nx>=Lxf||ny<0||ny>=Lyf||nz<0||nz>=Lzf) continue;
        float sxc = (cx<0) ? sx_neg : sx_pos;
        float syc = (cy<0) ? sy_neg : sy_pos;
        float szc = (cz<0) ? sz_neg : sz_pos;
        float cs = pow(max(sxc*syc*szc, 0.0), 1.0/3.0) * CORNER_P;
        if (cs > 0.0) Lphi += cs * (phi_fine[idxFine(uint(nx),uint(ny),uint(nz))] - phi_i);
    }

    return rho_fine[i] + Lphi;
}

void main() {
    uint cIdx = gl_GlobalInvocationID.x;
    uint N_coarse = pc.Lx_coarse * pc.Ly_coarse * pc.Lz_coarse;
    if (cIdx >= N_coarse) return;

    uint cx = cIdx % pc.Lx_coarse;
    uint rem = cIdx / pc.Lx_coarse;
    uint cy = rem % pc.Ly_coarse;
    uint cz = rem / pc.Ly_coarse;

    uint fx0 = cx * 2u;
    uint fy0 = cy * 2u;
    uint fz0 = cz * 2u;

    // v2: 導率加權殘差 restriction
    // 殘差按面導率權重平均 → 高剛度區域信號不被低剛度稀釋
    float sumWeightedResidual = 0.0;
    float totalWeight = 0.0;
    float sumUnweightedResidual = 0.0;
    int count = 0;

    for (uint dz = 0u; dz < 2u; dz++) {
        for (uint dy = 0u; dy < 2u; dy++) {
            for (uint dx = 0u; dx < 2u; dx++) {
                uint fx = fx0 + dx;
                uint fy = fy0 + dy;
                uint fz = fz0 + dz;
                if (fx < pc.Lx_fine && fy < pc.Ly_fine && fz < pc.Lz_fine) {
                    float r = residual(fx, fy, fz);
                    float w = faceCondSum(fx, fy, fz);
                    sumWeightedResidual += w * r;
                    totalWeight += w;
                    sumUnweightedResidual += r;
                    count++;
                }
            }
        }
    }

    // 加權殘差：若有導率 → 使用加權；否則 fallback 到簡單求和
    if (totalWeight > 1e-20) {
        // 加權平均 × count = 保留殘差積分量級
        rho_coarse[cIdx] = (sumWeightedResidual / totalWeight) * float(count);
    } else {
        rho_coarse[cIdx] = (count > 0) ? sumUnweightedResidual : 0.0;
    }

    // v2: 導率加權 phi 初始化
    float sumWeightedPhi = 0.0;
    float totalPhiWeight = 0.0;
    float sumPhi = 0.0;
    count = 0;

    for (uint dz = 0u; dz < 2u; dz++) {
        for (uint dy = 0u; dy < 2u; dy++) {
            for (uint dx = 0u; dx < 2u; dx++) {
                uint fx = fx0 + dx;
                uint fy = fy0 + dy;
                uint fz = fz0 + dz;
                if (fx < pc.Lx_fine && fy < pc.Ly_fine && fz < pc.Lz_fine) {
                    uint i = idxFine(fx, fy, fz);
                    float p = phi_fine[i];
                    float w = faceCondSum(fx, fy, fz);
                    sumWeightedPhi += w * p;
                    totalPhiWeight += w;
                    sumPhi += p;
                    count++;
                }
            }
        }
    }

    if (totalPhiWeight > 1e-20) {
        phi_coarse[cIdx] = sumWeightedPhi / totalPhiWeight;
    } else {
        phi_coarse[cIdx] = (count > 0) ? sumPhi / float(count) : 0.0;
    }
}
