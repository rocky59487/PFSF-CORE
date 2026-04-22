#version 450
#extension GL_GOOGLE_include_directive : enable
#include "stencil_constants.glsl"

// ═══════════════════════════════════════════════════════════════
//  PFSF PCG Precompute Kernel (Memory Optimization)
//
//  v3.1: 預計算面鄰居對角線之倒數 (Inverse Diagonal)
//  目的：消除 pcg_update.comp.glsl 在 SSOR 修正項中重複讀取 6 次 sigma 
//       的全域記憶體頻寬瓶頸，並將除法轉為乘法。
//
//  Workgroup: 256 threads (1D flat)
// ═══════════════════════════════════════════════════════════════

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(push_constant) uniform PushConstants {
    uint Lx, Ly, Lz;
} pc;

layout(set = 0, binding = 0) readonly  buffer Cond      { float sigma[];     };
layout(set = 0, binding = 1) writeonly buffer InvDiag6  { float invDiag6[];  };

void main() {
    uint gid = gl_GlobalInvocationID.x;
    uint N = pc.Lx * pc.Ly * pc.Lz;
    if (gid >= N) return;

    // 還原 3D 座標用於 26-鄰居邊界判定
    uint gx = gid % pc.Lx;
    uint rem = gid / pc.Lx;
    uint gy = rem % pc.Ly;
    uint gz = rem / pc.Ly;

    // 修正預處理子不匹配 (Point 5)：原本僅計算 6-face 對角線，
    // 但 A 矩陣 (pcg_matvec) 是 26-連通的。
    // SSOR 預處理子 M = diag(A) 必須包含所有 26 個鄰居的貢獻。
    
    // 1. 6-face neighbors
    float d26 = sigma[0u * N + gid] + sigma[1u * N + gid] + sigma[2u * N + gid]
              + sigma[3u * N + gid] + sigma[4u * N + gid] + sigma[5u * N + gid];

    // 2. 26-connectivity 剪力貢獻 (Edge + Corner)
    // 必須與 rbgs_smooth.comp.glsl / pcg_matvec.comp.glsl 中的 Stencil 權重完全一致
    {
        bool valid[6] = bool[6](
            gx > 0u, gx + 1u < pc.Lx,
            gy > 0u, gy + 1u < pc.Ly,
            gz > 0u, gz + 1u < pc.Lz
        );
        float sx_neg = sigma[0 * N + gid]; float sx_pos = sigma[1 * N + gid];
        float sy_neg = sigma[2 * N + gid]; float sy_pos = sigma[3 * N + gid];
        float sz_neg = sigma[4 * N + gid]; float sz_pos = sigma[5 * N + gid];

        // 12 edges
        if (valid[0]&&valid[2]) d26 += sqrt(max(sx_neg*sy_neg, 0.0)) * EDGE_P;
        if (valid[1]&&valid[3]) d26 += sqrt(max(sx_pos*sy_pos, 0.0)) * EDGE_P;
        if (valid[0]&&valid[3]) d26 += sqrt(max(sx_neg*sy_pos, 0.0)) * EDGE_P;
        if (valid[1]&&valid[2]) d26 += sqrt(max(sx_pos*sy_neg, 0.0)) * EDGE_P;
        if (valid[0]&&valid[4]) d26 += sqrt(max(sx_neg*sz_neg, 0.0)) * EDGE_P;
        if (valid[1]&&valid[5]) d26 += sqrt(max(sx_pos*sz_pos, 0.0)) * EDGE_P;
        if (valid[0]&&valid[5]) d26 += sqrt(max(sx_neg*sz_pos, 0.0)) * EDGE_P;
        if (valid[1]&&valid[4]) d26 += sqrt(max(sx_pos*sz_neg, 0.0)) * EDGE_P;
        if (valid[2]&&valid[4]) d26 += sqrt(max(sy_neg*sz_neg, 0.0)) * EDGE_P;
        if (valid[3]&&valid[5]) d26 += sqrt(max(sy_pos*sz_pos, 0.0)) * EDGE_P;
        if (valid[2]&&valid[5]) d26 += sqrt(max(sy_neg*sz_pos, 0.0)) * EDGE_P;
        if (valid[3]&&valid[4]) d26 += sqrt(max(sy_pos*sz_neg, 0.0)) * EDGE_P;

        // 8 corners
        for (int ci = 0; ci < 8; ci++) {
            int cx = ((ci & 1) != 0) ? 1 : -1;
            int cy = (((ci >> 1) & 1) != 0) ? 1 : -1;
            int cz = (((ci >> 2) & 1) != 0) ? 1 : -1;
            if ((cx < 0 && !valid[0]) || (cx > 0 && !valid[1])) continue;
            if ((cy < 0 && !valid[2]) || (cy > 0 && !valid[3])) continue;
            if ((cz < 0 && !valid[4]) || (cz > 0 && !valid[5])) continue;
            float sxc = (cx < 0) ? sx_neg : sx_pos;
            float syc = (cy < 0) ? sy_neg : sy_pos;
            float szc = (cz < 0) ? sz_neg : sz_pos;
            d26 += pow(max(sxc * syc * szc, 0.0), 1.0/3.0) * CORNER_P;
        }
    }

    // 預先計算倒數，用於 SSOR 修正項
    invDiag6[gid] = (d26 > 1e-20) ? (1.0 / d26) : 0.0;
}
