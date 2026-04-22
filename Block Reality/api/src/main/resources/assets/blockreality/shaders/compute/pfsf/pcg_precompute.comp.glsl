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

    // 計算 6-face 對角線和 (diag6)
    float d6 = sigma[0u * N + gid] + sigma[1u * N + gid] + sigma[2u * N + gid]
             + sigma[3u * N + gid] + sigma[4u * N + gid] + sigma[5u * N + gid];

    // 預先計算倒數，用於 SSOR 修正項
    invDiag6[gid] = (d6 > 1e-20) ? (1.0 / d6) : 0.0;
}
