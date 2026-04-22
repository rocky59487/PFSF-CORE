#version 450

#extension GL_GOOGLE_include_directive : enable
#include "stencil_constants.glsl"

// ═══════════════════════════════════════════════════════════════
//  AMG Coarse Jacobi: e_c = D_c⁻¹ · r_c
//
//  粗網格 Jacobi 求解：每個 coarse node j 的修正量 e_c[j] = r_c[j] / D_c[j]。
//
//  r_c 由 amg_scatter_restrict 透過 atomicCompSwap float-add 寫入，
//  因此儲存為 uint32（IEEE 754 bits）。
//  D_c 由 CPU side (amg_builder) 預計算並上傳（Galerkin 對角線近似）。
//
//  Workgroup: 256 threads (1D flat over N_coarse)
//
//  Bindings:
//   0: DiagC      float[N_coarse] (readonly)  Galerkin 對角線 D_c
//   1: CoarseR    uint[N_coarse]  (readonly)  float bits（來自 CAS atomicAdd）
//   2: CoarsePhi  float[N_coarse] (write)     粗網格修正量 e_c
// ═══════════════════════════════════════════════════════════════

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint N_coarse;
} pc;

layout(set = 0, binding = 0) readonly buffer DiagC     { float diagC[];   };
layout(set = 0, binding = 1) readonly buffer CoarseR   { uint  rcBits[];  };
layout(set = 0, binding = 2)          buffer CoarsePhi { float ec[];      };

void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= pc.N_coarse) return;

    float rc = uintBitsToFloat(rcBits[i]);
    float dc = diagC[i];
    ec[i] = (dc > 1e-20) ? rc / dc : 0.0;
}
