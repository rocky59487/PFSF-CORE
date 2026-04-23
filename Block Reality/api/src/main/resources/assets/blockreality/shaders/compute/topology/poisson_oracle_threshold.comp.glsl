#version 450

// ═══════════════════════════════════════════════════════════════
//  Tier-2 Poisson Oracle — threshold pass.
//
//  After the Jacobi outer loop converges, this kernel emits the
//  boolean fracture mask consumed by Tier 3:
//      mask[i] = 1  iff  type[i] == SOLID  AND  phi[i] < epsilon
//  The mask is packed 1 bit per voxel into a uint32[] buffer
//  (ceil(N/32) uints). Reading and writing individual bits via
//  atomicOr means the host reads only ~N/8 bytes per Tier-2 tick.
// ═══════════════════════════════════════════════════════════════

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint  Lx;
    uint  Ly;
    uint  Lz;
    float epsilon;
} pc;

layout(set = 0, binding = 0) readonly buffer VType      { uint  vtype[]; };
layout(set = 0, binding = 1) readonly buffer Phi        { float phi[];   };
layout(set = 0, binding = 2)          buffer MaskBits   { uint  mask[];  };

const uint VOXEL_SOLID = 1u;

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint N = pc.Lx * pc.Ly * pc.Lz;
    if (i >= N) return;

    if (vtype[i] != VOXEL_SOLID) return;
    if (phi[i] >= pc.epsilon) return;
    // Orphan: set bit i in the packed mask.
    uint word = i >> 5;
    uint bit  = 1u << (i & 31u);
    atomicOr(mask[word], bit);
}
