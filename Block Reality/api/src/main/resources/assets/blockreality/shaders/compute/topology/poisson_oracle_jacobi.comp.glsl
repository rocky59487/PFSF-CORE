#version 450
#extension GL_GOOGLE_include_directive : enable

// ═══════════════════════════════════════════════════════════════
//  Tier-2 Poisson Oracle — one Jacobi sweep on an anchor-diffusion
//  problem over a 26-connected voxel sub-region.
//
//  The solver mirrors PoissonOracleCPU bit-for-bit. For every voxel:
//      ANCHOR → φ = 1           (Dirichlet boundary)
//      AIR    → φ = 0           (source-free, walled off)
//      SOLID  → φ = mean of connected (non-AIR) 26 neighbours
//
//  Read phiIn, write phiOut. Host records this kernel in a fixed
//  unrolled loop (typically 64–256 sweeps) — the same unroll pattern
//  used by label_prop_iterate so there is no per-iteration PCIe
//  readback. Convergence bound: diameter of the sub-region; for a
//  16³ Tier-2 AABB that is ~28 iterations worst case.
//
//  After the outer loop, a threshold kernel (R.5.2) produces the
//  boolean fractureMask. That kernel is trivially a
//  `mask[i] = (type[i] == SOLID) && (phi[i] < EPSILON)` scan and is
//  included below as `poisson_oracle_threshold.comp.glsl`.
// ═══════════════════════════════════════════════════════════════

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint Lx;
    uint Ly;
    uint Lz;
} pc;

layout(set = 0, binding = 0) readonly  buffer VType  { uint  vtype[]; };
layout(set = 0, binding = 1) readonly  buffer PhiIn  { float phiIn[]; };
layout(set = 0, binding = 2) writeonly buffer PhiOut { float phiOut[]; };

const uint VOXEL_AIR    = 0u;
const uint VOXEL_SOLID  = 1u;
const uint VOXEL_ANCHOR = 2u;

// 26-connected face+edge+corner offsets — must match
// PFSFStencil.NEIGHBOR_OFFSETS and label_prop_iterate so CPU and GPU
// paths cannot drift on the stencil.
const ivec3 OFFSETS[26] = ivec3[](
    ivec3( 1, 0, 0), ivec3(-1, 0, 0),
    ivec3( 0, 1, 0), ivec3( 0,-1, 0),
    ivec3( 0, 0, 1), ivec3( 0, 0,-1),
    ivec3( 1, 1, 0), ivec3( 1,-1, 0), ivec3(-1, 1, 0), ivec3(-1,-1, 0),
    ivec3( 1, 0, 1), ivec3( 1, 0,-1), ivec3(-1, 0, 1), ivec3(-1, 0,-1),
    ivec3( 0, 1, 1), ivec3( 0, 1,-1), ivec3( 0,-1, 1), ivec3( 0,-1,-1),
    ivec3( 1, 1, 1), ivec3( 1, 1,-1), ivec3( 1,-1, 1), ivec3( 1,-1,-1),
    ivec3(-1, 1, 1), ivec3(-1, 1,-1), ivec3(-1,-1, 1), ivec3(-1,-1,-1)
);

uint flatIdx(uint x, uint y, uint z) {
    return x + pc.Lx * (y + pc.Ly * z);
}

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint N = pc.Lx * pc.Ly * pc.Lz;
    if (i >= N) return;

    uint t = vtype[i];
    if (t == VOXEL_ANCHOR) { phiOut[i] = 1.0; return; }
    if (t == VOXEL_AIR)    { phiOut[i] = 0.0; return; }

    uint x   = i % pc.Lx;
    uint rem = i / pc.Lx;
    uint y   = rem % pc.Ly;
    uint z   = rem / pc.Ly;

    float sum = 0.0;
    int count = 0;
    for (int k = 0; k < 26; k++) {
        ivec3 o = OFFSETS[k];
        int nx = int(x) + o.x;
        int ny = int(y) + o.y;
        int nz = int(z) + o.z;
        if (nx < 0 || nx >= int(pc.Lx)) continue;
        if (ny < 0 || ny >= int(pc.Ly)) continue;
        if (nz < 0 || nz >= int(pc.Lz)) continue;
        uint j = flatIdx(uint(nx), uint(ny), uint(nz));
        if (vtype[j] == VOXEL_AIR) continue;
        sum += phiIn[j];
        count++;
    }
    phiOut[i] = (count > 0) ? (sum / float(count)) : 0.0;
}
