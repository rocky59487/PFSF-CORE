#version 450
#extension GL_GOOGLE_include_directive : enable

// ═══════════════════════════════════════════════════════════════
//  PFSF Label Propagation — one iteration of Shiloach–Vishkin
//  (hook-to-min + single pointer jump).
//
//  Dispatched from the host in a FIXED UNROLLED LOOP — no CPU
//  round-trip per iteration. The host records a fixed number of
//  (hook, jump) pairs bounded above by ceil(log2(maximum expected
//  component diameter)). For Minecraft structures a diameter of
//  4096 voxels is already impossibly large, so 12 iterations
//  (2^12 = 4096) is the recorder's chosen cap. This eliminates the
//  0.5–2 ms vkQueueSubmit + vkWaitForFences + map/unmap cost that
//  a per-iteration readback of a "changedFlag" would have incurred
//  — a cost fatal to a 60 Hz tick budget.
//
//  Push constant `pass` selects which sub-pass to run:
//      pass == 0  →  hook-to-min over 26 neighbours
//      pass == 1  →  pointer jump islandId[i] <- islandId[islandId[i]-1]
//
//  Race handling: concurrent atomicMin is the natural fit here. If
//  multiple threads try to lower the same voxel's label, the smallest
//  wins and the write order is immaterial. We do NOT maintain a
//  per-iteration convergence flag because we are not reading back —
//  the fixed iteration count is the termination criterion.
//
//  Correctness oracle: this kernel's output is validated bit-for-bit
//  against the CPU reference `LabelPropagation.shiloachVishkin` in
//  `Block Reality/api/src/main/java/.../pfsf/LabelPropagation.java`
//  (see GpuLabelPropCorrectnessTest — Phase B.2f).
// ═══════════════════════════════════════════════════════════════

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint Lx;
    uint Ly;
    uint Lz;
    uint pass;       // 0 = hook-to-min; 1 = pointer jump
} pc;

layout(set = 0, binding = 0) readonly buffer VType    { uint vtype[];    };
layout(set = 0, binding = 1) coherent  buffer IslandId { uint islandId[]; };

const uint NO_ISLAND = 0xFFFFFFFFu;

// 26-connected face + edge + corner offsets.
// Order matches PFSFStencil.NEIGHBOR_OFFSETS exactly.
const ivec3 OFFSETS[26] = ivec3[](
    // face (6)
    ivec3( 1, 0, 0), ivec3(-1, 0, 0),
    ivec3( 0, 1, 0), ivec3( 0,-1, 0),
    ivec3( 0, 0, 1), ivec3( 0, 0,-1),
    // edge XY (4)
    ivec3( 1, 1, 0), ivec3( 1,-1, 0), ivec3(-1, 1, 0), ivec3(-1,-1, 0),
    // edge XZ (4)
    ivec3( 1, 0, 1), ivec3( 1, 0,-1), ivec3(-1, 0, 1), ivec3(-1, 0,-1),
    // edge YZ (4)
    ivec3( 0, 1, 1), ivec3( 0, 1,-1), ivec3( 0,-1, 1), ivec3( 0,-1,-1),
    // corner (8)
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

    uint myLbl = islandId[i];
    if (myLbl == NO_ISLAND) return;

    uint x   = i % pc.Lx;
    uint rem = i / pc.Lx;
    uint y   = rem % pc.Ly;
    uint z   = rem / pc.Ly;

    if (pc.pass == 0u) {
        // ─── Hook-to-min ──────────────────────────────────────────
        uint minLbl = myLbl;
        for (int k = 0; k < 26; k++) {
            ivec3 o = OFFSETS[k];
            int nx = int(x) + o.x;
            int ny = int(y) + o.y;
            int nz = int(z) + o.z;
            if (nx < 0 || nx >= int(pc.Lx)) continue;
            if (ny < 0 || ny >= int(pc.Ly)) continue;
            if (nz < 0 || nz >= int(pc.Lz)) continue;
            uint j = flatIdx(uint(nx), uint(ny), uint(nz));
            uint nLbl = islandId[j];
            if (nLbl == NO_ISLAND) continue;
            if (nLbl < minLbl) minLbl = nLbl;
        }
        if (minLbl < myLbl) {
            atomicMin(islandId[i], minLbl);
        }
    } else {
        // ─── Pointer jump (single hop) ────────────────────────────
        // myLbl == (some flat index) + 1; parentIdx is that flat index.
        uint parentIdx = myLbl - 1u;
        if (parentIdx < N) {
            uint parentLbl = islandId[parentIdx];
            if (parentLbl != NO_ISLAND && parentLbl < myLbl) {
                atomicMin(islandId[i], parentLbl);
            }
        }
    }
}
