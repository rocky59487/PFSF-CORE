#version 450
#extension GL_GOOGLE_include_directive : enable

// ═══════════════════════════════════════════════════════════════
//  PFSF Label Propagation — anchor-map summarise pass.
//
//  After Shiloach–Vishkin iteration has converged, every live voxel
//  carries its component's root label. This kernel produces a compact
//  "anchor bitmap": bit k of anchorBitmap is set iff there exists at
//  least one voxel with type == ANCHOR whose final label is k.
//
//  Downstream, CollapseManager reads the bitmap together with the
//  failure-compact readback: for each block position recovered from
//  failure_compact, it looks up that block's islandId and tests the
//  corresponding bit. Missing bit ⇒ the component is orphan ⇒
//  immediate collapse. This eliminates the need to readback the full
//  N × 4 byte islandId buffer every tick.
//
//  Sizing: labels live in [1, N], so the bitmap needs ceil(N/32) uints
//  = ~N/8 bytes. For a 64³ island that is ~32 KB — well within a
//  single staging-buffer hop.
//
//  Race model: atomicOr is commutative / idempotent; no ordering
//  constraint on concurrent writers.
// ═══════════════════════════════════════════════════════════════

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint Lx;
    uint Ly;
    uint Lz;
} pc;

layout(set = 0, binding = 0) readonly buffer VType        { uint vtype[];         };
layout(set = 0, binding = 1) readonly buffer IslandId     { uint islandId[];      };
layout(set = 0, binding = 2)          buffer AnchorBitmap { uint anchorBitmap[];  };

const uint VOXEL_ANCHOR = 2u;
const uint NO_ISLAND    = 0xFFFFFFFFu;

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint N = pc.Lx * pc.Ly * pc.Lz;
    if (i >= N) return;

    if (vtype[i] != VOXEL_ANCHOR) return;
    uint lbl = islandId[i];
    if (lbl == NO_ISLAND) return;

    // Labels are (flat index + 1) in [1, N]; map to bitmap positions [0, N-1]
    // by subtracting 1. Choose the uint word and bit within it.
    uint pos  = lbl - 1u;
    uint word = pos >> 5;          // / 32
    uint bit  = 1u << (pos & 31u); // % 32

    atomicOr(anchorBitmap[word], bit);
}
