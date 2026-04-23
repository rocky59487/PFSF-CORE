#version 450
#extension GL_GOOGLE_include_directive : enable

// ═══════════════════════════════════════════════════════════════
//  PFSF Label Propagation — initialisation pass
//
//  One thread per voxel. Every live voxel (SOLID or ANCHOR) is seeded
//  with a unique label equal to (flat index + 1). AIR voxels are
//  sentinelled with NO_ISLAND = 0xFFFFFFFF.
//
//  This mirrors the CPU reference
//  `LabelPropagation.shiloachVishkin` so GPU output can be validated
//  bit-for-bit against it on the same input domain. The companion
//  iteration kernel is `label_prop_iterate.comp.glsl`.
//
//  Anchor classification is NOT folded into the initial label here —
//  assigning anchors a shared low label would merge spatially-disjoint
//  anchored clusters into a single pseudo-component, which is the wrong
//  answer for the split path in StructureIslandRegistry. A future
//  summarise pass (`label_prop_summarise.comp.glsl`) will do the
//  anchor rollup after convergence instead.
// ═══════════════════════════════════════════════════════════════

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint Lx;
    uint Ly;
    uint Lz;
} pc;

layout(set = 0, binding = 0) readonly buffer VType    { uint  vtype[];    };
layout(set = 0, binding = 1) writeonly buffer IslandId { uint  islandId[]; };

const uint VOXEL_AIR    = 0u;
const uint VOXEL_SOLID  = 1u;
const uint VOXEL_ANCHOR = 2u;
const uint NO_ISLAND    = 0xFFFFFFFFu;

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint N = pc.Lx * pc.Ly * pc.Lz;
    if (i >= N) return;

    uint t = vtype[i];
    if (t == VOXEL_SOLID || t == VOXEL_ANCHOR) {
        islandId[i] = i + 1u;
    } else {
        islandId[i] = NO_ISLAND;
    }
}
