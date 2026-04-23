#version 450
#extension GL_GOOGLE_include_directive : enable

// ═══════════════════════════════════════════════════════════════
//  PFSF Label Propagation — Summarise Phase 1: aggregate.
//
//  Runs AFTER label_prop_summarise_alloc has populated rootToSlot and
//  initialised each component's metadata record. Visits every live
//  voxel and folds its contribution into its component's aggregate:
//
//    blockCount += 1                        (atomicAdd)
//    anchored   |= (type == ANCHOR ? 1 : 0) (atomicOr)
//    aabbMin     = min(aabbMin, (x,y,z))    (atomicMin per component)
//    aabbMax     = max(aabbMax, (x,y,z))    (atomicMax per component)
//
//  Output consumed by PFSFLabelPropRecorder CPU readback, which in
//  turn feeds:
//    - CollapseManager.enqueueCollapse(...) for any component with
//      anchored == 0 (orphan → immediate fall)
//    - StructureIslandRegistry.applyGpuComponent(...) for any
//      component whose (rootLabel, blockCount, anchored, AABB) does
//      not match the existing CPU-side island record (indicating a
//      legitimate split into one or more still-anchored fragments —
//      the case the earlier "orphan-only" design missed).
// ═══════════════════════════════════════════════════════════════

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint Lx;
    uint Ly;
    uint Lz;
    uint maxComponents;
} pc;

layout(set = 0, binding = 0) readonly buffer VType       { uint vtype[];      };
layout(set = 0, binding = 1) readonly buffer IslandId    { uint islandId[];   };
layout(set = 0, binding = 2) readonly buffer RootToSlot  { uint rootToSlot[]; };
layout(set = 0, binding = 3)          buffer Components  { uint components[]; };

const uint VOXEL_ANCHOR            = 2u;
const uint NO_ISLAND               = 0xFFFFFFFFu;
const uint UNSET_SLOT              = 0xFFFFFFFFu;
const uint FIELDS_PER_COMPONENT    = 12u;

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint N = pc.Lx * pc.Ly * pc.Lz;
    if (i >= N) return;

    uint lbl = islandId[i];
    if (lbl == NO_ISLAND) return;

    uint rootIdx = lbl - 1u;               // label l was set to (root+1)
    if (rootIdx >= N) return;              // safety net against corruption
    uint slot = rootToSlot[rootIdx];
    if (slot == UNSET_SLOT) return;        // root wasn't allocated (overflow)
    if (slot >= pc.maxComponents) return;  // belt-and-braces

    uint base = slot * FIELDS_PER_COMPONENT;

    // Count + anchor flag
    atomicAdd(components[base + 1u], 1u);
    if (vtype[i] == VOXEL_ANCHOR) {
        atomicOr(components[base + 2u], 1u);
    }

    // AABB
    uint x   = i % pc.Lx;
    uint rem = i / pc.Lx;
    uint y   = rem % pc.Ly;
    uint z   = rem / pc.Ly;
    atomicMin(components[base + 4u], x);
    atomicMin(components[base + 5u], y);
    atomicMin(components[base + 6u], z);
    atomicMax(components[base + 8u], x);
    atomicMax(components[base + 9u], y);
    atomicMax(components[base + 10u], z);
}
