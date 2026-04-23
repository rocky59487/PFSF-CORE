#version 450
#extension GL_GOOGLE_include_directive : enable

// ═══════════════════════════════════════════════════════════════
//  PFSF Label Propagation — Summarise Phase 0: slot allocation.
//
//  After Shiloach–Vishkin iteration has converged, every live voxel
//  carries its component's root label. A voxel `i` is its own root
//  iff islandId[i] == i+1. This kernel visits only those roots,
//  allocates them a dense slot index via atomicAdd on a shared
//  counter, and initialises the component's metadata record.
//
//  Output contract:
//    numComponents[0]    — total number of allocated slots, ≤ MAX_COMPONENTS
//    rootToSlot[rootIdx] — dense slot for the component whose root voxel
//                          is at flat index rootIdx; UNSET_SLOT if the
//                          voxel is not a root
//    components[slot]    — per-component aggregate record (rootLabel,
//                          blockCount=0, anchored=0, aabbMin=UINT_MAX,
//                          aabbMax=0); later filled by the aggregate pass.
//
//  If a component-heavy fracture exceeds MAX_COMPONENTS, the
//  overflow flag bit is set on numComponents[1]; the CPU readback
//  path treats overflow by falling back to the CPU SV reference
//  for the affected island that tick (graceful degradation, never
//  silent data loss).
// ═══════════════════════════════════════════════════════════════

layout(local_size_x = 256) in;

layout(push_constant) uniform PC {
    uint Lx;
    uint Ly;
    uint Lz;
    uint maxComponents;  // MAX_COMPONENTS constant, passed for guard check
} pc;

// 32-byte ComponentMeta record, std430 layout:
//   uint  rootLabel     offset  0
//   uint  blockCount    offset  4
//   uint  anchored      offset  8   (0 or 1; updated by aggregate pass via atomicOr)
//   uint  _pad0         offset 12
//   uvec3 aabbMin       offset 16
//   uvec3 aabbMax       offset 28  — oops, std430 pads uvec3 to 16B; use 4 uints each instead
// For safe cross-driver portability, we lay out as 8 plain uints:
//   [rootLabel, blockCount, anchored, _pad, aabbMinX, aabbMinY, aabbMinZ, _pad,
//    aabbMaxX, aabbMaxY, aabbMaxZ, _pad] => 48 bytes.
// Simpler: declare as uint[12] per component to sidestep std430 vector padding.

layout(set = 0, binding = 0) readonly buffer IslandId     { uint islandId[];       };
layout(set = 0, binding = 1)          buffer NumComponents{ uint numComponents[];  }; // [0]=count, [1]=overflow
layout(set = 0, binding = 2)          buffer RootToSlot   { uint rootToSlot[];     }; // size N
layout(set = 0, binding = 3)          buffer Components   { uint components[];     }; // flattened: MAX_COMPONENTS × 12 uints

const uint NO_ISLAND   = 0xFFFFFFFFu;
const uint UNSET_SLOT  = 0xFFFFFFFFu;
const uint FIELDS_PER_COMPONENT = 12u;  // matches struct layout in aggregate pass

void main() {
    uint i = gl_GlobalInvocationID.x;
    uint N = pc.Lx * pc.Ly * pc.Lz;
    if (i >= N) return;

    // Default: mark this flat index as not-a-root.
    rootToSlot[i] = UNSET_SLOT;

    uint lbl = islandId[i];
    if (lbl == NO_ISLAND) return;
    // Voxel i is a root iff its label equals its own flat index + 1.
    if (lbl != i + 1u) return;

    // Attempt to allocate a dense slot for this component.
    uint slot = atomicAdd(numComponents[0], 1u);
    if (slot >= pc.maxComponents) {
        // Overflow — the aggregate pass will also check and skip.
        atomicOr(numComponents[1], 1u);
        return;
    }
    rootToSlot[i] = slot;

    // Initialise the aggregate record.
    uint base = slot * FIELDS_PER_COMPONENT;
    components[base + 0u] = lbl;          // rootLabel
    components[base + 1u] = 0u;           // blockCount
    components[base + 2u] = 0u;           // anchored
    components[base + 3u] = 0u;           // pad
    components[base + 4u] = 0xFFFFFFFFu;  // aabbMin.x
    components[base + 5u] = 0xFFFFFFFFu;  // aabbMin.y
    components[base + 6u] = 0xFFFFFFFFu;  // aabbMin.z
    components[base + 7u] = 0u;           // pad
    components[base + 8u] = 0u;           // aabbMax.x
    components[base + 9u] = 0u;           // aabbMax.y
    components[base + 10u] = 0u;          // aabbMax.z
    components[base + 11u] = 0u;          // pad
}
