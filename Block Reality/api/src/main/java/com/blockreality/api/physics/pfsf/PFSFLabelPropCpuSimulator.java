package com.blockreality.api.physics.pfsf;

import java.util.Arrays;

/**
 * Bit-accurate CPU simulator of the four-kernel Vulkan label-propagation
 * pipeline (`label_prop_init`, `label_prop_iterate`, `label_prop_summarise_alloc`,
 * `label_prop_summarise_aggregate`). Serves two purposes:
 * <ol>
 *   <li><b>Correctness oracle.</b> Real GPU dispatch is not observable
 *       in the test sandbox, so we cannot run the kernels themselves
 *       during CI. Instead, the simulator replays each kernel's work
 *       in single-threaded Java with the exact memory model (atomicMin,
 *       atomicOr, atomicAdd) collapsed to plain sequential operations.
 *       Two separate tests — {@code LabelPropagationTest} for SV ≡ BFS
 *       and {@code GpuLabelPropSimulatorTest} for simulator ≡ BFS —
 *       wedge the GPU algorithm into the same correctness contract
 *       Phase A already validates.</li>
 *   <li><b>Implementation reference.</b> The Java body of each
 *       simulator method mirrors the GLSL kernel line-by-line, so an
 *       engineer porting or debugging the GPU path on a real machine
 *       has a single authoritative Java translation of what each
 *       kernel "should" be computing on each voxel.</li>
 * </ol>
 *
 * <p>This simulator is NOT a production path — at run-time the real
 * GPU kernels are what fire. It stays in {@code src/main} so
 * {@code StructureIslandRegistry}'s GPU-result decode path can share
 * record types (e.g. {@link ComponentMeta}) with it without awkward
 * test-jar plumbing.
 */
public final class PFSFLabelPropCpuSimulator {

    /** Maximum slots allocated for unique components in the aggregate readback. */
    public static final int MAX_COMPONENTS = 64;

    /** Sentinel for AIR voxels in the islandId array. Matches GLSL NO_ISLAND. */
    public static final int NO_ISLAND = 0xFFFFFFFF;
    /** Sentinel for "voxel is not a component root" in the rootToSlot array. */
    public static final int UNSET_SLOT = 0xFFFFFFFF;

    /** Number of fixed (hook, jump) iterations run by the host loop. */
    public static final int DEFAULT_SV_ITERATIONS = 12;

    /**
     * One component's aggregate record as produced by the
     * {@code label_prop_summarise_aggregate} shader. Matches the 48-byte
     * std430 layout described in the shader: (rootLabel, blockCount,
     * anchored, pad, aabbMin.x/y/z, pad, aabbMax.x/y/z, pad).
     */
    public record ComponentMeta(int rootLabel,
                                int blockCount,
                                boolean anchored,
                                int aabbMinX, int aabbMinY, int aabbMinZ,
                                int aabbMaxX, int aabbMaxY, int aabbMaxZ) {}

    /** Full result of one simulator run: per-voxel labels plus per-component metadata. */
    public record SimulatorResult(int[] islandId,
                                  int numComponents,
                                  boolean overflow,
                                  ComponentMeta[] components) {}

    private PFSFLabelPropCpuSimulator() {}

    /**
     * Runs init → iterate×DEFAULT_SV_ITERATIONS → summarise_alloc →
     * summarise_aggregate end-to-end and returns the result that
     * would be read back from GPU.
     *
     * @param type  voxel type array (AIR/SOLID/ANCHOR), length Lx*Ly*Lz.
     * @param Lx    grid width.
     * @param Ly    grid height.
     * @param Lz    grid depth.
     * @return simulator result with islandId, numComponents, overflow
     *         flag, and component records.
     */
    public static SimulatorResult run(byte[] type, int Lx, int Ly, int Lz) {
        return run(type, Lx, Ly, Lz, DEFAULT_SV_ITERATIONS, MAX_COMPONENTS);
    }

    /** Full-parameter run; exposed so tests can exercise pathological iteration counts. */
    public static SimulatorResult run(byte[] type, int Lx, int Ly, int Lz,
                                      int svIterations, int maxComponents) {
        int[] islandId = kernelInit(type, Lx, Ly, Lz);
        for (int k = 0; k < svIterations; k++) {
            kernelIterateHook(islandId, Lx, Ly, Lz);
            kernelIterateJump(islandId, Lx, Ly, Lz);
        }
        AllocResult alloc = kernelSummariseAlloc(islandId, Lx, Ly, Lz, maxComponents);
        ComponentMeta[] components = kernelSummariseAggregate(
                type, islandId, alloc.rootToSlot, alloc.components, alloc.numComponents, Lx, Ly, Lz);
        return new SimulatorResult(islandId, alloc.numComponents, alloc.overflow, components);
    }

    // ═════════════════════════════════════════════════════════════════
    //  Individual kernel simulations (line-for-line mirror of GLSL)
    // ═════════════════════════════════════════════════════════════════

    /** Mirror of {@code label_prop_init.comp.glsl}. */
    public static int[] kernelInit(byte[] type, int Lx, int Ly, int Lz) {
        int n = Lx * Ly * Lz;
        int[] islandId = new int[n];
        for (int i = 0; i < n; i++) {
            byte t = type[i];
            if (t == LabelPropagation.TYPE_SOLID || t == LabelPropagation.TYPE_ANCHOR) {
                islandId[i] = i + 1;
            } else {
                islandId[i] = NO_ISLAND;
            }
        }
        return islandId;
    }

    /** Mirror of {@code label_prop_iterate.comp.glsl} pass=0 (hook-to-min). */
    public static void kernelIterateHook(int[] islandId, int Lx, int Ly, int Lz) {
        int[] next = islandId.clone();
        int[][] offs = PFSFStencil.NEIGHBOR_OFFSETS;
        for (int z = 0; z < Lz; z++) {
            for (int y = 0; y < Ly; y++) {
                for (int x = 0; x < Lx; x++) {
                    int i = flatIdx(x, y, z, Lx, Ly);
                    int myLbl = next[i];
                    if (myLbl == NO_ISLAND) continue;
                    int minLbl = myLbl;
                    for (int[] o : offs) {
                        int nx = x + o[0], ny = y + o[1], nz = z + o[2];
                        if (nx < 0 || nx >= Lx || ny < 0 || ny >= Ly || nz < 0 || nz >= Lz) continue;
                        int j = flatIdx(nx, ny, nz, Lx, Ly);
                        int nLbl = next[j];
                        if (nLbl == NO_ISLAND) continue;
                        if (unsignedLess(nLbl, minLbl)) minLbl = nLbl;
                    }
                    if (unsignedLess(minLbl, myLbl)) {
                        // atomicMin semantics — since we are single-threaded,
                        // assigning the computed min is equivalent.
                        next[i] = minLbl;
                    }
                }
            }
        }
        System.arraycopy(next, 0, islandId, 0, islandId.length);
    }

    /** Mirror of {@code label_prop_iterate.comp.glsl} pass=1 (pointer jump). */
    public static void kernelIterateJump(int[] islandId, int Lx, int Ly, int Lz) {
        int n = Lx * Ly * Lz;
        int[] next = islandId.clone();
        for (int i = 0; i < n; i++) {
            int myLbl = islandId[i];
            if (myLbl == NO_ISLAND) continue;
            // myLbl == rootIdx + 1 (in unsigned arithmetic)
            int parentIdx = myLbl - 1;
            if (parentIdx < 0 || parentIdx >= n) continue;
            int parentLbl = islandId[parentIdx];
            if (parentLbl == NO_ISLAND) continue;
            if (unsignedLess(parentLbl, myLbl)) {
                next[i] = parentLbl;
            }
        }
        System.arraycopy(next, 0, islandId, 0, islandId.length);
    }

    /** Mirror of {@code label_prop_summarise_alloc.comp.glsl}. */
    public static AllocResult kernelSummariseAlloc(int[] islandId, int Lx, int Ly, int Lz, int maxComponents) {
        int n = Lx * Ly * Lz;
        int[] rootToSlot = new int[n];
        Arrays.fill(rootToSlot, UNSET_SLOT);
        int[] components = new int[maxComponents * 12];
        int numComponents = 0;
        boolean overflow = false;
        for (int i = 0; i < n; i++) {
            int lbl = islandId[i];
            if (lbl == NO_ISLAND) continue;
            if (lbl != i + 1) continue;  // not a root
            int slot = numComponents++;
            if (slot >= maxComponents) {
                overflow = true;
                numComponents = maxComponents;  // clamp
                continue;
            }
            rootToSlot[i] = slot;
            int base = slot * 12;
            components[base] = lbl;                    // rootLabel
            components[base + 1] = 0;                  // blockCount
            components[base + 2] = 0;                  // anchored
            components[base + 3] = 0;                  // pad
            components[base + 4] = 0xFFFFFFFF;         // aabbMin.x = UINT_MAX
            components[base + 5] = 0xFFFFFFFF;         // aabbMin.y
            components[base + 6] = 0xFFFFFFFF;         // aabbMin.z
            components[base + 7] = 0;                  // pad
            components[base + 8] = 0;                  // aabbMax.x
            components[base + 9] = 0;                  // aabbMax.y
            components[base + 10] = 0;                 // aabbMax.z
            components[base + 11] = 0;                 // pad
        }
        return new AllocResult(numComponents, rootToSlot, components, overflow);
    }

    /** Mirror of {@code label_prop_summarise_aggregate.comp.glsl}. */
    public static ComponentMeta[] kernelSummariseAggregate(byte[] type, int[] islandId,
                                                           int[] rootToSlot, int[] components,
                                                           int numComponents,
                                                           int Lx, int Ly, int Lz) {
        int n = Lx * Ly * Lz;
        for (int i = 0; i < n; i++) {
            int lbl = islandId[i];
            if (lbl == NO_ISLAND) continue;
            int rootIdx = lbl - 1;
            if (rootIdx < 0 || rootIdx >= n) continue;
            int slot = rootToSlot[rootIdx];
            if (slot == UNSET_SLOT) continue;
            if (slot >= numComponents) continue;
            int base = slot * 12;
            components[base + 1]++;                                                       // blockCount
            if (type[i] == LabelPropagation.TYPE_ANCHOR) components[base + 2] |= 1;       // anchored
            int x = i % Lx, rem = i / Lx;
            int y = rem % Ly, z = rem / Ly;
            components[base + 4] = unsignedMin(components[base + 4], x);
            components[base + 5] = unsignedMin(components[base + 5], y);
            components[base + 6] = unsignedMin(components[base + 6], z);
            components[base + 8] = unsignedMax(components[base + 8], x);
            components[base + 9] = unsignedMax(components[base + 9], y);
            components[base + 10] = unsignedMax(components[base + 10], z);
        }
        ComponentMeta[] out = new ComponentMeta[numComponents];
        for (int s = 0; s < numComponents; s++) {
            int base = s * 12;
            out[s] = new ComponentMeta(
                    components[base],
                    components[base + 1],
                    (components[base + 2] & 1) != 0,
                    components[base + 4], components[base + 5], components[base + 6],
                    components[base + 8], components[base + 9], components[base + 10]);
        }
        return out;
    }

    /** Intermediate product of the alloc kernel. */
    public record AllocResult(int numComponents, int[] rootToSlot, int[] components, boolean overflow) {}

    // ═════════════════════════════════════════════════════════════════
    //  Small helpers — GLSL is unsigned; Java uses signed int so we
    //  defend against 0xFFFFFFFF-style sentinels via explicit unsigned
    //  comparisons.
    // ═════════════════════════════════════════════════════════════════

    private static boolean unsignedLess(int a, int b) {
        return Integer.compareUnsigned(a, b) < 0;
    }

    private static int unsignedMin(int a, int b) {
        return Integer.compareUnsigned(a, b) <= 0 ? a : b;
    }

    private static int unsignedMax(int a, int b) {
        return Integer.compareUnsigned(a, b) >= 0 ? a : b;
    }

    private static int flatIdx(int x, int y, int z, int Lx, int Ly) {
        return x + Lx * (y + Ly * z);
    }
}
