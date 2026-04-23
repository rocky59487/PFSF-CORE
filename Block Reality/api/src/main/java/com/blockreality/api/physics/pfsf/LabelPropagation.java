package com.blockreality.api.physics.pfsf;

import net.minecraft.core.BlockPos;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Shiloach–Vishkin–style parallel label propagation for voxel island
 * connectivity, together with a sparse BFS fallback that operates on
 * {@link BlockPos} sets (the form used throughout {@code StructureIslandRegistry}).
 *
 * <p>This class is the <b>single source of truth</b> for two questions
 * previously answered by two separate code paths:
 * <ol>
 *   <li>Which SOLID voxels belong to the same connected component?</li>
 *   <li>Does that component reach at least one ANCHOR?</li>
 * </ol>
 * Prior to this class, the registry performed (1) via a per-island BFS
 * after every destruction and answered (2) implicitly through the PFSF
 * potential-field solver (an orphan island's φ blows up after a few
 * ticks and fails the {@code PHI_ORPHAN_THRESHOLD} check). That indirect
 * detection path is the "floating blocks for several ticks" bug
 * documented in {@code research/PFSF_GPU_ACADEMIC_AUDIT_2026-04-23.md}:
 * a split-off island is structurally orphaned <i>the tick it is born</i>,
 * but the old pipeline only discovers this after the physics field has
 * had time to diverge.
 *
 * <p>Both algorithms below answer (1) and (2) in a single pass.
 * Anchors are represented by a reserved {@link #ANCHORED_ID} label (0);
 * every SOLID voxel starts with a unique non-zero label. After
 * convergence, any component whose root label is still
 * {@code ANCHORED_ID} is anchored; any component with a non-zero root
 * label is orphan and must be collapsed immediately.
 *
 * <h2>CPU production path — {@link #bfsComponents}</h2>
 * Operates on {@code Set<BlockPos>} directly. Used by
 * {@code StructureIslandRegistry.checkAndSplitIsland} because its
 * memory cost scales with the number of solid/anchor blocks, not the
 * island's axis-aligned bounding volume.
 *
 * <h2>GPU-parity reference path — {@link #shiloachVishkin}</h2>
 * Operates on a flat {@code byte[]} voxel-type array with the exact
 * semantics we will port to {@code label_prop.comp.glsl} (Phase B of
 * the implementation plan). Exists primarily as the oracle against
 * which the GPU kernel will be validated.
 *
 * <p>{@link #bfsComponents} and {@link #shiloachVishkin} are kept in
 * sync by {@code LabelPropagationTest}, which exercises both on
 * randomly generated voxel domains and asserts that the partition
 * into anchored / orphan components is identical.
 */
public final class LabelPropagation {

    /** Special label shared by every ANCHOR voxel; the smallest possible value. */
    public static final int ANCHORED_ID = 0;
    /** Label returned for AIR voxels in the flat-array API. */
    public static final int NO_ISLAND = -1;

    /** Voxel type constants, aligned with {@code VoxelPhysicsCpuReference}. */
    public static final byte TYPE_AIR    = 0;
    public static final byte TYPE_SOLID  = 1;
    public static final byte TYPE_ANCHOR = 2;

    /** Neighbour sets used by both paths. */
    public enum NeighborPolicy {
        /** 6 face neighbours; matches the existing face-based Minecraft connectivity. */
        FACE_6,
        /** 26 face+edge+corner neighbours; matches the PFSF Laplacian stencil. */
        FULL_26
    }

    /** One connected component plus its anchored/orphan classification. */
    public record Component(Set<BlockPos> members, boolean anchored) {}

    /**
     * Result of {@link #bfsComponents}: the full partition of SOLID
     * blocks into connected components, each tagged anchored/orphan.
     */
    public record PartitionResult(List<Component> components) {
        /** Convenience: the union of all orphan (non-anchored) components. */
        public Set<BlockPos> orphans() {
            Set<BlockPos> out = new HashSet<>();
            for (Component c : components) if (!c.anchored) out.addAll(c.members);
            return out;
        }
    }

    private LabelPropagation() {}

    // ═════════════════════════════════════════════════════════════════
    //  CPU production: sparse-set BFS with anchor detection (O(|members|))
    // ═════════════════════════════════════════════════════════════════

    /**
     * Partition {@code members} into connected components and classify
     * each as anchored (touches a block in {@code anchors}) or orphan.
     *
     * <p>A block in {@code anchors} is considered part of the component
     * that contains it (if it is also in {@code members}), or adjacent
     * to it (if not — e.g. bedrock outside the registered island).
     *
     * <p>Complexity: O(|members| · K) where K is the neighbour count of
     * the chosen policy (6 or 26). No hidden BFS-budget truncation.
     *
     * @param members the SOLID blocks whose connectivity is being queried
     * @param anchors blocks treated as ANCHOR (typically bedrock, barriers,
     *                or explicitly pinned voxels); may overlap with members
     *                or lie just outside them.
     * @param policy  neighbour-connectivity policy; {@link NeighborPolicy#FACE_6}
     *                matches the existing registry semantics.
     */
    public static PartitionResult bfsComponents(Set<BlockPos> members,
                                                Set<BlockPos> anchors,
                                                NeighborPolicy policy) {
        if (members == null || members.isEmpty()) {
            return new PartitionResult(Collections.emptyList());
        }
        Set<BlockPos> unassigned = new HashSet<>(members);
        List<Component> out = new ArrayList<>();
        int[][] offsets = offsetsFor(policy);

        while (!unassigned.isEmpty()) {
            BlockPos seed = unassigned.iterator().next();
            Set<BlockPos> group = new HashSet<>();
            boolean anchored = false;
            Deque<BlockPos> q = new ArrayDeque<>();
            q.add(seed);
            unassigned.remove(seed);
            group.add(seed);

            while (!q.isEmpty()) {
                BlockPos p = q.poll();
                // A block that sits on top of / next to an anchor counts as anchored.
                if (anchors != null && !anchored) {
                    if (anchors.contains(p)) {
                        anchored = true;
                    } else {
                        for (int[] off : offsets) {
                            BlockPos n = p.offset(off[0], off[1], off[2]);
                            if (anchors.contains(n)) { anchored = true; break; }
                        }
                    }
                }
                for (int[] off : offsets) {
                    BlockPos n = p.offset(off[0], off[1], off[2]);
                    if (unassigned.remove(n)) {
                        group.add(n);
                        q.add(n);
                    }
                }
            }
            out.add(new Component(group, anchored));
        }
        return new PartitionResult(out);
    }

    // ═════════════════════════════════════════════════════════════════
    //  GPU-parity reference: Shiloach–Vishkin on a flat byte[] domain
    // ═════════════════════════════════════════════════════════════════

    /**
     * Classical hook-to-min SV label propagation with pointer jumping,
     * written as the CPU oracle for the forthcoming {@code label_prop.comp.glsl}
     * Vulkan kernel. Correctness is what matters here, not raw speed:
     * on CPU, {@link #bfsComponents} is strictly cheaper.
     *
     * <p>Output semantics:
     * <ul>
     *   <li>{@code islandId[i] = NO_ISLAND} — voxel {@code i} is AIR.</li>
     *   <li>{@code islandId[i] > 0} — voxel {@code i}'s component root
     *       label, equal to (smallest participating flat-index + 1).
     *       Two voxels are in the same connected component iff they
     *       carry the same label.</li>
     * </ul>
     *
     * <p>Anchor classification is <b>post-hoc</b>: a component is
     * anchored iff at least one of its voxels has {@code type == TYPE_ANCHOR}.
     * Initially assigning a shared low label to all ANCHOR voxels would
     * merge spatially-disjoint anchored clusters into one pseudo-component,
     * which is incorrect for the split path in
     * {@code StructureIslandRegistry}. Keeping each voxel unique during
     * propagation preserves the true component partition; {@link #partitionFromFlat}
     * does the anchor rollup afterwards.
     *
     * <p>The planned {@code label_prop_summarise.comp.glsl} GPU pass
     * mirrors this: it computes, per unique root label, whether any
     * member voxel has {@code type == TYPE_ANCHOR}, producing the
     * compact anchor map expected by CollapseManager.
     */
    public static int[] shiloachVishkin(byte[] type, int Lx, int Ly, int Lz, NeighborPolicy policy) {
        int n = Lx * Ly * Lz;
        if (type.length != n) {
            throw new IllegalArgumentException("type length " + type.length + " != Lx*Ly*Lz = " + n);
        }
        int[] id = new int[n];
        // Init: every live voxel (SOLID or ANCHOR) gets a unique label i+1;
        // AIR is NO_ISLAND. Anchor status is NOT folded into the label —
        // it is recovered post-hoc via {@link #partitionFromFlat}.
        for (int i = 0; i < n; i++) {
            byte t = type[i];
            if (t == TYPE_SOLID || t == TYPE_ANCHOR) id[i] = i + 1;
            else                                     id[i] = NO_ISLAND;
        }
        int[][] offs = offsetsFor(policy);

        // Safety cap: the algorithm converges in O(diameter) iterations,
        // bounded above by n itself. Cap at 2n so runaway loops are
        // observable in tests without risking an infinite loop in prod.
        int maxIters = Math.max(64, 2 * n);
        boolean changed = true;
        int iters = 0;
        while (changed && iters++ < maxIters) {
            changed = false;
            // Hook-to-min pass.
            int[] next = id.clone();
            for (int z = 0; z < Lz; z++) {
                for (int y = 0; y < Ly; y++) {
                    for (int x = 0; x < Lx; x++) {
                        int i = flatIdx(x, y, z, Lx, Ly);
                        int myId = next[i];
                        if (myId == NO_ISLAND) continue;
                        int minId = myId;
                        for (int[] off : offs) {
                            int nx = x + off[0], ny = y + off[1], nz = z + off[2];
                            if (nx < 0 || nx >= Lx || ny < 0 || ny >= Ly || nz < 0 || nz >= Lz) continue;
                            int j = flatIdx(nx, ny, nz, Lx, Ly);
                            int nId = next[j];
                            if (nId == NO_ISLAND) continue;
                            if (nId < minId) minId = nId;
                        }
                        if (minId < myId) {
                            next[i] = minId;
                            changed = true;
                        }
                    }
                }
            }
            id = next;

            // Pointer-jumping pass (one jump per iter to match the GPU
            // kernel we plan to write). Skips AIR and guards against an
            // AIR-pointing parent index (can happen if lbl = i+1 names
            // a voxel that was never SOLID/ANCHOR at that flat index,
            // though after the hook pass it will not be reached).
            int[] jumped = id.clone();
            for (int i = 0; i < n; i++) {
                int lbl = id[i];
                if (lbl == NO_ISLAND) continue;
                int parentIdx = lbl - 1;
                if (parentIdx < 0 || parentIdx >= n) continue;
                int parentLbl = id[parentIdx];
                if (parentLbl == NO_ISLAND) continue;
                if (parentLbl < lbl) {
                    jumped[i] = parentLbl;
                    changed = true;
                }
            }
            id = jumped;
        }
        return id;
    }

    // ═════════════════════════════════════════════════════════════════
    //  Helpers
    // ═════════════════════════════════════════════════════════════════

    /** Face / face+edge+corner offsets. 26-conn follows PFSFStencil order (face,edge,corner). */
    private static int[][] offsetsFor(NeighborPolicy policy) {
        if (policy == NeighborPolicy.FACE_6) {
            return FACE_6_OFFSETS;
        }
        return PFSFStencil.NEIGHBOR_OFFSETS;
    }

    private static final int[][] FACE_6_OFFSETS = {
            { 1,  0,  0}, {-1,  0,  0},
            { 0,  1,  0}, { 0, -1,  0},
            { 0,  0,  1}, { 0,  0, -1},
    };

    private static int flatIdx(int x, int y, int z, int Lx, int Ly) {
        return x + Lx * (y + Ly * z);
    }

    /**
     * Convenience: decode a flat SV output into per-component voxel
     * sets, tagging each component anchored (iff at least one member
     * voxel has {@code type == TYPE_ANCHOR}) or orphan. Used by
     * {@code LabelPropagationTest} to compare SV and BFS outputs on
     * the same domain.
     */
    public static FlatPartition partitionFromFlat(int[] islandId, byte[] type, int Lx, int Ly, int Lz) {
        int n = Lx * Ly * Lz;
        Map<Integer, Set<BlockPos>> groups = new HashMap<>();
        Map<Integer, Boolean> anchored = new HashMap<>();
        for (int i = 0; i < n; i++) {
            int lbl = islandId[i];
            if (lbl == NO_ISLAND) continue;
            int x = i % Lx;
            int rem = i / Lx;
            int y = rem % Ly;
            int z = rem / Ly;
            groups.computeIfAbsent(lbl, k -> new HashSet<>()).add(new BlockPos(x, y, z));
            boolean isAnchor = type[i] == TYPE_ANCHOR;
            anchored.merge(lbl, isAnchor, (a, b) -> a || b);
        }
        List<Component> comps = new ArrayList<>(groups.size());
        for (Map.Entry<Integer, Set<BlockPos>> e : groups.entrySet()) {
            comps.add(new Component(e.getValue(), anchored.getOrDefault(e.getKey(), Boolean.FALSE)));
        }
        return new FlatPartition(comps);
    }

    /** Parallel of {@link PartitionResult} for flat-array inputs. */
    public record FlatPartition(List<Component> components) {}
}
