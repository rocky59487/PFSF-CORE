package com.blockreality.api.physics.pfsf;

import net.minecraft.core.BlockPos;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Equivalence + correctness tests for {@link LabelPropagation}.
 *
 * <p>The class exposes two algorithms that must agree on every legal
 * input: a sparse BFS over {@code Set<BlockPos>} (production path) and
 * a Shiloach–Vishkin propagation over a flat {@code byte[]} (reference
 * for the forthcoming GPU kernel). If they ever diverge, the GPU port
 * cannot be trusted against the CPU oracle, so the invariant is worth
 * guarding with a wide battery of randomised cases.
 */
public class LabelPropagationTest {

    // ─── deterministic random scenarios ─────────────────────────────

    @Test
    @DisplayName("SV ≡ BFS on 100 random FACE_6 voxel domains")
    public void svEquivalentToBfs_face6() {
        runRandomEquivalence(100, LabelPropagation.NeighborPolicy.FACE_6, 20251215L);
    }

    @Test
    @DisplayName("SV ≡ BFS on 100 random FULL_26 voxel domains")
    public void svEquivalentToBfs_full26() {
        runRandomEquivalence(100, LabelPropagation.NeighborPolicy.FULL_26, 20251216L);
    }

    private void runRandomEquivalence(int trials,
                                      LabelPropagation.NeighborPolicy policy,
                                      long seed) {
        Random rng = new Random(seed);
        for (int trial = 0; trial < trials; trial++) {
            int Lx = 4 + rng.nextInt(8); // [4, 11]
            int Ly = 4 + rng.nextInt(8);
            int Lz = 4 + rng.nextInt(8);
            byte[] type = new byte[Lx * Ly * Lz];
            Set<BlockPos> members = new HashSet<>();
            Set<BlockPos> anchors = new HashSet<>();
            for (int z = 0; z < Lz; z++) {
                for (int y = 0; y < Ly; y++) {
                    for (int x = 0; x < Lx; x++) {
                        int i = x + Lx * (y + Ly * z);
                        double r = rng.nextDouble();
                        if (r < 0.55) {
                            type[i] = LabelPropagation.TYPE_SOLID;
                            members.add(new BlockPos(x, y, z));
                        } else if (r < 0.65) {
                            type[i] = LabelPropagation.TYPE_ANCHOR;
                            anchors.add(new BlockPos(x, y, z));
                            // ANCHORs inside the domain also count as members from the
                            // BFS's point of view (they are solid-like).
                            members.add(new BlockPos(x, y, z));
                        } else {
                            type[i] = LabelPropagation.TYPE_AIR;
                        }
                    }
                }
            }

            int[] svIds = LabelPropagation.shiloachVishkin(type, Lx, Ly, Lz, policy);
            LabelPropagation.FlatPartition svPart =
                    LabelPropagation.partitionFromFlat(svIds, type, Lx, Ly, Lz);
            LabelPropagation.PartitionResult bfsPart =
                    LabelPropagation.bfsComponents(members, anchors, policy);

            assertPartitionsEquivalent(
                    "trial=" + trial + " policy=" + policy + " dims=" + Lx + "x" + Ly + "x" + Lz,
                    svPart.components(), bfsPart.components());
        }
    }

    // ─── targeted scenarios ─────────────────────────────────────────

    @Test
    @DisplayName("Empty island => empty partition")
    public void emptyMembers() {
        LabelPropagation.PartitionResult r =
                LabelPropagation.bfsComponents(new HashSet<>(), new HashSet<>(),
                        LabelPropagation.NeighborPolicy.FACE_6);
        assertTrue(r.components().isEmpty());
        assertTrue(r.orphans().isEmpty());
    }

    @Test
    @DisplayName("Single SOLID block with ANCHOR next to it ⇒ 1 anchored component")
    public void singleBlockAdjacentAnchor() {
        Set<BlockPos> members = Set.of(new BlockPos(1, 1, 1));
        Set<BlockPos> anchors = Set.of(new BlockPos(1, 0, 1));
        LabelPropagation.PartitionResult r =
                LabelPropagation.bfsComponents(members, anchors,
                        LabelPropagation.NeighborPolicy.FACE_6);
        assertEquals(1, r.components().size());
        assertTrue(r.components().get(0).anchored(), "Should be anchored via downward face-neighbour");
        assertTrue(r.orphans().isEmpty());
    }

    @Test
    @DisplayName("Two-tower bridge: cut the middle beam → both towers remain anchored, beam fragment is orphan")
    public void twoTowerBridgeSplit() {
        // Two towers at x=0 and x=6, each with a bedrock anchor at (x, 0, 0);
        // connected by a beam y=3 from x=1..5. After removing beam voxels
        // (3, 3, 0) and (4, 3, 0), the surviving tips of the beam at
        // (1, 3, 0), (2, 3, 0) and (5, 3, 0) should remain attached to
        // their respective tower and therefore anchored.
        Set<BlockPos> members = new HashSet<>();
        Set<BlockPos> anchors = new HashSet<>();
        // left tower
        for (int y = 0; y <= 3; y++) members.add(new BlockPos(0, y, 0));
        anchors.add(new BlockPos(0, -1, 0));
        // right tower
        for (int y = 0; y <= 3; y++) members.add(new BlockPos(6, y, 0));
        anchors.add(new BlockPos(6, -1, 0));
        // beam fragments that survive the cut
        members.add(new BlockPos(1, 3, 0));
        members.add(new BlockPos(2, 3, 0));
        members.add(new BlockPos(5, 3, 0));
        // no (3,3,0), no (4,3,0) — beam is cut

        LabelPropagation.PartitionResult r =
                LabelPropagation.bfsComponents(members, anchors,
                        LabelPropagation.NeighborPolicy.FACE_6);
        // Expect exactly two components, both anchored (left tower+stub, right tower+stub)
        assertEquals(2, r.components().size());
        for (LabelPropagation.Component c : r.components()) {
            assertTrue(c.anchored(), "Surviving tower stub must be anchored: " + c);
        }
        assertTrue(r.orphans().isEmpty(), "No orphaned blocks expected");
    }

    @Test
    @DisplayName("Detached floating block: 1 anchored column + 1 orphan cluster")
    public void floatingFragmentIsOrphan() {
        Set<BlockPos> members = new HashSet<>();
        Set<BlockPos> anchors = new HashSet<>();
        // anchored column x=0, y=0..4
        for (int y = 0; y <= 4; y++) members.add(new BlockPos(0, y, 0));
        anchors.add(new BlockPos(0, -1, 0));
        // disconnected fragment high in the air
        members.add(new BlockPos(4, 8, 4));
        members.add(new BlockPos(5, 8, 4));
        members.add(new BlockPos(5, 8, 5));

        LabelPropagation.PartitionResult r =
                LabelPropagation.bfsComponents(members, anchors,
                        LabelPropagation.NeighborPolicy.FACE_6);
        assertEquals(2, r.components().size());
        long anchoredCount = r.components().stream().filter(LabelPropagation.Component::anchored).count();
        long orphanCount   = r.components().stream().filter(c -> !c.anchored()).count();
        assertEquals(1, anchoredCount);
        assertEquals(1, orphanCount);
        assertEquals(3, r.orphans().size());
    }

    @Test
    @DisplayName("SV determinism: same input ⇒ same output across 10 runs")
    public void svIsDeterministic() {
        int Lx = 8, Ly = 8, Lz = 8;
        byte[] type = new byte[Lx * Ly * Lz];
        Random rng = new Random(42);
        for (int i = 0; i < type.length; i++) {
            double r = rng.nextDouble();
            type[i] = r < 0.5 ? LabelPropagation.TYPE_SOLID
                    : r < 0.55 ? LabelPropagation.TYPE_ANCHOR
                    : LabelPropagation.TYPE_AIR;
        }
        int[] first = LabelPropagation.shiloachVishkin(type, Lx, Ly, Lz,
                LabelPropagation.NeighborPolicy.FULL_26);
        for (int run = 1; run < 10; run++) {
            int[] other = LabelPropagation.shiloachVishkin(type, Lx, Ly, Lz,
                    LabelPropagation.NeighborPolicy.FULL_26);
            assertArrayEquals(first, other, "SV must be deterministic under identical input (run " + run + ")");
        }
    }

    @Test
    @DisplayName("SV: AIR voxels are labelled NO_ISLAND; live voxels get a component root label")
    public void svLabelSemantics() {
        int Lx = 2, Ly = 2, Lz = 2;
        byte[] type = new byte[]{
                // z=0
                LabelPropagation.TYPE_ANCHOR, LabelPropagation.TYPE_SOLID,
                LabelPropagation.TYPE_SOLID,  LabelPropagation.TYPE_AIR,
                // z=1
                LabelPropagation.TYPE_AIR,    LabelPropagation.TYPE_AIR,
                LabelPropagation.TYPE_AIR,    LabelPropagation.TYPE_SOLID,
        };
        int[] ids = LabelPropagation.shiloachVishkin(type, Lx, Ly, Lz,
                LabelPropagation.NeighborPolicy.FACE_6);
        // Index 3 (z=0, y=1, x=1) is AIR ⇒ NO_ISLAND
        assertEquals(LabelPropagation.NO_ISLAND, ids[3]);
        // Live voxels get a non-NO_ISLAND label
        assertTrue(ids[0] != LabelPropagation.NO_ISLAND);
        assertTrue(ids[1] != LabelPropagation.NO_ISLAND);
        assertTrue(ids[2] != LabelPropagation.NO_ISLAND);
        assertTrue(ids[7] != LabelPropagation.NO_ISLAND);
        // Indices 0, 1, 2 are face-connected (anchor + two solids) so share a label;
        // Index 7 is isolated so carries its own.
        assertEquals(ids[0], ids[1]);
        assertEquals(ids[0], ids[2]);
        assertTrue(ids[7] != ids[0], "isolated cluster must have a different label");
        // Post-hoc anchor rollup: component of index 0 is anchored (contains TYPE_ANCHOR);
        // component of index 7 is orphan.
        LabelPropagation.FlatPartition part =
                LabelPropagation.partitionFromFlat(ids, type, Lx, Ly, Lz);
        assertEquals(2, part.components().size());
        boolean anchoredSeen = false, orphanSeen = false;
        for (LabelPropagation.Component c : part.components()) {
            if (c.anchored()) anchoredSeen = true; else orphanSeen = true;
        }
        assertTrue(anchoredSeen && orphanSeen, "expected one anchored + one orphan component");
    }

    // ─── helpers ────────────────────────────────────────────────────

    /**
     * Assert two component partitions are equivalent modulo the
     * arbitrary ordering of component lists. Two components match if
     * they cover the same BlockPos set AND share the same anchored flag.
     */
    private static void assertPartitionsEquivalent(String label,
                                                   List<LabelPropagation.Component> a,
                                                   List<LabelPropagation.Component> b) {
        assertEquals(a.size(), b.size(),
                label + " component count mismatch: SV=" + a.size() + " BFS=" + b.size());
        // Build canonical key: sorted BlockPos strings
        Map<String, Boolean> keyedA = toKeyed(a);
        Map<String, Boolean> keyedB = toKeyed(b);
        assertEquals(keyedA.keySet(), keyedB.keySet(),
                label + " component membership differs");
        for (String k : keyedA.keySet()) {
            assertEquals(keyedA.get(k), keyedB.get(k),
                    label + " anchored flag differs for component " + k);
        }
    }

    private static Map<String, Boolean> toKeyed(List<LabelPropagation.Component> comps) {
        Map<String, Boolean> out = new HashMap<>();
        for (LabelPropagation.Component c : comps) {
            List<String> poses = new ArrayList<>();
            for (BlockPos p : c.members()) poses.add(p.getX() + "," + p.getY() + "," + p.getZ());
            poses.sort(String::compareTo);
            out.put(String.join("|", poses), c.anchored());
        }
        return out;
    }
}
