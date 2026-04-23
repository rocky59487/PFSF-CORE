package com.blockreality.api.physics.topology;

import com.blockreality.api.physics.pfsf.LabelPropagation;
import net.minecraft.core.BlockPos;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Correctness tests for the Tier-2 Poisson oracle. The contract:
 * {@code fractureMask[i]} is true ⇔ voxel {@code i} is SOLID and lies
 * in a connected component (26-conn, AIR excluded) that contains no
 * ANCHOR. Ground truth comes from {@link LabelPropagation#bfsComponents},
 * which we already validated against Shiloach-Vishkin.
 */
public class PoissonOracleCPUTest {

    @Test
    @DisplayName("empty region — no voxels, no fractures")
    public void emptyRegion() {
        byte[] type = new byte[4 * 4 * 4];
        PoissonOracleCPU.Result r = PoissonOracleCPU.solve(type, 4, 4, 4);
        assertEquals(64, r.phi().length);
        for (float f : r.phi()) assertEquals(0f, f, 1e-7f);
        for (boolean b : r.fractureMask()) assertFalse(b);
    }

    @Test
    @DisplayName("single anchor — one-voxel component, no fracture")
    public void singleAnchor() {
        byte[] type = new byte[4 * 4 * 4];
        type[0] = TopologicalSVDAG.TYPE_ANCHOR;
        PoissonOracleCPU.Result r = PoissonOracleCPU.solve(type, 4, 4, 4);
        assertEquals(1f, r.phi()[0], 1e-5f);
        for (boolean b : r.fractureMask()) assertFalse(b);
    }

    @Test
    @DisplayName("anchored line — every member converges to φ=1, no fracture")
    public void anchoredLine() {
        int L = 8;
        byte[] type = new byte[L * L * L];
        for (int x = 0; x < L; x++) type[x] = TopologicalSVDAG.TYPE_SOLID;
        type[0] = TopologicalSVDAG.TYPE_ANCHOR;
        PoissonOracleCPU.Result r = PoissonOracleCPU.solve(type, L, L, L);
        for (int x = 0; x < L; x++) {
            assertTrue(r.phi()[x] > 0.99f,
                    "expected φ near 1 at x=" + x + ", got " + r.phi()[x]);
            assertFalse(r.fractureMask()[x]);
        }
    }

    @Test
    @DisplayName("disconnected cluster — orphan voxels converge to φ=0 and are in fractureMask")
    public void disconnectedCluster() {
        int L = 8;
        byte[] type = new byte[L * L * L];
        // Anchored column at y=0..L-1, x=0
        for (int y = 0; y < L; y++) type[idx(0, y, 0, L)] = TopologicalSVDAG.TYPE_SOLID;
        type[idx(0, 0, 0, L)] = TopologicalSVDAG.TYPE_ANCHOR;
        // Disconnected fragment at (L-1, L-1, L-1)
        type[idx(L - 1, L - 1, L - 1, L)] = TopologicalSVDAG.TYPE_SOLID;

        PoissonOracleCPU.Result r = PoissonOracleCPU.solve(type, L, L, L);
        // Column voxels must have φ ≈ 1
        for (int y = 0; y < L; y++) {
            float f = r.phi()[idx(0, y, 0, L)];
            assertTrue(f > 0.99f, "anchored column voxel y=" + y + " has low φ=" + f);
        }
        // Orphan fragment in fractureMask
        assertTrue(r.fractureMask()[idx(L - 1, L - 1, L - 1, L)],
                "orphan fragment should appear in fractureMask");
        assertEquals(0f, r.phi()[idx(L - 1, L - 1, L - 1, L)], 1e-5f);
    }

    @Test
    @DisplayName("bridge cut — removing the bridge flips fracture classification of the severed side")
    public void bridgeCut() {
        int L = 12;
        byte[] type = new byte[L * L * L];
        // Solid line y=5, x=0..9. Anchor at x=0.
        for (int x = 0; x <= 9; x++) type[idx(x, 5, 0, L)] = TopologicalSVDAG.TYPE_SOLID;
        type[idx(0, 5, 0, L)] = TopologicalSVDAG.TYPE_ANCHOR;

        // Before: all solid voxels connected to anchor.
        PoissonOracleCPU.Result before = PoissonOracleCPU.solve(type, L, L, L);
        for (int x = 0; x <= 9; x++) assertFalse(before.fractureMask()[idx(x, 5, 0, L)]);

        // Cut the bridge at x=5.
        type[idx(5, 5, 0, L)] = TopologicalSVDAG.TYPE_AIR;

        PoissonOracleCPU.Result after = PoissonOracleCPU.solve(type, L, L, L);
        // x=0..4 still anchored (x=0 is anchor, x=1..4 reachable via 26-conn)
        for (int x = 0; x <= 4; x++) {
            assertFalse(after.fractureMask()[idx(x, 5, 0, L)],
                    "anchored half should NOT be in fractureMask at x=" + x);
        }
        // x=6..9 now severed
        for (int x = 6; x <= 9; x++) {
            assertTrue(after.fractureMask()[idx(x, 5, 0, L)],
                    "orphan half should be in fractureMask at x=" + x);
        }
    }

    @Test
    @DisplayName("agreement with LabelPropagation.bfsComponents on 20 random domains")
    public void agreementWithBfsComponents() {
        Random rng = new Random(20260130L);
        for (int trial = 0; trial < 20; trial++) {
            int L = 6 + rng.nextInt(4);  // [6, 9]
            byte[] type = new byte[L * L * L];
            Set<BlockPos> members = new HashSet<>();
            Set<BlockPos> anchors = new HashSet<>();
            for (int i = 0; i < type.length; i++) {
                double r = rng.nextDouble();
                if (r < 0.5) {
                    type[i] = TopologicalSVDAG.TYPE_SOLID;
                } else if (r < 0.58) {
                    type[i] = TopologicalSVDAG.TYPE_ANCHOR;
                }
                if (type[i] != TopologicalSVDAG.TYPE_AIR) {
                    int x = i % L, rem = i / L, y = rem % L, z = rem / L;
                    members.add(new BlockPos(x, y, z));
                    if (type[i] == TopologicalSVDAG.TYPE_ANCHOR) anchors.add(new BlockPos(x, y, z));
                }
            }
            PoissonOracleCPU.Result r = PoissonOracleCPU.solve(type, L, L, L);
            LabelPropagation.PartitionResult bfs = LabelPropagation.bfsComponents(
                    members, anchors, LabelPropagation.NeighborPolicy.FULL_26);
            Set<BlockPos> bfsOrphans = bfs.orphans();

            for (int i = 0; i < type.length; i++) {
                int x = i % L, rem = i / L, y = rem % L, z = rem / L;
                boolean maskSaysOrphan = r.fractureMask()[i];
                boolean bfsSaysOrphan = bfsOrphans.contains(new BlockPos(x, y, z));
                assertEquals(bfsSaysOrphan, maskSaysOrphan,
                        "trial " + trial + " voxel (" + x + "," + y + "," + z + ") "
                        + "mask=" + maskSaysOrphan + " bfs=" + bfsSaysOrphan + " type=" + type[i]
                        + " phi=" + r.phi()[i]);
            }
        }
    }

    private static int idx(int x, int y, int z, int L) {
        return x + L * (y + L * z);
    }
}
