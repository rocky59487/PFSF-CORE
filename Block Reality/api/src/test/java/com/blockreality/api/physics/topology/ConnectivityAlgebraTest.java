package com.blockreality.api.physics.topology;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Correctness tests for {@link ConnectivityAlgebra}. The ground truth is
 * a brute-force 26-connected BFS on whatever voxel cube we happen to be
 * checking; the algebra must agree on which voxels share a component
 * and which components are anchored.
 */
public class ConnectivityAlgebraTest {

    private static final int LEAF = TopologicalSVDAG.LEAF_SIZE;

    @Test
    @DisplayName("buildFromLeaf — empty input => 0 components, 0 blocks")
    public void emptyLeaf() {
        byte[] voxels = new byte[LEAF * LEAF * LEAF];
        ConnectivitySummary s = ConnectivityAlgebra.buildFromLeaf(voxels);
        assertEquals(0, s.componentCount);
        assertEquals(0, s.blockCount);
        for (int f = 0; f < 6; f++) {
            for (int i = 0; i < LEAF * LEAF; i++) assertEquals(0, s.faceLabels[f][i]);
        }
    }

    @Test
    @DisplayName("buildFromLeaf — fully solid leaf => 1 component of block count 512")
    public void fullSolidLeaf() {
        byte[] voxels = new byte[LEAF * LEAF * LEAF];
        java.util.Arrays.fill(voxels, TopologicalSVDAG.TYPE_SOLID);
        ConnectivitySummary s = ConnectivityAlgebra.buildFromLeaf(voxels);
        assertEquals(1, s.componentCount);
        assertEquals(LEAF * LEAF * LEAF, s.blockCount);
        // Every face voxel should carry label 1.
        for (int f = 0; f < 6; f++)
            for (int i = 0; i < LEAF * LEAF; i++) assertEquals(1, s.faceLabels[f][i]);
        assertFalse(s.isComponentAnchored(1));
    }

    @Test
    @DisplayName("buildFromLeaf — single anchor voxel => 1 anchored component")
    public void singleAnchor() {
        byte[] voxels = new byte[LEAF * LEAF * LEAF];
        voxels[0] = TopologicalSVDAG.TYPE_ANCHOR;
        ConnectivitySummary s = ConnectivityAlgebra.buildFromLeaf(voxels);
        assertEquals(1, s.componentCount);
        assertEquals(1, s.blockCount);
        assertTrue(s.isComponentAnchored(1));
    }

    @Test
    @DisplayName("buildFromLeaf — two separated clusters => 2 components")
    public void twoClusters() {
        byte[] voxels = new byte[LEAF * LEAF * LEAF];
        // cluster A at (0, 0, 0..1)
        voxels[flat(0, 0, 0)] = TopologicalSVDAG.TYPE_SOLID;
        voxels[flat(0, 0, 1)] = TopologicalSVDAG.TYPE_SOLID;
        // cluster B at (7, 7, 7) — far corner
        voxels[flat(7, 7, 7)] = TopologicalSVDAG.TYPE_ANCHOR;
        ConnectivitySummary s = ConnectivityAlgebra.buildFromLeaf(voxels);
        assertEquals(2, s.componentCount);
        assertEquals(3, s.blockCount);
        // One of the two is anchored.
        int anchored = (s.isComponentAnchored(1) ? 1 : 0) + (s.isComponentAnchored(2) ? 1 : 0);
        assertEquals(1, anchored);
    }

    @Test
    @DisplayName("buildFromLeaf — random — face labels match brute-force BFS on the same leaf")
    public void randomLeafVsBruteBFS() {
        Random rng = new Random(20260120L);
        for (int trial = 0; trial < 30; trial++) {
            byte[] voxels = randomVoxels(LEAF, rng);
            ConnectivitySummary s = ConnectivityAlgebra.buildFromLeaf(voxels);
            int[] brute = bruteForceLabels(voxels, LEAF);
            // Compare face labels: same component iff same brute label for every pair.
            for (int f = 0; f < 6; f++) {
                for (int v = 0; v < LEAF; v++) {
                    for (int u = 0; u < LEAF; u++) {
                        int lbl = s.faceLabels[f][u + LEAF * v];
                        int[] xyz = faceToVoxel(f, u, v, LEAF);
                        int bf = brute[flat(xyz[0], xyz[1], xyz[2])];
                        if (bf == 0) {
                            assertEquals(0, lbl,
                                    "trial " + trial + " face " + f + " (u,v)=(" + u + "," + v + ") expected AIR");
                        } else {
                            assertTrue(lbl > 0, "trial " + trial + " expected live");
                        }
                    }
                }
            }
            assertEquals(countBlocks(voxels), s.blockCount);
        }
    }

    @Test
    @DisplayName("combine — two X-adjacent leaves merge components that straddle the shared face")
    public void combineTwoLeaves() {
        // Left child: solid column at x=LEAF-1 (right face)
        byte[] leftVox = new byte[LEAF * LEAF * LEAF];
        for (int y = 0; y < LEAF; y++)
            for (int z = 0; z < LEAF; z++) leftVox[flat(LEAF - 1, y, z)] = TopologicalSVDAG.TYPE_SOLID;
        ConnectivitySummary left = ConnectivityAlgebra.buildFromLeaf(leftVox);
        // Right child: solid column at x=0 (left face), plus an anchor cell.
        byte[] rightVox = new byte[LEAF * LEAF * LEAF];
        for (int y = 0; y < LEAF; y++)
            for (int z = 0; z < LEAF; z++) rightVox[flat(0, y, z)] = TopologicalSVDAG.TYPE_SOLID;
        rightVox[flat(0, 0, 0)] = TopologicalSVDAG.TYPE_ANCHOR;
        ConnectivitySummary right = ConnectivityAlgebra.buildFromLeaf(rightVox);
        assertEquals(1, left.componentCount);
        assertEquals(1, right.componentCount);
        assertTrue(right.isComponentAnchored(1));

        ConnectivitySummary[] children = new ConnectivitySummary[8];
        children[(0 << 2) | (0 << 1) | 0] = left;   // octant ox=0
        children[(1 << 2) | (0 << 1) | 0] = right;  // octant ox=1
        ConnectivitySummary parent = ConnectivityAlgebra.combine(children, LEAF);

        // Shared face connected ⇒ merge to single component, anchored.
        assertEquals(1, parent.componentCount);
        assertTrue(parent.isComponentAnchored(1));
        assertEquals(left.blockCount + right.blockCount, parent.blockCount);
    }

    @Test
    @DisplayName("combine — two non-touching leaves stay as two components")
    public void combineDisconnectedLeaves() {
        byte[] leftVox = new byte[LEAF * LEAF * LEAF];
        leftVox[flat(0, 0, 0)] = TopologicalSVDAG.TYPE_SOLID;  // interior only
        byte[] rightVox = new byte[LEAF * LEAF * LEAF];
        rightVox[flat(LEAF - 1, LEAF - 1, LEAF - 1)] = TopologicalSVDAG.TYPE_ANCHOR;
        ConnectivitySummary left  = ConnectivityAlgebra.buildFromLeaf(leftVox);
        ConnectivitySummary right = ConnectivityAlgebra.buildFromLeaf(rightVox);
        ConnectivitySummary[] children = new ConnectivitySummary[8];
        children[0] = left;
        children[(1 << 2) | (1 << 1) | 1] = right; // octant (1,1,1) — diagonally opposite
        ConnectivitySummary parent = ConnectivityAlgebra.combine(children, LEAF);
        assertEquals(2, parent.componentCount);
    }

    @Test
    @DisplayName("combine — random 2×2×2 octree, parent face labels ≡ brute-force BFS over merged 16³ cube")
    public void combineRandomVsBruteForce() {
        Random rng = new Random(20260121L);
        int parentSize = LEAF * 2;
        for (int trial = 0; trial < 10; trial++) {
            // Build merged voxel cube
            byte[] merged = new byte[parentSize * parentSize * parentSize];
            for (int i = 0; i < merged.length; i++) {
                double r = rng.nextDouble();
                if (r < 0.45) merged[i] = TopologicalSVDAG.TYPE_SOLID;
                else if (r < 0.5) merged[i] = TopologicalSVDAG.TYPE_ANCHOR;
            }
            // Slice into 8 children
            ConnectivitySummary[] children = new ConnectivitySummary[8];
            for (int oz = 0; oz < 2; oz++)
                for (int oy = 0; oy < 2; oy++)
                    for (int ox = 0; ox < 2; ox++) {
                        byte[] childVox = new byte[LEAF * LEAF * LEAF];
                        for (int z = 0; z < LEAF; z++)
                            for (int y = 0; y < LEAF; y++)
                                for (int x = 0; x < LEAF; x++) {
                                    int mx = ox * LEAF + x;
                                    int my = oy * LEAF + y;
                                    int mz = oz * LEAF + z;
                                    int mi = mx + parentSize * (my + parentSize * mz);
                                    childVox[flat(x, y, z)] = merged[mi];
                                }
                        ConnectivitySummary s = ConnectivityAlgebra.buildFromLeaf(childVox);
                        children[(ox << 2) | (oy << 1) | oz] = s;
                    }

            ConnectivitySummary parent = ConnectivityAlgebra.combine(children, LEAF);
            int[] brute = bruteForceLabels(merged, parentSize);

            // Verify face labels: same brute label ⇔ same parent label, AIR ⇔ AIR.
            for (int f = 0; f < 6; f++) {
                Map<Integer, Integer> parentToBrute = new HashMap<>();
                for (int v = 0; v < parentSize; v++) {
                    for (int u = 0; u < parentSize; u++) {
                        int[] xyz = faceToVoxel(f, u, v, parentSize);
                        int bf = brute[xyz[0] + parentSize * (xyz[1] + parentSize * xyz[2])];
                        int pl = parent.faceLabels[f][u + parentSize * v];
                        if (bf == 0) {
                            assertEquals(0, pl, "trial " + trial + " face " + f + " AIR mismatch");
                        } else {
                            assertTrue(pl > 0);
                            Integer mapped = parentToBrute.putIfAbsent(pl, bf);
                            if (mapped != null) {
                                assertEquals((int) mapped, bf,
                                        "trial " + trial + " label " + pl + " mapped ambiguously");
                            }
                        }
                    }
                }
            }
            assertEquals(countBlocks(merged), parent.blockCount,
                    "trial " + trial + " blockCount mismatch");
        }
    }

    // ═════════════════════════════════════════════════════════════════
    //  Helpers
    // ═════════════════════════════════════════════════════════════════

    private static int flat(int x, int y, int z) { return flat(x, y, z, LEAF); }
    private static int flat(int x, int y, int z, int size) { return x + size * (y + size * z); }

    private static byte[] randomVoxels(int size, Random rng) {
        byte[] voxels = new byte[size * size * size];
        for (int i = 0; i < voxels.length; i++) {
            double r = rng.nextDouble();
            if (r < 0.45) voxels[i] = TopologicalSVDAG.TYPE_SOLID;
            else if (r < 0.5) voxels[i] = TopologicalSVDAG.TYPE_ANCHOR;
        }
        return voxels;
    }

    private static int countBlocks(byte[] voxels) {
        int c = 0;
        for (byte b : voxels) if (b != TopologicalSVDAG.TYPE_AIR) c++;
        return c;
    }

    private static final int[][] N26 = {
            { 1, 0, 0}, {-1, 0, 0}, { 0, 1, 0}, { 0,-1, 0}, { 0, 0, 1}, { 0, 0,-1},
            { 1, 1, 0}, { 1,-1, 0}, {-1, 1, 0}, {-1,-1, 0},
            { 1, 0, 1}, { 1, 0,-1}, {-1, 0, 1}, {-1, 0,-1},
            { 0, 1, 1}, { 0, 1,-1}, { 0,-1, 1}, { 0,-1,-1},
            { 1, 1, 1}, { 1, 1,-1}, { 1,-1, 1}, { 1,-1,-1},
            {-1, 1, 1}, {-1, 1,-1}, {-1,-1, 1}, {-1,-1,-1},
    };

    /**
     * Flood-fill label assignment over {@code voxels} at the given
     * cube side. Returns an int array where AIR voxels are 0 and
     * every other voxel carries a component id in [1, K].
     */
    private static int[] bruteForceLabels(byte[] voxels, int size) {
        int[] lbl = new int[voxels.length];
        int next = 1;
        for (int z = 0; z < size; z++) {
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    int i = x + size * (y + size * z);
                    if (lbl[i] != 0) continue;
                    if (voxels[i] == TopologicalSVDAG.TYPE_AIR) continue;
                    int cid = next++;
                    Deque<int[]> q = new ArrayDeque<>();
                    q.add(new int[]{x, y, z});
                    lbl[i] = cid;
                    while (!q.isEmpty()) {
                        int[] p = q.poll();
                        for (int[] o : N26) {
                            int nx = p[0] + o[0], ny = p[1] + o[1], nz = p[2] + o[2];
                            if (nx < 0 || nx >= size || ny < 0 || ny >= size || nz < 0 || nz >= size) continue;
                            int j = nx + size * (ny + size * nz);
                            if (lbl[j] != 0) continue;
                            if (voxels[j] == TopologicalSVDAG.TYPE_AIR) continue;
                            lbl[j] = cid;
                            q.add(new int[]{nx, ny, nz});
                        }
                    }
                }
            }
        }
        return lbl;
    }

    /**
     * Map a face index + (u, v) to the 3D voxel coordinate on that face.
     * Must match {@link ConnectivitySummary#faceU}/{@link ConnectivitySummary#faceV}.
     */
    private static int[] faceToVoxel(int face, int u, int v, int size) {
        return switch (face) {
            case ConnectivitySummary.FACE_X_MINUS -> new int[]{0, u, v};
            case ConnectivitySummary.FACE_X_PLUS  -> new int[]{size - 1, u, v};
            case ConnectivitySummary.FACE_Y_MINUS -> new int[]{u, 0, v};
            case ConnectivitySummary.FACE_Y_PLUS  -> new int[]{u, size - 1, v};
            case ConnectivitySummary.FACE_Z_MINUS -> new int[]{u, v, 0};
            case ConnectivitySummary.FACE_Z_PLUS  -> new int[]{u, v, size - 1};
            default -> throw new IllegalArgumentException("bad face " + face);
        };
    }
}
