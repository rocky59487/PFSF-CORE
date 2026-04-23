package com.blockreality.api.physics.topology;

import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;

/**
 * The leaf builder and octree combiner for {@link ConnectivitySummary}.
 *
 * <p>Split from {@code ConnectivitySummary} so the data record stays
 * small and side-effect free. This class owns the two heavy operations
 * that Tier 1 actually calls:
 * <ul>
 *   <li>{@link #buildFromLeaf} — compute a summary from the flat voxel
 *       array of a leaf, via 26-conn BFS bounded to the 8³ cube.</li>
 *   <li>{@link #combine} — fold up to 8 child summaries into a summary
 *       for a parent cube of double the side length, respecting the
 *       face-adjacency relation implied by the octant positions.</li>
 * </ul>
 *
 * <p>Both operations are deterministic, allocation-lean, and validated
 * against a brute-force 3D BFS in {@code ConnectivityAlgebraTest}.
 */
public final class ConnectivityAlgebra {

    /** 26-neighbour offsets used for in-leaf connectivity, matches PFSF stencil. */
    private static final int[][] NEIGHBOR_26 = {
            { 1, 0, 0}, {-1, 0, 0}, { 0, 1, 0}, { 0,-1, 0}, { 0, 0, 1}, { 0, 0,-1},
            { 1, 1, 0}, { 1,-1, 0}, {-1, 1, 0}, {-1,-1, 0},
            { 1, 0, 1}, { 1, 0,-1}, {-1, 0, 1}, {-1, 0,-1},
            { 0, 1, 1}, { 0, 1,-1}, { 0,-1, 1}, { 0,-1,-1},
            { 1, 1, 1}, { 1, 1,-1}, { 1,-1, 1}, { 1,-1,-1},
            {-1, 1, 1}, {-1, 1,-1}, {-1,-1, 1}, {-1,-1,-1},
    };

    private ConnectivityAlgebra() {}

    // ═════════════════════════════════════════════════════════════════
    //  Leaf build
    // ═════════════════════════════════════════════════════════════════

    /**
     * Run a 26-connected BFS on the leaf's voxel array, assigning each
     * SOLID/ANCHOR voxel a local component ID starting at 1 (0 is the
     * AIR sentinel). Returns a summary with populated {@code faceLabels},
     * {@code anchorBits}, {@code componentCount}, and {@code blockCount}.
     */
    public static ConnectivitySummary buildFromLeaf(byte[] voxels) {
        int size = TopologicalSVDAG.LEAF_SIZE;
        int n = size * size * size;
        if (voxels.length != n) {
            throw new IllegalArgumentException("voxels length != LEAF_SIZE^3");
        }
        ConnectivitySummary s = new ConnectivitySummary(size);

        int[] label = new int[n];              // 0 = unassigned; > 0 = component id
        int nextId = 1;

        for (int z = 0; z < size; z++) {
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    int i = flat(x, y, z, size);
                    if (label[i] != 0) continue;
                    byte t = voxels[i];
                    if (t == TopologicalSVDAG.TYPE_AIR) continue;
                    int cid = nextId++;
                    // BFS this component.
                    Deque<int[]> q = new ArrayDeque<>();
                    q.add(new int[]{x, y, z});
                    label[i] = cid;
                    boolean anchored = (t == TopologicalSVDAG.TYPE_ANCHOR);
                    int count = 1;
                    while (!q.isEmpty()) {
                        int[] p = q.poll();
                        for (int[] o : NEIGHBOR_26) {
                            int nx = p[0] + o[0], ny = p[1] + o[1], nz = p[2] + o[2];
                            if (nx < 0 || nx >= size || ny < 0 || ny >= size || nz < 0 || nz >= size) continue;
                            int j = flat(nx, ny, nz, size);
                            if (label[j] != 0) continue;
                            byte tj = voxels[j];
                            if (tj == TopologicalSVDAG.TYPE_AIR) continue;
                            label[j] = cid;
                            count++;
                            if (tj == TopologicalSVDAG.TYPE_ANCHOR) anchored = true;
                            q.add(new int[]{nx, ny, nz});
                        }
                    }
                    if (anchored) s.markAnchored(cid);
                    s.blockCount += count;
                }
            }
        }
        s.componentCount = nextId - 1;

        // Populate face labels from the voxel label array.
        for (int z = 0; z < size; z++) {
            for (int y = 0; y < size; y++) {
                for (int x = 0; x < size; x++) {
                    int lbl = label[flat(x, y, z, size)];
                    if (lbl == 0) continue;
                    writeIfOnFace(s, x, y, z, lbl, ConnectivitySummary.FACE_X_MINUS, x == 0);
                    writeIfOnFace(s, x, y, z, lbl, ConnectivitySummary.FACE_X_PLUS, x == size - 1);
                    writeIfOnFace(s, x, y, z, lbl, ConnectivitySummary.FACE_Y_MINUS, y == 0);
                    writeIfOnFace(s, x, y, z, lbl, ConnectivitySummary.FACE_Y_PLUS, y == size - 1);
                    writeIfOnFace(s, x, y, z, lbl, ConnectivitySummary.FACE_Z_MINUS, z == 0);
                    writeIfOnFace(s, x, y, z, lbl, ConnectivitySummary.FACE_Z_PLUS, z == size - 1);
                }
            }
        }
        return s;
    }

    private static void writeIfOnFace(ConnectivitySummary s, int x, int y, int z, int lbl, int face, boolean onFace) {
        if (!onFace) return;
        int u = s.faceU(face, x, y, z);
        int v = s.faceV(face, x, y, z);
        s.faceLabels[face][u + s.size * v] = lbl;
    }

    // ═════════════════════════════════════════════════════════════════
    //  Internal-node combine
    // ═════════════════════════════════════════════════════════════════

    /**
     * Combine up to 8 child summaries of side {@code childSize} into a
     * parent summary of side {@code 2 * childSize}. Children may be
     * {@code null} — those octants contribute nothing and their slot
     * on the parent's boundary stays AIR (label 0).
     *
     * <p>The algorithm:
     * <ol>
     *   <li>Assign each non-null child a contiguous block of "global
     *       ids" in a union-find the size of (sum of child
     *       componentCount).</li>
     *   <li>For each of the 12 adjacent child octant pairs, union the
     *       global ids wherever two opposing boundary voxels are both
     *       non-AIR.</li>
     *   <li>Compact the union-find roots into parent component ids,
     *       propagate anchor flags, and project the merged labels
     *       onto the parent's 6 outer faces.</li>
     * </ol>
     *
     * @param children  length-8 array indexed by octant
     *                  {@code (ox << 2) | (oy << 1) | oz}; nulls allowed.
     * @param childSize side of each child cube.
     */
    public static ConnectivitySummary combine(ConnectivitySummary[] children, int childSize) {
        if (children.length != 8) throw new IllegalArgumentException("children.length != 8");
        int parentSize = childSize * 2;
        ConnectivitySummary parent = new ConnectivitySummary(parentSize);

        // Step 1: global id allocation per child.
        int[] childOffset = new int[8];
        int totalComps = 0;
        for (int i = 0; i < 8; i++) {
            childOffset[i] = totalComps;
            if (children[i] != null) totalComps += children[i].componentCount;
        }
        if (totalComps == 0) return parent;  // empty parent

        // Step 2: union-find with unions along 12 child-child faces.
        int[] ufParent = new int[totalComps];
        int[] ufRank   = new int[totalComps];
        for (int i = 0; i < totalComps; i++) ufParent[i] = i;

        for (PairSpec pair : ADJACENT_PAIRS) {
            ConnectivitySummary a = children[pair.aOctant];
            ConnectivitySummary b = children[pair.bOctant];
            if (a == null || b == null) continue;
            for (int v = 0; v < childSize; v++) {
                for (int u = 0; u < childSize; u++) {
                    int la = a.getLabel(pair.aFace, u, v);
                    int lb = b.getLabel(pair.bFace, u, v);
                    if (la <= 0 || lb <= 0) continue;
                    int ga = childOffset[pair.aOctant] + (la - 1);
                    int gb = childOffset[pair.bOctant] + (lb - 1);
                    ufUnion(ufParent, ufRank, ga, gb);
                }
            }
        }

        // Step 3: renumber roots → dense parent component ids.
        int[] rootToParentId = new int[totalComps];
        Arrays.fill(rootToParentId, -1);
        int nextParentId = 1;
        for (int i = 0; i < 8; i++) {
            if (children[i] == null) continue;
            for (int j = 1; j <= children[i].componentCount; j++) {
                int global = childOffset[i] + (j - 1);
                int root = ufFind(ufParent, global);
                if (rootToParentId[root] == -1) rootToParentId[root] = nextParentId++;
            }
        }
        parent.componentCount = nextParentId - 1;

        // Anchor flags: a parent component is anchored iff any of its
        // constituent child components was anchored.
        for (int i = 0; i < 8; i++) {
            if (children[i] == null) continue;
            for (int j = 1; j <= children[i].componentCount; j++) {
                if (!children[i].isComponentAnchored(j)) continue;
                int global = childOffset[i] + (j - 1);
                int pid = rootToParentId[ufFind(ufParent, global)];
                parent.markAnchored(pid);
            }
            parent.blockCount += children[i].blockCount;
        }

        // Step 4: project child face labels onto the parent's outer faces.
        for (int face = 0; face < ConnectivitySummary.FACE_COUNT; face++) {
            for (int v = 0; v < parentSize; v++) {
                for (int u = 0; u < parentSize; u++) {
                    int childOctant = parentFaceToChildOctant(face, u, v, childSize);
                    ConnectivitySummary c = children[childOctant];
                    if (c == null) continue;
                    int cu = u % childSize;
                    int cv = v % childSize;
                    int local = c.getLabel(face, cu, cv);
                    if (local == 0) continue;
                    int global = childOffset[childOctant] + (local - 1);
                    int pid = rootToParentId[ufFind(ufParent, global)];
                    parent.faceLabels[face][u + parentSize * v] = pid;
                }
            }
        }

        return parent;
    }

    /**
     * Given a parent-face index and a parent-face-local (u, v), return
     * the octant index of the child that owns that outer boundary voxel.
     */
    private static int parentFaceToChildOctant(int face, int u, int v, int childSize) {
        int ox, oy, oz;
        switch (face) {
            case ConnectivitySummary.FACE_X_MINUS -> {
                ox = 0; oy = (u >= childSize) ? 1 : 0; oz = (v >= childSize) ? 1 : 0;
            }
            case ConnectivitySummary.FACE_X_PLUS -> {
                ox = 1; oy = (u >= childSize) ? 1 : 0; oz = (v >= childSize) ? 1 : 0;
            }
            case ConnectivitySummary.FACE_Y_MINUS -> {
                oy = 0; ox = (u >= childSize) ? 1 : 0; oz = (v >= childSize) ? 1 : 0;
            }
            case ConnectivitySummary.FACE_Y_PLUS -> {
                oy = 1; ox = (u >= childSize) ? 1 : 0; oz = (v >= childSize) ? 1 : 0;
            }
            case ConnectivitySummary.FACE_Z_MINUS -> {
                oz = 0; ox = (u >= childSize) ? 1 : 0; oy = (v >= childSize) ? 1 : 0;
            }
            case ConnectivitySummary.FACE_Z_PLUS -> {
                oz = 1; ox = (u >= childSize) ? 1 : 0; oy = (v >= childSize) ? 1 : 0;
            }
            default -> throw new IllegalArgumentException("bad face " + face);
        }
        return (ox << 2) | (oy << 1) | oz;
    }

    /** Pair of adjacent child octants that share an internal face. */
    private record PairSpec(int aOctant, int bOctant, int aFace, int bFace) {}

    /** The 12 adjacent child-pair specs in a 2×2×2 octree cell. */
    private static final PairSpec[] ADJACENT_PAIRS;
    static {
        PairSpec[] ps = new PairSpec[12];
        int idx = 0;
        // X-axis pairs: ox=0 touches ox=1, A's X+ meets B's X-
        for (int oy = 0; oy < 2; oy++)
            for (int oz = 0; oz < 2; oz++)
                ps[idx++] = new PairSpec((0 << 2) | (oy << 1) | oz, (1 << 2) | (oy << 1) | oz,
                        ConnectivitySummary.FACE_X_PLUS, ConnectivitySummary.FACE_X_MINUS);
        // Y-axis pairs: oy=0 touches oy=1, A's Y+ meets B's Y-
        for (int ox = 0; ox < 2; ox++)
            for (int oz = 0; oz < 2; oz++)
                ps[idx++] = new PairSpec((ox << 2) | (0 << 1) | oz, (ox << 2) | (1 << 1) | oz,
                        ConnectivitySummary.FACE_Y_PLUS, ConnectivitySummary.FACE_Y_MINUS);
        // Z-axis pairs: oz=0 touches oz=1, A's Z+ meets B's Z-
        for (int ox = 0; ox < 2; ox++)
            for (int oy = 0; oy < 2; oy++)
                ps[idx++] = new PairSpec((ox << 2) | (oy << 1) | 0, (ox << 2) | (oy << 1) | 1,
                        ConnectivitySummary.FACE_Z_PLUS, ConnectivitySummary.FACE_Z_MINUS);
        ADJACENT_PAIRS = ps;
    }

    // ─────────────────────────────────────────────────────────────────
    //  Small helpers
    // ─────────────────────────────────────────────────────────────────

    private static int flat(int x, int y, int z, int size) { return x + size * (y + size * z); }

    private static int ufFind(int[] p, int i) {
        while (p[i] != i) { p[i] = p[p[i]]; i = p[i]; }
        return i;
    }

    private static void ufUnion(int[] p, int[] r, int a, int b) {
        int ra = ufFind(p, a), rb = ufFind(p, b);
        if (ra == rb) return;
        if (r[ra] < r[rb]) { p[ra] = rb; }
        else if (r[ra] > r[rb]) { p[rb] = ra; }
        else { p[rb] = ra; r[ra]++; }
    }
}
