package com.blockreality.api.physics.topology;

import java.util.Arrays;

/**
 * Cached connectivity metadata attached to every {@link TopologicalSVDAG}
 * node. The goal, following the plan's Tier 1 role, is to let a caller
 * answer two questions cheaply after a single destruction event, without
 * re-running global BFS:
 * <ol>
 *   <li>How many connected components does this subtree contain, and
 *       which of them touch an ANCHOR voxel?</li>
 *   <li>For every voxel on this subtree's outer boundary, which
 *       component is it part of? — so that parent nodes can compose
 *       their own summary by matching boundary voxels across shared
 *       child faces.</li>
 * </ol>
 *
 * <h2>Face layout</h2>
 * The subtree is a cube of side {@code size}. Its 6 axis-aligned faces
 * are indexed as: X- = 0, X+ = 1, Y- = 2, Y+ = 3, Z- = 4, Z+ = 5.
 * Each face carries {@code size × size} boundary voxels, flattened into
 * {@link #faceLabels} in face-local (u, v) order (see
 * {@link #faceIndex(int, int, int)}).
 *
 * <h2>Label semantics</h2>
 * {@code faceLabels[f][u + size*v] == 0} ⇔ that boundary voxel is AIR.
 * Any positive value is an opaque component ID valid only inside this
 * subtree; equality of labels across two boundary voxels (in the same
 * summary) is the only way to infer in-subtree connectivity.
 *
 * <h2>Anchor bits</h2>
 * {@link #anchorBits} packs one bit per component, bit {@code k}
 * set ⇔ component ID {@code k} contains at least one ANCHOR voxel in
 * this subtree. Length is {@code ceil((componentCount+1)/64)} longs;
 * entry 0 always encodes bit 0 which represents the AIR sentinel
 * component and is never set.
 */
public final class ConnectivitySummary {

    /** Number of axis-aligned faces on a cube. */
    public static final int FACE_COUNT = 6;
    /** Face index constants, matched by {@link #faceIndex}. */
    public static final int FACE_X_MINUS = 0;
    public static final int FACE_X_PLUS  = 1;
    public static final int FACE_Y_MINUS = 2;
    public static final int FACE_Y_PLUS  = 3;
    public static final int FACE_Z_MINUS = 4;
    public static final int FACE_Z_PLUS  = 5;

    /** Side length of this subtree in voxels. Power of 2 ≥ {@link TopologicalSVDAG#LEAF_SIZE}. */
    public final int size;

    /** Number of live components (SOLID + ANCHOR reachable classes). */
    public int componentCount;

    /** Total SOLID + ANCHOR voxel count inside this subtree. */
    public int blockCount;

    /**
     * Per-face component labels, length {@link #FACE_COUNT}, each inner
     * array length {@code size * size}. Label 0 is the AIR sentinel;
     * positive labels are component IDs in [1, {@link #componentCount}].
     */
    public final int[][] faceLabels;

    /** Bitmask with bit k set iff component id k contains any ANCHOR voxel. */
    public long[] anchorBits;

    public ConnectivitySummary(int size) {
        if (Integer.bitCount(size) != 1 || size < TopologicalSVDAG.LEAF_SIZE) {
            throw new IllegalArgumentException("ConnectivitySummary size must be a power of 2 ≥ LEAF_SIZE, got " + size);
        }
        this.size = size;
        this.faceLabels = new int[FACE_COUNT][size * size];
        this.anchorBits = new long[1];
        this.componentCount = 0;
        this.blockCount = 0;
    }

    public int getLabel(int face, int u, int v) {
        return faceLabels[face][u + size * v];
    }

    public boolean isComponentAnchored(int componentId) {
        if (componentId <= 0) return false;
        int word = componentId >>> 6;
        int bit = componentId & 63;
        return word < anchorBits.length && (anchorBits[word] & (1L << bit)) != 0;
    }

    /**
     * Set bit {@code componentId} in the anchor bitmask, growing the
     * backing array as needed. Callers that build summaries use this
     * to flag components that contain anchor voxels during leaf build
     * or during composition of child summaries.
     */
    public void markAnchored(int componentId) {
        if (componentId <= 0) return;
        int word = componentId >>> 6;
        if (word >= anchorBits.length) {
            anchorBits = Arrays.copyOf(anchorBits, word + 1);
        }
        anchorBits[word] |= (1L << (componentId & 63));
    }

    /**
     * Map a 3D in-subtree voxel coordinate to a face index + (u, v)
     * pair if the voxel sits exactly on one of the subtree's faces.
     * Returns -1 if the voxel is strictly inside. Interior voxels on
     * the intersection of two faces (edges / corners) are reported on
     * their lowest-indexed face; callers that need the full incidence
     * relation must iterate all six faces.
     */
    public int faceIndex(int x, int y, int z) {
        if (x == 0) return FACE_X_MINUS;
        if (x == size - 1) return FACE_X_PLUS;
        if (y == 0) return FACE_Y_MINUS;
        if (y == size - 1) return FACE_Y_PLUS;
        if (z == 0) return FACE_Z_MINUS;
        if (z == size - 1) return FACE_Z_PLUS;
        return -1;
    }

    /**
     * (u, v) local coords on a face, given the face index and the
     * full (x, y, z) in subtree-local coordinates. The chosen axes
     * are:
     * <pre>
     *   FACE_X_± : u = y, v = z
     *   FACE_Y_± : u = x, v = z
     *   FACE_Z_± : u = x, v = y
     * </pre>
     * which keeps the ordering right-handed when viewed from outside
     * the subtree. The convention matters for
     * {@link ConnectivityAlgebra#combine} because it reads face labels
     * pair-wise across child boundaries and must agree with the leaf
     * build path below.
     */
    public int faceU(int face, int x, int y, int z) {
        return switch (face) {
            case FACE_X_MINUS, FACE_X_PLUS -> y;
            case FACE_Y_MINUS, FACE_Y_PLUS -> x;
            case FACE_Z_MINUS, FACE_Z_PLUS -> x;
            default -> throw new IllegalArgumentException("bad face " + face);
        };
    }

    public int faceV(int face, int x, int y, int z) {
        return switch (face) {
            case FACE_X_MINUS, FACE_X_PLUS -> z;
            case FACE_Y_MINUS, FACE_Y_PLUS -> z;
            case FACE_Z_MINUS, FACE_Z_PLUS -> y;
            default -> throw new IllegalArgumentException("bad face " + face);
        };
    }
}
