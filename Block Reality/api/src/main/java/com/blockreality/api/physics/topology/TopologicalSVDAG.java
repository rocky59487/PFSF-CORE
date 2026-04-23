package com.blockreality.api.physics.topology;

import net.minecraft.core.BlockPos;

import java.util.ArrayList;
import java.util.List;

/**
 * Topologically-augmented Sparse Voxel DAG — Tier 1 of the three-tier
 * island-rewrite architecture. This skeleton ({@code R.1}) implements
 * the tree structure, per-voxel get/set, and leaf iteration; the
 * {@code ConnectivitySummary} / boundary bitmask propagation lands
 * separately in {@code R.2} and the dirty-region queue in {@code R.3}.
 *
 * <h2>Shape</h2>
 * <ul>
 *   <li>Dyadic octree.</li>
 *   <li>Leaf = {@link #LEAF_SIZE}³ voxel cube (default 8³).</li>
 *   <li>Each internal node has up to 8 children, one per octant.</li>
 *   <li>Domain side = {@code LEAF_SIZE · 2^rootLevel}; the root's AABB
 *       contains every voxel the tree will ever hold.</li>
 * </ul>
 *
 * <h2>Voxel type encoding</h2>
 * Same semantics as {@code LabelPropagation}: {@code 0 = AIR},
 * {@code 1 = SOLID}, {@code 2 = ANCHOR}. Tier 2 (PFSF Poisson
 * oracle) and Tier 3 (persistence tracker) will read this array
 * directly.
 *
 * <h2>Sparse memory</h2>
 * AIR-only subtrees are never allocated — a null child means "this
 * octant contains no live voxels". Writing a non-AIR voxel into a null
 * subtree walks down creating leaves lazily. Writing AIR into a leaf
 * may trim that leaf; aggressive trimming is deferred to a future
 * {@code prune()} call to keep this skeleton simple.
 */
public final class TopologicalSVDAG {

    /** Side length of a leaf node in voxels. Must be a power of 2. */
    public static final int LEAF_SIZE = 8;
    /** Voxel count per leaf ({@link #LEAF_SIZE}³). */
    public static final int LEAF_VOXEL_COUNT = LEAF_SIZE * LEAF_SIZE * LEAF_SIZE;

    // Type constants — mirror LabelPropagation for cross-layer parity.
    public static final byte TYPE_AIR    = 0;
    public static final byte TYPE_SOLID  = 1;
    public static final byte TYPE_ANCHOR = 2;

    // ─────────────────────────────────────────────────────────────
    //  Node hierarchy
    // ─────────────────────────────────────────────────────────────

    /**
     * Abstract base. Both internal and leaf nodes store their
     * world-space origin (min corner) and their side length in
     * voxels. Keeping the origin/size explicit on each node makes
     * neighbour lookups across octant boundaries straightforward.
     */
    public static abstract sealed class Node permits InternalNode, LeafNode {
        final int originX, originY, originZ;
        final int size;  // in voxels; size == LEAF_SIZE for leaves

        /**
         * Cached connectivity summary populated by {@code R.3}'s refresh
         * pass. Null until first populated; stale between a voxel write
         * and the matching {@code recomputePath} call. {@code R.3}'s
         * {@link TopologicalSVDAG#setVoxel} keeps it in lock-step.
         */
        ConnectivitySummary summary;

        Node(int ox, int oy, int oz, int size) {
            this.originX = ox; this.originY = oy; this.originZ = oz;
            this.size = size;
        }

        public final int getOriginX() { return originX; }
        public final int getOriginY() { return originY; }
        public final int getOriginZ() { return originZ; }
        public final int getSize()    { return size;    }
        public final ConnectivitySummary getSummary() { return summary; }

        public boolean contains(int x, int y, int z) {
            return x >= originX && x < originX + size
                && y >= originY && y < originY + size
                && z >= originZ && z < originZ + size;
        }
    }

    /**
     * An axis-aligned dirty region surfaced by {@link #drainDirtyRegions}.
     * Represents a subtree whose connectivity summary changed as a
     * result of voxel writes since the last drain. Tier 2 (the Poisson
     * oracle) uses this AABB to decide how much of the voxel field to
     * re-solve.
     */
    public record DirtyRegion(int minX, int minY, int minZ, int size) {
        public int maxX() { return minX + size; }
        public int maxY() { return minY + size; }
        public int maxZ() { return minZ + size; }
    }

    /** Internal octree node. Up to 8 children, indexed by octant bits. */
    public static final class InternalNode extends Node {
        /**
         * Octant children indexed as {@code (ox << 2) | (oy << 1) | oz}
         * where each bit is 0 if the child lies in the lower half of
         * the corresponding axis, 1 otherwise. Slot is {@code null} if
         * the octant currently contains no live voxels.
         */
        final Node[] children = new Node[8];

        InternalNode(int ox, int oy, int oz, int size) {
            super(ox, oy, oz, size);
        }

        public Node getChild(int octant) { return children[octant]; }
        public int childSize() { return size / 2; }
    }

    /** Leaf node. Stores {@link #LEAF_VOXEL_COUNT} voxels in a flat byte array. */
    public static final class LeafNode extends Node {
        /** Flat byte array, index = x + LEAF_SIZE * (y + LEAF_SIZE * z) where (x,y,z) is leaf-local. */
        final byte[] voxels = new byte[LEAF_VOXEL_COUNT];

        LeafNode(int ox, int oy, int oz) {
            super(ox, oy, oz, LEAF_SIZE);
        }

        public byte[] getVoxels() { return voxels; }

        public byte getLocal(int lx, int ly, int lz) {
            return voxels[lx + LEAF_SIZE * (ly + LEAF_SIZE * lz)];
        }

        public void setLocal(int lx, int ly, int lz, byte type) {
            voxels[lx + LEAF_SIZE * (ly + LEAF_SIZE * lz)] = type;
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  Tree state
    // ─────────────────────────────────────────────────────────────

    /** Non-null once {@link #ensureRoot(int, int, int)} has been invoked. */
    private Node root;

    /**
     * Number of octree levels above the leaves. Tree height in voxels
     * is {@code LEAF_SIZE · 2^rootLevel}. A rootLevel of 0 means the
     * tree holds a single leaf (8³ voxels).
     */
    private int rootLevel;

    /** True once any voxel has been written. */
    private boolean initialised = false;

    /**
     * Subtree AABBs whose connectivity summary changed since the last
     * {@link #drainDirtyRegions} call. Using a list means duplicates are
     * possible if the same subtree is touched twice, but each drain
     * deduplicates into a set before returning.
     */
    private final java.util.List<DirtyRegion> pendingDirty = new java.util.ArrayList<>();

    public Node getRoot() { return root; }
    public int getRootLevel() { return rootLevel; }
    public int getDomainSize() { return root == null ? 0 : root.size; }
    public boolean isInitialised() { return initialised; }

    /** Current total domain AABB, inclusive-min / exclusive-max. */
    public int getDomainMinX() { return root == null ? 0 : root.originX; }
    public int getDomainMinY() { return root == null ? 0 : root.originY; }
    public int getDomainMinZ() { return root == null ? 0 : root.originZ; }
    public int getDomainMaxX() { return root == null ? 0 : root.originX + root.size; }
    public int getDomainMaxY() { return root == null ? 0 : root.originY + root.size; }
    public int getDomainMaxZ() { return root == null ? 0 : root.originZ + root.size; }

    // ─────────────────────────────────────────────────────────────
    //  Public API
    // ─────────────────────────────────────────────────────────────

    /**
     * Set the voxel type at world-space {@code (x, y, z)}. The tree
     * grows its root upward as needed to cover the position, and
     * creates leaves lazily on non-AIR writes. AIR writes into a
     * non-existent subtree are no-ops.
     */
    public void setVoxel(int x, int y, int z, byte type) {
        if (!initialised) {
            ensureRoot(x, y, z);
            initialised = true;
        } else {
            growRootToContain(x, y, z);
        }
        if (type == TYPE_AIR) {
            clearVoxel(root, x, y, z);
        } else {
            writeVoxel(root, x, y, z, type);
        }
        // Propagate connectivity metadata from leaf to root. We short-
        // circuit as soon as a node's summary is semantically unchanged,
        // so writes inside a dense block (e.g. filling in a pre-SOLID
        // voxel that the leaf already counted through its neighbours)
        // cost O(1) above the leaf.
        recomputePath(root, x, y, z);
    }

    /**
     * Drain the accumulated set of subtree AABBs whose connectivity
     * summary changed since the previous call. Deduplicates; the
     * returned set is a fresh instance and not aliased to internal state.
     */
    public java.util.Set<DirtyRegion> drainDirtyRegions() {
        java.util.Set<DirtyRegion> out = new java.util.LinkedHashSet<>(pendingDirty);
        pendingDirty.clear();
        return out;
    }

    /**
     * Visible for testing: inspect the current pending set without
     * clearing it. Unit tests use this to assert summary-change
     * behaviour without committing to the drain semantics.
     */
    public java.util.List<DirtyRegion> peekDirtyRegions() {
        return java.util.Collections.unmodifiableList(pendingDirty);
    }

    /**
     * Bottom-up summary refresh for the node containing {@code (x,y,z)}.
     * Returns {@code true} iff this node's summary semantically changed.
     * If not, parent short-circuits (its own summary cannot have
     * changed either because only this child was touched).
     */
    private boolean recomputePath(Node n, int x, int y, int z) {
        if (n instanceof LeafNode leaf) {
            ConnectivitySummary old = leaf.summary;
            ConnectivitySummary fresh = ConnectivityAlgebra.buildFromLeaf(leaf.voxels);
            if (old != null && old.semanticallyEquals(fresh)) return false;
            leaf.summary = fresh;
            pendingDirty.add(new DirtyRegion(leaf.originX, leaf.originY, leaf.originZ, leaf.size));
            return true;
        }
        InternalNode internal = (InternalNode) n;
        int half = internal.childSize();
        int ox = (x >= internal.originX + half) ? 1 : 0;
        int oy = (y >= internal.originY + half) ? 1 : 0;
        int oz = (z >= internal.originZ + half) ? 1 : 0;
        int octant = (ox << 2) | (oy << 1) | oz;
        Node child = internal.children[octant];
        if (child == null) return false;
        boolean childChanged = recomputePath(child, x, y, z);
        if (!childChanged) return false;
        ConnectivitySummary[] childSums = new ConnectivitySummary[8];
        for (int i = 0; i < 8; i++) {
            if (internal.children[i] != null) childSums[i] = internal.children[i].summary;
        }
        ConnectivitySummary old = internal.summary;
        ConnectivitySummary fresh = ConnectivityAlgebra.combine(childSums, half);
        if (old != null && old.semanticallyEquals(fresh)) return false;
        internal.summary = fresh;
        pendingDirty.add(new DirtyRegion(internal.originX, internal.originY, internal.originZ, internal.size));
        return true;
    }

    /**
     * Read the voxel type at world-space {@code (x, y, z)}. Returns
     * {@link #TYPE_AIR} if the voxel is outside the current domain or
     * in a subtree that has not been allocated.
     */
    public byte getVoxel(int x, int y, int z) {
        if (root == null || !root.contains(x, y, z)) return TYPE_AIR;
        return readVoxel(root, x, y, z);
    }

    /** Visit every allocated leaf in depth-first octant order. */
    public void forEachLeaf(java.util.function.Consumer<LeafNode> visitor) {
        if (root != null) visitLeaves(root, visitor);
    }

    /** Collect every allocated leaf into a flat list. Primarily for tests. */
    public List<LeafNode> collectLeaves() {
        List<LeafNode> out = new ArrayList<>();
        forEachLeaf(out::add);
        return out;
    }

    // ─────────────────────────────────────────────────────────────
    //  Internal: tree growth + descent
    // ─────────────────────────────────────────────────────────────

    /**
     * Allocate a root node large enough to contain {@code (x, y, z)}.
     * Starts with a single leaf aligned to the LEAF_SIZE lattice
     * around the target. Caller guarantees this is the first write.
     */
    private void ensureRoot(int x, int y, int z) {
        int leafOX = Math.floorDiv(x, LEAF_SIZE) * LEAF_SIZE;
        int leafOY = Math.floorDiv(y, LEAF_SIZE) * LEAF_SIZE;
        int leafOZ = Math.floorDiv(z, LEAF_SIZE) * LEAF_SIZE;
        root = new LeafNode(leafOX, leafOY, leafOZ);
        rootLevel = 0;
    }

    /**
     * Grow the root upward until it contains {@code (x, y, z)}. Each
     * iteration doubles the root's side length and wraps the current
     * root as one of the new root's octant children.
     *
     * <p>Alignment invariant: every internal node's origin is a
     * multiple of its <b>child</b> size (not its own size — that
     * stricter constraint breaks trees whose extent straddles zero).
     * So per iteration the new parent's origin on each axis is one of
     * two options: keep the old root at this axis's low half
     * ({@code parent.origin = root.origin}) or place it in the high
     * half ({@code parent.origin = root.origin - root.size}). Each
     * axis picks independently based on where the target sits.
     */
    private void growRootToContain(int x, int y, int z) {
        while (!root.contains(x, y, z)) {
            int rootSize = root.size;
            int newSize = rootSize * 2;
            int newOX = (x >= root.originX) ? root.originX : root.originX - rootSize;
            int newOY = (y >= root.originY) ? root.originY : root.originY - rootSize;
            int newOZ = (z >= root.originZ) ? root.originZ : root.originZ - rootSize;
            InternalNode parent = new InternalNode(newOX, newOY, newOZ, newSize);
            int prevOctant = octantFromChildOrigin(parent, root.originX, root.originY, root.originZ);
            parent.children[prevOctant] = root;
            root = parent;
            rootLevel++;
        }
    }

    /**
     * Return the octant index within {@code parent} whose origin
     * corresponds to the child at world coords ({@code cox, coy, coz}).
     */
    private static int octantFromChildOrigin(InternalNode parent, int cox, int coy, int coz) {
        int half = parent.size / 2;
        int ox = (cox >= parent.originX + half) ? 1 : 0;
        int oy = (coy >= parent.originY + half) ? 1 : 0;
        int oz = (coz >= parent.originZ + half) ? 1 : 0;
        return (ox << 2) | (oy << 1) | oz;
    }

    private void writeVoxel(Node node, int x, int y, int z, byte type) {
        if (node instanceof LeafNode leaf) {
            leaf.setLocal(x - leaf.originX, y - leaf.originY, z - leaf.originZ, type);
            return;
        }
        InternalNode internal = (InternalNode) node;
        int half = internal.childSize();
        int ox = (x >= internal.originX + half) ? 1 : 0;
        int oy = (y >= internal.originY + half) ? 1 : 0;
        int oz = (z >= internal.originZ + half) ? 1 : 0;
        int octant = (ox << 2) | (oy << 1) | oz;
        Node child = internal.children[octant];
        if (child == null) {
            int cox = internal.originX + (ox == 1 ? half : 0);
            int coy = internal.originY + (oy == 1 ? half : 0);
            int coz = internal.originZ + (oz == 1 ? half : 0);
            child = (half == LEAF_SIZE)
                    ? new LeafNode(cox, coy, coz)
                    : new InternalNode(cox, coy, coz, half);
            internal.children[octant] = child;
        }
        writeVoxel(child, x, y, z, type);
    }

    private byte readVoxel(Node node, int x, int y, int z) {
        if (node instanceof LeafNode leaf) {
            return leaf.getLocal(x - leaf.originX, y - leaf.originY, z - leaf.originZ);
        }
        InternalNode internal = (InternalNode) node;
        int half = internal.childSize();
        int ox = (x >= internal.originX + half) ? 1 : 0;
        int oy = (y >= internal.originY + half) ? 1 : 0;
        int oz = (z >= internal.originZ + half) ? 1 : 0;
        Node child = internal.children[(ox << 2) | (oy << 1) | oz];
        if (child == null) return TYPE_AIR;
        return readVoxel(child, x, y, z);
    }

    private void clearVoxel(Node node, int x, int y, int z) {
        if (node instanceof LeafNode leaf) {
            leaf.setLocal(x - leaf.originX, y - leaf.originY, z - leaf.originZ, TYPE_AIR);
            return;
        }
        InternalNode internal = (InternalNode) node;
        int half = internal.childSize();
        int ox = (x >= internal.originX + half) ? 1 : 0;
        int oy = (y >= internal.originY + half) ? 1 : 0;
        int oz = (z >= internal.originZ + half) ? 1 : 0;
        Node child = internal.children[(ox << 2) | (oy << 1) | oz];
        if (child == null) return;  // nothing to clear
        clearVoxel(child, x, y, z);
    }

    private void visitLeaves(Node node, java.util.function.Consumer<LeafNode> visitor) {
        if (node instanceof LeafNode leaf) {
            visitor.accept(leaf);
            return;
        }
        InternalNode internal = (InternalNode) node;
        for (Node c : internal.children) {
            if (c != null) visitLeaves(c, visitor);
        }
    }

    // ─────────────────────────────────────────────────────────────
    //  Convenience for tests / future tiers
    // ─────────────────────────────────────────────────────────────

    public byte getVoxel(BlockPos p) { return getVoxel(p.getX(), p.getY(), p.getZ()); }
    public void setVoxel(BlockPos p, byte type) { setVoxel(p.getX(), p.getY(), p.getZ(), type); }
}
