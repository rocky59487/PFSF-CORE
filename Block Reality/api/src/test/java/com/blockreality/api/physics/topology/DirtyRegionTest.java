package com.blockreality.api.physics.topology;

import net.minecraft.core.BlockPos;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the {@code R.3} dirty-region machinery on
 * {@link TopologicalSVDAG}. The contract: every voxel write that
 * materially changes the connectivity summary of one or more enclosing
 * subtrees surfaces those subtrees' AABBs through
 * {@link TopologicalSVDAG#drainDirtyRegions}. Writes that do not
 * change the summary (rare but possible — e.g. writing a redundant
 * SOLID where one already existed) surface nothing.
 */
public class DirtyRegionTest {

    @Test
    @DisplayName("first write marks the containing leaf dirty")
    public void firstWriteDirties() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        t.setVoxel(3, 3, 3, TopologicalSVDAG.TYPE_SOLID);
        List<TopologicalSVDAG.DirtyRegion> dirty = t.peekDirtyRegions();
        assertFalse(dirty.isEmpty(), "expected the leaf that received the write to be dirty");
        TopologicalSVDAG.DirtyRegion leafRegion = dirty.get(0);
        assertEquals(TopologicalSVDAG.LEAF_SIZE, leafRegion.size());
        assertTrue(leafRegion.minX() <= 3 && 3 < leafRegion.maxX());
    }

    @Test
    @DisplayName("redundant write (same type as already present) does NOT mark dirty")
    public void redundantWriteQuiet() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        t.setVoxel(5, 5, 5, TopologicalSVDAG.TYPE_SOLID);
        t.drainDirtyRegions();  // flush the initial write
        t.setVoxel(5, 5, 5, TopologicalSVDAG.TYPE_SOLID); // same type, same place
        assertTrue(t.peekDirtyRegions().isEmpty(),
                "rewriting an identical type must be a no-op for Tier 1");
    }

    @Test
    @DisplayName("drainDirtyRegions empties the state and a second drain is empty")
    public void drainClears() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        t.setVoxel(0, 0, 0, TopologicalSVDAG.TYPE_SOLID);
        Set<TopologicalSVDAG.DirtyRegion> first = t.drainDirtyRegions();
        assertFalse(first.isEmpty());
        Set<TopologicalSVDAG.DirtyRegion> second = t.drainDirtyRegions();
        assertTrue(second.isEmpty(), "expected drained set to be empty on second call");
    }

    @Test
    @DisplayName("removing a bridging voxel splits the root summary from 1 to 2 components")
    public void bridgeSplitUpdatesRootSummary() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        // Two solid clusters connected by a single bridge voxel.
        // Cluster A at (0..3, 0..0, 0..0); bridge at (4, 0, 0); cluster B at (5..8, 0, 0).
        for (int x = 0; x <= 3; x++) t.setVoxel(x, 0, 0, TopologicalSVDAG.TYPE_SOLID);
        t.setVoxel(4, 0, 0, TopologicalSVDAG.TYPE_SOLID); // bridge
        for (int x = 5; x <= 8; x++) t.setVoxel(x, 0, 0, TopologicalSVDAG.TYPE_SOLID);
        t.drainDirtyRegions(); // clear build noise

        // Pre-condition: root summary has a single component.
        ConnectivitySummary beforeCut = t.getRoot().getSummary();
        assertNotNull(beforeCut);
        assertEquals(1, beforeCut.componentCount,
                "expected single bridged component; got " + beforeCut.componentCount);

        // Cut the bridge.
        t.setVoxel(4, 0, 0, TopologicalSVDAG.TYPE_AIR);

        ConnectivitySummary afterCut = t.getRoot().getSummary();
        assertEquals(2, afterCut.componentCount,
                "expected two components after cutting the bridge; got " + afterCut.componentCount);
        assertFalse(t.peekDirtyRegions().isEmpty(),
                "bridge cut must produce at least one dirty region");
    }

    @Test
    @DisplayName("anchor-removal propagates the anchored-bit change up to the root summary")
    public void anchorRemovalReflectedInRoot() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        for (int x = 0; x <= 5; x++) t.setVoxel(x, 0, 0, TopologicalSVDAG.TYPE_SOLID);
        t.setVoxel(0, 0, 0, TopologicalSVDAG.TYPE_ANCHOR);  // mark one end as anchor
        t.drainDirtyRegions();

        ConnectivitySummary before = t.getRoot().getSummary();
        assertEquals(1, before.componentCount);
        assertTrue(before.isComponentAnchored(1), "column should be anchored before removal");

        // Remove the anchor voxel (and its only path to anchoring).
        t.setVoxel(0, 0, 0, TopologicalSVDAG.TYPE_AIR);

        ConnectivitySummary after = t.getRoot().getSummary();
        // Column is now 5 voxels, no anchor in registry (nothing flagged anchor).
        assertEquals(1, after.componentCount);
        assertFalse(after.isComponentAnchored(1),
                "after removing the only anchor the root component must be orphan");
    }

    @Test
    @DisplayName("dirty regions include at least one AABB containing the written voxel")
    public void dirtyCoversWrite() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        t.setVoxel(50, 60, 70, TopologicalSVDAG.TYPE_SOLID);
        t.drainDirtyRegions();

        t.setVoxel(50, 60, 70, TopologicalSVDAG.TYPE_AIR);
        Set<TopologicalSVDAG.DirtyRegion> dirty = t.drainDirtyRegions();
        boolean covered = dirty.stream().anyMatch(r ->
                50 >= r.minX() && 50 < r.maxX()
             && 60 >= r.minY() && 60 < r.maxY()
             && 70 >= r.minZ() && 70 < r.maxZ());
        assertTrue(covered, "no dirty region covers the written voxel: " + dirty);
    }
}
