package com.blockreality.api.physics.topology;

import net.minecraft.core.BlockPos;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Invariants the {@code R.1} {@link TopologicalSVDAG} skeleton must satisfy
 * before later passes (summary propagation, dirty regions, Tier 2 Poisson)
 * can trust it.
 */
public class TopologicalSVDAGTest {

    @Test
    @DisplayName("empty tree reports domain size 0 and not initialised")
    public void emptyTree() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        assertEquals(0, t.getDomainSize());
        assertFalse(t.isInitialised());
        assertEquals(TopologicalSVDAG.TYPE_AIR, t.getVoxel(0, 0, 0));
        assertTrue(t.collectLeaves().isEmpty());
    }

    @Test
    @DisplayName("single-voxel write lazily allocates exactly one leaf")
    public void singleVoxelWrite() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        t.setVoxel(5, 10, 17, TopologicalSVDAG.TYPE_SOLID);
        assertTrue(t.isInitialised());
        assertEquals(1, t.collectLeaves().size(), "only one leaf should be allocated");
        assertEquals(TopologicalSVDAG.TYPE_SOLID, t.getVoxel(5, 10, 17));
        // Neighbouring voxels inside the same leaf are still AIR
        assertEquals(TopologicalSVDAG.TYPE_AIR, t.getVoxel(6, 10, 17));
    }

    @Test
    @DisplayName("out-of-domain read returns AIR without growing tree")
    public void outOfDomainRead() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        t.setVoxel(0, 0, 0, TopologicalSVDAG.TYPE_SOLID);
        int leavesBefore = t.collectLeaves().size();
        assertEquals(TopologicalSVDAG.TYPE_AIR, t.getVoxel(9999, 9999, 9999));
        assertEquals(leavesBefore, t.collectLeaves().size(), "read must not allocate");
    }

    @Test
    @DisplayName("writes far apart grow the root and keep both voxels readable")
    public void growRoot() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        t.setVoxel(0, 0, 0, TopologicalSVDAG.TYPE_SOLID);
        // 200 voxels is ≥ 25 leaves away ⇒ must grow root several times
        t.setVoxel(200, 200, 200, TopologicalSVDAG.TYPE_ANCHOR);
        assertTrue(t.getRootLevel() >= 5, "expected ≥ 5 levels above leaf, got " + t.getRootLevel());
        assertEquals(TopologicalSVDAG.TYPE_SOLID, t.getVoxel(0, 0, 0));
        assertEquals(TopologicalSVDAG.TYPE_ANCHOR, t.getVoxel(200, 200, 200));
        assertEquals(2, t.collectLeaves().size());
    }

    @Test
    @DisplayName("negative coordinates round-trip correctly")
    public void negativeCoordinates() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        t.setVoxel(-7, -3, -11, TopologicalSVDAG.TYPE_ANCHOR);
        t.setVoxel(4, 2, 5, TopologicalSVDAG.TYPE_SOLID);
        assertEquals(TopologicalSVDAG.TYPE_ANCHOR, t.getVoxel(-7, -3, -11));
        assertEquals(TopologicalSVDAG.TYPE_SOLID, t.getVoxel(4, 2, 5));
        // Voxels inside the negative leaf stay AIR
        assertEquals(TopologicalSVDAG.TYPE_AIR, t.getVoxel(-8, -4, -12));
    }

    @Test
    @DisplayName("write then clear returns AIR and does not corrupt other voxels")
    public void writeThenClear() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        t.setVoxel(3, 3, 3, TopologicalSVDAG.TYPE_SOLID);
        t.setVoxel(4, 3, 3, TopologicalSVDAG.TYPE_SOLID);
        t.setVoxel(3, 3, 3, TopologicalSVDAG.TYPE_AIR);
        assertEquals(TopologicalSVDAG.TYPE_AIR, t.getVoxel(3, 3, 3));
        assertEquals(TopologicalSVDAG.TYPE_SOLID, t.getVoxel(4, 3, 3));
    }

    @Test
    @DisplayName("sparse voxel set round-trip — 1000 random writes read back")
    public void sparseRoundTrip() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        Random rng = new Random(20260115L);
        Map<BlockPos, Byte> expected = new HashMap<>();
        for (int i = 0; i < 1000; i++) {
            int x = rng.nextInt(512) - 256;
            int y = rng.nextInt(512) - 256;
            int z = rng.nextInt(512) - 256;
            byte type = (byte) (1 + rng.nextInt(2)); // SOLID or ANCHOR
            BlockPos p = new BlockPos(x, y, z);
            t.setVoxel(p, type);
            expected.put(p, type);
        }
        for (Map.Entry<BlockPos, Byte> e : expected.entrySet()) {
            BlockPos p = e.getKey();
            assertEquals(e.getValue().byteValue(), t.getVoxel(p),
                    "mismatch at " + p);
        }
    }

    @Test
    @DisplayName("forEachLeaf visits exactly the allocated leaves")
    public void leafVisitor() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        // Three voxels in three different 8³ leaves
        t.setVoxel(1, 1, 1, TopologicalSVDAG.TYPE_SOLID);  // leaf origin (0, 0, 0)
        t.setVoxel(17, 1, 1, TopologicalSVDAG.TYPE_SOLID); // leaf origin (16, 0, 0)
        t.setVoxel(1, 1, 33, TopologicalSVDAG.TYPE_SOLID); // leaf origin (0, 0, 32)

        Set<String> leafOrigins = new HashSet<>();
        t.forEachLeaf(leaf ->
                leafOrigins.add(leaf.getOriginX() + "," + leaf.getOriginY() + "," + leaf.getOriginZ()));
        assertEquals(3, leafOrigins.size());
        assertTrue(leafOrigins.contains("0,0,0"));
        assertTrue(leafOrigins.contains("16,0,0"));
        assertTrue(leafOrigins.contains("0,0,32"));
    }

    @Test
    @DisplayName("memory efficiency: 1 voxel in 512³ domain uses < 20 leaves")
    public void memoryEfficiency() {
        TopologicalSVDAG t = new TopologicalSVDAG();
        // Force a 512³ domain by writing far-apart voxels
        t.setVoxel(0, 0, 0, TopologicalSVDAG.TYPE_SOLID);
        t.setVoxel(500, 500, 500, TopologicalSVDAG.TYPE_SOLID);
        // Only 2 leaves should hold real voxels; internal nodes don't count
        assertEquals(2, t.collectLeaves().size(),
                "sparse representation must allocate one leaf per live voxel cluster");
    }
}
