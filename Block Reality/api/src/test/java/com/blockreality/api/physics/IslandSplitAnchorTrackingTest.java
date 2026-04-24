package com.blockreality.api.physics;

import com.blockreality.api.physics.StructureIslandRegistry.OrphanIslandEvent;
import net.minecraft.core.BlockPos;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Regression tests for same-tick orphan classification in the registry path.
 */
public class IslandSplitAnchorTrackingTest {

    private final List<OrphanIslandEvent> received = new ArrayList<>();

    @BeforeEach
    public void setUp() {
        StructureIslandRegistry.resetForTesting();
        received.clear();
        StructureIslandRegistry.setOrphanListener(received::add);
    }

    @AfterEach
    public void tearDown() {
        StructureIslandRegistry.resetForTesting();
    }

    @Test
    @DisplayName("Cutting a two-tower bridge produces no orphan fragments when both sides stay anchored")
    public void bridgeCutBothSidesAnchored() {
        for (int y = 0; y <= 3; y++) StructureIslandRegistry.registerBlock(new BlockPos(0, y, 0), 1L);
        for (int y = 0; y <= 3; y++) StructureIslandRegistry.registerBlock(new BlockPos(6, y, 0), 1L);
        StructureIslandRegistry.registerAnchor(new BlockPos(0, -1, 0));
        StructureIslandRegistry.registerAnchor(new BlockPos(6, -1, 0));
        for (int x = 1; x <= 5; x++) StructureIslandRegistry.registerBlock(new BlockPos(x, 3, 0), 1L);

        int islandId = StructureIslandRegistry.getIslandId(new BlockPos(0, 3, 0));
        assertTrue(islandId > 0);
        assertEquals(islandId, StructureIslandRegistry.getIslandId(new BlockPos(6, 3, 0)));

        StructureIslandRegistry.unregisterBlock(null, new BlockPos(3, 3, 0), 2L);

        assertTrue(received.isEmpty(), "No orphan events expected when both halves retain anchors");
    }

    @Test
    @DisplayName("Removing the sole support block of a fragment fires an orphan event in the same call")
    public void removingSoleSupportFiresOrphanImmediately() {
        for (int y = 0; y <= 3; y++) StructureIslandRegistry.registerBlock(new BlockPos(0, y, 0), 1L);
        StructureIslandRegistry.registerAnchor(new BlockPos(0, -1, 0));
        StructureIslandRegistry.registerBlock(new BlockPos(1, 3, 0), 1L);
        StructureIslandRegistry.registerBlock(new BlockPos(2, 3, 0), 1L);
        StructureIslandRegistry.registerBlock(new BlockPos(3, 3, 0), 1L);

        StructureIslandRegistry.unregisterBlock(null, new BlockPos(0, 3, 0), 2L);

        assertFalse(received.isEmpty(), "Expected at least one orphan event for the overhang");
        Set<BlockPos> orphanBlocks = new HashSet<>();
        for (OrphanIslandEvent e : received) orphanBlocks.addAll(e.members());
        assertTrue(orphanBlocks.contains(new BlockPos(1, 3, 0)));
        assertTrue(orphanBlocks.contains(new BlockPos(2, 3, 0)));
        assertTrue(orphanBlocks.contains(new BlockPos(3, 3, 0)));
        assertFalse(orphanBlocks.contains(new BlockPos(0, 2, 0)));
    }

    @Test
    @DisplayName("Cross-shaped island produces four orphan arms after the center is removed")
    public void crossShapeMultiWayFracture() {
        for (int y = 0; y <= 5; y++) StructureIslandRegistry.registerBlock(new BlockPos(0, y, 0), 1L);
        StructureIslandRegistry.registerAnchor(new BlockPos(0, -1, 0));
        for (int x = 1; x <= 3; x++) {
            StructureIslandRegistry.registerBlock(new BlockPos( x, 5, 0), 1L);
            StructureIslandRegistry.registerBlock(new BlockPos(-x, 5, 0), 1L);
            StructureIslandRegistry.registerBlock(new BlockPos( 0, 5,  x), 1L);
            StructureIslandRegistry.registerBlock(new BlockPos( 0, 5, -x), 1L);
        }

        StructureIslandRegistry.unregisterBlock(null, new BlockPos(0, 5, 0), 2L);

        int totalOrphanBlocks = 0;
        int armsWithExactlyThree = 0;
        for (OrphanIslandEvent e : received) {
            totalOrphanBlocks += e.members().size();
            if (e.members().size() == 3) armsWithExactlyThree++;
        }
        assertEquals(12, totalOrphanBlocks);
        assertEquals(4, armsWithExactlyThree);

        for (OrphanIslandEvent e : received) {
            for (int y = 0; y <= 4; y++) {
                assertFalse(e.members().contains(new BlockPos(0, y, 0)));
            }
        }
    }

    @Test
    @DisplayName("Listener is invoked synchronously in the same call as unregisterBlock")
    public void listenerFiresSynchronouslyInSameCall() {
        AtomicInteger callsBefore = new AtomicInteger();
        AtomicInteger callsAfter = new AtomicInteger();

        StructureIslandRegistry.setOrphanListener(e -> callsAfter.incrementAndGet());

        StructureIslandRegistry.registerBlock(new BlockPos(0, 0, 0), 1L);
        StructureIslandRegistry.registerAnchor(new BlockPos(0, -1, 0));
        StructureIslandRegistry.registerBlock(new BlockPos(1, 0, 0), 1L);
        StructureIslandRegistry.registerBlock(new BlockPos(2, 0, 0), 1L);

        assertEquals(0, callsAfter.get());
        callsBefore.set(callsAfter.get());

        StructureIslandRegistry.unregisterBlock(null, new BlockPos(0, 0, 0), 2L);

        assertEquals(callsBefore.get() + 1, callsAfter.get());
    }

    @Test
    @DisplayName("Direct fracture path reports orphan fragments even when the cut leaves zero anchors")
    public void directFractureStillReportsOrphansWhenAnchorsEmpty() {
        for (int x = 0; x <= 4; x++) StructureIslandRegistry.registerBlock(new BlockPos(x, 0, 0), 1L);

        StructureIslandRegistry.unregisterBlock(null, new BlockPos(2, 0, 0), 2L);

        assertFalse(received.isEmpty(),
                "Direct unregisterBlock path must still surface orphan fragments when no anchors remain");
    }

    @Test
    @DisplayName("Anchor registration still protects the connected half after a cut")
    public void safetyValveLiftsWhenAnchorRegistered() {
        for (int x = 0; x <= 4; x++) StructureIslandRegistry.registerBlock(new BlockPos(x, 0, 0), 1L);
        StructureIslandRegistry.registerAnchor(new BlockPos(0, -1, 0));

        StructureIslandRegistry.unregisterBlock(null, new BlockPos(2, 0, 0), 2L);

        assertFalse(received.isEmpty());
        Set<BlockPos> orphanBlocks = new HashSet<>();
        for (OrphanIslandEvent e : received) orphanBlocks.addAll(e.members());
        assertTrue(orphanBlocks.contains(new BlockPos(3, 0, 0))
                || orphanBlocks.contains(new BlockPos(4, 0, 0)));
        assertFalse(orphanBlocks.contains(new BlockPos(0, 0, 0)));
    }

    @Test
    @DisplayName("External support loss reclassifies the island when the last anchor disappears")
    public void externalSupportLossRefreshesAnchorState() {
        for (int x = 0; x <= 2; x++) {
            StructureIslandRegistry.registerBlock(new BlockPos(x, 0, 0), 1L);
        }
        StructureIslandRegistry.registerAnchor(new BlockPos(0, 0, 0));

        int islandId = StructureIslandRegistry.getIslandId(new BlockPos(1, 0, 0));
        assertTrue(islandId > 0);

        StructureIslandRegistry.unregisterAnchor(new BlockPos(0, 0, 0));
        StructureIslandRegistry.refreshAnchorState(null, islandId, 2L);

        assertFalse(received.isEmpty(),
                "Losing the last external support must orphan the island immediately");
        Set<BlockPos> orphanBlocks = new HashSet<>();
        for (OrphanIslandEvent e : received) orphanBlocks.addAll(e.members());
        assertEquals(Set.of(
                new BlockPos(0, 0, 0),
                new BlockPos(1, 0, 0),
                new BlockPos(2, 0, 0)), orphanBlocks);
    }
}
