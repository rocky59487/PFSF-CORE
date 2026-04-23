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
 * Behavioural regression test for the "floating blocks persist for
 * several ticks" bug reported against PFSF-CORE.
 *
 * <p>The bug's mechanism:
 * <ol>
 *   <li>Player destroys a structural block.</li>
 *   <li>{@link StructureIslandRegistry#unregisterBlock} runs, the
 *       island fragments, but the fragments' anchor status is never
 *       examined — the registry has no concept of anchors.</li>
 *   <li>The orphan fragment is left registered as a normal island.
 *       Its members physically hang in mid-air.</li>
 *   <li>Only after several PFSF ticks does the potential field
 *       diverge past {@code PHI_ORPHAN_THRESHOLD}, at which point
 *       {@code failure_scan} finally flags the voxels as orphan and
 *       the CollapseManager kicks in — typically 3–10 ticks late.</li>
 * </ol>
 *
 * <p>The fix wires {@code StructureIslandRegistry} to
 * {@link com.blockreality.api.physics.pfsf.LabelPropagation}: every
 * split now partitions blocks by connectivity AND classifies each
 * component as anchored or orphan in the <b>same</b> call, and notifies
 * {@link StructureIslandRegistry#setOrphanListener an orphan listener}
 * in the <b>same tick</b> as the fracture.
 *
 * <p>Each test below asserts that the orphan listener fires with the
 * correct fragment the instant the supporting block is removed, not
 * several ticks later.
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

    // ─── (1) canonical floating fragment after a beam cut ──────────

    @Test
    @DisplayName("Cutting a two-tower bridge produces exactly zero orphan fragments when both sides stay anchored")
    public void bridgeCutBothSidesAnchored() {
        // Two towers; a beam spans between them at y=3. Cutting the middle
        // of the beam leaves two stubs, each still reachable from its
        // tower's bedrock anchor. No orphan events expected.
        for (int y = 0; y <= 3; y++) StructureIslandRegistry.registerBlock(new BlockPos(0, y, 0), 1L);
        for (int y = 0; y <= 3; y++) StructureIslandRegistry.registerBlock(new BlockPos(6, y, 0), 1L);
        StructureIslandRegistry.registerAnchor(new BlockPos(0, -1, 0));
        StructureIslandRegistry.registerAnchor(new BlockPos(6, -1, 0));
        for (int x = 1; x <= 5; x++) StructureIslandRegistry.registerBlock(new BlockPos(x, 3, 0), 1L);

        // Sanity: everything lives in one island
        int islandId = StructureIslandRegistry.getIslandId(new BlockPos(0, 3, 0));
        assertTrue(islandId > 0);
        assertEquals(islandId, StructureIslandRegistry.getIslandId(new BlockPos(6, 3, 0)),
                "Both towers should start in the same island via the beam");

        // Cut: remove (3,3,0) — beam halved but both stubs still anchored
        StructureIslandRegistry.unregisterBlock(null, new BlockPos(3, 3, 0), 2L);

        assertTrue(received.isEmpty(),
                "No orphan events expected; both halves retain anchor. Got: " + received);
    }

    // ─── (2) cutting the beam's last bridge creates an orphan ─────

    @Test
    @DisplayName("Removing the sole support block of a fragment fires an orphan event in the same call")
    public void removingSoleSupportFiresOrphanImmediately() {
        // Column at x=0 anchored at (0,-1,0). A 3-block overhang at
        // (1,3,0), (2,3,0), (3,3,0) rests on (0,3,0). Removing (0,3,0)
        // leaves the overhang unreachable from any anchor.
        for (int y = 0; y <= 3; y++) StructureIslandRegistry.registerBlock(new BlockPos(0, y, 0), 1L);
        StructureIslandRegistry.registerAnchor(new BlockPos(0, -1, 0));
        StructureIslandRegistry.registerBlock(new BlockPos(1, 3, 0), 1L);
        StructureIslandRegistry.registerBlock(new BlockPos(2, 3, 0), 1L);
        StructureIslandRegistry.registerBlock(new BlockPos(3, 3, 0), 1L);

        StructureIslandRegistry.unregisterBlock(null, new BlockPos(0, 3, 0), 2L);

        assertFalse(received.isEmpty(),
                "Expected at least one orphan event for the overhang");
        Set<BlockPos> orphanBlocks = new HashSet<>();
        for (OrphanIslandEvent e : received) orphanBlocks.addAll(e.members());
        assertTrue(orphanBlocks.contains(new BlockPos(1, 3, 0)));
        assertTrue(orphanBlocks.contains(new BlockPos(2, 3, 0)));
        assertTrue(orphanBlocks.contains(new BlockPos(3, 3, 0)));
        assertFalse(orphanBlocks.contains(new BlockPos(0, 2, 0)),
                "The still-anchored column must not appear in any orphan event");
    }

    // ─── (3) multi-way fracture: one island becomes three ─────────

    @Test
    @DisplayName("Cross-shaped island with 4 arms ⇒ 4 orphan fragments after centre is removed")
    public void crossShapeMultiWayFracture() {
        // '+' shape at y=5, centre at (0,5,0), arms extending ±x and ±z.
        // Only the centre stands on a column anchored at (0,-1,0).
        // Removing (0,5,0) and (0,4,0..) — actually the centre column
        // (0, 0..4, 0) is anchored via (0,-1,0), and (0,5,0) is the
        // cross centre. Remove the whole column at once to isolate
        // 4 arm fragments.
        for (int y = 0; y <= 5; y++) StructureIslandRegistry.registerBlock(new BlockPos(0, y, 0), 1L);
        StructureIslandRegistry.registerAnchor(new BlockPos(0, -1, 0));
        for (int x = 1; x <= 3; x++) {
            StructureIslandRegistry.registerBlock(new BlockPos( x, 5, 0), 1L);
            StructureIslandRegistry.registerBlock(new BlockPos(-x, 5, 0), 1L);
            StructureIslandRegistry.registerBlock(new BlockPos( 0, 5,  x), 1L);
            StructureIslandRegistry.registerBlock(new BlockPos( 0, 5, -x), 1L);
        }

        // Remove the centre of the cross
        StructureIslandRegistry.unregisterBlock(null, new BlockPos(0, 5, 0), 2L);

        // Expect 4 orphan events, one per arm, with 3 blocks each
        int totalOrphanBlocks = 0;
        int armsWithExactlyThree = 0;
        for (OrphanIslandEvent e : received) {
            totalOrphanBlocks += e.members().size();
            if (e.members().size() == 3) armsWithExactlyThree++;
        }
        assertEquals(12, totalOrphanBlocks, "4 arms × 3 blocks = 12 orphan blocks total");
        assertEquals(4, armsWithExactlyThree, "Each arm must be its own 3-block orphan event");

        // The anchored column must NOT appear in any orphan event
        for (OrphanIslandEvent e : received) {
            for (int y = 0; y <= 4; y++) {
                assertFalse(e.members().contains(new BlockPos(0, y, 0)),
                        "Anchored column block " + new BlockPos(0, y, 0) + " leaked into orphan event " + e);
            }
        }
    }

    // ─── (4) orphan event count matches observation, not heuristic ─

    @Test
    @DisplayName("Listener is invoked synchronously in the same call as unregisterBlock, not deferred")
    public void listenerFiresSynchronouslyInSameCall() {
        AtomicInteger callsBefore = new AtomicInteger();
        AtomicInteger callsAfter  = new AtomicInteger();

        StructureIslandRegistry.setOrphanListener(e -> callsAfter.incrementAndGet());

        // Build a floating-after-removal pattern
        StructureIslandRegistry.registerBlock(new BlockPos(0, 0, 0), 1L);
        StructureIslandRegistry.registerAnchor(new BlockPos(0, -1, 0));
        StructureIslandRegistry.registerBlock(new BlockPos(1, 0, 0), 1L);
        StructureIslandRegistry.registerBlock(new BlockPos(2, 0, 0), 1L);

        assertEquals(0, callsAfter.get(), "Listener should not fire on registration");
        callsBefore.set(callsAfter.get());

        // Remove the bridging block; orphan notification MUST happen
        // before this method returns. If the notification were deferred
        // to the next tick, callsAfter would still be zero immediately
        // after unregisterBlock returns.
        StructureIslandRegistry.unregisterBlock(null, new BlockPos(0, 0, 0), 2L);

        assertEquals(callsBefore.get() + 1, callsAfter.get(),
                "Orphan notification must fire synchronously, in the same method call as the fracture — " +
                "this is the key correctness property against the 'floating blocks for several ticks' bug");
    }

    // ─── (5) safety valve: anchorBlocks empty → no orphan flood ────

    @Test
    @DisplayName("Safety valve — with zero registered anchors, fracturing an island produces NO orphan events")
    public void safetyValveSuppressesOrphansWhenAnchorsEmpty() {
        // Deliberately register ZERO anchors. This reproduces the
        // production state flagged by the P0 review: anchor
        // registration is not yet wired to natural-anchor tracking,
        // so anchorBlocks stays empty. Without the safety valve,
        // every fractured component would be classified as orphan
        // and CollapseManager would be flooded on ordinary block
        // breaks.
        for (int x = 0; x <= 4; x++) StructureIslandRegistry.registerBlock(new BlockPos(x, 0, 0), 1L);

        // Cut the bar in the middle — would normally create two
        // orphan fragments (no anchors means no component anchored).
        StructureIslandRegistry.unregisterBlock(null, new BlockPos(2, 0, 0), 2L);

        assertTrue(received.isEmpty(),
                "Safety valve must suppress orphan events when anchorBlocks is empty. "
                + "Got " + received.size() + " events: " + received);
    }

    @Test
    @DisplayName("Safety valve disengages the moment a single anchor is registered")
    public void safetyValveLiftsWhenAnchorRegistered() {
        // Same shape as above, but we register exactly one anchor
        // that belongs to the LEFT half. The right half should then
        // register as orphan when the middle is cut.
        for (int x = 0; x <= 4; x++) StructureIslandRegistry.registerBlock(new BlockPos(x, 0, 0), 1L);
        StructureIslandRegistry.registerAnchor(new BlockPos(0, -1, 0));

        StructureIslandRegistry.unregisterBlock(null, new BlockPos(2, 0, 0), 2L);

        assertFalse(received.isEmpty(),
                "Once anchorBlocks is non-empty, classification must resume and orphans fire");
        Set<BlockPos> orphanBlocks = new HashSet<>();
        for (OrphanIslandEvent e : received) orphanBlocks.addAll(e.members());
        assertTrue(orphanBlocks.contains(new BlockPos(3, 0, 0))
                        || orphanBlocks.contains(new BlockPos(4, 0, 0)),
                "Right half should be reported as orphan; received voxels: " + orphanBlocks);
        assertFalse(orphanBlocks.contains(new BlockPos(0, 0, 0)),
                "Anchored left half must not appear in any orphan event");
    }
}
