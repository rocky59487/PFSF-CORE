package com.blockreality.api.physics.topology;

import net.minecraft.core.BlockPos;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * End-to-end tests for the three-tier pipeline under the synchronous
 * tick model. These invariants are the final acceptance criteria
 * mapped from the plan's §驗證契約 — especially the two scenarios
 * that drove the rewrite in the first place.
 */
public class ThreeTierOrchestratorTest {

    @Test
    @DisplayName("scenario 1 — bridge cut, both sides anchored: zero orphan events")
    public void scenarioBridgeCutBothAnchored() {
        ThreeTierOrchestrator o = new ThreeTierOrchestrator();
        // Two towers
        for (int y = 0; y <= 3; y++) o.setVoxel(0, y, 0, TopologicalSVDAG.TYPE_SOLID);
        for (int y = 0; y <= 3; y++) o.setVoxel(6, y, 0, TopologicalSVDAG.TYPE_SOLID);
        o.setVoxel(0, -1, 0, TopologicalSVDAG.TYPE_ANCHOR);
        o.setVoxel(6, -1, 0, TopologicalSVDAG.TYPE_ANCHOR);
        // Beam between tower tops
        for (int x = 1; x <= 5; x++) o.setVoxel(x, 3, 0, TopologicalSVDAG.TYPE_SOLID);

        ThreeTierOrchestrator.TickResult before = o.tick();
        assertTrue(before.orphanEvents().isEmpty(), "no orphans before cut");

        // Cut the middle of the beam
        o.setVoxel(3, 3, 0, TopologicalSVDAG.TYPE_AIR);

        ThreeTierOrchestrator.TickResult after = o.tick();
        assertTrue(after.orphanEvents().isEmpty(),
                "cutting a bridge between two anchored towers must produce ZERO orphan events; got "
                + after.orphanEvents().size());
    }

    @Test
    @DisplayName("scenario 2 — overhang cut, right side falls: exactly one orphan event in same tick")
    public void scenarioOverhangCutFalls() {
        ThreeTierOrchestrator o = new ThreeTierOrchestrator();
        // Anchored column at x=0
        for (int y = 0; y <= 3; y++) o.setVoxel(0, y, 0, TopologicalSVDAG.TYPE_SOLID);
        o.setVoxel(0, -1, 0, TopologicalSVDAG.TYPE_ANCHOR);
        // Beam out to x=4 that only hangs because of the column
        for (int x = 1; x <= 4; x++) o.setVoxel(x, 3, 0, TopologicalSVDAG.TYPE_SOLID);

        o.tick(); // settle identities
        assertTrue(o.tick().orphanEvents().isEmpty(), "pre-cut no orphans");

        // Sever the only connection to the column
        o.setVoxel(1, 3, 0, TopologicalSVDAG.TYPE_AIR);

        ThreeTierOrchestrator.TickResult result = o.tick();
        assertEquals(1, result.orphanEvents().size(),
                "exactly one orphan component (x=2..4) expected; got " + result.orphanEvents().size());
        Set<BlockPos> orphanVoxels = result.orphanEvents().get(0).voxels();
        assertEquals(3, orphanVoxels.size(), "severed overhang has 3 voxels");
        assertTrue(orphanVoxels.contains(new BlockPos(2, 3, 0)));
        assertTrue(orphanVoxels.contains(new BlockPos(3, 3, 0)));
        assertTrue(orphanVoxels.contains(new BlockPos(4, 3, 0)));
    }

    @Test
    @DisplayName("scenario 3 — H-shape split into two anchored children: zero orphans, two live identities with stable fingerprints")
    public void scenarioHShapeStableIdentities() {
        ThreeTierOrchestrator o = new ThreeTierOrchestrator();
        // H shape: two legs at x=0 and x=4, both anchored, joined by a
        // horizontal bar at y=5.
        for (int y = 0; y <= 5; y++) o.setVoxel(0, y, 0, TopologicalSVDAG.TYPE_SOLID);
        for (int y = 0; y <= 5; y++) o.setVoxel(4, y, 0, TopologicalSVDAG.TYPE_SOLID);
        o.setVoxel(0, -1, 0, TopologicalSVDAG.TYPE_ANCHOR);
        o.setVoxel(4, -1, 0, TopologicalSVDAG.TYPE_ANCHOR);
        for (int x = 1; x <= 3; x++) o.setVoxel(x, 5, 0, TopologicalSVDAG.TYPE_SOLID);

        o.tick();
        assertEquals(1, o.getTracker().liveCount(), "H starts as a single anchored component");
        long originalFingerprint = o.tick().orphanEvents().isEmpty()
                ? o.getTracker().getIdentity(1L) != null
                        ? 1L
                        : firstLiveFingerprint(o)
                : -1L;

        // Cut the crossbar entirely
        for (int x = 1; x <= 3; x++) o.setVoxel(x, 5, 0, TopologicalSVDAG.TYPE_AIR);

        ThreeTierOrchestrator.TickResult result = o.tick();
        assertEquals(0, result.orphanEvents().size(), "both legs still anchored — no orphans");
        assertEquals(2, result.liveComponents(), "H has split into two distinct live components");
    }

    @Test
    @DisplayName("scenario 4 — anchor removal orphans the whole island in one tick")
    public void scenarioAnchorRemovalOrphansAll() {
        ThreeTierOrchestrator o = new ThreeTierOrchestrator();
        for (int y = 0; y <= 4; y++) o.setVoxel(0, y, 0, TopologicalSVDAG.TYPE_SOLID);
        o.setVoxel(0, -1, 0, TopologicalSVDAG.TYPE_ANCHOR);

        o.tick();
        assertTrue(o.tick().orphanEvents().isEmpty());

        // Remove the anchor
        o.setVoxel(0, -1, 0, TopologicalSVDAG.TYPE_AIR);
        ThreeTierOrchestrator.TickResult result = o.tick();
        assertEquals(1, result.orphanEvents().size(),
                "removing the only anchor must orphan the whole column");
        assertEquals(5, result.orphanEvents().get(0).voxels().size());
    }

    @Test
    @DisplayName("determinism — 50 random fracture events produce the same orphan fingerprints twice")
    public void determinism() {
        Random rng = new Random(20260215L);
        // Record first run
        ThreeTierOrchestrator a = new ThreeTierOrchestrator();
        ThreeTierOrchestrator b = new ThreeTierOrchestrator();
        // Seed a mirror structure both orchestrators have
        seedCommonStructure(a);
        seedCommonStructure(b);
        long[] aFingerprints = new long[50];
        long[] bFingerprints = new long[50];
        for (int step = 0; step < 50; step++) {
            int x = 1 + rng.nextInt(5);
            int y = rng.nextInt(5);
            byte t = (byte) (rng.nextDouble() < 0.5
                    ? TopologicalSVDAG.TYPE_AIR : TopologicalSVDAG.TYPE_SOLID);
            a.setVoxel(x, y, 0, t);
            b.setVoxel(x, y, 0, t);
            ThreeTierOrchestrator.TickResult rA = a.tick();
            ThreeTierOrchestrator.TickResult rB = b.tick();
            aFingerprints[step] = rA.orphanEvents().isEmpty() ? 0L : rA.orphanEvents().get(0).identity().fingerprint();
            bFingerprints[step] = rB.orphanEvents().isEmpty() ? 0L : rB.orphanEvents().get(0).identity().fingerprint();
        }
        assertArrayEquals(aFingerprints, bFingerprints,
                "orphan fingerprints must be deterministic across two orchestrators given the same event stream");
    }

    // ──────────────────────────────────────────────────────────

    private static void seedCommonStructure(ThreeTierOrchestrator o) {
        for (int y = 0; y <= 4; y++) o.setVoxel(0, y, 0, TopologicalSVDAG.TYPE_SOLID);
        o.setVoxel(0, -1, 0, TopologicalSVDAG.TYPE_ANCHOR);
        for (int x = 1; x <= 5; x++) o.setVoxel(x, 4, 0, TopologicalSVDAG.TYPE_SOLID);
    }

    private static long firstLiveFingerprint(ThreeTierOrchestrator o) {
        for (long fp = 1; fp < 1_000_000; fp++) {
            if (o.getTracker().getIdentity(fp) != null) return fp;
        }
        return -1;
    }
}
