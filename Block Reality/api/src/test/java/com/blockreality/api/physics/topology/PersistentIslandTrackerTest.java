package com.blockreality.api.physics.topology;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static com.blockreality.api.physics.topology.PersistentIslandTracker.IslandIdentity;
import static com.blockreality.api.physics.topology.PersistentIslandTracker.encode;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Elder Rule + split/merge invariants for {@link PersistentIslandTracker}.
 */
public class PersistentIslandTrackerTest {

    @Test
    @DisplayName("first tick — every component gets a fresh identity")
    public void firstTick() {
        PersistentIslandTracker t = new PersistentIslandTracker();
        Set<Long> a = setOf(encode(0, 0, 0), encode(1, 0, 0));
        Set<Long> b = setOf(encode(10, 0, 0), encode(11, 0, 0));
        List<IslandIdentity> ids = t.update(List.of(a, b));
        assertEquals(2, ids.size());
        assertNotEquals(ids.get(0).fingerprint(), ids.get(1).fingerprint());
        assertEquals(2, t.liveCount());
    }

    @Test
    @DisplayName("persistence — same component across ticks keeps its identity")
    public void persistence() {
        PersistentIslandTracker t = new PersistentIslandTracker();
        Set<Long> comp = setOf(encode(0, 0, 0), encode(1, 0, 0), encode(2, 0, 0));
        IslandIdentity id1 = t.update(List.of(comp)).get(0);
        IslandIdentity id2 = t.update(List.of(comp)).get(0);
        IslandIdentity id3 = t.update(List.of(comp)).get(0);
        assertEquals(id1.fingerprint(), id2.fingerprint());
        assertEquals(id1.fingerprint(), id3.fingerprint());
        assertEquals(id1.birthTick(), id3.birthTick());
    }

    @Test
    @DisplayName("split — child containing birth voxel inherits; other gets fresh identity")
    public void split() {
        PersistentIslandTracker t = new PersistentIslandTracker();
        // Birth voxel will be (0,0,0) (smallest flat index)
        Set<Long> whole = setOf(encode(0, 0, 0), encode(1, 0, 0), encode(2, 0, 0), encode(3, 0, 0));
        IslandIdentity parent = t.update(List.of(whole)).get(0);

        // Split into {(0,0,0), (1,0,0)} and {(2,0,0), (3,0,0)}
        Set<Long> left  = setOf(encode(0, 0, 0), encode(1, 0, 0));
        Set<Long> right = setOf(encode(2, 0, 0), encode(3, 0, 0));
        List<IslandIdentity> ids = t.update(List.of(left, right));
        assertEquals(2, ids.size());
        // Left contains birth voxel ⇒ inherits.
        assertEquals(parent.fingerprint(), ids.get(0).fingerprint());
        // Right is fresh.
        assertNotEquals(parent.fingerprint(), ids.get(1).fingerprint());
        assertTrue(ids.get(1).birthTick() > parent.birthTick());
    }

    @Test
    @DisplayName("merge — elder component's identity survives; younger closed into persistence diagram")
    public void merge() {
        PersistentIslandTracker t = new PersistentIslandTracker();
        Set<Long> a = setOf(encode(0, 0, 0), encode(1, 0, 0));
        Set<Long> b = setOf(encode(10, 0, 0), encode(11, 0, 0));
        List<IslandIdentity> tick1 = t.update(List.of(a, b));
        IslandIdentity elder   = tick1.get(0);
        IslandIdentity younger = tick1.get(1);
        assertTrue(elder.birthTick() == younger.birthTick(), "both born on same tick");
        // Elder by tie-break (smaller birthVoxel): elder = tick1.get(0) with birthVoxel=(0,0,0)
        assertTrue(elder.birthVoxel() < younger.birthVoxel());

        // Merge: new component contains both prior birth voxels.
        Set<Long> merged = new HashSet<>();
        merged.addAll(a); merged.addAll(b); merged.add(encode(5, 0, 0));
        List<IslandIdentity> tick2 = t.update(List.of(merged));
        assertEquals(1, tick2.size());
        // Winner = elder by birthVoxel tie-break
        assertEquals(elder.fingerprint(), tick2.get(0).fingerprint());
        // Younger closed.
        assertEquals(1, t.getClosedIntervals().size());
        assertEquals(younger.fingerprint(), t.getClosedIntervals().get(0).identity().fingerprint());
        assertEquals(2L, t.getClosedIntervals().get(0).deathTick());
    }

    @Test
    @DisplayName("component disappearance — identity closed in persistence diagram")
    public void disappearance() {
        PersistentIslandTracker t = new PersistentIslandTracker();
        Set<Long> comp = setOf(encode(0, 0, 0), encode(1, 0, 0));
        IslandIdentity id = t.update(List.of(comp)).get(0);
        // All voxels removed next tick
        t.update(List.of());
        assertEquals(0, t.liveCount());
        assertEquals(1, t.getClosedIntervals().size());
        assertEquals(id.fingerprint(), t.getClosedIntervals().get(0).identity().fingerprint());
    }

    @Test
    @DisplayName("determinism — same event sequence yields same fingerprints")
    public void determinism() {
        List<List<Set<Long>>> timeline = randomTimeline(20, 30, new Random(20260210L));
        PersistentIslandTracker first = new PersistentIslandTracker();
        PersistentIslandTracker second = new PersistentIslandTracker();
        List<List<IslandIdentity>> aResults = new ArrayList<>();
        List<List<IslandIdentity>> bResults = new ArrayList<>();
        for (List<Set<Long>> tick : timeline) {
            aResults.add(first.update(tick));
            bResults.add(second.update(tick));
        }
        for (int i = 0; i < aResults.size(); i++) {
            assertEquals(aResults.get(i), bResults.get(i), "tick " + i + " diverged");
        }
    }

    @Test
    @DisplayName("voxel encode/decode round-trip for negative and positive coords")
    public void encodeRoundTrip() {
        int[][] probes = {{0,0,0}, {1,2,3}, {-1,-2,-3}, {100,-50,200}, {-1000,1000,-1000}};
        for (int[] p : probes) {
            long e = encode(p[0], p[1], p[2]);
            assertEquals(p[0], PersistentIslandTracker.decodeX(e));
            assertEquals(p[1], PersistentIslandTracker.decodeY(e));
            assertEquals(p[2], PersistentIslandTracker.decodeZ(e));
        }
    }

    // ─────────────────────────────────────────────────────────────

    private static Set<Long> setOf(long... vals) {
        Set<Long> s = new HashSet<>();
        for (long v : vals) s.add(v);
        return s;
    }

    /** Generate a deterministic multi-tick component timeline for the determinism test. */
    private static List<List<Set<Long>>> randomTimeline(int ticks, int maxVoxels, Random rng) {
        List<List<Set<Long>>> timeline = new ArrayList<>();
        for (int t = 0; t < ticks; t++) {
            int compCount = 1 + rng.nextInt(3);
            List<Set<Long>> comps = new ArrayList<>();
            for (int c = 0; c < compCount; c++) {
                int size = 1 + rng.nextInt(4);
                Set<Long> s = new HashSet<>();
                for (int v = 0; v < size; v++) {
                    int x = c * 100 + rng.nextInt(10);
                    int y = rng.nextInt(10);
                    int z = rng.nextInt(10);
                    s.add(encode(x, y, z));
                }
                if (!s.isEmpty()) comps.add(s);
            }
            timeline.add(comps);
        }
        return timeline;
    }
}
