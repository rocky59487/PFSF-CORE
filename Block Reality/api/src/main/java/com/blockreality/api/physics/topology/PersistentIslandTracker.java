package com.blockreality.api.physics.topology;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Tier-3 persistent-homology island identity tracker.
 *
 * <p>Maintains a stable "name" for every connected component (island)
 * that survives topology changes across ticks. When an island splits
 * into two children, the Elder Rule awards the parent's identity to
 * whichever child still contains the parent's <i>birth voxel</i>; the
 * other child gets a fresh identity. When two islands merge, the elder
 * (smaller birth tick, ties broken by birth voxel index) survives; the
 * younger is closed and its persistence interval (birth, death) is
 * committed to the {@link #closedIntervals} diagram.
 *
 * <p>This is the pared-down, GPU-friendly specialization of
 * Edelsbrunner–Harer 0-dimensional persistence for voxel fracture:
 * voxels act as simplices, components are H₀ classes, and the
 * filtration is the monotonically advancing tick timeline. The
 * "fingerprint" field is a stable opaque long per component that
 * downstream systems (CollapseJournal, StructureIslandRegistry) use
 * to refer to the same island across many ticks.
 *
 * <p>Design note. Each voxel is addressed by a packed 64-bit flat index
 * (x, y, z → long). The tracker is deterministic: given the same input
 * sequence of component sets, it produces the same fingerprints in the
 * same order. No RNG, no hash collisions possible.
 */
public final class PersistentIslandTracker {

    /**
     * Unique opaque handle for one live component. {@code fingerprint}
     * is assigned at birth and never reused. {@code birthVoxel} is the
     * smallest flat-index voxel in the component at its creation tick;
     * it is the anchor used by the Elder Rule to decide which child
     * inherits during a split.
     */
    public record IslandIdentity(
            long fingerprint,
            long birthTick,
            long birthVoxel,
            int  birthBlockCount
    ) {}

    /**
     * One closed persistence bar: born at {@code birthTick}, died at
     * {@code deathTick}, identity preserved in {@code identity}.
     * Produced whenever a split creates an orphan child or a merge
     * retires a younger identity.
     */
    public record PersistenceInterval(
            IslandIdentity identity,
            long deathTick
    ) {
        public long birthTick()   { return identity.birthTick(); }
        public int  lifetime()    { return (int)(deathTick - identity.birthTick()); }
    }

    private final Map<Long, IslandIdentity> liveByFingerprint = new HashMap<>();
    private final Map<Long, IslandIdentity> liveByBirthVoxel  = new HashMap<>();
    private final List<PersistenceInterval> closedIntervals   = new ArrayList<>();
    private long nextFingerprint = 1L;
    private long currentTick = 0L;

    public long currentTick() { return currentTick; }
    public int  liveCount()   { return liveByFingerprint.size(); }
    public List<PersistenceInterval> getClosedIntervals() { return Collections.unmodifiableList(closedIntervals); }
    public IslandIdentity getIdentity(long fingerprint) { return liveByFingerprint.get(fingerprint); }

    /**
     * Advance the tracker by one tick. {@code newComponents} is the
     * complete partition of currently live voxels into connected
     * components — each entry is the set of packed flat-index voxels
     * belonging to one component.
     *
     * <p>Returns a mapping from the input component index to the
     * IslandIdentity that represents it after the tick. The returned
     * list has the same length as {@code newComponents}.
     */
    public List<IslandIdentity> update(List<Set<Long>> newComponents) {
        currentTick++;

        List<IslandIdentity> assigned = new ArrayList<>(newComponents.size());
        Set<Long> survivingFingerprints = new HashSet<>();

        for (Set<Long> comp : newComponents) {
            // Find every live identity whose birth voxel is in this component.
            List<IslandIdentity> candidates = new ArrayList<>();
            for (long v : comp) {
                IslandIdentity id = liveByBirthVoxel.get(v);
                if (id != null) candidates.add(id);
            }

            IslandIdentity winner;
            if (candidates.isEmpty()) {
                // Fresh component — allocate a new identity.
                long birthVoxel = Collections.min(comp);
                winner = new IslandIdentity(nextFingerprint++, currentTick, birthVoxel, comp.size());
                liveByFingerprint.put(winner.fingerprint(), winner);
                liveByBirthVoxel.put(birthVoxel, winner);
            } else {
                // Elder Rule: lowest birthTick, ties broken by birthVoxel.
                winner = candidates.stream()
                        .min(Comparator.<IslandIdentity>comparingLong(IslandIdentity::birthTick)
                                .thenComparingLong(IslandIdentity::birthVoxel))
                        .orElseThrow();
                // All other candidates are merged in → their identities die.
                for (IslandIdentity loser : candidates) {
                    if (loser.fingerprint() != winner.fingerprint()) {
                        retireIdentity(loser);
                    }
                }
            }
            assigned.add(winner);
            survivingFingerprints.add(winner.fingerprint());
        }

        // Any previously live identity not picked up by any new component dies now.
        List<IslandIdentity> toRetire = new ArrayList<>();
        for (IslandIdentity id : liveByFingerprint.values()) {
            if (!survivingFingerprints.contains(id.fingerprint())) toRetire.add(id);
        }
        for (IslandIdentity id : toRetire) retireIdentity(id);

        return assigned;
    }

    private void retireIdentity(IslandIdentity id) {
        liveByFingerprint.remove(id.fingerprint());
        liveByBirthVoxel.remove(id.birthVoxel());
        closedIntervals.add(new PersistenceInterval(id, currentTick));
    }

    /**
     * Encode a 3D voxel coordinate into a packed long flat index.
     * 21 bits per axis (supports ±1M) leaves 1 sign bit unused.
     * The encoding must be stable across runs; downstream systems
     * that serialise IslandIdentity to disk rely on it.
     */
    public static long encode(int x, int y, int z) {
        long mx = x & 0x1FFFFF;
        long my = y & 0x1FFFFF;
        long mz = z & 0x1FFFFF;
        return (mx << 42) | (my << 21) | mz;
    }

    public static int decodeX(long e) { int v = (int)((e >> 42) & 0x1FFFFF); return (v & 0x100000) != 0 ? v | ~0x1FFFFF : v; }
    public static int decodeY(long e) { int v = (int)((e >> 21) & 0x1FFFFF); return (v & 0x100000) != 0 ? v | ~0x1FFFFF : v; }
    public static int decodeZ(long e) { int v = (int)(e & 0x1FFFFF);         return (v & 0x100000) != 0 ? v | ~0x1FFFFF : v; }
}
