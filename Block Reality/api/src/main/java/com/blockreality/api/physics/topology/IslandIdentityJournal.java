package com.blockreality.api.physics.topology;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Append-only event journal for island identity changes produced by
 * {@link PersistentIslandTracker} and routed through
 * {@link ThreeTierOrchestrator}. This is the persistence-diagram
 * counterpart of {@code CollapseJournal}: every birth, death, split
 * and merge gets recorded with its tick, identity, and (for orphan
 * events) the associated voxel set.
 *
 * <p>Purpose for the paper: provides the material for the "identity
 * stability across rewirings" evaluation. Demonstrating that an
 * island's fingerprint survives N fracture events → reappears with
 * the same fingerprint when partially rebuilt is the key result that
 * Union-Find-based systems cannot produce.
 */
public final class IslandIdentityJournal {

    public enum EventType {
        /** A fresh identity was minted at this tick (no ancestor). */
        BIRTH,
        /** An identity disappeared — either orphan-collapsed or fully destroyed. */
        DEATH,
        /** This identity inherited a parent's fingerprint across a split (Elder Rule). */
        SPLIT_INHERIT,
        /** A new identity spawned from the "other half" of a split. */
        SPLIT_FRESH,
        /** This identity absorbed another at a merge (Elder Rule). */
        MERGE_SURVIVOR,
        /** This identity was retired at a merge (younger side). */
        MERGE_RETIRED,
        /** This identity was flagged orphan (emitted as an OrphanEvent). */
        ORPHAN
    }

    public record Entry(
            long tick,
            EventType event,
            PersistentIslandTracker.IslandIdentity identity,
            int voxelCount,
            Long parentFingerprint  // null unless event is SPLIT_* or MERGE_*
    ) {}

    private final List<Entry> entries = new ArrayList<>();
    private final Map<Long, List<Long>> ancestryByFingerprint = new HashMap<>();

    public void record(Entry e) {
        entries.add(e);
        if (e.parentFingerprint() != null) {
            ancestryByFingerprint
                    .computeIfAbsent(e.identity().fingerprint(), k -> new ArrayList<>())
                    .add(e.parentFingerprint());
        }
    }

    public void recordBirth(long tick, PersistentIslandTracker.IslandIdentity id, int voxelCount) {
        record(new Entry(tick, EventType.BIRTH, id, voxelCount, null));
    }

    public void recordDeath(long tick, PersistentIslandTracker.IslandIdentity id, int lastVoxelCount) {
        record(new Entry(tick, EventType.DEATH, id, lastVoxelCount, null));
    }

    public void recordOrphan(long tick, PersistentIslandTracker.IslandIdentity id, Set<?> voxels) {
        record(new Entry(tick, EventType.ORPHAN, id, voxels.size(), null));
    }

    public void recordSplitInherit(long tick,
                                   PersistentIslandTracker.IslandIdentity survivor,
                                   int voxelCount,
                                   long parentFingerprint) {
        record(new Entry(tick, EventType.SPLIT_INHERIT, survivor, voxelCount, parentFingerprint));
    }

    public void recordSplitFresh(long tick,
                                 PersistentIslandTracker.IslandIdentity fresh,
                                 int voxelCount,
                                 long parentFingerprint) {
        record(new Entry(tick, EventType.SPLIT_FRESH, fresh, voxelCount, parentFingerprint));
    }

    public List<Entry> entries() { return Collections.unmodifiableList(entries); }

    /**
     * Recursive ancestry trace: returns every fingerprint this identity
     * descends from (including itself). Useful for proving "this
     * current component is a descendant of the original anchor-rooted
     * island" in the paper's identity-stability experiment.
     */
    public List<Long> ancestry(long fingerprint) {
        List<Long> result = new ArrayList<>();
        result.add(fingerprint);
        List<Long> parents = ancestryByFingerprint.get(fingerprint);
        if (parents != null) {
            for (long p : parents) result.addAll(ancestry(p));
        }
        return result;
    }

    public int size() { return entries.size(); }
}
