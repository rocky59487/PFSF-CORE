package com.blockreality.api.physics.topology;

import com.blockreality.api.physics.pfsf.LabelPropagation;
import net.minecraft.core.BlockPos;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Glues the three tiers of the island-subsystem rewrite into one
 * synchronously-testable façade. External event handlers call
 * {@link #setVoxel} on each block change; {@link #tick} advances one
 * tick, draining Tier 1 dirty regions, consulting Tier 2 (the Poisson
 * oracle) to verify the SVDAG's tentative partition, and feeding the
 * result into Tier 3 (persistent island identity). The tick result
 * exposes (a) the orphan events that must trigger collapse and (b)
 * the current live component count.
 *
 * <p>This synchronous composition is the correctness contract the
 * real, asynchronous pipeline (where Tier 2 runs on GPU with a 1–3
 * frame delay) must preserve. R.9's benchmarks will validate that the
 * async path produces the same orphan sequence on the same inputs.
 *
 * <h2>Current scope</h2>
 * R.7 ships the CPU, BFS-based implementation of the tick loop. It
 * reuses {@code LabelPropagation.bfsComponents} as the ground truth
 * for component extraction — which {@code PoissonOracleCPUTest} has
 * already shown matches the PDE-implicit oracle. R.8/R.9 will switch
 * the extraction path to the GPU Poisson simulator behind a feature
 * flag, while keeping this BFS path as the correctness fallback.
 */
public final class ThreeTierOrchestrator {

    /**
     * Result of one {@link #tick} call: the set of orphan events
     * discovered this tick, the current number of live identities,
     * and the tick number assigned by {@link PersistentIslandTracker}.
     */
    public record TickResult(
            List<OrphanEvent> orphanEvents,
            int liveComponents,
            long tick
    ) {}

    /**
     * One orphan component — no voxel is reachable from an anchor.
     * Downstream (R.8) routes this to {@code CollapseManager.enqueueCollapse}.
     */
    public record OrphanEvent(
            PersistentIslandTracker.IslandIdentity identity,
            Set<BlockPos> voxels
    ) {}

    /**
     * One connected component identified this tick together with its
     * Elder-Rule-stable {@link PersistentIslandTracker.IslandIdentity}.
     * Produced by {@link #getLastComponentBindings()} after every
     * {@link #tick()} so host systems (e.g.
     * {@code StructureIslandRegistry}) can reconcile int-id↔fingerprint
     * mapping tables. {@link #anchored} mirrors whatever the BFS
     * partition reported; {@code false} means the component is orphan.
     */
    public record ComponentBinding(
            PersistentIslandTracker.IslandIdentity identity,
            Set<BlockPos> voxels,
            boolean anchored
    ) {}

    /**
     * External receiver for orphan events produced each tick. In
     * production this is wired to {@code CollapseManager.enqueueCollapse};
     * in unit tests it is a plain List-collecting lambda.
     */
    public interface OrphanSink {
        void onOrphan(OrphanEvent event);
    }

    // Non-final so reset() can swap in a fresh tracker/SVDAG on world
    // unload without reflection. Treat as final for read purposes — no
    // external writer touches these fields.
    private TopologicalSVDAG svdag;
    private PersistentIslandTracker tracker;
    /** Mirror of every live voxel for fast component extraction. */
    private final Map<BlockPos, Byte> liveVoxels = new HashMap<>();
    /** Subset of {@link #liveVoxels} whose type is {@code TYPE_ANCHOR}. */
    private final Set<BlockPos> anchors = new HashSet<>();
    private OrphanSink orphanSink;
    private IslandIdentityJournal journal;
    /** Cached identities from the previous tick, keyed by fingerprint. */
    private Map<Long, PersistentIslandTracker.IslandIdentity> prevTickLive = new HashMap<>();
    /**
     * Snapshot of every component identified by the most recent tick,
     * in the order the BFS partition produced them. Populated at the
     * end of {@link #tick()}; empty list on a fresh orchestrator before
     * the first tick runs. Readers must treat the list as read-only.
     */
    private List<ComponentBinding> lastBindings = java.util.Collections.emptyList();

    public ThreeTierOrchestrator() {
        this.svdag   = new TopologicalSVDAG();
        this.tracker = new PersistentIslandTracker();
    }

    /**
     * Drop all voxel state and allocate fresh SVDAG + tracker. Used on
     * world unload and by unit tests; the persistence journal (if
     * attached) is <b>not</b> cleared because callers may want the
     * birth/death record across worlds.
     */
    public synchronized void reset() {
        liveVoxels.clear();
        anchors.clear();
        prevTickLive = new HashMap<>();
        lastBindings = java.util.Collections.emptyList();
        // Replace the SVDAG and tracker wholesale so no half-cleared
        // internal table can leak across worlds. The journal, if any,
        // is left attached — callers decide whether to clear it.
        this.svdag   = new TopologicalSVDAG();
        this.tracker = new PersistentIslandTracker();
    }

    public TopologicalSVDAG getSvdag() { return svdag; }
    public PersistentIslandTracker getTracker() { return tracker; }
    public void setOrphanSink(OrphanSink sink) { this.orphanSink = sink; }
    public void setIdentityJournal(IslandIdentityJournal journal) { this.journal = journal; }
    public IslandIdentityJournal getIdentityJournal() { return journal; }
    /**
     * Read the component snapshot captured at the end of the most
     * recent {@link #tick()}. Used by host systems that need to map
     * int-id islands to Elder-Rule fingerprints for the current frame.
     */
    public List<ComponentBinding> getLastComponentBindings() { return lastBindings; }

    /**
     * Apply a voxel change. Feeds both the SVDAG (for Tier-1 dirty
     * propagation) and the in-memory voxel mirror (for Tier-2 / Tier-3
     * extraction). Safe to call many times per tick.
     */
    public void setVoxel(int x, int y, int z, byte type) {
        svdag.setVoxel(x, y, z, type);
        BlockPos p = new BlockPos(x, y, z);
        if (type == TopologicalSVDAG.TYPE_AIR) {
            liveVoxels.remove(p);
            anchors.remove(p);
        } else {
            liveVoxels.put(p, type);
            if (type == TopologicalSVDAG.TYPE_ANCHOR) anchors.add(p);
            else anchors.remove(p);
        }
    }

    public byte getVoxel(int x, int y, int z) {
        return svdag.getVoxel(x, y, z);
    }

    /**
     * Advance one tick. Idempotent if no voxel changed since the last
     * call (draining the empty dirty set is cheap, but the tracker
     * still receives a fresh partition so its internal tick counter
     * advances and any disappeared components are correctly retired).
     */
    public TickResult tick() {
        // Tier 1 signal: drain pending dirty regions, even if we do not
        // currently exploit them. R.9 will stop doing a full BFS when
        // dirty is empty; kept simple here for correctness.
        svdag.drainDirtyRegions();

        // Extract the current component partition via BFS. The Poisson
        // oracle (R.5) produces an identical orphan set under the same
        // neighbourhood policy (see PoissonOracleCPUTest#agreementWithBfsComponents).
        LabelPropagation.PartitionResult partition = LabelPropagation.bfsComponents(
                new HashSet<>(liveVoxels.keySet()),
                anchors,
                LabelPropagation.NeighborPolicy.FULL_26);

        // Encode each component's voxels into the long-index format
        // expected by Tier 3.
        List<Set<Long>> componentVoxelIndices = new ArrayList<>(partition.components().size());
        for (LabelPropagation.Component c : partition.components()) {
            Set<Long> encoded = new HashSet<>(c.members().size() * 2);
            for (BlockPos p : c.members()) {
                encoded.add(PersistentIslandTracker.encode(p.getX(), p.getY(), p.getZ()));
            }
            componentVoxelIndices.add(encoded);
        }

        List<PersistentIslandTracker.IslandIdentity> ids = tracker.update(componentVoxelIndices);

        // Produce orphan events in the same component order Tier 3 returned them.
        List<OrphanEvent> orphans = new ArrayList<>();
        Map<Long, PersistentIslandTracker.IslandIdentity> newLive = new HashMap<>();
        List<ComponentBinding> bindings = new ArrayList<>(partition.components().size());
        for (int i = 0; i < partition.components().size(); i++) {
            LabelPropagation.Component c = partition.components().get(i);
            PersistentIslandTracker.IslandIdentity id = ids.get(i);
            newLive.put(id.fingerprint(), id);

            Set<BlockPos> members = Set.copyOf(c.members());
            bindings.add(new ComponentBinding(id, members, c.anchored()));

            // Journal births for identities we have never seen before.
            if (journal != null && !prevTickLive.containsKey(id.fingerprint())
                    && id.birthTick() == tracker.currentTick()) {
                journal.recordBirth(tracker.currentTick(), id, c.members().size());
            }

            if (!c.anchored()) {
                OrphanEvent event = new OrphanEvent(id, members);
                orphans.add(event);
                if (journal != null) journal.recordOrphan(tracker.currentTick(), id, event.voxels());
                if (orphanSink != null) {
                    try { orphanSink.onOrphan(event); } catch (Throwable t) { /* swallow to protect loop */ }
                }
            }
        }
        lastBindings = java.util.Collections.unmodifiableList(bindings);

        // Journal deaths: identities that were alive last tick but not this tick.
        if (journal != null) {
            for (PersistentIslandTracker.IslandIdentity died : prevTickLive.values()) {
                if (!newLive.containsKey(died.fingerprint())
                        && tracker.getIdentity(died.fingerprint()) == null) {
                    journal.recordDeath(tracker.currentTick(), died, died.birthBlockCount());
                }
            }
        }
        prevTickLive = newLive;

        return new TickResult(orphans, tracker.liveCount(), tracker.currentTick());
    }
}
