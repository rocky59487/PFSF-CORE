package com.blockreality.api.physics.topology;

import net.minecraft.core.BlockPos;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Verifies the R.8 glue: the orchestrator's orphan sink fires
 * synchronously on the same tick that produces the orphan, and the
 * identity journal records births, orphans, and deaths with the
 * correct tick stamps. In production the sink calls
 * {@code CollapseManager.enqueueCollapse}; in these tests the sink
 * appends to a list.
 */
public class OrphanSinkIntegrationTest {

    @Test
    @DisplayName("orphan sink fires in same tick as fracture")
    public void sinkFiresSameTick() {
        ThreeTierOrchestrator o = new ThreeTierOrchestrator();
        List<ThreeTierOrchestrator.OrphanEvent> received = new ArrayList<>();
        o.setOrphanSink(received::add);

        // Anchored column + overhang
        for (int y = 0; y <= 2; y++) o.setVoxel(0, y, 0, TopologicalSVDAG.TYPE_SOLID);
        o.setVoxel(0, -1, 0, TopologicalSVDAG.TYPE_ANCHOR);
        o.setVoxel(1, 2, 0, TopologicalSVDAG.TYPE_SOLID);
        o.setVoxel(2, 2, 0, TopologicalSVDAG.TYPE_SOLID);
        o.tick();
        received.clear();

        // Cut the overhang's sole support
        o.setVoxel(1, 2, 0, TopologicalSVDAG.TYPE_AIR);
        ThreeTierOrchestrator.TickResult result = o.tick();
        assertEquals(1, received.size(),
                "sink must receive exactly one orphan event, got " + received.size());
        assertEquals(result.orphanEvents().size(), received.size());
        assertTrue(received.get(0).voxels().contains(new BlockPos(2, 2, 0)));
    }

    @Test
    @DisplayName("journal records BIRTH on first appearance and ORPHAN on fracture")
    public void journalRecordsBirthAndOrphan() {
        ThreeTierOrchestrator o = new ThreeTierOrchestrator();
        IslandIdentityJournal j = new IslandIdentityJournal();
        o.setIdentityJournal(j);

        // Create an anchored column
        for (int y = 0; y <= 2; y++) o.setVoxel(0, y, 0, TopologicalSVDAG.TYPE_SOLID);
        o.setVoxel(0, -1, 0, TopologicalSVDAG.TYPE_ANCHOR);
        o.tick();

        long births = j.entries().stream()
                .filter(e -> e.event() == IslandIdentityJournal.EventType.BIRTH).count();
        assertEquals(1, births, "one live component → one birth entry");

        // Orphan: add an isolated cluster, tick, check ORPHAN entry
        o.setVoxel(100, 100, 100, TopologicalSVDAG.TYPE_SOLID);
        o.setVoxel(101, 100, 100, TopologicalSVDAG.TYPE_SOLID);
        o.tick();

        long orphans = j.entries().stream()
                .filter(e -> e.event() == IslandIdentityJournal.EventType.ORPHAN).count();
        assertTrue(orphans >= 1, "isolated cluster without anchor must record ORPHAN, got " + orphans);
    }

    @Test
    @DisplayName("journal records DEATH when a component vanishes")
    public void journalRecordsDeath() {
        ThreeTierOrchestrator o = new ThreeTierOrchestrator();
        IslandIdentityJournal j = new IslandIdentityJournal();
        o.setIdentityJournal(j);

        o.setVoxel(0, 0, 0, TopologicalSVDAG.TYPE_SOLID);
        o.tick();
        o.setVoxel(0, 0, 0, TopologicalSVDAG.TYPE_AIR);
        o.tick();

        long deaths = j.entries().stream()
                .filter(e -> e.event() == IslandIdentityJournal.EventType.DEATH).count();
        assertTrue(deaths >= 1, "fully removed component should record DEATH");
    }

    @Test
    @DisplayName("no sink installed — orchestrator degrades gracefully")
    public void noSinkInstalled() {
        ThreeTierOrchestrator o = new ThreeTierOrchestrator();
        // Deliberately do NOT install a sink.
        o.setVoxel(0, 0, 0, TopologicalSVDAG.TYPE_SOLID);
        // This must not throw even though the single SOLID is orphan.
        ThreeTierOrchestrator.TickResult r = o.tick();
        assertEquals(1, r.orphanEvents().size(),
                "orphan still detected in TickResult even without sink");
    }

    @Test
    @DisplayName("sink exception does not break the tick loop — other orphans still recorded in TickResult")
    public void sinkExceptionIsolated() {
        ThreeTierOrchestrator o = new ThreeTierOrchestrator();
        o.setOrphanSink(e -> { throw new RuntimeException("boom"); });
        o.setVoxel(0, 0, 0, TopologicalSVDAG.TYPE_SOLID);
        o.setVoxel(10, 10, 10, TopologicalSVDAG.TYPE_SOLID);
        ThreeTierOrchestrator.TickResult r = o.tick();
        assertEquals(2, r.orphanEvents().size(),
                "TickResult reports every orphan even if the sink throws for one");
    }
}
