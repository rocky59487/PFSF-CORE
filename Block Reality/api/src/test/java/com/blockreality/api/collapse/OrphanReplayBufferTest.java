package com.blockreality.api.collapse;

import net.minecraft.core.BlockPos;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Behaviour of the FIFO buffer that backs
 * {@code CollapseManager.onOrphanIsland}'s null-level path.
 *
 * <p>The P1 Codex review flagged that without a buffer, every orphan
 * event emitted from {@code flushDestructions} — which does not know
 * its {@link net.minecraft.server.level.ServerLevel} — was silently
 * dropped. These tests pin the new semantics: the event is stored,
 * survives until drained, and preserves its members verbatim.
 */
public class OrphanReplayBufferTest {

    @BeforeEach
    public void setUp() {
        OrphanReplayBuffer.clear();
    }

    @AfterEach
    public void tearDown() {
        OrphanReplayBuffer.clear();
    }

    @Test
    @DisplayName("empty buffer — size 0, drain returns empty list")
    public void initialEmpty() {
        assertEquals(0, OrphanReplayBuffer.size());
        assertTrue(OrphanReplayBuffer.drain().isEmpty());
    }

    @Test
    @DisplayName("add then drain returns the same event in FIFO order")
    public void addAndDrainPreservesFifo() {
        OrphanReplayBuffer.add(7, Set.of(new BlockPos(1, 2, 3)));
        OrphanReplayBuffer.add(8, Set.of(new BlockPos(4, 5, 6), new BlockPos(7, 8, 9)));
        assertEquals(2, OrphanReplayBuffer.size());

        List<OrphanReplayBuffer.PendingOrphan> drained = OrphanReplayBuffer.drain();
        assertEquals(2, drained.size());
        assertEquals(7, drained.get(0).islandId());
        assertEquals(1, drained.get(0).members().size());
        assertTrue(drained.get(0).members().contains(new BlockPos(1, 2, 3)));
        assertEquals(8, drained.get(1).islandId());
        assertEquals(2, drained.get(1).members().size());
    }

    @Test
    @DisplayName("drain empties the buffer — subsequent size is 0")
    public void drainEmpties() {
        OrphanReplayBuffer.add(1, Set.of(new BlockPos(0, 0, 0)));
        OrphanReplayBuffer.add(2, Set.of(new BlockPos(1, 1, 1)));
        OrphanReplayBuffer.drain();
        assertEquals(0, OrphanReplayBuffer.size());
        assertTrue(OrphanReplayBuffer.drain().isEmpty());
    }

    @Test
    @DisplayName("add rejects empty and null member sets (no buffer growth)")
    public void addIgnoresEmpty() {
        OrphanReplayBuffer.add(1, Set.of());
        OrphanReplayBuffer.add(2, null);
        assertEquals(0, OrphanReplayBuffer.size());
    }

    @Test
    @DisplayName("stored member set is immutable — external mutation of the input does not corrupt the buffered copy")
    public void defensiveCopy() {
        java.util.HashSet<BlockPos> mutable = new java.util.HashSet<>();
        mutable.add(new BlockPos(0, 0, 0));
        OrphanReplayBuffer.add(42, mutable);
        // Mutate the original after buffering
        mutable.add(new BlockPos(99, 99, 99));
        mutable.clear();

        List<OrphanReplayBuffer.PendingOrphan> drained = OrphanReplayBuffer.drain();
        assertEquals(1, drained.size());
        assertEquals(1, drained.get(0).members().size(),
                "buffered copy must be immune to later mutation of the input set");
        assertTrue(drained.get(0).members().contains(new BlockPos(0, 0, 0)));
    }

    @Test
    @DisplayName("clear purges without replaying")
    public void clearPurges() {
        OrphanReplayBuffer.add(1, Set.of(new BlockPos(0, 0, 0)));
        OrphanReplayBuffer.add(2, Set.of(new BlockPos(1, 1, 1)));
        OrphanReplayBuffer.clear();
        assertEquals(0, OrphanReplayBuffer.size());
        assertTrue(OrphanReplayBuffer.drain().isEmpty());
    }
}
