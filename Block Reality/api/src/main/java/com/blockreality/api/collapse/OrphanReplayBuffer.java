package com.blockreality.api.collapse;

import net.minecraft.core.BlockPos;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentLinkedDeque;

/**
 * FIFO buffer for orphan-island events whose originating path could
 * not supply a {@link net.minecraft.server.level.ServerLevel}. The
 * Codex P1 review flagged that {@code CollapseManager.onOrphanIsland}
 * used to silently drop these events — batched removals from
 * {@code flushDestructions} therefore never collapsed in their
 * fracture tick. This buffer holds them instead;
 * {@code CollapseManager.flushPendingOrphans(level)} drains the
 * buffer on the next server tick once a level is known.
 *
 * <p>Split out from {@code CollapseManager} so that the buffer's
 * semantics can be verified in the sandbox without pulling in every
 * Minecraft Forge class the manager depends on.
 *
 * <p>All operations are thread-safe via {@link ConcurrentLinkedDeque}.
 */
public final class OrphanReplayBuffer {

    /** One buffered orphan event. Preserves the original island id and
     *  member block set for the eventual replay. */
    public record PendingOrphan(int islandId, Set<BlockPos> members, long bufferedMillis) {}

    private static final ConcurrentLinkedDeque<PendingOrphan> pending = new ConcurrentLinkedDeque<>();

    private OrphanReplayBuffer() {}

    /**
     * Append an orphan event to the buffer. No-op if {@code members}
     * is empty. The stored set is an immutable snapshot so later
     * mutations on the original set cannot corrupt the buffered copy.
     */
    public static void add(int islandId, Set<BlockPos> members) {
        if (members == null || members.isEmpty()) return;
        pending.add(new PendingOrphan(islandId, Set.copyOf(members), System.currentTimeMillis()));
    }

    /**
     * Remove and return every currently-buffered event in FIFO order.
     * Subsequent {@link #size()} calls return 0 until new events are
     * {@link #add added}.
     */
    public static List<PendingOrphan> drain() {
        List<PendingOrphan> out = new ArrayList<>();
        PendingOrphan o;
        while ((o = pending.pollFirst()) != null) out.add(o);
        return out;
    }

    /** Current number of buffered events (lock-free approximation). */
    public static int size() {
        return pending.size();
    }

    /** Purge the buffer without replaying. Primarily for tests. */
    public static void clear() {
        pending.clear();
    }
}
