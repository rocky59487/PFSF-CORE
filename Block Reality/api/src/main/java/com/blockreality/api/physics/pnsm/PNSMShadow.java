package com.blockreality.api.physics.pnsm;

import com.blockreality.api.config.BRConfig;
import net.minecraft.core.BlockPos;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Phase 1.0 shadow mirror for the PNSM RFC
 * ({@code docs/design/pfsf-native-structural-memory.md}).
 *
 * <p>This class is the minimum-viable version: a flat {@code leafKey →
 * occupancy bitmap} map, not yet an SO-DAG. The purpose of this phase
 * is to prove the wiring — every {@code StructureIslandRegistry}
 * mutation reaches this shadow, and at tick end a reconstructed voxel
 * set diffed against the registry's members agrees. Phase 1.1 will
 * layer the actual octree + hash-cons structure on top of this same
 * mutation API.</p>
 *
 * <p>Activation is gated by {@link BRConfig#isPNSMShadowEnabled}; when
 * disabled every public method short-circuits so the overhead in a
 * production build is a single volatile read per call. Log output is
 * rate-limited so a diff storm cannot flood the console.</p>
 *
 * <p>Thread-safety: mutations are expected from the server tick thread
 * only, but the backing map is a {@link ConcurrentHashMap} so that the
 * /br status command (or any future async diagnostic) can read it from
 * a worker thread without external synchronisation. Diffs acquire no
 * locks; the snapshot is eventually-consistent, which is acceptable
 * because this is a development-time tool that tolerates race noise
 * in its warnings.</p>
 */
public final class PNSMShadow {

    private static final Logger LOG = LoggerFactory.getLogger("PNSM-Shadow");

    /** 4×4×4 leaf span. Voxel -> leaf by arithmetic shift of 2 bits. */
    public static final int LEAF_EDGE = 4;
    private static final int LEAF_EDGE_LOG2 = 2;
    private static final int LEAF_VOXELS = LEAF_EDGE * LEAF_EDGE * LEAF_EDGE;

    /**
     * Packed leaf key: lower 21 bits of (x,y,z) leaf coordinates. Fits
     * a ±1 M-block world on a single long, which covers every
     * plausible Minecraft range ({@code ±30 M} is well inside ±1 M when
     * shifted by 2). The key encoding is designed so two voxels in the
     * same 4×4×4 region produce identical keys regardless of their
     * sub-leaf offset.
     */
    private static long leafKey(BlockPos pos) {
        long lx = (pos.getX() >> LEAF_EDGE_LOG2) & 0x1FFFFFL; // 21 bits
        long ly = (pos.getY() >> LEAF_EDGE_LOG2) & 0x1FFFFFL;
        long lz = (pos.getZ() >> LEAF_EDGE_LOG2) & 0x1FFFFFL;
        return (lx << 42) | (ly << 21) | lz;
    }

    /** Bit index within the 64-bit occupancy word for a given voxel. */
    private static int leafBit(BlockPos pos) {
        int dx = pos.getX() & (LEAF_EDGE - 1);
        int dy = pos.getY() & (LEAF_EDGE - 1);
        int dz = pos.getZ() & (LEAF_EDGE - 1);
        return (dx << 4) | (dy << 2) | dz; // 6 bits → 0..63
    }

    /** Expand a leaf key and one bit back into a BlockPos. Inverse of
     *  {@link #leafKey}/{@link #leafBit}. Sign-extends the 21-bit
     *  leaf coordinates so negative-Z worlds round-trip correctly. */
    private static BlockPos expand(long key, int bit) {
        long lx = (key >>> 42) & 0x1FFFFFL;
        long ly = (key >>> 21) & 0x1FFFFFL;
        long lz = key & 0x1FFFFFL;
        int sx = signExtend21((int) lx);
        int sy = signExtend21((int) ly);
        int sz = signExtend21((int) lz);
        int dx = (bit >> 4) & 0x3;
        int dy = (bit >> 2) & 0x3;
        int dz = bit & 0x3;
        return new BlockPos((sx << LEAF_EDGE_LOG2) | dx,
                            (sy << LEAF_EDGE_LOG2) | dy,
                            (sz << LEAF_EDGE_LOG2) | dz);
    }

    private static int signExtend21(int v) {
        return (v << 11) >> 11;
    }

    /** leafKey → 64-bit occupancy bitmap. A leaf with zero occupancy
     *  is removed from the map so {@link #reconstruct} does not walk
     *  empty regions. */
    private static final Map<Long, Long> leaves = new ConcurrentHashMap<>();

    /** Per-JVM mutation counter — lets operators see at a glance in a
     *  debug dump whether any mirroring happened at all. */
    private static volatile long mutationCount = 0;

    /** Rate-limit guard for diff warnings; we only want one log line
     *  per second even during a diff storm to avoid drowning the
     *  console during a regression. */
    private static volatile long lastWarnAtMillis = 0;
    private static final long WARN_INTERVAL_MS = 1000;

    private PNSMShadow() {}

    /** Called from {@code StructureIslandRegistry.registerBlock}
     *  after the legacy map has been updated. No-op when the flag is
     *  off so the shadow path has zero allocation cost in production. */
    public static void mirrorInsert(BlockPos pos) {
        if (!BRConfig.isPNSMShadowEnabled()) return;
        long key = leafKey(pos);
        int bit = leafBit(pos);
        leaves.merge(key, 1L << bit, (oldOcc, addBit) -> oldOcc | addBit);
        mutationCount++;
    }

    /** Called from {@code StructureIslandRegistry.unregisterBlock}
     *  after the legacy map has been updated. Clears the voxel bit
     *  and drops the leaf entirely if it becomes empty. */
    public static void mirrorRemove(BlockPos pos) {
        if (!BRConfig.isPNSMShadowEnabled()) return;
        long key = leafKey(pos);
        int bit = leafBit(pos);
        leaves.compute(key, (k, oldOcc) -> {
            if (oldOcc == null) return null;
            long next = oldOcc & ~(1L << bit);
            return next == 0L ? null : next;
        });
        mutationCount++;
    }

    /** Full wipe — called from
     *  {@code StructureIslandRegistry.clear} / {@code resetForTesting}
     *  so world unload and JUnit isolation stay consistent. */
    public static void reset() {
        leaves.clear();
        mutationCount = 0;
    }

    /** Rebuild the full live voxel set from the leaf map. Intended
     *  only for diff verification and debug dumps — never called on
     *  the hot tick path. Cost is O(live voxels). */
    public static Set<BlockPos> reconstruct() {
        Set<BlockPos> out = new HashSet<>(leaves.size() * LEAF_VOXELS / 4);
        for (Map.Entry<Long, Long> e : leaves.entrySet()) {
            long occ = e.getValue();
            long key = e.getKey();
            while (occ != 0L) {
                int bit = Long.numberOfTrailingZeros(occ);
                out.add(expand(key, bit));
                occ &= occ - 1;
            }
        }
        return out;
    }

    /** Compare the shadow's reconstructed voxel set against the
     *  caller-supplied legacy voxel set (typically
     *  {@code StructureIslandRegistry.blockToIsland.keySet()}). Logs
     *  a rate-limited warning on any disagreement. Returns true when
     *  the sets agree exactly, false otherwise — allowing test code
     *  to assert agreement without stringifying the log output. */
    public static boolean diffAgainst(Set<BlockPos> legacy) {
        if (!BRConfig.isPNSMShadowEnabled()) return true;
        Set<BlockPos> mine = reconstruct();
        if (mine.equals(legacy)) return true;

        Set<BlockPos> onlyInPNSM = new HashSet<>(mine);
        onlyInPNSM.removeAll(legacy);
        Set<BlockPos> onlyInLegacy = new HashSet<>(legacy);
        onlyInLegacy.removeAll(mine);

        long now = System.currentTimeMillis();
        if (now - lastWarnAtMillis >= WARN_INTERVAL_MS) {
            lastWarnAtMillis = now;
            LOG.warn("[PNSM] shadow-mirror diverged from registry: +{} in PNSM, +{} in registry (sampled: {} / {})",
                    onlyInPNSM.size(), onlyInLegacy.size(),
                    firstOrNull(onlyInPNSM), firstOrNull(onlyInLegacy));
        }
        return false;
    }

    private static Object firstOrNull(Set<BlockPos> s) {
        return s.isEmpty() ? null : s.iterator().next();
    }

    // ─── Diagnostics ───

    public static int leafCount() { return leaves.size(); }
    public static long mutationCount() { return mutationCount; }

    /** Total live voxel count across all leaves. O(leaf count). */
    public static int voxelCount() {
        int total = 0;
        for (long occ : leaves.values()) total += Long.bitCount(occ);
        return total;
    }
}
