package com.blockreality.api.physics;

import com.blockreality.api.block.RBlock;
import com.blockreality.api.block.RBlockEntity;
import com.blockreality.api.config.BRConfig;
import com.blockreality.api.material.BlockType;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.state.BlockState;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;

/**
 * 錨定點識別器 — 純幾何判定。
 *
 * Java 端不做結構力學分析；錨定條件僅依賴幾何位置：
 *   1. y ≤ level.getMinBuildHeight() + 1（底層地板）
 *   2. 正下方是基岩/屏障
 *   3. 自身是基岩/屏障
 *   4. BlockType.ANCHOR_PILE（手動錨定樁）
 *
 * 沿鋼筋路徑的 BFS 傳播判斷已移除 — 結構連通性由 PFSF GPU phi 場處理。
 *
 * 實作 IAnchorChecker 介面（快照版本）。
 */
public class AnchorContinuityChecker implements IAnchorChecker {

    private static final Direction[] ALL_DIRS = Direction.values();

    /** 全域單例 */
    private static final AnchorContinuityChecker INSTANCE = new AnchorContinuityChecker();
    public static AnchorContinuityChecker getInstance() { return INSTANCE; }

    // ═══════════════════════════════════════════════════════
    //  IAnchorChecker 介面實作（快照版本）
    // ═══════════════════════════════════════════════════════

    @Override
    public AnchorResult check(RWorldSnapshot snapshot, Set<BlockPos> anchorSeeds) {
        Set<BlockPos> anchored = new HashSet<>();
        Set<BlockPos> orphans = new HashSet<>();

        int maxDepth = BRConfig.INSTANCE.anchorBfsMaxDepth.get();
        Deque<BlockPos> queue = new ArrayDeque<>(anchorSeeds);
        Set<BlockPos> visited = new HashSet<>(anchorSeeds);
        anchored.addAll(anchorSeeds);

        // BFS 從種子出發
        int steps = 0;
        while (!queue.isEmpty() && steps < maxDepth * 100) {
            BlockPos current = queue.poll();
            steps++;

            for (Direction dir : ALL_DIRS) {
                BlockPos neighbor = current.relative(dir);
                if (visited.contains(neighbor)) continue;

                int lx = neighbor.getX() - snapshot.getStartX();
                int ly = neighbor.getY() - snapshot.getStartY();
                int lz = neighbor.getZ() - snapshot.getStartZ();

                if (lx < 0 || ly < 0 || lz < 0
                    || lx >= snapshot.getSizeX()
                    || ly >= snapshot.getSizeY()
                    || lz >= snapshot.getSizeZ()) continue;

                RBlockState rbs = snapshot.getBlock(lx, ly, lz);
                if (rbs == null || rbs.blockId().equals("minecraft:air")) continue;

                visited.add(neighbor);
                anchored.add(neighbor);
                queue.add(neighbor);
            }
        }

        // 快照中所有非空氣方塊若沒被 BFS 觸及 = 孤島
        for (int lx = 0; lx < snapshot.getSizeX(); lx++) {
            for (int ly = 0; ly < snapshot.getSizeY(); ly++) {
                for (int lz = 0; lz < snapshot.getSizeZ(); lz++) {
                    RBlockState rbs = snapshot.getBlock(lx, ly, lz);
                    if (rbs == null || rbs.blockId().equals("minecraft:air")) continue;
                    BlockPos worldPos = new BlockPos(
                        snapshot.getStartX() + lx,
                        snapshot.getStartY() + ly,
                        snapshot.getStartZ() + lz
                    );
                    if (!anchored.contains(worldPos)) {
                        orphans.add(worldPos);
                    }
                }
            }
        }

        return new AnchorResult(anchored, orphans);
    }

    // ═══════════════════════════════════════════════════════
    //  Live Level 版本（即時事件用）
    // ═══════════════════════════════════════════════════════

    /**
     * 判定某個方塊是否為錨定點（純幾何，無 BFS）。
     * 結構連通性分析由 PFSF GPU phi 場負責。
     */
    public boolean isAnchored(ServerLevel level, BlockPos pos) {
        return isNaturalAnchor(level, pos) || (level.getBlockEntity(pos) instanceof RBlockEntity rbe && rbe.isAnchored());
    }

    // ─── 錨定源判定 ──────────────────────────────────────────

    /**
     * 判定是否為天然錨定點。
     *
     * 三種錨定源（v3fix spec）：
     *   1. y ≤ minBuildHeight + 1
     *   2. 正下方是基岩
     *   3. BlockType.ANCHOR_PILE（手動錨定樁）
     */
    public static boolean isNaturalAnchor(ServerLevel level, BlockPos pos) {
        // 1. 最底層
        if (pos.getY() <= level.getMinBuildHeight() + 1) return true;

        // 2. 正下方是基岩或屏障
        BlockState below = level.getBlockState(pos.below());
        if (below.is(Blocks.BEDROCK) || below.is(Blocks.BARRIER)) return true;

        // 3. 本身是基岩
        BlockState state = level.getBlockState(pos);
        if (state.is(Blocks.BEDROCK) || state.is(Blocks.BARRIER)) return true;

        // 4. ANCHOR_PILE BlockType（手動指定錨定點）
        BlockEntity be = level.getBlockEntity(pos);
        if (be instanceof RBlockEntity rbe && rbe.getBlockType() == BlockType.ANCHOR_PILE) {
            return true;
        }

        // Treat contact with a sturdy non-physics world block BELOW as a
        // natural foundation. This covers the common case where RBlocks are
        // built on terrain rather than directly on bedrock.
        //
        // Only DOWN counts: gravity pulls down, so structural support comes
        // from below. Lateral contact (an RBlock placed next to a hill) does
        // NOT anchor — there is no friction or adhesion model. Contact from
        // above (a "stalactite" hanging from terrain) likewise does not hold
        // a block up.
        //
        // The previous version walked all 6 directions. For a horizontal
        // cantilever built near terrain, every block whose 6-neighbourhood
        // happened to clip grass/dirt became a "natural anchor" —
        // failure_scan and SupportPathAnalyzer both skipped them, so only
        // the literal tip extending past terrain ever fell, and the cantilever
        // appeared to "shed" one block at a time instead of breaking.
        if (hasStableExternalSupport(level, pos, Direction.DOWN)) {
            return true;
        }

        return false;
    }

    private static boolean hasStableExternalSupport(ServerLevel level, BlockPos pos, Direction dir) {
        BlockPos supportPos = pos.relative(dir);
        BlockState supportState = level.getBlockState(supportPos);
        if (supportState.isAir() || !supportState.getFluidState().isEmpty()) {
            return false;
        }
        if (supportState.getBlock() instanceof RBlock) {
            return false;
        }
        return supportState.isFaceSturdy(level, supportPos, dir.getOpposite());
    }

    // ─── 快取管理（no-op — 純幾何判定無需快取）──────────────

    /** No-op：已移除 BFS 快取，保留簽名供舊呼叫端相容。 */
    public void markDirty(BlockPos pos) {}

    /** No-op：已移除 BFS 快取。 */
    public void clearCache() {}

    public String getCacheStats() {
        return "AnchorCache: disabled (geometric-only mode)";
    }
}
