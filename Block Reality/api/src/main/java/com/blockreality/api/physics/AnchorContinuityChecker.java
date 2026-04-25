package com.blockreality.api.physics;

import com.blockreality.api.block.RBlock;
import com.blockreality.api.block.RBlockEntity;
import com.blockreality.api.material.BlockType;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.state.BlockState;

/**
 * 錨定點識別器 — 純幾何判定。
 *
 * <p>Java 端不做結構力學分析；錨定條件僅依賴幾何位置：</p>
 * <ol>
 *   <li>{@code y ≤ level.getMinBuildHeight() + 1}（底層地板）</li>
 *   <li>正下方是基岩 / 屏障</li>
 *   <li>自身是基岩 / 屏障</li>
 *   <li>{@link BlockType#ANCHOR_PILE}（手動錨定樁）</li>
 *   <li>正下方為非物理世界穩固面（地形支撐）</li>
 * </ol>
 *
 * <p>沿鋼筋路徑的 BFS 傳播判斷已移除 — 結構連通性由 PFSF GPU phi 場處理。
 * audit-fixes 階段進一步刪除了 {@code IAnchorChecker} 介面與快照版
 * {@code check(snapshot, ...)} 實作，因為 RWorldSnapshot 流程已被
 * PFSF 直接從 {@link StructureIslandRegistry} 取代。</p>
 */
public final class AnchorContinuityChecker {

    /** 全域單例 */
    private static final AnchorContinuityChecker INSTANCE = new AnchorContinuityChecker();
    public static AnchorContinuityChecker getInstance() { return INSTANCE; }

    private AnchorContinuityChecker() {}

    // ═══════════════════════════════════════════════════════
    //  Live Level 版本（即時事件用）
    // ═══════════════════════════════════════════════════════

    /**
     * 判定某個方塊是否為錨定點（純幾何，無 BFS）。
     * 結構連通性分析由 PFSF GPU phi 場負責。
     */
    public boolean isAnchored(ServerLevel level, BlockPos pos) {
        return isNaturalAnchor(level, pos)
                || (level.getBlockEntity(pos) instanceof RBlockEntity rbe && rbe.isAnchored());
    }

    // ─── 錨定源判定 ──────────────────────────────────────────

    /**
     * 判定是否為天然錨定點。
     *
     * <p>錨定源（v3fix spec）：</p>
     * <ol>
     *   <li>{@code y ≤ minBuildHeight + 1}</li>
     *   <li>正下方是基岩 / 屏障</li>
     *   <li>自身是基岩 / 屏障</li>
     *   <li>{@link BlockType#ANCHOR_PILE}（手動錨定樁）</li>
     *   <li>正下方為非物理世界穩固面 — 蓋在地形上的常見情況</li>
     * </ol>
     *
     * <p>只算「下方」：重力把方塊往下拉，結構支撐來自下方。橫向接觸（蓋在
     * 山坡旁的 RBlock）不算錨定 — 沒有摩擦或黏附模型。從上方接觸（從地形
     * 垂掛下來的「鐘乳石」）同樣不算把方塊撐住。先前版本走全部 6 個方向，
     * 對於蓋在地形附近的水平懸臂，每個 6 鄰居恰好碰到草地/泥土的方塊都成了
     * 「天然錨定」 — failure_scan 與 SupportPathAnalyzer 都會跳過它們，
     * 所以只有真正延伸到地形外的方塊頂端會掉，懸臂看起來會「一塊一塊掉」
     * 而不是斷裂。</p>
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

        // 5. 正下方是非物理世界穩固面（地形支撐）
        return hasStableExternalSupport(level, pos, Direction.DOWN);
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
}
