package com.blockreality.api.event;

import com.blockreality.api.BlockRealityMod;
import com.blockreality.api.block.RBlock;
import com.blockreality.api.block.RBlockEntity;
import com.blockreality.api.collapse.CollapseManager;
import com.blockreality.api.config.BRConfig;
import com.blockreality.api.material.DefaultMaterial;
import com.blockreality.api.material.RMaterial;
import com.blockreality.api.material.VanillaMaterialMap;
import com.blockreality.api.physics.AnchorContinuityChecker;
import com.blockreality.api.physics.ConnectivityCache;
import com.blockreality.api.physics.RCFusionDetector;
import com.blockreality.api.physics.StructureIslandRegistry;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.block.Blocks;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.level.BlockEvent;
import net.minecraftforge.eventbus.api.EventPriority;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * Forge 事件監聽器 — 連接 StructureIslandRegistry + RCFusionDetector。
 *
 * 結構完整性由 PFSF GPU 物理引擎在 ServerTickHandler 中處理。
 * 此處僅負責：
 *   1. Island Registry 登錄/註銷
 *   2. RC 融合偵測/降級
 *   3. ConnectivityCache epoch 更新
 */
@Mod.EventBusSubscriber(modid = BlockRealityMod.MOD_ID, bus = Mod.EventBusSubscriber.Bus.FORGE)
public class BlockPhysicsEventHandler {

    private static final Logger LOGGER = LogManager.getLogger("BR-Events");

    // ─── 放置事件 ───────────────────────────────────────────

    @SubscribeEvent(priority = EventPriority.NORMAL)
    public static void onBlockPlaced(BlockEvent.EntityPlaceEvent event) {
        if (!(event.getLevel() instanceof ServerLevel level)) return;
        if (!(event.getPlacedBlock().getBlock() instanceof RBlock)) return;

        // H6-fix revised: 創造模式仍執行物理計算，但抑制崩塌
        final boolean creativeMode = event.getEntity() instanceof net.minecraft.world.entity.player.Player player
                && player.isCreative();
        if (creativeMode) {
            CollapseManager.setSuppressCollapse(true);
        }

        final BlockPos pos = event.getPos().immutable();

        // ★ Phase 1a: 天然錨點預判 — 必須在 registerBlock 之前，
        // 否則 StructureIslandRegistry.registerBlock 在分類 voxel 類型時
        // 會把實際上該錨定的方塊當成一般體素（見 registerBlock 第 408 行的
        // anchorBlocks.contains(pos) 檢查）。這裡同時是從 test-only 的
        // registerAnchor 路徑接到生產端的唯一觸發點 — 一旦 anchorBlocks
        // 有內容，advanceTopology 的 safety-valve 就放行，
        // ThreeTierOrchestrator 才會真的輸出 orphan events。
        if (AnchorContinuityChecker.isNaturalAnchor(level, pos)) {
            StructureIslandRegistry.registerAnchor(pos);
        }

        // ★ Phase 1b: 登錄到 Island Registry（同步）
        long epoch = ConnectivityCache.getStructureEpoch();
        int islandId = StructureIslandRegistry.registerBlock(pos, epoch);

        level.getServer().execute(() -> {
            if (!level.getServer().isRunning()) return;
            ConnectivityCache.notifyStructureChanged(pos);

            // 驗證方塊確實存在（可能被低優先級事件取消放置）
            BlockState placedState = level.getBlockState(pos);
            if (!(placedState.getBlock() instanceof RBlock)) {
                StructureIslandRegistry.unregisterBlock(level, pos, epoch);
                // 同步 rollback 錨點登錄，避免 anchorBlocks 殘留 ghost 條目。
                StructureIslandRegistry.unregisterAnchor(pos);
                LOGGER.debug("[BR-Events] Block placement at {} was cancelled, rolled back island registration", pos);
                return;
            }

            // RC 融合偵測
            int fusions = RCFusionDetector.checkAndFuse(level, pos);
            if (fusions > 0) {
                LOGGER.debug("[BR-Events] RC fusion at {}: {} pairs fused", pos, fusions);
                BlockEntity be = level.getBlockEntity(pos);
                if (be instanceof RBlockEntity rbe) {
                    MinecraftForge.EVENT_BUS.post(new FusionCompletedEvent(level, pos,
                        DefaultMaterial.CONCRETE, rbe.getMaterial()));
                }
            }

            // PFSF 透過 onServerTick 自動檢測結構變化
        });
    }

    // ─── 破壞事件 ───────────────────────────────────────────

    @SubscribeEvent(priority = EventPriority.HIGHEST)
    public static void onBlockBreak(BlockEvent.BreakEvent event) {
        if (!(event.getLevel() instanceof ServerLevel level)) return;

        // H6-fix revised: 創造模式仍執行物理，但抑制崩塌
        final boolean creativeMode = event.getPlayer() != null && event.getPlayer().isCreative();
        if (creativeMode) {
            CollapseManager.setSuppressCollapse(true);
        }

        final BlockPos pos = event.getPos().immutable();
        BlockEntity be = level.getBlockEntity(pos);
        if (!(be instanceof RBlockEntity rbe)) return;

        // 快取 BlockType 供降級檢查
        final com.blockreality.api.material.BlockType cachedBlockType = rbe.getBlockType();

        // Phase 1: 取得 epoch（在方塊消失前）
        long epoch = ConnectivityCache.getStructureEpoch();

        // 延遲到方塊實際消失後執行
        level.getServer().execute(() -> {
            if (!level.getServer().isRunning()) return;
            ConnectivityCache.notifyStructureChanged(pos);

            // 從 Island Registry 註銷，取回所有分裂後的 island ID
            java.util.List<Integer> resultIds = StructureIslandRegistry.unregisterBlock(level, pos, epoch);
            // 同步清除錨點登錄 — 方塊已經破壞，若曾經是天然錨點就一併移除。
            // 呼叫 unregisterAnchor 是 idempotent 的（內部 set.remove）所以
            // 即使該位置從來不是錨點也安全。
            StructureIslandRegistry.unregisterAnchor(pos);

            // ★ Fix 1: 通知 PFSF 所有分裂後的 island（原 island + 新 island），
            // 觸發每個 island 的 sparse full-rebuild，確保 GPU buffer 同步正確拓撲。
            for (int id : resultIds) {
                com.blockreality.api.physics.pfsf.PFSFEngine.notifyBlockChange(id, pos, null, java.util.Set.of());
            }

            // RC 融合降級檢查
            int downgrades = RCFusionDetector.checkAndDowngrade(level, pos, cachedBlockType);
            if (downgrades > 0) {
                LOGGER.info("[BR-Events] Break at {} caused {} RC_NODE downgrades", pos, downgrades);
            }

            // 結構完整性由 PFSF GPU 引擎在 ServerTickHandler 中自動檢測
        });
    }
}
