package com.blockreality.api.event;

import com.blockreality.api.BlockRealityMod;
import com.blockreality.api.block.RBlock;
import com.blockreality.api.block.RBlockEntity;
import com.blockreality.api.collapse.CollapseManager;
import com.blockreality.api.material.DefaultMaterial;
import com.blockreality.api.physics.AnchorContinuityChecker;
import com.blockreality.api.physics.ConnectivityCache;
import com.blockreality.api.physics.RCFusionDetector;
import com.blockreality.api.physics.StructureIslandRegistry;
import com.blockreality.api.physics.pfsf.PFSFEngine;
import com.blockreality.api.physics.pfsf.PFSFLockdown;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.network.chat.Component;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.entity.player.Player;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.level.BlockEvent;
import net.minecraftforge.eventbus.api.EventPriority;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

@Mod.EventBusSubscriber(modid = BlockRealityMod.MOD_ID, bus = Mod.EventBusSubscriber.Bus.FORGE)
public final class BlockPhysicsEventHandler {

    private static final Logger LOGGER = LogManager.getLogger("BR-Events");

    private BlockPhysicsEventHandler() {}

    @SubscribeEvent(priority = EventPriority.HIGHEST)
    public static void onBlockPlaced(BlockEvent.EntityPlaceEvent event) {
        if (!(event.getLevel() instanceof ServerLevel level)) return;

        BlockPos pos = event.getPos().immutable();
        boolean isRBlock = event.getPlacedBlock().getBlock() instanceof RBlock;

        // No-fallback contract: PFSF is the sole arbiter of RBlock physics. When PFSF is
        // locked (GPU init failed / numerical divergence), RBlock placement is blocked so
        // the world cannot accumulate physics-bearing structures the engine cannot solve.
        if (isRBlock && PFSFLockdown.isLocked()) {
            event.setCanceled(true);
            if (event.getEntity() instanceof Player p) {
                p.displayClientMessage(
                        Component.literal("§c[Block Reality] §fRBlock placement blocked: PFSF unavailable (")
                                .append(Component.literal(PFSFLockdown.getReason()))
                                .append(Component.literal(")")),
                        true);
            }
            return;
        }

        if (!isRBlock) {
            level.getServer().execute(() -> refreshAdjacentNaturalAnchors(level, pos));
            return;
        }

        boolean creativeMode = event.getEntity() instanceof net.minecraft.world.entity.player.Player player
                && player.isCreative();
        if (creativeMode) {
            CollapseManager.setSuppressCollapse(true);
        }

        if (AnchorContinuityChecker.isNaturalAnchor(level, pos)) {
            StructureIslandRegistry.registerAnchor(pos);
        }

        long epoch = ConnectivityCache.getStructureEpoch();
        StructureIslandRegistry.registerBlock(pos, epoch);

        level.getServer().execute(() -> {
            if (!level.getServer().isRunning()) return;

            ConnectivityCache.notifyStructureChanged(pos);

            BlockState placedState = level.getBlockState(pos);
            if (!(placedState.getBlock() instanceof RBlock)) {
                StructureIslandRegistry.unregisterBlock(level, pos, epoch);
                StructureIslandRegistry.unregisterAnchor(pos);
                LOGGER.debug("[BR-Events] Block placement at {} was cancelled, rolled back island registration", pos);
                return;
            }

            int fusions = RCFusionDetector.checkAndFuse(level, pos);
            if (fusions <= 0) return;

            LOGGER.debug("[BR-Events] RC fusion at {}: {} pairs fused", pos, fusions);
            BlockEntity be = level.getBlockEntity(pos);
            if (be instanceof RBlockEntity rbe) {
                MinecraftForge.EVENT_BUS.post(new FusionCompletedEvent(
                        level,
                        pos,
                        DefaultMaterial.CONCRETE,
                        rbe.getMaterial()));
            }
        });
    }

    @SubscribeEvent(priority = EventPriority.HIGHEST)
    public static void onBlockBreak(BlockEvent.BreakEvent event) {
        if (!(event.getLevel() instanceof ServerLevel level)) return;

        BlockPos pos = event.getPos().immutable();
        BlockEntity be = level.getBlockEntity(pos);
        if (!(be instanceof RBlockEntity rbe)) {
            level.getServer().execute(() -> refreshAdjacentNaturalAnchors(level, pos));
            return;
        }

        // Mirror the place-side guard: while PFSF is locked, RBlock break is also blocked
        // so the registry's island bookkeeping cannot drift away from a physics state the
        // engine cannot recompute. Creative-mode players still see the chat warning.
        if (PFSFLockdown.isLocked()) {
            event.setCanceled(true);
            if (event.getPlayer() != null) {
                event.getPlayer().displayClientMessage(
                        Component.literal("§c[Block Reality] §fRBlock break blocked: PFSF unavailable (")
                                .append(Component.literal(PFSFLockdown.getReason()))
                                .append(Component.literal(")")),
                        true);
            }
            return;
        }

        boolean creativeMode = event.getPlayer() != null && event.getPlayer().isCreative();
        if (creativeMode) {
            CollapseManager.setSuppressCollapse(true);
        }

        long epoch = ConnectivityCache.getStructureEpoch();
        var cachedBlockType = rbe.getBlockType();

        level.getServer().execute(() -> {
            if (!level.getServer().isRunning()) return;

            ConnectivityCache.notifyStructureChanged(pos);

            java.util.List<Integer> resultIds = StructureIslandRegistry.unregisterBlock(level, pos, epoch);
            StructureIslandRegistry.unregisterAnchor(pos);

            for (int id : resultIds) {
                PFSFEngine.notifyBlockChange(id, pos, null, java.util.Set.of());
            }

            int downgrades = RCFusionDetector.checkAndDowngrade(level, pos, cachedBlockType);
            if (downgrades > 0) {
                LOGGER.info("[BR-Events] Break at {} caused {} RC_NODE downgrades", pos, downgrades);
            }
        });
    }

    private static void refreshAdjacentNaturalAnchors(ServerLevel level, BlockPos changedPos) {
        if (level == null || level.getServer() == null || !level.getServer().isRunning()) {
            return;
        }

        java.util.Map<Integer, BlockPos> affectedIslands = new java.util.LinkedHashMap<>();
        boolean anchorChanged = false;

        for (Direction dir : Direction.values()) {
            BlockPos neighborPos = changedPos.relative(dir).immutable();
            BlockState neighborState = level.getBlockState(neighborPos);
            if (!(neighborState.getBlock() instanceof RBlock)) {
                continue;
            }

            boolean anchoredNow = AnchorContinuityChecker.isNaturalAnchor(level, neighborPos);
            boolean anchoredBefore = StructureIslandRegistry.isAnchorRegistered(neighborPos);
            if (anchoredNow == anchoredBefore) {
                continue;
            }

            anchorChanged = true;
            if (anchoredNow) {
                StructureIslandRegistry.registerAnchor(neighborPos);
            } else {
                StructureIslandRegistry.unregisterAnchor(neighborPos);
            }

            ConnectivityCache.notifyStructureChanged(neighborPos);
            int islandId = StructureIslandRegistry.getIslandId(neighborPos);
            if (islandId >= 0) {
                affectedIslands.putIfAbsent(islandId, neighborPos);
            }
        }

        if (!anchorChanged) {
            return;
        }

        long epoch = ConnectivityCache.getStructureEpoch();
        for (var entry : affectedIslands.entrySet()) {
            java.util.List<Integer> resultIds =
                    StructureIslandRegistry.refreshAnchorState(level, entry.getKey(), epoch);
            if (resultIds.isEmpty()) {
                resultIds = java.util.List.of(entry.getKey());
            }
            for (int islandId : resultIds) {
                PFSFEngine.notifyBlockChange(islandId, entry.getValue(), null, java.util.Set.of());
            }
        }
    }
}
