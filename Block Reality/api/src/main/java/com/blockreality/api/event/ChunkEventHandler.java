package com.blockreality.api.event;

import com.blockreality.api.BlockRealityMod;
import com.blockreality.api.block.RBlockEntity;
import com.blockreality.api.physics.AnchorContinuityChecker;
import com.blockreality.api.physics.ConnectivityCache;
import com.blockreality.api.physics.StructureIslandRegistry;
import com.blockreality.api.spi.ModuleRegistry;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.block.entity.BlockEntity;
import net.minecraft.world.level.chunk.LevelChunk;
import net.minecraftforge.event.level.ChunkEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

/**
 * 區塊事件處理器 — chunk 載入時重新登錄已儲存的 RBlock，
 * chunk 卸載時清理纜索。結構載重由 PFSF GPU 引擎管理。
 */
@Mod.EventBusSubscriber(modid = BlockRealityMod.MOD_ID, bus = Mod.EventBusSubscriber.Bus.FORGE)
public class ChunkEventHandler {

    private static final Logger LOGGER = LogManager.getLogger("BR-ChunkEvent");

    /**
     * Re-hydrate RBlock registrations from the chunk's persisted
     * BlockEntity map. Without this, any world reloaded from disk
     * starts with an empty StructureIslandRegistry — saved structures
     * get no physics until a block is placed or broken, at which
     * point spurious orphan events fire for the parts the registry
     * never learned about. Running every Load event keeps the
     * in-memory registry in sync with what is actually on disk.
     *
     * <p>The iteration walks {@code getBlockEntities()} rather than
     * scanning every voxel, because only RBlocks have block entities
     * and only RBlocks need registration. Natural-anchor detection
     * reuses {@link AnchorContinuityChecker#isNaturalAnchor} to stay
     * consistent with the place-event path in
     * {@link BlockPhysicsEventHandler}.</p>
     */
    @SubscribeEvent
    public static void onChunkLoad(ChunkEvent.Load event) {
        if (!(event.getLevel() instanceof ServerLevel level)) return;
        if (!(event.getChunk() instanceof LevelChunk levelChunk)) return;

        long epoch = ConnectivityCache.getStructureEpoch();
        int registered = 0;
        int anchored = 0;

        // getBlockEntities returns a live view; copy to avoid
        // concurrent-mod issues if something else touches the chunk
        // mid-iteration.
        java.util.Map<BlockPos, BlockEntity> entities =
                new java.util.HashMap<>(levelChunk.getBlockEntities());

        for (java.util.Map.Entry<BlockPos, BlockEntity> e : entities.entrySet()) {
            if (!(e.getValue() instanceof RBlockEntity)) continue;
            BlockPos pos = e.getKey().immutable();

            // Skip already-registered positions — chunks can re-fire
            // Load on teleport / world transitions; re-registering
            // would push the same pos into the StructureIsland member
            // set twice and confuse the dirty/epoch accounting.
            if (StructureIslandRegistry.getIslandId(pos) >= 0) continue;

            // Anchor first so registerBlock sees the flag (matches
            // the ordering in BlockPhysicsEventHandler.onBlockPlaced).
            if (AnchorContinuityChecker.isNaturalAnchor(level, pos)) {
                StructureIslandRegistry.registerAnchor(pos);
                anchored++;
            }
            StructureIslandRegistry.registerBlock(pos, epoch);
            registered++;
        }

        if (registered > 0) {
            ChunkPos cp = event.getChunk().getPos();
            LOGGER.debug("[BR-ChunkEvent] Chunk [{}, {}] load: re-hydrated {} RBlocks ({} anchors)",
                    cp.x, cp.z, registered, anchored);
        }
    }

    @SubscribeEvent
    public static void onChunkUnload(ChunkEvent.Unload event) {
        if (!(event.getLevel() instanceof ServerLevel level)) return;

        ChunkPos chunkPos = event.getChunk().getPos();

        // 清理纜索（兩端都在此 chunk 內的纜索）
        int cablesRemoved = ModuleRegistry.getCableManager().removeChunkCables(chunkPos);

        if (cablesRemoved > 0) {
            LOGGER.debug("[BR-ChunkEvent] Chunk [{}, {}] unload: removed {} cables",
                chunkPos.x, chunkPos.z, cablesRemoved);
        }
    }
}
