package com.blockreality.api.event;

import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraftforge.eventbus.api.Event;

import java.util.Set;

/**
 * Block Reality 結構坍方事件。
 *
 * 在 FORGE event bus 上 post 此事件，讓外部模組（CI、視覺效果等）可以掛接。
 *
 * 事件流程：
 *   1. PFSF failure_scan 偵測到失效並由 CollapseManager.triggerPFSFCollapse 觸發
 *   2. post RStructureCollapseEvent（每次崩塌一個方塊，collapsingBlocks 為單元素）
 *   3. CollapseManager 接管視覺效果（粒子、音效）
 */
public class RStructureCollapseEvent extends Event {

    private final ServerLevel level;
    private final BlockPos triggerPos;
    private final Set<BlockPos> collapsingBlocks;
    private final int affectedRadius;

    public RStructureCollapseEvent(ServerLevel level, BlockPos triggerPos,
                                    Set<BlockPos> collapsingBlocks) {
        this.level = level;
        this.triggerPos = triggerPos;
        this.collapsingBlocks = collapsingBlocks;
        // 從坍方方塊計算影響半徑
        int maxDist = 0;
        for (BlockPos pos : collapsingBlocks) {
            int dist = Math.abs(pos.getX() - triggerPos.getX())
                     + Math.abs(pos.getY() - triggerPos.getY())
                     + Math.abs(pos.getZ() - triggerPos.getZ());
            if (dist > maxDist) maxDist = dist;
        }
        this.affectedRadius = maxDist;
    }

    public ServerLevel getLevel() { return level; }
    public BlockPos getTriggerPos() { return triggerPos; }
    public Set<BlockPos> getCollapsingBlocks() { return collapsingBlocks; }
    public int getAffectedRadius() { return affectedRadius; }
    public int getCollapseCount() { return collapsingBlocks.size(); }
}
