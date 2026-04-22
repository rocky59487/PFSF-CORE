package com.blockreality.api.client;

import com.blockreality.api.network.CollapseEffectPacket.CollapseInfo;
import com.blockreality.api.physics.FailureType;
import net.minecraft.client.Minecraft;
import net.minecraft.core.BlockPos;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.api.distmarker.OnlyIn;

import java.util.Map;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * ★ review-fix ICReM-5: 客戶端崩塌效果快取
 *
 * 接收伺服器端的崩塌效果資料，並分發到對應的視覺效果渲染器。
 *
 * 不同破壞模式的視覺效果：
 *   - CANTILEVER_BREAK: 大型斷裂碎片 + 斷裂線動畫 + 整段下墜
 *   - CRUSHING:         漸進式裂紋擴散 + 壓碎粉塵 + 材質碎裂
 *   - NO_SUPPORT:       快速分散掉落 + 輕量粒子
 */
@OnlyIn(Dist.CLIENT)
public class ClientCollapseCache {

    /** 待處理的崩塌效果佇列（網路線程寫入，渲染線程讀取） */
    private static final ConcurrentLinkedQueue<CollapseEffect> pendingEffects = new ConcurrentLinkedQueue<>();

    public record CollapseEffect(BlockPos pos, FailureType type, int materialId) {}

    /**
     * 處理從伺服器收到的崩塌效果封包。
     * 在網路線程上呼叫，排入佇列。
     */
    public static void processCollapseEffects(Map<BlockPos, CollapseInfo> data) {
        for (Map.Entry<BlockPos, CollapseInfo> entry : data.entrySet()) {
            pendingEffects.add(new CollapseEffect(
                entry.getKey(),
                entry.getValue().type(),
                entry.getValue().materialId()
            ));
        }
    }


    /** 材質 ID → RGB 顏色 */
    private static float[] getMaterialColor(int materialId) {
        return switch (materialId) {
            case 0 -> new float[]{0.7f, 0.7f, 0.68f};   // concrete
            case 1 -> new float[]{0.55f, 0.6f, 0.65f};   // steel
            case 2 -> new float[]{0.6f, 0.4f, 0.2f};     // wood
            case 3 -> new float[]{0.5f, 0.5f, 0.55f};    // rebar
            case 4 -> new float[]{0.9f, 0.75f, 0.2f};    // RC node
            case 5 -> new float[]{0.4f, 0.7f, 1.0f};     // anchor pile
            default -> new float[]{0.8f, 0.8f, 0.8f};    // unknown
        };
    }

    /**
     * Fix 2: 客戶端崩塌動畫效果。
     * <ul>
     *   <li>CRUSHING: 近距離（16 格內）螢幕輕微震動，模擬衝擊波</li>
     *   <li>所有類型: 觸發 BRAnimationEngine 的 structure collapse clip</li>
     * </ul>
     */
    private static void triggerClientCollapseEffect(BlockPos pos, FailureType type) {
        Minecraft mc = Minecraft.getInstance();
        if (mc.player == null || mc.level == null) return;

        double distSq = mc.player.blockPosition().distSqr(pos);
        if (distSq > 64 * 64) return;


        // ── 環境粉塵粒子 ──
        if (distSq < 32 * 32) {
            double px = pos.getX() + 0.5, py = pos.getY() + 0.5, pz = pos.getZ() + 0.5;
            int dustCount = type == FailureType.CRUSHING ? 8 : 4;
            for (int i = 0; i < dustCount; i++) {
                double dx = (mc.level.random.nextDouble() - 0.5) * 2.5;
                double dy = mc.level.random.nextDouble() * 0.5;
                double dz = (mc.level.random.nextDouble() - 0.5) * 2.5;
                mc.level.addParticle(
                        net.minecraft.core.particles.ParticleTypes.CAMPFIRE_COSY_SMOKE,
                        px + dx, py + dy, pz + dz,
                        dx * 0.015, 0.03, dz * 0.015);
            }
        }
    }
}
