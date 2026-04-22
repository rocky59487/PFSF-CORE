package com.blockreality.api.client;

import com.blockreality.api.BlockRealityMod;
import com.blockreality.api.spi.ModuleRegistry;
import com.mojang.blaze3d.platform.InputConstants;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;
import net.minecraft.client.KeyMapping;
import net.minecraft.client.Minecraft;
import net.minecraft.network.chat.Component;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.api.distmarker.OnlyIn;
import net.minecraftforge.client.event.RegisterKeyMappingsEvent;
import net.minecraftforge.client.event.RenderLevelStageEvent;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.TickEvent;
import net.minecraftforge.event.level.LevelEvent;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.client.event.EntityRenderersEvent;
import net.minecraftforge.fml.common.Mod;
import org.lwjgl.glfw.GLFW;

import com.blockreality.api.registry.BRBlockEntities;
import com.blockreality.api.registry.BREntities;

import java.util.concurrent.atomic.AtomicBoolean;

/**
 * 客戶端初始化 — v3fix §1.8
 *
 * MOD event bus 註冊：
 *   - R 鍵 KeyMapping（切換應力熱圖）
 *   - RenderLevelStageEvent 渲染掛接
 *
 * 重要：此類僅在 CLIENT 端載入（@OnlyIn + @Dist.CLIENT）
 */
@OnlyIn(Dist.CLIENT)
@Mod.EventBusSubscriber(modid = BlockRealityMod.MOD_ID, bus = Mod.EventBusSubscriber.Bus.MOD, value = Dist.CLIENT)
public class ClientSetup {

    private static final Logger LOGGER = LogManager.getLogger("BlockReality/ClientSetup");
    private static boolean pipelineInitFailed = false;
    private static boolean diagnosticSent = false;  // ★ 一次性診斷訊息

    /** 應力熱圖切換鍵 — 預設 R */
    public static final KeyMapping STRESS_OVERLAY_KEY = new KeyMapping(
        "key.blockreality.stress_overlay",   // 翻譯 key
        InputConstants.Type.KEYSYM,
        GLFW.GLFW_KEY_R,
        "key.categories.blockreality"        // 分類
    );

    /**
     * MOD bus: 註冊按鍵映射。
     */
    @SubscribeEvent
    public static void onRegisterKeyMappings(RegisterKeyMappingsEvent event) {
        event.register(STRESS_OVERLAY_KEY);
    }


    /**
     * 初始化 FORGE bus 的客戶端事件監聽。
     * 應在 FMLClientSetupEvent 中呼叫。
     */
    public static void initForgeEvents() {
        MinecraftForge.EVENT_BUS.register(ClientForgeEvents.class);
    }

    /**
     * FORGE event bus 的客戶端事件處理。
     * 分離到內部類，避免 MOD bus / FORGE bus 混用。
     */
    @OnlyIn(Dist.CLIENT)
    public static class ClientForgeEvents {

        /**
         * 渲染掛接 — 轉發到 StressHeatmapRenderer、AnchorPathRenderer、GhostBlockRenderer
         */
        @SubscribeEvent
        public static void onRenderLevel(RenderLevelStageEvent event) {
            StressHeatmapRenderer.onRenderLevelStage(event);
            AnchorPathRenderer.render(event);
            GhostBlockRenderer.onRenderLevel(event);
        }

        /**
         * 客戶端 tick — 檢測按鍵切換。
         * 僅在非 GUI 狀態下響應（避免打字時誤觸）。
         */
        @SubscribeEvent
        public static void onClientTick(TickEvent.ClientTickEvent event) {
            if (event.phase != TickEvent.Phase.END) return;

            Minecraft mc = Minecraft.getInstance();
            if (mc.player == null) return;

            // 檢查 R 鍵是否被按下
            while (STRESS_OVERLAY_KEY.consumeClick()) {
                StressHeatmapRenderer.toggleOverlay();

                // HUD 提示訊息
                String state = StressHeatmapRenderer.isOverlayEnabled() ? "ON" : "OFF";
                mc.player.displayClientMessage(
                    Component.literal("§6[BR] §fStress Heatmap: §" +
                        (StressHeatmapRenderer.isOverlayEnabled() ? "a" : "c") + state),
                    true // actionbar
                );
            }
        }

}
}
