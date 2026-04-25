package com.blockreality.api.client;

import com.blockreality.api.BlockRealityMod;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.Font;
import net.minecraft.client.gui.GuiGraphics;
import net.minecraft.network.chat.Component;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.api.distmarker.OnlyIn;
import net.minecraftforge.client.event.RenderGuiOverlayEvent;
import net.minecraftforge.client.gui.overlay.VanillaGuiOverlay;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.fml.common.Mod;

import java.util.concurrent.atomic.AtomicReference;

/**
 * 客戶端 HUD：PFSF 鎖定狀態橫幅。
 *
 * <p>當 server 透過 {@link com.blockreality.api.network.PFSFLockdownPacket} 通知鎖定，
 * 在螢幕頂部中央渲染紅色橫幅顯示「PFSF UNAVAILABLE」+ 失敗原因。</p>
 *
 * <p>橫幅持續顯示直到 server 解除鎖定（送出 locked=false 封包）或玩家離線。</p>
 */
@OnlyIn(Dist.CLIENT)
@Mod.EventBusSubscriber(modid = BlockRealityMod.MOD_ID, bus = Mod.EventBusSubscriber.Bus.FORGE, value = Dist.CLIENT)
public final class PFSFLockdownHud {

    private static final AtomicReference<String> LOCK_REASON = new AtomicReference<>(null);

    private static final int BG_COLOR    = 0xC0_8B0000; // 深紅半透明
    private static final int TITLE_COLOR = 0xFFFF5555; // 鮮紅
    private static final int REASON_COLOR = 0xFFFFFFFF; // 白
    private static final int PADDING_X = 12;
    private static final int PADDING_Y = 6;
    private static final int TOP_OFFSET = 8;

    private PFSFLockdownHud() {}

    /**
     * 由 {@link com.blockreality.api.network.PFSFLockdownPacket} 在客戶端執行緒呼叫，
     * 設定當前鎖定狀態。
     */
    public static void setLocked(boolean locked, String reason) {
        LOCK_REASON.set(locked ? (reason != null && !reason.isEmpty() ? reason : "physics engine unavailable") : null);
    }

    /** @return true 若客戶端目前認為 server 處於 PFSF 鎖定狀態。 */
    public static boolean isLocked() {
        return LOCK_REASON.get() != null;
    }

    @SubscribeEvent
    public static void onRenderOverlay(RenderGuiOverlayEvent.Post event) {
        // 只在 hotbar 之後渲染一次（避免同一 frame 多次繪製）
        if (event.getOverlay() != VanillaGuiOverlay.HOTBAR.type()) return;

        String reason = LOCK_REASON.get();
        if (reason == null) return;

        Minecraft mc = Minecraft.getInstance();
        if (mc.options.hideGui) return;

        GuiGraphics g = event.getGuiGraphics();
        Font font = mc.font;

        Component title = Component.literal("⚠ PFSF UNAVAILABLE — physics frozen");
        Component detail = Component.literal(reason);

        int titleWidth  = font.width(title);
        int detailWidth = font.width(detail);
        int boxWidth = Math.max(titleWidth, detailWidth) + PADDING_X * 2;
        int boxHeight = font.lineHeight * 2 + PADDING_Y * 3;

        int screenWidth = event.getWindow().getGuiScaledWidth();
        int boxX = (screenWidth - boxWidth) / 2;
        int boxY = TOP_OFFSET;

        g.fill(boxX, boxY, boxX + boxWidth, boxY + boxHeight, BG_COLOR);
        g.drawString(font, title,
                boxX + (boxWidth - titleWidth) / 2,
                boxY + PADDING_Y, TITLE_COLOR, false);
        g.drawString(font, detail,
                boxX + (boxWidth - detailWidth) / 2,
                boxY + PADDING_Y + font.lineHeight + PADDING_Y, REASON_COLOR, false);
    }
}
