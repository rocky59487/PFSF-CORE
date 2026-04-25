package com.blockreality.api.physics.pfsf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicReference;

/**
 * PFSF 物理引擎鎖定狀態 — server 全域。
 *
 * <p>當 GPU/Vulkan 初始化失敗、shader 編譯出錯、ABI 不相符或數值發散時，
 * 由各個失敗點呼叫 {@link #lock(String)} 進入鎖定狀態。此後：</p>
 * <ul>
 *   <li>{@link PFSFEngine#isAvailable()} 回傳 false（停止 tick 求解）</li>
 *   <li>RBlock 放置/破壞事件被 {@code BlockPhysicsEventHandler} 取消</li>
 *   <li>玩家上線時收到聊天紅字警告 + HUD 橫幅</li>
 *   <li>{@code /br status} 顯示鎖定原因</li>
 * </ul>
 *
 * <p>此類取代過去 GPU 失敗時靜默退回到圖論連通性簡易模式的行為 —
 * PFSF 是物理判定的唯一來源，不可用時系統明示失敗，而非偽裝通過。</p>
 *
 * <p>狀態僅存在記憶體；server 重啟後重置為未鎖定。鎖定解除須透過
 * {@link #unlock()}（測試 / debug 命令使用）。</p>
 */
public final class PFSFLockdown {

    private static final Logger LOGGER = LoggerFactory.getLogger("PFSF-Lockdown");

    private static final AtomicReference<String> REASON = new AtomicReference<>(null);

    private PFSFLockdown() {}

    /**
     * 進入鎖定狀態。第一次呼叫設定原因；後續呼叫保留首次原因（先發生的失敗優先）。
     *
     * @param reason 人類可讀的失敗原因（會被顯示給玩家，避免技術術語）
     */
    public static void lock(String reason) {
        if (reason == null || reason.isEmpty()) {
            reason = "PFSF physics engine unavailable";
        }
        if (REASON.compareAndSet(null, reason)) {
            LOGGER.error("[PFSF] LOCKDOWN ENGAGED: {}", reason);
        } else {
            LOGGER.warn("[PFSF] Additional lockdown trigger ignored ({}); first reason retained: {}",
                    reason, REASON.get());
        }
    }

    /**
     * 解除鎖定。僅供測試 / 除錯命令使用 — 一般情況下鎖定後須重啟 server。
     */
    public static void unlock() {
        String prev = REASON.getAndSet(null);
        if (prev != null) {
            LOGGER.warn("[PFSF] Lockdown released: {}", prev);
        }
    }

    /** @return true 若 PFSF 已鎖定（GPU 不可用或數值失效）。 */
    public static boolean isLocked() {
        return REASON.get() != null;
    }

    /** @return 當前鎖定原因；未鎖定時回傳 null。 */
    public static String getReason() {
        return REASON.get();
    }
}
