package com.blockreality.api.physics.pfsf;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Phase C — GPU 能量讀回與 EMA 不變式判定器。
 *
 * <p>每個 island 維護一個 {@link EnergyEmaState}，追蹤 E_elastic / E_external /
 * E_phaseField 的指數移動平均 (EMA) 與方差，供 {@link com.blockreality.api.physics.pfsf.PFSFScheduler}
 * 做 Z-score 單步異常偵測。
 *
 * <h2>為何用 EMA 而非 raw threshold（P1 警告修正 2026-04-22）</h2>
 * PCG 迭代初期高頻誤差消除會讓 E 短暫逆流（數值上非單調）；若用
 * {@code E_after > E_before + 1e-4 × |E|} 這類 raw threshold 會被淹沒。
 * EMA + Z-score 作法：
 * <pre>
 *   μ_k    = α E_k + (1-α) μ_{k-1}
 *   σ²_k   = α (E_k - μ_{k-1})² + (1-α) σ²_{k-1}
 *   Z_k    = (E_k - μ_{k-1}) / (√σ²_{k-1} + ε)
 *   警告：k > WARMUP 且 Z_k > Z_THRESHOLD
 * </pre>
 *
 * <p>預熱 8 tick 內完全不告警；啟動後 3σ 外才寫 telemetry。
 *
 * <h2>GPU readback API（預留）</h2>
 * Phase C 實作分兩步：
 * <ol>
 *   <li>目前：純 Java 端記錄 + EMA 判定（可用 {@link com.blockreality.api.physics.effective.EnergyEvaluatorCPU}
 *       做 CPU 端 fallback）</li>
 *   <li>後續：在 PFSFDispatcher 加入 {@code recordEnergyReduction()}，呼叫 energy_reduce.comp.glsl
 *       並把 GPU 讀回的值透過 {@link #recordSample} 送入</li>
 * </ol>
 *
 * <h2>執行緒安全</h2>
 * {@code perIsland} 為 ConcurrentHashMap；每個 island 的 {@link EnergyEmaState} 由單執行緒
 * （PFSFScheduler tick thread）寫入，無鎖。
 */
public final class PFSFEnergyRecorder {

    /** EMA 學習率 α ∈ (0, 1) — 值越高對新觀測越敏感 */
    public static final double EMA_ALPHA = 0.2;

    /** 預熱期 tick 數，期間不告警 */
    public static final int WARMUP_TICKS = 8;

    /** Z-score 告警閾值（3σ 以上視為異常） */
    public static final double Z_THRESHOLD = 3.0;

    /** 方差下界，避免 σ = 0 除零 */
    private static final double VAR_EPS = 1e-9;

    private final Map<Long, EnergyEmaState> perIsland = new ConcurrentHashMap<>();

    /**
     * 記錄一筆新樣本並更新 EMA。
     *
     * @param islandId        島嶼 id
     * @param tick            目前 tick
     * @param eElastic        GPU（或 CPU fallback）算出的彈性能
     * @param eExternal       外力能
     * @param ePhaseField     相場能
     * @return 單步 Z-score（前 WARMUP_TICKS tick 回傳 0）
     */
    public double recordSample(long islandId, long tick,
                               double eElastic, double eExternal, double ePhaseField) {
        EnergyEmaState state = perIsland.computeIfAbsent(islandId, k -> new EnergyEmaState());
        double total = eElastic + eExternal + ePhaseField;
        return state.update(tick, eElastic, eExternal, ePhaseField, total);
    }

    /** 回傳 island 最近一筆樣本（或 null）。 */
    public EnergySample getLatestSample(long islandId) {
        EnergyEmaState s = perIsland.get(islandId);
        return (s == null) ? null : s.lastSample;
    }

    /** 給 scheduler 查詢 EMA 狀態（決定是否告警） */
    public EnergyEmaState getState(long islandId) {
        return perIsland.computeIfAbsent(islandId, k -> new EnergyEmaState());
    }

    /** 當 island 移除時清理狀態 */
    public void forget(long islandId) {
        perIsland.remove(islandId);
    }

    // ═══════════════════════════════════════════════════════════════
    //  EnergySample record
    // ═══════════════════════════════════════════════════════════════
    public record EnergySample(
        long tick, long islandId,
        double eElastic, double eExternal, double ePhaseField,
        double total
    ) {}

    // ═══════════════════════════════════════════════════════════════
    //  EnergyEmaState — 每 island 的 EMA 追蹤器
    // ═══════════════════════════════════════════════════════════════
    public static final class EnergyEmaState {
        private double meanTotal = Double.NaN;
        private double varTotal  = 0.0;
        private int    tickCount = 0;
        private EnergySample lastSample;

        /**
         * 更新 EMA 並回傳本步 Z-score。
         * 預熱期內（tickCount ≤ WARMUP_TICKS）回傳 0 以避免誤判。
         */
        double update(long tick, double eElastic, double eExt, double ePhase, double total) {
            EnergySample prev = lastSample;
            lastSample = new EnergySample(tick, -1L, eElastic, eExt, ePhase, total);
            tickCount++;

            if (Double.isNaN(meanTotal)) {
                // 初始化：第一筆直接作為均值
                meanTotal = total;
                varTotal  = 0.0;
                return 0.0;
            }

            double dev = total - meanTotal;
            // 方差先算（用舊均值，新樣本的殘差）避免後續更新污染
            varTotal  = EMA_ALPHA * dev * dev + (1.0 - EMA_ALPHA) * varTotal;
            meanTotal = EMA_ALPHA * total    + (1.0 - EMA_ALPHA) * meanTotal;

            if (tickCount <= WARMUP_TICKS) {
                return 0.0; // 預熱期不告警
            }
            double sigma = Math.sqrt(varTotal + VAR_EPS);
            return dev / sigma;
        }

        public double mean()       { return meanTotal; }
        public double variance()   { return varTotal; }
        public double stddev()     { return Math.sqrt(varTotal + VAR_EPS); }
        public int    tickCount()  { return tickCount; }

        /** 是否應告警此 Z-score（封裝判定邏輯） */
        public boolean shouldWarn(double z) {
            return tickCount > WARMUP_TICKS && Math.abs(z) > Z_THRESHOLD;
        }
    }

    /**
     * 違反事件 record，可寫入 CollapseJournal telemetry。
     *
     * @param islandId  島嶼 id
     * @param meanPrev  違反前的 EMA 均值
     * @param eCurrent  當前能量（觸發事件的值）
     * @param zScore    單步 Z-score
     * @param tick      發生 tick
     */
    public record EnergyViolation(
        long islandId,
        double meanPrev,
        double eCurrent,
        double zScore,
        long tick
    ) {}
}
