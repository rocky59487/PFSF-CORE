package com.blockreality.api.physics.pfsf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static com.blockreality.api.physics.pfsf.PFSFConstants.*;

/**
 * PFSF 迭代排程器 — Chebyshev 半迭代加速 + 保守重啟 + 發散熔斷。
 *
 * <p>Chebyshev 半迭代（Wang 2015, SIGGRAPH Asia）透過動態調整每步步長 ω，
 * 使 Jacobi 迭代收斂速度提升約一個數量級。</p>
 *
 * <p>保守重啟機制在崩塌發生時重置 ω 回 1.0（純 Jacobi），
 * 避免拓撲突變導致 Chebyshev 過度外插。</p>
 *
 * 參考：PFSF 手冊 §4.3
 */
public final class PFSFScheduler {

    private static final Logger LOGGER = LoggerFactory.getLogger("PFSF-Scheduler");

    /** 預算 omega 表（最多 64 步） */
    private static final int OMEGA_TABLE_SIZE = 64;

    private PFSFScheduler() {}

    // ═══════════════════════════════════════════════════════════════
    //  Chebyshev Omega 排程
    // ═══════════════════════════════════════════════════════════════

    /**
     * 計算 Chebyshev 半迭代的 omega 值。
     *
     * <pre>
     * iter=0: ω = 1.0
     * iter=1: ω = 2 / (2 - ρ²)
     * iter≥2: ω = 4 / (4 - ρ² × ω_{k-1})
     * </pre>
     *
     * <p>v0.3d Phase 4: routes through {@code libpfsf_compute.pfsf_chebyshev_omega}
     * when {@link NativePFSFBridge#hasComputeV4()} resolves {@code true};
     * {@link #computeOmegaJavaRef(int, float)} is preserved verbatim as the
     * parity oracle.</p>
     *
     * @param iter     當前迭代索引（從 0 開始，已扣除 warmup）
     * @param rhoSpec  頻譜半徑估計
     * @return omega 值（≥ 1.0）
     */
    public static float computeOmega(int iter, float rhoSpec) {
        if (NativePFSFBridge.hasComputeV4()) {
            try {
                return NativePFSFBridge.nativeChebyshevOmega(iter, rhoSpec);
            } catch (UnsatisfiedLinkError e) {
                // fall through to the Java ref.
            }
        }
        return computeOmegaJavaRef(iter, rhoSpec);
    }

    /** Java reference implementation — never deleted (golden-vector oracle). */
    static float computeOmegaJavaRef(int iter, float rhoSpec) {
        if (iter <= 0) return 1.0f;

        float rhoSq = rhoSpec * rhoSpec;
        // 安全檢查：rhoSq >= 1 時 Chebyshev 無法收斂，退回純 Jacobi
        if (rhoSq >= 1.0f) {
            LOGGER.warn("[PFSF] Invalid rhoSpec={} (rhoSq >= 1), falling back to omega=1.0", rhoSpec);
            return 1.0f;
        }

        if (iter == 1) {
            return 2.0f / (2.0f - rhoSq);
        }

        // 遞推計算（避免存全表）
        float omegaPrev = 1.0f;
        float omega = 2.0f / (2.0f - rhoSq); // iter=1
        for (int k = 2; k <= iter; k++) {
            float denom = 4.0f - rhoSq * omega;
            if (denom < OMEGA_DENOM_EPSILON) {
                // A6-fix: 分母趨近零 → 回傳上一步的穩定值（非當前 stale 值）
                omega = omegaPrev;
                break;
            }
            omegaPrev = omega;
            float omegaNew = 4.0f / denom;
            // 超過 2.0 表示遞迴已發散，不應繼續
            if (omegaNew > 2.0f || Float.isNaN(omegaNew)) {
                LOGGER.debug("[PFSF] Chebyshev omega diverged at k={}, using omegaPrev={}", k, omegaPrev);
                omega = omegaPrev;
                break;
            }
            omega = omegaNew;
        }
        return Math.min(omega, MAX_OMEGA);
    }

    /**
     * 預算 omega 表（靜態查找，避免每步遞推）。
     *
     * <p>v0.3d Phase 4: routes through {@code pfsf_precompute_omega_table}
     * when available; {@link #precomputeOmegaTableJavaRef(float)} is
     * preserved as the parity oracle.</p>
     */
    public static float[] precomputeOmegaTable(float rhoSpec) {
        if (NativePFSFBridge.hasComputeV4()) {
            try {
                float[] table = new float[OMEGA_TABLE_SIZE];
                int n = NativePFSFBridge.nativePrecomputeOmegaTable(rhoSpec, table);
                if (n == OMEGA_TABLE_SIZE) return table;
                // On partial fill fall back to the Java ref — never return
                // a half-populated schedule to the caller.
            } catch (UnsatisfiedLinkError e) {
                // fall through.
            }
        }
        return precomputeOmegaTableJavaRef(rhoSpec);
    }

    /** Java reference implementation — never deleted (golden-vector oracle). */
    static float[] precomputeOmegaTableJavaRef(float rhoSpec) {
        float[] table = new float[OMEGA_TABLE_SIZE];
        float rhoSq = rhoSpec * rhoSpec;

        table[0] = 1.0f;
        if (OMEGA_TABLE_SIZE > 1) {
            table[1] = 2.0f / (2.0f - rhoSq);
        }
        for (int k = 2; k < OMEGA_TABLE_SIZE; k++) {
            float denom = 4.0f - rhoSq * table[k - 1];
            if (denom < OMEGA_DENOM_EPSILON) { table[k] = table[k - 1]; continue; }
            table[k] = Math.min(4.0f / denom, 1.98f);
            if (Float.isNaN(table[k])) table[k] = 1.0f;
        }
        return table;
    }

    // ═══════════════════════════════════════════════════════════════
    //  頻譜半徑估計
    // ═══════════════════════════════════════════════════════════════

    /**
     * 估計 3D 正規網格上 Laplacian 的頻譜半徑。
     *
     * <pre>
     * ρ_spec = cos(π / Lmax) × SAFETY_MARGIN
     * </pre>
     *
     * D2 注：此為正規均勻網格的近似值。不規則 island 形狀或材料
     * 變化可能導致實際頻譜半徑偏離。SAFETY_MARGIN (0.95) 和
     * 保守重啟機制（§4.3.1）共同確保數值穩定性。
     *
     * @param Lmax 網格最大維度
     * @return 頻譜半徑估計 ∈ (0, 1)
     */
    public static float estimateSpectralRadius(int Lmax) {
        if (NativePFSFBridge.hasComputeV4()) {
            try {
                return NativePFSFBridge.nativeEstimateSpectralRadius(Lmax, SAFETY_MARGIN);
            } catch (UnsatisfiedLinkError e) {
                // fall through.
            }
        }
        return estimateSpectralRadiusJavaRef(Lmax);
    }

    /** Java reference implementation — never deleted (golden-vector oracle). */
    static float estimateSpectralRadiusJavaRef(int Lmax) {
        if (Lmax <= 1) return 0.5f;
        return (float) (Math.cos(Math.PI / Lmax) * SAFETY_MARGIN);
    }

    // ═══════════════════════════════════════════════════════════════
    //  迭代步數推薦
    // ═══════════════════════════════════════════════════════════════

    /**
     * 決定每 tick 此 island 應跑多少步迭代。
     *
     * @param buf     island buffer
     * @param isDirty 是否有結構變更
     * @param hasCollapse 是否正在連鎖崩塌
     * @return 推薦步數（0 = 已收斂，無需更新）
     */
    public static int recommendSteps(PFSFIslandBuffer buf, boolean isDirty, boolean hasCollapse) {
        if (NativePFSFBridge.hasComputeV4()) {
            try {
                return NativePFSFBridge.nativeRecommendSteps(
                        buf.getLy(), buf.chebyshevIter,
                        isDirty, hasCollapse,
                        STEPS_MINOR, STEPS_MAJOR, STEPS_COLLAPSE);
            } catch (UnsatisfiedLinkError e) {
                // fall through.
            }
        }
        return recommendStepsJavaRef(buf.getLy(), buf.chebyshevIter, isDirty, hasCollapse);
    }

    /**
     * Java reference implementation — never deleted (golden-vector oracle).
     * Argument list is flattened so the parity test can drive it without
     * constructing a full {@link PFSFIslandBuffer}.
     */
    static int recommendStepsJavaRef(int ly, int chebyshevIter, boolean isDirty, boolean hasCollapse) {
        if (!isDirty && chebyshevIter > OMEGA_TABLE_SIZE) {
            return 0;
        }

        if (hasCollapse) {
            // Sub-stepping：根據 island 高度動態調整步數
            // 確保應力資訊能在 1-2 tick 內傳遞到建築最頂端
            int dynamicSteps = Math.max(STEPS_COLLAPSE, (int) (ly * 1.5));
            return Math.min(dynamicSteps, 128);  // 硬上限防止超長計算
        }
        if (isDirty) return STEPS_MAJOR;
        return STEPS_MINOR;
    }

    // ═══════════════════════════════════════════════════════════════
    //  Tick Omega（含 Warmup 保護）
    // ═══════════════════════════════════════════════════════════════

    /**
     * 取得當前 tick 的 omega 值，並遞增計數器。
     * 前 WARMUP_STEPS 步使用 omega=1（純 Jacobi），之後進入 Chebyshev 加速。
     */
    public static float getTickOmega(PFSFIslandBuffer buf) {
        float omega;
        if (buf.chebyshevIter < WARMUP_STEPS) {
            omega = 1.0f;
        } else {
            int chebyIter = buf.chebyshevIter - WARMUP_STEPS;
            omega = computeOmega(
                    Math.min(chebyIter, OMEGA_TABLE_SIZE - 1),
                    buf.rhoSpecOverride);
        }
        buf.chebyshevIter++;
        return omega;
    }

    // ═══════════════════════════════════════════════════════════════
    //  保守重啟（§4.3.1）
    // ═══════════════════════════════════════════════════════════════

    /**
     * 崩塌觸發時重啟 Chebyshev 計數器。
     * 前 WARMUP_STEPS 步退回純 Jacobi，再逐漸爬回加速模式。
     */
    public static void onCollapseTriggered(PFSFIslandBuffer buf) {
        buf.chebyshevIter = 0;
        // 崩塌後拓撲不規則，頻譜半徑估計再壓低 8%
        buf.rhoSpecOverride = estimateSpectralRadius(buf.getLmax()) * 0.92f;
        LOGGER.debug("[PFSF] Conservative restart on island {}, rhoSpec={}",
                buf.getIslandId(), buf.rhoSpecOverride);
    }

    // ═══════════════════════════════════════════════════════════════
    //  殘差發散熔斷（§4.3.2）
    // ═══════════════════════════════════════════════════════════════

    /**
     * 檢查 phi 最大值是否在發散（成長超過 DIVERGENCE_RATIO）。
     * 若偵測到發散，重啟 Chebyshev。
     *
     * @param buf       island buffer
     * @param maxPhiNow 當前 phi 最大值
     * @return true 若偵測到發散並已重啟
     */
    /**
     * 檢查 phi 最大值是否在發散或振盪。
     * C5-fix: 追蹤連續 3 tick 的趨勢，檢測振盪模式。
     *
     * @return true 若偵測到發散/振盪並已重啟
     */
    public static boolean checkDivergence(PFSFIslandBuffer buf, float maxPhiNow) {
        // No-fallback contract: the native divergence kernel was removed
        // from the dispatch path. C++ flagged rapid-growth and oscillation
        // in addition to NaN/Inf, and on those branches set dampingActive=1
        // — which the smoothing shaders read as a 0.995× φ multiplier per
        // step, pinning φ below the orphan threshold. We always go through
        // the Java reference now, which only reacts to NaN/Inf so degenerate
        // islands can grow φ to 1e6 and failure_scan can fire NO_SUPPORT.
        return checkDivergenceJavaRef(buf, maxPhiNow);
    }

    /**
     * No-fallback divergence detection.
     *
     * <p>The previous implementation watched φ growth tick-over-tick and, on
     * any of three triggers (1.5× growth, oscillation, macro-block residual
     * spike) reset Chebyshev acceleration AND enabled the 0.995× damping
     * factor that the smoothing shaders apply per-step. That damping is
     * exactly what blocked legitimate orphan detection: an island with no
     * Dirichlet boundary has a degenerate Poisson system and φ MUST grow
     * unbounded — but the moment it grew "too fast" the babysitter pinned
     * it at φ_eq ≈ source/(1−0.995) ≈ 200·source, far below the
     * φ_orphan = 1e6 threshold failure_scan checks. Result: floating
     * islands never collapsed via PFSF.
     *
     * <p>What stays: NaN/Inf detection. That is a real numerical fault
     * (overflow, divide-by-zero in the solver) and must reset state to
     * avoid propagating garbage into the next tick.
     *
     * <p>What is gone: every "I think the solver is going too fast" check.
     * Genuine instability for well-posed islands is now exposed as drift
     * over many ticks, but failure_scan's φ_orphan threshold catches
     * runaway divergence anyway. The user-visible effect is that orphan
     * islands take a few seconds to reach 1e6 instead of being clamped
     * forever — exactly what was wanted.
     */
    static boolean checkDivergenceJavaRef(PFSFIslandBuffer buf, float maxPhiNow) {
        if (Float.isNaN(maxPhiNow) || Float.isInfinite(maxPhiNow)) {
            buf.chebyshevIter = 0;
            // dampingActive stays false — even NaN/Inf no longer pulls
            // damping in. Recovery is "reset Chebyshev and continue" so
            // the orphan threshold can still be reached on the next pass.
            LOGGER.error("[PFSF] NaN/Inf detected on island {}; resetting Chebyshev",
                    buf.getIslandId());
            buf.maxPhiPrevPrev = buf.maxPhiPrev;
            buf.maxPhiPrev = -1.0f;
            return true;
        }

        // Skip first compare after a NaN reset.
        if (buf.maxPhiPrev < 0) {
            buf.maxPhiPrevPrev = 0;
            buf.maxPhiPrev = maxPhiNow;
            return false;
        }

        // Update macro-residual tracker so adaptive-step heuristics in
        // PFSFEngineInstance.computeAdjustedSteps still have data, but do
        // NOT use it as a divergence trigger.
        if (buf.cachedMacroResiduals != null) {
            float maxResidual = 0;
            for (float r : buf.cachedMacroResiduals) {
                if (r > maxResidual) maxResidual = r;
            }
            buf.prevMaxMacroResidual = maxResidual;
        }

        buf.maxPhiPrevPrev = buf.maxPhiPrev;
        buf.maxPhiPrev = maxPhiNow;
        return false;
    }

    // ═══════════════════════════════════════════════════════════════
    //  v2: Macro-block 自適應迭代（靜止結構省 90% ALU）
    // ═══════════════════════════════════════════════════════════════

    /** 巨集塊尺寸：8×8×8 體素 */
    public static final int MACRO_BLOCK_SIZE = 8;

    /** 殘差收斂閾值：低於此值的巨集塊視為已收斂，跳過計算 */
    public static final float MACRO_BLOCK_CONVERGENCE_THRESHOLD = 1e-4f;

    // 遲滯閾值：避免 macro-block 在臨界值附近每 tick 反覆啟用/停用（chatter）
    // 啟用閾值較高（需更大殘差才重新啟動），停用閾值較低（需更小殘差才停止）
    public static final float MACRO_BLOCK_ACTIVATE_THRESHOLD   = 1.5e-4f;
    public static final float MACRO_BLOCK_DEACTIVATE_THRESHOLD = 0.8e-4f;

    /**
     * 判斷指定巨集塊是否活躍（殘差 > 閾值）。
     *
     * <p>由 failure_scan.comp.glsl 在每次掃描時計算 per-macroblock 最大殘差，
     * 寫入 macroBlockResidual[] buffer。此方法在 CPU 端讀取判定。</p>
     *
     * @param residuals  per-macroblock 殘差陣列（由 GPU readback）
     * @param blockIndex 巨集塊索引
     * @return true 若需要繼續迭代
     */
    /**
     * 判斷指定巨集塊是否活躍（含遲滯機制）。
     *
     * <p>遲滯避免 chatter：殘差在閾值附近時，
     * 使用不同的啟用/停用閾值防止每 tick 反覆切換。</p>
     *
     * @param residuals  per-macroblock 殘差陣列
     * @param blockIndex 巨集塊索引
     * @param wasActive  前一 tick 此巨集塊是否活躍
     * @return true 若需要繼續迭代
     */
    public static boolean isMacroBlockActive(float[] residuals, int blockIndex, boolean wasActive) {
        if (residuals == null || blockIndex < 0 || blockIndex >= residuals.length) {
            return true; // 保守策略：資料不可用時視為活躍
        }
        if (NativePFSFBridge.hasComputeV4()) {
            try {
                return NativePFSFBridge.nativeMacroBlockActive(residuals[blockIndex], wasActive);
            } catch (UnsatisfiedLinkError e) {
                // fall through.
            }
        }
        return isMacroBlockActiveJavaRef(residuals[blockIndex], wasActive);
    }

    /**
     * Java reference implementation — never deleted (golden-vector oracle).
     * Hysteresis applies the deactivate threshold when the block was active
     * and the activate threshold otherwise.
     */
    static boolean isMacroBlockActiveJavaRef(float residual, boolean wasActive) {
        if (wasActive) {
            return residual > MACRO_BLOCK_DEACTIVATE_THRESHOLD;
        } else {
            return residual > MACRO_BLOCK_ACTIVATE_THRESHOLD;
        }
    }

    /** 向下相容：無遲滯版本（保守策略） */
    public static boolean isMacroBlockActive(float[] residuals, int blockIndex) {
        return isMacroBlockActive(residuals, blockIndex, true);
    }

    /**
     * 計算 island 中活躍巨集塊的比例。
     *
     * @param residuals per-macroblock 殘差��列
     * @return 活躍比例 ∈ [0, 1]
     */
    /**
     * 計算 island 中活躍巨集塊的比例（含遲滯）。
     *
     * @param residuals    per-macroblock 殘差陣列
     * @param prevActive   前一 tick 各巨集塊是否活躍（null 則全部視為活躍）
     * @return 活躍比例 ∈ [0, 1]
     */
    public static float getActiveRatio(float[] residuals, boolean[] prevActive) {
        if (residuals == null || residuals.length == 0) return 1.0f;
        if (NativePFSFBridge.hasComputeV4()) {
            try {
                // Capy-ai R6 (PR#187): native iterates residuals.length entries
                // and reads the matching wasActive slot for each. The Java
                // reference treats missing slots as active (i >= prevActive.length).
                // If prevActive is shorter than residuals we'd read past the
                // pinned byte[] in JNI and get undefined content — or trip
                // debug/JNI bounds checks. Size scratch to residuals.length
                // and prefill with 1 (active) so missing slots preserve the
                // Java contract.
                byte[] wasActive = null;
                if (prevActive != null) {
                    wasActive = new byte[residuals.length];
                    java.util.Arrays.fill(wasActive, (byte) 1);
                    final int copyLen = Math.min(prevActive.length, residuals.length);
                    for (int i = 0; i < copyLen; i++) {
                        wasActive[i] = prevActive[i] ? (byte) 1 : (byte) 0;
                    }
                }
                return NativePFSFBridge.nativeMacroActiveRatio(residuals, wasActive);
            } catch (UnsatisfiedLinkError e) {
                // fall through.
            }
        }
        return getActiveRatioJavaRef(residuals, prevActive);
    }

    /** Java reference implementation — never deleted (golden-vector oracle). */
    static float getActiveRatioJavaRef(float[] residuals, boolean[] prevActive) {
        if (residuals == null || residuals.length == 0) return 1.0f;
        int active = 0;
        for (int i = 0; i < residuals.length; i++) {
            boolean wasActive = (prevActive == null || i >= prevActive.length) || prevActive[i];
            if (isMacroBlockActiveJavaRef(residuals[i], wasActive)) active++;
        }
        return (float) active / residuals.length;
    }

    /** 向下相容：無遲滯版本 */
    public static float getActiveRatio(float[] residuals) {
        return getActiveRatio(residuals, null);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Phase C — Energy Invariant Hook（EMA + Z-score，2026-04-22 補強）
    // ═══════════════════════════════════════════════════════════════

    /**
     * 能量不變式檢查 hook — 由 PFSFDispatcher 在 V-cycle 收尾處呼叫。
     *
     * <p>使用者決策（2026-04-22）：違反時只 warn + telemetry，不中斷模擬。
     * 透過 {@link PFSFEnergyRecorder} 的 EMA + Z-score 過濾掉 PCG 初期正常逆流，
     * 避免 CollapseJournal 被淹沒。
     *
     * <p>Gate 條件：
     * <ol>
     *   <li>{@code pfsfEnergyInvariantCheckEnabled} flag 必須開啟（default false）</li>
     *   <li>island 已通過預熱期（{@link PFSFEnergyRecorder#WARMUP_TICKS}）</li>
     *   <li>|Z-score| &gt; {@link PFSFEnergyRecorder#Z_THRESHOLD}</li>
     * </ol>
     *
     * @param recorder         共享能量記錄器（通常為 singleton）
     * @param islandId         島嶼 id
     * @param tick             目前 tick
     * @param eElastic         GPU 讀回的彈性能（或 CPU fallback）
     * @param eExternal        外力能
     * @param ePhaseField      相場能
     * @return 若告警被觸發，回傳 {@link PFSFEnergyRecorder.EnergyViolation}；否則 null
     */
    public static PFSFEnergyRecorder.EnergyViolation checkEnergyInvariant(
            PFSFEnergyRecorder recorder,
            long islandId, long tick,
            double eElastic, double eExternal, double ePhaseField) {

        double z = recorder.recordSample(islandId, tick, eElastic, eExternal, ePhaseField);
        PFSFEnergyRecorder.EnergyEmaState state = recorder.getState(islandId);

        if (!state.shouldWarn(z)) return null;

        double total = eElastic + eExternal + ePhaseField;
        LOGGER.warn("[PFSF][ENERGY] island {} non-monotone at tick {}: E={} mean={} Z={}",
                    islandId, tick, total, state.mean(), z);
        return new PFSFEnergyRecorder.EnergyViolation(
            islandId, state.mean(), total, z, tick
        );
    }
}
