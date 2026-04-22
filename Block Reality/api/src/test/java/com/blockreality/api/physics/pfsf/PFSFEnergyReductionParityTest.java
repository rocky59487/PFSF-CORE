package com.blockreality.api.physics.pfsf;

import com.blockreality.api.physics.effective.EnergyEvaluatorCPU;
import com.blockreality.api.physics.effective.GraphEnergyFunctional;
import com.blockreality.api.physics.effective.MaterialCalibration;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase C — GPU/CPU 能量 reduction 對等性 + Kahan + EMA 不變式行為驗證。
 *
 * <p>沙箱環境無 Vulkan GPU，本測試不直接呼叫 GPU kernel。改以：
 * <ol>
 *   <li>CPU 內模擬「Kahan Summation + pair-wise tree reduction」路徑，
 *       驗證 Kahan 比 naive float32 累加有明顯更低誤差（對應 energy_reduce.comp.glsl 邏輯）</li>
 *   <li>直接驗證 {@link PFSFEnergyRecorder} 的 EMA + Z-score 行為</li>
 *   <li>驗證 {@link PFSFScheduler#checkEnergyInvariant} 的 warning gate 正確生效</li>
 * </ol>
 *
 * <p>實機 GPU parity（CPU vs shader bit-match，相對誤差 &lt; 1e-4）會在有 GPU 的
 * CI runner 上另外以 integration test 驗證，本單元測試保底 CPU reference 的數學正確。
 */
class PFSFEnergyReductionParityTest {

    private final GraphEnergyFunctional cpuEvaluator = new EnergyEvaluatorCPU();

    // ═══════════════════════════════════════════════════════════════
    //  Kahan vs Naive 累加誤差比較
    // ═══════════════════════════════════════════════════════════════

    /**
     * 模擬 energy_reduce.comp.glsl 的 Kahan 單 workgroup 累加路徑。
     * 對極端差距輸入（大值 + 海量小值），Kahan 相對 naive 應有顯著優勢。
     */
    @Test
    void kahanSummationBeatsNaiveOnExtremeScaleInput() {
        // 情境：1 個大值 1e8 + 10_000 個小值 1e-3
        //  truth = 1e8 + 10_000 × 1e-3 = 100_000_010
        //  naive float32 累加會丟失小值（因為 1e8 的尾數解析度 ≈ 8）
        int smallCount = 10_000;
        float bigVal = 1e8f;
        float smallVal = 1e-3f;
        double truth = (double) bigVal + smallCount * (double) smallVal;

        // naive
        float naive = bigVal;
        for (int i = 0; i < smallCount; i++) naive += smallVal;

        // Kahan（對應 shader 實作）
        float kSum = bigVal, kCmp = 0.0f;
        for (int i = 0; i < smallCount; i++) {
            float y = smallVal - kCmp;
            float t = kSum + y;
            kCmp = (t - kSum) - y;
            kSum = t;
        }
        float kFinal = kSum - kCmp;

        double naiveErr = Math.abs(truth - naive);
        double kahanErr = Math.abs(truth - kFinal);

        assertTrue(kahanErr < naiveErr,
            "Kahan 誤差 " + kahanErr + " 應小於 naive " + naiveErr);
        // float32 尾數提供約 7 位十進數精度；1e8 的 ULP ≈ 8，
        // Kahan 誤差應遠小於 naive 且在 float32 可表示的童差內（< 16）
        assertTrue(kahanErr < 16.0,
            "Kahan 累加誤差應在 float32 精度冇內（truth=" + truth +
            " kahan=" + kFinal + " naive=" + naive + ")");
    }

    /**
     * 確認 pair-wise 樹狀歸約（含 Kahan）在全隨機輸入下仍維持 &lt; 1e-6 相對誤差。
     */
    @Test
    void pairwiseKahanReductionKeepsPrecisionOnRandomInput() {
        int N = 1 << 16;
        Random r = new Random(0xDEAF);
        float[] vals = new float[N];
        double truth = 0.0;
        for (int i = 0; i < N; i++) {
            vals[i] = (r.nextFloat() - 0.5f) * 10.0f;
            truth += vals[i];
        }
        float kahanResult = simulatePairwiseKahanReduction(vals);
        double rel = Math.abs(truth - kahanResult) / (Math.abs(truth) + 1e-9);
        assertTrue(rel < 1e-6, "Pairwise Kahan 相對誤差應 < 1e-6, actual=" + rel);
    }

    /**
     * 模擬 energy_reduce.comp.glsl pass2 的 pair-wise Kahan 樹狀歸約。
     */
    private float simulatePairwiseKahanReduction(float[] vals) {
        int n = vals.length;
        float[] sum = new float[n];
        float[] cmp = new float[n];
        System.arraycopy(vals, 0, sum, 0, n);

        for (int stride = n / 2; stride > 0; stride >>= 1) {
            for (int i = 0; i < stride; i++) {
                float ps = sum[i + stride];
                float pcm = cmp[i + stride];
                float s = sum[i], c = cmp[i];
                float y1 = ps - c;
                float t1 = s + y1;
                c = (t1 - s) - y1;
                s = t1;
                float y2 = -pcm - c;
                float t2 = s + y2;
                c = (t2 - s) - y2;
                s = t2;
                sum[i] = s;
                cmp[i] = c;
            }
        }
        return sum[0] - cmp[0];
    }

    // ═══════════════════════════════════════════════════════════════
    //  EMA + Z-score 行為
    // ═══════════════════════════════════════════════════════════════

    @Test
    void warmupPeriodSuppressesWarnings() {
        PFSFEnergyRecorder rec = new PFSFEnergyRecorder();
        // 預熱期內即使丟入極端值也不應告警
        for (int t = 0; t < PFSFEnergyRecorder.WARMUP_TICKS; t++) {
            double z = rec.recordSample(42L, t, 100.0, 0.0, 0.0);
            PFSFEnergyRecorder.EnergyEmaState state = rec.getState(42L);
            assertFalse(state.shouldWarn(z),
                "預熱期 tick=" + t + " 不應告警，Z=" + z);
        }
    }

    @Test
    void postWarmupSmallFluctuationsDoNotWarn() {
        PFSFEnergyRecorder rec = new PFSFEnergyRecorder();
        // 預熱：平穩的 50.0
        for (int t = 0; t < PFSFEnergyRecorder.WARMUP_TICKS + 5; t++) {
            rec.recordSample(42L, t, 50.0 + (t % 2), 0.0, 0.0);
        }
        // 小幅波動（±2%）不應觸發 3σ 告警
        double z = rec.recordSample(42L, 100L, 51.0, 0.0, 0.0);
        PFSFEnergyRecorder.EnergyEmaState state = rec.getState(42L);
        assertFalse(state.shouldWarn(z),
            "小幅波動不應告警，Z=" + z);
    }

    @Test
    void postWarmupLargeSpikeTriggersWarning() {
        PFSFEnergyRecorder rec = new PFSFEnergyRecorder();
        // 預熱：交替 50.0/51.0 讓 sigma 積累到有意義水平（約 σ ≈ 0.45）
        for (int t = 0; t < PFSFEnergyRecorder.WARMUP_TICKS + 20; t++) {
            rec.recordSample(42L, t, (t % 2 == 0) ? 50.0 : 51.0, 0.0, 0.0);
        }
        // 突然跳到 10000（約 200x 均值，必然 >> 3σ）
        double z = rec.recordSample(42L, 100L, 10000.0, 0.0, 0.0);
        PFSFEnergyRecorder.EnergyEmaState state = rec.getState(42L);
        assertTrue(state.shouldWarn(z),
            "大幅跳動應告警，Z=" + z + " (threshold=" + PFSFEnergyRecorder.Z_THRESHOLD + ")");
    }

    @Test
    void schedulerHookReturnsNullOnHealthy() {
        PFSFEnergyRecorder rec = new PFSFEnergyRecorder();
        // 預熱：交替 10.0/10.1 讓 sigma 積累（約 σ≈0.05）
        for (int t = 0; t < PFSFEnergyRecorder.WARMUP_TICKS + 30; t++) {
            PFSFScheduler.checkEnergyInvariant(rec, 7L, t, (t % 2 == 0) ? 10.0 : 10.1, 0.0, 0.0);
        }
        // 1% 波動（dev ≈0.05 < 3σ ≈ 0.15）不應告警
        PFSFEnergyRecorder.EnergyViolation v =
            PFSFScheduler.checkEnergyInvariant(rec, 7L, 100L, 10.05, 0.0, 0.0);
        assertNull(v, "微小波動不應回傳 violation");
    }

    @Test
    void schedulerHookEmitsViolationOnSpike() {
        PFSFEnergyRecorder rec = new PFSFEnergyRecorder();
        // 預熱：交替 10.0/10.1 讓 sigma 積累（約 σ≈0.05）
        for (int t = 0; t < PFSFEnergyRecorder.WARMUP_TICKS + 30; t++) {
            PFSFScheduler.checkEnergyInvariant(
                rec, 7L, t, (t % 2 == 0) ? 10.0 : 10.1, 0.0, 0.0);
        }
        // 極端跳動：10x 內約4000x，必然 >> 3σ
        PFSFEnergyRecorder.EnergyViolation v =
            PFSFScheduler.checkEnergyInvariant(rec, 7L, 100L, 40000.0, 0.0, 0.0);
        assertNotNull(v, "極端跳動應產生 violation");
        assertEquals(7L, v.islandId());
        assertEquals(40000.0, v.eCurrent(), 1e-9);
        assertTrue(Math.abs(v.zScore()) > PFSFEnergyRecorder.Z_THRESHOLD);
    }

    @Test
    void perIslandStateIsIsolated() {
        PFSFEnergyRecorder rec = new PFSFEnergyRecorder();
        Random r = new Random(11L);
        // island 1: 平穩
        for (int t = 0; t < 20; t++) {
            rec.recordSample(1L, t, 10.0 + r.nextGaussian() * 0.05, 0.0, 0.0);
        }
        // island 2: 劇烈
        for (int t = 0; t < 20; t++) {
            rec.recordSample(2L, t, 100.0 * r.nextDouble(), 0.0, 0.0);
        }
        PFSFEnergyRecorder.EnergyEmaState s1 = rec.getState(1L);
        PFSFEnergyRecorder.EnergyEmaState s2 = rec.getState(2L);
        assertTrue(s1.stddev() < s2.stddev(),
            "兩 island 的 EMA 狀態應彼此獨立（s1.σ=" + s1.stddev() +
            " s2.σ=" + s2.stddev() + "）");
    }

    // ═══════════════════════════════════════════════════════════════
    //  與 CPU EnergyEvaluator 整合合理性
    // ═══════════════════════════════════════════════════════════════

    @Test
    void recorderAcceptsCpuEvaluatorOutput() {
        int L = 8;
        int N = L * L * L;
        Random r = new Random(0xABCDL);
        float[] phi = randArr(r, N, -1.0f, 1.0f);
        float[] sigma = randArr(r, N, 0.5f, 1.5f);
        float[] d = randArr(r, N, 0.0f, 0.2f);
        float[] h = randArr(r, N, 0.9f, 1.0f);
        float[] rho = randArr(r, N, -0.01f, 0.01f);

        MaterialCalibration calib = MaterialCalibration.defaultFor("test");
        GraphEnergyFunctional.EnergyBreakdown b =
            cpuEvaluator.evaluateBreakdown(phi, d, sigma, rho, h, L, L, L, calib);

        PFSFEnergyRecorder rec = new PFSFEnergyRecorder();
        rec.recordSample(99L, 0L, b.eElastic(), b.eExternal(), b.ePhaseField());
        PFSFEnergyRecorder.EnergySample s = rec.getLatestSample(99L);
        assertNotNull(s);
        assertEquals(b.eElastic(), s.eElastic(), 1e-12);
        assertEquals(b.total(), s.total(), 1e-12);
    }

    private static float[] randArr(Random r, int n, float lo, float hi) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = lo + r.nextFloat() * (hi - lo);
        return a;
    }
}
