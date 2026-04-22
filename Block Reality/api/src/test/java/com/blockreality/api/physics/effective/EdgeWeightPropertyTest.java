package com.blockreality.api.physics.effective;

import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase A — {@link EdgeWeight} 四性質驗證（10k 隨機輸入）。
 *
 * <p>驗證 {@link DefaultEdgeWeight} 在各種隨機輸入下都滿足：
 * <ol>
 *   <li>Symmetry：w(i,j) == w(j,i)</li>
 *   <li>Positivity：w ≥ 0</li>
 *   <li>Monotonicity in damage：d ↑ → w ↓</li>
 *   <li>Extreme cases：d=1 → w=0；d=0, h=1 → w=σ</li>
 * </ol>
 *
 * <p>Locality 由 caller 保證，本測試不驗證（範疇外）。
 */
class EdgeWeightPropertyTest {

    private static final int TRIALS = 10_000;
    private static final long SEED  = 0xCA11B0A1DL;
    private static final double EPS = 1e-9;

    private final EdgeWeight edgeWeight = DefaultEdgeWeight.INSTANCE;
    private final MaterialCalibration calibConcrete =
        MaterialCalibration.defaultFor("concrete");
    private final MaterialCalibration calibSteel = MaterialCalibration.legacy(
        MaterialCalibration.SCHEMA_V1, "steel", 1,
        MaterialCalibration.BoundaryProfile.ANCHORED_BOTTOM,
        1.0, 0.5, 1.5, MaterialCalibration.DEFAULT_PHASE_FIELD_EXPONENT_STEEL,
        0L, "default"
    );

    @Test
    void symmetry_randomInputs() {
        Random r = new Random(SEED);
        for (int t = 0; t < TRIALS; t++) {
            double sigma = r.nextDouble() * 10.0;
            double dI = r.nextDouble();
            double dJ = r.nextDouble();
            double hI = r.nextDouble();
            double hJ = r.nextDouble();
            MaterialCalibration calib = (t % 2 == 0) ? calibConcrete : calibSteel;

            double wIJ = edgeWeight.weight(sigma, dI, dJ, hI, hJ, calib);
            double wJI = edgeWeight.weight(sigma, dJ, dI, hJ, hI, calib);

            assertEquals(wIJ, wJI, EPS,
                "Symmetry violated at trial " + t +
                " (sigma=" + sigma + " dI=" + dI + " dJ=" + dJ +
                " hI=" + hI + " hJ=" + hJ + ")");
        }
    }

    @Test
    void positivity_randomInputs() {
        Random r = new Random(SEED + 1);
        for (int t = 0; t < TRIALS; t++) {
            double sigma = r.nextDouble() * 10.0;
            double dI = r.nextDouble();
            double dJ = r.nextDouble();
            double hI = r.nextDouble();
            double hJ = r.nextDouble();

            double w = edgeWeight.weight(sigma, dI, dJ, hI, hJ, calibConcrete);
            assertTrue(w >= 0.0,
                "Positivity violated at trial " + t + ": w=" + w +
                " (sigma=" + sigma + " dI=" + dI + " dJ=" + dJ + ")");
            assertFalse(Double.isNaN(w), "NaN at trial " + t);
            assertFalse(Double.isInfinite(w), "Inf at trial " + t);
        }
    }

    @Test
    void monotonicityInDamageI_randomInputs() {
        Random r = new Random(SEED + 2);
        int violations = 0;
        for (int t = 0; t < TRIALS; t++) {
            double sigma = r.nextDouble() * 10.0 + 0.01; // 避免 σ=0 退化
            double dILow  = r.nextDouble() * 0.5;
            double dIHigh = dILow + r.nextDouble() * (1.0 - dILow);
            double dJ = r.nextDouble();
            double hI = r.nextDouble() * 0.9 + 0.1;
            double hJ = r.nextDouble() * 0.9 + 0.1;
            MaterialCalibration calib = (t % 2 == 0) ? calibConcrete : calibSteel;

            double wLow  = edgeWeight.weight(sigma, dILow,  dJ, hI, hJ, calib);
            double wHigh = edgeWeight.weight(sigma, dIHigh, dJ, hI, hJ, calib);

            // dI 上升 → w 單調下降（允許相等：dILow == dIHigh 時）
            if (wHigh > wLow + EPS) {
                violations++;
                fail("Monotonicity violated at trial " + t +
                    " dILow=" + dILow + " dIHigh=" + dIHigh +
                    " wLow=" + wLow + " wHigh=" + wHigh);
            }
        }
        assertEquals(0, violations);
    }

    @Test
    void monotonicityInDamageJ_randomInputs() {
        Random r = new Random(SEED + 3);
        for (int t = 0; t < TRIALS; t++) {
            double sigma = r.nextDouble() * 10.0 + 0.01;
            double dI = r.nextDouble();
            double dJLow  = r.nextDouble() * 0.5;
            double dJHigh = dJLow + r.nextDouble() * (1.0 - dJLow);
            double hI = r.nextDouble() * 0.9 + 0.1;
            double hJ = r.nextDouble() * 0.9 + 0.1;

            double wLow  = edgeWeight.weight(sigma, dI, dJLow,  hI, hJ, calibConcrete);
            double wHigh = edgeWeight.weight(sigma, dI, dJHigh, hI, hJ, calibConcrete);

            assertTrue(wHigh <= wLow + EPS,
                "Monotonicity in dJ violated at trial " + t);
        }
    }

    @Test
    void extremeCase_fullDamage_zeroWeight() {
        // d = 1 任一端 → w 必為 0（連線移除）
        double w1 = edgeWeight.weight(5.0, 1.0, 0.5, 1.0, 1.0, calibConcrete);
        double w2 = edgeWeight.weight(5.0, 0.5, 1.0, 1.0, 1.0, calibConcrete);
        double w3 = edgeWeight.weight(5.0, 1.0, 1.0, 1.0, 1.0, calibConcrete);

        assertEquals(0.0, w1, EPS, "d_i = 1 應使 w = 0");
        assertEquals(0.0, w2, EPS, "d_j = 1 應使 w = 0");
        assertEquals(0.0, w3, EPS, "雙端 d = 1 應使 w = 0");
    }

    @Test
    void extremeCase_noDamage_fullCuring_returnsSigma() {
        // d = 0 且 h = 1 → w = σ
        double sigma = 7.5;
        double w = edgeWeight.weight(sigma, 0.0, 0.0, 1.0, 1.0, calibConcrete);
        assertEquals(sigma, w, EPS, "d=0 h=1 時 w 應等於 σ");
    }

    @Test
    void extremeCase_noCuring_zeroWeight() {
        double w1 = edgeWeight.weight(5.0, 0.0, 0.0, 0.0, 1.0, calibConcrete);
        double w2 = edgeWeight.weight(5.0, 0.0, 0.0, 1.0, 0.0, calibConcrete);
        assertEquals(0.0, w1, EPS, "h_i = 0 應使 w = 0");
        assertEquals(0.0, w2, EPS, "h_j = 0 應使 w = 0");
    }

    @Test
    void negativeInputs_clampedToZero() {
        // 異常輸入不應破壞 positivity
        double w = edgeWeight.weight(-1.0, -0.1, -0.1, -0.1, -0.1, calibConcrete);
        assertTrue(w >= 0.0, "負輸入應被下夾，positivity 必須保持");
        assertEquals(0.0, w, EPS, "σ=0 效應 → w=0");
    }

    @Test
    void damageAbove1_clampedToZero() {
        double w = edgeWeight.weight(5.0, 1.5, 0.0, 1.0, 1.0, calibConcrete);
        assertEquals(0.0, w, EPS, "d > 1 應 clamp 到 1 → w = 0");
    }

    @Test
    void phaseFieldExponentMatters() {
        // p=2 (concrete) 與 p=4 (steel) 在 d=0.5 時應給出不同衰減
        double wConcrete = edgeWeight.weight(1.0, 0.5, 0.0, 1.0, 1.0, calibConcrete);
        double wSteel    = edgeWeight.weight(1.0, 0.5, 0.0, 1.0, 1.0, calibSteel);

        // (1-0.5)^2 = 0.25；(1-0.5)^4 = 0.0625
        assertEquals(0.25, wConcrete, EPS, "p=2: (1-0.5)^2 = 0.25");
        assertEquals(0.0625, wSteel,  EPS, "p=4: (1-0.5)^4 = 0.0625");
        assertTrue(wSteel < wConcrete, "較高 p 應產生較銳利的衰減（wSteel < wConcrete）");
    }
}
