package com.blockreality.api.physics.effective;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase E — {@link CalibrationRunner} 閉合解驗證。
 *
 * <p>驗證三組內建解析解的量綱與邊界情境符合教科書公式，並測試 L2 error /
 * grid search 工具的數學正確性。
 *
 * <h2>教科書引用</h2>
 * <ul>
 *   <li>Gere &amp; Timoshenko, Mechanics of Materials §9.3（懸臂樑撓度）</li>
 *   <li>Hibbeler, Structural Analysis §10.5（拱橋）</li>
 *   <li>Eurocode 2 / AISC Steel Manual（材料常數）</li>
 * </ul>
 */
class CalibrationRunnerTest {

    private static final double EPS = 1e-6;
    private static final double REL_EPS = 1e-3; // 0.1% 相對誤差

    // ═══════════════════════════════════════════════════════════════
    //  懸臂樑
    // ═══════════════════════════════════════════════════════════════

    @Test
    void cantileverMaxStress_matchesTextbookFormula() {
        // 標準 C30 混凝土懸臂: L=2m, h=0.2m, ρ=2400
        //   σ_max = 6 × 2400 × 9.81 × 4 / 0.04 = 14,126,400 Pa ≈ 14.13 MPa
        double rho = 2400.0;
        double L = 2.0;
        double h = 0.2;
        double sigma = CalibrationRunner.cantileverMaxStress(rho, L, h);
        double expected = 6.0 * rho * 9.81 * L * L / (h * h);
        assertEquals(expected, sigma, EPS, "cantileverMaxStress 手算核對");
        assertTrue(sigma > 0, "應力必須為正");
    }

    @Test
    void cantileverTipDeflection_scalesAsL4() {
        // δ ∝ L⁴ 的量綱驗證：L 加倍 → δ 16 倍
        double rho = 2400.0;
        double h = 0.2;
        double b = 0.3;
        double E = 30e9;
        double d1 = CalibrationRunner.cantileverTipDeflection(rho, 1.0, h, b, E);
        double d2 = CalibrationRunner.cantileverTipDeflection(rho, 2.0, h, b, E);
        double ratio = d2 / d1;
        assertEquals(16.0, ratio, ratio * REL_EPS, "δ ∝ L⁴ 應滿足 2^4=16");
    }

    @Test
    void cantileverMoment_atRootEqualsHalfWLsq() {
        double rho = 2400.0, L = 3.0, b = 0.3, h = 0.2;
        double M_root = CalibrationRunner.cantileverMoment(rho, L, b, h, 0.0);
        double w = rho * 9.81 * b * h;
        double expected = 0.5 * w * L * L;
        assertEquals(expected, M_root, expected * REL_EPS, "根部彎矩 = wL²/2");
    }

    @Test
    void cantileverMoment_atTipIsZero() {
        double M_tip = CalibrationRunner.cantileverMoment(2400.0, 3.0, 0.3, 0.2, 3.0);
        assertEquals(0.0, M_tip, EPS, "自由端彎矩應為 0");
    }

    @Test
    void cantileverMoment_outOfRangeThrows() {
        assertThrows(IllegalArgumentException.class,
            () -> CalibrationRunner.cantileverMoment(2400.0, 3.0, 0.3, 0.2, -1.0));
        assertThrows(IllegalArgumentException.class,
            () -> CalibrationRunner.cantileverMoment(2400.0, 3.0, 0.3, 0.2, 3.5));
    }

    // ═══════════════════════════════════════════════════════════════
    //  簡支樑
    // ═══════════════════════════════════════════════════════════════

    @Test
    void simplySupportedMaxStress_matchesFormula() {
        double w = 1000.0, L = 4.0, h = 0.3;
        double sigma = CalibrationRunner.simplySupportedMaxStress(w, L, h);
        double expected = 3.0 * w * L * L / (2.0 * h * h);
        assertEquals(expected, sigma, EPS);
    }

    @Test
    void simplySupportedCenterDeflection_scalesAsL4() {
        double w = 1000.0, h = 0.2, b = 0.3, E = 30e9;
        double d1 = CalibrationRunner.simplySupportedCenterDeflection(w, 1.0, h, b, E);
        double d2 = CalibrationRunner.simplySupportedCenterDeflection(w, 2.0, h, b, E);
        assertEquals(16.0, d2 / d1, d2/d1 * REL_EPS);
    }

    // ═══════════════════════════════════════════════════════════════
    //  半圓拱
    // ═══════════════════════════════════════════════════════════════

    @Test
    void semiArchCrownMoment_matchesClosedForm() {
        double w = 500.0, R = 8.0;
        double M = CalibrationRunner.semiArchCrownMoment(w, R);
        double expected = w * R * R * (1.0 - 2.0 / Math.PI);
        assertEquals(expected, M, EPS);
        // 係數 1 − 2/π ≈ 0.3634 → M > 0 但遠小於 wR²
        assertTrue(M < w * R * R, "M_crown 應 < wR²");
        assertTrue(M > 0, "M_crown 應為正");
    }

    @Test
    void semiArchHorizontalThrust_inverselyProportionalToHeight() {
        double w = 500.0, R = 8.0;
        double H1 = CalibrationRunner.semiArchHorizontalThrust(w, R, 4.0);
        double H2 = CalibrationRunner.semiArchHorizontalThrust(w, R, 8.0);
        assertEquals(2.0, H1 / H2, REL_EPS * 2.0, "H ∝ 1/h_arch");
    }

    @Test
    void semiArchCrownAxial_combinesThrustAndLoad() {
        double w = 500.0, R = 8.0, h_arch = 8.0;
        double N = CalibrationRunner.semiArchCrownAxial(w, R, h_arch);
        double H = CalibrationRunner.semiArchHorizontalThrust(w, R, h_arch);
        assertEquals(H + w * R, N, EPS);
    }

    // ═══════════════════════════════════════════════════════════════
    //  L2 error
    // ═══════════════════════════════════════════════════════════════

    @Test
    void l2Error_zeroWhenExactMatch() {
        double[] ref = {1.0, 2.0, 3.0, 4.0};
        float[] dis = {1.0f, 2.0f, 3.0f, 4.0f};
        assertEquals(0.0, CalibrationRunner.l2Error(dis, ref), 1e-6);
    }

    @Test
    void l2Error_relativeToReferenceMagnitude() {
        double[] ref = {10.0, 10.0};
        float[] dis = {11.0f, 9.0f};
        // Σ Δ² = 1 + 1 = 2；Σ ref² = 200；err = √(2/200) = 0.1
        double e = CalibrationRunner.l2Error(dis, ref);
        assertEquals(0.1, e, 1e-6);
    }

    @Test
    void l2Error_lengthMismatchThrows() {
        assertThrows(IllegalArgumentException.class,
            () -> CalibrationRunner.l2Error(new float[2], new double[3]));
    }

    // ═══════════════════════════════════════════════════════════════
    //  Grid search
    // ═══════════════════════════════════════════════════════════════

    @Test
    void gridSearch_findsMinimumOverCandidates() {
        // 目標最小點在 (0.35, 0.15)，用人造 quadratic: err = (e-0.35)² + (c-0.15)²
        double[] edges   = {0.20, 0.35, 0.50};
        double[] corners = {0.05, 0.15, 0.25};
        CalibrationRunner.GridSearchResult r = CalibrationRunner.gridSearch(
            edges, corners,
            (e, c) -> (e - 0.35) * (e - 0.35) + (c - 0.15) * (c - 0.15)
        );
        assertEquals(0.35, r.bestEdgePenalty(), 1e-9);
        assertEquals(0.15, r.bestCornerPenalty(), 1e-9);
        assertEquals(0.0, r.bestError(), 1e-9);
    }

    @Test
    void gridSearch_rejectsEmptyCandidates() {
        assertThrows(IllegalArgumentException.class,
            () -> CalibrationRunner.gridSearch(new double[0], new double[]{0.1},
                (e, c) -> 0.0));
    }

    // ═══════════════════════════════════════════════════════════════
    //  整合驗證：解析解 → L2 比對 → grid search
    // ═══════════════════════════════════════════════════════════════

    @Test
    void analyticMomentProfile_matchesCantileverFormula() {
        // 驗證：懸臂樑彎矩從根部到自由端單調遞減
        double rho = 2400.0, L = 4.0, b = 0.3, h = 0.2;
        double[] xs = {0.0, 1.0, 2.0, 3.0, 4.0};
        double prev = Double.POSITIVE_INFINITY;
        for (double x : xs) {
            double M = CalibrationRunner.cantileverMoment(rho, L, b, h, x);
            assertTrue(M <= prev, "彎矩應從根部到自由端單調遞減：x=" + x + " M=" + M);
            prev = M;
        }
    }
}
