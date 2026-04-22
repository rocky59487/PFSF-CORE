package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L1: Phase 1 stateless compute primitives — no Vulkan or GPU required.
 *
 * <p>These tests call C++ functions that run entirely on the CPU
 * ({@code pfsf_compute.h} Phase 1 group). If L0 passes but L1 fails,
 * the native library loaded but has a logic error in its compute kernels.</p>
 *
 * <p>L1-05 specifically validates the normalization contract documented in
 * CLAUDE.md "常見陷阱 #9": {@code rcomp} and {@code rtens} <em>must</em> be
 * divided by {@code sigmaMax} inside {@code nativeNormalizeSoA6}. If they are
 * not, the {@code failure_scan} shader compares phi (in normalized units) against
 * thresholds in raw MPa — the scales differ by ~20×, destroying failure detection.</p>
 */
@DisplayName("L1: Phase 1 Stateless Compute Primitives")
class L1_Phase1PrimitivesTest {

    @BeforeEach
    void requireComputeV1() {
        assumeTrue(NativePFSFBridge.isAvailable(),
                "libblockreality_pfsf not loaded — run L0 first");
        assumeTrue(NativePFSFBridge.hasComputeV1(),
                "compute.v1 absent in this native build — Phase 1 primitives unavailable");
    }

    // ─── Wind pressure ───────────────────────────────────────────────────────

    @Test
    @DisplayName("L1-01: windPressureSource ≈ Bernoulli ½ρv² (±20%)")
    void windPressureBernoulli() {
        float v   = 10.0f;    // m/s
        float rho = 1.225f;   // kg/m³ (sea-level air)
        float expected = 0.5f * rho * v * v; // 61.25 Pa

        float result = NativePFSFBridge.nativeWindPressureSource(v, rho, /*exposed=*/ true);
        System.out.println("[L1] windPressureSource=" + result + "  expected≈" + expected);
        assertTrue(result > 0,
                "Wind pressure must be positive, got: " + result);
        assertEquals(expected, result, expected * 0.20f,
                "Wind pressure deviates >20% from Bernoulli ½ρv²=" + expected + ", got " + result);
    }

    @Test
    @DisplayName("L1-02: windPressureSource(v=0) = 0")
    void windPressureZeroWind() {
        float result = NativePFSFBridge.nativeWindPressureSource(0.0f, 1.225f, true);
        assertEquals(0.0f, result, 1e-5f, "Zero wind must produce zero pressure");
    }

    // ─── Timoshenko moment factor ─────────────────────────────────────────────

    @Test
    @DisplayName("L1-03: timoshenkoMomentFactor 介於 0.5 ~ 2.0")
    void timoshenkoMomentFactorRange() {
        // 1×1 cross-section, arm=4 blocks, concrete: E=30GPa, ν=0.2
        float factor = NativePFSFBridge.nativeTimoshenkoMomentFactor(
                1.0f, 1.0f, 4, 30.0f, 0.2f);
        System.out.println("[L1] timoshenkoMomentFactor=" + factor);
        assertTrue(factor > 0.5f && factor < 2.0f,
                "Timoshenko factor out of sensible range [0.5, 2.0]: " + factor);
    }

    // ─── SoA-6 normalization ──────────────────────────────────────────────────

    @Test
    @DisplayName("L1-04: normalizeSoA6 sigmaMax > 0 且正規化後 max(cond) ≈ 1.0")
    void normalizeSoA6SigmaMax() {
        int N = 4;
        float[] source = { 10.0f, 5.0f, 3.0f, 1.0f };
        float[] rcomp  = { 20.0f, 20.0f, 20.0f, 20.0f };
        float[] rtens  = {  2.0f,  2.0f,  2.0f,  2.0f };
        float[] cond   = new float[6 * N];
        for (int i = 0; i < N; i++) for (int d = 0; d < 6; d++) cond[d * N + i] = 15.0f;

        float sigmaMax = NativePFSFBridge.nativeNormalizeSoA6(source, rcomp, rtens, cond, null, N);
        System.out.println("[L1] normalizeSoA6 sigmaMax=" + sigmaMax);

        assertTrue(sigmaMax > 0, "sigmaMax must be positive, got: " + sigmaMax);

        float maxCond = 0;
        for (float c : cond) maxCond = Math.max(maxCond, c);
        assertEquals(1.0f, maxCond, 0.01f,
                "After normalize, max(conductivity) should be 1.0, got: " + maxCond);
    }

    @Test
    @DisplayName("L1-05: normalizeSoA6 同步正規化 rcomp/rtens（CLAUDE.md 陷阱 #9）")
    void normalizeSoA6RcompRtensScaled() {
        int N = 2;
        // Set sigmaMax = 30 by using conductivity = 30
        float[] source = { 1.0f, 1.0f };
        float[] rcomp  = { 30.0f, 30.0f };
        float[] rtens  = {  3.0f,  3.0f };
        float[] cond   = new float[6 * N];
        for (int i = 0; i < N; i++) for (int d = 0; d < 6; d++) cond[d * N + i] = 30.0f;

        float sigmaMax = NativePFSFBridge.nativeNormalizeSoA6(source, rcomp, rtens, cond, null, N);

        System.out.println("[L1] rcomp normalization check: sigmaMax=" + sigmaMax
                + "  rcomp[0]=" + rcomp[0] + "  rtens[0]=" + rtens[0]);

        assertEquals(30.0f, sigmaMax, 1.0f,
                "Expected sigmaMax ≈ 30.0, got: " + sigmaMax);
        assertEquals(1.0f, rcomp[0], 0.02f,
                "[TRAP #9] rcomp was NOT normalized: expected 1.0 (30/sigmaMax), got " + rcomp[0] +
                "\nThis means failure_scan compares phi (normalized) vs rcomp (raw MPa) — physics broken!");
        assertEquals(0.1f, rtens[0], 0.01f,
                "[TRAP #9] rtens was NOT normalized: expected 0.1 (3/sigmaMax), got " + rtens[0]);
    }

    // ─── Wind bias ────────────────────────────────────────────────────────────

    @Test
    @DisplayName("L1-06: applyWindBias 修改 conductivity（不全為原始值）")
    void applyWindBiasModifiesConductivity() {
        int N = 8;
        float[] cond = new float[6 * N];
        for (int i = 0; i < 6 * N; i++) cond[i] = 1.0f;
        float[] original = cond.clone();

        assertDoesNotThrow(() ->
                NativePFSFBridge.nativeApplyWindBias(cond, N, 1.0f, 0.0f, 0.0f, 1.5f));

        boolean anyChanged = false;
        for (int i = 0; i < 6 * N; i++) {
            if (Math.abs(cond[i] - original[i]) > 1e-6f) {
                anyChanged = true;
                break;
            }
        }
        assertTrue(anyChanged, "applyWindBias must modify at least one conductivity entry");
    }
}
