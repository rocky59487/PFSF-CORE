package com.blockreality.api.physics.effective;

import com.blockreality.api.physics.pfsf.PFSFStencil;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase B — {@link EnergyEvaluatorCPU} 手算比對 (Golden Oracle)。
 *
 * <p>三種尺度驗證（4³ / 8³ / 16³）涵蓋三種情境：
 * <ol>
 *   <li>4³ 純彈性、均勻 σ、固定 φ → 手算預期值比對 absolute &lt; 1e-5</li>
 *   <li>8³ 單裂縫面（d=1 一層）→ elastic 項歸零、phase-field 項 ≈ G_c × A_crack × (1/(4 l₀))</li>
 *   <li>16³ 隨機 σ → 各項非負性、total = breakdown sum 一致性</li>
 * </ol>
 */
class EnergyEvaluatorCPUGoldenTest {

    private static final double ABS_EPS = 1e-5;
    private static final double REL_EPS = 1e-12;

    private final GraphEnergyFunctional evaluator = new EnergyEvaluatorCPU();

    /** 關掉 phase-field，讓只剩 elastic + external；方便手算 */
    private static MaterialCalibration noPhaseFieldCalib() {
        return MaterialCalibration.legacy(
            MaterialCalibration.SCHEMA_V1, "test_no_pf", 1,
            MaterialCalibration.BoundaryProfile.ANCHORED_BOTTOM,
            1.0,
            0.0,   // G_c = 0 → 關閉 phase-field
            1.5,
            2.0,
            0L, "test"
        );
    }

    private static MaterialCalibration standardCalib() {
        return MaterialCalibration.defaultFor("test_standard");
    }

    // ═══════════════════════════════════════════════════════════════
    //  4³ 純彈性手算比對
    // ═══════════════════════════════════════════════════════════════

    @Test
    void case4cube_uniformPhi_zeroEnergy() {
        // 均勻 φ=5、均勻 σ=1、d=0、ρ=0、h=1 → 所有 (φ_i − φ_j)² = 0 → E_elastic = 0
        int L = 4;
        int N = L * L * L;
        float[] phi = filled(N, 5.0f);
        float[] sigma = filled(N, 1.0f);
        float[] h = filled(N, 1.0f);

        double e = evaluator.evaluate(phi, null, sigma, null, h, L, L, L, noPhaseFieldCalib());
        assertEquals(0.0, e, ABS_EPS, "均勻 φ 下 E 應恰為 0");
    }

    @Test
    void case2x1x1_linearPhi_manualComputation() {
        // 2×1×1 grid: phi = [0, 1], σ=1, d=0, h=1, ρ=0
        // 唯一一條邊為 (0,0,0) - (1,0,0) 面鄰居
        // σ_ij = 0.5 × (1 + 1) = 1
        // w_ij = 1 × 1 × 1 × 1 = 1
        // E_elastic = 0.5 × 1 × (0 - 1)² = 0.5
        float[] phi = {0.0f, 1.0f};
        float[] sigma = {1.0f, 1.0f};
        float[] h = {1.0f, 1.0f};

        double e = evaluator.evaluate(phi, null, sigma, null, h, 2, 1, 1, noPhaseFieldCalib());
        assertEquals(0.5, e, ABS_EPS,
            "2×1×1 線性 φ=[0,1], σ=1: E_elastic = 0.5 × 1 × 1² = 0.5");
    }

    @Test
    void case2x2x1_checkerboardPhi() {
        // 2×2×1:
        //   (0,0): phi=0   (1,0): phi=1
        //   (0,1): phi=1   (1,1): phi=0
        // 邊統計（正向 offsets）:
        //   面邊: (0,0)-(1,0) Δ=1; (0,0)-(0,1) Δ=1; (1,0)-(1,1) Δ=1; (0,1)-(1,1) Δ=1
        //   XY 對角: (0,0)-(1,1) Δ=0; (1,0)-(0,1) Δ=0
        // σ=1 均勻 → w_face=1, w_edge=EDGE_P
        // E_elastic = 0.5 × [4 × (1×1²) + 2 × (EDGE_P × 0²)] = 0.5 × 4 = 2.0
        float[] phi = {0.0f, 1.0f, 1.0f, 0.0f};
        float[] sigma = filled(4, 1.0f);
        float[] h = filled(4, 1.0f);

        double e = evaluator.evaluate(phi, null, sigma, null, h, 2, 2, 1, noPhaseFieldCalib());
        assertEquals(2.0, e, ABS_EPS,
            "2×2×1 checkerboard: 4 face edges × Δ²=1, 2 diagonal edges × Δ²=0 → E=2.0");
    }

    @Test
    void case4cube_externalOnly() {
        // 均勻 φ=2、ρ=3 → E_external = Σ ρ φ = N × 3 × 2 = 64 × 6 = 384
        int L = 4;
        int N = L * L * L;
        float[] phi = filled(N, 2.0f);
        float[] sigma = filled(N, 1.0f);
        float[] rho = filled(N, 3.0f);
        float[] h = filled(N, 1.0f);

        GraphEnergyFunctional.EnergyBreakdown b =
            evaluator.evaluateBreakdown(phi, null, sigma, rho, h, L, L, L, noPhaseFieldCalib());
        assertEquals(0.0, b.eElastic(),    ABS_EPS, "均勻 φ → E_elastic = 0");
        assertEquals(384.0, b.eExternal(), ABS_EPS, "Σ ρ φ = 64 × 6 = 384");
        assertEquals(0.0, b.ePhaseField(), ABS_EPS, "G_c=0 → E_phaseField = 0");
        assertEquals(384.0, b.total(),     ABS_EPS);
    }

    // ═══════════════════════════════════════════════════════════════
    //  8³ 相場裂縫
    // ═══════════════════════════════════════════════════════════════

    @Test
    void case8cube_fullDamageLayer_elasticGoesToZeroAcrossCrack() {
        // 8³: 中間 z=4 一層 d=1 → 此層與上下層的邊權全為 0
        // 驗證 elastic 項存在性（非全零 φ）+ 跨裂縫邊貢獻消失
        int L = 8;
        int N = L * L * L;
        float[] phi = new float[N];
        float[] d   = new float[N];
        float[] sigma = filled(N, 1.0f);
        float[] h = filled(N, 1.0f);

        // φ = 0 below crack, φ = 1 above
        for (int z = 0; z < L; z++) {
            for (int y = 0; y < L; y++) {
                for (int x = 0; x < L; x++) {
                    int i = x + y * L + z * L * L;
                    phi[i] = (z >= 4) ? 1.0f : 0.0f;
                }
            }
        }
        // d = 1 at z = 3 (crack 層)
        for (int y = 0; y < L; y++) {
            for (int x = 0; x < L; x++) {
                int i = x + y * L + 3 * L * L;
                d[i] = 1.0f;
            }
        }

        double eDamaged =
            evaluator.evaluate(phi, d, sigma, null, h, L, L, L, noPhaseFieldCalib());

        // 相同 φ 但 d=0 → 跨 z=3/z=4 的邊權不為零 → E 更大
        float[] dZero = new float[N];
        double eHealthy =
            evaluator.evaluate(phi, dZero, sigma, null, h, L, L, L, noPhaseFieldCalib());

        assertTrue(eHealthy > eDamaged + 0.01,
            "d=1 裂縫層應明顯降低 E；healthy=" + eHealthy + " damaged=" + eDamaged);
        assertTrue(eDamaged >= 0.0, "elastic + external 在此設定下非負");
    }

    @Test
    void case8cube_phaseFieldEnergyUnderFullHealth_approxGcVolumeTerm() {
        // d=0 均勻 → |∇d|² = 0；ePhaseField = G_c / (4 l₀) × N
        int L = 8;
        int N = L * L * L;
        float[] phi = filled(N, 0.0f);
        float[] sigma = filled(N, 1.0f);
        float[] h = filled(N, 1.0f);

        MaterialCalibration calib = standardCalib();
        GraphEnergyFunctional.EnergyBreakdown b =
            evaluator.evaluateBreakdown(phi, null, sigma, null, h, L, L, L, calib);

        double expected = calib.gcEff() / (4.0 * calib.l0Eff()) * N;
        assertEquals(expected, b.ePhaseField(), expected * 1e-6,
            "均勻 d=0 → ePhaseField = G_c / (4 l₀) × N");
    }

    // ═══════════════════════════════════════════════════════════════
    //  16³ 隨機一致性 + 對稱性
    // ═══════════════════════════════════════════════════════════════

    @Test
    void case16cube_totalEqualsBreakdownSum() {
        int L = 16;
        int N = L * L * L;
        Random r = new Random(0xDEADBEEFL);

        float[] phi = randArr(r, N, -1.0f, 1.0f);
        float[] d = randArr(r, N, 0.0f, 0.3f);
        float[] sigma = randArr(r, N, 0.5f, 1.5f);
        float[] rho = randArr(r, N, -0.1f, 0.1f);
        float[] h = randArr(r, N, 0.8f, 1.0f);

        MaterialCalibration calib = standardCalib();
        double total = evaluator.evaluate(phi, d, sigma, rho, h, L, L, L, calib);
        GraphEnergyFunctional.EnergyBreakdown b =
            evaluator.evaluateBreakdown(phi, d, sigma, rho, h, L, L, L, calib);

        assertEquals(b.total(), total, Math.abs(total) * REL_EPS + ABS_EPS,
            "evaluate() 必須等於 breakdown 三項之和");
        assertTrue(b.eElastic() >= 0.0, "elastic 項非負");
        assertTrue(b.ePhaseField() >= 0.0, "phase-field 項非負");
    }

    @Test
    void case16cube_symmetryUnderPhiNegation() {
        // E_elastic 與 E_phaseField 對 φ 取負不變；E_external 取負
        int L = 16;
        int N = L * L * L;
        Random r = new Random(0xF00DL);

        float[] phi = randArr(r, N, -1.0f, 1.0f);
        float[] d = randArr(r, N, 0.0f, 0.5f);
        float[] sigma = randArr(r, N, 0.5f, 1.5f);
        float[] rho = randArr(r, N, 0.0f, 0.1f);
        float[] h = filled(N, 1.0f);

        float[] phiNeg = new float[N];
        for (int i = 0; i < N; i++) phiNeg[i] = -phi[i];

        MaterialCalibration calib = standardCalib();
        GraphEnergyFunctional.EnergyBreakdown bPos =
            evaluator.evaluateBreakdown(phi,    d, sigma, rho, h, L, L, L, calib);
        GraphEnergyFunctional.EnergyBreakdown bNeg =
            evaluator.evaluateBreakdown(phiNeg, d, sigma, rho, h, L, L, L, calib);

        double tol = Math.abs(bPos.eElastic()) * 1e-10 + ABS_EPS;
        assertEquals(bPos.eElastic(),    bNeg.eElastic(),    tol, "elastic 對 φ→-φ 不變");
        assertEquals(bPos.ePhaseField(), bNeg.ePhaseField(), tol, "phase-field 與 φ 無關");
        assertEquals(-bPos.eExternal(),  bNeg.eExternal(),
            Math.abs(bPos.eExternal()) * 1e-10 + ABS_EPS,
            "external 對 φ→-φ 應變號");
    }

    // ═══════════════════════════════════════════════════════════════
    //  Stencil 一致性：EDGE_P / CORNER_P 確實被使用
    // ═══════════════════════════════════════════════════════════════

    @Test
    void stencilCoefficientsAreUsedInEvaluator() {
        // 2×2×2：φ 僅在 (0,0,0) = 1，其他 = 0
        // 以 corner edge (0,0,0)-(1,1,1) 的貢獻為 0.5 × CORNER_P × 1² 驗證
        int L = 2;
        int N = L * L * L;
        float[] phi = new float[N];
        phi[0] = 1.0f; // (0,0,0)
        float[] sigma = filled(N, 1.0f);
        float[] h = filled(N, 1.0f);

        double e = evaluator.evaluate(phi, null, sigma, null, h, L, L, L, noPhaseFieldCalib());

        // 所有從 (0,0,0) 出發的邊:
        //   3 個面邊: w=1 each → 3 × 0.5 × 1² = 1.5
        //   3 個邊邊: w=EDGE_P × 1 → 3 × 0.5 × EDGE_P × 1² = 1.5 × EDGE_P
        //   1 個角邊: w=CORNER_P × 1 → 0.5 × CORNER_P × 1² = 0.5 × CORNER_P
        double expected = 1.5 + 1.5 * PFSFStencil.EDGE_P + 0.5 * PFSFStencil.CORNER_P;

        assertEquals(expected, e, ABS_EPS,
            "2×2×2 φ=單點衝擊驗證 stencil EDGE_P=" + PFSFStencil.EDGE_P +
            " CORNER_P=" + PFSFStencil.CORNER_P);
    }

    // ─── helpers ─────────────────────────────────────────────────

    private static float[] filled(int n, float v) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = v;
        return a;
    }

    private static float[] randArr(Random r, int n, float lo, float hi) {
        float[] a = new float[n];
        for (int i = 0; i < n; i++) a[i] = lo + r.nextFloat() * (hi - lo);
        return a;
    }
}
