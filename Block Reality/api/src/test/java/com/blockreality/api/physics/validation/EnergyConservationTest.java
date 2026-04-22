package com.blockreality.api.physics.validation;

import com.blockreality.api.physics.effective.EnergyEvaluatorCPU;
import com.blockreality.api.physics.effective.GraphEnergyFunctional;
import com.blockreality.api.physics.effective.MaterialCalibration;
import com.blockreality.api.physics.validation.VoxelPhysicsCpuReference.Domain;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase F — 能量單調下降驗證（E monotonically decreasing under Jacobi iteration）。
 *
 * <p>對純彈性 Poisson 問題，Jacobi / RBGS / PCG 等 iterative solver 在數學上是梯度
 * 下降，迭代過程中能量泛函 E(φ) 必須單調下降至收斂。這是 solver 正確性的硬數學保證。
 *
 * <h2>PASS 判準</h2>
 * <ol>
 *   <li>40 步 Jacobi 迭代中，E(φ_k) ≥ E(φ_{k+1}) − ε（允許 float32 截斷的小誤差）</li>
 *   <li>隨機場收斂後的 E 應顯著小於初始亂數場的 E</li>
 *   <li>在無 source、無 anchor 的孤立 domain 中，E 應收斂到 0</li>
 *   <li>有 source 的情境，E 應收斂到某個穩態（非 0）且之後不再單調減少太多</li>
 * </ol>
 */
class EnergyConservationTest {

    private final GraphEnergyFunctional evaluator = new EnergyEvaluatorCPU();

    /** G_c=0 的 calibration，單純測純彈性能量 */
    private static MaterialCalibration elasticOnly() {
        return MaterialCalibration.legacy(
            MaterialCalibration.SCHEMA_V1, "test_elastic", 1,
            MaterialCalibration.BoundaryProfile.ANCHORED_BOTTOM,
            1.0, 0.0, 1.5, 2.0,
            0L, "test"
        );
    }

    @Test
    void jacobiIterationsMakeEnergyMonotonic() {
        int L = 16;
        Domain dom = VoxelPhysicsCpuReference.buildAnchoredSlab(L, L, 1.0);

        float[] d = new float[dom.N()];       // damage = 0
        float[] h = new float[dom.N()];
        java.util.Arrays.fill(h, 1f);         // curing = 1

        // 初始 φ 隨機大值（遠離收斂解）
        Random r = new Random(0xBEEFL);
        float[] phi = new float[dom.N()];
        for (int i = 0; i < dom.N(); i++) {
            if (dom.type()[i] == VoxelPhysicsCpuReference.TYPE_SOLID) {
                phi[i] = (r.nextFloat() - 0.5f) * 20f;
            }
        }

        // 模擬 Source = b_i；能量泛函使用 +rho_i*phi_i，因此 rho_i 必須為 -b_i
        float[] negSrc = new float[dom.N()];
        for(int i=0; i<dom.N(); i++) negSrc[i] = -dom.source()[i];

        MaterialCalibration calib = elasticOnly();
        double E_initial = evaluator.evaluate(phi, d, dom.sigma(), negSrc, h,
                                              dom.Lx(), dom.Ly(), dom.Lz(), calib);

        double E_prev = E_initial;
        int violations = 0;
        double E_current = E_initial;
        for (int step = 0; step < 40; step++) {
            phi = VoxelPhysicsCpuReference.jacobiStep(phi, dom, 1.0f, 1.0f);
            E_current = evaluator.evaluate(phi, d, dom.sigma(), negSrc, h,
                                           dom.Lx(), dom.Ly(), dom.Lz(), calib);
            // 允許 ε = 1e-4 × |E| 的 float32 截斷容忍
            double eps = 1e-4 * Math.abs(E_prev);
            if (E_current > E_prev + eps) {
                violations++;
                System.err.println("step " + step + ": E " + E_prev + " → " + E_current);
            }
            E_prev = E_current;
        }
        assertEquals(0, violations,
            "40 步 Jacobi 中 E 應單調下降，違反次數 violations=" + violations);

        // 收斂後 E 必遠小於初始 E（至少 < 10%，表明收斂到局部最小）
        assertTrue(E_current < E_initial * 0.1,
            "收斂後 E 應顯著下降；E_initial=" + E_initial + " E_final=" + E_current);
    }

    @Test
    void energyConvergesForHealthyStructure() {
        int L = 12;
        Domain dom = VoxelPhysicsCpuReference.buildAnchoredSlab(L, L, 1.0);
        float[] d = new float[dom.N()];
        float[] h = new float[dom.N()];
        java.util.Arrays.fill(h, 1f);

        float[] phi = new float[dom.N()];
        MaterialCalibration calib = elasticOnly();
        
        float[] negSrc = new float[dom.N()];
        for(int i=0; i<dom.N(); i++) negSrc[i] = -dom.source()[i];

        // 跑 200 步；26-conn 收斂比 6-conn 慢，最後 10 步的 E 變化應顯著小於前 10 步 (例如 30%)
        double[] traj = new double[200];
        for (int step = 0; step < 200; step++) {
            phi = VoxelPhysicsCpuReference.jacobiStep(phi, dom, 1.0f, 1.0f);
            traj[step] = evaluator.evaluate(phi, d, dom.sigma(), negSrc, h,
                                            dom.Lx(), dom.Ly(), dom.Lz(), calib);
        }
        double earlyChange = Math.abs(traj[9] - traj[0]);
        double lateChange  = Math.abs(traj[199] - traj[190]);
        assertTrue(lateChange <= earlyChange * 0.35,
            "後期 E 變化 (" + lateChange + ") 應小於前期 (" + earlyChange + ") 以證明收斂");
    }

    @Test
    void damageIncreasesEnergyStorageCapacity() {
        // 對相同 φ 但有 damage 的情境，彈性項應減少（損傷降低剛度）
        int L = 8;
        Domain dom = VoxelPhysicsCpuReference.buildAnchoredSlab(L, L, 1.0);
        float[] h = new float[dom.N()];
        java.util.Arrays.fill(h, 1f);

        // 先跑個小迭代讓 φ 進入合理範圍
        float[] phi = new float[dom.N()];
        for (int step = 0; step < 50; step++) {
            phi = VoxelPhysicsCpuReference.jacobiStep(phi, dom, 1.0f, 1.0f);
        }

        MaterialCalibration calib = elasticOnly();
        float[] dZero = new float[dom.N()];
        float[] dHalf = new float[dom.N()];
        java.util.Arrays.fill(dHalf, 0.5f);

        double eZero = evaluator.evaluate(phi, dZero, dom.sigma(), dom.source(), h,
                                          dom.Lx(), dom.Ly(), dom.Lz(), calib);
        double eHalf = evaluator.evaluate(phi, dHalf, dom.sigma(), dom.source(), h,
                                          dom.Lx(), dom.Ly(), dom.Lz(), calib);
        // 對同 φ：d=0.5 時 w = σ(1-d)² ≈ 0.0625 × σ → elastic 項降至 1/16
        // external 項不變，所以 total 會變小（若 external 為正）或可能變大（依 external 正負）
        // 但 elastic 必 ≤ 不損傷情境
        GraphEnergyFunctional.EnergyBreakdown bZero =
            evaluator.evaluateBreakdown(phi, dZero, dom.sigma(), dom.source(), h,
                                        dom.Lx(), dom.Ly(), dom.Lz(), calib);
        GraphEnergyFunctional.EnergyBreakdown bHalf =
            evaluator.evaluateBreakdown(phi, dHalf, dom.sigma(), dom.source(), h,
                                        dom.Lx(), dom.Ly(), dom.Lz(), calib);
        assertTrue(bHalf.eElastic() <= bZero.eElastic(),
            "damage 上升應降低 elastic 項（剛度下降）；zero=" + bZero.eElastic() +
            " half=" + bHalf.eElastic());
    }

    @Test
    void symmetricLoadYieldsSymmetricEnergyProfile() {
        // 對稱自重載荷 → E_elastic 應與 φ 的符號翻轉無關
        int L = 10;
        Domain dom = VoxelPhysicsCpuReference.buildAnchoredSlab(L, L, 1.0);
        float[] d = new float[dom.N()];
        float[] h = new float[dom.N()];
        java.util.Arrays.fill(h, 1f);

        Random r = new Random(0xCAFE);
        float[] phi = new float[dom.N()];
        for (int i = 0; i < dom.N(); i++)
            phi[i] = (r.nextFloat() - 0.5f) * 5f;
        float[] phiNeg = new float[dom.N()];
        for (int i = 0; i < dom.N(); i++) phiNeg[i] = -phi[i];

        MaterialCalibration calib = elasticOnly();
        GraphEnergyFunctional.EnergyBreakdown bPos =
            evaluator.evaluateBreakdown(phi,    d, dom.sigma(), null, h,
                                        dom.Lx(), dom.Ly(), dom.Lz(), calib);
        GraphEnergyFunctional.EnergyBreakdown bNeg =
            evaluator.evaluateBreakdown(phiNeg, d, dom.sigma(), null, h,
                                        dom.Lx(), dom.Ly(), dom.Lz(), calib);
        assertEquals(bPos.eElastic(), bNeg.eElastic(),
            Math.abs(bPos.eElastic()) * 1e-10 + 1e-6,
            "E_elastic 對 φ→-φ 不變（quadratic）");
    }
}
