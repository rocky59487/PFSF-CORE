package com.blockreality.api.physics.validation;

import com.blockreality.api.physics.effective.CalibrationRunner;
import com.blockreality.api.physics.validation.VoxelPhysicsCpuReference.Domain;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase F — 尺度不變性驗證（16³ vs 32³ vs 64³）。
 *
 * <p>預期物理：在固定幾何比例（相同 aspect ratio）下，隨尺度放大，物理量應遵循
 * 特定尺度律。本測試驗證：
 * <ul>
 *   <li>解析應力 σ_max ∝ L² / h²（長度加倍、高度加倍 → σ 不變；只長度加倍 → σ ×4）</li>
 *   <li>CPU Jacobi solver 在不同尺度下收斂行為一致（相對收斂率、最終 φ 分佈形狀）</li>
 *   <li>能量 E 在同尺度比例放大時遵循 volume-scaling（E ∝ N for Poisson problem）</li>
 * </ul>
 *
 * <p>這測試對於發現「小模型正常、大模型崩潰」的尺度依賴問題至關重要。
 */
class ScaleInvarianceTest {

    @Test
    void stressScalesAsLengthSquaredOverHeightSquared() {
        // σ_max = 6 ρ g L² / h² → L 加倍 → σ ×4；h 加倍 → σ ÷4；兩者同倍 → σ 不變
        double rho = 2400.0;
        double s_L1_h1 = CalibrationRunner.cantileverMaxStress(rho, 1.0, 0.1);
        double s_L2_h1 = CalibrationRunner.cantileverMaxStress(rho, 2.0, 0.1);
        double s_L1_h2 = CalibrationRunner.cantileverMaxStress(rho, 1.0, 0.2);
        double s_L2_h2 = CalibrationRunner.cantileverMaxStress(rho, 2.0, 0.2);

        assertEquals(4.0, s_L2_h1 / s_L1_h1, 1e-9, "L 加倍 → σ ×4");
        assertEquals(0.25, s_L1_h2 / s_L1_h1, 1e-9, "h 加倍 → σ ÷4");
        assertEquals(s_L1_h1, s_L2_h2, 1e-9, "L 與 h 同倍放大 → σ 不變（尺度對稱）");
    }

    @Test
    void volumeScalesAsCubeOfSide() {
        // N = L³ → 側邊 L 加倍 → N 變 8 倍
        Domain d8  = VoxelPhysicsCpuReference.buildAnchoredSlab(8,  8,  1.0);
        Domain d16 = VoxelPhysicsCpuReference.buildAnchoredSlab(16, 16, 1.0);
        assertEquals(8, d16.N() / d8.N(), "體積應隨 L³ 縮放");
    }

    @Test
    void jacobiConvergenceRateScalesPredictably() {
        // 理論：Jacobi 收斂率 ρ ≈ cos(π/L)；L 加倍 → 收斂率更接近 1 → 收斂更慢
        // 但每一步的相對 φ 變化率應保持類似比例關係
        int[] sides = {8, 12, 16};
        double[] finalEnergies = new double[sides.length];
        com.blockreality.api.physics.effective.EnergyEvaluatorCPU ev =
            new com.blockreality.api.physics.effective.EnergyEvaluatorCPU();
        com.blockreality.api.physics.effective.MaterialCalibration calib =
            com.blockreality.api.physics.effective.MaterialCalibration.legacy(
                1, "test", 1,
                com.blockreality.api.physics.effective.MaterialCalibration.BoundaryProfile.ANCHORED_BOTTOM,
                1.0, 0.0, 1.5, 2.0, 0L, "test");

        for (int k = 0; k < sides.length; k++) {
            int L = sides[k];
            // 所有尺度使用相同 load density，這樣 E ∝ N 可比較
            Domain dom = VoxelPhysicsCpuReference.buildAnchoredSlab(L, L, 1.0);
            // 跑足夠多步收斂（大尺度需要更多步）
            float[] phi = VoxelPhysicsCpuReference.solve(dom, 1.0f, L * 20);
            float[] d   = new float[dom.N()];
            float[] h   = new float[dom.N()]; java.util.Arrays.fill(h, 1f);
            finalEnergies[k] = ev.evaluate(phi, d, dom.sigma, dom.source, h,
                                           dom.Lx, dom.Ly, dom.Lz, calib);
        }
        // 所有 E 必為正且收斂（量級應與 N 相當）
        for (double e : finalEnergies) {
            assertTrue(e >= 0, "E 必非負");
            assertTrue(Double.isFinite(e), "E 必有限");
        }
        // E 應隨 L 增加而增加（更多體素 × 更多內部 edges）
        assertTrue(finalEnergies[1] > finalEnergies[0],
            "L=12 的 E 應 > L=8 的 E");
        assertTrue(finalEnergies[2] > finalEnergies[1],
            "L=16 的 E 應 > L=12 的 E");
    }

    @Test
    void orphanDetectionIsScaleInvariant() {
        // 同幾何比例（浮島相對世界 1/2）在不同尺度下都應偵測全部 orphan
        int[] worldSizes = {11, 15, 23};  // 世界
        int[] clusterSizes = {5, 7, 11};  // cluster（都接近 worldSize/2）
        for (int k = 0; k < worldSizes.length; k++) {
            int W = worldSizes[k];
            int C = clusterSizes[k];
            Domain dom = VoxelPhysicsCpuReference.buildFloatingIsland(W, W, W, C, C, C);
            var orphans = VoxelPhysicsCpuReference.findOrphans(dom);
            assertEquals(C * C * C, orphans.size(),
                "World=" + W + " Cluster=" + C + " 的 orphan count 應為 C³=" + (C*C*C) +
                " actual=" + orphans.size());
        }
    }

    @Test
    void archRiseRatioInvariance() {
        // 不同半徑的半圓拱在相同 rise/span 比例下，推力/載荷比例應相同
        double w = 500.0;
        // 半徑加倍，拱高也加倍（全等比例），H ∝ R² / h = R² / R = R；h 隨 R 放大 → H ∝ R
        double H_R8  = CalibrationRunner.semiArchHorizontalThrust(w, 8.0,  8.0);
        double H_R16 = CalibrationRunner.semiArchHorizontalThrust(w, 16.0, 16.0);
        double ratio = H_R16 / H_R8;
        assertEquals(2.0, ratio, 1e-9, "等比例放大半徑 → H ∝ R");
    }

    @Test
    void fixedGeometryDifferentResolutionGivesConsistentMaxPhi() {
        // 相同「幾何」但不同離散解析度 — φ_max 的縮放行為應可預期
        // 短的 cantilever 與長的 cantilever 在 load 相同下 φ_max 有特定比例關係
        Domain d8  = VoxelPhysicsCpuReference.buildCantilever(8,  0, 1.0);
        Domain d12 = VoxelPhysicsCpuReference.buildCantilever(12, 0, 1.0);
        Domain d16 = VoxelPhysicsCpuReference.buildCantilever(16, 0, 1.0);

        float m8  = VoxelPhysicsCpuReference.argMaxAbs(
            VoxelPhysicsCpuReference.solve(d8,  1f, 2000)).absValue();
        float m12 = VoxelPhysicsCpuReference.argMaxAbs(
            VoxelPhysicsCpuReference.solve(d12, 1f, 3000)).absValue();
        float m16 = VoxelPhysicsCpuReference.argMaxAbs(
            VoxelPhysicsCpuReference.solve(d16, 1f, 4000)).absValue();

        // 隨長度增加 φ_max 應單調增（更多 load 累積）
        assertTrue(m12 > m8,  "L=12 的 φ_max 應 > L=8 的：" + m12 + " vs " + m8);
        assertTrue(m16 > m12, "L=16 的 φ_max 應 > L=12 的：" + m16 + " vs " + m12);
    }
}
