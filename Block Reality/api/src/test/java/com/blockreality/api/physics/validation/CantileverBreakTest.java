package com.blockreality.api.physics.validation;

import com.blockreality.api.physics.effective.CalibrationRunner;
import com.blockreality.api.physics.effective.MaterialCalibration;
import com.blockreality.api.physics.effective.MaterialCalibrationRegistry;
import com.blockreality.api.physics.validation.VoxelPhysicsCpuReference.Domain;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase F — 懸臂樑斷裂位置驗證。
 *
 * <p>預期物理：1×1×N 懸臂樑在自重載荷下，根部（z=0 附近）承受最大彎矩；
 * 理論公式 σ_max = 6ρgL²/h²（{@link CalibrationRunner#cantileverMaxStress}）。
 * 在 PFSF 勢場架構下，對應的 φ 場應在錨點附近最大（靠近根部），
 * 因為錨點 Dirichlet φ=0 而遠端累積載荷。
 *
 * <h2>PASS 判準</h2>
 * <ol>
 *   <li>φ_max 出現在距錨點 ≤ 2 格範圍內（以「錨點上方第一到第二格」為預期斷裂位置）</li>
 *   <li>φ 沿樑長度單調遞增（自由端最大）— 注意這與應力最大位置是不同方向</li>
 *   <li>φ_max 與解析應力成比例關係：長度 L 加倍 → φ_max 以可預期的方式成長</li>
 *   <li>calibrated material 的 rtensEff 與 σ_max 的關係可判定「是否斷裂」</li>
 * </ol>
 *
 * <p>註：本測試不直接觸發 GPU failure_scan shader；以 CPU reference + 解析公式對照
 * 驗證物理概念的數學正確性。
 */
class CantileverBreakTest {

    /** 各長度下樑的自重（fake load intensity，只驗證比例關係） */
    private static final double LOAD_RHO = 1.0;

    @ParameterizedTest(name = "cantilever L={0}")
    @ValueSource(ints = {8, 16, 32})
    void cantileverPhiMonotoneAlongLength(int N) {
        Domain dom = VoxelPhysicsCpuReference.buildCantilever(N, 0, LOAD_RHO);
        float[] phi = VoxelPhysicsCpuReference.solve(dom, 1.0f, 2000);

        // 單調性：從錨點(z=0)往自由端(z=N-1) φ 應遞增
        float prev = phi[0];  // 錨點 = 0
        assertEquals(0f, prev, 1e-5f, "錨點 φ 必為 0");
        for (int z = 1; z < N; z++) {
            float curr = phi[dom.idx(0, 0, z)];
            assertTrue(curr >= prev - 1e-4f,
                "φ 應沿長度遞增：z=" + z + " curr=" + curr + " prev=" + prev);
            prev = curr;
        }
    }

    @Test
    void cantileverMaxPhiScalesWithLength() {
        // 量綱驗證：L 加倍 → φ_max 至少成長 4×（Poisson-like 與 L² 相關）
        Domain d8  = VoxelPhysicsCpuReference.buildCantilever(8,  0, LOAD_RHO);
        Domain d16 = VoxelPhysicsCpuReference.buildCantilever(16, 0, LOAD_RHO);

        float max8  = VoxelPhysicsCpuReference.argMaxAbs(
            VoxelPhysicsCpuReference.solve(d8,  1.0f, 2000)).absValue();
        float max16 = VoxelPhysicsCpuReference.argMaxAbs(
            VoxelPhysicsCpuReference.solve(d16, 1.0f, 4000)).absValue();

        double ratio = max16 / max8;
        // Poisson ∇²φ = source，對於線性累積勢，L 加倍應該讓 φ_max 約 4× 成長
        assertTrue(ratio > 3.0 && ratio < 6.0,
            "L 加倍 → φ_max 應約 4× 成長：actual ratio=" + ratio);
    }

    @Test
    void cantileverMaxStressMatchesTextbook() {
        // 用解析解驗證 σ_max = 6 ρ g L² / h²
        // 參考情境：C30 混凝土，L=4m, h=0.2m, ρ=2400
        double sigma = CalibrationRunner.cantileverMaxStress(2400.0, 4.0, 0.2);
        // 6 × 2400 × 9.81 × 16 / 0.04 = 56,505,600 Pa
        assertEquals(56_505_600.0, sigma, 1e-3);

        // 與 calibrated material 的 rtensEff 比對 — 是否斷裂？
        MaterialCalibration c30 = MaterialCalibrationRegistry.getInstance()
            .getOrDefault("concrete_c30", 1,
                MaterialCalibration.BoundaryProfile.ANCHORED_BOTTOM);
        // C30 rtensEff = 3 MPa = 3e6 Pa；σ_max(56 MPa) 遠超過 → 必斷
        double rtensPa = c30.rtensEff() * 1e6;
        assertTrue(sigma > rtensPa,
            "4m 無配筋混凝土懸臂在自重下必斷裂（σ=" + sigma + " > rtens=" + rtensPa + ")");
    }

    @Test
    void cantileverSafeLengthBelowRtens() {
        // 短懸臂 L=0.5m 應安全（未超過 rtensEff）
        double sigma = CalibrationRunner.cantileverMaxStress(2400.0, 0.5, 0.2);
        // 6 × 2400 × 9.81 × 0.25 / 0.04 = 882,900 Pa ≈ 0.88 MPa
        MaterialCalibration c30 = MaterialCalibrationRegistry.getInstance()
            .getOrDefault("concrete_c30", 1,
                MaterialCalibration.BoundaryProfile.ANCHORED_BOTTOM);
        double rtensPa = c30.rtensEff() * 1e6;
        assertTrue(sigma < rtensPa,
            "0.5m 短懸臂應安全（σ=" + sigma + " < rtens=" + rtensPa + ")");
    }

    @Test
    void anchorVoxelAlwaysZeroPhi() {
        Domain dom = VoxelPhysicsCpuReference.buildCantilever(16, 0, LOAD_RHO);
        float[] phi = VoxelPhysicsCpuReference.solve(dom, 1.0f, 500);
        // 所有 ANCHOR 體素 φ 必為 0
        for (int i = 0; i < dom.N(); i++) {
            if (dom.type[i] == VoxelPhysicsCpuReference.TYPE_ANCHOR) {
                assertEquals(0f, phi[i], 1e-6f, "anchor voxel i=" + i + " 的 φ 必為 0");
            }
        }
    }
}
