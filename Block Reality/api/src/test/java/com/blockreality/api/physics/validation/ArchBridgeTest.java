package com.blockreality.api.physics.validation;

import com.blockreality.api.physics.effective.CalibrationRunner;
import com.blockreality.api.physics.validation.VoxelPhysicsCpuReference.Domain;
import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase F — 半圓拱橋壓縮流驗證。
 *
 * <p>預期物理：半圓無鉸拱橋在均勻垂直載荷下，結構主要承受軸向壓縮。
 * 皇冠位置中央彎矩為 M_c = wR²(1 − 2/π)；水平推力 H = wR²/(π h_arch)。
 *
 * <h2>PASS 判準</h2>
 * <ol>
 *   <li>拱幾何完整：arc 體素齊全、兩端有 anchor</li>
 *   <li>所有拱體素與 anchor 連通（不產生 orphan）</li>
 *   <li>解析公式：M_c 與 H 的比例關係符合閉合解（cross-check CalibrationRunner）</li>
 *   <li>Jacobi 收斂後 φ 分佈呈現「中央高、兩端錨低」的對稱性</li>
 * </ol>
 */
class ArchBridgeTest {

    @Test
    void archGeometryHasAnchorsAtBothFootings() {
        int R = 8;
        Domain dom = VoxelPhysicsCpuReference.buildSemiArch(R, 1);
        int cx = dom.Lx / 2;
        int iL = (cx - R) + dom.Lx * (0 + dom.Ly * 0);
        int iR = (cx + R) + dom.Lx * (0 + dom.Ly * 0);
        assertEquals(VoxelPhysicsCpuReference.TYPE_ANCHOR, dom.type[iL],
            "左端點必為 anchor");
        assertEquals(VoxelPhysicsCpuReference.TYPE_ANCHOR, dom.type[iR],
            "右端點必為 anchor");
    }

    @Test
    void archBodyFullyConnected_noOrphans() {
        Domain dom = VoxelPhysicsCpuReference.buildSemiArch(8, 1);
        Set<Integer> orphans = VoxelPhysicsCpuReference.findOrphans(dom);
        assertEquals(0, orphans.size(),
            "半圓拱（兩端錨）不應有 orphan：actual=" + orphans.size());
    }

    @Test
    void archCrownMomentMatchesClosedForm() {
        // 解析：M_c = w R² (1 − 2/π)
        double w = 500.0, R = 8.0;
        double M = CalibrationRunner.semiArchCrownMoment(w, R);
        double expected = w * R * R * (1 - 2.0 / Math.PI);
        assertEquals(expected, M, 1e-6);
        // 量級檢查：M_c 應遠小於 wR² 但為正
        assertTrue(M > 0);
        assertTrue(M < w * R * R);
    }

    @Test
    void archHorizontalThrustScalesInverselyWithRise() {
        // H = w R² / (π h_arch) — 拱高 (rise) 越小 → 推力越大（淺拱原理）
        double w = 500.0, R = 8.0;
        double H_deep   = CalibrationRunner.semiArchHorizontalThrust(w, R, 8.0);
        double H_medium = CalibrationRunner.semiArchHorizontalThrust(w, R, 4.0);
        double H_shallow = CalibrationRunner.semiArchHorizontalThrust(w, R, 2.0);

        assertTrue(H_medium > H_deep, "中等拱的推力應大於深拱");
        assertTrue(H_shallow > H_medium, "淺拱的推力應更大");
        // 1/h_arch 比例：h 減半 → H 加倍
        assertEquals(2.0, H_medium / H_deep, 1e-6);
        assertEquals(4.0, H_shallow / H_deep, 1e-6);
    }

    @Test
    void archSolverConvergesToSymmetricField() {
        int R = 6;
        Domain dom = VoxelPhysicsCpuReference.buildSemiArch(R, 1);
        float[] phi = VoxelPhysicsCpuReference.solve(dom, 1.0f, 2000);

        int cx = dom.Lx / 2;
        // 對稱性：cx 左右對應 voxel 的 φ 應大約相等
        int mismatchCount = 0;
        for (int z = 0; z <= R; z++) {
            for (int dx = 1; dx <= R; dx++) {
                int iL = (cx - dx) + dom.Lx * (0 + dom.Ly * z);
                int iR = (cx + dx) + dom.Lx * (0 + dom.Ly * z);
                if (iL < 0 || iR >= dom.N()) continue;
                if (dom.type[iL] != VoxelPhysicsCpuReference.TYPE_SOLID &&
                    dom.type[iL] != VoxelPhysicsCpuReference.TYPE_ANCHOR) continue;
                if (dom.type[iR] != VoxelPhysicsCpuReference.TYPE_SOLID &&
                    dom.type[iR] != VoxelPhysicsCpuReference.TYPE_ANCHOR) continue;
                float vL = phi[iL], vR = phi[iR];
                float relative = Math.abs(vL - vR) / (Math.abs(vL) + Math.abs(vR) + 1e-6f);
                if (relative > 0.05f) mismatchCount++;
            }
        }
        assertEquals(0, mismatchCount,
            "半圓拱收斂後 φ 必對稱（左右對應誤差 < 5%）：mismatch=" + mismatchCount);
    }

    @Test
    void archCrownAxialExceedsHorizontalThrust() {
        // 皇冠軸向力 N = H + wR > H（因為 w 是垂直分量的貢獻）
        double w = 500.0, R = 8.0, h = 8.0;
        double N = CalibrationRunner.semiArchCrownAxial(w, R, h);
        double H = CalibrationRunner.semiArchHorizontalThrust(w, R, h);
        assertTrue(N > H, "N_crown 應大於 H");
        assertEquals(H + w * R, N, 1e-6);
    }
}
