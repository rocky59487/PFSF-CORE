package com.blockreality.api.physics.effective;

/**
 * Phase E — 以內建解析解做校準的執行器。
 *
 * <p>使用者決策（2026-04-22）：**不引入外部 FEM library**，完全以閉合解做
 * reference。這裡提供三個常用 benchmark case 的解析公式與 L2 比對工具。
 *
 * <h2>解析解公式</h2>
 * <h3>1. 懸臂樑（Euler-Bernoulli，自重 ρg）</h3>
 * <pre>
 *   σ_max     = 6 ρ g L² / h²          // 根部最大正應力
 *   δ_tip     = ρ g L⁴ / (8 E I)       // 自由端撓度
 *   M(x)      = (1/2) ρ g (L - x)²      // 距根部 x 處的彎矩
 *   I (矩形) = b h³ / 12
 * </pre>
 *
 * <h3>2. 簡支樑（中央集中載荷 P，或分佈 w）</h3>
 * <pre>
 *   中央彎矩    M_c = (1/4) P L    （集中） 或  M_c = (1/8) w L²（分佈）
 *   σ_max      = 3 w L² / (2 h²)
 *   δ_center   = 5 w L⁴ / (384 E I)
 * </pre>
 *
 * <h3>3. 半圓拱橋（無鉸拱，均布載荷 w）</h3>
 * <pre>
 *   中央彎矩   M_c = w R² (1 - 2/π)
 *   水平推力   H   = w R² / (π h_arch)
 *   皇冠軸向   N_crown = H + w R
 * </pre>
 *
 * <h2>校準流程（Phase F 接入後使用）</h2>
 * <ol>
 *   <li>建立幾何 + 載荷 → PFSF solver 產生 φ_discrete[]</li>
 *   <li>本 class 計算 φ_reference[]</li>
 *   <li>{@link #l2Error(float[], double[])} 計算相對誤差</li>
 *   <li>外層以網格搜尋 edgePenaltyEff / cornerPenaltyEff 最小化 L2</li>
 * </ol>
 *
 * <h2>單位約定</h2>
 * <p>所有輸入使用 SI：長度 m、應力 Pa、密度 kg/m³、E Pa、g m/s²。
 * 呼叫端若用 MPa / GPa 需自行換算。
 */
public final class CalibrationRunner {

    public static final double GRAVITY = 9.81;

    private CalibrationRunner() {}

    // ═══════════════════════════════════════════════════════════════
    //  懸臂樑（Cantilever Beam）
    // ═══════════════════════════════════════════════════════════════

    /**
     * Euler-Bernoulli 懸臂樑根部最大正應力（自重載荷）。
     *
     * @param density  材料密度 (kg/m³)
     * @param length   樑長 L (m)
     * @param height   截面高 h (m)
     * @return σ_max (Pa)
     */
    public static double cantileverMaxStress(double density, double length, double height) {
        requirePositive(density, "density");
        requirePositive(length, "length");
        requirePositive(height, "height");
        return 6.0 * density * GRAVITY * length * length / (height * height);
    }

    /**
     * 懸臂樑自由端撓度 (自重)。
     *
     * @param density  材料密度 (kg/m³)
     * @param length   樑長 L (m)
     * @param height   截面高 h (m)
     * @param breadth  截面寬 b (m)
     * @param youngsPa 楊氏模量 E (Pa)
     * @return 自由端撓度 δ (m)
     */
    public static double cantileverTipDeflection(
            double density, double length, double height, double breadth, double youngsPa) {
        requirePositive(density, "density");
        requirePositive(length, "length");
        requirePositive(height, "height");
        requirePositive(breadth, "breadth");
        requirePositive(youngsPa, "youngsPa");
        double I = breadth * Math.pow(height, 3) / 12.0;
        double w = density * GRAVITY * breadth * height;  // 每單位長度的重量（N/m）
        return w * Math.pow(length, 4) / (8.0 * youngsPa * I);
    }

    /**
     * 懸臂樑彎矩分布（距根部 x）。
     *
     * @param density  材料密度 (kg/m³)
     * @param length   樑長 L (m)
     * @param breadth  截面寬 b (m)
     * @param height   截面高 h (m)
     * @param x        距根部位置 (m), 0 ≤ x ≤ L
     * @return M(x) (N·m)
     */
    public static double cantileverMoment(
            double density, double length, double breadth, double height, double x) {
        if (x < 0.0 || x > length) {
            throw new IllegalArgumentException("x 必須在 [0, L] 範圍內");
        }
        double w = density * GRAVITY * breadth * height;
        double d = length - x;
        return 0.5 * w * d * d;
    }

    // ═══════════════════════════════════════════════════════════════
    //  簡支樑（Simply Supported Beam）
    // ═══════════════════════════════════════════════════════════════

    /**
     * 簡支樑中央最大正應力（均布載荷 w，N/m）。
     *
     * @param loadPerLen 均布線性載荷 w (N/m)
     * @param length     跨距 L (m)
     * @param height     截面高 h (m)
     * @return σ_max (Pa)
     */
    public static double simplySupportedMaxStress(double loadPerLen, double length, double height) {
        requirePositive(loadPerLen, "loadPerLen");
        requirePositive(length, "length");
        requirePositive(height, "height");
        // σ = M / W = (wL²/8) / (h²/6) = 3 w L² / (2 h²) (單位寬度近似)
        return 3.0 * loadPerLen * length * length / (2.0 * height * height);
    }

    /**
     * 簡支樑中央撓度（均布載荷）。
     */
    public static double simplySupportedCenterDeflection(
            double loadPerLen, double length, double height, double breadth, double youngsPa) {
        requirePositive(loadPerLen, "loadPerLen");
        double I = breadth * Math.pow(height, 3) / 12.0;
        return 5.0 * loadPerLen * Math.pow(length, 4) / (384.0 * youngsPa * I);
    }

    // ═══════════════════════════════════════════════════════════════
    //  半圓拱（Semi-Circular Arch）
    // ═══════════════════════════════════════════════════════════════

    /**
     * 半圓無鉸拱中央（皇冠）彎矩 — 無鉸拱閉合解。
     *
     * @param loadPerLen 均布垂直載荷 w (N/m)
     * @param radius     拱半徑 R (m)
     * @return M_c (N·m)
     */
    public static double semiArchCrownMoment(double loadPerLen, double radius) {
        requirePositive(loadPerLen, "loadPerLen");
        requirePositive(radius, "radius");
        return loadPerLen * radius * radius * (1.0 - 2.0 / Math.PI);
    }

    /**
     * 半圓拱支座水平推力。
     *
     * @param loadPerLen 均布垂直載荷 w (N/m)
     * @param radius     拱半徑 R (m)
     * @param archHeight 拱高 h_arch (m)，通常 = R
     * @return H (N)
     */
    public static double semiArchHorizontalThrust(
            double loadPerLen, double radius, double archHeight) {
        requirePositive(loadPerLen, "loadPerLen");
        requirePositive(radius, "radius");
        requirePositive(archHeight, "archHeight");
        return loadPerLen * radius * radius / (Math.PI * archHeight);
    }

    /**
     * 半圓拱皇冠軸向力（壓縮，結構主要承載方式）。
     */
    public static double semiArchCrownAxial(
            double loadPerLen, double radius, double archHeight) {
        return semiArchHorizontalThrust(loadPerLen, radius, archHeight)
             + loadPerLen * radius;
    }

    // ═══════════════════════════════════════════════════════════════
    //  L2 比對（用於網格搜尋參數）
    // ═══════════════════════════════════════════════════════════════

    /**
     * 計算 φ_discrete 與 φ_reference 的相對 L2 誤差：
     * <pre>
     *   e = sqrt( Σ (d_i - r_i)² / Σ r_i² )
     * </pre>
     * @return 相對誤差（0 = 完美匹配）
     */
    public static double l2Error(float[] phiDiscrete, double[] phiReference) {
        if (phiDiscrete.length != phiReference.length) {
            throw new IllegalArgumentException("長度不一致：discrete=" + phiDiscrete.length +
                " reference=" + phiReference.length);
        }
        double num = 0.0, den = 0.0;
        for (int i = 0; i < phiDiscrete.length; i++) {
            double d = phiDiscrete[i] - phiReference[i];
            num += d * d;
            den += phiReference[i] * phiReference[i];
        }
        if (den <= 0.0) return Math.sqrt(num);  // 完全零參考 → 回傳絕對誤差
        return Math.sqrt(num / den);
    }

    /**
     * 網格搜尋最佳 edgePenalty / cornerPenalty 的最小化框架。
     *
     * <p>實際計算由呼叫者提供（callback 接收 (edgeP, cornerP) → L2 error），
     * 因為 discrete solution 需要呼叫 PFSF solver 而它在 test 範疇外。本方法
     * 只負責列舉格點並追蹤最小值。
     *
     * @param edgeCandidates    edgePenalty 搜尋候選值
     * @param cornerCandidates  cornerPenalty 搜尋候選值
     * @param errorFn           給定 (edgeP, cornerP) 回傳 L2 error
     * @return (bestEdge, bestCorner, bestError)
     */
    public static GridSearchResult gridSearch(
            double[] edgeCandidates,
            double[] cornerCandidates,
            java.util.function.ToDoubleBiFunction<Double, Double> errorFn) {
        if (edgeCandidates.length == 0 || cornerCandidates.length == 0) {
            throw new IllegalArgumentException("候選值陣列不得為空");
        }
        double bestE = edgeCandidates[0], bestC = cornerCandidates[0];
        double bestErr = Double.POSITIVE_INFINITY;
        for (double e : edgeCandidates) {
            for (double c : cornerCandidates) {
                double err = errorFn.applyAsDouble(e, c);
                if (err < bestErr) {
                    bestErr = err;
                    bestE = e;
                    bestC = c;
                }
            }
        }
        return new GridSearchResult(bestE, bestC, bestErr);
    }

    public record GridSearchResult(double bestEdgePenalty, double bestCornerPenalty, double bestError) {}

    // ─── helpers ─────────────────────────────────────────────────

    private static void requirePositive(double v, String name) {
        if (v <= 0.0) throw new IllegalArgumentException(name + " 必須為正數，實際 " + v);
    }
}
