package com.blockreality.api.physics.effective;

/**
 * 體素間邊權 w_ij 抽象。
 *
 * <h2>數學定義</h2>
 * <p>在 26-connectivity 圖上，每對相鄰體素 (i, j) 有一個非負的能量耦合強度 w_ij。
 * 圖能量泛函的彈性項為：
 * <pre>
 *   E_elastic = (1/2) Σ_{edges} w_ij × (φ_i − φ_j)²
 * </pre>
 *
 * <h2>四性質保證（實作必須滿足）</h2>
 * <ol>
 *   <li><b>Symmetry（對稱性）</b>：{@code weight(i, j, σ, d_i, d_j, h_i, h_j, calib) ==
 *       weight(j, i, σ, d_j, d_i, h_j, h_i, calib)}。
 *       ⇒ 確保離散算子 A 對稱 ⇒ CG 可收斂。</li>
 *
 *   <li><b>Positivity（正定性）</b>：{@code weight(...) ≥ 0}。
 *       ⇒ A 為半正定 ⇒ 能量二次型 ≥ 0。</li>
 *
 *   <li><b>Locality（局部性）</b>：僅對 26-conn 鄰居呼叫（由 caller 保證）；
 *       非鄰居 w_ij ≡ 0。
 *       ⇒ 影響力不瞬間躍遷。</li>
 *
 *   <li><b>Monotonicity in damage（損傷單調性）</b>：d_i 或 d_j 上升 → weight 單調下降；
 *       當 d_i = 1 或 d_j = 1 → weight = 0（連線移除）。
 *       ⇒ 損壞必然導致剛度下降，不可能越壞越強。</li>
 * </ol>
 *
 * <p>這四性質由 {@link EdgeWeightPropertyTest} 以 10k 隨機輸入驗證。
 *
 * @see DefaultEdgeWeight 混凝土/鋼材通用預設實作
 * @see PFSFStencil 26-connectivity 常數源
 */
@FunctionalInterface
public interface EdgeWeight {

    /**
     * 計算兩體素間的邊權。
     *
     * @param sigmaIJ       兩端點導率的幾何平均（或面鄰居時取直接 σ）
     * @param dI            體素 i 的相場損傷 ∈ [0, 1]
     * @param dJ            體素 j 的相場損傷 ∈ [0, 1]
     * @param hI            體素 i 的水化/養護程度 ∈ [0, 1]
     * @param hJ            體素 j 的水化/養護程度 ∈ [0, 1]
     * @param calib         有效參數（含 phaseFieldExponent）
     * @return 非負邊權 w_ij
     */
    double weight(double sigmaIJ, double dI, double dJ, double hI, double hJ,
                  MaterialCalibration calib);
}
