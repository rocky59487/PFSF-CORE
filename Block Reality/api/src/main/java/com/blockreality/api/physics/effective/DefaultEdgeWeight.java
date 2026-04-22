package com.blockreality.api.physics.effective;

/**
 * 預設 {@link EdgeWeight} 實作 — 混凝土/鋼材通用。
 *
 * <h2>公式</h2>
 * <pre>
 *   w_ij = σ_ij × (1 − d_i)^p × (1 − d_j)^p × √(h_i × h_j)
 * </pre>
 *
 * <p>其中 {@code p} 來自 {@link MaterialCalibration#phaseFieldExponent()}，
 * 混凝土通常 p=2（較鈍的損傷衰減），鋼材 p=4（更銳利的脆性過渡）。
 *
 * <h2>四性質證明</h2>
 *
 * <h3>1. Symmetry</h3>
 * <p>{@code (1-d_i)^p × (1-d_j)^p} 與 {@code √(h_i × h_j)} 皆對 (i, j) 交換不變。
 * σ_ij 本身由呼叫者以幾何平均產生，對稱。∴ w_ij = w_ji。∎
 *
 * <h3>2. Positivity</h3>
 * <p>σ_ij ≥ 0（導率）；(1-d)^p ≥ 0 因為 d ∈ [0, 1]；√(h_i × h_j) ≥ 0。
 * 全部非負 → 乘積非負。∎
 *
 * <h3>3. Locality</h3>
 * <p>本實作不檢查拓撲；locality 由 caller（{@link EnergyEvaluatorCPU}、GPU shader）保證，
 * 僅對 26-conn 鄰居呼叫。
 *
 * <h3>4. Monotonicity in d</h3>
 * <p>對 d_i 微分：
 * <pre>
 *   ∂w/∂d_i = σ_ij × (−p) × (1−d_i)^(p−1) × (1−d_j)^p × √(h_i × h_j)
 *           = −p × (1−d_i)^(p−1) × [其餘非負項]
 *           ≤ 0    對所有 d_i ∈ [0, 1] 且 p ≥ 1。
 * </pre>
 * <p>∴ d_i 上升 → w 單調下降。邊界情形：d_i = 1 → (1-d_i)^p = 0 → w = 0。∎
 *
 * <h2>Numerical safeguards</h2>
 * <ul>
 *   <li>d, h 以 {@code Math.max(0, ...)} 下夾確保輸入異常不會破壞正定性</li>
 *   <li>h_i × h_j 以 {@code Math.max(0, ...)} 下夾再開方</li>
 *   <li>當 p < 1 時仍數學正確（單調性由 (1-d)^p 的遞減性保證），但不建議使用</li>
 * </ul>
 */
public final class DefaultEdgeWeight implements EdgeWeight {

    /** 單一共享實例（無狀態） */
    public static final DefaultEdgeWeight INSTANCE = new DefaultEdgeWeight();

    public DefaultEdgeWeight() {}

    @Override
    public double weight(double sigmaIJ, double dI, double dJ,
                         double hI, double hJ, MaterialCalibration calib) {
        // 下夾防負值（positivity 保證）
        double dICl = clamp01(dI);
        double dJCl = clamp01(dJ);
        double hICl = Math.max(0.0, hI);
        double hJCl = Math.max(0.0, hJ);
        double sigmaCl = Math.max(0.0, sigmaIJ);

        double p = calib.phaseFieldExponent();
        double damageI = Math.pow(1.0 - dICl, p);
        double damageJ = Math.pow(1.0 - dJCl, p);
        double curing  = Math.sqrt(hICl * hJCl);

        return sigmaCl * damageI * damageJ * curing;
    }

    private static double clamp01(double v) {
        if (v < 0.0) return 0.0;
        if (v > 1.0) return 1.0;
        return v;
    }
}
