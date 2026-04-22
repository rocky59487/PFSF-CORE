package com.blockreality.api.physics.effective;

/**
 * 離散圖能量泛函 E(φ, d) — 標量能量評估器介面。
 *
 * <h2>數學定義</h2>
 * <pre>
 *   E(φ, d, σ, h) =   (1/2) Σ_{edges (i,j)} w_ij × (φ_i − φ_j)²        ← E_elastic
 *                   + Σ_i   ρ_i × φ_i                                    ← E_external
 *                   + Σ_i   G_c × [ l₀ |∇d_i|² + (1 / (4 l₀)) (1 − d_i)² ]  ← E_phaseField (AT2)
 * </pre>
 *
 * <p>其中：
 * <ul>
 *   <li>w_ij 由 {@link EdgeWeight} 產生；26-conn 面/邊/角鄰居按 {@link com.blockreality.api.physics.pfsf.PFSFStencil}
 *       的 EDGE_P / CORNER_P 係數加權</li>
 *   <li>ρ_i 為體素 i 的載荷源（自重、外力等），單位與 φ 同</li>
 *   <li>G_c, l₀ 來自 {@link MaterialCalibration}</li>
 *   <li>|∇d|² 使用中心差分（與 phase_field_evolve.comp.glsl 一致）</li>
 * </ul>
 *
 * <h2>用途</h2>
 * <ol>
 *   <li><b>Phase B CPU Golden Oracle</b> — {@link EnergyEvaluatorCPU} 是唯一權威實作，
 *       所有 GPU reduction kernel 必須與之 bit-compatible（1e-4 相對誤差）</li>
 *   <li><b>Phase C 能量不變式檢查</b> — V-cycle 收斂後比對 E_before / E_after，
 *       違反 monotonicity 時寫入 telemetry（不中斷模擬）</li>
 *   <li><b>Phase F Voxel Benchmark</b> — EnergyConservationTest 使用此介面</li>
 * </ol>
 *
 * <h2>數值性質</h2>
 * <ul>
 *   <li><b>非負性</b>：{@code E_elastic ≥ 0}（邊權非負 × 平方），{@code E_phaseField ≥ 0}；
 *       但 {@code E_external} 可為任意符號（∵ φ 符號任意）</li>
 *   <li><b>對稱性</b>：對任意置換滿足 E(φ, d) 不變</li>
 *   <li><b>單調性（求解收斂中）</b>：對固定 (ρ, d, σ, h)，RBGS/PCG 迭代下 E(φ_k) ≥ E(φ_{k+1})
 *       （至少至離散誤差 ε）</li>
 * </ul>
 *
 * <h2>離散誤差 ε_discrete</h2>
 * <p>對 float32 累加在 N ≤ 2²⁴ 體素，建議 {@code ε = 1e-4 × |E|}。
 *
 * @see EnergyEvaluatorCPU Phase B 參考實作
 * @see EdgeWeight
 * @see MaterialCalibration
 */
public interface GraphEnergyFunctional {

    /**
     * 計算 E(φ, d) 完整總能量（三項相加）。
     *
     * @param phi       體素 φ 場，長度 Lx×Ly×Lz，indexing i = x + y×Lx + z×Lx×Ly
     * @param d         相場損傷 ∈ [0, 1]，同長度；可為 null → 視為全 0
     * @param sigma     體素導率，同長度（未歸一化）
     * @param rho       載荷源（外力密度），同長度；可為 null → 視為全 0
     * @param hField    水化/養護場 ∈ [0, 1]，同長度；可為 null → 視為全 1
     * @param Lx        x 維度
     * @param Ly        y 維度
     * @param Lz        z 維度
     * @param calib     有效參數（提供 G_c, l₀, phaseFieldExponent）
     * @return 總能量 E（double 累加，避免 float32 精度損失）
     */
    double evaluate(float[] phi, float[] d, float[] sigma, float[] rho, float[] hField,
                    int Lx, int Ly, int Lz, MaterialCalibration calib);

    /**
     * 分項能量結果，便於除錯與 Phase C 的 EnergySample record 組裝。
     */
    record EnergyBreakdown(double eElastic, double eExternal, double ePhaseField) {
        public double total() { return eElastic + eExternal + ePhaseField; }
    }

    /**
     * 分項版本（預設以 evaluate 包裝；實作可覆寫為單次掃描提升效率）。
     */
    default EnergyBreakdown evaluateBreakdown(float[] phi, float[] d, float[] sigma,
                                              float[] rho, float[] hField,
                                              int Lx, int Ly, int Lz,
                                              MaterialCalibration calib) {
        throw new UnsupportedOperationException(
            "evaluateBreakdown() 須由實作覆寫 — 預設介面無法拆分"
        );
    }
}
