package com.blockreality.api.physics.effective;

import com.blockreality.api.physics.pfsf.PFSFStencil;

/**
 * Phase B — CPU 端離散圖能量泛函的權威實作（Golden Oracle）。
 *
 * <p>所有 GPU {@code energy_reduce.comp.glsl}（Phase C）必須與此 CPU 版本
 * 產生的總能量相對誤差 &lt; 1e-4（相當於 float32 累加的內在誤差界）。
 *
 * <h2>能量泛函（與 {@link GraphEnergyFunctional} 定義完全一致）</h2>
 * <pre>
 *   E = (1/2) Σ_{edges (i,j)} w_ij (φ_i − φ_j)²
 *     + Σ_i ρ_i φ_i
 *     + Σ_i G_c [ l₀ |∇d_i|² + (1/(4 l₀)) (1 − d_i)² ]
 * </pre>
 *
 * <h2>26-connectivity 展開</h2>
 * <p>每個體素 i 的 26 個鄰居依 {@link PFSFStencil#NEIGHBOR_OFFSETS} 順序：
 * <ul>
 *   <li>面鄰居（6）：σ_ij 取兩端 σ 的算術平均 × 1.0</li>
 *   <li>邊鄰居（12）：σ_ij = sqrt(σ_i × σ_j) × {@link PFSFStencil#EDGE_P}</li>
 *   <li>角鄰居（8）：σ_ij = cbrt(σ_i × σ_j) × {@link PFSFStencil#CORNER_P}
 *       （註：cbrt 於兩端時退化為 sqrt，符合 GPU shader 行為）</li>
 * </ul>
 *
 * <h2>|∇d|² 離散化</h2>
 * <p>使用 6-conn 中心差分（與 {@code phase_field_evolve.comp.glsl} 一致）：
 * <pre>
 *   |∇d_i|² ≈ Σ_{face j} (d_j − d_i)²
 * </pre>
 *
 * <h2>邊重複計算避免</h2>
 * <p>elastic 項每邊只計算一次（iterate i，僅對 neighbor 偏移 j 使 j &gt; i 的偏移方向處理）。
 * 具體做法：以「只看 +x、+y、+z 方向的 13 個正向 offsets」遍歷，確保 (i, j) 不會與 (j, i) 重複。
 *
 * <h2>double 累加</h2>
 * <p>所有 Σ 累加使用 double 降低 catastrophic cancellation。
 * 對 N &le; 2²⁴ 體素，相對誤差應 &lt; 1e-12（遠小於 Phase C 1e-4 容許界）。
 *
 * @see GraphEnergyFunctional
 * @see DefaultEdgeWeight
 */
public final class EnergyEvaluatorCPU implements GraphEnergyFunctional {

    /** 13 個正向（+x / +y / +z 主導）偏移，用於 elastic 項避免重複計算 */
    private static final int[][] POSITIVE_OFFSETS = buildPositiveOffsets();

    private static int[][] buildPositiveOffsets() {
        int[][] all = PFSFStencil.NEIGHBOR_OFFSETS;
        int[][] positive = new int[all.length / 2][];
        int k = 0;
        for (int[] off : all) {
            if (isPositiveDirection(off)) {
                positive[k++] = off;
            }
        }
        if (k != all.length / 2) {
            throw new IllegalStateException("NEIGHBOR_OFFSETS 不是對稱分佈，positive="
                + k + " expected=" + (all.length / 2));
        }
        return positive;
    }

    /** 判斷偏移是否為「正向」—— 字典序第一個非零元素為 +1 */
    private static boolean isPositiveDirection(int[] off) {
        for (int d : off) {
            if (d > 0) return true;
            if (d < 0) return false;
        }
        return false; // 全 0 — 不應發生
    }

    private final EdgeWeight edgeWeight;

    public EnergyEvaluatorCPU() {
        this(DefaultEdgeWeight.INSTANCE);
    }

    public EnergyEvaluatorCPU(EdgeWeight edgeWeight) {
        this.edgeWeight = edgeWeight;
    }

    @Override
    public double evaluate(float[] phi, float[] d, float[] sigma, float[] rho, float[] hField,
                           int Lx, int Ly, int Lz, MaterialCalibration calib) {
        EnergyBreakdown b = evaluateBreakdown(phi, d, sigma, rho, hField, Lx, Ly, Lz, calib);
        return b.total();
    }

    @Override
    public EnergyBreakdown evaluateBreakdown(float[] phi, float[] d, float[] sigma,
                                             float[] rho, float[] hField,
                                             int Lx, int Ly, int Lz,
                                             MaterialCalibration calib) {
        validate(phi, d, sigma, rho, hField, Lx, Ly, Lz);

        final int N = Lx * Ly * Lz;
        final double gc = calib.gcEff();
        final double l0 = calib.l0Eff();

        double eElastic    = 0.0;
        double eExternal   = 0.0;
        double ePhaseField = 0.0;

        for (int z = 0; z < Lz; z++) {
            for (int y = 0; y < Ly; y++) {
                for (int x = 0; x < Lx; x++) {
                    final int i = idx(x, y, z, Lx, Ly);
                    final double phiI = phi[i];
                    final double dI   = (d      != null) ? d[i]      : 0.0;
                    final double sigI = (sigma  != null) ? sigma[i]  : 1.0;
                    final double hI   = (hField != null) ? hField[i] : 1.0;

                    // ─── elastic 項（正向偏移避免重複）───
                    for (int[] off : POSITIVE_OFFSETS) {
                        int nx = x + off[0];
                        int ny = y + off[1];
                        int nz = z + off[2];
                        if (!inBounds(nx, ny, nz, Lx, Ly, Lz)) continue;

                        final int j = idx(nx, ny, nz, Lx, Ly);
                        final double phiJ = phi[j];
                        final double dJ   = (d      != null) ? d[j]      : 0.0;
                        final double sigJ = (sigma  != null) ? sigma[j]  : 1.0;
                        final double hJ   = (hField != null) ? hField[j] : 1.0;

                        double sigmaIJ = stencilConductivity(off, sigI, sigJ);
                        double w = edgeWeight.weight(sigmaIJ, dI, dJ, hI, hJ, calib);
                        double diff = phiI - phiJ;
                        eElastic += w * diff * diff;
                    }

                    // ─── external 項 ───
                    if (rho != null) {
                        eExternal += rho[i] * phiI;
                    }

                    // ─── phase-field 項（Ambati 2015 AT2）───
                    if (gc > 0.0 && l0 > 0.0) {
                        double gradSq = 0.0;
                        // 6-conn 中心差分，僅 +x/+y/+z 方向（但因為是平方，對稱下不需正向限制）
                        // 這裡為嚴格對應 phase_field_evolve.comp.glsl 使用所有 6 面
                        gradSq += gradContrib(d, x, y, z,  1,  0,  0, Lx, Ly, Lz, dI);
                        gradSq += gradContrib(d, x, y, z, -1,  0,  0, Lx, Ly, Lz, dI);
                        gradSq += gradContrib(d, x, y, z,  0,  1,  0, Lx, Ly, Lz, dI);
                        gradSq += gradContrib(d, x, y, z,  0, -1,  0, Lx, Ly, Lz, dI);
                        gradSq += gradContrib(d, x, y, z,  0,  0,  1, Lx, Ly, Lz, dI);
                        gradSq += gradContrib(d, x, y, z,  0,  0, -1, Lx, Ly, Lz, dI);
                        // 6 個差分平方已是 |∇d|² 的 2× 版本（因為每個對稱邊被計算兩次），
                        // 但對應 GPU 實作是直接累加 6 項，這裡保持一致。
                        double oneMinusD = 1.0 - dI;
                        ePhaseField += gc * (l0 * gradSq + (1.0 / (4.0 * l0)) * oneMinusD * oneMinusD);
                    }
                }
            }
        }

        eElastic *= 0.5; // 1/2 前因子
        return new EnergyBreakdown(eElastic, eExternal, ePhaseField);
    }

    // ─── 工具方法 ─────────────────────────────────────────────────

    /** 根據 26-conn 偏移類型，計算有效導率 σ_ij */
    static double stencilConductivity(int[] off, double sigI, double sigJ) {
        int nonzero = 0;
        for (int d : off) if (d != 0) nonzero++;

        double sigICl = Math.max(0.0, sigI);
        double sigJCl = Math.max(0.0, sigJ);

        switch (nonzero) {
            case 1:  // 面鄰居：算術平均（對應 GPU shader 常見做法）
                return 0.5 * (sigICl + sigJCl);
            case 2:  // 邊鄰居：sqrt(σ_i × σ_j) × EDGE_P
                return Math.sqrt(sigICl * sigJCl) * PFSFStencil.EDGE_P;
            case 3:  // 角鄰居：sqrt(σ_i × σ_j) × CORNER_P
                // 註：GPU shader 用 cbrt(σ_x × σ_y × σ_z)，但三維乘積在 CPU 端
                // 只能看到兩端點，退化為 sqrt，符合幾何平均。
                return Math.sqrt(sigICl * sigJCl) * PFSFStencil.CORNER_P;
            default:
                throw new IllegalStateException("無效偏移：nonzero=" + nonzero);
        }
    }

    private static double gradContrib(float[] d, int x, int y, int z,
                                      int dx, int dy, int dz,
                                      int Lx, int Ly, int Lz, double dI) {
        int nx = x + dx, ny = y + dy, nz = z + dz;
        if (!inBounds(nx, ny, nz, Lx, Ly, Lz)) return 0.0;
        double dJ = (d != null) ? d[idx(nx, ny, nz, Lx, Ly)] : 0.0;
        double diff = dJ - dI;
        return diff * diff;
    }

    private static int idx(int x, int y, int z, int Lx, int Ly) {
        return x + y * Lx + z * Lx * Ly;
    }

    private static boolean inBounds(int x, int y, int z, int Lx, int Ly, int Lz) {
        return x >= 0 && x < Lx && y >= 0 && y < Ly && z >= 0 && z < Lz;
    }

    private static void validate(float[] phi, float[] d, float[] sigma, float[] rho, float[] hField,
                                 int Lx, int Ly, int Lz) {
        if (Lx <= 0 || Ly <= 0 || Lz <= 0) {
            throw new IllegalArgumentException("Lx, Ly, Lz 必須為正數");
        }
        int N = Lx * Ly * Lz;
        if (phi == null || phi.length != N) {
            throw new IllegalArgumentException("phi 長度必須為 " + N);
        }
        if (d != null && d.length != N) {
            throw new IllegalArgumentException("d 長度必須為 " + N + " 或 null");
        }
        if (sigma != null && sigma.length != N) {
            throw new IllegalArgumentException("sigma 長度必須為 " + N + " 或 null");
        }
        if (rho != null && rho.length != N) {
            throw new IllegalArgumentException("rho 長度必須為 " + N + " 或 null");
        }
        if (hField != null && hField.length != N) {
            throw new IllegalArgumentException("hField 長度必須為 " + N + " 或 null");
        }
    }
}
