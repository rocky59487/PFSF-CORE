package com.blockreality.api.physics.validation;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashSet;
import java.util.Set;

/**
 * Phase F — 體素物理驗證用 CPU reference solver + 幾何構造工具。
 *
 * <p>本 class 為 test-only 範疇，不屬於 production API。提供與 GPU solver 同構的
 * 簡化 CPU 版本，供 validation benchmark 做為 ground truth 比對。
 *
 * <h2>Solver</h2>
 * <ul>
 *   <li>6-connectivity Jacobi（對應 jacobi_smooth.comp.glsl 的 6-face 部分）</li>
 *   <li>Dirichlet anchor 邊界（type == ANCHOR ⇒ φ = 0）</li>
 *   <li>Air voxel 不參與求解（type == AIR ⇒ φ = 0）</li>
 *   <li>均勻 σ：faces 6 向皆 = sigma 常數</li>
 * </ul>
 *
 * <h2>幾何構造器</h2>
 * <ul>
 *   <li>{@link #buildCantilever(int, int, double)} — 1×1×N 懸臂（底端 z=0 錨）</li>
 *   <li>{@link #buildFloatingIsland(int, int, int, int, int, int)} —
 *       在地面上但與地面不接觸的懸浮 cluster</li>
 *   <li>{@link #buildSemiArch(int, int)} — 半圓拱（兩端錨）</li>
 *   <li>{@link #buildAnchoredSlab(int, int, int)} — 整個底面錨定的立方體（能量測試用）</li>
 * </ul>
 */
public final class VoxelPhysicsCpuReference {

    public static final byte TYPE_AIR    = 0;
    public static final byte TYPE_SOLID  = 1;
    public static final byte TYPE_ANCHOR = 2;

    private VoxelPhysicsCpuReference() {}

    // ═══════════════════════════════════════════════════════════════
    //  Domain record
    // ═══════════════════════════════════════════════════════════════

    public record Domain(int Lx, int Ly, int Lz, byte[] type, float[] sigma, float[] source) {
        public int N() { return Lx * Ly * Lz; }
        public int idx(int x, int y, int z) { return x + Lx * (y + Ly * z); }
    }

    // ═══════════════════════════════════════════════════════════════
    //  6-connectivity Jacobi iteration
    // ═══════════════════════════════════════════════════════════════

    /**
     * 學術嚴謹的 26 連通 Jacobi 迭代步。
     *
     * <p>精確計算離散 Poisson 方程：A φ = b 
     * 其中 A_ij = -σ_ij, A_ii = Σ σ_ij。
     * σ_ij 計算與 EnergyEvaluatorCPU 完全對齊，保證數學嚴格性。
     * 採用 double 累加避免在大量聚合時發生浮點數災難性抵消 (catastrophic cancellation)。
     *
     * @param phiPrev         上一步的勢場陣列
     * @param dom             領域幾何與物理屬性
     * @param legacySigmaFace (已廢棄) 過去用於強制固定導率，現已改為直接讀取 dom.sigma()
     * @param omega           鬆弛因子 (1.0 = 標準 Jacobi)
     * @return 新的勢場陣列
     */
    public static float[] jacobiStep(float[] phiPrev, Domain dom, float legacySigmaFace, float omega) {
        int Lx = dom.Lx(), Ly = dom.Ly(), Lz = dom.Lz();
        float[] phi = new float[dom.N()];
        
        int[][] offs = com.blockreality.api.physics.pfsf.PFSFStencil.NEIGHBOR_OFFSETS;
        float EDGE_P = com.blockreality.api.physics.pfsf.PFSFStencil.EDGE_P;
        float CORNER_P = com.blockreality.api.physics.pfsf.PFSFStencil.CORNER_P;

        for (int z = 0; z < Lz; z++) {
            for (int y = 0; y < Ly; y++) {
                for (int x = 0; x < Lx; x++) {
                    int i = dom.idx(x, y, z);
                    byte t = dom.type()[i];
                    if (t == TYPE_ANCHOR || t == TYPE_AIR) { 
                        phi[i] = 0f; 
                        continue; 
                    }

                    double diagA = 0.0;
                    double offDiagSum = 0.0;
                    float sigI = Math.max(0f, dom.sigma()[i]);

                    for (int[] off : offs) {
                        int nx = x + off[0], ny = y + off[1], nz = z + off[2];
                        if (nx < 0 || nx >= Lx || ny < 0 || ny >= Ly || nz < 0 || nz >= Lz) continue;
                        int j = dom.idx(nx, ny, nz);
                        if (dom.type()[j] == TYPE_AIR) continue;
                        
                        float sigJ = Math.max(0f, dom.sigma()[j]);
                        
                        // 計算與 EnergyEvaluatorCPU 一致的導率，嚴謹遵守 26 連拓撲
                        int nonzero = (off[0] != 0 ? 1 : 0) + (off[1] != 0 ? 1 : 0) + (off[2] != 0 ? 1 : 0);
                        double sigIJ = 0.0;
                        if (nonzero == 1) { // 面
                            sigIJ = 0.5 * (sigI + sigJ);
                        } else if (nonzero == 2) { // 邊
                            sigIJ = Math.sqrt(sigI * sigJ) * EDGE_P;
                        } else if (nonzero == 3) { // 角
                            sigIJ = Math.sqrt(sigI * sigJ) * CORNER_P; // 退化為 sqrt(因點只有兩端)
                        }
                        
                        diagA += sigIJ;
                        offDiagSum += sigIJ * phiPrev[j];
                    }
                    
                    double b = dom.source()[i];
                    double phiJ = (diagA > 1e-15) ? (b + offDiagSum) / diagA : phiPrev[i];
                    phi[i] = (float) (omega * (phiJ - phiPrev[i]) + phiPrev[i]);
                }
            }
        }
        return phi;
    }

    /** 跑 N 步 Jacobi，初始 phi 為 0。回傳最終 phi。 */
    public static float[] solve(Domain dom, float legacySigmaFace, int steps) {
        float[] phi = new float[dom.N()];
        for (int s = 0; s < steps; s++) {
            phi = jacobiStep(phi, dom, legacySigmaFace, 1.0f);
        }
        return phi;
    }

    // ═══════════════════════════════════════════════════════════════
    //  Orphan / support detection
    // ═══════════════════════════════════════════════════════════════

    /**
     * BFS from all anchor voxels，回傳不可達（orphan）的 solid voxel 索引集合。
     * 用於 FloatingIslandTest 驗證 NO_SUPPORT 偵測。
     * 已升級至 26-connectivity，能正確穿越對角線識別結構連續性。
     */
    public static Set<Integer> findOrphans(Domain dom) {
        Set<Integer> reached = new HashSet<>();
        Deque<int[]> q = new ArrayDeque<>();
        for (int i = 0; i < dom.N(); i++) {
            if (dom.type()[i] == TYPE_ANCHOR) {
                reached.add(i);
                int x = i % dom.Lx();
                int y = (i / dom.Lx()) % dom.Ly();
                int z = i / (dom.Lx() * dom.Ly());
                q.add(new int[]{x, y, z});
            }
        }
        int[][] offs = com.blockreality.api.physics.pfsf.PFSFStencil.NEIGHBOR_OFFSETS;
        while (!q.isEmpty()) {
            int[] p = q.pop();
            for (int[] off : offs) {
                int nx = p[0]+off[0], ny = p[1]+off[1], nz = p[2]+off[2];
                if (nx<0||nx>=dom.Lx()||ny<0||ny>=dom.Ly()||nz<0||nz>=dom.Lz()) continue;
                int j = dom.idx(nx, ny, nz);
                if (dom.type()[j] != TYPE_SOLID && dom.type()[j] != TYPE_ANCHOR) continue;
                if (reached.add(j)) q.add(new int[]{nx, ny, nz});
            }
        }
        Set<Integer> orphans = new HashSet<>();
        for (int i = 0; i < dom.N(); i++) {
            if (dom.type()[i] == TYPE_SOLID && !reached.contains(i)) orphans.add(i);
        }
        return orphans;
    }

    // ═══════════════════════════════════════════════════════════════
    //  Geometry builders
    // ═══════════════════════════════════════════════════════════════

    /**
     * 1×1×N 懸臂樑（z=0 底層 anchor，z=1..N-1 solid）。
     *
     * @param N       樑長（voxel 格數）
     * @param padXY   x/y 方向預留 padding（通常取 0 或 1）
     * @param loadRho 均勻自重 source 密度
     */
    public static Domain buildCantilever(int N, int padXY, double loadRho) {
        int L = 1 + 2 * padXY;
        int Lz = N;
        int nTotal = L * L * Lz;
        byte[] type = new byte[nTotal];
        float[] sigma = new float[nTotal];
        float[] source = new float[nTotal];
        int cx = padXY, cy = padXY;
        for (int z = 0; z < Lz; z++) {
            int i = cx + L * (cy + L * z);
            type[i] = (z == 0) ? TYPE_ANCHOR : TYPE_SOLID;
            sigma[i] = 1.0f;
            source[i] = (z == 0) ? 0f : (float) loadRho;
        }
        return new Domain(L, L, Lz, type, sigma, source);
    }

    /**
     * 懸浮 cluster：在一個 Lx×Ly×Lz 空間中，整塊 (gx, gy, gz) 大小的 solid cluster 放在
     * (ox, oy, oz) 為最小角；地面 z=0 一層為 anchor（但 cluster 若 oz &gt; 0 則完全不接觸地面）。
     */
    public static Domain buildFloatingIsland(
            int worldLx, int worldLy, int worldLz,
            int gx, int gy, int gz) {
        int nTotal = worldLx * worldLy * worldLz;
        byte[] type = new byte[nTotal];
        float[] sigma = new float[nTotal];
        float[] source = new float[nTotal];

        // z=0 一整層 anchor
        for (int y = 0; y < worldLy; y++) {
            for (int x = 0; x < worldLx; x++) {
                int i = x + worldLx * (y + worldLy * 0);
                type[i] = TYPE_ANCHOR;
                sigma[i] = 1.0f;
            }
        }
        // 懸浮 cluster 放在中央上方（oz 至少為 2 與地面隔離）
        int ox = (worldLx - gx) / 2;
        int oy = (worldLy - gy) / 2;
        int oz = 2;
        for (int z = oz; z < oz + gz; z++) {
            for (int y = oy; y < oy + gy; y++) {
                for (int x = ox; x < ox + gx; x++) {
                    int i = x + worldLx * (y + worldLy * z);
                    type[i] = TYPE_SOLID;
                    sigma[i] = 1.0f;
                    source[i] = 1.0f; // 自重
                }
            }
        }
        return new Domain(worldLx, worldLy, worldLz, type, sigma, source);
    }

    /**
     * 半圓拱（2D，在 y=0 平面）— 半徑 R，兩端 (−R, 0) 與 (+R, 0) 為 anchor。
     * 世界尺寸約 (2R+3)×1×(R+2)。
     */
    public static Domain buildSemiArch(int R, int loadIntensity) {
        int Lx = 2 * R + 3;
        int Ly = 1;
        int Lz = R + 2;
        byte[] type = new byte[Lx * Ly * Lz];
        float[] sigma = new float[Lx * Ly * Lz];
        float[] source = new float[Lx * Ly * Lz];

        int cx = Lx / 2;
        int cy = 0;

        // 拱 arc：所有 (x, y, z) s.t. round(sqrt((x-cx)² + z²)) == R 且 z ≥ 0
        for (int z = 0; z <= R; z++) {
            for (int x = 0; x < Lx; x++) {
                double dx = x - cx;
                double dz = z;
                double d = Math.sqrt(dx * dx + dz * dz);
                if (Math.abs(d - R) < 0.5) {
                    int i = x + Lx * (cy + Ly * z);
                    type[i] = TYPE_SOLID;
                    sigma[i] = 1.0f;
                    source[i] = (float) loadIntensity;
                }
            }
        }
        // 兩端錨點：(cx - R, 0) 與 (cx + R, 0)
        int iL = (cx - R) + Lx * (cy + Ly * 0);
        int iR = (cx + R) + Lx * (cy + Ly * 0);
        if (cx - R >= 0) { type[iL] = TYPE_ANCHOR; source[iL] = 0f; }
        if (cx + R < Lx) { type[iR] = TYPE_ANCHOR; source[iR] = 0f; }

        return new Domain(Lx, Ly, Lz, type, sigma, source);
    }

    /** 單純 L×L×L 立方體，底面 (z=0) 全 anchor，其他 solid，均勻自重。 */
    public static Domain buildAnchoredSlab(int L, int LyUse, double load) {
        int nTotal = L * LyUse * L;
        byte[] type = new byte[nTotal];
        float[] sigma = new float[nTotal];
        float[] source = new float[nTotal];
        for (int z = 0; z < L; z++) {
            for (int y = 0; y < LyUse; y++) {
                for (int x = 0; x < L; x++) {
                    int i = x + L * (y + LyUse * z);
                    type[i] = (z == 0) ? TYPE_ANCHOR : TYPE_SOLID;
                    sigma[i] = 1.0f;
                    source[i] = (z == 0) ? 0f : (float) load;
                }
            }
        }
        return new Domain(L, LyUse, L, type, sigma, source);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Utility
    // ═══════════════════════════════════════════════════════════════

    /** 求 phi 的最大絕對值與其索引 */
    public static record MaxInfo(int index, float absValue) {}
    public static MaxInfo argMaxAbs(float[] phi) {
        int best = 0;
        float bestAbs = 0f;
        for (int i = 0; i < phi.length; i++) {
            float a = Math.abs(phi[i]);
            if (a > bestAbs) { bestAbs = a; best = i; }
        }
        return new MaxInfo(best, bestAbs);
    }
}
