package com.blockreality.api.physics.validation;

import com.blockreality.api.physics.pfsf.*;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Paper-data collector: multi-scenario analytical validation.
 *
 * <p>Each geometry is solved by {@link VoxelPhysicsCpuReference} (the CPU
 * reference solver that mirrors the GPU 26-connected stencil) and compared
 * against a closed-form analytical reference.
 *
 * <h2>Analytical references</h2>
 * <ul>
 *   <li><b>CANTILEVER</b> — 1D Poisson on [0, L] with Dirichlet φ(0)=0 and
 *       Neumann φ'(L)=0 under a uniform source q=1.
 *       Analytic solution: φ(z) = q·L·z − 0.5·q·z².</li>
 *   <li><b>ARCH</b> — Semi-circular arch as 1D Poisson along arc length s
 *       with Dirichlet φ(0) = φ(L_arc) = 0 under a uniform line load q.
 *       Analytic solution: φ(s) = 0.5·q·s·(L_arc − s).</li>
 *   <li><b>SLAB</b> — Anchored slab: Dirichlet only at z=0, free elsewhere.
 *       The solution depends only on z (horizontal homogeneity) and
 *       reduces to the cantilever analytic form φ(z) = q·Lz·z − 0.5·q·z²
 *       at every (x, y) column. Error is measured on the per-z average.</li>
 * </ul>
 *
 * <h2>Output</h2>
 * <p>Writes {@code research/paper_data/raw/validation_results.csv} with
 * columns {@code Geometry,L2RelErrorPercent,Provenance}. The
 * {@code Provenance} column is {@code ANALYTIC_COMPARE} when a real
 * closed-form comparison was computed (never a hard-coded placeholder).
 */
public class PaperDataCollectorTest {

    private static final int STEPS = 5000;
    private static final String DATA_PATH = "../../research/paper_data/raw/validation_results.csv";

    @Test
    @DisplayName("Collect paper validation data (analytic compare, all scenarios)")
    public void collectAllPaperData() throws IOException {
        Files.writeString(Paths.get(DATA_PATH),
                "Geometry,L2RelErrorPercent,Provenance\n",
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

        runCantilever();
        runArch();
        runSlab();
    }

    // ═══════════════════════════════════════════════════════════════
    //  CANTILEVER — 1D Poisson with φ(0)=0 (Dirichlet), φ'(L)=0 (Neumann)
    // ═══════════════════════════════════════════════════════════════

    private void runCantilever() throws IOException {
        VoxelPhysicsCpuReference.Domain dom = VoxelPhysicsCpuReference.buildCantilever(64, 0, 1.0);
        float[] phi = solve(dom);

        int L = dom.Lz();
        double l2Sq = 0.0;
        double refNormSq = 0.0;
        for (int z = 0; z < L; z++) {
            double analytic = (double) L * z - 0.5 * z * z;
            double diff = analytic - phi[z];
            l2Sq += diff * diff;
            refNormSq += analytic * analytic;
        }
        double err = 100.0 * Math.sqrt(l2Sq / Math.max(refNormSq, 1e-30));
        writeRow("CANTILEVER", err, "ANALYTIC_COMPARE");
    }

    // ═══════════════════════════════════════════════════════════════
    //  ARCH — 1D Poisson along arc length with Dirichlet at both ends
    // ═══════════════════════════════════════════════════════════════

    private void runArch() throws IOException {
        int R = 24;
        VoxelPhysicsCpuReference.Domain dom = VoxelPhysicsCpuReference.buildSemiArch(R, 1);
        float[] phi = solve(dom);

        // BFS from one anchor endpoint along solid voxels to recover arc-length
        // ordering. With both feet Dirichlet=0 and uniform source q=1, the
        // analytic solution along the discrete arc length is a parabola:
        //   phi(s) = 0.5 * s * (N_arc - 1 - s)
        int[] arcOrder = buildArcOrder(dom);
        int nArc = arcOrder.length;
        if (nArc < 4) {
            // degenerate — record NaN rather than a fabricated number
            writeRow("ARCH", Double.NaN, "DEGENERATE_ARC_LEN<4");
            return;
        }

        double l2Sq = 0.0;
        double refNormSq = 0.0;
        // Exclude the two anchor endpoints (phi == 0 by construction)
        for (int k = 1; k < nArc - 1; k++) {
            double s = k;
            double analytic = 0.5 * s * ((nArc - 1) - s);
            double diff = analytic - phi[arcOrder[k]];
            l2Sq += diff * diff;
            refNormSq += analytic * analytic;
        }
        double err = 100.0 * Math.sqrt(l2Sq / Math.max(refNormSq, 1e-30));
        writeRow("ARCH", err, "ANALYTIC_COMPARE");
    }

    /**
     * Build an arc-length ordering of solid voxels along the semi-arch.
     * Starts from one anchor, walks 26-conn neighbours greedily. Returns the
     * list of flat-voxel indices in traversal order; degenerate domains
     * produce a short list which the caller reports explicitly.
     */
    private static int[] buildArcOrder(VoxelPhysicsCpuReference.Domain dom) {
        int Lx = dom.Lx(), Ly = dom.Ly(), Lz = dom.Lz();
        byte[] type = dom.type();
        int start = -1;
        for (int i = 0; i < dom.N(); i++) {
            if (type[i] == VoxelPhysicsCpuReference.TYPE_ANCHOR) { start = i; break; }
        }
        if (start < 0) return new int[0];

        int[] order = new int[dom.N()];
        int count = 0;
        boolean[] seen = new boolean[dom.N()];
        Deque<Integer> q = new ArrayDeque<>();
        q.add(start); seen[start] = true;
        int[][] offs = PFSFStencil.NEIGHBOR_OFFSETS;
        while (!q.isEmpty()) {
            int i = q.poll();
            order[count++] = i;
            int x = i % Lx, y = (i / Lx) % Ly, z = i / (Lx * Ly);
            for (int[] off : offs) {
                int nx = x + off[0], ny = y + off[1], nz = z + off[2];
                if (nx < 0 || nx >= Lx || ny < 0 || ny >= Ly || nz < 0 || nz >= Lz) continue;
                int j = dom.idx(nx, ny, nz);
                if (seen[j]) continue;
                byte t = type[j];
                if (t != VoxelPhysicsCpuReference.TYPE_SOLID && t != VoxelPhysicsCpuReference.TYPE_ANCHOR) continue;
                seen[j] = true;
                q.add(j);
            }
        }
        return Arrays.copyOf(order, count);
    }

    // ═══════════════════════════════════════════════════════════════
    //  SLAB — z-column homogeneous 1D Poisson (per-z mean comparison)
    // ═══════════════════════════════════════════════════════════════

    private void runSlab() throws IOException {
        int Lxy = 16;
        VoxelPhysicsCpuReference.Domain dom = VoxelPhysicsCpuReference.buildAnchoredSlab(Lxy, Lxy, 1.0);
        float[] phi = solve(dom);

        int Lx = dom.Lx(), Ly = dom.Ly(), Lz = dom.Lz();
        double[] sum = new double[Lz];
        int[] cnt = new int[Lz];
        byte[] type = dom.type();
        for (int z = 0; z < Lz; z++) {
            for (int y = 0; y < Ly; y++) {
                for (int x = 0; x < Lx; x++) {
                    int i = dom.idx(x, y, z);
                    if (type[i] == VoxelPhysicsCpuReference.TYPE_SOLID ||
                        type[i] == VoxelPhysicsCpuReference.TYPE_ANCHOR) {
                        sum[z] += phi[i];
                        cnt[z]++;
                    }
                }
            }
        }
        double l2Sq = 0.0;
        double refNormSq = 0.0;
        for (int z = 0; z < Lz; z++) {
            if (cnt[z] == 0) continue;
            double meanPhi = sum[z] / cnt[z];
            double analytic = (double) Lz * z - 0.5 * z * z;
            double diff = analytic - meanPhi;
            l2Sq += diff * diff;
            refNormSq += analytic * analytic;
        }
        double err = 100.0 * Math.sqrt(l2Sq / Math.max(refNormSq, 1e-30));
        writeRow("SLAB", err, "ANALYTIC_COMPARE");
    }

    // ═══════════════════════════════════════════════════════════════
    //  helpers
    // ═══════════════════════════════════════════════════════════════

    private static float[] solve(VoxelPhysicsCpuReference.Domain dom) {
        float[] phi = new float[dom.N()];
        for (int s = 0; s < STEPS; s++) {
            phi = VoxelPhysicsCpuReference.jacobiStep(phi, dom, 1.0f, 1.0f);
        }
        return phi;
    }

    private static void writeRow(String name, double errPct, String provenance) throws IOException {
        String row = String.format("%s,%.4f,%s%n", name,
                Double.isNaN(errPct) ? Double.NaN : errPct, provenance);
        Files.writeString(Paths.get(DATA_PATH), row, StandardOpenOption.APPEND);
        System.out.printf(">>> [PaperData] %s  L2RelErr=%s%%  provenance=%s%n",
                name, Double.isNaN(errPct) ? "NaN" : String.format("%.4f", errPct), provenance);
    }
}
