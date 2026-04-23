package com.blockreality.api.physics.topology;

import com.blockreality.api.physics.pfsf.PFSFStencil;

/**
 * CPU reference implementation of the Tier-2 Poisson oracle: given a
 * voxel sub-region with {@code TYPE_AIR / TYPE_SOLID / TYPE_ANCHOR}
 * labels, converge the discrete Poisson system
 * <pre>
 *     L · φ = 0    (Dirichlet φ = 1 at ANCHOR, AIR voxels forced to 0)
 * </pre>
 * using Jacobi iteration over the 26-connected Laplacian, and
 * threshold the result into a boolean fracture mask.
 *
 * <p><b>Mathematical claim.</b> At steady state, {@code φ_i = 1} ⇔ voxel
 * {@code i} lies in the connected component of some ANCHOR voxel (under
 * 26-connectivity restricted to non-AIR cells). SOLID voxels that are
 * not connected to any anchor converge to {@code φ = 0}. A SOLID
 * voxel with {@code φ < ε} (for small ε > 0) is therefore orphan.
 *
 * <p><b>Role in the rewrite.</b> This class is the CPU oracle against
 * which {@code R.5}'s GPU port will be validated. It reuses the same
 * neighbour offsets as {@code PFSFStencil.NEIGHBOR_OFFSETS} so the CPU
 * and GPU paths cannot drift on the stencil. Production eventually
 * runs the GPU version; this class stays as a test oracle and
 * fallback.
 */
public final class PoissonOracleCPU {

    /** Default steady-state tolerance; anything below counts as orphan. */
    public static final float DEFAULT_EPSILON = 0.5f;

    /** Default Jacobi iteration count. Diameter of a 64³ island ≤ ~110;
     *  256 iterations give ample margin for signal to propagate. */
    public static final int DEFAULT_ITERATIONS = 256;

    private PoissonOracleCPU() {}

    /**
     * Post-solve result. {@code phi} is the converged scalar field;
     * {@code fractureMask[i] == true} iff voxel {@code i} is SOLID and
     * disconnected from every ANCHOR (i.e. orphan).
     */
    public record Result(float[] phi, boolean[] fractureMask) {}

    public static Result solve(byte[] type, int Lx, int Ly, int Lz) {
        return solve(type, Lx, Ly, Lz, DEFAULT_ITERATIONS, DEFAULT_EPSILON);
    }

    public static Result solve(byte[] type, int Lx, int Ly, int Lz, int iterations, float epsilon) {
        int n = Lx * Ly * Lz;
        if (type.length != n) {
            throw new IllegalArgumentException("type length " + type.length + " != Lx·Ly·Lz = " + n);
        }
        float[] phi = new float[n];
        float[] next = new float[n];
        for (int i = 0; i < n; i++) {
            if (type[i] == TopologicalSVDAG.TYPE_ANCHOR) phi[i] = 1f;
        }
        int[][] offs = PFSFStencil.NEIGHBOR_OFFSETS;

        for (int iter = 0; iter < iterations; iter++) {
            // Jacobi sweep: write into `next`, then swap.
            for (int z = 0; z < Lz; z++) {
                for (int y = 0; y < Ly; y++) {
                    for (int x = 0; x < Lx; x++) {
                        int i = x + Lx * (y + Ly * z);
                        byte t = type[i];
                        if (t == TopologicalSVDAG.TYPE_ANCHOR) { next[i] = 1f; continue; }
                        if (t == TopologicalSVDAG.TYPE_AIR)    { next[i] = 0f; continue; }
                        // SOLID: average over connected (non-AIR) neighbours.
                        float sum = 0f;
                        int count = 0;
                        for (int[] o : offs) {
                            int nx = x + o[0], ny = y + o[1], nz = z + o[2];
                            if (nx < 0 || nx >= Lx) continue;
                            if (ny < 0 || ny >= Ly) continue;
                            if (nz < 0 || nz >= Lz) continue;
                            int j = nx + Lx * (ny + Ly * nz);
                            if (type[j] == TopologicalSVDAG.TYPE_AIR) continue;
                            sum += phi[j];
                            count++;
                        }
                        next[i] = count > 0 ? sum / count : 0f;
                    }
                }
            }
            // Swap refs (cheapest) — do this by copying since `phi` must
            // hold the authoritative current state on loop exit.
            float[] tmp = phi; phi = next; next = tmp;
        }

        boolean[] mask = new boolean[n];
        for (int i = 0; i < n; i++) {
            if (type[i] == TopologicalSVDAG.TYPE_SOLID && phi[i] < epsilon) mask[i] = true;
        }
        return new Result(phi, mask);
    }
}
