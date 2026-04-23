package com.blockreality.api.physics.topology;

import com.blockreality.api.physics.pfsf.PFSFStencil;

/**
 * Bit-accurate Java reproduction of the Tier-2 Vulkan Poisson oracle
 * ({@code poisson_oracle_jacobi.comp.glsl} +
 * {@code poisson_oracle_threshold.comp.glsl}). Serves the same role for
 * Tier 2 that {@code PFSFLabelPropCpuSimulator} serves for label
 * propagation: the sandbox cannot observe GPU dispatches, so the
 * simulator replays each kernel's memory model in sequential Java, then
 * gets exercised by unit tests against {@link PoissonOracleCPU}.
 *
 * <p>Differences from {@link PoissonOracleCPU}:
 * <ul>
 *   <li>Uses the same ping-pong buffer pattern as the GPU kernel. The
 *       CPU oracle does too, but the simulator is explicit about which
 *       buffer is phiIn vs phiOut at each step to mirror what the
 *       shader sees.</li>
 *   <li>The threshold pass writes into a packed uint32 bitmap
 *       (ceil(N/32) uints), the same layout as the shader output
 *       {@code maskBits}.</li>
 * </ul>
 */
public final class PoissonOracleGpuSimulator {

    private PoissonOracleGpuSimulator() {}

    /** Output of a full simulator run: converged φ field + packed mask bits. */
    public record Result(float[] phi, int[] maskBits) {
        public boolean getMaskBit(int i) {
            int word = i >>> 5;
            int bit  = 1 << (i & 31);
            return word < maskBits.length && (maskBits[word] & bit) != 0;
        }
    }

    public static Result run(byte[] type, int Lx, int Ly, int Lz) {
        return run(type, Lx, Ly, Lz, PoissonOracleCPU.DEFAULT_ITERATIONS, PoissonOracleCPU.DEFAULT_EPSILON);
    }

    public static Result run(byte[] type, int Lx, int Ly, int Lz, int iterations, float epsilon) {
        int n = Lx * Ly * Lz;
        float[] phiIn  = new float[n];
        float[] phiOut = new float[n];
        // Seed ANCHOR Dirichlet; rest initialised to zero by Java.
        for (int i = 0; i < n; i++) {
            if (type[i] == TopologicalSVDAG.TYPE_ANCHOR) phiIn[i] = 1f;
        }

        // Jacobi outer loop: exactly what the host records as
        //   for k in 0..iterations-1:
        //       bind phiIn = buf[k & 1], phiOut = buf[(k^1) & 1]
        //       dispatch jacobi; barrier
        for (int k = 0; k < iterations; k++) {
            kernelJacobi(type, phiIn, phiOut, Lx, Ly, Lz);
            // ping-pong
            float[] tmp = phiIn; phiIn = phiOut; phiOut = tmp;
        }
        // phiIn holds the final field (one extra swap after the last iter).
        float[] phi = phiIn;

        int[] maskBits = new int[(n + 31) >>> 5];
        kernelThreshold(type, phi, maskBits, Lx, Ly, Lz, epsilon);
        return new Result(phi, maskBits);
    }

    /** Mirror of poisson_oracle_jacobi.comp.glsl for a single sweep. */
    public static void kernelJacobi(byte[] type, float[] phiIn, float[] phiOut,
                                    int Lx, int Ly, int Lz) {
        int n = Lx * Ly * Lz;
        int[][] offs = PFSFStencil.NEIGHBOR_OFFSETS;
        for (int i = 0; i < n; i++) {
            byte t = type[i];
            if (t == TopologicalSVDAG.TYPE_ANCHOR) { phiOut[i] = 1f; continue; }
            if (t == TopologicalSVDAG.TYPE_AIR)    { phiOut[i] = 0f; continue; }
            int x   = i % Lx;
            int rem = i / Lx;
            int y   = rem % Ly;
            int z   = rem / Ly;
            float sum = 0f;
            int count = 0;
            for (int[] o : offs) {
                int nx = x + o[0], ny = y + o[1], nz = z + o[2];
                if (nx < 0 || nx >= Lx) continue;
                if (ny < 0 || ny >= Ly) continue;
                if (nz < 0 || nz >= Lz) continue;
                int j = nx + Lx * (ny + Ly * nz);
                if (type[j] == TopologicalSVDAG.TYPE_AIR) continue;
                sum += phiIn[j];
                count++;
            }
            phiOut[i] = count > 0 ? sum / count : 0f;
        }
    }

    /** Mirror of poisson_oracle_threshold.comp.glsl. */
    public static void kernelThreshold(byte[] type, float[] phi, int[] maskBits,
                                       int Lx, int Ly, int Lz, float epsilon) {
        int n = Lx * Ly * Lz;
        for (int i = 0; i < n; i++) {
            if (type[i] != TopologicalSVDAG.TYPE_SOLID) continue;
            if (phi[i] >= epsilon) continue;
            maskBits[i >>> 5] |= 1 << (i & 31);
        }
    }
}
