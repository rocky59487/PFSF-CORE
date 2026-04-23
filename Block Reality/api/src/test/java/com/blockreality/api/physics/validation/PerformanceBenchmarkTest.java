package com.blockreality.api.physics.validation;

import com.blockreality.api.physics.pfsf.*;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

/**
 * Bandwidth-model prediction of GPU throughput versus measured CPU Jacobi.
 *
 * <p><b>Honest scope:</b> this test does NOT dispatch any GPU work. The CPU
 * rows are wall-clock measurements (JVM {@link System#nanoTime()}); the
 * GPU rows are model predictions computed from a nominal DRAM bandwidth
 * and the 26-connected stencil access pattern. See {@link #predictGpuMs}.
 *
 * <p>Output: {@code research/paper_data/raw/performance_metrics_predicted.csv}
 * with columns {@code Size_N,Role,TimePerStep_ms,Provenance} — every row is
 * explicitly labelled so downstream readers cannot confuse predicted GPU
 * numbers with real measurements.
 */
public class PerformanceBenchmarkTest {

    private static final String PERF_DATA_PATH =
            "../../research/paper_data/raw/performance_metrics_predicted.csv";

    /** Nominal GPU DRAM bandwidth used in the prediction model (GB/s). */
    private static final double GPU_BANDWIDTH_GB_S = 400.0;
    /** Optimisation factor applied to the raw bandwidth bound (0.3 = 70% reuse). */
    private static final double GPU_OPT_FACTOR = 0.3;
    /** Assumed fixed dispatch overhead per step (ms). */
    private static final double GPU_DISPATCH_OVERHEAD_MS = 0.05;

    @Test
    @DisplayName("CPU measured vs GPU predicted (bandwidth model)")
    public void runPerformanceComparison() throws IOException {
        Files.writeString(Paths.get(PERF_DATA_PATH),
                "Size_N,Role,TimePerStep_ms,Provenance\n",
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

        // Voxel island sizes L ∈ {16, 32, 64, 128} ⇒ N ∈ {~4k, ~33k, ~262k, ~2M}
        int[] sizes = {16, 32, 64, 128};
        for (int L : sizes) {
            benchmarkSize(L);
        }
    }

    private void benchmarkSize(int L) throws IOException {
        VoxelPhysicsCpuReference.Domain dom =
                VoxelPhysicsCpuReference.buildCantilever(L, L / 4, 1.0);
        int N = dom.N();
        int iterations = 100;

        // 1. CPU measured
        float[] phiCpu = new float[N];
        long startCpu = System.nanoTime();
        for (int i = 0; i < iterations; i++) {
            phiCpu = VoxelPhysicsCpuReference.jacobiStep(phiCpu, dom, 1.0f, 1.0f);
        }
        long endCpu = System.nanoTime();
        double cpuMs = (endCpu - startCpu) / 1_000_000.0 / iterations;

        // 2. GPU predicted (bandwidth model, no GPU dispatch occurs)
        double gpuPredictedMs = predictGpuMs(N);

        writeRow(N, "CPU_JAVA", cpuMs, "MEASURED_JVM_NANOTIME");
        writeRow(N, "GPU_FNO", gpuPredictedMs,
                "PREDICTED_BANDWIDTH_MODEL(bw=" + GPU_BANDWIDTH_GB_S + "GB/s,opt=" + GPU_OPT_FACTOR + ")");

        System.out.printf(">>> [Perf] N=%d  CPU=%.3fms(measured)  GPU=%.3fms(predicted)%n",
                N, cpuMs, gpuPredictedMs);
    }

    /**
     * Roofline-style upper bound for a single Jacobi step under the
     * 26-connected stencil:
     *   bytes_per_voxel = 26 neighbours × 4 bytes × 2 (read φ, write new φ)
     *   t_bw_ms = bytes_total / bandwidth
     *   t_gpu_ms = t_bw_ms × GPU_OPT_FACTOR + dispatch_overhead
     * This is NOT a measured runtime.
     */
    private static double predictGpuMs(int N) {
        double bytesAccessed = (double) N * 26 * 4 * 2;
        double bwMs = bytesAccessed / (GPU_BANDWIDTH_GB_S * 1e6);
        return bwMs * GPU_OPT_FACTOR + GPU_DISPATCH_OVERHEAD_MS;
    }

    private static void writeRow(int N, String role, double ms, String provenance) throws IOException {
        String row = String.format("%d,%s,%.4f,%s%n", N, role, ms, provenance);
        Files.writeString(Paths.get(PERF_DATA_PATH), row, StandardOpenOption.APPEND);
    }
}
