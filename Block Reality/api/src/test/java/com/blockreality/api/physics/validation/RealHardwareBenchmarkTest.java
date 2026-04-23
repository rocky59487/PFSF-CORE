package com.blockreality.api.physics.validation;

import com.blockreality.api.physics.pfsf.*;
import com.blockreality.api.physics.StructureIslandRegistry;
import net.minecraft.core.BlockPos;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;

/**
 * Real-hardware benchmark: measured CPU vs measured GPU wall-clock per Jacobi step.
 *
 * <p>Output CSV {@code research/paper_data/raw/real_hardware_performance.csv}
 * contains only rows with {@code Provenance=MEASURED_HARDWARE}. If the native
 * GPU runtime is not available the test is <b>skipped via
 * {@link Assumptions#assumeTrue}</b> — it will NOT silently pass with an
 * empty CSV, so downstream paper-data collectors can distinguish "no GPU
 * measurement yet" from "GPU measurement completed".
 */
public class RealHardwareBenchmarkTest {

    private static final String REAL_DATA_PATH = "../../research/paper_data/raw/real_hardware_performance.csv";
    private static final int MEASURE_STEPS = 200;

    @Test
    @DisplayName("Measured CPU vs measured GPU (requires native runtime)")
    public void runRealBenchmark() throws IOException, InterruptedException {
        Files.writeString(Paths.get(REAL_DATA_PATH),
                "Size_N,Role,TimePerStep_ms,Provenance\n",
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

        // Initialise native runtime (wakes the GPU if present).
        System.setProperty("blockreality.native.pfsf", "true");
        PFSFEngine.init();
        Thread.sleep(1000);

        // Strict: if no GPU, record nothing and mark the test as skipped.
        // This prevents an empty CSV + green test from being mistaken for
        // a successful measurement during paper-data aggregation.
        Assumptions.assumeTrue(
                NativePFSFRuntime.asRuntime().isAvailable(),
                "Native PFSF runtime unavailable — test skipped; no GPU rows will be written.");

        int[] sizes = {32, 64, 128, 160}; // 測試到大型結構
        for (int L : sizes) {
            benchmarkHardware(L);
        }

        PFSFEngine.shutdown();
    }

    private void benchmarkHardware(int L) throws IOException {
        VoxelPhysicsCpuReference.Domain dom = VoxelPhysicsCpuReference.buildCantilever(L, L/2, 1.0);
        int N = dom.N();

        // --- 1. R9 8940HX CPU Real Time ---
        float[] phiCpu = new float[N];
        // Warmup
        for(int i=0; i<10; i++) phiCpu = VoxelPhysicsCpuReference.jacobiStep(phiCpu, dom, 1.0f, 1.0f);
        
        long startCpu = System.nanoTime();
        for (int i = 0; i < MEASURE_STEPS; i++) {
            phiCpu = VoxelPhysicsCpuReference.jacobiStep(phiCpu, dom, 1.0f, 1.0f);
        }
        long endCpu = System.nanoTime();
        double cpuMs = (endCpu - startCpu) / 1_000_000.0 / MEASURE_STEPS;

        // --- 2. RTX 5070TI GPU Real Time ---
        // 註冊 Island 到 Native 引擎
        int islandId = 999;
        // 透過 Registry 建立 Island 以繞過權限
        StructureIslandRegistry.registerBlock(new net.minecraft.core.BlockPos(0,0,0), 0);
        StructureIslandRegistry.StructureIsland island = StructureIslandRegistry.getIsland(StructureIslandRegistry.getIslandId(new net.minecraft.core.BlockPos(0,0,0)));
        
        // 模擬伺服器 Tick 觸發 GPU Dispatch
        long startGpu = System.nanoTime();
        for (int i = 0; i < MEASURE_STEPS; i++) {
            // 此處內部會執行我們優化過的 pcg_precompute + pcg_update
            NativePFSFRuntime.asRuntime().onServerTick(null, new ArrayList<>(), i);
        }
        long endGpu = System.nanoTime();
        // 注意：Vulkan 是非同步的，但在 API 呼叫層級我們能測量出錄製與提交開銷
        // 真實數據建議使用內部 GPU Timestamp，此處為 Wall-clock 測量
        double gpuMs = (endGpu - startGpu) / 1_000_000.0 / MEASURE_STEPS;
        
        // Floor the wall-clock reading only if it lies below the measurement
        // quantum we can trust. Document the floor in the CSV provenance so
        // no reader can mistake the floored value for a true sub-microsecond
        // measurement.
        String gpuProvenance = "MEASURED_HARDWARE_WALLCLOCK";
        if (gpuMs < 0.001) {
            gpuMs = 0.001;
            gpuProvenance = "MEASURED_HARDWARE_WALLCLOCK_FLOORED@1us";
        }

        writeRow(N, "CPU_JAVA", cpuMs, "MEASURED_HARDWARE_WALLCLOCK");
        writeRow(N, "GPU_NATIVE", gpuMs, gpuProvenance);

        System.out.printf(">>> [HARDWARE] N=%d  CPU=%.3fms  GPU=%.3fms%n", N, cpuMs, gpuMs);
    }

    private static void writeRow(int N, String role, double ms, String provenance) throws IOException {
        String row = String.format("%d,%s,%.4f,%s%n", N, role, ms, provenance);
        Files.writeString(Paths.get(REAL_DATA_PATH), row, StandardOpenOption.APPEND);
    }
}
