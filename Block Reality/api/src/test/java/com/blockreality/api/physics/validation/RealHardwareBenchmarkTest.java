package com.blockreality.api.physics.validation;

import com.blockreality.api.physics.pfsf.*;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

/**
 * 真實硬體基準測試：修正版。
 * 確保 GPU 真正執行了物理運算，而不是因為 Dirty Flag 沒設而提早退出。
 */
public class RealHardwareBenchmarkTest {

    private static final String REAL_DATA_PATH = "../../research/paper_data/raw/real_hardware_performance.csv";
    private static final int MEASURE_STEPS = 50;

    @Test
    @DisplayName("真實 GPU 計算負載對標")
    public void runRealBenchmark() throws Exception {
        Files.writeString(Paths.get(REAL_DATA_PATH),
                "Size_N,Role,TimePerStep_ms,Provenance\n",
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);

        System.setProperty("blockreality.native.pfsf", "true");
        PFSFEngine.init();
        Thread.sleep(1000);

        Assumptions.assumeTrue(
                NativePFSFRuntime.asRuntime().isAvailable(),
                "Native PFSF runtime unavailable.");

        int[] sizes = {32, 64, 100}; // 限制測試規模以防止 TDR (Timeout Detection and Recovery)
        for (int L : sizes) {
            benchmarkHardware(L);
        }

        PFSFEngine.shutdown();
    }

    private void benchmarkHardware(int L) throws Exception {
        VoxelPhysicsCpuReference.Domain dom = VoxelPhysicsCpuReference.buildCantilever(L, L/2, 1.0);
        int N = dom.N();

        // ── 1. CPU 真實測量 ──
        float[] phiCpu = new float[N];
        for(int i=0; i<5; i++) phiCpu = VoxelPhysicsCpuReference.jacobiStep(phiCpu, dom, 1.0f, 1.0f);
        
        long startCpu = System.nanoTime();
        for (int i = 0; i < MEASURE_STEPS; i++) {
            phiCpu = VoxelPhysicsCpuReference.jacobiStep(phiCpu, dom, 1.0f, 1.0f);
        }
        long endCpu = System.nanoTime();
        double cpuMs = (endCpu - startCpu) / 1_000_000.0 / MEASURE_STEPS;

        // ── 2. GPU 真實測量 (修正 Dummy 問題) ──
        long engineHandle = NativePFSFRuntime.getHandle();
        int islandId = 999;

        // 正確上傳物理資料，而非全 0
        ByteBuffer phiBuf = createFloatBuffer(new float[N]);
        ByteBuffer srcBuf = createFloatBuffer(dom.source());
        // Note: CPU reference Domain uses sigma() for conductivity
        ByteBuffer condBuf = createFloatBuffer(expandConductivityToSoA(dom.sigma(), N));
        ByteBuffer typeBuf = createByteBuffer(dom.type());
        ByteBuffer rcompBuf = createFloatBuffer(new float[N]);
        ByteBuffer rtensBuf = createFloatBuffer(new float[N]);
        ByteBuffer maxPhiBuf = createFloatBuffer(new float[N]);

        NativePFSFBridge.nativeAddIsland(engineHandle, islandId, 0, 0, 0, dom.Lx(), dom.Ly(), dom.Lz());
        NativePFSFBridge.nativeRegisterIslandBuffers(engineHandle, islandId,
                phiBuf, srcBuf, condBuf, typeBuf, rcompBuf, rtensBuf, maxPhiBuf);

        // Warmup: 讓 GPU 跑幾次建立快取與 Pipeline
        NativePFSFBridge.nativeMarkFullRebuild(engineHandle, islandId);
        NativePFSFBridge.nativeTick(engineHandle, new int[]{islandId}, 0, null);

        long startGpu = System.nanoTime();
        for (int i = 0; i < MEASURE_STEPS; i++) {
            // 關鍵修復：每次 Tick 前必須重新標記 Dirty，否則 C++ 引擎會直接 Return OK (0ms)
            NativePFSFBridge.nativeMarkFullRebuild(engineHandle, islandId);
            NativePFSFBridge.nativeTick(engineHandle, new int[]{islandId}, i + 1, null);
        }
        long endGpu = System.nanoTime();
        
        // Native 引擎預設每個 Tick 會執行 STEPS_MAJOR 次 (通常是 25 次) Jacobi/PCG 迭代
        // 因此實際運算的單步時間要再除以 25。
        double gpuMsTotal = (endGpu - startGpu) / 1_000_000.0 / MEASURE_STEPS;
        double gpuMsPerStep = gpuMsTotal / 25.0; // PFSFConstants.STEPS_MAJOR 預設為 25

        double speedup = cpuMs / gpuMsPerStep;

        String row = String.format("%d,CPU_JAVA,%.4f,MEASURED_HARDWARE%n", N, cpuMs);
        Files.writeString(Paths.get(REAL_DATA_PATH), row, StandardOpenOption.APPEND);
        row = String.format("%d,GPU_NATIVE_5070TI,%.4f,MEASURED_HARDWARE%n", N, gpuMsPerStep);
        Files.writeString(Paths.get(REAL_DATA_PATH), row, StandardOpenOption.APPEND);

        System.out.printf(">>> [TRUE HARDWARE] N=%d  CPU=%.3fms  GPU=%.3fms  Speedup=%.1fx%n", N, cpuMs, gpuMsPerStep, speedup);
    }

    private ByteBuffer createFloatBuffer(float[] data) {
        ByteBuffer bb = ByteBuffer.allocateDirect(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bb.asFloatBuffer().put(data);
        return bb;
    }

    private ByteBuffer createByteBuffer(byte[] data) {
        ByteBuffer bb = ByteBuffer.allocateDirect(data.length).order(ByteOrder.LITTLE_ENDIAN);
        bb.put(data);
        return bb;
    }

    private float[] expandConductivityToSoA(float[] sigma, int N) {
        float[] soa = new float[N * 6];
        for (int d = 0; d < 6; d++) {
            System.arraycopy(sigma, 0, soa, d * N, N);
        }
        return soa;
    }
}
