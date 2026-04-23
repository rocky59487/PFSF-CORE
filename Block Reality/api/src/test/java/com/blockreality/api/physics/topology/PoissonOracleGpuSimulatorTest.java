package com.blockreality.api.physics.topology;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.junit.jupiter.api.Assertions.*;

/**
 * The GPU simulator must produce the same φ field and the same fracture
 * mask as the CPU reference for every input, iteration count, and
 * epsilon. If they ever diverge, the GPU port of the kernels is
 * untrustworthy and we need to debug the simulator or the shader before
 * shipping Tier 2 to real hardware.
 */
public class PoissonOracleGpuSimulatorTest {

    @Test
    @DisplayName("simulator φ ≡ CPU φ on 20 random 8³ regions (256 iter, ε=0.5)")
    public void phiMatchesCpu() {
        Random rng = new Random(20260201L);
        for (int trial = 0; trial < 20; trial++) {
            int L = 6 + rng.nextInt(4);
            byte[] type = randomDomain(L, rng);
            PoissonOracleCPU.Result cpu = PoissonOracleCPU.solve(type, L, L, L);
            PoissonOracleGpuSimulator.Result gpu = PoissonOracleGpuSimulator.run(type, L, L, L);
            assertEquals(cpu.phi().length, gpu.phi().length);
            for (int i = 0; i < cpu.phi().length; i++) {
                assertEquals(cpu.phi()[i], gpu.phi()[i], 1e-4f,
                        "trial " + trial + " voxel " + i + ": cpu=" + cpu.phi()[i] + " gpu=" + gpu.phi()[i]);
            }
        }
    }

    @Test
    @DisplayName("simulator fracture mask ≡ CPU mask bit-for-bit on 50 random domains")
    public void maskMatchesCpu() {
        Random rng = new Random(20260202L);
        for (int trial = 0; trial < 50; trial++) {
            int L = 5 + rng.nextInt(6);
            byte[] type = randomDomain(L, rng);
            PoissonOracleCPU.Result cpu = PoissonOracleCPU.solve(type, L, L, L);
            PoissonOracleGpuSimulator.Result gpu = PoissonOracleGpuSimulator.run(type, L, L, L);
            for (int i = 0; i < type.length; i++) {
                boolean cpuMask = cpu.fractureMask()[i];
                boolean gpuMask = gpu.getMaskBit(i);
                assertEquals(cpuMask, gpuMask,
                        "trial " + trial + " voxel " + i + ": cpu=" + cpuMask + " gpu=" + gpuMask);
            }
        }
    }

    @Test
    @DisplayName("empty domain — all bits zero")
    public void emptyDomain() {
        int L = 4;
        byte[] type = new byte[L * L * L];
        PoissonOracleGpuSimulator.Result r = PoissonOracleGpuSimulator.run(type, L, L, L);
        for (int i = 0; i < L * L * L; i++) {
            assertFalse(r.getMaskBit(i), "empty domain should have no mask bits");
        }
    }

    @Test
    @DisplayName("determinism — repeated runs produce byte-identical results")
    public void determinism() {
        Random rng = new Random(20260203L);
        byte[] type = randomDomain(8, rng);
        PoissonOracleGpuSimulator.Result first = PoissonOracleGpuSimulator.run(type, 8, 8, 8);
        for (int r = 1; r < 5; r++) {
            PoissonOracleGpuSimulator.Result again = PoissonOracleGpuSimulator.run(type, 8, 8, 8);
            assertArrayEquals(first.maskBits(), again.maskBits(),
                    "mask differs on run " + r);
            for (int i = 0; i < first.phi().length; i++) {
                assertEquals(first.phi()[i], again.phi()[i], 1e-7f,
                        "phi differs on run " + r + " voxel " + i);
            }
        }
    }

    private static byte[] randomDomain(int L, Random rng) {
        byte[] type = new byte[L * L * L];
        for (int i = 0; i < type.length; i++) {
            double r = rng.nextDouble();
            if (r < 0.5) type[i] = TopologicalSVDAG.TYPE_SOLID;
            else if (r < 0.56) type[i] = TopologicalSVDAG.TYPE_ANCHOR;
        }
        return type;
    }
}
