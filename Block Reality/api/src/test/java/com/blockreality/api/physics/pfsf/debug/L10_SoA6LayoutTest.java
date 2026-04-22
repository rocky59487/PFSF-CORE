package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.lwjgl.system.MemoryUtil;
import org.junit.jupiter.api.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L10: SoA-6 memory layout verification — sets only Y-direction conductivity
 * (sigma[2*N+i] = NEG_Y, sigma[3*N+i] = POS_Y) to zero, keeps X/Z at 1.0,
 * and verifies that phi varies ONLY along Y after solving.
 *
 * <p>Expected: all voxels at the same Y-level have identical phi (relVar &lt; 1%).</p>
 *
 * <p>If this test fails: the SoA-6 index formula {@code sigma[d*N+i]} is wrong —
 * the shader is reading conductivity from the wrong memory slot (AoS vs SoA bug).
 * See CLAUDE.md §PFSF §26-connectivity consistency requirement.</p>
 */
@DisplayName("L10: SoA-6 Conductivity Layout Verification")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class L10_SoA6LayoutTest {

    private static final int LX = 4, LY = 4, LZ = 4, N = LX * LY * LZ;
    private static final int ISLAND_ID = 100;

    long handle = 0L;
    ByteBuffer phi, source, cond, type, rcomp, rtens, maxPhi;
    ByteBuffer matId, anchor, fluid, curing;

    @BeforeAll
    void setup() {
        assumeTrue(NativePFSFBridge.isAvailable(), "library not loaded");
        handle = NativePFSFBridge.nativeCreate(N, 10, PhysicsDebugFixtures.ENGINE_VRAM_BYTES, true, true);
        assumeTrue(handle != 0L, "no Vulkan device");
        assumeTrue(NativePFSFBridge.nativeInit(handle) == NativePFSFBridge.PFSFResult.OK, "pfsf_init failed");

        phi    = PhysicsDebugFixtures.allocAligned(N * 4);
        source = PhysicsDebugFixtures.allocAligned(N * 4);
        cond   = PhysicsDebugFixtures.allocAligned(6 * N * 4);
        type   = PhysicsDebugFixtures.allocAligned(N);
        rcomp  = PhysicsDebugFixtures.allocAligned(N * 4);
        rtens  = PhysicsDebugFixtures.allocAligned(N * 4);
        maxPhi = PhysicsDebugFixtures.allocAligned(N * 4);
        matId  = PhysicsDebugFixtures.allocAligned(N * 4);
        anchor = PhysicsDebugFixtures.allocAligned(N * 8);
        fluid  = PhysicsDebugFixtures.allocAligned(N * 4);
        curing = PhysicsDebugFixtures.allocAligned(N * 4);

        for (int x = 0; x < LX; x++) {
            for (int y = 0; y < LY; y++) {
                for (int z = 0; z < LZ; z++) {
                    int i = x + LX * (y + LY * z);
                    type.put(i, (byte) 1);
                    source.putFloat(i * 4, 0.5f);
                    rcomp.putFloat(i * 4, 20.0f);
                    rtens.putFloat(i * 4, 2.0f);
                    maxPhi.putFloat(i * 4, 10.0f);
                    curing.putFloat(i * 4, 1.0f);
                    // Only Y-direction conductivity — d=2(NEG_Y), d=3(POS_Y)
                    cond.putFloat((2 * N + i) * 4, 1.0f);
                    cond.putFloat((3 * N + i) * 4, 1.0f);
                    // X and Z conductivity stays at 0 (already zeroed by allocAligned)
                }
            }
        }
        // Anchor y=0 layer
        for (int x = 0; x < LX; x++)
            for (int z = 0; z < LZ; z++)
                anchor.putLong((x + LX * (0 + LY * z)) * 8, 1L);

        assumeTrue(NativePFSFBridge.nativeAddIsland(handle, ISLAND_ID, 0, 0, 0, LX, LY, LZ)
                == NativePFSFBridge.PFSFResult.OK, "addIsland failed");
        assumeTrue(NativePFSFBridge.nativeRegisterIslandBuffers(handle, ISLAND_ID,
                phi, source, cond, type, rcomp, rtens, maxPhi)
                == NativePFSFBridge.PFSFResult.OK, "registerBuffers failed");
        NativePFSFBridge.nativeRegisterIslandLookups(handle, ISLAND_ID, matId, anchor, fluid, curing);
        NativePFSFBridge.nativeRegisterStressReadback(handle, ISLAND_ID, phi);

        // Run 5 ticks so phi can propagate
        ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        for (int epoch = 1; epoch <= 5; epoch++) {
            fb.putInt(0, 0);
            NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, epoch, fb);
        }
    }

    @AfterAll
    void teardown() {
        PhysicsDebugFixtures.free(phi);    PhysicsDebugFixtures.free(source);
        PhysicsDebugFixtures.free(cond);   PhysicsDebugFixtures.free(type);
        PhysicsDebugFixtures.free(rcomp);  PhysicsDebugFixtures.free(rtens);
        PhysicsDebugFixtures.free(maxPhi); PhysicsDebugFixtures.free(matId);
        PhysicsDebugFixtures.free(anchor); PhysicsDebugFixtures.free(fluid);
        PhysicsDebugFixtures.free(curing);
        if (handle != 0L) {
            try { NativePFSFBridge.nativeShutdown(handle); } catch (Throwable ignored) {}
            try { NativePFSFBridge.nativeDestroy(handle);  } catch (Throwable ignored) {}
            handle = 0L;
        }
    }

    @Test
    @DisplayName("L10-01: Y-only 導率 → phi 在同一 Y 層內 X 方向無變化（SoA layout 正確）")
    void yOnlyConductivityMeansNoXVariation() {
        phi.rewind();
        float[] phiArr = new float[N];
        for (int i = 0; i < N; i++) phiArr[i] = phi.getFloat();

        // Check that at each (y, z) slice, all x-values are equal
        float maxXVariation = 0f;
        float maxAbsPhi = 0f;
        for (int y = 1; y < LY; y++) {   // skip y=0 anchors
            for (int z = 0; z < LZ; z++) {
                float ref = phiArr[0 + LX * (y + LY * z)];
                for (int x = 1; x < LX; x++) {
                    float v = phiArr[x + LX * (y + LY * z)];
                    maxXVariation = Math.max(maxXVariation, Math.abs(v - ref));
                    maxAbsPhi = Math.max(maxAbsPhi, Math.abs(v));
                }
            }
        }
        float relVar = maxAbsPhi > 0 ? maxXVariation / maxAbsPhi : 0f;
        System.out.printf("[L10] maxAbsPhi=%.4f  maxXVariation=%.6f  relVar=%.4f%n",
                maxAbsPhi, maxXVariation, relVar);
        assertTrue(maxAbsPhi > 0f, "phi all-zero — solver did not run");
        assertTrue(relVar < 0.01f,
                "SoA-6 layout broken: phi varies in X with Y-only conductivity. relVar=" + relVar);
    }
}
