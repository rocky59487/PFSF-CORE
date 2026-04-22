package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.junit.jupiter.api.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L12: Source-free island — all source terms are zero.
 *
 * <p>The linear system A×φ = source becomes A×φ = 0.
 * φ = 0 is the unique solution when the system has a ground (anchor at φ=0).
 * If phi is non-zero after solving, the solver has a numerical drift bug
 * or the host→GPU buffer upload did not clear old values.</p>
 */
@DisplayName("L12: Source-Free Island (phi must be zero)")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class L12_SourceFreeIslandTest {

    private static final int LX = 4, LY = 4, LZ = 4, N = LX * LY * LZ;
    private static final int ISLAND_ID = 120;

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
        source = PhysicsDebugFixtures.allocAligned(N * 4); // all-zero
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
                    // source = 0 (already zero)
                    rcomp.putFloat(i * 4, 20.0f);
                    rtens.putFloat(i * 4, 2.0f);
                    maxPhi.putFloat(i * 4, 10.0f);
                    curing.putFloat(i * 4, 1.0f);
                    for (int d = 0; d < 6; d++)
                        cond.putFloat((d * N + i) * 4, 1.0f);
                }
            }
        }
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

        ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        fb.putInt(0, 0);
        NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, 1L, fb);
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
    @DisplayName("L12-01: 零源項 → phi 精確為零（A×0=0 正確解）")
    void zeroSourceMeansZeroPhi() {
        phi.rewind();
        float maxAbsPhi = 0f;
        for (int i = 0; i < N; i++) maxAbsPhi = Math.max(maxAbsPhi, Math.abs(phi.getFloat()));
        System.out.printf("[L12] maxAbsPhi=%.8f%n", maxAbsPhi);
        assertTrue(maxAbsPhi < 1e-3f,
                "phi should be ~0 when source=0, but maxAbsPhi=" + maxAbsPhi);
    }
}
