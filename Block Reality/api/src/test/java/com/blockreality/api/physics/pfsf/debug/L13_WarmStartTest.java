package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.junit.jupiter.api.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L13: Warm-start epoch consistency — runs 20 consecutive ticks and verifies
 * that phi converges stably without drift.
 *
 * <p>Each tick, {@code nativeRegisterStressReadback} causes the C++ engine to
 * write the computed phi back into the Java-visible buffer. The next tick reads
 * this warm solution instead of starting from φ=0, which should accelerate
 * convergence. If phi drifts more than 1% between tick 1 and tick 20, the
 * warm-start path is re-reading stale or garbage values.</p>
 */
@DisplayName("L13: Warm-Start Epoch Consistency (20 ticks)")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class L13_WarmStartTest {

    private static final int LX = 4, LY = 4, LZ = 4, N = LX * LY * LZ;
    private static final int ISLAND_ID = 130;
    private static final int TICKS = 20;

    long handle = 0L;
    PhysicsDebugFixtures.IslandBuffers bufs;

    @BeforeAll
    void setup() {
        assumeTrue(NativePFSFBridge.isAvailable(), "library not loaded");
        handle = NativePFSFBridge.nativeCreate(N, 10, PhysicsDebugFixtures.ENGINE_VRAM_BYTES, true, true);
        assumeTrue(handle != 0L, "no Vulkan device");
        assumeTrue(NativePFSFBridge.nativeInit(handle) == NativePFSFBridge.PFSFResult.OK, "pfsf_init failed");

        bufs = PhysicsDebugFixtures.buildIslandBuffers(LX, LY, LZ, false);
        assumeTrue(NativePFSFBridge.nativeAddIsland(handle, ISLAND_ID, 0, 0, 0, LX, LY, LZ)
                == NativePFSFBridge.PFSFResult.OK, "addIsland failed");
        assumeTrue(NativePFSFBridge.nativeRegisterIslandBuffers(handle, ISLAND_ID,
                bufs.phi(), bufs.source(), bufs.conductivity(), bufs.voxelType(),
                bufs.rcomp(), bufs.rtens(), bufs.maxPhi()) == NativePFSFBridge.PFSFResult.OK,
                "registerBuffers failed");
        NativePFSFBridge.nativeRegisterIslandLookups(handle, ISLAND_ID,
                bufs.materialId(), bufs.anchorBitmap(), bufs.fluidPressure(), bufs.curing());
        NativePFSFBridge.nativeRegisterStressReadback(handle, ISLAND_ID, bufs.phi());
    }

    @AfterAll
    void teardown() {
        if (bufs   != null) { bufs.free(); bufs = null; }
        if (handle != 0L) {
            try { NativePFSFBridge.nativeShutdown(handle); } catch (Throwable ignored) {}
            try { NativePFSFBridge.nativeDestroy(handle);  } catch (Throwable ignored) {}
            handle = 0L;
        }
    }

    @Test @Order(1)
    @DisplayName("L13-01: 20 ticks 中 phi 無 NaN/Inf")
    void noNanOrInfOver20Ticks() {
        ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        for (int epoch = 1; epoch <= TICKS; epoch++) {
            fb.putInt(0, 0);
            int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, epoch, fb);
            assertEquals(NativePFSFBridge.PFSFResult.OK, rc, "tick " + epoch + " failed");
            bufs.phi().rewind();
            for (int i = 0; i < N; i++) {
                float v = bufs.phi().getFloat();
                assertFalse(Float.isNaN(v),   "phi[" + i + "] is NaN at epoch " + epoch);
                assertFalse(Float.isInfinite(v), "phi[" + i + "] is Inf at epoch " + epoch);
            }
        }
    }

    @Test @Order(2)
    @DisplayName("L13-02: tick 1 與 tick 20 的 maxPhi 漂移 < 1%（暖啟動穩定）")
    void phiStableOver20Ticks() {
        ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);

        fb.putInt(0, 0);
        NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, 100L, fb);
        bufs.phi().rewind();
        float maxPhi1 = 0f;
        for (int i = 0; i < N; i++) maxPhi1 = Math.max(maxPhi1, bufs.phi().getFloat());

        for (int epoch = 101; epoch <= 119; epoch++) {
            fb.putInt(0, 0);
            NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, epoch, fb);
        }
        fb.putInt(0, 0);
        NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, 120L, fb);
        bufs.phi().rewind();
        float maxPhi20 = 0f;
        for (int i = 0; i < N; i++) maxPhi20 = Math.max(maxPhi20, bufs.phi().getFloat());

        assumeTrue(maxPhi1 > 0f, "phi still zero — solver did not run (check L4 first)");
        float drift = Math.abs(maxPhi20 - maxPhi1) / maxPhi1;
        System.out.printf("[L13] tick1=%.4f  tick20=%.4f  drift=%.4f%n", maxPhi1, maxPhi20, drift);
        assertTrue(drift < 0.01f, "Warm-start drift " + drift + " exceeds 1% over 20 ticks");
    }
}
