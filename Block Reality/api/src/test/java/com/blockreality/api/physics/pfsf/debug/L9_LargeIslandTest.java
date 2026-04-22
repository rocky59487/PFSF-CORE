package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.junit.jupiter.api.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L9: Large island (8×8×8, 512 voxels) — verifies that the solver handles a
 * production-scale island within a reasonable time budget and produces non-zero phi.
 *
 * <p>On lavapipe the tick takes ~35-40 ms. On a real GPU it should be &lt;5 ms.</p>
 */
@DisplayName("L9: Large Island (8×8×8)")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class L9_LargeIslandTest {

    private static final int LX = 8, LY = 8, LZ = 8;
    private static final int N = LX * LY * LZ; // 512
    private static final int ISLAND_ID = 90;
    private static final long TICK_MS_LIMIT = 2000L;

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
    @DisplayName("L9-01: 8×8×8 tick 完成且時間 < " + TICK_MS_LIMIT + " ms")
    void tickCompletesInTime() {
        ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        fb.putInt(0, 0);
        long t0 = System.currentTimeMillis();
        int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, 1L, fb);
        long elapsed = System.currentTimeMillis() - t0;
        System.out.printf("[L9] tick time=%d ms%n", elapsed);
        assertEquals(NativePFSFBridge.PFSFResult.OK, rc, "tick rc=" + rc);
        assertTrue(elapsed < TICK_MS_LIMIT,
                "Tick took " + elapsed + " ms (limit " + TICK_MS_LIMIT + " ms)");
    }

    @Test @Order(2)
    @DisplayName("L9-02: 8×8×8 phi 非零（求解器有效）")
    void phiNonZero() {
        bufs.phi().rewind();
        int nonZero = 0;
        float maxV = 0f;
        for (int i = 0; i < N; i++) {
            float v = bufs.phi().getFloat();
            if (v != 0f) nonZero++;
            if (v > maxV) maxV = v;
        }
        System.out.printf("[L9] non-zero phi: %d/%d  max=%.4f%n", nonZero, N, maxV);
        assertTrue(nonZero > 0, "phi all-zero after 8×8×8 tick");
    }
}
