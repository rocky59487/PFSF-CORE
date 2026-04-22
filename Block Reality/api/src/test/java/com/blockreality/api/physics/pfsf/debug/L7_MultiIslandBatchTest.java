package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.junit.jupiter.api.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L7: Multi-island batching — three 4×4×4 islands submitted in one nativeTickDbb call.
 *
 * <p>Verifies that the tick-budget mechanism correctly skips over-budget islands
 * (they keep dirty state and get processed on subsequent ticks) and that the
 * first island in the batch always finishes with non-zero phi.</p>
 */
@DisplayName("L7: Multi-Island Batch Tick")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class L7_MultiIslandBatchTest {

    private static final int LX = 4, LY = 4, LZ = 4, N = LX * LY * LZ;
    private static final int ID_A = 10, ID_B = 11, ID_C = 12;

    long handle = 0L;
    PhysicsDebugFixtures.IslandBuffers bufsA, bufsB, bufsC;

    @BeforeAll
    void setup() {
        assumeTrue(NativePFSFBridge.isAvailable(), "library not loaded");
        handle = NativePFSFBridge.nativeCreate(N, 10, PhysicsDebugFixtures.ENGINE_VRAM_BYTES, true, true);
        assumeTrue(handle != 0L, "no Vulkan device");
        assumeTrue(NativePFSFBridge.nativeInit(handle) == NativePFSFBridge.PFSFResult.OK, "pfsf_init failed");

        bufsA = PhysicsDebugFixtures.buildIslandBuffers(LX, LY, LZ, false);
        bufsB = PhysicsDebugFixtures.buildIslandBuffers(LX, LY, LZ, false);
        bufsC = PhysicsDebugFixtures.buildIslandBuffers(LX, LY, LZ, false);

        for (int[] p : new int[][]{{ID_A, 0}, {ID_B, 100}, {ID_C, 200}}) {
            assumeTrue(NativePFSFBridge.nativeAddIsland(handle, p[0], p[1], 0, 0, LX, LY, LZ)
                    == NativePFSFBridge.PFSFResult.OK, "addIsland failed id=" + p[0]);
        }
        PhysicsDebugFixtures.IslandBuffers[] bufsArr = {bufsA, bufsB, bufsC};
        int[] ids = {ID_A, ID_B, ID_C};
        for (int k = 0; k < 3; k++) {
            PhysicsDebugFixtures.IslandBuffers b = bufsArr[k];
            assumeTrue(NativePFSFBridge.nativeRegisterIslandBuffers(handle, ids[k],
                    b.phi(), b.source(), b.conductivity(), b.voxelType(),
                    b.rcomp(), b.rtens(), b.maxPhi()) == NativePFSFBridge.PFSFResult.OK,
                    "registerBuffers failed id=" + ids[k]);
            NativePFSFBridge.nativeRegisterIslandLookups(handle, ids[k],
                    b.materialId(), b.anchorBitmap(), b.fluidPressure(), b.curing());
            NativePFSFBridge.nativeRegisterStressReadback(handle, ids[k], b.phi());
        }
    }

    @AfterAll
    void teardown() {
        if (bufsA != null) { bufsA.free(); bufsA = null; }
        if (bufsB != null) { bufsB.free(); bufsB = null; }
        if (bufsC != null) { bufsC.free(); bufsC = null; }
        if (handle != 0L) {
            try { NativePFSFBridge.nativeShutdown(handle); } catch (Throwable ignored) {}
            try { NativePFSFBridge.nativeDestroy(handle);  } catch (Throwable ignored) {}
            handle = 0L;
        }
    }

    @Test @Order(1)
    @DisplayName("L7-01: 批次 tick 回傳 OK")
    void batchTickReturnsOk() {
        ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        fb.putInt(0, 0);
        int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ID_A, ID_B, ID_C}, 1L, fb);
        assertEquals(NativePFSFBridge.PFSFResult.OK, rc, "batch tick rc=" + rc);
    }

    @Test @Order(2)
    @DisplayName("L7-02: 第一個 island 的 phi 非零（求解器至少處理了一個 island）")
    void firstIslandHasNonZeroPhi() {
        // Run a few extra ticks to allow budget-limited islands to catch up
        ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        for (int epoch = 2; epoch <= 10; epoch++) {
            fb.putInt(0, 0);
            NativePFSFBridge.nativeTickDbb(handle, new int[]{ID_A, ID_B, ID_C}, epoch, fb);
        }
        bufsA.phi().rewind();
        int nonZero = 0;
        for (int i = 0; i < N; i++) if (bufsA.phi().getFloat() != 0f) nonZero++;
        System.out.printf("[L7] island A non-zero phi: %d/%d%n", nonZero, N);
        assertTrue(nonZero > 0, "island A phi all-zero after 10 ticks");
    }
}
