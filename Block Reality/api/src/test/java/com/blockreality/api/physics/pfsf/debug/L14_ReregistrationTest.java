package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.junit.jupiter.api.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L14: Re-registration after removal — same island ID re-added with a different AABB.
 *
 * <p>This exercises the {@code IslandBufferPool.getOrCreate()} re-allocation path:
 * when lx/ly/lz change for the same ID, the C++ engine must release and reallocate
 * the GPU SSBOs. Verifies that the new island produces non-zero phi after re-registration.</p>
 *
 * <p>This path is triggered in-game whenever a structure expands (block placed at edge
 * of AABB) or contracts (AABB recalculated after block removal).</p>
 */
@DisplayName("L14: Re-registration After Removal (same ID, different AABB)")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class L14_ReregistrationTest {

    private static final int LX1 = 4, LY1 = 4, LZ1 = 4, N1 = LX1 * LY1 * LZ1;
    private static final int LX2 = 6, LY2 = 6, LZ2 = 6, N2 = LX2 * LY2 * LZ2;
    private static final int ISLAND_ID = 140;

    long handle = 0L;

    @BeforeAll
    void setup() {
        assumeTrue(NativePFSFBridge.isAvailable(), "library not loaded");
        // max_island_size must accommodate N2=216
        handle = NativePFSFBridge.nativeCreate(N2, 10, PhysicsDebugFixtures.ENGINE_VRAM_BYTES, true, true);
        assumeTrue(handle != 0L, "no Vulkan device");
        assumeTrue(NativePFSFBridge.nativeInit(handle) == NativePFSFBridge.PFSFResult.OK, "pfsf_init failed");
    }

    @AfterAll
    void teardown() {
        if (handle != 0L) {
            NativePFSFBridge.nativeRemoveIsland(handle, ISLAND_ID);
            try { NativePFSFBridge.nativeShutdown(handle); } catch (Throwable ignored) {}
            try { NativePFSFBridge.nativeDestroy(handle);  } catch (Throwable ignored) {}
            handle = 0L;
        }
    }

    @Test @Order(1)
    @DisplayName("L14-01: 4×4×4 首次 tick 成功，phi 非零")
    void initialRegistrationWorks() {
        PhysicsDebugFixtures.IslandBuffers b = PhysicsDebugFixtures.buildIslandBuffers(LX1, LY1, LZ1, false);
        try {
            assumeTrue(NativePFSFBridge.nativeAddIsland(handle, ISLAND_ID, 0, 0, 0, LX1, LY1, LZ1)
                    == NativePFSFBridge.PFSFResult.OK, "addIsland 4×4×4 failed");
            assumeTrue(NativePFSFBridge.nativeRegisterIslandBuffers(handle, ISLAND_ID,
                    b.phi(), b.source(), b.conductivity(), b.voxelType(),
                    b.rcomp(), b.rtens(), b.maxPhi()) == NativePFSFBridge.PFSFResult.OK, "registerBuffers failed");
            NativePFSFBridge.nativeRegisterIslandLookups(handle, ISLAND_ID,
                    b.materialId(), b.anchorBitmap(), b.fluidPressure(), b.curing());
            NativePFSFBridge.nativeRegisterStressReadback(handle, ISLAND_ID, b.phi());

            ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
            fb.putInt(0, 0);
            int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, 1L, fb);
            assertEquals(NativePFSFBridge.PFSFResult.OK, rc, "first tick failed");

            b.phi().rewind();
            int nonZero = 0;
            for (int i = 0; i < N1; i++) if (b.phi().getFloat() != 0f) nonZero++;
            assertTrue(nonZero > 0, "4×4×4 phi all-zero");
        } finally {
            b.free();
        }
    }

    @Test @Order(2)
    @DisplayName("L14-02: remove → re-add 6×6×6 同一 ID → phi 非零")
    void reregistrationWithDifferentAabbWorks() {
        NativePFSFBridge.nativeRemoveIsland(handle, ISLAND_ID);

        PhysicsDebugFixtures.IslandBuffers b2 = PhysicsDebugFixtures.buildIslandBuffers(LX2, LY2, LZ2, false);
        try {
            int rc = NativePFSFBridge.nativeAddIsland(handle, ISLAND_ID, 0, 64, 0, LX2, LY2, LZ2);
            assertEquals(NativePFSFBridge.PFSFResult.OK, rc, "re-add 6×6×6 failed rc=" + rc);
            assumeTrue(NativePFSFBridge.nativeRegisterIslandBuffers(handle, ISLAND_ID,
                    b2.phi(), b2.source(), b2.conductivity(), b2.voxelType(),
                    b2.rcomp(), b2.rtens(), b2.maxPhi()) == NativePFSFBridge.PFSFResult.OK, "re-registerBuffers failed");
            NativePFSFBridge.nativeRegisterIslandLookups(handle, ISLAND_ID,
                    b2.materialId(), b2.anchorBitmap(), b2.fluidPressure(), b2.curing());
            NativePFSFBridge.nativeRegisterStressReadback(handle, ISLAND_ID, b2.phi());

            ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
            fb.putInt(0, 0);
            rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, 2L, fb);
            assertEquals(NativePFSFBridge.PFSFResult.OK, rc, "re-registered tick failed");

            b2.phi().rewind();
            float maxV = 0f;
            for (int i = 0; i < N2; i++) maxV = Math.max(maxV, b2.phi().getFloat());
            System.out.printf("[L14] 6×6×6 maxPhi=%.4f%n", maxV);
            assertTrue(maxV > 0f, "6×6×6 phi still all-zero after re-registration");
        } finally {
            b2.free();
        }
    }
}
