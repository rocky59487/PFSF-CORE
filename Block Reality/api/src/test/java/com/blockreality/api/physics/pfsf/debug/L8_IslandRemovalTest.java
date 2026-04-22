package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.junit.jupiter.api.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L8: Island removal — verifies that removing an island while others are active does
 * not crash the engine and that ticking the removed island ID is handled gracefully.
 */
@DisplayName("L8: Island Removal During Solve")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class L8_IslandRemovalTest {

    private static final int LX = 4, LY = 4, LZ = 4, N = LX * LY * LZ;
    private static final int ID_ALIVE = 81, ID_DEAD = 82;

    long handle = 0L;
    PhysicsDebugFixtures.IslandBuffers bufsAlive, bufsDead;

    @BeforeAll
    void setup() {
        assumeTrue(NativePFSFBridge.isAvailable(), "library not loaded");
        handle = NativePFSFBridge.nativeCreate(N, 10, PhysicsDebugFixtures.ENGINE_VRAM_BYTES, true, true);
        assumeTrue(handle != 0L, "no Vulkan device");
        assumeTrue(NativePFSFBridge.nativeInit(handle) == NativePFSFBridge.PFSFResult.OK, "pfsf_init failed");

        bufsAlive = PhysicsDebugFixtures.buildIslandBuffers(LX, LY, LZ, false);
        bufsDead  = PhysicsDebugFixtures.buildIslandBuffers(LX, LY, LZ, false);

        for (int[] p : new int[][]{{ID_ALIVE, 0}, {ID_DEAD, 200}}) {
            assumeTrue(NativePFSFBridge.nativeAddIsland(handle, p[0], p[1], 0, 0, LX, LY, LZ)
                    == NativePFSFBridge.PFSFResult.OK, "addIsland failed id=" + p[0]);
        }
        for (int[] p : new int[][]{{ID_ALIVE, 0}, {ID_DEAD, 1}}) {
            PhysicsDebugFixtures.IslandBuffers b = p[1] == 0 ? bufsAlive : bufsDead;
            int id = p[0];
            assumeTrue(NativePFSFBridge.nativeRegisterIslandBuffers(handle, id,
                    b.phi(), b.source(), b.conductivity(), b.voxelType(),
                    b.rcomp(), b.rtens(), b.maxPhi()) == NativePFSFBridge.PFSFResult.OK,
                    "registerBuffers failed id=" + id);
            NativePFSFBridge.nativeRegisterIslandLookups(handle, id,
                    b.materialId(), b.anchorBitmap(), b.fluidPressure(), b.curing());
            NativePFSFBridge.nativeRegisterStressReadback(handle, id, b.phi());
        }
    }

    @AfterAll
    void teardown() {
        if (bufsAlive != null) { bufsAlive.free(); bufsAlive = null; }
        if (bufsDead  != null) { bufsDead.free();  bufsDead  = null; }
        if (handle != 0L) {
            try { NativePFSFBridge.nativeShutdown(handle); } catch (Throwable ignored) {}
            try { NativePFSFBridge.nativeDestroy(handle);  } catch (Throwable ignored) {}
            handle = 0L;
        }
    }

    @Test @Order(1)
    @DisplayName("L8-01: 移除 island 不崩潰，存活 island 繼續正常 tick")
    void removeIslandNoCrash() {
        NativePFSFBridge.nativeRemoveIsland(handle, ID_DEAD);

        ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        fb.putInt(0, 0);
        int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ID_ALIVE}, 1L, fb);
        assertEquals(NativePFSFBridge.PFSFResult.OK, rc,
                "tick after removal should succeed for the alive island");
    }

    @Test @Order(2)
    @DisplayName("L8-02: tick 已移除的 island ID 不崩潰（回傳 OK 或 ISLAND_NOT_FOUND）")
    void tickRemovedIslandGraceful() {
        ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        fb.putInt(0, 0);
        int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ID_DEAD}, 2L, fb);
        // Acceptable: OK (island silently skipped) or -2 (ISLAND_NOT_FOUND)
        assertTrue(rc == NativePFSFBridge.PFSFResult.OK || rc == -2,
                "unexpected rc=" + rc + " when ticking removed island");
    }
}
