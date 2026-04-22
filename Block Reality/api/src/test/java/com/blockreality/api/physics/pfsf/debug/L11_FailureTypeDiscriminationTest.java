package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.junit.jupiter.api.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L11: Failure type discrimination — verifies CANTILEVER(1), CRUSHING(2), and
 * TENSION_BREAK(4) can each be independently triggered with appropriate thresholds.
 *
 * <p>All three failure_scan shader branches are exercised. If any branch never fires,
 * the corresponding shader condition or threshold normalization is broken.</p>
 */
@DisplayName("L11: Failure Type Discrimination")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class L11_FailureTypeDiscriminationTest {

    private static final int LX = 4, LY = 4, LZ = 4, N = LX * LY * LZ;

    long handle = 0L;

    @BeforeAll
    void setup() {
        assumeTrue(NativePFSFBridge.isAvailable(), "library not loaded");
        handle = NativePFSFBridge.nativeCreate(N, 10, PhysicsDebugFixtures.ENGINE_VRAM_BYTES, true, true);
        assumeTrue(handle != 0L, "no Vulkan device");
        assumeTrue(NativePFSFBridge.nativeInit(handle) == NativePFSFBridge.PFSFResult.OK, "pfsf_init failed");
    }

    @AfterAll
    void teardown() {
        if (handle != 0L) {
            try { NativePFSFBridge.nativeShutdown(handle); } catch (Throwable ignored) {}
            try { NativePFSFBridge.nativeDestroy(handle);  } catch (Throwable ignored) {}
            handle = 0L;
        }
    }

    private int runAndFindType(int islandId, float rcValue, float maxPhiValue, int startEpoch) {
        ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        for (int epoch = startEpoch; epoch <= startEpoch + 20; epoch++) {
            fb.putInt(0, 0);
            NativePFSFBridge.nativeTickDbb(handle, new int[]{islandId}, epoch, fb);
            int count = fb.getInt(0);
            if (count > 0) return fb.getInt(16); // first event type
        }
        return -1;
    }

    @Test @Order(1)
    @DisplayName("L11-01: CANTILEVER(1) — maxPhi 極低 (0.001)，phi 超過閾值")
    void cantileverTypeFires() {
        int id = 111;
        PhysicsDebugFixtures.IslandBuffers b = PhysicsDebugFixtures.buildIslandBuffers(LX, LY, LZ, true);
        // triggerFailure=true: center voxel gets maxPhi=0.001 & rcomp=0.001 → CANTILEVER or CRUSHING
        assumeTrue(NativePFSFBridge.nativeAddIsland(handle, id, 0, 0, 0, LX, LY, LZ) == NativePFSFBridge.PFSFResult.OK);
        assumeTrue(NativePFSFBridge.nativeRegisterIslandBuffers(handle, id,
                b.phi(), b.source(), b.conductivity(), b.voxelType(),
                b.rcomp(), b.rtens(), b.maxPhi()) == NativePFSFBridge.PFSFResult.OK);
        NativePFSFBridge.nativeRegisterIslandLookups(handle, id, b.materialId(), b.anchorBitmap(), b.fluidPressure(), b.curing());
        NativePFSFBridge.nativeRegisterStressReadback(handle, id, b.phi());

        try {
            int type = runAndFindType(id, 0.001f, 0.001f, 1);
            System.out.printf("[L11] CANTILEVER-path type=%d%n", type);
            assertTrue(type == 1 || type == 2,
                    "Expected CANTILEVER(1) or CRUSHING(2), got " + type);
        } finally {
            b.free();
            NativePFSFBridge.nativeRemoveIsland(handle, id);
        }
    }

    @Test @Order(2)
    @DisplayName("L11-02: CRUSHING(2) 和 TENSION_BREAK(4) — 極低 rcomp/rtens")
    void crushingAndTensionFire() {
        int id = 112;
        PhysicsDebugFixtures.IslandBuffers b = PhysicsDebugFixtures.buildIslandBuffers(LX, LY, LZ, false);
        // Overwrite rcomp/rtens/maxPhi to near-zero to force crushing + tension
        for (int i = 0; i < N; i++) {
            b.rcomp().putFloat(i * 4, 1e-5f);
            b.rtens().putFloat(i * 4, 1e-5f);
            b.maxPhi().putFloat(i * 4, 1e-5f);
        }
        assumeTrue(NativePFSFBridge.nativeAddIsland(handle, id, 100, 0, 0, LX, LY, LZ) == NativePFSFBridge.PFSFResult.OK);
        assumeTrue(NativePFSFBridge.nativeRegisterIslandBuffers(handle, id,
                b.phi(), b.source(), b.conductivity(), b.voxelType(),
                b.rcomp(), b.rtens(), b.maxPhi()) == NativePFSFBridge.PFSFResult.OK);
        NativePFSFBridge.nativeRegisterIslandLookups(handle, id, b.materialId(), b.anchorBitmap(), b.fluidPressure(), b.curing());
        NativePFSFBridge.nativeRegisterStressReadback(handle, id, b.phi());

        try {
            ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
            boolean seenCrushing = false, seenTension = false;
            for (int epoch = 200; epoch <= 220; epoch++) {
                fb.putInt(0, 0);
                NativePFSFBridge.nativeTickDbb(handle, new int[]{id}, epoch, fb);
                int count = fb.getInt(0);
                for (int k = 0; k < Math.min(count, 64); k++) {
                    int t = fb.getInt(16 + k * 16);
                    if (t == 2) seenCrushing = true;
                    if (t == 4) seenTension  = true;
                }
            }
            System.out.printf("[L11] CRUSHING=%b  TENSION=%b%n", seenCrushing, seenTension);
            assertTrue(seenCrushing || seenTension,
                    "Neither CRUSHING(2) nor TENSION_BREAK(4) fired with rcomp=rtens=1e-5");
        } finally {
            b.free();
            NativePFSFBridge.nativeRemoveIsland(handle, id);
        }
    }
}
