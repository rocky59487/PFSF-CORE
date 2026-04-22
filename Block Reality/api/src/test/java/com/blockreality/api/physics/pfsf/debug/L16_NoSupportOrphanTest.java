package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.junit.jupiter.api.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L16: NO_SUPPORT (type=3) orphan voxel trigger.
 *
 * <p>The RBGS solver B4-fix (rbgs_smooth.comp.glsl line ~180):
 * when {@code sumSigma == 0} (all neighbor conductivities are zero), the shader
 * sets {@code phi_gs = 1e7}. Since {@code phi_orphan = 1e6} (from
 * {@code constants.h:PHI_ORPHAN_THRESHOLD}), the failure_scan shader fires
 * {@code fail_flags[i] = 3} (NO_SUPPORT) for any isolated solid voxel.</p>
 *
 * <p>This test creates a 3×3×3 island with one fully isolated center voxel
 * (zero conductivity to all 26 neighbors) and verifies that after one tick
 * the readback phi exceeds 1e6 and at least one NO_SUPPORT event is reported.</p>
 */
@DisplayName("L16: NO_SUPPORT(type=3) — Orphan Voxel phi=1e7 > phi_orphan=1e6")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class L16_NoSupportOrphanTest {

    private static final int LS = 3, NS = LS * LS * LS; // 27
    // (x=1, y=1, z=1) — center voxel
    private static final int ORPHAN_IDX = 1 + LS * (1 + LS * 1);
    private static final int ISLAND_ID = 160;
    private static final float PHI_ORPHAN = 1e6f;

    long handle = 0L;
    ByteBuffer phi, source, cond, type, rcomp, rtens, maxPhi;
    ByteBuffer matId, anchor, fluid, curing;

    @BeforeAll
    void setup() {
        assumeTrue(NativePFSFBridge.isAvailable(), "library not loaded");
        handle = NativePFSFBridge.nativeCreate(NS, 10, PhysicsDebugFixtures.ENGINE_VRAM_BYTES, true, true);
        assumeTrue(handle != 0L, "no Vulkan device");
        assumeTrue(NativePFSFBridge.nativeInit(handle) == NativePFSFBridge.PFSFResult.OK, "pfsf_init failed");

        phi    = PhysicsDebugFixtures.allocAligned(NS * 4);
        source = PhysicsDebugFixtures.allocAligned(NS * 4);
        cond   = PhysicsDebugFixtures.allocAligned(6 * NS * 4); // all zero — fully isolated center
        type   = PhysicsDebugFixtures.allocAligned(NS);
        rcomp  = PhysicsDebugFixtures.allocAligned(NS * 4);
        rtens  = PhysicsDebugFixtures.allocAligned(NS * 4);
        maxPhi = PhysicsDebugFixtures.allocAligned(NS * 4);
        matId  = PhysicsDebugFixtures.allocAligned(NS * 4);
        anchor = PhysicsDebugFixtures.allocAligned(NS * 8);
        fluid  = PhysicsDebugFixtures.allocAligned(NS * 4);
        curing = PhysicsDebugFixtures.allocAligned(NS * 4);

        for (int i = 0; i < NS; i++) {
            int y = (i / LS) % LS;
            curing.putFloat(i * 4, 1.0f);

            if (y == 0) {
                type.put(i, (byte) 2); // ANCHOR
                anchor.putLong(i * 8, 0xFFFFFFFFFFFFFFFFL);
            } else if (i == ORPHAN_IDX) {
                type.put(i, (byte) 1);    // SOLID isolated
                source.putFloat(i * 4, 1.0f);
                // conductivity stays 0 — B4-fix will set phi=1e7
                rcomp.putFloat(i * 4, 1.0f);
                rtens.putFloat(i * 4, 1.0f);
                maxPhi.putFloat(i * 4, 0.001f); // far below 1e7
            } else {
                type.put(i, (byte) 1); // SOLID normal — high thresholds so they don't fail
                source.putFloat(i * 4, 0.1f);
                for (int d = 0; d < 6; d++) cond.putFloat((d * NS + i) * 4, 1.0f);
                rcomp.putFloat(i * 4, 1e9f);
                rtens.putFloat(i * 4, 1e9f);
                maxPhi.putFloat(i * 4, 1e9f);
            }
        }

        assumeTrue(NativePFSFBridge.nativeAddIsland(handle, ISLAND_ID, 400, 0, 0, LS, LS, LS)
                == NativePFSFBridge.PFSFResult.OK, "addIsland 3×3×3 failed");
        assumeTrue(NativePFSFBridge.nativeRegisterIslandBuffers(handle, ISLAND_ID,
                phi, source, cond, type, rcomp, rtens, maxPhi)
                == NativePFSFBridge.PFSFResult.OK, "registerBuffers failed");
        NativePFSFBridge.nativeRegisterIslandLookups(handle, ISLAND_ID, matId, anchor, fluid, curing);
        NativePFSFBridge.nativeRegisterStressReadback(handle, ISLAND_ID, phi);

        ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        fb.putInt(0, 0);
        int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, 1L, fb);
        assumeTrue(rc == NativePFSFBridge.PFSFResult.OK, "tick failed rc=" + rc);
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
            NativePFSFBridge.nativeRemoveIsland(handle, ISLAND_ID);
            try { NativePFSFBridge.nativeShutdown(handle); } catch (Throwable ignored) {}
            try { NativePFSFBridge.nativeDestroy(handle);  } catch (Throwable ignored) {}
            handle = 0L;
        }
    }

    @Test @Order(1)
    @DisplayName("L16-01: 孤立體素 phi > phi_orphan (1e6) — RBGS B4-fix 正確寫入")
    void orphanVoxelExceedsPhiOrphanThreshold() {
        phi.rewind();
        float orphanPhi = 0f;
        for (int i = 0; i < NS; i++) {
            float v = phi.getFloat();
            if (i == ORPHAN_IDX) orphanPhi = v;
        }
        System.out.printf("[L16] orphan voxel phi=%.4e%n", orphanPhi);
        assertTrue(orphanPhi > PHI_ORPHAN,
                "Orphan phi=" + orphanPhi + " should be > phi_orphan=" + PHI_ORPHAN +
                "\n  Check RBGS B4-fix: 'else { phi_gs = 1e7; }' in rbgs_smooth.comp.glsl");
    }

    @Test @Order(2)
    @DisplayName("L16-02: failure type=3 (NO_SUPPORT) 事件存在")
    void noSupportFailureEventFired() {
        // Re-tick to get fresh failure events
        ByteBuffer fb = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        fb.putInt(0, 0);
        NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, 2L, fb);

        int count = fb.getInt(0);
        assertTrue(count >= 1, "No failure events — expected at least 1 NO_SUPPORT event");

        boolean foundNoSupport = false;
        for (int i = 0; i < Math.min(count, 1024); i++) {
            if (fb.getInt(16 + i * 16) == 3) { foundNoSupport = true; break; }
        }
        assertTrue(foundNoSupport,
                "failure type=3 (NO_SUPPORT) not found in " + count + " events. " +
                "Check failure_scan.comp.glsl: 'fail_flags[i] = (p > phi_orphan) ? 3u : 1u'");
    }
}
