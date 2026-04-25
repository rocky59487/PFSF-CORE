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
 * <p>{@code rbgs_smooth.comp.glsl} writes {@code phi_gs = 0.0} for a
 * voxel with {@code sumSigma == 0} (the "energy explosion fix" — see
 * the comment block at line ~170 of rbgs_smooth). Because the orphan
 * φ never reaches the legacy {@code phi_orphan = 1e6} threshold,
 * failure_scan's NO_SUPPORT detection now uses a topological test
 * (six face σ all zero) in addition to the legacy φ-divergence path —
 * the two paths together cover both single-voxel isolation and
 * whole-component orphans.</p>
 *
 * <p>This test creates a 3×3×3 island with one fully isolated center voxel
 * (zero conductivity to all 6 face neighbours) and verifies that the
 * topological branch produces a NO_SUPPORT event. The voxel's φ stays at 0
 * by design and is not asserted upon.</p>
 */
@DisplayName("L16: NO_SUPPORT(type=3) — Topologically Isolated Voxel")
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
    /**
     * Failure-buffer captured during {@link #setup()} after the first tick.
     * Re-running {@code nativeTickDbb} in a later @Test would early-return
     * (the engine marks the island clean after dispatch and skips clean
     * islands), so we keep the readback from the only tick the engine
     * actually runs on this fixture and assert against it across tests.
     */
    ByteBuffer setupFailBuf;

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

        setupFailBuf = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        setupFailBuf.putInt(0, 0);
        int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, 1L, setupFailBuf);
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
    @DisplayName("L16-01: 孤立體素 phi 維持 0（rbgs energy-explosion fix 合約）")
    void orphanVoxelPhiStaysZero() {
        // After rbgs_smooth's energy-explosion fix, an isolated voxel's φ
        // is held at 0 instead of being driven to 1e7 — the previous 1e7
        // overshoot leaked through the energy field hField and corrupted
        // its neighbours. Failure detection migrated to a topological
        // test (σ-based) in failure_scan; see L16-02.
        phi.rewind();
        float orphanPhi = 0f;
        for (int i = 0; i < NS; i++) {
            float v = phi.getFloat();
            if (i == ORPHAN_IDX) orphanPhi = v;
        }
        System.out.printf("[L16] orphan voxel phi=%.4e%n", orphanPhi);
        assertEquals(0.0f, orphanPhi, 1e-3f,
                "Orphan voxel φ must stay at 0 per rbgs_smooth.comp.glsl L173 contract; got " + orphanPhi);
    }

    @Test @Order(2)
    @DisplayName("L16-02: failure type=3 (NO_SUPPORT) 由拓撲分支觸發")
    void noSupportFailureEventFired() {
        // Asserts against the failure buffer the engine produced during the
        // single tick that ran in @BeforeAll. The engine marks the island
        // clean after dispatch and skips it on subsequent ticks, so any
        // re-tick here would not re-run failure_scan.
        int count = setupFailBuf.getInt(0);
        assertTrue(count >= 1,
                "No failure events — expected at least 1 NO_SUPPORT event from the "
                        + "topologically-isolated centre voxel");

        boolean foundNoSupport = false;
        for (int i = 0; i < Math.min(count, 1024); i++) {
            // event tuple layout: int x, y, z, type at offsets 4+i*16..16+i*16
            if (setupFailBuf.getInt(16 + i * 16) == 3) { foundNoSupport = true; break; }
        }
        assertTrue(foundNoSupport,
                "failure type=3 (NO_SUPPORT) not found in " + count + " events. "
                        + "Check failure_scan.comp.glsl topological orphan branch (sumSigma6==0)");
    }
}
