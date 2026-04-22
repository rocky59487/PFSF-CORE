package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.junit.jupiter.api.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L15: Failure buffer saturation — a 12×12×12 structure with rcomp=maxPhi=1e-6
 * should produce far more than 1024 failure events, saturating the Java-side DBB.
 *
 * <p>Verifies that:</p>
 * <ol>
 *   <li>count is exactly capped at the DBB capacity (no overflow).</li>
 *   <li>At least some failures were detected (solver ran).</li>
 * </ol>
 *
 * <p>If count is 0: the solver ran but threshold is not low enough relative to phi.
 * If count overflows the buffer: the C++ {@code readbackFailures} cap check is broken.</p>
 */
@DisplayName("L15: Failure Buffer Saturation")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class L15_FailureBufferSaturationTest {

    private static final int LS = 12, NS = LS * LS * LS; // 1728
    private static final int CAP = 1024;
    private static final int ISLAND_ID = 150;

    long handle = 0L;
    ByteBuffer phi, source, cond, type, rcomp, rtens, maxPhi;
    ByteBuffer matId, anchor, fluid, curing;

    @BeforeAll
    void setup() {
        assumeTrue(NativePFSFBridge.isAvailable(), "library not loaded");
        // max_island_size must be >= 1728
        handle = NativePFSFBridge.nativeCreate(2048, 10, PhysicsDebugFixtures.ENGINE_VRAM_BYTES, true, true);
        assumeTrue(handle != 0L, "no Vulkan device");
        assumeTrue(NativePFSFBridge.nativeInit(handle) == NativePFSFBridge.PFSFResult.OK, "pfsf_init failed");

        phi    = PhysicsDebugFixtures.allocAligned(NS * 4);
        source = PhysicsDebugFixtures.allocAligned(NS * 4);
        cond   = PhysicsDebugFixtures.allocAligned(6 * NS * 4);
        type   = PhysicsDebugFixtures.allocAligned(NS);
        rcomp  = PhysicsDebugFixtures.allocAligned(NS * 4);
        rtens  = PhysicsDebugFixtures.allocAligned(NS * 4);
        maxPhi = PhysicsDebugFixtures.allocAligned(NS * 4);
        matId  = PhysicsDebugFixtures.allocAligned(NS * 4);
        anchor = PhysicsDebugFixtures.allocAligned(NS * 8);
        fluid  = PhysicsDebugFixtures.allocAligned(NS * 4);
        curing = PhysicsDebugFixtures.allocAligned(NS * 4);

        for (int x = 0; x < LS; x++) {
            for (int y = 0; y < LS; y++) {
                for (int z = 0; z < LS; z++) {
                    int i = x + LS * (y + LS * z);
                    curing.putFloat(i * 4, 1.0f);
                    if (y == 0) {
                        type.put(i, (byte) 2); // ANCHOR
                        anchor.putLong(i * 8, 0xFFFFFFFFFFFFFFFFL);
                    } else {
                        type.put(i, (byte) 1); // SOLID
                        source.putFloat(i * 4, 1.0f);
                        for (int d = 0; d < 6; d++) cond.putFloat((d * NS + i) * 4, 1.0f);
                        rcomp.putFloat(i * 4, 1e-6f);
                        rtens.putFloat(i * 4, 1e-6f);
                        maxPhi.putFloat(i * 4, 1e-6f);
                    }
                }
            }
        }

        assumeTrue(NativePFSFBridge.nativeAddIsland(handle, ISLAND_ID, 0, 0, 0, LS, LS, LS)
                == NativePFSFBridge.PFSFResult.OK, "addIsland 12×12×12 failed");
        assumeTrue(NativePFSFBridge.nativeRegisterIslandBuffers(handle, ISLAND_ID,
                phi, source, cond, type, rcomp, rtens, maxPhi)
                == NativePFSFBridge.PFSFResult.OK, "registerBuffers failed");
        NativePFSFBridge.nativeRegisterIslandLookups(handle, ISLAND_ID, matId, anchor, fluid, curing);
        NativePFSFBridge.nativeRegisterStressReadback(handle, ISLAND_ID, phi);
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

    @Test
    @DisplayName("L15-01: failure count 不超過 DBB 容量 CAP=" + CAP + "（無緩衝區溢位）")
    void failureCountCappedAtBufferCapacity() {
        ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + CAP * 16).order(ByteOrder.LITTLE_ENDIAN);
        failBuf.putInt(0, 0);

        int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, 1L, failBuf);
        assertEquals(NativePFSFBridge.PFSFResult.OK, rc, "tick failed rc=" + rc);

        int count = failBuf.getInt(0);
        System.out.printf("[L15] count=%d / CAP=%d (%.1f%% 飽和)%n",
                count, CAP, count * 100.0 / CAP);

        assertTrue(count >= 0 && count <= CAP,
                "count=" + count + " exceeds CAP=" + CAP + " — buffer overflow");
        assertTrue(count > 0,
                "No failure events — either solver did not run or thresholds not met");
    }
}
