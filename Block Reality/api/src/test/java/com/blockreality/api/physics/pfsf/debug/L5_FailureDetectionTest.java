package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.junit.jupiter.api.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L5: Failure detection on a known-bad structure.
 *
 * <p>Uses a 4×4×4 island where the center floating voxel has extreme-low
 * {@code maxPhi} (0.001) and {@code rcomp} (0.001), so after a few ticks the
 * failure scan shader must flag it. If no failure events are produced after
 * 32 ticks, the problem is one of:</p>
 * <ul>
 *   <li>phi never reaching the voxel (DataBuilder source/conductivity issue)</li>
 *   <li>rcomp/maxPhi not normalized (CLAUDE.md trap #9): 0.001 raw MPa becomes
 *       0.001/sigmaMax ≈ 0.00005 after normalize — so failure would fire immediately.
 *       But if normalize is skipped, rcomp = 0.001 is already in normalized units
 *       and failure may never occur if phi < 0.001.</li>
 *   <li>failure_scan shader disabled or not wired</li>
 * </ul>
 */
@DisplayName("L5: Failure Detection")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class L5_FailureDetectionTest {

    private static final int LX = 4, LY = 4, LZ = 4;
    private static final int N  = LX * LY * LZ;
    private static final int ISLAND_ID = PhysicsDebugFixtures.ISLAND_ID;
    private static final int MAX_TICKS = 32;

    long handle = 0L;
    PhysicsDebugFixtures.IslandBuffers bufs;

    @BeforeAll
    void setup() {
        assumeTrue(NativePFSFBridge.isAvailable(), "library not loaded");

        handle = NativePFSFBridge.nativeCreate(N, 1, PhysicsDebugFixtures.ENGINE_VRAM_BYTES, true, true);
        assumeTrue(handle != 0L, "no Vulkan device");
        assumeTrue(NativePFSFBridge.nativeInit(handle) == NativePFSFBridge.PFSFResult.OK,
                "pfsf_init failed");

        // triggerFailure=true: center floating voxel gets maxPhi=0.001 & rcomp=0.001
        bufs = PhysicsDebugFixtures.buildIslandBuffers(LX, LY, LZ, /*triggerFailure=*/ true);

        assumeTrue(NativePFSFBridge.nativeAddIsland(handle, ISLAND_ID, 0, 0, 0, LX, LY, LZ)
                == NativePFSFBridge.PFSFResult.OK, "nativeAddIsland failed");
        assumeTrue(NativePFSFBridge.nativeRegisterIslandBuffers(handle, ISLAND_ID,
                bufs.phi(), bufs.source(), bufs.conductivity(),
                bufs.voxelType(), bufs.rcomp(), bufs.rtens(), bufs.maxPhi())
                == NativePFSFBridge.PFSFResult.OK, "registerIslandBuffers failed");
        assumeTrue(NativePFSFBridge.nativeRegisterIslandLookups(handle, ISLAND_ID,
                bufs.materialId(), bufs.anchorBitmap(), bufs.fluidPressure(), bufs.curing())
                == NativePFSFBridge.PFSFResult.OK, "registerIslandLookups failed");
        assumeTrue(NativePFSFBridge.nativeRegisterStressReadback(handle, ISLAND_ID, bufs.phi())
                == NativePFSFBridge.PFSFResult.OK, "registerStressReadback failed");
    }

    @AfterAll
    void teardown() {
        if (bufs   != null) { bufs.free(); bufs = null; }
        if (handle != 0L)   {
            try { NativePFSFBridge.nativeShutdown(handle); } catch (Throwable ignored) {}
            try { NativePFSFBridge.nativeDestroy(handle);  } catch (Throwable ignored) {}
            handle = 0L;
        }
    }

    @Test
    @DisplayName("L5-01: 最多 32 次 tick 後必須產生至少一個 failure event")
    void failureEventFiredWithinMaxTicks() {
        // failBuf layout: int count at [0], then count × {x, y, z, type} × int32
        ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + 1024 * 16)
                .order(ByteOrder.LITTLE_ENDIAN);

        int totalEvents = 0;
        for (int epoch = 1; epoch <= MAX_TICKS; epoch++) {
            failBuf.putInt(0, 0); // reset count header each tick
            int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, epoch, failBuf);
            assertEquals(NativePFSFBridge.PFSFResult.OK, rc,
                    "Tick " + epoch + " failed: " + NativePFSFBridge.PFSFResult.describe(rc));

            int count = failBuf.getInt(0);
            totalEvents += count;
            if (count > 0) {
                System.out.println("[L5] Failure events at epoch=" + epoch + ": count=" + count);
                for (int i = 0; i < Math.min(count, 4); i++) {
                    int x = failBuf.getInt(4 + i * 16);
                    int y = failBuf.getInt(8 + i * 16);
                    int z = failBuf.getInt(12 + i * 16);
                    int t = failBuf.getInt(16 + i * 16);
                    System.out.printf("    [%d] pos=(%d,%d,%d) type=%d%n", i, x, y, z, t);
                }
                break; // found failure, stop early
            }
        }

        assertTrue(totalEvents > 0,
                "[L5-FAIL] No failure events after " + MAX_TICKS + " ticks.\n" +
                "  Structure has voxel with maxPhi=0.001 & rcomp=0.001 — should fail immediately.\n\n" +
                "  Possible causes:\n" +
                "  1. failure_scan shader is not running or not wired to nativeTickDbb\n" +
                "  2. phi never propagates to the center voxel (source=0 or anchor blocks phi escape)\n" +
                "  3. rcomp WAS normalized by sigmaMax inside nativeNormalizeSoA6,\n" +
                "     so the value already got divided: 0.001/sigmaMax << phi threshold.\n" +
                "     Check: L1-05 should show rcomp[0] ≈ 0.001/sigmaMax after normalize.\n" +
                "  4. failureBuffer count byte-order or layout mismatch");
    }

    @Test
    @DisplayName("L5-02: failure event 位置在預期的 island 範圍內")
    void failureEventPositionInBounds() {
        ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + 1024 * 16)
                .order(ByteOrder.LITTLE_ENDIAN);

        int foundX = -1, foundY = -1, foundZ = -1;
        outer:
        for (int epoch = 100; epoch <= 100 + MAX_TICKS; epoch++) {
            failBuf.putInt(0, 0);
            int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, epoch, failBuf);
            assumeTrue(rc == NativePFSFBridge.PFSFResult.OK, "tick failed");
            int count = failBuf.getInt(0);
            if (count > 0) {
                foundX = failBuf.getInt(4);
                foundY = failBuf.getInt(8);
                foundZ = failBuf.getInt(12);
                break outer;
            }
        }

        assumeTrue(foundX >= 0, "no failure event found yet — run L5-01 first");
        System.out.printf("[L5] failure pos=(%d,%d,%d)%n", foundX, foundY, foundZ);
        // Position must be within island world bounds (origin 0,0,0 + lx,ly,lz)
        assertTrue(foundX >= 0 && foundX < LX, "failure X out of island bounds: " + foundX);
        assertTrue(foundY >= 0 && foundY < LY, "failure Y out of island bounds: " + foundY);
        assertTrue(foundZ >= 0 && foundZ < LZ, "failure Z out of island bounds: " + foundZ);
    }
}
