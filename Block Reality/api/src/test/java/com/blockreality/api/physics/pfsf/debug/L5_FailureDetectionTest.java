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
 * <p>Builds a 4×4×4 island where the centre voxel is topologically isolated
 * (six face-conductivities forced to zero), then runs ONE tick. The
 * {@code failure_scan} shader's topological NO_SUPPORT branch
 * ({@code sumSigma6 == 0}) should fire on that voxel.</p>
 *
 * <p>Why a single tick: the native engine marks an island clean after each
 * dispatch and skips it on subsequent ticks until something re-marks it
 * dirty. Looping epoch=1..32 used to be harmless when the engine
 * re-dispatched on every tick, but with the markClean optimisation only
 * the first tick produces work — so we read its result directly.</p>
 *
 * <p>If no failure events are produced after the single tick, the problem
 * is one of:</p>
 * <ul>
 *   <li>{@code failure_scan} shader is not running or not wired to
 *       {@code nativeTickDbb}</li>
 *   <li>The shader's NO_SUPPORT branch (sumSigma6==0) was reverted</li>
 *   <li>The fixture's centre-voxel sigma override is failing to land in
 *       the SoA buffer (check {@link PhysicsDebugFixtures})</li>
 * </ul>
 */
@DisplayName("L5: Failure Detection")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class L5_FailureDetectionTest {

    private static final int LX = 4, LY = 4, LZ = 4;
    private static final int N  = LX * LY * LZ;
    private static final int ISLAND_ID = PhysicsDebugFixtures.ISLAND_ID;

    /** Centre voxel index in the SoA flat layout: (LX/2, 1, LZ/2). */
    private static final int ORPHAN_IDX = (LX / 2) + LX * (1 + LY * (LZ / 2));

    long handle = 0L;
    PhysicsDebugFixtures.IslandBuffers bufs;
    /** Failure buffer captured after the only tick the engine actually runs. */
    ByteBuffer setupFailBuf;

    @BeforeAll
    void setup() {
        assumeTrue(NativePFSFBridge.isAvailable(), "library not loaded");

        handle = NativePFSFBridge.nativeCreate(N, 1, PhysicsDebugFixtures.ENGINE_VRAM_BYTES, true, true);
        assumeTrue(handle != 0L, "no Vulkan device");
        assumeTrue(NativePFSFBridge.nativeInit(handle) == NativePFSFBridge.PFSFResult.OK,
                "pfsf_init failed");

        bufs = PhysicsDebugFixtures.buildIslandBuffers(LX, LY, LZ, /*triggerFailure=*/ true);

        // Force the centre voxel into a topologically-isolated state by
        // zeroing its six face conductivities. Direct buffer write keeps
        // the rest of the fixture (anchor row, source field, neighbour
        // conductivities) intact, so the orphan emerges via
        // failure_scan's sumSigma6==0 branch and not as a side-effect.
        ByteBuffer cond = bufs.conductivity();
        for (int d = 0; d < 6; d++) {
            cond.putFloat((d * N + ORPHAN_IDX) * 4, 0.0f);
        }

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

        // Run the one tick the engine will accept. After this the island is
        // clean and re-ticking would early-return without re-dispatching.
        setupFailBuf = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        setupFailBuf.putInt(0, 0);
        int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, 1L, setupFailBuf);
        assumeTrue(rc == NativePFSFBridge.PFSFResult.OK,
                "tick failed: " + NativePFSFBridge.PFSFResult.describe(rc));
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
    @DisplayName("L5-01: 第一個 tick 必須產生至少一個 failure event")
    void failureEventFiredOnFirstTick() {
        int count = setupFailBuf.getInt(0);
        if (count > 0) {
            System.out.println("[L5] Failure events: count=" + count);
            for (int i = 0; i < Math.min(count, 4); i++) {
                int x = setupFailBuf.getInt(4 + i * 16);
                int y = setupFailBuf.getInt(8 + i * 16);
                int z = setupFailBuf.getInt(12 + i * 16);
                int t = setupFailBuf.getInt(16 + i * 16);
                System.out.printf("    [%d] pos=(%d,%d,%d) type=%d%n", i, x, y, z, t);
            }
        }

        assertTrue(count > 0,
                "[L5-FAIL] No failure events from the single dispatched tick.\n"
                + "  Centre voxel was forced into topological isolation (six face σ = 0)\n"
                + "  and failure_scan should flag it as NO_SUPPORT (type=3) immediately.\n\n"
                + "  Possible causes:\n"
                + "  1. failure_scan shader is not running or not wired to nativeTickDbb\n"
                + "  2. Topological NO_SUPPORT branch (sumSigma6==0) reverted in the shader\n"
                + "  3. Fixture's centre-voxel σ override missed the SoA layout\n"
                + "  4. failureBuffer count byte-order or layout mismatch");
    }

    @Test
    @DisplayName("L5-02: failure event 位置在預期的 island 範圍內")
    void failureEventPositionInBounds() {
        int count = setupFailBuf.getInt(0);
        assumeTrue(count > 0, "no failure event found — run L5-01 first");

        int foundX = setupFailBuf.getInt(4);
        int foundY = setupFailBuf.getInt(8);
        int foundZ = setupFailBuf.getInt(12);

        System.out.printf("[L5] failure pos=(%d,%d,%d)%n", foundX, foundY, foundZ);
        assertTrue(foundX >= 0 && foundX < LX, "failure X out of island bounds: " + foundX);
        assertTrue(foundY >= 0 && foundY < LY, "failure Y out of island bounds: " + foundY);
        assertTrue(foundZ >= 0 && foundZ < LZ, "failure Z out of island bounds: " + foundZ);
    }
}
