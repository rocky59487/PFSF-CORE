package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.junit.jupiter.api.*;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L4: One solver tick on a minimal 4×4×4 island.
 *
 * <p>Verifies that after {@code nativeTickDbb} the phi field contains non-zero
 * values (the solver ran) and that 5 consecutive ticks produce a stable,
 * non-NaN phi field (the solver converges rather than diverges).</p>
 *
 * <p>If phi is all-zero after tick: the source/conductivity data was not
 * reaching the GPU — check buffer fills in {@link PhysicsDebugFixtures}.</p>
 *
 * <p>If phi contains NaN/Inf: the solver is diverging — likely a normalization
 * bug (CLAUDE.md trap #9) or a 26-connectivity stencil inconsistency (trap #10).</p>
 */
@DisplayName("L4: Single Tick Correctness")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class L4_SingleTickTest {

    private static final int LX = 4, LY = 4, LZ = 4;
    private static final int N  = LX * LY * LZ;
    private static final int ISLAND_ID = PhysicsDebugFixtures.ISLAND_ID;

    long handle = 0L;
    PhysicsDebugFixtures.IslandBuffers bufs;

    @BeforeAll
    void setup() {
        assumeTrue(NativePFSFBridge.isAvailable(), "library not loaded");

        handle = NativePFSFBridge.nativeCreate(N, 1, PhysicsDebugFixtures.ENGINE_VRAM_BYTES, true, true);
        assumeTrue(handle != 0L, "no Vulkan device");
        assumeTrue(NativePFSFBridge.nativeInit(handle) == NativePFSFBridge.PFSFResult.OK,
                "pfsf_init failed");

        bufs = PhysicsDebugFixtures.buildIslandBuffers(LX, LY, LZ, false);

        // Register island + all buffers (full registration as in NativePFSFRuntime)
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

    private ByteBuffer makeFailBuf() {
        return ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
    }

    @Test
    @Order(1)
    @DisplayName("L4-01: 首次 tick 回傳 PFSFResult.OK，耗時 < 5s")
    void firstTickReturnsOk() {
        ByteBuffer failBuf = makeFailBuf();
        failBuf.putInt(0, 0);

        long t0 = System.currentTimeMillis();
        int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, 1L, failBuf);
        long elapsed = System.currentTimeMillis() - t0;
        System.out.println("[L4] firstTick rc=" + NativePFSFBridge.PFSFResult.describe(rc)
                + "  elapsed=" + elapsed + "ms");

        assertEquals(NativePFSFBridge.PFSFResult.OK, rc,
                "nativeTickDbb failed: " + NativePFSFBridge.PFSFResult.describe(rc)
                + "\n  If ERROR_VULKAN: compute pipeline setup failed (check shaders/SPIR-V)");
        assertTrue(elapsed < 5000,
                "Tick took >5s (" + elapsed + "ms) — possible GPU hang");
    }

    @Test
    @Order(2)
    @DisplayName("L4-02: tick 後 phi 場至少有一個非零值（求解器有計算）")
    void phiFieldNonZeroAfterTick() {
        // Tick again to ensure we read from a fresh run
        ByteBuffer failBuf = makeFailBuf();
        failBuf.putInt(0, 0);
        int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, 2L, failBuf);
        assumeTrue(rc == NativePFSFBridge.PFSFResult.OK, "tick failed in L4-02");

        // Read phi from the registered stress readback buffer (same as bufs.phi)
        float maxPhi = 0f;
        bufs.phi().position(0);
        for (int i = 0; i < N; i++) {
            float v = bufs.phi().getFloat(i * 4);
            if (v > maxPhi) maxPhi = v;
        }
        System.out.println("[L4] maxPhi after tick=" + maxPhi);
        assertTrue(maxPhi > 0f,
                "[L4-FAIL] phi field is all-zero after tick.\n" +
                "  Possible causes:\n" +
                "  1. Source terms are all zero in the buffers (check PhysicsDebugFixtures.buildIslandBuffers)\n" +
                "  2. Conductivity is all zero\n" +
                "  3. The C++ solver ran but wrote phi back to a different buffer");
    }

    @Test
    @Order(3)
    @DisplayName("L4-03: 5 次連續 tick phi 不含 NaN/Inf（求解器穩定）")
    void fiveTicksStableNonNaN() {
        ByteBuffer failBuf = makeFailBuf();
        for (int epoch = 10; epoch < 15; epoch++) {
            failBuf.putInt(0, 0);
            int rc = NativePFSFBridge.nativeTickDbb(handle, new int[]{ISLAND_ID}, epoch, failBuf);
            assertEquals(NativePFSFBridge.PFSFResult.OK, rc,
                    "Tick " + epoch + " failed: " + NativePFSFBridge.PFSFResult.describe(rc));
        }

        // Check for NaN/Inf in phi
        float maxPhi = 0f;
        boolean hasNaN = false, hasInf = false;
        bufs.phi().position(0);
        for (int i = 0; i < N; i++) {
            float v = bufs.phi().getFloat(i * 4);
            if (Float.isNaN(v))      hasNaN = true;
            else if (Float.isInfinite(v)) hasInf = true;
            else if (v > maxPhi)          maxPhi = v;
        }
        System.out.println("[L4] After 5 ticks: maxPhi=" + maxPhi
                + "  hasNaN=" + hasNaN + "  hasInf=" + hasInf);

        assertFalse(hasNaN,
                "[L4-FAIL] phi contains NaN after 5 ticks — solver DIVERGING.\n" +
                "  Likely causes:\n" +
                "  1. CLAUDE.md trap #9: rcomp/rtens not normalized by sigmaMax\n" +
                "  2. CLAUDE.md trap #10: RBGS/PCG stencil mismatch (26-connectivity inconsistency)\n" +
                "  Run L1-05 to check normalization.");
        assertFalse(hasInf,
                "[L4-FAIL] phi contains Infinity after 5 ticks — solver DIVERGING.");
    }
}
