package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L2: Vulkan engine lifecycle — requires a GPU/lavapipe Vulkan device.
 *
 * <p>Tests {@code nativeCreate → nativeInit → getStats → nativeShutdown → nativeDestroy}.
 * If L1 passes but this layer FAILs or SKIPs with "no Vulkan device", the GPU stack is
 * absent. In GPU-less CI, these tests are expected to SKIP.</p>
 */
@DisplayName("L2: Vulkan Engine Init")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class L2_VulkanInitTest {

    long handle = 0L;
    boolean initOk = false;

    @BeforeAll
    void requireLibrary() {
        assumeTrue(NativePFSFBridge.isAvailable(),
                "libblockreality_pfsf not loaded — L0 must pass first");
    }

    @AfterAll
    void cleanup() {
        if (handle != 0L) {
            try { NativePFSFBridge.nativeShutdown(handle); } catch (Throwable ignored) {}
            try { NativePFSFBridge.nativeDestroy(handle);  } catch (Throwable ignored) {}
            handle = 0L;
        }
    }

    @Test
    @Order(1)
    @DisplayName("L2-01: nativeCreate 回傳非零 handle")
    void createReturnsNonZeroHandle() {
        handle = NativePFSFBridge.nativeCreate(
                /*maxIslandSize*/ 512,
                /*tickBudgetMs */ 1,
                /*vramBytes    */ PhysicsDebugFixtures.ENGINE_VRAM_BYTES,
                /*phaseField   */ true,
                /*multigrid    */ true);
        System.out.println("[L2] nativeCreate handle=0x" + Long.toHexString(handle));
        assumeTrue(handle != 0L,
                "pfsf_create returned 0 — no suitable Vulkan device found (expected skip in GPU-less CI)");
        assertNotEquals(0L, handle);
    }

    @Test
    @Order(2)
    @DisplayName("L2-02: nativeInit 回傳 PFSFResult.OK")
    void initReturnsOk() {
        assumeTrue(handle != 0L, "nativeCreate did not produce a handle (L2-01 skipped)");
        int rc = NativePFSFBridge.nativeInit(handle);
        System.out.println("[L2] nativeInit rc=" + NativePFSFBridge.PFSFResult.describe(rc));
        assumeTrue(rc == NativePFSFBridge.PFSFResult.OK,
                "pfsf_init failed: " + NativePFSFBridge.PFSFResult.describe(rc)
                + " — Vulkan device may lack a compute queue");
        initOk = true;
    }

    @Test
    @Order(3)
    @DisplayName("L2-03: nativeIsAvailable 初始化後為 true")
    void isAvailableAfterInit() {
        assumeTrue(initOk, "pfsf_init did not succeed (L2-02 skipped)");
        assertTrue(NativePFSFBridge.nativeIsAvailable(handle),
                "nativeIsAvailable must return true after successful init");
    }

    @Test
    @Order(4)
    @DisplayName("L2-04: nativeGetStats 回傳 long[5]，vramBudget > 0")
    void getStatsReturnsValidArray() {
        assumeTrue(initOk, "pfsf_init did not succeed");
        long[] stats = NativePFSFBridge.nativeGetStats(handle);
        assertNotNull(stats, "nativeGetStats must not return null on a live handle");
        assertEquals(5, stats.length,
                "nativeGetStats contract: {islandCount, totalVoxels, vramUsed, vramBudget, lastTickMicros}");
        System.out.printf("[L2] stats: islands=%d voxels=%d vramUsed=%d vramBudget=%d lastTickUs=%d%n",
                stats[0], stats[1], stats[2], stats[3], stats[4]);
        assertTrue(stats[3] > 0, "vramBudget (stats[3]) must be > 0, got: " + stats[3]);
    }

    @Test
    @Order(5)
    @DisplayName("L2-05: nativeShutdown + nativeDestroy 不 crash")
    void shutdownDestroyClean() {
        assumeTrue(handle != 0L, "No handle to shut down");
        assertDoesNotThrow(() -> {
            NativePFSFBridge.nativeShutdown(handle);
            NativePFSFBridge.nativeDestroy(handle);
            handle = 0L;
            initOk = false;
        });
    }
}
