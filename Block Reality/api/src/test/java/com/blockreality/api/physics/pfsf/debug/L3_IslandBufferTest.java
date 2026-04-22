package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L3: Island registration and buffer management.
 *
 * <p>Validates that the C++ side accepts island dimensions, receives the
 * DirectByteBuffer registrations without returning {@code ERROR_INVALID_ARG},
 * and that the sparse upload buffer is correctly aliased. If L2 passes but
 * this layer fails, the issue is in buffer sizing, alignment, or ABI layout
 * (e.g., wrong conductivity stride or buffer capacity).</p>
 */
@DisplayName("L3: Island & Buffer Registration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class L3_IslandBufferTest {

    private static final int LX = 4, LY = 4, LZ = 4;
    private static final int ISLAND_ID = PhysicsDebugFixtures.ISLAND_ID;

    long handle = 0L;
    PhysicsDebugFixtures.IslandBuffers bufs;

    @BeforeAll
    void setup() {
        assumeTrue(NativePFSFBridge.isAvailable(), "library not loaded");

        handle = NativePFSFBridge.nativeCreate(LX * LY * LZ, 1,
                PhysicsDebugFixtures.ENGINE_VRAM_BYTES, true, true);
        assumeTrue(handle != 0L, "pfsf_create returned 0 — no Vulkan device");

        int rc = NativePFSFBridge.nativeInit(handle);
        assumeTrue(rc == NativePFSFBridge.PFSFResult.OK,
                "pfsf_init failed: " + NativePFSFBridge.PFSFResult.describe(rc));

        bufs = PhysicsDebugFixtures.buildIslandBuffers(LX, LY, LZ, false);
    }

    @AfterAll
    void teardown() {
        if (bufs != null)   { bufs.free(); bufs = null; }
        if (handle != 0L)   {
            try { NativePFSFBridge.nativeShutdown(handle); } catch (Throwable ignored) {}
            try { NativePFSFBridge.nativeDestroy(handle);  } catch (Throwable ignored) {}
            handle = 0L;
        }
    }

    @Test
    @Order(1)
    @DisplayName("L3-01: nativeAddIsland 回傳 OK")
    void addIslandSucceeds() {
        int rc = NativePFSFBridge.nativeAddIsland(handle, ISLAND_ID, 0, 0, 0, LX, LY, LZ);
        System.out.println("[L3] nativeAddIsland rc=" + NativePFSFBridge.PFSFResult.describe(rc));
        assertEquals(NativePFSFBridge.PFSFResult.OK, rc,
                "nativeAddIsland failed: " + NativePFSFBridge.PFSFResult.describe(rc));
    }

    @Test
    @Order(2)
    @DisplayName("L3-02: nativeRegisterIslandBuffers 回傳 OK（7 個主 buffer）")
    void registerIslandBuffersSucceeds() {
        int rc = NativePFSFBridge.nativeRegisterIslandBuffers(handle, ISLAND_ID,
                bufs.phi(), bufs.source(), bufs.conductivity(),
                bufs.voxelType(), bufs.rcomp(), bufs.rtens(), bufs.maxPhi());
        System.out.println("[L3] nativeRegisterIslandBuffers rc="
                + NativePFSFBridge.PFSFResult.describe(rc));
        assertEquals(NativePFSFBridge.PFSFResult.OK, rc,
                "registerIslandBuffers failed: " + NativePFSFBridge.PFSFResult.describe(rc)
                + "\n  If ERROR_INVALID_ARG: check buffer alignment (must be 256-byte) and sizes.");
    }

    @Test
    @Order(3)
    @DisplayName("L3-03: nativeRegisterIslandLookups 回傳 OK（4 個 lookup buffer）")
    void registerIslandLookupsSucceeds() {
        int rc = NativePFSFBridge.nativeRegisterIslandLookups(handle, ISLAND_ID,
                bufs.materialId(), bufs.anchorBitmap(), bufs.fluidPressure(), bufs.curing());
        System.out.println("[L3] nativeRegisterIslandLookups rc="
                + NativePFSFBridge.PFSFResult.describe(rc));
        assertEquals(NativePFSFBridge.PFSFResult.OK, rc,
                "registerIslandLookups failed: " + NativePFSFBridge.PFSFResult.describe(rc));
    }

    @Test
    @Order(4)
    @DisplayName("L3-04: nativeRegisterStressReadback 回傳 OK")
    void registerStressReadbackSucceeds() {
        int rc = NativePFSFBridge.nativeRegisterStressReadback(handle, ISLAND_ID, bufs.phi());
        System.out.println("[L3] nativeRegisterStressReadback rc="
                + NativePFSFBridge.PFSFResult.describe(rc));
        assertEquals(NativePFSFBridge.PFSFResult.OK, rc,
                "registerStressReadback failed: " + NativePFSFBridge.PFSFResult.describe(rc));
    }

    @Test
    @Order(5)
    @DisplayName("L3-05: getStats islandCount = 1 後新增後")
    void statsShowOneIsland() {
        long[] stats = NativePFSFBridge.nativeGetStats(handle);
        assertNotNull(stats);
        System.out.println("[L3] islandCount=" + stats[0]);
        assertEquals(1L, stats[0], "Expected exactly 1 island, got: " + stats[0]);
    }

    @Test
    @Order(6)
    @DisplayName("L3-06: nativeGetSparseUploadBuffer isDirect && capacity ≥ 24576")
    void sparseUploadBufferAliasing() {
        java.nio.ByteBuffer upload = NativePFSFBridge.nativeGetSparseUploadBuffer(handle, ISLAND_ID);
        assertNotNull(upload, "Sparse upload buffer must alias a live VMA allocation");
        assertTrue(upload.isDirect(), "Sparse upload buffer must be direct");
        assertTrue(upload.capacity() >= 512 * 48,
                "Capacity below 24576: " + upload.capacity());
        System.out.println("[L3] sparseUploadBuffer capacity=" + upload.capacity());
    }

    @Test
    @Order(7)
    @DisplayName("L3-07: nativeNotifySparseUpdates(0 records) = OK")
    void sparseUpdateZeroRecordsIsNoop() {
        int rc = NativePFSFBridge.nativeNotifySparseUpdates(handle, ISLAND_ID, 0);
        assertEquals(NativePFSFBridge.PFSFResult.OK, rc,
                "Zero-record sparse notify should be no-op: "
                        + NativePFSFBridge.PFSFResult.describe(rc));
    }

    @Test
    @Order(8)
    @DisplayName("L3-08: nativeRemoveIsland 後 islandCount = 0")
    void removeIslandCleansUp() {
        assertDoesNotThrow(() -> NativePFSFBridge.nativeRemoveIsland(handle, ISLAND_ID));
        long[] stats = NativePFSFBridge.nativeGetStats(handle);
        assertNotNull(stats);
        System.out.println("[L3] islandCount after remove=" + stats[0]);
        assertEquals(0L, stats[0], "Expected 0 islands after remove, got: " + stats[0]);
    }
}
