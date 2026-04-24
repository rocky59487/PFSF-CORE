package com.blockreality.api.physics.pfsf;

import net.minecraft.core.BlockPos;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PFSFEngineIntegrationTest {

    @Test
    @DisplayName("uninitialized engine reports unavailable")
    void testUninitializedEngine() {
        assertFalse(PFSFEngine.isAvailable(),
                "Engine should not be available without Vulkan context");
    }

    @Test
    @DisplayName("native runtime counts as available")
    void testNativeRuntimeCountsAsAvailable() throws Exception {
        Field instanceField = PFSFEngine.class.getDeclaredField("instance");
        instanceField.setAccessible(true);
        Object previousInstance = instanceField.get(null);

        Field nativeActiveField = NativePFSFRuntime.class.getDeclaredField("active");
        nativeActiveField.setAccessible(true);
        boolean previousNativeActive = nativeActiveField.getBoolean(null);

        try {
            instanceField.set(null, new PFSFEngineInstance());
            nativeActiveField.setBoolean(null, true);

            assertTrue(PFSFEngine.isAvailable(),
                    "Active native runtime must count as GPU availability");
            assertSame(NativePFSFRuntime.asRuntime(), PFSFEngine.getRuntime(),
                    "Native runtime should be selected when it is active");
        } finally {
            nativeActiveField.setBoolean(null, previousNativeActive);
            instanceField.set(null, previousInstance);
        }
    }

    @Test
    @DisplayName("native-only mode allocates host-backed island buffers")
    void testNativeOnlyIslandBufferAllocation() throws Exception {
        Field nativeActiveField = NativePFSFRuntime.class.getDeclaredField("active");
        nativeActiveField.setAccessible(true);
        boolean previousNativeActive = nativeActiveField.getBoolean(null);

        try {
            nativeActiveField.setBoolean(null, true);

            PFSFIslandBuffer buf = new PFSFIslandBuffer(77);
            buf.allocate(4, 4, 4, BlockPos.ZERO);

            assertTrue(buf.isAllocated(), "Native-only mode should still allocate host-side buffers");
            assertEqualsZero(buf.getPhiBuf(), "Host-only native allocation must not depend on Java Vulkan buffers");
            assertNotNull(buf.getPhiBufAsBB(), "Native runtime still needs host DBBs for phi");
            assertNotNull(buf.getSourceBufAsBB(), "Native runtime still needs host DBBs for source");
            assertNotNull(buf.getLookupCuringBB(), "Native runtime still needs lookup DBBs");
            assertFalse(buf.getPhaseField().isAllocated(), "Java phase-field GPU buffers must stay disabled in host-only mode");

            buf.release();
        } finally {
            nativeActiveField.setBoolean(null, previousNativeActive);
        }
    }

    @Test
    @DisplayName("getStats returns disabled when engine is off")
    void testGetStatsDisabled() {
        String stats = PFSFEngine.getStats();
        assertNotNull(stats);
        assertTrue(stats.contains("DISABLED"),
                "Stats should indicate DISABLED when engine is not available: " + stats);
    }

    @Test
    @DisplayName("shutdown is idempotent")
    void testShutdownIdempotent() {
        assertDoesNotThrow(() -> {
            PFSFEngine.shutdown();
            PFSFEngine.shutdown();
        });
    }

    @Test
    @DisplayName("lookup setters accept null")
    void testLookupSettersAcceptNull() {
        assertDoesNotThrow(() -> PFSFEngine.setMaterialLookup(null));
        assertDoesNotThrow(() -> PFSFEngine.setAnchorLookup(null));
        assertDoesNotThrow(() -> PFSFEngine.setFillRatioLookup(null));
    }

    @Test
    @DisplayName("removeBuffer ignores unknown island ids")
    void testRemoveNonExistentBuffer() {
        assertDoesNotThrow(() -> PFSFEngine.removeBuffer(999999));
    }

    @Test
    @DisplayName("onServerTick is safe when no GPU solver is active")
    void testOnServerTickNoGpu() {
        assertDoesNotThrow(() -> PFSFEngine.onServerTick(null, null, 0));
    }

    @Test
    @DisplayName("extracted helper classes remain loadable")
    void testExtractedClassesExist() {
        assertDoesNotThrow(() -> {
            Class.forName("com.blockreality.api.physics.pfsf.PFSFPipelineFactory");
            Class.forName("com.blockreality.api.physics.pfsf.PFSFDataBuilder");
            Class.forName("com.blockreality.api.physics.pfsf.PFSFVCycleRecorder");
            Class.forName("com.blockreality.api.physics.pfsf.PFSFFailureRecorder");
            Class.forName("com.blockreality.api.physics.pfsf.PFSFBufferManager");
            Class.forName("com.blockreality.api.physics.pfsf.PFSFStressExtractor");
        });
    }

    @Test
    @DisplayName("failure type mappings remain intact")
    void testFailureTypeMapping() {
        var failTypes = com.blockreality.api.physics.FailureType.values();
        assertTrue(failTypes.length >= 4,
                "FailureType should have at least 4 values for PFSF modes");

        assertNotNull(com.blockreality.api.physics.FailureType.CANTILEVER_BREAK);
        assertNotNull(com.blockreality.api.physics.FailureType.CRUSHING);
        assertNotNull(com.blockreality.api.physics.FailureType.NO_SUPPORT);
        assertNotNull(com.blockreality.api.physics.FailureType.TENSION_BREAK);
    }

    private static void assertEqualsZero(long value, String message) {
        assertTrue(value == 0L, message + " (actual=" + value + ")");
    }
}
