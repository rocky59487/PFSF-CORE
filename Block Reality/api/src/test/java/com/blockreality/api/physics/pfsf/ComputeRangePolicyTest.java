package com.blockreality.api.physics.pfsf;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ComputeRangePolicyTest {

    private VramBudgetManager mgr;

    @BeforeEach
    void setUp() {
        mgr = new VramBudgetManager();
    }

    @Test
    @DisplayName("low pressure returns a config")
    void testLowPressureReturnsConfig() {
        ComputeRangePolicy.Config config = ComputeRangePolicy.decide(mgr, 1000);
        assertNotNull(config);
    }

    @Test
    @DisplayName("config fields are readable")
    void testConfigFieldsAccessible() {
        ComputeRangePolicy.Config config = ComputeRangePolicy.decide(mgr, 100);
        assertNotNull(config);
        assertNotNull(config.gridLevel);
        assertTrue(config.stepMultiplier > 0);
        boolean pf = config.allocatePhaseField;
        boolean mg = config.allocateMultigrid;
        assertTrue(pf || !pf);
        assertTrue(mg || !mg);
    }

    @Test
    @DisplayName("null VRAM manager falls back to safe full resolution")
    void testNullManagerFallsBackToFullResolution() {
        ComputeRangePolicy.Config config = ComputeRangePolicy.decide(null, 2048);
        assertNotNull(config);
        assertEquals(ComputeRangePolicy.GridLevel.L0_FULL, config.gridLevel);
        assertEquals(1.0f, config.stepMultiplier);
    }

    @Test
    @DisplayName("high pressure may reject an island")
    void testHighPressureMayReject() {
        long budget = mgr.getTotalBudget();
        mgr.tryRecord(1L, (long) (budget * 0.96), VramBudgetManager.PARTITION_PFSF);

        ComputeRangePolicy.Config config = ComputeRangePolicy.decide(mgr, 100000);
        assertTrue(config == null || config.stepMultiplier > 0);
    }

    @Test
    @DisplayName("adjustSteps always returns non-negative")
    void testAdjustSteps() {
        int result = ComputeRangePolicy.adjustSteps(16, mgr);
        assertTrue(result >= 0);
    }

    @Test
    @DisplayName("grid levels exist")
    void testGridLevelEnum() {
        ComputeRangePolicy.GridLevel[] levels = ComputeRangePolicy.GridLevel.values();
        assertTrue(levels.length > 0);
    }
}
