package com.blockreality.api.physics.effective;

import org.junit.jupiter.api.Test;

import java.util.Optional;

import static com.blockreality.api.physics.effective.MaterialCalibration.BoundaryProfile.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase E — {@link MaterialCalibrationRegistry} 行為驗證。
 *
 * <p>驗證 JSON schema 解析、fallback 路徑、v1/v2 schema 相容性、thread-safety
 * (透過大量並發 register + lookup)。
 */
class MaterialCalibrationRegistryTest {

    @Test
    void lookupMissingEntry_returnsEmpty() {
        MaterialCalibrationRegistry reg = MaterialCalibrationRegistry.newEmpty();
        Optional<MaterialCalibration> c = reg.lookup("unknown_mat", 1, ANCHORED_BOTTOM);
        assertTrue(c.isEmpty(), "不存在的 key 應回傳 empty Optional");
    }

    @Test
    void getOrDefault_returnsFallback_whenMissing() {
        MaterialCalibrationRegistry reg = MaterialCalibrationRegistry.newEmpty();
        MaterialCalibration fallback = reg.getOrDefault("unknown", 1, ANCHORED_BOTTOM);
        assertNotNull(fallback);
        assertEquals("unknown", fallback.materialId());
        assertEquals(MaterialCalibration.DEFAULT_GC, fallback.gcEff(), 1e-9);
    }

    @Test
    void registerThenLookup_roundtrip() {
        MaterialCalibrationRegistry reg = MaterialCalibrationRegistry.newEmpty();
        MaterialCalibration c = new MaterialCalibration(
            MaterialCalibration.SCHEMA_V2, "test_mat", 2, ANCHORED_BOTH_ENDS,
            1.2, 0.15, 1.7, 2.5,
            50.0, 5.0, 0.40, 0.18,
            1L, "abc123"
        );
        reg.register(c);
        MaterialCalibration loaded = reg.getOrDefault("test_mat", 2, ANCHORED_BOTH_ENDS);
        assertEquals(c, loaded);
    }

    @Test
    void schemaV1Entry_loadsWithZeroedV2Fields() {
        MaterialCalibrationRegistry reg = MaterialCalibrationRegistry.newEmpty();
        String json = """
            {
              "schemaVersion": 1,
              "entries": [
                {
                  "materialId": "legacy_concrete",
                  "voxelScale": 1,
                  "boundary": "ANCHORED_BOTTOM",
                  "sigmaEff": 1.0,
                  "gcEff": 0.12,
                  "l0Eff": 1.5,
                  "phaseFieldExponent": 2.0,
                  "timestamp": 1000,
                  "solverCommit": "v1"
                }
              ]
            }
            """;
        int n = reg.parseAndLoad(json);
        assertEquals(1, n, "應載入 1 個 entry");

        MaterialCalibration c = reg.getOrDefault("legacy_concrete", 1, ANCHORED_BOTTOM);
        assertEquals(0.12, c.gcEff(), 1e-9);
        assertEquals(0.0, c.rcompEff(), 1e-9, "v1 entry 的 rcompEff 應為 0");
        assertEquals(0.0, c.edgePenaltyEff(), 1e-9);
        assertFalse(c.hasEdgePenaltyOverride(), "edgePenalty=0 應視為未覆寫");
    }

    @Test
    void schemaV2Entry_loadsAllFields() {
        MaterialCalibrationRegistry reg = MaterialCalibrationRegistry.newEmpty();
        String json = """
            {
              "schemaVersion": 2,
              "entries": [
                {
                  "materialId": "c30",
                  "voxelScale": 1,
                  "boundary": "ANCHORED_BOTTOM",
                  "sigmaEff": 1.0,
                  "gcEff": 0.12,
                  "l0Eff": 1.5,
                  "phaseFieldExponent": 2.0,
                  "rcompEff": 30.0,
                  "rtensEff": 3.0,
                  "edgePenaltyEff": 0.35,
                  "cornerPenaltyEff": 0.15,
                  "timestamp": 1713772800,
                  "solverCommit": "seed"
                }
              ]
            }
            """;
        int n = reg.parseAndLoad(json);
        assertEquals(1, n);

        MaterialCalibration c = reg.getOrDefault("c30", 1, ANCHORED_BOTTOM);
        assertEquals(MaterialCalibration.SCHEMA_V2, c.schemaVersion());
        assertEquals(30.0, c.rcompEff(), 1e-9);
        assertEquals(0.35, c.edgePenaltyEff(), 1e-9);
        assertTrue(c.hasEdgePenaltyOverride());
        assertTrue(c.hasCornerPenaltyOverride());
    }

    @Test
    void malformedEntry_skippedWithoutCrash() {
        MaterialCalibrationRegistry reg = MaterialCalibrationRegistry.newEmpty();
        String json = """
            {
              "entries": [
                { "materialId": "no_fields_except_id" },
                {
                  "materialId": "good",
                  "voxelScale": 1,
                  "boundary": "ANCHORED_BOTTOM",
                  "sigmaEff": 1.0,
                  "gcEff": 0.12,
                  "l0Eff": 1.5,
                  "phaseFieldExponent": 2.0,
                  "timestamp": 0,
                  "solverCommit": "ok"
                }
              ]
            }
            """;
        int n = reg.parseAndLoad(json);
        assertEquals(1, n, "僅 1 個有效 entry");
        assertTrue(reg.lookup("good", 1, ANCHORED_BOTTOM).isPresent());
    }

    @Test
    void defaultJsonShipsWithResources() {
        // 真實使用 singleton 觸發 classpath 載入
        MaterialCalibrationRegistry reg = MaterialCalibrationRegistry.getInstance();
        assertTrue(reg.isLoaded(), "default.json 應在 classpath 中並成功載入");
        // seed 中至少有 concrete_c30@scale=1
        Optional<MaterialCalibration> c = reg.lookup("concrete_c30", 1, ANCHORED_BOTTOM);
        assertTrue(c.isPresent(), "default.json seed 應含 concrete_c30@scale=1");
        assertEquals(30.0, c.get().rcompEff(), 1e-9);
    }

    @Test
    void concurrentRegisterAndLookup_noDataLoss() throws InterruptedException {
        MaterialCalibrationRegistry reg = MaterialCalibrationRegistry.newEmpty();
        int threads = 8;
        int perThread = 500;
        Thread[] ts = new Thread[threads];
        for (int i = 0; i < threads; i++) {
            final int tid = i;
            ts[i] = new Thread(() -> {
                for (int k = 0; k < perThread; k++) {
                    reg.register(new MaterialCalibration(
                        MaterialCalibration.SCHEMA_V2,
                        "m" + tid + "_" + k, 1, ANCHORED_BOTTOM,
                        1.0, 0.1, 1.5, 2.0,
                        10.0, 1.0, 0.35, 0.15,
                        0L, "c"
                    ));
                }
            });
            ts[i].start();
        }
        for (Thread t : ts) t.join();
        assertEquals(threads * perThread, reg.size());
    }
}
