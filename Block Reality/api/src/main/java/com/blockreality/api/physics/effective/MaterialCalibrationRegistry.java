package com.blockreality.api.physics.effective;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Scanner;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Phase E — 材料有效參數資料庫（執行緒安全）。
 *
 * <p>Registry key = {@link Key}(materialId, voxelScale, boundary)；value = {@link MaterialCalibration}。
 * 找不到時回傳 {@link MaterialCalibration#defaultFor(String)} fallback 並記錄 warning
 * 計數（避免 spam logger）。
 *
 * <h2>JSON 載入</h2>
 * <p>啟動時從 classpath 的 {@code blockreality/calibration/default.json} 載入。
 * 為避免引入第三方 JSON 依賴，使用輕量正則解析（僅支援我們產出的扁平 entries 格式）。
 * 若需複雜 schema，可升級為 Jackson / Gson（留待未來）。
 *
 * <h2>JSON Schema (flat)</h2>
 * <pre>
 * {
 *   "schemaVersion": 2,
 *   "entries": [
 *     {
 *       "materialId": "concrete_c30",
 *       "voxelScale": 1,
 *       "boundary": "ANCHORED_BOTTOM",
 *       "sigmaEff": 1.0,
 *       "gcEff": 0.12,
 *       "l0Eff": 1.5,
 *       "phaseFieldExponent": 2.0,
 *       "rcompEff": 30.0,
 *       "rtensEff": 3.0,
 *       "edgePenaltyEff": 0.35,
 *       "cornerPenaltyEff": 0.15,
 *       "timestamp": 1713772800,
 *       "solverCommit": "abc123"
 *     }
 *   ]
 * }
 * </pre>
 *
 * <h2>Thread-safety</h2>
 * {@link ConcurrentHashMap} 支援 concurrent reads + single-writer register。
 *
 * <h2>Fallback 計數</h2>
 * 第 1 次 fallback 打 warning、之後每 100 次才再打一次，避免 log spam。
 */
public final class MaterialCalibrationRegistry {

    private static final org.slf4j.Logger LOGGER =
        org.slf4j.LoggerFactory.getLogger(MaterialCalibrationRegistry.class);

    public static final String DEFAULT_RESOURCE_PATH = "blockreality/calibration/default.json";

    private static final MaterialCalibrationRegistry INSTANCE = new MaterialCalibrationRegistry();
    /** 是否已嘗試載入過 default.json（避免反覆重試） */
    private final AtomicInteger initStatus = new AtomicInteger(0);  // 0=未載入 1=已載入 -1=載入失敗
    private final Map<Key, MaterialCalibration> store = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<String, AtomicLong> fallbackCounters = new ConcurrentHashMap<>();

    public static MaterialCalibrationRegistry getInstance() {
        INSTANCE.ensureLoaded();
        return INSTANCE;
    }

    /** 測試用：建立空 registry。不觸發 default.json 自動載入。 */
    public static MaterialCalibrationRegistry newEmpty() {
        MaterialCalibrationRegistry r = new MaterialCalibrationRegistry();
        r.initStatus.set(-1); // 視為「不要嘗試載入」
        return r;
    }

    private MaterialCalibrationRegistry() {}

    // ═══════════════════════════════════════════════════════════════
    //  Public API
    // ═══════════════════════════════════════════════════════════════

    public Optional<MaterialCalibration> lookup(
            String materialId, int voxelScale, MaterialCalibration.BoundaryProfile boundary) {
        MaterialCalibration c = store.get(new Key(materialId, voxelScale, boundary));
        return Optional.ofNullable(c);
    }

    /**
     * 查詢校準值；找不到時回 fallback 預設。
     */
    public MaterialCalibration getOrDefault(
            String materialId, int voxelScale, MaterialCalibration.BoundaryProfile boundary) {
        Objects.requireNonNull(materialId, "materialId");
        Objects.requireNonNull(boundary, "boundary");
        MaterialCalibration c = store.get(new Key(materialId, voxelScale, boundary));
        if (c != null) return c;

        warnFallbackOnce(materialId, voxelScale, boundary);
        return MaterialCalibration.defaultFor(materialId);
    }

    /** 註冊/更新一筆校準值 */
    public void register(MaterialCalibration calib) {
        Objects.requireNonNull(calib, "calib");
        Key k = new Key(calib.materialId(), calib.voxelScale(), calib.boundary());
        store.put(k, calib);
    }

    /** 目前已註冊的條目數（測試用） */
    public int size() {
        return store.size();
    }

    /** 清空（測試用） */
    public void clear() {
        store.clear();
        fallbackCounters.clear();
    }

    /** 是否已完成 default.json 載入（或確認載入失敗） */
    public boolean isLoaded() {
        return initStatus.get() == 1;
    }

    // ═══════════════════════════════════════════════════════════════
    //  JSON 載入（正則 parser，依賴零，schema v1/v2 兼容）
    // ═══════════════════════════════════════════════════════════════

    private void ensureLoaded() {
        if (initStatus.get() != 0) return;
        synchronized (this) {
            if (initStatus.get() != 0) return;
            try (InputStream in = Thread.currentThread()
                    .getContextClassLoader()
                    .getResourceAsStream(DEFAULT_RESOURCE_PATH)) {
                if (in == null) {
                    LOGGER.info("[MaterialCalibrationRegistry] {} 不存在，啟用純 fallback 模式",
                                DEFAULT_RESOURCE_PATH);
                    initStatus.set(-1);
                    return;
                }
                String content = new Scanner(in, StandardCharsets.UTF_8).useDelimiter("\\A").next();
                int loaded = parseAndLoad(content);
                LOGGER.info("[MaterialCalibrationRegistry] 載入 {} entries 自 {}",
                            loaded, DEFAULT_RESOURCE_PATH);
                initStatus.set(1);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }

    /**
     * 解析 JSON entries。正則解析每個 `{...}` block 為 key=value pair。
     * 支援 v1（缺 rcompEff 等欄位，以 0 填充）與 v2（全 14 欄位）。
     * @return 成功載入的 entry 數
     */
    int parseAndLoad(String json) {
        Pattern entryPattern = Pattern.compile("\\{([^{}]+)}", Pattern.DOTALL);
        Matcher em = entryPattern.matcher(json);
        int count = 0;
        // 跳過最外層 { ... entries: [...] }；直接掃描所有 inner braces
        while (em.find()) {
            String body = em.group(1);
            if (!body.contains("materialId")) continue; // 濾掉非 entry 的 brace
            try {
                MaterialCalibration c = parseEntry(body);
                register(c);
                count++;
            } catch (RuntimeException e) {
                LOGGER.warn("[MaterialCalibrationRegistry] 跳過無效 entry: {}", e.getMessage());
            }
        }
        return count;
    }

    private MaterialCalibration parseEntry(String body) {
        String materialId = stringField(body, "materialId");
        int voxelScale    = (int) doubleField(body, "voxelScale");
        String boundaryS  = stringField(body, "boundary");
        double sigma      = doubleField(body, "sigmaEff");
        double gc         = doubleField(body, "gcEff");
        double l0         = doubleField(body, "l0Eff");
        double p          = doubleField(body, "phaseFieldExponent");
        double rcomp      = optDouble(body, "rcompEff", 0.0);
        double rtens      = optDouble(body, "rtensEff", 0.0);
        double edgeP      = optDouble(body, "edgePenaltyEff", 0.0);
        double cornerP    = optDouble(body, "cornerPenaltyEff", 0.0);
        long ts           = (long) optDouble(body, "timestamp", 0.0);
        String commit     = optString(body, "solverCommit", "unknown");

        int schema = (rcomp != 0.0 || rtens != 0.0) ? MaterialCalibration.SCHEMA_V2
                                                     : MaterialCalibration.SCHEMA_V1;
        return new MaterialCalibration(
            schema, materialId, voxelScale,
            MaterialCalibration.BoundaryProfile.valueOf(boundaryS),
            sigma, gc, l0, p, rcomp, rtens, edgeP, cornerP,
            ts, commit
        );
    }

    private static String stringField(String body, String key) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"").matcher(body);
        if (!m.find()) throw new IllegalArgumentException("缺少 string 欄位 " + key);
        return m.group(1);
    }

    private static double doubleField(String body, String key) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*(-?[\\d.eE+]+)").matcher(body);
        if (!m.find()) throw new IllegalArgumentException("缺少數字欄位 " + key);
        return Double.parseDouble(m.group(1));
    }

    private static double optDouble(String body, String key, double dflt) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*(-?[\\d.eE+]+)").matcher(body);
        return m.find() ? Double.parseDouble(m.group(1)) : dflt;
    }

    private static String optString(String body, String key, String dflt) {
        Matcher m = Pattern.compile("\"" + key + "\"\\s*:\\s*\"([^\"]*)\"").matcher(body);
        return m.find() ? m.group(1) : dflt;
    }

    // ═══════════════════════════════════════════════════════════════
    //  Fallback log rate-limit
    // ═══════════════════════════════════════════════════════════════

    private void warnFallbackOnce(String materialId, int voxelScale,
                                   MaterialCalibration.BoundaryProfile boundary) {
        String keyStr = materialId + ":" + voxelScale + ":" + boundary;
        AtomicLong counter = fallbackCounters.computeIfAbsent(keyStr, k -> new AtomicLong(0));
        long n = counter.incrementAndGet();
        if (n == 1 || n % 100 == 0) {
            LOGGER.warn("[MaterialCalibrationRegistry] Fallback #{} for (mat={}, scale={}, bnd={})",
                        n, materialId, voxelScale, boundary);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    //  Key record
    // ═══════════════════════════════════════════════════════════════

    public record Key(String materialId, int voxelScale,
                      MaterialCalibration.BoundaryProfile boundary) {}
}
