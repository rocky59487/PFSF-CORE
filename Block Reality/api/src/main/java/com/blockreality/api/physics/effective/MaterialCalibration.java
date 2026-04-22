package com.blockreality.api.physics.effective;

/**
 * 材料在特定離散尺度下的有效參數三元組 key：(materialId, voxelScale, boundary)。
 *
 * <p>Phase E 完整欄位版本，向後相容 Phase A 的 10-欄位 constructor（見
 * {@link #legacy(int, String, int, BoundaryProfile, double, double, double, double, long, String)}）。
 *
 * <h2>Calibration 流程</h2>
 * <p>{@code CalibrationRunner}（見同套件）對每個 (material, scale, boundary) 組合：
 * <ol>
 *   <li>用內建解析解（Euler-Bernoulli beam、半圓拱閉合解）計算參考 φ_reference</li>
 *   <li>以網格搜尋 edgePenaltyEff / cornerPenaltyEff / gcEff 最小化 L2(φ_discrete − φ_reference)</li>
 *   <li>產生 record 並寫入 JSON</li>
 * </ol>
 *
 * <h2>Fallback</h2>
 * <p>當 Registry 找不到 (mat, scale, boundary) 時，用 {@link #defaultFor(String)} 產生
 * 保守預設值。
 *
 * <h2>Schema version</h2>
 * <ul>
 *   <li>v1：Phase A 初始 10 欄位（基礎 sigmaEff/gcEff/l0Eff/phaseFieldExponent）</li>
 *   <li>v2：Phase E 擴充到 14 欄位（+ rcompEff/rtensEff/edgePenaltyEff/cornerPenaltyEff）</li>
 * </ul>
 */
public record MaterialCalibration(
    int schemaVersion,
    String materialId,
    int voxelScale,
    BoundaryProfile boundary,
    double sigmaEff,
    double gcEff,
    double l0Eff,
    double phaseFieldExponent,
    // ─── v2 擴充欄位（Phase E） ───
    double rcompEff,         // 有效壓強（MPa 或歸一化）
    double rtensEff,         // 有效拉強（MPa）
    double edgePenaltyEff,   // per-(mat, scale) 校準後的 EDGE_P override（null/0 → 走 PFSFStencil 全域值）
    double cornerPenaltyEff, // 同上
    long timestamp,
    String solverCommit
) {

    /** Schema 版本常數 */
    public static final int SCHEMA_V1 = 1;
    public static final int SCHEMA_V2 = 2;

    public static final double DEFAULT_PHASE_FIELD_EXPONENT_CONCRETE = 2.0;
    public static final double DEFAULT_PHASE_FIELD_EXPONENT_STEEL    = 4.0;
    public static final double DEFAULT_GC = 0.12;
    public static final double DEFAULT_L0 = 1.5;

    /** Phase A 向後相容建構子：10 欄位，v1 schema，擴充欄位取合理預設 */
    public static MaterialCalibration legacy(
            int schemaVersion, String materialId, int voxelScale,
            BoundaryProfile boundary,
            double sigmaEff, double gcEff, double l0Eff, double phaseFieldExponent,
            long timestamp, String solverCommit) {
        return new MaterialCalibration(
            schemaVersion, materialId, voxelScale, boundary,
            sigmaEff, gcEff, l0Eff, phaseFieldExponent,
            0.0, 0.0, 0.0, 0.0,   // v2 欄位先放 0 表示「未校準 → 走全域值」
            timestamp, solverCommit
        );
    }

    /** 產生未校準的保守預設值（fallback path） */
    public static MaterialCalibration defaultFor(String materialId) {
        return new MaterialCalibration(
            SCHEMA_V2, materialId, /*voxelScale*/ 1, BoundaryProfile.ANCHORED_BOTTOM,
            /*sigmaEff*/ 1.0,
            DEFAULT_GC, DEFAULT_L0, DEFAULT_PHASE_FIELD_EXPONENT_CONCRETE,
            /*rcompEff*/ 25.0, /*rtensEff*/ 2.5,
            /*edgePenaltyEff*/ 0.0, /*cornerPenaltyEff*/ 0.0,
            /*timestamp*/ 0L, /*solverCommit*/ "default"
        );
    }

    /** 檢查 edge penalty override 是否啟用（0 表未校準，走全域 PFSFStencil） */
    public boolean hasEdgePenaltyOverride() {
        return edgePenaltyEff > 0.0;
    }

    public boolean hasCornerPenaltyOverride() {
        return cornerPenaltyEff > 0.0;
    }

    /** 邊界條件類型 */
    public enum BoundaryProfile {
        /** 底面錨定（Dirichlet φ=0） */
        ANCHORED_BOTTOM,
        /** 兩端錨定（拱橋、樑等結構） */
        ANCHORED_BOTH_ENDS,
        /** 完全自由 */
        FREE,
        /** 週期性邊界 */
        PERIODIC
    }
}
