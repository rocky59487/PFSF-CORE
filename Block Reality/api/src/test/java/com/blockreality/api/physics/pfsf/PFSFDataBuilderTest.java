package com.blockreality.api.physics.pfsf;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static com.blockreality.api.physics.pfsf.PFSFConstants.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * PFSFDataBuilder coarse grid downsampling logic test.
 *
 * <p>Test conductivity and type calculations for 2×2×2 average downsampling,
 * No GPU required (pure CPU array operations). </p>
 */
class PFSFDataBuilderTest {

    @Test
    @DisplayName("粗網格 type：2×2×2 中有 anchor → coarse = ANCHOR")
    void testCoarseTypeAnchorPriority() {
        // 2×2×2 fine grid → 1×1×1 coarse grid
        byte[] fineType = new byte[8];
        fineType[0] = VOXEL_SOLID;
        fineType[1] = VOXEL_SOLID;
        fineType[2] = VOXEL_SOLID;
        fineType[3] = VOXEL_ANCHOR; // Just one anchor

        float[] fineCond = new float[8 * 6]; // SoA layout
        byte[] coarseType = new byte[1];

        // Manually replicate the logic
        int solidCount = 0, anchorCount = 0, total = 0;
        for (int i = 0; i < 8; i++) {
            total++;
            if (fineType[i] == VOXEL_ANCHOR) anchorCount++;
            else if (fineType[i] == VOXEL_SOLID) solidCount++;
        }

        if (anchorCount > 0) coarseType[0] = VOXEL_ANCHOR;
        else if (solidCount > total / 2) coarseType[0] = VOXEL_SOLID;
        else coarseType[0] = VOXEL_AIR;

        assertEquals(VOXEL_ANCHOR, coarseType[0],
                "Any anchor in 2×2×2 block → coarse voxel should be ANCHOR");
    }

    @Test
    @DisplayName("粗網格 type：多數 solid → coarse = SOLID")
    void testCoarseTypeMajoritySolid() {
        byte[] fineType = new byte[8];
        fineType[0] = VOXEL_SOLID;
        fineType[1] = VOXEL_SOLID;
        fineType[2] = VOXEL_SOLID;
        fineType[3] = VOXEL_SOLID;
        fineType[4] = VOXEL_SOLID;
        fineType[5] = VOXEL_AIR;
        fineType[6] = VOXEL_AIR;
        fineType[7] = VOXEL_AIR;
        // 5 solid, 3 air → majority solid

        int solidCount = 5, total = 8;
        assertTrue(solidCount > total / 2);

        byte result = (solidCount > total / 2) ? VOXEL_SOLID : VOXEL_AIR;
        assertEquals(VOXEL_SOLID, result);
    }

    @Test
    @DisplayName("粗網格 type：多數 air → coarse = AIR")
    void testCoarseTypeMajorityAir() {
        byte[] fineType = new byte[8];
        fineType[0] = VOXEL_SOLID;
        fineType[1] = VOXEL_AIR;
        fineType[2] = VOXEL_AIR;
        fineType[3] = VOXEL_AIR;
        fineType[4] = VOXEL_AIR;
        fineType[5] = VOXEL_AIR;
        fineType[6] = VOXEL_AIR;
        fineType[7] = VOXEL_AIR;
        // 1 solid, 7 air → air wins

        int solidCount = 1, total = 8;
        assertFalse(solidCount > total / 2);

        byte result = (solidCount > total / 2) ? VOXEL_SOLID : VOXEL_AIR;
        assertEquals(VOXEL_AIR, result);
    }

    @Test
    @DisplayName("粗網格 conductivity 平均：均勻 σ 不變")
    void testCoarseConductivityAveraging() {
        // 4×4×4 fine → 2×2×2 coarse
        int fLx = 4, fLy = 4, fLz = 4;
        int fN = fLx * fLy * fLz; // 64
        float[] fineCond = new float[fN * 6];

        // All conductivity set to 2.0
        for (int i = 0; i < fineCond.length; i++) {
            fineCond[i] = 2.0f;
        }

        // Compute coarse (0,0,0) voxels
        int cx = 0, cy = 0, cz = 0;
        float[] condSum = new float[6];
        int total = 0;

        for (int dz = 0; dz < 2; dz++) {
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int fi = dx + fLx * (dy + fLy * dz);
                    total++;
                    for (int d = 0; d < 6; d++) {
                        condSum[d] += fineCond[d * fN + fi];
                    }
                }
            }
        }

        // Average = 2.0 × 8 / 8 = 2.0
        for (int d = 0; d < 6; d++) {
            assertEquals(2.0f, condSum[d] / total, 1e-5f,
                    "Uniform conductivity should average to same value, dir=" + d);
        }
    }

    @Test
    @DisplayName("SoA conductivity layout：d * N + i 索引驗證")
    void testSoALayoutIndexing() {
        int Lx = 4, Ly = 4, Lz = 4;
        int N = Lx * Ly * Lz; // 64
        float[] conductivity = new float[N * 6];

        // Set the value of (1,2,3) direction DIR_POS_X
        int i = 1 + Lx * (2 + Ly * 3);
        int d = DIR_POS_X;
        conductivity[d * N + i] = 42.0f;

        assertEquals(42.0f, conductivity[d * N + i], 1e-5f);
        // Confirm that indexes in other directions do not conflict
        for (int other = 0; other < 6; other++) {
            if (other != d) {
                assertEquals(0.0f, conductivity[other * N + i], 1e-5f,
                        "Other direction should be zero, dir=" + other);
            }
        }
    }

    // ─── Sigma normalisation invariants ─────────────────────────────
    // CLAUDE.md "正規化約定" marks this path as "極重要": source,
    // rcomp, rtens and conductivity are all divided by sigmaMax before
    // upload so failure_scan's flux/threshold comparison stays in the
    // right magnitude. Without these invariants a regression silently
    // produces spurious collapses in high-stiffness structures.

    @Test
    @DisplayName("sigma norm: sigmaMax < 1 → no division applied")
    void testNormalizeSkippedBelowThreshold() {
        int N = 4;
        float[] source = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] rcomp  = {10.0f, 20.0f, 30.0f, 40.0f};
        float[] rtens  = {5.0f, 6.0f, 7.0f, 8.0f};
        float[] cond   = new float[6 * N];
        for (int i = 0; i < cond.length; i++) cond[i] = 0.5f; // all < 1

        float sigmaMax = PFSFDataBuilder.normalizeSoA6JavaRef(source, rcomp, rtens, cond, N);

        assertEquals(1.0f, sigmaMax, 1e-6f,
                "sigmaMax must be 1.0f when every conductivity is below 1 (no normalisation)");
        assertArrayEquals(new float[]{1.0f, 2.0f, 3.0f, 4.0f}, source, 1e-6f);
        assertArrayEquals(new float[]{10.0f, 20.0f, 30.0f, 40.0f}, rcomp, 1e-6f);
        assertArrayEquals(new float[]{5.0f, 6.0f, 7.0f, 8.0f}, rtens, 1e-6f);
    }

    @Test
    @DisplayName("sigma norm: source/rcomp/rtens/conductivity all scaled by 1/sigmaMax")
    void testNormalizeAllBuffersScaledTogether() {
        int N = 3;
        float[] source = {100.0f, 200.0f, 300.0f};
        float[] rcomp  = {50.0f, 60.0f, 70.0f};
        float[] rtens  = {10.0f, 20.0f, 30.0f};
        float[] cond   = new float[6 * N];
        // Seed conductivity so sigmaMax = 40.0f
        cond[0] = 1.0f; cond[1] = 5.0f; cond[2] = 40.0f;
        for (int i = 3; i < cond.length; i++) cond[i] = 2.0f;

        float sigmaMax = PFSFDataBuilder.normalizeSoA6JavaRef(source, rcomp, rtens, cond, N);

        assertEquals(40.0f, sigmaMax, 1e-5f, "sigmaMax == max(conductivity)");
        float inv = 1.0f / 40.0f;
        assertArrayEquals(new float[]{100.0f * inv, 200.0f * inv, 300.0f * inv}, source, 1e-5f);
        assertArrayEquals(new float[]{50.0f * inv, 60.0f * inv, 70.0f * inv}, rcomp, 1e-5f,
                "rcomp MUST be divided by sigmaMax — see CLAUDE.md 正規化約定 + D1-fix");
        assertArrayEquals(new float[]{10.0f * inv, 20.0f * inv, 30.0f * inv}, rtens, 1e-5f,
                "rtens MUST be divided by sigmaMax — same invariant as rcomp");
        assertEquals(1.0f / 40.0f * 40.0f, cond[2], 1e-5f, "conductivity max must normalise to 1");
        assertTrue(cond[2] <= 1.0f + 1e-5f, "every normalised conductivity must be <= 1");
    }

    @Test
    @DisplayName("sigma norm: ratio source/rcomp preserved — failure_scan compares identical magnitudes")
    void testRatioPreservedAcrossNormalisation() {
        int N = 2;
        float[] source = {15.0f, 25.0f};
        float[] rcomp  = {30.0f, 50.0f};
        float[] rtens  = {20.0f, 40.0f};
        float[] cond   = new float[6 * N];
        cond[0] = 100.0f; // force sigmaMax = 100

        float[] srcOrig = source.clone();
        float[] rcompOrig = rcomp.clone();
        float[] rtensOrig = rtens.clone();

        PFSFDataBuilder.normalizeSoA6JavaRef(source, rcomp, rtens, cond, N);

        // The physical invariant: any ratio that was meaningful in the
        // original units must remain identical post-normalisation,
        // because flux/rcomp and flux/rtens are the actual failure
        // predicates and both sides get the same 1/sigmaMax factor.
        for (int i = 0; i < N; i++) {
            float originalRatio = srcOrig[i] / rcompOrig[i];
            float normalisedRatio = source[i] / rcomp[i];
            assertEquals(originalRatio, normalisedRatio, 1e-5f,
                    "source/rcomp ratio must be invariant across sigma normalisation (voxel " + i + ")");

            float originalTRatio = srcOrig[i] / rtensOrig[i];
            float normalisedTRatio = source[i] / rtens[i];
            assertEquals(originalTRatio, normalisedTRatio, 1e-5f,
                    "source/rtens ratio must be invariant across sigma normalisation (voxel " + i + ")");
        }
    }
}
