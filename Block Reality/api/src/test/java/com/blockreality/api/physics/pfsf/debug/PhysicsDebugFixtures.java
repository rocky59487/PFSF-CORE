package com.blockreality.api.physics.pfsf.debug;

import org.lwjgl.system.MemoryUtil;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Shared buffer helpers for the incremental physics debug test suite (L0–L6).
 *
 * <p>All allocations are 256-byte aligned via LWJGL {@link MemoryUtil#memAlignedAlloc}
 * to satisfy the {@code nativeRegisterIslandBuffers} ABI contract. Callers must call
 * {@link IslandBuffers#free()} in {@code @AfterEach} / {@code @AfterAll} to avoid
 * off-heap leaks.</p>
 */
final class PhysicsDebugFixtures {

    static final long ENGINE_VRAM_BYTES = 64L * 1024 * 1024;
    static final int  ISLAND_ID        = 1;

    private static final int ALIGN = 256;

    /**
     * Allocates a 256-byte-aligned, zero-initialised, LITTLE_ENDIAN direct ByteBuffer.
     * Size is rounded up to the next multiple of 256.
     */
    static ByteBuffer allocAligned(int bytes) {
        int aligned = Math.max(ALIGN, (bytes + ALIGN - 1) & ~(ALIGN - 1));
        ByteBuffer buf = MemoryUtil.memAlignedAlloc(ALIGN, aligned);
        MemoryUtil.memSet(buf, 0);
        return buf.order(ByteOrder.LITTLE_ENDIAN);
    }

    static void free(ByteBuffer buf) {
        if (buf != null) MemoryUtil.memAlignedFree(buf);
    }

    /**
     * All island buffers for one (lx × ly × lz) region, matching the
     * {@code nativeRegisterIslandBuffers} + {@code nativeRegisterIslandLookups} ABI.
     */
    record IslandBuffers(
        ByteBuffer phi,
        ByteBuffer source,
        ByteBuffer conductivity,   // float32 × 6N  (SoA: dir*N + i)
        ByteBuffer voxelType,      // uint8   × N
        ByteBuffer rcomp,          // float32 × N
        ByteBuffer rtens,          // float32 × N
        ByteBuffer maxPhi,         // float32 × N
        ByteBuffer materialId,     // int32   × N
        ByteBuffer anchorBitmap,   // int64   × N
        ByteBuffer fluidPressure,  // float32 × N
        ByteBuffer curing,         // float32 × N
        int lx, int ly, int lz
    ) {
        int n() { return lx * ly * lz; }

        void free() {
            PhysicsDebugFixtures.free(phi);
            PhysicsDebugFixtures.free(source);
            PhysicsDebugFixtures.free(conductivity);
            PhysicsDebugFixtures.free(voxelType);
            PhysicsDebugFixtures.free(rcomp);
            PhysicsDebugFixtures.free(rtens);
            PhysicsDebugFixtures.free(maxPhi);
            PhysicsDebugFixtures.free(materialId);
            PhysicsDebugFixtures.free(anchorBitmap);
            PhysicsDebugFixtures.free(fluidPressure);
            PhysicsDebugFixtures.free(curing);
        }
    }

    /**
     * Builds a minimal solid-block island where the bottom layer (y=0) is anchored.
     *
     * <ul>
     *   <li>All voxels: type=1 (solid), conductivity=1.0, source=0.5, rcomp=20, rtens=2, curing=1</li>
     *   <li>y=0 layer: anchorBitmap set</li>
     *   <li>When {@code triggerFailure=true}: the center floating voxel at
     *       ({@code lx/2}, 1, {@code lz/2}) gets maxPhi=0.001 and rcomp=0.001 so the
     *       failure scan fires after one tick.</li>
     * </ul>
     */
    static IslandBuffers buildIslandBuffers(int lx, int ly, int lz, boolean triggerFailure) {
        int N = lx * ly * lz;

        ByteBuffer phi          = allocAligned(N * 4);
        ByteBuffer source       = allocAligned(N * 4);
        ByteBuffer conductivity = allocAligned(6 * N * 4);
        ByteBuffer voxelType    = allocAligned(N);
        ByteBuffer rcomp        = allocAligned(N * 4);
        ByteBuffer rtens        = allocAligned(N * 4);
        ByteBuffer maxPhiBuf    = allocAligned(N * 4);
        ByteBuffer materialId   = allocAligned(N * 4);
        ByteBuffer anchorBitmap = allocAligned(N * 8);
        ByteBuffer fluidPressure = allocAligned(N * 4);
        ByteBuffer curing       = allocAligned(N * 4);

        for (int x = 0; x < lx; x++) {
            for (int y = 0; y < ly; y++) {
                for (int z = 0; z < lz; z++) {
                    int i = x + lx * (y + ly * z);
                    voxelType.put(i, (byte) 1);
                    source.putFloat(i * 4, 0.5f);
                    rcomp.putFloat(i * 4, 20.0f);
                    rtens.putFloat(i * 4, 2.0f);
                    maxPhiBuf.putFloat(i * 4, 10.0f);
                    curing.putFloat(i * 4, 1.0f);
                    for (int d = 0; d < 6; d++) {
                        conductivity.putFloat((d * N + i) * 4, 1.0f);
                    }
                }
            }
        }

        // Anchor the bottom layer
        for (int x = 0; x < lx; x++) {
            for (int z = 0; z < lz; z++) {
                int i = x + lx * (0 + ly * z);
                anchorBitmap.putLong(i * 8, 1L);
            }
        }

        // Failure trigger: center floating voxel gets near-zero thresholds
        if (triggerFailure && ly >= 2) {
            int cx = lx / 2, cz = lz / 2;
            int fi = cx + lx * (1 + ly * cz);
            maxPhiBuf.putFloat(fi * 4, 0.001f);
            rcomp.putFloat(fi * 4, 0.001f);
        }

        return new IslandBuffers(phi, source, conductivity, voxelType, rcomp, rtens,
                maxPhiBuf, materialId, anchorBitmap, fluidPressure, curing, lx, ly, lz);
    }

    private PhysicsDebugFixtures() {}
}
