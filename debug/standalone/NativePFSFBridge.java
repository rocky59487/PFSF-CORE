package com.blockreality.api.physics.pfsf;

import java.nio.ByteBuffer;

/**
 * Minimal standalone NativePFSFBridge — matches JNI symbol names exactly.
 * No Forge/SLF4J/NativeLibLoader dependency: loads .so files directly.
 */
public final class NativePFSFBridge {

    static final String BUILD_DIR =
        "/home/user/Block-Realityapi-Fast-design/build/native-build";

    static boolean LOADED = false;
    public static String  LOAD_ERROR = "not attempted";

    static {
        try {
            System.load(BUILD_DIR + "/libbr_core/libbr_core.so");
            System.load(BUILD_DIR + "/libpfsf/libpfsf.so");
            System.load(BUILD_DIR + "/libpfsf/libblockreality_pfsf.so");
            LOADED = true;
            LOAD_ERROR = null;
        } catch (Throwable t) {
            LOAD_ERROR = t.getMessage();
        }
    }

    private NativePFSFBridge() {}

    public static boolean isAvailable() { return LOADED; }

    // ── static queries (no handle) ────────────────────────────────────────
    public static native String nativeVersion();
    public static native String nativeAbiContractVersion();
    public static native boolean nativeHasFeature(String name);
    public static native String nativeBuildInfo();

    // ── engine lifecycle ─────────────────────────────────────────────────
    public static native long  nativeCreate(int maxIslandSize, int tickBudgetMs,
                                             long vramBudgetBytes,
                                             boolean enablePhaseField,
                                             boolean enableMultigrid);
    public static native int   nativeInit(long handle);
    public static native void  nativeShutdown(long handle);
    public static native void  nativeDestroy(long handle);
    public static native boolean nativeIsAvailable(long handle);
    public static native long[] nativeGetStats(long handle);
    public static native void  nativeSetPCGEnabled(long handle, boolean enabled);

    // ── island management ─────────────────────────────────────────────────
    public static native int nativeAddIsland(long handle, int islandId,
                                              int ox, int oy, int oz,
                                              int lx, int ly, int lz);
    public static native void nativeRemoveIsland(long handle, int islandId);

    // ── buffer registration ───────────────────────────────────────────────
    public static native int nativeRegisterIslandBuffers(long handle, int islandId,
                                                          ByteBuffer phi, ByteBuffer source,
                                                          ByteBuffer conductivity,
                                                          ByteBuffer voxelType,
                                                          ByteBuffer rcomp, ByteBuffer rtens,
                                                          ByteBuffer maxPhi);
    public static native int nativeRegisterIslandLookups(long handle, int islandId,
                                                          ByteBuffer materialId,
                                                          ByteBuffer anchorBitmap,
                                                          ByteBuffer fluidPressure,
                                                          ByteBuffer curing);
    public static native int nativeRegisterStressReadback(long handle, int islandId,
                                                           ByteBuffer stress);

    // ── sparse update ────────────────────────────────────────────────────
    public static native ByteBuffer nativeGetSparseUploadBuffer(long handle, int islandId);
    public static native int nativeNotifySparseUpdates(long handle, int islandId,
                                                        int updateCount);

    // ── tick ─────────────────────────────────────────────────────────────
    public static native int nativeTickDbb(long handle, int[] dirtyIslandIds,
                                            long currentEpoch, ByteBuffer failureBuffer);

    // ── compute primitives ────────────────────────────────────────────────
    public static native float nativeNormalizeSoA6(float[] source, float[] rcomp,
                                                    float[] rtens, float[] conductivity,
                                                    float[] hydrationOrNull, int n);
    public static native float nativeWindPressureSource(float windSpeed, float density,
                                                         boolean exposed);
    public static native float nativeTimoshenkoMomentFactor(float w, float h,
                                                             int arm, float youngsGPa,
                                                             float poisson);
}
