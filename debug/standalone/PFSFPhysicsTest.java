import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;

/**
 * Standalone L0-L5 physics pipeline test.
 *
 * 每層失敗會直接 FAIL 並跳出，顯示最深通過的層次。
 * 不需要 Forge / SLF4J / Minecraft — 只靠 JNI + libblockreality_pfsf.so。
 *
 * 執行方式：
 *   VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
 *   java -cp . PFSFPhysicsTest
 */
public class PFSFPhysicsTest {

    // ── 共用結構尺寸 ─────────────────────────────────────────────────
    static final int LX = 4, LY = 4, LZ = 4;
    static final int N  = LX * LY * LZ;   // 64 voxels

    // voxel types (mirrors VOXEL_AIR / VOXEL_SOLID / VOXEL_ANCHOR in C++)
    static final int VTYPE_AIR    = 0;
    static final int VTYPE_SOLID  = 1;
    static final int VTYPE_ANCHOR = 2;

    // PFSF result codes
    static final int OK = 0;

    static long handle = 0;

    // ── 結果計數 ─────────────────────────────────────────────────────
    static int passed = 0;
    static int failed = 0;

    public static void main(String[] args) {
        System.setProperty("blockreality.native.pfsf", "true");

        System.out.println("═══════════════════════════════════════════════");
        System.out.println("  PFSF 物理管線測試  L0→L16");
        System.out.println("═══════════════════════════════════════════════");

        try {
            testL0_LibraryLoad();
            testL1_ComputePrimitives();
            testL2_VulkanInit();
            testL3_IslandBuffers();
            testL4_SingleTick();
            testL5_FailureDetection();
            testL6_PhiConvergence();
            testL7_MultiIslandBatch();
            testL8_IslandRemoveDuringSolve();
            testL9_LargeIsland();
            testL10_AnisotropicConductivity();
            testL11_FailureTypeDiscrimination();
            testL12_SourceFreeIsland();
            testL13_WarmStartEpochConsistency();
            testL14_ReregistrationAfterRemove();
            testL15_FailureBufferSaturation();
            testL16_NoSupportOrphanTrigger();
        } catch (TestFailException e) {
            System.out.println("\n[ABORT] " + e.getMessage());
        } finally {
            shutdownEngine();
        }

        System.out.println("\n═══════════════════════════════════════════════");
        System.out.printf("  結果：%d PASS  %d FAIL%n", passed, failed);
        System.out.println("═══════════════════════════════════════════════");
        System.exit(failed > 0 ? 1 : 0);
    }

    // ════════════════════════════════════════════════════════════════
    //  L0 — 庫載入 + 版本字串
    // ════════════════════════════════════════════════════════════════
    static void testL0_LibraryLoad() {
        header("L0", "庫載入 / 版本字串");

        if (!NativePFSFBridge.isAvailable()) {
            fail("L0.1", "libblockreality_pfsf.so 載入失敗: " + NativePFSFBridge.LOAD_ERROR);
            throw new TestFailException("L0 失敗 — 無法繼續");
        }
        pass("L0.1", "libblockreality_pfsf.so 載入成功");

        String ver = NativePFSFBridge.nativeVersion();
        check("L0.2", "nativeVersion() 非空", ver != null && !ver.isEmpty(), "got: " + ver);

        String abi = NativePFSFBridge.nativeAbiContractVersion();
        check("L0.3", "abiContractVersion 格式",
              abi != null && (abi.matches("\\d+\\.\\d+\\.\\d+") || abi.equals("n/a")),
              "got: " + abi);

        System.out.println("       版本=" + ver + "  ABI合約=" + abi);

        boolean v1 = NativePFSFBridge.nativeHasFeature("compute.v1");
        System.out.println("       compute.v1=" + v1
            + "  v2=" + NativePFSFBridge.nativeHasFeature("compute.v2")
            + "  v3=" + NativePFSFBridge.nativeHasFeature("compute.v3")
            + "  v4=" + NativePFSFBridge.nativeHasFeature("compute.v4"));
    }

    // ════════════════════════════════════════════════════════════════
    //  L1 — 無 Vulkan 的純 CPU 計算 primitives
    // ════════════════════════════════════════════════════════════════
    static void testL1_ComputePrimitives() {
        header("L1", "CPU Compute Primitives (無 Vulkan)");

        if (!NativePFSFBridge.nativeHasFeature("compute.v1")) {
            System.out.println("  [SKIP] compute.v1 不可用");
            return;
        }

        // 風壓：½ρv² = 0.5 * 1.225 * 100 = 61.25 Pa
        float wp = NativePFSFBridge.nativeWindPressureSource(10.0f, 1.225f, true);
        check("L1.1", "windPressureSource ½ρv²≈61.25",
              Math.abs(wp - 61.25f) < 5.0f, "got: " + wp);

        // Timoshenko 修正因子：合理範圍 0.5–2.0
        float tf = NativePFSFBridge.nativeTimoshenkoMomentFactor(1.0f, 1.0f, 2, 30.0f, 0.2f);
        check("L1.2", "timoshenkoMomentFactor 介於 0.5–2.0",
              tf > 0.5f && tf < 2.0f, "got: " + tf);

        // normalizeSoA6：各向同性 sigma=1.0，sigmaMax 應 ≈ 1.0
        float[] src  = new float[N];
        float[] rc   = new float[N];
        float[] rt   = new float[N];
        float[] cond = new float[6 * N];
        for (int i = 0; i < N; i++) {
            src[i] = 0.5f;
            rc[i]  = 30.0f;
            rt[i]  = 3.0f;
            for (int d = 0; d < 6; d++) cond[d * N + i] = 1.0f;
        }
        float sigmaMax = NativePFSFBridge.nativeNormalizeSoA6(src, rc, rt, cond, null, N);
        check("L1.3", "normalizeSoA6 sigmaMax > 0", sigmaMax > 0, "got: " + sigmaMax);
        System.out.println("       sigmaMax=" + sigmaMax);
    }

    // ════════════════════════════════════════════════════════════════
    //  L2 — Vulkan 初始化
    // ════════════════════════════════════════════════════════════════
    static void testL2_VulkanInit() {
        header("L2", "Vulkan 初始化");

        handle = NativePFSFBridge.nativeCreate(2048, 10, 64L * 1024 * 1024, false, true);
        if (handle == 0) {
            System.out.println("  [SKIP] nativeCreate 回傳 0 (無 Vulkan device)");
            throw new TestFailException("L2 SKIP — 無 Vulkan device");
        }
        pass("L2.1", "nativeCreate handle=" + Long.toHexString(handle));

        int initRes = NativePFSFBridge.nativeInit(handle);
        if (initRes != OK) {
            fail("L2.2", "nativeInit 失敗: " + initRes);
            NativePFSFBridge.nativeDestroy(handle);
            handle = 0;
            throw new TestFailException("L2 FAIL — Vulkan init 失敗");
        }
        pass("L2.2", "nativeInit OK");

        check("L2.3", "nativeIsAvailable == true",
              NativePFSFBridge.nativeIsAvailable(handle), "回傳 false");

        long[] stats = NativePFSFBridge.nativeGetStats(handle);
        check("L2.4", "nativeGetStats 非空 && vramBudget>0",
              stats != null && stats.length >= 4 && stats[3] > 0,
              stats == null ? "null" : "vramBudget=" + (stats.length >= 4 ? stats[3] : "?"));
        if (stats != null)
            System.out.printf("       islands=%d  voxels=%d  vramUsed=%d  vramBudget=%d%n",
                stats[0], stats[1], stats[2], stats[3]);
    }

    // ════════════════════════════════════════════════════════════════
    //  L3 — Island buffer 生命週期
    // ════════════════════════════════════════════════════════════════
    static void testL3_IslandBuffers() {
        header("L3", "Island Buffer 生命週期");
        if (handle == 0) { System.out.println("  [SKIP] L2 未通過"); return; }

        int r = NativePFSFBridge.nativeAddIsland(handle, 99, 0, 0, 0, LX, LY, LZ);
        check("L3.1", "nativeAddIsland OK", r == OK, "result=" + r);

        // 分配暫用緩衝區
        IslandBuffers bufs = allocBuffers(1.0f);
        r = NativePFSFBridge.nativeRegisterIslandBuffers(handle, 99,
            bufs.phi, bufs.source, bufs.cond, bufs.type,
            bufs.rcomp, bufs.rtens, bufs.maxPhi);
        check("L3.2", "nativeRegisterIslandBuffers OK", r == OK, "result=" + r);

        r = NativePFSFBridge.nativeRegisterIslandLookups(handle, 99,
            bufs.matId, bufs.anchor, bufs.fluid, bufs.curing);
        check("L3.3", "nativeRegisterIslandLookups OK", r == OK, "result=" + r);

        r = NativePFSFBridge.nativeRegisterStressReadback(handle, 99, bufs.phi);
        check("L3.4", "nativeRegisterStressReadback OK", r == OK, "result=" + r);

        long[] stats = NativePFSFBridge.nativeGetStats(handle);
        check("L3.5", "stats islandCount==1",
              stats != null && stats[0] == 1, "got=" + (stats == null ? "null" : stats[0]));

        ByteBuffer sparse = NativePFSFBridge.nativeGetSparseUploadBuffer(handle, 99);
        check("L3.6", "getSparseUploadBuffer 非空 isDirect capacity>=24576",
              sparse != null && sparse.isDirect() && sparse.capacity() >= 24576,
              sparse == null ? "null" : "cap=" + sparse.capacity());

        r = NativePFSFBridge.nativeNotifySparseUpdates(handle, 99, 0);
        check("L3.7", "notifySparseUpdates(0) OK", r == OK, "result=" + r);

        NativePFSFBridge.nativeRemoveIsland(handle, 99);
        stats = NativePFSFBridge.nativeGetStats(handle);
        check("L3.8", "removeIsland → islandCount==0",
              stats != null && stats[0] == 0, "got=" + (stats == null ? "null" : stats[0]));
    }

    // ════════════════════════════════════════════════════════════════
    //  L4 — 單 tick，驗證 phi 場非零
    // ════════════════════════════════════════════════════════════════
    static void testL4_SingleTick() {
        header("L4", "單 tick — phi 場非零");
        if (handle == 0) { System.out.println("  [SKIP] L2 未通過"); return; }

        IslandBuffers bufs = allocBuffers(1.0f);
        fillCantilever(bufs, 1.0f);   // 一般強度懸臂

        int r;
        r = NativePFSFBridge.nativeAddIsland(handle, 1, 0, 0, 0, LX, LY, LZ);
        check("L4.0", "addIsland", r == OK, "r=" + r);
        r = NativePFSFBridge.nativeRegisterIslandBuffers(handle, 1,
            bufs.phi, bufs.source, bufs.cond, bufs.type,
            bufs.rcomp, bufs.rtens, bufs.maxPhi);
        check("L4.0b", "registerBuffers", r == OK, "r=" + r);
        NativePFSFBridge.nativeRegisterIslandLookups(handle, 1,
            bufs.matId, bufs.anchor, bufs.fluid, bufs.curing);
        // stress readback destination = same phi buffer; tick writes computed phi back here
        NativePFSFBridge.nativeRegisterStressReadback(handle, 1, bufs.phi);

        ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + 4 * 200 * 4)
                                       .order(ByteOrder.nativeOrder());
        failBuf.putInt(0, 0);  // count = 0

        long t0 = System.currentTimeMillis();
        r = NativePFSFBridge.nativeTickDbb(handle, new int[]{1}, 1L, failBuf);
        long elapsed = System.currentTimeMillis() - t0;
        check("L4.1", "nativeTickDbb OK time<500ms",
              r == OK && elapsed < 500, "result=" + r + " time=" + elapsed + "ms");
        System.out.println("       tick 耗時: " + elapsed + " ms");

        // 讀回 phi：至少一個非零
        bufs.phi.order(ByteOrder.nativeOrder());
        float maxPhi = 0;
        int nonZero = 0;
        for (int i = 0; i < N; i++) {
            float v = bufs.phi.getFloat(i * 4);
            if (Math.abs(v) > 1e-10f) nonZero++;
            if (v > maxPhi) maxPhi = v;
        }
        check("L4.2", "phi 場有非零值",
              nonZero > 0, "全部為零 (nonZero=" + nonZero + ")");
        System.out.println("       phi 非零數=" + nonZero + "/" + N
                         + "  maxPhi=" + maxPhi);

        // 5 ticks — 觀察 phi 是否在收斂（不發散）
        float[] phiSamples = new float[5];
        for (int k = 0; k < 5; k++) {
            NativePFSFBridge.nativeTickDbb(handle, new int[]{1}, 2L + k, failBuf);
            float pkMax = 0;
            for (int i = 0; i < N; i++) {
                float v = bufs.phi.getFloat(i * 4);
                if (Float.isNaN(v) || Float.isInfinite(v)) {
                    fail("L4.3", "phi 發散 NaN/Inf at i=" + i + " tick=" + (k + 2));
                    return;
                }
                if (v > pkMax) pkMax = v;
            }
            phiSamples[k] = pkMax;
        }
        pass("L4.3", "5 ticks phi 無 NaN/Inf");
        System.out.println("       maxPhi after 5 more ticks = " + phiSamples[4]);
    }

    // ════════════════════════════════════════════════════════════════
    //  L5 — 失效偵測：弱懸臂 → failure event
    // ════════════════════════════════════════════════════════════════
    static void testL5_FailureDetection() {
        header("L5", "失效偵測 (弱懸臂 rcomp=0.001)");
        if (handle == 0) { System.out.println("  [SKIP] L2 未通過"); return; }

        // 先移掉 L4 的 island 1
        NativePFSFBridge.nativeRemoveIsland(handle, 1);

        IslandBuffers bufs = allocBuffers(1.0f);
        fillCantilever(bufs, 0.001f);   // 極低強度 → 必然失效

        NativePFSFBridge.nativeAddIsland(handle, 2, 0, 0, 0, LX, LY, LZ);
        NativePFSFBridge.nativeRegisterIslandBuffers(handle, 2,
            bufs.phi, bufs.source, bufs.cond, bufs.type,
            bufs.rcomp, bufs.rtens, bufs.maxPhi);
        NativePFSFBridge.nativeRegisterIslandLookups(handle, 2,
            bufs.matId, bufs.anchor, bufs.fluid, bufs.curing);
        NativePFSFBridge.nativeRegisterStressReadback(handle, 2, bufs.phi);

        // failure buffer: int header + 200 tuples * 4 ints
        ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + 200 * 4 * 4)
                                       .order(ByteOrder.nativeOrder());

        int totalFailures = 0;
        int firstFailTick = -1;

        for (int tick = 1; tick <= 32 && totalFailures == 0; tick++) {
            failBuf.putInt(0, 0);  // 重置計數
            int r = NativePFSFBridge.nativeTickDbb(handle, new int[]{2}, tick, failBuf);
            if (r != OK) {
                fail("L5.0", "tickDbb 非 OK at tick=" + tick + " r=" + r);
                return;
            }
            int count = failBuf.getInt(0);
            if (count > 0) {
                totalFailures += count;
                firstFailTick = tick;
            }
        }

        check("L5.1", "至少產生 1 個 failure event",
              totalFailures > 0,
              "跑了 32 ticks 沒有任何 failure (phi solver 可能未通過 rcomp 閾值)");

        if (totalFailures > 0) {
            int x = failBuf.getInt(4 * 1);
            int y = failBuf.getInt(4 * 2);
            int z = failBuf.getInt(4 * 3);
            int type = failBuf.getInt(4 * 4);
            System.out.println("       第一個 failure: pos=(" + x + "," + y + "," + z + ")"
                             + "  type=" + type
                             + "  tick=" + firstFailTick);

            // failure 位置應在島內 (0..3)
            check("L5.2", "failure 座標在 island 範圍內",
                  x >= 0 && x < LX && y >= 0 && y < LY && z >= 0 && z < LZ,
                  "pos=(" + x + "," + y + "," + z + ")");

            // failure type 為 CANTILEVER(1) 或 CRUSHING(2)
            check("L5.3", "failure type == CANTILEVER(1) 或 CRUSHING(2)",
                  type == 1 || type == 2, "type=" + type);
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  L6 — 30-tick 收斂穩定性
    // ════════════════════════════════════════════════════════════════
    static void testL6_PhiConvergence() {
        header("L6", "30-tick phi 收斂穩定性");
        if (handle == 0) { System.out.println("  [SKIP] L2 未通過"); return; }

        NativePFSFBridge.nativeRemoveIsland(handle, 2);

        IslandBuffers bufs = allocBuffers(1.0f);
        fillCantilever(bufs, 1.0f);
        NativePFSFBridge.nativeAddIsland(handle, 3, 0, 0, 0, LX, LY, LZ);
        NativePFSFBridge.nativeRegisterIslandBuffers(handle, 3,
            bufs.phi, bufs.source, bufs.cond, bufs.type,
            bufs.rcomp, bufs.rtens, bufs.maxPhi);
        NativePFSFBridge.nativeRegisterIslandLookups(handle, 3,
            bufs.matId, bufs.anchor, bufs.fluid, bufs.curing);
        NativePFSFBridge.nativeRegisterStressReadback(handle, 3, bufs.phi);

        ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + 256 * 16)
                                       .order(ByteOrder.nativeOrder());

        float[] maxPhiHistory = new float[30];
        for (int tick = 0; tick < 30; tick++) {
            failBuf.putInt(0, 0);
            NativePFSFBridge.nativeTickDbb(handle, new int[]{3}, tick + 100L, failBuf);
            float mx = 0;
            for (int i = 0; i < N; i++) {
                float v = bufs.phi.getFloat(i * 4);
                if (Float.isNaN(v) || Float.isInfinite(v)) {
                    fail("L6.1", "NaN/Inf at i=" + i + " tick=" + (tick+1));
                    return;
                }
                if (v > mx) mx = v;
            }
            maxPhiHistory[tick] = mx;
        }
        pass("L6.1", "30 ticks 無 NaN/Inf");

        // 最後 5 tick 的 maxPhi 變化幅度 < 5%（收斂判定）
        float last5Max = 0, last5Min = Float.MAX_VALUE;
        for (int i = 25; i < 30; i++) {
            if (maxPhiHistory[i] > last5Max) last5Max = maxPhiHistory[i];
            if (maxPhiHistory[i] < last5Min) last5Min = maxPhiHistory[i];
        }
        float variation = (last5Max > 0) ? (last5Max - last5Min) / last5Max : 0;
        check("L6.2", "最後 5 tick maxPhi 變化 < 5%",
              variation < 0.05f, String.format("variation=%.3f maxPhi=%.4f", variation, last5Max));
        System.out.printf("       tick[0]=%.4f  tick[14]=%.4f  tick[29]=%.4f  conv=%.3f%n",
            maxPhiHistory[0], maxPhiHistory[14], maxPhiHistory[29], variation);
    }

    // ════════════════════════════════════════════════════════════════
    //  L7 — 多島批次 tick
    // ════════════════════════════════════════════════════════════════
    static void testL7_MultiIslandBatch() {
        header("L7", "多島批次 tick（3 個島同時）");
        if (handle == 0) { System.out.println("  [SKIP] L2 未通過"); return; }

        NativePFSFBridge.nativeRemoveIsland(handle, 3);

        IslandBuffers[] bufsArr = new IslandBuffers[3];
        int[] ids = {10, 11, 12};
        for (int k = 0; k < 3; k++) {
            bufsArr[k] = allocBuffers(1.0f);
            fillCantilever(bufsArr[k], 1.0f);
            int r = NativePFSFBridge.nativeAddIsland(handle, ids[k],
                k * 10, 0, 0, LX, LY, LZ);
            check("L7.0." + k, "addIsland " + ids[k], r == OK, "r=" + r);
            NativePFSFBridge.nativeRegisterIslandBuffers(handle, ids[k],
                bufsArr[k].phi, bufsArr[k].source, bufsArr[k].cond, bufsArr[k].type,
                bufsArr[k].rcomp, bufsArr[k].rtens, bufsArr[k].maxPhi);
            NativePFSFBridge.nativeRegisterIslandLookups(handle, ids[k],
                bufsArr[k].matId, bufsArr[k].anchor, bufsArr[k].fluid, bufsArr[k].curing);
            NativePFSFBridge.nativeRegisterStressReadback(handle, ids[k], bufsArr[k].phi);
        }

        ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + 256 * 16)
                                       .order(ByteOrder.nativeOrder());
        failBuf.putInt(0, 0);
        int r = NativePFSFBridge.nativeTickDbb(handle, ids, 200L, failBuf);
        check("L7.1", "批次 tick OK", r == OK, "r=" + r);

        int nonZeroIslands = 0;
        for (int k = 0; k < 3; k++) {
            int nz = 0;
            for (int i = 0; i < N; i++) {
                if (Math.abs(bufsArr[k].phi.getFloat(i * 4)) > 1e-10f) nz++;
            }
            if (nz > 0) nonZeroIslands++;
            System.out.printf("       island %d: nonZeroPhi=%d/%d%n", ids[k], nz, N);
        }
        // Tick budget (10ms) limits per-tick processing — run extra ticks until
        // all islands have been solved (each skipped island stays dirty)
        if (nonZeroIslands < 3) {
            for (int extra = 0; extra < 10 && nonZeroIslands < 3; extra++) {
                failBuf.putInt(0, 0);
                NativePFSFBridge.nativeTickDbb(handle, ids, 201L + extra, failBuf);
                nonZeroIslands = 0;
                for (int k = 0; k < 3; k++) {
                    int nz2 = 0;
                    for (int i = 0; i < N; i++)
                        if (Math.abs(bufsArr[k].phi.getFloat(i * 4)) > 1e-10f) nz2++;
                    if (nz2 > 0) nonZeroIslands++;
                }
            }
        }
        check("L7.2", "3 個島在 ≤10 個 tick 內都有非零 phi", nonZeroIslands == 3,
              "只有 " + nonZeroIslands + "/3 個島有 phi");
    }

    // ════════════════════════════════════════════════════════════════
    //  L8 — 移除中的島嶼不崩潰
    // ════════════════════════════════════════════════════════════════
    static void testL8_IslandRemoveDuringSolve() {
        header("L8", "移除後再 tick 不崩潰");
        if (handle == 0) { System.out.println("  [SKIP] L2 未通過"); return; }

        // 移除 island 10
        NativePFSFBridge.nativeRemoveIsland(handle, 10);
        // 繼續 tick 其他兩個島，不包含 10
        ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + 64 * 16)
                                       .order(ByteOrder.nativeOrder());
        failBuf.putInt(0, 0);
        int r = NativePFSFBridge.nativeTickDbb(handle, new int[]{11, 12}, 201L, failBuf);
        check("L8.1", "移除 island 10 後 tick [11,12] OK", r == OK, "r=" + r);

        // tick 已移除的 island 10
        failBuf.putInt(0, 0);
        r = NativePFSFBridge.nativeTickDbb(handle, new int[]{10}, 202L, failBuf);
        check("L8.2", "tick 已移除 island → OK 或非 crash (stale 直接忽略)",
              r == OK || r == -2 /* INVALID_ARG */, "r=" + r);
        System.out.println("       tick stale island 10 result=" + r + " (expected 0 or -2)");

        // 清理
        NativePFSFBridge.nativeRemoveIsland(handle, 11);
        NativePFSFBridge.nativeRemoveIsland(handle, 12);
    }

    // ════════════════════════════════════════════════════════════════
    //  L9 — 大型島嶼 (8×8×8 = 512 voxels)
    // ════════════════════════════════════════════════════════════════
    static final int BIG = 8;
    static final int BIG_N = BIG * BIG * BIG;

    static void testL9_LargeIsland() {
        header("L9", "大型島嶼 8×8×8=512 voxels");
        if (handle == 0) { System.out.println("  [SKIP] L2 未通過"); return; }

        IslandBuffers b = allocBigBuffers(1.0f);
        fillBigCantilever(b, 1.0f);

        int r = NativePFSFBridge.nativeAddIsland(handle, 20, 0, 0, 0, BIG, BIG, BIG);
        check("L9.1", "addIsland 8×8×8 OK", r == OK, "r=" + r);
        r = NativePFSFBridge.nativeRegisterIslandBuffers(handle, 20,
            b.phi, b.source, b.cond, b.type,
            b.rcomp, b.rtens, b.maxPhi);
        check("L9.2", "registerBuffers OK", r == OK, "r=" + r);
        NativePFSFBridge.nativeRegisterIslandLookups(handle, 20,
            b.matId, b.anchor, b.fluid, b.curing);
        NativePFSFBridge.nativeRegisterStressReadback(handle, 20, b.phi);

        ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + 512 * 16)
                                       .order(ByteOrder.nativeOrder());
        failBuf.putInt(0, 0);
        long t0 = System.currentTimeMillis();
        r = NativePFSFBridge.nativeTickDbb(handle, new int[]{20}, 300L, failBuf);
        long ms = System.currentTimeMillis() - t0;
        check("L9.3", "tick OK time<2000ms", r == OK && ms < 2000, "r="+r+" time="+ms+"ms");

        int nz = 0;
        float mx = 0;
        for (int i = 0; i < BIG_N; i++) {
            float v = b.phi.getFloat(i * 4);
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                fail("L9.4", "NaN/Inf at i=" + i); return;
            }
            if (Math.abs(v) > 1e-10f) nz++;
            if (v > mx) mx = v;
        }
        check("L9.4", "8×8×8 phi 非零", nz > 0, "全零");
        System.out.printf("       tick=%dms  nonZero=%d/%d  maxPhi=%.4f%n",
            ms, nz, BIG_N, mx);

        NativePFSFBridge.nativeRemoveIsland(handle, 20);
    }

    // ════════════════════════════════════════════════════════════════
    //  L10 — 各向異性導率：只有 Y 方向 → phi 沿 X/Z 不變
    // ════════════════════════════════════════════════════════════════
    static void testL10_AnisotropicConductivity() {
        header("L10", "SoA6 各向異性導率驗證（只有 Y 方向）");
        if (handle == 0) { System.out.println("  [SKIP] L2 未通過"); return; }

        IslandBuffers b = allocBuffers(1.0f);
        // 只填 Y 方向（d=2 NEG_Y, d=3 POS_Y），X 和 Z 的導率為 0
        for (int gz = 0; gz < LZ; gz++) {
            for (int gy = 0; gy < LY; gy++) {
                for (int gx = 0; gx < LX; gx++) {
                    int i = gx + LX * (gy + LY * gz);
                    if (gy == 0) {
                        b.type.put(i, (byte) VTYPE_ANCHOR);
                        b.anchor.putLong(i * 8, 0xFFFFFFFFFFFFFFFFL);
                    } else {
                        b.type.put(i, (byte) VTYPE_SOLID);
                        b.source.putFloat(i * 4, 1.0f);
                        b.cond.putFloat((2 * N + i) * 4, 1.0f); // NEG_Y
                        b.cond.putFloat((3 * N + i) * 4, 1.0f); // POS_Y
                        // X, Z 方向全部保持 0
                        b.rcomp.putFloat(i * 4, 100.0f);  // 高強度 → 不失效
                        b.rtens.putFloat(i * 4, 100.0f);
                        b.maxPhi.putFloat(i * 4, 100.0f);
                    }
                    b.curing.putFloat(i * 4, 1.0f);
                }
            }
        }

        NativePFSFBridge.nativeAddIsland(handle, 30, 0, 0, 0, LX, LY, LZ);
        NativePFSFBridge.nativeRegisterIslandBuffers(handle, 30,
            b.phi, b.source, b.cond, b.type,
            b.rcomp, b.rtens, b.maxPhi);
        NativePFSFBridge.nativeRegisterIslandLookups(handle, 30,
            b.matId, b.anchor, b.fluid, b.curing);
        NativePFSFBridge.nativeRegisterStressReadback(handle, 30, b.phi);

        ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + 64 * 16)
                                       .order(ByteOrder.nativeOrder());
        // 執行 10 ticks 讓場收斂
        for (int tick = 0; tick < 10; tick++) {
            failBuf.putInt(0, 0);
            NativePFSFBridge.nativeTickDbb(handle, new int[]{30}, 400L + tick, failBuf);
        }

        // 對每個 y，讀取 phi(0,y,0), phi(1,y,0), phi(2,y,0), phi(3,y,0)
        // 若 SoA6 方向正確，這些值應相等（1D diffusion，與 x 無關）
        float maxXVariation = 0;
        float maxAbsPhi = 0;
        for (int gy = 1; gy < LY; gy++) {
            float refPhi = b.phi.getFloat((0 + LX * (gy + LY * 0)) * 4);
            for (int gx = 1; gx < LX; gx++) {
                int i = gx + LX * (gy + LY * 0);
                float v = b.phi.getFloat(i * 4);
                float diff = Math.abs(v - refPhi);
                if (diff > maxXVariation) maxXVariation = diff;
                if (Math.abs(v) > maxAbsPhi) maxAbsPhi = Math.abs(v);
            }
        }
        float relVariation = (maxAbsPhi > 0) ? maxXVariation / maxAbsPhi : 0;
        check("L10.1", "Y-only 導率 → phi 沿 X 方向不變 (variation < 1%)",
              relVariation < 0.01f,
              String.format("relVariation=%.4f maxPhi=%.4f", relVariation, maxAbsPhi));
        System.out.printf("       maxAbsPhi=%.4f  maxXVariation=%.6f  relVar=%.4f%n",
            maxAbsPhi, maxXVariation, relVariation);

        NativePFSFBridge.nativeRemoveIsland(handle, 30);
    }

    // ════════════════════════════════════════════════════════════════
    //  L11 — 失效類型分辨
    // ════════════════════════════════════════════════════════════════
    static void testL11_FailureTypeDiscrimination() {
        header("L11", "失效類型分辨 CANTILEVER/CRUSHING/TENSION");
        if (handle == 0) { System.out.println("  [SKIP] L2 未通過"); return; }

        // ── 11A CANTILEVER: phi > maxPhi → type=1 ──
        {
            IslandBuffers b = allocBuffers(1.0f);
            fillCantilever(b, 0.001f); // 極低 maxPhi → 必 CANTILEVER
            NativePFSFBridge.nativeAddIsland(handle, 41, 0, 0, 0, LX, LY, LZ);
            NativePFSFBridge.nativeRegisterIslandBuffers(handle, 41,
                b.phi, b.source, b.cond, b.type,
                b.rcomp, b.rtens, b.maxPhi);
            NativePFSFBridge.nativeRegisterIslandLookups(handle, 41,
                b.matId, b.anchor, b.fluid, b.curing);
            NativePFSFBridge.nativeRegisterStressReadback(handle, 41, b.phi);

            ByteBuffer fb = ByteBuffer.allocateDirect(4 + 200 * 16)
                                      .order(ByteOrder.nativeOrder());
            int cantTick = -1;
            boolean hasCant = false;
            for (int t = 1; t <= 10 && !hasCant; t++) {
                fb.putInt(0, 0);
                NativePFSFBridge.nativeTickDbb(handle, new int[]{41}, 500L + t, fb);
                int cnt = fb.getInt(0);
                for (int ei = 0; ei < cnt; ei++) {
                    int type = fb.getInt(4 + ei * 16 + 12);
                    if (type == 1) { hasCant = true; cantTick = t; break; }
                }
            }
            check("L11.1", "CANTILEVER(1) 失效被偵測", hasCant, "未觸發");
            System.out.println("       CANTILEVER tick=" + cantTick);
            NativePFSFBridge.nativeRemoveIsland(handle, 41);
        }

        // ── 11B CRUSHING: flux_in > rcomp → type=2 ──
        // 製造高壓縮：中間層有大量 source（重力），rcomp 極低
        {
            IslandBuffers b = allocBuffers(1.0f);
            for (int gz = 0; gz < LZ; gz++) {
                for (int gy = 0; gy < LY; gy++) {
                    for (int gx = 0; gx < LX; gx++) {
                        int i = gx + LX * (gy + LY * gz);
                        b.curing.putFloat(i * 4, 1.0f);
                        if (gy == 0) {
                            b.type.put(i, (byte) VTYPE_ANCHOR);
                            b.anchor.putLong(i * 8, 0xFFFFFFFFFFFFFFFFL);
                        } else {
                            b.type.put(i, (byte) VTYPE_SOLID);
                            b.source.putFloat(i * 4, 1000.0f); // 超大源項
                            for (int d = 0; d < 6; d++) b.cond.putFloat((d*N+i)*4, 1.0f);
                            b.rcomp.putFloat(i * 4, 0.001f); // 極低壓縮強度
                            b.rtens.putFloat(i * 4, 0.001f);
                            b.maxPhi.putFloat(i * 4, 1e6f);  // 高 maxPhi → 不走 CANTILEVER 路徑
                        }
                    }
                }
            }
            NativePFSFBridge.nativeAddIsland(handle, 42, 0, 0, 0, LX, LY, LZ);
            NativePFSFBridge.nativeRegisterIslandBuffers(handle, 42,
                b.phi, b.source, b.cond, b.type,
                b.rcomp, b.rtens, b.maxPhi);
            NativePFSFBridge.nativeRegisterIslandLookups(handle, 42,
                b.matId, b.anchor, b.fluid, b.curing);
            NativePFSFBridge.nativeRegisterStressReadback(handle, 42, b.phi);

            ByteBuffer fb = ByteBuffer.allocateDirect(4 + 200 * 16)
                                      .order(ByteOrder.nativeOrder());
            boolean hasCrush = false, hasTension = false;
            for (int t = 1; t <= 10 && !(hasCrush && hasTension); t++) {
                fb.putInt(0, 0);
                NativePFSFBridge.nativeTickDbb(handle, new int[]{42}, 600L + t, fb);
                int cnt = fb.getInt(0);
                for (int ei = 0; ei < cnt; ei++) {
                    int type = fb.getInt(4 + ei * 16 + 12);
                    if (type == 2) hasCrush = true;
                    if (type == 4) hasTension = true;
                }
            }
            check("L11.2", "CRUSHING(2) 失效被偵測", hasCrush, "未觸發");
            System.out.println("       CRUSHING=" + hasCrush + "  TENSION=" + hasTension);
            NativePFSFBridge.nativeRemoveIsland(handle, 42);
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  L12 — 零源項島嶼：phi 應保持在 0 附近
    // ════════════════════════════════════════════════════════════════
    static void testL12_SourceFreeIsland() {
        header("L12", "零源項島嶼 phi ≈ 0");
        if (handle == 0) { System.out.println("  [SKIP] L2 未通過"); return; }

        IslandBuffers b = allocBuffers(1.0f);
        // 設定結構：y=0 anchor，y=1..3 solid，但 source=0
        for (int gz = 0; gz < LZ; gz++) {
            for (int gy = 0; gy < LY; gy++) {
                for (int gx = 0; gx < LX; gx++) {
                    int i = gx + LX * (gy + LY * gz);
                    b.curing.putFloat(i * 4, 1.0f);
                    if (gy == 0) {
                        b.type.put(i, (byte) VTYPE_ANCHOR);
                        b.anchor.putLong(i * 8, 0xFFFFFFFFFFFFFFFFL);
                    } else {
                        b.type.put(i, (byte) VTYPE_SOLID);
                        // source = 0 (預設為 0，allocBuffers 已清零)
                        for (int d = 0; d < 6; d++)
                            b.cond.putFloat((d * N + i) * 4, 1.0f);
                        b.rcomp.putFloat(i * 4, 100.0f);
                        b.rtens.putFloat(i * 4, 100.0f);
                        b.maxPhi.putFloat(i * 4, 100.0f);
                    }
                }
            }
        }

        NativePFSFBridge.nativeAddIsland(handle, 50, 0, 0, 0, LX, LY, LZ);
        NativePFSFBridge.nativeRegisterIslandBuffers(handle, 50,
            b.phi, b.source, b.cond, b.type, b.rcomp, b.rtens, b.maxPhi);
        NativePFSFBridge.nativeRegisterIslandLookups(handle, 50,
            b.matId, b.anchor, b.fluid, b.curing);
        NativePFSFBridge.nativeRegisterStressReadback(handle, 50, b.phi);

        ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + 64 * 16)
                                       .order(ByteOrder.nativeOrder());
        for (int tick = 0; tick < 5; tick++) {
            failBuf.putInt(0, 0);
            NativePFSFBridge.nativeTickDbb(handle, new int[]{50}, 700L + tick, failBuf);
        }

        float maxAbsPhi = 0;
        for (int i = 0; i < N; i++) {
            float v = b.phi.getFloat(i * 4);
            if (Float.isNaN(v) || Float.isInfinite(v)) {
                fail("L12.1", "NaN/Inf at i=" + i); return;
            }
            if (Math.abs(v) > maxAbsPhi) maxAbsPhi = Math.abs(v);
        }
        // 零源項 → phi 應無限接近 0（Ax=0 的唯一解是 x=0，錨點 BC 強制 phi=0）
        check("L12.1", "零源項 → maxAbsPhi < 1e-3",
              maxAbsPhi < 1e-3f, String.format("maxAbsPhi=%.6f", maxAbsPhi));
        System.out.printf("       maxAbsPhi=%.8f%n", maxAbsPhi);
        NativePFSFBridge.nativeRemoveIsland(handle, 50);
    }

    // ════════════════════════════════════════════════════════════════
    //  L13 — Warm-start epoch 一致性：phi 在多 tick 後保持有效
    //         （模擬遊戲內 PFSF 20-tick 暖啟動行為）
    // ════════════════════════════════════════════════════════════════
    static void testL13_WarmStartEpochConsistency() {
        header("L13", "Warm-start 20 ticks phi 保持有效且遞增");
        if (handle == 0) { System.out.println("  [SKIP] L2 未通過"); return; }

        IslandBuffers b = allocBuffers(1.0f);
        fillCantilever(b, 1.0f);

        NativePFSFBridge.nativeAddIsland(handle, 60, 0, 0, 0, LX, LY, LZ);
        NativePFSFBridge.nativeRegisterIslandBuffers(handle, 60,
            b.phi, b.source, b.cond, b.type, b.rcomp, b.rtens, b.maxPhi);
        NativePFSFBridge.nativeRegisterIslandLookups(handle, 60,
            b.matId, b.anchor, b.fluid, b.curing);
        NativePFSFBridge.nativeRegisterStressReadback(handle, 60, b.phi);

        ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + 200 * 16)
                                       .order(ByteOrder.nativeOrder());

        float firstPhi = 0, lastPhi = 0;
        boolean anyNaN = false;
        int nanTick = -1;

        for (int tick = 1; tick <= 20; tick++) {
            failBuf.putInt(0, 0);
            // 模擬遊戲 warm-start: 每 tick 呼叫 nativeMarkFullRebuild
            NativePFSFBridge.nativeTickDbb(handle, new int[]{60}, 800L + tick, failBuf);

            float mx = 0;
            for (int i = 0; i < N; i++) {
                float v = b.phi.getFloat(i * 4);
                if (Float.isNaN(v) || Float.isInfinite(v)) {
                    anyNaN = true; nanTick = tick; break;
                }
                if (v > mx) mx = v;
            }
            if (anyNaN) break;
            if (tick == 1) firstPhi = mx;
            lastPhi = mx;
        }

        check("L13.1", "20 ticks 無 NaN/Inf", !anyNaN,
              "tick " + nanTick + " 發散");
        // phi 在暖啟動後應保持收斂（lastPhi ≈ firstPhi，誤差 < 1%）
        float drift = (firstPhi > 0) ? Math.abs(lastPhi - firstPhi) / firstPhi : 0;
        check("L13.2", "20-tick phi 漂移 < 1%", drift < 0.01f,
              String.format("drift=%.4f first=%.4f last=%.4f", drift, firstPhi, lastPhi));
        System.out.printf("       tick1=%.4f  tick20=%.4f  drift=%.4f%n",
            firstPhi, lastPhi, drift);
        NativePFSFBridge.nativeRemoveIsland(handle, 60);
    }

    // ════════════════════════════════════════════════════════════════
    //  輔助：8×8×8 大型島嶼緩衝區
    // ════════════════════════════════════════════════════════════════
    static IslandBuffers allocBigBuffers(float ignored) {
        IslandBuffers b = new IslandBuffers();
        b.phi    = ByteBuffer.allocateDirect(BIG_N * 4).order(ByteOrder.nativeOrder());
        b.source = ByteBuffer.allocateDirect(BIG_N * 4).order(ByteOrder.nativeOrder());
        b.cond   = ByteBuffer.allocateDirect(6 * BIG_N * 4).order(ByteOrder.nativeOrder());
        b.type   = ByteBuffer.allocateDirect(BIG_N * 1).order(ByteOrder.nativeOrder());
        b.rcomp  = ByteBuffer.allocateDirect(BIG_N * 4).order(ByteOrder.nativeOrder());
        b.rtens  = ByteBuffer.allocateDirect(BIG_N * 4).order(ByteOrder.nativeOrder());
        b.maxPhi = ByteBuffer.allocateDirect(BIG_N * 4).order(ByteOrder.nativeOrder());
        b.matId  = ByteBuffer.allocateDirect(BIG_N * 4).order(ByteOrder.nativeOrder());
        b.anchor = ByteBuffer.allocateDirect(BIG_N * 8).order(ByteOrder.nativeOrder());
        b.fluid  = ByteBuffer.allocateDirect(BIG_N * 4).order(ByteOrder.nativeOrder());
        b.curing = ByteBuffer.allocateDirect(BIG_N * 4).order(ByteOrder.nativeOrder());
        for (int i = 0; i < BIG_N; i++) b.curing.putFloat(i * 4, 1.0f);
        return b;
    }

    static void fillBigCantilever(IslandBuffers b, float rcValue) {
        for (int gz = 0; gz < BIG; gz++) {
            for (int gy = 0; gy < BIG; gy++) {
                for (int gx = 0; gx < BIG; gx++) {
                    int i = gx + BIG * (gy + BIG * gz);
                    if (gy == 0) {
                        b.type.put(i, (byte) VTYPE_ANCHOR);
                        b.anchor.putLong(i * 8, 0xFFFFFFFFFFFFFFFFL);
                    } else {
                        b.type.put(i, (byte) VTYPE_SOLID);
                        b.source.putFloat(i * 4, 1.0f);
                        for (int d = 0; d < 6; d++)
                            b.cond.putFloat((d * BIG_N + i) * 4, 1.0f);
                        b.rcomp.putFloat(i * 4, rcValue);
                        b.rtens.putFloat(i * 4, rcValue * 0.1f);
                        b.maxPhi.putFloat(i * 4, rcValue);
                    }
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  輔助：建立懸臂結構緩衝區
    // ════════════════════════════════════════════════════════════════

    static class IslandBuffers {
        ByteBuffer phi, source, cond, type, rcomp, rtens, maxPhi;
        ByteBuffer matId, anchor, fluid, curing;
    }

    /** 分配所有必要的 DirectByteBuffer，尺寸符合 NxFloat 規格。 */
    static IslandBuffers allocBuffers(float sigmaMax) {
        IslandBuffers b = new IslandBuffers();
        b.phi    = ByteBuffer.allocateDirect(N * 4).order(ByteOrder.nativeOrder());
        b.source = ByteBuffer.allocateDirect(N * 4).order(ByteOrder.nativeOrder());
        b.cond   = ByteBuffer.allocateDirect(6 * N * 4).order(ByteOrder.nativeOrder());
        b.type   = ByteBuffer.allocateDirect(N * 1).order(ByteOrder.nativeOrder());
        b.rcomp  = ByteBuffer.allocateDirect(N * 4).order(ByteOrder.nativeOrder());
        b.rtens  = ByteBuffer.allocateDirect(N * 4).order(ByteOrder.nativeOrder());
        b.maxPhi = ByteBuffer.allocateDirect(N * 4).order(ByteOrder.nativeOrder());
        b.matId  = ByteBuffer.allocateDirect(N * 4).order(ByteOrder.nativeOrder());
        b.anchor = ByteBuffer.allocateDirect(N * 8).order(ByteOrder.nativeOrder()); // int64 per voxel
        b.fluid  = ByteBuffer.allocateDirect(N * 4).order(ByteOrder.nativeOrder());
        b.curing = ByteBuffer.allocateDirect(N * 4).order(ByteOrder.nativeOrder());
        // 初始化 curing 為 1.0 (fully cured)
        for (int i = 0; i < N; i++) b.curing.putFloat(i * 4, 1.0f);
        return b;
    }

    /**
     * 填充 4×4×4 懸臂結構。
     *
     * y=0 底層全部為錨點 (VTYPE_ANCHOR)。
     * y=1..3 為實心 (VTYPE_SOLID)。
     * conductivity 各向同性 1.0（正規化後）。
     * source = gravity 驅動 = 1.0/sigmaMax per solid voxel。
     * rcomp = rcValue (已正規化，調用方傳入)。
     *
     * @param rcValue 壓縮強度（未正規化時可傳 sigmaMax=1.0 的相對值）。
     */
    static void fillCantilever(IslandBuffers b, float rcValue) {
        final float SIGMA    = 1.0f;  // 正規化後的導率
        final float SOURCE_G = 1.0f;  // 重力驅動源項（正規化後）
        final float RTENS    = rcValue * 0.1f;

        // 先清零
        b.phi.clear();
        b.source.clear();
        b.cond.clear();
        b.type.clear();
        b.rcomp.clear();
        b.rtens.clear();
        b.maxPhi.clear();
        b.anchor.clear();
        b.matId.clear();

        for (int gz = 0; gz < LZ; gz++) {
            for (int gy = 0; gy < LY; gy++) {
                for (int gx = 0; gx < LX; gx++) {
                    int i = gx + LX * (gy + LY * gz);

                    if (gy == 0) {
                        // 底層錨點
                        b.type.put(i, (byte) VTYPE_ANCHOR);
                        // anchor bitmap：全方向錨定
                        b.anchor.putLong(i * 8, 0xFFFFFFFFFFFFFFFFL);
                        // 不設導率/源項（錨點無物理計算）
                    } else {
                        // 實心
                        b.type.put(i, (byte) VTYPE_SOLID);
                        b.source.putFloat(i * 4, SOURCE_G);
                        // 各向同性導率 SoA: sigma[dir*N + i]
                        for (int d = 0; d < 6; d++) {
                            b.cond.putFloat((d * N + i) * 4, SIGMA);
                        }
                        b.rcomp.putFloat(i * 4, rcValue);
                        b.rtens.putFloat(i * 4, RTENS);
                        b.maxPhi.putFloat(i * 4, rcValue);  // maxPhi = rcomp (cantilever threshold)
                    }
                }
            }
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  L14 — 移除後重新註冊（相同 ID，不同 AABB）
    // ════════════════════════════════════════════════════════════════
    static void testL14_ReregistrationAfterRemove() {
        header("L14", "移除後重新註冊（相同 ID，不同 AABB）");

        final int ID = 14;

        // 初次：4×4×4
        int rc = NativePFSFBridge.nativeAddIsland(handle, ID, 0, 0, 0, LX, LY, LZ);
        check("L14.0", "首次 nativeAddIsland 4×4×4", rc == OK, "rc=" + rc);

        IslandBuffers b1 = allocBuffers(1.0f);
        fillCantilever(b1, 5.0f);
        rc = NativePFSFBridge.nativeRegisterIslandBuffers(handle, ID,
                b1.phi, b1.source, b1.cond, b1.type,
                b1.rcomp, b1.rtens, b1.maxPhi);
        check("L14.1", "首次 registerBuffers", rc == OK, "rc=" + rc);
        NativePFSFBridge.nativeRegisterIslandLookups(handle, ID,
                b1.matId, b1.anchor, b1.fluid, b1.curing);
        NativePFSFBridge.nativeRegisterStressReadback(handle, ID, b1.phi);

        int[] ids1 = {ID};
        ByteBuffer fb1 = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        fb1.putInt(0, 0);
        rc = NativePFSFBridge.nativeTickDbb(handle, ids1, 1001L, fb1);
        check("L14.2", "首次 tick OK", rc == OK, "rc=" + rc);

        // 移除
        NativePFSFBridge.nativeRemoveIsland(handle, ID);
        pass("L14.3", "nativeRemoveIsland 不崩潰");

        // 重新以不同尺寸（6×6×6 = 216 voxels）註冊相同 ID
        final int L2 = 6, N2 = L2 * L2 * L2;
        rc = NativePFSFBridge.nativeAddIsland(handle, ID, 0, 64, 0, L2, L2, L2);
        check("L14.4", "重新 nativeAddIsland 6×6×6", rc == OK, "rc=" + rc);

        ByteBuffer phi2    = ByteBuffer.allocateDirect(N2 * 4).order(ByteOrder.nativeOrder());
        ByteBuffer source2 = ByteBuffer.allocateDirect(N2 * 4).order(ByteOrder.nativeOrder());
        ByteBuffer cond2   = ByteBuffer.allocateDirect(6 * N2 * 4).order(ByteOrder.nativeOrder());
        ByteBuffer type2   = ByteBuffer.allocateDirect(N2).order(ByteOrder.nativeOrder());
        ByteBuffer rcomp2  = ByteBuffer.allocateDirect(N2 * 4).order(ByteOrder.nativeOrder());
        ByteBuffer rtens2  = ByteBuffer.allocateDirect(N2 * 4).order(ByteOrder.nativeOrder());
        ByteBuffer maxPhi2 = ByteBuffer.allocateDirect(N2 * 4).order(ByteOrder.nativeOrder());
        ByteBuffer matId2  = ByteBuffer.allocateDirect(N2 * 4).order(ByteOrder.nativeOrder());
        ByteBuffer anchor2 = ByteBuffer.allocateDirect(N2 * 8).order(ByteOrder.nativeOrder());
        ByteBuffer fluid2  = ByteBuffer.allocateDirect(N2 * 4).order(ByteOrder.nativeOrder());
        ByteBuffer curing2 = ByteBuffer.allocateDirect(N2 * 4).order(ByteOrder.nativeOrder());
        for (int i = 0; i < N2; i++) {
            int gx = i % L2, rem = i / L2, gy = rem % L2;
            type2.put(i, (byte)(gy == 0 ? VTYPE_ANCHOR : VTYPE_SOLID));
            if (gy == 0) {
                anchor2.putLong(i * 8, 0xFFFFFFFFFFFFFFFFL);
            } else {
                source2.putFloat(i * 4, 1.0f);
                for (int d = 0; d < 6; d++) cond2.putFloat((d * N2 + i) * 4, 1.0f);
                rcomp2.putFloat(i * 4, 5.0f);
                rtens2.putFloat(i * 4, 0.5f);
                maxPhi2.putFloat(i * 4, 5.0f);
            }
            curing2.putFloat(i * 4, 1.0f);
        }

        rc = NativePFSFBridge.nativeRegisterIslandBuffers(handle, ID,
                phi2, source2, cond2, type2, rcomp2, rtens2, maxPhi2);
        check("L14.5", "重新 registerBuffers 6×6×6", rc == OK, "rc=" + rc);
        NativePFSFBridge.nativeRegisterIslandLookups(handle, ID,
                matId2, anchor2, fluid2, curing2);
        NativePFSFBridge.nativeRegisterStressReadback(handle, ID, phi2);

        ByteBuffer fb2 = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        fb2.putInt(0, 0);
        rc = NativePFSFBridge.nativeTickDbb(handle, ids1, 1002L, fb2);
        check("L14.6", "重新 tick 後 6×6×6 正常完成", rc == OK, "rc=" + rc);

        float maxPhi6 = 0f;
        phi2.rewind();
        for (int i = 0; i < N2; i++) maxPhi6 = Math.max(maxPhi6, phi2.getFloat());
        check("L14.7", "6×6×6 phi 非零（求解器有效）", maxPhi6 > 0f,
              String.format("maxPhi=%.5f", maxPhi6));

        NativePFSFBridge.nativeRemoveIsland(handle, ID);
    }

    // ════════════════════════════════════════════════════════════════
    //  L15 — failure 緩衝區飽和（>1024 事件，驗證 count 上限）
    // ════════════════════════════════════════════════════════════════
    static void testL15_FailureBufferSaturation() {
        header("L15", "failure 緩衝區飽和（>1024 個失效事件）");

        // 建立 12×12×12 = 1728 個體素的結構，全部 rcomp 極低 → 大量失效
        final int LS = 12, NS = LS * LS * LS;
        final int ID = 15;

        int rc = NativePFSFBridge.nativeAddIsland(handle, ID, 200, 0, 0, LS, LS, LS);
        check("L15.0", "nativeAddIsland 12×12×12", rc == OK, "rc=" + rc);
        if (rc != OK) return;

        ByteBuffer phi    = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer source = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer cond   = ByteBuffer.allocateDirect(6 * NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer type   = ByteBuffer.allocateDirect(NS).order(ByteOrder.nativeOrder());
        ByteBuffer rcomp  = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer rtens  = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer maxPhi = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer matId  = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer anchor = ByteBuffer.allocateDirect(NS * 8).order(ByteOrder.nativeOrder());
        ByteBuffer fluid  = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer curing = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());

        for (int i = 0; i < NS; i++) {
            int gx = i % LS, rem = i / LS, gy = rem % LS;
            if (gy == 0) {
                type.put(i, (byte) VTYPE_ANCHOR);
                anchor.putLong(i * 8, 0xFFFFFFFFFFFFFFFFL);
            } else {
                type.put(i, (byte) VTYPE_SOLID);
                source.putFloat(i * 4, 1.0f);
                for (int d = 0; d < 6; d++) cond.putFloat((d * NS + i) * 4, 1.0f);
                // rcomp/maxPhi 極低 → 所有體素都超過閾值 → 大量失效
                rcomp.putFloat(i * 4, 1e-6f);
                rtens.putFloat(i * 4, 1e-6f);
                maxPhi.putFloat(i * 4, 1e-6f);
            }
            curing.putFloat(i * 4, 1.0f);
        }

        rc = NativePFSFBridge.nativeRegisterIslandBuffers(handle, ID,
                phi, source, cond, type, rcomp, rtens, maxPhi);
        check("L15.1", "registerBuffers 12×12×12", rc == OK, "rc=" + rc);
        if (rc != OK) { NativePFSFBridge.nativeRemoveIsland(handle, ID); return; }
        NativePFSFBridge.nativeRegisterIslandLookups(handle, ID,
                matId, anchor, fluid, curing);
        NativePFSFBridge.nativeRegisterStressReadback(handle, ID, phi);

        // failBuf 只分配 1024 個 slot（4 + 1024×16 bytes）
        final int CAP = 1024;
        ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + CAP * 16).order(ByteOrder.LITTLE_ENDIAN);
        failBuf.putInt(0, 0);

        int[] ids = {ID};
        rc = NativePFSFBridge.nativeTickDbb(handle, ids, 2001L, failBuf);
        check("L15.2", "tick 完成", rc == OK, "rc=" + rc);

        int count = failBuf.getInt(0);
        // C++ 端限制最多寫入 CAP 個事件（max_failure_per_tick 與 Java 緩衝區容量雙重保護）
        check("L15.3", "failure count 不超過 CAP=" + CAP,
              count >= 0 && count <= CAP,
              "count=" + count);
        check("L15.4", "至少產生部分失效事件（結構應失效）",
              count > 0,
              "count=" + count + " (期望 > 0，可能求解未在 1 tick 內穩定)");
        System.out.printf("    → count=%d / CAP=%d (%.1f%% 飽和)%n",
                count, CAP, count * 100.0 / CAP);

        NativePFSFBridge.nativeRemoveIsland(handle, ID);
    }

    // ════════════════════════════════════════════════════════════════
    //  L16 — phi_orphan / NO_SUPPORT(type=3) 孤立體素觸發
    // ════════════════════════════════════════════════════════════════
    static void testL16_NoSupportOrphanTrigger() {
        header("L16", "NO_SUPPORT(type=3) — 孤立體素 phi=1e7 > phi_orphan=1e6");

        // 孤立體素結構：3×3×3 = 27 個體素
        // y=0 底層為錨點，(1,1,1) 中心為孤立體素（所有 sigma=0）
        // RBGS B4-fix: sumSigma==0 → phi_gs=1e7 > phi_orphan=1e6 → NO_SUPPORT(3)
        final int LS = 3, NS = LS * LS * LS;
        final int ID = 16;

        int rc = NativePFSFBridge.nativeAddIsland(handle, ID, 400, 0, 0, LS, LS, LS);
        check("L16.0", "nativeAddIsland 3×3×3", rc == OK, "rc=" + rc);
        if (rc != OK) return;

        ByteBuffer phi    = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer source = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer cond   = ByteBuffer.allocateDirect(6 * NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer type   = ByteBuffer.allocateDirect(NS).order(ByteOrder.nativeOrder());
        ByteBuffer rcomp  = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer rtens  = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer maxPhi = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer matId  = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer anchor = ByteBuffer.allocateDirect(NS * 8).order(ByteOrder.nativeOrder());
        ByteBuffer fluid  = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());
        ByteBuffer curing = ByteBuffer.allocateDirect(NS * 4).order(ByteOrder.nativeOrder());

        final int ORPHAN_IDX = 1 + LS * (1 + LS * 1); // (x=1, y=1, z=1) — 孤立體素

        for (int i = 0; i < NS; i++) {
            int gx = i % LS, rem = i / LS, gy = rem % LS;
            curing.putFloat(i * 4, 1.0f);

            if (gy == 0) {
                // 底層錨點
                type.put(i, (byte) VTYPE_ANCHOR);
                anchor.putLong(i * 8, 0xFFFFFFFFFFFFFFFFL);
            } else if (i == ORPHAN_IDX) {
                // 孤立體素：實心 + 零導率 + 低閾值
                // sigma 全 0 → sumSigma=0 → RBGS B4-fix → phi=1e7
                type.put(i, (byte) VTYPE_SOLID);
                source.putFloat(i * 4, 1.0f);   // 有源項確保 vtype≠AIR
                // cond 已預設為全 0 — 不設導率
                rcomp.putFloat(i * 4, 1.0f);
                rtens.putFloat(i * 4, 1.0f);
                maxPhi.putFloat(i * 4, 0.001f); // 遠小於 phi=1e7
            } else {
                // 其他體素：實心，有導率但低閾值（不期望失效）
                type.put(i, (byte) VTYPE_SOLID);
                source.putFloat(i * 4, 0.1f);
                for (int d = 0; d < 6; d++) cond.putFloat((d * NS + i) * 4, 1.0f);
                rcomp.putFloat(i * 4, 1e9f);  // 極高閾值 → 不失效
                rtens.putFloat(i * 4, 1e9f);
                maxPhi.putFloat(i * 4, 1e9f);
            }
        }

        rc = NativePFSFBridge.nativeRegisterIslandBuffers(handle, ID,
                phi, source, cond, type, rcomp, rtens, maxPhi);
        check("L16.1", "registerBuffers 3×3×3", rc == OK, "rc=" + rc);
        if (rc != OK) { NativePFSFBridge.nativeRemoveIsland(handle, ID); return; }
        NativePFSFBridge.nativeRegisterIslandLookups(handle, ID,
                matId, anchor, fluid, curing);
        NativePFSFBridge.nativeRegisterStressReadback(handle, ID, phi);

        ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + 1024 * 16).order(ByteOrder.LITTLE_ENDIAN);
        failBuf.putInt(0, 0);
        int[] ids = {ID};
        rc = NativePFSFBridge.nativeTickDbb(handle, ids, 3001L, failBuf);
        check("L16.2", "tick 完成", rc == OK, "rc=" + rc);

        // 讀取孤立體素的 phi（應為 ~1e7）
        phi.rewind();
        float orphanPhi = Float.MIN_VALUE;
        for (int i = 0; i < NS; i++) {
            float v = phi.getFloat();
            if (i == ORPHAN_IDX) orphanPhi = v;
        }
        check("L16.3", "孤立體素 phi > phi_orphan(1e6)",
              orphanPhi > 1e6f,
              String.format("orphanPhi=%.4e (期望 > 1e6)", orphanPhi));
        System.out.printf("    → orphan voxel phi=%.4e%n", orphanPhi);

        int count = failBuf.getInt(0);
        check("L16.4", "至少 1 個 NO_SUPPORT failure", count >= 1, "count=" + count);

        boolean foundNoSupport = false;
        for (int i = 0; i < Math.min(count, 1024); i++) {
            int type3 = failBuf.getInt(16 + i * 16);
            if (type3 == 3) { foundNoSupport = true; break; }
        }
        check("L16.5", "failure type=3 (NO_SUPPORT) 存在", foundNoSupport,
              "events=" + count + " but none with type=3");

        NativePFSFBridge.nativeRemoveIsland(handle, ID);
    }

    // ════════════════════════════════════════════════════════════════
    //  引擎清理
    // ════════════════════════════════════════════════════════════════
    static void shutdownEngine() {
        if (handle != 0) {
            NativePFSFBridge.nativeShutdown(handle);
            NativePFSFBridge.nativeDestroy(handle);
            handle = 0;
        }
    }

    // ════════════════════════════════════════════════════════════════
    //  輸出工具
    // ════════════════════════════════════════════════════════════════
    static void header(String layer, String desc) {
        System.out.println("\n── " + layer + ": " + desc + " ──");
    }

    static void pass(String id, String msg) {
        System.out.println("  [PASS] " + id + ": " + msg);
        passed++;
    }

    static void fail(String id, String msg) {
        System.out.println("  [FAIL] " + id + ": " + msg);
        failed++;
    }

    static void check(String id, String desc, boolean cond, String failDetail) {
        if (cond) pass(id, desc);
        else fail(id, desc + " — " + failDetail);
    }

    static class TestFailException extends RuntimeException {
        TestFailException(String msg) { super(msg); }
    }
}
