package com.blockreality.api.physics.pfsf;

import com.blockreality.api.config.BRConfig;
import com.blockreality.api.material.RMaterial;
import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.phys.Vec3;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;
import java.util.List;
import java.util.Map;
import java.util.Set;
import com.blockreality.api.physics.StructureIslandRegistry;
import com.blockreality.api.physics.StructureIslandRegistry.StructureIsland;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Function;

/**
 * Façade + {@link IPFSFRuntime} adapter for the native {@code libblockreality_pfsf}
 * runtime.
 */
public final class NativePFSFRuntime {

    private static final Logger LOGGER = LoggerFactory.getLogger("PFSF-NativeRT");

    public static final String ACTIVATION_PROPERTY = "blockreality.native.pfsf";
    public static final String DISABLE_PROPERTY    = "blockreality.native.pfsf.disable";

    private static final boolean KERNELS_PORTED = true;

    // Default-on: users should not need to set a JVM flag for physics to work.
    // Explicit -Dblockreality.native.pfsf=false or -Dblockreality.native.pfsf.disable=true still disables.
    private static final boolean FLAG_ENABLED = resolveFlagEnabled();
    private static final AtomicBoolean INIT_ATTEMPTED = new AtomicBoolean(false);

    private static boolean resolveFlagEnabled() {
        if (Boolean.getBoolean(DISABLE_PROPERTY)) return false;
        String explicit = System.getProperty(ACTIVATION_PROPERTY);
        if (explicit != null) return Boolean.parseBoolean(explicit);
        return true;
    }

    private static volatile long    handle = 0L;
    private static volatile boolean active = false;
    private static volatile int     shadersRegistered = 0;
    private static volatile int     shadersMissing    = 0;

    private static final RuntimeView VIEW = new RuntimeView();

    private NativePFSFRuntime() {}

    public static long getHandle() { return handle; }

    public static boolean isActive()        { return active;      }
    public static boolean isFlagEnabled()   { return FLAG_ENABLED; }
    public static boolean isLibraryLoaded() { return NativePFSFBridge.isAvailable(); }
    public static boolean areKernelsPorted(){ return KERNELS_PORTED; }
    public static int     getShadersRegistered() { return shadersRegistered; }
    public static int     getShadersMissing()    { return shadersMissing; }

    public static IPFSFRuntime asRuntime() { return VIEW; }

    private static boolean registerRequiredShaders() {
        String[] shaders = {
            "pfsf/jacobi_smooth.comp",
            "pfsf/rbgs_smooth.comp",
            "pfsf/phase_field_evolve.comp",
            "pfsf/pcg_matvec.comp",
            "pfsf/pcg_update.comp",
            "pfsf/pcg_direction.comp",
            "pfsf/pcg_dot.comp",
            "pfsf/pcg_precompute.comp",
            "pfsf/failure_scan.comp",
            "pfsf/sparse_scatter.comp"
        };

        int registered = 0, missing = 0;
        for (String s : shaders) {
            String resourcePath = "/assets/blockreality/shaders/compute/" + s + ".spv";
            try (java.io.InputStream in = NativePFSFRuntime.class.getResourceAsStream(resourcePath)) {
                if (in == null) {
                    missing++;
                    LOGGER.error("Shader missing from jar: {} (run :api:precompileShaders?)", resourcePath);
                    continue;
                }
                byte[] bytes = in.readAllBytes();
                java.nio.ByteBuffer dbb = java.nio.ByteBuffer.allocateDirect(bytes.length);
                dbb.put(bytes);
                // Native side reads via GetDirectBufferAddress + GetDirectBufferCapacity
                // (both position-independent), so flip() is not strictly required.
                // Call it anyway so the buffer round-trips cleanly if a future
                // position-aware reader is introduced on either side.
                dbb.flip();
                String canonicalName = "compute/" + s;
                NativePFSFBridge.nativeRegisterShader(canonicalName, dbb);
                registered++;
            } catch (Exception e) {
                missing++;
                LOGGER.error("Failed to register shader: {}", s, e);
            }
        }
        shadersRegistered = registered;
        shadersMissing    = missing;
        if (registered == 0) {
            LOGGER.error("No PFSF shaders registered; physics will not function. Expected {} shaders.", shaders.length);
            return false;
        }
        if (missing > 0) {
            LOGGER.warn("PFSF shader registration incomplete: registered={}, missing={}", registered, missing);
        } else {
            LOGGER.info("PFSF shaders registered: {}/{}", registered, shaders.length);
        }
        return true;
    }

    public static synchronized void init() {
        if (!INIT_ATTEMPTED.compareAndSet(false, true)) return;

        if (!FLAG_ENABLED) {
            LOGGER.info("Native PFSF runtime disabled via -D{}=false or -D{}=true.",
                    ACTIVATION_PROPERTY, DISABLE_PROPERTY);
            return;
        }
        if (!NativePFSFBridge.isAvailable()) {
            LOGGER.warn("Native PFSF runtime enabled but libblockreality_pfsf was not loaded; physics will not run.");
            return;
        }
        LOGGER.info("Native PFSF runtime enabling (KERNELS_PORTED={})", KERNELS_PORTED);

        if (!registerRequiredShaders()) {
            return;
        }

        long h = 0L;
        try {
            // 修正：在測試環境中 BRConfig 可能未載入，提供安全回退值
            int budget = 10;
            boolean pcgEnabled = true;
            try {
                budget = BRConfig.getPFSFTickBudgetMs();
                pcgEnabled = BRConfig.isPFSFPCGEnabled();
            } catch (Throwable ignored) {}

            long vkInst  = VulkanComputeContext.getVkInstanceHandle();
            long vkPhys  = VulkanComputeContext.getVkPhysicalDeviceHandle();
            long vkDev   = VulkanComputeContext.getVkDeviceHandle();
            long vkQueue = VulkanComputeContext.getVkComputeQueueHandle();
            int  vkQf    = VulkanComputeContext.getComputeQueueFamily();
            boolean shareVk = VulkanComputeContext.isComputeSupported()
                    && vkInst != 0L && vkPhys != 0L && vkDev != 0L && vkQueue != 0L && vkQf >= 0;

            if (shareVk) {
                try {
                    h = NativePFSFBridge.nativeCreateWithVulkan(
                            50_000, Math.max(1, budget), 512L * 1024 * 1024,
                            true, true,
                            vkInst, vkPhys, vkDev, vkQf, vkQueue);
                    LOGGER.info("Native PFSF sharing Vulkan instance/device with Java host.");
                } catch (UnsatisfiedLinkError missing) {
                    // Older native binary without the shared-handle entrypoint.
                    LOGGER.warn("nativeCreateWithVulkan not present in native lib; falling back to standalone.");
                    h = 0L;
                }
            }
            if (h == 0L) {
                h = NativePFSFBridge.nativeCreate(50_000, Math.max(1, budget), 512L * 1024 * 1024, true, true);
            }
            if (h == 0L) return;

            int rc = NativePFSFBridge.nativeInit(h);
            if (rc != NativePFSFBridge.PFSFResult.OK) {
                NativePFSFBridge.nativeDestroy(h);
                return;
            }

            handle = h;
            active = true;
            try {
                NativePFSFBridge.nativeSetPCGEnabled(h, pcgEnabled);
            } catch (UnsatisfiedLinkError ignored) {}
            LOGGER.info("Native PFSF runtime attached (handle=0x{}, kernels_ported={})", Long.toHexString(h), KERNELS_PORTED);
        } catch (Throwable t) {
            LOGGER.error("Native PFSF init failed", t);
            if (h != 0L) NativePFSFBridge.nativeDestroy(h);
            active = false;
            handle = 0L;
        }
    }

    public static synchronized void shutdown() {
        long h = handle;
        handle = 0L; active = false;
        INIT_ATTEMPTED.set(false);
        if (h != 0L) NativePFSFBridge.nativeDestroy(h);
    }

    public static String getStatus() {
        if (active) return String.format("Native PFSF: %s", KERNELS_PORTED ? "ROUTING" : "ATTACHED");
        return "Native PFSF: INACTIVE";
    }

    public static ByteBuffer getSparseUploadBuffer(int islandId) {
        if (!active || handle == 0L) return null;
        return NativePFSFBridge.nativeGetSparseUploadBuffer(handle, islandId);
    }

    public static int notifySparseUpdates(int islandId, int count) {
        if (!active || handle == 0L) return NativePFSFBridge.PFSFResult.ERROR_NOT_INIT;
        return NativePFSFBridge.nativeNotifySparseUpdates(handle, islandId, count);
    }

    /**
     * Notifies the native engine that an island is being removed (eviction, world unload, etc.).
     * Called from PFSFEngine.removeBuffer so IslandBufferEvictor and AABB-expansion paths
     * also clean up the native GPU allocation — not just explicit API callers.
     */
    static void notifyIslandRemoved(int islandId) {
        if (active && handle != 0L) {
            NativePFSFBridge.nativeRemoveIsland(handle, islandId);
            VIEW.nativeIslandDims.remove(islandId);
            VIEW.solveTicksLeft.remove(islandId);
        }
    }

    static final class RuntimeView implements IPFSFRuntime {
        private long lastProcessedEpoch = -1;

        // Tracks (lx, ly, lz) registered per island in the native engine.
        // nativeAddIsland is called only when an island is new or its AABB changed.
        // C++ getOrCreate re-allocates GPU buffers when dims differ, so re-adding is safe.
        final java.util.Map<Integer, int[]> nativeIslandDims =
                new java.util.concurrent.ConcurrentHashMap<>();

        // Warm-start convergence budget: number of additional ticks remaining after the
        // initial block-change tick. The RBGS solver needs several ticks to propagate
        // stress through large islands — without this, physics stops after one tick and
        // large structures never fully converge.
        // Entry absent → island is not in the warm-start phase (clean / just processed).
        // Entry present → island keeps being ticked; decremented each tick; removed at 0.
        // Budget scales with island volume: n/64 ticks, clamped to [4, 20].
        // 4×4×4 (64 vox) → 4 ticks; 8×8×8 (512 vox) → 8 ticks; 16×16×16+ → 20 ticks.
        private static final int MAX_SOLVE_TICKS = 20;
        private static final int MIN_SOLVE_TICKS = 4;
        final java.util.Map<Integer, Integer> solveTicksLeft = new java.util.HashMap<>();

        // Pre-allocated failure readback buffer — capacity for 1024 events.
        // Re-used every tick to avoid per-tick direct memory allocation (16 KB).
        private final ByteBuffer failBuf = ByteBuffer.allocateDirect(4 + 1024 * 16)
                .order(java.nio.ByteOrder.LITTLE_ENDIAN);

        @Override public void init() { NativePFSFRuntime.init(); }
        @Override public void shutdown() { NativePFSFRuntime.shutdown(); }
        // No-fallback contract: native runtime borrows Vulkan handles from
        // VulkanComputeContext (see init() lines 147-152). Without Vulkan the
        // C++ tick path is a no-op (pfsf_engine.cpp gates dispatch on
        // descPool != VK_NULL_HANDLE), so reporting "available" would silently
        // suppress all failure detection. Require Vulkan explicitly.
        @Override public boolean isAvailable() {
            return active && KERNELS_PORTED && VulkanComputeContext.isAvailable();
        }
        @Override public String getStats() { return getStatus(); }

        @Override
        public void onServerTick(ServerLevel level, List<ServerPlayer> players, long currentEpoch) {
            if (!isAvailable()) return;

            Map<Integer, StructureIsland> dirtyIslands = StructureIslandRegistry.getDirtyIslands(lastProcessedEpoch);
            if (dirtyIslands.isEmpty()) {
                lastProcessedEpoch = currentEpoch;
                return;
            }

            // Two-pass: first prepare all islands, then tick together.
            // This lets nativeTickDbb batch all dirty islands in one submit.
            java.util.List<Integer> tickableIds = new java.util.ArrayList<>(dirtyIslands.size());
            for (int id : dirtyIslands.keySet()) {
                StructureIsland island = dirtyIslands.get(id);
                PFSFIslandBuffer buf = PFSFBufferManager.getBuffer(id);
                if (buf == null) buf = PFSFBufferManager.getOrCreateBuffer(island);
                if (buf == null) continue;  // VRAM budget rejected

                if (solveTicksLeft.containsKey(id)) {
                    // Warm-start tick: island is continuing convergence from a previous
                    // block-change event. Buffer registrations are already in place; just
                    // re-dirty the C-side engine so uploadFromHosts carries the warm phi.
                    NativePFSFBridge.nativeMarkFullRebuild(handle, id);
                    tickableIds.add(id);
                    continue;
                }

                // First tick after a block-change event: full registration path.

                // Register island with native engine (or re-register on AABB change).
                int lx = buf.getLx(), ly = buf.getLy(), lz = buf.getLz();
                int[] registered = nativeIslandDims.get(id);
                if (registered == null || registered[0] != lx || registered[1] != ly || registered[2] != lz) {
                    net.minecraft.core.BlockPos origin = buf.getOrigin();
                    int rc = NativePFSFBridge.nativeAddIsland(handle, id,
                            origin.getX(), origin.getY(), origin.getZ(), lx, ly, lz);
                    if (rc != NativePFSFBridge.PFSFResult.OK) {
                        LOGGER.warn("nativeAddIsland failed for island {} (rc={})", id, rc);
                        continue;
                    }
                    nativeIslandDims.put(id, new int[]{lx, ly, lz});
                }

                // Compute & normalise source/conductivity into hostCoalescedBuf (zero-copy DBB).
                PFSFDataBuilder.updateSourceAndConductivity(buf, island, level,
                        PFSFEngine.getMaterialLookup(), PFSFEngine.getAnchorLookup(),
                        PFSFEngine.getFillRatioLookup(), PFSFEngine.getCuringLookup(),
                        PFSFEngine.getCurrentWindVec(), null);

                // Register persistent host-side DBBs so native reads from hostCoalescedBuf.
                java.nio.ByteBuffer phiBB = buf.getPhiBufAsBB();
                int regRc = NativePFSFBridge.nativeRegisterIslandBuffers(handle, id,
                        phiBB, buf.getSourceBufAsBB(), buf.getCondBufAsBB(),
                        buf.getTypeBufAsBB(), buf.getRcompBufAsBB(), buf.getRtensBufAsBB(),
                        buf.getMaxPhiBufAsBB());
                if (regRc != NativePFSFBridge.PFSFResult.OK) {
                    LOGGER.warn("nativeRegisterIslandBuffers failed for island {} (rc={})", id, regRc);
                    continue;
                }

                // Phi warm-start: after each dispatch C++ writes GPU phi back into phiBB
                // (hostCoalescedBuf[phiOffset..]). The next tick's uploadFromHosts reads
                // this warm solution, avoiding cold-start convergence from phi=0 each tick.
                // phiBB is a duplicate slice of hostCoalescedBuf — same backing memory.
                NativePFSFBridge.nativeRegisterStressReadback(handle, id, phiBB);

                // Augmentation lookups — curing is wired; materialId/anchorBitmap/fluidPressure
                // are zero-filled stubs (not used in uploadFromHosts but required non-null by ABI).
                // PFSFDataBuilder.writeLookupCuring() was already called inside updateSourceAndConductivity.
                NativePFSFBridge.nativeRegisterIslandLookups(handle, id,
                        buf.getLookupMaterialIdBB(), buf.getLookupAnchorBitmapBB(),
                        buf.getLookupFluidPressureBB(), buf.getLookupCuringBB());

                tickableIds.add(id);
                int n = buf.getLx() * buf.getLy() * buf.getLz();
                solveTicksLeft.put(id, Math.min(MAX_SOLVE_TICKS, Math.max(MIN_SOLVE_TICKS, n / 64)));
            }

            if (tickableIds.isEmpty()) {
                lastProcessedEpoch = currentEpoch;
                return;
            }

            int[] dirtyIds = new int[tickableIds.size()];
            for (int i = 0; i < dirtyIds.length; i++) dirtyIds[i] = tickableIds.get(i);
            failBuf.putInt(0, 0); // C++ reads header count and appends; must start at 0
            int rc = NativePFSFBridge.nativeTickDbb(handle, dirtyIds, currentEpoch, failBuf);
            if (rc == NativePFSFBridge.PFSFResult.OK) {
                int count = failBuf.getInt(0);
                for (int i = 0; i < Math.min(count, 1024); i++) {
                    int x        = failBuf.getInt(4  + i * 16);
                    int y        = failBuf.getInt(8  + i * 16);
                    int z        = failBuf.getInt(12 + i * 16);
                    int nativeType = failBuf.getInt(16 + i * 16);
                    BlockPos pos = new BlockPos(x, y, z);
                    com.blockreality.api.physics.FailureType type = switch (nativeType) {
                        case 2  -> com.blockreality.api.physics.FailureType.CRUSHING;
                        case 3  -> com.blockreality.api.physics.FailureType.NO_SUPPORT;
                        case 4  -> com.blockreality.api.physics.FailureType.TENSION_BREAK;
                        default -> com.blockreality.api.physics.FailureType.CANTILEVER_BREAK;
                    };
                    com.blockreality.api.collapse.CollapseManager.triggerPFSFCollapse(level, pos, type);
                }
                for (int id : dirtyIds) {
                    int left = solveTicksLeft.getOrDefault(id, 0) - 1;
                    if (left <= 0) {
                        solveTicksLeft.remove(id);
                        StructureIslandRegistry.markProcessed(id);
                    } else {
                        solveTicksLeft.put(id, left);
                        StructureIslandRegistry.markDirty(id); // keep in next tick's dirty list
                    }
                }
            }
            lastProcessedEpoch = currentEpoch;
        }

        @Override
        public void notifyBlockChange(int islandId, BlockPos pos, RMaterial newMaterial, Set<BlockPos> anchors) {
            if (!active) return;
            NativePFSFBridge.nativeMarkFullRebuild(handle, islandId);
            // Invalidate warm-start so the next tick runs full re-registration
            // with freshly computed source/conductivity for the changed island.
            solveTicksLeft.remove(islandId);
            // Mark dirty in the Java registry so getDirtyIslands picks it up on the next
            // tick regardless of game-time epoch vs structure-change-counter mismatch.
            StructureIslandRegistry.markDirty(islandId);
        }

        @Override public void setMaterialLookup(Function<BlockPos, RMaterial> lookup) {}
        @Override public void setAnchorLookup(Function<BlockPos, Boolean> lookup) {}
        @Override public void setFillRatioLookup(Function<BlockPos, Float> lookup) {}
        @Override public void setCuringLookup(Function<BlockPos, Float> lookup) {}

        @Override
        public void setWindVector(Vec3 wind) {
            if (active && wind != null) NativePFSFBridge.nativeSetWind(handle, (float)wind.x, (float)wind.y, (float)wind.z);
        }

        @Override
        public void removeBuffer(int islandId) {
            // Native engine GPU cleanup is handled by notifyIslandRemoved (called from PFSFEngine).
            // This override exists for IPFSFRuntime contract completeness.
            notifyIslandRemoved(islandId);
        }
    }
}
