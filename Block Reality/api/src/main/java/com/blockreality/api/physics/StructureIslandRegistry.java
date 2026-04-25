package com.blockreality.api.physics;

import com.blockreality.api.block.RBlockEntity;
import com.blockreality.api.material.DefaultMaterial;
import com.blockreality.api.material.RMaterial;
import com.blockreality.api.physics.PhysicsConstants;
import com.blockreality.api.physics.pfsf.LabelPropagation;
import com.blockreality.api.physics.topology.ThreeTierOrchestrator;
import com.blockreality.api.physics.topology.TopologicalSVDAG;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.block.entity.BlockEntity;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import javax.annotation.Nullable;
import javax.annotation.concurrent.ThreadSafe;
import java.util.ArrayDeque;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Consumer;
import java.util.function.Function;

/**
 * 結構島嶼登錄 — 追蹤所有 RBlock 連通分量（「島嶼」）。
 *
 * 設計目標：
 *   1. 放置/破壞時 O(1)~O(K) 增量更新（K = 鄰居數 ≤ 6）
 *   2. 支援跨 chunk 的大型結構（突破 40³ 快照上限）
 *   3. 為並行 PhysicsExecutor 提供獨立工作單元
 *
 * 核心概念：
 *   - 每個 RBlock 屬於恰好一個 island
 *   - 相鄰的 RBlock 屬於同一個 island（6 方向連通）
 *   - 放置方塊時合併相鄰 island
 *   - 破壞方塊時可能分裂 island（使用 BFS 驗證）
 *
 * 執行緒安全：
 *   - 所有修改操作（register/unregister）應在 server thread 呼叫
 *   - 查詢操作（getIsland/getIslandId）可從任意執行緒呼叫
 */
public class StructureIslandRegistry {

    private static final Logger LOGGER = LogManager.getLogger("BR-IslandRegistry");

    /** 島嶼 ID 生成器 */
    private static final AtomicInteger nextIslandId = new AtomicInteger(1);

    /** 方塊位置 → 島嶼 ID 映射 */
    private static final ConcurrentHashMap<BlockPos, Integer> blockToIsland = new ConcurrentHashMap<>();

    /** 島嶼 ID → 島嶼資訊 */
    private static final ConcurrentHashMap<Integer, StructureIsland> islands = new ConcurrentHashMap<>();

    /**
     * 錨點方塊集合（bedrock / barrier / explicitly-pinned）。用於 split 後判定
     * 新 island 是否仍連通至錨點；無錨者即為 orphan，必須立即觸發掉落。
     * 維護由 {@link #registerAnchor} / {@link #unregisterAnchor} 負責；呼叫方為
     * {@code AnchorContinuityChecker} 所在的 chunk-load / block-update 流程。
     */
    private static final Set<BlockPos> anchorBlocks = ConcurrentHashMap.newKeySet();

    /**
     * Logged once per JVM when {@link #orphanClassificationEnabled}
     * first observes an empty anchor set. Codex review on the initial
     * land of this classifier flagged (P0) that
     * {@link #registerAnchor} is only ever called from tests, so in
     * production {@link #anchorBlocks} stays empty and every fractured
     * component would be treated as orphan. The safety valve below
     * keeps {@code notifyOrphan} silent until anchor registration is
     * wired from the natural-anchor flow; the rewrite under
     * {@code physics/topology/} supersedes this code path entirely but
     * the valve protects today's production builds in the meantime.
     */
    private static final java.util.concurrent.atomic.AtomicBoolean SAFETY_VALVE_LOGGED =
            new java.util.concurrent.atomic.AtomicBoolean(false);

    /**
     * Split 後產生、無任何錨點連通的新 island 會透過此 listener 通知。
     * 實務上由 {@code CollapseManager} 安裝，立即整個拆除該 island。
     * 若未安裝（如單元測試情境），不會有任何副作用，僅記錄 WARN log。
     */
    @Nullable
    private static volatile Consumer<OrphanIslandEvent> orphanListener = null;

    // ═══════════════════════════════════════════════════════════════
    //  Topology v2: 正式上線 ThreeTierOrchestrator
    // ═══════════════════════════════════════════════════════════════
    // The safety-valve in orphanClassificationEnabled() has been keeping
    // orphan detection off in production because anchorBlocks is empty.
    // The topology/ rewrite solves this properly with Elder Rule identity
    // and anchor-aware BFS; we mirror every register/unregister into it
    // here and route its orphan events back through orphanListener so
    // existing subscribers (CollapseManager) see the same OrphanIslandEvent
    // shape. GPU int-id plumbing is untouched — the orchestrator only
    // informs orphan decisions, it does not re-own the int-id registry.
    private static final ThreeTierOrchestrator topology = createTopology();

    /**
     * Elder-Rule fingerprint → legacy int islandId. Populated each tick
     * by {@link #advanceTopology(long)} from the orchestrator's
     * component bindings. Queryable from any thread; updates happen
     * only on the server tick thread.
     */
    private static final ConcurrentHashMap<Long, Integer> fingerprintToIntId = new ConcurrentHashMap<>();
    /** Reverse: int islandId → last seen Elder-Rule fingerprint. */
    private static final ConcurrentHashMap<Integer, Long> intIdToFingerprint = new ConcurrentHashMap<>();

    private static ThreeTierOrchestrator createTopology() {
        ThreeTierOrchestrator t = new ThreeTierOrchestrator();
        t.setOrphanSink(event -> {
            // Map new-system orphan voxels to the int-id the rest of the
            // engine (GPU, CollapseManager subscribers) speaks. We look up
            // the int island by any voxel in the orphan set; all members
            // share the same int id because the CPU-side BFS always keeps
            // blockToIsland coherent before this sink runs.
            Set<BlockPos> voxels = event.voxels();
            if (voxels.isEmpty()) return;
            int islandId = -1;
            for (BlockPos p : voxels) {
                Integer id = blockToIsland.get(p);
                if (id != null) { islandId = id; break; }
            }
            if (islandId < 0) {
                // Voxels are known to the orchestrator but not the int
                // registry — race between tick() and a concurrent
                // unregister that already removed them. Ignore; next
                // tick will reconcile.
                return;
            }
            long epoch = ConnectivityCache.getStructureEpoch();
            notifyOrphan(islandId, voxels, epoch, null);
        });
        return t;
    }

    /** Accessor for tests and advanced diagnostics. */
    public static ThreeTierOrchestrator getTopology() { return topology; }

    /**
     * Look up the Elder-Rule-stable fingerprint for a legacy int
     * island id. Returns {@code 0L} when the island has not yet been
     * reconciled (first tick not yet run) or has been destroyed.
     * Downstream systems that want ID stability across split/merge —
     * collapse journaling, persistence checkpoints, test oracles —
     * should key on this value rather than the int id.
     */
    public static long getStableFingerprint(int islandId) {
        Long fp = intIdToFingerprint.get(islandId);
        return fp != null ? fp : 0L;
    }

    /**
     * Reverse lookup: find the current int island id that represents
     * the component with the given Elder-Rule fingerprint. Returns
     * {@code -1} when the fingerprint is unknown.
     */
    public static int getIslandIdByFingerprint(long fingerprint) {
        Integer id = fingerprintToIntId.get(fingerprint);
        return id != null ? id : -1;
    }

    /** Records the last topology tick at which each int id was seen dirty. */
    private static final ConcurrentHashMap<Integer, Long> lastMutationTick = new ConcurrentHashMap<>();

    /**
     * Current topology tick from the underlying {@link PersistentIslandTracker}.
     * Advances once per {@link #advanceTopology(long)} call. Downstream
     * systems use this as the universal clock for "has anything
     * changed since I last ran" queries, replacing the ad-hoc
     * ALWAYS_DIRTY / {@code lastModifiedEpoch} Long.MAX_VALUE sentinel
     * which never reflected the actual topology clock.
     */
    public static long getTopologyTick() {
        return topology.getTracker().currentTick();
    }

    /**
     * True when the island's voxel set has not mutated since
     * {@code sinceTick}. Callers that already processed this island on
     * or after {@code sinceTick} can use this as an early-skip — the
     * component is topologically clean, no GPU re-solve is required
     * for shape-change reasons (force / material changes still need
     * to be handled by their own paths).
     */
    public static boolean topologicallyClean(int islandId, long sinceTick) {
        Long t = lastMutationTick.get(islandId);
        return t == null || t <= sinceTick;
    }

    /**
     * 將於 split 結束時傳送給 {@link #orphanListener} 的事件。
     * {@code level} 為觸發本次 split 的 ServerLevel（若呼叫者有提供，
     * 例如 {@link #unregisterBlock}）；{@code flushDestructions} 無此資訊
     * 時為 {@code null}，由 listener 自行 fallback 查詢。
     */
    public record OrphanIslandEvent(int islandId,
                                    Set<BlockPos> members,
                                    long epoch,
                                    @Nullable ServerLevel level) {}

    /** 安裝 orphan listener（無則以 null 解除）。 */
    public static void setOrphanListener(@Nullable Consumer<OrphanIslandEvent> listener) {
        orphanListener = listener;
    }

    /** 將 {@code pos} 標記為錨點（若此前未登錄）。執行緒安全；可重複呼叫。 */
    public static void registerAnchor(BlockPos pos) {
        BlockPos p = pos.immutable();
        anchorBlocks.add(p);
        // Mirror to topology so the new-system orphan classifier sees
        // the same anchor set as the legacy isBlockAnchored() check.
        topology.setVoxel(p.getX(), p.getY(), p.getZ(), TopologicalSVDAG.TYPE_ANCHOR);
    }

    /** Returns whether {@code pos} is currently registered as an anchor voxel. */
    public static boolean isAnchorRegistered(BlockPos pos) {
        return anchorBlocks.contains(pos);
    }

    /** 取消錨點登錄。若此前未登錄則為 no-op。 */
    public static void unregisterAnchor(BlockPos pos) {
        anchorBlocks.remove(pos);
        // Demote back to a plain solid voxel if the block itself is
        // still registered; otherwise reflect as air. The caller is
        // expected to follow up with unregisterBlock if the block has
        // actually been broken.
        byte t = blockToIsland.containsKey(pos)
                ? TopologicalSVDAG.TYPE_SOLID
                : TopologicalSVDAG.TYPE_AIR;
        topology.setVoxel(pos.getX(), pos.getY(), pos.getZ(), t);
    }

    /** 僅供測試：清除全部狀態（含錨點與 listener）以確保測試獨立性。 */
    public static void resetForTesting() {
        blockToIsland.clear();
        islands.clear();
        anchorBlocks.clear();
        orphanListener = null;
        nextIslandId.set(1);
        topology.reset();
        fingerprintToIntId.clear();
        intIdToFingerprint.clear();
        lastMutationTick.clear();
        com.blockreality.api.physics.pnsm.PNSMShadow.reset();
    }

    /**
     * 結構島嶼 — 一組連通的 RBlock。
     */
    public static class StructureIsland {
        private final int id;
        private final Set<BlockPos> members = ConcurrentHashMap.newKeySet();
        private volatile int minX, minY, minZ, maxX, maxY, maxZ;
        private volatile long lastModifiedEpoch;

        /**
         * The oldest surviving member, used as the Elder-Rule seed when
         * {@link StructureIslandRegistry#checkAndSplitIsland} needs to
         * pick which fragment keeps this island's int id. Set to the
         * voxel that caused the island's creation; refreshed on the
         * next addMember when the original birth voxel is removed.
         * {@code null} briefly between full-decay and removal.
         */
        private volatile BlockPos birthVoxel;

        // ─── B1: Centre-of-mass cache ───
        // NaN signals "needs recomputation". Invalidated on every addMember/removeMember.
        private volatile double cachedComX = Double.NaN;
        private volatile double cachedComY = Double.NaN;
        private volatile double cachedComZ = Double.NaN;

        StructureIsland(int id) {
            this.id = id;
            this.minX = Integer.MAX_VALUE;
            this.minY = Integer.MAX_VALUE;
            this.minZ = Integer.MAX_VALUE;
            this.maxX = Integer.MIN_VALUE;
            this.maxY = Integer.MIN_VALUE;
            this.maxZ = Integer.MIN_VALUE;
        }

        public int getId() { return id; }
        public Set<BlockPos> getMembers() { return Collections.unmodifiableSet(members); }
        public int getBlockCount() { return members.size(); }
        public long getLastModifiedEpoch() { return lastModifiedEpoch; }

        public BlockPos getMinCorner() { return new BlockPos(minX, minY, minZ); }
        public BlockPos getMaxCorner() { return new BlockPos(maxX, maxY, maxZ); }

        /** AABB 體積（用於判斷是否超過快照上限） */
        public int getAABBVolume() {
            if (members.isEmpty()) return 0;
            return (maxX - minX + 1) * (maxY - minY + 1) * (maxZ - minZ + 1);
        }

        synchronized void addMember(BlockPos pos) {
            members.add(pos);
            minX = Math.min(minX, pos.getX());
            minY = Math.min(minY, pos.getY());
            minZ = Math.min(minZ, pos.getZ());
            maxX = Math.max(maxX, pos.getX());
            maxY = Math.max(maxY, pos.getY());
            maxZ = Math.max(maxZ, pos.getZ());
            if (birthVoxel == null) birthVoxel = pos;
            invalidateCoM();
        }

        synchronized void removeMember(BlockPos pos) {
            members.remove(pos);
            if (pos.equals(birthVoxel)) {
                // Elder-Rule seed was just broken. Pick any surviving
                // member as the new seed; on the next split we fall
                // back to "anchored+largest" if the new seed is also
                // absent from the kept component, which is still a
                // monotonic improvement over no seed at all.
                birthVoxel = members.isEmpty() ? null : members.iterator().next();
            }
            invalidateCoM();
        }

        /** Invalidate CoM cache — must be called after any membership change. */
        void invalidateCoM() {
            cachedComX = Double.NaN;
        }

        /**
         * Mass-weighted centre of mass (world space, block-centre offsets).
         * First call is O(N); subsequent calls return cached value until membership changes.
         *
         * @param matLookup optional material lookup; pass {@code null} to use CONCRETE density
         * @return [comX, comY, comZ] in world coordinates
         */
        public synchronized double[] getCoM(Function<BlockPos, RMaterial> matLookup) {
            if (!Double.isNaN(cachedComX)) {
                return new double[]{ cachedComX, cachedComY, cachedComZ };
            }
            double cx = 0, cy = 0, cz = 0, totalMass = 0;
            for (BlockPos p : members) {
                RMaterial mat = matLookup != null ? matLookup.apply(p) : null;
                double density = (mat != null ? mat.getDensity()
                        : DefaultMaterial.CONCRETE.getDensity());
                double m = density * PhysicsConstants.BLOCK_AREA;
                cx += (p.getX() + 0.5) * m;
                cy += (p.getY() + 0.5) * m;
                cz += (p.getZ() + 0.5) * m;
                totalMass += m;
            }
            if (totalMass < 1e-9) totalMass = 1.0;
            cachedComX = cx / totalMass;
            cachedComY = cy / totalMass;
            cachedComZ = cz / totalMass;
            return new double[]{ cachedComX, cachedComY, cachedComZ };
        }

        /** 重新計算 AABB（在成員變動後） */
        synchronized void recalculateBounds() {
            if (members.isEmpty()) {
                minX = minY = minZ = 0;
                maxX = maxY = maxZ = 0;
                return;
            }
            int mnX = Integer.MAX_VALUE, mnY = Integer.MAX_VALUE, mnZ = Integer.MAX_VALUE;
            int mxX = Integer.MIN_VALUE, mxY = Integer.MIN_VALUE, mxZ = Integer.MIN_VALUE;
            for (BlockPos p : members) {
                mnX = Math.min(mnX, p.getX());
                mnY = Math.min(mnY, p.getY());
                mnZ = Math.min(mnZ, p.getZ());
                mxX = Math.max(mxX, p.getX());
                mxY = Math.max(mxY, p.getY());
                mxZ = Math.max(mxZ, p.getZ());
            }
            minX = mnX; minY = mnY; minZ = mnZ;
            maxX = mxX; maxY = mxY; maxZ = mxZ;
        }

        void touch(long epoch) { this.lastModifiedEpoch = epoch; }
    }

    // ═══════════════════════════════════════════════════════
    //  放置：登錄方塊 + 合併相鄰 island
    // ═══════════════════════════════════════════════════════

    /**
     * 登錄新放置的 RBlock。
     * 檢查 6 鄰居，找到所有相鄰 island 並合併。
     *
     * @param pos   新方塊位置
     * @param epoch 當前結構 epoch
     * @return 此方塊所屬的 island ID
     */
    public static int registerBlock(BlockPos pos, long epoch) {
        // Mirror into topology as a solid voxel before the int-id BFS
        // runs; the orchestrator's own tick consumes this on the next
        // advanceTopology() invocation.
        {
            byte t = anchorBlocks.contains(pos)
                    ? TopologicalSVDAG.TYPE_ANCHOR
                    : TopologicalSVDAG.TYPE_SOLID;
            topology.setVoxel(pos.getX(), pos.getY(), pos.getZ(), t);
        }
        // Mirror into the PNSM shadow (Phase 1 of the RFC at
        // docs/design/pfsf-native-structural-memory.md). No-op when
        // the flag is off; idempotent when it is on, so re-registering
        // an already-live pos produces no drift.
        com.blockreality.api.physics.pnsm.PNSMShadow.mirrorInsert(pos);

        Set<Integer> neighborIslands = new HashSet<>();
        for (Direction dir : Direction.values()) {
            Integer id = blockToIsland.get(pos.relative(dir));
            if (id != null) {
                neighborIslands.add(id);
            }
        }

        if (neighborIslands.isEmpty()) {
            // 孤立方塊 → 建立新 island
            int newId = nextIslandId.getAndIncrement();
            StructureIsland island = new StructureIsland(newId);
            island.addMember(pos);
            islands.put(newId, island);
            blockToIsland.put(pos, newId);
            markDirty(newId); // ALWAYS_DIRTY — ensures physics runs regardless of game epoch vs structure epoch
            LOGGER.debug("[IslandRegistry] New island {} at {}", newId, pos.toShortString());
            return newId;
        }

        if (neighborIslands.size() == 1) {
            // 只有一個相鄰 island → 直接加入
            // 使用 computeIfPresent 保證原子性：若 island 已被並發移除則跳過，避免方塊加入孤立物件
            int targetId = neighborIslands.iterator().next();
            boolean added = islands.computeIfPresent(targetId, (k, island) -> {
                island.addMember(pos);
                return island;
            }) != null;
            if (added) {
                blockToIsland.put(pos, targetId);
                markDirty(targetId); // ALWAYS_DIRTY — block added, must re-solve
            }
            return targetId;
        }

        // 多個相鄰 island → 合併到最大的 island 中
        int targetId = -1;
        int maxSize = -1;
        for (int id : neighborIslands) {
            StructureIsland island = islands.get(id);
            if (island != null && island.getBlockCount() > maxSize) {
                maxSize = island.getBlockCount();
                targetId = id;
            }
        }

        StructureIsland target = islands.get(targetId);
        if (target == null) return -1;

        // 合併其他 island 到 target
        for (int id : neighborIslands) {
            if (id == targetId) continue;
            StructureIsland other = islands.remove(id);
            if (other != null) {
                for (BlockPos member : other.members) {
                    target.addMember(member);
                    blockToIsland.put(member, targetId);
                }
                LOGGER.debug("[IslandRegistry] Merged island {} ({} blocks) into {} at {}",
                    id, other.getBlockCount(), targetId, pos.toShortString());
            }
        }

        // 使用 computeIfPresent 保護：若 target 已被並發移除則跳過，避免方塊加入孤立物件
        boolean added = islands.computeIfPresent(targetId, (k, t) -> {
            t.addMember(pos);
            return t;
        }) != null;
        if (added) {
            blockToIsland.put(pos, targetId);
            markDirty(targetId); // ALWAYS_DIRTY — merged block, must re-solve
        }
        return targetId;
    }

    // ═══════════════════════════════════════════════════════
    //  破壞：移除方塊 + 可能分裂 island
    // ═══════════════════════════════════════════════════════

    /**
     * 註銷被破壞的 RBlock。
     * 移除後檢查鄰居是否仍然連通，必要時分裂 island。
     *
     * @param level 伺服器世界（用於 BFS 驗證連通性）
     * @param pos   被破壞方塊位置
     * @param epoch 當前結構 epoch
     * @return 操作後仍存在的所有 island ID（原 island + 分裂出的新 island）；若 island 消失則空列表
     */
    public static List<Integer> unregisterBlock(ServerLevel level, BlockPos pos, long epoch) {
        // Mirror into topology first; the orchestrator treats AIR as
        // "not part of any component" and will drop the voxel from the
        // next tick's BFS.
        topology.setVoxel(pos.getX(), pos.getY(), pos.getZ(), TopologicalSVDAG.TYPE_AIR);
        // PNSM shadow mirror — runs unconditionally so a remove of a
        // pos that was never inserted still leaves both sides in sync
        // (the shadow's compute() drops empty leaves cleanly).
        com.blockreality.api.physics.pnsm.PNSMShadow.mirrorRemove(pos);

        Integer removedIslandId = blockToIsland.remove(pos);
        if (removedIslandId == null) return Collections.emptyList();

        StructureIsland island = islands.get(removedIslandId);
        if (island == null) return Collections.emptyList();

        island.removeMember(pos);

        if (island.getBlockCount() == 0) {
            islands.remove(removedIslandId);
            LOGGER.debug("[IslandRegistry] Island {} removed (empty)", removedIslandId);
            return Collections.emptyList();
        }
        markDirty(removedIslandId); // ALWAYS_DIRTY — block removed, must re-solve

        // Whether or not the removal looks like it could fragment the
        // island, delegate to the LabelPropagation-backed split path,
        // which (a) finds all connected components in one pass,
        // (b) classifies each as anchored / orphan using the registry's
        // anchor set, (c) notifies the orphan listener for any orphan
        // fragment in the same tick as the fracture. This is the
        // correctness fix for the "floating blocks persist for several
        // ticks" symptom reported by the user.
        return checkAndSplitIsland(island, removedIslandId, epoch, level, true);
    }

    /**
     * Re-run anchor/orphan classification for an island whose surrounding
     * world support changed without directly adding/removing one of the island's
     * own voxels.
     */
    public static List<Integer> refreshAnchorState(@Nullable ServerLevel level, int islandId, long epoch) {
        StructureIsland island = islands.get(islandId);
        if (island == null || island.getBlockCount() == 0) {
            return Collections.emptyList();
        }
        markDirty(islandId);
        return checkAndSplitIsland(island, islandId, epoch, level, true);
    }

    // ═══════════════════════════════════════════════════════
    //  查詢
    // ═══════════════════════════════════════════════════════

    /** 取得方塊所屬的 island ID（不存在回傳 -1） */
    public static int getIslandId(BlockPos pos) {
        Integer id = blockToIsland.get(pos);
        return id != null ? id : -1;
    }

    /** 取得 island 資訊 */
    public static StructureIsland getIsland(int islandId) {
        return islands.get(islandId);
    }

    /** 取得所有 island */
    public static Map<Integer, StructureIsland> getAllIslands() {
        return Collections.unmodifiableMap(new HashMap<>(islands));
    }

    /** 取得已登錄的方塊總數 */
    public static int getTotalRegisteredBlocks() {
        return blockToIsland.size();
    }

    /**
     * Number of anchor voxels currently registered. Cheap (set size)
     * so callable from status / HUD paths without a scan.
     */
    public static int getAnchorCount() {
        return anchorBlocks.size();
    }

    /**
     * Snapshot of every currently-registered voxel. Only intended for
     * diagnostic consumers (the PNSM shadow-mode diff, /br status) —
     * each call copies the keySet into a new HashSet, so do not call
     * this in the physics hot loop. Returning a copy rather than the
     * live keySet keeps the caller safe against concurrent mutation
     * from the server tick thread while iterating.
     */
    public static java.util.Set<BlockPos> snapshotRegisteredVoxels() {
        return new java.util.HashSet<>(blockToIsland.keySet());
    }

    /** 取得 island 數量 */
    public static int getIslandCount() {
        return islands.size();
    }

    // ═══════════════════════════════════════════════════════
    //  生命週期
    // ═══════════════════════════════════════════════════════

    /**
     * 標記指定 island 為 dirty，觸發物理重算。
     *
     * <p>Epoch 設計說明（雙軌 epoch）：
     * <ul>
     *   <li>{@code touch(counter)} — 正常生命週期更新，counter 來自 ConnectivityCache
     *       的遞增計數器（通常在百至千的數量級）。
     *   <li>{@code touch(ALWAYS_DIRTY)} — 強制標記為 dirty，使用 Long.MAX_VALUE 哨兵值，
     *       確保 {@code getDirtyIslands(sinceEpoch)} 的比較 {@code lastModifiedEpoch > sinceEpoch}
     *       永遠成立（Long.MAX_VALUE > 任何 counter 值）。
     * </ul>
     * 舊版錯誤地使用 System.nanoTime()（≈10¹⁸），雖然偶然正確（nanotime >> counter），
     * 但語義混亂且在極端情況下可能溢位比較。改為明確的哨兵值。
     *
     * @param islandId island ID
     */
    public static void markDirty(int islandId) {
        StructureIsland island = islands.get(islandId);
        if (island != null) {
            island.touch(ALWAYS_DIRTY); // Long.MAX_VALUE sentinel — 永遠比任何 counter epoch 大
            // Stamp the topology-clock mutation tick too so consumers
            // using topologicallyClean(id, sinceTick) see the change.
            // We read currentTick without waiting for the next
            // advanceTopology call because voxel mutation happens
            // before tick() runs and should be observable immediately.
            lastMutationTick.put(islandId, topology.getTracker().currentTick());
        }
    }

    /** Sentinel epoch：保證 {@code lastModifiedEpoch > sinceEpoch} 永遠成立。 */
    private static final long ALWAYS_DIRTY = Long.MAX_VALUE;

    /**
     * 取得自指定 epoch 後有變化（dirty）的 island 集合。
     * PFSF 引擎每 tick 呼叫此方法取得待計算的島嶼。
     */
    public static java.util.Map<Integer, StructureIsland> getDirtyIslands(long sinceEpoch) {
        java.util.Map<Integer, StructureIsland> dirty = new java.util.LinkedHashMap<>();
        for (var entry : islands.entrySet()) {
            if (entry.getValue().lastModifiedEpoch > sinceEpoch) {
                dirty.put(entry.getKey(), entry.getValue());
            }
        }
        return dirty;
    }

    /**
     * 標記 island 為已處理（重置 dirty 狀態）。
     * PFSF 引擎完成計算後呼叫。
     */
    public static void markProcessed(int islandId) {
        StructureIsland island = islands.get(islandId);
        if (island != null) {
            island.touch(0L); // 重置 epoch 以避免重複處理
        }
    }

    /** 清除所有登錄（世界卸載時呼叫） */
    public static void clear() {
        blockToIsland.clear();
        islands.clear();
        pendingDestructions.clear(); // P2-C: 清除批次緩衝，避免跨世界殘留
        topology.reset();
        fingerprintToIntId.clear();
        intIdToFingerprint.clear();
        lastMutationTick.clear();
        com.blockreality.api.physics.pnsm.PNSMShadow.reset();
        LOGGER.info("[IslandRegistry] Cleared all islands");
    }

    /**
     * Advance the topology v2 pipeline by one tick.
     *
     * <p>Called once per server tick by {@code ServerTickHandler} right
     * after {@link #flushDestructions()} has applied all block changes
     * for this tick. The orchestrator runs its BFS partition + Elder
     * Rule update and fires {@link OrphanIslandEvent}s through the
     * existing {@link #orphanListener} for any component that lost
     * anchor connectivity this tick.
     *
     * <p>Safe to call on worlds without any registered anchors: the
     * orchestrator simply reports every component as orphan, but the
     * installed {@link com.blockreality.api.physics.topology.ThreeTierOrchestrator.OrphanSink}
     * gates on {@link #anchorBlocks} non-emptiness the same way the
     * legacy classifier did — so we never flood the collapse manager
     * before anchor registration is wired from chunk-load.
     */
    public static void advanceTopology(long epoch) {
        if (anchorBlocks.isEmpty()) {
            // Without anchors every component looks orphan; honour the
            // legacy safety-valve behaviour instead of spamming the
            // collapse queue. Still drain the dirty set so the tracker
            // stays in sync for the moment anchors do come online.
            topology.getSvdag().drainDirtyRegions();
            return;
        }
        topology.tick();
        reconcileFingerprintMap();
    }

    /**
     * Walk the orchestrator's last-tick component bindings and refresh
     * the fingerprint↔int maps so every int island observed by this
     * registry has an Elder-Rule fingerprint attached. Any fingerprint
     * whose component has dissolved this tick drops out; any int id
     * whose underlying voxels all vanished likewise drops from the
     * reverse map. Runs on the server tick thread only; the maps
     * themselves are ConcurrentHashMap so readers on other threads see
     * a consistent snapshot per entry.
     */
    private static void reconcileFingerprintMap() {
        // Build the new forward mapping from this tick's bindings.
        java.util.Map<Long, Integer> freshForward = new HashMap<>();
        for (ThreeTierOrchestrator.ComponentBinding b : topology.getLastComponentBindings()) {
            long fp = b.identity().fingerprint();
            int islandId = -1;
            for (BlockPos p : b.voxels()) {
                Integer id = blockToIsland.get(p);
                if (id != null) { islandId = id; break; }
            }
            if (islandId >= 0) {
                freshForward.put(fp, islandId);
            }
        }
        // Replace both maps atomically-per-entry: drop stale, refresh live.
        fingerprintToIntId.keySet().retainAll(freshForward.keySet());
        fingerprintToIntId.putAll(freshForward);
        intIdToFingerprint.clear();
        for (var e : freshForward.entrySet()) {
            intIdToFingerprint.put(e.getValue(), e.getKey());
        }
    }

    // ═══════════════════════════════════════════════════════
    //  P2-C: 批次破壞 — 延遲 BFS，同 tick 多塊合一次連通性檢查
    // ═══════════════════════════════════════════════════════

    /**
     * 本 tick 待移除的方塊緩衝（pos → 最大 epoch）。
     * 爆炸等批量事件可呼叫 {@link #queueBlockDestruction} 而非 {@link #unregisterBlock}，
     * 由 {@link #flushDestructions} 統一處理，降低 BFS 次數（N 塊 → 每 island 1 次）。
     */
    private static final ConcurrentHashMap<BlockPos, Long> pendingDestructions = new ConcurrentHashMap<>();

    /**
     * 將方塊排入批次破壞佇列。不立即執行 BFS。
     * 同一位置多次排入時保留最大 epoch。
     *
     * @param pos   方塊位置
     * @param epoch 當前結構 epoch
     */
    public static void queueBlockDestruction(BlockPos pos, long epoch) {
        pendingDestructions.merge(pos, epoch, Math::max);
    }

    /**
     * 處理所有排入的批次破壞事件。
     * 同一 island 的多個移除合併為一次 BFS，降低爆炸場景的 TPS 衝擊。
     *
     * <p>由 ServerTickHandler.onServerTick() 在物理引擎運行前呼叫。</p>
     */
    public static void flushDestructions() {
        if (pendingDestructions.isEmpty()) return;

        // 快照 + 清空緩衝（允許本 tick 繼續排入下一個緩衝週期）
        Map<BlockPos, Long> snapshot = new HashMap<>(pendingDestructions);
        pendingDestructions.clear();

        // Mirror the full batch into topology up front so a single
        // advanceTopology() call at the end of the tick sees every
        // removal at once rather than piecewise.
        for (BlockPos p : snapshot.keySet()) {
            topology.setVoxel(p.getX(), p.getY(), p.getZ(), TopologicalSVDAG.TYPE_AIR);
        }

        // 按 island 分組
        Map<Integer, List<BlockPos>> byIsland = new HashMap<>();
        for (BlockPos pos : snapshot.keySet()) {
            Integer islandId = blockToIsland.get(pos);
            if (islandId != null) {
                byIsland.computeIfAbsent(islandId, k -> new java.util.ArrayList<>()).add(pos);
            }
        }

        // 每個受影響的 island 只做一次 BFS
        for (Map.Entry<Integer, List<BlockPos>> entry : byIsland.entrySet()) {
            int islandId = entry.getKey();
            List<BlockPos> removals = entry.getValue();
            long epoch = removals.stream()
                .mapToLong(p -> snapshot.getOrDefault(p, 0L))
                .max().orElse(0L);

            // 從追蹤移除所有方塊
            for (BlockPos pos : removals) {
                blockToIsland.remove(pos);
                com.blockreality.api.physics.pnsm.PNSMShadow.mirrorRemove(pos);
            }

            StructureIsland island = islands.get(islandId);
            if (island == null) continue;

            for (BlockPos pos : removals) {
                island.removeMember(pos);
            }

            if (island.getBlockCount() == 0) {
                islands.remove(islandId);
                LOGGER.debug("[IslandRegistry] Island {} removed (batch destruction, {} blocks)",
                    islandId, removals.size());
                continue;
            }

            markDirty(islandId); // ALWAYS_DIRTY — batch block removal, must re-solve
            // 單次連通性 BFS，取代原本 N 次獨立 BFS
            checkAndSplitIsland(island, islandId, epoch);
        }
    }

    /**
     * 連通性檢查 + 必要時分裂 island 的私有輔助方法。
     * 從任意成員 BFS，不可達成員建立新 island。
     * 由 {@link #unregisterBlock} 和 {@link #flushDestructions} 共用。
     *
     * @param island   已移除方塊後的 island（成員集合已更新）
     * @param islandId island ID
     * @param epoch    本次操作的 epoch
     * @return 操作後仍存在的所有 island ID（原 island + 分裂出的新 island）
     */
    /**
     * Recompute the island's connectivity using {@link LabelPropagation}
     * and, if the set has fragmented, create new islands for each
     * disconnected component.
     *
     * <p>The critical correctness improvement over the prior single-seed
     * BFS: each resulting component is independently classified as
     * anchored or orphan by consulting {@link #anchorBlocks}. Any
     * orphan component is reported to {@link #orphanListener}
     * <b>in the same tick as the fracture</b>, so CollapseManager can
     * collapse the fragment immediately instead of waiting for the
     * PFSF potential field to diverge past {@code PHI_ORPHAN_THRESHOLD}
     * over the next several ticks — that delay is the "floating-block
     * ghost" documented in the GPU audit.
     */
    private static List<Integer> checkAndSplitIsland(StructureIsland island, int islandId, long epoch) {
        return checkAndSplitIsland(island, islandId, epoch, null);
    }

    private static List<Integer> checkAndSplitIsland(StructureIsland island,
                                                     int islandId,
                                                     long epoch,
                                                     @Nullable ServerLevel level) {
        return checkAndSplitIsland(island, islandId, epoch, level, false);
    }

    private static List<Integer> checkAndSplitIsland(StructureIsland island,
                                                     int islandId,
                                                     long epoch,
                                                     @Nullable ServerLevel level,
                                                     boolean allowAnchorlessClassification) {
        if (island.getBlockCount() == 0) {
            islands.remove(islandId);
            return Collections.emptyList();
        }
        if (island.getBlockCount() == 1) {
            island.recalculateBounds();
            BlockPos only = island.members.iterator().next();
            boolean anchored = isBlockAnchored(only);
            if (!anchored && orphanClassificationEnabled(allowAnchorlessClassification)) {
                notifyOrphan(islandId, Set.of(only), epoch, level);
            }
            return Collections.singletonList(islandId);
        }

        // Authoritative partition: SV-equivalent BFS over the island's
        // current member set with anchor-adjacency check folded in.
        LabelPropagation.PartitionResult partition = LabelPropagation.bfsComponents(
                new HashSet<>(island.members),
                anchorBlocks,
                LabelPropagation.NeighborPolicy.FACE_6);

        if (partition.components().size() == 1) {
            // Still a single connected component. Update AABB; if the
            // whole component is now orphan (e.g. its only anchor was
            // just removed), surface it to the orphan listener.
            island.recalculateBounds();
            LabelPropagation.Component sole = partition.components().get(0);
            if (!sole.anchored() && orphanClassificationEnabled(allowAnchorlessClassification)) {
                notifyOrphan(islandId, sole.members(), epoch, level);
            }
            return Collections.singletonList(islandId);
        }

        // Fragmented: pick one component to keep in the original island.
        // Elder Rule first — the fragment that still contains the
        // island's birth voxel inherits the int id, so downstream
        // systems (CollapseJournal, persistence) keep seeing the same
        // id for the "same" structure even after repeated splits.
        // When the birth voxel is absent (it was the voxel that just
        // broke, or the island lost it to an earlier split), fall back
        // to the legacy "prefer anchored, then largest" rule so we
        // still degrade gracefully.
        int keepIdx = -1;
        BlockPos elder = island.birthVoxel;
        if (elder != null) {
            for (int i = 0; i < partition.components().size(); i++) {
                if (partition.components().get(i).members().contains(elder)) {
                    keepIdx = i;
                    break;
                }
            }
        }
        if (keepIdx < 0) {
            keepIdx = 0;
            boolean keepAnchored = partition.components().get(0).anchored();
            int keepSize = partition.components().get(0).members().size();
            for (int i = 1; i < partition.components().size(); i++) {
                LabelPropagation.Component c = partition.components().get(i);
                boolean better = (c.anchored() && !keepAnchored)
                        || (c.anchored() == keepAnchored && c.members().size() > keepSize);
                if (better) {
                    keepIdx = i;
                    keepAnchored = c.anchored();
                    keepSize = c.members().size();
                }
            }
        }

        LabelPropagation.Component keep = partition.components().get(keepIdx);
        island.members.retainAll(keep.members());
        island.recalculateBounds();
        markDirty(islandId);

        List<Integer> resultIds = new java.util.ArrayList<>();
        resultIds.add(islandId);
        if (!keep.anchored() && orphanClassificationEnabled(allowAnchorlessClassification)) {
            // Even the "kept" component is orphan — tell CollapseManager.
            notifyOrphan(islandId, keep.members(), epoch, level);
        }

        for (int i = 0; i < partition.components().size(); i++) {
            if (i == keepIdx) continue;
            LabelPropagation.Component c = partition.components().get(i);
            int newId = nextIslandId.getAndIncrement();
            StructureIsland newIsland = new StructureIsland(newId);
            for (BlockPos p : c.members()) {
                newIsland.addMember(p);
                blockToIsland.put(p, newId);
            }
            newIsland.touch(epoch);
            islands.put(newId, newIsland);
            markDirty(newId);
            resultIds.add(newId);
            LOGGER.info("[IslandRegistry] Split: new island {} with {} blocks from island {} (anchored={})",
                    newId, c.members().size(), islandId, c.anchored());
            if (!c.anchored() && orphanClassificationEnabled(allowAnchorlessClassification)) {
                notifyOrphan(newId, c.members(), epoch, level);
            }
        }
        return resultIds;
    }

    /**
     * True iff {@code pos} itself is a registered anchor, or a face-
     * neighbour is. Matches the semantics of
     * {@link LabelPropagation#bfsComponents} with
     * {@link LabelPropagation.NeighborPolicy#FACE_6}.
     */
    /**
     * Gate that surrounds every {@code notifyOrphan} call in
     * {@link #checkAndSplitIsland}. When {@link #anchorBlocks} is
     * empty — the current production state — every component appears
     * orphan to the classifier and CollapseManager gets flooded with
     * spurious entries on every block break. Returning false here
     * silences orphan notification until anchor registration is wired.
     * Logs the first trigger per JVM so operators can tell the valve
     * is in effect.
     */
    private static boolean orphanClassificationEnabled(boolean allowAnchorlessClassification) {
        if (allowAnchorlessClassification) return true;
        if (!anchorBlocks.isEmpty()) return true;
        if (SAFETY_VALVE_LOGGED.compareAndSet(false, true)) {
            LOGGER.warn(
                "[IslandRegistry] Safety valve active: anchorBlocks is empty; "
                + "neither the legacy BFS classifier nor the topology v2 "
                + "orchestrator (now wired in advanceTopology) will fire "
                + "OrphanIslandEvent until at least one anchor is registered "
                + "via registerAnchor(...). Chunk-load natural-anchor "
                + "registration is the outstanding wiring task."
            );
        }
        return false;
    }

    private static boolean isBlockAnchored(BlockPos pos) {
        if (anchorBlocks.contains(pos)) return true;
        for (Direction d : Direction.values()) {
            if (anchorBlocks.contains(pos.relative(d))) return true;
        }
        return false;
    }

    private static void notifyOrphan(int islandId,
                                     Set<BlockPos> members,
                                     long epoch,
                                     @Nullable ServerLevel level) {
        Consumer<OrphanIslandEvent> listener = orphanListener;
        if (listener == null) {
            // No-fallback contract: production has no orphan listener. Topology
            // BFS still runs (its dirty-region tracking serves other purposes)
            // but the result is dropped — orphan detection now happens on the
            // GPU via failure_scan's φ_orphan threshold. Logged at debug only;
            // in production every fracture would otherwise spam this line.
            LOGGER.debug("[IslandRegistry] Orphan island {} ({} blocks) detected; PFSF will collapse via φ divergence",
                    islandId, members.size());
            return;
        }
        try {
            listener.accept(new OrphanIslandEvent(islandId, Set.copyOf(members), epoch, level));
        } catch (Throwable t) {
            LOGGER.error("[IslandRegistry] Orphan listener threw for island " + islandId, t);
        }
    }

    /** 診斷用統計資訊 */
    public static String getStats() {
        return String.format("islands=%d, totalBlocks=%d",
            islands.size(), blockToIsland.size());
    }
}
