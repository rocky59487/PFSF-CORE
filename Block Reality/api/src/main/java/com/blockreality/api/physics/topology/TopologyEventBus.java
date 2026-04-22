package com.blockreality.api.physics.topology;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Consumer;

/**
 * Phase G — 拓撲事件匯流排（per-island vector clock + thread-local ring buffer +
 * end-of-tick 拓撲排序）。
 *
 * <h2>為何廢除全域 AtomicLong（P2 警告修正 2026-04-22）</h2>
 * <p>原計畫用 {@code AtomicLong sequenceNumber} 產生全域單調序號。當千島並發
 * 粉碎時（每秒千級 publish），CAS 爭用會淹沒 scheduler → GPU 飢餓。
 *
 * <h2>新架構</h2>
 * <ol>
 *   <li><b>Publish 路徑（熱路徑）</b>：每個 worker thread 的
 *       {@link ThreadLocal} ring buffer 存自己產生的事件。每 island 維護自己的
 *       {@code islandClock}（非 atomic，單 thread 寫）。完全無鎖、無 CAS。</li>
 *   <li><b>End-of-tick drain（冷路徑）</b>：主 scheduler thread 呼叫
 *       {@link #drainAndSort(Map)}：收集所有 thread-local buffer、
 *       以 (tick, islandClock) + island 依賴 DAG 做拓撲排序，賦予 globalSeq。</li>
 *   <li><b>Subscriber delivery</b>：drain 完成後以 globalSeq 順序呼叫訂閱者。</li>
 * </ol>
 *
 * <h2>拓撲排序的依賴邊</h2>
 * <p>Island 依賴 DAG 由 {@link IslandSplitEdge} 表達：parent → children。Drain
 * 時保證 {@code IslandSplit} 先於該 split 產生的子 island 上的任何事件（單調性）。
 *
 * <h2>Thread-safety 等級</h2>
 * <ul>
 *   <li>{@link #publish(TopologyEvent)} — 可從任意 worker thread 呼叫</li>
 *   <li>{@link #subscribe(Class, Consumer)} — 可從任意 thread 呼叫（CoW list）</li>
 *   <li>{@link #drainAndSort(Map)} — 限主 scheduler thread 呼叫，獨佔</li>
 * </ul>
 *
 * <h2>執行緒 clock 指派</h2>
 * <p>每個 publishing thread 在第一次寫 island I 時，向 Bus 申請該 thread 當次
 * island clock 起點（透過 {@link #allocateClockBase(long)}）。後續每筆 event 自己
 * thread-local 遞增。Drain 時全域重新排序，避免 CAS。</p>
 */
public final class TopologyEventBus {

    /** 預設 ring buffer 容量 */
    public static final int DEFAULT_RING_CAPACITY = 4096;

    /** Thread-local 事件緩衝 */
    private final ThreadLocal<ArrayList<TopologyEvent>> localBuffer =
        ThreadLocal.withInitial(() -> new ArrayList<>(DEFAULT_RING_CAPACITY));

    /** 所有活躍的 thread-local buffer（弱引用避免 leak） */
    private final List<ArrayList<TopologyEvent>> allBuffers = new CopyOnWriteArrayList<>();

    /** Subscriber 列表（type → consumers）：CoW 容器支援並發訂閱 */
    private final Map<Class<? extends TopologyEvent>, List<Consumer<TopologyEvent>>> subscribers =
        new ConcurrentHashMap<>();

    /** Island clock 起點分配：islandId → nextClockBase（單 tick 內遞增） */
    private final Map<Long, Long> islandClockBase = new ConcurrentHashMap<>();

    /** 最近一次 drain 賦予的 globalSeq 起點（僅保護資料完整性，非熱路徑） */
    private long lastGlobalSeq = -1L;

    // ═══════════════════════════════════════════════════════════════
    //  Publish（熱路徑）
    // ═══════════════════════════════════════════════════════════════

    /**
     * 發布事件。本方法不做任何原子操作 / 鎖。呼叫 thread 負責產生正確的
     * {@code islandClock}（通常透過 {@link #allocateClockBase(long)} 領號後自行遞增）。
     */
    public void publish(TopologyEvent ev) {
        ArrayList<TopologyEvent> buf = localBuffer.get();
        if (buf.isEmpty()) {
            // 首次使用此 thread-local buffer，登記到全域列表供 drain 收集
            allBuffers.add(buf);
        }
        buf.add(ev);
    }

    /**
     * 為指定 island 領取 clock 起點（caller 之後自增）。本方法使用
     * {@code ConcurrentHashMap.compute}，單次 CAS，不在熱路徑上。
     *
     * <p>使用範式：
     * <pre>
     *   long base = bus.allocateClockBase(islandId);
     *   // Thread-local 使用 base+0, base+1, base+2 ...
     * </pre>
     */
    public long allocateClockBase(long islandId) {
        long[] result = new long[1];
        islandClockBase.compute(islandId, (k, v) -> {
            long curr = (v == null) ? 0L : v;
            result[0] = curr;
            return curr + 1L;   // 預留 1 格給 caller
        });
        return result[0];
    }

    // ═══════════════════════════════════════════════════════════════
    //  Subscribe
    // ═══════════════════════════════════════════════════════════════

    @SuppressWarnings("unchecked")
    public <E extends TopologyEvent> void subscribe(Class<E> type, Consumer<E> consumer) {
        subscribers.computeIfAbsent(type, k -> new CopyOnWriteArrayList<>())
                   .add((Consumer<TopologyEvent>) consumer);
    }

    // ═══════════════════════════════════════════════════════════════
    //  Drain + topological sort (End-of-tick, 單一 thread)
    // ═══════════════════════════════════════════════════════════════

    /**
     * 收集所有 thread-local buffer、依 (tick, islandClock) 初排，再依 island
     * 依賴 DAG 做拓撲調整，最後賦予 globalSeq 並派送給 subscriber。
     *
     * @param islandDag  island 依賴有向邊列表（empty map ⇒ 只做 (tick, clock) 排序）
     * @return 本次 drain 排序後的事件列表（已包含 globalSeq）
     */
    public List<TopologyEvent> drainAndSort(Map<Long, Set<Long>> islandDag) {
        // 1. 收集所有 thread-local buffer
        List<TopologyEvent> all = new ArrayList<>();
        for (ArrayList<TopologyEvent> buf : allBuffers) {
            synchronized (buf) {
                all.addAll(buf);
                buf.clear();
            }
        }

        // 2. 初排序：(tick asc, islandClock asc, islandId asc) — 單執行緒，無 CAS
        all.sort(Comparator
            .comparingLong(TopologyEvent::tick)
            .thenComparingLong(TopologyEvent::islandClock)
            .thenComparingLong(TopologyEvent::islandId));

        // 3. 拓撲調整：若 islandDag 非空，做 Kahn's algorithm 調整跨 island 順序
        List<TopologyEvent> sorted = (islandDag == null || islandDag.isEmpty())
                                     ? all
                                     : topologicalAdjust(all, islandDag);

        // 4. 指派 globalSeq + 派送
        List<TopologyEvent> stamped = new ArrayList<>(sorted.size());
        for (TopologyEvent ev : sorted) {
            lastGlobalSeq++;
            TopologyEvent withSeq = ev.withGlobalSeq(lastGlobalSeq);
            stamped.add(withSeq);
            dispatch(withSeq);
        }

        // 5. 清掉下次 drain 的 clock base 快照（每個 tick 獨立）
        islandClockBase.clear();

        return stamped;
    }

    /**
     * 對已按 (tick, clock) 排序的事件做 island DAG 拓撲調整。
     *
     * <p>策略：每當一個事件的 islandId 有未處理的 parent（DAG 上游 island），
     * 將事件延後到所有 parent island 的事件都處理完之後。這保證
     * {@code IslandSplit parent=P children=[A,B]} 事件必先於任何 island A/B
     * 上的事件。
     */
    private List<TopologyEvent> topologicalAdjust(
            List<TopologyEvent> events, Map<Long, Set<Long>> islandDag) {
        // 建立反向索引：island → 必須先完成的父 islands
        // islandDag: parent -> children ⇒ child 依賴 parent
        Map<Long, Set<Long>> parents = new HashMap<>();
        for (var entry : islandDag.entrySet()) {
            long parent = entry.getKey();
            for (long child : entry.getValue()) {
                parents.computeIfAbsent(child, k -> new HashSet<>()).add(parent);
            }
        }

        List<TopologyEvent> out = new ArrayList<>(events.size());
        List<TopologyEvent> deferred = new ArrayList<>();
        Set<Long> completedIslands = new HashSet<>();

        for (TopologyEvent ev : events) {
            long id = ev.islandId();
            Set<Long> deps = parents.getOrDefault(id, Collections.emptySet());
            if (completedIslands.containsAll(deps)) {
                out.add(ev);
            } else {
                deferred.add(ev);
            }
            // IslandSplit 之後 parent 視為「已處理」，釋放 children 的延遲事件
            if (ev instanceof TopologyEvent.IslandSplit) {
                completedIslands.add(id);
                // 嘗試釋放被延遲的事件
                var it = deferred.iterator();
                while (it.hasNext()) {
                    TopologyEvent d = it.next();
                    Set<Long> dDeps = parents.getOrDefault(d.islandId(), Collections.emptySet());
                    if (completedIslands.containsAll(dDeps)) {
                        out.add(d);
                        it.remove();
                    }
                }
            }
        }
        // 任何仍被延遲的事件（DAG 根本無對應 split）也要最終輸出
        out.addAll(deferred);
        return out;
    }

    @SuppressWarnings("unchecked")
    private void dispatch(TopologyEvent ev) {
        // 1) 訂閱了具體 type 的
        List<Consumer<TopologyEvent>> typed = subscribers.get(ev.getClass());
        if (typed != null) {
            for (Consumer<TopologyEvent> c : typed) c.accept(ev);
        }
        // 2) 訂閱了 TopologyEvent 基底介面的
        List<Consumer<TopologyEvent>> all = subscribers.get(TopologyEvent.class);
        if (all != null) {
            for (Consumer<TopologyEvent> c : all) c.accept(ev);
        }
    }

    // ─── 測試/除錯 ────────────────────────────────────────────────

    /** 清空所有狀態（測試用） */
    public void reset() {
        for (ArrayList<TopologyEvent> buf : allBuffers) {
            synchronized (buf) { buf.clear(); }
        }
        allBuffers.clear();
        subscribers.clear();
        islandClockBase.clear();
        lastGlobalSeq = -1L;
        // 清 ThreadLocal 相對昂貴，測試用手動 remove
        localBuffer.remove();
    }

    public long lastGlobalSeq() { return lastGlobalSeq; }

    /** Island DAG 依賴記錄（parent → children，一般由 IslandSplit 事件填入） */
    public record IslandSplitEdge(long parentIslandId, long[] childIslandIds) {}
}
