package com.blockreality.api.physics.topology;

import com.blockreality.api.physics.FailureType;
import net.minecraft.core.BlockPos;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase G — {@link TopologyEventBus} 順序與確定性驗證。
 *
 * <p>100 組隨機崩塌情境驗證：
 * <ol>
 *   <li>drain 後 globalSeq 嚴格遞增</li>
 *   <li>相同輸入下兩次 drain 得相同結果（replay determinism）</li>
 *   <li>IslandSplit 發生後，子 island 的事件必排在後面（拓撲正確性）</li>
 * </ol>
 */
class TopologyEventOrderingTest {

    @Test
    void drainProducesMonotonicGlobalSeq() {
        TopologyEventBus bus = new TopologyEventBus();
        // 發佈 20 筆事件於 3 個 island
        for (int i = 0; i < 20; i++) {
            long island = i % 3L + 1L;
            long clockBase = bus.allocateClockBase(island);
            bus.publish(new TopologyEvent.SupportLost(
                100L, clockBase, -1L, island, BlockPos.ZERO, FailureType.NO_SUPPORT
            ));
        }
        List<TopologyEvent> sorted = bus.drainAndSort(Collections.emptyMap());
        assertEquals(20, sorted.size());
        long prev = -2L;
        for (TopologyEvent ev : sorted) {
            assertTrue(ev.globalSeq() > prev,
                "globalSeq 應嚴格遞增：prev=" + prev + " curr=" + ev.globalSeq());
            prev = ev.globalSeq();
        }
    }

    @Test
    void replayDeterminism_identicalRunsProduceIdenticalSequences() {
        Random r = new Random(0xDEADBEEFL);
        List<EventBlueprint> script = buildRandomScript(r, 100);

        List<Long> run1 = executeAndCollect(script);
        List<Long> run2 = executeAndCollect(script);

        assertEquals(run1, run2,
            "相同 script 兩次執行的 globalSeq 序列應完全一致（replay determinism）");
    }

    @Test
    void islandSplitDagRespected_childEventsAfterSplit() {
        TopologyEventBus bus = new TopologyEventBus();

        // parent island = 1, children = [10, 11]
        // 發佈時故意先發 child events (clock 較小) 再發 split (clock 較大)
        // topological adjust 應把 split 拉到所有 child events 之前
        long childBase1 = bus.allocateClockBase(10L);
        long childBase2 = bus.allocateClockBase(11L);
        long splitBase  = bus.allocateClockBase(1L);

        bus.publish(new TopologyEvent.SupportLost(
            50L, childBase1, -1L, 10L, BlockPos.ZERO, FailureType.NO_SUPPORT));
        bus.publish(new TopologyEvent.SupportLost(
            50L, childBase2, -1L, 11L, BlockPos.ZERO, FailureType.NO_SUPPORT));
        bus.publish(new TopologyEvent.IslandSplit(
            50L, splitBase, -1L, 1L, new long[]{10L, 11L}));

        // DAG: parent 1 -> children {10, 11}
        Map<Long, Set<Long>> dag = new HashMap<>();
        dag.put(1L, new HashSet<>(List.of(10L, 11L)));

        List<TopologyEvent> sorted = bus.drainAndSort(dag);
        assertEquals(3, sorted.size());

        // IslandSplit 必出現在所有 island 10 / 11 事件之前
        int splitIdx = -1;
        int maxChildIdx = -1;
        for (int i = 0; i < sorted.size(); i++) {
            TopologyEvent ev = sorted.get(i);
            if (ev instanceof TopologyEvent.IslandSplit) splitIdx = i;
            else if (ev.islandId() == 10L || ev.islandId() == 11L) maxChildIdx = Math.max(maxChildIdx, i);
        }
        assertTrue(splitIdx >= 0, "找不到 IslandSplit 事件");
        assertTrue(maxChildIdx > splitIdx,
            "IslandSplit 必須在所有 child island 事件之前 (splitIdx=" + splitIdx +
            ", maxChildIdx=" + maxChildIdx + ")");
    }

    @Test
    void bufferClearedAfterDrain() {
        TopologyEventBus bus = new TopologyEventBus();
        bus.publish(new TopologyEvent.SupportLost(1L, 0L, -1L, 1L, BlockPos.ZERO, FailureType.NO_SUPPORT));
        bus.drainAndSort(Collections.emptyMap());

        List<TopologyEvent> second = bus.drainAndSort(Collections.emptyMap());
        assertEquals(0, second.size(), "drain 後 buffer 必須被清空");
    }

    @Test
    void sameTickSameClockSameIsland_deterministicTieBreakByIslandId() {
        // 刻意製造 tie：兩事件同 tick 同 clock，但不同 islandId
        // 排序規則：islandId asc
        TopologyEventBus bus = new TopologyEventBus();
        bus.publish(new TopologyEvent.SupportLost(10L, 5L, -1L, 7L, BlockPos.ZERO, FailureType.NO_SUPPORT));
        bus.publish(new TopologyEvent.SupportLost(10L, 5L, -1L, 3L, BlockPos.ZERO, FailureType.NO_SUPPORT));
        bus.publish(new TopologyEvent.SupportLost(10L, 5L, -1L, 5L, BlockPos.ZERO, FailureType.NO_SUPPORT));

        List<TopologyEvent> sorted = bus.drainAndSort(Collections.emptyMap());
        assertEquals(3L, sorted.get(0).islandId());
        assertEquals(5L, sorted.get(1).islandId());
        assertEquals(7L, sorted.get(2).islandId());
    }

    // ═══════════════════════════════════════════════════════════════
    //  Helper：script-based replay test
    // ═══════════════════════════════════════════════════════════════

    private record EventBlueprint(long tick, long island) {}

    private List<EventBlueprint> buildRandomScript(Random r, int n) {
        List<EventBlueprint> list = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            list.add(new EventBlueprint(r.nextInt(100), r.nextInt(10) + 1));
        }
        return list;
    }

    private List<Long> executeAndCollect(List<EventBlueprint> script) {
        TopologyEventBus bus = new TopologyEventBus();
        for (EventBlueprint bp : script) {
            long clock = bus.allocateClockBase(bp.island);
            bus.publish(new TopologyEvent.SupportLost(
                bp.tick, clock, -1L, bp.island, BlockPos.ZERO, FailureType.NO_SUPPORT));
        }
        List<TopologyEvent> sorted = bus.drainAndSort(Collections.emptyMap());
        List<Long> result = new ArrayList<>(sorted.size());
        for (TopologyEvent ev : sorted) {
            // 以 (tick, islandId, islandClock) 作為 signature 驗證 script 執行一致性
            result.add((ev.tick() * 1_000_000L + ev.islandId() * 1000L + ev.islandClock()));
        }
        return result;
    }
}
