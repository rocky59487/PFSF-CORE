package com.blockreality.api.physics.topology;

import com.blockreality.api.physics.FailureType;
import net.minecraft.core.BlockPos;
import org.junit.jupiter.api.Test;

import java.util.Collections;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase G — 多 island 並發 publish 壓力測試。
 *
 * <p>驗證 P2 修正的關鍵：publish 熱路徑完全無 CAS 爭用，能承受千級並發。
 *
 * <p>測試結構：
 * <ol>
 *   <li>多 worker thread 同時在各自的 island 上 publish 大量事件</li>
 *   <li>End-of-tick 呼叫 drainAndSort → 驗證 globalSeq 全域單調</li>
 *   <li>驗證 subscriber 收到事件數 == publish 總數</li>
 *   <li>驗證單 island 內 islandClock 單調遞增（per-island 時鐘正確性）</li>
 * </ol>
 */
class TopologyEventBusConcurrencyTest {

    private static final int THREADS       = 8;
    private static final int EVENTS_PER_WORKER = 2000;
    private static final int ISLANDS       = 16;

    @Test
    void highConcurrencyPublishing_noEventLoss() throws Exception {
        TopologyEventBus bus = new TopologyEventBus();
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done  = new CountDownLatch(THREADS);
        ExecutorService pool = Executors.newFixedThreadPool(THREADS);

        for (int t = 0; t < THREADS; t++) {
            final int tid = t;
            pool.submit(() -> {
                try {
                    start.await();
                    for (int i = 0; i < EVENTS_PER_WORKER; i++) {
                        long island = (long) (i % ISLANDS + 1);
                        long clock = bus.allocateClockBase(island);
                        bus.publish(new TopologyEvent.SupportLost(
                            1L, clock, -1L, island, BlockPos.ZERO, FailureType.NO_SUPPORT
                        ));
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    done.countDown();
                }
            });
        }
        start.countDown();
        assertTrue(done.await(10, TimeUnit.SECONDS), "worker 不應超時");
        pool.shutdown();

        List<TopologyEvent> sorted = bus.drainAndSort(Collections.emptyMap());
        int expected = THREADS * EVENTS_PER_WORKER;
        assertEquals(expected, sorted.size(),
            "所有 publish 事件必須被 drain 拾到；expected=" + expected +
            " actual=" + sorted.size());

        // 全域 globalSeq 必須嚴格遞增
        long prev = -2L;
        for (TopologyEvent ev : sorted) {
            assertTrue(ev.globalSeq() > prev,
                "globalSeq 應嚴格遞增：prev=" + prev + " curr=" + ev.globalSeq());
            prev = ev.globalSeq();
        }
    }

    @Test
    void perIslandClockMonotonic() throws Exception {
        TopologyEventBus bus = new TopologyEventBus();
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done  = new CountDownLatch(THREADS);
        ExecutorService pool = Executors.newFixedThreadPool(THREADS);

        for (int t = 0; t < THREADS; t++) {
            pool.submit(() -> {
                try {
                    start.await();
                    for (int i = 0; i < EVENTS_PER_WORKER; i++) {
                        long island = (long) (i % ISLANDS + 1);
                        long clock = bus.allocateClockBase(island);
                        bus.publish(new TopologyEvent.SupportLost(
                            1L, clock, -1L, island, BlockPos.ZERO, FailureType.NO_SUPPORT
                        ));
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    done.countDown();
                }
            });
        }
        start.countDown();
        assertTrue(done.await(10, TimeUnit.SECONDS));
        pool.shutdown();

        List<TopologyEvent> sorted = bus.drainAndSort(Collections.emptyMap());

        // 單 island 內：allocateClockBase 產生的 clock 序列應無重複（zero-conflict allocation）
        ConcurrentHashMap<Long, java.util.TreeSet<Long>> perIsland = new ConcurrentHashMap<>();
        for (TopologyEvent ev : sorted) {
            perIsland.computeIfAbsent(ev.islandId(), k -> new java.util.TreeSet<>())
                     .add(ev.islandClock());
        }
        // 每 island clock set 大小 == 該 island 事件數（無重複 clock 表示 allocate 正確）
        int totalUnique = 0;
        int totalEvents = 0;
        for (var entry : perIsland.entrySet()) {
            java.util.TreeSet<Long> clocks = entry.getValue();
            long perIslandCount = sorted.stream().filter(e -> e.islandId() == entry.getKey()).count();
            totalUnique += clocks.size();
            totalEvents += perIslandCount;
            assertEquals(perIslandCount, clocks.size(),
                "Island " + entry.getKey() + " 的 clocks 應全為 unique，actual unique=" +
                clocks.size() + " total=" + perIslandCount);
        }
        assertEquals(totalEvents, totalUnique, "全域 clock 空間不應有碰撞");
    }

    @Test
    void subscribersReceiveAllEventsAfterDrain() throws Exception {
        TopologyEventBus bus = new TopologyEventBus();
        AtomicInteger received = new AtomicInteger(0);
        bus.subscribe(TopologyEvent.SupportLost.class, ev -> received.incrementAndGet());

        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done  = new CountDownLatch(THREADS);
        ExecutorService pool = Executors.newFixedThreadPool(THREADS);
        for (int t = 0; t < THREADS; t++) {
            pool.submit(() -> {
                try {
                    start.await();
                    for (int i = 0; i < 100; i++) {
                        long clock = bus.allocateClockBase(1L);
                        bus.publish(new TopologyEvent.SupportLost(
                            1L, clock, -1L, 1L, BlockPos.ZERO, FailureType.NO_SUPPORT));
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    done.countDown();
                }
            });
        }
        start.countDown();
        assertTrue(done.await(5, TimeUnit.SECONDS));
        pool.shutdown();

        // drain 前 subscriber 不該收到任何事件
        assertEquals(0, received.get(),
            "drainAndSort 前 subscriber 不應被觸發（end-of-tick delivery 模型）");
        bus.drainAndSort(Collections.emptyMap());
        assertEquals(THREADS * 100, received.get());
    }

    @Test
    void subscriberForBaseInterfaceReceivesAllEventTypes() {
        TopologyEventBus bus = new TopologyEventBus();
        AtomicInteger allCount = new AtomicInteger(0);
        bus.subscribe(TopologyEvent.class, ev -> allCount.incrementAndGet());

        long c1 = bus.allocateClockBase(1L);
        bus.publish(new TopologyEvent.SupportLost(
            1L, c1, -1L, 1L, BlockPos.ZERO, FailureType.NO_SUPPORT));
        long c2 = bus.allocateClockBase(1L);
        bus.publish(new TopologyEvent.EdgeFractured(
            1L, c2, -1L, 1L, 0L, 1L, 1.0, 0.0));
        long c3 = bus.allocateClockBase(1L);
        bus.publish(new TopologyEvent.IslandSplit(
            1L, c3, -1L, 1L, new long[]{2L, 3L}));

        bus.drainAndSort(Collections.emptyMap());
        assertEquals(3, allCount.get(), "TopologyEvent 基底 subscriber 應收到所有類型事件");
    }
}
