package com.blockreality.api.physics.topology;

import com.blockreality.api.physics.FailureType;
import net.minecraft.core.BlockPos;
import net.minecraft.world.phys.Vec3;

/**
 * Phase G — 拓撲事件匯流排的 sealed event hierarchy。
 *
 * <p>所有體素物理拓撲變更都透過此 sealed interface 表達；實作者為 4 個
 * record：{@link SupportLost}、{@link EdgeFractured}、{@link IslandSplit}、
 * {@link RigidBodyReleased}。
 *
 * <h2>排序欄位（per-island vector clock + end-of-tick global seq）</h2>
 * <ul>
 *   <li>{@link #tick()} — 發生的 server tick</li>
 *   <li>{@link #islandClock()} — 該 island 自己的遞增計數器（非 atomic；
 *       由 island 的 tick thread 獨佔寫）</li>
 *   <li>{@link #globalSeq()} — 由 {@link TopologyEventBus#drainAndSort} 在 tick 結尾
 *       以拓撲排序賦予的全域單調序號；raw publish 時為 -1，drain 後才填入</li>
 * </ul>
 *
 * <p>P2 警告修正（2026-04-22）：廢除原計畫的全域 {@code AtomicLong}。
 * 千島並發破壞時 CAS 會淹沒 scheduler → GPU 飢餓。改用 per-island clock
 * + thread-local buffer + end-of-tick 單執行緒拓撲排序。
 */
public sealed interface TopologyEvent
    permits TopologyEvent.SupportLost,
            TopologyEvent.EdgeFractured,
            TopologyEvent.IslandSplit,
            TopologyEvent.RigidBodyReleased {

    /** 事件發生的 server tick */
    long tick();

    /** 此 island 自己的 vector clock（非 atomic，單 thread 寫） */
    long islandClock();

    /** End-of-tick 拓撲排序後的全域單調序號；raw 狀態為 -1 */
    long globalSeq();

    /** 主要受影響的 island id；IslandSplit 使用 parent */
    long islandId();

    /** 提供給 bus 使用：產生帶 globalSeq 的 copy */
    TopologyEvent withGlobalSeq(long seq);

    // ═══════════════════════════════════════════════════════════════
    //  Record 實作
    // ═══════════════════════════════════════════════════════════════

    /** Island 因失去錨定路徑而喪失支撐。 */
    record SupportLost(
        long tick, long islandClock, long globalSeq,
        long islandId, BlockPos pos, FailureType reason
    ) implements TopologyEvent {
        @Override public TopologyEvent withGlobalSeq(long seq) {
            return new SupportLost(tick, islandClock, seq, islandId, pos, reason);
        }
    }

    /** 某條邊（兩體素間耦合）因相場損傷 d 達臨界而斷裂（w_ij 降至 0）。 */
    record EdgeFractured(
        long tick, long islandClock, long globalSeq,
        long islandId, long i, long j, double wBefore, double wAfter
    ) implements TopologyEvent {
        @Override public TopologyEvent withGlobalSeq(long seq) {
            return new EdgeFractured(tick, islandClock, seq, islandId, i, j, wBefore, wAfter);
        }
    }

    /** 因斷裂導致 island 拓撲切分為多個子 islands。 */
    record IslandSplit(
        long tick, long islandClock, long globalSeq,
        long parentIslandId, long[] childIslandIds
    ) implements TopologyEvent {
        /** parent 為主 island，也是依賴 DAG 上游 */
        @Override public long islandId() { return parentIslandId; }
        @Override public TopologyEvent withGlobalSeq(long seq) {
            return new IslandSplit(tick, islandClock, seq, parentIslandId, childIslandIds);
        }
    }

    /** Island 轉為剛體飛出（collapse 的最後一步）。 */
    record RigidBodyReleased(
        long tick, long islandClock, long globalSeq,
        long islandId, Vec3 initialVelocity
    ) implements TopologyEvent {
        @Override public TopologyEvent withGlobalSeq(long seq) {
            return new RigidBodyReleased(tick, islandClock, seq, islandId, initialVelocity);
        }
    }
}
