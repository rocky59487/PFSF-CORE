/**
 * Phase G — 拓撲事件驅動架構（per-island vector clock + thread-local buffer +
 * end-of-tick 拓撲排序）。
 *
 * <h2>核心類別</h2>
 * <ul>
 *   <li>{@link com.blockreality.api.physics.topology.TopologyEvent} —
 *       sealed 事件 hierarchy：SupportLost / EdgeFractured / IslandSplit /
 *       RigidBodyReleased</li>
 *   <li>{@link com.blockreality.api.physics.topology.TopologyEventBus} —
 *       無 AtomicLong 的 event bus，熱路徑完全無鎖</li>
 * </ul>
 *
 * <h2>整合 roadmap</h2>
 * <p>Sprint 4 本包只建立核心基礎設施（bus + 4 event record）。後續 Sprint
 * 以 adapter 模式接入：
 * <ul>
 *   <li>{@code UnionFindEngine} merge/split → publish IslandSplit</li>
 *   <li>{@code PFSFFailureRecorder} → publish SupportLost / EdgeFractured</li>
 *   <li>{@code CollapseJournal} → subscribe TopologyEvent 並 append</li>
 *   <li>{@code CollapseManager} → subscribe SupportLost 並進入 collapse queue</li>
 * </ul>
 *
 * <h2>設計決策歷史</h2>
 * <p>原計畫使用全域 {@code AtomicLong} 產生單調序號；千島並發破壞下 CAS 爭用
 * 會淹沒 GPU scheduler（2026-04-22 P2 警告）。改為：
 * <ol>
 *   <li>Publish 熱路徑：Thread-Local ring buffer + per-island vector clock，零鎖</li>
 *   <li>End-of-tick 冷路徑：單執行緒收集所有 buffer、以 (tick, clock, island) 初排、
 *       再依 island DAG 拓撲調整、最後賦 globalSeq 並派送 subscriber</li>
 * </ol>
 */
package com.blockreality.api.physics.topology;
