/**
 * Block Reality 物理引擎核心模組。
 *
 * <h2>架構（PFSF GPU-only）</h2>
 * <p>所有結構物理由 {@link com.blockreality.api.physics.pfsf PFSF GPU 求解器}
 * 處理。此套件保留共用基礎設施：</p>
 * <ul>
 *   <li>連通性與孤兒偵測：{@link com.blockreality.api.physics.StructureIslandRegistry}、
 *       {@link com.blockreality.api.physics.AnchorContinuityChecker}、
 *       {@link com.blockreality.api.physics.ConnectivityCache}</li>
 *   <li>應力場結果與快照：{@link com.blockreality.api.physics.StressField}、
 *       {@link com.blockreality.api.physics.ResultApplicator}、
 *       {@link com.blockreality.api.physics.SnapshotBuilder}</li>
 *   <li>失效類型與 LoadType：{@link com.blockreality.api.physics.FailureType}、
 *       {@link com.blockreality.api.physics.LoadType}</li>
 *   <li>LOD / 排程：{@link com.blockreality.api.physics.PhysicsTier}、
 *       {@link com.blockreality.api.physics.PhysicsScheduler}</li>
 *   <li>有效物性參數：{@link com.blockreality.api.physics.effective}</li>
 * </ul>
 *
 * <p>歷史的 CPU 圖論求解器（ForceEquilibriumSolver、LoadPathEngine、
 * UnionFindEngine、BeamStressEngine 等）已於 audit-fixes 階段移除；
 * 失效偵測與崩塌觸發的單一權威來源是 PFSF 的 failure_scan shader 加上
 * {@link com.blockreality.api.collapse.CollapseManager#triggerPFSFCollapse}。</p>
 *
 * @since 3.0
 */
package com.blockreality.api.physics;
