/**
 * <h2>下層 GPU 求解器（Solver Layer）</h2>
 *
 * <p>三層物理架構的最底層，是實際執行能量最小化的 GPU compute pipeline。
 * 本套件為概念性標記層；實體程式碼位於 {@code com.blockreality.api.physics.pfsf}。
 *
 * <h3>現有求解器（位於 pfsf/ 套件）</h3>
 * <ul>
 *   <li>RBGS 8-color Smoother — 高頻消除（Phase 1）</li>
 *   <li>PCG Jacobi-preconditioned — 低頻收斂（Phase 2）</li>
 *   <li>Multigrid V-Cycle — 多尺度加速（每 MG_INTERVAL 步）</li>
 *   <li>Phase-Field Evolution — Ambati 2015 AT2 損傷場演化</li>
 *   <li>Failure Scan — 斷裂/壓碎/無支撐偵測</li>
 *   <li>Energy Reduction — GPU 能量標量化（energy_reduce.comp.glsl，Phase C）</li>
 * </ul>
 *
 * <h3>為何不實體搬移 pfsf/ 到本套件</h3>
 * <p>實體搬移會破壞現有 144+ JUnit test 的 import 路徑，成本 &gt; 收益。
 * 本 package-info.java 僅作架構文件用途，說明 {@code pfsf/} 屬於 Solver Layer。
 *
 * <h3>數學定義</h3>
 * <p>求解器在最小化離散能量泛函 E(φ, d)，詳見：
 * {@code docs/L1-api/L2-physics/L3-graph-energy-model.md}
 *
 * @see com.blockreality.api.physics.pfsf.PFSFEngine
 * @see com.blockreality.api.physics.pfsf.PFSFStencil
 * @see com.blockreality.api.physics.effective
 */
package com.blockreality.api.physics.solver;
