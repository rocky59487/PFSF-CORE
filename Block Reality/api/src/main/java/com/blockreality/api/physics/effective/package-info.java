/**
 * <h2>中層有效參數（Effective Parameter Layer）</h2>
 *
 * <p>三層物理架構的中間層，負責把材料常數轉換為在特定離散尺度下
 * 數學自洽的「有效參數」，以修正離散化引入的尺度偏差。
 *
 * <h3>核心類別</h3>
 * <ul>
 *   <li>{@link com.blockreality.api.physics.effective.GraphEnergyFunctional} —
 *       定義 E(φ, d, σ, h) 標量能量泛函介面</li>
 *   <li>{@link com.blockreality.api.physics.effective.EdgeWeight} —
 *       邊權 w_ij 介面，保證四性質（對稱、正定、局部、單調）</li>
 *   <li>{@link com.blockreality.api.physics.effective.MaterialCalibration} —
 *       (material, voxelScale, boundary) → 有效參數三元組</li>
 *   <li>{@link com.blockreality.api.physics.effective.MaterialCalibrationRegistry} —
 *       執行緒安全的校準資料庫，fallback 到 {@link com.blockreality.api.material.DefaultMaterial}</li>
 *   <li>{@link com.blockreality.api.physics.effective.EnergyEvaluatorCPU} —
 *       CPU 端能量評估器（GPU energy_reduce.comp.glsl 的 Java golden oracle）</li>
 * </ul>
 *
 * <h3>Stencil 常數</h3>
 * <p>所有邊權計算必須引用 {@link com.blockreality.api.physics.pfsf.PFSFStencil}
 * 的常數（EDGE_P、CORNER_P），不得硬編。
 *
 * @see com.blockreality.api.physics.pfsf.PFSFStencil
 * @see com.blockreality.api.physics.semantic
 * @see com.blockreality.api.physics.solver
 */
package com.blockreality.api.physics.effective;
