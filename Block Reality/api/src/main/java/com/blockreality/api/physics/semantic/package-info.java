/**
 * <h2>上層語意規則（Semantic Layer）</h2>
 *
 * <p>三層物理架構的最上層，負責將材料特性轉換為高層語意規則。
 * 例如：砂土只能受壓碎、木材可拉裂、玻璃脆性剝落。
 *
 * <h3>核心擴展點</h3>
 * <ul>
 *   <li>{@link com.blockreality.api.physics.semantic.ISemanticRule} —
 *       每種材料族群的物理行為規則（Java SPI 擴展）</li>
 *   <li>{@link com.blockreality.api.physics.semantic.SemanticRuleRegistry} —
 *       中央規則注冊表，透過 {@code ServiceLoader} 自動發現</li>
 * </ul>
 *
 * <h3>三層架構關係</h3>
 * <pre>
 * semantic/    ← 本套件：高層規則（ISemanticRule）
 *     ↓
 * effective/   ← 有效參數（EdgeWeight, MaterialCalibration）
 *     ↓
 * solver/ → pfsf/  ← GPU RBGS + PCG + Multigrid（能量最小化器）
 * </pre>
 *
 * @see com.blockreality.api.physics.effective
 * @see com.blockreality.api.physics.solver
 */
package com.blockreality.api.physics.semantic;
