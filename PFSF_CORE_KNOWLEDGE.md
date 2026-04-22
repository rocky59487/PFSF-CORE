# PFSF Core Research Knowledge Base

## 1. 系統定位 (Core Objective)
證明基於 Phase-Field 的 Voxel 物理系統在離散環境下具備高性能與物理正確性。

## 2. 核心機制 (Core Mechanisms)

### A. 應力解算 (Stress Solver)
- **求解器：** GPU 加速的 Conjugate Gradient (PCG) 與 Jacobi 迭代。
- **目標：** 求解位移場 $u$，計算各 Voxel 的應力張量 $\sigma$。
- **收斂判據：** 殘差向量的 $L_2$ 範數。

### B. 損害演化 (Damage Evolution)
- **機制：** 基於能量釋放率的相場更新。
- **持久性：** 損害具有不可逆性 ($\phi_{t} \ge \phi_{t-1}$)。
- **臨界值：** 當 $\phi$ 超過預設閾值時，觸發方塊破壞與結構坍塌。

### C. 島嶼與倒塌 (Island & Collapse)
- **島嶼系統：** 追蹤結構的連通性塊。
- **倒塌觸發：** 當島嶼失去所有地基支撐（Anchors）時，將整個島嶼轉換為動態實體（Entity）。

## 3. 驗證與 Benchmarks (Validation)

### 靜態測試 (Static Tests)
- **懸臂樑 (Cantilever Beam)：** 對比解析解的撓度與應力分佈。
- **垂直載重 (Vertical Load)：** 驗證受壓穩定性。

### 動態測試 (Dynamic Tests)
- **橋樑倒塌 (Bridge Collapse)：** 驗證結構斷裂點與倒塌順序。
- **支撐移除：** 瞬時移除支撐後的結構反應。

## 4. 開發守則 (Implementation Rules)
1. **拒絕抽象：** 直接呼叫 Vulkan kernels，不使用 Universal Layer。
2. **精度優先：** GPU 版本必須與 CPU Reference 版本在 $10^{-5}$ 誤差範圍內保持一致。
3. **無損傳輸：** Voxel 數據上傳至 GPU 需保持拓撲版本一致。
