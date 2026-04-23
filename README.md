# PFSF-CORE — Potential-Field Structure Failure Engine

![Minecraft Forge](https://img.shields.io/badge/Forge-1.20.1--47.4.13-orange)
![Java 17](https://img.shields.io/badge/Java-17-blue)
![Vulkan](https://img.shields.io/badge/Vulkan-Compute%20%2B%20RT-red)
![License: MIT](https://img.shields.io/badge/License-MIT-green)

**A GPU-accelerated structural-physics simulation engine for Minecraft Forge 1.20.1, built on an isotropic 26-connected potential-field solver with Ambati/Miehe-style phase-field fracture.**
**以各向同性 26 連通勢場求解器與 Ambati/Miehe 相場斷裂為核心的 Minecraft Forge 1.20.1 GPU 結構物理模擬引擎。**

> *If it would not stand in the real world, it will not stand here.*
> *現實中撐不住的，這裡也撐不住。*

---

## Abstract / 摘要

PFSF-CORE maps every Minecraft block to a structural element with engineering-scale properties (compressive strength in MPa, Young's modulus in GPa, density in kg·m⁻³) and solves the steady-state potential field `A φ = b` on the voxel grid every server tick. The stiffness operator `A` is a conductivity-weighted **26-connected isotropic Laplacian** with face/edge/corner weights `1, 0.5, 1/6` (Shinozaki–Oono isotropic stencil); `b` is assembled from per-block gravity and applied loads. Damage follows a history-field phase-field update in the Ambati/Miehe family, with a monotone update `H_{n+1} = max(H_n, ψ_e)` that enforces irreversibility. Solver progression is an **adaptive RBGS → PCG switch** with Chebyshev semi-iteration and periodic V-Cycle multigrid; all kernels run on Vulkan Compute through a shared `br_core` device context.

This README is **paper-grade**: every numeric claim is traceable to a source file or a CSV row tagged with an explicit `Provenance` field (`ANALYTIC_COMPARE`, `MEASURED_JVM_NANOTIME`, `PREDICTED_BANDWIDTH_MODEL`, `MEASURED_HARDWARE_WALLCLOCK`). Unmeasured data is never replaced by hard-coded placeholders — see §7 *Validation* and §8 *Performance*. Known limitations (§9) disclose the current gaps honestly, including the state of the native GPU path and the absence of measured real-hardware timings at time of writing.

PFSF-CORE 將每個 Minecraft 方塊對映為具工程尺度屬性（抗壓 MPa、楊氏模量 GPa、密度 kg·m⁻³）的結構元素，並在每個 server tick 於體素格上求解穩態勢場 `A φ = b`。剛度算子 `A` 為導率加權的 **26 連通各向同性 Laplacian**，面/邊/角權重為 `1、0.5、1/6`（Shinozaki–Oono 各向同性模板）；`b` 由每方塊重力與外加荷載組裝。損傷演化採 Ambati/Miehe 族系的歷史場相場更新，以單調條件 `H_{n+1} = max(H_n, ψ_e)` 強制不可逆性。求解器採 **RBGS → PCG 自適應切換**，結合 Chebyshev 半迭代與週期性 V-Cycle 多網格；所有 kernel 皆透過共享的 `br_core` 裝置上下文執行於 Vulkan Compute。

本 README 為**論文等級**：每個數值宣告皆可追溯至原始碼檔案或標註明確 `Provenance` 欄位（`ANALYTIC_COMPARE`、`MEASURED_JVM_NANOTIME`、`PREDICTED_BANDWIDTH_MODEL`、`MEASURED_HARDWARE_WALLCLOCK`）的 CSV 列。未測量之資料絕不以硬編碼佔位符取代——見 §7 驗證 與 §8 效能。現行限制（§9）誠實揭露包含 native GPU 路徑狀態與撰寫當下尚無實測硬體時間在內的缺口。

---

## Table of Contents / 目次

1. [Introduction / 引言](#1-introduction--引言)
2. [Problem Statement / 問題陳述](#2-problem-statement--問題陳述)
3. [Isotropic 26-Stencil / 各向同性 26 鄰接模板](#3-isotropic-26-stencil--各向同性-26-鄰接模板)
4. [Adaptive Solver / 自適應求解器](#4-adaptive-solver--自適應求解器)
5. [Phase-Field Damage / 相場損傷](#5-phase-field-damage--相場損傷)
6. [Implementation / 實作](#6-implementation--實作)
7. [Validation / 驗證](#7-validation--驗證)
8. [Performance / 效能](#8-performance--效能)
9. [Limitations / 局限](#9-limitations--局限)
10. [Reproducibility / 可重現性](#10-reproducibility--可重現性)
11. [Material System / 材料系統](#11-material-system--材料系統)
12. [References / 參考文獻](#12-references--參考文獻)
13. [Appendix A — Stencil Weight Table](#appendix-a--stencil-weight-table)
14. [Appendix B — File Map](#appendix-b--file-map)

## 1. Introduction / 引言

Most voxel games model block durability with scalar hit-points. Structural games either pre-bake collapse outcomes with static integrity-range rules or delegate to a rigid-body physics engine whose internal state is unrelated to material stiffness. Neither approach predicts the *mode* of failure (crushing, tension rupture, cantilever, buckling-adjacent) from the continuum response of the structure.

PFSF-CORE takes a different path: it discretises a scalar stiffness PDE on the block grid and lets the solution field drive both collapse triggering and the graphical fracture evolution. The design goals are (i) engineering-unit faithfulness, (ii) a single source of truth for the discretisation stencil so the Java/CPU reference and the GLSL/GPU kernels cannot drift, and (iii) per-tick amortisation cheap enough to run alongside vanilla Minecraft ticking on a single server thread with GPU offload for the heavy numerics.

多數體素遊戲以純量血量建模方塊耐久度。結構向遊戲要麼用靜態整合半徑規則預先烘焙坍塌結果，要麼交由剛體物理引擎處理——但剛體引擎的內部狀態與材料剛度無關。兩者都無法從結構的連體回應預測**失效模態**（壓碎、拉斷、懸臂、近屈曲）。PFSF-CORE 選擇另一條路：於方塊格上離散純量剛度 PDE，並以其解場同時驅動坍塌觸發與圖形斷裂演化。設計目標為：(i) 工程單位忠實、(ii) 離散化模板在 Java/CPU 參考與 GLSL/GPU kernel 間擁有單一真值來源（避免數值分歧）、(iii) 每個 tick 的均攤成本夠低，可在單一伺服器執行緒上與原版 Minecraft tick 並行，並將重度數值交由 GPU。

## 2. Problem Statement / 問題陳述

Let `Ω ⊂ ℤ³` be the set of occupied voxels in a connected structural island, and let `A ⊂ ∂Ω` be the anchored (world-grounded) subset. We solve for a scalar potential `φ : Ω → ℝ` satisfying the discrete conductivity-weighted Poisson equation

```
- ∇·( κ ∇φ )  =  s       in Ω
            φ  =  0       on A        (Dirichlet anchor)
        n·∇φ  =  0       on ∂Ω \ A   (homogeneous Neumann)
```

with per-voxel conductivity `κ_i` proportional to Young's modulus via a calibration mapping in `MaterialCalibrationRegistry`, and per-voxel source `s_i` assembled from gravity and external loads (Timoshenko beam contributions where applicable). After discretisation this becomes the symmetric positive-semidefinite linear system `A φ = b` where `A` is the conductivity-weighted 26-connected Laplacian stiffness matrix described in §3.

Failure is reported by `failure_scan.comp.glsl`, which compares outward flux (`flux_i = Σ_j a_ij |φ_i − φ_j|`) against the per-voxel resistance `rcomp_i` / `rtens_i`. Eight failure-mode codes are currently emitted — cantilever, crushing, no-support, tension-break, hydrostatic, thermal, wind-overturning, and lightning-strike — with two further modes reserved for torsion and fatigue (see §11). All resistances and loads live in engineering units throughout: **strength in MPa, Young's modulus in GPa, density in kg·m⁻³**.

設 `Ω ⊂ ℤ³` 為連通結構島的佔用體素集合，`A ⊂ ∂Ω` 為錨定（接地）子集合。我們求解純量勢場 `φ : Ω → ℝ` 使其滿足上式之離散導率加權 Poisson 方程：在 Ω 內部滿足 `−∇·(κ∇φ) = s`，錨定點上 `φ = 0`（Dirichlet），其他邊界採齊次 Neumann。每體素導率 `κ_i` 由 `MaterialCalibrationRegistry` 之映射依楊氏模量校準，源項 `s_i` 由重力與外加荷載（適用時含 Timoshenko 梁貢獻）組裝。離散化後即為對稱半正定線性系統 `A φ = b`，`A` 為 §3 所述之導率加權 26 連通 Laplacian 剛度矩陣。失效由 `failure_scan.comp.glsl` 回報——以外向通量 `flux_i = Σ_j a_ij |φ_i − φ_j|` 與各體素抗力 `rcomp_i / rtens_i` 比較。目前已實作 8 種失效模式碼（懸臂、壓碎、無支撐、拉斷、靜水壓、熱、風傾覆、雷擊），另保留扭斷與疲勞兩種（見 §11）。所有抗力與荷載全程採工程單位：**強度 MPa、楊氏模量 GPa、密度 kg·m⁻³**。

## 3. Isotropic 26-Stencil / 各向同性 26 鄰接模板

**Single source of truth:** `Block Reality/api/src/main/java/com/blockreality/api/physics/pfsf/PFSFStencil.java:42-43`. GPU shaders consume this via `#include "stencil_constants.glsl"`, which is regenerated from the Java SSOT by the Gradle task `:api:generateStencilGlsl`. Any divergence between CPU and GPU weights is blocked by `PFSFStencilConsistencyTest`.

**唯一真值來源：** `PFSFStencil.java:42-43`。GPU shader 透過 `#include "stencil_constants.glsl"` 使用之；該 header 由 Gradle task `:api:generateStencilGlsl` 從 Java 端重新產生。CPU 與 GPU 間任何分歧都會被 `PFSFStencilConsistencyTest` 攔截。

| Neighbour class 鄰居類別 | Distance 距離 | Count 數 | Weight 權重 | Symbol |
|---|---|---|---|---|
| Face 面     | 1       | 6  | 1.0        | (base) |
| Edge 邊     | √2      | 12 | 0.5        | `EDGE_P`   |
| Corner 角   | √3      | 8  | 1/6 ≈ 0.1666667 | `CORNER_P` |

The weight choice `1, ½, ⅙` is the **Shinozaki–Oono isotropic Laplacian**. Its discrete 3-D Taylor expansion around a voxel `i` cancels the leading-order anisotropic terms of the second-derivative operator, so the leading truncation error is `O(h²)·∇⁴φ` with isotropic coefficient — meaning diagonal directions are not advantaged over axis-aligned directions. This matters for fracture: an anisotropic Laplacian biases crack propagation along the grid axes, producing a staircasing artefact that a user can see as "all cracks run straight north-south". With the Shinozaki–Oono weights, the discrete operator is rotationally symmetric at the stencil level.

此 `1、½、⅙` 權重組合即為 **Shinozaki–Oono 各向同性 Laplacian**。其 3D 離散 Taylor 展開在體素 `i` 附近抵消二階導算子的首階各向異性項，使首階截斷誤差為 `O(h²)·∇⁴φ`（係數各向同性）——意即對角方向不會相對於軸向方向被優待。此點對斷裂尤為關鍵：各向異性的 Laplacian 會使裂縫偏好沿格軸傳播，產生使用者能直觀看見的「所有裂縫都走南北正向」階梯化瑕疵。採用 Shinozaki–Oono 權重後，離散算子在模板層次具旋轉對稱性。

**Critical invariant / 關鍵不變式.** RBGS, Jacobi, and PCG matvec MUST share the identical 26-connected stencil (same `EDGE_P` = 0.5, same `CORNER_P` = 0.1666667). If any one shader uses a different weight, PCG will converge to the wrong minimum (because the preconditioner no longer matches the operator being smoothed) and multigrid transfer operators can cause divergence. The consistency test in `PFSFStencilConsistencyTest` reads the GLSL header at build time and asserts byte-for-byte agreement with `PFSFStencil.java`. 

For the mathematical derivation (including the Taylor cancellation), see `research/paper_data/STENCIL_MATHEMATICS.md`.

**關鍵不變式.** RBGS、Jacobi 與 PCG matvec **必須**共用完全相同的 26 連通模板（相同 `EDGE_P = 0.5`、相同 `CORNER_P = 0.1666667`）。任一 shader 採不同權重，PCG 會收斂至錯誤極小（因預條件子不再匹配被平滑的算子），多網格轉移算子也可能造成發散。`PFSFStencilConsistencyTest` 會於建置時讀取 GLSL header，斷言其與 `PFSFStencil.java` 在位元層級完全一致。數學推導（含 Taylor 展開抵消細節）詳見 `research/paper_data/STENCIL_MATHEMATICS.md`。

## 4. Adaptive Solver / 自適應求解器

The linear system `A φ = b` is solved per island per tick by `PFSFDispatcher`, which composes three kernel classes — a **Red-Black Gauss-Seidel (RBGS) smoother**, a **Preconditioned Conjugate Gradient (PCG) solver**, and an optional **V-Cycle multigrid** — under an adaptive switching policy driven by the residual norm.

`PFSFDispatcher` 每 tick 對每座島求解 `A φ = b`，組合三類 kernel——**Red-Black Gauss-Seidel (RBGS) smoother**、**預條件共軛梯度 (PCG)**、可選的 **V-Cycle 多網格**——並由殘差驅動的自適應策略切換。

### 4.1 RBGS smoother with Chebyshev semi-iteration

`rbgs_smooth.comp.glsl` performs an in-place 8-colour Gauss-Seidel sweep (color index = `(x%2) | (y%2)<<1 | (z%2)<<2`). After two warm-up pure-Jacobi steps, the iterates are accelerated by Chebyshev semi-iteration with the classical recurrence

```
ω₀ = 1
ω₁ = 2 / (2 − ρ²)
ω_k = 4 / (4 − ρ² · ω_{k−1})     (k ≥ 2)
ρ   = cos(π / L_max) · SAFETY_MARGIN      (SAFETY_MARGIN = 0.95)
```

where `ρ` is the spectral-radius estimate of the smoothed iteration matrix. Chebyshev is a polynomial accelerator: it minimises the maximum of the residual polynomial over the spectral interval, giving asymptotic convergence ∝ `(√κ − 1) / (√κ + 1)` per iteration versus Jacobi's `(κ − 1) / (κ + 1)`.

`rbgs_smooth.comp.glsl` 執行原地 8 色 Gauss-Seidel 掃瞄（顏色索引 = `(x%2) | (y%2)<<1 | (z%2)<<2`）。經 2 步純 Jacobi 暖機後，以 Chebyshev 半迭代加速；`ρ` 為被平滑迭代矩陣之譜半徑估計。Chebyshev 為多項式加速器：在譜區間上極小化殘差多項式之最大值，漸近收斂率由 Jacobi 的 `(κ−1)/(κ+1)` 提升為 `(√κ−1)/(√κ+1)`。

### 4.2 V-Cycle multigrid

Every `MG_INTERVAL = 4` smoothing steps a V-Cycle is inserted: `mg_restrict` projects the residual to a coarser grid via conductivity-weighted restriction, `jacobi_smooth` solves on the coarse grid, and `mg_prolong` interpolates the correction back by trilinear prolongation. This attacks low-frequency error modes that RBGS alone damps slowly.

每 `MG_INTERVAL = 4` 步平滑後插入一次 V-Cycle：`mg_restrict` 以導率加權 restriction 將殘差投影至粗網格，`jacobi_smooth` 在粗網格求解，`mg_prolong` 以三線性插值將修正 prolong 回細網格。此步專門攻擊 RBGS 單獨時阻尼緩慢的低頻誤差模態。

### 4.3 RBGS → PCG switch

When the RBGS residual drop ratio falls below 5 % (stagnation), `PFSFDispatcher` switches to PCG. The PCG kernels (`pcg_matvec.comp.glsl`, `pcg_update.comp.glsl`, `pcg_direction.comp.glsl`, `pcg_dot.comp.glsl`) implement the standard Saad 2003 formulation with a Jacobi preconditioner `M⁻¹ = diag(A₂₆)⁻¹`, computed on-the-fly so no extra buffer is required. The inner product uses `r·z` (preconditioned), not `r·r`, as required for convergence proofs of PCG.

當 RBGS 殘差下降比率跌破 5%（停滯），`PFSFDispatcher` 切換至 PCG。PCG kernel 實作標準 Saad 2003 公式，以 Jacobi 預條件子 `M⁻¹ = diag(A₂₆)⁻¹`（即時計算，不需額外 buffer）；內積採 `r·z`（預條件後），而非 `r·r`——此為 PCG 收斂性證明之要求。

### 4.4 Divergence and convergence detection

`PFSFScheduler` watches the residual history. If the ratio `‖r_k‖ / ‖r_{k-1}‖ > DIVERGENCE_RATIO (1.5)`, the tick is abandoned and marked for fallback. When the macro-block residual falls below `CONVERGENCE_SKIP_THRESHOLD (0.01)`, the island enters LOD_DORMANT and no kernels are dispatched for `STABLE_TICK_SKIP_COUNT (3)` subsequent ticks, amortising the per-tick cost over static structures.

`PFSFScheduler` 監控殘差歷史。若 `‖r_k‖ / ‖r_{k-1}‖ > DIVERGENCE_RATIO (1.5)`，放棄本 tick 並標記 fallback；若 macro-block 殘差跌破 `CONVERGENCE_SKIP_THRESHOLD (0.01)`，島進入 LOD_DORMANT，後續 `STABLE_TICK_SKIP_COUNT (3)` tick 不派工，將靜態結構的每 tick 成本均攤掉。

## 5. Phase-Field Damage / 相場損傷

Damage evolution follows the Ambati/Miehe history-field family. Each voxel carries two additional scalar fields: the history strain-energy density `H` and the damage variable `d ∈ [0, 1]`. After each tick's solver phase, the smoother (`rbgs_smooth.comp.glsl` or `jacobi_smooth.comp.glsl`) writes the updated history via the **monotone rule**

```
H_{n+1}_i  =  max( H_n_i ,  ψ_e_i )
```

where `ψ_e_i = ½ · (∇φ)_i · κ_i · (∇φ)_i` is the elastic strain-energy density at voxel *i*. This guarantees irreversibility `H_{n+1} ≥ H_n` voxel-wise, which propagates to damage monotonicity `d_{n+1} ≥ d_n` because the damage update is

```
d_new_i  =  ( H_i + l₀² · ∇²d_i ) / ( H_i + G_c / (2 l₀) )
```

with regularisation length `l₀ = 1.5` blocks, fracture energy `G_c` scaled by the Bažant hydration law `G_c(t) = G_{c,base} · H_hyd^{1.5}`, and relaxation factor `0.3`. When `d_i > 0.95` the voxel is reported as fully cracked and removed by `CollapseManager` at the end of the tick.

損傷演化採 Ambati/Miehe 族系的歷史場更新。每體素另攜帶歷史應變能密度 `H` 與損傷變數 `d ∈ [0, 1]`。每 tick 求解階段結束後，smoother 以**單調規則** `H_{n+1}_i = max(H_n_i, ψ_e_i)` 更新歷史；其中 `ψ_e_i = ½ (∇φ)_i · κ_i · (∇φ)_i` 為體素 *i* 的彈性應變能密度。此規則逐體素保證 `H_{n+1} ≥ H_n`，並透過上式損傷更新傳遞為 `d_{n+1} ≥ d_n`。正則化長度 `l₀ = 1.5` 方塊；斷裂能 `G_c` 依 Bažant 水化律 `G_c(t) = G_{c,base}·H_hyd^{1.5}` 縮放；鬆弛因子 `0.3`。當 `d_i > 0.95` 時體素回報為全裂，於 tick 結束由 `CollapseManager` 移除。

**Write-ownership invariant / 寫入權不變式.** `H` is written **exclusively** by the smoother shaders. `phase_field_evolve.comp.glsl` declares `H` as `readonly` and never touches it. This prevents a GPU race condition that would otherwise give non-deterministic damage evolution because compute work-group scheduling across dispatches does not enforce an implicit barrier on shared buffers.

**寫入權不變式.** `H` 僅由 smoother shader 寫入。`phase_field_evolve.comp.glsl` 將 `H` 宣告為 `readonly` 完全不觸碰——此安排避免 GPU race condition；若由兩 shader 同寫，因 compute work-group 在多 dispatch 間排程不保證對共享 buffer 隱式屏障，會產生非確定性損傷演化。

## 6. Implementation / 實作

### 6.1 Per-tick pipeline

```
PFSFEngine.onServerTick()
  │
  ├─ PFSFAsyncCompute.pollCompleted()     (non-blocking fence check, triple-buffered)
  ├─ StructureIslandRegistry.getDirty…()  (incremental dirty-epoch diff)
  │
  ├─ PFSFDataBuilder (per dirty island)
  │   ├─ Assemble source/conductivity/type
  │   ├─ Morton-tiled 8×8×8 Z-order layout (mortonBlockSize=8)
  │   └─ σ_max normalisation (see §6.2)
  │
  ├─ PFSFDispatcher                       (adaptive RBGS → PCG, V-Cycle every 4 steps; see §4)
  ├─ PFSFFailureRecorder                  (failure_scan → failure_compact)
  ├─ PFSFPhaseFieldRecorder               (Ambati/Miehe; see §5)
  └─ submitAsync() → fence → CollapseManager.triggerPFSFCollapse()
```

### 6.2 σ_max normalisation contract / σ_max 正規化約定

`PFSFDataBuilder.normalizeSoA6()` rescales every buffer carrying a *stress-scale* threshold by the maximum conductivity value on the island before GPU upload. This keeps Vulkan buffer values in a well-conditioned numerical range (`[0, 1]` for `κ`). Because `A · φ = b` is **scale-invariant** under simultaneous scaling of `A` and `b` (divide both sides by σ_max and `φ` is unchanged), the potential field itself needs no rescaling — but comparisons against *physical* thresholds must happen in the same normalised frame.

`PFSFDataBuilder.normalizeSoA6()` 於 GPU 上傳前，以島上最大導率值 σ_max 對每個載有**應力尺度**閾值的 buffer 進行縮放；此舉使 Vulkan buffer 值保持在條件數良好的數值區間（`κ ∈ [0, 1]`）。因 `A·φ = b` 對 `A` 與 `b` 同比縮放具**標度不變性**（兩邊同除 σ_max 後 `φ` 不變），故勢場本身不需縮放；但對**物理**閾值的比較，必須落在同一縮放後的參考框架。

| Buffer | Normalisation | Rationale |
|---|---|---|
| `conductivity[i]` | `/= σ_max` | maps to [0, 1] |
| `source[i]`       | `/= σ_max` | keeps `A φ = b` balanced |
| `rcomp[i]`        | `/= σ_max` | `failure_scan` compares `flux > rcomp[i]` in normalised frame |
| `rtens[i]`        | `/= σ_max` | same as rcomp |
| `maxPhi[i]`       | **NOT scaled** | φ is scale-invariant ⇒ its threshold must not scale |
| `phi` field       | unchanged | as above |

The `maxPhi` row deserves special attention. `PFSFDataBuilder.java:261-269` explicitly retains `maxPhi` at its physical value with the comment *"移除對 maxPhi 的正規化縮放 (Normalization Bug)"* — earlier revisions divided `maxPhi` by σ_max and saw stiff structures collapse spuriously because their cantilever threshold shrank with conductivity. The corrected argument: since the PDE is scale-invariant in `φ`, any threshold on `φ` must be scale-invariant too.

`maxPhi` 一列值得特別說明。`PFSFDataBuilder.java:261-269` 明確保留 `maxPhi` 於物理值並附註「移除對 maxPhi 的正規化縮放 (Normalization Bug)」——先前版本曾對 `maxPhi` 除以 σ_max，結果高剛性結構因其懸臂閾值隨導率縮小而莫名崩塌。修正後的論述為：既然 PDE 對 `φ` 標度不變，`φ` 的任何閾值也必須標度不變。

### 6.3 Buffer layout & async compute

Each island owns one `PFSFIslandBuffer` allocated through VMA (`VkMemoryAllocator`) as a **single contiguous block** with sub-region offsets for `phi(×2)`, `source`, `conductivity[26N]`, `type`, `fail_flags`, `maxPhi`, `rcomp`, `rtens`, `hField`, `dField`, `hydrationBuf`, `macroResidual`. Triple-buffered fences (`PFSFAsyncCompute`) allow three in-flight ticks without readback stalls. Shader kernels share the `br_core::VulkanComputeContext` to avoid per-pipeline command-pool allocation.

每座島擁有一份透過 VMA (`VkMemoryAllocator`) 配置的 `PFSFIslandBuffer`，為**單一連續記憶體塊**，內部以子區偏移存放 `phi(×2)、source、conductivity[26N]、type、fail_flags、maxPhi、rcomp、rtens、hField、dField、hydrationBuf、macroResidual`。三重緩衝 fence (`PFSFAsyncCompute`) 允許 3 個 tick 同時飛行而不會因 readback 卡住。Shader kernel 共用 `br_core::VulkanComputeContext` 避免每 pipeline 重複配置 command pool。

## 7. Validation / 驗證

Validation compares the CPU reference solver `VoxelPhysicsCpuReference` (which uses the identical 26-connected stencil as the GPU kernels) against closed-form analytical solutions on three canonical structural primitives.

驗證以 CPU 參考求解器 `VoxelPhysicsCpuReference`（使用與 GPU kernel 完全相同之 26 連通模板）對照三個標準結構原型的閉式解析解。

### 7.1 Analytic references / 解析參考

| Scenario | Geometry | Boundary | Analytic solution |
|---|---|---|---|
| **CANTILEVER** | 1×1×L column | φ(0)=0 (Dirichlet), φ'(L)=0 (Neumann) | `φ(z) = q·L·z − ½·q·z²` |
| **ARCH**   | semi-circular arch R=24 | φ(0) = φ(L_arc) = 0 (both feet Dirichlet) | `φ(s) = ½·q·s·(L_arc − s)` along arc length |
| **SLAB**   | L×L×L with Dirichlet at z=0 | free elsewhere | same as cantilever per-z-column: `φ(z) = q·L·z − ½·q·z²` |

### 7.2 Results / 結果

Results live at `research/paper_data/raw/validation_results.csv`, regenerated by

```bash
./gradlew :api:test --tests "com.blockreality.api.physics.validation.PaperDataCollectorTest"
```

with CSV schema `Geometry, L2RelErrorPercent, Provenance`. Every row carries `Provenance = ANALYTIC_COMPARE` — the earlier release hard-coded `1.25` for ARCH and SLAB because the analytic comparisons were not implemented; this release removes that placeholder entirely and computes a real L² relative error from the converged reference solution. See `PaperDataCollectorTest.java` for the exact comparison routine (arc-length BFS ordering for ARCH; per-z-layer mean for SLAB).

結果於 `research/paper_data/raw/validation_results.csv`，CSV 欄位為 `Geometry, L2RelErrorPercent, Provenance`。上一版對 ARCH 與 SLAB 硬編碼 `1.25` 是因解析對比尚未實作；本版完全移除該佔位符，改以收斂後參考解計算真實 L² 相對誤差。比對常式詳見 `PaperDataCollectorTest.java`（ARCH 採弧長 BFS 排序；SLAB 採每 z 層均值）。

> **Reproducibility note.** The CSV is regenerated each time the test runs; we therefore do not list specific percentages in this README — numbers live next to the code that produces them so they cannot drift. The L² relative error is expected to be ≲ 1–5 % on CANTILEVER (grid-resolution dependent), with ARCH and SLAB slightly higher because their analytic references are themselves continuum approximations of a voxelised geometry.
>
> **可重現性說明.** CSV 由測試每次執行重新產生；本 README 因此不列具體百分比——數字緊挨產生它們的程式碼存放，不會漂移。L² 相對誤差預期於 CANTILEVER ≲ 1–5 %（視網格解析度），ARCH/SLAB 略高，因其解析參考本身為體素化幾何的連體近似。

## 8. Performance / 效能

Performance data is split into two CSVs, each with a `Provenance` column so predicted and measured numbers cannot be confused.

效能資料分為兩份 CSV，各帶 `Provenance` 欄位以免預測值與實測值混淆。

### 8.1 CPU measured vs GPU predicted

`research/paper_data/raw/performance_metrics_predicted.csv` (regenerated by `PerformanceBenchmarkTest`) contains two roles per island size:

- `CPU_JAVA` — `Provenance = MEASURED_JVM_NANOTIME`. Wall-clock `System.nanoTime()` average of 100 Jacobi steps on the CPU reference solver.
- `GPU_FNO`  — `Provenance = PREDICTED_BANDWIDTH_MODEL(bw=400GB/s,opt=0.3)`. **No GPU is dispatched.** The value is a roofline-style upper bound computed from a nominal DRAM bandwidth and the 26-connected stencil byte traffic `N × 26 × 4 B × 2`. The formula lives in `PerformanceBenchmarkTest.predictGpuMs()` and is disclosed in every CSV row.

`research/paper_data/raw/performance_metrics_predicted.csv`（由 `PerformanceBenchmarkTest` 重新產生）每島尺寸含兩列：`CPU_JAVA`（`Provenance = MEASURED_JVM_NANOTIME`，即 100 次 Jacobi 步之 wall-clock 均值）；`GPU_FNO`（`Provenance = PREDICTED_BANDWIDTH_MODEL(bw=400GB/s,opt=0.3)`，**不派任何 GPU 工作**，僅以名義 DRAM 頻寬與 26 連通模板位元組流量 `N × 26 × 4 B × 2` 計算之 roofline 上界，公式在 `PerformanceBenchmarkTest.predictGpuMs()` 並於每列 CSV 中明示）。

### 8.2 Real-hardware measured

`research/paper_data/raw/real_hardware_performance.csv` is populated only when `RealHardwareBenchmarkTest` runs on a host with a working native runtime. If the runtime is unavailable the test is **skipped via `Assumptions.assumeTrue`** — it will NOT silently pass with an empty CSV. Every written row is tagged `Provenance = MEASURED_HARDWARE_WALLCLOCK`, or `MEASURED_HARDWARE_WALLCLOCK_FLOORED@1us` when the recorded wall-clock drops below 1 µs (the measurement quantum we trust). The earlier release returned silently on runtime unavailability and shipped an empty CSV that looked like a completed measurement; §9.2 documents this as fixed.

`research/paper_data/raw/real_hardware_performance.csv` 僅在 `RealHardwareBenchmarkTest` 於具備可用 native runtime 之主機執行時填入。Runtime 不可用時，測試**經由 `Assumptions.assumeTrue` 被 skip** —— 不會以空 CSV 靜默通過。每列 Provenance 為 `MEASURED_HARDWARE_WALLCLOCK`，或當 wall-clock 低於 1 μs（我們信任的測量量子）時為 `MEASURED_HARDWARE_WALLCLOCK_FLOORED@1us`。上一版於 runtime 不可用時靜默 `return`，結果 CSV 空白卻看似完成測量；§9.2 記錄此處已修。

### 8.3 What we do NOT claim

At time of writing, this repository contains **no measured GPU timings**. Any speedup factor stated in prior releases (e.g. "72.76× speedup") originated from the bandwidth prediction and is not a real measurement. A reader wishing to quote GPU performance must either (a) run `RealHardwareBenchmarkTest` on hardware and cite the resulting `MEASURED_HARDWARE_WALLCLOCK` row, or (b) rely on the predicted values with the predicted-model provenance visible in the same sentence.

本 repository **目前並無任何實測 GPU 時間**。先前版本出現的加速倍數（如 "72.76× speedup"）源自頻寬預測，並非真實測量。欲引用 GPU 效能者，請（a）於硬體執行 `RealHardwareBenchmarkTest` 並引用其 `MEASURED_HARDWARE_WALLCLOCK` 列，或（b）使用預測值但同句中明示 predicted-model provenance。

## 9. Limitations / 局限

A full audit of the repository's state against its documentation was completed on 2026-04-23 and is archived at `research/PFSF_GPU_ACADEMIC_AUDIT_2026-04-23.md`. This section lists the limitations currently disclosed as open items and the ones this revision addresses.

2026-04-23 對本倉庫對照其文件的完整審核存於 `research/PFSF_GPU_ACADEMIC_AUDIT_2026-04-23.md`。本節列出目前仍揭露的公開項以及本次修訂處理之項目。

### 9.1 Addressed in this revision / 本次已處理

| Item | Fix |
|---|---|
| Stencil weights doc↔code drift (README said 0.35/0.15, code had 0.5/1/6) | README §3 and `CLAUDE.md`, `AGENTS.md` now cite the Shinozaki–Oono values 0.5, 1/6 that match `PFSFStencil.java:42-43` and `stencil_constants.glsl`. |
| `maxPhi /= σ_max` described in docs but explicitly disabled in code | README §6.2 and `CLAUDE.md`, `AGENTS.md` now cite `PFSFDataBuilder.java:261-269` and explain the scale-invariance argument. |
| ARCH/SLAB validation used hard-coded `1.25` placeholder | `PaperDataCollectorTest` now computes analytic L² relative error for all three scenarios; the CSV carries `Provenance=ANALYTIC_COMPARE`. |
| `PerformanceBenchmarkTest` wrote a column labelled `GPUTime_ms` that was a prediction | CSV renamed to `performance_metrics_predicted.csv`, schema now `Size_N, Role, TimePerStep_ms, Provenance`; predicted rows carry `PREDICTED_BANDWIDTH_MODEL(…)`. |
| `RealHardwareBenchmarkTest` silently passed with an empty CSV when no GPU | Test now SKIPS via `Assumptions.assumeTrue` when runtime unavailable; provenance labels wall-clock vs floored. |
| Native GPU path reported `SPIR-V blob missing` because `br_shaders` was never added | `L1-native/CMakeLists.txt` now `add_subdirectory(shaders)` behind `BR_BUILD_SHADERS=ON` (default); `libpfsf/CMakeLists.txt` force-links `br_shaders` via `$<LINK_LIBRARY:WHOLE_ARCHIVE,br_shaders>`; `Block Reality/api/build.gradle` drops `-DBR_BUILD_SHADERS=OFF` and enables `-DPFSF_ALLOW_PREBUILT_SPV=ON` when glslang is missing. |
| `build.gradle` unconditionally set `VK_ICD_FILENAMES` to Linux paths, breaking Windows tests | Override is now guarded by `org.gradle.internal.os.OperatingSystem.current().isLinux()`. |

### 9.2 Still open / 尚未解決

- **No measured real-hardware GPU timings in the repository.** `real_hardware_performance.csv` is empty because the CI host does not have a usable native PFSF runtime. The test now SKIPS (see §8.2) rather than silently passing, but the CSV must be populated on a developer workstation before any GPU-speedup claim can be made in paper text.
- **`performance_metrics_predicted.csv` is a model, not a measurement.** Readers must not quote its numbers as empirical evidence. A true on-GPU benchmark that emits MEASURED rows is the correct way to supersede this file.
- **Failure-mode coverage is partial.** `FailureType` enumerates torsion and fatigue modes, but `failure_scan.comp.glsl` does not yet implement per-voxel cycle counting or asymmetric load detection. They remain on the roadmap.
- **Fluid GPU path is disabled.** `FluidGPUEngine.GPU_PATH_ENABLED = false`; `FluidCPUSolver` handles diffusion on the server thread with the 1-tick fluid-structure coupling delay acknowledged as intentional (circular-dependency avoidance).
- **Citation tightening.** The Shinozaki–Oono label appears in the repo source comments but the primary 1995 reference has not been cross-checked line-by-line against the derivation in `research/paper_data/STENCIL_MATHEMATICS.md`. The stencil is mathematically a valid isotropic choice; the attribution is what needs verification before paper submission.

- **倉庫中無實測硬體 GPU 時間。** `real_hardware_performance.csv` 為空——CI 主機無可用 native PFSF runtime。測試現在改為 SKIP（見 §8.2）而非靜默通過，但論文行文欲做任何 GPU 加速宣稱前，該 CSV 必須先於開發者工作站上填入。
- **`performance_metrics_predicted.csv` 為模型，非測量。** 請勿將其數值引為經驗證據；欲取代之，應執行真正於 GPU 上 benchmark 並寫入 MEASURED 列。
- **失效模態涵蓋不完整。** `FailureType` 列舉扭斷與疲勞，但 `failure_scan.comp.glsl` 尚未實作逐體素循環計數與非對稱荷載偵測；列於 roadmap。
- **流體 GPU 路徑已停用。** `FluidGPUEngine.GPU_PATH_ENABLED = false`；`FluidCPUSolver` 在伺服器執行緒處理擴散，流體-結構耦合有 1 tick 延遲（刻意避免循環依賴）。
- **引用標籤待收緊。** Shinozaki–Oono 標籤出現於倉庫註解，但尚未逐行對照 1995 原始文獻與 `research/paper_data/STENCIL_MATHEMATICS.md` 之推導。此模板在數學上為有效各向同性選擇；投稿前需核對的是**歸因標籤**而非算法本身。

## 10. Reproducibility / 可重現性

### 10.1 Environment

- Java 17 (Temurin recommended)
- Gradle 8.8 wrapper (daemon disabled; heap `-Xmx3G`)
- Minecraft Forge 1.20.1 (47.4.13), Official Mappings
- Vulkan SDK with `glslangValidator` on PATH (optional; prebuilt SPIR-V is used as fallback — see §9.1)
- Linux recommended for CI tests; Windows supported for runtime (§9.1 loader fix)
- Optional: CMake 3.20+ for the native `libpfsf` build

### 10.2 Build commands

All Gradle commands run from `Block Reality/`:

```bash
cd "Block Reality"

# Full build
./gradlew build

# Merged JAR (for dropping into mods/)
./gradlew mergedJar

# Run Minecraft
./gradlew :fastdesign:runClient      # recommended
./gradlew :api:runClient
./gradlew :api:runServer

# Deploy to PrismLauncher dev instance
./gradlew :api:copyToDevInstance
./gradlew :fastdesign:copyToDevInstance
```

### 10.3 Paper-data regeneration

Each CSV under `research/paper_data/raw/` is regenerated by a single JUnit test. Columns and `Provenance` values are guaranteed by those tests.

| CSV file | Generated by | Provenance values |
|---|---|---|
| `validation_results.csv` | `PaperDataCollectorTest` | `ANALYTIC_COMPARE`, or `DEGENERATE_ARC_LEN<4` for pathological ARCH radii |
| `performance_metrics_predicted.csv` | `PerformanceBenchmarkTest` | `MEASURED_JVM_NANOTIME` for CPU rows; `PREDICTED_BANDWIDTH_MODEL(…)` for GPU rows |
| `real_hardware_performance.csv` | `RealHardwareBenchmarkTest` (**skipped when no GPU**) | `MEASURED_HARDWARE_WALLCLOCK` or `MEASURED_HARDWARE_WALLCLOCK_FLOORED@1us` |

```bash
./gradlew :api:test --tests "com.blockreality.api.physics.validation.PaperDataCollectorTest"
./gradlew :api:test --tests "com.blockreality.api.physics.validation.PerformanceBenchmarkTest"
./gradlew :api:test --tests "com.blockreality.api.physics.validation.RealHardwareBenchmarkTest" -Dblockreality.native.pfsf=true
```

### 10.4 Native build

```bash
cd L1-native
mkdir build && cd build
cmake .. -DBR_BUILD_PFSF=ON          \
         -DBR_BUILD_SHADERS=ON       \
         -DPFSF_ALLOW_PREBUILT_SPV=ON    # set to OFF when glslangValidator is present
cmake --build .
```

### 10.5 Stencil regeneration

If `PFSFStencil.java` changes, re-emit the GLSL header and let `PFSFStencilConsistencyTest` verify agreement:

```bash
./gradlew :api:generateStencilGlsl
./gradlew :api:test --tests "com.blockreality.api.physics.pfsf.PFSFStencilConsistencyTest"
```

## 11. Material System / 材料系統

All values are engineering-unit (MPa strength, GPa Young's modulus, kg·m⁻³ density). Safety factors follow Eurocode.

### 11.1 Built-in materials / 內建材料

| Material | R_c (MPa) | R_t (MPa) | R_sh (MPa) | ρ (kg/m³) | E (GPa) | ν | Safety |
|---|---|---|---|---|---|---|---|
| `plain_concrete` | 25  | 2.5  | 3.5  | 2400 | 25   | 0.18 | 1.5 (EN 1992) |
| `concrete`       | 30  | 3.0  | 4.0  | 2350 | 30   | 0.20 | 1.5 |
| `rc_node` (RC composite) | 33 | 5.9 | 5.0 | 2500 | 32 | 0.20 | 1.5 |
| `rebar`          | 250 | 400  | 150  | 7850 | 200  | 0.29 | 1.15 (EN 1993) |
| `steel`          | 350 | 500  | 200  | 7850 | 200  | 0.29 | 1.15 |
| `brick`          | 10  | 0.5  | 1.5  | 1800 | 5    | 0.15 | 2.5 (EN 1996) |
| `timber`         | 5   | 8.0  | 2.0  | 600  | 11   | 0.35 | 1.3 (EN 1995) |
| `stone`          | 30  | 3.0  | 4.0  | 2400 | 50   | 0.25 | 1.5 |
| `glass`          | 100 | 30.0 | 1.0  | 2500 | 70   | 0.22 | 1.6 (prEN 16612) |
| `sand`           | 0.1 | 0    | 0.05 | 1600 | 0.01 | 0.30 | 1.4 (EN 1997) |
| `obsidian`       | 200 | 5.0  | 20   | 2600 | 70   | 0.20 | 1.5 |
| `bedrock`        | 1e9 | 1e9  | 1e9  | 3000 | 1e6  | 0.10 | 1.0 (indestructible) |

### 11.2 RC fusion / 鋼筋混凝土融合

When rebar and concrete are adjacent, `RCFusionDetector` promotes them to a `DynamicMaterial` composite with fixed `97 % concrete + 3 % rebar` weights. This ratio is not configurable.

當鋼筋與混凝土相鄰時，`RCFusionDetector` 將其提升為 `DynamicMaterial` 複合材——固定比 `97 % 混凝土 + 3 % 鋼筋`，不可調整。

### 11.3 Failure modes / 失效模式

```
FailureType ∈ {
  CANTILEVER_BREAK, CRUSHING, NO_SUPPORT, TENSION_BREAK,
  HYDROSTATIC_PRESSURE, THERMAL_STRESS, THERMAL_SPALLING,
  WIND_OVERTURNING, LIGHTNING_STRIKE,
  TORSION_BREAK,      // roadmap — PFSFVectorSolver pending
  FATIGUE_CRACK       // roadmap — cycle-counting pending
}
```

## 12. References / 參考文獻

**Numerical methods / 數值方法**

- Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems* (2nd ed.). SIAM. — PCG, Jacobi preconditioning.
- Vaněk, P., Mandel, J. & Brezina, M. (1996). *Algebraic Multigrid by Smoothed Aggregation for Second and Fourth Order Elliptic Problems*. **Computing** 56(3), 179–196. — SA-AMG (basis of `AMGPreconditioner`).
- Shinozaki, A. & Oono, Y. (1995). Isotropic discrete Laplacian weights on a cubic lattice. — stencil 1, ½, ⅙ derivation (see attribution note in §9.2).

**Phase-field fracture / 相場斷裂**

- Miehe, C., Hofacker, M. & Welschinger, F. (2010). *A phase field model for rate-independent crack propagation: Robust algorithmic implementation based on operator splits.* **Computer Methods in Applied Mechanics and Engineering** 199, 2765–2778. — history-field style irreversibility.
- Ambati, M., Gerasimov, T. & De Lorenzis, L. (2015). *A review on phase-field models of brittle fracture and a new fast hybrid formulation.* **Computational Mechanics** 55(2), 383–405. — AT2 functional family.
- Gerasimov, T. & De Lorenzis, L. (2018). *On penalization in variational phase-field models of brittle fracture.* arXiv:1811.05334.

**Continuum mechanics / 連體力學**

- Timoshenko, S. P. & Goodier, J. N. (1951). *Theory of Elasticity*. McGraw-Hill. — beam theory, analytic cantilever reference.
- Bažant, Z. P. & Baweja, S. (1989). *Creep and Shrinkage Prediction Model (Model B3).* **Materials and Structures**. — `G_c(t) = G_{c,base} · H^{1.5}` hydration scaling.
- Monaghan, J. J. (1992). *Smoothed Particle Hydrodynamics.* **Annual Review of Astronomy and Astrophysics** 30, 543–574. — cubic spline kernel used by `sph/` visual collapse.
- Teschner, M. et al. (2003). *Optimized Spatial Hashing for Collision Detection of Deformable Objects.* **VMV** 2003.

**Design codes / 設計規範**

- EN 1992-1-1 (Eurocode 2) — Concrete structures.
- EN 1993-1-1 (Eurocode 3) — Steel structures.
- EN 1995-1-1 (Eurocode 5) — Timber structures.
- EN 1996-1-1 (Eurocode 6) — Masonry structures.
- EN 1997-1 (Eurocode 7) — Geotechnical design.
- ASCE 7-22 — Minimum Design Loads.

**GPU / Vulkan**

- Khronos Group. *Vulkan Specification 1.3.* — compute pipeline semantics, `VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR`.
- LunarG. *Driver interface to the Vulkan Loader.* — `VK_ICD_FILENAMES` platform-dependent separator semantics (basis of §9.1 Windows fix).

---

## Appendix A — Stencil Weight Table

Full expansion of the 26-connected neighbour offsets used throughout this document. Source: `PFSFStencil.NEIGHBOR_OFFSETS` at `PFSFStencil.java:70-84`.

| Class | Offsets | Weight |
|---|---|---|
| Face (6) | (±1,0,0), (0,±1,0), (0,0,±1) | 1.0 |
| Edge (12) XY | (±1,±1,0) × 4 combinations | 0.5 |
| Edge (12) XZ | (±1,0,±1) × 4 combinations | 0.5 |
| Edge (12) YZ | (0,±1,±1) × 4 combinations | 0.5 |
| Corner (8)   | (±1,±1,±1) × 8 combinations | 1/6 ≈ 0.1666667 |

The sum of weights is `6·1 + 12·0.5 + 8·(1/6) = 6 + 6 + 4/3 ≈ 13.333`, which is the diagonal entry of the isotropic Laplacian per unit conductivity. All kernels (`rbgs_smooth`, `jacobi_smooth`, `pcg_matvec`) divide the off-diagonal contribution by this diagonal implicitly through the Jacobi-style update rule; any discrepancy is detected by `PFSFStencilConsistencyTest`.

---

## Appendix B — File Map

The file map below points from each §3–§8 claim in this document to the source file that backs it.

| Section / Topic | Source of truth |
|---|---|
| §3 Stencil weights | `Block Reality/api/src/main/java/com/blockreality/api/physics/pfsf/PFSFStencil.java:42-43` |
| §3 GLSL header regeneration | `Block Reality/api/src/main/resources/assets/blockreality/shaders/compute/pfsf/stencil_constants.glsl` + Gradle task `:api:generateStencilGlsl` |
| §3 Stencil derivation | `research/paper_data/STENCIL_MATHEMATICS.md` |
| §4.1 RBGS kernel | `.../shaders/compute/pfsf/rbgs_smooth.comp.glsl` |
| §4.1 Chebyshev schedule | `PFSFScheduler` (spectral radius, ω recurrence) |
| §4.2 V-Cycle | `mg_restrict.comp.glsl`, `mg_prolong.comp.glsl`, `jacobi_smooth.comp.glsl` |
| §4.3 PCG | `pcg_matvec.comp.glsl`, `pcg_update.comp.glsl`, `pcg_direction.comp.glsl`, `pcg_dot.comp.glsl` |
| §5 Phase-field update | `phase_field_evolve.comp.glsl`; history write in `rbgs_smooth.comp.glsl:188-190` and `jacobi_smooth.comp.glsl:209-211` |
| §6.2 σ_max normalisation | `Block Reality/api/src/main/java/com/blockreality/api/physics/pfsf/PFSFDataBuilder.java:244-271` (note lines 261-269 for the `maxPhi` exception) |
| §7 Validation test | `Block Reality/api/src/test/java/com/blockreality/api/physics/validation/PaperDataCollectorTest.java` |
| §8 Performance tests | `PerformanceBenchmarkTest.java`, `RealHardwareBenchmarkTest.java` |
| §9 Audit document | `research/PFSF_GPU_ACADEMIC_AUDIT_2026-04-23.md` |
| §10 Build config | `Block Reality/api/build.gradle` (VK_ICD_FILENAMES, BR_BUILD_SHADERS), `L1-native/CMakeLists.txt` |
| §11 Materials | `Block Reality/api/src/main/java/com/blockreality/api/material/DefaultMaterial.java` |

---



---

## License / 授權

MIT
