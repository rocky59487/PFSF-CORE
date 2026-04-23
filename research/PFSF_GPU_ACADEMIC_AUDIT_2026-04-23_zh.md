# PFSF 核心 GPU 故障與學術稽核報告

日期：2026-04-23  
範圍：`Block Reality/api`、`L1-native/libpfsf`、`research/`、相關 `README` 與驗證測試。  
不含範圍：`fastdesign`、`brml`、`BR-NeXT`、`HYBR`、`reborn-ml`。

## 執行摘要 (Executive Summary)

本次稽核發現了兩個獨立原因，導致目前的 PFSF GPU 路徑不可信：

1. `NativeIsolationTest` 重現的 Java/LWJGL Vulkan 故障，主因是 Windows 上的測試環境配置問題，不足以證明是全域性的驅動程式故障。
2. 原生 (native)/JNI Vulkan 路徑可以偵測到真實 GPU，但其 compute shaders 未在 `br_core::SpirvRegistry` 中註冊，導致 `build_compute_pipeline()` 無法建立 RBGS 管線。

在學術方面，核心的相場不可逆性 (phase-field irreversibility) 在概念層面上與文獻廣泛一致，但基準測試 (benchmark) 和論文數據 (paper-data) 輸出尚不符合發布安全性要求：

- `performance_metrics.csv` 包含的是預測的 GPU 耗時，而非實際測得的耗時。
- `real_hardware_performance.csv` 輸出的內容僅有標題列，但測試仍顯示通過。
- `validation_results.csv` 混合了一個計算案例與兩個寫死的佔位符數值。
- 儲存庫 (repo) 文件在 stencil 權重和校準狀態上存在矛盾。

## GPU 根本原因矩陣 (GPU Root Cause Matrix)

| 症狀 | 範圍 | 證據 | 根本原因 | 優先級 |
|---|---|---|---|---|
| Java 測試路徑中 `vkCreateInstance -> VK_ERROR_INCOMPATIBLE_DRIVER` | `僅限測試 (test-only)` (除非另有證明) | `Block Reality/api/build.gradle:146-176`, `:190-210`；Windows 上的 `vulkaninfo --summary` 成功，但一旦強制 `VK_ICD_FILENAMES` 為 Linux 路徑，相同指令即失敗 | 測試和 `pfsfBench` 在 Windows 上注入了 `/usr/share/vulkan/...` 的清單檔案；導致 Vulkan loader 找不到有效的驅動程式 | P0 |
| 原生運行時除非顯式啟用否則不活動 | `全域運行時閘控` | `Block Reality/api/src/main/java/com/blockreality/api/physics/pfsf/NativePFSFRuntime.java:29-55` | 原生後端被 `-Dblockreality.native.pfsf=true` 強制閘控 | P1 |
| 原生後端偵測到 RTX 5070 Ti 但無法建立計算管線 | `僅限原生 (native-only)` | `Block Reality/api/build.gradle:709-722`; `L1-native/CMakeLists.txt:12-32`; `L1-native/libpfsf/CMakeLists.txt:116-124`; `L1-native/libbr_core/src/compute_pipeline.cpp:175-188`; `:api:pfsfValidate` 日誌顯示 `SPIR-V blob missing` | Shader 註冊管線斷開：頂層 CMake 未加入 `shaders/`，Gradle 配置 `BR_BUILD_SHADERS=OFF`，且 `blockreality_pfsf` 未強制連結 `br_shaders` | P0 |

## 重現與解讀 (Reproduction and Interpretation)

### 1. Java/LWJGL 路徑

觀測到的指令：

```powershell
vulkaninfo --summary
$env:VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.json:/usr/share/vulkan/icd.d/lvp_icd.x86_64.json:/usr/share/vulkan/icd.d/lvp_icd.aarch64.json'
vulkaninfo --summary
.\gradlew.bat :api:test --tests "com.blockreality.test.NativeIsolationTest" --stacktrace --no-daemon
```

觀測到的行為：

- `vulkaninfo --summary` 在這台 Windows 機器上列舉出 NVIDIA RTX 5070 Ti Laptop GPU 和 AMD 610M。
- 當 `VK_ICD_FILENAMES` 被設定為 Linux 清單路徑後，相同指令立即失敗，loader 錯誤訊息以 `ERROR_INCOMPATIBLE_DRIVER` 結束。
- `NativeIsolationTest` 在原生程式庫載入成功後，於 `VulkanComputeContext.init()` 期間失敗。

為何這尚非全域性驅動程式故障：

- `Block Reality/api/build.gradle:42-57` 中的 `minecraft { runs { client/server } }` 並未注入 `VK_ICD_FILENAMES`。
- 僅在 `Block Reality/api/build.gradle:146-176` 和 `:190-210` 的 `test {}` 與 `pfsfBench` 任務中存在 Linux 專用的清單覆蓋 (manifest override)。
- Vulkan loader 文件指出 `VK_ICD_FILENAMES` 和 `VK_DRIVER_FILES` 會覆蓋正常的驅動偵測，且路徑分隔符取決於平台。Windows 上使用的是分號 (`;`) 而非冒號 (`:`)。因此在 Windows 上使用 Linux 路徑列表會引導 loader 指向無效的清單，進而完全禁用驅動偵測。來源：[Driver interface to the Vulkan Loader](https://vulkan.lunarg.com/doc/view/latest/windows/LoaderDriverInterface.html)。

結論：

- 目前的 Java 故障應歸類為 Windows 上損壞的驗證環境 (verification harness)。
- 不應將其引用為「真實 Windows/Forge 運行時無法建立 Vulkan 實例」的證據。

### 2. 原生 (Native)/JNI 路徑

觀測到的指令：

```powershell
.\gradlew.bat :api:pfsfValidate --stacktrace --no-daemon
```

觀測到的行為：

- 驗證套件總體報告成功。
- 運行結束時的原生記錄顯示：
  - Vulkan 已在 `NVIDIA GeForce RTX 5070 Ti Laptop GPU` 上初始化。
  - `build_compute_pipeline(compute/pfsf/rbgs_smooth.comp): SPIR-V blob missing`
  - `RBGS pipeline build failed` (RBGS 管線建立失敗)

為何遺失 blob：

- `L1-native/shaders/CMakeLists.txt` 定義了 `br_shaders` 並透過 `br_core::SpirvRegistry::add_deferred_blob(...)` 自動註冊 blob。
- `L1-native/CMakeLists.txt:26-32` 加入了 `libbr_core` 和 `libpfsf`，但從未加入 `shaders/`。
- `Block Reality/api/build.gradle:713-721` 硬編碼了 `-DBR_BUILD_SHADERS=OFF`。
- `L1-native/libpfsf/CMakeLists.txt:116-124` 將 `blockreality_pfsf` 連結至 `pfsf` 和 `pfsf_compute`，但未連結 `br_shaders`。
- `libbr_core/src/compute_pipeline.cpp:184-188` 在活動註冊表和延遲隊列皆不包含該 blob 時，顯式返回失敗。

結論：

- 原生路徑已正確接觸到 GPU。
- 故障在於著色器 (shader) 的打包與註冊，而非 Vulkan 裝置的偵測。
- 這是導致「GPU 無法實際執行 PFSF 內核」可信度最高的根本原因。

## 論文數據來源稽核 (Paper Data Provenance Audit)

| 產出物 | 當前狀態 | 原因 |
|---|---|---|
| `research/paper_data/raw/performance_metrics.csv` | `預測值` | `PerformanceBenchmarkTest` 根據假設的頻寬和手動調整的係數計算 GPU 時間，而非實際分派 GPU 任務：`Block Reality/api/src/test/java/com/blockreality/api/physics/validation/PerformanceBenchmarkTest.java:47-61` |
| `research/paper_data/raw/real_hardware_performance.csv` | `空值但通過` | 當運行時不可用時，`RealHardwareBenchmarkTest` 會提前返回，但 JUnit 測試仍顯示通過：`.../RealHardwareBenchmarkTest.java:24-38`；產生的 CSV 目前僅包含標題列 |
| `research/paper_data/raw/validation_results.csv` | `混合：計算+佔位符` | `PaperDataCollectorTest` 僅計算 `CANTILEVER` (懸臂)；非懸臂案例則從實體佔位符返回 `1.25`：`.../PaperDataCollectorTest.java:50-66` |
| `research/CALIBRATION_REPORT.md` | `過時 / 矛盾` | 仍報告 `Best Error: 999.0%` 且權重為 `0.35 / 0.15`，這與當前的 shader SSOT (單一事實來源) 和當前的驗證 CSV 不符 |

目前的原始輸出：

- `research/paper_data/raw/performance_metrics.csv:1-5`
- `research/paper_data/raw/real_hardware_performance.csv:1`
- `research/paper_data/raw/validation_results.csv:1-4`

## 內部一致性調查結果 (Internal Consistency Findings)

### 1. Stencil 權重內部不一致

目前儲存庫中存在兩個不相容的版本：

- `README.md:208` 指出關鍵不變量為 `EDGE_PENALTY=0.35` 與 `CORNER_PENALTY=0.15`。
- `research/CALIBRATION_REPORT.md:3-5` 重複了 `0.35 / 0.15`。
- `research/paper_data/STENCIL_MATHEMATICS.md:3-10` 則稱等向性 stencil 為 `0.5 / 1/6`。
- `Block Reality/api/src/main/resources/assets/blockreality/shaders/compute/pfsf/stencil_constants.glsl:1-10` 亦標註為 `0.5 / 0.1666667`。
- `Block Reality/api/src/main/java/com/blockreality/api/physics/pfsf/PFSFStencil.java:42-43` 設定 `EDGE_P = 0.5f` 且 `CORNER_P = 0.1666667f`。

影響：

- 現行的 shader/Java 事實來源為 `0.5 / 1/6`。
- README 和校準報告仍描述舊有的校準狀態。
- 任何同時引用兩者的論文文本或基準測試討論目前均不具備自洽性。

### 2. 「真實硬體基準測試」語義不受測試邏輯支持

`Block Reality/api/build/test-results/pfsfValidate/TEST-com.blockreality.api.physics.validation.RealHardwareBenchmarkTest.xml:1-10` 顯示：

- 測試已通過，
- `VulkanComputeContext` 無法使用，
- stderr 仍印出 `ERROR: 5070TI Native Runtime NOT available!`

影響：

- 通過 `:api:pfsfValidate` 的「綠燈」並不代表硬體基準測試已實際執行。
- 將此測試套件視為「GPU 已驗證」在學術上是不安全的。

## 宣稱矩陣 (Claim Matrix)

| 宣稱項目 | 狀態 | 依據 |
|---|---|---|
| 已實現透過 `max(old, psi_e)` 達成 `hField` 的不可逆性 | `支持 (Supported)` | `rbgs_smooth.comp.glsl:188-190` 與 `jacobi_smooth.comp.glsl:209-211` 單調更新 `hField`；`phase_field_evolve.comp.glsl:39` 宣告 `hField` 為唯讀，且 `:109-113` 記錄了唯讀用途 |
| 相場演化在概念上屬於 Miehe/Ambati 歷史場系列 | `概念層面支持` | 儲存庫使用了與文獻系列一致的歷史場和不可逆更新模式；可與 [Miehe et al. 2010](https://crm.sns.it/media/course/3060/miehe%2Bhofacker%2Bwelschinger10.pdf) 和 [Gerasimov & De Lorenzis 2018](https://arxiv.org/abs/1811.05334) 進行對比 |
| 該實現可作為忠實的 "Ambati 2015" 實現呈現 | `需要重新標註` | `phase_field_evolve.comp.glsl:10-19` 和 README 用語直接引用了 Ambati 的評論文章，但儲存庫並未證明其與公式逐一對應；較安全的表述為「Ambati/Miehe 風格的相場演化」 |
| 精確的 `0.5 / 1/6` stencil 獲得儲存庫引用來源的學術證明 | `需要重新標註` | 當前儲存庫稱為 "Shinozaki-Oono standard"，但本次稽核未驗證該標籤是否有精確的主要來源對應；該實現可能仍是合理的等向性工程 stencil，但引用需要更精確 |
| `performance_metrics.csv` 是經驗性的 CPU-vs-GPU 基準測試 | `與儲存庫數據矛盾` | GPU 數值為模型輸出，而非實際分派耗時 |
| `real_hardware_performance.csv` 是當前驗證套件的真實測量基準測試 | `與儲存庫數據矛盾` | 當前測試運行通過但產出空 CSV |
| `validation_results.csv` 是三個情境的解析驗證集 | `過度宣稱` | 僅計算了懸臂樑；拱橋和平板為佔位符 |
| 儲存庫對於 stencil 權重具有連貫的校準狀態 | `與儲存庫數據矛盾` | README/報告稱 `0.35 / 0.15`；現行事實來源稱 `0.5 / 1/6` |

## 建議修復順序 (Recommended Fix Order)

1. 修復 Windows 測試套件的 Vulkan 環境處理。
   - 將 `VK_ICD_FILENAMES` 的覆蓋限制在 Linux CI，或切換至平台上正確的數值。
   - 除非在測試之外重現了真實的運行時故障，否則保持 `runClient/runServer` 不變。
2. 修復原生著色器建置連結。
   - 將 `L1-native/shaders/` 連接到頂層 CMake 圖中。
   - 當預期進行原生著色器打包時，停止強制設定 `BR_BUILD_SHADERS=OFF`。
   - 確保 `blockreality_pfsf` 或其依賴鏈強制連結 `br_shaders`。
3. 修復基準測試語義。
   - 將預測指標重新命名為「預測值」而非「測量值」。
   - 當 `RealHardwareBenchmarkTest` 未輸出數據時，應判定為失敗。
   - 從論文數據收集移除實體佔位符，或將其標註為佔位符。
4. 修復過時的研究/文件產出物。
   - 協調 `README`、`CALIBRATION_REPORT.md`、`STENCIL_MATHEMATICS.md` 與現行 shader 的事實來源。

## 外部參考文獻 (External References)

- Vulkan loader 環境變數行為：[Driver interface to the Vulkan Loader](https://vulkan.lunarg.com/doc/view/latest/windows/LoaderDriverInterface.html)
- Vulkan 實例建立與初始化規則：[Vulkan Initialization Spec](https://github.khronos.org/Vulkan-Site/spec/latest/chapters/initialization.html)
- 相場斷裂中的歷史場風格不可逆性：[Miehe et al. 2010](https://crm.sns.it/media/course/3060/miehe%2Bhofacker%2Bwelschinger10.pdf)
- 後續關於不可逆性強制執行與歷史場技術對比的討論：[Gerasimov & De Lorenzis 2018](https://arxiv.org/abs/1811.05334)

## 底線結論 (Bottom Line)

目前儲存庫並不支持「由於機器驅動程式損壞而無法使用 GPU」的武斷說法。更強有力、且具備證據支持的陳述應為：

- Java 驗證路徑因 Windows 不相容的測試環境覆蓋而損壞。
- 原生驗證路徑可以觸及真實 GPU，但由於其 SPIR-V 註冊表連結不完整，無法執行 PFSF 計算核心。

學術核心在建模層面部分健全，但在預測、缺失和佔位符輸出與真實測量值明確區分開之前，基準測試和論文數據管線尚不具備發布安全性。
