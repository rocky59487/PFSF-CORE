# PFSF Core GPU Failure and Academic Audit

Date: 2026-04-23  
Scope: `Block Reality/api`, `L1-native/libpfsf`, `research/`, related `README` and validation tests.  
Out of scope: `fastdesign`, `brml`, `BR-NeXT`, `HYBR`, `reborn-ml`.

## Executive Summary

This audit found two independent reasons the PFSF GPU path is currently not trustworthy:

1. The Java/LWJGL Vulkan failure reproduced by `NativeIsolationTest` is primarily a test harness configuration problem on Windows, not sufficient evidence of a runtime-wide driver failure.
2. The native/JNI Vulkan path can see the real GPU, but its compute shaders are not registered into `br_core::SpirvRegistry`, so `build_compute_pipeline()` cannot create the RBGS pipeline.

On the academic side, the core phase-field irreversibility story is broadly aligned with the literature at the concept level, but the benchmark and paper-data outputs are not publication-safe:

- `performance_metrics.csv` contains predicted GPU timings, not measured ones.
- `real_hardware_performance.csv` is emitted with only a header while the test still passes.
- `validation_results.csv` mixes one computed case with two hardcoded placeholder values.
- repo documents disagree on stencil weights and calibration status.

## GPU Root Cause Matrix

| Symptom | Scope | Evidence | Root Cause | Priority |
|---|---|---|---|---|
| `vkCreateInstance -> VK_ERROR_INCOMPATIBLE_DRIVER` in Java test path | `test-only` until proven otherwise | `Block Reality/api/build.gradle:146-176`, `:190-210`; `vulkaninfo --summary` succeeds on Windows, but the same command fails once `VK_ICD_FILENAMES` is forced to Linux paths | test and `pfsfBench` inject `/usr/share/vulkan/...` manifests on Windows; Vulkan loader then finds no valid drivers | P0 |
| Native runtime inactive unless explicitly enabled | `runtime-wide gate` | `Block Reality/api/src/main/java/com/blockreality/api/physics/pfsf/NativePFSFRuntime.java:29-55` | native backend is hard-gated by `-Dblockreality.native.pfsf=true` | P1 |
| Native backend sees RTX 5070 Ti but cannot build compute pipeline | `native-only` | `Block Reality/api/build.gradle:709-722`; `L1-native/CMakeLists.txt:12-32`; `L1-native/libpfsf/CMakeLists.txt:116-124`; `L1-native/libbr_core/src/compute_pipeline.cpp:175-188`; `:api:pfsfValidate` log prints `SPIR-V blob missing` | shader registry pipeline is disconnected: top-level CMake never adds `shaders/`, Gradle configures `BR_BUILD_SHADERS=OFF`, and `blockreality_pfsf` does not force-link `br_shaders` | P0 |

## Reproduction and Interpretation

### 1. Java/LWJGL path

Observed commands:

```powershell
vulkaninfo --summary
$env:VK_ICD_FILENAMES='/usr/share/vulkan/icd.d/lvp_icd.json:/usr/share/vulkan/icd.d/lvp_icd.x86_64.json:/usr/share/vulkan/icd.d/lvp_icd.aarch64.json'
vulkaninfo --summary
.\gradlew.bat :api:test --tests "com.blockreality.test.NativeIsolationTest" --stacktrace --no-daemon
```

Observed behavior:

- `vulkaninfo --summary` enumerates both the NVIDIA RTX 5070 Ti Laptop GPU and the AMD 610M on this Windows machine.
- The same command fails immediately after `VK_ICD_FILENAMES` is set to Linux manifest paths, with loader errors ending in `ERROR_INCOMPATIBLE_DRIVER`.
- `NativeIsolationTest` fails after native library load succeeds, during `VulkanComputeContext.init()`.

Why this is not yet a runtime-wide driver failure:

- `minecraft { runs { client/server } }` in `Block Reality/api/build.gradle:42-57` does not inject `VK_ICD_FILENAMES`.
- The Linux-only manifest override is present only in the `test {}` and `pfsfBench` tasks at `Block Reality/api/build.gradle:146-176` and `:190-210`.
- The Vulkan loader documentation states that `VK_ICD_FILENAMES` and `VK_DRIVER_FILES` override normal driver discovery, and that the list separator is platform-dependent. On Windows the list is semicolon-separated, not colon-separated. A Linux path list on Windows therefore points the loader to invalid manifests and can disable driver discovery entirely. Source: [Driver interface to the Vulkan Loader](https://vulkan.lunarg.com/doc/view/latest/windows/LoaderDriverInterface.html).

Conclusion:

- The current Java failure is best classified as a broken verification harness on Windows.
- It should not be cited as proof that the real Windows/Forge runtime cannot create a Vulkan instance.

### 2. Native/JNI path

Observed command:

```powershell
.\gradlew.bat :api:pfsfValidate --stacktrace --no-daemon
```

Observed behavior:

- Validation suite reports success overall.
- End-of-run native log states:
  - Vulkan initialized on `NVIDIA GeForce RTX 5070 Ti Laptop GPU`
  - `build_compute_pipeline(compute/pfsf/rbgs_smooth.comp): SPIR-V blob missing`
  - `RBGS pipeline build failed`

Why the blob is missing:

- `L1-native/shaders/CMakeLists.txt` defines `br_shaders` and auto-registers blobs through `br_core::SpirvRegistry::add_deferred_blob(...)`.
- `L1-native/CMakeLists.txt:26-32` adds `libbr_core` and `libpfsf`, but never adds `shaders/`.
- `Block Reality/api/build.gradle:713-721` hardcodes `-DBR_BUILD_SHADERS=OFF`.
- `L1-native/libpfsf/CMakeLists.txt:116-124` links `blockreality_pfsf` against `pfsf` and `pfsf_compute`, but not `br_shaders`.
- `libbr_core/src/compute_pipeline.cpp:184-188` explicitly returns failure when neither the live registry nor the deferred queue contains the blob.

Conclusion:

- The native path is reaching the GPU correctly.
- The failure is in shader packaging and registration, not in Vulkan device discovery.
- This is the highest-confidence root cause for “GPU cannot actually run PFSF kernels”.

## Paper Data Provenance Audit

| Artifact | Current status | Why |
|---|---|---|
| `research/paper_data/raw/performance_metrics.csv` | `Predicted` | `PerformanceBenchmarkTest` computes GPU time from assumed bandwidth and a hand-tuned factor instead of dispatching GPU work: `Block Reality/api/src/test/java/com/blockreality/api/physics/validation/PerformanceBenchmarkTest.java:47-61` |
| `research/paper_data/raw/real_hardware_performance.csv` | `Empty-but-pass` | `RealHardwareBenchmarkTest` returns early when runtime is unavailable, but the JUnit test still passes: `.../RealHardwareBenchmarkTest.java:24-38`; the generated CSV currently contains only the header |
| `research/paper_data/raw/validation_results.csv` | `Mixed: computed + placeholder` | `PaperDataCollectorTest` computes only `CANTILEVER`; non-cantilever cases return `1.25` from a literal placeholder: `.../PaperDataCollectorTest.java:50-66` |
| `research/CALIBRATION_REPORT.md` | `Stale / contradicted` | still reports `Best Error: 999.0%` and weights `0.35 / 0.15`, which disagree with the current shader SSOT and current validation CSV |

Current raw outputs:

- `research/paper_data/raw/performance_metrics.csv:1-5`
- `research/paper_data/raw/real_hardware_performance.csv:1`
- `research/paper_data/raw/validation_results.csv:1-4`

## Internal Consistency Findings

### 1. Stencil weights are internally inconsistent

The repo currently contains two incompatible stories:

- `README.md:208` says the critical invariant is `EDGE_PENALTY=0.35` and `CORNER_PENALTY=0.15`.
- `research/CALIBRATION_REPORT.md:3-5` repeats `0.35 / 0.15`.
- `research/paper_data/STENCIL_MATHEMATICS.md:3-10` says the isotropic stencil is `0.5 / 1/6`.
- `Block Reality/api/src/main/resources/assets/blockreality/shaders/compute/pfsf/stencil_constants.glsl:1-10` also says `0.5 / 0.1666667`.
- `Block Reality/api/src/main/java/com/blockreality/api/physics/pfsf/PFSFStencil.java:42-43` sets `EDGE_P = 0.5f` and `CORNER_P = 0.1666667f`.

Implication:

- The live shader/Java source of truth is `0.5 / 1/6`.
- The README and calibration report still describe an older calibration state.
- Any paper text or benchmark discussion that cites both is currently not self-consistent.

### 2. “Real hardware benchmark” wording is not supported by the test semantics

`Block Reality/api/build/test-results/pfsfValidate/TEST-com.blockreality.api.physics.validation.RealHardwareBenchmarkTest.xml:1-10` shows:

- the test passed,
- `VulkanComputeContext` was not available,
- stderr still printed `ERROR: 5070TI Native Runtime NOT available!`

Implication:

- A green `:api:pfsfValidate` run does not prove the hardware benchmark executed.
- Treating this suite as “GPU validated” is academically unsafe.

## Claim Matrix

| Claim | Status | Basis |
|---|---|---|
| `hField` irreversibility via `max(old, psi_e)` is implemented | `Supported` | `rbgs_smooth.comp.glsl:188-190` and `jacobi_smooth.comp.glsl:209-211` update `hField` monotonically; `phase_field_evolve.comp.glsl:39` declares `hField` readonly and `:109-113` documents read-only use |
| phase-field evolution is conceptually in the Miehe/Ambati history-field family | `Supported at concept level` | the repo uses a history field and irreversible update pattern consistent with the literature family; compare with [Miehe et al. 2010](https://crm.sns.it/media/course/3060/miehe%2Bhofacker%2Bwelschinger10.pdf) and [Gerasimov & De Lorenzis 2018](https://arxiv.org/abs/1811.05334) |
| the implementation can be presented as a faithful “Ambati 2015” implementation | `Needs relabeling` | `phase_field_evolve.comp.glsl:10-19` and README language cite Ambati’s review directly, but the repo does not demonstrate equation-by-equation equivalence; safer wording is “Ambati/Miehe-style phase-field evolution” |
| the exact `0.5 / 1/6` stencil is academically justified by the repo’s cited sources | `Needs relabeling` | current repo says “Shinozaki-Oono standard”, but this audit did not verify an exact primary-source match for that label; the implementation may still be a reasonable isotropic engineering stencil, but the citation needs tightening |
| `performance_metrics.csv` is an empirical CPU-vs-GPU benchmark | `Contradicted by repo data` | GPU values are model outputs, not dispatch timings |
| `real_hardware_performance.csv` is a real measured benchmark from the current validation suite | `Contradicted by repo data` | current test run passes while producing an empty CSV |
| `validation_results.csv` is a three-scenario analytic validation set | `Overclaimed` | only cantilever is computed; arch and slab are placeholders |
| the repo has one coherent calibration state for stencil weights | `Contradicted by repo data` | README/report say `0.35 / 0.15`; live SSOT says `0.5 / 1/6` |

## Recommended Fix Order

1. Fix Windows test harness Vulkan env handling.
   - Restrict `VK_ICD_FILENAMES` overrides to Linux CI only, or switch to platform-correct values.
   - Keep `runClient/runServer` unchanged unless a real runtime failure is reproduced outside tests.
2. Repair native shader build wiring.
   - connect `L1-native/shaders/` into the top-level CMake graph,
   - stop forcing `BR_BUILD_SHADERS=OFF` when native shader packaging is intended,
   - ensure `blockreality_pfsf` or its dependency chain force-links `br_shaders`.
3. Repair benchmark semantics.
   - rename predicted metrics as predicted, not measured,
   - fail `RealHardwareBenchmarkTest` when it emits no data,
   - remove literal placeholders from paper-data collection or label them as placeholders.
4. Repair stale research/docs artifacts.
   - reconcile `README`, `CALIBRATION_REPORT.md`, `STENCIL_MATHEMATICS.md`, and the live shader SSOT.

## External References

- Vulkan loader environment-variable behavior: [Driver interface to the Vulkan Loader](https://vulkan.lunarg.com/doc/view/latest/windows/LoaderDriverInterface.html)
- Vulkan instance creation and initialization rules: [Vulkan Initialization Spec](https://github.khronos.org/Vulkan-Site/spec/latest/chapters/initialization.html)
- History-field style irreversibility in phase-field fracture: [Miehe et al. 2010](https://crm.sns.it/media/course/3060/miehe%2Bhofacker%2Bwelschinger10.pdf)
- Later discussion of irreversibility enforcement and comparison with history-field techniques: [Gerasimov & De Lorenzis 2018](https://arxiv.org/abs/1811.05334)

## Bottom Line

The repo does not currently support the blanket statement “GPU cannot be used because the machine driver is broken”. The stronger, evidence-backed statement is:

- the Java verification path is broken by a Windows-incompatible test environment override, and
- the native verification path reaches the real GPU but cannot execute PFSF compute kernels because its SPIR-V registry wiring is incomplete.

The academic core is partially sound at the modeling level, but the benchmark and paper-data pipeline is not publication-safe until predicted, missing, and placeholder outputs are explicitly separated from real measurements.
