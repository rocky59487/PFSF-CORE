cmake_minimum_required(VERSION 3.20)
project(blockreality_native
    VERSION 0.3.0
    DESCRIPTION "Block Reality v0.3c — external C++ engine (core/pfsf)"
    LANGUAGES C CXX
)

# ═════════════════════════════════════════════════════════════════════════════
#  編譯選項 (Build Options)
# ═════════════════════════════════════════════════════════════════════════════

option(BR_BUILD_PFSF    "Build Phase-Field Structural Fracture (PFSF)" ON)
option(BR_BUILD_FLUID   "Build Fluid Simulation" OFF)
option(BR_BUILD_RENDER  "Build Ray-Trace Renderer" OFF)
# Build the SPIR-V registry library (br_shaders). Default ON so the native
# runtime can actually dispatch compute pipelines; can be disabled for
# headless-only builds that don't touch Vulkan. Requires either
# glslangValidator on PATH or -DPFSF_ALLOW_PREBUILT_SPV=ON.
option(BR_BUILD_SHADERS "Build SPIR-V registry (br_shaders)" ON)
option(BR_BUILD_TESTS   "Build native gtest suite" OFF)
option(BR_VK_STRICT     "Enable Vulkan strict validation layers" OFF)

# When glslang is unavailable locally, fall back to pre-built SPIR-V
# shipped in L1-native/shaders/prebuilt/. CI stores those blobs and
# a drift check keeps them honest.
option(PFSF_ALLOW_PREBUILT_SPV
       "Fall back to L1-native/shaders/prebuilt/*.spv when glslang is missing"
       ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif()
    endforeach()
endif()

# ★ 修正 SxS / VCRUNTIME 衝突：強制靜態連結 CRT
# 因為 Java JVM 已經載入了舊版的 msvcp140.dll，而 VS2026 編譯出的 DLL 需要新版，
# 導致 Side-by-Side (SxS) 錯誤。靜態連結 (/MT) 可徹底解決此問題。
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE
        CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif()
    endforeach()
endif()

# ═════════════════════════════════════════════════════════════════════════════
#  子目錄 (Subdirectories)
# ═════════════════════════════════════════════════════════════════════════════

# libbr_core — SHARED foundation (Vulkan context, compute wrappers).
add_subdirectory(libbr_core)

# br_shaders — SPIR-V blob registry. Must be added BEFORE libpfsf so the
# pfsf target can link against it. The previous release wired shaders
# through a separate path, which left `br_shaders` undefined at link time
# and triggered "SPIR-V blob missing" at runtime
# (see research/PFSF_GPU_ACADEMIC_AUDIT_2026-04-23.md §3).
if(BR_BUILD_SHADERS)
    add_subdirectory(shaders)
endif()

# libpfsf — core solver logic (SoA structure, Jacobi/RBGS, phase-field).
if(BR_BUILD_PFSF)
    add_subdirectory(libpfsf)
endif()

# ═════════════════════════════════════════════════════════════════════════════
#  Diagnostic Summary
# ═════════════════════════════════════════════════════════════════════════════

message(STATUS "[blockreality_native] v${PROJECT_VERSION}")
message(STATUS "  BR_BUILD_PFSF    = ${BR_BUILD_PFSF}")
message(STATUS "  BR_BUILD_FLUID   = ${BR_BUILD_FLUID}")
message(STATUS "  BR_BUILD_RENDER  = ${BR_BUILD_RENDER}")
message(STATUS "  BR_BUILD_SHADERS = ${BR_BUILD_SHADERS}")
message(STATUS "  BR_BUILD_TESTS   = ${BR_BUILD_TESTS}")
message(STATUS "  BR_VK_STRICT     = ${BR_VK_STRICT}")
