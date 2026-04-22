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
option(BR_BUILD_SHADERS "Precompile GLSL shaders to SPIR-V (needs glslang)" OFF)
option(BR_BUILD_TESTS   "Build native gtest suite" OFF)
option(BR_VK_STRICT     "Enable Vulkan strict validation layers" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ═════════════════════════════════════════════════════════════════════════════
#  子目錄 (Subdirectories)
# ═════════════════════════════════════════════════════════════════════════════

# libbr_core — SHARED foundation (Vulkan context, compute wrappers).
add_subdirectory(libbr_core)

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
