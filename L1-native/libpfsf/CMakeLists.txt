cmake_minimum_required(VERSION 3.20)
project(pfsf
    VERSION 0.1.0
    DESCRIPTION "Block Reality PFSF physics solver — standalone C++ library"
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── v0.4 M1c — pinned ABI contract version ──────────────────────────
if(NOT DEFINED PFSF_ABI_VERSION_STRING OR PFSF_ABI_VERSION_STRING STREQUAL "")
    file(READ "${CMAKE_CURRENT_SOURCE_DIR}/abi/pfsf_v1.abi.json" _abi_json)
    string(REGEX MATCH "\"abi_version\"[ \t]*:[ \t]*\"([0-9]+\\.[0-9]+\\.[0-9]+)\""
           _abi_match "${_abi_json}")
    if(_abi_match)
        set(PFSF_ABI_VERSION_STRING "${CMAKE_MATCH_1}")
    else()
        message(FATAL_ERROR "libpfsf: could not parse abi_version from pfsf_v1.abi.json.")
    endif()
endif()
message(STATUS "libpfsf: pinned ABI contract version = ${PFSF_ABI_VERSION_STRING}")

# ── Vulkan SDK ──
find_package(Vulkan REQUIRED)

# ── Vulkan Memory Allocator (VMA) ──
if(NOT TARGET VulkanMemoryAllocator)
    include(FetchContent)
    FetchContent_Declare(vma
        GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
        GIT_TAG        v3.1.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(vma)
endif()

# ── JNI ──
find_package(JNI QUIET)
option(PFSF_BUILD_JNI "Build JNI bridge for Java integration" ${JNI_FOUND})

# ── Build options ──
option(PFSF_BUILD_TESTS   "Build unit tests"           OFF)
option(PFSF_BUILD_EXAMPLE "Build standalone CLI example" ON)
option(BUILD_SHARED_LIBS   "Build shared library"       ON)

# ── Sources ──
set(PFSF_SOURCES
    src/pfsf_api.cpp
    src/pfsf_engine.cpp
    src/core/vulkan_context.cpp
    src/core/island_buffer.cpp
    src/core/buffer_manager.cpp
    src/solver/jacobi_solver.cpp
    src/solver/vcycle_solver.cpp
    src/solver/phase_field.cpp
    src/solver/pcg_solver.cpp
    src/solver/failure_scan.cpp
    src/solver/sparse_scatter.cpp
    src/solver/dispatcher.cpp
    src/solver/dispatcher_pcg.cpp
    src/solver/dispatcher_vcycle.cpp
    src/solver/amg_builder.cpp
    src/solver/dispatcher_amg.cpp
)

# ── pfsf_compute static target ──
add_library(pfsf_compute STATIC
    src/compute/version.cpp
    src/compute/normalize.cpp
    src/compute/wind_bias.cpp
    src/compute/timoshenko.cpp
    src/compute/wind_pressure.cpp
    src/compute/arm_map.cpp
    src/compute/arch_factor.cpp
    src/compute/phantom_edges.cpp
    src/compute/morton.cpp
    src/compute/downsample.cpp
    src/compute/chebyshev.cpp
    src/compute/divergence.cpp
    src/compute/features.cpp
    src/compute/conductivity.cpp
    src/compute/aug_kernels.cpp
    src/ext/augmentation.cpp
    src/ext/hooks.cpp
    src/ext/plan_dispatcher.cpp
    src/trace/ring_buffer.cpp
    src/diag/crash.cpp
)
target_include_directories(pfsf_compute PUBLIC include PRIVATE src/compute)
set_target_properties(pfsf_compute PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(pfsf_compute PRIVATE PFSF_ABI_VERSION_STRING="${PFSF_ABI_VERSION_STRING}")

# ── Library target ──
add_library(pfsf_static STATIC ${PFSF_SOURCES})
target_include_directories(pfsf_static PUBLIC include PRIVATE src)
target_link_libraries(pfsf_static PUBLIC Vulkan::Vulkan PRIVATE VulkanMemoryAllocator)

add_library(pfsf SHARED ${PFSF_SOURCES})
target_include_directories(pfsf PUBLIC include PRIVATE src)
target_link_libraries(pfsf PUBLIC Vulkan::Vulkan PRIVATE VulkanMemoryAllocator)

if(NOT TARGET br_core)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../libbr_core ${CMAKE_CURRENT_BINARY_DIR}/libbr_core)
endif()
target_link_libraries(pfsf_static PUBLIC br_core)
target_link_libraries(pfsf PUBLIC br_core)
target_compile_definitions(pfsf_static PRIVATE PFSF_USE_BR_CORE=1)
target_compile_definitions(pfsf PRIVATE PFSF_USE_BR_CORE=1)

# ── JNI bridge ──
if(PFSF_BUILD_JNI AND JNI_FOUND)
    add_library(blockreality_pfsf SHARED src/jni/native_pfsf_bridge.cpp)
    target_link_libraries(blockreality_pfsf PRIVATE pfsf JNI::JNI)
    target_include_directories(blockreality_pfsf PRIVATE ${JNI_INCLUDE_DIRS})
    target_link_libraries(blockreality_pfsf PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,pfsf_compute>)
    target_compile_definitions(blockreality_pfsf PRIVATE PFSF_USE_BR_CORE=1)
    # Force-link the SPIR-V registry so its static initialisers (which
    # register each compute kernel into br_core::SpirvRegistry at library
    # load time) are NOT dead-stripped. Without WHOLE_ARCHIVE, modern
    # linkers drop the translation units whose symbols aren't directly
    # referenced, and build_compute_pipeline() reports
    # "SPIR-V blob missing" at first dispatch.
    if(TARGET br_shaders)
        target_link_libraries(blockreality_pfsf PRIVATE $<LINK_LIBRARY:WHOLE_ARCHIVE,br_shaders>)
    endif()
    if(NOT WIN32)
        set_target_properties(blockreality_pfsf PROPERTIES PREFIX "lib")
    endif()
    set_target_properties(blockreality_pfsf PROPERTIES MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

# --- HARDWARE PROOF ---
# Research-only executable. Disabled by default because the source file
# lives under research/ and is not part of shipping builds; was also the
# only target pulling in a developer-absolute VMA path.
option(PFSF_BUILD_HARDWARE_PROOF "Build the research-only hardware-proof executable" OFF)
if(PFSF_BUILD_HARDWARE_PROOF)
    add_executable(pfsf_hardware_proof ../../research/gpu_hardproof.cpp)
    # Resolve VMA headers through the FetchContent_Declare target populated
    # earlier (vma is MakeAvailable'd near the top of this file); this
    # removes the hardcoded C:/Users/... include path that broke every
    # non-dev-machine build.
    target_include_directories(pfsf_hardware_proof PRIVATE src include)
    target_link_libraries(pfsf_hardware_proof PRIVATE
        pfsf_static pfsf_compute br_core Vulkan::Vulkan VulkanMemoryAllocator)
endif()

# ── Standalone CLI example ──
if(PFSF_BUILD_EXAMPLE)
    add_executable(pfsf_cli
        src/example/main.cpp
        src/example/pfsf_cli_args.cpp
        src/example/fixture_loader.cpp
        src/example/cpu_backend.cpp
        src/example/vk_backend.cpp
    )
    target_link_libraries(pfsf_cli PRIVATE pfsf_static $<LINK_LIBRARY:WHOLE_ARCHIVE,pfsf_compute>)
endif()

# ── Install ──
install(TARGETS pfsf LIBRARY DESTINATION lib ARCHIVE DESTINATION lib RUNTIME DESTINATION bin)
install(DIRECTORY include/pfsf DESTINATION include)

message(STATUS "[libpfsf] v${PROJECT_VERSION} — Vulkan ${Vulkan_VERSION}")
