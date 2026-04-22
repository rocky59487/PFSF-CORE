/**
 * @file island_buffer.h
 * @brief GPU buffer set for a single structure island.
 *
 * Mirrors Java PFSFIslandBuffer — flat 3D array layout:
 *   flat_index = x + Lx * (y + Ly * z)
 *
 * SoA conductivity layout:
 *   conductivity[dir * N + flat_index]  (dir ∈ [0,5])
 */
#pragma once

#include <vulkan/vulkan.h>
#include <pfsf/pfsf_types.h>
#include <cstdint>

namespace pfsf {

class VulkanContext;

struct IslandBuffer {
    // ── Grid dimensions ──
    int32_t  island_id = -1;
    pfsf_pos origin{};
    int32_t  lx = 0, ly = 0, lz = 0;

    int64_t N() const { return static_cast<int64_t>(lx) * ly * lz; }

    int64_t flatIndex(int32_t x, int32_t y, int32_t z) const {
        int64_t ix = static_cast<int64_t>(x) - origin.x;
        int64_t iy = static_cast<int64_t>(y) - origin.y;
        int64_t iz = static_cast<int64_t>(z) - origin.z;
        if (ix < 0 || ix >= lx || iy < 0 || iy >= ly || iz < 0 || iz >= lz) return -1;
        return ix + static_cast<int64_t>(lx) * (iy + static_cast<int64_t>(ly) * iz);
    }

    // ── GPU buffer handles (buffer + memory pairs) ──

    // Phi flip buffers (Chebyshev)
    VkBuffer phi_buf_a     = VK_NULL_HANDLE; VkDeviceMemory phi_mem_a     = VK_NULL_HANDLE;
    VkBuffer phi_buf_b     = VK_NULL_HANDLE; VkDeviceMemory phi_mem_b     = VK_NULL_HANDLE;
    bool     phi_flip      = false;   // false → A is current, true → B is current

    // Source (self-weight)
    VkBuffer source_buf    = VK_NULL_HANDLE; VkDeviceMemory source_mem    = VK_NULL_HANDLE;

    // Conductivity (6N floats, SoA)
    VkBuffer cond_buf      = VK_NULL_HANDLE; VkDeviceMemory cond_mem      = VK_NULL_HANDLE;

    // Type (N bytes)
    VkBuffer type_buf      = VK_NULL_HANDLE; VkDeviceMemory type_mem      = VK_NULL_HANDLE;

    // Failure flags (N bytes)
    VkBuffer fail_buf      = VK_NULL_HANDLE; VkDeviceMemory fail_mem      = VK_NULL_HANDLE;

    // MaxPhi (per-voxel limit)
    VkBuffer max_phi_buf   = VK_NULL_HANDLE; VkDeviceMemory max_phi_mem   = VK_NULL_HANDLE;

    // Rcomp (compression strength)
    VkBuffer rcomp_buf     = VK_NULL_HANDLE; VkDeviceMemory rcomp_mem     = VK_NULL_HANDLE;

    // Rtens (tension strength)
    VkBuffer rtens_buf     = VK_NULL_HANDLE; VkDeviceMemory rtens_mem     = VK_NULL_HANDLE;

    // Phase-field fracture
    VkBuffer h_field_buf   = VK_NULL_HANDLE; VkDeviceMemory h_field_mem   = VK_NULL_HANDLE;
    VkBuffer d_field_buf   = VK_NULL_HANDLE; VkDeviceMemory d_field_mem   = VK_NULL_HANDLE;

    // Hydration (curing)
    VkBuffer hydration_buf = VK_NULL_HANDLE; VkDeviceMemory hydration_mem = VK_NULL_HANDLE;

    // Macro-block residual bits — written by RBGS (binding 5) and by
    // failure_scan (binding 7). Must be a dedicated buffer; aliasing onto
    // fail_buf corrupts per-voxel failure codes.
    VkBuffer macro_residual_buf = VK_NULL_HANDLE; VkDeviceMemory macro_residual_mem = VK_NULL_HANDLE;

    // ── PCG state (Jacobi-preconditioned CG) ──
    VkBuffer pcg_r_buf         = VK_NULL_HANDLE; VkDeviceMemory pcg_r_mem         = VK_NULL_HANDLE;
    VkBuffer pcg_z_buf         = VK_NULL_HANDLE; VkDeviceMemory pcg_z_mem         = VK_NULL_HANDLE;
    VkBuffer pcg_p_buf         = VK_NULL_HANDLE; VkDeviceMemory pcg_p_mem         = VK_NULL_HANDLE;
    VkBuffer pcg_ap_buf        = VK_NULL_HANDLE; VkDeviceMemory pcg_ap_mem        = VK_NULL_HANDLE;
    VkBuffer pcg_partial_buf   = VK_NULL_HANDLE; VkDeviceMemory pcg_partial_mem   = VK_NULL_HANDLE;
    VkBuffer pcg_reduction_buf = VK_NULL_HANDLE; VkDeviceMemory pcg_reduction_mem = VK_NULL_HANDLE;
    VkBuffer pcg_inv_diag_buf  = VK_NULL_HANDLE; VkDeviceMemory pcg_inv_diag_mem  = VK_NULL_HANDLE;

    bool hasPCGBuffers() const {
        return pcg_r_buf         != VK_NULL_HANDLE
            && pcg_z_buf         != VK_NULL_HANDLE
            && pcg_p_buf         != VK_NULL_HANDLE
            && pcg_ap_buf        != VK_NULL_HANDLE
            && pcg_partial_buf   != VK_NULL_HANDLE
            && pcg_reduction_buf != VK_NULL_HANDLE
            && pcg_inv_diag_buf  != VK_NULL_HANDLE;
    }

    // ── Aggregation Multigrid (AMG) coarse correction ──
    int32_t  amg_n_coarse = 0;
    VkBuffer amg_aggregation_buf = VK_NULL_HANDLE; VkDeviceMemory amg_aggregation_mem = VK_NULL_HANDLE;
    VkBuffer amg_weights_buf     = VK_NULL_HANDLE; VkDeviceMemory amg_weights_mem     = VK_NULL_HANDLE;
    VkBuffer amg_coarse_r_buf    = VK_NULL_HANDLE; VkDeviceMemory amg_coarse_r_mem    = VK_NULL_HANDLE;
    VkBuffer amg_coarse_diag_buf = VK_NULL_HANDLE; VkDeviceMemory amg_coarse_diag_mem = VK_NULL_HANDLE;
    VkBuffer amg_coarse_phi_buf  = VK_NULL_HANDLE; VkDeviceMemory amg_coarse_phi_mem  = VK_NULL_HANDLE;
    bool     amg_dirty           = true;

    bool hasAMGBuffers() const {
        return amg_aggregation_buf != VK_NULL_HANDLE
            && amg_weights_buf     != VK_NULL_HANDLE
            && amg_coarse_r_buf    != VK_NULL_HANDLE
            && amg_coarse_diag_buf != VK_NULL_HANDLE
            && amg_coarse_phi_buf  != VK_NULL_HANDLE;
    }

    bool allocateAMG(VulkanContext& vk, int32_t n_coarse);
    bool uploadAMGData(VulkanContext& vk,
                       const int32_t* aggregation, const float* weights,
                       const float* diag_c, int64_t n_fine, int32_t n_coarse);

    // ── Multigrid coarse-grid state ──
    int32_t  lx_l1 = 0, ly_l1 = 0, lz_l1 = 0;
    int32_t  lx_l2 = 0, ly_l2 = 0, lz_l2 = 0;

    VkBuffer mg_phi_l1      = VK_NULL_HANDLE; VkDeviceMemory mg_phi_l1_mem      = VK_NULL_HANDLE;
    VkBuffer mg_phi_prev_l1 = VK_NULL_HANDLE; VkDeviceMemory mg_phi_prev_l1_mem = VK_NULL_HANDLE;
    VkBuffer mg_source_l1   = VK_NULL_HANDLE; VkDeviceMemory mg_source_l1_mem   = VK_NULL_HANDLE;
    VkBuffer mg_cond_l1     = VK_NULL_HANDLE; VkDeviceMemory mg_cond_l1_mem     = VK_NULL_HANDLE;
    VkBuffer mg_type_l1     = VK_NULL_HANDLE; VkDeviceMemory mg_type_l1_mem     = VK_NULL_HANDLE;

    VkBuffer mg_phi_l2      = VK_NULL_HANDLE; VkDeviceMemory mg_phi_l2_mem      = VK_NULL_HANDLE;
    VkBuffer mg_phi_prev_l2 = VK_NULL_HANDLE; VkDeviceMemory mg_phi_prev_l2_mem = VK_NULL_HANDLE;
    VkBuffer mg_source_l2   = VK_NULL_HANDLE; VkDeviceMemory mg_source_l2_mem   = VK_NULL_HANDLE;
    VkBuffer mg_cond_l2     = VK_NULL_HANDLE; VkDeviceMemory mg_cond_l2_mem     = VK_NULL_HANDLE;
    VkBuffer mg_type_l2     = VK_NULL_HANDLE; VkDeviceMemory mg_type_l2_mem     = VK_NULL_HANDLE;

    bool hasMultigridL1() const {
        return mg_phi_l1    != VK_NULL_HANDLE
            && mg_source_l1 != VK_NULL_HANDLE
            && mg_cond_l1   != VK_NULL_HANDLE
            && mg_type_l1   != VK_NULL_HANDLE;
    }

    bool mg_coarse_dirty = true;
    bool hasMultigridL2() const {
        return mg_phi_l2    != VK_NULL_HANDLE
            && mg_source_l2 != VK_NULL_HANDLE
            && mg_cond_l2   != VK_NULL_HANDLE
            && mg_type_l2   != VK_NULL_HANDLE;
    }
    int64_t nL1() const { return static_cast<int64_t>(lx_l1) * ly_l1 * lz_l1; }
    int64_t nL2() const { return static_cast<int64_t>(lx_l2) * ly_l2 * lz_l2; }

    // ── Sparse voxel update ──
    static constexpr std::int32_t MAX_SPARSE_UPDATES_PER_TICK = 512;
    static constexpr std::int32_t SPARSE_RECORD_BYTES         = 48;

    VkBuffer sparse_upload_buf    = VK_NULL_HANDLE;
    void*    sparse_upload_mapped = nullptr;
    std::int32_t sparse_upload_capacity = 0;

    bool hasSparseUpload() const {
        return sparse_upload_buf != VK_NULL_HANDLE && sparse_upload_mapped != nullptr;
    }

    VkBuffer staging_buf   = VK_NULL_HANDLE; VkDeviceMemory staging_mem   = VK_NULL_HANDLE;

    bool     dirty           = true;
    bool     allocated       = false;
    int      chebyshev_iter  = 0;
    float    max_phi_prev    = 0.0f;
    float    max_phi_prev2   = 0.0f;
    bool     damping_active  = false;

    float    prev_max_macro_residual = 0.0f;
    float    last_max_macro_residual = 0.0f;

    struct HostAddrs {
        const void* phi          = nullptr; std::int64_t phi_bytes          = 0;
        const void* source       = nullptr; std::int64_t source_bytes       = 0;
        const void* conductivity = nullptr; std::int64_t conductivity_bytes = 0;
        const void* voxel_type   = nullptr; std::int64_t voxel_type_bytes   = 0;
        const void* rcomp        = nullptr; std::int64_t rcomp_bytes        = 0;
        const void* rtens        = nullptr; std::int64_t rtens_bytes        = 0;
        const void* max_phi      = nullptr; std::int64_t max_phi_bytes      = 0;
        bool registered          = false;

        const void*  material_id     = nullptr; std::int64_t material_id_bytes     = 0;
        const void*  anchor_bitmap   = nullptr; std::int64_t anchor_bitmap_bytes   = 0;
        const void*  fluid_pressure  = nullptr; std::int64_t fluid_pressure_bytes  = 0;
        const void*  curing          = nullptr; std::int64_t curing_bytes          = 0;
        bool         lookups_registered = false;

        void*        stress_out  = nullptr;
        std::int64_t stress_bytes = 0;
    } hosts;

    void markDirty()  { dirty = true; }
    void markClean()  { dirty = false; }

    bool allocate(VulkanContext& vk, bool with_phase_field);
    bool allocatePCG(VulkanContext& vk);
    bool allocateMultigrid(VulkanContext& vk);
    bool allocateSparseUpload(VulkanContext& vk);
    bool uploadFromHosts(VulkanContext& vk);
    bool uploadMultigridData(VulkanContext& vk);
    bool readbackPhi(VulkanContext& vk, float* out, std::int32_t cap_floats,
                     std::int32_t* out_count);
    bool readbackFailures(VulkanContext& vk, void* dbb_addr, std::int64_t dbb_bytes);

    std::int32_t numMacroBlocks() const {
        constexpr std::int32_t MB = 8;
        const std::int32_t mbx = (lx + MB - 1) / MB;
        const std::int32_t mby = (ly + MB - 1) / MB;
        const std::int32_t mbz = (lz + MB - 1) / MB;
        return mbx * mby * mbz;
    }

    void recordClearMacroResiduals(VkCommandBuffer cmd);
    float readbackMacroResidualMax(VulkanContext& vk);
    void free(VulkanContext& vk);
};

} // namespace pfsf
