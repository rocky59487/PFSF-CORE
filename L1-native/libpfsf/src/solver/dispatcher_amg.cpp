/**
 * @file dispatcher_amg.cpp
 * @brief AMG coarse-grid correction recording.
 *
 * One application per PCG initialization:
 *   zero(amg_coarse_r) → restrict(r_f → r_c) → coarse_jacobi(r_c → e_c) → prolong(e_c → phi)
 *
 * The correction phi += P·D_c^{-1}·P^T·r reduces the initial low-frequency
 * error before PCG iterates, giving faster convergence especially on
 * near-isotropic islands where the V-cycle coarsening is not aligned with
 * the error modes.
 */

#include "dispatcher.h"
#include "pcg_solver.h"
#include "core/island_buffer.h"
#include "core/vulkan_context.h"

#include <cstdint>
#include <cstdio>

namespace pfsf {

namespace {

void computeBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

void transferToComputeBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{};
    mb.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

VkDescriptorSet allocSet(VkDevice dev, VkDescriptorPool pool,
                          VkDescriptorSetLayout layout) {
    if (dev == VK_NULL_HANDLE || pool == VK_NULL_HANDLE || layout == VK_NULL_HANDLE)
        return VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkDescriptorSetAllocateInfo ai{};
    ai.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    ai.descriptorPool     = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts        = &layout;
    if (vkAllocateDescriptorSets(dev, &ai, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    return set;
}

void writeStorage(VkDevice dev, VkDescriptorSet set,
                   std::uint32_t binding, VkBuffer buf) {
    VkDescriptorBufferInfo bi{ buf, 0, VK_WHOLE_SIZE };
    VkWriteDescriptorSet w{};
    w.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w.dstSet          = set;
    w.dstBinding      = binding;
    w.descriptorCount = 1;
    w.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    w.pBufferInfo     = &bi;
    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
}

constexpr std::uint32_t kWGSize = 256;

std::uint32_t ceilDiv(std::int64_t n, std::uint32_t wg) {
    return static_cast<std::uint32_t>((n + wg - 1) / wg);
}

} // namespace

void Dispatcher::recordAMGCorrection(VkCommandBuffer cmd, IslandBuffer& buf,
                                      VkDescriptorPool pool) {
    if (!pcg_.amgReady() || !buf.hasAMGBuffers()) return;
    if (!buf.hasPCGBuffers()) return;

    VkDevice dev = vk_.device();
    if (dev == VK_NULL_HANDLE) return;

    const std::uint32_t N_fine   = static_cast<std::uint32_t>(buf.N());
    const std::uint32_t N_coarse = static_cast<std::uint32_t>(buf.amg_n_coarse);
    if (N_fine == 0 || N_coarse == 0) return;

    // Step 1: zero coarse_r_buf (uint for atomic CAS; 0 bits = 0.0f).
    // vkCmdFillBuffer requires size multiple of 4; float == 4 bytes, so fine.
    const VkDeviceSize coarseBytes = static_cast<VkDeviceSize>(N_coarse) * sizeof(float);
    vkCmdFillBuffer(cmd, buf.amg_coarse_r_buf, 0, coarseBytes, 0u);
    vkCmdFillBuffer(cmd, buf.amg_coarse_phi_buf, 0, coarseBytes, 0u);
    transferToComputeBarrier(cmd);

    // Step 2: scatter restrict — r_c[agg[i]] += P[i] * r_f[i]
    {
        VkDescriptorSet set = allocSet(dev, pool, pcg_.amgRestrictLayout());
        if (set == VK_NULL_HANDLE) return;
        writeStorage(dev, set, 0, buf.pcg_r_buf);            // FineResidual
        writeStorage(dev, set, 1, buf.amg_aggregation_buf);  // Aggregation
        writeStorage(dev, set, 2, buf.amg_weights_buf);      // PWeights
        writeStorage(dev, set, 3, buf.amg_coarse_r_buf);     // CoarseSrcBits

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pcg_.amgRestrictPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pcg_.amgRestrictPipelineLayout(), 0, 1, &set, 0, nullptr);

        AMGFinePushConstants pc{ N_fine, N_coarse };
        vkCmdPushConstants(cmd, pcg_.amgRestrictPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, ceilDiv(buf.N(), kWGSize), 1, 1);
        computeBarrier(cmd);
    }

    // Step 3: coarse Jacobi — e_c[i] = r_c[i] / D_c[i]
    {
        VkDescriptorSet set = allocSet(dev, pool, pcg_.amgCoarseJacobiLayout());
        if (set == VK_NULL_HANDLE) return;
        writeStorage(dev, set, 0, buf.amg_coarse_diag_buf);  // DiagC
        writeStorage(dev, set, 1, buf.amg_coarse_r_buf);     // CoarseR (uint bits)
        writeStorage(dev, set, 2, buf.amg_coarse_phi_buf);   // CoarsePhi (ec)

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pcg_.amgCoarseJacobiPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pcg_.amgCoarseJacobiPipelineLayout(), 0, 1, &set, 0, nullptr);

        AMGCoarseJacobiPushConstants pc{ N_coarse };
        vkCmdPushConstants(cmd, pcg_.amgCoarseJacobiPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, ceilDiv(N_coarse, kWGSize), 1, 1);
        computeBarrier(cmd);
    }

    // Step 4: gather prolong — phi_f[i] += P[i] * e_c[agg[i]]
    {
        VkDescriptorSet set = allocSet(dev, pool, pcg_.amgProlongLayout());
        if (set == VK_NULL_HANDLE) return;
        VkBuffer phi = buf.phi_flip ? buf.phi_buf_b : buf.phi_buf_a;
        writeStorage(dev, set, 0, buf.amg_coarse_phi_buf);   // CoarsePhi
        writeStorage(dev, set, 1, buf.amg_aggregation_buf);  // Aggregation
        writeStorage(dev, set, 2, buf.amg_weights_buf);      // PWeights
        writeStorage(dev, set, 3, phi);                       // FinePhi

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pcg_.amgProlongPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
            pcg_.amgProlongPipelineLayout(), 0, 1, &set, 0, nullptr);

        AMGFinePushConstants pc{ N_fine, N_coarse };
        vkCmdPushConstants(cmd, pcg_.amgProlongPipelineLayout(),
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, ceilDiv(buf.N(), kWGSize), 1, 1);
        computeBarrier(cmd);
    }
}

} // namespace pfsf
