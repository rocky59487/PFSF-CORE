/**
 * @file dispatcher_pcg.cpp
 * @brief PCG (Jacobi-preconditioned Conjugate Gradient) recording.
 */
#include "dispatcher.h"
#include "pcg_solver.h"
#include "core/island_buffer.h"
#include "core/constants.h"
#include "core/vulkan_context.h"
#include <vulkan/vulkan.h>
#include <cstdio>
#include <algorithm>

namespace pfsf {

namespace {

/** Helper: ceiling division. */
inline std::uint32_t ceilDiv(std::int64_t a, std::uint32_t b) {
    return static_cast<std::uint32_t>((a + b - 1) / b);
}

/** Matches the workgroup size used in pcg_matvec / pcg_update shaders. */
constexpr std::uint32_t kWGScan = 256;
/** Matches the dot product reduction element count per workgroup (2 per thread). */
constexpr std::uint32_t kElPerWG = 512;

/** Helper: allocate and bind a storage buffer to a descriptor set. */
void writeStorage(VkDevice dev, VkDescriptorSet set, int binding, VkBuffer buf) {
    if (buf == VK_NULL_HANDLE) return;
    VkDescriptorBufferInfo info{};
    info.buffer = buf;
    info.offset = 0;
    info.range  = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write{};
    write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet          = set;
    write.dstBinding      = binding;
    write.descriptorCount = 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo     = &info;
    vkUpdateDescriptorSets(dev, 1, &write, 0, nullptr);
}

VkDescriptorSet allocSet(VkDevice dev, VkDescriptorPool pool, VkDescriptorSetLayout layout) {
    VkDescriptorSetAllocateInfo alloc{};
    alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool     = pool;
    alloc.descriptorSetCount = 1;
    alloc.pSetLayouts        = &layout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (vkAllocateDescriptorSets(dev, &alloc, &set) != VK_SUCCESS) return VK_NULL_HANDLE;
    return set;
}

void computeBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier b{};
    b.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &b, 0, nullptr, 0, nullptr);
}

void recordDotPass1(VkCommandBuffer cmd, VkDevice dev, VkDescriptorPool pool,
                    const PCGSolver& pcg, IslandBuffer& buf,
                    VkBuffer vecA, VkBuffer vecB, VkBuffer partials) {
    VkDescriptorSet set = allocSet(dev, pool, pcg.dotLayout());
    if (set == VK_NULL_HANDLE) return;
    writeStorage(dev, set, 0, vecA);
    writeStorage(dev, set, 1, vecB);
    writeStorage(dev, set, 2, partials);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pcg.dotPipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pcg.dotPipelineLayout(), 0, 1, &set, 0, nullptr);

    PCGDotPushConstants pc{};
    pc.N       = static_cast<std::uint32_t>(buf.N());
    pc.isPass2 = 0u;
    vkCmdPushConstants(cmd, pcg.dotPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, ceilDiv(buf.N(), kElPerWG), 1, 1);
    computeBarrier(cmd);
}

void recordDotPass2(VkCommandBuffer cmd, VkDevice dev, VkDescriptorPool pool,
                    const PCGSolver& pcg, IslandBuffer& buf,
                    VkBuffer partials, VkBuffer reduction,
                    std::uint32_t numPartials, std::uint32_t outputSlot) {
    VkDescriptorSet set = allocSet(dev, pool, pcg.dotLayout());
    if (set == VK_NULL_HANDLE) return;
    writeStorage(dev, set, 0, partials);
    writeStorage(dev, set, 1, partials); // unused in pass2
    writeStorage(dev, set, 2, reduction);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pcg.dotPipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pcg.dotPipelineLayout(), 0, 1, &set, 0, nullptr);

    PCGDotPushConstants pc{};
    pc.N       = numPartials;
    pc.isPass2 = 1u | (outputSlot << 16); // Encode slot in upper 16 bits for shader
    vkCmdPushConstants(cmd, pcg.dotPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    if (numPartials > kElPerWG) return;
    vkCmdDispatch(cmd, 1u, 1, 1);
    computeBarrier(cmd);
    (void) buf;
}

void recordMatvec(VkCommandBuffer cmd, VkDevice dev, VkDescriptorPool pool,
                   const PCGSolver& pcg, IslandBuffer& buf,
                   VkBuffer input, VkBuffer output) {
    VkDescriptorSet set = allocSet(dev, pool, pcg.matvecLayout());
    if (set == VK_NULL_HANDLE) return;
    writeStorage(dev, set, 0, input);
    writeStorage(dev, set, 1, output);
    writeStorage(dev, set, 2, buf.cond_buf);
    writeStorage(dev, set, 3, buf.type_buf);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pcg.matvecPipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pcg.matvecPipelineLayout(), 0, 1, &set, 0, nullptr);

    PCGMatvecPushConstants pc{};
    pc.Lx = static_cast<std::uint32_t>(buf.lx);
    pc.Ly = static_cast<std::uint32_t>(buf.ly);
    pc.Lz = static_cast<std::uint32_t>(buf.lz);
    vkCmdPushConstants(cmd, pcg.matvecPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, ceilDiv(buf.N(), kWGScan), 1, 1);
    computeBarrier(cmd);
}

void recordPrecompute(VkCommandBuffer cmd, VkDevice dev, VkDescriptorPool pool,
                      const PCGSolver& pcg, IslandBuffer& buf) {
    VkDescriptorSet set = allocSet(dev, pool, pcg.precomputeLayout());
    if (set == VK_NULL_HANDLE) return;
    writeStorage(dev, set, 0, buf.cond_buf);
    writeStorage(dev, set, 1, buf.pcg_inv_diag_buf);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pcg.precomputePipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pcg.precomputePipelineLayout(), 0, 1, &set, 0, nullptr);

    PCGPrecomputePushConstants pc{};
    pc.Lx = static_cast<std::uint32_t>(buf.lx);
    pc.Ly = static_cast<std::uint32_t>(buf.ly);
    pc.Lz = static_cast<std::uint32_t>(buf.lz);
    vkCmdPushConstants(cmd, pcg.precomputePipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, ceilDiv(buf.N(), kWGScan), 1, 1);
    computeBarrier(cmd);
}

void recordUpdate(VkCommandBuffer cmd, VkDevice dev, VkDescriptorPool pool,
                   const PCGSolver& pcg, IslandBuffer& buf, bool isInit,
                   float alphaInit) {
    VkDescriptorSet set = allocSet(dev, pool, pcg.updateLayout());
    if (set == VK_NULL_HANDLE) return;
    VkBuffer phi = buf.phi_flip ? buf.phi_buf_b : buf.phi_buf_a;
    writeStorage(dev, set, 0, phi);
    writeStorage(dev, set, 1, buf.pcg_r_buf);
    writeStorage(dev, set, 2, buf.pcg_p_buf);
    writeStorage(dev, set, 3, buf.pcg_ap_buf);
    writeStorage(dev, set, 4, buf.source_buf);
    writeStorage(dev, set, 5, buf.type_buf);
    writeStorage(dev, set, 6, buf.pcg_partial_buf);
    writeStorage(dev, set, 7, buf.pcg_reduction_buf);
    writeStorage(dev, set, 8, buf.cond_buf);
    writeStorage(dev, set, 9, buf.pcg_inv_diag_buf);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pcg.updatePipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pcg.updatePipelineLayout(), 0, 1, &set, 0, nullptr);

    PCGUpdatePushConstants pc{};
    pc.Lx      = static_cast<std::uint32_t>(buf.lx);
    pc.Ly      = static_cast<std::uint32_t>(buf.ly);
    pc.Lz      = static_cast<std::uint32_t>(buf.lz);
    pc.alpha   = alphaInit;
    pc.isInit  = isInit ? 1u : 0u;
    pc.padding = 0;
    vkCmdPushConstants(cmd, pcg.updatePipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, ceilDiv(buf.N(), kWGScan), 1, 1);
    computeBarrier(cmd);
}

void recordDirection(VkCommandBuffer cmd, VkDevice dev, VkDescriptorPool pool,
                      const PCGSolver& pcg, IslandBuffer& buf) {
    VkDescriptorSet set = allocSet(dev, pool, pcg.directionLayout());
    if (set == VK_NULL_HANDLE) return;
    writeStorage(dev, set, 0, buf.pcg_r_buf);
    writeStorage(dev, set, 1, buf.pcg_p_buf);
    writeStorage(dev, set, 2, buf.type_buf);
    writeStorage(dev, set, 3, buf.pcg_reduction_buf);
    writeStorage(dev, set, 4, buf.cond_buf);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pcg.directionPipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
        pcg.directionPipelineLayout(), 0, 1, &set, 0, nullptr);

    PCGDirectionPushConstants pc{};
    pc.Lx = static_cast<std::uint32_t>(buf.lx);
    pc.Ly = static_cast<std::uint32_t>(buf.ly);
    pc.Lz = static_cast<std::uint32_t>(buf.lz);
    vkCmdPushConstants(cmd, pcg.directionPipelineLayout(),
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vkCmdDispatch(cmd, ceilDiv(buf.N(), kWGScan), 1, 1);
    computeBarrier(cmd);
}

void recordReductionRotate(VkCommandBuffer cmd, IslandBuffer& buf) {
    if (buf.pcg_reduction_buf == VK_NULL_HANDLE) return;

    VkMemoryBarrier pre{};
    pre.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    pre.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pre.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &pre, 0, nullptr, 0, nullptr);

    VkBufferCopy region{};
    region.srcOffset = 2 * sizeof(float);
    region.dstOffset = 0;
    region.size      = sizeof(float);
    vkCmdCopyBuffer(cmd, buf.pcg_reduction_buf, buf.pcg_reduction_buf, 1, &region);

    VkMemoryBarrier post{};
    post.sType         = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    post.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    post.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &post, 0, nullptr, 0, nullptr);
}

} // namespace

void Dispatcher::recordPCGInitialResidual(VkCommandBuffer cmd, IslandBuffer& buf,
                                           VkDescriptorPool pool) {
    if (!pcg_.isReady() || !buf.hasPCGBuffers()) return;
    if (buf.cond_buf == VK_NULL_HANDLE || buf.type_buf == VK_NULL_HANDLE ||
        buf.source_buf == VK_NULL_HANDLE) return;

    VkDevice dev = vk_.device();
    if (dev == VK_NULL_HANDLE) return;

    VkBuffer phi = buf.phi_flip ? buf.phi_buf_b : buf.phi_buf_a;

    recordPrecompute(cmd, dev, pool, pcg_, buf);
    recordMatvec(cmd, dev, pool, pcg_, buf, phi, buf.pcg_ap_buf);
    recordUpdate(cmd, dev, pool, pcg_, buf, true, -1.0f);

    const std::uint32_t groups = ceilDiv(buf.N(), kElPerWG);
    recordDotPass2(cmd, dev, pool, pcg_, buf,
                    buf.pcg_partial_buf, buf.pcg_reduction_buf, groups, 0);
}

void Dispatcher::recordPCGStep(VkCommandBuffer cmd, IslandBuffer& buf,
                                VkDescriptorPool pool) {
    if (!pcg_.isReady() || !buf.hasPCGBuffers()) return;

    VkDevice dev = vk_.device();
    if (dev == VK_NULL_HANDLE) return;

    const std::uint32_t groups = ceilDiv(buf.N(), kElPerWG);

    recordMatvec(cmd, dev, pool, pcg_, buf, buf.pcg_p_buf, buf.pcg_ap_buf);

    recordDotPass1(cmd, dev, pool, pcg_, buf,
                    buf.pcg_p_buf, buf.pcg_ap_buf, buf.pcg_partial_buf);
    recordDotPass2(cmd, dev, pool, pcg_, buf,
                    buf.pcg_partial_buf, buf.pcg_reduction_buf, groups, 1);

    recordUpdate(cmd, dev, pool, pcg_, buf, false, 0.0f);

    recordDotPass2(cmd, dev, pool, pcg_, buf,
                    buf.pcg_partial_buf, buf.pcg_reduction_buf, groups, 2);

    recordDirection(cmd, dev, pool, pcg_, buf);
    recordReductionRotate(cmd, buf);
}

} // namespace pfsf
