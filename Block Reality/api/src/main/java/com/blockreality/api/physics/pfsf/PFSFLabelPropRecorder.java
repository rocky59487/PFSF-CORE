package com.blockreality.api.physics.pfsf;

import org.lwjgl.system.MemoryStack;
import org.lwjgl.vulkan.VkCommandBuffer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.ByteBuffer;

import static com.blockreality.api.physics.pfsf.PFSFPipelineFactory.*;
import static com.blockreality.api.physics.pfsf.PFSFVCycleRecorder.ceilDiv;
import static org.lwjgl.vulkan.VK10.*;

/**
 * Records the four-pass Shiloach–Vishkin label-propagation pipeline
 * into a Vulkan command buffer:
 *
 * <pre>
 *   init
 *     └─ for k in 0..SV_ITERATIONS-1:
 *          iterate pass=0 (hook-to-min)
 *          computeBarrier
 *          iterate pass=1 (pointer jump)
 *          computeBarrier
 *     └─ clear numComponents + rootToSlot
 *     └─ summarise_alloc
 *     └─ computeBarrier
 *     └─ summarise_aggregate
 *     └─ computeBarrier
 * </pre>
 *
 * <p>The (iterate, barrier, iterate, barrier) pairs all live inside
 * ONE submit. No changedFlag readback per iteration, no per-pass
 * fence wait — just 2·SV_ITERATIONS + 3 dispatches and 2·SV_ITERATIONS
 * + 2 barriers on a single command buffer, eliminating PCIe round-trip
 * cost that would otherwise dominate.
 *
 * <p>Post-fence (after the outer
 * {@link com.blockreality.api.physics.pfsf.PFSFAsyncCompute} fence
 * signals) the host reads {@code labelNumCompBuf} (8 bytes) and
 * {@code labelComponentsBuf} (48 × MAX_COMPONENTS bytes) back to CPU
 * and decodes into {@link PFSFLabelPropCpuSimulator.ComponentMeta}
 * records. See {@link #decodeComponents} for the decode logic.
 *
 * <p>This recorder runs only when
 * {@link PFSFIslandBuffer#isLabelPropEnabled()} is true AND the
 * island has successfully allocated its label-prop buffers. When
 * disabled, the Phase A CPU path in
 * {@code StructureIslandRegistry.checkAndSplitIsland} is the sole
 * source of orphan detection, which is already correct and tested.
 */
public final class PFSFLabelPropRecorder {

    private static final Logger LOGGER = LoggerFactory.getLogger("PFSF-LabelProp");

    /** Matches the GLSL / simulator constant. */
    public static final int SV_ITERATIONS = PFSFLabelPropCpuSimulator.DEFAULT_SV_ITERATIONS;

    private PFSFLabelPropRecorder() {}

    /**
     * Record init → iterate×SV_ITERATIONS → summarise_alloc → summarise_aggregate
     * into {@code cmdBuf}. Safe to call from the dispatcher's normal submission
     * chain; obeys the same descriptor-pool-exhaustion fail-safes as the
     * other recorders (logs and returns).
     */
    public static void recordLabelProp(VkCommandBuffer cmdBuf,
                                       PFSFIslandBuffer buf,
                                       long descriptorPool) {
        if (!buf.isLabelPropAllocated()) {
            // Feature off or allocation failed — the Phase A CPU path
            // still runs and keeps correctness. No work to do here.
            return;
        }

        int N = buf.getN();
        int Lx = buf.getLx(), Ly = buf.getLy(), Lz = buf.getLz();
        int maxComponents = PFSFLabelPropCpuSimulator.MAX_COMPONENTS;

        // Step 0: zero the numComponents (count + overflow) and rootToSlot buffers.
        vkCmdFillBuffer(cmdBuf, buf.getLabelNumCompBuf(), 0, buf.getLabelNumCompSize(), 0);
        vkCmdFillBuffer(cmdBuf, buf.getRootToSlotBuf(),   0, buf.getRootToSlotSize(),   0xFFFFFFFF);
        // Zero the aggregate region so the alloc pass's UINT_MAX-seeded
        // aabbMin values aren't poisoned by prior tick residue.
        vkCmdFillBuffer(cmdBuf, buf.getLabelComponentsBuf(), 0, buf.getLabelComponentsSize(), 0);
        VulkanComputeContext.computeBarrier(cmdBuf);

        // Step 1: init (per-voxel label = i+1 for live, NO_ISLAND for AIR).
        try (MemoryStack stack = MemoryStack.stackPush()) {
            vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, labelPropInitPipeline);
            long ds = VulkanComputeContext.allocateDescriptorSet(descriptorPool, labelPropInitDSLayout);
            if (ds == 0) { LOGGER.error("[PFSF] Label-prop init: descriptor set alloc failed"); return; }
            VulkanComputeContext.bindBufferToDescriptor(ds, 0, buf.getTypeBuf(),   buf.getTypeOffset(),  buf.getTypeSize());
            VulkanComputeContext.bindBufferToDescriptor(ds, 1, buf.getLabelIdBuf(), 0, buf.getLabelIdSize());
            vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, labelPropInitPipelineLayout, 0, stack.longs(ds), null);

            ByteBuffer pc = stack.malloc(12);
            pc.putInt(Lx).putInt(Ly).putInt(Lz);
            pc.flip();
            vkCmdPushConstants(cmdBuf, labelPropInitPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);
            vkCmdDispatch(cmdBuf, ceilDiv(N, 256), 1, 1);
        }
        VulkanComputeContext.computeBarrier(cmdBuf);

        // Step 2: fixed unrolled SV loop — (hook, barrier, jump, barrier) × SV_ITERATIONS.
        // One descriptor set is reused across all 2·SV_ITERATIONS dispatches
        // because the iterate kernel reads/writes the same two buffers every
        // pass (type + islandId); only the push-constant `pass` switches.
        // This avoids exhausting the descriptor pool in large islands.
        try (MemoryStack stack = MemoryStack.stackPush()) {
            long iterDs = VulkanComputeContext.allocateDescriptorSet(descriptorPool, labelPropIterateDSLayout);
            if (iterDs == 0) {
                LOGGER.error("[PFSF] Label-prop iterate: descriptor set alloc failed");
                return;
            }
            VulkanComputeContext.bindBufferToDescriptor(iterDs, 0, buf.getTypeBuf(),    buf.getTypeOffset(), buf.getTypeSize());
            VulkanComputeContext.bindBufferToDescriptor(iterDs, 1, buf.getLabelIdBuf(), 0, buf.getLabelIdSize());
            vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, labelPropIteratePipeline);
            vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, labelPropIteratePipelineLayout, 0, stack.longs(iterDs), null);

            for (int k = 0; k < SV_ITERATIONS; k++) {
                // pass = 0: hook-to-min
                ByteBuffer pc0 = stack.malloc(16);
                pc0.putInt(Lx).putInt(Ly).putInt(Lz).putInt(0);
                pc0.flip();
                vkCmdPushConstants(cmdBuf, labelPropIteratePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc0);
                vkCmdDispatch(cmdBuf, ceilDiv(N, 256), 1, 1);
                VulkanComputeContext.computeBarrier(cmdBuf);

                // pass = 1: pointer jump
                ByteBuffer pc1 = stack.malloc(16);
                pc1.putInt(Lx).putInt(Ly).putInt(Lz).putInt(1);
                pc1.flip();
                vkCmdPushConstants(cmdBuf, labelPropIteratePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc1);
                vkCmdDispatch(cmdBuf, ceilDiv(N, 256), 1, 1);
                VulkanComputeContext.computeBarrier(cmdBuf);
            }
        }

        // Step 3: summarise_alloc — voxel-is-root → dense slot.
        try (MemoryStack stack = MemoryStack.stackPush()) {
            vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, labelPropAllocPipeline);
            long ds = VulkanComputeContext.allocateDescriptorSet(descriptorPool, labelPropAllocDSLayout);
            if (ds == 0) { LOGGER.error("[PFSF] Label-prop alloc: descriptor set alloc failed"); return; }
            VulkanComputeContext.bindBufferToDescriptor(ds, 0, buf.getLabelIdBuf(),        0, buf.getLabelIdSize());
            VulkanComputeContext.bindBufferToDescriptor(ds, 1, buf.getLabelNumCompBuf(),   0, buf.getLabelNumCompSize());
            VulkanComputeContext.bindBufferToDescriptor(ds, 2, buf.getRootToSlotBuf(),     0, buf.getRootToSlotSize());
            VulkanComputeContext.bindBufferToDescriptor(ds, 3, buf.getLabelComponentsBuf(), 0, buf.getLabelComponentsSize());
            vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, labelPropAllocPipelineLayout, 0, stack.longs(ds), null);

            ByteBuffer pc = stack.malloc(16);
            pc.putInt(Lx).putInt(Ly).putInt(Lz).putInt(maxComponents);
            pc.flip();
            vkCmdPushConstants(cmdBuf, labelPropAllocPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);
            vkCmdDispatch(cmdBuf, ceilDiv(N, 256), 1, 1);
        }
        VulkanComputeContext.computeBarrier(cmdBuf);

        // Step 4: summarise_aggregate — fold blockCount / anchored / AABB per slot.
        try (MemoryStack stack = MemoryStack.stackPush()) {
            vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, labelPropAggregatePipeline);
            long ds = VulkanComputeContext.allocateDescriptorSet(descriptorPool, labelPropAggregateDSLayout);
            if (ds == 0) { LOGGER.error("[PFSF] Label-prop aggregate: descriptor set alloc failed"); return; }
            VulkanComputeContext.bindBufferToDescriptor(ds, 0, buf.getTypeBuf(),    buf.getTypeOffset(), buf.getTypeSize());
            VulkanComputeContext.bindBufferToDescriptor(ds, 1, buf.getLabelIdBuf(), 0, buf.getLabelIdSize());
            VulkanComputeContext.bindBufferToDescriptor(ds, 2, buf.getRootToSlotBuf(), 0, buf.getRootToSlotSize());
            VulkanComputeContext.bindBufferToDescriptor(ds, 3, buf.getLabelComponentsBuf(), 0, buf.getLabelComponentsSize());
            vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, labelPropAggregatePipelineLayout, 0, stack.longs(ds), null);

            ByteBuffer pc = stack.malloc(16);
            pc.putInt(Lx).putInt(Ly).putInt(Lz).putInt(maxComponents);
            pc.flip();
            vkCmdPushConstants(cmdBuf, labelPropAggregatePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pc);
            vkCmdDispatch(cmdBuf, ceilDiv(N, 256), 1, 1);
        }
        VulkanComputeContext.computeBarrier(cmdBuf);
    }

    // ═════════════════════════════════════════════════════════════════
    //  Host-side readback decode — post-fence helper.
    // ═════════════════════════════════════════════════════════════════

    /**
     * Decode the raw bytes of {@code labelNumCompBuf} (8 bytes) and
     * {@code labelComponentsBuf} (48 × MAX_COMPONENTS bytes) into
     * {@link PFSFLabelPropCpuSimulator.ComponentMeta} records. Used by
     * the dispatcher's post-fence callback.
     *
     * @param numCompBytes     buffer with {@code [numComponents, overflowFlag]} (8 bytes, little-endian)
     * @param componentsBytes  per-slot 48-byte records (12 uint32 each)
     * @return {@link DecodedComponents} with count, overflow flag, and the array of slots actually populated
     */
    public static DecodedComponents decodeComponents(ByteBuffer numCompBytes, ByteBuffer componentsBytes) {
        numCompBytes.order(java.nio.ByteOrder.LITTLE_ENDIAN);
        componentsBytes.order(java.nio.ByteOrder.LITTLE_ENDIAN);
        int numComponents = numCompBytes.getInt(0);
        int overflowFlag  = numCompBytes.getInt(4);
        int cap = Math.min(numComponents, PFSFLabelPropCpuSimulator.MAX_COMPONENTS);
        PFSFLabelPropCpuSimulator.ComponentMeta[] metas =
                new PFSFLabelPropCpuSimulator.ComponentMeta[cap];
        for (int s = 0; s < cap; s++) {
            int base = s * 12 * Integer.BYTES;
            int rootLabel  = componentsBytes.getInt(base);
            int blockCount = componentsBytes.getInt(base + 4);
            int anchored   = componentsBytes.getInt(base + 8);
            // offset 12 = pad
            int minX = componentsBytes.getInt(base + 16);
            int minY = componentsBytes.getInt(base + 20);
            int minZ = componentsBytes.getInt(base + 24);
            // offset 28 = pad
            int maxX = componentsBytes.getInt(base + 32);
            int maxY = componentsBytes.getInt(base + 36);
            int maxZ = componentsBytes.getInt(base + 40);
            metas[s] = new PFSFLabelPropCpuSimulator.ComponentMeta(
                    rootLabel, blockCount, (anchored & 1) != 0,
                    minX, minY, minZ, maxX, maxY, maxZ);
        }
        return new DecodedComponents(numComponents, (overflowFlag & 1) != 0, metas);
    }

    /** Post-readback decoded result. */
    public record DecodedComponents(int numComponents, boolean overflow,
                                    PFSFLabelPropCpuSimulator.ComponentMeta[] components) {}
}
