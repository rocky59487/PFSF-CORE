/**
 * @file vulkan_context.h
 * @brief Internal Vulkan compute context — device, queue, command pool.
 */
#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <unordered_map>

// ── VMA 類型與函數手動聲明 ──
typedef struct VmaAllocator_T* VmaAllocator;
typedef struct VmaAllocation_T* VmaAllocation;

namespace pfsf {

class VulkanContext {
public:
    VulkanContext();
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    bool init();
    /// Adopt Vulkan handles owned by another module (typically the Java
    /// host). This context will NOT destroy them on shutdown; only the
    /// VMA allocator + cmdPool it creates here are torn down.
    bool initFromExisting(VkInstance inst,
                          VkPhysicalDevice phys,
                          VkDevice dev,
                          uint32_t queueFamily,
                          VkQueue computeQueue);
    void shutdown();

    bool isAvailable() const { return available_; }
    const std::string& deviceName() const { return deviceName_; }

    VkInstance       instance()     const { return instance_; }
    VkPhysicalDevice physDevice()   const { return physDevice_; }
    VkDevice         device()       const { return device_; }
    VkQueue          computeQueue() const { return computeQueue_; }
    uint32_t         queueFamily()  const { return computeQueueFamily_; }
    VkCommandPool    cmdPool()      const { return cmdPool_; }

    bool allocBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                     VkBuffer* outBuffer, VkDeviceMemory* outMemory);
    bool allocHostVisibleStorage(VkDeviceSize size, VkBuffer* outBuffer, void** outMappedPtr);
    void freeBuffer(VkBuffer buffer, VkDeviceMemory memory);
    void* mapBuffer(VkBuffer buffer, VkDeviceSize size);
    void unmapBuffer(VkBuffer buffer);

    VkCommandBuffer allocCmdBuffer();
    VkResult submitAndWait(VkCommandBuffer cmdBuf);

    VkShaderModule createShaderModule(const uint32_t* spirv, size_t sizeBytes);
    VkDescriptorPool createDescriptorPool(uint32_t maxSets, uint32_t maxDescriptors);
    VkDescriptorSet allocDescriptorSet(VkDescriptorPool pool, VkDescriptorSetLayout layout);

    // 新增：獲取內部 VMA 分配器（用於 standalone 測試）
    VmaAllocator allocator() const { return allocator_; }

private:
    bool selectPhysicalDevice();
    int  findComputeQueueFamily();
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props);

    bool        available_ = false;
    std::string deviceName_ = "unknown";

    VkInstance       instance_       = VK_NULL_HANDLE;
    VkPhysicalDevice physDevice_     = VK_NULL_HANDLE;
    VkDevice         device_         = VK_NULL_HANDLE;
    VkQueue          computeQueue_   = VK_NULL_HANDLE;
    uint32_t         computeQueueFamily_ = UINT32_MAX;
    VkCommandPool    cmdPool_        = VK_NULL_HANDLE;

    VmaAllocator     allocator_      = nullptr;
    std::unordered_map<VkBuffer, VmaAllocation> allocationMap_;

    /// True when this context created the instance/device and should
    /// destroy them on shutdown. False when adopted via initFromExisting.
    bool             ownsHandles_    = true;

    /// Timeline semaphore used to serialise submits without stalling
    /// unrelated work queued on the same VkQueue. VK_NULL_HANDLE when
    /// unavailable; submitAndWait falls back to vkQueueWaitIdle.
    VkSemaphore      timelineSem_    = VK_NULL_HANDLE;
    uint64_t         timelineValue_  = 0;
    /// True when we know the device has timelineSemaphore enabled.
    /// In the owned-init path this is set after feature negotiation.
    /// In initFromExisting we don't know for sure (Java host may or may
    /// not have enabled it), so we optimistically try and fall back
    /// gracefully when vkCreateSemaphore returns VK_ERROR_FEATURE_NOT_PRESENT.
    bool             timelineEnabled_ = false;

    void tryCreateTimelineSemaphore();
};

} // namespace pfsf
