/**
 * @file vulkan_context.cpp
 * @brief 5070TI 專用高效能 Vulkan 上下文 (含手動載入邏輯)。
 */
#include <vk_mem_alloc.h>
#include "vulkan_context.h"
#include <cstring>
#include <vector>
#include <cstdio>
#include <windows.h>

namespace pfsf {

VulkanContext::VulkanContext() = default;
VulkanContext::~VulkanContext() { shutdown(); }

// ── 內部手動載入器 ──
static void* manual_vulkan_handle = nullptr;
static bool force_load_vulkan() {
    if (manual_vulkan_handle) return true;
    
    // 優先嘗試 System32
    const char* path = "C:\\Windows\\System32\\vulkan-1.dll";
    manual_vulkan_handle = (void*)LoadLibraryA(path);

    if (!manual_vulkan_handle) {
        manual_vulkan_handle = (void*)LoadLibraryA("vulkan-1.dll");
    }

    if (manual_vulkan_handle) {
        fprintf(stderr, "[libpfsf] Vulkan Loader successfully linked.\n");
        return true;
    } else {
        fprintf(stderr, "[libpfsf] FATAL: Vulkan Loader not found! (Error: %lu)\n", GetLastError());
        return false;
    }
}

bool VulkanContext::init() {
    if (available_) return true;

    fprintf(stderr, "[libpfsf] --- VULKAN INIT ---\n");

    if (!force_load_vulkan()) return false;

    // ── Instance: try 1.3 → 1.2 → 1.1.
    // On 1.1 we *try* to enable VK_KHR_get_physical_device_properties2 so
    // vkGetPhysicalDeviceFeatures2 works as a promoted-from-KHR call, but
    // only after verifying the driver actually exposes the extension.
    // VulkanMod (Minecraft's reference Vulkan renderer) simply stays on
    // 1.1 + vkGetPhysicalDeviceFeatures and skips features2 entirely; we
    // fall back to that same conservative path when the extension is
    // missing rather than failing vkCreateInstance.
    bool have_properties2_ext = false;
    {
        uint32_t extCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
        std::vector<VkExtensionProperties> exts(extCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extCount, exts.data());
        for (auto& e : exts) {
            if (strcmp(e.extensionName, "VK_KHR_get_physical_device_properties2") == 0) {
                have_properties2_ext = true; break;
            }
        }
    }

    const char* kProperties2Ext = "VK_KHR_get_physical_device_properties2";
    uint32_t versions[] = { VK_API_VERSION_1_3, VK_API_VERSION_1_2, VK_API_VERSION_1_1 };
    VkResult last_res = VK_SUCCESS;
    uint32_t instanceApiVersion = 0;

    for (uint32_t ver : versions) {
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "PFSF_Core";
        appInfo.apiVersion = ver;

        VkInstanceCreateInfo instCI = {};
        instCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instCI.pApplicationInfo = &appInfo;
        if (ver == VK_API_VERSION_1_1 && have_properties2_ext) {
            // Promote features2 access on 1.1 driver. On 1.2+ it's core.
            instCI.enabledExtensionCount = 1;
            instCI.ppEnabledExtensionNames = &kProperties2Ext;
        }

        last_res = vkCreateInstance(&instCI, nullptr, &instance_);
        if (last_res == VK_SUCCESS) {
            instanceApiVersion = ver;
            fprintf(stderr, "[libpfsf] Instance created with API %d.%d (properties2=%s)\n",
                    VK_VERSION_MAJOR(ver), VK_VERSION_MINOR(ver),
                    (ver >= VK_API_VERSION_1_2 || have_properties2_ext) ? "yes" : "no");
            break;
        }
    }

    if (!instance_) {
        fprintf(stderr, "[libpfsf] vkCreateInstance FAILED: %d\n", (int)last_res);
        return false;
    }

    if (!selectPhysicalDevice()) {
        fprintf(stderr, "[libpfsf] No suitable Vulkan physical device found.\n");
        return false;
    }

    int qf = findComputeQueueFamily();
    if (qf < 0) return false;
    computeQueueFamily_ = (uint32_t)qf;

    // features2 is only safe when instance API >= 1.2 OR properties2 ext.
    // When neither, stay on 1.0-style vkGetPhysicalDeviceFeatures path.
    const bool features2_available =
        (instanceApiVersion >= VK_API_VERSION_1_2) || have_properties2_ext;
    const bool canUse12Chain = (instanceApiVersion >= VK_API_VERSION_1_2);

    VkPhysicalDeviceFeatures wanted10{};
    VkPhysicalDeviceFeatures2 wanted2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    VkPhysicalDeviceVulkan12Features wanted12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };

    if (features2_available) {
        VkPhysicalDeviceFeatures2 feats2 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
        VkPhysicalDeviceVulkan12Features feats12 = { VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
        if (canUse12Chain) feats2.pNext = &feats12;
        vkGetPhysicalDeviceFeatures2(physDevice_, &feats2);

        wanted2.features.shaderInt64   = feats2.features.shaderInt64;
        wanted2.features.shaderFloat64 = feats2.features.shaderFloat64;
        if (canUse12Chain) {
            wanted12.timelineSemaphore       = feats12.timelineSemaphore;
            wanted12.storageBuffer8BitAccess = feats12.storageBuffer8BitAccess;
            wanted12.shaderInt8              = feats12.shaderInt8;
            wanted2.pNext = &wanted12;
        }
    } else {
        // 1.1 without properties2 ext: conservative 1.0 feature query.
        VkPhysicalDeviceFeatures have{};
        vkGetPhysicalDeviceFeatures(physDevice_, &have);
        wanted10.shaderInt64   = have.shaderInt64;
        wanted10.shaderFloat64 = have.shaderFloat64;
    }
    timelineEnabled_ = (canUse12Chain && wanted12.timelineSemaphore == VK_TRUE);

    float qp = 1.0f;
    VkDeviceQueueCreateInfo qCI = {};
    qCI.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qCI.queueFamilyIndex = computeQueueFamily_;
    qCI.queueCount = 1;
    qCI.pQueuePriorities = &qp;

    VkDeviceCreateInfo devCI = {};
    devCI.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    devCI.queueCreateInfoCount = 1;
    devCI.pQueueCreateInfos = &qCI;
    if (features2_available) {
        devCI.pNext = &wanted2;  // features2 chain — don't set pEnabledFeatures here
    } else {
        devCI.pEnabledFeatures = &wanted10;
    }

    if (vkCreateDevice(physDevice_, &devCI, nullptr, &device_) != VK_SUCCESS) {
        fprintf(stderr, "[libpfsf] vkCreateDevice FAILED\n");
        return false;
    }

    vkGetDeviceQueue(device_, computeQueueFamily_, 0, &computeQueue_);

    if (features2_available) {
        fprintf(stderr, "[libpfsf] Device features enabled: int64=%d float64=%d timelineSem=%d int8=%d storage8bit=%d\n",
                wanted2.features.shaderInt64, wanted2.features.shaderFloat64,
                wanted12.timelineSemaphore, wanted12.shaderInt8, wanted12.storageBuffer8BitAccess);
    } else {
        fprintf(stderr, "[libpfsf] Device features enabled (1.0 path): int64=%d float64=%d\n",
                wanted10.shaderInt64, wanted10.shaderFloat64);
    }

    VkCommandPoolCreateInfo cpCI = {};
    cpCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpCI.queueFamilyIndex = computeQueueFamily_;
    cpCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device_, &cpCI, nullptr, &cmdPool_) != VK_SUCCESS) {
        fprintf(stderr, "[libpfsf] vkCreateCommandPool FAILED\n");
        return false;
    }

    VmaAllocatorCreateInfo vmaCI = {};
    vmaCI.vulkanApiVersion = instanceApiVersion >= VK_API_VERSION_1_2 ? VK_API_VERSION_1_2 : VK_API_VERSION_1_1;
    vmaCI.physicalDevice = physDevice_;
    vmaCI.device = device_;
    vmaCI.instance = instance_;
    if (vmaCreateAllocator(&vmaCI, &allocator_) != VK_SUCCESS) {
        fprintf(stderr, "[libpfsf] VMA FAILED\n");
        return false;
    }

    if (timelineEnabled_) {
        tryCreateTimelineSemaphore();
    }

    available_ = true;
    fprintf(stderr, "[libpfsf] Vulkan ready on device: %s\n", deviceName_.c_str());
    return true;
}

bool VulkanContext::initFromExisting(VkInstance inst,
                                     VkPhysicalDevice phys,
                                     VkDevice dev,
                                     uint32_t queueFamily,
                                     VkQueue computeQueue) {
    if (available_) return true;
    if (inst == VK_NULL_HANDLE || phys == VK_NULL_HANDLE ||
        dev  == VK_NULL_HANDLE || computeQueue == VK_NULL_HANDLE) {
        fprintf(stderr, "[libpfsf] initFromExisting: null handle rejected\n");
        return false;
    }

    instance_           = inst;
    physDevice_         = phys;
    device_             = dev;
    computeQueueFamily_ = queueFamily;
    computeQueue_       = computeQueue;
    ownsHandles_        = false;

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(phys, &props);
    deviceName_ = props.deviceName;

    VkCommandPoolCreateInfo cpCI = {};
    cpCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpCI.queueFamilyIndex = queueFamily;
    cpCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device_, &cpCI, nullptr, &cmdPool_) != VK_SUCCESS) {
        fprintf(stderr, "[libpfsf] initFromExisting: vkCreateCommandPool FAILED\n");
        return false;
    }

    VmaAllocatorCreateInfo vmaCI = {};
    vmaCI.vulkanApiVersion = VK_API_VERSION_1_2;
    vmaCI.physicalDevice = physDevice_;
    vmaCI.device = device_;
    vmaCI.instance = instance_;
    if (vmaCreateAllocator(&vmaCI, &allocator_) != VK_SUCCESS) {
        fprintf(stderr, "[libpfsf] initFromExisting: VMA FAILED\n");
        vkDestroyCommandPool(device_, cmdPool_, nullptr);
        cmdPool_ = VK_NULL_HANDLE;
        return false;
    }

    tryCreateTimelineSemaphore();

    available_ = true;
    fprintf(stderr, "[libpfsf] Vulkan adopted from Java host on device: %s\n", deviceName_.c_str());
    return true;
}

void VulkanContext::shutdown() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        if (timelineSem_ != VK_NULL_HANDLE) {
            vkDestroySemaphore(device_, timelineSem_, nullptr);
            timelineSem_ = VK_NULL_HANDLE;
        }
        if (allocator_) vmaDestroyAllocator(allocator_);
        if (cmdPool_) vkDestroyCommandPool(device_, cmdPool_, nullptr);
        if (ownsHandles_) vkDestroyDevice(device_, nullptr);
    }
    if (instance_ && ownsHandles_) vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE; device_ = VK_NULL_HANDLE; cmdPool_ = VK_NULL_HANDLE;
    allocator_ = nullptr;
    timelineValue_ = 0;
    timelineEnabled_ = false;
    available_ = false;
    ownsHandles_ = true;
}

bool VulkanContext::selectPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) return false;
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());

    VkPhysicalDevice best = VK_NULL_HANDLE;
    int bestScore = -1;
    std::string bestName;

    for (auto& pd : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);

        // Compute queue family must exist for this device to be usable at all.
        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qfs(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qfs.data());
        bool hasCompute = false;
        for (uint32_t i = 0; i < qfCount; i++) {
            if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { hasCompute = true; break; }
        }
        if (!hasCompute) continue;

        VkPhysicalDeviceFeatures feats;
        vkGetPhysicalDeviceFeatures(pd, &feats);

        int score = 0;
        switch (props.deviceType) {
            case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   score += 1000; break;
            case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: score += 100;  break;
            case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    score += 50;   break;
            default: break;
        }
        if (feats.shaderInt64)   score += 50;
        if (feats.shaderFloat64) score += 25;
        if (props.apiVersion >= VK_API_VERSION_1_2) score += 30;

        if (score > bestScore) {
            bestScore = score;
            best = pd;
            bestName = props.deviceName;
        }
    }

    if (best == VK_NULL_HANDLE) {
        fprintf(stderr, "[libpfsf] No Vulkan device with compute queue found among %u candidate(s).\n", count);
        return false;
    }
    physDevice_ = best;
    deviceName_ = bestName;
    fprintf(stderr, "[libpfsf] Selected device: %s (score=%d)\n", deviceName_.c_str(), bestScore);
    return true;
}

int VulkanContext::findComputeQueueFamily() {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice_, &count, nullptr);
    std::vector<VkQueueFamilyProperties> props(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physDevice_, &count, props.data());
    for (uint32_t i = 0; i < count; i++) {
        if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return (int)i;
    }
    return -1;
}

bool VulkanContext::allocBuffer(VkDeviceSize s, VkBufferUsageFlags u, VkBuffer* b, VkDeviceMemory* m) {
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size = s; ci.usage = u;
    VmaAllocationCreateInfo vi = {}; vi.usage = VMA_MEMORY_USAGE_AUTO;
    if (u & VK_BUFFER_USAGE_TRANSFER_SRC_BIT) vi.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocation a; 
    if (vmaCreateBuffer(allocator_, &ci, &vi, b, &a, nullptr) != VK_SUCCESS) return false;
    allocationMap_[*b] = a;
    if (m) *m = VK_NULL_HANDLE;
    return true;
}

void VulkanContext::freeBuffer(VkBuffer b, VkDeviceMemory) {
    auto it = allocationMap_.find(b);
    if (it != allocationMap_.end()) { vmaDestroyBuffer(allocator_, b, it->second); allocationMap_.erase(it); }
}

void* VulkanContext::mapBuffer(VkBuffer b, VkDeviceSize) {
    auto it = allocationMap_.find(b);
    if (it == allocationMap_.end()) return nullptr;
    void* data; vmaMapMemory(allocator_, it->second, &data); return data;
}

void VulkanContext::unmapBuffer(VkBuffer b) {
    auto it = allocationMap_.find(b);
    if (it != allocationMap_.end()) vmaUnmapMemory(allocator_, it->second);
}

VkCommandBuffer VulkanContext::allocCmdBuffer() {
    VkCommandBufferAllocateInfo ai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cmdPool_; ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ai.commandBufferCount = 1;
    VkCommandBuffer cb; vkAllocateCommandBuffers(device_, &ai, &cb);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi); return cb;
}

void VulkanContext::tryCreateTimelineSemaphore() {
    if (timelineSem_ != VK_NULL_HANDLE) return;
    // In the owned-init path we only call this when timelineEnabled_ was
    // set true during feature negotiation. In initFromExisting we don't
    // know whether the Java host enabled the feature; we still attempt
    // and fall back on VK_ERROR_FEATURE_NOT_PRESENT rather than guessing.
    VkSemaphoreTypeCreateInfo stci = {};
    stci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    stci.initialValue = 0;
    VkSemaphoreCreateInfo sci = {};
    sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    sci.pNext = &stci;
    VkSemaphore sem = VK_NULL_HANDLE;
    VkResult r = vkCreateSemaphore(device_, &sci, nullptr, &sem);
    if (r == VK_SUCCESS) {
        timelineSem_ = sem;
        timelineValue_ = 0;
        fprintf(stderr, "[libpfsf] Timeline semaphore enabled for async submits.\n");
    } else {
        // Feature unavailable (or validation layer complaint) — stay on
        // vkQueueWaitIdle path in submitAndWait. Not fatal.
        timelineSem_ = VK_NULL_HANDLE;
        fprintf(stderr, "[libpfsf] Timeline semaphore unavailable (vkResult=%d); using vkQueueWaitIdle fallback.\n",
                (int)r);
    }
}

VkResult VulkanContext::submitAndWait(VkCommandBuffer cb) {
    vkEndCommandBuffer(cb);

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;

    VkResult res;
    if (timelineSem_ != VK_NULL_HANDLE) {
        // Timeline path: wait only on THIS submit, not the entire queue.
        uint64_t signalValue = ++timelineValue_;
        VkTimelineSemaphoreSubmitInfo tssi = {};
        tssi.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
        tssi.signalSemaphoreValueCount = 1;
        tssi.pSignalSemaphoreValues = &signalValue;

        si.pNext = &tssi;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &timelineSem_;

        if (vkQueueSubmit(computeQueue_, 1, &si, VK_NULL_HANDLE) != VK_SUCCESS) {
            vkFreeCommandBuffers(device_, cmdPool_, 1, &cb);
            return VK_ERROR_DEVICE_LOST;
        }

        VkSemaphoreWaitInfo wi = {};
        wi.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wi.semaphoreCount = 1;
        wi.pSemaphores = &timelineSem_;
        wi.pValues = &signalValue;
        // 5-second timeout is generous for a compute dispatch; treat
        // timeout as DEVICE_LOST so callers can reset rather than hang.
        res = vkWaitSemaphores(device_, &wi, 5'000'000'000ULL);
    } else {
        vkQueueSubmit(computeQueue_, 1, &si, VK_NULL_HANDLE);
        res = vkQueueWaitIdle(computeQueue_);
    }

    vkFreeCommandBuffers(device_, cmdPool_, 1, &cb);
    return res;
}

VkShaderModule VulkanContext::createShaderModule(const uint32_t* p, size_t s) {
    VkShaderModuleCreateInfo ci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    ci.codeSize = s; ci.pCode = p;
    VkShaderModule m; return (vkCreateShaderModule(device_, &ci, nullptr, &m) == VK_SUCCESS) ? m : VK_NULL_HANDLE;
}

VkDescriptorPool VulkanContext::createDescriptorPool(uint32_t ms, uint32_t md) {
    VkDescriptorPoolSize ps = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, md};
    VkDescriptorPoolCreateInfo ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.maxSets = ms; ci.poolSizeCount = 1; ci.pPoolSizes = &ps;
    VkDescriptorPool p; return (vkCreateDescriptorPool(device_, &ci, nullptr, &p) == VK_SUCCESS) ? p : VK_NULL_HANDLE;
}

VkDescriptorSet VulkanContext::allocDescriptorSet(VkDescriptorPool p, VkDescriptorSetLayout l) {
    VkDescriptorSetAllocateInfo ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = p; ai.descriptorSetCount = 1; ai.pSetLayouts = &l;
    VkDescriptorSet s; return (vkAllocateDescriptorSets(device_, &ai, &s) == VK_SUCCESS) ? s : VK_NULL_HANDLE;
}

bool VulkanContext::allocHostVisibleStorage(VkDeviceSize s, VkBuffer* b, void** p) {
    VkBufferCreateInfo ci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; ci.size = s;
    ci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo vi = {}; vi.usage = VMA_MEMORY_USAGE_AUTO;
    vi.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocation a; VmaAllocationInfo info;
    if (vmaCreateBuffer(allocator_, &ci, &vi, b, &a, &info) != VK_SUCCESS) return false;
    allocationMap_[*b] = a; if (p) *p = info.pMappedData; return true;
}

} // namespace pfsf
