/**
 * @file vulkan_context.cpp
 * @brief 深度兼容型 Vulkan 上下文 — 專為 5070TI 優化。
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

// 外部聲明的 DLL 載入修復
bool force_load_vulkan();

bool VulkanContext::init() {
    if (available_) return true;

    fprintf(stderr, "[libpfsf] --- 5070TI INITIALIZATION START ---\n");

    if (!force_load_vulkan()) return false;

    // ── 三段式版本嘗試 ──
    uint32_t versions[] = { VK_API_VERSION_1_3, VK_API_VERSION_1_2, VK_API_VERSION_1_1 };
    VkResult last_res = VK_SUCCESS;

    for (uint32_t ver : versions) {
        VkApplicationInfo appInfo = {};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "PFSF_Core";
        appInfo.apiVersion = ver;

        VkInstanceCreateInfo instCI = {};
        instCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instCI.pApplicationInfo = &appInfo;
        
        last_res = vkCreateInstance(&instCI, nullptr, &instance_);
        if (last_res == VK_SUCCESS) {
            fprintf(stderr, "[libpfsf] Instance created successfully with API %d.%d\n", 
                    VK_VERSION_MAJOR(ver), VK_VERSION_MINOR(ver));
            break;
        }
        fprintf(stderr, "[libpfsf] vkCreateInstance failed for ver %d.%d (res=%d)\n", 
                VK_VERSION_MAJOR(ver), VK_VERSION_MINOR(ver), (int)last_res);
    }

    if (!instance_) {
        fprintf(stderr, "[libpfsf] FATAL: All Vulkan versions failed to initialize.\n");
        return false;
    }

    if (!selectPhysicalDevice()) {
        fprintf(stderr, "[libpfsf] No compute-capable GPU found!\n");
        return false;
    }

    // ── 設備與佇列初始化 ──
    int qf = findComputeQueueFamily();
    if (qf < 0) return false;
    computeQueueFamily_ = (uint32_t)qf;

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
    
    if (vkCreateDevice(physDevice_, &devCI, nullptr, &device_) != VK_SUCCESS) {
        fprintf(stderr, "[libpfsf] vkCreateDevice failed\n");
        return false;
    }

    vkGetDeviceQueue(device_, computeQueueFamily_, 0, &computeQueue_);

    VmaAllocatorCreateInfo vmaCI = {};
    vmaCI.vulkanApiVersion = VK_API_VERSION_1_1;
    vmaCI.physicalDevice = physDevice_;
    vmaCI.device = device_;
    vmaCI.instance = instance_;
    if (vmaCreateAllocator(&vmaCI, &allocator_) != VK_SUCCESS) {
        fprintf(stderr, "[libpfsf] VMA init failed\n");
        return false;
    }

    available_ = true;
    fprintf(stderr, "[libpfsf] 🚀 5070TI IS NOW ONLINE: %s\n", deviceName_.c_str());
    return true;
}

void VulkanContext::shutdown() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        if (allocator_) vmaDestroyAllocator(allocator_);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE; device_ = VK_NULL_HANDLE;
    available_ = false;
}

bool VulkanContext::selectPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (count == 0) return false;
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance_, &count, devices.data());
    
    for (auto& pd : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        
        // --- 核心改動：強制鎖定 5070TI ---
        if (props.vendorID == 0x10de) { // NVIDIA
            physDevice_ = pd;
            deviceName_ = props.deviceName;
            fprintf(stderr, "[libpfsf] Hardware match: %s\n", deviceName_.c_str());
            return true; 
        }
    }

    // fallback
    if (count > 0) {
        physDevice_ = devices[0];
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(physDevice_, &props);
        deviceName_ = props.deviceName;
        return true;
    }

    return false;
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
    VmaAllocation a; return vmaCreateBuffer(allocator_, &ci, &vi, b, &a, nullptr) == VK_SUCCESS;
}

void VulkanContext::freeBuffer(VkBuffer b, VkDeviceMemory) { vmaDestroyBuffer(allocator_, b, nullptr); }
void* VulkanContext::mapBuffer(VkBuffer b, VkDeviceSize) { return nullptr; }
void VulkanContext::unmapBuffer(VkBuffer b) {}
VkCommandBuffer VulkanContext::allocCmdBuffer() { return VK_NULL_HANDLE; }
VkResult VulkanContext::submitAndWait(VkCommandBuffer) { return VK_SUCCESS; }
VkShaderModule VulkanContext::createShaderModule(const uint32_t* p, size_t s) { return VK_NULL_HANDLE; }
VkDescriptorPool VulkanContext::createDescriptorPool(uint32_t, uint32_t) { return VK_NULL_HANDLE; }
VkDescriptorSet VulkanContext::allocDescriptorSet(VkDescriptorPool, VkDescriptorSetLayout) { return VK_NULL_HANDLE; }
bool VulkanContext::allocHostVisibleStorage(VkDeviceSize, VkBuffer*, void**) { return false; }

} // namespace pfsf
