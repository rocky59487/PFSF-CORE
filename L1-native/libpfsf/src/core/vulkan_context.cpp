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

    fprintf(stderr, "[libpfsf] --- VULKAN INIT: TARGETING 5070TI ---\n");

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
            fprintf(stderr, "[libpfsf] Instance created with API %d.%d\n", 
                    VK_VERSION_MAJOR(ver), VK_VERSION_MINOR(ver));
            break;
        }
    }

    if (!instance_) {
        fprintf(stderr, "[libpfsf] vkCreateInstance FAILED: %d\n", (int)last_res);
        return false;
    }

    if (!selectPhysicalDevice()) {
        fprintf(stderr, "[libpfsf] 5070TI NOT FOUND!\n");
        return false;
    }

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
        fprintf(stderr, "[libpfsf] vkCreateDevice FAILED\n");
        return false;
    }

    vkGetDeviceQueue(device_, computeQueueFamily_, 0, &computeQueue_);

    VkCommandPoolCreateInfo cpCI = {};
    cpCI.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpCI.queueFamilyIndex = computeQueueFamily_;
    cpCI.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device_, &cpCI, nullptr, &cmdPool_) != VK_SUCCESS) {
        fprintf(stderr, "[libpfsf] vkCreateCommandPool FAILED\n");
        return false;
    }

    VmaAllocatorCreateInfo vmaCI = {};
    vmaCI.vulkanApiVersion = VK_API_VERSION_1_1;
    vmaCI.physicalDevice = physDevice_;
    vmaCI.device = device_;
    vmaCI.instance = instance_;
    if (vmaCreateAllocator(&vmaCI, &allocator_) != VK_SUCCESS) {
        fprintf(stderr, "[libpfsf] VMA FAILED\n");
        return false;
    }

    available_ = true;
    fprintf(stderr, "[libpfsf] 🚀 5070TI SUCCESS: %s\n", deviceName_.c_str());
    return true;
}

void VulkanContext::shutdown() {
    if (device_) {
        vkDeviceWaitIdle(device_);
        if (allocator_) vmaDestroyAllocator(allocator_);
        if (cmdPool_) vkDestroyCommandPool(device_, cmdPool_, nullptr);
        vkDestroyDevice(device_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
    instance_ = VK_NULL_HANDLE; device_ = VK_NULL_HANDLE; cmdPool_ = VK_NULL_HANDLE;
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
        if (props.vendorID == 0x10de) { 
            physDevice_ = pd; deviceName_ = props.deviceName; return true; 
        }
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

VkResult VulkanContext::submitAndWait(VkCommandBuffer cb) {
    vkEndCommandBuffer(cb);
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    vkQueueSubmit(computeQueue_, 1, &si, VK_NULL_HANDLE);
    VkResult res = vkQueueWaitIdle(computeQueue_);
    vkFreeCommandBuffers(device_, cmdPool_, 1, &cb); return res;
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
