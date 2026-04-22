#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>

/**
 * 終極 Standalone 探針：直接測試系統對 Vulkan 的訪問權限。
 */
int main() {
    std::cout << "--- STANDALONE VULKAN HARDWARE PROBE ---" << std::endl;

    VkApplicationInfo appInfo = {};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "HardwareProbe";
    appInfo.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo instCI = {};
    instCI.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instCI.pApplicationInfo = &appInfo;

    VkInstance instance;
    VkResult res = vkCreateInstance(&instCI, nullptr, &instance);

    if (res == VK_SUCCESS) {
        std::cout << "✅ SUCCESS: Vulkan Instance Created!" << std::endl;
        
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        std::cout << "Found " << deviceCount << " GPU(s):" << std::endl;
        for (auto d : devices) {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(d, &props);
            std::cout << "   - " << props.deviceName << " (Vendor: " << props.vendorID << ")" << std::endl;
        }

        vkDestroyInstance(instance, nullptr);
        return 0;
    } else {
        std::cerr << "❌ FATAL: vkCreateInstance failed with code: " << (int)res << std::endl;
        return (int)res;
    }
}
