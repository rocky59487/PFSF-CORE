/**
 * @file vulkan_loader_fix.cpp
 * @brief 強制手動加載系統 Vulkan 驅動。
 */
#include <windows.h>
#include <cstdio>

namespace pfsf {
    void* manual_vulkan_handle = nullptr;

    bool force_load_vulkan() {
        if (manual_vulkan_handle) return true;

        // 優先嘗試 System32
        const char* path = "C:\\Windows\\System32\\vulkan-1.dll";
        manual_vulkan_handle = (void*)LoadLibraryA(path);

        if (!manual_vulkan_handle) {
            manual_vulkan_handle = (void*)LoadLibraryA("vulkan-1.dll");
        }

        if (manual_vulkan_handle) {
            fprintf(stderr, "[libpfsf] Vulkan Loader found.\n");
            return true;
        } else {
            fprintf(stderr, "[libpfsf] Vulkan Loader NOT FOUND (Error: %lu)\n", GetLastError());
            return false;
        }
    }
}
