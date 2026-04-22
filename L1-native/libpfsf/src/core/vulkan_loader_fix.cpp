/**
 * @file vulkan_loader_fix.cpp
 * @brief 絕對路徑載入器 — 最終調試版。
 */
#include <windows.h>
#include <cstdio>
#include <vulkan/vulkan.h>

namespace pfsf {
    void* manual_vulkan_handle = nullptr;

    bool force_load_vulkan() {
        if (manual_vulkan_handle) return true;

        const char* path = "C:\\Windows\\System32\\vulkan-1.dll";
        HMODULE hModule = LoadLibraryA(path);

        if (!hModule) {
            fprintf(stderr, "[libpfsf] >>> FATAL: LoadLibraryA(%s) failed! Error: %lu\n", path, GetLastError());
            return false;
        }

        // 檢查符號是否存在
        auto pfn = (void*)GetProcAddress(hModule, "vkCreateInstance");
        if (!pfn) {
            fprintf(stderr, "[libpfsf] >>> FATAL: vkCreateInstance symbol NOT FOUND in %s!\n", path);
            FreeLibrary(hModule);
            return false;
        }

        manual_vulkan_handle = (void*)hModule;
        fprintf(stderr, "[libpfsf] >>> SUCCESS: Vulkan Loader & Symbols verified in System32\n");
        return true;
    }
}
