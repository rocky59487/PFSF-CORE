#include <iostream>
#include <windows.h>

/**
 * 最小化真理證明器 (v2)：排除所有靜態初始化與複雜類別。
 */
int main() {
    std::cout << "PFSF STANDALONE PROBE START" << std::endl;
    
    const char* path = "C:\\Windows\\System32\\vulkan-1.dll";
    HMODULE h = LoadLibraryA(path);
    
    if (h) {
        std::cout << ">>> [PROBE] Vulkan DLL found and loaded at " << path << std::endl;
        auto pfn = GetProcAddress(h, "vkCreateInstance");
        if (pfn) {
            std::cout << ">>> [PROBE] vkCreateInstance symbol found!" << std::endl;
        } else {
            std::cout << ">>> [PROBE] ERROR: vkCreateInstance symbol missing!" << std::endl;
        }
        FreeLibrary(h);
    } else {
        std::cout << ">>> [PROBE] FATAL: Could not load " << path << ". Error: " << GetLastError() << std::endl;
    }

    std::cout << "PROBE END" << std::endl;
    return 0;
}
