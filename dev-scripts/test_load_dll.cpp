#include <windows.h>
#include <iostream>

int main() {
    SetDllDirectoryA("C:\\VulkanSDK\\1.4.341.1\\Bin");
    const char* path = "C:\\Users\\wmc02\\Desktop\\pr_review\\pfsf_core\\Block Reality\\api\\build\\native-out\\META-INF\\native\\win-x64\\br_core.dll";
    HMODULE h = LoadLibraryA(path);
    if (!h) {
        std::cerr << "Failed to load br_core.dll. Error: " << GetLastError() << std::endl;
        return 1;
    }
    std::cout << "Loaded br_core.dll successfully!" << std::endl;
    FreeLibrary(h);
    return 0;
}
