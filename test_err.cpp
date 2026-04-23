#include <windows.h>
#include <iostream>
#include <string>

int main() {
    DWORD err = 4551;
    LPSTR messageBuffer = nullptr;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, err, MAKELANGID(LANG_ENGLISH, SUBLANG_ENGLISH_US), (LPSTR)&messageBuffer, 0, NULL);
    if (size) {
        std::cout << "Error 4551 means: " << messageBuffer << std::endl;
        LocalFree(messageBuffer);
    } else {
        std::cout << "FormatMessage failed." << std::endl;
    }
    return 0;
}
