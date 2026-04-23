@echo off
title Block Reality - C++ Engine Client
echo =======================================================
echo  Block Reality - Local C++ Environment Initializer
echo =======================================================
echo.

:: 1. 尋找並初始化 MSVC x64 編譯環境
echo [1/3] Initializing MSVC x64 Environment...
if exist "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
) else if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" (
    call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat" x64
) else (
    echo [!] WARNING: MSVC vcvarsall.bat not found. Native C++ build might fail.
)
echo.

:: 2. 配置 Vulkan SDK (針對您的電腦環境)
echo [2/3] Configuring Vulkan SDK...
set VULKAN_SDK=C:\VulkanSDK\1.4.341.1
if not exist "%VULKAN_SDK%" (
    echo [!] WARNING: Vulkan SDK not found at %VULKAN_SDK%
) else (
    set "PATH=%VULKAN_SDK%\Bin;%PATH%"
    set "VK_ICD_FILENAMES=C:\Windows\System32\DriverStore\FileRepository\nvami.inf_amd64_2c2a10115c2556ed\nv-vk64.json"
)
echo.

:: 3. 啟動 Gradle (編譯 C++ 原生庫並啟動 Minecraft 客戶端)
echo [3/3] Building C++ Core and Starting Minecraft...
cd "Block Reality"
call gradlew.bat :api:runClient "-Pblockreality.native.build=true" "-Pblockreality.native.shaders=true" "-Dblockreality.native.pfsf=true"

echo.
echo Minecraft client closed.
pause
