import ctypes
import os

try:
    os.add_dll_directory(r"C:\VulkanSDK\1.4.341.1\Bin")
    dll_path = r"C:\Users\wmc02\Desktop\pr_review\pfsf_core\Block Reality\api\build\native-out\META-INF\native\win-x64\br_core.dll"
    print(f"Loading {dll_path}...")
    lib = ctypes.CDLL(dll_path)
    print("SUCCESS!")
except Exception as e:
    print(f"FAILED: {e}")
