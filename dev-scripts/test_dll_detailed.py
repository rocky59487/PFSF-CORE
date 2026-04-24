import ctypes
import traceback

try:
    print("Loading Vulkan SDK DLL...")
    ctypes.CDLL(r"C:\VulkanSDK\1.4.341.1\Bin\vulkan-1.dll")
    
    dll_path = r"C:\Users\wmc02\Desktop\pr_review\pfsf_core\Block Reality\api\build\native-out\META-INF\native\win-x64\br_core.dll"
    print(f"Loading {dll_path}...")
    lib = ctypes.CDLL(dll_path)
    print("SUCCESS!")
except OSError as e:
    print(f"FAILED (OSError): {e.winerror} - {e.strerror}")
except Exception as e:
    print(f"FAILED (Exception): {e}")
    traceback.print_exc()
