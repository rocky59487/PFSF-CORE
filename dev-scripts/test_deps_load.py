import ctypes
deps = ["vulkan-1.dll", "MSVCP140.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll", "KERNEL32.dll"]
for dep in deps:
    try:
        ctypes.CDLL(dep)
        print(f"OK: {dep}")
    except OSError as e:
        print(f"FAILED: {dep} - {e}")
