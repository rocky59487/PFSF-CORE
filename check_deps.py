import os
import subprocess

deps = [
    "vulkan-1.dll",
    "MSVCP140.dll",
    "VCRUNTIME140.dll",
    "VCRUNTIME140_1.dll"
]

for dep in deps:
    res = subprocess.run(["where", dep], capture_output=True, text=True)
    if res.returncode == 0:
        print(f"FOUND {dep}: {res.stdout.strip().splitlines()[0]}")
    else:
        print(f"MISSING {dep}")
