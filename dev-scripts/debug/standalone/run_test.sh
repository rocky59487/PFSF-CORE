#!/usr/bin/env bash
# Standalone PFSF physics test — no Forge/Gradle/JDK17 required.
# Tests the full Java→JNI→C++→Vulkan pipeline using lavapipe.
#
# Usage: ./run_test.sh
#   Optional: BUILD_DIR=/path/to/native-build ./run_test.sh

set -e
cd "$(dirname "$0")"

BUILD_DIR="${BUILD_DIR:-/home/user/Block-Realityapi-Fast-design/build/native-build}"
JAVA="${JAVA:-java}"

echo "Building..."
mkdir -p out/com/blockreality/api/physics/pfsf
"${JAVA}c" -d out NativePFSFBridge.java PFSFPhysicsTest.java 2>&1

# Edit NativePFSFBridge.java BUILD_DIR if your native build is elsewhere
sed -i "s|/home/user/Block-Realityapi-Fast-design/build/native-build|${BUILD_DIR}|g" NativePFSFBridge.java 2>/dev/null || true

echo "Running L0–L5 physics test..."
VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json \
    "${JAVA}" -cp out PFSFPhysicsTest
