package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import com.blockreality.api.physics.pfsf.NativePFSFRuntime;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * L0: JNI bridge smoke test — no GPU or Vulkan required.
 *
 * <p>Tests library loading, version strings, feature-gate probes, and static constants.
 * If any test here FAILs (rather than SKIPs), the native library itself is missing or
 * has an ABI mismatch. Fix by running:</p>
 * <pre>
 *   ./gradlew nativeBuild -Pblockreality.native.build=true
 * </pre>
 * <p>Then re-run with {@code -Dblockreality.native.pfsf=true} forwarded to the test JVM.</p>
 */
@DisplayName("L0: JNI Bridge Smoke")
class L0_JniBridgeSmokeTest {

    @Test
    @DisplayName("L0-01: 庫載入不拋例外，getVersion/getAbiContractVersion 非 null")
    void libraryLoadDoesNotThrow() {
        assertDoesNotThrow(() -> {
            boolean available = NativePFSFBridge.isAvailable();
            String  version   = NativePFSFBridge.getVersion();
            String  contract  = NativePFSFBridge.getAbiContractVersion();
            assertNotNull(version,  "getVersion() must never return null");
            assertNotNull(contract, "getAbiContractVersion() must never return null");
            System.out.println("[L0] isAvailable=" + available
                    + "  version=" + version + "  abiContract=" + contract);
        });
    }

    @Test
    @DisplayName("L0-02: isAvailable = true（庫已成功載入）")
    void libraryIsAvailable() {
        assertTrue(NativePFSFBridge.isAvailable(),
                "[L0-FAIL] libblockreality_pfsf not loaded.\n" +
                "  • Check java.library.path contains the native build output directory.\n" +
                "  • Or run: ./gradlew nativeBuild -Pblockreality.native.build=true\n" +
                "  • Then re-test with: -Dblockreality.native.pfsf=true");
    }

    @Test
    @DisplayName("L0-03: ABI 合約版本格式為 X.Y.Z（或 n/a 降級）")
    void abiContractVersionFormat() {
        String v = NativePFSFBridge.getAbiContractVersion();
        assertTrue(v.matches("\\d+\\.\\d+\\.\\d+") || v.equals("n/a") || v.equals("0.0.0"),
                "ABI contract version must be semver or sentinel, got: " + v);
    }

    @Test
    @DisplayName("L0-04: feature gate 狀態可讀取（不拋例外）")
    void featureGatesReadable() {
        assertDoesNotThrow(() -> {
            boolean v1 = NativePFSFBridge.hasComputeV1();
            boolean v2 = NativePFSFBridge.hasComputeV2();
            boolean v3 = NativePFSFBridge.hasComputeV3();
            System.out.println("[L0] compute.v1=" + v1 + "  v2=" + v2 + "  v3=" + v3);
            // Diagnostic: warn if v1 absent so L1 skip is anticipated
            if (!v1) System.out.println("[L0] WARNING: compute.v1 absent — L1 tests will skip");
        });
    }

    @Test
    @DisplayName("L0-05: PFSFResult.describe 覆蓋所有已知代碼")
    void pfsfResultDescribeCoverage() {
        assertEquals("OK",          NativePFSFBridge.PFSFResult.describe(NativePFSFBridge.PFSFResult.OK));
        assertEquals("VULKAN",      NativePFSFBridge.PFSFResult.describe(NativePFSFBridge.PFSFResult.ERROR_VULKAN));
        assertEquals("NO_DEVICE",   NativePFSFBridge.PFSFResult.describe(NativePFSFBridge.PFSFResult.ERROR_NO_DEVICE));
        assertEquals("OUT_OF_VRAM", NativePFSFBridge.PFSFResult.describe(NativePFSFBridge.PFSFResult.ERROR_OUT_OF_VRAM));
        assertEquals("INVALID_ARG", NativePFSFBridge.PFSFResult.describe(NativePFSFBridge.PFSFResult.ERROR_INVALID_ARG));
        assertEquals("NOT_INIT",    NativePFSFBridge.PFSFResult.describe(NativePFSFBridge.PFSFResult.ERROR_NOT_INIT));
        assertEquals("ISLAND_FULL", NativePFSFBridge.PFSFResult.describe(NativePFSFBridge.PFSFResult.ERROR_ISLAND_FULL));
        assertTrue(NativePFSFBridge.PFSFResult.describe(99).startsWith("UNKNOWN"),
                "Unknown code must produce UNKNOWN prefix");
    }

    @Test
    @DisplayName("L0-06: KERNELS_PORTED = true（CI parity job 已翻轉旗標）")
    void kernelsPortedIsTrue() {
        boolean active  = NativePFSFRuntime.isActive();
        boolean flag    = NativePFSFRuntime.isFlagEnabled();
        boolean kernels = NativePFSFRuntime.areKernelsPorted();
        System.out.println("[L0] NativePFSFRuntime: active=" + active
                + "  flag=" + flag + "  kernels=" + kernels);
        assertTrue(kernels,
                "[L0-FAIL] KERNELS_PORTED is false in NativePFSFRuntime.java:31.\n" +
                "  This blocks isAvailable() even when native is loaded and init succeeds.\n" +
                "  Set KERNELS_PORTED = true to enable the native routing.");
    }
}
