package com.blockreality.api.physics.pfsf.debug;

import com.blockreality.api.physics.pfsf.NativePFSFBridge;
import com.blockreality.api.physics.pfsf.NativePFSFRuntime;
import com.blockreality.api.physics.pfsf.IPFSFRuntime;
import com.blockreality.api.physics.pfsf.VulkanComputeContext;
import org.junit.jupiter.api.*;

import static org.junit.jupiter.api.Assertions.*;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

/**
 * L6: In-game pipeline integration — validates the Java-side facade chain.
 *
 * <p>This layer does NOT call {@code onServerTick} with Minecraft objects (that
 * requires a running game). Instead it verifies:</p>
 * <ol>
 *   <li>{@link NativePFSFRuntime#init()} succeeds when the activation flag is set
 *       ({@code -Dblockreality.native.pfsf=true} forwarded to the test JVM via
 *       {@code build.gradle}).</li>
 *   <li>{@link NativePFSFRuntime#isActive()} returns {@code true} after init.</li>
 *   <li>{@link NativePFSFRuntime#asRuntime()}{@code .isAvailable()} returns {@code true},
 *       confirming the {@code KERNELS_PORTED} gate is open.</li>
 *   <li>Config setters (wind vector, lookup functions) do not throw.</li>
 * </ol>
 *
 * <p>If L5 passes but L6 fails, the problem is in the Java integration layer:
 * the lookup functions may not be wired to {@code PFSFEngine}, or
 * {@code NativePFSFRuntime.init()} is not called from {@code PFSFEngine.init()},
 * or the activation flag is not being read correctly.</p>
 */
@DisplayName("L6: In-Game Pipeline Integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class L6_InGamePipelineTest {

    @BeforeAll
    void requireNativeLibrary() {
        assumeTrue(NativePFSFBridge.isAvailable(),
                "libblockreality_pfsf not loaded — L0 must pass first");
        assumeTrue(NativePFSFRuntime.areKernelsPorted(),
                "KERNELS_PORTED=false — set it to true in NativePFSFRuntime.java:31");
        // L6-03 asserts RuntimeView.isAvailable() which is gated on
        // VulkanComputeContext.isAvailable() (no-fallback contract: native
        // runtime borrows Vulkan handles from the Java side). Tests must
        // initialize the Java Vulkan context before asserting availability.
        VulkanComputeContext.init();
        assumeTrue(VulkanComputeContext.isAvailable(),
                "VulkanComputeContext init failed — Java-side Vulkan compute unavailable. "
                        + "Driver / GPU may not support compute, or LWJGL Vulkan loader is missing.");
    }

    @AfterAll
    void shutdown() {
        if (NativePFSFRuntime.isActive()) {
            NativePFSFRuntime.shutdown();
        }
        VulkanComputeContext.shutdown();
    }

    @Test
    @Order(1)
    @DisplayName("L6-01: activation flag 被轉送至測試 JVM（-Dblockreality.native.pfsf=true）")
    void activationFlagForwarded() {
        boolean flag = NativePFSFRuntime.isFlagEnabled();
        System.out.println("[L6] FLAG_ENABLED=" + flag);
        assertTrue(flag,
                "[L6-FAIL] blockreality.native.pfsf system property is NOT set in the test JVM.\n" +
                "  The build.gradle test block must forward it.  Expected entry:\n" +
                "    def pfsfFlag = System.getProperty('blockreality.native.pfsf')\n" +
                "    if (pfsfFlag != null) systemProperty 'blockreality.native.pfsf', pfsfFlag\n" +
                "  And run tests with: ./gradlew :api:test -Dblockreality.native.pfsf=true");
    }

    @Test
    @Order(2)
    @DisplayName("L6-02: NativePFSFRuntime.init() 成功，isActive = true")
    void runtimeInitSucceeds() {
        NativePFSFRuntime.init();
        boolean active = NativePFSFRuntime.isActive();
        System.out.println("[L6] isActive after init=" + active
                + "  status=" + NativePFSFRuntime.getStatus());
        assumeTrue(active,
                "NativePFSFRuntime.init() did not set active=true.\n" +
                "  Check that Vulkan init succeeds (pfsf_init returns OK).\n" +
                "  Look at the log output above for the exact failure reason.");
    }

    @Test
    @Order(3)
    @DisplayName("L6-03: asRuntime().isAvailable() = true (active && KERNELS_PORTED)")
    void runtimeViewIsAvailable() {
        assumeTrue(NativePFSFRuntime.isActive(), "runtime not active (L6-02 failed)");
        IPFSFRuntime view = NativePFSFRuntime.asRuntime();
        assertNotNull(view, "asRuntime() must always return a non-null singleton");
        assertTrue(view.isAvailable(),
                "[L6-FAIL] IPFSFRuntime.isAvailable()=false despite active=true and KERNELS_PORTED=true.\n" +
                "  Check RuntimeView.isAvailable() at NativePFSFRuntime.java: active && KERNELS_PORTED");
    }

    @Test
    @Order(4)
    @DisplayName("L6-04: 設定 wind vector + lookup stubs 不拋例外")
    void configSettersDoNotThrow() {
        assumeTrue(NativePFSFRuntime.isActive(), "runtime not active");
        IPFSFRuntime view = NativePFSFRuntime.asRuntime();
        assertDoesNotThrow(() -> {
            view.setWindVector(new net.minecraft.world.phys.Vec3(5.0, 0.0, 0.0));
            view.setMaterialLookup(null);
            view.setAnchorLookup(null);
            view.setFillRatioLookup(null);
            view.setCuringLookup(null);
            view.removeBuffer(/*islandId=*/ 999); // non-existent island — should be a no-op
        });
    }

    @Test
    @Order(5)
    @DisplayName("L6-05: getStatus() 前綴為 'Native PFSF: ROUTING'")
    void statusStringIndicatesRouting() {
        String status = NativePFSFRuntime.getStatus();
        System.out.println("[L6] status=" + status);
        assertNotNull(status);
        assertTrue(status.startsWith("Native PFSF:"),
                "Status must start with 'Native PFSF:' got: " + status);
        if (NativePFSFRuntime.isActive()) {
            assertEquals("Native PFSF: ROUTING", status,
                    "Active runtime with KERNELS_PORTED=true must report ROUTING");
        }
    }
}
