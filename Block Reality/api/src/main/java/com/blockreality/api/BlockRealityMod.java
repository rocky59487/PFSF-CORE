package com.blockreality.api;

import com.blockreality.api.command.BrCommand;
import com.blockreality.api.collapse.CollapseManager;
import com.blockreality.api.config.BRConfig;
// import com.blockreality.api.diagnostic.BrCrashReporter;
import com.blockreality.api.diagnostic.BrLogCapture;
import com.blockreality.api.material.VanillaMaterialMap;
import com.blockreality.api.network.BRNetwork;
import com.blockreality.api.physics.AnchorContinuityChecker;
import com.blockreality.api.physics.ConnectivityCache;
import com.blockreality.api.physics.pfsf.PFSFEngine;
import com.blockreality.api.fragment.StructureFragmentManager;
import com.blockreality.api.registry.BRBlockEntities;
import com.blockreality.api.registry.BRBlocks;
import com.blockreality.api.registry.BREntities;

import com.blockreality.api.spi.ModuleRegistry;
import com.google.gson.JsonObject;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.world.item.CreativeModeTab;
import net.minecraft.world.item.ItemStack;
import net.minecraftforge.common.MinecraftForge;
import net.minecraftforge.event.RegisterCommandsEvent;
import net.minecraftforge.event.server.ServerStartedEvent;
import net.minecraftforge.event.server.ServerStartingEvent;
import net.minecraftforge.event.server.ServerStoppingEvent;
import net.minecraftforge.eventbus.api.IEventBus;
import net.minecraftforge.eventbus.api.SubscribeEvent;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.fml.DistExecutor;
import net.minecraftforge.fml.common.Mod;
import net.minecraftforge.fml.config.ModConfig;
import net.minecraftforge.fml.event.lifecycle.FMLCommonSetupEvent;
import net.minecraftforge.registries.DeferredRegister;
import net.minecraftforge.registries.ForgeRegistries;
import net.minecraftforge.registries.RegistryObject;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

@Mod(BlockRealityMod.MOD_ID)
public class BlockRealityMod {
    public static final String MOD_ID = "blockreality";
    private static final Logger LOGGER = LogManager.getLogger("BlockReality");

    static {
        // LWJGL MemoryStack 預設 64KB / thread。Blackwell + 近期 NVIDIA 驅動回報的
        // device extension 數量會讓 new VkInstance(...) 內部的
        // VkExtensionProperties.malloc(stack) 一次需要 ~80KB，必爆。
        // 在 mod class 初始化時拉高，確保 server thread 首次取得 MemoryStack 時用 256KB。
        // 若使用者透過 -Dorg.lwjgl.system.stackSize=N 設過更大，尊重該值。
        Integer current = org.lwjgl.system.Configuration.STACK_SIZE.get();
        if (current == null || current < 256) {
            org.lwjgl.system.Configuration.STACK_SIZE.set(256);
        }
    }

    // ─── Creative Tab ───
    // Use ResourceLocation form to avoid NoSuchFieldError on Registries.CREATIVE_MODE_TAB
    // in production Forge (the Registries class field uses SRG names in the universal jar).
    public static final DeferredRegister<CreativeModeTab> CREATIVE_TABS =
        DeferredRegister.create(ResourceLocation.parse("creative_mode_tab"), MOD_ID);
        // B1-fix: ResourceLocation(String) deprecated → ResourceLocation.parse()

    public static final RegistryObject<CreativeModeTab> BR_TAB = CREATIVE_TABS.register("br_tab",
        () -> CreativeModeTab.builder()
            .icon(() -> new ItemStack(BRBlocks.R_CONCRETE.get()))
            .title(Component.literal("Block Reality"))
            .displayItems((params, output) -> {
                output.accept(BRBlocks.R_CONCRETE_ITEM.get());
                output.accept(BRBlocks.R_REBAR_ITEM.get());
                output.accept(BRBlocks.R_STEEL_ITEM.get());
                output.accept(BRBlocks.R_TIMBER_ITEM.get());
                output.accept(BRBlocks.CHISEL.get());
            })
            .build()
    );

    // NOTE: FMLJavaModLoadingContext.get() is deprecated but there is no non-deprecated
    // alternative in Forge 1.20.1. The replacement (constructor injection) was added in 1.21.1.
    // Suppress until we upgrade the Forge target version.
    @SuppressWarnings("removal")
    public BlockRealityMod() {
        // ─── 最優先：安裝崩潰報告器 ───
        // BrCrashReporter.install();
        BrLogCapture.install();

        // B2-fix: FMLJavaModLoadingContext.get() deprecated in 1.20.6+
        //         Use the IEventBus injected into the mod constructor instead.
        IEventBus modBus = net.minecraftforge.fml.javafmlmod.FMLJavaModLoadingContext.get().getModEventBus();

        // ─── 註冊 Deferred Registers ───
        BRBlocks.BLOCKS.register(modBus);
        BRBlocks.ITEMS.register(modBus);
        BRBlockEntities.BLOCK_ENTITIES.register(modBus);
        BREntities.ENTITIES.register(modBus);
        CREATIVE_TABS.register(modBus);

        // ─── 註冊 Config ───
        // B3-fix: ModLoadingContext.get() deprecated → use FMLJavaModLoadingContext directly
        net.minecraftforge.fml.javafmlmod.FMLJavaModLoadingContext.get().registerConfig(ModConfig.Type.COMMON, BRConfig.SPEC);

        // ─── Lifecycle events ───
        modBus.addListener(this::commonSetup);

        // ─── 客戶端初始化（安全分離，伺服器不載入 client 類）───
        DistExecutor.unsafeRunWhenOn(Dist.CLIENT, () -> () -> {
            com.blockreality.api.client.ClientSetup.initForgeEvents();
        });

        MinecraftForge.EVENT_BUS.register(this);
        LOGGER.info("[BlockReality] Mod 初始化完成 — v0.2.0-alpha (R-unit system)");
    }

    private void commonSetup(FMLCommonSetupEvent event) {
        event.enqueueWork(() -> {
            BRNetwork.register();
            VanillaMaterialMap.getInstance().init();
                        LOGGER.info("[BlockReality] Network channel registered, VanillaMaterialMap loaded ({} entries)",
                VanillaMaterialMap.getInstance().size());

            // C7: Initialize VS2 bridge if VS2 is installed alongside Block Reality.
            // The bridge handles fragment dynamics (rotation, rolling) while Block Reality
            // handles static analysis and initial velocity computation.
            if (net.minecraftforge.fml.ModList.get().isLoaded("valkyrienskies")) {
                try {
                    com.blockreality.api.spi.IVS2Bridge bridge =
                        new com.blockreality.api.vs2.VS2ShipBridge();
                    com.blockreality.api.spi.ModuleRegistry.setVS2Bridge(bridge);
                    LOGGER.info("[BlockReality] VS2 detected — VS2ShipBridge activated for fragment dynamics");
                } catch (Exception e) {
                    LOGGER.warn("[BlockReality] VS2 detected but bridge init failed — " +
                        "using built-in StructureFragmentEntity fallback", e);
                }
            } else {
                LOGGER.info("[BlockReality] VS2 not detected — built-in StructureFragmentEntity active");
            }
        });
    }

    @SubscribeEvent
    public void onRegisterCommands(RegisterCommandsEvent event) {
        BrCommand.register(event.getDispatcher());
        com.blockreality.api.command.StressAnalysisCommand.register(event.getDispatcher());
        LOGGER.info("[BlockReality] 已註冊指令: /br (toggle|status|vulkan_test), /br_stress");

        // ★ Register commands from all modules
        for (var provider : ModuleRegistry.getCommandProviders()) {
            try {
                provider.registerCommands(event.getDispatcher());
                LOGGER.debug("[BlockReality] Registered commands from module: {}", provider.getModuleId());
            } catch (RuntimeException e) {
                LOGGER.error("[BlockReality] Error registering commands from module {}: {}",
                    provider.getModuleId(), e.getMessage(), e);
            }
        }
    }

    @SubscribeEvent
    public void onServerStarting(ServerStartingEvent event) {

        // ─── B1-fix: 初始化 PFSF GPU 物理引擎 ───
        // No-fallback contract: any failure during init engages PFSFLockdown so
        // RBlock interactions are blocked and players see a red HUD warning.
        // The mod still loads — the rest of vanilla survives — but physics is off.
        try {
            com.blockreality.api.physics.pfsf.VulkanComputeContext.init();
            PFSFEngine.init();
            if (PFSFEngine.isAvailable()) {
                // 設定材料/錨點/填充率查詢函數
                PFSFEngine.setMaterialLookup(pos -> {
                    net.minecraft.server.MinecraftServer srv =
                            net.minecraftforge.server.ServerLifecycleHooks.getCurrentServer();
                    if (srv == null) return null;
                    net.minecraft.server.level.ServerLevel level = srv.overworld();
                    if (level == null) return null;
                    net.minecraft.world.level.block.state.BlockState state = level.getBlockState(pos);
                    String blockId = net.minecraftforge.registries.ForgeRegistries.BLOCKS
                            .getKey(state.getBlock()).toString();
                    return VanillaMaterialMap.getInstance().getMaterial(blockId);
                });
                PFSFEngine.setAnchorLookup(pos -> {
                    net.minecraft.server.MinecraftServer srv =
                            net.minecraftforge.server.ServerLifecycleHooks.getCurrentServer();
                    if (srv == null) return false;
                    net.minecraft.server.level.ServerLevel level = srv.overworld();
                    if (level == null) return false;
                    return AnchorContinuityChecker.getInstance().isAnchored(level, pos);
                });
                PFSFEngine.setFillRatioLookup(pos -> 1.0f); // 預設滿填充
                LOGGER.info("[BlockReality] PFSF GPU 物理引擎已啟動");
            } else {
                // VulkanComputeContext or PFSFEngineInstance already engaged lockdown
                // when they failed; assert it here so a missing wiring upstream still
                // trips the lockdown for the player-visible red HUD path.
                if (!com.blockreality.api.physics.pfsf.PFSFLockdown.isLocked()) {
                    com.blockreality.api.physics.pfsf.PFSFLockdown.lock(
                            "PFSF GPU engine unavailable (Vulkan compute required)");
                }
                LOGGER.error("[BlockReality] ════════════════════════════════════════════");
                LOGGER.error("[BlockReality] PFSF GPU LOCKDOWN: {}",
                        com.blockreality.api.physics.pfsf.PFSFLockdown.getReason());
                LOGGER.error("[BlockReality] RBlock 互動已凍結；玩家上線時會看到紅字警告");
                LOGGER.error("[BlockReality] 請檢視 [PFSF-VulkanCtx] 訊息以取得失敗原因");
                LOGGER.error("[BlockReality] ════════════════════════════════════════════");
            }
        } catch (Exception e) {
            String reason = "PFSF init exception: " + e.getClass().getSimpleName()
                    + (e.getMessage() != null ? " — " + e.getMessage() : "");
            LOGGER.error("[BlockReality] {}", reason, e);
            com.blockreality.api.physics.pfsf.PFSFLockdown.lock(reason);
        }
    }

    @SubscribeEvent
    public void onServerStarted(ServerStartedEvent event) {
        // Single-line summary of the physics chain so server operators
        // and modpack authors can see at a glance whether physics is
        // actually going to run for this world, without digging for
        // the individual PFSF init log lines. Matches /br status
        // output so the in-game command reproduces what the log shows.
        boolean physicsOn = com.blockreality.api.config.BRConfig.isPhysicsEnabled();
        boolean pfsfOn    = com.blockreality.api.config.BRConfig.isPFSFEnabled();
        boolean pfsfAvail = PFSFEngine.isAvailable();
        boolean locked    = com.blockreality.api.physics.pfsf.PFSFLockdown.isLocked();
        boolean nativeOn  = com.blockreality.api.physics.pfsf.NativePFSFRuntime.isActive();
        int shaders = com.blockreality.api.physics.pfsf.NativePFSFRuntime.getShadersRegistered();
        boolean pnsmShadow = com.blockreality.api.config.BRConfig.isPNSMShadowEnabled();
        LOGGER.info("[BlockReality] Physics status — physics={} pfsf_cfg={} pfsf_gpu={} locked={} native={} shaders={} pnsm_shadow={}",
                physicsOn ? "ON" : "OFF",
                pfsfOn ? "ON" : "OFF",
                pfsfAvail ? "AVAILABLE" : "UNAVAILABLE",
                locked ? "YES (" + com.blockreality.api.physics.pfsf.PFSFLockdown.getReason() + ")" : "NO",
                nativeOn ? "ACTIVE" : "INACTIVE",
                shaders,
                pnsmShadow ? "ON" : "OFF");
        if (locked) {
            LOGGER.warn("[BlockReality] PFSF LOCKDOWN ACTIVE — RBlock interactions blocked, players will see red HUD banner");
        }
    }

    @SubscribeEvent
    public void onServerStopping(ServerStoppingEvent event) {
        
        
        PFSFEngine.shutdown();

        // ─── Vulkan 計算環境關閉（最後：所有 pipeline/buffer 已釋放後才銷毀 VkDevice）───
        com.blockreality.api.physics.pfsf.VulkanComputeContext.shutdown();

        // 清理快取（避免跨世界洩漏）
        ConnectivityCache.clearCache();
        CollapseManager.clearQueue();
        StructureFragmentManager.clearAll();

        // 卸載日誌捕捉 Appender（伺服器正常關閉時）
        BrLogCapture.uninstall();

        LOGGER.info("[BlockReality] All engines stopped, caches cleared");
    }
}
