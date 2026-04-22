package com.blockreality.api.spi;

import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import javax.annotation.Nonnull;

/**
 * No-Op Implementation of IThermalManager.
 * Used as a placeholder to safely disable thermal logic.
 */
public class NoOpThermalManager implements IThermalManager {
    public static final NoOpThermalManager INSTANCE = new NoOpThermalManager();

    @Override public void init(@Nonnull ServerLevel level) {}
    @Override public void tick(@Nonnull ServerLevel level, int tickBudgetMs) {}
    @Override public void shutdown() {}
    @Override public float getTemperatureAt(@Nonnull BlockPos pos) { return 20.0f; }
    @Override public void setHeatSource(@Nonnull BlockPos pos, float temperature) {}
    @Override public void removeHeatSource(@Nonnull BlockPos pos) {}
    @Override public int getActiveRegionCount() { return 0; }
}
