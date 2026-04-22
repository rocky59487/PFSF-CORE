package com.blockreality.api.spi;

import net.minecraft.core.BlockPos;
import net.minecraft.server.level.ServerLevel;
import javax.annotation.Nonnull;
import java.util.Collection;

/**
 * No-Op Implementation of IFluidManager.
 * Used as a placeholder to safely disable fluid logic.
 */
public class NoOpFluidManager implements IFluidManager {
    public static final NoOpFluidManager INSTANCE = new NoOpFluidManager();

    @Override public void init(@Nonnull ServerLevel level) {}
    @Override public void tick(@Nonnull ServerLevel level, int tickBudgetMs) {}
    @Override public void shutdown() {}
    @Override public float getFluidPressureAt(@Nonnull BlockPos pos) { return 0.0f; }
    @Override public float getFluidVolumeAt(@Nonnull BlockPos pos) { return 0.0f; }
    @Override public void notifyBarrierBreach(@Nonnull BlockPos pos) {}
    @Override public void notifyBarrierBreachBatch(@Nonnull Collection<BlockPos> positions) {}
    @Override public void setFluidSource(@Nonnull BlockPos pos, int type, float volume) {}
    @Override public void removeFluid(@Nonnull BlockPos pos) {}
    @Override public int getActiveRegionCount() { return 0; }
    @Override public int getTotalFluidVoxelCount() { return 0; }
}
