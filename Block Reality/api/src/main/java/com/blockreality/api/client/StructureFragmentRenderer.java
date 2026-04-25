package com.blockreality.api.client;

import com.blockreality.api.fragment.StructureFragmentEntity;
import com.mojang.blaze3d.vertex.PoseStack;
import net.minecraft.client.renderer.MultiBufferSource;
import net.minecraft.client.renderer.culling.Frustum;
import net.minecraft.client.renderer.entity.EntityRenderer;
import net.minecraft.client.renderer.entity.EntityRendererProvider;
import net.minecraft.resources.ResourceLocation;
import net.minecraftforge.api.distmarker.Dist;
import net.minecraftforge.api.distmarker.OnlyIn;

/**
 * Placeholder renderer for {@link StructureFragmentEntity}.
 *
 * The fragment entity carries the block snapshot of a collapsed sub-structure
 * for server-side rigid-body simulation; once it settles, blocks are placed
 * back into the world by {@code StructureFragmentManager.onFragmentSettle()}.
 *
 * Visual rendering of the tumbling fragment is not yet implemented. This
 * renderer must still exist so the dispatcher does not NPE when the spawn
 * packet arrives on the client (vanilla {@code EntityRenderDispatcher.shouldRender}
 * calls {@code renderers.get(type).shouldRender(...)} with no null guard).
 */
@OnlyIn(Dist.CLIENT)
public class StructureFragmentRenderer extends EntityRenderer<StructureFragmentEntity> {

    private static final ResourceLocation NO_TEXTURE =
        new ResourceLocation("minecraft", "textures/misc/white.png");

    public StructureFragmentRenderer(EntityRendererProvider.Context ctx) {
        super(ctx);
    }

    @Override
    public boolean shouldRender(StructureFragmentEntity entity, Frustum frustum,
                                double camX, double camY, double camZ) {
        return false;
    }

    @Override
    public void render(StructureFragmentEntity entity, float yaw, float partialTicks,
                       PoseStack pose, MultiBufferSource buffers, int light) {
        // intentionally empty — see class javadoc
    }

    @Override
    public ResourceLocation getTextureLocation(StructureFragmentEntity entity) {
        return NO_TEXTURE;
    }
}
