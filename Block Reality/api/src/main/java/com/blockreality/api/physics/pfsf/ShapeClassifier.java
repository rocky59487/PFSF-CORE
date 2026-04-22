package com.blockreality.api.physics.pfsf;

import net.minecraft.core.BlockPos;
import java.util.Set;

/**
 * Classifies structure islands as "regular" or "irregular" to decide
 * which physics backend to use:
 *
 * <ul>
 *   <li>Regular (score < threshold) → PFSF iterative solver (fast, proven)</li>
 *   <li>Irregular (score >= threshold) → FNO ML surrogate (handles cantilevers, arches, etc.)</li>
 * </ul>
 *
 * <p>Irregularity score ∈ [0, 1] based on:</p>
 * <ol>
 *   <li>Fill ratio vs AABB (empty space = irregular)</li>
 *   <li>Surface-to-volume ratio (jagged = irregular)</li>
 *   <li>Vertical profile variance (inconsistent layers = irregular)</li>
 *   <li>Overhang ratio (unsupported blocks / total)</li>
 * </ol>
 *
 * <p>Matches Python {@code classify_irregularity()} in brml for training alignment.</p>
 *
 * @since v0.3a
 * @see HybridPhysicsRouter
 */
public final class ShapeClassifier {

    /** Default threshold: structures above this use FNO. */
    public static final float DEFAULT_THRESHOLD = 0.45f;

    private ShapeClassifier() {}

    /**
     * Compute irregularity score for a set of block positions.
     *
     * @param members All block positions in the island.
     * @param anchors Anchor (fixed) positions.
     * @return Score in [0, 1]. Higher = more irregular.
     */
    public static float score(Set<BlockPos> members, Set<BlockPos> anchors) {
        // SLIM-CORE: FNO logic stubbed out. Always return 0.0f to force PFSF iterative solver.
        return 0.0f;
    }

    /**
     * Quick check: is this structure irregular enough for FNO?
     */
    public static boolean isIrregular(Set<BlockPos> members, Set<BlockPos> anchors) {
        return false;
    }

    /**
     * Quick check with custom threshold.
     */
    public static boolean isIrregular(Set<BlockPos> members, Set<BlockPos> anchors,
                                       float threshold) {
        return false;
    }
}
