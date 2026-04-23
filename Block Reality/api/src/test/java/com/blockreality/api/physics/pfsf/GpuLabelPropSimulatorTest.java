package com.blockreality.api.physics.pfsf;

import net.minecraft.core.BlockPos;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Correctness harness for {@link PFSFLabelPropCpuSimulator}, the
 * bit-accurate reproduction of the Vulkan label-propagation pipeline.
 *
 * <p>Two layers of assertion:
 * <ol>
 *   <li>The per-voxel islandId array produced by the simulator labels
 *       exactly the same connected components as
 *       {@link LabelPropagation#bfsComponents} and
 *       {@link LabelPropagation#shiloachVishkin}.</li>
 *   <li>The component-meta records produced by the summarise kernels
 *       (block count, anchored flag, AABB min/max) match the values
 *       obtained by re-scanning the domain independently.</li>
 * </ol>
 *
 * <p>If the real GPU kernels are ever observed to disagree with this
 * simulator under identical input, the divergence is a GPU driver bug
 * or an unintended SSBO race — both cases the simulator will help
 * triage since it ships the "should-be" numbers alongside the actual
 * readback.
 */
public class GpuLabelPropSimulatorTest {

    @Test
    @DisplayName("simulator partitions ≡ bfsComponents on 100 random FULL_26 domains")
    public void simulatorEquivalentToBfs() {
        Random rng = new Random(20260101L);
        int trials = 100;
        for (int trial = 0; trial < trials; trial++) {
            int Lx = 4 + rng.nextInt(7);
            int Ly = 4 + rng.nextInt(7);
            int Lz = 4 + rng.nextInt(7);
            byte[] type = new byte[Lx * Ly * Lz];
            Set<BlockPos> members = new HashSet<>();
            Set<BlockPos> anchors = new HashSet<>();
            for (int i = 0; i < type.length; i++) {
                double r = rng.nextDouble();
                if (r < 0.55) type[i] = LabelPropagation.TYPE_SOLID;
                else if (r < 0.65) type[i] = LabelPropagation.TYPE_ANCHOR;
                else type[i] = LabelPropagation.TYPE_AIR;
                if (type[i] != LabelPropagation.TYPE_AIR) {
                    int x = i % Lx, rem = i / Lx, y = rem % Ly, z = rem / Ly;
                    members.add(new BlockPos(x, y, z));
                    if (type[i] == LabelPropagation.TYPE_ANCHOR) {
                        anchors.add(new BlockPos(x, y, z));
                    }
                }
            }

            PFSFLabelPropCpuSimulator.SimulatorResult gpu =
                    PFSFLabelPropCpuSimulator.run(type, Lx, Ly, Lz);
            LabelPropagation.PartitionResult bfs = LabelPropagation.bfsComponents(
                    members, anchors, LabelPropagation.NeighborPolicy.FULL_26);

            // Build keyed partition from simulator output for comparison.
            Map<Integer, Set<BlockPos>> simGroups = new HashMap<>();
            for (int i = 0; i < gpu.islandId().length; i++) {
                int lbl = gpu.islandId()[i];
                if (lbl == PFSFLabelPropCpuSimulator.NO_ISLAND) continue;
                int x = i % Lx, rem = i / Lx, y = rem % Ly, z = rem / Ly;
                simGroups.computeIfAbsent(lbl, k -> new HashSet<>()).add(new BlockPos(x, y, z));
            }
            assertEquals(bfs.components().size(), simGroups.size(),
                    "trial=" + trial + " component count mismatch: simulator="
                            + simGroups.size() + " bfs=" + bfs.components().size());

            // Each simulator group must match some BFS component exactly.
            for (Set<BlockPos> simMembers : simGroups.values()) {
                boolean found = false;
                for (LabelPropagation.Component c : bfs.components()) {
                    if (c.members().equals(simMembers)) { found = true; break; }
                }
                assertTrue(found, "trial=" + trial + " simulator group has no BFS match: " + simMembers);
            }
        }
    }

    @Test
    @DisplayName("component-meta records (count, anchored, AABB) match independent re-scan")
    public void summariseRecordsAgreeWithRescan() {
        Random rng = new Random(20260102L);
        int trials = 50;
        for (int trial = 0; trial < trials; trial++) {
            int L = 6 + rng.nextInt(6);
            byte[] type = new byte[L * L * L];
            for (int i = 0; i < type.length; i++) {
                double r = rng.nextDouble();
                type[i] = (byte) (r < 0.5 ? LabelPropagation.TYPE_SOLID
                        : r < 0.6 ? LabelPropagation.TYPE_ANCHOR
                        : LabelPropagation.TYPE_AIR);
            }
            PFSFLabelPropCpuSimulator.SimulatorResult gpu =
                    PFSFLabelPropCpuSimulator.run(type, L, L, L);
            // For each component slot, independently scan voxels with
            // islandId == rootLabel to recover count / anchored / AABB
            // and compare to the simulator's record.
            for (PFSFLabelPropCpuSimulator.ComponentMeta meta : gpu.components()) {
                int count = 0;
                boolean anchored = false;
                int minX = Integer.MAX_VALUE, minY = Integer.MAX_VALUE, minZ = Integer.MAX_VALUE;
                int maxX = 0, maxY = 0, maxZ = 0;
                for (int i = 0; i < gpu.islandId().length; i++) {
                    if (gpu.islandId()[i] != meta.rootLabel()) continue;
                    count++;
                    if (type[i] == LabelPropagation.TYPE_ANCHOR) anchored = true;
                    int x = i % L, rem = i / L, y = rem % L, z = rem / L;
                    minX = Math.min(minX, x); minY = Math.min(minY, y); minZ = Math.min(minZ, z);
                    maxX = Math.max(maxX, x); maxY = Math.max(maxY, y); maxZ = Math.max(maxZ, z);
                }
                assertEquals(count, meta.blockCount(),
                        "trial=" + trial + " root=" + meta.rootLabel() + " blockCount mismatch");
                assertEquals(anchored, meta.anchored(),
                        "trial=" + trial + " root=" + meta.rootLabel() + " anchored flag mismatch");
                assertEquals(minX, meta.aabbMinX(), "aabbMin.x mismatch for root=" + meta.rootLabel());
                assertEquals(minY, meta.aabbMinY(), "aabbMin.y mismatch for root=" + meta.rootLabel());
                assertEquals(minZ, meta.aabbMinZ(), "aabbMin.z mismatch for root=" + meta.rootLabel());
                assertEquals(maxX, meta.aabbMaxX(), "aabbMax.x mismatch for root=" + meta.rootLabel());
                assertEquals(maxY, meta.aabbMaxY(), "aabbMax.y mismatch for root=" + meta.rootLabel());
                assertEquals(maxZ, meta.aabbMaxZ(), "aabbMax.z mismatch for root=" + meta.rootLabel());
            }
        }
    }

    @Test
    @DisplayName("simulator is deterministic under repeated identical input")
    public void simulatorIsDeterministic() {
        int L = 10;
        byte[] type = new byte[L * L * L];
        Random rng = new Random(20260103L);
        for (int i = 0; i < type.length; i++) {
            double r = rng.nextDouble();
            type[i] = (byte) (r < 0.5 ? LabelPropagation.TYPE_SOLID
                    : r < 0.6 ? LabelPropagation.TYPE_ANCHOR
                    : LabelPropagation.TYPE_AIR);
        }
        PFSFLabelPropCpuSimulator.SimulatorResult first =
                PFSFLabelPropCpuSimulator.run(type, L, L, L);
        for (int run = 1; run < 10; run++) {
            PFSFLabelPropCpuSimulator.SimulatorResult other =
                    PFSFLabelPropCpuSimulator.run(type, L, L, L);
            assertArrayEquals(first.islandId(), other.islandId(),
                    "islandId differs on run " + run);
            assertEquals(first.numComponents(), other.numComponents());
            assertEquals(first.overflow(), other.overflow());
        }
    }

    @Test
    @DisplayName("overflow flag triggers when unique components exceed MAX_COMPONENTS")
    public void overflowFlagOnExcessComponents() {
        // Build a domain with > MAX_COMPONENTS isolated SOLID voxels (each
        // surrounded by AIR ⇒ each is its own component).
        int L = 10;  // 10³ = 1000 voxels, enough for > 64 isolated components
        byte[] type = new byte[L * L * L];
        int target = PFSFLabelPropCpuSimulator.MAX_COMPONENTS + 8;
        int placed = 0;
        // Place isolated SOLID voxels on positions where no neighbour is set.
        for (int i = 0; i < type.length && placed < target; i += 3) {
            // Skip every 3rd voxel so none are adjacent under 26-conn.
            if (placed * 3 + 2 >= type.length) break;
            type[placed * 3] = LabelPropagation.TYPE_SOLID;
            placed++;
        }
        // Force them isolated by resetting any neighbours of the solids to AIR
        // (the stride-3 loop along flat index is not guaranteed to avoid
        // 3D adjacency). Safer: use a dedicated sparse placement that is
        // spatially isolated — every 3rd voxel along x with y=0, z=0 only.
        java.util.Arrays.fill(type, (byte) 0);
        int cnt = 0;
        for (int x = 0; x < L && cnt < target; x += 2) {
            type[x] = LabelPropagation.TYPE_SOLID; cnt++;
        }
        // The x-axis has 5 slots (0,2,4,6,8) at L=10 — not enough. Use 3D.
        java.util.Arrays.fill(type, (byte) 0);
        cnt = 0;
        outer:
        for (int z = 0; z < L; z += 2) {
            for (int y = 0; y < L; y += 2) {
                for (int x = 0; x < L; x += 2) {
                    int i = x + L * (y + L * z);
                    type[i] = LabelPropagation.TYPE_SOLID;
                    if (++cnt >= target) break outer;
                }
            }
        }

        PFSFLabelPropCpuSimulator.SimulatorResult r =
                PFSFLabelPropCpuSimulator.run(type, L, L, L);
        assertTrue(r.overflow(),
                "with " + cnt + " isolated components (> MAX_COMPONENTS=" + PFSFLabelPropCpuSimulator.MAX_COMPONENTS + "), overflow flag should be set");
        assertEquals(PFSFLabelPropCpuSimulator.MAX_COMPONENTS, r.numComponents(),
                "numComponents is clamped to MAX_COMPONENTS on overflow");
    }
}
