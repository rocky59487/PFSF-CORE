package com.blockreality.api.physics.validation;

import com.blockreality.api.physics.validation.VoxelPhysicsCpuReference.Domain;
import org.junit.jupiter.api.Test;

import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Phase F — 浮島 NO_SUPPORT 偵測驗證。
 *
 * <p>預期物理：一塊 solid cluster 與地面 anchor 無連通路徑 → 所有體素應被標記為
 * orphan（對應 GPU failure_scan 的 {@code FAIL_NO_SUPPORT}）。
 *
 * <h2>PASS 判準</h2>
 * <ol>
 *   <li>懸浮 5×5×5 cluster 的所有體素 BFS 不可達 → orphans.size() == 125</li>
 *   <li>BFS 不會誤把地面 anchor 視為 orphan</li>
 *   <li>一旦加一根「支柱」連到地面，orphan 消失</li>
 *   <li>接觸地面（oz=1）的 cluster 不是 orphan</li>
 * </ol>
 */
class FloatingIslandTest {

    @Test
    void fullyDetachedCluster_allBlocksAreOrphans() {
        // worldLx/Ly/Lz = 11, cluster 5×5×5 浮於 oz=2 上方
        Domain dom = VoxelPhysicsCpuReference.buildFloatingIsland(11, 11, 11, 5, 5, 5);
        Set<Integer> orphans = VoxelPhysicsCpuReference.findOrphans(dom);
        assertEquals(5 * 5 * 5, orphans.size(),
            "5³ 浮島應全部為 orphan：actual=" + orphans.size());
    }

    @Test
    void groundAnchorsNeverFlaggedAsOrphans() {
        Domain dom = VoxelPhysicsCpuReference.buildFloatingIsland(11, 11, 11, 5, 5, 5);
        Set<Integer> orphans = VoxelPhysicsCpuReference.findOrphans(dom);
        // 所有 type == ANCHOR 的 voxel 都不能在 orphan 集合
        for (int i = 0; i < dom.N(); i++) {
            if (dom.type[i] == VoxelPhysicsCpuReference.TYPE_ANCHOR) {
                assertFalse(orphans.contains(i),
                    "anchor i=" + i + " 不應被視為 orphan");
            }
        }
    }

    @Test
    void airVoxelsNeverFlaggedAsOrphans() {
        Domain dom = VoxelPhysicsCpuReference.buildFloatingIsland(11, 11, 11, 5, 5, 5);
        Set<Integer> orphans = VoxelPhysicsCpuReference.findOrphans(dom);
        for (int i = 0; i < dom.N(); i++) {
            if (dom.type[i] == VoxelPhysicsCpuReference.TYPE_AIR) {
                assertFalse(orphans.contains(i),
                    "air voxel i=" + i + " 不應被視為 orphan");
            }
        }
    }

    @Test
    void clusterTouchingGround_noOrphans() {
        // 直接建一個從 z=0 anchor 延伸到 z=5 的柱子 → 全連通
        int Lx = 7, Ly = 7, Lz = 7;
        byte[] type = new byte[Lx * Ly * Lz];
        float[] sigma = new float[Lx * Ly * Lz];
        float[] source = new float[Lx * Ly * Lz];
        int cx = Lx / 2, cy = Ly / 2;

        // 地面全 anchor（底層）
        for (int y = 0; y < Ly; y++)
            for (int x = 0; x < Lx; x++) {
                int i = x + Lx * (y + Ly * 0);
                type[i] = VoxelPhysicsCpuReference.TYPE_ANCHOR;
                sigma[i] = 1f;
            }
        // 中心柱 z=1..5
        for (int z = 1; z < 6; z++) {
            int i = cx + Lx * (cy + Ly * z);
            type[i] = VoxelPhysicsCpuReference.TYPE_SOLID;
            sigma[i] = 1f;
            source[i] = 1f;
        }
        Domain dom = new Domain(Lx, Ly, Lz, type, sigma, source);
        Set<Integer> orphans = VoxelPhysicsCpuReference.findOrphans(dom);
        assertEquals(0, orphans.size(), "貼地柱子不應有 orphan");
    }

    @Test
    void addingSupportColumn_removesOrphans() {
        // Step 1：浮島
        Domain floating = VoxelPhysicsCpuReference.buildFloatingIsland(11, 11, 11, 5, 5, 5);
        Set<Integer> orphans1 = VoxelPhysicsCpuReference.findOrphans(floating);
        assertEquals(125, orphans1.size());

        // Step 2：在 cluster 中心柱填補從 z=0 (ground) 到 cluster 底部 z=2 的連線
        // cluster oz=2，所以需要填 z=1 一層貫穿到 cluster
        byte[] type = floating.type().clone();
        float[] sigma = floating.sigma().clone();
        float[] source = floating.source().clone();
        int cx = 11 / 2;  // cluster 中心
        int cy = 11 / 2;
        // 填 z=1 一個 solid 連到 cluster (oz=2) — ground 的 z=0 已是 ANCHOR，z=2 是 SOLID
        int i = cx + 11 * (cy + 11 * 1);
        type[i] = VoxelPhysicsCpuReference.TYPE_SOLID;
        sigma[i] = 1f;
        source[i] = 1f;

        Domain connected = new Domain(11, 11, 11, type, sigma, source);
        Set<Integer> orphans2 = VoxelPhysicsCpuReference.findOrphans(connected);
        assertEquals(0, orphans2.size(),
            "加入支柱後不應有 orphan，actual=" + orphans2.size());
    }

    @Test
    void detachedSubcluster_onlyThatClusterIsOrphan() {
        // 8 格地面 anchor + 連著 2 格柱子 + 另外一個 3 格的完全獨立 cluster
        int Lx = 8, Ly = 8, Lz = 6;
        byte[] type = new byte[Lx * Ly * Lz];
        float[] sigma = new float[Lx * Ly * Lz];
        float[] source = new float[Lx * Ly * Lz];

        // Ground z=0
        for (int y = 0; y < Ly; y++)
            for (int x = 0; x < Lx; x++) {
                int i = x + Lx * (y + Ly * 0);
                type[i] = VoxelPhysicsCpuReference.TYPE_ANCHOR;
                sigma[i] = 1f;
            }
        // Connected pillar (x=1, y=1, z=1,2)
        for (int z = 1; z <= 2; z++) {
            int i = 1 + Lx * (1 + Ly * z);
            type[i] = VoxelPhysicsCpuReference.TYPE_SOLID;
            sigma[i] = 1f; source[i] = 1f;
        }
        // Detached cluster: (x=6, y=6, z=3..5) 三個體素
        for (int z = 3; z <= 5; z++) {
            int i = 6 + Lx * (6 + Ly * z);
            type[i] = VoxelPhysicsCpuReference.TYPE_SOLID;
            sigma[i] = 1f; source[i] = 1f;
        }
        Domain dom = new Domain(Lx, Ly, Lz, type, sigma, source);

        Set<Integer> orphans = VoxelPhysicsCpuReference.findOrphans(dom);
        assertEquals(3, orphans.size(), "只有 3-voxel 獨立 cluster 應為 orphan");
    }
}
