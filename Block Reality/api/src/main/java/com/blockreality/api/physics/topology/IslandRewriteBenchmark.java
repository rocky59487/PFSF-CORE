package com.blockreality.api.physics.topology;

import com.blockreality.api.physics.pfsf.LabelPropagation;
import net.minecraft.core.BlockPos;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/**
 * R.9 — benchmark + ablation harness for the three-tier rewrite. The
 * intent is to produce the paper's evaluation table: average and max
 * tick time across domain sizes, for the full hybrid and each
 * individual tier in isolation.
 *
 * <p>Each ablation measures the cost of answering the SAME question —
 * "which voxels are orphan this tick?" — using only one tier's
 * machinery. This exposes what the hybrid actually buys:
 * <ul>
 *   <li><b>Tier 1 only</b>: AABB dirty tracking with no content
 *       decision. Useful to measure the bare SVDAG cost; not a real
 *       solver.</li>
 *   <li><b>Tier 2 only</b>: full-domain Poisson oracle every tick. No
 *       incremental benefit from AABB locality; no identity.</li>
 *   <li><b>Tier 3 only</b>: full-domain BFS every tick to derive
 *       components, then tracker update. No AABB locality, no PDE;
 *       the current-tick Union-Find-style baseline.</li>
 *   <li><b>Full hybrid</b>: Tier 1 dirty region narrows the Tier 2
 *       work to small AABBs, and Tier 3 maintains identity across
 *       splits/merges.</li>
 * </ul>
 *
 * <p>Benchmark run:
 * <pre>
 *   cd "Block Reality"
 *   ./gradlew :api:test --tests "com.blockreality.api.physics.topology.IslandRewriteBenchmark"
 *   # or standalone
 *   java ...IslandRewriteBenchmark 128
 * </pre>
 * Output is a Markdown table printed to stdout so it can be pasted into
 * the paper's evaluation section directly.
 */
public final class IslandRewriteBenchmark {

    public static final int[] DEFAULT_SIZES = {16, 32, 64};
    public static final int DEFAULT_FRACTURES = 50;

    public enum Ablation { FULL_HYBRID, TIER1_ONLY, TIER2_ONLY, TIER3_ONLY }

    /** One row of the evaluation table. */
    public record Row(
            Ablation ablation,
            int domainSize,
            double avgTickMs,
            double maxTickMs,
            int totalOrphansDetected,
            boolean correctness
    ) {}

    private IslandRewriteBenchmark() {}

    public static List<Row> runAll() { return runAll(DEFAULT_SIZES, DEFAULT_FRACTURES, 42L); }

    public static List<Row> runAll(int[] sizes, int fractures, long seed) {
        List<Row> rows = new ArrayList<>();
        for (int s : sizes) {
            for (Ablation a : Ablation.values()) {
                rows.add(runOne(a, s, fractures, seed));
            }
        }
        return rows;
    }

    public static Row runOne(Ablation ablation, int domainSize, int fractures, long seed) {
        Scenario sc = buildScenario(domainSize, fractures, seed);
        return switch (ablation) {
            case FULL_HYBRID -> measureFullHybrid(sc);
            case TIER1_ONLY  -> measureTier1Only(sc);
            case TIER2_ONLY  -> measureTier2Only(sc);
            case TIER3_ONLY  -> measureTier3Only(sc);
        };
    }

    // ─────────────────────────────────────────────────────────────────
    //  Scenario: an anchored lattice structure with a scripted stream
    //  of fracture events (remove voxels one by one).
    // ─────────────────────────────────────────────────────────────────

    private record Scenario(
            int size,
            byte[] initialVoxels,
            List<int[]> fractures
    ) {}

    private static Scenario buildScenario(int size, int fractures, long seed) {
        byte[] voxels = new byte[size * size * size];
        // Fill a plus-shape across the middle plane — guarantees several
        // anchored paths that can genuinely fracture.
        int mid = size / 2;
        for (int x = 0; x < size; x++) voxels[flat(x, mid, mid, size)] = TopologicalSVDAG.TYPE_SOLID;
        for (int y = 0; y < size; y++) voxels[flat(mid, y, mid, size)] = TopologicalSVDAG.TYPE_SOLID;
        for (int z = 0; z < size; z++) voxels[flat(mid, mid, z, size)] = TopologicalSVDAG.TYPE_SOLID;
        // Anchor the two ends of the x-axis bar.
        voxels[flat(0,        mid, mid, size)] = TopologicalSVDAG.TYPE_ANCHOR;
        voxels[flat(size - 1, mid, mid, size)] = TopologicalSVDAG.TYPE_ANCHOR;

        Random rng = new Random(seed);
        List<int[]> events = new ArrayList<>();
        for (int i = 0; i < fractures; i++) {
            int axis = rng.nextInt(3);
            int pos  = 1 + rng.nextInt(size - 2);
            int[] xyz = switch (axis) {
                case 0 -> new int[]{pos, mid, mid};
                case 1 -> new int[]{mid, pos, mid};
                default -> new int[]{mid, mid, pos};
            };
            events.add(xyz);
        }
        return new Scenario(size, voxels, events);
    }

    // ─────────────────────────────────────────────────────────────────
    //  Ablation measurements
    // ─────────────────────────────────────────────────────────────────

    private static Row measureFullHybrid(Scenario sc) {
        ThreeTierOrchestrator o = new ThreeTierOrchestrator();
        applyInitial(o, sc);
        o.tick();
        double[] tickTimes = new double[sc.fractures.size()];
        int totalOrphans = 0;
        for (int i = 0; i < sc.fractures.size(); i++) {
            int[] xyz = sc.fractures.get(i);
            o.setVoxel(xyz[0], xyz[1], xyz[2], TopologicalSVDAG.TYPE_AIR);
            long t0 = System.nanoTime();
            ThreeTierOrchestrator.TickResult r = o.tick();
            long t1 = System.nanoTime();
            tickTimes[i] = (t1 - t0) / 1_000_000.0;
            totalOrphans += r.orphanEvents().size();
        }
        return summarise(Ablation.FULL_HYBRID, sc.size, tickTimes, totalOrphans, true);
    }

    private static Row measureTier1Only(Scenario sc) {
        TopologicalSVDAG svdag = new TopologicalSVDAG();
        for (int i = 0; i < sc.initialVoxels.length; i++) {
            byte t = sc.initialVoxels[i];
            if (t == TopologicalSVDAG.TYPE_AIR) continue;
            int x = i % sc.size, rem = i / sc.size, y = rem % sc.size, z = rem / sc.size;
            svdag.setVoxel(x, y, z, t);
        }
        svdag.drainDirtyRegions();
        double[] tickTimes = new double[sc.fractures.size()];
        for (int i = 0; i < sc.fractures.size(); i++) {
            int[] xyz = sc.fractures.get(i);
            svdag.setVoxel(xyz[0], xyz[1], xyz[2], TopologicalSVDAG.TYPE_AIR);
            long t0 = System.nanoTime();
            svdag.drainDirtyRegions();
            long t1 = System.nanoTime();
            tickTimes[i] = (t1 - t0) / 1_000_000.0;
        }
        // Tier 1 alone cannot produce orphan events.
        return summarise(Ablation.TIER1_ONLY, sc.size, tickTimes, 0, false);
    }

    private static Row measureTier2Only(Scenario sc) {
        byte[] voxels = sc.initialVoxels.clone();
        double[] tickTimes = new double[sc.fractures.size()];
        int totalOrphans = 0;
        for (int i = 0; i < sc.fractures.size(); i++) {
            int[] xyz = sc.fractures.get(i);
            voxels[flat(xyz[0], xyz[1], xyz[2], sc.size)] = TopologicalSVDAG.TYPE_AIR;
            long t0 = System.nanoTime();
            PoissonOracleCPU.Result r = PoissonOracleCPU.solve(voxels, sc.size, sc.size, sc.size,
                    Math.min(64, sc.size * 4), PoissonOracleCPU.DEFAULT_EPSILON);
            long t1 = System.nanoTime();
            tickTimes[i] = (t1 - t0) / 1_000_000.0;
            for (boolean b : r.fractureMask()) if (b) totalOrphans++;
        }
        return summarise(Ablation.TIER2_ONLY, sc.size, tickTimes, totalOrphans, true);
    }

    private static Row measureTier3Only(Scenario sc) {
        byte[] voxels = sc.initialVoxels.clone();
        PersistentIslandTracker tracker = new PersistentIslandTracker();
        double[] tickTimes = new double[sc.fractures.size()];
        int totalOrphans = 0;
        for (int i = 0; i < sc.fractures.size(); i++) {
            int[] xyz = sc.fractures.get(i);
            voxels[flat(xyz[0], xyz[1], xyz[2], sc.size)] = TopologicalSVDAG.TYPE_AIR;
            long t0 = System.nanoTime();
            Set<BlockPos> members = new HashSet<>();
            Set<BlockPos> anchors = new HashSet<>();
            for (int k = 0; k < voxels.length; k++) {
                byte t = voxels[k];
                if (t == TopologicalSVDAG.TYPE_AIR) continue;
                int x = k % sc.size, rem = k / sc.size, y = rem % sc.size, z = rem / sc.size;
                BlockPos p = new BlockPos(x, y, z);
                members.add(p);
                if (t == TopologicalSVDAG.TYPE_ANCHOR) anchors.add(p);
            }
            LabelPropagation.PartitionResult part = LabelPropagation.bfsComponents(
                    members, anchors, LabelPropagation.NeighborPolicy.FULL_26);
            List<Set<Long>> comps = new ArrayList<>();
            for (LabelPropagation.Component c : part.components()) {
                Set<Long> enc = new HashSet<>();
                for (BlockPos p : c.members()) enc.add(PersistentIslandTracker.encode(p.getX(), p.getY(), p.getZ()));
                comps.add(enc);
                if (!c.anchored()) totalOrphans++;
            }
            tracker.update(comps);
            long t1 = System.nanoTime();
            tickTimes[i] = (t1 - t0) / 1_000_000.0;
        }
        return summarise(Ablation.TIER3_ONLY, sc.size, tickTimes, totalOrphans, true);
    }

    private static Row summarise(Ablation a, int size, double[] ticks, int totalOrphans, boolean correct) {
        double sum = 0, max = 0;
        for (double t : ticks) { sum += t; if (t > max) max = t; }
        double avg = sum / Math.max(1, ticks.length);
        return new Row(a, size, avg, max, totalOrphans, correct);
    }

    private static void applyInitial(ThreeTierOrchestrator o, Scenario sc) {
        for (int i = 0; i < sc.initialVoxels.length; i++) {
            byte t = sc.initialVoxels[i];
            if (t == TopologicalSVDAG.TYPE_AIR) continue;
            int x = i % sc.size, rem = i / sc.size, y = rem % sc.size, z = rem / sc.size;
            o.setVoxel(x, y, z, t);
        }
    }

    private static int flat(int x, int y, int z, int s) { return x + s * (y + s * z); }

    // ─────────────────────────────────────────────────────────────────
    //  Markdown pretty-print for paper insertion
    // ─────────────────────────────────────────────────────────────────

    public static String toMarkdown(List<Row> rows) {
        StringBuilder sb = new StringBuilder();
        sb.append("| Ablation | Domain | avg ms/tick | max ms/tick | orphans | correct? |\n");
        sb.append("|---|---|---|---|---|---|\n");
        for (Row r : rows) {
            sb.append(String.format("| %s | %d³ | %.3f | %.3f | %d | %s |%n",
                    r.ablation(), r.domainSize(), r.avgTickMs(), r.maxTickMs(),
                    r.totalOrphansDetected(), r.correctness() ? "✓" : "(partial)"));
        }
        return sb.toString();
    }

    public static void main(String[] args) {
        int[] sizes = DEFAULT_SIZES;
        if (args.length > 0) {
            sizes = new int[args.length];
            for (int i = 0; i < args.length; i++) sizes[i] = Integer.parseInt(args[i]);
        }
        List<Row> rows = runAll(sizes, DEFAULT_FRACTURES, 42L);
        System.out.println(toMarkdown(rows));
    }
}
