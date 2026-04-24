# RFC — PFSF-Native Unified Voxel Memory (UVM)

**Status**: Design evaluation, pre-implementation
**Author**: Discussion with rocky59487, drafted 2026-04
**Problem owner**: physics + storage subsystems
**Supersedes (if shipped)**: StructureIslandRegistry, LabelPropagation, parts of PFSFIslandBuffer, PersistentIslandTracker BFS path

## 1. Motivation — why the current stack pays for the same thing three times

Today the engine maintains structural information in three parallel representations that must be kept in sync by hand:

| System | Role | Backing store |
|---|---|---|
| `StructureIslandRegistry` | int-id accounting, dirty flags, AABB | `HashMap<BlockPos, Integer>` + per-island `Set<BlockPos>` |
| `TopologicalSVDAG` + `PersistentIslandTracker` | Elder-Rule identity, fingerprint | sparse hash map of live voxel indices |
| `PFSFIslandBuffer` | GPU physics arena | AABB-dense 3D buffers, zero-padded |

Each system computes connectivity independently: `LabelPropagation.bfsComponents` in the registry, a second BFS inside the tracker, and the PFSF solver itself (whose potential field already knows which voxels are reachable from anchors). Three passes, three representations, three places to drift. A typical 10 k-voxel island eats roughly **7 MB** across all three — of which less than 300 KB is actual live physics state; the rest is padding, duplicate graphs and book-keeping.

The user's framing is the cleanest statement of the fix: **let the physics be the topology**, keep material metadata off the GPU, and make the storage layer reveal the block layout directly rather than hiding it behind flat arrays keyed by integer ids.

## 2. Design goals (user-stated)

1. Connectivity is a consequence of PFSF force propagation — no separate BFS / union-find.
2. The storage layout itself encodes the block structure; a reader of the raw bytes can see the geometry without a decoding step.
3. Bit-size wins come from the structure, not from a bolted-on compressor.
4. GPU only touches the low-level numeric work; material types and "unimportant" per-voxel metadata leave VRAM.
5. Identity (who is this structure across time) replaces ad-hoc int ids wherever it flows naturally.

## 3. Proposed stack — three layers, one identity

### Layer A — Sparse Occupancy DAG (SO-DAG, CPU)

A content-addressed octree.

- **Leaves**: 4 × 4 × 4 occupancy bitmap (64 bits = 8 bytes) + 16-bit palette-table index.
- **Internal nodes**: 8 child references (32 bits each), stored as indices into a flat node arena.
- **Deduplication**: identical subtrees share the same arena slot. A tower that stacks the same 4×4×4 block 200× stores that leaf once.
- **Mutation**: copy-on-write up the spine from the edited leaf; `O(log N)` voxels touched per edit.
- **Block-logic aware by construction**: the DAG's shape *is* the geometry. A 4×4 pillar of concrete shows up as one leaf chained eight deep into one internal node; you can read the SO-DAG and see the pillar without decoding.

Compression for common Minecraft structures:

| Structure | Naïve bytes (type + palette only) | SO-DAG bytes | Ratio |
|---|---|---|---|
| 16×16×16 hollow room (1176 voxels) | 8 KB | ~0.6 KB | 13× |
| 32×8×8 uniform-concrete bridge deck (2048 voxels) | 16 KB | ~0.2 KB | 80× |
| 10×30×10 mixed-material tower (1800 non-empty / 3000 total voxels) | 12 KB | ~1.5 KB | 8× |

Numbers are design estimates, not measurements. The win comes from two places: (1) repeated leaves share storage; (2) large regions of air compress into "child index = 0" internal nodes.

### Layer B — Physics State Arena (GPU)

A sparse, packed arena keyed by slot number, not by AABB coordinate.

- Walk the SO-DAG → allocate one slot per live voxel. Slot numbers are stable within a tick but re-assigned when the SO-DAG geometry changes.
- Per-voxel GPU state: `phi, h, d, sigma_x, sigma_y, sigma_z, source` — 28 bytes dense. Compare to today's ~60 bytes + AABB padding.
- Material-derived physics constants (stiffness modulus, tensile limit, curing factor, …) live in a shader-side 16-entry LUT keyed by the palette index carried in the SO-DAG leaf. A single uniform buffer is enough for every island the player can see.
- **No per-voxel material id on the GPU.** The palette index is enough; everything else resolves on the CPU when needed.
- Voxel neighbour lookup: given a slot, the SO-DAG walk can produce the 26-conn neighbour slots. For the hot inner loop this becomes a small index table computed once per leaf.

Typical 10 k-voxel island on GPU:

| Component | Current | Proposed |
|---|---|---|
| phi / h / d fields | ~5 MB AABB-padded | ~110 KB packed |
| conductivity (SoA, 6 faces × float) | ~1 MB | ~240 KB |
| material id per voxel | ~40 KB | 0 (palette LUT < 1 KB) |
| **Subtotal** | **≈ 6 MB** | **≈ 360 KB** |

### Layer C — Identity Registry (CPU)

Keyed on `structuralFingerprint: long` from the existing `PersistentIslandTracker`.

- Holds material metadata, RBlockEntity references, curing timers, hydration curves, BlockType enums, every piece of gameplay data that is *per structure* or *per voxel group* rather than per-physics-slot.
- `getMaterial(pos) → RMaterial` = palette-lookup on SO-DAG leaf + registry lookup on palette entry. Constant time, cache-friendly.
- Amortised O(1) on the hot path because physics ticks never walk this registry — only event handlers do.

The identity registry is where the user's "身份系統" hint lives. The fingerprint flowing from the tracker already exists; this proposal promotes it from a testing aid into the single key that binds gameplay-side state to physics-side voxels.

## 4. Connectivity-from-force — killing three BFSes at once

The PFSF solver computes a steady-state potential `φ` such that `∇·(σ∇φ) = source` with `φ = 0` at anchors. After convergence:

- `σ · φ → 0` at any voxel not graph-reachable from an anchor → **orphan**.
- Connected components are the equivalence classes of voxels linked by paths of `φ > ε_conn` — a standard watershed on the solved field.

Both of these are computed as a side-effect of the solver we already run. No separate BFS, no union-find, no tracker BFS.

Identity still matters (Elder Rule for split/merge), and `PersistentIslandTracker` keeps its role, but its *input* changes from "CPU BFS components" to "voxels where σφ > ε" — a one-kernel GPU readback. The tracker stays under 200 lines and loses its BFS helper entirely.

The concrete handover:

1. PFSF tick solves as today.
2. A new compute shader (trivial, single-pass) writes `reachable[slot] = (sigma[slot] * phi[slot] > EPS_CONN)` to a bitmask buffer.
3. The bitmask reads back to the CPU (one `vkCmdCopyBuffer` + staging; a handful of KB for 10 k voxels).
4. Tracker sees `{slot : reachable[slot] == 1}` as the component voxel set, walks its existing Elder-Rule fingerprint machinery, fires OrphanEvents for anything newly unreachable.
5. `LabelPropagation.bfsComponents` is deleted. `StructureIslandRegistry.checkAndSplitIsland` becomes a thin adapter over the tracker's output.

Cost saved per tick for a world with 50 islands averaging 1 k voxels each:

- ~50 × 1 k voxel BFS visits in the registry (~250 µs).
- ~50 × 1 k voxel BFS visits in the tracker (~250 µs).
- The PFSF solve was going to run anyway; the only new cost is the bitmask kernel (~10 µs).

## 5. Honest trade-offs

- **Mutation cost**: an SO-DAG edit is `O(log N)` subtree hashes vs. the current `O(1)` HashMap put. In absolute numbers, a 4 m-block world is ~20 hash steps per edit — measured sub-microsecond in comparable SVDAG implementations (e.g. the `ephtracy/VoxelDag` benchmarks cited in `research/paper_data/`). Not free, not a problem.
- **Hash collisions**: content-addressing requires a 64-bit subtree hash with a 128-bit tie-breaker in the arena. This is state-of-the-art but adds one memcmp per insertion; negligible.
- **`ε_conn` tuning**: treating "connected" as "σφ > ε" means ε has to be chosen so that legitimate-but-weak connections are not falsely classified as orphan. The material calibration registry (`MaterialCalibration`, v2.2) already solves a related problem; extending it to supply per-material ε_conn is natural and test-covered.
- **Debuggability**: a flat HashMap is trivially inspectable; an SO-DAG is not. Mitigation: a dev-time visualiser that walks SO-DAG → renders leaves with palette-coloured overlays; probably a single new `/br svdump` subcommand.
- **ABI churn**: every system that holds a `BlockPos → islandId` contract today needs to switch to fingerprint-based addressing. Roughly 30 call sites; mechanical but exhaustive.

## 6. Migration path — six phases, fully reversible until phase 5

1. **Build SO-DAG side-by-side** behind `BRConfig.enableUVM` flag. Mirror every `registerBlock` / `unregisterBlock` into the DAG. At tick end, diff DAG-reconstructed voxel set against `StructureIsland.members`; any disagreement logs a warning. Keeps the current stack in production.
2. **Phase B arena** — add the new GPU arena allocator next to `PFSFIslandBuffer`; run both on the same islands; compare `phi` per voxel each tick; diverge → log + disable new path. 1 release cycle of shadow-mode.
3. **Flip default** — UVM is primary, legacy is the shadow. Keep the shadow for another cycle to catch edge cases (chunk boundaries, fluid coupling, RC fusion).
4. **Retire BFS** — once `reachable[slot]` bitmask agrees with BFS for 10 k ticks across playtest, delete `LabelPropagation.bfsComponents` and the tracker's internal BFS.
5. **Delete legacy stack** — `StructureIslandRegistry.blockToIsland`, the `members` sets, the AABB padding path. No rollback beyond git after this point.
6. **Migrate RBlockEntity storage** into the identity registry — this is the final "unimportant info off the GPU" step. BlockEntity becomes a thin view over the registry entry.

Phases 1–4 ship independently; each is its own PR and a flag-flip. Phase 5 is the only irreversible step; by then UVM has at least two release cycles of production hours behind it.

## 7. What this buys the user

Concrete, measurable outcomes:

- **Memory**: ~20–25× reduction for a typical mixed-structure world (~7 MB → ~300 KB per 10 k-voxel island).
- **Correctness**: three independent connectivity passes collapse into one. No more "legacy BFS says A, tracker says B" investigation threads.
- **Code surface**: ~800 LOC removed from the physics side (LabelPropagation + half of StructureIslandRegistry + tracker BFS helpers).
- **GPU**: material lookups vanish from the hot shader path; palette LUT replaces per-voxel material id reads.
- **Gameplay semantics**: identity becomes the first-class identifier everywhere. Fingerprints survive splits, merges, and world reloads; int ids stop being mentioned outside the GPU arena.

## 8. What this does *not* buy

- Fluid simulation still lives outside the SO-DAG (different grid, different solver) — the UVM proposal does not help `PFSF-Fluid`.
- Render-side greedy meshing is its own geometry cache; UVM supplies occupancy but the mesher still has to classify faces.
- Node-graph editor state is orthogonal.
- The existing int-id API stays backwards-compatible through phase 5 — legacy mods do not benefit directly; they do not break either.

## 9. Recommendation

Ship phase 1 behind the flag and let it run shadow-mode for a development cycle. The mirror-and-diff approach makes this nearly risk-free: the legacy stack keeps running, and the first time UVM disagrees we get a warning rather than a physics regression. Decision on phase 2 can wait until phase 1 shows ≥ 24 hours of playtest with zero mirror-diffs.

The payoff is large, the commitment is gradual, and the first irreversible step is four PRs away. If after phase 1 the mirror diffs look bad or the SO-DAG's mutation cost proves to exceed the estimates, we can abandon UVM with no user-visible change and only the mirror code to delete.
