# RFC — PFSF-Native Structural Memory (PNSM)

**Status**: Design evaluation, v2 after review
**Previous revision**: `pfsf-native-unified-memory.md` (committed at `84829c1`, renamed + substantially revised by this document per review feedback)
**Alternative name considered**: Force-Bonded Voxel Memory (FBVM) — rejected for this revision because PNSM frames the proposal as a storage architecture first and a physics contract second, which matches the three-layer split below.
**Problem owner**: physics + storage subsystems
**Supersedes (if shipped)**: `StructureIslandRegistry`, `LabelPropagation`, parts of `PFSFIslandBuffer`, the CPU-side BFS helper inside `PersistentIslandTracker`

## 0. Core declaration

> Connectivity is derived from PFSF's active transmissible bond state, not from independent geometric BFS passes. Component labeling runs once on the GPU-side bond graph, and the CPU identity tracker consumes only the resulting labels and orphan events.

Every section below is an unpacking of this sentence. The previous revision's "connectivity-from-force" framing is abandoned: a scalar `σφ > ε` threshold cannot discriminate multiple separately-anchored components, unstable weak bridges, or symmetric potential plateaus, and should never have been offered as a sole oracle. Bond-graph connectivity — well-studied, correct under all the edge cases — is the right primitive.

## 1. Motivation — the current stack does the same work in three places

Today the engine maintains structural information in three parallel representations that must be kept in sync by hand:

| System | Role | Backing store |
|---|---|---|
| `StructureIslandRegistry` | int-id accounting, dirty flags, AABB | `HashMap<BlockPos, Integer>` + per-island `Set<BlockPos>` |
| `TopologicalSVDAG` + `PersistentIslandTracker` | Elder-Rule identity, fingerprint | sparse hash map of live voxel indices |
| `PFSFIslandBuffer` | GPU physics arena | AABB-dense 3D buffers, zero-padded |

Each system computes connectivity independently: `LabelPropagation.bfsComponents` in the registry, a second BFS inside the tracker, and the PFSF solver itself whose potential field already knows which voxels are reachable from anchors. Three geometric passes, three representations, three places to drift. A typical 10 k-voxel island eats roughly **7 MB** across all three — of which less than 300 KB is live physics state; the rest is padding, duplicate graphs and book-keeping.

The fix is to let the PFSF solver's own data — specifically the per-bond transmissibility state it already has to compute — be the single source of truth for connectivity, and to reshape the storage side so block geometry is self-describing rather than hidden behind flat arrays keyed by integer ids.

## 2. Design goals

1. **Connectivity is not a separate subsystem.** It is derived from the PFSF bond graph (active transmissible edges) via a single GPU component-labeling kernel.
2. **Block-logic aware storage.** A reader of the raw storage bytes can see the geometry without a decoding pass — repeated structural motifs (pillars, wall panels, deck sections) fold into shared subtrees.
3. **Bit-size wins come from the structure**, not from a bolted-on compressor.
4. **GPU holds only hot physics.** Per-voxel material ids, block-type enums, gameplay metadata and curing timers leave VRAM; physics-relevant material constants come from a tiny palette LUT.
5. **Identity is first-class.** Fingerprints from `PersistentIslandTracker` become the primary key that binds gameplay-side state to physics-side voxels, surviving splits / merges / world reloads. Int ids persist only inside the GPU arena.
6. **Slots are stable across local edits.** History-dependent fields (h-field, damage, failure events, debug replay) reference slots; slot renaming is deferred, batched, and announced.

## 3. Three-layer stack

### Layer A — Sparse Occupancy DAG (SO-DAG) — cold path only

A content-addressed octree.

- **Leaves**: 4 × 4 × 4 occupancy bitmap (64 bits = 8 bytes) + 16-bit palette-table index.
- **Internal nodes**: 8 child references (32 bits each), stored as indices into a flat node arena.
- **Deduplication**: identical subtrees share the same arena slot. A tower stacking the same 4×4×4 block 200× stores that leaf once.
- **Mutation**: copy-on-write up the spine from the edited leaf; `O(log N)` voxels touched per edit.
- **Block-logic aware by construction**: the DAG's shape *is* the geometry. A 4×4 pillar of concrete shows up as one leaf chained eight deep into one internal node; the storage and the structure coincide.

**Scope restriction — SO-DAG never enters the GPU hot loop.** Its purpose is:

- snapshots and world-save compression,
- fingerprint generation for `PersistentIslandTracker`,
- cold-path queries (block-type lookups, material metadata resolution, debug dumps),
- rebuild of the Layer-B packed arena when geometry changes.

Physics ticks walk the packed arena (Layer B), not the DAG. Any temptation to have a shader traverse the DAG is resolved here: it is not allowed.

Compression estimates for common Minecraft structures (design numbers, not measurements):

| Structure | Naïve bytes | SO-DAG bytes | Ratio |
|---|---|---|---|
| 16×16×16 hollow room (1176 voxels) | 8 KB | ~0.6 KB | 13× |
| 32×8×8 uniform-concrete bridge deck (2048 voxels) | 16 KB | ~0.2 KB | 80× |
| 10×30×10 mixed-material tower (1800 non-empty / 3000 total) | 12 KB | ~1.5 KB | 8× |

### Layer B — GPU Packed Arena — hot physics only

Flat, SoA-friendly arrays of exactly the physics quantities the solver touches:

```
cells[slot]          — phi, h, d, source        (7 floats)
ports[bond]          — face direction + two endpoint slots
bonds[bond]          — conductance, damage, bondActive
neighborIndex[slot]  — offsets into ports[] for each slot's 6 faces
events[]             — ring buffer of orphan / split / failure events
```

- Slot numbers are opaque `uint32` handles, stable across local edits (see §5).
- Material-derived physics constants (stiffness modulus, tensile limit, curing factor, …) live in a shader-side 16-entry LUT keyed by the palette index carried from Layer A.
- **No per-voxel material id on the GPU.** Palette index is enough; everything else resolves through Layer C when actually needed.
- `bondActive[bond]` is the single state variable the component-labeling kernel reads (§4).
- Neighbour lookup for the inner loop uses the pre-computed `neighborIndex` table, not SO-DAG traversal.

Typical 10 k-voxel island on GPU:

| Component | Current | Proposed |
|---|---|---|
| phi / h / d fields | ~5 MB AABB-padded | ~110 KB packed |
| conductivity (6 faces × float SoA) | ~1 MB | ~240 KB |
| material id per voxel | ~40 KB | 0 (palette LUT < 1 KB) |
| bond / port tables (new) | — | ~180 KB |
| **Subtotal** | **≈ 6 MB** | **≈ 530 KB** |

### Layer C — Identity Registry — CPU, off-GPU

Keyed on `structuralFingerprint: long` from the existing `PersistentIslandTracker`.

- Holds material metadata, `RBlockEntity` references, curing timers, hydration curves, `BlockType` enums — every piece of gameplay data that is per-structure or per-voxel-group rather than per-physics-slot.
- `getMaterial(pos) → RMaterial` resolves as palette-lookup on the SO-DAG leaf + registry-lookup on the palette entry. Amortised `O(1)`; cache-friendly because physics ticks never walk the registry — only event handlers do.
- The fingerprint flowing from the tracker already exists; this layer promotes it from a testing aid into the single key binding gameplay state to physics state.

## 4. Connectivity-from-transmission

Replace the three parallel BFS passes with one GPU-side component labeling pass over the active-bond graph.

### 4.1 What the PFSF solver contributes

At the end of each solve, the solver has already computed — or can trivially derive — a boolean `bondActive[bond]` for every bond in the island:

```
bondActive[bond] = (conductance[bond] * (1 - damage[bond])) > ε_bond
```

`conductance` and `damage` are both already in the GPU arena (they drive the stress solve). `ε_bond` is a per-material threshold seeded from the `MaterialCalibration` registry — same mechanism we already use for calibrating `maxPhi`. No new tunable lands in a vacuum.

### 4.2 Component labeling kernel

A single GPU pass running standard parallel connected-component labeling (Stopar–Kofler or Playne–Hawick variant — both map well onto Vulkan compute and are already partially implemented in `PFSFLabelPropRecorder`, `PFSFLabelPropCpuSimulator`, and the Phase-B.2 pipeline landed at `1c56149` + `cbf7e8b`). Output is `label[slot] → uint32` where anchored components get low ids and isolated components get synthetic ids.

### 4.3 Why this is correct where `σφ > ε` is not

The scalar-field threshold abandoned by the previous revision fails in four predictable ways:

- **Multiple anchors** — all anchored components look identical under a `σφ` test; it cannot discriminate them.
- **Weak connections** — a bond close to `ε` toggles between connected and orphan tick-to-tick, producing false events.
- **Symmetric potential fields** — two equal-strength sources create plateaus where the threshold has no information.
- **Multiple separately-grounded structures** — each is anchored and reaches `φ = 0`; a threshold cannot tell them apart.

Bond-graph labeling is a topological algorithm on a boolean graph. It answers the question correctly for all four cases, and it is what the codebase has been quietly building toward since the label-prop work in `1c56149`.

### 4.4 CPU handover

```
GPU: solve → derive bondActive[] → label[slot] kernel → readback small buffer
CPU: PersistentIslandTracker consumes labels, runs Elder-Rule identity, emits OrphanEvent / SplitEvent
```

`LabelPropagation.bfsComponents` is deleted. `StructureIslandRegistry.checkAndSplitIsland` becomes a thin adapter over the tracker's output. The tracker's own BFS helper becomes dead code and goes with it. Net removal: ~800 lines on the connectivity side.

## 5. Slot stability semantics

History-dependent fields — h-field evolution, damage accumulation, failure event logs, debug replay — reference voxels by slot number. A slot that silently renames breaks every one of them. The allocator therefore guarantees:

- **Stability across local edits.** Inserting, removing or mutating a voxel does not renumber surviving slots. A neighbour's slot stays identical tick-over-tick unless the voxel itself is deleted.
- **Tombstones on deletion.** A deleted voxel becomes a tombstone slot: its physics fields zero out, its `bondActive` clears, but the slot id remains reserved. Downstream systems that hold a slot reference (e.g. a pending failure event addressed by slot) receive a defined "this slot is dead" answer rather than a silent re-use.
- **Deferred compaction.** Tombstones accumulate until either (a) the tombstone ratio exceeds a tunable threshold (`BRConfig.pnsmCompactionThreshold`, default 25 %), or (b) a world save is about to occur. Compaction is an explicit batch operation; it never runs inside a physics tick.
- **Relocation table on compaction.** A compaction pass emits `oldSlot → newSlot` for every slot that moved (tombstones are dropped; only live slots are renumbered contiguously). Every consumer that keyed off slots — history buffers, event ring, label output, debug replay — translates through the table atomically. Consumers that miss the translation window are flagged as stale by a per-compaction generation counter.
- **Generation counter.** Every slot reference carries an implicit generation tag. Post-compaction code that tries to dereference a pre-compaction slot gets an "expired" result and a telemetry ping, not silent corruption.

This is the promise the previous revision failed to make. Without it, history fields become meaningless the moment geometry changes, and the proposal's memory wins are paid for by losing damage accumulation across edits — a trade no one wants.

## 6. Honest trade-offs

- **Mutation cost**: an SO-DAG edit is `O(log N)` subtree hashes vs. the current `O(1)` HashMap put. In absolute numbers, a 4 m-block world is ~20 hash steps per edit — sub-microsecond in comparable SVDAG implementations. Not free, not a problem.
- **Hash collisions**: content-addressing requires a 64-bit subtree hash with a 128-bit tie-breaker in the arena. One memcmp per insertion; negligible.
- **`ε_bond` tuning**: per-material threshold from the `MaterialCalibration` registry means this lands with defaults and calibration tooling already in place. Integration test: verify bond-graph labeling agrees with reference BFS across a corpus of seeded structures.
- **Debuggability**: a flat HashMap is trivially inspectable; an SO-DAG is not. Mitigation: a dev-time visualiser that walks SO-DAG → renders leaves with palette-coloured overlays, plus a `/br slotdump <islandId>` subcommand that dumps live slots with tombstone status.
- **Compaction stalls**: deferred compaction can land at world-save time, adding perceptible pause if islands are large. Mitigation: compaction runs in a background worker and double-buffers the arena; the tick loop sees a stable snapshot until the swap atom.
- **ABI churn**: every system holding a `BlockPos → islandId` contract today needs to switch to fingerprint-based addressing. Roughly 30 call sites; mechanical but exhaustive.

## 7. Migration path — five phases, shadow-gated

Matches the sequence you proposed in the review: each phase is a PR or small PR train, and phases 1–4 are fully reversible because legacy and new run side-by-side with a diff-and-warn reconciliation.

1. **Phase 1 — SO-DAG shadow mirror + diff.** Build Layer A behind `BRConfig.enablePNSM` flag. Mirror every `registerBlock` / `unregisterBlock` into the DAG. At tick end, diff DAG-reconstructed voxel set against `StructureIsland.members`; any disagreement logs a warning. Current stack stays in production; zero user-visible change.
2. **Phase 2 — GPU packed arena shadow path.** Layer B allocator runs alongside `PFSFIslandBuffer`. Same islands, same inputs, compare `phi` per-voxel each tick; diverge → log + disable new path. One release cycle of shadow-mode.
3. **Phase 3 — active-bond component labeling on GPU.** Derive `bondActive[]` from existing conductance/damage buffers and run the label-prop kernel (leveraging `PFSFLabelPropRecorder` + `cbf7e8b`). Feed labels into a parallel tracker instance; compare its output against the BFS-driven tracker every tick. Diverge → warn.
4. **Phase 4 — retire CPU BFS.** After phase 3 shows ≥ 10 k ticks of agreement across playtest, delete `LabelPropagation.bfsComponents` and the tracker's internal BFS helper. Component labels now come from the GPU path as the sole source of truth. This is the point the code-surface reduction lands.
5. **Phase 5 — delete legacy registry + AABB buffers.** `StructureIslandRegistry.blockToIsland`, the `members` sets, and the AABB-padding GPU buffers come out. This is the only irreversible step; by this point PNSM has two release cycles of production hours behind it.

Phases 1–4 ship independently; each is its own PR and a flag-flip. The first irreversible step is four PRs away. If phase 1's mirror-diff looks bad, we can delete the mirror code and walk away with no user-visible change.

## 8. What this buys

- **Memory**: ~11–13× reduction for a typical mixed-structure world (~6 MB → ~530 KB GPU per 10 k-voxel island; CPU registry bookkeeping drops proportionally).
- **Correctness**: three independent connectivity passes collapse into one GPU kernel with a well-known algorithm. No more "legacy BFS says A, tracker says B" investigation threads.
- **Code surface**: ~800 LOC removed from the physics side (`LabelPropagation` + half of `StructureIslandRegistry` + tracker BFS helpers).
- **GPU**: material lookups vanish from the hot shader path; palette LUT replaces per-voxel material id reads.
- **Gameplay semantics**: identity becomes the first-class identifier everywhere. Fingerprints survive splits, merges, and world reloads; int ids stop being mentioned outside the GPU arena.
- **History-safe edits**: slot stability + relocation tables let h-field / damage / failure events survive structural mutations — today, any geometry change silently invalidates them.

## 9. What this does not buy

- Fluid simulation still lives outside the SO-DAG (different grid, different solver) — PNSM does not help `PFSF-Fluid`.
- Render-side greedy meshing is its own geometry cache; PNSM supplies occupancy but the mesher still classifies faces.
- Node-graph editor state is orthogonal.
- The existing int-id API stays backwards-compatible through phase 5 — legacy mods do not benefit directly; they do not break either.

## 10. Recommendation

Ship phase 1 behind the flag and let it run shadow-mode for one development cycle. The mirror-and-diff approach makes this nearly risk-free: the legacy stack keeps running, and the first time PNSM disagrees we get a warning rather than a physics regression. Decision on phase 2 waits until phase 1 shows ≥ 24 hours of playtest with zero mirror-diffs.

The payoff is large, the commitment is gradual, and the first irreversible step is four PRs away. The most valuable part of this v2 revision — from an engineering risk standpoint — is §5 (slot stability) and the §4 move from scalar threshold to bond-graph labeling. Both address correctness holes the v1 RFC had, and both are non-negotiable before any phase 3 code lands.
