# Proposal: Graph-Parallel / Replicate Strategy (alongside Halo DD)

**Status:** scoping (no code yet). Motivated by the A6000 head-to-head where
fairchem's graph-parallel UMA scaled ~2:1 better than our spatial halo DD at
small/dense systems (w2 0.52× memory + 1.8× speed + 2× max-N vs our 0.88× / no
gain). Root cause: our spatial halo pays ~95% ghost overhead when the box (~29 Å)
≈ the halo shell; fairchem's edge-by-atom-index partition has no halo penalty.

## TL;DR
Add a second, selectable distributed strategy — **edge-partition + replicate-nodes**
(fairchem `gp_utils` style) — beside the existing spatial halo. **~2/3 of the
plumbing already exists** as a half-wired "sharded" path; this is largely *activating
and completing* it, not building from scratch. Keep BOTH strategies: GP wins
small/dense N, halo wins large N — complementary, not competing.

## Architecture fit
The spec/policy/placement/context abstractions cleanly admit a 3rd strategy.
"Halo" is baked into exactly **four** places:
1. `distributed_model.py:421-427` — `__call__` raises unless policy is `HaloStoragePolicy` (the one hard gate).
2. `distributed_model.py:524-533` — `_ensure_initialized` hard-builds `SpatialPartitioner` + `ParticleHaloConfig`.
3. `helpers.py:176-263` — `refresh_neighbors` / `scatter_to_owners` raise `NotImplementedError` when `not ctx.is_halo`.
4. output consolidation picks `consolidate_padded_outputs` (halo-reverse) — GP wants the sibling `consolidate_sharded_outputs`.

Already strategy-agnostic: `StoragePolicy` Protocol (`PlainShard`=Shard(0), halo, and
a future `Replicate` are all admissible), `ShardRouting.from_assignment` (consumes any
rank→row map — "spatial, contiguous block, or any custom partitioner"), and
`DistributedContext` which already has `is_sharded`/`gather_meta` (the non-halo slot).

## Component inventory
| Piece | Status |
|---|---|
| Index partitioner `tensor_split(arange(N),W)[rank]` | BUILD (trivial, ~30 lines) → feeds existing `ShardRouting.from_assignment` |
| Edge-by-owned-target filter | REUSE pattern — per-model `adapt_input` NL filter (like halo's `_mark_halo_receiver_edges_as_padding`) |
| Per-layer all-gather→replicate (fwd) + reduce_scatter (bwd) | **BUILD — the real work**: `_GatherToReplicate` autograd.Function in `gather_primitives.py`, ~60 lines, mirrors existing `_DistributedAllReduceSum`/`_DistributedScatterAdd`; reuses `_all_gather_v_rows` + `_all_to_all_v_rows` |
| Output energy/force all_reduce | REUSE as-is: `consolidate_sharded_outputs` + `distributed_all_reduce` + `system_sum(OWNED)` |
| Strategy switch | BUILD (small): `GraphParallelPolicy` in `storage_policy.py` + `__call__` dispatch + `_ensure_initialized` branch |
| `_call_graph_parallel` forward | BUILD: modeled on `_call_halo_storage` minus halo exchange, populates `gather_meta` |
| Per-layer intent verbs | REUSE call sites, BUILD `is_sharded` branch in `refresh_neighbors`/`scatter_to_owners` |

### ShardTensor `redistribute([Replicate()])`? — NO.
Use a **standalone plain-tensor `autograd.Function`**, not a ShardTensor placement
transition. Rationale: the halo per-layer correction is deliberately NOT routed through
ShardTensor (UMA sets `shard_fields=()`, does block-boundary comm on plain tensors via
`current_dd_context()`/`compile_routing`). GP is the same shape of problem → same
pattern. Avoids `__torch_dispatch__` mixed-Tensor/DTensor backward hazards and the
`@torch.compiler.disable` graph-break interaction; reuses the proven distributed-primitive
autograd family.

## MVP: MACE (not UMA)
MACE uses `FRAMEWORK_FROM_NODE_ENERGY` → framework owns energy forward, forces = −dE/dx
via autograd, so GP force reduction falls out of autograd of the replicated energy (no
per-model force code). Per-layer aggregation is a plain `scatter_sum`. UMA (MoLE, fairchem
graph padder, Triton) and AIMNet2 (dense nbmat, ConvSV MethodAdapter) are heavier — defer.

## Phasing
- **P0 (S) DONE** strategy seam + `IndexPartitioner` + `GraphParallelPolicy` + dispatch.
- **P1 (M) DONE** `_GatherToReplicate` autograd.Function; toy MPNN equivalence (`test/distributed/_core/test_graph_parallel.py`) w=2/3.
- **P2 (M) DONE** intent verbs delegate to `ctx.policy` (polymorphism refactor, no `is_sharded` branch) + `SPEC_MPNN_GP` + owned-target NL filter (`_graph_parallel_owned_edges`) + `_call_graph_parallel`. Eager. Gated end-to-end through `DistributedModel` by a BYO toy GP wrapper (`test/distributed/model/test_graph_parallel_model.py`): energy + owned forces == single-process, machine precision, box-green w=2/3. **Energy reduce must be `system_sum(LOCAL)` + plain all_reduce — `OWNED`'s autograd all_reduce inflates forces ×world_size.** Next: real A6000 GP-vs-halo data point.
- **P3 (L, deferred)** compile path (GP is *friendlier* than halo: replicated shape `[N_total,*F]` is static → no per-step routing tensors).
- **P4 (M)** real MACE-GP (needs `neighbor_refresh_adapters` extended to GP so a pure wrapper takes the asymmetric global-sender/owned-receiver edges) + UMA + AIMNet2 + validator coverage.

## Risks / open questions
- **Compile graph-breaks:** GP collectives graph-break like halo's, but GP's static gathered
  shape is *easier* under fixed-N MD. MVP eager; revisit P3.
- **Autograd over-count:** all-gather replicates energy on every rank → backward ×world_size,
  exactly the factor `consolidate_sharded_outputs` already divides out. reduce_scatter must be
  the exact adjoint of the gather. `gradcheck` is the backstop.
- **Load-balance vs locality:** GP pays a full `[N_total,hidden]` all-gather *every* layer
  (vs halo's one boundary exchange/layer) → GP loses at large N, wins at small/dense N where
  halo's ghost fraction → 1. **Keep both, selectable by spec policy.**
- **Coexistence:** share the adapter mechanism + intent-verb call sites; differ only in the
  policy branch inside the verb. A wrapper picks halo-vs-GP purely by which `MLIPSpec` policy
  it declares → per-model GP cost ≈ zero once framework pieces land.
- **Worth it?** Yes — mostly completing a half-wired path; the sharded scaffolding
  (`PlainShard`, `ShardRouting`, `consolidate_sharded_outputs`, `is_sharded`, distributed
  gather/scatter autograd) is already built + tested.

## Critical files
- `nvalchemi/distributed/distributed_model.py` — `__call__` gate :421-427, `_ensure_initialized`, new `_call_graph_parallel`
- `nvalchemi/distributed/_core/gather_primitives.py` — new `_GatherToReplicate` (reuses `_all_gather_v_rows`, `_all_to_all_v_rows`, `_DistributedAllReduceSum`)
- `nvalchemi/distributed/_core/storage_policy.py` — add `GraphParallelPolicy`
- `nvalchemi/distributed/helpers.py` — `is_sharded` branch in `refresh_neighbors`/`scatter_to_owners`/`system_sum`
- `nvalchemi/distributed/partitioner.py` — `IndexPartitioner` → `_core/placement.py:ShardRouting.from_assignment`
- (per-model) `nvalchemi/models/mace.py` — `SPEC_MACE_GP` + owned-target NL filter
