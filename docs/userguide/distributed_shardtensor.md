<!-- markdownlint-disable MD014 -->

(distributed_shardtensor_guide)=

# ShardTensor: How Per-Atom Fields Flow Across Ranks

Spatial domain decomposition partitions per-atom tensors across ranks
— but without further machinery, every operation on those tensors
would be a regular per-rank op with no knowledge of the partition.
:class:`~nvalchemi.distributed._core.shard_tensor.ShardTensor` is the
{py:class}`torch.Tensor` subclass that carries the partition's
metadata along with the data, and routes select operations through
distribution-aware handlers via PyTorch's `__torch_function__`
protocol.

This guide is about the *mechanism*. It complements the
{doc}`distributed user guide <distributed>` (which covers when to
use which storage strategy) and {doc}`distributed_byo` (which covers
authoring a spec for a new model).

## The subclass approach

`ShardTensor` is an "almost transparent" Tensor subclass: it stores
the same underlying data buffer as a regular `torch.Tensor`, plus a
small bag of metadata describing the partition:

| Field | What it carries | Used by |
|---|---|---|
| `_spec` | The :class:`MLIPSpec` governing dispatch behaviour | All op-level handlers |
| `_meta` | Halo-storage metadata: `n_owned`, `n_padded`, halo routing index | Halo-exchange, halo-correction |
| `_gather_meta` | Sharded-storage metadata: per-row global IDs, rank assignments | Sharded `index_select`, `scatter_add` |
| `_config` | The per-rank :class:`ParticleHaloConfig` (process group, ghost width) | All collective ops |
| `_n_systems` | Number of systems on this rank | Per-system reductions |

The wrap is zero-copy: `ShardTensor.wrap(t, spec=...)` calls
`t.as_subclass(ShardTensor)` and attaches the metadata. The
underlying storage is shared; the wrap survives in-place mutations
and the autograd graph.

```python
from nvalchemi.distributed.ops import ShardTensor
from nvalchemi.distributed.spec import SPEC_MPNN_HALO

local_positions = torch.zeros(n_padded, 3, device="cuda")
shard = ShardTensor.wrap(
    local_positions,
    spec=SPEC_MPNN_HALO,
    meta=halo_meta,         # ParticleHaloMetadata describing this rank's slice
    config=halo_config,     # ParticleHaloConfig with the process group
    n_systems=n_systems,
)
```

Most code never constructs `ShardTensor` directly:
:class:`~nvalchemi.distributed.distributed_model.DistributedModel`
promotes `data.positions` (and `data.charges` if present) to
`ShardTensor` in its halo-storage call path before invoking the
wrapper, so per-atom ops inside the wrapper's forward see a
`ShardTensor` and dispatch accordingly.

## `__torch_function__` dispatch

Every torch op called on a `ShardTensor` runs through
`ShardTensor.__torch_function__(func, types, args, kwargs)` — the
standard subclass-hook PyTorch provides. The dispatch is
predicate-based: a small registry of handlers, each tagged with a
predicate `(func, args, kwargs) -> bool`, is consulted in order. The
first matching handler runs; if none match, the op falls back to
the default `torch.Tensor.__torch_function__`.

```{graphviz}
:caption: Dispatch decision tree for an op called on a ShardTensor.

digraph dispatch {
    rankdir=TB
    fontname="Helvetica"
    node [fontname="Helvetica" fontsize=11 shape=box style="rounded,filled"]
    edge [fontname="Helvetica" fontsize=10]

    Op [label="op(shard_tensor, ...)" fillcolor="#dce6f1"]
    Pred [label="any registered handler\npredicate matches?" fillcolor="#f9e2ae" shape=diamond]
    Handler [label="handler runs:\n• unwrap inputs\n• run cross-rank logic\n• promote outputs"
             fillcolor="#82b366"]
    Fallback [label="super().__torch_function__\n(plain torch.Tensor path)" fillcolor="#dce6f1"]

    Op -> Pred
    Pred -> Handler [label="yes"]
    Pred -> Fallback [label="no"]
}
```

Three handler families are registered today:

- **Halo correction.** Fires on `scatter_add_` / `index_add_` calls
  where the destination is a `ShardTensor` carrying halo metadata.
  After the local scatter, the handler does
  `halo_reverse_exchange + halo_forward_exchange` so halo rows
  contribute their partial sums back to owners and the halo is
  re-populated with the corrected owner values for downstream ops.
- **Per-system reduce.** Fires on `scatter_add_` calls whose target
  is a per-system buffer (shape `(n_systems, F)`) and whose source
  is per-atom. Slices halo rows off the source first (so each atom
  contributes once), then scatters locally and all-reduces across
  the mesh.
- **Distributed scatter / index_select.** Fires for sharded-storage
  models. Routes the op via global IDs: an `index_select` with
  cross-rank target rows gathers them via `all_to_all_v`; a
  `scatter_add` with cross-rank source rows likewise.

The registry is in
{py:mod}`nvalchemi.distributed._core.shard_tensor` and is keyed by
op + predicate so multiple handlers can coexist for the same op
(e.g. halo-correction for one shape, per-system-reduce for another).

## Halo storage in detail

```{graphviz}
:caption: Halo-storage layout across two ranks for a 6-atom system.

digraph halo_storage {
    rankdir=LR
    fontname="Helvetica"
    node [fontname="Helvetica" fontsize=10 shape=box style="filled,rounded"]
    rank0 [label="rank 0\\nowned: {0,1,2}\\nhalo: {3, 4}\\n(copies of rank 1's owned)"
           fillcolor="#dce6f1"]
    rank1 [label="rank 1\\nowned: {3,4,5}\\nhalo: {1, 2}\\n(copies of rank 0's owned)"
           fillcolor="#dce6f1"]
}
```

At step start, each rank's halo rows are stale. The halo exchange
populates them by all-to-all-v of owned-row data into the partner
ranks' halo slots, with `_meta.halo_routing` carrying the index
table. For the duration of the step, every read of `positions[j]`
where `j` is a halo row resolves to a *current* copy of rank
`r(j)`'s owned atom — so cross-rank pair distances are computed
locally with no further communication.

Halo writes are different. When a model writes into a halo row via
`out.scatter_add_(0, receiver, msg)` and `receiver[e]` happens to be
a halo atom, the write only contributes a partial sum on this rank.
The corresponding owner on the other rank holds its own partial sum
from its own edges. **Halo correction** reverses this: after the
scatter, halo-row partial sums are routed back via
`halo_reverse_exchange` and added to the owner's value; then
`halo_forward_exchange` repopulates this rank's halo with the
combined owner result so downstream ops in the same forward pass
see consistent per-atom features.

The handler is registered on `scatter_add_` / `index_add_`. A model
author who writes a standard PyTorch MPNN

```python
out = torch.zeros_like(x)
out.scatter_add_(0, receivers, msg)
```

gets halo correction for free *iff* `out` is a `ShardTensor` —
which it is automatically when `x` is, because
`torch.zeros_like(x)` propagates the subclass.

## Sharded storage in detail

Sharded-storage models hold only owned rows on each rank — there's
no halo. Cross-rank lookups happen on demand.
{py:class}`~nvalchemi.distributed.sharded_batch.ShardedBatch`'s
`_atom_fields` carry per-row global-ID metadata; when a model does
`x.index_select(0, idx)` with `idx` containing cross-rank global
IDs, the dispatch handler:

1. Inspects `idx` against the `_gather_meta.rank_assignment` table
   to figure out which rank owns each requested row.
2. Issues an `all_to_all_v` to ship the requested rows to this rank.
3. Reorders the result back into the order `idx` requested.

The reverse holds for `scatter_add` with cross-rank receivers:
locally-grouped contributions are shipped to the owner ranks where
the actual scatter occurs.

`_gather_meta` carries a `n_global` sentinel for "this slot is
padding" — out-of-range indices in `idx` resolve to a known
empty contribution rather than triggering a CUDA out-of-bounds
assertion.

## Subclass propagation guarantees

Two PyTorch behaviours that the framework relies on:

1. **Like-shaped allocator ops preserve subclass.** `torch.zeros_like(x)`,
   `torch.empty_like(x)`, `x.new_zeros(...)`, etc. return a
   `ShardTensor` when `x` is a `ShardTensor`, with the same
   `_spec` / `_meta` / `_config`. This is what makes the toy MPNN
   pattern in {doc}`distributed_byo` work without explicit wrap calls
   inside the model body.
2. **Most ops downcast to `torch.Tensor`.** `x[i]`, `x + y`,
   `linear(x)` — these go through the default
   `__torch_function__` path which produces a plain Tensor view of
   the underlying storage. The autograd graph flows through this
   view; the ShardTensor subclass identity is dropped. This is fine
   for ops that don't need cross-rank communication.

The split between "preserves subclass" and "drops subclass" is
deliberate. Halo-correction needs the destination of `scatter_add_`
to be a `ShardTensor` (so it can reach the metadata); intermediate
features after a `Linear` layer don't, because Linear is a per-rank
op.

## Custom ops and `OpAdapter`

Warp / Triton / Numba / generic CUDA kernels wrapped as
`@torch.library.custom_op` are *opaque* to subclass dispatch: the
kernel does `wp.from_torch(t)` (or the equivalent) internally,
bypassing `__torch_function__`. Without further help, calling such
an op on a `ShardTensor` would unwrap to a plain Tensor (losing
the metadata) and run the kernel as if the input were single-process.

{py:class}`~nvalchemi.distributed.spec.OpAdapter` declares the
distribution semantics for one such kernel:

```python
from nvalchemi.distributed.ops import GatherInputs, ScatterOutputs
from nvalchemi.distributed.spec import OpAdapter

OpAdapter(
    op=torch.ops.mymodel.fused_kernel.default,
    arg_transforms={0: GatherInputs()},     # halo-pad input position 0
    output_transforms={0: ScatterOutputs()}, # halo-correct output position 0
)
```

The adapter goes on the spec's `distribution.custom_ops`. At
scope-entry the framework's
{py:class}`~nvalchemi.distributed.AdapterRegistry`
walks the spec, installs a ShardTensor handler on each op handle,
and the kernel becomes distribution-aware: when called with a
ShardTensor input, the handler runs the declared
`arg_transforms`, calls the kernel on plain tensors, then runs
the declared `output_transforms` and re-promotes outputs.

The available transforms are:

| Transform | Pre-/post-kernel action |
|---|---|
| `GatherInputs` | halo-pad an owned input to `(n_padded, *F)` |
| `GatherInputsFull` | full-gather a sharded input to `(n_global + 1, *F)` |
| `SliceOwned` | slice a halo-padded input to `(n_owned, *F)` |
| `ScatterOutputs` | `halo_reverse + halo_forward` on a per-atom output |
| `AllReduceSum` | cross-mesh `SUM` all-reduce on a partial output |
| `SliceOutputsOwned` | slice an `(n_global + 1, *F)` output back to `(n_owned + 1, *F)` |

See {doc}`distributed_byo` for an end-to-end OpAdapter
authoring example with a Warp kernel.

## When you don't need ShardTensor

If your wrapper's forward is built entirely from torch ops that the
framework already handles (`scatter_add_`, `index_select`,
`scatter_add`, etc.) and you stay within a single storage strategy,
you generally don't touch `ShardTensor` directly. The framework
promotes `data.positions` and the subclass propagation handles the
rest.

You *do* need `ShardTensor` when:

- Your model has a per-layer node-feature buffer (e.g. message-passing
  state) that scatter writes target. Wrapping that buffer once via
  `ShardTensor.wrap(...)` in `adapt_input` is enough — see the MACE
  wrapper's `adapt_input` for the canonical pattern.
- You're authoring a custom op via `OpAdapter` and need to declare
  what shape the kernel expects and produces.

You *don't* need `ShardTensor` for:

- Pure per-atom ops with no aggregation (per-atom MLP, embeddings).
- Ops on per-system tensors (`scatter_add_` with target shape
  `(n_systems, F)` is automatically routed via per-system-reduce
  when `system_reductions=True` on the spec).

## Reference: where ShardTensor lives

The full implementation is in
{py:mod}`nvalchemi.distributed._core.shard_tensor`. The
upstream-candidate boundary linter
({py:mod}`tools.check_core_imports`) keeps this module
chemistry-vocabulary-free; it's the basis for any future upstream
contribution to PhysicsNeMo or related projects.

## Next steps

- {doc}`distributed_byo` walks through declaring a spec for a new
  wrapper, authoring an `OpAdapter` for a Warp kernel, and using
  `trace_and_validate` to confirm distributed correctness.
- The runnable examples in `examples/distributed/04_*` and
  `examples/distributed/05_*` exercise both patterns end-to-end.
