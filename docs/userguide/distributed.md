<!-- markdownlint-disable MD014 -->

(distributed_guide)=

# Distributed Simulations

The `nvalchemi.distributed` package extends the toolkit's dynamics + model
machinery to run across multiple GPUs via spatial domain decomposition. A
single :class:`~nvalchemi.distributed.DomainParallel` wrapper takes any
:class:`~nvalchemi.dynamics.base.BaseDynamics` integrator or optimizer and
makes it run on a partitioned :class:`~nvalchemi.distributed.ShardedBatch`,
with halo exchanges + cross-rank reductions handled automatically.

```{tip}
The distributed API is intentionally separate from the single-process
dynamics API: the same {py:class}`~nvalchemi.models.base.BaseModelMixin`
wrapper, the same hooks, and the same integrators run unchanged. The
only addition at the user layer is one
{py:class}`~nvalchemi.distributed.DomainConfig` and a
{py:class}`~nvalchemi.distributed.DomainParallel` wrap.
```

This guide covers:

1. **Why** spatial domain decomposition and **what** it gets you.
2. The **two storage strategies** the framework supports — halo storage
   and sharded storage — and when to pick each.
3. The **runtime architecture**: how
   {py:class}`~nvalchemi.distributed.DomainParallel`,
   {py:class}`~nvalchemi.distributed.ShardedBatch`, and the
   `DistributedModel` adapter cooperate per step.
4. A **minimal usage example** end-to-end.

Two companion guides go deeper:

- {doc}`distributed_shardtensor` — how
  {py:class}`~nvalchemi.distributed._core.shard_tensor.ShardTensor`
  represents a partitioned per-atom field and how its
  `__torch_function__` dispatch routes operations through
  distribution-aware handlers.
- {doc}`distributed_byo` — bringing your own model under
  domain decomposition: writing the wrapper, authoring or deriving an
  :class:`MLIPSpec`, and using `trace_and_validate` to confirm
  correctness.

## Why partition?

A standard MLIP forward on a single GPU lays out every atom's per-atom
fields (`positions`, `forces`, `node_features`) as a single
`(N, F)` tensor and computes everything in one process. That's optimal
for systems up to a few thousand atoms but breaks down past that:

- **Memory.** Per-atom node features can dominate the activation budget
  in modern message-passing networks; the largest production MACE /
  UMA configurations OOM at < 50k atoms on an H100.
- **Throughput.** Even when memory fits, a single GPU's neighbor-list
  build, message passing, and force consolidation are sequential — no
  amount of batching helps a single trajectory.
- **Latency.** Multi-thousand-step MD trajectories on a single GPU
  measure in days; spatial parallelism cuts wall-clock proportionally
  to GPU count.

`nvalchemi.distributed` answers all three by **partitioning atoms across
GPUs by spatial location**, replicating only the small *halo* of atoms
within the model's interaction cutoff so each rank evaluates its
subdomain independently. Cross-rank communication happens once per
step (the halo exchange) plus a handful of collectives for
per-system reductions.

## Two storage strategies

The choice of how per-atom fields are laid out across ranks is the
single biggest architectural decision in any distributed MD framework.
`nvalchemi.distributed` supports two:

### Halo storage

Each rank holds *all* the per-atom rows it needs to evaluate its
owned atoms — that's `n_owned` owned rows plus a `n_halo`-row halo of
copies of neighbouring ranks' atoms within `ghost_width` of any owned
atom. The padded layout is:

```text
rank 0:   [ owned_0 | halo_from_1, halo_from_2, ... ]  # shape (n_padded, F)
rank 1:   [ owned_1 | halo_from_0, halo_from_2, ... ]  # shape (n_padded, F)
…
```

A halo exchange at the start of each step refreshes the halo rows.
The model then evaluates each rank's `n_padded` rows as a regular
forward pass — every cross-rank pair distance is computed locally
because the partner atom is already in the halo. The only
distributed mechanics on the model's hot path are halo-correction
scatters (when a `scatter_add_` writes into halo rows that should
be reverse-summed back to owners) and per-system reductions (when
the model produces a per-graph quantity like total energy).

**Pick halo storage when:**

- The model is a scatter-heavy MPNN (MACE, NequIP, Allegro, ORB, UMA).
  Every message-passing layer does a `scatter_sum` into per-atom
  features; halo-correction handles the cross-rank case naturally.
- The model has a clear interaction cutoff (typically `< 6 Å` for
  modern MLIPs). The cutoff bounds the halo width; long-range
  models (Ewald, PME) can still use halo storage with a separate
  reciprocal-space dispatch.

This is the default for the `MACE` / `LJ` / `UMA` / `Ewald` / `PME`
wrappers shipped with the toolkit.

### Sharded storage

Each rank holds *only* its `n_owned` rows. Cross-rank operations
(`index_select`, `scatter_add`) route through global IDs at dispatch
time — the rank that needs row `j` gathers it on demand, computes,
and either returns the result to the original rank or scatters
locally.

```text
rank 0:   [ owned_0 ]  # shape (n_owned_0, F)
rank 1:   [ owned_1 ]  # shape (n_owned_1, F)
…
```

**Pick sharded storage when:**

- The model is a charge-equilibration network (AIMNet2). Equilibrium
  charges depend on a global mol_sum across all atoms in a system —
  no spatial cutoff bounds the dependence, so halo storage would need
  the entire box as halo on every rank.
- Per-atom out-of-place updates dominate the forward (each
  message-passing layer rewrites `x` rather than scatter-adding into
  it). Halo-storage's halo-correction would need a refresh after
  every update; gather avoids that bookkeeping.

The sharded path is only enabled by AIMNet2 in the current toolkit.

### Choosing

The strategy is declared on the model's
:class:`~nvalchemi.distributed.spec.MLIPSpec`. The shipped presets are:

| Preset | Storage | Models |
|---|---|---|
| `SPEC_MPNN_HALO` | halo, halo-correction scatter, halo-read gather | MACE, NequIP, generic MPNN |
| `SPEC_LJ_HALO` | halo, halo-correction scatter | Lennard-Jones, pair potentials |
| `SPEC_UMA_GATHER` | halo, local scatter (eSCN backbone is halo-unaware) | UMA |
| `SPEC_EWALD_HALO` | halo, with custom-op adapters for reciprocal-space | Ewald |
| `SPEC_PME_HALO` | halo, with custom-op adapters for charge spreading | PME |
| `SPEC_AIMNET2_GATHER` | sharded, distributed scatter+gather | AIMNet2 |

If your model fits one of these patterns, the preset is a one-line
declaration on your wrapper's `distribution_spec` property. If it
doesn't, see {doc}`distributed_byo` for the authoring workflow.

## Runtime architecture

```{graphviz}
:caption: Per-step flow under DomainParallel.

digraph distributed_step {
    rankdir=TB
    fontname="Helvetica"
    node [fontname="Helvetica" fontsize=11 shape=box style="rounded,filled"]
    edge [fontname="Helvetica" fontsize=10]

    DP [label="DomainParallel.step()" fillcolor="#dce6f1"]
    Halo [label="halo_exchange\n(populate halo rows)" fillcolor="#f9e2ae"]
    NL [label="NeighborListHook\n(NL on padded batch)" fillcolor="#f9e2ae"]
    Wrap [label="DistributedModel\n(spec dispatch)" fillcolor="#dce6f1"]
    Inner [label="wrapper(padded_batch)" fillcolor="#dce6f1"]
    Cons [label="output_consolidation\n(slice / halo_reverse / all_reduce)" fillcolor="#f9e2ae"]
    Integ [label="inner integrator\npost_update + atom migration" fillcolor="#dce6f1"]

    DP -> Halo -> NL -> Wrap -> Inner -> Cons -> Integ
}
```

The pieces:

- {py:class}`~nvalchemi.distributed.ShardedBatch` is the persistent
  rank-local store of owned atoms (positions, velocities, forces,
  cell, etc.) plus the rank-assignment map needed to migrate atoms
  across ranks when they cross domain boundaries. It's built once on
  rank 0 from the full batch and scattered via
  {py:meth}`~nvalchemi.distributed.DomainParallel.partition`.
- :class:`~nvalchemi.distributed.distributed_model.DistributedModel`
  is the per-step adapter wrapping a single-process
  {py:class}`~nvalchemi.models.base.BaseModelMixin`. Its `__call__`
  takes a `ShardedBatch`, runs the appropriate storage path
  (`_call_halo_storage` or `_call_sharded_storage`), and returns
  consolidated outputs in the standard
  {py:class}`~nvalchemi._typing.ModelOutputs` format.
- {py:class}`~nvalchemi.distributed.DomainParallel` is the integrator
  wrapper. It composes a `DistributedModel` with any
  {py:class}`~nvalchemi.dynamics.base.BaseDynamics` subclass and
  drives the per-step loop.

The user-facing API is `DomainParallel`; the layers below are
internal but exposed for advanced users (e.g. running a single
forward without an integrator).

## Minimal example

A complete distributed MACE NVT trajectory:

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed import DomainConfig, DomainParallel
from nvalchemi.dynamics import HostMemory, NVTLangevin
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import SnapshotHook
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper

# torchrun populates RANK / WORLD_SIZE / LOCAL_RANK
dist.init_process_group(backend="nccl")
device = torch.device(f"cuda:{dist.get_rank()}")
torch.cuda.set_device(device)

mesh = DeviceMesh(
    "cuda", list(range(dist.get_world_size())), mesh_dim_names=("domain",)
)

# 1. Wrap the model — same wrapper as single-process.
wrapper = MACEWrapper.from_checkpoint("medium-0b2", device=device).eval()

# 2. Build the inner integrator with a NeighborListHook.
nl_hook = NeighborListHook(
    wrapper.model_config.neighbor_config,
    skin=0.5,
    stage=DynamicsStage.BEFORE_COMPUTE,
)
sink = HostMemory(capacity=100)
snap_hook = SnapshotHook(sink=sink, frequency=10)

integrator = NVTLangevin(
    model=wrapper,
    dt=0.5,        # fs
    temperature=300.0,
    friction=0.01,
    hooks=[nl_hook, snap_hook],
    n_steps=200,
)

# 3. Wrap with DomainParallel + a DomainConfig describing the mesh
#    and the halo width (``cutoff = wrapper.cutoff`` for an exact match).
domain_cfg = DomainConfig(cutoff=float(wrapper.cutoff), skin=0.5, mesh=mesh)
dynamics = DomainParallel(dynamics=integrator, config=domain_cfg, n_steps=200)

# 4. Build the full batch on rank 0; partition.
batch = build_my_batch(device) if dist.get_rank() == 0 else None
owned = dynamics.partition(batch)

# 5. Run. ``DomainParallel.run`` is the canonical entry point; the
#    SnapshotHook accumulates per-step batches in ``sink``.
dynamics.run(owned)

dynamics.close()
dist.destroy_process_group()
```

The full version, with xyz trajectory persistence and CLI arguments,
ships as `examples/distributed/03_mace_nvt_distributed.py`.

## DomainConfig

{py:class}`~nvalchemi.distributed.DomainConfig` carries the runtime
parameters every rank needs:

- `cutoff` — the model's interaction cutoff (Å). Sets the minimum halo
  width.
- `skin` — extra ghost-region padding (Å) so the halo doesn't need
  rebuilding every step. Set to 0 for one-shot inference; set to
  `0.3 – 1.0 Å` for MD where atoms drift between rebuilds.
- `mesh` — the
  {py:class}`~torch.distributed.device_mesh.DeviceMesh`. Construct
  manually or derive from `dist.get_world_size()`.

Phase 7 of the distributed refactor split `DomainConfig` into focused
sub-configs ({py:class}`~nvalchemi.distributed.config.HaloConfig`,
{py:class}`~nvalchemi.distributed.config.MeshConfig`,
{py:class}`~nvalchemi.distributed.config.PartitionConfig`); the legacy
flat constructor still works for callsites that pass `cutoff=...`,
`mesh=...`, etc.

## Next steps

- The {doc}`ShardTensor walkthrough <distributed_shardtensor>`
  explains how per-atom tensors flow through halo exchange and
  per-system reductions, and what subclass propagation guarantees the
  framework relies on.
- The {doc}`Bring-your-own-model walkthrough <distributed_byo>`
  shows how to declare an
  {py:class}`~nvalchemi.distributed.spec.MLIPSpec` for a new wrapper,
  validate it via :func:`trace_and_validate`, and persist the
  resulting spec for production use.
- The runnable example in
  `examples/distributed/03_mace_nvt_distributed.py` is the
  end-to-end version of the snippet above.
