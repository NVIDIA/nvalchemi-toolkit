<!-- markdownlint-disable MD014 -->

(distributed_guide)=

# Distributed Simulations

The `nvalchemi.distributed` package extends the toolkit's dynamics + model
machinery to run across multiple GPUs via spatial domain decomposition. A
single {py:class}`~nvalchemi.distributed.DomainParallel` wrapper takes any
{py:class}`~nvalchemi.dynamics.base.BaseDynamics` integrator or optimizer and
makes it run on a partitioned {py:class}`~nvalchemi.distributed.ShardedBatch`,
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
  {py:class}`MLIPSpec`, and using `trace_and_validate` to confirm
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

## Two parallelization strategies

The choice of how per-atom fields are laid out across ranks is the
single biggest architectural decision in any distributed MD framework.
`nvalchemi.distributed` ships two, selected with `DomainConfig.strategy`
({class}`~nvalchemi.distributed.config.StrategyKind`): **halo** (the
default, spatial domain decomposition) and **graph-parallel** (a
node partition for models that build their own neighbour list).

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

### Graph-parallel storage (node partition)

Instead of a spatial halo, atoms are split by *index* into balanced
contiguous blocks. Each rank owns `n_owned` atoms but holds the full
geometry **replicated**, so a model that builds its own neighbour list
inside `forward` — UMA / eSCN-family models emit their own `edge_index`
via an internal `radius_pbc` kernel — still sees every position.

```text
rank 0:   positions = ALL n_global rows (replicated);  owns nodes [0 .. n0)
rank 1:   positions = ALL n_global rows (replicated);  owns nodes [n0 .. n0+n1)
…
```

Each rank runs the model on its owned block; a per-message-passing-layer
feature `all_gather` reconstructs the full node set the convolution
needs, and a reduce-scatter adjoint routes each owned atom's cross-rank
gradient back on the backward pass. Per-system quantities (energy,
stress) sum the owned slices with an `all_reduce`; forces come from the
model's own autograd. Because the partition is by index rather than
geometry, the cell is an ordinary model input, atoms never migrate
between ranks, and only the edge count drifts under MD (compiled runs
cap edges, not atoms).

**Pick graph-parallel storage when:**

- The model rebuilds its own neighbour list inside `forward` and can't
  be handed a pre-padded halo view (UMA / eSCN-family). There is no
  pre-forward seam to attach a halo to, but the full replicated
  geometry gives the internal builder everything it needs.
- You want to scale a *single* system past one GPU's memory: per-rank
  message-passing activations span only `n_owned` rows even though the
  positions are replicated, so peak memory falls with rank count.

Select it with `DomainConfig(strategy=StrategyKind.GRAPH_PARTITION)`; the
default is halo. The `SPEC_MPNN_GP` preset declares this layout for
generic MPNNs, and UMA's wrapper returns a node-partition spec when the
config selects it.

### Choosing

The strategy is declared on the model's
{py:class}`~nvalchemi.distributed.spec.MLIPSpec`. The shipped presets are:

| Preset | Storage | Models |
|---|---|---|
| `SPEC_MPNN_HALO` | halo, halo-correction scatter, halo-read gather | MACE, NequIP, generic MPNN |
| `SPEC_LJ_HALO` | halo, halo-correction scatter | Lennard-Jones, pair potentials |
| `SPEC_UMA_HALO` | halo, local scatter (eSCN backbone is halo-unaware) | UMA |
| `SPEC_EWALD_HALO` | halo, with custom-op adapters for reciprocal-space | Ewald |
| `SPEC_PME_HALO` | halo, with custom-op adapters for charge spreading | PME |
| `SPEC_DFTD3_HALO` | halo, standard energy/force outputs | DFTD3 dispersion |
| `SPEC_MPNN_GP` | graph-partition (node partition + per-layer feature all-gather) | MACE / generic MPNN, graph-parallel |

If your model fits one of these patterns, the preset is a one-line
declaration on your wrapper's `distribution_spec` property. If it
doesn't, see {doc}`distributed_byo` for the authoring workflow.

AIMNet2 is supported as well, but its wrapper builds its (halo) spec
inline rather than exposing a shipped `SPEC_*` preset.

## Runtime architecture

```{graphviz}
:caption: Per-step flow under DomainParallel.

digraph distributed_step {
    rankdir=TB
    fontname="Helvetica"
    node [fontname="Helvetica" fontsize=11 shape=box style="rounded,filled"]
    edge [fontname="Helvetica" fontsize=10]

    DP [label="DomainParallel.step()" fillcolor="#dce6f1" fontcolor="#111111"]
    Halo [label="halo_exchange\n(populate halo rows)" fillcolor="#f9e2ae" fontcolor="#111111"]
    NL [label="NeighborListHook\n(NL on padded batch)" fillcolor="#f9e2ae" fontcolor="#111111"]
    Wrap [label="DistributedModel\n(spec dispatch)" fillcolor="#dce6f1" fontcolor="#111111"]
    Inner [label="wrapper(padded_batch)" fillcolor="#dce6f1" fontcolor="#111111"]
    Cons [label="output_consolidation\n(slice / halo_reverse / all_reduce)" fillcolor="#f9e2ae" fontcolor="#111111"]
    Integ [label="inner integrator\npost_update + atom migration" fillcolor="#dce6f1" fontcolor="#111111"]

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
- {py:class}`~nvalchemi.distributed.distributed_model.DistributedModel`
  is the per-step adapter wrapping a single-process
  {py:class}`~nvalchemi.models.base.BaseModelMixin`. Its `__call__`
  takes a `ShardedBatch` and dispatches to the active parallelization
  strategy's `run_forward` (halo exchange or graph-partition), returning
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

`DomainConfig` also carries optional fields for advanced runs — `compile`
(enable a compiled distributed forward), `strategy` (select the
parallelization strategy, e.g. halo vs. graph-partition), and finer
partition/migration tuning (`ghost_width`, `grid_dims`,
`require_nondegenerate`, `migration_hysteresis`). See the class for the
full list; the three fields above are all a typical run needs.

## Compiled distributed runs

Set `compile=True` on the `DomainConfig` to run the per-rank forward under
`torch.compile`:

```python
domain_cfg = DomainConfig(
    cutoff=wrapper.cutoff, skin=0.5, mesh=mesh, compile=True
)
```

The distributed forward is **fixed-shape**: the framework pads each rank's
graph to per-rank capacity caps so tensor shapes stay static across MD
steps. The caps grow a few times during warm-up and then settle, so a
trajectory reaches a **recompile-free steady state** — the compiled graph
is reused every step. That is what makes compiled DD practical for MD
rather than one-shot inference. The shipped MACE (including
cuEquivariance), AIMNet2, and UMA wrappers all support it; a BYO model
opts in by declaring a
{py:class}`~nvalchemi.distributed.CompilePolicy` on its spec (see
{doc}`distributed_byo`).

## Distributed dynamics

Any {py:class}`~nvalchemi.dynamics.base.BaseDynamics` integrator or
optimizer runs under `DomainParallel` — you wrap it exactly like the NVT
example above and call `partition()` / `run()`. What changes under domain
decomposition is *where global quantities come from*.

### What the inner integrator sees

Under `DomainParallel`, each rank's inner integrator sees **only its
owned atoms** — never ghosts, never the whole system. The owned + ghost
(halo) view exists only inside the model forward; your `pre_update` /
`post_update` are handed the owned `Batch`. Two consequences:

- **Per-atom operations are correct as-is.** Position integration,
  velocity half-kicks, per-atom Langevin friction and noise, applying
  forces — anything that only touches per-atom fields works under DD with
  no changes. `NVE` and `NVTLangevin` are exact under DD for exactly this
  reason.
- **Global reductions are not.** Any quantity that reduces across *all*
  atoms in a system — total kinetic energy, temperature, degrees of
  freedom, a global force dot-product (FIRE), a convergence test, a
  barostat's kinetic pressure — is wrong if computed as a local `.sum()`
  / `.max()` over one rank's shard. Each rank would see only its slice.

### Supported ensembles

The shipped thermostats and barostats declare their global quantities as
*intent*; `DomainParallel`'s dynamics coordinator supplies the cross-rank
reduction, so the integrator body stays free of any distributed code. For
these, there is nothing to do — wrap in `DomainParallel` and run:

| Integrator | Under DD | Example |
|---|---|---|
| `NVE` | Exact; per-atom only, no reduction | — |
| `NVTLangevin` | Exact; per-atom only, no reduction | `03_mace_nvt_distributed.py` |
| Nosé–Hoover NVT | Global KE + DOF reduced by the coordinator | — |
| `NPT` | Global KE + DOF + pressure; replicated barostat + cell | `06_mace_npt_distributed.py` |
| `NPH` | Global KE + pressure; replicated cell | — |
| `FIRE` / `FIRE2` | Global `v·f`, `v·v`, `f·f` dot-products | `07_fire_nvt_dd.py` |

You can also run **two-dimensional parallelism** — a pipeline of stages,
each stage itself domain-decomposed — by feeding `DomainParallel` stages
to a `DistributedPipeline` over a `(pipeline, domain)` mesh;
`07_fire_nvt_dd.py` shows a FIRE→NVT pipeline where each stage is its own
DD sub-mesh.

```{note}
Global thermostat/barostat support (Nosé–Hoover, NPT, NPH) is recent.
Validate long production barostat trajectories before relying on them.
```

### Bring your own integrator under DD

A custom `BaseDynamics` subclass (see
{doc}`dynamics_simulations` → *Writing your own dynamics*) runs under
`DomainParallel` unchanged **as long as every operation is per-atom.** If
your integrator needs a *global* scalar, make it DD-aware — in order of
preference:

1. **Reuse a shipped ensemble** when your scheme is Nosé–Hoover / NPT /
   NPH / FIRE-shaped: the coordinator already globalizes its
   thermodynamic quantities.

2. **Compute the global scalar in a `HookScope.GLOBAL` hook.** This is the
   sanctioned public seam. A hook whose `scope` is `HookScope.GLOBAL`
   receives the *full gathered system on every rank*, so it can compute
   any global quantity identically everywhere and hand it to the
   integrator, which then reads a plain Python value (no shard math):

   ```python
   from nvalchemi.dynamics.base import DynamicsStage
   from nvalchemi.distributed import HookScope

   class GlobalKineticEnergyHook:
       """Whole-system KE, computed identically on every rank.

       ``scope = GLOBAL`` makes DomainParallel gather the full system onto
       every rank before the hook runs, so every rank writes the same
       value onto the integrator it wraps.
       """
       stage = DynamicsStage.BEFORE_STEP
       scope = HookScope.GLOBAL
       frequency = 1

       def __init__(self, integrator):
           self._integrator = integrator

       def __call__(self, ctx, stage):
           b = ctx.batch  # the FULL system, identical on every rank
           ke = 0.5 * (b.atomic_masses * (b.velocities**2).sum(-1)).sum()
           self._integrator.global_kinetic_energy = ke

   # Register on the OUTER DomainParallel so the GLOBAL gather fires;
   # the integrator reads ``self.global_kinetic_energy`` in post_update.
   dynamics = DomainParallel(
       dynamics=my_integrator,
       config=domain_cfg,
       n_steps=n_steps,
       hooks=[GlobalKineticEnergyHook(my_integrator)],
   )
   ```

   Because every rank computed the same value from the same gathered
   system, the integrator stays in lockstep. A `GLOBAL` hook gathers the
   whole system when it fires, so it suits step-boundary couplings — not a
   per-step hot-loop scalar on very large systems.

3. **Read `energy` directly** — the forward already reduces per-system
   energy to its global value and replicates it to every rank, so
   `batch.energy` is global. Never re-sum it.

**Don'ts under DD:**

- Don't call `torch.distributed.all_reduce` yourself over the world
  group. Route global reductions through the framework (a shipped
  ensemble, or a `GLOBAL` hook) so they stay correct under whichever
  strategy is active — a hand-rolled reduction can double-count on a
  replicated layout, where every rank already holds the full data.
- Don't try to read owned + ghost atoms inside `pre_update` /
  `post_update` — you only ever get owned atoms; the halo view lives
  inside the forward.
- Don't make a per-rank control-flow decision (like "converged") from a
  local scalar. `DomainParallel` already reduces convergence mesh-wide; a
  divergent local decision desyncs collectives.

## Next steps

- The {doc}`ShardTensor walkthrough <distributed_shardtensor>`
  explains how per-atom tensors flow through halo exchange and
  per-system reductions, and what subclass propagation guarantees the
  framework relies on.
- The {doc}`Bring-your-own-model walkthrough <distributed_byo>`
  shows how to declare an
  {py:class}`~nvalchemi.distributed.spec.MLIPSpec` for a new wrapper,
  validate it via {py:func}`trace_and_validate`, and persist the
  resulting spec for production use.
- The runnable example in
  `examples/distributed/03_mace_nvt_distributed.py` is the
  end-to-end version of the snippet above.
