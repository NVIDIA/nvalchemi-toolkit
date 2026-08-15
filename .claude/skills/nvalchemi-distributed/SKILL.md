---
name: nvalchemi-distributed
description: How to run domain-decomposed (multi-GPU) MLIP simulations with DomainParallel — choose between the halo and graph-partition strategies, author a distribution_spec so a bring-your-own model runs under domain decomposition, and write a custom dynamics integrator that stays correct across ranks.
---

# nvalchemi Distributed (Domain Decomposition)

## Overview

Domain decomposition (DD) splits *one* atomic system across several GPUs so a
simulation that doesn't fit — or doesn't run fast enough — on a single card can
scale out. The **same** model wrapper, hooks, and integrators you use
single-process run unchanged; you add one wrapper around each.

- {class}`~nvalchemi.distributed.DomainParallel` wraps any
  {class}`~nvalchemi.dynamics.base.BaseDynamics` integrator/optimizer and drives
  it across the mesh.
- Under the hood it wraps the model in a `DistributedModel`, which reads the
  model's `distribution_spec` to know how to shard/gather each field.

```python
from nvalchemi.distributed import (
    DistributedManager, DomainConfig, DomainParallel, HookScope,
)
```

Launch with **torchrun** (or SLURM): DD is one process per GPU.

```bash
torchrun --nproc_per_node=4 my_distributed_md.py
```

There are four things you may need to do. Pick the section you need:

1. **Run a shipped model under DD** → §1
2. **Choose halo vs graph-partition** → §2
3. **Make your own model run under DD** (author a `distribution_spec`) → §3
4. **Write a custom integrator that stays correct under DD** → §4

---

## 1. Run a domain-decomposed model

Bootstrap the process group + mesh with `DistributedManager`, wrap the model as
usual, then wrap the integrator in `DomainParallel`. Build the full system on
rank 0, `partition()` it, and `run()`.

```python
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed import (
    DistributedManager, DomainConfig, DomainParallel, HookScope,
)
from nvalchemi.dynamics import NVTLangevin, HostMemory
from nvalchemi.dynamics.hooks import SnapshotHook
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.mace import MACEWrapper

# 1. Bootstrap (reads RANK / WORLD_SIZE / LOCAL_RANK from torchrun).
DistributedManager.initialize()
dm = DistributedManager()
mesh = dm.initialize_mesh(mesh_shape=(dm.world_size,), mesh_dim_names=("domain",))
device = torch.device(dm.device)

# 2. Wrap the model — identical to single-process.
wrapper = MACEWrapper.from_checkpoint("medium-mpa-0", device=device).eval()

# 3. Build the inner integrator (its NeighborListHook is an INNER hook).
integrator = NVTLangevin(
    model=wrapper, dt=1.0, temperature=300.0, friction=0.01, n_steps=200,
    hooks=[NeighborListHook(wrapper.model_config.neighbor_config, skin=0.5,
                            stage=DynamicsStage.BEFORE_COMPUTE)],
)

# 4. Trajectory snapshot: gather to rank 0 (an OUTER hook).
snapshot = SnapshotHook(sink=HostMemory(capacity=201), frequency=10)
snapshot.scope = HookScope.RANK_ZERO

# 5. Wrap + run. cutoff = wrapper.cutoff makes the halo width exact.
domain_cfg = DomainConfig(cutoff=float(wrapper.cutoff), skin=0.5, mesh=mesh)
with DomainParallel(dynamics=integrator, config=domain_cfg,
                    n_steps=200, hooks=[snapshot]) as dynamics:
    full_batch = build_full_system(device) if dm.rank == 0 else None
    owned = dynamics.partition(full_batch)   # returns THIS rank's owned atoms
    dynamics.run(owned)

DistributedManager.cleanup()
```

Key points:

- `partition()` takes the full system on **rank 0** (`None` elsewhere) and returns
  each rank's owned `Batch`. `run()` loops `step()` for `n_steps`.
- `with DomainParallel(...)` makes teardown exception-safe; keep the process-group
  lifecycle (`initialize` / `cleanup`) at launcher scope.
- **The initial system** (`build_full_system`, built on rank 0) is a `Batch` whose
  per-atom fields must include everything the integrator reads. For MD that means
  `positions`, `atomic_numbers`, `atomic_masses`, `cell`, `pbc`, **and
  `velocities`**. Masses/positions/cell/pbc are `AtomicData(...)` constructor args,
  but **velocities are attached separately** — `data.add_node_property("velocities",
  v)` — then `Batch.from_data_list([data])`. `NVTLangevin` / `NVE` read
  `batch.velocities` + `batch.atomic_masses` and crash at step 0 if they're absent:

  ```python
  data = AtomicData(positions=pos, atomic_numbers=z, atomic_masses=m,
                    cell=box.unsqueeze(0), pbc=pbc)     # (masses in the ctor)
  data.add_node_property("velocities", v)               # velocities added after
  full = Batch.from_data_list([data], device=device)
  ```

- **Hook placement:** neighbor-list / compute hooks go on the **inner** integrator
  (they fire on the padded compute view); trajectory/logging hooks go on the
  **outer** `DomainParallel` with `HookScope.RANK_ZERO` (gather to rank 0) or
  `HookScope.GLOBAL` (gather to every rank).

Runnable references: `examples/distributed/03_mace_nvt_distributed.py` (NVT),
`06_mace_npt_distributed.py` (NPT), `07_fire_nvt_dd.py` (FIRE + 2-D pipeline×DD).

---

## 2. Choose the strategy: halo vs graph-partition

Two strategies ship, selected with `DomainConfig.strategy`
({class}`~nvalchemi.distributed.config.StrategyKind`). The default is halo.

| | **Halo** (default) | **Graph-partition** |
|---|---|---|
| Split by | space (spatial domains) | atom index (balanced node blocks) |
| Each rank holds | owned atoms + a ghost halo | the **full geometry** (replicated) + owns a node slice |
| Best for | models with a bounded cutoff you can hand a padded view (MACE, NequIP, LJ, Ewald, PME) | models that build their own neighbour list inside `forward` (UMA / eSCN-family) |
| Comms | one halo exchange per step | per-layer feature all-gather + reduce-scatter |
| Cell / migration | cell load-bearing; atoms migrate | cell is a plain input; no migration |

```python
from nvalchemi.distributed.config import StrategyKind

# Halo (default):
cfg = DomainConfig(cutoff=wrapper.cutoff, skin=0.5, mesh=mesh)
# Graph-partition:
cfg = DomainConfig(cutoff=wrapper.cutoff, skin=0.5, mesh=mesh,
                   strategy=StrategyKind.GRAPH_PARTITION)
```

**Rule of thumb:** use **halo** for a short-range MPNN with a real cutoff (it
scales to large N as a capacity play); use **graph-partition** for a model that
rebuilds its own graph and can't take a pre-padded view, or to fit a single
system past one GPU's memory.

**Shipped presets** (from `nvalchemi.distributed.spec`) — a wrapper declares one
of these as its `distribution_spec` (§3):

| Preset | Strategy |
|---|---|
| `SPEC_MPNN_HALO` | halo — scatter-heavy MPNNs (MACE, NequIP, generic) **and any differentiable, autograd-force model** (incl. a BYO pure-PyTorch pair potential) |
| `SPEC_LJ_HALO` | halo — the *shipped* Lennard-Jones wrapper's **opaque Warp kernel** (carries `OpAdapter`s) |
| `SPEC_UMA_HALO` | halo — UMA/eSCN (local scatter) |
| `SPEC_EWALD_HALO` / `SPEC_PME_HALO` | halo — long-range electrostatics |
| `SPEC_DFTD3_HALO` | halo — DFTD3 dispersion |
| `SPEC_MPNN_GP` | graph-partition — MPNNs under node partition |

Note: `SPEC_LJ_HALO` is for the shipped LJ's opaque kernel, **not** a BYO pair
potential — if you write a differentiable pair potential in plain PyTorch, its
forces come from autograd and the correct preset is `SPEC_MPNN_HALO`.

**Compiled runs:** add `compile=True` to the `DomainConfig`. The forward is
fixed-shape (padded to per-rank caps), so after a short warm-up the trajectory
runs recompile-free — MD-ready. Supported by the MACE (incl. cuEquivariance),
AIMNet2, and UMA wrappers.

---

## 3. Add a `distribution_spec` (bring your own model)

To make an arbitrary wrapped model (see `nvalchemi-model-wrapping`) run under DD,
give it a `distribution_spec` that returns an
{class}`~nvalchemi.distributed.spec.MLIPSpec`. The framework reads it — the
wrapper's `forward` stays free of any distributed code. Four steps:

**(1) Wrap** your model with `BaseModelMixin` as usual.

**(2) Declare a spec.** If your model is a scatter-heavy MPNN with autograd
forces, return a preset:

```python
from nvalchemi.distributed.spec import SPEC_MPNN_HALO

class MyWrapper(nn.Module, BaseModelMixin):
    def distribution_spec(self, strategy=None):
        return SPEC_MPNN_HALO
```

**Getting the wrapper right under halo.** For an autograd/scatter model that
returns `SPEC_MPNN_HALO`, four things in your `forward` / `adapt_input` decide
correctness — miss them and you get *silently wrong* energies, not a crash:

- **Emit per-atom energies over ALL local atoms (owned + ghost).** Return an
  `(n_local,)` atomic-energy tensor and let the framework reduce ghost rows; don't
  pre-slice to owned.
- **Rebind in-place scatters:** write `ae = ae.scatter_add_(0, recv, e_edge)` and
  use the *returned* tensor — under DD the dispatch returns a new, cross-rank
  halo-corrected tensor; the pre-rebind value is wrong.
- **Filter neighbor-list sentinels in `adapt_input`:** halo padding can leave
  receiver indices ≥ `n_atoms`; drop them with
  `valid = (edge_index[0] < n) & (edge_index[1] < n)`.
- **Periodic systems:** compute displacements with the minimum-image convention
  from the cell (`frac = rij @ inv_cell; frac -= frac.round(); rij = frac @ cell`).
  It is a no-op on halo ghost edges (the ghost already sits at its image) and
  correct single-process, so no explicit shift vectors are needed.

Your wrapper's `ModelConfig` should declare forces as autograd —
`autograd_outputs=frozenset({"forces"})`, `autograd_inputs=frozenset({"positions"})`,
`neighbor_config=NeighborConfig(cutoff=…, format=NeighborListFormat.COO)`. See
`nvalchemi-model-wrapping` and `examples/distributed/04_byo_pytorch_mpnn.py` for the
full wrapper.

If the model calls a **custom op** (a Warp/Triton kernel, a fused scatter) the
tracer can't see through, build the spec explicitly and declare per-op behaviour
with an adapter:

```python
import torch
from nvalchemi.distributed.ops import HaloStoragePolicy, ScatterOutputs
from nvalchemi.distributed.spec import (
    DistributionSpec, MLIPSpec, OpAdapter, OutputKind,
)

def _build_spec():
    return MLIPSpec(
        distribution=DistributionSpec(
            policy=HaloStoragePolicy(),        # halo layout
            custom_ops=(                       # one OpAdapter per opaque kernel
                OpAdapter(
                    op=torch.ops.mypkg.my_energy_op.default,
                    arg_transforms={},                       # inputs already halo-padded
                    output_transforms={0: ScatterOutputs()}, # halo-correct output 0
                ),
            ),
        ),
        output_kinds={"energy": OutputKind.PER_GRAPH,
                      "forces": OutputKind.PER_NODE},
    )
```

- **`policy`** — storage layout: `HaloStoragePolicy` (halo) or
  `GraphParallelPolicy` (node-partition). `output_kinds` tags each named output's
  shape (`PER_NODE`, `PER_GRAPH`, `GLOBAL`) so consolidation knows whether to
  halo-reverse, all-reduce, or pass it through.
- **Adapters** declare how a black-box op distributes. `OpAdapter(op=,
  arg_transforms={i: …}, output_transforms={i: …})` goes in
  `DistributionSpec.custom_ops`; `MethodAdapter` (swap a third-party module
  method) and `PythonAdapter` / `JitAdapter` (replace a helper) go in
  `third_party_helpers`. Transforms — `ScatterOutputs` / `GatherInputs` /
  `SliceOwned` / `AllReduceSum` — say what to do per arg/output.

**(3) Validate.** `trace_and_validate` runs the model under a simulated 2-rank
mesh, checks force-equivalence vs single-process, and — when a bare op needs an
adapter — tells you which one to add:

```python
from nvalchemi.distributed.validate import trace_and_validate

report = trace_and_validate(MyWrapper(model), world_size=2)
assert report.ok, report.next_action   # e.g. "add OpAdapter(..., ScatterOutputs())"
```

**(4) Persist.** `MLIPSpec.save(path)` / `MLIPSpec.load(path)` round-trip the spec
so production runs skip re-discovery.

Full walkthroughs: `examples/distributed/04_byo_pytorch_mpnn.py` (autograd MPNN,
no custom op) and `05_byo_graph_transformer.py` (a custom-op kernel needing an
`OpAdapter`). Deep reference: the *Bring Your Own Model* user guide.

---

## 4. Bring your own dynamics under DD

A custom {class}`~nvalchemi.dynamics.base.BaseDynamics` integrator (see
`nvalchemi-dynamics-implementation`) runs under `DomainParallel` **unchanged as
long as every operation is per-atom.** The one rule:

> Under DD each rank's `pre_update` / `post_update` sees **only its owned
> atoms**. Per-atom math is correct as-is; any **global** reduction is not.

**Safe locally (no change needed):** position/velocity integration, per-atom
Langevin friction + noise, applying forces. `NVE` and `NVTLangevin` are exact
under DD for exactly this reason. Reading `batch.energy` is also fine — the
forward already reduces + replicates it globally.

**Must be globalized (a shard `.sum()` / `.max()` / `.dot()` is WRONG):** total
kinetic energy, temperature, degrees of freedom, FIRE dot-products (`v·f`,
`v·v`, `f·f`), a global convergence test, barostat pressure.

For the shipped ensembles this is automatic: Nosé–Hoover, `NPT`, `NPH`, and
`FIRE` declare their global quantities as intent and `DomainParallel`'s
coordinator reduces them — **prefer reusing them.** For a genuinely custom global
scalar, compute it in a `HookScope.GLOBAL` hook, which sees the full gathered
system on every rank:

```python
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.distributed import HookScope

class GlobalKineticEnergyHook:
    stage = DynamicsStage.BEFORE_STEP
    scope = HookScope.GLOBAL       # gather the FULL system to every rank first
    frequency = 1
    def __init__(self, integrator):
        self._integrator = integrator
    def __call__(self, ctx, stage):
        b = ctx.batch              # full system, identical on every rank
        self._integrator.global_ke = 0.5 * (
            b.atomic_masses * (b.velocities**2).sum(-1)
        ).sum()

# Register on the OUTER DomainParallel; the integrator reads self.global_ke.
dynamics = DomainParallel(dynamics=my_integrator, config=cfg, n_steps=n,
                          hooks=[GlobalKineticEnergyHook(my_integrator)])
```

Because every rank computed the same value from the same gathered system, the
integrator stays in lockstep.

**Don'ts:**

- Don't hand-roll `torch.distributed.all_reduce` over the world group — route
  reductions through a shipped ensemble or a `GLOBAL` hook so they stay correct
  under a replicated layout (where every rank holds the full data).
- Don't read owned + ghost atoms inside `pre_update` / `post_update` — you only
  ever get owned atoms.
- Don't make a per-rank control-flow decision (like "converged") from a local
  scalar — `DomainParallel` already reduces convergence mesh-wide; a divergent
  local decision desyncs collectives.

---

## Reference: key imports

| Symbol | Import from | Role |
|---|---|---|
| `DistributedManager` | `nvalchemi.distributed` | Process-group + mesh bootstrap |
| `DomainParallel` | `nvalchemi.distributed` | Wrap an integrator to run across the mesh |
| `DomainConfig` | `nvalchemi.distributed` | `cutoff` / `skin` / `mesh` / `strategy` / `compile` |
| `StrategyKind` | `nvalchemi.distributed.config` | `HALO` (default) vs `GRAPH_PARTITION` |
| `HookScope` | `nvalchemi.distributed` | `LOCAL` / `GLOBAL` / `RANK_ZERO` hook gather |
| `MLIPSpec` / `DistributionSpec` | `nvalchemi.distributed.spec` | The `distribution_spec` a BYO model returns |
| `HaloStoragePolicy` / `GraphParallelPolicy` | `nvalchemi.distributed.ops` / `.spec` | Per-field storage layout (the two strategies' policies) |
| `OpAdapter` / `MethodAdapter` / `PythonAdapter` | `nvalchemi.distributed` | Declare how a black-box op/method/helper distributes |
| `OutputKind`, `ScatterOutputs`, `GatherInputs` | `nvalchemi.distributed` (`.ops`) | Output shape tags + per-arg/output transforms |
| `CompilePolicy` | `nvalchemi.distributed` | Opt a BYO model into a compiled DD forward |
| `trace_and_validate` | `nvalchemi.distributed.validate` | Simulate a 2-rank mesh + check force-equivalence |
| `SPEC_MPNN_HALO`, `SPEC_MPNN_GP`, … | `nvalchemi.distributed.spec` | Shipped presets |

## See also

- User guide: *Overview: Domain Decomposition*, *Bring Your Own Model*,
  *ShardTensor*, and the *Architecture & design* deep dive.
- Examples: `examples/distributed/03`–`07`.
- Related skills: `nvalchemi-model-wrapping` (wrap a model),
  `nvalchemi-dynamics-implementation` (write an integrator),
  `nvalchemi-dynamics-hooks` (hooks), `nvalchemi-dynamics-api` (run/scale).
