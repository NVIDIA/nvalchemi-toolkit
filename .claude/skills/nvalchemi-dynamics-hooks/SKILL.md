---
name: nvalchemi-dynamics-hooks
description: How to use and write dynamics hooks — callbacks that observe or modify batch state at specific points during each simulation step.
---

# nvalchemi Hooks

## Overview

Hooks are callbacks that fire at specific points during each workflow step.
They observe or modify batch state without changing the engine itself.
The hook system is framework-wide: the same `Hook` protocol works for
dynamics and custom pipelines — only the stage enum changes.

```python
from nvalchemi.hooks import Hook, HookContext, HookRegistryMixin
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import (
    BiasedPotentialHook,
    EnergyDriftMonitorHook,
    LoggingHook,
    MaxForceClampHook,
    NaNDetectorHook,
    ProfilerHook,
    SnapshotHook,
    WrapPeriodicHook,
)
```

---

## Hook protocol

Any object with these attributes satisfies the `Hook` protocol (runtime-checkable):

```python
class Hook(Protocol):
    frequency: int   # execute every N steps (1 = every step)
    stage: Enum      # stage enum value at which this hook runs

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Called with a context snapshot and the current stage."""
        ...
```

A hook fires when `step_count % hook.frequency == 0` (so all hooks fire at step 0).

**HookContext** — unified snapshot of workflow state:

```python
@dataclass
class HookContext:
    batch: Batch              # current batch (all engines)
    step_count: int           # current step number
    model: BaseModelMixin | None = None
    converged_mask: Tensor | None = None      # dynamics only
    global_rank: int = 0                      # distributed rank
```

Access batch data via `ctx.batch`, step info via `ctx.step_count`, etc.

---

## Execution stages

### Dynamics — `DynamicsStage`

Each `step()` call fires hooks at 9 stages in this order:

```
BEFORE_STEP (0)
  BEFORE_PRE_UPDATE (1)  →  pre_update()  →  AFTER_PRE_UPDATE (2)
  BEFORE_COMPUTE (3)     →  compute()      →  AFTER_COMPUTE (4)
  BEFORE_POST_UPDATE (5) →  post_update()  →  AFTER_POST_UPDATE (6)
AFTER_STEP (7)
ON_CONVERGE (8)   ← only if convergence detected
```

**Stage selection guidelines (dynamics):**

| Goal | Stage |
|------|-------|
| Modify forces/energy after model | `DynamicsStage.AFTER_COMPUTE` |
| Observe final state (logging, snapshots) | `DynamicsStage.AFTER_STEP` |
| Wrap positions after velocity update | `DynamicsStage.AFTER_POST_UPDATE` |
| Instrument timing / profiling | `DynamicsStage.BEFORE_STEP` |
| React to convergence | `DynamicsStage.ON_CONVERGE` |

---

## Registering hooks

```python
from nvalchemi.dynamics.demo import DemoDynamics

# At construction
dynamics = DemoDynamics(
    model=model,
    n_steps=1000,
    dt=0.5,
    hooks=[
        MaxForceClampHook(max_force=10.0),
        LoggingHook(frequency=100),
    ],
)

# After construction
dynamics.register_hook(NaNDetectorHook(frequency=10))
```

Multiple hooks at the same stage fire in registration order.

**Stage type enforcement**: each engine declares `_stage_type` to restrict
which enum types are accepted. For example, `BaseDynamics` sets
`_stage_type = DynamicsStage`.

---

## Built-in hooks

### Safety hooks (stage: AFTER_COMPUTE)

**NaNDetectorHook** — detect NaN/Inf in forces and energy.

```python
NaNDetectorHook(
    frequency=1,              # check every N steps
    extra_keys=["stress"],    # additional batch keys to check (optional)
)
```

**MaxForceClampHook** — clamp per-atom force vectors to a maximum L2 norm.

```python
MaxForceClampHook(
    max_force=10.0,     # max force norm (eV/A)
    frequency=1,
    log_clamps=False,   # log a warning when clamping occurs
)
```

### Bias hook (stage: AFTER_COMPUTE)

**BiasedPotentialHook** — add an external bias potential for enhanced sampling.

```python
def my_bias(batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (bias_energy [B, 1], bias_forces [V, 3])."""
    bias_e = torch.zeros(batch.num_graphs, 1, device=batch.device)
    bias_f = torch.zeros_like(batch.positions)
    # ... compute bias ...
    return bias_e, bias_f

BiasedPotentialHook(
    bias_fn=my_bias,
    frequency=1,
)
```

### Observer hooks (stage: AFTER_STEP)

**LoggingHook** — log scalar observables.

```python
LoggingHook(
    frequency=100,
    backend="loguru",              # "loguru", "csv", "tensorboard", "custom"
    log_path="md_log.csv",         # for file-based backends
    custom_scalars={               # additional scalars to log
        "max_velocity": lambda ctx, stage: ctx.batch.velocities.norm(dim=-1).max().item(),
    },
    writer_fn=None,                # custom writer for "custom" backend
)
```

**SnapshotHook** — save full batch state to a `DataSink`.

```python
from nvalchemi.dynamics.sinks import GPUBuffer, HostMemory, ZarrData

SnapshotHook(
    sink=ZarrData("trajectory.zarr", capacity=10000),
    frequency=10,
)
```

**EnergyDriftMonitorHook** — track total energy drift.

```python
EnergyDriftMonitorHook(
    threshold=1e-4,                          # drift threshold
    metric="per_atom_per_step",              # or "absolute"
    action="warn",                           # or "raise"
    frequency=1,
    include_kinetic=True,                    # include kinetic energy
)
```

### Periodic boundary hook (stage: AFTER_POST_UPDATE)

**WrapPeriodicHook** — wrap positions back into the unit cell.

```python
WrapPeriodicHook(frequency=10)  # wrap every 10 steps
```

### Profiling hook (multi-stage, uses plum dispatch)

**ProfilerHook** — NVTX ranges and wall-clock timing. Fires at multiple
stages via `_runs_on_stage` and uses `plum.dispatch` to support
dynamics and custom workflows with appropriate domain annotations.

```python
ProfilerHook(
    frequency=1,
    enable_nvtx=True,                        # NVTX annotation for Nsight Systems
    enable_timer=True,                       # wall-clock step timing
    timer_backend="cuda_event",              # or "perf_counter"
)
```

---

## Writing a custom hook

### Option 1: Simple single-stage hook (dynamics)

Implement the protocol directly — no inheritance needed.

```python
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import HookContext

class TemperatureLogger:
    stage = DynamicsStage.AFTER_STEP
    frequency = 50

    def __call__(self, ctx: HookContext, stage: DynamicsStage) -> None:
        ke = ctx.batch.kinetic_energies.sum()
        n_atoms = ctx.batch.num_nodes
        temp = 2.0 * ke / (3.0 * n_atoms * 8.617e-5)  # kB in eV/K
        print(f"Step {ctx.step_count}: T = {temp:.1f} K")
```

### Option 2: Multi-stage hook with `_runs_on_stage`

Fire at multiple stages by defining `_runs_on_stage(stage) -> bool`:

```python
from enum import Enum
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import HookContext

class StepTimerHook:
    stage = DynamicsStage.BEFORE_STEP  # primary stage (protocol compliance)
    frequency = 1

    def __init__(self):
        self._stages = {DynamicsStage.BEFORE_STEP, DynamicsStage.AFTER_STEP}
        self._t0 = None

    def _runs_on_stage(self, stage: Enum) -> bool:
        return stage in self._stages

    def __call__(self, ctx: HookContext, stage: DynamicsStage) -> None:
        import time
        if stage == DynamicsStage.BEFORE_STEP:
            self._t0 = time.perf_counter()
        elif stage == DynamicsStage.AFTER_STEP and self._t0 is not None:
            dt = time.perf_counter() - self._t0
            print(f"Step {ctx.step_count}: {dt*1000:.1f} ms")
```

### Option 3: Cross-category hook with `plum` dispatch

For hooks that work with multiple stage enum types (e.g. `DynamicsStage` and
a custom enum), use `plum.dispatch` to overload `__call__` with different
stage types:

```python
from enum import Enum
from plum import dispatch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import HookContext

# Example custom stage enum for a hypothetical pipeline
class MyPipelineStage(Enum):
    BEFORE_PROCESS = 0
    AFTER_PROCESS = 1

class UniversalLoggerHook:
    stage = DynamicsStage.AFTER_STEP
    frequency = 10

    def __init__(self):
        self._stages = {DynamicsStage.AFTER_STEP, MyPipelineStage.AFTER_PROCESS}

    def _runs_on_stage(self, stage: Enum) -> bool:
        return stage in self._stages

    @dispatch
    def __call__(self, ctx: HookContext, stage: DynamicsStage) -> None:
        fmax = ctx.batch.forces.norm(dim=-1).max().item()
        print(f"[dynamics] step {ctx.step_count}: fmax={fmax:.4f}")

    @dispatch
    def __call__(self, ctx: HookContext, stage: MyPipelineStage) -> None:
        print(f"[pipeline] step {ctx.step_count}: processed")

    @dispatch
    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        print(f"[custom] step {ctx.step_count}: stage={stage.name}")
```

The built-in `ProfilerHook` uses exactly this pattern to instrument
dynamics and custom workflows with appropriate NVTX domain annotations.

---

## Hook ordering recommendations

Register hooks in this order for correct behavior:

```python
hooks = [
    # 1. Bias (modifies forces/energy)
    BiasedPotentialHook(bias_fn=my_bias),
    # 2. Safety (clamp after all force modifications)
    MaxForceClampHook(max_force=10.0),
    # 3. NaN detection (check final forces)
    NaNDetectorHook(),
    # 4. Periodic wrapping
    WrapPeriodicHook(frequency=10),
    # 5. Observers (read final state)
    LoggingHook(frequency=100),
    SnapshotHook(sink=my_sink, frequency=50),
    EnergyDriftMonitorHook(threshold=1e-4),
    # 6. Profiling
    ProfilerHook(),
]

dynamics = DemoDynamics(model=model, n_steps=10000, dt=0.5, hooks=hooks)
```

---

## Complete example

```python
import torch
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.demo import DemoModelWrapper
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import HookContext
from nvalchemi.dynamics.hooks import MaxForceClampHook, NaNDetectorHook

# Custom hook
class StepPrinter:
    stage = DynamicsStage.AFTER_STEP
    frequency = 10

    def __call__(self, ctx: HookContext, stage: DynamicsStage) -> None:
        fmax = ctx.batch.forces.norm(dim=-1).max().item()
        print(f"Step {ctx.step_count}: fmax={fmax:.4f}")

# Setup
model = DemoModelWrapper()
dynamics = DemoDynamics(
    model=model,
    n_steps=100,
    dt=0.5,
    hooks=[
        MaxForceClampHook(max_force=10.0),
        NaNDetectorHook(),
        StepPrinter(),
    ],
)

data = AtomicData(
    atomic_numbers=torch.tensor([6, 6, 8], dtype=torch.long),
    positions=torch.randn(3, 3),
)
batch = Batch.from_data_list([data])
batch.forces = torch.zeros(3, 3)
batch.energy = torch.zeros(1, 1)

dynamics.run(batch)
```
