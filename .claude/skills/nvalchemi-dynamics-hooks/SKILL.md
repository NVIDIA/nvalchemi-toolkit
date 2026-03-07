# nvalchemi Dynamics Hooks

## Overview

Hooks are callbacks that fire at specific points during each dynamics step.
They observe or modify batch state without changing the integrator itself.

```python
from nvalchemi.dynamics.base import HookStageEnum
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
    frequency: int           # execute every N steps (1 = every step)
    stage: HookStageEnum     # when in the step to fire

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Called with the current batch and dynamics instance."""
        ...
```

A hook fires when `step_count % hook.frequency == 0` (so all hooks fire at step 0).

---

## Execution stages

Each `step()` call fires hooks at 9 stages in this order:

```
BEFORE_STEP (0)
  BEFORE_PRE_UPDATE (1)  →  pre_update()  →  AFTER_PRE_UPDATE (2)
  BEFORE_COMPUTE (3)     →  compute()      →  AFTER_COMPUTE (4)
  BEFORE_POST_UPDATE (5) →  post_update()  →  AFTER_POST_UPDATE (6)
AFTER_STEP (7)
ON_CONVERGE (8)   ← only if convergence detected
```

**Stage selection guidelines:**

| Goal | Stage |
|------|-------|
| Modify forces/energies after model | `AFTER_COMPUTE` |
| Observe final state (logging, snapshots) | `AFTER_STEP` |
| Wrap positions after velocity update | `AFTER_POST_UPDATE` |
| Instrument timing / profiling | `BEFORE_STEP` |
| React to convergence | `ON_CONVERGE` |

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

---

## Built-in hooks

### Safety hooks (stage: AFTER_COMPUTE)

**NaNDetectorHook** — detect NaN/Inf in forces and energies.

```python
NaNDetectorHook(
    frequency=1,              # check every N steps
    extra_keys=["stresses"],  # additional batch keys to check (optional)
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
        "max_velocity": lambda batch, dyn: batch.velocities.norm(dim=-1).max().item(),
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

### Profiling hook (stage: BEFORE_STEP)

**ProfilerHook** — NVTX ranges and wall-clock timing.

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

### Option 1: Subclass a base class

Use `_ObserverHook` (read-only, `AFTER_STEP`) or `_PostComputeHook` (modifies batch, `AFTER_COMPUTE`).

```python
from nvalchemi.dynamics.hooks._base import _PostComputeHook

class MyForceScaleHook(_PostComputeHook):
    """Scale all forces by a constant factor."""

    def __init__(self, scale: float, frequency: int = 1) -> None:
        super().__init__(frequency=frequency)
        self.scale = scale

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        batch.forces.mul_(self.scale)
```

### Option 2: Implement the protocol directly

Any object with `frequency`, `stage`, and `__call__` works — no inheritance needed.

```python
class TemperatureLogger:
    stage = HookStageEnum.AFTER_STEP
    frequency = 50

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        ke = batch.kinetic_energies.sum()
        n_atoms = batch.num_nodes
        temp = 2.0 * ke / (3.0 * n_atoms * 8.617e-5)  # kB in eV/K
        print(f"Step {dynamics.step_count}: T = {temp:.1f} K")
```

### Option 3: Use a lambda/closure (for quick one-offs)

```python
from types import SimpleNamespace

log_hook = SimpleNamespace(
    stage=HookStageEnum.AFTER_STEP,
    frequency=100,
    __call__=lambda self, batch, dyn: print(f"Step {dyn.step_count}"),
)
# Note: SimpleNamespace.__call__ doesn't work as method — use a class instead
```

---

## Hook ordering recommendations

Register hooks in this order for correct behavior:

```python
hooks = [
    # 1. Bias (modifies forces/energies)
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
from nvalchemi.dynamics.base import HookStageEnum
from nvalchemi.dynamics.hooks import MaxForceClampHook, NaNDetectorHook

# Custom hook
class StepPrinter:
    stage = HookStageEnum.AFTER_STEP
    frequency = 10

    def __call__(self, batch, dynamics):
        fmax = batch.forces.norm(dim=-1).max().item()
        print(f"Step {dynamics.step_count}: fmax={fmax:.4f}")

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
batch.energies = torch.zeros(1, 1)

dynamics.run(batch)
```
