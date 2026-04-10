---
name: nvalchemi-dynamics-implementation
description: How to implement a dynamics integrator by subclassing BaseDynamics and overriding pre_update() and post_update() methods.
---

# nvalchemi Dynamics Implementation

## Overview

To implement a dynamics class (integrator) in `nvalchemi`, subclass `BaseDynamics`
and override two methods: `pre_update()` and `post_update()`. The base class handles
the model forward pass, hook dispatch, convergence checking, and the step/run loop.

```python
from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook
from nvalchemi.data import Batch
```

---

## Step execution flow

Each call to `step(batch)` executes:

```
1. BEFORE_STEP hooks
2. BEFORE_PRE_UPDATE hooks  →  pre_update(batch)  →  AFTER_PRE_UPDATE hooks
3. BEFORE_COMPUTE hooks     →  compute(batch)      →  AFTER_COMPUTE hooks
4. BEFORE_POST_UPDATE hooks →  post_update(batch)  →  AFTER_POST_UPDATE hooks
5. AFTER_STEP hooks
6. Check convergence → ON_CONVERGE hooks if converged
7. Increment step_count
```

- `pre_update()` and `post_update()` run inside `torch.no_grad()`
- `compute()` calls the model forward pass and writes forces/energy to the batch in-place
- You implement `pre_update()` and `post_update()`; everything else is inherited

---

## Implementation guide

### 1. Define the class

Set `__needs_keys__` (model outputs your integrator requires) and `__provides_keys__`
(state your integrator produces).

```python
class MyDynamics(BaseDynamics):
    __needs_keys__: set[str] = {"forces"}
    __provides_keys__: set[str] = {"velocities", "positions"}
```

### 2. Implement `__init__`

Store integrator parameters. Always call `super().__init__()` and forward `**kwargs`
(needed for cooperative multiple inheritance with the communication mixin).

```python
def __init__(
    self,
    model: BaseModelMixin,
    n_steps: int,
    dt: float = 1.0,
    hooks: list[Hook] | None = None,
    convergence_hook: ConvergenceHook | dict | None = None,
    **kwargs: Any,
) -> None:
    super().__init__(
        model=model,
        hooks=hooks,
        convergence_hook=convergence_hook,
        n_steps=n_steps,
        **kwargs,
    )
    self.dt = dt
```

**BaseDynamics constructor parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `BaseModelMixin` | The neural network potential |
| `hooks` | `list[Hook] \| None` | Hooks to register (organized by `stage`) |
| `convergence_hook` | `ConvergenceHook \| dict \| None` | Convergence detection |
| `n_steps` | `int \| None` | Default step count for `run()` |
| `exit_status` | `int` | Status value for graduated samples (default: 1) |
| `**kwargs` | `Any` | Forwarded to communication mixin |

### 3. Implement `pre_update(batch)`

Update **positions** based on current velocities and forces. Modify the batch in-place.

```python
def pre_update(self, batch: Batch) -> None:
    positions = batch.positions       # [V, 3]
    velocities = batch.velocities     # [V, 3]
    forces = batch.forces             # [V, 3] or None
    masses = batch.atomic_masses.unsqueeze(-1)  # [V] -> [V, 1]

    with torch.no_grad():
        if forces is not None and not torch.all(forces == 0):
            accelerations = forces / masses
            # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2
            positions.add_(velocities * self.dt + 0.5 * accelerations * self.dt**2)
        else:
            # First step fallback (no forces yet)
            positions.add_(velocities * self.dt)
```

### 4. Implement `post_update(batch)`

Update **velocities** based on new forces (computed between `pre_update` and `post_update`
by the inherited `compute()` method). Modify the batch in-place.

```python
def post_update(self, batch: Batch) -> None:
    velocities = batch.velocities     # [V, 3]
    forces = batch.forces             # [V, 3]
    masses = batch.atomic_masses.unsqueeze(-1)

    with torch.no_grad():
        new_accelerations = forces / masses
        # v(t+dt) = v(t) + a(t+dt)*dt
        velocities.add_(new_accelerations * self.dt)
```

---

## Inherited methods (do NOT override)

| Method | Description |
|--------|-------------|
| `compute(batch)` | Model forward pass → validates outputs → writes forces/energy to batch |
| `step(batch)` | Full step with hook dispatch (see flow above) |
| `run(batch, n_steps=None)` | Loop calling `step()` for `n_steps` iterations |
| `register_hook(hook)` | Register a hook at its declared stage |
| `_check_convergence(batch)` | Check convergence criteria, return converged indices |
| `_validate_model_outputs(outputs)` | Verify `__needs_keys__` are present in model output |

---

## Inherited attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | `BaseModelMixin` | The wrapped model |
| `step_count` | `int` | Current step (starts at 0, incremented after each step) |
| `hooks` | `dict[DynamicsStage, list[Hook]]` | Registered hooks by stage |
| `convergence_hook` | `ConvergenceHook \| None` | Convergence detector |
| `n_steps` | `int \| None` | Default step count |
| `exit_status` | `int` | Status threshold for graduated samples |
| `model_is_conservative` | `bool` | Whether forces use autograd |

---

## Usage

```python
from nvalchemi.models.demo import DemoModelWrapper
from nvalchemi.data import AtomicData, Batch
import torch

# Create model and dynamics
model = DemoModelWrapper()
dynamics = MyDynamics(model=model, n_steps=100, dt=0.5)

# Create batch
data = AtomicData(
    atomic_numbers=torch.tensor([6, 6, 8], dtype=torch.long),
    positions=torch.randn(3, 3),
)
batch = Batch.from_data_list([data])

# Initialize required fields (forces/energy must exist for copy_())
batch.forces = torch.zeros(3, 3)
batch.energy = torch.zeros(1, 1)

# Run
result = dynamics.run(batch)
# Or step-by-step
dynamics.step(batch)
```

---

## Complete example: Velocity Verlet

This mirrors `DemoDynamics`, the reference implementation.

```python
from __future__ import annotations
from typing import Any, TYPE_CHECKING
import torch
from nvalchemi.data import Batch
from nvalchemi.dynamics.base import BaseDynamics, ConvergenceHook

if TYPE_CHECKING:
    from nvalchemi.dynamics.base import Hook
    from nvalchemi.models.base import BaseModelMixin


class VelocityVerlet(BaseDynamics):
    """Velocity Verlet integrator."""

    __needs_keys__: set[str] = {"forces"}
    __provides_keys__: set[str] = {"velocities", "positions"}

    def __init__(
        self,
        model: BaseModelMixin,
        n_steps: int,
        dt: float = 1.0,
        hooks: list[Hook] | None = None,
        convergence_hook: ConvergenceHook | dict | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model, hooks=hooks, convergence_hook=convergence_hook,
            n_steps=n_steps, **kwargs,
        )
        self.dt = dt
        self._prev_accelerations: torch.Tensor | None = None

    def pre_update(self, batch: Batch) -> None:
        """x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt^2"""
        positions = batch.positions
        velocities = batch.velocities
        forces = batch.forces
        masses = batch.atomic_masses.unsqueeze(-1)

        with torch.no_grad():
            if forces is not None and not torch.all(forces == 0):
                accelerations = forces / masses
                self._prev_accelerations = accelerations.clone()
                positions.add_(velocities * self.dt + 0.5 * accelerations * self.dt**2)
            else:
                positions.add_(velocities * self.dt)

    def post_update(self, batch: Batch) -> None:
        """v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt"""
        velocities = batch.velocities
        forces = batch.forces
        masses = batch.atomic_masses.unsqueeze(-1)

        with torch.no_grad():
            new_accelerations = forces / masses
            if self._prev_accelerations is not None:
                velocities.add_(
                    0.5 * (self._prev_accelerations + new_accelerations) * self.dt
                )
            else:
                velocities.add_(new_accelerations * self.dt)
```

---

## Convergence

Use `ConvergenceHook` to stop early or migrate samples in a pipeline:

```python
from nvalchemi.dynamics.base import ConvergenceHook

hook = ConvergenceHook(
    criteria=[
        {"key": "fmax", "threshold": 0.05},
        {"key": "energy_change", "threshold": 1e-6},
    ],
    source_status=0,   # check samples with this status
    target_status=1,   # migrate converged samples to this status
    frequency=1,       # check every N steps
)

dynamics = MyDynamics(model=model, n_steps=1000, convergence_hook=hook)
```

---

## Composition with FusedStage

Chain multiple dynamics stages that share a single model forward pass:

```python
relax = MyDynamics(model, n_steps=100, dt=0.5)
md = MyDynamics(model, n_steps=500, dt=0.1)

# Compose with + operator
fused = relax + md

# Samples start in relax, converge, then move to md
fused.run(batch)
```

---

## Distributed pipeline

Chain stages across ranks with the `|` operator:

```python
opt_stage = MyDynamics(model, n_steps=100, dt=0.5)   # rank 0
md_stage = MyDynamics(model, n_steps=500, dt=0.1)    # rank 1

pipeline = opt_stage | md_stage

with pipeline:
    pipeline.run()
```
