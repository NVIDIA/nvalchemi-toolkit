<!-- markdownlint-disable MD014 -->

(hooks_guide)=
(dynamics_hooks_guide)=

# Hooks

Hooks let you observe or modify workflow state at specific points in any
engine's execution loop — dynamics simulations, training loops, or custom
pipelines — without touching the engine code itself. They are the primary
extension mechanism for logging, convergence checking, trajectory recording,
and any custom per-step logic.

```{tip}
**AI coding assistant?** Load the ``nvalchemi-dynamics-hooks``
{ref}`agent skill <agent_skills>` for concise instructions on writing
and registering hooks for dynamics simulations.
```

## The Hook protocol

A hook is any object that satisfies the
{py:class}`~nvalchemi.hooks.Hook` protocol (a
`@runtime_checkable` `Protocol`). The required interface is:

| Attribute / Method | Type | Purpose |
|--------------------|------|---------|
| `stage` | `Enum` | Which stage of the execution loop this hook fires at (e.g. `DynamicsStage`) |
| `frequency` | `int` | Execute every *n* steps (1 = every step) |
| `__call__(ctx, stage)` | `None` | The hook's logic, called with a {py:class}`~nvalchemi.hooks.HookContext` or workflow-specific subclass and the current stage |

The `Hook` protocol lives in {py:mod}`nvalchemi.hooks` and is
stage-enum agnostic — the same protocol works for dynamics, training,
or any custom workflow.

```python
from nvalchemi.hooks import DynamicsContext, Hook
from nvalchemi.dynamics.base import DynamicsStage

class MyHook:
    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        print(f"Step {ctx.step_count}: energy = {ctx.batch.energy.mean():.4f}")

assert isinstance(MyHook(), Hook)  # True --- structural subtyping
```

Hooks are attached at construction time via the `hooks` parameter:

```python
from nvalchemi.dynamics import FIRE, ConvergenceHook
from nvalchemi.dynamics.hooks import LoggingHook

opt = FIRE(
    model=model,
    dt=0.1,
    n_steps=500,
    hooks=[
        ConvergenceHook.from_fmax(0.05),
        LoggingHook(backend="csv", log_path="hooks.csv", frequency=10),
    ],
)
```

During each `step()`, the engine iterates over hooks and calls those whose
`stage` matches the current point in the loop and whose `frequency` divides the
current step count.

```{tip}
A ``Hook`` is implemented as a Python ``Protocol``, which represents structural
subtyping: for those wanting to write custom ``Hook``s,
it's not necessary to subclass the base ``Hook``, providing that your
custom class contains the same attributes and methods — as long as it
quacks like a duck.
```

## Context Objects

Every hook receives a {py:class}`~nvalchemi.hooks.HookContext` object, or a
workflow-specific subclass, that provides a snapshot of the current workflow
state. The base context contains only fields that are meaningful to all
hook-enabled workflows:

| Field | Type | Populated by |
|-------|------|-------------|
| `batch` | `Batch` | All engines |
| `model` | `BaseModelMixin \| None` | All engines |
| `global_rank` | `int` | All engines (distributed) |
| `workflow` | `Any` | Back-reference to the engine |

Dynamics engines pass {py:class}`~nvalchemi.hooks.DynamicsContext`, which adds:

| Field | Type | Meaning |
|-------|------|---------|
| `step_count` | `int` | Current dynamics step |
| `converged_mask` | `torch.Tensor \| None` | Samples that converged at the current hook stage |

Training loops pass {py:class}`~nvalchemi.hooks.TrainContext`, which adds:

| Field | Type | Meaning |
|-------|------|---------|
| `step_count` | `int` | Current optimizer step on this worker |
| `global_step_count` | `int` | Current optimizer step across all data-parallel workers |
| `batch_count` | `int` | Training batches consumed, including skipped optimizer steps |
| `epoch_step_count` | `int` | Batches consumed within the current epoch |
| `epoch` | `int` | Current epoch |
| `loss` | `torch.Tensor \| None` | Aggregate loss |
| `losses` | `dict[str, torch.Tensor] \| None` | Named loss components |
| `models` | `dict[str, BaseModelMixin] \| ModuleDict[str, BaseModelMixin] \| None` | Models in the training step |
| `optimizers` | `list[torch.optim.Optimizer] \| None` | Optimizers in the training step |
| `lr_schedulers` | `list[object] \| None` | Learning-rate schedulers |
| `gradients` | `dict[str, torch.Tensor] \| None` | Parameter gradients |

The engine builds this context object at each stage via an overridable
`_build_context(batch)` method. Custom engines should return their own
`HookContext` subclass when hooks need workflow-specific fields.

### Optional context manager support

Hooks may optionally implement `__enter__` and `__exit__`. If present, the
engine calls them when the workflow starts and ends (or when using the engine
as a context manager). This is useful for hooks that manage resources like
open files or logger instances — for example,
{py:class}`~nvalchemi.dynamics.hooks.LoggingHook` uses this to set up and tear down
its logger.

## Task-category specialization

The hook system supports multiple **task categories** through stage enums:

- **Dynamics**: {py:class}`~nvalchemi.dynamics.base.DynamicsStage` — 9 stages from
  `BEFORE_STEP` through `ON_CONVERGE`
- **Custom pipelines**: Any custom `Enum` type — the hook system accepts arbitrary
  enum types via the `Enum` fallback

Each engine declares which stage enum type(s) it accepts via
{py:attr}`~nvalchemi.hooks.HookRegistryMixin._stage_type`. For example,
`BaseDynamics` sets `_stage_type = DynamicsStage`.

For multi-stage hooks, define a `_runs_on_stage` method so the registry knows
the hook fires at more than just `self.stage`. Hooks that need to support
multiple enum types can use [plum-dispatch](https://github.com/wesselb/plum)
to overload `__call__` for each stage enum type, plus a fallback overload typed
as `Enum` for any stage type not explicitly handled.

## Built-in hooks

### ConvergenceHook

{py:class}`~nvalchemi.dynamics.base.ConvergenceHook` evaluates one or more
convergence criteria at each step and marks systems as converged when **all** of
them are satisfied (AND semantics).

The simplest way to create one is with the convenience classmethods:

```python
from nvalchemi.dynamics import ConvergenceHook

# Check whether fmax (pre-computed scalar on the batch) is below a threshold
hook = ConvergenceHook.from_fmax(threshold=0.05)

# Or check per-atom force norms directly (applies a norm reduction internally)
hook = ConvergenceHook.from_forces(threshold=0.05)
```

#### Multiple criteria

When a single scalar is not enough, pass a list of criterion specifications.
Each entry is a dictionary with the fields of the internal
`_ConvergenceCriterion` model:

| Field | Type | Default | Purpose |
|-------|------|---------|---------|
| `key` | `str` | *(required)* | Tensor key to read from the batch |
| `threshold` | `float` | *(required)* | Values at or below this are converged |
| `reduce_op` | `"min"` / `"max"` / `"norm"` / `"mean"` / `"sum"` / `None` | `None` | Reduction applied within each entry before graph-level aggregation |
| `reduce_dims` | `int` or `list[int]` | `-1` | Dimensions to reduce over |
| `custom_op` | callable or `None` | `None` | Custom function `Tensor -> Bool[B]`; when set, the other fields are ignored |

```python
hook = ConvergenceHook(criteria=[
    {"key": "fmax", "threshold": 0.05},
    {"key": "energy_change", "threshold": 1e-6},
])
```

All criteria must be satisfied for a system to converge. If you omit `criteria`
entirely, the hook defaults to a single `fmax < 0.05` criterion.

#### How evaluation works under the hood

For each criterion the hook:

1. Retrieves the tensor from the batch via its `key`.
2. If `reduce_op` is set, reduces within each entry (e.g. force vector norm).
3. If the tensor is node-level (first dim matches `num_nodes`), scatter-reduces to
   graph-level using the batch index.
4. Compares the resulting per-graph scalar against `threshold`.

The per-criterion boolean masks are stacked into a `(num_criteria, B)` tensor and
AND-reduced across criteria to produce a single `(B,)` convergence mask.

#### Status migration in multi-stage pipelines

When `source_status` and `target_status` are provided, the hook updates
`batch.status` for converged systems — this is how
{py:class}`~nvalchemi.dynamics.base.FusedStage` moves systems between stages:

```python
hook = ConvergenceHook(
    criteria=[{"key": "fmax", "threshold": 0.05}],
    source_status=0,   # only check systems currently in stage 0
    target_status=1,    # promote converged systems to stage 1
)
```

In a single-stage simulation (no status arguments), convergence simply causes
those systems to stop being updated.

### Dynamics LoggingHook

{py:class}`~nvalchemi.dynamics.hooks.LoggingHook` records scalar observables
(energy, temperature, maximum force, etc.) at a configurable interval:

```python
from nvalchemi.dynamics.hooks import LoggingHook

hook = LoggingHook(backend="csv", log_path="hooks.csv", frequency=10)  # log every 10 steps
```

The hook implements the context manager protocol to manage its logger lifecycle.
It is the current built-in dynamics logger, not the full logging abstraction for
all workflows.

### Logging vs. reporting

Use logging hooks when you want simple, direct records from a workflow: rows,
files, or lightweight backend writes that are easy to inspect later. Logging is
workflow-general; dynamics and training can each have loggers that understand
their own event model. For example, the built-in dynamics `LoggingHook` writes
per-graph dynamics observables to CSV, TensorBoard, or a custom sink without
imposing a higher-level analysis model.

Use reporting when you want workflow-level summaries: scalar collection,
rank-aware reductions, serialized reporting snapshots, live dashboards, or
analysis-facing output across training and dynamics. The reporting abstractions
are described separately in the {doc}`reporting user guide <reporting>`.

### SnapshotHook

{py:class}`~nvalchemi.dynamics.hooks.SnapshotHook` saves the full batch state to a
[data sink](dynamics_sinks_guide) at regular intervals. This is how you record
trajectories:

```python
from nvalchemi.dynamics.hooks import SnapshotHook
from nvalchemi.dynamics.sinks import ZarrData

hook = SnapshotHook(
    sink=ZarrData("/path/to/trajectory.zarr"),
    frequency=50,  # save every 50 steps
)
```

### ConvergedSnapshotHook

{py:class}`~nvalchemi.dynamics.hooks.ConvergedSnapshotHook` is a specialised variant
that only saves systems at the moment they satisfy their convergence criterion. This
is useful for collecting relaxed structures from a large batch without storing the
full trajectory:

```python
from nvalchemi.dynamics.hooks import ConvergedSnapshotHook
from nvalchemi.dynamics.sinks import ZarrData

hook = ConvergedSnapshotHook(sink=ZarrData("/path/to/relaxed.zarr"))
```

## Writing a custom hook

### Simple dynamics hook

To write your own hook, create a class that implements the three required members
(`stage`, `frequency`, `__call__`). Dynamics hooks receive a
{py:class}`~nvalchemi.hooks.DynamicsContext` and the current stage:

```python
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import DynamicsContext

class PrintFmaxHook:
    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        fmax = ctx.batch.forces.norm(dim=-1).max().item()
        print(f"Step {ctx.step_count}: fmax = {fmax:.4f} eV/A")
```

### Multi-stage hooks with `_runs_on_stage`

A hook can fire at multiple stages by defining a `_runs_on_stage` method.
The registry calls this instead of comparing `stage == hook.stage`:

```python
from enum import Enum
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import DynamicsContext

class StepTimerHook:
    stage = DynamicsStage.BEFORE_STEP  # primary stage (for protocol compliance)
    frequency = 1

    def __init__(self):
        self._stages = {DynamicsStage.BEFORE_STEP, DynamicsStage.AFTER_STEP}
        self._t0 = None

    def _runs_on_stage(self, stage: Enum) -> bool:
        return stage in self._stages

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        import time
        if stage == DynamicsStage.BEFORE_STEP:
            self._t0 = time.perf_counter()
        elif stage == DynamicsStage.AFTER_STEP and self._t0 is not None:
            dt = time.perf_counter() - self._t0
            print(f"Step {ctx.step_count}: {dt*1000:.1f} ms")
```

### Cross-category hooks with `plum` dispatch

For hooks that work with multiple stage enum types (e.g. `DynamicsStage` *and*
a custom enum), use `plum.dispatch` to overload `__call__` with different stage
types. This lets you customize behavior per category:

```python
from enum import Enum
from dataclasses import dataclass
from plum import dispatch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import DynamicsContext, HookContext

# Example custom stage enum for a hypothetical pipeline
class MyPipelineStage(Enum):
    BEFORE_PROCESS = 0
    AFTER_PROCESS = 1

@dataclass(kw_only=True)
class PipelineContext(HookContext):
    step_count: int = 0

class UniversalLoggerHook:
    stage = DynamicsStage.AFTER_STEP  # primary stage
    frequency = 10

    def __init__(self):
        self._stages = {DynamicsStage.AFTER_STEP, MyPipelineStage.AFTER_PROCESS}

    def _runs_on_stage(self, stage: Enum) -> bool:
        return stage in self._stages

    @dispatch
    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        fmax = ctx.batch.forces.norm(dim=-1).max().item()
        print(f"[dynamics] step {ctx.step_count}: fmax={fmax:.4f}")

    @dispatch
    def __call__(self, ctx: PipelineContext, stage: MyPipelineStage) -> None:
        print(f"[pipeline] step {ctx.step_count}: processed")

    @dispatch
    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        print(f"[custom] stage={stage.name}, graphs={ctx.batch.num_graphs}")
```

Cross-category hooks such as {py:class}`~nvalchemi.hooks.TorchProfilerHook` use
this pattern to claim the training and dynamics stages they support.
{py:class}`~nvalchemi.hooks.StageTimingHook` uses the same multi-stage hook
protocol for lightweight per-stage timing.

### Resource management with `__enter__` / `__exit__`

If your hook needs setup or teardown (e.g. opening a file), add `__enter__` and
`__exit__`:

```python
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import DynamicsContext

class FileWriterHook:
    stage = DynamicsStage.AFTER_STEP
    frequency = 10

    def __init__(self, path):
        self.path = path
        self._file = None

    def __enter__(self):
        self._file = open(self.path, "w")
        return self

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        energy = ctx.batch.energy.mean().item()
        self._file.write(f"{ctx.step_count},{energy}\n")

    def __exit__(self, *exc):
        if self._file is not None:
            self._file.close()
```

### Restartable hooks with `CheckpointableHook`

Hooks are stateless by default. If a hook owns state that changes training
semantics after a restart (for example EMA weights, a dynamic schedule, or a
history buffer), make it satisfy
{py:class}`~nvalchemi.hooks.CheckpointableHook` by adding `state_dict()` and
`load_state_dict()`. Training checkpoints discover this protocol at runtime and
store only hooks that opt in.

Pydantic-backed hooks should keep declarative configuration in model fields and
use `model_dump()` for the configuration part of `state_dict()`. Use
`model_dump_json()` when you need a JSON representation for logs or separate
configuration files. Runtime tensors or counters that are not Pydantic fields
can then be added explicitly.

```python
from collections.abc import Mapping
from typing import Any

import torch
from pydantic import BaseModel, Field, PrivateAttr

from nvalchemi.hooks import CheckpointableHook
from nvalchemi.training import TrainingStage
from nvalchemi.training.hooks import TrainingUpdateHook

class RunningLossHook(BaseModel, TrainingUpdateHook):
    window: int = Field(gt=0, default=100)
    num_updates: int = 0

    _loss_sum: torch.Tensor | None = PrivateAttr(default=None)

    def __call__(self, ctx, stage, will_skip):
        if (
            stage is TrainingStage.AFTER_OPTIMIZER_STEP
            and not will_skip
            and ctx.loss is not None
        ):
            value = ctx.loss.detach().to("cpu")
            self._loss_sum = (
                value if self._loss_sum is None else self._loss_sum + value
            )
            self.num_updates += 1
        return True, ctx.loss

    def state_dict(self) -> dict[str, Any]:
        state = self.model_dump()
        if self._loss_sum is not None:
            state["loss_sum"] = self._loss_sum
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "window" in state and state["window"] != self.window:
            raise ValueError("RunningLossHook checkpoint window does not match")
        self.num_updates = int(state.get("num_updates", self.num_updates))
        self._loss_sum = state.get("loss_sum")

assert isinstance(RunningLossHook(), CheckpointableHook)
```

Only implement this protocol for state that must survive restart. Temporary
resources, cached buffers that can be rebuilt, and bookkeeping derived from the
workflow counters should stay out of hook checkpoints.

## Composing hooks

Hooks are independent and composable. A typical production setup combines
convergence, logging, and trajectory recording:

```python
from nvalchemi.dynamics import FIRE, ConvergenceHook
from nvalchemi.dynamics.hooks import (
    ConvergedSnapshotHook,
    LoggingHook,
    SnapshotHook,
)
from nvalchemi.dynamics.sinks import ZarrData

with FIRE(
    model=model,
    dt=0.1,
    n_steps=500,
    hooks=[
        ConvergenceHook.from_fmax(0.05),
        LoggingHook(backend="csv", log_path="hooks.csv", frequency=10),
        SnapshotHook(sink=ZarrData("/tmp/traj.zarr"), frequency=50),
        ConvergedSnapshotHook(sink=ZarrData("/tmp/relaxed.zarr")),
    ],
) as opt:
    relaxed = opt.run(batch)
```

## See also

- **Dynamics overview**: The [execution loop](dynamics_guide) shows where hooks fire
  in the step sequence.
- **Data sinks**: The [Sinks guide](dynamics_sinks_guide) covers the storage backends
  used by snapshot hooks.
- **API**: {py:mod}`nvalchemi.hooks` for the core hook protocol, context, and registry.
- **API**: {py:mod}`nvalchemi.dynamics` for dynamics-specific hooks and stages.
