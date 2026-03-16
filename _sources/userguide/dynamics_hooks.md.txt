<!-- markdownlint-disable MD014 -->

(dynamics_hooks_guide)=

# Hooks

Hooks let you observe or modify the simulation state at specific points in the
[execution loop](dynamics_guide) without touching integrator code. They are the
primary extension mechanism for logging, convergence checking, trajectory recording,
and any custom per-step logic.

## The Hook protocol

A hook is any object that satisfies the
{py:class}`~nvalchemi.dynamics.base.Hook` protocol (a
`@runtime_checkable` `Protocol`). The required interface is:

| Attribute / Method | Type | Purpose |
|--------------------|------|---------|
| `stage` | {py:class}`~nvalchemi.dynamics.base.HookStageEnum` | Which stage of the step loop this hook fires at |
| `frequency` | `int` | Execute every *n* steps (1 = every step) |
| `__call__(batch, dynamics)` | `None` | The hook's logic, called with the current batch and dynamics object |

Hooks are attached at construction time via the `hooks` parameter:

```python
from nvalchemi.dynamics import FIRE, ConvergenceHook
from nvalchemi.dynamics.hooks import LoggingHook

opt = FIRE(
    model=model,
    dt=0.1,
    n_steps=500,
    hooks=[
        ConvergenceHook(fmax=0.05),
        LoggingHook(interval=10),
    ],
)
```

During each `step()`, the dynamics engine iterates over hooks and calls those whose
`stage` matches the current point in the loop and whose `frequency` divides the
current step count.

```{tip}
A ``Hook`` is implemented as a Python ``Protocol``, which represents structural
subtyping: for those wanting to write custom ``Hook``s,
it's not necessary to subclass the base ``Hook``, providing that your
custom class contains the same attributes and methods---as long as it
quacks like a duck.
```

### Optional context manager support

Hooks may optionally implement `__enter__` and `__exit__`. If present, the dynamics
engine calls them when the simulation starts and ends (or when using the dynamics
object as a context manager). This is useful for hooks that manage resources like
open files or logger instances --- for example,
{py:class}`~nvalchemi.dynamics.hooks.LoggingHook` uses this to set up and tear down
its logger.

## Built-in hooks

### ConvergenceHook

{py:class}`~nvalchemi.dynamics.hooks.ConvergenceHook` evaluates one or more
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
`batch.status` for converged systems --- this is how
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

### LoggingHook

{py:class}`~nvalchemi.dynamics.hooks.LoggingHook` records scalar observables
(energy, temperature, maximum force, etc.) at a configurable interval:

```python
from nvalchemi.dynamics.hooks import LoggingHook

hook = LoggingHook(interval=10)  # log every 10 steps
```

The hook implements the context manager protocol to manage its logger lifecycle.

### SnapshotHook

{py:class}`~nvalchemi.dynamics.hooks.SnapshotHook` saves the full batch state to a
[data sink](dynamics_sinks_guide) at regular intervals. This is how you record
trajectories:

```python
from nvalchemi.dynamics.hooks import SnapshotHook
from nvalchemi.dynamics.sinks import ZarrData

hook = SnapshotHook(
    sink=ZarrData("/path/to/trajectory.zarr"),
    interval=50,  # save every 50 steps
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

To write your own hook, create a class that implements the three required members
(`stage`, `frequency`, `__call__`). For example, a hook that prints the maximum
force every step:

```python
from nvalchemi.dynamics.base import HookStageEnum

class PrintFmaxHook:
    stage = HookStageEnum.AFTER_STEP
    frequency = 1

    def __call__(self, batch, dynamics):
        fmax = batch.forces.norm(dim=-1).max().item()
        print(f"Step {dynamics.step_count}: fmax = {fmax:.4f} eV/A")
```

If your hook needs setup or teardown (e.g. opening a file), add `__enter__` and
`__exit__`:

```python
class FileWriterHook:
    stage = HookStageEnum.AFTER_STEP
    frequency = 10

    def __init__(self, path):
        self.path = path
        self._file = None

    def __enter__(self):
        self._file = open(self.path, "w")
        return self

    def __call__(self, batch, dynamics):
        energy = batch.energies.mean().item()
        self._file.write(f"{dynamics.step_count},{energy}\n")

    def __exit__(self, *exc):
        if self._file is not None:
            self._file.close()
```

## Composing hooks

Hooks are independent and composable. A typical production setup combines
convergence, logging, and trajectory recording:

```python
from nvalchemi.dynamics import FIRE
from nvalchemi.dynamics.hooks import (
    ConvergenceHook,
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
        ConvergenceHook(fmax=0.05),
        LoggingHook(interval=10),
        SnapshotHook(sink=ZarrData("/tmp/traj.zarr"), interval=50),
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
- **API**: {py:mod}`nvalchemi.dynamics` for the full hook and dynamics API reference.
