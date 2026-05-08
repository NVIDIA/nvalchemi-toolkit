<!-- markdownlint-disable MD014 -->

(losses_guide)=

# Losses

Loss functions in ALCHEMI are tensor-first, composable
{py:class}`torch.nn.Module` objects. A **leaf loss** consumes a
prediction tensor and a target tensor and returns a scalar; a
{py:class}`~nvalchemi.training.ComposedLossFunction` *routes* keyed
mappings of predictions and targets into each leaf, applies the
composition's per-component weights, and returns a structured
{py:class}`~nvalchemi.training.ComposedLossOutput` with a `total_loss`
plus per-component contributions.

This page covers:

- the built-in leaf losses and how to call them directly;
- {py:class}`~nvalchemi.training.ComposedLossFunction` for multi-task
  training and where per-loss coefficients live;
- loss-weight scheduling via the
  {py:class}`~nvalchemi.training.LossWeightSchedule` protocol, applied
  at the composition level;
- how to write your own loss — first a pure tensor-to-tensor loss,
  then a metadata-aware one.

```{tip}
Leaves are tensor-first: they consume plain `(pred, target)` plus
optional `**kwargs`. For how graph metadata is threaded through, see
[Passing graph metadata](passing_graph_metadata).
```

## Built-in losses

The three provided losses cover the standard MLIP training targets.
Each is a {py:class}`torch.nn.Module` with an MSE-style `forward`,
configurable `target_key` / `prediction_key` attributes used by
composition, and an opt-in `ignore_nan` flag for batches with
missing labels.

| Class | Target | Key defaults | Extra knobs |
|-------|--------|--------------|-------------|
| {py:class}`~nvalchemi.training.EnergyLoss` | Per-graph energy `(B, 1)` | `"energy"` / `"predicted_energy"` | `per_atom` normalization, `ignore_nan` |
| {py:class}`~nvalchemi.training.ForceLoss` | Per-atom forces, dense `(V, 3)` or padded `(B, V_max, 3)` | `"forces"` / `"predicted_forces"` | `normalize_by_atom_count`, `ignore_nan` |
| {py:class}`~nvalchemi.training.StressLoss` | Per-graph stress `(B, 3, 3)` | `"stress"` / `"predicted_stress"` | `ignore_nan` |

### Calling a leaf loss directly

A leaf loss is a plain `nn.Module`. For losses that do not require
graph metadata — `EnergyLoss(per_atom=False)` (the default) and
`StressLoss` — call it with `(pred, target)` and get a scalar back.
Leaves carry no weight or schedule of their own; a direct call returns
the unweighted MSE-style value:

```python
import torch
from nvalchemi.training import EnergyLoss

loss_fn = EnergyLoss()
pred = torch.randn(4, 1, requires_grad=True)
target = torch.randn(4, 1)

loss = loss_fn(pred, target)         # scalar Tensor
loss.backward()
```

`ForceLoss()` (default `normalize_by_atom_count=True`) and
`EnergyLoss(per_atom=True)` require graph metadata and will raise
`ValueError` on a bare `(pred, target)` call. Either pass metadata
kwargs (see [Passing graph metadata](passing_graph_metadata)) or, for
dense `(V, 3)` forces, disable the per-graph normalization for a
tensor-only call:

```python
from nvalchemi.training import ForceLoss

force_fn = ForceLoss(normalize_by_atom_count=False)   # plain MSE over (V, 3)
force_pred = torch.randn(10, 3, requires_grad=True)
force_target = torch.randn(10, 3)
loss = force_fn(force_pred, force_target)             # no metadata needed
```

Padded `(B, V_max, 3)` forces still require `num_nodes_per_graph` even
with `normalize_by_atom_count=False`, since padding rows must be
masked before reduction.

#### Canonical shape layouts

Built-in leaves expect matching shapes. Use **exactly** the layouts
below; `assert_same_shape` allows broadcast-compatible mismatches
(see [Shape and dtype validation](shape_validation)) but broadcasting
silently produces wrong values for per-graph losses.

| Loss | `pred` shape | `target` shape |
|------|--------------|----------------|
| `EnergyLoss` | `(B, 1)` | `(B, 1)` |
| `ForceLoss` (dense) | `(V, 3)` | `(V, 3)` |
| `ForceLoss` (padded) | `(B, V_max, 3)` | `(B, V_max, 3)` |
| `StressLoss` | `(B, 3, 3)` | `(B, 3, 3)` |

```{warning}
`(B, 1)` versus `(B,)` is broadcast-compatible but **wrong** for
per-graph losses. `pred - target` will broadcast to `(B, B)` and
silently compute pairwise residuals across the batch, giving a
finite-looking but meaningless scalar. Keep the explicit trailing
`1` on per-graph tensors.
```

Leaf losses do not receive schedule counters. `step=` and `epoch=`
belong to {py:class}`~nvalchemi.training.ComposedLossFunction`, which
uses them to resolve schedule-driven weights before calling each leaf
(see [Composition weights and schedules](composition_weights)).

(passing_graph_metadata)=

### Passing graph metadata

Concrete losses may require graph metadata as keyword arguments. For
example, `ForceLoss` with the default graph-balanced normalization
needs `batch_idx` and `num_graphs` for dense `(V, 3)` forces:

```python
from nvalchemi.training import ForceLoss

force_fn = ForceLoss()                         # normalize_by_atom_count=True

pred = torch.randn(10, 3, requires_grad=True)
target = torch.randn(10, 3)
batch_idx = torch.tensor([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])

loss = force_fn(pred, target, batch_idx=batch_idx, num_graphs=3)
```

The same loss accepts a padded `(B, V_max, 3)` layout with per-graph
counts instead:

```python
pred_padded = torch.randn(3, 4, 3, requires_grad=True)
target_padded = torch.randn(3, 4, 3)
counts = torch.tensor([3, 4, 3])

loss = force_fn(pred_padded, target_padded, num_nodes_per_graph=counts)
```

{py:class}`~nvalchemi.training.EnergyLoss` (when `per_atom=True`) and
{py:class}`~nvalchemi.training.ForceLoss` also accept an optional
`batch=` keyword argument as a convenience source for that metadata.
When `batch=` is provided, the loss pulls `batch_idx`, `num_graphs`,
and `num_nodes_per_graph` directly from it:

```python
# Batch-derived metadata — shorter callsite
loss = force_fn(pred, target, batch=batch)

# Equivalent explicit call — fine-grained control
loss = force_fn(
    pred, target,
    batch_idx=batch.batch_idx,
    num_graphs=batch.num_graphs,
)
```

Explicit kwargs always win when both are provided — useful if you want
to override `num_graphs` for a sub-batch without rebuilding a `Batch`.
A duck-typed `batch` that's missing a required attribute still falls
through to the descriptive `ValueError` raised by the metadata
resolver, so you don't have to pre-validate it.

### Ignoring missing labels with `ignore_nan`

Every built-in loss has an `ignore_nan=False` flag. When `True`, target
entries equal to `NaN` contribute zero to both the loss value and the
gradient — a "nanmean"-style reduction implemented with branch-free
tensor ops so it stays `torch.compile`-safe:

```python
energy_loss = EnergyLoss(ignore_nan=True)

target = torch.tensor([[1.0], [float("nan")], [3.0]])
pred = torch.zeros_like(target, requires_grad=True)

loss = energy_loss(pred, target)
loss.backward()

assert torch.isfinite(loss)
assert pred.grad[1].item() == 0.0   # masked row has zero gradient
```

`NaN` targets contribute zero loss and zero gradient; a graph whose
target is entirely `NaN` contributes exactly `0.0` because the numerator
and denominator both go to zero and the denominator is clamp-min'd to
`1`. The default (`ignore_nan=False`) lets `NaN` propagate, which is
usually what you want during development when a label *shouldn't* be
missing.

```{warning}
Only target `NaN`s are treated as missing labels. Prediction `NaN`s still
propagate whenever the corresponding target is finite; if the target is
`NaN`, that position contributes zero loss and zero gradient. Do not
rely on `ignore_nan` to hide model explosions.
```

(shape_validation)=

### Shape and dtype validation

Built-in leaves opt in to shape and dtype validation by calling the
public helper
{py:func}`nvalchemi.training.losses.assert_same_shape` at the top of
`forward`:

```python
from nvalchemi.training.losses import assert_same_shape

assert_same_shape(
    pred, target,
    name="MyLoss",
    prediction_key="predicted_energy",
    target_key="energy",
)
```

`assert_same_shape` checks strict `dtype` equality first and then uses
`torch.broadcast_shapes` to verify shape compatibility — so `(B, 1)`
vs. `(B,)` passes (broadcastable) but mismatched dtypes do not. The
helper raises `ValueError` with the component `name` and the
prediction/target keys embedded in the message.

Validation is opt-in because some legitimate losses (e.g. dipole
derived from per-atom charges) have `pred.shape != target.shape` by
design. When writing a custom loss, call `assert_same_shape` at the
top of your `forward` if and only if pred and target are supposed to
have matching shapes; skip the call when they don't. Note that
`assert_same_shape` is exported from `nvalchemi.training.losses` only —
it is not re-exported from the top-level `nvalchemi.training`.

## Composition

Real training objectives typically combine several targets. The idiomatic way is
to add leaves together and use the resulting
{py:class}`~nvalchemi.training.ComposedLossFunction`:

```python
from nvalchemi.training import EnergyLoss, ForceLoss, StressLoss

loss_fn = EnergyLoss() + ForceLoss() + StressLoss()
```

`loss_fn` is an `nn.Module` whose components sit in an
`nn.ModuleList`, so `.to(device)`, `.state_dict()`, `.modules()`, and
the nested `__repr__` work the way you'd expect. Adding a
`ComposedLossFunction` to another loss flattens transparently:

```python
loss_fn_a = EnergyLoss() + ForceLoss()
loss_fn_b = loss_fn_a + StressLoss()   # still 3 flat components
```

### The call signature

A composed loss takes **keyed mappings**, not tensors:

```python
def loss_fn(
    predictions: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    step: int = 0,
    epoch: int | None = None,
    **kwargs,
) -> ComposedLossOutput: ...
```

Each component reads its own `prediction_key` and `target_key`
attributes to pull tensors out of the two mappings. Any extra `**kwargs`
(graph metadata, for example) are forwarded unchanged to every leaf;
each leaf consumes the kwargs it needs and ignores the rest.

```python
predictions = {
    "predicted_energy": model_outputs["energy"],
    "predicted_forces": model_outputs["forces"],
    "predicted_stress": model_outputs["stress"],
}
targets = {
    "energy": batch.energy,
    "forces": batch.forces,
    "stress": batch.stress,
}

out = loss_fn(
    predictions, targets,
    step=global_step, epoch=epoch,
    batch_idx=batch.batch_idx,
    num_graphs=batch.num_graphs,
    num_nodes_per_graph=batch.num_nodes_per_graph,
)

out["total_loss"].backward()
```

Or equivalently `loss_fn(predictions, targets, step=..., epoch=...,
batch=batch)`; see [Passing graph metadata](passing_graph_metadata).

### The return type

`ComposedLossFunction.forward` returns a
{py:class}`~nvalchemi.training.ComposedLossOutput` — a
{py:class}`typing.TypedDict` with five fields:

| Field | Type | Meaning |
|-------|------|---------|
| `total_loss` | `torch.Tensor` | Scalar sum of `effective_weight * component_loss` across components. `.backward()` on this. |
| `per_component_total` | `dict[str, torch.Tensor]` | Per-component **weighted** loss (after applying the effective weight). Keyed by component class name with suffixes on duplicates. |
| `per_component_weight` | `dict[str, float]` | Effective (post-normalization) weights actually applied at this call. |
| `per_component_raw_weight` | `dict[str, float]` | Raw (pre-normalization) weights, equal to `per_component_weight` when `normalize_weights=False`. |
| `per_component_sample` | `dict[str, torch.Tensor]` | Weighted, detached `(B,)` tensors for components that populate `per_sample_loss`. Absent when the leaf stores `None`. See [Per-sample loss diagnostics](#per-sample-loss-diagnostics) below for details (including aggregation caveats). |

```python
out = loss_fn(predictions, targets)
out["total_loss"].backward()

for name, value in out["per_component_total"].items():
    logger.log_scalar(f"loss/{name}", value.detach(), step=global_step)
for name, w in out["per_component_weight"].items():
    logger.log_scalar(f"loss_weight/{name}", w, step=global_step)
```

Duplicate class names get numeric suffixes (`StressLoss_0`,
`StressLoss_1`, …) so keys remain unique.

### Per-sample loss diagnostics

Every leaf carries an optional `per_sample_loss: torch.Tensor | None` attribute.
Concrete losses populate it as a side effect of `forward` with a detached
per-graph tensor of shape `(B,)`, cleared to `None` at the top of every call.
The scalar return still carries gradients — this attribute is for logging and
diagnostics only.

Which built-ins populate it:

- `EnergyLoss`: populated when the residual has shape `(B,)` or `(B, 1)`. Left
  as `None` on unexpected broadcast-trap shapes (see the warning in
  [Canonical shape layouts](#canonical-shape-layouts)).
- `StressLoss`: always populated (Frobenius MSE is already per-graph).
- `ForceLoss`: populated whenever `normalize_by_atom_count=True` (dense +
  `batch_idx` or padded + `num_nodes_per_graph`), and for padded inputs with
  `normalize_by_atom_count=False`. **Not** populated for dense `(V, 3)` with
  `normalize_by_atom_count=False`, since the scalar path does not need
  `batch_idx` and requiring it just for diagnostics would change the call
  contract.

`ComposedLossOutput["per_component_sample"]` carries
`effective_weight * component.per_sample_loss` (detached) for each component
that populated the attribute. Components whose `per_sample_loss` was `None`
are **absent** from the dict:

```python
out = loss(predictions, targets)
if "EnergyLoss" in out["per_component_sample"]:
    per_graph_energy_loss = out["per_component_sample"]["EnergyLoss"]
    # shape (B,), detached, weighted by the effective energy weight at this step
```

```{note}
For most losses `per_sample_loss.mean()` equals the scalar return, but two
built-in paths populate a per-graph metric whose mean does **not** coincide
with the global scalar: `EnergyLoss(ignore_nan=True)` (global divisor =
count of valid entries) and padded `ForceLoss(normalize_by_atom_count=False)`
(global divisor = total valid components across graphs). Inspect individual
components rather than comparing aggregates.
```

### Routing errors

`ComposedLossFunction` validates its inputs eagerly and fails with a
focused error when a contract is broken:

- A missing `prediction_key` or `target_key` in the input mappings
  raises `KeyError`.
- A mapping entry that is not a `torch.Tensor` raises `TypeError`.
- A component class without `prediction_key` / `target_key`
  attributes (e.g. a bespoke loss you forgot to configure) raises
  `AttributeError`.
- A non-finite or non-strictly-positive **sum** of resolved weights
  (when `normalize_weights=True`) raises `ValueError` — see
  [Weight normalization](weight_normalization) for details.

(composition_weights)=

## Composition weights and schedules

Per-loss coefficients live on
{py:class}`~nvalchemi.training.ComposedLossFunction`, not on leaves.
Leaves have no `weight` argument. A composition stores a parallel
`weights` list — one entry per top-level component — of
`float | LossWeightSchedule | None`. `None` defaults to `1.0`.

The idiomatic way to assemble a weighted composition is with operator
sugar:

```python
from nvalchemi.training import EnergyLoss, ForceLoss, StressLoss

loss_fn = 1.0 * EnergyLoss() + 10.0 * ForceLoss() + 0.1 * StressLoss()
```

`3.0 * EnergyLoss()` returns a one-component
`ComposedLossFunction([EnergyLoss()], weights=[3.0])`. Multiplying a
leaf attaches a weight; subsequent additions combine weights into a
single flat composition.

For a direct construction with named arguments:

```python
from nvalchemi.training import ComposedLossFunction, LinearWeight

loss_fn = ComposedLossFunction(
    [EnergyLoss(), ForceLoss(), StressLoss()],
    weights=[1.0, LinearWeight(start=0.0, end=10.0, num_steps=1000), 0.1],
    normalize_weights=True,
)
```

(weight_normalization)=

### Weight normalization

`ComposedLossFunction` normalizes its resolved weights to sum to `1.0`
at every call by default (`normalize_weights=True`). That keeps the
loss magnitude independent of how many terms you add and puts
scheduling in control of relative weighting rather than absolute
magnitude.

Opt out when you want raw arithmetic sums (e.g. if you're reproducing
results from a paper that hard-codes coefficients):

```python
loss_fn = ComposedLossFunction(
    [EnergyLoss(), ForceLoss()],
    weights=[1.0, 10.0],
    normalize_weights=False,
)
```

When `normalize_weights=True`, the raw-weight sum must be finite and
strictly positive at every call; otherwise a `ValueError` fires before
any gradient can be computed.

### Operator sugar and its constraints

Common forms: `3.0 * EnergyLoss()` to attach a weight,
`schedule * EnergyLoss()` to attach a schedule, `a + b + c` and
`sum([a, b, c])` to compose. A handful of non-obvious constraints:

- **`composition + composition`** requires both sides to share the
  same `normalize_weights` flag. Mismatch raises `ValueError`;
  construct the combined composition explicitly with
  `ComposedLossFunction(..., normalize_weights=...)` to choose.
- **`schedule * composition`** is **rejected** with `TypeError`.
  Scale each component individually (`schedule * EnergyLoss()` and
  compose the results) or multiply the composition by a plain float.
- **`bool * loss`** is **rejected** to avoid `True` silently
  coercing to `1.0`. Pass `1.0` explicitly.

### Weight schedules

Any entry in `weights` may be a
{py:class}`~nvalchemi.training.LossWeightSchedule` instead of a
float. The composition evaluates it at every call with the `(step,
epoch)` you pass to `forward`:

```python
from nvalchemi.training import (
    ConstantWeight,
    CosineWeight,
    EnergyLoss,
    ForceLoss,
    LinearWeight,
    PiecewiseWeight,
    StressLoss,
)

energy_sched = ConstantWeight(value=1.0)
force_sched = LinearWeight(start=0.0, end=1.0, num_steps=1000)
stress_sched = PiecewiseWeight(
    boundaries=(0, 10, 20),
    values=(0.0, 0.5, 1.0, 1.0),
    per_epoch=True,
)

loss_fn = (
    energy_sched * EnergyLoss()
    + force_sched * ForceLoss()
    + stress_sched * StressLoss()
)

out = loss_fn(predictions, targets, step=500, epoch=7, batch=batch)
```

| Schedule | Shape | Typical use |
|----------|-------|-------------|
| {py:class}`~nvalchemi.training.ConstantWeight` | Flat | Static task weight |
| {py:class}`~nvalchemi.training.LinearWeight` | `start` → `end` over `num_steps`, clamped | Curriculum warm-up |
| {py:class}`~nvalchemi.training.CosineWeight` | Half-cosine `start` → `end`, clamped | Smooth curriculum |
| {py:class}`~nvalchemi.training.PiecewiseWeight` | Step function over boundaries | Phase changes |

### Step vs. epoch

Every schedule has a `per_epoch: bool` field. When `False` (the default)
the schedule advances by the `step` argument passed to the loss; when
`True`, it advances by `epoch`. Mixing the two lets most schedules
advance per batch while keeping others, such as a stress-weight
curriculum, aligned with learning-rate epochs.

A `per_epoch=True` schedule called with `epoch=None` raises
`ValueError` — passing `epoch` is required whenever any attached
schedule opts in.

### Bring your own schedule

{py:class}`~nvalchemi.training.LossWeightSchedule` is a
`runtime_checkable` {py:class}`typing.Protocol`: any object with a
`per_epoch` attribute and a `__call__(step: int, epoch: int) -> float`
method qualifies. You don't need to subclass anything to use a custom
schedule in a composition; it just has to quack like one.

```python
class CappedInverse:
    """Return min(1.0, 1.0 / max(step, 1)) — reciprocal step decay."""

    per_epoch = False

    def __call__(self, step: int, epoch: int) -> float:
        return min(1.0, 1.0 / max(step, 1))

loss_fn = CappedInverse() * ForceLoss() + EnergyLoss()
```

Subclass the internal `_BaseWeightSchedule` (from
`nvalchemi.training.losses.base`) instead when you want Pydantic
validation and `create_model_spec` round-tripping for checkpoints.

## Writing your own loss

Writing a custom loss is a matter of subclassing
{py:class}`~nvalchemi.training.BaseLossFunction` and implementing
`forward(pred, target, **kwargs) -> torch.Tensor`.
`forward` is the sole override point — the base class is abstract and
does no pre- or post-processing. Weight scheduling lives on
`ComposedLossFunction`, so your `forward` returns the unweighted loss
value only.

Four conventions worth knowing:

1. **Accept `**kwargs`.** `ComposedLossFunction` forwards extra metadata
   kwargs to every component. Swallowing the ones you don't use keeps
   your loss composable with any other loss in the mix.
2. **Define `target_key` and `prediction_key`.** These attributes tell
   `ComposedLossFunction` which slots in the prediction/target mappings
   to wire into your `forward`. Without them, your loss works
   standalone but cannot participate in a composition.
3. **Keep `forward` tensor-first.** See
   [Passing graph metadata](passing_graph_metadata) for the kwarg
   contract.
4. **Call `assert_same_shape` for MSE-style losses** (skip it when
   `pred.shape != target.shape` by design).

### Example 1: a Huber energy loss

A simple drop-in replacement for MSE. Pure tensor-to-tensor, no graph
metadata, works standalone or inside a composition out of the box.

```python
from typing import Any

import torch
import torch.nn.functional as F

from nvalchemi.training import BaseLossFunction
from nvalchemi.training.losses import assert_same_shape


class HuberEnergyLoss(BaseLossFunction):
    def __init__(
        self,
        *,
        target_key: str = "energy",
        prediction_key: str = "predicted_energy",
        delta: float = 1.0,
    ) -> None:
        super().__init__()
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.delta = delta

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,            # noqa: ARG002 — swallow composition kwargs
    ) -> torch.Tensor:
        assert_same_shape(
            pred, target,
            name=type(self).__name__,
            prediction_key=self.prediction_key,
            target_key=self.target_key,
        )
        return F.huber_loss(pred, target, delta=self.delta)
```

Override `extra_repr()` if you want `print(loss_fn)` to show
`delta=...` alongside the default fields.

Compose it with any other leaf:

```python
from nvalchemi.training import ForceLoss

loss_fn = 1.0 * HuberEnergyLoss(delta=0.5) + 10.0 * ForceLoss()
```

### Example 2: a metadata-aware masked-energy loss

When your loss depends on graph structure, pull the pieces you need out
of `**kwargs`. The established pattern is:

1. Declare typed keyword arguments with defaults of `None`.
2. Validate presence with a focused error — raise `ValueError` with a
   clear message naming the required metadata.
3. Use the reduction helpers in `nvalchemi.training.losses.reductions`
   for scatter-based per-graph reductions.

The example below is a per-atom-count-normalized energy loss: both
`pred` and `target` are per-graph `(B, 1)`, so it passes shape
validation and drops into any `ComposedLossFunction`.

```python
from typing import Any

import torch

from nvalchemi.training import BaseLossFunction
from nvalchemi.training.losses import assert_same_shape


class MaskedEnergyLoss(BaseLossFunction):
    """Energy MSE that uses node-count metadata to normalize per atom."""

    target_key = "energy"
    prediction_key = "predicted_energy"

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        num_nodes_per_graph: torch.Tensor | None = None,
        **kwargs: Any,            # noqa: ARG002
    ) -> torch.Tensor:
        assert_same_shape(
            pred, target,
            name=type(self).__name__,
            prediction_key=self.prediction_key,
            target_key=self.target_key,
        )
        if num_nodes_per_graph is None:
            raise ValueError(
                "MaskedEnergyLoss requires num_nodes_per_graph=... metadata."
            )
        # Accept counts (B,) or a padded node-validity mask (B, V_max).
        nodes = num_nodes_per_graph.to(pred)
        counts = nodes.clamp_min(1.0) if nodes.ndim == 1 else nodes.sum(dim=-1).clamp_min(1.0)
        return torch.mean(((pred - target) / counts.unsqueeze(-1)) ** 2)
```

`target_key` and `prediction_key` are resolved by composition via
`getattr`, so class-level defaults are enough when a loss has no other
constructor state. If you want callers to override routing keys or
configure additional fields, expose those via `__init__` the way
`HuberEnergyLoss` does above.

### Populating `per_sample_loss` (optional)

Custom leaves may set `self.per_sample_loss` to a detached `(B,)` tensor at
the end of `forward` to expose per-graph diagnostics through
`ComposedLossOutput["per_component_sample"]`. See
[Per-sample loss diagnostics](#per-sample-loss-diagnostics) for the full
contract; leave it `None` when a per-graph decomposition is unavailable.

### Testing a custom loss

Two checks usually suffice:

1. A direct call returns a scalar of the expected dtype and gradient
   flows back to `pred`.
2. If `ignore_nan` semantics matter for your loss, assert that a
   `NaN`-filled target row contributes zero to `pred.grad`.

```python
import torch

loss_fn = HuberEnergyLoss(delta=1.0)
pred = torch.randn(4, 1, requires_grad=True)
target = torch.randn(4, 1)

value = loss_fn(pred, target)
assert value.ndim == 0
value.backward()
assert pred.grad is not None
```

For composed losses, assert `total_loss` equals the expected weighted
sum of per-component values on a tiny batch — inspect
`out["per_component_total"]` and `out["per_component_weight"]` to see
exactly what the composition applied.

## See also

- **API**: {ref}`losses-api` for the full class and schedule reference.
- **Reductions**: the `nvalchemi.training.losses.reductions` module for
  scatter-based per-graph helpers usable in custom losses.
- **Models**: the [models guide](models) covers the model-side of the
  contract (how `predictions` mappings are produced).
- **Hooks**: the [hooks guide](hooks_guide) covers the
  {py:class}`~nvalchemi.hooks.HookContext` fields a training loop
  makes available, including `ctx.loss`.
