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

The built-in losses cover standard MLIP training targets and additional
MAE/L2 norm tensor reductions. Each is a {py:class}`torch.nn.Module` with
configurable `target_key` / `prediction_key` attributes used by
composition. The MSE-style losses expose an opt-in `ignore_nonfinite` flag;
the MAE/L2 norm losses expose `ignore_nonfinite` and mask target `NaN`
and `inf` values.

| Class | Target | Key defaults | Extra knobs |
|-------|--------|--------------|-------------|
| {py:class}`~nvalchemi.training.EnergyMSELoss` | Per-graph energy `(B, 1)` | `"energy"` / `"predicted_energy"` | `per_atom` normalization, `ignore_nonfinite` |
| {py:class}`~nvalchemi.training.EnergyMAELoss` | Per-graph energy `(B, 1)` or `(B,)` | `"energy"` / `"predicted_energy"` | MAE reduction, `per_atom`, `ignore_nonfinite` |
| {py:class}`~nvalchemi.training.EnergyHuberLoss` | Per-graph energy `(B, 1)` | `"energy"` / `"predicted_energy"` | Huber residual, `per_atom`, `delta`, `ignore_nonfinite` |
| {py:class}`~nvalchemi.training.ForceMSELoss` | Per-atom forces, dense `(V, 3)` or padded `(B, V_max, 3)` | `"forces"` / `"predicted_forces"` | `normalize_by_atom_count`, `ignore_nonfinite` |
| {py:class}`~nvalchemi.training.ForceHuberLoss` | Per-atom forces, dense `(V, 3)` or padded `(B, V_max, 3)` | `"forces"` / `"predicted_forces"` | Huber residual, `normalize_by_atom_count`, `delta`, `ignore_nonfinite` |
| {py:class}`~nvalchemi.training.ForceL2NormLoss` | Per-atom forces, dense `(V, 3)` or padded `(B, V_max, 3)` | `"forces"` / `"predicted_forces"` | Vector-L2 reduction, `normalize_by_atom_count`, `ignore_nonfinite` |
| {py:class}`~nvalchemi.training.StressMSELoss` | Per-graph stress `(B, 3, 3)` | `"stress"` / `"predicted_stress"` | `ignore_nonfinite` |
| {py:class}`~nvalchemi.training.StressHuberLoss` | Per-graph stress `(B, 3, 3)` | `"stress"` / `"predicted_stress"` | Huber residual, `delta`, `ignore_nonfinite` |

### Calling a leaf loss directly

A leaf loss is a plain `nn.Module`. For losses that do not require
graph metadata — `EnergyMSELoss(per_atom=False)` (the default), dense
`ForceMSELoss(normalize_by_atom_count=False)`,
`ForceHuberLoss(normalize_by_atom_count=False)`,
`StressMSELoss`, `StressHuberLoss`, `EnergyMAELoss(per_atom=False)`,
and dense `ForceL2NormLoss(normalize_by_atom_count=False)` — call it
with `(pred, target)` and get a scalar back. Leaves carry no weight or
schedule of their own; a direct call returns the unweighted value:

```python
import torch
from nvalchemi.training import EnergyMSELoss

loss_fn = EnergyMSELoss()
pred = torch.randn(4, 1, requires_grad=True)
target = torch.randn(4, 1)

loss = loss_fn(pred, target)         # scalar Tensor
loss.backward()
```

`ForceMSELoss()` and `ForceL2NormLoss()` (default
`normalize_by_atom_count=True`), `EnergyHuberLoss()` (default
`per_atom=True`), and both energy losses with `per_atom=True` require
graph metadata and will raise `ValueError` on a bare `(pred, target)`
call. Either pass metadata kwargs (see
[Passing graph metadata](passing_graph_metadata)) or, for dense `(V, 3)`
forces, disable the per-graph normalization for a tensor-only call:

```python
from nvalchemi.training import ForceL2NormLoss, ForceMSELoss

force_fn = ForceMSELoss(normalize_by_atom_count=False)   # plain MSE over (V, 3)
force_pred = torch.randn(10, 3, requires_grad=True)
force_target = torch.randn(10, 3)
loss = force_fn(force_pred, force_target)             # no metadata needed

l2_fn = ForceL2NormLoss(normalize_by_atom_count=False)
l2_loss = l2_fn(force_pred, force_target)             # no metadata needed
```

Padded `(B, V_max, 3)` forces still require `num_nodes_per_graph` even
with `normalize_by_atom_count=False`, since padding rows must be
masked before reduction.

(canonical-shape-layouts)=

#### Expected shape layouts

Built-in leaves call `assert_same_shape(..., strict=True)`, so
prediction and target shapes must match exactly. The table below lists
the layouts these losses are designed for.

| Loss | `pred` shape | `target` shape |
|------|--------------|----------------|
| `EnergyMSELoss` | `(B, 1)` | `(B, 1)` |
| `EnergyMAELoss` | `(B, 1)` or `(B,)` | exact same shape as `pred` |
| `EnergyHuberLoss` | `(B, 1)` | `(B, 1)` |
| `ForceMSELoss` (dense) | `(V, 3)` | `(V, 3)` |
| `ForceMSELoss` (padded) | `(B, V_max, 3)` | `(B, V_max, 3)` |
| `ForceHuberLoss` (dense) | `(V, 3)` | `(V, 3)` |
| `ForceHuberLoss` (padded) | `(B, V_max, 3)` | `(B, V_max, 3)` |
| `ForceL2NormLoss` (dense) | `(V, 3)` | `(V, 3)` |
| `ForceL2NormLoss` (padded) | `(B, V_max, 3)` | `(B, V_max, 3)` |
| `StressMSELoss` | `(B, 3, 3)` | `(B, 3, 3)` |
| `StressHuberLoss` | `(B, 3, 3)` | `(B, 3, 3)` |

```{warning}
`(B, 1)` versus `(B,)` is broadcast-compatible but rejected by the
built-ins. Keep the explicit trailing `1` on per-graph tensors unless
both prediction and target intentionally use the `(B,)` layout supported
by `EnergyMAELoss`.
```

Leaf losses do not receive schedule counters. `step=` and `epoch=`
belong to {py:class}`~nvalchemi.training.ComposedLossFunction`, which
uses them to resolve schedule-driven weights before calling each leaf
(see [Composition weights and schedules](composition_weights)).

(passing_graph_metadata)=

### Passing graph metadata

Concrete losses may require graph metadata as keyword arguments. For
example, `ForceMSELoss` with the default graph-balanced normalization
needs `batch_idx` and `num_graphs` for dense `(V, 3)` forces:

```python
from nvalchemi.training import ForceMSELoss

force_fn = ForceMSELoss()                         # normalize_by_atom_count=True

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

{py:class}`~nvalchemi.training.EnergyMSELoss`,
{py:class}`~nvalchemi.training.EnergyMAELoss`,
{py:class}`~nvalchemi.training.EnergyHuberLoss`,
{py:class}`~nvalchemi.training.ForceMSELoss`,
{py:class}`~nvalchemi.training.ForceHuberLoss`, and
{py:class}`~nvalchemi.training.ForceL2NormLoss` accept an optional
`batch=` keyword argument as a convenience source for metadata when the
selected reduction needs it. When `batch=` is provided, the loss pulls
`batch_idx`, `num_graphs`, and `num_nodes_per_graph` directly from it:

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

### Ignoring missing labels

`EnergyMSELoss`, `ForceMSELoss`, and `StressMSELoss` have an `ignore_nonfinite=False`
flag. When `True`, target entries equal to `NaN` contribute zero to both
the loss value and the gradient — a "nanmean"-style reduction
implemented with branch-free tensor ops so it stays `torch.compile`-safe:

```python
energy_loss = EnergyMSELoss(ignore_nonfinite=True)

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
`1`. The default (`ignore_nonfinite=False`) lets `NaN` propagate, which is
usually what you want during development when a label *shouldn't* be
missing.

```{warning}
For these MSE-style losses, only target `NaN`s are treated as missing
labels. Prediction `NaN`s still propagate whenever the corresponding
target is finite; if the target is `NaN`, that position contributes zero
loss and zero gradient. Do not rely on `ignore_nonfinite` to hide model
explosions.
```

### MAE and force-L2 reductions

`EnergyMAELoss` and `ForceL2NormLoss` implement tensor reductions only.
They do not apply dataset normalization, target transforms,
element-reference corrections, or any other preprocessing; apply those
outside the loss before passing tensors in.

`EnergyMAELoss` computes absolute energy residuals and defaults to
`per_atom=True`: prediction and target are divided by
`num_nodes_per_graph`, then reduced with atom-count weights so that
larger graphs contribute in proportion to their size — matching the
reduction semantics of `EnergyMSELoss(per_atom=True)`.

`ForceL2NormLoss` computes a per-atom vector norm before reduction:

```python
per_atom = torch.linalg.vector_norm(predicted_forces - forces, ord=2, dim=-1)
```

With `normalize_by_atom_count=True`, dense forces use `batch_idx` and
`num_graphs` to compute a valid-atom mean per graph, then mean over
graphs; padded forces use `num_nodes_per_graph` counts or a node mask to
exclude padding before the same per-graph reduction. With
`normalize_by_atom_count=False`, the scalar is a global mean over valid
atom L2 norms.

Both MAE/L2 norm losses have `ignore_nonfinite=True` by default and use
`torch.isfinite(target)` (`.all(dim=-1)` for force vectors), excluding
target `NaN` and `inf` labels while preserving gradients through valid
prediction entries.

(shape_validation)=

### Shape and dtype validation

Built-in leaves opt in to shape and dtype validation via the
{py:meth}`~nvalchemi.training.BaseLossFunction.validate` hook, which
calls {py:func}`nvalchemi.training.losses.assert_same_shape`:

```python
from nvalchemi.training.losses import assert_same_shape

assert_same_shape(
    pred, target,
    name="MyLoss",
    prediction_key="predicted_energy",
    target_key="energy",
)
```

`assert_same_shape` checks strict `dtype` equality first. With its
default `strict=False`, it then uses `torch.broadcast_shapes` to verify
shape compatibility — so `(B, 1)` vs. `(B,)` passes (broadcastable) but
mismatched dtypes do not. With `strict=True`, it requires exact shape
equality. The helper raises `ValueError` with the component `name` and
the prediction/target keys embedded in the message.

Validation is opt-in because some legitimate losses (e.g. dipole
derived from per-atom charges) have `pred.shape != target.shape` by
design. When writing a custom loss, call `assert_same_shape` at the
top of your `forward` with `strict=True` if pred and target are supposed
to match exactly; use the default broadcast-compatible policy only when
that is intentional. Skip the call when they don't. Note that
`assert_same_shape` is exported from `nvalchemi.training.losses` only —
it is not re-exported from the top-level `nvalchemi.training`.

## Composition

Real training objectives typically combine several targets. The idiomatic way is
to add leaves together and use the resulting
{py:class}`~nvalchemi.training.ComposedLossFunction`:

```python
from nvalchemi.training import EnergyMSELoss, ForceMSELoss, StressMSELoss

loss_fn = EnergyMSELoss() + ForceMSELoss() + StressMSELoss()
```

`loss_fn` is an `nn.Module` whose components sit in an
`nn.ModuleList`, so `.to(device)`, `.state_dict()`, `.modules()`, and
the nested `__repr__` work the way you'd expect. Adding a
`ComposedLossFunction` to another loss flattens transparently:

```python
loss_fn_a = EnergyMSELoss() + ForceMSELoss()
loss_fn_b = loss_fn_a + StressMSELoss()   # still 3 flat components
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
| `per_component_unweighted` | `dict[str, torch.Tensor]` | Raw per-component loss before applying the effective weight. Keyed by component class name with suffixes on duplicates. |
| `per_component_weight` | `dict[str, float]` | Effective (post-normalization) weights actually applied at this call. |
| `per_component_raw_weight` | `dict[str, float]` | Raw (pre-normalization) weights, equal to `per_component_weight` when `normalize_weights=False`. |
| `per_component_sample` | `dict[str, torch.Tensor]` | Weighted, detached `(B,)` tensors for components that populate `per_sample_loss`. Absent when the leaf stores `None`. See [Per-sample loss diagnostics](#per-sample-loss-diagnostics) below for details (including aggregation caveats). |

```python
out = loss_fn(predictions, targets)
out["total_loss"].backward()

for name, value in out["per_component_unweighted"].items():
    logger.log_scalar(f"loss/{name}", value.detach(), step=global_step)
for name, w in out["per_component_weight"].items():
    logger.log_scalar(f"loss_weight/{name}", w, step=global_step)
```

Duplicate class names get numeric suffixes (`StressMSELoss_0`,
`StressMSELoss_1`, …) so keys remain unique.

(per-sample-loss-diagnostics)=

### Per-sample loss diagnostics

Every leaf carries an optional `per_sample_loss: torch.Tensor | None` attribute.
Concrete losses populate it as a side effect of `forward` with a detached
per-graph tensor of shape `(B,)`, cleared to `None` at the top of every call.
The scalar return still carries gradients — this attribute is for logging and
diagnostics only.

| Loss | When populated | Aggregation caveat |
|------|----------------|--------------------|
| `EnergyMSELoss` | Recognizable `(B,)` or `(B, 1)` residuals | `per_atom=True` stores per-graph squared per-atom residuals; scalar applies atom-count weights. `ignore_nonfinite=True` uses a global valid-entry divisor. |
| `EnergyMAELoss` | Supported `(B,)` or `(B, 1)` layouts | `per_atom=True` stores per-graph absolute per-atom residuals; scalar applies atom-count weights. `ignore_nonfinite=True` stores masked entries as zero; scalar divides by valid atom-count-weighted sum. |
| `EnergyHuberLoss` | Recognizable `(B,)` or `(B, 1)` residuals | Same layout caveats as `EnergyMSELoss`; scalar is a graph-balanced mean over labeled structures when `per_atom=True`. |
| `StressMSELoss` | Always | None; per-graph Frobenius MSE is already the scalar mean input. |
| `StressHuberLoss` | Always | Same as `StressMSELoss`; per-graph component Huber mean, then mean over graphs. |
| `ForceMSELoss` | Graph-balanced paths and padded global path | Dense `normalize_by_atom_count=False` leaves it absent. Padded global path divides by total valid components. |
| `ForceHuberLoss` | Same paths as `ForceMSELoss` | Inherits `ForceMSELoss` reduction; default global component mean leaves `per_sample_loss` absent for dense inputs. |
| `ForceL2NormLoss` | Graph-balanced paths and padded global path | Dense `normalize_by_atom_count=False` leaves it absent. Padded global path divides by total valid atoms. |

`ComposedLossOutput["per_component_sample"]` carries
`effective_weight * component.per_sample_loss` (detached) for each component
that populated the attribute. Components whose `per_sample_loss` was `None`
are **absent** from the dict:

```python
out = loss(predictions, targets)
if "EnergyMSELoss" in out["per_component_sample"]:
    per_graph_energy_loss = out["per_component_sample"]["EnergyMSELoss"]
    # shape (B,), detached, weighted by the effective energy weight at this step
```

```{note}
For paths with an aggregation caveat, inspect individual components rather than
assuming `per_sample_loss.mean()` equals the scalar return.
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
from nvalchemi.training import EnergyMSELoss, ForceMSELoss, StressMSELoss

loss_fn = 1.0 * EnergyMSELoss() + 10.0 * ForceMSELoss() + 0.1 * StressMSELoss()
```

`3.0 * EnergyMSELoss()` returns a one-component
`ComposedLossFunction([EnergyMSELoss()], weights=[3.0])`. Multiplying a
leaf attaches a weight; subsequent additions combine weights into a
single flat composition.

For a direct construction with named arguments:

```python
from nvalchemi.training import ComposedLossFunction, LinearWeight

loss_fn = ComposedLossFunction(
    [EnergyMSELoss(), ForceMSELoss(), StressMSELoss()],
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
    [EnergyMSELoss(), ForceMSELoss()],
    weights=[1.0, 10.0],
    normalize_weights=False,
)
```

For direct summed task losses, construct the composition
explicitly and set `normalize_weights=False` so coefficients are applied
as raw multipliers rather than renormalized relative weights:

```python
from nvalchemi.training import ComposedLossFunction, EnergyMAELoss, ForceL2NormLoss

loss_fn = ComposedLossFunction(
    [EnergyMAELoss(), ForceL2NormLoss()],
    weights=[1.0, 10.0],
    normalize_weights=False,
)
```

When `normalize_weights=True`, the raw-weight sum must be finite and
strictly positive at every call; otherwise a `ValueError` fires before
any gradient can be computed.

### Operator sugar and its constraints

Common forms: `3.0 * EnergyMSELoss()` to attach a weight,
`schedule * EnergyMSELoss()` to attach a schedule, `a + b + c` and
`sum([a, b, c])` to compose. A handful of non-obvious constraints:

- **`composition + composition`** requires both sides to share the
  same `normalize_weights` flag. Mismatch raises `ValueError`;
  construct the combined composition explicitly with
  `ComposedLossFunction(..., normalize_weights=...)` to choose.
- **`schedule * composition`** is **rejected** with `TypeError`.
  Scale each component individually (`schedule * EnergyMSELoss()` and
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
    EnergyMSELoss,
    ForceMSELoss,
    LinearWeight,
    PiecewiseWeight,
    StressMSELoss,
)

energy_sched = ConstantWeight(value=1.0)
force_sched = LinearWeight(start=0.0, end=1.0, num_steps=1000)
stress_sched = PiecewiseWeight(
    boundaries=(0, 10, 20),
    values=(0.0, 0.5, 1.0, 1.0),
    per_epoch=True,
)

loss_fn = (
    energy_sched * EnergyMSELoss()
    + force_sched * ForceMSELoss()
    + stress_sched * StressMSELoss()
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
`per_epoch` attribute, a `__call__(step: int, epoch: int) -> float` method,
and a `to_spec()` method qualifies. You don't need to subclass anything to
use a custom schedule in a composition; it just needs to be callable and
rebuildable from its spec.

```python
class CappedInverse:
    """Return min(1.0, 1.0 / max(step, 1)) — reciprocal step decay."""

    per_epoch = False

    def __call__(self, step: int, epoch: int) -> float:
        return min(1.0, 1.0 / max(step, 1))

loss_fn = CappedInverse() * ForceMSELoss() + EnergyMSELoss()
```

When a custom schedule is part of a `TrainingStrategy`, it must also be
serializable into the strategy checkpoint spec. Match the general
pattern by implementing `to_spec()`:

```python
from nvalchemi.training import create_model_spec

class CappedInverse:
    per_epoch = False

    def __init__(self, cap: float = 1.0) -> None:
        self.cap = float(cap)

    def __call__(self, step: int, epoch: int) -> float:
        return min(self.cap, 1.0 / max(step, 1))

    def to_spec(self):
        return create_model_spec(type(self), cap=self.cap)
```

Subclass the internal `_BaseWeightSchedule` (from
`nvalchemi.training.losses.base`) when you want Pydantic validation and a
default `to_spec()` implementation backed by `model_dump()`.

## Writing your own loss

{py:class}`~nvalchemi.training.BaseLossFunction` uses a **template-method**
`forward` that orchestrates five hooks:

1. {py:meth}`~nvalchemi.training.BaseLossFunction.validate` — shape/dtype
   checks (default calls `assert_same_shape`).
2. {py:meth}`~nvalchemi.training.BaseLossFunction.normalize` — pre-process
   `pred` and `target` (e.g. per-atom energy division) and return a
   {py:class}`~nvalchemi.training.ReductionContext` for downstream hooks.
3. {py:meth}`~nvalchemi.training.BaseLossFunction.mask` — produce a boolean
   validity tensor (e.g. `torch.isfinite`, padding masks).
4. {py:meth}`~nvalchemi.training.BaseLossFunction.compute_residual` —
   **abstract**, the only method every leaf must implement.
5. {py:meth}`~nvalchemi.training.BaseLossFunction.reduce` — collapse the
   residual + validity mask to a scalar (default: validity-weighted mean,
   incorporating optional `ctx["weights"]`).

Subclass `BaseLossFunction` and override `compute_residual` at a
minimum. The default hooks handle shape validation, all-valid masking,
and weighted-mean reduction out of the box. Override individual hooks
when you need domain-specific behaviour (per-atom normalization in
`normalize`, padding-aware masking in `mask`, graph-balanced reduction
in `reduce`). Weight scheduling lives on `ComposedLossFunction`, so
your hooks return unweighted values only.

You may also override `forward` directly to bypass the template — useful
for losses with non-standard signatures — but you lose the composable
hook structure.

Four conventions worth knowing:

1. **Define `target_key` and `prediction_key`.** These attributes tell
   `ComposedLossFunction` which slots in the prediction/target mappings
   to wire into your loss. Without them, your loss works standalone but
   cannot participate in a composition.
2. **Accept `**kwargs` in hooks that receive them.** `ComposedLossFunction`
   forwards extra metadata kwargs to every component. Swallowing the ones
   you don't use keeps your loss composable with any other loss in the mix.
3. **Keep hooks tensor-first.** See
   [Passing graph metadata](passing_graph_metadata) for the kwarg
   contract.
4. **Override `validate` for non-standard shapes** (skip or customize it
   when `pred.shape != target.shape` by design).

### Example 1: a metadata-aware per-atom energy loss (normalize + compute_residual)

When your loss depends on graph structure, override `normalize` to
inject per-atom division and return atom-count weights via
{py:class}`~nvalchemi.training.ReductionContext`. The base `reduce`
picks up `ctx["weights"]` automatically.

```python
from typing import Any

import torch

from nvalchemi.training import BaseLossFunction, ReductionContext


class PerAtomEnergyMSELoss(BaseLossFunction):
    """Energy MSE normalized by atom count, with atom-count-weighted reduction."""

    target_key = "energy"
    prediction_key = "predicted_energy"

    def normalize(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, ReductionContext]:
        ctx = ReductionContext()
        counts = kwargs.get("num_nodes_per_graph")
        if counts is None:
            raise ValueError(
                "PerAtomEnergyMSELoss requires num_nodes_per_graph=... metadata."
            )
        counts = counts.to(dtype=pred.dtype).unsqueeze(-1).clamp_min(1.0)
        ctx["weights"] = counts
        return pred / counts, target / counts, ctx

    def compute_residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        return residual.pow(2)
```

`target_key` and `prediction_key` are resolved by composition via
`getattr`, so class-level defaults are enough when a loss has no other
constructor state. If you want callers to override routing keys or
configure additional fields, expose those via `__init__` (for example
`delta` on {py:class}`~nvalchemi.training.EnergyHuberLoss`).

### Example 2: custom masking (mask override)

Override `mask` when your loss needs validity logic beyond the base
default (all-True). The mask is a boolean tensor broadcast-compatible
with `pred`/`target`; entries where `mask` is `False` are zeroed in
`compute_residual` and excluded from the reduction denominator.

A common pattern is excluding non-finite targets so that missing labels
contribute zero loss and zero gradient. The built-in
`EnergyMSELoss.mask` is a one-liner:

```python
def mask(
    self,
    pred: torch.Tensor,
    target: torch.Tensor,
    ctx: ReductionContext,
    **kwargs: Any,
) -> torch.Tensor:
    if self.ignore_nonfinite:
        return torch.isfinite(target)
    return torch.ones_like(target, dtype=torch.bool)
```

For padded tensor layouts, the mask must also exclude padding rows. The
built-in force losses combine a node-validity mask (derived from
`num_nodes_per_graph`) with an optional `isfinite` check:

```python
def mask(self, pred, target, ctx, **kwargs):
    num_nodes_per_graph = kwargs.get("num_nodes_per_graph")
    # Build a (B, V_max) node mask from counts, expand to (B, V_max, 3)
    node_mask = _padded_node_mask(num_nodes_per_graph, pred, pred.shape[1])
    valid = node_mask.unsqueeze(-1).expand_as(pred)
    if self.ignore_nonfinite:
        valid = valid & torch.isfinite(target)
    return valid
```

The key contract: `mask` returns a boolean tensor, and `compute_residual`
receives it as the `valid` argument. Your `compute_residual` should use
`torch.where(valid, ..., torch.zeros_like(...))` to zero invalid
entries, and the base `reduce` weights the denominator by
`valid.to(dtype=residual.dtype)`.

### Example 3: custom reduction (reduce override)

Override `reduce` when the base validity-weighted mean is not the
reduction you need — for example, a graph-balanced reduction that
computes a per-graph mean first, then averages over graphs:

```python
import torch

from nvalchemi.training import BaseLossFunction, ReductionContext
from nvalchemi.training.losses.reductions import per_graph_mean, per_graph_sum


class GraphBalancedForceMSE(BaseLossFunction):
    """Force MSE with graph-balanced reduction for dense (V, 3) forces."""

    target_key = "forces"
    prediction_key = "predicted_forces"

    def compute_residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        return residual.pow(2)

    def reduce(
        self,
        residual: torch.Tensor,
        valid: torch.Tensor,
        ctx: ReductionContext,
        **kwargs,
    ) -> torch.Tensor:
        batch_idx = kwargs["batch_idx"]
        num_graphs = kwargs["num_graphs"]
        valid_f = valid.to(dtype=residual.dtype)
        # Per-atom squared error summed over xyz, then per-graph mean
        per_atom_se = residual.sum(dim=-1)
        per_atom_valid = valid_f.sum(dim=-1)
        per_graph_num = per_graph_sum(per_atom_se, batch_idx, num_graphs)
        per_graph_den = per_graph_sum(per_atom_valid, batch_idx, num_graphs)
        per_sample = per_graph_num / per_graph_den.clamp_min(1.0)
        self.per_sample_loss = per_sample.detach()
        return per_sample.mean()
```

When overriding `reduce`, populate `self.per_sample_loss` with a
detached `(B,)` tensor for diagnostics, or leave it `None` when a
per-graph decomposition is not meaningful.

### Layout dispatch with plum (advanced)

The built-in force losses (`ForceMSELoss`, `ForceHuberLoss`, `ForceL2NormLoss`)
accept both dense `(V, 3)` and padded `(B, V_max, 3)` inputs. Rather than
branching on `pred.ndim` inside each hook, they use
[plum-dispatch](https://github.com/beartype/plum) to route to
type-annotated overloads. For example, `ForceMSELoss._valid_force_components`
has two `@overload` implementations — one for `Forces` (dense, 2-D) and
one for `_PaddedForces` (padded, 3-D) — plus a `@dispatch` fallback:

```python
from plum import dispatch, overload

class ForceMSELoss(BaseLossFunction):
    # ...

    @overload
    def _valid_force_components(self, pred: Forces, target: Forces, ...):
        """Dense (V, 3) path — no padding mask needed."""
        ...

    @overload
    def _valid_force_components(self, pred: _PaddedForces, target: _PaddedForces, ...):
        """Padded (B, V_max, 3) path — build node mask from counts."""
        ...

    @dispatch
    def _valid_force_components(self, pred, target, num_nodes_per_graph):
        pass  # plum routes to the matching overload at runtime
```

The `mask` and `reduce` hooks delegate to these dispatched helpers,
keeping each layout's logic in a focused, testable overload. If you are
writing a loss that handles multiple tensor layouts, the `ForceMSELoss`
and `ForceL2NormLoss` implementations in
`nvalchemi/training/losses/terms.py` are the reference patterns to
follow.

### Populating `per_sample_loss` (optional)

The base `reduce` populates `self.per_sample_loss` automatically for
residuals with a recognizable `(B,)` or `(B, 1)` shape. For custom
`reduce` overrides, set `self.per_sample_loss` to a detached `(B,)` tensor
to expose per-graph diagnostics through
`ComposedLossOutput["per_component_sample"]`. See
[Per-sample loss diagnostics](#per-sample-loss-diagnostics) for the full
contract; leave it `None` when a per-graph decomposition is unavailable.

### Testing a custom loss

Two checks usually suffice:

1. A direct call returns a scalar of the expected dtype and gradient
   flows back to `pred`.
2. If `ignore_nonfinite` semantics matter for your loss, assert that a
   `NaN`-filled target row contributes zero to `pred.grad`.

```python
import torch

from nvalchemi.training import EnergyMSELoss

loss_fn = EnergyMSELoss()
pred = torch.randn(4, 1, requires_grad=True)
target = torch.randn(4, 1)

value = loss_fn(pred, target)
assert value.ndim == 0
value.backward()
assert pred.grad is not None
```

For composed losses, assert `total_loss` equals the expected weighted
sum of per-component values on a tiny batch — inspect
`out["per_component_unweighted"]` and `out["per_component_weight"]` to see
exactly what the composition applied.

## See also

- **API**: {ref}`losses-api` for the full class and schedule reference.
- **Reductions**: the `nvalchemi.training.losses.reductions` module for
  scatter-based per-graph helpers usable in custom losses.
- **Models**: the {doc}`models guide <models>` covers the model-side of the
  contract (how `predictions` mappings are produced).
- **Hooks**: the {ref}`hooks guide <hooks_guide>` covers the
  {py:class}`~nvalchemi.hooks.HookContext` fields a training loop
  makes available, including `ctx.loss`.
