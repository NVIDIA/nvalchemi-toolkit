<!-- markdownlint-disable MD014 -->

(losses_guide)=

# Losses

Computing the loss - how well a model's prediction lines up with its
targets - is a central part of model training. In particular, how we
design and execute the loss computation can play an extremely large
role in how we design model architectures as it dictates what kind of
signal propagates back to the weight updates. NVIDIA ALCHEMI Toolkit
provides a flexible abstraction with these requirements in mind: users
and developers can use existing loss functions or implement their own
logic in a straightforward manner, and freely compose them with a modular
scheduling system that allows curricula to be designed exactly for a wide
range of training styles.

This page covers:

- the built-in leaf (or terms) losses and how to call them directly;
- {py:class}`~nvalchemi.training.ComposedLossFunction` for multi-task
  training and where per-loss coefficients live;
- loss-weight scheduling via the
  {py:class}`~nvalchemi.training.LossWeightSchedule` protocol, applied
  at the composition level;
- how to write your own loss — first a pure tensor-to-tensor loss,
  then a metadata-aware one.

## Built-in losses

The choice of reduction shapes both gradient noise and interpretability.
MSE losses (`EnergyMSELoss`, `ForceMSELoss`, `StressMSELoss`) have smooth
gradients and are the default starting point for most training runs. Huber
variants (`EnergyHuberLoss`, `ForceHuberLoss`, `StressHuberLoss`) reduce
sensitivity to outlier labels — prefer them when your dataset has noisy
DFT references or a long tail of unusual configurations. MAE and L2-norm
reductions (`EnergyMAELoss`, `ForceL2NormLoss`) report in the same units
as the target and are easiest to interpret as validation metrics, though
their non-smooth gradients make them less common as the primary training
loss.

The built-in losses cover standard MLIP training targets and additional
MAE/L2 norm tensor reductions. Each is a {py:class}`torch.nn.Module` with
configurable `target_key` / `prediction_key` attributes used by
composition. All built-in losses expose `dtype_policy` for optional
prediction/target dtype alignment before validation. The MSE-style losses
expose an opt-in `ignore_nonfinite` flag; the MAE/L2 norm losses expose
`ignore_nonfinite` and mask target `NaN` and `inf` values.

| Class | Target | Key defaults | Extra knobs |
|-------|--------|--------------|-------------|
| {py:class}`~nvalchemi.training.EnergyMSELoss` | Per-graph energy `(B, 1)` | `"energy"` / `"predicted_energy"` | `per_atom` normalization, `ignore_nonfinite`, `dtype_policy` |
| {py:class}`~nvalchemi.training.EnergyMAELoss` | Per-graph energy `(B, 1)` or `(B,)` | `"energy"` / `"predicted_energy"` | MAE reduction, `per_atom`, `ignore_nonfinite`, `dtype_policy` |
| {py:class}`~nvalchemi.training.EnergyHuberLoss` | Per-graph energy `(B, 1)` | `"energy"` / `"predicted_energy"` | Huber residual, `per_atom`, `delta`, `ignore_nonfinite`, `dtype_policy` |
| {py:class}`~nvalchemi.training.ForceMSELoss` | Per-atom forces, dense `(V, 3)` or padded `(B, V_max, 3)` | `"forces"` / `"predicted_forces"` | `normalize_by_atom_count`, `ignore_nonfinite`, `dtype_policy` |
| {py:class}`~nvalchemi.training.ForceHuberLoss` | Per-atom forces, dense `(V, 3)` or padded `(B, V_max, 3)` | `"forces"` / `"predicted_forces"` | Huber residual, `normalize_by_atom_count`, `delta`, `ignore_nonfinite`, `dtype_policy` |
| {py:class}`~nvalchemi.training.ForceL2NormLoss` | Per-atom forces, dense `(V, 3)` or padded `(B, V_max, 3)` | `"forces"` / `"predicted_forces"` | Vector-L2 reduction, `normalize_by_atom_count`, `ignore_nonfinite`, `dtype_policy` |
| {py:class}`~nvalchemi.training.StressMSELoss` | Per-graph stress `(B, 3, 3)` | `"stress"` / `"predicted_stress"` | `ignore_nonfinite`, `dtype_policy` |
| {py:class}`~nvalchemi.training.StressHuberLoss` | Per-graph stress `(B, 3, 3)` | `"stress"` / `"predicted_stress"` | Huber residual, `delta`, `ignore_nonfinite`, `dtype_policy` |

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
(dtype_alignment)=

### Data type alignment

In some workflows, it is desireable to train models with a different
precision from the more common ML ones like `bfloat16` and `float32`;
for long term integration, double (`float64`) may be required.

To serve this purpose, `BaseLossFunction` provides a mechanism for
defining the policy for casting:

- By default, leaf losses use `dtype_policy="strict"`: prediction and target
tensors must already have the same dtype, and mismatches raise before shape
validation. This catches accidental mixed precision labels early.
- Set `dtype_policy="prediction_to_target"` when labels define the desired loss
dtype and model outputs should be cast to match them, or
- `dtype_policy="target_to_prediction"` when labels should follow the model output
dtype. Casting happens before each leaf's normal shape and dtype validation.

```python
from nvalchemi.training import EnergyMSELoss

loss_fn = EnergyMSELoss(dtype_policy="prediction_to_target")
loss = loss_fn(predicted_energy, energy_labels)
```

For multi-component objectives, set the policy on the composition when every
strict leaf should share the same behavior:

```python
from nvalchemi.training import ComposedLossFunction, EnergyMSELoss, ForceMSELoss

loss_fn = ComposedLossFunction(
    [EnergyMSELoss(), ForceMSELoss()],
    dtype_policy="prediction_to_target",
)
```

When using operator sugar, set the same property after constructing the composed
loss:

```python
loss_fn = EnergyMSELoss() + ForceMSELoss()
loss_fn.dtype_policy = "prediction_to_target"
```

A composed-level policy is applied at call time only to leaves whose own
`dtype_policy` is still `"strict"`; an explicitly configured leaf keeps its own
policy. This lets you set a broad default without mutating reusable leaf loss
instances or overriding a component that needs different dtype handling.

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

If you need a different exclusion strategy — for example, masking based on
per-graph quality flags or confidence scores rather than per-entry NaN checks —
override `mask` in a custom leaf. See [Masking](#masking-mask).

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

If the built-in reductions don't match your objective — for instance, if you want
a per-graph median instead of a mean, or a reduction that accounts for
per-structure uncertainty weights — override `reduce` in a custom leaf. See
[Reduction](#reduction-reduce).

(shape_validation)=

### Shape and dtype validation

Built-in leaves validate inputs via the
{py:meth}`~nvalchemi.training.BaseLossFunction.validate` hook, which calls
{py:func}`nvalchemi.training.losses.assert_same_shape` with `strict=True`. This
requires exact shape and dtype equality between `pred` and `target`, raising a
`ValueError` that embeds the component name and routing keys in the message for
easy diagnosis.

```python
from nvalchemi.training.losses import assert_same_shape

assert_same_shape(
    pred, target,
    name="MyLoss",
    prediction_key="predicted_energy",
    target_key="energy",
)
```

`assert_same_shape` checks `dtype` equality first. With `strict=False` it uses
`torch.broadcast_shapes` to verify shape compatibility — so `(B, 1)` vs. `(B,)`
passes — while `strict=True` requires exact shape equality.

Some legitimate losses have `pred.shape != target.shape` by design, so validation
is opt-in per-leaf rather than enforced globally. When writing a custom loss,
`validate` is the place to enforce whatever shape invariant your loss expects — or
to skip the check entirely when the shapes are intentionally asymmetric. See
[Shape and dtype validation](#shape-and-dtype-validation-validate) in the custom
loss section for override examples. Note that `assert_same_shape` is exported from
`nvalchemi.training.losses` only — it is not re-exported from the top-level
`nvalchemi.training`.

## Composition

Real training objectives typically combine several targets. The idiomatic way is
to literally add leaves together and use the resulting
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

The subsections below first cover the call signature and return type, then
weights and scheduling — how to control the relative importance of each term.

### The call signature

`ComposedLossFunction` takes **keyed mappings** rather than raw tensors. This is the
design that allows routing: each leaf reads its own `prediction_key` and
`target_key` attributes to pull the tensors it needs from the two mappings, so you
pass one unified set of predictions and targets and the composition handles the
dispatch automatically. You never have to manually split tensors and call each leaf
in turn.

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

Any extra `**kwargs` — graph metadata, batch index, atom counts — are forwarded
unchanged to every leaf. Each leaf consumes what it needs and ignores the rest, so
you pass metadata once at the call site rather than threading it through each loss
individually:

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

Calling `.backward()` on `total_loss` is all you need for training, but
`ComposedLossFunction` returns a richer
{py:class}`~nvalchemi.training.ComposedLossOutput` — a
{py:class}`typing.TypedDict` — specifically so you can observe the composition's
internals without recomputing anything. The per-component fields are the primary
hook for logging and debugging: they tell you what each task is contributing to
the objective and what weight was actually applied at this step (which differs from
the raw weight when `normalize_weights=True` or a schedule is active).

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

`per_component_weight` is particularly useful when a schedule is active: logging it
alongside the raw loss tells you whether a sudden loss spike came from the model or
from a weight that jumped. Duplicate class names get numeric suffixes
(`StressMSELoss_0`, `StressMSELoss_1`, …) so keys remain unique.

(per-sample-loss-diagnostics)=

### Per-sample loss diagnostics

The scalar `total_loss` is what the optimizer sees, but understanding *which
structures* are driving the loss is a separate concern — useful for identifying
hard samples, debugging dataset quality issues, or building curriculum strategies
that weight structures by difficulty. Every leaf exposes this through an optional
`per_sample_loss: torch.Tensor | None` attribute, populated as a detached `(B,)`
side effect of `forward` and cleared to `None` at the start of each call.

The built-in losses populate it when the residual shape makes a per-graph
decomposition well-defined. Whether a given configuration populates it and any
aggregation caveats are listed below:

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

`ComposedLossOutput["per_component_sample"]` surfaces these per-graph tensors
weighted by the effective composition weight, making them ready to log or rank
directly. Components whose `per_sample_loss` was `None` are **absent** from the
dict, so check before reading:

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

Custom leaves control this field through `self.per_sample_loss` inside their
`reduce` override. See [Reduction](#reduction-reduce) for the contract.

### Routing errors

Before computing anything, `ComposedLossFunction` validates that its inputs
satisfy the routing contract. Errors here almost always indicate a mismatch between
what `training_fn` returned and what the loss components expect — catching it at
the call site is faster than debugging a silent wrong result or a downstream
shape error.

- A missing `prediction_key` or `target_key` in the input mappings
  raises `KeyError`.
- A mapping entry that is not a `torch.Tensor` raises `TypeError`.
- A component class without `prediction_key` / `target_key`
  attributes (e.g. a custom loss you forgot to configure) raises
  `AttributeError`.
- A non-finite or non-strictly-positive **sum** of resolved weights
  (when `normalize_weights=True`) raises `ValueError` — see
  [Weight normalization](weight_normalization) for details.

(composition_weights)=

### Weights

More often than not, it is desirable to assign weights to different terms/leaves
of the composed loss, for example training on more numerically stable targets
first before introducing more complex/difficult outputs that are hard to optimize.

The weighting values are stored within the composed loss, rather than on the
individual term/loss; this is so that dynamic schedules can be orchestrated.
By default, if no weight is provided, any given loss term/leaf is multiplied
by `1.0`. Static weights are most easily and intuitively applied by multiplying
a term in the composition like shown below:

```python
from nvalchemi.training import EnergyMSELoss, ForceMSELoss, StressMSELoss

loss_fn = 1.0 * EnergyMSELoss() + 10.0 * ForceMSELoss() + 0.1 * StressMSELoss()
```

While the leading `1.0` is not necessary, it is used to illustrate how it should
just resemble an equation where we train with a 10x weighting on the forces,
but only 10% emphasis on the stresses in a periodic system.

Alternatively, the weights can be specified as a list of values;
`3.0 * EnergyMSELoss()` produces a one-component
`ComposedLossFunction([EnergyMSELoss()], weights=[3.0])`; subsequent additions
flatten components and weights into a single composition.

(weight_normalization)=

By default, the values above are **not** applied literally: the default behavior,
when `normalize_weights` is set to `True`, has each weighting factor normalized
by the sum of the weights. This is desirable when trying to reason about the
relative weighting of each term and to keep things numerically stable, however
sometimes the actual desired behavior is to have each term scaled literally
because it has physical meaning or needs the amplification. In those cases,
set `normalize_weights=False`.

When `normalize_weights=True`, the resolved weight sum must be finite and
strictly positive at every call; a zero or NaN sum raises `ValueError` before
any gradient is computed.

### Weight schedules

Static weights fix the relative importance of each task for the entire run.
When your training objective should change over time — warming up force weight
over the first thousand steps, introducing stress loss at epoch 10, annealing
energy weight as the model converges — use a schedule instead of a float.

Any entry in the `weights` list may be a
{py:class}`~nvalchemi.training.LossWeightSchedule`. The composition evaluates it
at every call with the `(step, epoch)` you pass to `forward`:

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

Every schedule has a `per_epoch: bool` field. When `False` (the default), the
schedule advances by the `step` argument; when `True`, by `epoch`. Mixing the two
lets most schedules advance per batch while keeping others — such as a
stress-weight curriculum — aligned with learning-rate epochs. A `per_epoch=True`
schedule called with `epoch=None` raises `ValueError`.

```{tip}
**Operator constraints** — a few non-obvious restrictions worth knowing:

- **`composition + composition`** requires both sides to share the same
  `normalize_weights` flag. Mismatch raises `ValueError`; construct the
  combined composition explicitly to choose.
- **`schedule * composition`** is rejected with `TypeError`. Scale each
  component individually and compose the results, or multiply the
  composition by a plain float.
- **`bool * loss`** is rejected to prevent `True` silently coercing to
  `1.0`. Pass `1.0` explicitly.

```

### Bring your own schedule

When none of the built-in schedules fit — reciprocal-step decay, exponential decay
with a floor, cyclic oscillation, or a schedule driven by external state —
implement {py:class}`~nvalchemi.training.LossWeightSchedule` directly.
The protocol is `runtime_checkable`, so any object with a `per_epoch` attribute
and a `__call__(step: int, epoch: int) -> float` method plugs into a composition
without subclassing anything:

```python
class CappedInverse:
    """Return min(1.0, 1.0 / max(step, 1)) — reciprocal step decay."""

    per_epoch = False

    def __call__(self, step: int, epoch: int) -> float:
        return min(1.0, 1.0 / max(step, 1))

loss_fn = CappedInverse() * ForceMSELoss() + EnergyMSELoss()
```

When a custom schedule is part of a `TrainingStrategy`, it must also be
serializable into the strategy checkpoint spec. Add `to_spec()` to meet the
full protocol:

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

Scheduling controls *when* each objective matters; the next section covers
how to change *what* is computed inside a leaf — residuals, normalization,
masking, and reduction — by writing your own loss.

## Writing your own loss

{py:class}`~nvalchemi.training.BaseLossFunction` is a **template-method** class:
its `forward` orchestrates five hooks in a fixed order, each with a default
implementation you can selectively override. This design lets you implement only
the logic your loss actually requires — from a single residual formula to a fully
custom reduction pipeline — without rewriting the parts the base class handles
correctly.

Four conventions apply to every custom loss:

1. **Define `target_key` and `prediction_key`.** These class-level attributes tell
   `ComposedLossFunction` which slots in the prediction/target mappings to wire
   into your loss. Without them, your loss works standalone but cannot participate
   in a composition.
2. **Accept `**kwargs` in hooks that receive them.** `ComposedLossFunction`
   forwards extra metadata kwargs to every component. Swallowing the ones you
   don't use keeps your loss composable with any other loss in the mix.
3. **Keep hooks tensor-first.** See
   [Passing graph metadata](passing_graph_metadata) for the kwarg contract.
4. **Weight scheduling lives on `ComposedLossFunction`.** Your hooks return
   unweighted values. Override `forward` directly to bypass the template when a
   loss has a fundamentally non-standard signature — but doing so means the
   composition hook structure no longer applies.

### Residuals (compute_residual)

`compute_residual` is the one hook every leaf must implement — it has no default
and the base class raises if you omit it. It receives the (optionally normalized)
`pred` and `target` tensors and the boolean `valid` mask produced by `mask`, and
returns an element-wise residual tensor of the same shape. Because `normalize` and
`mask` have already run, `compute_residual` can focus entirely on the residual
formula and safely zero invalid positions with `torch.where`.

The minimum viable leaf overrides nothing but `compute_residual`. The base class
provides validation, an all-valid mask, and a validity-weighted mean reduction —
you supply only the residual formula:

```python
import torch
from nvalchemi.training import BaseLossFunction


class EnergyMSELoss(BaseLossFunction):
    target_key = "energy"
    prediction_key = "predicted_energy"

    def compute_residual(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        return residual.pow(2)
```

Expose `target_key` and `prediction_key` via `__init__` when callers should be
able to override routing keys or configure extra fields — for example, `delta` on
{py:class}`~nvalchemi.training.EnergyHuberLoss`.

### Normalization (normalize)

The default `normalize` is a pass-through: `pred` and `target` flow unchanged into
`compute_residual`, and an empty {py:class}`~nvalchemi.training.ReductionContext`
is forwarded to the downstream hooks. Override it when the residual should be
computed on transformed inputs — the most common case being per-atom normalization,
where both energy tensors are divided by atom count before the squared-error is
taken. The {py:class}`~nvalchemi.training.ReductionContext` is a `dict`-like
container that flows from `normalize` through `mask` and into `reduce`; storing
normalization factors as `ctx["weights"]` instructs the base `reduce` to apply a
correspondingly weighted mean automatically.

Override `normalize` to return a `(pred, target, ctx)` triple. Here, each energy
prediction and target is divided by atom count, and the counts are stored in the
context so the final reduction is also atom-count-weighted:

```python
from typing import Any

import torch

from nvalchemi.training import BaseLossFunction, ReductionContext


class PerAtomEnergyMSELoss(BaseLossFunction):
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

`ctx["weights"]` here is a `(B, 1)` atom-count tensor. The base `reduce`
multiplies per-graph residuals by these weights when computing the mean, giving
larger graphs proportionally more influence — the same semantics as
`EnergyMSELoss(per_atom=True)`.

### Masking (mask)

The default `mask` returns an all-True tensor — every position is valid, with no
padding exclusion and no NaN handling. The mask flows to `compute_residual` as the
`valid` argument, and the base `reduce` excludes `False` positions from both the
numerator and denominator. Override it when entries should be excluded from the
loss entirely. Two cases come up most often: missing labels, where NaN or inf
targets should contribute zero loss and zero gradient; and padded batches, where
padding rows occupy valid memory but should not contribute to the loss. You can
also combine both, or define completely custom validity criteria.

Override `mask` to return a boolean tensor broadcast-compatible with `pred` and
`target`. For missing labels, `torch.isfinite(target)` is usually sufficient:

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

For padded tensor layouts, also exclude padding rows. The built-in force losses
combine a node-validity mask with an optional `isfinite` check:

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

The key contract: `mask` returns a boolean tensor received by `compute_residual`
as `valid`. Use `torch.where(valid, ..., torch.zeros_like(...))` in
`compute_residual` to zero invalid entries; the base `reduce` handles the
denominator.

### Reduction (reduce)

The default `reduce` computes a validity-weighted global mean: valid entries
contribute to both the numerator and the denominator, with `ctx["weights"]`
applied if `normalize` set them. Override it when the default aggregation would
bias your training signal. The most common case is a heterogeneous batch — one
very large graph and many small ones — where a global atom-position mean gives
disproportionate gradient signal to the large graph. A graph-balanced reduction
computes a per-graph mean first, then means over graphs, giving equal weight to
each structure regardless of size. This matters most for force losses, where atom
counts vary significantly across structures.

`reduce` receives the element-wise `residual` and boolean `valid` tensors, along
with any graph metadata forwarded as kwargs. The
`nvalchemi.training.losses.reductions` module provides scatter-based helpers for
building graph-level aggregations:

```python
import torch

from nvalchemi.training import BaseLossFunction, ReductionContext
from nvalchemi.training.losses.reductions import per_graph_sum


class GraphBalancedForceMSE(BaseLossFunction):
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
        per_atom_se = residual.sum(dim=-1)
        per_atom_valid = valid_f.sum(dim=-1)
        per_graph_num = per_graph_sum(per_atom_se, batch_idx, num_graphs)
        per_graph_den = per_graph_sum(per_atom_valid, batch_idx, num_graphs)
        per_sample = per_graph_num / per_graph_den.clamp_min(1.0)
        self.per_sample_loss = per_sample.detach()
        return per_sample.mean()
```

Populate `self.per_sample_loss` with a detached `(B,)` tensor to expose
per-graph diagnostics through `ComposedLossOutput["per_component_sample"]`.
See [Per-sample loss diagnostics](#per-sample-loss-diagnostics) for the full
contract; leave it `None` when a per-graph decomposition is unavailable.

### Shape and dtype validation (validate)

By default, `validate` calls
{py:func}`nvalchemi.training.losses.assert_same_shape` with `strict=True`,
requiring exact shape and dtype equality between `pred` and `target` before any
residual computation runs. Override it when your loss has `pred.shape !=
target.shape` by design — a dipole loss derived from per-atom charges might produce
`(V,)` charge predictions and compare them against `(B, 3)` dipole targets, which
`assert_same_shape` would reject even though the asymmetry is intentional.
Replace `validate` with whatever check is meaningful for your shapes, or skip it
entirely:

```python
def validate(self, pred, target, **kwargs):
    # pred is (V,) charges, target is (B, 3) dipoles — shapes are intentional
    if pred.dtype != target.dtype:
        raise ValueError(
            f"dtype mismatch: pred {pred.dtype} vs target {target.dtype}"
        )
```

When `pred` and `target` should always match exactly, keep the default.

### Layout dispatch with plum (advanced)

```{note}
This section covers an advanced pattern used by the built-in force losses.
You do not need plum-dispatch to write a custom loss; most custom losses
branch on `pred.ndim` directly or accept a single layout. Read on only if
you need a loss that cleanly handles both dense `(V, 3)` and padded
`(B, V_max, 3)` inputs with separate, testable code paths per layout.
```

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
