---
name: nvalchemi-loss-api
description: >-
  How to use built-in loss functions and implement custom losses using the
  BaseLossFunction template-method pattern — residual types, per-atom
  normalization, masking, and graph-balanced reductions. Use when choosing or
  weighting energy, force, or stress objectives for training or fine-tuning,
  masking atoms or graphs out of the loss, or writing a custom loss term.
---

# nvalchemi Loss API

## Overview

Loss functions are `torch.nn.Module` subclasses rooted at `BaseLossFunction`.
Each leaf consumes `(pred, target, **kwargs)` and returns a scalar.
`ComposedLossFunction` routes keyed prediction/target mappings to leaves,
applies per-component weights (float or `LossWeightSchedule`), and returns
a `ComposedLossOutput` TypedDict.

```python
from nvalchemi.training import (
    BaseLossFunction,
    ComposedLossFunction,
    ReductionContext,
    EnergyMSELoss,
    EnergyMAELoss,
    ForceMSELoss,
    ForceL2NormLoss,
    StressMSELoss,
)
```

---

## Built-in losses

Choose losses by the training signal you want:

- `EnergyMSELoss`: default for smooth energy regression when larger errors should
  dominate early training; combine with `per_atom=True` when system sizes vary.
- `EnergyMAELoss`: more robust to outlier energies and often useful for reporting
  or late-stage fitting when median absolute accuracy matters.
- `EnergyHuberLoss`: compromise between MSE and MAE; use when energy labels have
  occasional noisy outliers but small residuals should remain smooth.
- `ForceMSELoss`: default force objective; component-wise squared residuals give
  strong gradients for geometry-sensitive fitting.
- `ForceL2NormLoss`: use when vector direction/magnitude per atom is the desired
  error signal rather than independent xyz components.
- `ForceHuberLoss`: robust force fitting when some force labels are noisy or
  contain rare large residuals.
- `StressMSELoss` / `StressHuberLoss`: add only when stress labels are reliable
  and the model is configured to produce stresses.

**Composition sugar:**

```python
loss_fn = 1.0 * EnergyMSELoss() + 10.0 * ForceMSELoss() + 0.1 * StressMSELoss()
out = loss_fn(predictions, targets, step=step, epoch=epoch, batch=batch)
out["total_loss"].backward()
```

**Graph metadata:** losses that need graph structure (`per_atom=True`,
`normalize_by_atom_count=True`, or padded layouts) accept `batch=`
(pulls `batch_idx`, `num_graphs`, `num_nodes_per_graph` automatically)
or explicit kwargs.

---

## Template-method pattern

`BaseLossFunction.forward` orchestrates five hooks:

```text
forward(pred, target, **kwargs)
  1. validate(pred, target)                         # shape checks
  2. pred, target, ctx = normalize(pred, target, **kwargs)  # pre-processing
  3. valid = mask(pred, target, ctx, **kwargs)       # boolean validity mask
  4. residual = compute_residual(pred, target, valid) # ABSTRACT — must override
  5. scalar = reduce(residual, valid, ctx, **kwargs)  # collapse to scalar
```

**Minimum implementation:** override `compute_residual` only. Defaults
handle shape validation, all-True masking, and validity-weighted mean reduction.

---

## Writing a custom loss

### Minimal: compute_residual only

```python
class HuberEnergyLoss(BaseLossFunction):
    def __init__(self, *, target_key="energy", prediction_key="predicted_energy", delta=1.0):
        super().__init__()
        self.target_key = target_key
        self.prediction_key = prediction_key
        self.delta = delta

    def compute_residual(self, pred, target, valid):
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        abs_r = residual.abs()
        return torch.where(
            abs_r < self.delta,
            0.5 * residual.pow(2),
            self.delta * (abs_r - 0.5 * self.delta),
        )
```

### Per-atom normalization (normalize override)

Override `normalize` to divide by atom counts and pass weights via
`ReductionContext["weights"]`. The base `reduce` picks up weights
automatically.

```python
class PerAtomEnergyMSE(BaseLossFunction):
    target_key = "energy"
    prediction_key = "predicted_energy"

    def normalize(self, pred, target, **kwargs):
        ctx = ReductionContext()
        counts = kwargs["num_nodes_per_graph"].to(dtype=pred.dtype).unsqueeze(-1).clamp_min(1.0)
        ctx["weights"] = counts  # base reduce uses this for atom-count-weighted mean
        return pred / counts, target / counts, ctx

    def compute_residual(self, pred, target, valid):
        residual = torch.where(valid, pred - target, torch.zeros_like(pred))
        return residual.pow(2)
```

### Custom masking (mask override)

Override `mask` to exclude non-finite targets, padding, or other invalid entries.
Return a boolean tensor broadcast-compatible with `pred`/`target`.

```python
def mask(self, pred, target, ctx, **kwargs):
    if self.ignore_nonfinite:
        return torch.isfinite(target)
    return torch.ones_like(target, dtype=torch.bool)
```

For padded force layouts `(B, V_max, 3)`, combine a node mask with nonfinite check:

```python
def mask(self, pred, target, ctx, **kwargs):
    num_nodes_per_graph = kwargs.get("num_nodes_per_graph")
    node_mask = _padded_node_mask(num_nodes_per_graph, pred, pred.shape[1])
    valid = node_mask.unsqueeze(-1).expand_as(pred)
    if self.ignore_nonfinite:
        valid = valid & torch.isfinite(target)
    return valid
```

The `valid` tensor flows into `compute_residual` as the third argument.
Zero invalid entries with `torch.where(valid, ..., torch.zeros_like(...))`.

### Custom reduction (reduce override)

Override `reduce` for graph-balanced or other non-mean reductions.
Populate `self.per_sample_loss` with a detached `(B,)` tensor for diagnostics.

```python
from nvalchemi.training.losses.reductions import per_graph_sum

def reduce(self, residual, valid, ctx, **kwargs):
    batch_idx = kwargs["batch_idx"]
    num_graphs = kwargs["num_graphs"]
    valid_f = valid.to(dtype=residual.dtype)
    # Per-atom SE summed over xyz, then per-graph mean, then mean over graphs
    per_atom_se = residual.sum(dim=-1)
    per_atom_valid = valid_f.sum(dim=-1)
    per_graph_num = per_graph_sum(per_atom_se, batch_idx, num_graphs)
    per_graph_den = per_graph_sum(per_atom_valid, batch_idx, num_graphs)
    per_sample = per_graph_num / per_graph_den.clamp_min(1.0)
    self.per_sample_loss = per_sample.detach()
    return per_sample.mean()
```

### Layout dispatch with plum (dense vs padded forces)

`ForceMSELoss` and `ForceL2NormLoss` use `plum-dispatch` to handle both
dense `(V, 3)` and padded `(B, V_max, 3)` layouts without `if/else` on
`ndim`. Their `mask` and `reduce` hooks delegate to `@overload`/`@dispatch`
helper methods — one overload per layout. See these implementations in
`nvalchemi/training/losses/terms.py` as the reference pattern for
multi-layout losses.

```python
from plum import dispatch, overload

@overload
def _my_helper(self, pred: Forces, target: Forces, ...):
    """Dense (V, 3) path."""
    ...

@overload
def _my_helper(self, pred: _PaddedForces, target: _PaddedForces, ...):
    """Padded (B, V_max, 3) path."""
    ...

@dispatch
def _my_helper(self, pred, target, ...):
    pass  # plum routes to matching overload at runtime
```

---

## Conventions

1. **Define `target_key` and `prediction_key`** on any loss that participates
   in `ComposedLossFunction` — these route tensors from the prediction/target
   mappings.
2. **Accept `**kwargs`** in hooks that receive them — `ComposedLossFunction`
   forwards metadata kwargs to every component.
3. **`compute_residual` must zero invalid entries** using the `valid` mask
   argument — the base `reduce` handles weighting but not masking.
4. **`ReductionContext`** is a `dict` subclass (not TypedDict) for
   `torch.compile` compatibility. Conventional key: `"weights"` for
   atom-count weights consumed by the base `reduce`.

---

## Key files

| File | Contents |
|---|---|
| `nvalchemi/training/losses/composition.py` | `BaseLossFunction`, `ComposedLossFunction`, `ReductionContext` |
| `nvalchemi/training/losses/terms.py` | All 5 built-in leaf losses |
| `nvalchemi/training/losses/reductions.py` | `per_graph_sum`, `per_graph_mean`, `frobenius_mse` |
| `nvalchemi/training/losses/schedules.py` | `ConstantWeight`, `LinearWeight`, `CosineWeight`, `PiecewiseWeight` |
| `nvalchemi/training/losses/base.py` | `LossWeightSchedule` protocol, re-exports |
| `test/training/test_losses.py` | Comprehensive tests for all loss terms |
| `docs/userguide/losses.md` | Full user guide with examples |
