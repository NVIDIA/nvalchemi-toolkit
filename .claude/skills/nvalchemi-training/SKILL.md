---
name: nvalchemi-training
description: How to train MLIPs using the nvalchemi Trainer, compose loss functions, use hooks and checkpointing, and leverage nvalchemiops GPU kernels for high-performance training.
---

# nvalchemi Training

## Overview

The training module provides a complete loop for training Machine Learning
Interatomic Potentials (MLIPs). It wraps PyTorch's optimizer/scheduler
ecosystem with MLIP-specific features: composable physics losses, GPU-accelerated
segmented reductions (via `nvalchemiops`), mixed precision, distributed
training (DDP/FSDP), and a hook system for extensibility.

```python
from nvalchemi.training import (
    Trainer,
    TrainingResult,
    TrainingConfig,
    MixedPrecisionConfig,
    GradClipConfig,
    TrainingContext,
    TrainingHook,
    TrainingStageEnum,
    EarlyStoppingHook,
    SWAHook,
    SWAFinalizeHook,
    StopTraining,
    TerminateOnStepsHook,
)
from nvalchemi.training.losses import (
    CompositeLoss,
    LossComponent,
    EnergyLoss,
    ForceLoss,
    StressLoss,
)
```

---

## Composing loss functions

Losses follow a two-level design: individual `LossComponent` subclasses define
per-element error metrics, while `CompositeLoss` aggregates multiple weighted
terms. Use arithmetic operators to compose them ergonomically.

### Built-in loss components

| Class | Level | Default keys | Default `normalize_by_atoms` | Error metric |
|-------|-------|--------------|------------------------------|--------------|
| `EnergyLoss` | system | `energies` / `energies` | `True` | MSE, MAE, or Huber |
| `ForceLoss` | node | `forces` / `forces` | `False` | Per-component squared error |
| `StressLoss` | system | `stresses` / `stresses` | `False` | Frobenius norm squared |

### Composing with arithmetic operators

```python
# Weighted multi-task loss — returns a CompositeLoss
loss = 1.0 * EnergyLoss() + 10.0 * ForceLoss() + 0.1 * StressLoss()

# With custom error function and keys
loss = (
    1.0 * EnergyLoss(error_fn="huber", huber_delta=0.5)
    + 10.0 * ForceLoss(pred_key="predicted_forces", target_key="forces")
)

# sum() also works (starts with 0 + first term)
terms = [1.0 * EnergyLoss(), 10.0 * ForceLoss()]
loss = sum(terms)
```

### EnergyLoss reference

```python
EnergyLoss(
    name="energy",                # human-readable name for logging
    pred_key="energies",          # key in ModelOutputs dict
    target_key="energies",        # key in Batch
    weight=1.0,                   # per-component weight
    reduction="mean",             # "mean" or "sum" over batch
    normalize_by_atoms=True,      # divide per-graph loss by atom count
    error_fn="mse",               # "mse", "mae", or "huber"
    huber_delta=1.0,              # Huber threshold (only for error_fn="huber")
)
```

### ForceLoss reference

```python
ForceLoss(
    name="forces",
    pred_key="forces",
    target_key="forces",
    weight=1.0,
    reduction="mean",
    normalize_by_atoms=False,     # forces are already per-atom
)
```

### StressLoss reference

```python
StressLoss(
    name="stress",
    pred_key="stresses",          # shape (B, 3, 3)
    target_key="stresses",
    weight=1.0,
    reduction="mean",
    normalize_by_atoms=False,
)
```

### How losses are computed internally

1. **Key extraction** — `pred = outputs[pred_key]`, `target = batch[target_key]`
2. **Elementwise error** — subclass-defined (e.g. `(pred - target).square()`)
3. **Segmented reduction** — node-level losses are reduced per graph via
   `nvalchemiops.segment_ops.segmented_sum` (wrapped as a `torch.library.custom_op`
   for `torch.compile` compatibility)
4. **Atom-count normalization** — if `normalize_by_atoms=True`, each per-graph
   loss is divided by its atom count
5. **Batch reduction** — `"mean"` or `"sum"` over graphs

If a key is missing from `outputs` or `batch`, the component returns `None`
and is skipped (a debug log is emitted, not an error).

### CompositeLoss forward signature

```python
total_loss, per_term_dict = loss(batch, model_outputs)
# total_loss: scalar Tensor (weighted sum of all terms)
# per_term_dict: dict[str, Tensor | None] mapping component names to individual losses
```

---

## Writing custom loss components

Subclass `LossComponent` and implement `elementwise_error`:

```python
from nvalchemi.training.losses import LossComponent

class DipoleMAE(LossComponent):
    """Mean absolute error on predicted dipole moments."""

    def __init__(self, weight: float = 1.0) -> None:
        super().__init__(
            name="dipole_mae",
            pred_key="dipoles",
            target_key="dipoles",
            weight=weight,
            level="system",       # dipoles are system-level (B, 3)
            reduction="mean",
            normalize_by_atoms=False,
        )

    def elementwise_error(self, pred, target):
        return (pred - target).abs()

# Compose with built-in losses
loss = 1.0 * EnergyLoss() + 5.0 * DipoleMAE()
```

For node-level custom losses, set `level="node"` — the base class will
automatically apply segmented reduction to aggregate per-atom errors into
per-graph losses.

---

## Segmented reductions (nvalchemiops integration)

The training module wraps `nvalchemiops` Warp kernels as PyTorch custom ops
to serve as opaque boundaries for `torch.compile`.

```python
from nvalchemi.training.losses._reductions import (
    _segmented_sum,      # (values, idx, num_segments) -> Tensor
    _segmented_mean,     # (values, idx, num_segments) -> Tensor
    segmented_mse,       # (pred, target, batch_idx, num_segments, reduction) -> Tensor
    segmented_mae,       # (pred, target, batch_idx, num_segments, reduction) -> Tensor
)
```

These are registered as `torch.library.custom_op` under the
`nvalchemi_training::` namespace, preventing `torch.compile` from tracing
into Warp interop code. The underlying Warp kernels (`nvalchemiops.segment_ops`)
handle GPU-accelerated per-segment reductions over sorted `batch_idx` arrays.

### When to use directly

Most users should compose losses via `LossComponent` subclasses, which
call `_segmented_sum` internally. Use `segmented_mse` / `segmented_mae`
directly only for custom metrics outside the loss system.

---

## Configuring training

### TrainingConfig

```python
from nvalchemi.training import TrainingConfig, MixedPrecisionConfig, GradClipConfig

config = TrainingConfig(
    max_epochs=100,
    grad_clip=GradClipConfig(method="norm", max_value=1.0),
    grad_accumulation_steps=4,
    val_every_n_epochs=1,
    checkpoint_every_n_epochs=5,
    checkpoint_dir=Path("checkpoints"),
    resume_from=None,             # or Path("checkpoints/epoch_49.pt")
    mixed_precision=MixedPrecisionConfig(
        enabled=True,
        amp_type="bf16",          # "fp16" or "bf16"
        grad_scaler=True,         # auto-disabled for bf16
    ),
    strategy="ddp",               # "ddp" or "fsdp"
    torch_compile=False,
    compile_kwargs={},            # e.g. {"mode": "reduce-overhead"}
    log_every_n_steps=10,
    seed=42,
)
```

### MixedPrecisionConfig

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `enabled` | `bool` | `False` | Enable AMP autocast |
| `amp_type` | `"fp16"` \| `"bf16"` | `"bf16"` | Precision for autocast |
| `grad_scaler` | `bool` | `True` | Gradient scaling; auto-forced to `False` for bf16 |

Access the `torch.dtype` via `config.mixed_precision.torch_dtype`.

### GradClipConfig

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `method` | `"norm"` \| `"value"` | `"norm"` | `clip_grad_norm_` vs `clip_grad_value_` |
| `max_value` | `float` | required | Threshold for clipping |

---

## The Trainer

### Constructor

```python
from nvalchemi.training import Trainer

trainer = Trainer(
    model=model,                  # BaseModelMixin instance
    loss=loss,                    # CompositeLoss
    optimizer=optimizer,          # or [opt1, opt2] for multi-optimizer
    scheduler=scheduler,          # or [sched1, sched2], or None
    train_loader=train_loader,
    val_loader=val_loader,        # optional
    config=config,                # TrainingConfig
    dist_manager=None,            # physicsnemo DistributedManager, or None
)
```

### Running training

Always use the context manager — it sets up a dedicated CUDA stream and
optional autocast context:

```python
with Trainer(model, loss, optimizer, config=config,
             train_loader=train_loader) as trainer:
    result = trainer.fit()

# result: TrainingResult
print(result.epochs_completed)
print(result.best_val_loss)
print(result.best_checkpoint)    # Path or None
print(result.final_metrics)      # dict[str, float]
print(result.history)            # list[dict[str, float]]
```

### What `fit()` does

1. Creates a `TrainingContext` (mutable state bag)
2. For each epoch:
   a. Fires `BEFORE_EPOCH` hooks
   b. Calls `_train_one_epoch(ctx)` — iterates train_loader
   c. Fires `AFTER_EPOCH` hooks
   d. Runs validation if `val_every_n_epochs` matches
   e. Saves checkpoint if `checkpoint_every_n_epochs` matches
3. Catches `StopTraining` for graceful early termination
4. Fires `ON_TRAINING_END` hooks in `finally` block
5. Returns `TrainingResult`

### Single training step internals

```
BEFORE_STEP
  → load batch
AFTER_DATA_LOAD
BEFORE_FORWARD
  → model(batch)
AFTER_FORWARD
BEFORE_LOSS
  → loss(batch, outputs) + reduce_loss_across_ranks
  → scale for gradient accumulation
AFTER_LOSS
BEFORE_BACKWARD
  → scaled_loss.backward() (with optional GradScaler)
AFTER_BACKWARD
  [every grad_accumulation_steps:]
  BEFORE_OPTIMIZER_STEP
    → clip gradients
    → optimizer.step()
    → scheduler.step()
    → model.zero_grad(set_to_none=True)
  AFTER_OPTIMIZER_STEP
AFTER_STEP
  → ctx.global_step += 1
```

---

## Hook system

### TrainingStageEnum

18 hook points grouped by tens:

| Group | Stages |
|-------|--------|
| Epoch | `BEFORE_EPOCH` (0), `AFTER_EPOCH` (1) |
| Step | `BEFORE_STEP` (10), `AFTER_DATA_LOAD` (11), `BEFORE_FORWARD` (12), `AFTER_FORWARD` (13), `BEFORE_LOSS` (14), `AFTER_LOSS` (15), `BEFORE_BACKWARD` (16), `AFTER_BACKWARD` (17), `BEFORE_OPTIMIZER_STEP` (18), `AFTER_OPTIMIZER_STEP` (19), `AFTER_STEP` (20) |
| Validation | `BEFORE_VALIDATION` (30), `AFTER_VALIDATION` (31) |
| Checkpoint | `BEFORE_CHECKPOINT` (40), `AFTER_CHECKPOINT` (41) |
| End | `ON_TRAINING_END` (50) |

### TrainingContext

Mutable dataclass passed to every hook:

| Field | Type | Description |
|-------|------|-------------|
| `epoch` | `int` | Current epoch (0-based) |
| `global_step` | `int` | Cumulative step counter |
| `batch` | `Batch \| None` | Current mini-batch |
| `model_outputs` | `ModelOutputs \| None` | Forward pass outputs |
| `losses` | `TensorDict` | Named per-term losses |
| `total_loss` | `Tensor \| None` | Scalar loss for backward |
| `metrics` | `dict[str, float]` | Accumulated metrics |
| `extra` | `dict[str, Any]` | Free-form hook-to-hook state |
| `stage_counts` | `dict[TrainingStageEnum, int]` | Per-stage fire counter |

### TrainingHook protocol

```python
class TrainingHook(Protocol):
    frequency: int                   # execute every N steps
    stage: TrainingStageEnum         # single stage, or expose `stages` frozenset

    def __call__(self, ctx: TrainingContext, model, trainer) -> None: ...
```

### Writing a custom hook

```python
from nvalchemi.training import TrainingHook, TrainingContext, TrainingStageEnum

class GradNormLogger:
    """Log gradient norm after each backward pass."""

    frequency: int = 1
    stage = TrainingStageEnum.AFTER_BACKWARD

    def __call__(self, ctx: TrainingContext, model, trainer) -> None:
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        ctx.metrics["grad_norm"] = total_norm ** 0.5
```

### Multi-stage hooks

Expose a `stages` frozenset instead of a single `stage`:

```python
class StepTimer:
    frequency: int = 1
    stages = frozenset({TrainingStageEnum.BEFORE_STEP, TrainingStageEnum.AFTER_STEP})

    def __call__(self, ctx, model, trainer):
        import time
        if ctx.stage_counts.get(TrainingStageEnum.BEFORE_STEP, 0) > \
           ctx.stage_counts.get(TrainingStageEnum.AFTER_STEP, 0):
            ctx.extra["step_start"] = time.monotonic()
        else:
            elapsed = time.monotonic() - ctx.extra.get("step_start", 0)
            ctx.metrics["step_time_ms"] = elapsed * 1000
```

### Registering hooks

```python
trainer.register_hook(GradNormLogger())
trainer.register_hook(EarlyStoppingHook(patience=10))
```

---

## Built-in hooks

### EarlyStoppingHook

```python
from nvalchemi.training import EarlyStoppingHook

hook = EarlyStoppingHook(
    metric="val_loss",    # key in ctx.metrics
    patience=5,           # epochs without improvement before stopping
    mode="min",           # "min" or "max"
)
# Stage: AFTER_VALIDATION
# Raises StopTraining when patience is exhausted
```

### SWAHook + SWAFinalizeHook

Stochastic Weight Averaging:

```python
from torch.optim.swa_utils import AveragedModel
from nvalchemi.training import SWAHook, SWAFinalizeHook

swa_model = AveragedModel(model)

trainer.register_hook(SWAHook(swa_model=swa_model, swa_start_step=1000))
trainer.register_hook(SWAFinalizeHook(
    swa_model=swa_model,
    train_loader=train_loader,
    device="cuda",
))
# SWAHook:          stage=AFTER_OPTIMIZER_STEP, calls swa_model.update_parameters()
# SWAFinalizeHook:  stage=ON_TRAINING_END, runs update_bn() for BN stats
```

### TerminateOnStepsHook

```python
from nvalchemi.training import TerminateOnStepsHook, TrainingStageEnum

hook = TerminateOnStepsHook(
    max_count=5000,
    stage=TrainingStageEnum.AFTER_STEP,   # default
)
# Raises StopTraining when ctx.stage_counts[stage] >= max_count
```

---

## Checkpointing

Checkpoints are managed automatically by the trainer at the configured
frequency. You can also use the low-level API directly:

```python
from nvalchemi.training._checkpoint import (
    save_training_checkpoint,
    load_training_checkpoint,
)

# Save — rank-0 only in distributed settings
save_training_checkpoint(
    path=Path("checkpoints/epoch_10.pt"),
    model=model,
    optimizer=optimizer,          # or [opt1, opt2]
    scheduler=scheduler,          # or [sched1, sched2], or None
    scaler=scaler,                # GradScaler or None
    epoch=10,
    metrics={"val_loss": 0.05},
    dist_manager=dist_manager,
)

# Load — returns dict with "epoch" and "metrics"
state = load_training_checkpoint(
    path=Path("checkpoints/epoch_10.pt"),
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    device=torch.device("cuda:0"),
)
# state == {"epoch": 10, "metrics": {"val_loss": 0.05}}
```

### Resume from checkpoint

```python
config = TrainingConfig(
    max_epochs=100,
    resume_from=Path("checkpoints/epoch_49.pt"),
)
# Trainer.__init__ calls _maybe_resume(), restoring model/optimizer/scheduler
# state and setting _start_epoch = 49
```

---

## Distributed training

### Wrapping the model

The trainer calls `wrap_model` internally, but you can also use it directly:

```python
from nvalchemi.training._distributed import wrap_model, reduce_loss_across_ranks

# Automatic wrapping in Trainer:
config = TrainingConfig(max_epochs=10, strategy="ddp")  # or "fsdp"

# Manual wrapping:
wrapped = wrap_model(model, strategy="ddp", dist_manager=dist_manager)
```

- Returns the model unchanged if `dist_manager` is `None` or world_size <= 1
- `"ddp"` wraps with `DistributedDataParallel`
- `"fsdp"` wraps with `FullyShardedDataParallel`

### Loss reduction across ranks

```python
# Called automatically in the training loop after loss computation
reduced = reduce_loss_across_ranks(loss, dist_manager)
```

Uses `physicsnemo.distributed.utils.reduce_loss` under the hood. Returns
the input tensor unchanged in single-process mode.

---

## nvalchemiops operations used in training

The training module directly consumes the following `nvalchemiops` operations:

| Operation | Source | Used by | Purpose |
|-----------|--------|---------|---------|
| `segmented_sum` | `nvalchemiops.segment_ops` | `_reductions._segmented_sum` | Aggregate node-level errors to per-graph |
| `segmented_mean` | `nvalchemiops.segment_ops` | `_reductions._segmented_mean` | Average node-level errors per graph |

These are wrapped as `torch.library.custom_op` to create opaque boundaries
for `torch.compile`, preventing it from tracing into Warp interop code.

### Pattern for wrapping nvalchemiops in training code

```python
import torch
import warp as wp
from nvalchemiops.segment_ops import segmented_sum

@torch.library.custom_op("nvalchemi_training::segmented_sum", mutates_args=())
def _segmented_sum(values: torch.Tensor, idx: torch.Tensor, num_segments: int) -> torch.Tensor:
    out = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    segmented_sum(
        wp.from_torch(values.contiguous()),
        wp.from_torch(idx.to(torch.int32)),
        wp.from_torch(out),
    )
    return out

@_segmented_sum.register_fake
def _(values: torch.Tensor, idx: torch.Tensor, num_segments: int) -> torch.Tensor:
    return torch.empty(num_segments, device=values.device, dtype=values.dtype)
```

Key points:
- Always pre-allocate the output buffer (`torch.zeros`)
- Convert indices to `int32` (Warp requirement)
- Use `.contiguous()` on input tensors
- Register a `fake` impl returning the correct shape/dtype for `torch.compile`

---

## Full workflow example

```python
import torch
from pathlib import Path
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import (
    Trainer,
    TrainingConfig,
    MixedPrecisionConfig,
    GradClipConfig,
    EarlyStoppingHook,
    TerminateOnStepsHook,
    TrainingStageEnum,
)
from nvalchemi.training.losses import EnergyLoss, ForceLoss, StressLoss

# 1. Model (must implement BaseModelMixin)
model = MyMLIP()  # see nvalchemi-model-wrapping skill

# 2. Loss
loss = 1.0 * EnergyLoss() + 10.0 * ForceLoss() + 0.1 * StressLoss()

# 3. Optimizer + scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

# 4. Data loaders (yield Batch objects)
train_loader = ...  # see nvalchemi-data-storage skill
val_loader = ...

# 5. Config
config = TrainingConfig(
    max_epochs=100,
    grad_clip=GradClipConfig(method="norm", max_value=1.0),
    grad_accumulation_steps=1,
    val_every_n_epochs=1,
    checkpoint_every_n_epochs=10,
    checkpoint_dir=Path("checkpoints"),
    mixed_precision=MixedPrecisionConfig(enabled=True, amp_type="bf16"),
    seed=42,
)

# 6. Train
with Trainer(
    model=model,
    loss=loss,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_loader,
    val_loader=val_loader,
    config=config,
) as trainer:
    trainer.register_hook(EarlyStoppingHook(patience=10))
    trainer.register_hook(TerminateOnStepsHook(max_count=50000))
    result = trainer.fit()

print(f"Completed {result.epochs_completed} epochs")
print(f"Best val loss: {result.best_val_loss:.6f}")
```

### Multi-optimizer example

```python
# Separate optimizers for backbone and head
backbone_params = [p for n, p in model.named_parameters() if "head" not in n]
head_params = [p for n, p in model.named_parameters() if "head" in n]

opt_backbone = torch.optim.AdamW(backbone_params, lr=1e-4)
opt_head = torch.optim.AdamW(head_params, lr=1e-3)

sched_backbone = torch.optim.lr_scheduler.StepLR(opt_backbone, step_size=30)
sched_head = torch.optim.lr_scheduler.StepLR(opt_head, step_size=30)

with Trainer(
    model=model,
    loss=loss,
    optimizer=[opt_backbone, opt_head],
    scheduler=[sched_backbone, sched_head],
    train_loader=train_loader,
    config=config,
) as trainer:
    result = trainer.fit()
```

### Distributed training example

```python
from physicsnemo.distributed import DistributedManager

DistributedManager.initialize()
dist = DistributedManager()

config = TrainingConfig(
    max_epochs=100,
    strategy="ddp",
    mixed_precision=MixedPrecisionConfig(enabled=True, amp_type="bf16"),
)

with Trainer(
    model=model,
    loss=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    config=config,
    dist_manager=dist,
) as trainer:
    result = trainer.fit()
```

### SWA (Stochastic Weight Averaging) example

```python
from torch.optim.swa_utils import AveragedModel
from nvalchemi.training import SWAHook, SWAFinalizeHook

swa_model = AveragedModel(model)

with Trainer(model=model, loss=loss, optimizer=optimizer,
             train_loader=train_loader, config=config) as trainer:
    trainer.register_hook(SWAHook(swa_model=swa_model, swa_start_step=5000))
    trainer.register_hook(SWAFinalizeHook(
        swa_model=swa_model,
        train_loader=train_loader,
    ))
    result = trainer.fit()

# swa_model now contains the averaged weights with updated BN stats
```
