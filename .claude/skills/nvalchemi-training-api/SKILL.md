---
name: nvalchemi-training-api
description: >-
  How to configure nvalchemi training workflows with TrainingStrategy, custom
  training functions, composed losses, loss-weight schedules, optimizer and
  scheduler configs, validation, hooks, and checkpoints. Use when creating or
  modifying training scripts, adding new loss functions to training, configuring
  optimizers, or wiring arbitrary wrapped models into the training API.
---

# nvalchemi Training API

## Overview

Use `TrainingStrategy` as the owner of one training job: model(s), dataloaders,
loss, optimizer/scheduler config, validation, hooks, runtime counters, and
checkpoints. For full details, link agents to `docs/userguide/training.md` and
`docs/userguide/losses.md`.

```python
from nvalchemi.training import (
    CheckpointHook,
    ComposedLossFunction,
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStrategy,
    ValidationConfig,
)
```

---

## Minimal Pattern

```python
import torch

loss_fn = ComposedLossFunction(
    [EnergyMSELoss(), ForceMSELoss()],
    weights=[1.0, 10.0],
    normalize_weights=False,
)

strategy = TrainingStrategy(
    models=model,
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-4, "weight_decay": 1e-5},
    ),
    loss_fn=loss_fn,
    validation=ValidationConfig(validation_data=val_loader, every_n_epochs=1),
    hooks=[CheckpointHook("runs/example/checkpoints", epoch_interval=1)],
    num_epochs=20,
)
strategy.run(train_loader)
```

---

## Model-Agnostic Inputs

Accept any `torch.nn.Module` that works with the selected `training_fn`.
Prefer wrapped `BaseModelMixin` models for standard `Batch` input/output
contracts; load `nvalchemi-model-wrapping` or `docs/userguide/models.md` when
adapting arbitrary MLIPs.

For multiple models, pass a named mapping and write `training_fn(models, batch)`.
Optimizer configs must use the same model keys. Models absent from
`optimizer_configs` are available in the forward path but frozen during training.

```python
def training_fn(models, batch):
    student = models["student"](batch)
    with torch.no_grad():
        teacher = models["teacher"](batch)
    return {
        "predicted_energy": student["energy"],
        "teacher_energy": teacher["energy"].detach(),
    }

strategy = TrainingStrategy(
    models={"student": student_model, "teacher": teacher_model},
    optimizer_configs={
        "student": [
            OptimizerConfig(
                optimizer_cls=torch.optim.AdamW,
                optimizer_kwargs={"lr": 3e-5},
            )
        ]
    },
    training_fn=training_fn,
    loss_fn=loss_fn,
    num_epochs=5,
)
```

---

## Losses And Scheduling

Use `ComposedLossFunction` for multi-target objectives. Leaf losses consume
unweighted tensors; weights and schedules live on the composition. Built-in
schedules include `ConstantWeight`, `LinearWeight`, `CosineWeight`, and
`PiecewiseWeight`.

```python
from nvalchemi.training import CosineWeight, LinearWeight, StressMSELoss

loss_fn = (
    1.0 * EnergyMSELoss()
    + LinearWeight(start=0.0, end=10.0, num_steps=1000) * ForceMSELoss()
    + CosineWeight(start=0.0, end=0.1, num_steps=5000) * StressMSELoss()
)
```

Caveats:

- `normalize_weights=True` is the default; set `False` for raw coefficient sums.
- `per_epoch=True` schedules require `epoch` during loss calls.
- Custom schedules must implement `per_epoch`, `__call__(step, epoch)`, and
  `to_spec()` if they are used in restartable strategy checkpoints.
- For custom leaf-loss internals, use the existing `nvalchemi-loss-api` skill and
  `docs/userguide/losses.md`.

---

## Optimizers And Schedulers

Use `OptimizerConfig(optimizer_cls=..., optimizer_kwargs=...)`; add
`scheduler_cls` and `scheduler_kwargs` when needed. Keyword arguments are
validated against class constructors before training starts.

Time-based schedulers step after optimizer steps. `ReduceLROnPlateau`-style
metric schedulers step after validation; set `scheduler_metric_adapter` to a
validation-summary key or callable when the default `"total_loss"` is not right.

---

## Checkpoints

Use `CheckpointHook` for periodic restart checkpoints and
`TrainingStrategy.save_checkpoint(...)` / `TrainingStrategy.load_checkpoint(...)`
for explicit save/load. Strategy checkpoints are restart packages: model weights,
optimizer and scheduler state, strategy counters, and checkpointable hook state.
For checkpoint reconstruction details, see
`docs/modules/training/checkpoints.rst`.
