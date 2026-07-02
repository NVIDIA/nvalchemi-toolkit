---
name: nvalchemi-training-api
description: >-
  How to configure nvalchemi training workflows with TrainingStrategy, custom
  training functions, standalone or composed losses, loss-weight schedules,
  optimizer and scheduler configs, validation, hooks, restartable checkpoints,
  and model-agnostic inputs. Use when training a model from scratch or setting
  up optimizers, schedulers, validation, or checkpointing for a training run;
  for adapting a pretrained model, see nvalchemi-fine-tuning.
---

# nvalchemi Training API

## Overview

Use `TrainingStrategy` as the owner of one training job: model(s), dataloaders,
loss, optimizer/scheduler config, validation, hooks, runtime counters, and
checkpoints. For full details, link agents to `docs/userguide/training.md`,
`docs/userguide/losses.md`, and `docs/modules/training/checkpoints.rst`.

```python
import torch

from nvalchemi.data import Batch
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import (
    CheckpointHook,
    ComposedLossFunction,
    CosineWeight,
    EnergyMSELoss,
    ForceMSELoss,
    LinearWeight,
    OptimizerConfig,
    StressMSELoss,
    TrainingStrategy,
    ValidationConfig,
    create_model_spec,
)
```

---

## Minimal Pattern

```python
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
    validation_config=ValidationConfig(validation_data=val_loader, every_n_epochs=1),
    hooks=[CheckpointHook("runs/example/checkpoints", epoch_interval=1)],
    num_epochs=20,
)
strategy.run(train_loader)
```

---

## Model-Agnostic Inputs

Accept any `torch.nn.Module` that works with the selected `training_fn`. Prefer
wrapped `BaseModelMixin` models for standard `AtomicData`/`Batch` contracts;
load `nvalchemi-model-wrapping` or `docs/userguide/models.md` when adapting
arbitrary MLIPs.

Make model construction reproducible when possible. Use native checkpoint
constructors that carry a spec, or store a `create_model_spec(...)` for custom
wrappers so strategy checkpoints can rebuild the model before loading weights.
Treat foreign checkpoints as imported weights until a fresh `TrainingStrategy`
checkpoint has been saved.

---

## Custom Training Functions

Use `training_fn` when the batch needs custom routing, multiple models, teacher
outputs, auxiliary predictions, or non-standard model outputs. It receives
`(model, batch)` for a single model or `(models, batch)` for named models and
returns the prediction mapping consumed by `loss_fn`.

For multiple models, pass a named mapping. `optimizer_configs` must use the same
model keys for trainable models. Models absent from `optimizer_configs` may be
used in the forward path but are frozen during training.

```python
def training_fn(models: dict[str, BaseModelMixin], batch: Batch):
    student = models["student"](batch)
    with torch.no_grad():
        teacher = models["teacher"](batch)
    return {
        "student_energy": student["energy"],
        "teacher_energy": teacher["energy"].detach(),
    }

loss_fn = ComposedLossFunction(
    [EnergyMSELoss(prediction_key="student_energy", target_key="teacher_energy")]
)

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

If targets do not come directly from the batch, also provide a
`loss_target_assembler`; see `docs/userguide/training.md`.

---

## Losses And Scheduling

A standalone leaf loss such as `EnergyMSELoss()` can be used when the objective
has one target. Use `ComposedLossFunction` or operator sugar for multi-target
objectives. Leaf losses consume unweighted tensors; weights and schedules live on
the composition. Built-in schedules include `ConstantWeight`, `LinearWeight`,
`CosineWeight`, and `PiecewiseWeight`.

Built-in losses default to `dtype_policy="strict"` and raise when prediction
and target dtypes differ. When building or reviewing workflows, check likely
label/model dtype alignment, such as float64 dataset labels with float32 model
outputs. If the mismatch is intentional, tell the user they can set
`dtype_policy="prediction_to_target"` to cast outputs to labels or
`dtype_policy="target_to_prediction"` to cast labels to outputs. Set the
policy on an explicit `ComposedLossFunction(...)`, on a leaf loss, or after
operator-sugar construction:

```python
loss_fn = EnergyMSELoss() + ForceMSELoss()
loss_fn.dtype_policy = "prediction_to_target"
```

A leaf loss with its own explicit `dtype_policy` overrides the composed-level
policy. The setting is included in serializable loss specs for restartable
training workflows. For CLI scaffolds, pass `--loss-dtype-policy strict`,
`--loss-dtype-policy prediction_to_target`, or
`--loss-dtype-policy target_to_prediction` to `nvalchemi-training train init`
or `nvalchemi-training finetune init ...`; `spec report` shows the selected
policy.

```python
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
- For custom leaf-loss internals, use `nvalchemi-loss-api` and
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

## Checkpoints And Reproducibility

Agent-created training workflows should be fully checkpointable and reproducible:

- Use deterministic model/wrapper constructors or `create_model_spec(...)`.
- Keep loss functions, schedules, optimizer configs, and restart-critical hooks
  serializable; implement `to_spec()` where protocols require it.
- Use `CheckpointHook` for periodic checkpoints and save early enough for preempted
  jobs, including Slurm-style cluster runs.
- Make data splits, sampler state, seeds, units, dtype/device choices, and config
  files explicit in the run directory.

Strategy checkpoints are restart packages: model weights, optimizer and scheduler
state, strategy counters, checkpointable hook state, and reconstruction metadata.

---

## Resume Training

Use resume when continuing the same run after interruption. This is different
from fine-tuning, which imports weights into a new objective or dataset.

```python
strategy = TrainingStrategy.load_checkpoint("runs/example/checkpoints", map_location="cuda")
strategy.run(train_loader)
```

Resume only from native `TrainingStrategy` checkpoints when optimizer, scheduler,
hook state, and counters matter. Plain pretrained weight files are not sufficient
for faithful continuation. To start a fresh fine-tuning run from native
checkpoint weights, use `FineTuningStrategy.from_pretrained_checkpoint(...)` from
`nvalchemi-fine-tuning`; opt into source loss or optimizer classes with
`use_original_loss=True` or `use_original_opt_class=True` when those defaults are
desired. See `docs/modules/training/checkpoints.rst`.
