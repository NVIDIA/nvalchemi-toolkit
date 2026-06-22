---
name: nvalchemi-fine-tuning
description: >-
  How to fine-tune arbitrary nvalchemi-compatible models from pretrained or
  user-provided checkpoints. Use when adapting MACE, AIMNet2, custom
  BaseModelMixin wrappers, PyTorch modules, or foreign checkpoint weights with
  TrainingStrategy, low learning rates, validation, restart checkpoints, and
  reproducible model reconstruction.
---

# nvalchemi Fine Tuning

## Overview

Fine-tuning is `TrainingStrategy` training that starts from pretrained weights and
usually changes the data, objective, optimizer setup, or trainable parameter set.
For full API details, link agents to `docs/userguide/training.md`,
`docs/userguide/models.md`, and `docs/userguide/losses.md`.

```python
import torch

from nvalchemi.training import (
    CheckpointHook,
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStrategy,
    ValidationConfig,
    create_model_spec,
)
```

---

## Workflow

1. Load or construct the pretrained model.
2. Prefer a wrapper class that accepts `AtomicData`/`Batch` and returns standard
   prediction keys; otherwise provide a custom `training_fn`.
3. Choose which parameters the fine-tuning strategy should train.
4. Configure conservative learning rates, validation from epoch 1, and restart
   checkpoints.

```python
for name, param in model.named_parameters():
    param.requires_grad_(not name.startswith("backbone."))

loss_fn = 1.0 * EnergyMSELoss() + 10.0 * ForceMSELoss()

strategy = TrainingStrategy(
    models=model,
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-5, "weight_decay": 1e-6},
    ),
    loss_fn=loss_fn,
    validation=ValidationConfig(validation_data=val_loader, every_n_epochs=1),
    hooks=[CheckpointHook("runs/finetune/checkpoints", epoch_interval=1)],
    num_epochs=10,
)
strategy.run(train_loader)
```

When a first-class fine-tuning strategy exists in the branch, prefer that API for
parameter selection and freezing instead of open-coded `requires_grad_` loops.

---

## Bring Your Own Model Or Checkpoint

Prefer native wrapper constructors for supported pretrained models, for example
`MACEWrapper.from_checkpoint(...)`, because they preserve reconstruction metadata
for later strategy checkpoints.

For arbitrary PyTorch checkpoints:

- Instantiate the architecture through a wrapper class when possible.
- Use `create_model_spec(wrapper_cls_or_factory, ...)` for reproducible rebuilds.
- Load weights with `state_dict`; use `strict=False` only for intentional head or
  adapter changes and inspect missing/unexpected keys.
- If output keys differ from loss keys, write a `training_fn` that returns the
  mapping expected by the loss.
- Treat foreign checkpoints as weight imports, not restart checkpoints. Save a
  fresh `TrainingStrategy` checkpoint before relying on resume behavior.

```python
state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model.load_state_dict(state["model"] if "model" in state else state, strict=False)
model_spec = create_model_spec(MyWrapper.from_pretrained, checkpoint_path=str(checkpoint_path))
```

---

## Multi-Model Fine-Tuning

For student-teacher, adapter, or frozen-reference workflows, pass named models.
Only put trainable models in `optimizer_configs`; unconfigured models are frozen
inside the training step.

```python
def training_fn(models, batch):
    student_out = models["student"](batch)
    with torch.no_grad():
        ref_out = models["reference"](batch)
    return {
        "predicted_energy": student_out["energy"],
        "reference_energy": ref_out["energy"].detach(),
    }

strategy = TrainingStrategy(
    models={"student": student, "reference": reference},
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

## Caveats

- Check target/prediction units, atom ordering, neighbor-list assumptions, PBC,
  dtype, device, and output shapes before training.
- Enable force/stress outputs in the model config when those losses need
  autograd-derived quantities.
- Start with validation and short checkpoint intervals; pretrained runs can
  regress quickly with mismatched data or too-large learning rates.
- Native strategy checkpoints can resume optimizer/scheduler/hook state; plain
  pretrained weight files cannot.
