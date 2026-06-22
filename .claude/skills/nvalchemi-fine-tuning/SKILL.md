---
name: nvalchemi-fine-tuning
description: >-
  How to fine-tune nvalchemi-compatible models with FineTuningStrategy,
  pretrained checkpoint initialization, module patches, trainable-parameter
  filters, conservative optimizer defaults, validation, restart checkpoints,
  and model-agnostic MACE, AIMNet2, custom BaseModelMixin, or PyTorch inputs.
---

# nvalchemi Fine Tuning

## Overview

Use `FineTuningStrategy` when adapting pretrained weights to a new dataset,
objective, trainable parameter set, or model head. Link users to
`docs/userguide/finetuning.md`, `docs/userguide/training.md`,
`docs/userguide/models.md`, and `docs/userguide/losses.md` for full details.

```python
import torch

from nvalchemi.training import (
    CheckpointHook,
    EnergyMSELoss,
    FineTuningStrategy,
    ForceMSELoss,
    OptimizerConfig,
    ValidationConfig,
    create_model_spec,
    default_training_fn,
)
```

---

## CLI Planning

Use `nvalchemi-finetune` when the user wants an offline JSON specification, an
endpoint scaffold, or a Rich intent report before execution. The main groups are
`init` (`checkpoint`, `mace`, `aimnet2`, `custom`), `schema` (`dump`,
`template`), and `spec` (`validate`, `report`). Treat the CLI as a planning and
review surface; user model and dataloader execution still belongs in scripts.
Expect `spec report` to include a warning section for common mistakes such
as high fine-tuning learning rates, missing validation data, unsafe
checkpoint output paths, or MACE compile settings.

---

## Choose The Entry Point

- Use `FineTuningStrategy(models=...)` when the user already loaded or built a
  trainable model.
- Use `FineTuningStrategy.from_pretrained_checkpoint(...)` to start a fresh
  fine-tuning run from model weights in a native nvalchemi checkpoint.
- Use `FineTuningStrategy.load_checkpoint(...)` only to resume an interrupted
  fine-tuning run with optimizer/scheduler/counters/hook state.

`from_pretrained_checkpoint` loads the complete checkpoint model set. A
single-model checkpoint becomes a single model input; multi-model checkpoints
preserve their named mapping. Source optimizer state, hooks, validation
settings, counters, and `num_epochs`/`num_steps` do not carry over. If the user
omits `loss_fn` or `optimizer_configs`, they may opt into source metadata with
`use_original_loss=True` or `use_original_opt_class=True`. Reused optimizer
configs get `optimizer_lr=1e-5` by default; pass `optimizer_lr=None` to keep the
checkpoint LR.

---

## Minimal Pattern

```python
strategy = FineTuningStrategy(
    models=pretrained_model,
    trainable_patterns=("main.model.readout.*",),
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 3e-4, "weight_decay": 1e-6},
    ),
    training_fn=default_training_fn,
    loss_fn=EnergyMSELoss() + ForceMSELoss(normalize_by_atom_count=True),
    validation_config=ValidationConfig(validation_data=val_loader, every_n_epochs=1),
    hooks=[CheckpointHook("runs/finetune/checkpoints", epoch_interval=1)],
    num_epochs=10,
)
strategy.run(train_loader)
```

Use low learning rates for full-model fine-tuning. Prefer `trainable_patterns`
for head-only or adapter-style workflows; patterns match fully qualified names
such as `"main.model.readout.weight"`.

---

## From A Native Checkpoint

Use this when a previous nvalchemi run produced a restartable checkpoint but the
new task should get fresh fine-tuning counters and optional source loss/optimizer
metadata.

```python
strategy = FineTuningStrategy.from_pretrained_checkpoint(
    "runs/pretrain/checkpoints",
    use_original_loss=True,
    use_original_opt_class=True,
    optimizer_lr=1e-5,
    training_fn=default_training_fn,
    trainable_patterns=("main.model.readout.*",),
    num_steps=2_000,
)
```

For multi-model checkpoints, write `training_fn(models, batch)` and pass
`optimizer_configs` keyed by the model(s) to update. Models omitted from
`optimizer_configs` are frozen/eval during training but can be used as teachers
or references. `use_original_loss` and `use_original_opt_class` require native
strategy metadata; they do not work with component-only checkpoints.

---

## Bring Your Own Model Or Foreign Checkpoint

Prefer native wrapper constructors for supported pretrained models, for example
`MACEWrapper.from_checkpoint(..., compile_model=False)`, because they preserve
reconstruction metadata for later strategy checkpoints. `compile_model=True` is
inference-only for MACE and freezes parameters.

For arbitrary PyTorch checkpoints:

- Instantiate the architecture through a wrapper class when possible.
- Use `create_model_spec(wrapper_cls_or_factory, ...)` for reproducible rebuilds.
- Load weights with `state_dict`; use `strict=False` only for intentional head or
  adapter changes and inspect missing/unexpected keys.
- If output keys differ from loss keys, write a `training_fn` that returns the
  mapping expected by the loss.
- Treat foreign checkpoints as weight imports, not restart checkpoints. Save a
  fresh `FineTuningStrategy` checkpoint before relying on resume behavior.

```python
state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
model.load_state_dict(state["model"] if "model" in state else state, strict=False)
model_spec = create_model_spec(MyWrapper.from_pretrained, checkpoint_path=str(checkpoint_path))
```

---

## Patch Or Freeze The Model

Use `module_patches` to replace or add child modules before optimizer
construction. Use `create_model_spec(...)` for patches that must serialize;
direct module instances are runtime-only.

```python
strategy = FineTuningStrategy(
    models=pretrained_model,
    module_patches={
        "main.model.readout": create_model_spec(
            torch.nn.Linear,
            in_features=128,
            out_features=1,
        )
    },
    freeze_patterns=("main.model.*",),
    trainable_patterns=("main.model.readout.*",),
    optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.AdamW),
    training_fn=default_training_fn,
    loss_fn=EnergyMSELoss(),
    num_steps=1_000,
)
```

`trainable_patterns` alone is an allow-list. `freeze_patterns` excludes broad
regions first, then `trainable_patterns` re-includes exceptions. Use
`freeze_mode="optimizer_only"` only when frozen parameters should still receive
gradients for diagnostics or custom hooks.

Typical strategies to fine-tune without catastrophic forgetting is to add
different readout/output heads, or to add a new atom embedding table. Users
will likely need a way to route based on dataset. If the user does not specify
a strategy, you should discuss this with them with proposed solutions tailored
to the model and fine-tuning dataset.

---

## Caveats

- Check target/prediction units, atom ordering, neighbor-list assumptions, PBC,
  dtype, device, and output shapes before training.
- Enable force/stress outputs in the model config when those losses need
  autograd-derived quantities.
- Start with validation and short checkpoint intervals; pretrained runs can
  regress quickly with mismatched data or too-large learning rates.
- Resume interrupted fine-tuning with `FineTuningStrategy.load_checkpoint`, not
  `from_pretrained_checkpoint`.
