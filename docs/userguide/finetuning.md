(finetuning_guide)=

# Fine-Tuning Pretrained Models

Fine-tuning adapts a pretrained interatomic potential to a new dataset while
reusing most of the training loop machinery from
{py:class}`~nvalchemi.training.TrainingStrategy`. The
{py:class}`~nvalchemi.training.FineTuningStrategy` convenience wrapper adds two
pieces that are common in transfer-learning workflows:

- patch one or more {py:class}`torch.nn.Module` children before optimizers are
  built;
- choose which parameters are trainable with glob patterns.

This guide assumes that you already have:

- a pretrained model wrapped with
  {py:class}`~nvalchemi.models.base.BaseModelMixin`;
- a dataset or dataloader that yields {py:class}`~nvalchemi.data.Batch` objects;
- target tensors whose keys match the configured losses.

For those prerequisites, see {ref}`models_guide`, {ref}`data_guide`,
{ref}`datapipes_guide`, and {ref}`losses_guide`.

```{warning}
Load pretrained models in a trainable form before passing them to
`FineTuningStrategy`. For MACE checkpoints,
{py:meth}`nvalchemi.models.mace.MACEWrapper.from_checkpoint` returns an
eval-mode wrapper, and the training strategy temporarily switches configured
models into train mode during `run`. However, `compile_model=True` is
inference-only: it freezes parameters before `torch.compile`. Use
`compile_model=False` for fine-tuning.
```

## Training CLI

Use `nvalchemi-training finetune` to scaffold, review, and start
fine-tuning specifications for quick experimentation without requiring full
knowledge of the Python training API. The CLI records the source model, dataset
intent, output paths, runtime hooks, and a JSON-ready
`FineTuningStrategy.to_spec_dict()` bundle. `spec report` renders a Rich
report card showing what the run will consume and write, validates local
dataset and checkpoint paths, checks that serialized hooks can be built as
`Hook` or `CheckpointableHook` instances, previews the learning-rate schedule,
and lists runtime hooks in chronological firing order. The report also includes
heuristic warnings for common fine-tuning mistakes, such as high learning
rates, missing validation data, unsafe checkpoint paths, or inference-oriented
wrapper settings.

For workflows that need arbitrary Python code, custom model construction,
programmatic data routing, dynamic loss logic, or non-standard orchestration,
write a script with `FineTuningStrategy` directly. The CLI optimizes the common
path; scripts remain the flexible power-user interface.

```bash
nvalchemi-training finetune init checkpoint runs/pretrain/checkpoints \
  --dataset data/domain.zarr \
  --output-dir runs/domain-ft \
  --trainable-pattern 'main.model.readout.*' \
  --out finetune.json

# Repeat --dataset to record a MultiDataset-backed workflow.
nvalchemi-training finetune init mace small-0b \
  --dataset data/domain-a.zarr \
  --dataset data/domain-b.zarr \
  --output-dir runs/mace-ft \
  --out mace-ft.json

nvalchemi-training schema dump --out finetune.schema.json
nvalchemi-training spec report finetune.json
nvalchemi-training spec run finetune.json
```

For distributed execution, launch the same spec through `torchrun` and pass
`--distributed`. The CLI initializes `DistributedManager`, prepends `DDPHook`,
builds the dataset or `MultiDataset`, constructs the strategy, and calls
`run(...)`.

```bash
torchrun --nproc_per_node=4 -m nvalchemi.training.cli \
  spec run finetune.json --distributed
```

Fine-tuning scaffold commands are available under `nvalchemi-training finetune init` for
`checkpoint`, `mace`, `aimnet2`, and `custom` sources. Training-from-scratch scaffolds
are available under `nvalchemi-training train init`. MACE scaffolds default to
`compile_model=false` because compiled MACE wrappers are inference-only for
fine-tuning.

Runtime hooks belong in `source.hooks`. Each hook entry contains a `spec`
object with the serialized `BaseSpec` fields (`cls_path`, `timestamp`, and the
hook constructor keyword fields). The optional `stages` list overrides the
`TrainingStage` values where the hook fires; multiple stages build one hook
instance per stage. For model-input transforms such as neighbor-list
construction, use `BEFORE_FORWARD`, which runs during both training and
strategy-owned validation.

```json
{
  "source": {
    "hooks": [
      {
        "spec": {
          "cls_path": "nvalchemi.hooks.neighbor_list.NeighborListHook",
          "timestamp": "2026-01-01T00:00:00+00:00",
          "config": {"cls_path": "...", "timestamp": "...", "cutoff": 5.0}
        },
        "stages": ["BEFORE_FORWARD"]
      }
    ]
  }
}
```

## Checkpoint workflows

Fine-tuning has two checkpoint workflows with different intent:

| Goal | API | Restores optimizer/scheduler/counters? |
| --- | --- | --- |
| Resume an interrupted fine-tuning run | `FineTuningStrategy.load_checkpoint(...)` | Yes |
| Start a new fine-tuning run from prior model weights | `FineTuningStrategy.from_pretrained_checkpoint(...)` | No |
| Fine-tune a model you loaded yourself | `FineTuningStrategy(models=...)` | No |

Use `load_checkpoint` when continuing the same fine-tuning job after
interruption. It restores the saved strategy state, including model weights,
optimizer state, scheduler state, runtime counters, checkpointable hook state,
and serialized fine-tuning configuration.

Use `from_pretrained_checkpoint` when branching a new fine-tuning job from a
checkpoint written by {py:class}`~nvalchemi.training.hooks.CheckpointHook` or
{py:meth}`~nvalchemi.training.TrainingStrategy.save_checkpoint`. It loads the
complete checkpoint model set as initialization, then builds fresh optimizers,
schedulers, counters, losses, module patches, trainable-parameter filters, and
runtime hooks from the arguments you pass to the new strategy. Source strategy
settings such as `num_epochs`, `num_steps`, optimizer classes, scheduler
configuration, hooks, and validation settings do not bleed into the new
fine-tuning strategy by default. Set `use_original_loss=True` to reuse the
checkpointed loss when `loss_fn` is omitted. Set `use_original_opt_class=True`
to reuse checkpointed optimizer/scheduler config when `optimizer_configs` is
omitted; reused optimizer configs get a conservative fine-tuning `optimizer_lr`
default of `1e-5` unless `optimizer_lr=None` is passed. For multi-model
checkpoints, the training function and optimizer configuration you pass to the
fine-tuning strategy decide which models participate in the new workflow.

```python
# Resume the same fine-tuning run.
resumed = FineTuningStrategy.load_checkpoint(
    "runs/domain-ft/checkpoints",
    training_fn=default_training_fn,
)

# Start a new fine-tuning run from a previous checkpoint's model weights.
strategy = FineTuningStrategy.from_pretrained_checkpoint(
    "runs/pretrain/checkpoints",
    trainable_patterns=("main.model.readout.*",),
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 3e-4},
    ),
    training_fn=default_training_fn,
    loss_fn=EnergyMSELoss(),
    num_steps=2_000,
)
```

For multi-model checkpoints, `from_pretrained_checkpoint` preserves the named
model mapping from the checkpoint. Use your new `training_fn`,
`optimizer_configs`, `module_patches`, and parameter filters to decide which
loaded models are trained, frozen, ignored, or used as references. Reusing the
source loss or optimizer config is explicit and initialization-only; it never
restores optimizer state or runtime counters.

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

## Simple full-model fine-tuning

The simplest workflow is to load a pretrained model and continue training all
of its parameters on your dataset. This is often a useful baseline, but it is
also the most likely workflow to cause catastrophic forgetting: the model may
adapt to the new domain while losing accuracy on the broader distribution it
was pretrained on.

```python
import torch

from nvalchemi.training import (
    EnergyMSELoss,
    FineTuningStrategy,
    ForceMSELoss,
    OptimizerConfig,
    default_training_fn,
)

pretrained_model = load_my_pretrained_model()
train_loader = make_my_batch_loader()
loss_fn = EnergyMSELoss() + ForceMSELoss(normalize_by_atom_count=True)
# Optional: align model outputs to label dtype before loss validation.
loss_fn.dtype_policy = "prediction_to_target"

strategy = FineTuningStrategy(
    models=pretrained_model,
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-5},
    ),
    training_fn=default_training_fn,
    loss_fn=loss_fn,
    num_epochs=5,
    devices=[torch.device("cuda")],
)

strategy.run(train_loader)
```

`FineTuningStrategy` uses the same forward and loss contracts as
`TrainingStrategy`. The default single-model training function calls
`model(batch)` and prefixes outputs with `"predicted_"` so the built-in
losses can consume keys such as `"predicted_energy"` and
`"predicted_forces"`. Data type alignment is configured on the loss exactly as in
regular training; with operator-composed losses, set `loss_fn.dtype_policy`
after constructing the loss. See {ref}`dtype_alignment` for the full policy
behavior.

```{warning}
Full-model fine-tuning updates every optimizer-visible parameter. Use a
small learning rate, early stopping, validation on the original domain, or a
frozen-base workflow when preserving pretrained behavior matters.
```

## Inspecting names for patches and filters

Fine-tuning fields use fully-qualified names. For a single model passed as
`models=pretrained_model`, the strategy stores it under the key `"main"`.
That means module and parameter names are prefixed with `"main."`.

Print the names before writing patches or freeze patterns:

```python
for name, module in pretrained_model.named_modules():
    print(f"main.{name}", type(module).__name__)

for name, parameter in pretrained_model.named_parameters():
    print(f"main.{name}", tuple(parameter.shape))
```

For a mapping such as `models={"student": student, "teacher": teacher}`, the
prefix is the mapping key instead of `"main"`. A patch target has the form
`"<model_key>.<parent_module_path>.<child_name>"`. Parameter filters use glob
patterns against names such as `"main.model.readout.weight"`.

## Freezing the base model

To reduce forgetting, freeze the pretrained body and train only a narrow set of
parameters. `trainable_patterns` alone acts as an allow-list: only matching
parameters enter the optimizer, and all other parameters are temporarily marked
`requires_grad=False` during `run`.

```python
strategy = FineTuningStrategy(
    models=pretrained_model,
    trainable_patterns=("main.model.readout.*",),
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 3e-4},
    ),
    training_fn=default_training_fn,
    loss_fn=EnergyMSELoss(),
    num_steps=2_000,
    devices=[torch.device("cuda")],
)

strategy.run(train_loader)
```

Use both `freeze_patterns` and `trainable_patterns` when the readable form
is "freeze this broad region, then unfreeze these exceptions":

```python
strategy = FineTuningStrategy(
    models=pretrained_model,
    freeze_patterns=("main.model.*",),
    trainable_patterns=("main.model.readout.*",),
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 3e-4},
    ),
    training_fn=default_training_fn,
    loss_fn=EnergyMSELoss(),
    num_epochs=20,
)
```

Each pattern must match at least one parameter. This catches misspellings and
model-version drift before a long run starts.

## Adding or replacing an output head

Use `module_patches` when the target task needs a new module, for example a
new output head for a dataset-specific property. Patches are applied before
parameter filters and before optimizers are constructed.

```python
import torch

from nvalchemi.training import create_model_spec

strategy = FineTuningStrategy(
    models=pretrained_model,
    module_patches={
        "main.model.readout": create_model_spec(
            torch.nn.Linear,
            in_features=128,
            out_features=1,
        ),
    },
    freeze_patterns=("main.model.*",),
    trainable_patterns=("main.model.readout.*",),
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-3},
    ),
    training_fn=default_training_fn,
    loss_fn=EnergyMSELoss(),
    num_epochs=25,
)

strategy.run(train_loader)
```

The parent module path must already exist. The final child can replace an
existing {py:class}`torch.nn.Module` or add a new child module, but the model's
forward pass must actually use that child. If you add a brand-new auxiliary
head, update the wrapper or provide a custom `training_fn` that calls it and
returns a prediction key consumed by your loss.

```python
def train_energy_and_band_gap(model, batch):
    outputs = default_training_fn(model, batch)
    # Implement this with the feature extraction path exposed by your wrapper.
    embeddings = extract_hidden_features(model, batch)
    outputs["predicted_band_gap"] = model.model.band_gap_head(embeddings)
    return outputs

strategy = FineTuningStrategy(
    models=pretrained_model,
    module_patches={
        "main.model.band_gap_head": create_model_spec(
            torch.nn.Linear,
            in_features=128,
            out_features=1,
        ),
    },
    trainable_patterns=("main.model.band_gap_head.*",),
    training_fn=train_energy_and_band_gap,
    loss_fn=(
        EnergyMSELoss()
        + EnergyMSELoss(prediction_key="predicted_band_gap", target_key="band_gap")
    ),
    optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.AdamW),
    num_steps=1_000,
)
```

## Adding or replacing an embedding table

Embedding-table patches are useful when a target dataset introduces species or
features that the pretrained model did not represent well. A common pattern is
to replace the table, freeze the rest of the model, and train the new table
together with the final head.

```python
strategy = FineTuningStrategy(
    models=pretrained_model,
    module_patches={
        "main.model.atomic_embedding": create_model_spec(
            torch.nn.Embedding,
            num_embeddings=119,
            embedding_dim=128,
        ),
    },
    freeze_patterns=("main.model.*",),
    trainable_patterns=(
        "main.model.atomic_embedding.*",
        "main.model.readout.*",
    ),
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 5e-4},
    ),
    training_fn=default_training_fn,
    loss_fn=EnergyMSELoss() + ForceMSELoss(normalize_by_atom_count=True),
    num_epochs=10,
)
```

When you need to initialize a replacement from the old table, build the module
yourself and pass the module instance directly:

```python
old = pretrained_model.model.atomic_embedding
replacement = torch.nn.Embedding(119, old.embedding_dim)
with torch.no_grad():
    replacement.weight[: old.num_embeddings].copy_(old.weight)
    torch.nn.init.normal_(replacement.weight[old.num_embeddings :], std=0.02)

strategy = FineTuningStrategy(
    models=pretrained_model,
    module_patches={"main.model.atomic_embedding": replacement},
    trainable_patterns=("main.model.atomic_embedding.*",),
    optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.AdamW),
    training_fn=default_training_fn,
    loss_fn=EnergyMSELoss(),
    num_steps=1_000,
)
```

Direct module instances are runtime-only. They are not serializable through
`to_spec_dict()` because their construction code and copied weights are not
captured. Use {py:func}`~nvalchemi.training.create_model_spec` for declarative
patches that must round-trip through JSON.

## Choosing a freeze mode

The default `freeze_mode="requires_grad"` removes excluded parameters from
optimizers and temporarily sets `requires_grad=False` while `run` executes.
This is the usual choice for transfer learning because it saves gradient memory
and makes accidental updates easy to detect.

Use `freeze_mode="optimizer_only"` when excluded parameters should still
receive gradients, but must not be updated by optimizers. This is useful for
diagnostics, gradient-based regularizers, or custom hooks that inspect frozen
base-model gradients.

```python
strategy = FineTuningStrategy(
    models=pretrained_model,
    freeze_patterns=("main.model.*",),
    trainable_patterns=("main.model.readout.*",),
    freeze_mode="optimizer_only",
    optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.AdamW),
    training_fn=default_training_fn,
    loss_fn=EnergyMSELoss(),
    num_steps=500,
)
```

## Operational notes

- Use `FineTuningStrategy.load_checkpoint(...)` to resume an interrupted
  fine-tuning run. Use `FineTuningStrategy.from_pretrained_checkpoint(...)` to
  start a fresh fine-tuning run from checkpointed model weights. Source loss
  and optimizer config reuse requires explicit `use_original_loss` or
  `use_original_opt_class`; optimizer state, hooks, counters, and epoch/step
  limits are not inherited.
- Compiled MACE checkpoint wrappers are inference-only. Load MACE sources with
  `compile_model=False` before fine-tuning, then compile/export a separate
  inference model after training if needed.
- Generated fine-tuning hooks are registered before explicit `hooks=` so
  later hooks see the patched module tree and optimizer parameter filter.
- Registering parameter filters after optimizers have already been built emits
  a warning. Construct a fresh strategy or rebuild optimizers when changing the
  trainable set.
- Fine-tuning hooks are registration-time hooks, not per-batch update hooks.
  Use {ref}`training-update-hooks` for policies that coordinate backward,
  optimizer steps, mixed precision, or scheduler stepping.

## API reference

See {ref}`training-finetuning-api` for the API reference for
{py:class}`~nvalchemi.training.FineTuningStrategy`,
{py:class}`~nvalchemi.training.hooks.ModulePatchHook`, and
{py:class}`~nvalchemi.training.hooks.TrainableParameterHook`.
