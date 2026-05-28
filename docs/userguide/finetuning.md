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

## Simple full-model fine-tuning

The simplest workflow is to load a pretrained model and continue training all
of its parameters on your dataset. This is often a useful baseline, but it is
also the most likely workflow to cause catastrophic forgetting: the model may
adapt to the new domain while losing accuracy on the broader distribution it
was pretrained on.

```python
import torch

from nvalchemi.training import (
    EnergyLoss,
    FineTuningStrategy,
    ForceLoss,
    OptimizerConfig,
    default_training_fn,
)

pretrained_model = load_my_pretrained_model()
train_loader = make_my_batch_loader()

strategy = FineTuningStrategy(
    models=pretrained_model,
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 1e-5},
    ),
    training_fn=default_training_fn,
    loss_fn=EnergyLoss() + ForceLoss(normalize_by_atom_count=True),
    num_epochs=5,
    devices=[torch.device("cuda")],
)

strategy.run(train_loader)
```

`FineTuningStrategy` uses the same forward and loss contracts as
`TrainingStrategy`. The default single-model training function calls
`model(batch)` and prefixes outputs with `"predicted_"` so the built-in
losses can consume keys such as `"predicted_energy"` and
`"predicted_forces"`.

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
    loss_fn=EnergyLoss(),
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
    loss_fn=EnergyLoss(),
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
    loss_fn=EnergyLoss(),
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
        EnergyLoss()
        + EnergyLoss(prediction_key="predicted_band_gap", target_key="band_gap")
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
    loss_fn=EnergyLoss() + ForceLoss(normalize_by_atom_count=True),
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
    loss_fn=EnergyLoss(),
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
    loss_fn=EnergyLoss(),
    num_steps=500,
)
```

## Operational notes

- `source_checkpoint` is reserved for a future checkpoint-loading workflow.
  For now, load the pretrained model yourself and pass it via `models=`.
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
