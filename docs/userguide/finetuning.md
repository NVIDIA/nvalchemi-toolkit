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

## Training CLI

The simplest and quickest way to get started with launching training
and fine-tuning experiments is through the command-line interface (CLI) available
after installing NVIDIA ALCHEMI Toolkit, via `nvalchemi-training`. The main
features of this CLI is the ability to generate, review, and run experiments
directly from JSON configuration files - you do not need an intricate knowledge
of the full training API (although we highly suggest that you do!) in order to
get started.

### Representative workflow: fine-tuning a MACE checkpoint

To give an overview of the CLI, we can look at fine-tuning a pre-trained MACE model
on your new dataset. Under the hood, the CLI effectively makes use of the fine-tuning
and training APIs so you don't have to build a script yourself, although we recommend
power users to do so for more flexibility.

The first step in the CLI workflow is to generate a reference JSON configuration
if you don't already have one. The JSON schema is tailored specifically for the CLI,
but its contents are used to subsequently construct the same objects as you would
if you were to write a script. The `nvalchemi-training finetune` group contains the
command to initialize a configuration for a given architecture, as well as an existing
public checkpoint:

```bash
# multiple datasets can be specified together
nvalchemi-training finetune init mace small-0b \
  --dataset data/domain-a.zarr \
  --dataset data/domain-b.zarr \
  --output-dir runs/mace-ft \
  --out mace-ft.json

# get options printed out
nvalchemi-training finetune init mace --help
```

We request a MACE model starting from the `small-0b` public checkpoint, and the
expected training outputs will go into `runs/mace-ft`. The configuration file
will be written out to `mace-ft.json` in the current working directory. You
can then make edits directly to `mace-ft.json` to match your requirements.

One important feature of the CLI is the ability to provide direct feedback
and validate your configuration *before* you allocate/launch the compute;
this is particularly handy so you do not need to wait for your GPU job
to queue, only to find out that you have a mistake in your dataset path or
something minor:

```bash
nvalchemi-training spec report mace-ft.json
```

This will create a terminal-based report that lets you review your intentions:
everything from batch size, dataset choice, and learning rate schedule, and for
supported models, specific hyperparameters like the `E0` values for MACE. Some
`nvalchemi` specific diagnostics are also included, such as what hooks are configured
and when they are expected to fire, and in the case of fine-tuning, which parameters
are expected to actually be updated via the `trainable_patterns` regular expressions.
Users should also pay close attention to the "Warnings" section of the report, which
will provide important heuristics for catching common mistakes.

```{tip}
Run `nvalchemi-training spec report <config>.json --json` to have the result dumped
to a JSON file, as opposed to just being in the terminal. This can be helpful for
bookkeeping, or for use with agents.
```

The base configuration will be missing some elements like hooks, which modify the
runtime behavior. An essential one for a graph-based model like MACE is the neighbor
list, which can be configured below:

```json
{
  "source": {
    "hooks": [
      {
        "spec": {
          "cls_path": "nvalchemi.hooks.neighbor_list.NeighborListHook",
          "config": {
            "cutoff": 6.0,
            "format": "coo",
            "half_list": false,
            "skin": 0.0,
          },
          "skin": 0.0
        },
        "stages": ["BEFORE_FORWARD"]
      }
    ]
  }
}
```

The configuration specifies a COO neighbor list with a cutoff radius of 6.0, and the
hook will fire at the `TrainingStage.BEFORE_FORWARD` stage. Other hooks can be
arbitrarily specified in the same way. Other useful hooks include {py:class}`~nvalchemi.training.CheckpointHook`,
and {py:class}`~nvalchemi.hooks.ReportingOrchestrator` - the former will create
regular training checkpoints that we can resume from (more on that later),
and the latter will provide metric logging utilities.

:::{admonition} Checkpoint and tensorboard configuration
:class: hint

The configuration can be copy-pasted into a separate JSON config file.
If you have `jq` installed, you can merge multiple JSON files together
using `jq -s 'add' file1.json file2.json > combined.json`!

```json
{
  "source": {
    "hooks": [
      {
        "spec": {
          "cls_path": "nvalchemi.hooks.CheckpointHook",
          "checkpoint_dir": "training-output/checkpoints",
          "step_interval": 1000
        }
      },
      {
        "spec": {
          "cls_path": "nvalchemi.hooks.ReportingOrchestrator",
          "reporters": [
            {
              "cls_path": "nvalchemi.hooks.TensorBoardReporter",
              "log_dir": "training-outputs/tensorboard",
              "include_losses": true,
              "include_optimizer_lrs": true,
              "tag_prefix": "train",
              "flush": true
            }
          ],
          "frequency": 10,
        },
        "stages": ["AFTER_OPTIMIZER_STEP"]
      }
    ]
  }
}
```

:::

Other settings you should consider modifying are the batch size and the number of steps.

Once your configuration is satisfactory, you can execute the training/fine-tuning:

```bash
nvalchemi-training spec run mace-ft.json
```

```{tip}
Distributed runs can simply be wrapped with `torchrun`, i.e.
`torchrun --nproc_per_node=4 -m nvalchemi.training.cli spec run ...`
```

For whatever reason, if your fine-tuning run was interrupted, you can easily
continue from the same session:

```bash
nvalchemi-training spec resume training-outputs/checkpoints \
  --spec mace-ft.json \
  --checkpoint_index 5
```

This will resume training at an arbitrary checkpoint index (in this case, the *6th* checkpoint
since we zero index).

Once you're done with your fine-tuning, you can access the model within Python simply
by using the {py:func}`~nvalchemi.training.load_checkpoint` method:

```python
from nvalchemi.training import load_checkpoint

checkpoint_data = load_checkpoint(
  "training-output/checkpoints",
  checkpoint_index=-1,  # load the last checkpoint
  map_location="cuda",  # or CPU, depending on your use case
)
# the hierarchy corresponds to: access the 'main' model within the
# checkpoint, and the 'model' key within 'main' yields the instance
# of MACEWrapper
model = checkpoint_data["models"]["main"]["model"]
model.eval()
```

The loaded model will then be usable like any other {py:class}`~nvalchemi.models.mace.MACEWrapper`;
you will be able to run batched dynamics, etc. to evaluate the behavior of your model.

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
- Generated fine-tuning hooks are registered before explicit `hooks=` so
  later hooks see the patched module tree and optimizer parameter filter.
- Registering parameter filters after optimizers have already been built emits
  a warning. Construct a fresh strategy or rebuild optimizers when changing the
  trainable set.
- Fine-tuning hooks are registration-time hooks, not per-batch update hooks.
  Use {ref}`training-update-hooks` for policies that coordinate backward,
  optimizer steps, mixed precision, or scheduler stepping.

## Notes on fine-tuning models

### MACE

Load pretrained models in a trainable form before passing them to
`FineTuningStrategy`. For MACE checkpoints,
{py:meth}`nvalchemi.models.mace.MACEWrapper.from_checkpoint` returns an
eval-mode wrapper, and the training strategy temporarily switches configured
models into train mode during `run`. However, `compile_model=True` is
inference-only: it freezes parameters before `torch.compile`. Use
`compile_model=False` for fine-tuning.

## API reference

See {ref}`training-finetuning-api` for the API reference for
{py:class}`~nvalchemi.training.FineTuningStrategy`,
{py:class}`~nvalchemi.training.hooks.ModulePatchHook`, and
{py:class}`~nvalchemi.training.hooks.TrainableParameterHook`.
