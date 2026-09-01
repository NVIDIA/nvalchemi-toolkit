(finetuning_guide)=

# Fine-Tuning Pretrained Models

Fine-tuning is the art of adapting a pretrained model to a new dataset
and/or domain, typically with access to less data and/or computational
resources than the original training run. The NVIDIA ALCHEMI Toolkit
provides a fine-tuning API that closely resembles regular training,
and provides a few functions that make the fine-tuning experience more
turn-key for non-experts and experts alike to help make productionizing
the model products more seamless.

```{tip}
**AI coding assistant?** Load the ``nvalchemi-fine-tuning``
{ref}`agent skill <agent_skills>` for concise instructions on configuring
fine-tuning workflows and adapting pretrained checkpoints.
```

This documentation is roughly broken up into two parts: an introduction
to the command-line interface which is designed for a more "on-the-rails"
experience to fine-tuning and training, and the core fine-tuning API
details for developers and ML engineers who need a better understanding
of the internals.

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
            "skin": 0.0
          }
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
          "frequency": 10
        },
        "stages": ["AFTER_OPTIMIZER_STEP"]
      }
    ]
  }
}
```

:::

Other settings you should consider modifying are the batch size and the number of steps.

The `--loss-dtype-policy` flag controls how the loss function handles dtype
mismatches between predictions and targets. Accepted values are `strict`,
`prediction_to_target`, and `target_to_prediction`. The value is stored in
`strategy.loss_fn_spec.dtype_policy`, reflected in `spec report`, and applied
at `spec run` time.

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

## Fine-tuning API

In this section, we go into detail about the core fine-tuning API within
`nvalchemi-toolki`, providing sufficient detail for users and developers to
build with and on top of the fine-tuning specific components.

{py:class}`~nvalchemi.training.FineTuningStrategy` is the Python entry point
for the fine-tuning abstraction: it inherits from {py:class}`~nvalchemi.training.TrainingStrategy`,
and therefore re-uses many of the same systems, configurations, etc. but
specializes it for fine-tuning workflows by being more opinionated on
some default values such as learning rate, and adding more API entry points
like the ability to add and modify existing layers, etc. For the general
training topics we refer the reader to {ref}`training_guide`.

### Simple full-model fine-tuning

The most straightforward entry point is to load a pretrained model and continue
training all of its parameters on your new dataset. Every weight is free to
adapt, which gives the model maximum flexibility — but it is also the most
likely workflow to cause *catastrophic forgetting*, where the model drifts
toward the new domain while losing accuracy on the distribution it was
pretrained on. A small learning rate and early stopping on a held-out
validation set can go a long way toward mitigating this.

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

The loss and training function conventions here are the same as in regular
training — see {ref}`losses_guide` and {ref}`training_guide` for details on
operator-composed losses, `dtype_policy`, and how `default_training_fn` maps
output keys. For regular fine-tuning, leave `peft_config=None`. Parameter-efficient
fine-tuning will be discussed later in the LoRA section.

```{tip}
If your pretrained model comes from an existing `nvalchemi-toolkit` checkpoint,
you can omit `loss_fn` entirely and pass `use_original_loss=True` to
{py:meth}`~nvalchemi.training.FineTuningStrategy.from_pretrained_checkpoint`
instead. This reuses the loss function that was serialized into the checkpoint,
saving you from having to reconstruct it manually. See
{ref}`checkpoint-workflows` for details.
```

```{warning}
Full-model fine-tuning updates every optimizer-visible parameter. Use a
small learning rate, early stopping, validation on the original domain, or a
frozen-base workflow when preserving pretrained behavior matters.
```

### Fine-tuning modifications

The additions specific to fine-tuning (over regular training) are three arguments
to {py:class}`~nvalchemi.training.FineTuningStrategy` that apply modifications to
the architecture and weight update methodology before the optimizer is constructed:

- **`module_patches`** : swap or graft `nn.Module` children in the model tree,
  so the optimizer sees the updated architecture from the start.
- **`trainable_patterns` / `freeze_patterns`** : glob-based allow and deny
  lists that control which parameters enter the optimizer. Every unmatched
  parameter is excluded from optimization and, by default, has
  `requires_grad` set to `False` for the duration of the run.
- **`freeze_mode`** : whether "excluded" means removed from the optimizer only,
  or also `requires_grad=False`.

The following sections cover each mechanism in turn, starting with how to
discover the parameter names that all three rely on.

### Inspecting names for patches and filters

All three modification mechanisms reference model components by their
fully-qualified names, so the first step before using any of them is to know
what those names are and ensure that the user-specified patterns will catch them.
The `FineTuningStrategy` validates every pattern at startup and
raises a `ValueError` if any of them match zero parameters — catching typos
and model-version drift before a run begins.

All fine-tuning fields use names prefixed with the model key. When you pass a
single model as `models=pretrained_model`, the strategy stores it under the key
`"main"` (or if you provide a dictionary of models, their corresponding key),
so every module and parameter name gains a prefix.
Everything after that prefix is determined by the model's own module hierarchy
— it depends entirely on how the wrapper and its children are laid out. The
only reliable way to discover the correct names is to print them before writing
your configuration:

```python
# note this is outside of the strategy wrapping
for name, module in pretrained_model.named_modules():
    # 'main.' prefix depends on the key and is the default;
    # if you pass a dict of models, substitute 'main' with the
    # corresponding key
    print(f"main.{name}", type(module).__name__)

for name, parameter in pretrained_model.named_parameters():
    print(f"main.{name}", tuple(parameter.shape))
```

Reading the output tells you exactly what names and types are available, so you
can write patterns against what the model actually exposes rather than guessing
at the hierarchy.

### Freezing the base model

With the parameter names in hand, the most common modification is to restrict
which of them are trained. `trainable_patterns` acts as an explicit allow-list:
only parameters whose fully-qualified names match at least one glob pattern are
passed to the optimizer. All others are temporarily marked `requires_grad=False`
for the duration of `run`. Freezing the pretrained body and updating only the
output head or a narrow set of domain-specific layers tends to converge faster,
require less data, and cause less catastrophic forgetting than full-model
fine-tuning.

```python
strategy = FineTuningStrategy(
    models=pretrained_model,
    # only allow readout layers to be updated
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

Only the readout layer's parameters enter the optimizer; the rest of the
pretrained body is frozen.

When the intent is easier to express as "freeze this broad region, then
carve out exceptions", combine `freeze_patterns` and `trainable_patterns`:
`freeze_patterns` excludes a broad set, and `trainable_patterns` re-admits a
subset of those exclusions.

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

Every pattern — in either field — must match at least one parameter. This
safety check catches typos and model-version drift before a long run begins.

Because patterns are plain strings, they can be generated programmatically
from `named_parameters()` output — for example, to implement progressive
unfreezing by running `FineTuningStrategy` in stages, each starting from the
previous stage's checkpoint via `from_pretrained_checkpoint` and widening
`trainable_patterns` to include the next layer group.

### Choosing a freeze mode

The `freeze_mode` argument controls what "frozen" means at the PyTorch level. The two
options trade memory efficiency against gradient visibility.

The default `freeze_mode="requires_grad"` sets `requires_grad=False` on
frozen parameters for the duration of `run` and removes them from the
optimizer. PyTorch does not allocate gradient buffers for them, which reduces
peak memory. This is the right choice for almost all transfer learning
workflows.

`freeze_mode="optimizer_only"` keeps `requires_grad=True`, so gradients are
still computed and held in memory, but the optimizer never updates frozen
parameters. This is useful when a hook or regularizer needs access to the
gradient of a frozen layer — for example, to monitor how much the frozen base
is being "pulled" by the new data as a domain-mismatch signal.

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

Because `optimizer_only` keeps gradient buffers allocated for frozen
parameters, it uses more memory than the default. Prefer `"requires_grad"`
unless you have a specific reason to inspect those gradients.

`freeze_mode="optimizer_only"` is specifically designed as an extension seam
for hooks that need gradient information from the frozen base. A hook firing at
`AFTER_BACKWARD` can read `.grad` directly from any frozen parameter — useful
for gradient-based regularizers such as elastic weight consolidation, or for
diagnostic logging that tracks how strongly the frozen base is being "pulled"
by the new domain. See {ref}`training-hooks` for how to write and register
such a hook.

### Adding or replacing an output head

Parameter filtering controls which weights get updated, but sometimes the
pretrained output head itself is the wrong shape for the new task — for
example, the source model predicts energy per atom and the target task adds a
band gap property. `module_patches` lets you swap or graft `nn.Module` children
before the optimizer is built, so the rest of the configuration sees the
updated model tree as if it were always there.

Each entry in `module_patches` maps a fully-qualified child path to a new
module. Use {py:func}`~nvalchemi.training.create_model_spec` to describe the
replacement declaratively — this keeps the patch serializable through
`to_spec_dict()` and round-trippable via JSON. The first argument can be any
`nn.Module` subclass, including ones defined in your own codebase; the spec
stores the fully-qualified class path and constructor arguments, so custom
architectures serialize exactly like built-in ones.

Patched modules are trainable by default, so you do not need to repeat their
parameter names in `trainable_patterns`. Use `trainable_patterns` only when you
also want to opt in additional, non-patched parameters from the pretrained base.

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

The patch replaces `main.model.readout` with a fresh `Linear` layer while
leaving the rest of the body frozen. Because the replacement is randomly
initialized, the learning rate can be higher than for a full-model run.

An important constraint to this approach: while replacing an existing layer
doesn't require further modification, if layers are *added* then the training
function must be modified from the default, so that the workflow knows to
actually use the new layer. To do so, pass a custom `training_fn` — a callable
with the same signature as {py:func}`~nvalchemi.training.default_training_fn`
— that calls the new head explicitly and returns a prediction key your loss can
consume:

```python
def train_energy_and_band_gap(model, batch: Batch) -> dict[str, torch.Tensor]:
    embeddings = model.compute_embeddings(batch)
    outputs = {}
    # compute the outputs manually
    outputs["predicted_band_gap"] = model.model.band_gap_head(embeddings)
    outputs["predicted_energy"] = model.model.readout(embeddings)
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
    # keep the pretrained base frozen; the patched head is trainable by default
    freeze_patterns=("main.model.*",),
    training_fn=train_energy_and_band_gap,
    # add an MSE loss based on the band gap
    loss_fn=(
        EnergyMSELoss()
        + YourMSELoss(prediction_key="predicted_band_gap", target_key="band_gap")
    ),
    optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.AdamW),
    num_steps=1_000,
)
```

In this example, the new `train_energy_and_band_gap` method replaces the regular
`default_training_fn`, where we route the embeddings manually to the band gap
and readout heads to obtain the energy and band gap values. The loss function
is then composed of the regular {py:class}`~nvalchemi.training.losses.terms.EnergyMSELoss`
and a custom (fictitious) `YourMSELoss` to compute against the band gap and provide
the key/value mapping out of the returned predictions dictionary from the training
function. The pretrained base is frozen by `freeze_patterns`, while the new
band gap head remains trainable because patched modules are trainable by
default.

### Adding or replacing an embedding table

A related scenario arises when the target dataset contains atomic species that
the pretrained model did not see during pretraining, or saw only rarely.
Replacing the embedding table — while keeping the pretrained message-passing
and readout layers frozen — adapts the model's input representation without
discarding the learned body.

The declarative approach via `create_model_spec` works the same way as for
output heads:

```python
strategy = FineTuningStrategy(
    models=pretrained_model,
    module_patches={
        "main.model.atomic_embedding": create_model_spec(
            torch.nn.Embedding,
            num_embeddings=100,
            # if you do not want to modify the remainder
            # of the model, keep this dimensionality the same
            embedding_dim=128,
        ),
    },
    freeze_patterns=("main.model.*",),
    trainable_patterns=("main.model.readout.*",),
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 5e-4},
    ),
    training_fn=default_training_fn,
    loss_fn=EnergyMSELoss() + ForceMSELoss(normalize_by_atom_count=True),
    num_epochs=10,
)
```

`create_model_spec` initializes the replacement table from scratch. When you
want to warm-start the new table from the existing weights — copying rows for
species the model already knew and randomly initializing the rest — build the
replacement yourself and pass the live module instance directly:

```python
old = pretrained_model.model.atomic_embedding
replacement = torch.nn.Embedding(100, old.embedding_dim)
with torch.no_grad():
    replacement.weight[: old.num_embeddings].copy_(old.weight)
    torch.nn.init.normal_(replacement.weight[old.num_embeddings :], std=0.02)

strategy = FineTuningStrategy(
    models=pretrained_model,
    module_patches={"main.model.atomic_embedding": replacement},
    freeze_patterns=("main.model.*",),
    optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.AdamW),
    training_fn=default_training_fn,
    loss_fn=EnergyMSELoss(),
    num_steps=1_000,
)
```

Passing a live module instance is runtime-only — it cannot be serialized
through `to_spec_dict()` because the construction code and copied weights are
not captured in the spec and so this is not the recommended approach, however
is possible. As described earlier, use `create_model_spec` for patches so
they can actually be JSON round-trippable.

### Multi-model fine-tuning

All of the examples above pass a single model to `FineTuningStrategy`, which
stores it internally under the key `"main"`. You can also pass a dictionary of
models when your workflow involves more than one — for example, a student and a
frozen teacher used as a reference:

```python
strategy = FineTuningStrategy(
    models={"student": student_model, "teacher": teacher_model},
    trainable_patterns=("student.model.readout.*",),
    optimizer_configs={"student": OptimizerConfig(optimizer_cls=torch.optim.AdamW)},
    training_fn=my_distillation_fn,
    loss_fn=my_distillation_loss,
    num_steps=2_000,
)
```

There are two sharp edges to be aware of.

**Pattern and patch names must use your model keys, not `"main"`.** Every
`trainable_patterns` glob, `freeze_patterns` glob, and `module_patches` key
must be prefixed with the corresponding dict key. In the example above, patterns
target `"student.*"` — writing `"main.*"` would raise a `ValueError` at startup
because no model is stored under `"main"` in a multi-model strategy.

**Partial pattern coverage is not validated.** The strategy only checks that
each pattern matches at least one parameter somewhere across all models. It does
not warn if your patterns leave an entire model's parameters untouched. In the
example above, `"teacher.*"` has no optimizer config and no trainable patterns,
which is intentional — the teacher is used as a frozen reference. But if you
accidentally wrote `"student.model.readuot.*"` (typo) instead, the teacher's
parameters would be the only match, and the student would train nothing without
any error. Print the matched parameter names after constructing the strategy to
verify coverage before committing to a long run.

**Differential learning rates across models.** When you provide `optimizer_configs`
as a dict keyed by model name, each participating model gets its own optimizer
and learning rate. Only include entries for models you actually want to update
— the teacher in a distillation setup needs no optimizer config at all, since
it is never updated. This is also the right pattern for two-stage
teacher-student workflows where the student's backbone and head are trained at
different rates: split the student's parameters across two optimizer groups using
the {ref}`training_guide` parameter-group API, or run separate strategies in
sequence via `from_pretrained_checkpoint`.

### LoRA fine-tuning

{py:class}`~nvalchemi.training.FineTuningStrategy` also supports
parameter-efficient fine-tuning through `peft_config`; LoRA is the only
supported PEFT method right now. Pass
{py:class}`~nvalchemi.training.peft.lora.LoRAConfig` to inject low-rank LoRA adapters
into selected modules before optimizer construction. The pretrained base is
frozen by default, and the optimizer sees only adapter parameters plus any
module patches or explicit extra parameters you opt in with
`trainable_patterns`.

This is useful when the pretrained model is large, the target dataset is
relatively small, or you want to keep the base model reusable while shipping a
small domain-specific adapter. Full-model fine-tuning gives the optimizer the
most freedom; LoRA trades some of that flexibility for lower memory use, fewer
trainable parameters, and smaller checkpoint or adapter artifacts.

```python
import torch

from nvalchemi.hooks import NeighborListHook
from nvalchemi.training import (
    EnergyMSELoss,
    FineTuningStrategy,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStage,
    ValidationConfig,
    default_training_fn,
)

from nvalchemi.training.peft.lora import LoRAConfig

pretrained_model = load_my_pretrained_model()
train_loader = make_my_batch_loader()
validation_loader = make_my_validation_loader()

loss_fn = EnergyMSELoss() + ForceMSELoss(normalize_by_atom_count=True)
loss_fn.dtype_policy = "prediction_to_target"

strategy = FineTuningStrategy(
    models=pretrained_model,
    optimizer_configs=OptimizerConfig(
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs={"lr": 3e-4, "weight_decay": 1e-6},
    ),
    training_fn=default_training_fn,
    loss_fn=loss_fn,
    num_steps=2_000,
    devices=[torch.device("cuda")],
    hooks=[
        NeighborListHook(
            pretrained_model.model_config.neighbor_config,
            stage=TrainingStage.BEFORE_FORWARD,
        ),
    ],
    validation_config=ValidationConfig(
        validation_data=validation_loader,
        validation_fn=default_training_fn,
        loss_fn=loss_fn,
        every_n_steps=250,
    ),
    peft_config=LoRAConfig(
        lora_target_patterns=("main.model.interactions.*.linear*",),
        rank=8,
        alpha=1.0,
        lora_dropout=0.0,
    ),
)

strategy.run(train_loader)
```

LoRA target patterns match module names, not parameter names. They use the same
shell-style glob syntax as `trainable_patterns`, but they are matched against
`named_modules()` output with the model key prepended. A single model passed as
`models=pretrained_model` is stored under `"main"`, so target names start with
`main.`. For multi-model workflows, use the keys from your model dictionary.

```python
for name, module in pretrained_model.named_modules():
    print(f"main.{name}", type(module).__name__)
```

Use the printed names to choose targets that the current model actually
contains. Exact names such as `"main.model.projection"` are accepted, and globs
such as `"main.model.*projection"` can select related modules. Patterns that
match no modules raise an error before training begins.

#### How LoRA is applied

LoRA injection happens before optimizer construction. When `peft_config` is
provided, `FineTuningStrategy` prepends PEFT registration hooks before user
hooks. When `compute_base_fingerprints=True`, the default, this starts with
{py:class}`~nvalchemi.training.hooks.BaseFingerprintHook`, which records
base-model fingerprints before adapters mutate the model. For LoRA, PEFT setup
then includes
{py:class}`~nvalchemi.training.peft.lora_hook.LoRAHook`, which registers built-in and
custom LoRA wrappers, then applies adapters to modules whose model-prefixed
names match the target patterns. For each match, the LoRA registry selects the
wrapper registered for that layer class.

Each wrapper replaces the base layer with a module containing the frozen original
layer and trainable low-rank adapter parameters. The net adapter weights start at zero,
so the wrapped model initially matches the base model. Training optimizes only
LoRA parameters, module patches, and explicit `trainable_patterns`. Mergeable
wrappers can later fold the learned update back into the base layer for inference
or export.

The public wrapper classes are available from `nvalchemi.training.peft.lora`:

- {py:class}`nvalchemi.training.peft.lora.LoRALinear` for `torch.nn.Linear`
  layers.
- `nvalchemi.training.peft.lora.TransformerEngineLoRALinear` for
  `transformer_engine.pytorch.Linear` layers when Transformer Engine is
  installed.
- {py:class}`nvalchemi.training.peft.lora.EquivariantLoRALinear` for
  `e3nn.o3.Linear` layers.
- {py:class}`nvalchemi.training.peft.lora.CuEquivarianceLoRALinear` for
  `cuequivariance_torch.operations.linear.Linear` layers.
- {py:class}`nvalchemi.training.peft.lora.E3NNFullyConnectedLoRALayer` for
  `e3nn.nn._fc._Layer` scalar fully connected layers.

Inspect the active layer-to-wrapper pairs with
{py:func}`~nvalchemi.training.peft.lora.available_lora_wrappers`.

```python
from nvalchemi.training.peft.lora import available_lora_wrappers

for layer_cls, wrapper_cls in available_lora_wrappers():
    print(layer_cls, "->", wrapper_cls)
```

To support a custom layer type, write a wrapper module that implements the
{py:class}`nvalchemi.training.peft.lora.LoRALayer` interface, then register that
wrapper for the base layer class through
`LoRAConfig(wrapper_registrations=...)`. Each registration is a pair of
`(LoRAWrappableLayer, LoRAWrapper)`: when LoRA finds an instance of the base class
matching your target patterns, it replaces that layer with your wrapper. The
strategy register these pairs immediately before adapter injection, so target
patterns can select the custom layer in the same constructor call. Registrations
are temporary: the process-wide registry is restored after injection.

A custom wrapper may replace an existing or built-in wrapper for the duration
of injection, with a `UserWarning`. Attempting to assign two different wrappers
to the same layer class in `LoRAConfig` will raise `ValueError` during
configuration validation.

```python
import torch

from nvalchemi.training import FineTuningStrategy, OptimizerConfig
from nvalchemi.training.peft.lora import LoRAConfig, LoRALayer, LoRAWrapperRegistrations


class MyBlock(torch.nn.Module):
    ...


class MyBlockLoRA(torch.nn.Module, LoRALayer):
    mergeable = True

    def __init__(
        self,
        base_layer: MyBlock,
        rank: int,
        alpha: float = 1.0,
        dropout: float = 0.0,
        init: str = "default",
        **kwargs,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        self.scaling = alpha / rank
        self.enabled = True
        # Define trainable LoRA parameters or submodules here. Initialize the
        # LoRA residual to zero.

    def forward(self, *args, **kwargs):
        # Return base_layer(...) plus the LoRA residual when enabled.
        ...

    def merge_into_base(self) -> None:
        # Fold the LoRA delta into base_layer for merged inference.
        ...


custom_lora_wrappers: LoRAWrapperRegistrations = (
    (MyBlock, MyBlockLoRA),
)

strategy = FineTuningStrategy(
    models=pretrained_model,
    peft_config=LoRAConfig(
        lora_target_patterns=("main.model.custom_blocks.*",),
        wrapper_registrations=custom_lora_wrappers,
    ),
    optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.AdamW),
    training_fn=default_training_fn,
    loss_fn=EnergyMSELoss(),
    num_steps=1_000,
)
```

If the strategy must round-trip through `to_spec_dict()` or restart from a
serialized checkpoint recipe, define custom layer and wrapper classes in
importable modules rather than in a notebook cell or local function. The recipe
stores class paths for `wrapper_registrations`, so those paths must be
importable when the strategy is rebuilt.

When `peft_config` is provided, LoRA freezes the pretrained base by default, so
`freeze_patterns` must stay empty. LoRA adapters and `module_patches` are
trainable automatically. As in regular fine-tuning, use `trainable_patterns`
to select any additional existing model parameters you also want to update.

For example, this strategy trains the LoRA adapters, the patched auxiliary
projection, and one explicitly selected base projection.

```python
from nvalchemi.training import FineTuningStrategy, create_model_spec
from nvalchemi.training.peft.lora import LoRAConfig

strategy = FineTuningStrategy(
    models=pretrained_model,
    module_patches={
        "main.model.aux_projection": create_model_spec(
            torch.nn.Linear,
            in_features=128,
            out_features=1,
        ),
    },
    trainable_patterns=("main.model.projection.*",),
    peft_config=LoRAConfig(
        lora_target_patterns=("main.model.interactions.*.linear*",),
    ),
    optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.AdamW),
    training_fn=default_training_fn,
    loss_fn=EnergyMSELoss(),
    num_steps=1_000,
)
```

The complete intermediate example
`examples/intermediate/08_lora_finetuning.py` shows a MACE workflow that
downloads the LPSC dataset, fits reference energies, inspects LoRA target
candidates, trains adapters, writes a trainable-state PEFT checkpoint, and
loads the PEFT checkpoint back into a fresh base model.

### Adding a PEFT method

A PEFT provider must supply all five arguments required by
{py:func}`~nvalchemi.training.peft.register_peft_method`. Together, these inputs
are the complete registry contract for a method:

- `method`: a unique, stable string stored in serialized strategy and checkpoint
  metadata.
- `config_cls`: a `PeftConfig` subclass whose frozen `peft_method` literal matches
  `method`. Its `to_spec_dict()` result must be JSON-safe; add Pydantic serializers
  or validators for class objects and other Python-only fields.
- `build_peft_setup_hooks`: a callable that accepts the validated config and
  strategy mapping, checks model-dependent restrictions, and returns hooks that
  install the PEFT structure before optimizer construction.
- `apply_peft_from_checkpoint_metadata`: a callable that accepts the base model,
  serialized strategy metadata, model name, and optional import-path validator.
  It must rebuild the saved config and recreate the PEFT module structure before
  weights are loaded.
- `merge_peft`: a callable that merges learned PEFT parameters into the model and
  honors `strict` when the structure cannot be merged completely.

The registration has this shape:

```python
from nvalchemi.training.peft import register_peft_method

register_peft_method(
    MY_PEFT_METHOD,
    config_cls=MyPeftConfig,
    build_peft_setup_hooks=build_my_peft_setup_hooks,
    apply_peft_from_checkpoint_metadata=apply_my_peft_from_checkpoint_metadata,
    merge_peft=merge_my_peft,
)
```

A minimal discriminator field looks like this:

```python
from typing import Literal

from pydantic import Field

from nvalchemi.training.peft import PeftConfig

MY_PEFT_METHOD = "my_peft"


class MyPeftConfig(PeftConfig):
    peft_method: Literal["my_peft"] = Field(
        default=MY_PEFT_METHOD,
        frozen=True,
    )
```

Use `nvalchemi/training/peft/lora.py` as a reference implementation and
`test/training/test_peft_lora.py` as its test example.

Finally, add the method registration callback to `_register_builtin_peft_methods()`
in `nvalchemi/training/peft/registry.py`.

(checkpoint-workflows)=

## Checkpoint workflows

Fine-tuning has two distinct checkpoint situations that call for different
APIs: resuming an interrupted run versus starting a fresh experiment from
prior model weights. Using the wrong one can silently discard optimizer state
or inherit unwanted settings from the source run.

| Goal | API | Restores optimizer/scheduler/counters? |
| --- | --- | --- |
| Resume an interrupted fine-tuning run | `FineTuningStrategy.load_checkpoint(...)` | Yes |
| Start a new fine-tuning run from prior model weights | `FineTuningStrategy.from_pretrained_checkpoint(...)` | No |
| Fine-tune a model you loaded yourself | `FineTuningStrategy(models=...)` | No |

### Resuming an interrupted run

`load_checkpoint` is for when a job was killed or preempted and you want to
continue exactly where it stopped. It restores the complete saved strategy
state — model weights, optimizer state, scheduler state, runtime counters,
checkpointable hook state, and the serialized fine-tuning configuration — so
the resumed run is indistinguishable from one that never stopped.

```python
resumed = FineTuningStrategy.load_checkpoint(
    "runs/domain-ft/checkpoints",
    training_fn=default_training_fn,
)
resumed.run(train_loader)
```

The path is the checkpoint directory written by
{py:class}`~nvalchemi.training.hooks.CheckpointHook`. The most recent
checkpoint is selected automatically; pass `checkpoint_index` to pin a
specific one.

```{tip}
This approach is what is used by the `resume` function in the training
CLI.
```

### Branching a new run from existing weights

`from_pretrained_checkpoint` is for starting a fresh experiment whose model
weights are initialized from a prior checkpoint — for example, branching from
a general-purpose pretrained model to fine-tune on a new domain, or iterating
on learning rate without retraining from scratch. It loads the checkpoint
model set for initialization, then builds entirely new optimizers, schedulers,
counters, losses, module patches, and parameter filters from the arguments you
supply. Nothing from the source run's optimizer state, epoch limits, hooks, or
validation settings carry over by default.

```python
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
strategy.run(train_loader)
```

Two convenience flags let you reuse parts of the source run explicitly.
`use_original_loss=True` copies the checkpointed loss when `loss_fn` is
not provided, which means that the fine-tuning run will use the same
method for computing the loss as the pretrained model, which is a particularly
desirable approach for continuity.

Another flag, `use_original_opt_class=True` copies the optimizer and scheduler
class when `optimizer_configs` is omitted which similarly means that users
do not need to explicitly set the same optimizer class manually. For the
purposes of fine-tuning, however, we do set the lower learning rate
of `1e-5` — pass `optimizer_lr=None` to suppress this and use the source
learning rate unchanged.

```python
strategy = FineTuningStrategy.from_pretrained_checkpoint(
    "runs/pretrain/checkpoints",
    # use the same loss function and optimizer class as the
    # original pretraining recipe
    use_original_loss=True,
    use_original_opt_class=True,
    # override the reused optimizer LR with a conservative 1e-5;
    # pass optimizer_lr=None instead to keep the checkpoint's LR
    optimizer_lr=1e-5,
    training_fn=default_training_fn,
    trainable_patterns=("main.model.readout.*",),
    num_steps=2_000,
)
```

`from_pretrained_checkpoint` is also the natural building block for
**progressive unfreezing**: after each stage completes and writes a checkpoint,
the next stage calls `from_pretrained_checkpoint` on that checkpoint with a
broader set of `trainable_patterns`, gradually opening up more of the model
without ever managing weights manually between stages.

### LoRA checkpoints and PEFT loading

LoRA workflows use the same `FineTuningStrategy` checkpoint APIs as regular
fine-tuning. {py:meth}`~nvalchemi.training.FineTuningStrategy.from_pretrained_checkpoint`
starts a fresh LoRA run from saved model weights when the new strategy includes
`peft_config=LoRAConfig(...)`. {py:meth}`~nvalchemi.training.FineTuningStrategy.load_checkpoint`,
{py:meth}`~nvalchemi.training.FineTuningStrategy.restore_checkpoint`,
{py:meth}`~nvalchemi.training.FineTuningStrategy.save_checkpoint`, and
{py:class}`~nvalchemi.training.hooks.CheckpointHook` handle restartable runs.
For parameter-efficient saving, especially with large base models or when
training only a small fraction of parameters, use
`save_trainable_state_only=True` in `save_checkpoint` or `CheckpointHook`.

```python
from nvalchemi.training import CheckpointHook, FineTuningStrategy
from nvalchemi.training.peft.lora import LoRAConfig

strategy = FineTuningStrategy(
    models=pretrained_model,
    peft_config=LoRAConfig(
        lora_target_patterns=("main.model.interactions.*.linear*",),
    ),
    hooks=[
        CheckpointHook(
            "runs/lora-ft/checkpoints",
            step_interval=500,
            save_trainable_state_only=True,
        ),
    ],
    ...
)
```

Pass `save_trainable_state_only=False` only when the checkpoint must carry
the full model state instead of relying on the original base model being
available. Trainable-only checkpoints are still training checkpoints; they are
not meant to be standalone deployable model artifacts unless they are loaded
with a compatible base model.

Load a trainable-state PEFT checkpoint into a compatible pristine base model
with {py:func}`~nvalchemi.training.peft.load_peft_checkpoint_into_model`. By
default, `load_peft_checkpoint_into_model(..., merge=True)` folds mergeable
adapter weights into the base layers, which is usually the right shape for
inference and export.

```python
from nvalchemi.training.peft import load_peft_checkpoint_into_model

checkpoint_index = strategy.save_checkpoint(
    "runs/lora-ft/checkpoints",
    save_trainable_state_only=True,
)

base_model = load_my_pretrained_model()
finetuned_model = load_peft_checkpoint_into_model(
    base_model,
    "runs/lora-ft/checkpoints",
    checkpoint_index=checkpoint_index,
    model_name="main",
    merge=True,
)
finetuned_model.eval()
```

LoRA checkpoints record base-model fingerprints when
`compute_base_fingerprints=True`, the default. Loading raises when the saved
checkpoint does not match the supplied base model. For loading,
`strict=False` turns supported compatibility or merge failures into warnings,
but the safest workflow is to load PEFT checkpoints onto the same base
checkpoint and model configuration used during fine-tuning.

## Hooks in fine-tuning

The hook system in `FineTuningStrategy` is the same as in
`TrainingStrategy` — see {ref}`training-hooks` for the full hook lifecycle,
available stages, and how to write custom hooks.

One ordering detail is specific to fine-tuning: the strategy internally
registers PEFT hooks first, including
{py:class}`~nvalchemi.training.hooks.BaseFingerprintHook` when base
fingerprinting is enabled and then PEFT method hooks such as
{py:class}`~nvalchemi.training.peft.lora_hook.LoRAHook`. It then registers
{py:class}`~nvalchemi.training.hooks.ModulePatchHook`, followed by
{py:class}`~nvalchemi.training.hooks.TrainableParameterHook`. When parameter
filtering is active, {py:class}`~nvalchemi.training.hooks.FineTuningSummaryHook`
caches the trainable-parameter summary before any hooks you supply via
`hooks=`. This means your custom hooks always observe the already-adapted
module tree, the already-filtered optimizer parameter groups, and the cached
fine-tuning summary. If a hook inspects which parameters are in the optimizer,
it will see the post-filter state.

For per-batch policies — mixed precision, gradient clipping, custom scheduler
stepping — use {ref}`training-update-hooks` rather than registration-time
hooks.

## Notes on fine-tuning models

### MACE

When loading a MACE checkpoint for fine-tuning, make sure the model is in a
trainable form before passing it to `FineTuningStrategy`.
{py:meth}`nvalchemi.models.mace.MACEWrapper.from_checkpoint` returns an
eval-mode wrapper by default; the training strategy switches it to train mode
during `run`, so that part is handled for you. The one flag to watch is
`compile_model`: setting it to `True` is inference-only because it freezes
parameters before `torch.compile`. Always use `compile_model=False` for
fine-tuning:

```python
from nvalchemi.models.mace import MACEWrapper

pretrained_model = MACEWrapper.from_checkpoint(
    "runs/pretrain/checkpoints",
    compile_model=False,
)

strategy = FineTuningStrategy(
    models=pretrained_model,
    trainable_patterns=("main.model.readout.*",),
    ...
)
```

If a compiled model is passed, the trainable parameter filter will match
nothing and the strategy will raise an error before the run begins.

## API reference

See {ref}`training-finetuning-api` for the API reference for
{py:class}`~nvalchemi.training.FineTuningStrategy`,
{py:class}`~nvalchemi.training.hooks.ModulePatchHook`,
{py:class}`~nvalchemi.training.hooks.TrainableParameterHook`,
{py:class}`~nvalchemi.training.hooks.FineTuningSummaryHook`,
{py:class}`~nvalchemi.training.peft.lora.LoRAConfig`,
{py:func}`~nvalchemi.training.peft.load_peft_checkpoint_into_model`,
and {py:class}`~nvalchemi.training.hooks.BaseFingerprintHook`.
