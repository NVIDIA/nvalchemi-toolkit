<!-- markdownlint-disable MD014 -->

(training_guide)=

# Training

Model training in NVIDIA ALCHEMI Toolkit is designed to be a highly modular,
extensible, and ergonomic set of core functionalities: everything was designed
with the explicit intention for users and developers to implement their atomistic
modeling intentions with as little friction as possible, whilst keeping
production readiness (e.g. traceability and reproducibility) a background constant.

As with many components in `nvalchemi-toolkit`, many of the core user-facing
interfaces are written with `pydantic`, making early validation and (de)serialization
first-class citizens. The core of the training utilities is organized around
{py:class}`~nvalchemi.training.TrainingStrategy` — a `pydantic` workflow engine
that orchestrates the moving pieces during training, as well as ensuring that
the "recipe" for training can be and is persisted, reproducible, and user-friendly.
This class is similar to a "trainer" abstraction that many are familiar with, albeit
with some key differences in what it comprises for the sake of reproducibility
as well as with modern model training concepts in mind such as multi-model
distillation, multitask loss weight scheduling, model averaging, etc. that have
become recently popular in atomistic AI/ML models.

## Minimal structure

Before getting into the details, it helps to see the overall anatomy at once.
Nearly every training script, however elaborate it eventually becomes, is built
from the same five parts:

1. Build or load one or more models.
2. Create a dataloader that emits {py:class}`~nvalchemi.data.Batch` objects.
3. Define the loss or training function that turns each batch into a scalar loss.
4. Configure optimizer, scheduler, validation, and hook behavior.
5. Execute the strategy, while leaving a trail of metrics and model weights.

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

# configure how the model will be trained
loss_fn = EnergyMSELoss() + ForceMSELoss() * 10.0

# pass a model, the optimizer configuration, and supporting
# functionality as hooks
strategy = TrainingStrategy(
    models=model,
    optimizer_configs=OptimizerConfig(lr=1e-4),
    loss_fn=loss_fn,
    validation_config=ValidationConfig(validation_data=val_loader, every_n_epochs=1),
    hooks=[CheckpointHook("runs/example/checkpoints", epoch_interval=1)],
    num_epochs=20,
)

strategy.run(train_loader)
```

The rest of this page walks through what happens after `run()` starts. The key
idea is that `TrainingStrategy` is not only a loop over batches; it is a small
workflow engine whose public extension points are the values of
{py:class}`~nvalchemi.training.TrainingStage`.

## Lifecycle Overview

`run()` expands into a fixed sequence of stages. The whole of it fits in the
single diagram below, which is useful as a reference when you are trying
to understand the orchestration flow as well as when you are trying to build
new workflows and components:

```{graphviz}
digraph training_lifecycle {
  graph [rankdir=TB, bgcolor="transparent", compound=true, nodesep=0.45, ranksep=0.55];
  node [
    shape=box,
    style="rounded,filled",
    fillcolor="#F8F9FA",
    color="#5C677D",
    fontname="Helvetica"
  ];
  edge [color="#5C677D", fontname="Helvetica"];

  setup [label="SETUP\nworkflow and dataloader preparation"];
  ddp [label="DDPHook\nwrap models, install distributed samplers", fillcolor="#EAF7EA"];
  before_training [label="BEFORE_TRAINING\nonce, before first batch"];

  subgraph cluster_epoch {
    label="epoch loop";
    color="#B7C4D6";
    style="rounded";

    before_epoch [label="BEFORE_EPOCH\nepoch-level initialization"];

    subgraph cluster_batch {
      label="batch/update loop";
      color="#D4DCE8";
      style="rounded";

      before_batch [label="BEFORE_BATCH\nzero-grad policy, accumulation setup"];
      before_forward [label="BEFORE_FORWARD\nlast chance to prepare batch/model inputs"];
      forward [label="training_fn(model, batch)\nmodel predictions", fillcolor="#EAF4FF"];
      after_forward [label="AFTER_FORWARD\npredictions are available"];
      before_loss [label="BEFORE_LOSS"];
      loss [label="loss_fn(predictions, batch)\nstructured loss output", fillcolor="#EAF4FF"];
      after_loss [label="AFTER_LOSS\nloss diagnostics are available"];
      before_backward [label="BEFORE_BACKWARD"];
      do_backward [
        label="DO_BACKWARD\nTrainingUpdateOrchestrator\nmay transform/own backward",
        fillcolor="#FFF4D6"
      ];
      after_backward [label="AFTER_BACKWARD\ngradients are available"];
      before_step [label="BEFORE_OPTIMIZER_STEP\nlast pre-step observation"];
      do_step [
        label="DO_OPTIMIZER_STEP\nTrainingUpdateOrchestrator\nmay veto/own step",
        fillcolor="#FFF4D6"
      ];
      after_step [label="AFTER_OPTIMIZER_STEP\nEMA, step diagnostics, step validation"];
      after_batch [label="AFTER_BATCH\nbatch logging, cleanup, checkpoint cadence"];
    }

    after_epoch [label="AFTER_EPOCH\nepoch logging/checkpoints, epoch validation"];
  }

  after_training [label="AFTER_TRAINING\nfinal training cleanup"];
  final_validation [label="final validation\nif configured", fillcolor="#EAF7EA"];
  after_validation [label="AFTER_VALIDATION\nvalidation loggers and metric schedulers", fillcolor="#EAF7EA"];

  setup -> ddp [label="setup hooks"];
  ddp -> before_training;
  before_training -> before_epoch;
  before_epoch -> before_batch;
  before_batch -> before_forward -> forward -> after_forward;
  after_forward -> before_loss -> loss -> after_loss;
  after_loss -> before_backward -> do_backward -> after_backward;
  after_backward -> before_step -> do_step -> after_step -> after_batch;
  after_batch -> before_batch [label="next batch", style=dashed];
  after_batch -> after_epoch [label="epoch exhausted"];
  after_epoch -> before_epoch [label="next epoch", style=dashed];
  after_step -> after_validation [label="every_n_steps", style=dotted];
  after_epoch -> after_validation [label="every_n_epochs", style=dotted];
  after_epoch -> after_training [label="target reached"];
  after_training -> final_validation -> after_validation;
}
```

The diagram is meant to be read as both execution order and API map. Stages are
where {doc}`hooks <hooks>` enter the workflow; the filled operation boxes are
where the strategy itself calls the model, loss, backward pass, optimizer,
scheduler, validation, or checkpoint machinery. Most stages are placed as
observation points: hooks can inspect the current
{py:class}`~nvalchemi.hooks.TrainContext`, log metrics, update side state, or
modify workflow-owned objects when that stage allows it. The exception is the two
replacement stages, `DO_BACKWARD` and `DO_OPTIMIZER_STEP`, which are unique to
training and are owned either by the strategy default path or by the
{py:class}`~nvalchemi.training.hooks.TrainingUpdateOrchestrator`.

The per-batch stages in the inner loop are covered in detail in subsequent
sections, but the outer stages are worth describing here since they are where
coarser-grained work belongs. `BEFORE_TRAINING` and `AFTER_TRAINING` fire
exactly once, wrapping the whole run: the former suits one-time setup that needs
the resolved runtime state, and the latter is the place for final teardown such
as flushing a reporting sink or closing a writer. `BEFORE_EPOCH` and
`AFTER_EPOCH` bracket each pass over the dataloader; the latter is the natural
home for epoch-level summaries, periodic checkpoints, and epoch-cadence
validation. The validation stages sit slightly apart from the main flow: a
validation pass runs on a step or epoch cadence, and the moment it finishes
`AFTER_VALIDATION` fires while its reduced summary is still in hand — which is
exactly where validation logging and metric-driven schedulers such as
`ReduceLROnPlateau` do their work.

## Configuring a training strategy

`TrainingStrategy` is organized around two groups of inputs. The forward path —
`models`, `training_fn`, `loss_fn`, and `loss_target_assembler` — controls what
happens to each batch. The loop control layer — `optimizer_configs`,
`validation_config`, `hooks`, and `num_epochs` or `num_steps` — controls when
and how training progresses.

The two components on the forward path are `training_fn` and
`loss_target_assembler`. `training_fn` owns everything from model call to
prediction mapping; replace it when the default `model(batch)` call is not
enough — multi-model workflows, distillation, or any forward pass that needs
more than one model to produce the output. `loss_target_assembler` controls how
those predictions are matched to targets before `loss_fn` is called; replace it
when _targets are not in the batch directly_, e.g. when they come from a teacher
model's output or must be assembled from other sources. Together they let you
replace the entire forward-to-loss path without touching the rest of the
training strategy.

A `training_fn` is a callable that receives either `(model, batch)` for a
single-model strategy or `(models, batch)` for a named multi-model strategy, and
returns a `Mapping` of model outputs. That mapping is passed into
{py:function}`~nvalchemi.training.losses.composition.compute_supervised_loss`,
which reads targets from the batch by `TrainingStrategy.target_keys` unless a
`loss_target_assembler` is supplied. A `loss_target_assembler` must satisfy
{py:class}`~nvalchemi.training.losses.composition.LossTargetAssemblyProtocol`:
it receives `loss_fn`, the prediction mapping, the batch, the current workflow,
and optional `target_keys`, then returns the target mapping.

The distillation example below shows both seams in use. `training_fn` runs both
models and returns their outputs under distinct keys; `teacher_targets`
implements the `loss_target_assembler` protocol and pulls
targets from the prediction mapping rather than the batch:

```python
from collections.abc import Mapping, Sequence

import torch

from nvalchemi.data import Batch
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training.losses import ComposedLossFunction, EnergyMSELoss

# example that employs a student-teacher workflow; the same
# batch is passed into a student and a teacher model. The
# teacher output is returned in the predictions mapping and
# routed into loss_fn by loss_target_assembler.


def training_fn(
    models: Mapping[str, BaseModelMixin],
    batch: Batch,
) -> Mapping[str, torch.Tensor]:
    """Implements the logic for computing predictions, given a set
    of models and an incoming ``Batch`` object.
    """
    student_out = models["student"](batch)
    # teacher is not part of the autograd graph
    with torch.no_grad():
        teacher_out = models["teacher"](batch)
    return {
        "student_energy": student_out["energy"],
        "teacher_energy": teacher_out["energy"].detach(),
    }


def teacher_targets(
    loss_fn: ComposedLossFunction,
    predictions: Mapping[str, torch.Tensor],
    batch: Batch,
    *,
    workflow: object | None = None,
    target_keys: Sequence[str] | None = None,
    batch_label: str = "Batch",
) -> Mapping[str, torch.Tensor]:
    """This method is used to inform the training workflow how
    to obtain the target values to train against.

    Normally, the values would be grabbed from the ``Batch`` object but in
    this case we retrieve them from the ``predictions`` as they
    were returned as part of ``training_fn``.
    """
    return {"teacher_energy": predictions["teacher_energy"]}


# at runtime, the loss function will rely on `teacher_targets`
# to provide the labels
loss_fn = EnergyMSELoss(
    prediction_key="student_energy",
    target_key="teacher_energy",
    per_atom=True,
  )
```

In this example, `prediction_key="student_energy"` is read from the mapping returned
by `training_fn`, while `target_key="teacher_energy"` names the target returned by
`teacher_targets`. Users opt into that routing by passing
`loss_target_assembler=teacher_targets` to `TrainingStrategy`. The strategy calls the
assembler with the configured loss, predictions, batch, and current workflow, then
passes the resulting target mapping into `loss_fn`.

```{warning}
Having `training_fn` and `loss_target_assembler` as a mere callable that's
passed into `TrainingStrategy` was
intentional for the sake of agility: when writing a script, you could simply
embed the function within the same script, or persist it in the package or workflow
you are developing. For security reasons, we do **not** serialize the function as
part of checkpointing, as there are no effective ways to guarantee your callable
function is safe to execute, and that it hasn't been replaced in-flight.

For this reason, it is up to the user/developer to ensure that their training function
is importable, and to ensure that their checkpoints and training function/recipe is
up to date with one another.
```

The hooks in the loop control layer participate at different lifecycle stages.
Hooks that need structural changes before the first optimizer is built —
DDP wrapping, reporter initialization, profiler setup — run during `SETUP`.
Per-batch output should wait for the batch stages that carry the relevant data.
At `run()`, the strategy resolves this startup sequence: it moves models to
devices, lets setup hooks mutate the workflow, normalizes update hooks into a
single orchestrator, and builds optimizers and schedulers. The loop then begins.

Understanding how the strategy tracks progress through that loop is the
foundation for writing hooks that fire at the right time.

## Training Counters

The training workflow tracks progress using a small set of counters:

- `batch_count` counts the number of completed batches on this worker,
- `step_count` counts completed optimizer/scheduler steps on this worker,
- `global_step_count` counts completed optimizer/scheduler steps across all
  data-parallel workers,
- `epoch_count` counts the number of times the dataloader has been exhausted,
- `epoch_step_count` counts the number of batches consumed in the current epoch.

The distinction between `batch_count` and `step_count` is important. A batch can
finish without an optimizer step if the training workflow uses gradient
accumulation, spike skipping, or any other update policy that defers or vetoes the
step. Code that cares about data throughput should usually read `batch_count`,
while code that cares about local optimizer state should usually read
`step_count`. Distributed code that needs aggregate optimizer progress, such as
fixed compute budgets (i.e. how many FLOPs have I utilized across all ranks) or
world-size-independent sampler restarts, should read
`global_step_count`; under DDP it advances by the current world size when an
optimizer step runs and is restored from checkpoints.

Inside hooks, these values are available from the
{py:class}`~nvalchemi.hooks.TrainContext` passed into the hook call:

```python
from nvalchemi.training import TrainingStage


# this is just to illustrate how a logger hook can access state
class ProgressLogger:
    stage = TrainingStage.AFTER_BATCH
    frequency = 1

    def __call__(self, ctx, stage):
        logger.info(
            "epoch=%s batch=%s step=%s",
            ctx.epoch,
            ctx.batch_count,
            ctx.step_count,
        )
```

Outside hooks, the same state is available on the strategy object as
`strategy.epoch_count`, `strategy.batch_count`, `strategy.step_count`,
`strategy.global_step_count`, and `strategy.epoch_step_count`. These values are
part of the strategy runtime state and are restored by checkpoints.

After setup, `BEFORE_TRAINING` fires once before the first batch. The epoch loop
then starts with `BEFORE_EPOCH`. At each epoch boundary, the strategy calls
`set_epoch(...)` on distributed samplers when available, so each epoch can use a
deterministic but distinct sample order.

## Batches: Forward, Loss, Backward, Update

With the counters and epoch loop in view, we can zoom in on what happens to a
single batch. Before the batch stages run, the strategy moves the batch onto the
primary training device — the device the model was placed on, which under DDP is
the current rank's GPU. Each stage then gives hooks access to a progressively
richer `TrainContext`, following the natural data-availability order of the
forward pass:

| Stage | Available on `ctx` | Extension opportunity |
|-------|--------------------|-----------------------|
| `BEFORE_BATCH` | Batch, counters | Per-batch setup, zero-gradient policy |
| `BEFORE_FORWARD` | Batch | Transform inputs before the model call |
| `AFTER_FORWARD` | `ctx.predictions` | Log prediction statistics; redirect outputs before loss |
| `AFTER_LOSS` | `ctx.loss` | Per-component and per-sample loss diagnostics |
| `AFTER_BACKWARD` | Gradients | Gradient norm logging, gradient-based monitoring |
| `AFTER_OPTIMIZER_STEP` | Updated weights | EMA updates, learning-rate logging, step metrics |
| `AFTER_BATCH` | Full context | Throughput logging, reporting, checkpoint cadence |

The two replacement stages, `DO_BACKWARD` and `DO_OPTIMIZER_STEP`, are not
observation points — they are owned by either the strategy default path or a
{py:class}`~nvalchemi.training.hooks.TrainingUpdateHook`. Extending those stages
requires the update orchestrator; see [Optimizer Orchestration](#optimizer-orchestration).

The default supervised path calls `training_fn` to produce a prediction mapping,
then calls
{py:function}`~nvalchemi.training.losses.composition.compute_supervised_loss` to
retrieve targets and evaluate `loss_fn`. The resulting structured loss contains
`total_loss` for backpropagation, along with per-component and per-sample
diagnostics accessible at `AFTER_LOSS`. See {doc}`losses` and
{doc}`/modules/training/losses` for the loss object contract.

## Optimizer Orchestration

Once the loss has been computed, the `TrainingStrategy` then needs to be able
to handle it, as well as provide the opportunity for developers to interact
with how backpropagation is performed, perform gradient surgery, etc.
{py:class}`~nvalchemi.training.hooks.TrainingUpdateHook` is the abstraction
for this: it owns the replacement stages `DO_BACKWARD` and
`DO_OPTIMIZER_STEP`, and is the right tool for any workflow that changes how
gradients are computed or applied — mixed precision, gradient accumulation,
gradient clipping, spike skipping, and EMA all fit here.

Optimizers and learning-rate schedulers are configured through
`optimizer_configs`. Each entry names the model parameters it owns and the
optimizer/scheduler objects that should be constructed for those parameters.
During setup, `TrainingStrategy` builds the configured optimizers and schedulers
once, stores them on the runtime context, and exposes them to hooks as
`ctx.optimizers` and `ctx.lr_schedulers`.

When no specialized update hooks are registered (these are discussed below), the
strategy owns the default update sequence, which runs on every batch:

1. zero gradients before the forward pass,
2. call `loss.backward()` after `AFTER_LOSS`,
3. call {py:function}`~nvalchemi.training.optimizers.step_optimizers` to apply
   the parameter update,
4. advance step-based learning-rate schedulers with
   {py:function}`~nvalchemi.training.optimizers.step_lr_schedulers`,
5. advance `step_count`, but only when the optimizer-step path actually executes.

That last point is the one worth internalizing: because `step_count` moves only
when an optimizer step is taken, gradient accumulation and similar policies can
defer an update without corrupting the step bookkeeping.

Metric-based schedulers, such as `ReduceLROnPlateau`, are the exception to this
fixed cadence. Rather than stepping on every optimizer step, they require a
validation quantity to track, and that quantity is only exposed on the
`TrainContext` once training reaches `TrainingStage.AFTER_VALIDATION`.

A `TrainingUpdateHook` can participate in four update stages: `BEFORE_BATCH`,
`DO_BACKWARD`, `DO_OPTIMIZER_STEP`, and `AFTER_OPTIMIZER_STEP`. When one or more
update hooks are registered, the strategy folds them into a single
{py:class}`~nvalchemi.training.hooks.TrainingUpdateOrchestrator`. The orchestrator
becomes the owner of the replacement stages, so the strategy does not also call its
default backward or optimizer-step implementation.

The update stages have distinct responsibilities:

| Stage | Responsibility |
|-------|----------------|
| `BEFORE_BATCH` | Zero-gradient policy and per-batch accumulation setup |
| `DO_BACKWARD` | Backward pass or a transformed version of it |
| `DO_OPTIMIZER_STEP` | Optimizer and scheduler stepping; can veto the step |
| `AFTER_OPTIMIZER_STEP` | Post-step updates such as EMA weights |

Update hooks can be registered directly in `hooks=[...]`; the strategy will wrap
bare update hooks into one orchestrator. They can also be composed explicitly
with `hook_a + hook_b` when a script wants to make the composition visible.

```{note}
Only one object may own `DO_BACKWARD` and only one object may own
`DO_OPTIMIZER_STEP`. The dividing line is ownership: a hook that only *observes*
gradients, learning rates, or counters — logging gradient norms at `AFTER_BACKWARD`,
say — should stay a standard hook, while one that *changes* whether or how
gradients are applied belongs in the update orchestrator.
```

See {doc}`/modules/training/hooks` for the stage contract and the built-in update
hooks.

## Validation, Schedulers, And Reporting

Once the update path is handling gradients and optimizer steps, the next
question is whether the model is actually improving. Validation, metric-driven
scheduling, and reporting all live in the outer lifecycle stages —
`AFTER_VALIDATION`, `AFTER_EPOCH`, `AFTER_BATCH` — so they observe fully
updated weights without interfering with the update path. Some schedulers
also cannot make their decisions without a validation signal.

Validation is configured through
{py:class}`~nvalchemi.training.ValidationConfig` on the strategy. It reuses the
same model, `training_fn`/`validation_fn`, loss function, and target assembly
language as training, but executes under validation semantics: evaluation mode by
default, configurable autograd, optional EMA weights, and distributed reduction of
summary metrics.

Step-cadence validation is checked after `AFTER_OPTIMIZER_STEP`, so it observes
the latest successfully updated weights. Epoch-cadence validation is checked
after `AFTER_EPOCH`. When training finishes, the strategy runs a final validation
pass if validation is configured. Immediately after each validation pass,
`AFTER_VALIDATION` fires while the reduced summary is still available on the
strategy.

```{tip}
When using model averaging (EMA), the hook will automatically use the averaged
model weights for computing validation. This will generally result in
significantly smoother validation curves than the training counterparts.
```

Use `AFTER_VALIDATION` for lifecycle-level validation logging and metric-driven
scheduler behavior. Use the per-batch callback on `ValidationConfig` only when
you need a tap into individual validation batches, predictions, or losses for a
custom sink or offline error analysis.

### Listening to validation results

Register a standard hook on `AFTER_VALIDATION` to read the reduced summary.
The summary is available on every rank; guard external side effects with a
rank check:

```python
from nvalchemi.training import TrainingStage

class SummaryLogger:
    stage = TrainingStage.AFTER_VALIDATION
    frequency = 1

    def __call__(self, ctx, stage):
        summary = ctx.validation
        if ctx.global_rank == 0 and summary is not None:
            my_tracker.log(val_loss=float(summary["total_loss"]))

strategy.register_hook(SummaryLogger())
```

### Per-batch validation callback

When you need more than the reduced summary — per-sample predictions,
domain-level breakdowns, or a custom sink — configure a `batch_callback`
on `ValidationConfig`. Any callable with the keyword-only signature
`(*, batch, predictions, loss, batch_count, step_count, epoch)` satisfies
the {py:class}`~nvalchemi.training.BatchValidationCallback` protocol:

```python
from nvalchemi.training import ValidationConfig

class ZarrBatchSink:
    def __init__(self, store):
        self._store = store

    def __call__(self, *, batch, predictions, loss, batch_count, step_count, epoch):
        group = self._store.require_group(f"step_{step_count}")
        group[f"batch_{batch_count}"] = predictions["energy"].cpu().numpy()

config = ValidationConfig(
    validation_data=val_data,
    batch_callback=ZarrBatchSink(my_zarr_store),
)
```

A plain function with the same keyword-only signature also satisfies the protocol.

### Logging And Reporting

Logging and reporting are observer behavior, so — unlike the update hooks above —
they belong in standard hooks rather than the update path. The only real design
choice is the stage at which a logger runs: late enough that the data it needs
already exists, but no later than necessary. The lifecycle offers a natural home
for each kind of output:

- `AFTER_LOSS` for loss components and per-sample loss summaries,
- `AFTER_BACKWARD` for gradient diagnostics,
- `AFTER_OPTIMIZER_STEP` for learning rate, step status, EMA state, or any
  optimizer-step-dependent metric,
- `AFTER_BATCH` for generic counters, throughput, and final per-batch logging,
- `AFTER_EPOCH` for epoch summaries,
- `AFTER_VALIDATION` for reduced validation summaries.

Because the hook receives `TrainContext`, it can read counters, losses, models,
optimizers, schedulers, the latest validation summary, and the owning workflow
from whichever stage it picks. For a complete guide to writing hooks, see
{doc}`hooks`; for the built-in reporting stack, which uses exactly these stages to
write Rich and TensorBoard output, see {doc}`reporting`.

## Checkpointing

A long run will eventually be interrupted — preemption, a crash, or a deliberate
pause — and resuming it faithfully takes more than the latest weights. While
this may sound straightforward to do with `pickle`, it is not recommended to do
so for security (arbitrary code execution) and reproducibility (code changes): for
these reasons, we designed the checkpointing workflow and abstraction heavily around
making use of `pydantic`, to enable developers and researchers to make reloading/restarting
training products as safely and turn-key as possible. We have tried to hide the
`pydantic` abstraction as much as possible for checkpointing, so users do not need
to be familiar with the framework.

A checkpoint captures four categories of state. Each has a developer-facing
requirement for the loader to reconstruct and restore it correctly:

- **Model weights and architecture**: the model `state_dict` and the
  hyperparameters needed to reconstruct the model class. Models based on the
  ALCHEMI model base classes expose a spec automatically. A custom architecture
  not derived from `BaseModelMixin` must implement the spec protocol for its
  weights and config to be reloadable.
- **Optimizer and scheduler state**: the optimizer `state_dict` and scheduler
  construction parameters, handled automatically when using `OptimizerConfig`.
  Custom `LossWeightSchedule` instances used in composed losses must implement
  `to_spec()` for their state to be included in the checkpoint.
- **Training counters**: `step_count`, `epoch_count`, `batch_count`, and
  `global_step_count` are always saved and restored with no action required.
- **Hook state**: hooks that own restart-critical state — accumulated
  diagnostics, step-conditioned buffers, custom EMA weights — must implement
  {py:class}`~nvalchemi.hooks.CheckpointableHook`. Hooks that do not implement
  the protocol are silently skipped; their state is neither saved nor restored.
  Logging hooks generally do not need this because their artifacts are already
  flushed to an external sink.

Note that `training_fn` and `loss_target_assembler` are not serialized (see the
warning above). The checkpoint cannot reconstruct them; they must be supplied
again at load time.

Use {py:class}`~nvalchemi.training.CheckpointHook` to write checkpoints
periodically from `AFTER_BATCH` or `AFTER_EPOCH`. Use
`TrainingStrategy.save_checkpoint(...)` to save at an explicit point in a
script. See {doc}`/modules/training/checkpoints` for strategy reconstruction,
hook state, model specs, and distributed checkpoint behavior.

### Restart semantics

There are two distinct restart scenarios, and the right API depends on which
applies.

**Resuming an interrupted run** restores the full training state: model
weights, optimizer state, scheduler state, training counters, and any
checkpointable hook state. The run continues from the step immediately after
the checkpoint. Supply the same hook objects the strategy was originally
constructed with — the loader maps saved state into those live objects:

```python
from nvalchemi.training import CheckpointHook, TrainingStrategy

strategy = TrainingStrategy.load_checkpoint(
    "runs/example/checkpoints/step_1000",
    # the checkpoint hook itself must be provided again for continuity
    # as it is stateless and is not kept with the checkpoint
    hooks=[CheckpointHook("runs/example/checkpoints"),],
)
strategy.run(train_loader)
```

**Starting fresh from pretrained weights** loads only model weights. Optimizer
state, training counters, and hook state are not restored — the run starts at
step zero with freshly built optimizers and schedulers. This is the right path
for fine-tuning a pretrained model on a new dataset or task:

```python
from nvalchemi.training import FineTuningStrategy, OptimizerConfig

strategy = FineTuningStrategy.from_pretrained_checkpoint(
    "runs/pretrained/checkpoints/final",
    loss_fn=loss_fn,
    optimizer_configs=OptimizerConfig(lr=1e-5),
    num_epochs=10,
)
strategy.run(finetune_loader)
```

The key difference: `load_checkpoint` resumes exactly where training stopped,
counters and all. `from_pretrained_checkpoint` gives the model its learned
weights but otherwise treats the run as new. See {doc}`finetuning` for the
full fine-tuning API, including parameter freezing and layer-wise learning-rate
configuration.

## Training CLI

The `nvalchemi-training` CLI offers a structured path to launching training
experiments — scaffold a JSON configuration, validate it, then execute — without
requiring a Python script. The `train` command group handles training from
scratch using the same configuration format as fine-tuning workflows, so the
JSON representations of hooks, optimizers, and loss functions map one-to-one
onto the Python API described in the rest of this page.

### Starting a training run

Initialize a spec scaffold for your training job:

```bash
nvalchemi-training train init \
    --dataset data/train.zarr \
    --output-dir runs/my-model \
    --lr 1e-4 \
    --num-steps 5000 \
    --out train.json
```

Multiple `--dataset` flags create a `MultiDataset`-backed dataloader across
the provided paths:

```bash
nvalchemi-training train init \
    --dataset data/domain-a.zarr \
    --dataset data/domain-b.zarr \
    --output-dir runs/my-model \
    --out train.json
```

The generated `train.json` includes a fully populated optimizer, loss
function, and output configuration. However, `strategy.model_specs` is left
empty — training from scratch requires you to supply the architecture. Fill
it in using the same `BaseSpec` JSON format used by checkpoints (`cls_path`
plus constructor keyword fields):

```json
{
  "strategy": {
    "model_specs": {
      "main": {
        "cls_path": "your.package.ModelClass",
        "hidden_size": 128,
        "num_layers": 4
      }
    }
  }
}
```

Use `jq` to merge a separately authored model spec into the base scaffold:

```bash
jq -s 'add' train.json model_spec.json > train_merged.json
```

### Validating and configuring hooks

Before allocating compute, validate the spec and review the training intent:

```bash
nvalchemi-training spec report train.json
```

The report renders an optimizer summary, loss function configuration, hook
list, and a learning-rate preview curve. It also surfaces warnings for
common configuration mistakes: an empty model spec, a missing checkpoint
hook, no validation dataset, or a learning rate that is out of the expected
range. Resolve warnings before executing.

Hooks use the same JSON spec format as the Python API objects: a `cls_path`
and constructor kwargs serialized with `BaseSpec`. A graph-based model
typically needs a neighbor list and a checkpoint hook at minimum:

```json
{
  "source": {
    "hooks": [
      {
        "spec": {
          "cls_path": "nvalchemi.hooks.neighbor_list.NeighborListHook",
          "cutoff": 6.0,
          "format": "coo"
        },
        "stages": ["BEFORE_FORWARD"]
      },
      {
        "spec": {
          "cls_path": "nvalchemi.training.hooks.checkpoint.CheckpointHook",
          "checkpoint_dir": "runs/my-model/checkpoints",
          "step_interval": 500
        }
      }
    ]
  }
}
```

The `stages` list specifies which `TrainingStage` values the hook fires at,
corresponding to the stages described in
[Batches: Forward, Loss, Backward, Update](#batches-forward-loss-backward-update).
Omit `stages` to use the hook's constructor default.

### Executing and resuming

Once the report shows no critical warnings, launch the run:

```bash
nvalchemi-training spec run train.json
```

For distributed training, wrap with `torchrun`:

```bash
torchrun --nproc_per_node=4 -m nvalchemi.training.cli \
    spec run train.json --distributed
```

If the run is interrupted, resume from the latest checkpoint:

```bash
nvalchemi-training spec resume runs/my-model/checkpoints --spec train.json
```

`spec resume` restores model weights, optimizer state, training counters,
and hook state, then continues training exactly where it stopped — the same
behavior as calling `TrainingStrategy.load_checkpoint(...)` directly from
the Python API (described in [Restart semantics](#restart-semantics)).
Specify `--checkpoint-index N` to resume from a particular checkpoint
instead of the latest.
