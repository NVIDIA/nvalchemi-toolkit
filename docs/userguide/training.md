<!-- markdownlint-disable MD014 -->

(training_guide)=

# Training

The training API in `nvalchemi-toolkit` is designed around flexibility for
ML researchers and production engineers to build out highly complex training
workflows. Using the `Hook` abstraction, users can construct modular ways to
change any step of the training process — from data loading and model
orchestration to parameter updates, logging, and reporting.

Training in ALCHEMI is organized around {py:class}`~nvalchemi.training.TrainingStrategy`.
A strategy object owns the runtime pieces for one training job: models, optimizers,
learning-rate schedulers, a training function, optional validation settings, and
hooks. Ultimately, the abstraction on a whole is designed such that each piece can
potentially be substituted for another, including user defined experimental components
to modify anything ranging from how optimization is performed, one or many models,
and how data is transformed throughout the whole training process.

## Minimal structure

Before getting into the details, it helps to see the overall structure at once.
Nearly every training script, however elaborate it eventually becomes, is built
from the same five parts:

1. Build or load one or more models.
2. Create a dataloader that emits {py:class}`~nvalchemi.data.Batch` objects.
3. Define the loss or training function that turns each batch into a scalar loss.
4. Configure optimizer, scheduler, validation, and hook behavior.
5. Run the strategy, optionally saving checkpoints along the way.

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

loss_fn = ComposedLossFunction(
    components=[EnergyMSELoss(), ForceMSELoss()],
    weights=[1.0, 10.0],
)

strategy = TrainingStrategy(
    models=model,
    optimizer_configs=OptimizerConfig(lr=1e-4),
    loss_fn=loss_fn,
    validation=ValidationConfig(validation_data=val_loader, every_n_epochs=1),
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
where {doc}`hooks <hooks>` enter the workflow; the filled operation boxes are where
the strategy
itself calls the model, loss, backward pass, optimizer, scheduler, validation, or
checkpoint machinery. Most stages are placed as observation points: hooks can inspect the
current {py:class}`~nvalchemi.hooks.TrainContext`, log metrics, update side
state, or modify workflow-owned objects when that stage allows it. An exception of this
general pattern is with the two
replacement stages, `DO_BACKWARD` and `DO_OPTIMIZER_STEP`, which are unique to training
and are owned either by the strategy default path or by the
{py:class}`~nvalchemi.training.hooks.TrainingUpdateOrchestrator`.

The per-batch stages in the inner loop are covered in detail in subsequent
sections, but the
outer stages are worth describing here, since they are where coarser-grained work
belongs. `BEFORE_TRAINING` and `AFTER_TRAINING` fire exactly once, wrapping the
whole run: the former suits one-time setup that needs the resolved runtime state,
and the latter is the place for final teardown such as flushing a reporting sink
or closing a writer. `BEFORE_EPOCH` and `AFTER_EPOCH` are run at the beginning of
and after exhausting the dataloader, and the latter in particular is
the natural home for epoch-level
summaries, periodic checkpoints, and epoch-cadence validation. The validation
stages sit slightly apart from the main flow: a validation pass runs on a step or
epoch cadence, and the moment it finishes `AFTER_VALIDATION` fires while its
reduced summary is still in hand — which is exactly where validation logging and
metric-driven schedulers such as `ReduceLROnPlateau` do their work.

## Configuring a training strategy

`TrainingStrategy` starts from declarative pieces:

- `models` define the modules being trained or used by the training function,
- `optimizer_configs` define which model parameters are optimized and how,
- `training_fn` optionally defines a user-supplied forward workflow for each batch,
- `loss_fn` defines how predictions and targets become a structured loss,
- `loss_target_assembler` optionally customizes where loss targets come from,
- `validation` configures optional held-out evaluation,
- `hooks` attach lifecycle behavior,
- `num_epochs` or `num_steps` defines the stopping condition.

A user-defined `training_fn` should be a callable that receives either
`(model, batch)` for a single-model strategy or `(models, batch)` for a named
multi-model strategy, and returns a `Mapping` comprising the model outputs
that will subsequently be used to compute the loss. This mapping is then
passed into {py:function}`~nvalchemi.training.losses.composition.compute_supervised_loss`.
This method is called by `TrainingStrategy` to orchestrate how to retrieve the
training target labels; by default, it uses `TrainingStrategy.target_keys` to
read from the batch, however more complex workflows can be facilitated by
passing in a {py:class}`~nvalchemi.training.losses.composition.LossTargetAssemblyProtocol`,
which allows users to define arbitrary logic for obtaining training targets.

The example snippet below illustrates how to specify custom logic in
the training output as well as the loss computation, on a student-teacher
knowledge transfer/distillation workflow.

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


loss_fn = ComposedLossFunction(
    [
        EnergyMSELoss(
            prediction_key="student_energy",
            target_key="teacher_energy",
            per_atom=True,
        )
    ]
)
```

In this example, `prediction_key="student_energy"` is read from the mapping returned
by `training_fn`, while `target_key="teacher_energy"` names the target returned by
`teacher_targets`. Users opt into that routing by passing
`loss_target_assembler=teacher_targets` to `TrainingStrategy`. The strategy calls the
assembler with the configured loss, predictions, batch, and current workflow, then
passes the resulting target mapping into `loss_fn`.

Before any batch is consumed, the strategy resolves runtime state: it moves models
to devices, lets setup hooks mutate the workflow, normalizes training-update
hooks into one orchestrator, and builds optimizers and schedulers for the
configured models.

Setup hooks should be used for structural changes that must happen before the
first optimizer is built or the first batch is consumed. DDP wrapping is the
canonical example. Runtime reporting sinks, profiler setup, and checkpoint roots
can also initialize here, but per-batch output should wait for the batch stages
that carry the data to be reported.

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
fixed compute budgets or world-size-independent sampler restarts, should read
`global_step_count`; under DDP it advances by the current world size when an
optimizer step runs and is restored from checkpoints.

Inside hooks, these values are available from the
{py:class}`~nvalchemi.hooks.TrainContext` passed into the hook call:

```python
from nvalchemi.training import TrainingStage


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

With the counters and the epoch loop in view, we can zoom in on the part of the
lifecycle you will touch most often: what happens to a single batch. A batch has
to live on the same device as the model that will consume it, so before the batch
stages run the strategy moves it onto the primary training device — the device the
model was placed on, which under DDP is the current rank's GPU. The batch is then
exposed to hooks through `TrainContext` as the strategy walks the same sequence
shown in the lifecycle diagram:

- `BEFORE_BATCH` is used for per-batch preparation and zero-gradient policy,
- `BEFORE_FORWARD` and `AFTER_FORWARD` comprise the call to `training_fn`,
- `BEFORE_LOSS` and `AFTER_LOSS` comprise supervised loss computation,
- `BEFORE_BACKWARD`, `DO_BACKWARD`, and `AFTER_BACKWARD` cover gradient
  computation,
- `BEFORE_OPTIMIZER_STEP`, `DO_OPTIMIZER_STEP`, and `AFTER_OPTIMIZER_STEP` cover
  optimizer/scheduler updates,
- `AFTER_BATCH` is the final per-batch observation and cleanup stage.

The default supervised path is to call `training_fn` to produce a prediction
mapping, then call
{py:function}`~nvalchemi.training.losses.composition.compute_supervised_loss` to
retrieve targets and evaluate `loss_fn`. The resulting structured loss contains
`total_loss`, which is used for backpropagation, along with per-component and
per-sample diagnostics that logging hooks can consume. See {doc}`losses` and
{doc}`/modules/training/losses` for the loss object contract.

The update path is intentionally centralized. Normal hooks can observe stages
around backward and stepping, but the replacement stages `DO_BACKWARD` and
`DO_OPTIMIZER_STEP` are owned by either the default strategy implementation or by
the {py:class}`~nvalchemi.training.hooks.TrainingUpdateOrchestrator`. This avoids
multiple hooks independently calling `backward()` or `optimizer.step()`.

## Optimizer Orchestration

The batch walkthrough showed *where* the optimizer step sits in the lifecycle;
this section is about *what* runs there and how to take it over when the default
is not enough. Optimizers and learning-rate schedulers are configured through
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

Custom update behavior is added with
{py:class}`~nvalchemi.training.hooks.TrainingUpdateHook`, which is the right tool
whenever a workflow changes backward behavior, gradient application, optimizer
stepping, or post-step state — mixed precision, gradient accumulation, gradient
clipping, spike skipping, and EMA are all built this way. Such a hook can
participate in four update stages: `BEFORE_BATCH`, `DO_BACKWARD`,
`DO_OPTIMIZER_STEP`, and `AFTER_OPTIMIZER_STEP`. When one or more update hooks are
registered, the strategy folds them into a single
{py:class}`~nvalchemi.training.hooks.TrainingUpdateOrchestrator`. The
orchestrator becomes the owner of the replacement stages, so the strategy does
not also call its default backward or optimizer-step implementation.

The update stages have separate responsibilities:

- `BEFORE_BATCH` handles zero-gradient policy and per-batch update state,
- `DO_BACKWARD` handles the backward pass or a transformed backward pass,
- `DO_OPTIMIZER_STEP` handles optimizer and scheduler stepping, and can veto the
  step for accumulation or spike-skipping workflows,
- `AFTER_OPTIMIZER_STEP` receives skip-aware state and can update post-step
  objects such as EMA weights.

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

Training the weights is only a small part of the model lifecycle; you also
want to know whether the
model is actually improving, and some schedulers cannot make their decisions
without that signal. Validation is configured through
{py:class}`~nvalchemi.training.ValidationConfig` on the strategy. It reuses the
same model, `training_fn`/`validation_fn`, loss function, and target assembly
language as training, but executes under validation semantics: evaluation mode by
default, configurable autograd, optional EMA weights, and distributed reduction of
summary metrics.

```{tip}
When using model averaging (EMA), the hook will automatically use the
averaged model weights for computing validation. This will generally
result in significantly smoother validation curves than the training
counterparts.
```

Step-cadence validation is checked after `AFTER_OPTIMIZER_STEP`, so it observes
the latest successfully updated weights. Epoch-cadence validation is checked
after `AFTER_EPOCH`. When training finishes, the strategy runs a final validation
pass if validation is configured. Immediately after each validation pass,
`AFTER_VALIDATION` fires while the reduced summary is still available on the
strategy.

Use `AFTER_VALIDATION` for lifecycle-level validation logging and metric-driven
scheduler behavior. Use the per-batch callback on `ValidationConfig` only when
you need a tap into individual validation batches, predictions, or losses for a
custom sink or offline error analysis. See {doc}`/modules/training/validation`
for gradient policy, EMA model selection, scheduler integration, and callback
details.

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

## Checkpoints: Saving Runtime State, Not Just Weights

A long run will eventually be interrupted — preemption, a crash, or a deliberate
pause — and resuming it faithfully takes more than the latest weights.
Checkpointing is implemented as part of the same strategy lifecycle, and a
checkpoint is a restart package, not just a model weight export. It can include model
weights, optimizer state, scheduler state, strategy counters, checkpointable hook
state, and reconstruction metadata for serializable components.

Use {py:class}`~nvalchemi.training.CheckpointHook` when checkpoints should be
written periodically from normal training stages such as `AFTER_BATCH` or
`AFTER_EPOCH`. Use `TrainingStrategy.save_checkpoint(...)` when a script needs to
save explicitly at a known point. Use `TrainingStrategy.load_checkpoint(...)` to
reconstruct a strategy and continue training from a saved checkpoint.

Hooks that own restart-critical state should implement
{py:class}`~nvalchemi.hooks.CheckpointableHook`. The checkpoint loader can then
restore that state into the runtime hook objects supplied by the caller. Logging
hooks often do not need checkpoint state because their external artifacts are
already durable. See {doc}`/modules/training/checkpoints` for strategy
reconstruction, hook state, model specs, and distributed checkpoint behavior.
