.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _training-hooks-api:
.. _training-hooks:
.. _training-update-hooks:

.. seealso::

   - **Core framework**: :ref:`hooks-api` — the ``Hook`` protocol,
     context dataclasses, and ``HookRegistryMixin``.
   - **User guide**: :ref:`training_guide` — training lifecycle and
     extension points.

Training update hooks
=====================

Training update hooks are for policies that need to participate in the
weight-update portion of a training batch. They are intentionally narrower than
general :class:`~nvalchemi.hooks.Hook` objects: a
:class:`~nvalchemi.training.hooks.TrainingUpdateHook` only runs on the stages
owned by :class:`~nvalchemi.training.hooks.TrainingUpdateOrchestrator`, and the
orchestrator performs the actual ``backward()``, optimizer step, scheduler step,
and gradient zeroing calls.

Use this hook family when multiple update policies need to coordinate around the
same batch update. Typical examples include gradient accumulation, mixed
precision, gradient clipping, spike skipping, and post-step model averaging.
Use a standard training hook for read-only observation or lifecycle logic that
does not need to own backward or optimizer-step behavior.

Most users register concrete :class:`~nvalchemi.training.hooks.TrainingUpdateHook`
instances, such as :class:`~nvalchemi.training.hooks.MixedPrecisionHook` or
:class:`~nvalchemi.training.hooks.EMAHook`, directly on the strategy. The
:class:`~nvalchemi.training.hooks.TrainingUpdateOrchestrator` is the coordination
object that runs those update hooks in one ordered update path; strategies create
it automatically for bare update hooks. Construct it yourself only when you need
to pre-compose update hooks before passing them to a strategy.

Fine-tuning helpers such as
:class:`~nvalchemi.training.hooks.ModulePatchHook` and
:class:`~nvalchemi.training.hooks.TrainableParameterHook` are
registration-time hooks. They adapt the model tree and optimizer parameter set
before training starts, but they do not own any batch update stages. See
:ref:`finetuning_guide` and :ref:`training-finetuning-api` for those workflows.

``ctx.step_count`` tracks completed optimizer/scheduler steps on this worker,
and ``ctx.global_step_count`` tracks completed optimizer/scheduler steps across
all data-parallel workers. If an update hook vetoes ``DO_OPTIMIZER_STEP`` for
gradient accumulation or spike skipping, the batch still advances
``ctx.batch_count`` and ``ctx.epoch_step_count`` but does not advance either
step counter.

TrainingStage
-------------

:class:`~nvalchemi.training.TrainingStage` enumerates the eighteen
hook-firing points within a training run:

.. graphviz::
   :caption: TrainingStage hook firing points across a training run.

   digraph training_stages {
       rankdir=TB
       compound=true
       fontname="Helvetica"
       node [fontname="Helvetica" fontsize=10 shape=box style="rounded,filled" fillcolor="#dce6f1"]
       edge [fontname="Helvetica" fontsize=9 style=bold]

       SETUP            [fillcolor="#f9e2ae"]
       BEFORE_TRAINING  [fillcolor="#f9e2ae"]
       AFTER_TRAINING   [fillcolor="#f9e2ae"]
       AFTER_VALIDATION [fillcolor="#f9e2ae" label="AFTER_VALIDATION\n(event-based)"]

       subgraph cluster_epoch {
           label="epoch loop"
           style=rounded
           color="#4a90d9"
           fontcolor="#4a90d9"
           fontname="Helvetica"
           fontsize=11

           BEFORE_EPOCH
           AFTER_EPOCH

           subgraph cluster_batch {
               label="batch loop"
               style=rounded
               color="#2a6090"
               fontcolor="#2a6090"
               fontname="Helvetica"
               fontsize=11

               BEFORE_BATCH
               BEFORE_FORWARD
               AFTER_FORWARD
               BEFORE_LOSS
               AFTER_LOSS
               BEFORE_BACKWARD
               DO_BACKWARD        [label="DO_BACKWARD\n(replacement slot)" fillcolor="#e8d5f5"]
               AFTER_BACKWARD
               BEFORE_OPTIMIZER_STEP
               DO_OPTIMIZER_STEP  [label="DO_OPTIMIZER_STEP\n(replacement slot)" fillcolor="#e8d5f5"]
               AFTER_OPTIMIZER_STEP
               AFTER_BATCH

               BEFORE_BATCH -> BEFORE_FORWARD -> AFTER_FORWARD
               AFTER_FORWARD -> BEFORE_LOSS -> AFTER_LOSS
               AFTER_LOSS -> BEFORE_BACKWARD -> DO_BACKWARD -> AFTER_BACKWARD
               AFTER_BACKWARD -> BEFORE_OPTIMIZER_STEP -> DO_OPTIMIZER_STEP -> AFTER_OPTIMIZER_STEP
               AFTER_OPTIMIZER_STEP -> AFTER_BATCH
           }

           BEFORE_EPOCH -> BEFORE_BATCH [lhead=cluster_batch]
           AFTER_BATCH -> AFTER_EPOCH [ltail=cluster_batch]
       }

       SETUP -> BEFORE_TRAINING
       BEFORE_TRAINING -> BEFORE_EPOCH [lhead=cluster_epoch]
       AFTER_EPOCH -> AFTER_TRAINING [ltail=cluster_epoch]
       AFTER_TRAINING -> AFTER_VALIDATION [style=dashed label="if validate()"]
   }

.. list-table:: Training stages reference
   :widths: 30 10 60
   :header-rows: 1

   * - Stage
     - Value
     - When it fires
   * - ``SETUP``
     - 1
     - Once before optimizer construction; setup hooks may mutate model wrappers
       and dataloaders before training begins.
   * - ``BEFORE_TRAINING``
     - 2
     - Once before the epoch loop, after the model is on device and optimizers
       are constructed.
   * - ``BEFORE_EPOCH``
     - 3
     - Start of each epoch, before the first batch.
   * - ``BEFORE_BATCH``
     - 4
     - Start of each batch, before gradient zeroing.
   * - ``BEFORE_FORWARD``
     - 5
     - Before the model forward pass; right stage for input transforms such as
       neighbor-list construction.
   * - ``AFTER_FORWARD``
     - 6
     - After the model forward pass; predictions are available.
   * - ``BEFORE_LOSS``
     - 7
     - Before the loss computation.
   * - ``AFTER_LOSS``
     - 8
     - After the loss computation; the loss tensor is populated.
   * - ``BEFORE_BACKWARD``
     - 9
     - Before the backward pass; typical slot for observers that need the
       pre-backward state.
   * - ``DO_BACKWARD``
     - 10
     - Replacement slot for the backward pass. At most one hook may claim this
       stage; observers should use ``BEFORE_BACKWARD`` / ``AFTER_BACKWARD``.
   * - ``AFTER_BACKWARD``
     - 11
     - After the backward pass; gradients are available. Typical slot for
       gradient clipping and gradient-norm logging.
   * - ``BEFORE_OPTIMIZER_STEP``
     - 12
     - Immediately before the optimizer step; last pre-step point for observers
       that need unscaled gradients.
   * - ``DO_OPTIMIZER_STEP``
     - 13
     - Replacement slot for the optimizer and LR-scheduler step. At most one hook
       may claim this stage; observers should use ``BEFORE_OPTIMIZER_STEP`` /
       ``AFTER_OPTIMIZER_STEP``.
   * - ``AFTER_OPTIMIZER_STEP``
     - 14
     - After the optimizer and scheduler step path completes; typical slot for EMA
       updates and post-step logging.
   * - ``AFTER_BATCH``
     - 15
     - End of each batch; generic batch cleanup.
   * - ``AFTER_EPOCH``
     - 16
     - End of each epoch, after the last batch.
   * - ``AFTER_TRAINING``
     - 17
     - Once after the final epoch.
   * - ``AFTER_VALIDATION``
     - 18
     - Event-based; fires inside ``TrainingStrategy.validate()`` after a
       validation pass produces its summary. Reliable slot for loggers that
       need the latest validation results.


Distributed data parallel
-------------------------

:class:`~nvalchemi.training.hooks.DDPHook` wraps optimized models in
``torch.nn.parallel.DistributedDataParallel`` during
``TrainingStage.SETUP``. This setup stage runs after distributed rank/device
resolution and before optimizer construction, so optimizers are built from the
DDP-wrapped model parameters.
See :ref:`distributed_manager_guide` for the workflow-level
``DistributedManager`` guide.

.. code-block:: python

   from nvalchemi.distributed import DistributedManager
   from nvalchemi.training.hooks import DDPHook, MixedPrecisionHook
   from nvalchemi.training.strategy import TrainingStrategy

   DistributedManager.initialize()
   manager = DistributedManager()

   strategy = TrainingStrategy(
       ...,
       distributed_manager=manager,
       hooks=[
           DDPHook(find_unused_parameters=False),
           MixedPrecisionHook(precision="bf16"),
       ],
   )

Launch single-node distributed training with ``torchrun``:

.. code-block:: bash

   torchrun --nproc_per_node=2 train.py

``DDPHook`` uses ``TrainingStrategy.distributed_manager`` when one is provided,
falling back to ``torch.distributed`` and torchrun environment variables.
Sampler injection is automatic: a ``DistributedSampler`` is added when the
dataloader does not already have one, and ``sampler.set_epoch()`` is called
each epoch when available.

``DDPHook`` is not a training-update hook, so it does not participate in
``DO_BACKWARD`` or ``DO_OPTIMIZER_STEP``. Register it alongside
``MixedPrecisionHook`` normally; DDP wrapping happens before AMP opens its
per-batch autocast/update path.

.. dataclass-table:: nvalchemi.training.hooks.DDPHook

PyTorch profiler traces
-----------------------

:class:`~nvalchemi.training.hooks.TorchProfilerHook` captures PyTorch profiler
Chrome traces through PhysicsNeMo's profiler wrapper. In training workflows it
starts at ``TrainingStage.BEFORE_TRAINING``, advances the profiler schedule at
``TrainingStage.AFTER_BATCH``, and finalizes at ``TrainingStage.AFTER_TRAINING``
or when the strategy context exits. Standalone ``train_batch()`` calls start
lazily at ``TrainingStage.BEFORE_BATCH`` and still finalize when the context
closes.

.. code-block:: python

   from torch.profiler import ProfilerActivity, schedule

   from nvalchemi.training.hooks import TorchProfilerHook
   from nvalchemi.training.strategy import TrainingStrategy

   profile_hook = TorchProfilerHook(
       output_dir="profiles/train-run",
       activities=(ProfilerActivity.CPU, ProfilerActivity.CUDA),
       schedule=schedule(wait=2, warmup=2, active=5, repeat=1),
       record_shapes=True,
       profile_memory=True,
       with_flops=True,
   )

   strategy = TrainingStrategy(..., hooks=[profile_hook])
   strategy.run(train_loader)

Each process writes to ``profiles/train-run/rank_<global_rank>/torch/`` unless
PhysicsNeMo's distributed manager is active and already owns rank suffixing.

Mixed precision
---------------

:class:`~nvalchemi.training.hooks.MixedPrecisionHook` enables
``torch.amp.autocast`` for the forward/loss portion of the batch and uses
``torch.amp.GradScaler`` when ``precision`` is ``torch.float16``. The
``precision`` argument is required so configs must choose one of the supported
policies explicitly:

.. code-block:: python

   import torch

   from nvalchemi.training.hooks import MixedPrecisionHook
   from nvalchemi.training.strategy import TrainingStrategy

   strategy = TrainingStrategy(
       ...,
       hooks=[MixedPrecisionHook(precision=torch.bfloat16)],
   )

``precision`` accepts the dtype objects ``torch.float32``, ``torch.bfloat16``,
and ``torch.float16``, the canonical strings ``"float32"``, ``"bfloat16"``,
and ``"float16"``, or the shorthand aliases ``"fp32"``, ``"bf16"``, and
``"fp16"``.

The policies are:

* ``torch.float32``: no autocast context is created and no scaler is used.
* ``torch.bfloat16``: eligible ops run under bf16 autocast and no scaler is used.
* ``torch.float16``: eligible forward/loss ops run under fp16 autocast, the hook
  scales the loss before backward, unscales gradients immediately before an
  optimizer step proceeds, and lets the scaler skip steps with ``inf`` or
  ``nan`` gradients.

Register at most one ``MixedPrecisionHook`` per strategy. The strategy rejects
multiple mixed-precision hooks so that autocast, loss scaling, unscale, scaler
step, and scaler update cannot be applied twice in one batch update.

Autocast scope
--------------

The autocast context spans ``BEFORE_BATCH`` through ``DO_BACKWARD`` and is
released before ``backward()``. Model components or loss functions that require
full precision for a specific subregion should use a local
``torch.amp.autocast(..., enabled=False)`` block within that region.

Gradient accumulation
---------------------

Veto ``DO_OPTIMIZER_STEP`` on intermediate microbatches to accumulate gradients
across a window of K batches. Under ``torch.float16``,
:class:`~nvalchemi.training.hooks.MixedPrecisionHook` suppresses
``GradScaler.unscale``, ``GradScaler.step``, and ``GradScaler.update`` for
vetoed batches; gradients remain scaled until the veto is lifted on the Kth
batch. Schedulers advance only when the paired optimizer actually stepped.

Zero-gradient policy is set by the ``BEFORE_BATCH`` return value: return
``proceed=False`` on intermediate microbatches to skip zeroing and accumulate
into existing gradients.

Validation
----------

``TrainingStrategy.validate()`` honors a registered
:class:`~nvalchemi.training.hooks.MixedPrecisionHook` automatically; the
``use_mixed_precision`` field on
:class:`~nvalchemi.training.ValidationConfig` controls whether autocast is
active during inference. The standalone
:class:`~nvalchemi.training.ValidationLoop` is hook-agnostic and takes an
explicit ``autocast`` callable instead. See :doc:`validation`.

Stage constraints
-----------------

Training update hooks always receive ``(ctx, stage, will_skip)`` and return
``(proceed, loss)``. The meaning of those values depends on the stage:

.. list-table:: Training update hook stage contract
   :widths: 18 22 22 38
   :header-rows: 1

   * - Stage
     - Hook responsibility
     - Return contract
     - Restrictions and expectations
   * - ``BEFORE_BATCH``
     - Decide whether the orchestrator should call
       :func:`~nvalchemi.training.optimizers.zero_gradients`.
     - ``proceed`` must be a strict ``bool``. Any ``False`` vetoes gradient
       zeroing. ``loss`` is ignored.
     - Do not call ``backward()``, ``optimizer.step()``, or
       ``scheduler.step()``. Use this stage for zero-grad policy, per-batch
       update bookkeeping, or resetting state that is safe before the forward
       pass.
   * - ``DO_BACKWARD``
     - Transform or replace ``ctx.loss`` before the orchestrator calls
       ``backward()`` once.
     - ``loss`` must be a :class:`torch.Tensor`. ``proceed`` is ignored.
     - Do not call ``backward()`` directly. Return the loss tensor the next
       update hook should see. This is the stage for loss scaling and other
       loss-space transforms.
   * - ``DO_OPTIMIZER_STEP``
     - Decide whether the orchestrator should call
       :func:`~nvalchemi.training.optimizers.step_optimizers` and
       :func:`~nvalchemi.training.optimizers.step_lr_schedulers`.
     - ``proceed`` must be a strict ``bool``. Any ``False`` vetoes both the
       optimizer and scheduler step. ``loss`` is ignored.
     - Do not call ``backward()``. Avoid side effects that assume a step will
       run when ``will_skip`` is ``True``. This is the stage for pre-step logic
       such as gradient clipping, scaler updates, and accumulation/spike-skip
       decisions.
   * - ``AFTER_OPTIMIZER_STEP``
     - Observe the final step decision and run post-step bookkeeping.
     - ``proceed`` and ``loss`` are ignored. ``will_skip`` tells the hook
       whether the optimizer/scheduler step was vetoed.
     - Do not call ``backward()`` or perform another optimizer/scheduler step.
       Use this stage for work that should happen after the step path, such as
       EMA updates, diagnostics, and state cleanup.

Composition rules
-----------------

All update hooks for a strategy are composed into one orchestrator. Lower
``priority`` values run first, and registration order breaks ties. The
orchestrator keeps calling later hooks after a veto so they can observe
``will_skip=True`` and update their own state consistently.

Only one object may own ``DO_BACKWARD`` or ``DO_OPTIMIZER_STEP`` in a
:class:`~nvalchemi.training.strategy.TrainingStrategy`. For convenience, the
strategy auto-wraps bare :class:`~nvalchemi.training.hooks.TrainingUpdateHook`
instances into one :class:`~nvalchemi.training.hooks.TrainingUpdateOrchestrator`.
Passing ``stage=...`` while registering an update hook is not supported because
update hooks declare their stages through the orchestrator.

Checkpointing
-------------

:class:`~nvalchemi.training.hooks.CheckpointHook` saves model weights,
optimizer state, scheduler state, training counters, and the state of any
:class:`~nvalchemi.hooks.CheckpointableHook` implementors to disk at a
configured cadence.

.. dataclass-table:: nvalchemi.training.hooks.CheckpointHook

Pass the same hook instance to ``TrainingStrategy.load_checkpoint()`` when
resuming so its internal state is restored alongside the model and optimizer:

.. code-block:: python

   from nvalchemi.training import CheckpointHook, TrainingStrategy

   checkpoint = CheckpointHook(
       "runs/my-model/checkpoints",
       step_interval=500,
       max_checkpoints=5,
   )

   # Resume from step 500
   strategy = TrainingStrategy.load_checkpoint(
       "runs/my-model/checkpoints/step_500",
       hooks=[checkpoint],
   )
   strategy.run(train_loader)

EMA model averaging
-------------------

:class:`~nvalchemi.training.hooks.EMAHook` maintains an exponential moving
average of one model's weights. The averaged weights are updated at
``AFTER_OPTIMIZER_STEP`` only when the optimizer step was not vetoed, so the
EMA step count stays in sync with the actual optimizer step count.

.. dataclass-table:: nvalchemi.training.hooks.EMAHook

Access the averaged model weights via ``ema.averaged_model`` after training.

.. code-block:: python

   from nvalchemi.training.hooks import EMAHook
   from nvalchemi.training.strategy import TrainingStrategy

   ema = EMAHook(model_key="main", decay=0.999)
   strategy = TrainingStrategy(..., hooks=[ema])
   strategy.run(train_loader)

   averaged_model = ema.averaged_model

Restartable update hooks
------------------------

If a training hook owns state that changes resumed training behavior — EMA
weights, a learned update policy, accumulated statistics — implement
:class:`~nvalchemi.hooks.CheckpointableHook` by adding ``state_dict()`` and
``load_state_dict()``. The strategy checkpoint loader restores state only into
hooks that satisfy this protocol.

.. code-block:: python

   from nvalchemi.hooks import CheckpointableHook
   from nvalchemi.training.hooks import TrainingUpdateHook

   class StatefulHook(TrainingUpdateHook):
       def __init__(self):
           self.step_count = 0

       def state_dict(self):
           return {"step_count": self.step_count}

       def load_state_dict(self, state):
           self.step_count = int(state["step_count"])

       def __call__(self, ctx, stage, will_skip):
           ...
           return True, ctx.loss

   assert isinstance(StatefulHook(), CheckpointableHook)

For Pydantic update hooks, call ``model_dump()`` inside ``state_dict()`` to
capture field values before appending non-field runtime tensors. Tensor state
must remain in ``state_dict()``; use ``model_dump_json()`` only for
configuration records or diagnostics.


API reference
-------------

.. currentmodule:: nvalchemi.training.hooks

.. autosummary::
   :toctree: generated
   :nosignatures:

   DDPHook
   MixedPrecisionHook
   TorchProfilerHook
   TrainingUpdateHook
   TrainingUpdateOrchestrator
   EMAHook
   CheckpointHook
