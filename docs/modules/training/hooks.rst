.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _training-update-hooks:

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

``ctx.step_count`` tracks completed optimizer/scheduler steps. If an update hook
vetoes ``DO_OPTIMIZER_STEP`` for gradient accumulation or spike skipping, the
batch still advances ``ctx.batch_count`` and ``ctx.epoch_step_count`` but does
not advance ``ctx.step_count``.

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
and ``torch.float16`` or the canonical strings ``"float32"``, ``"bfloat16"``,
and ``"float16"``.

The policies are:

* ``torch.float32``: autocast is disabled and no scaler is used.
* ``torch.bfloat16``: eligible ops run under bf16 autocast and no scaler is used.
* ``torch.float16``: eligible forward/loss ops run under fp16 autocast, the hook
  scales the loss before backward, unscales gradients immediately before an
  optimizer step proceeds, and lets the scaler skip steps with ``inf`` or
  ``nan`` gradients.

Autocast begins from the update-hook ``BEFORE_BATCH`` stage and is released
before ``backward()`` during ``DO_BACKWARD``. In normal strategy execution, that
covers the model forward and configured loss calculation while keeping backward
outside autocast. ``torch.float32`` is a no-op policy and does not create an
autocast context.

With fp16 gradient scaling, accumulated gradients stay scaled until the
effective batch is ready to step. A gradient-accumulation update hook should
veto ``TrainingStage.DO_OPTIMIZER_STEP`` on intermediate microbatches; that
suppresses AMP unscale, scaler step, and scaler update for those batches. When
the accumulation window is complete, the optimizer-step stage proceeds and
``MixedPrecisionHook`` unscales once per optimizer just before stepping.

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

Example
-------

.. code-block:: python

   import torch

   from nvalchemi.training import TrainingStage
   from nvalchemi.training.hooks import TrainingUpdateHook

   class ClipGradients(TrainingUpdateHook):
       priority = 30

       def __init__(self, max_norm: float) -> None:
           self.max_norm = max_norm

       def __call__(self, ctx, stage, will_skip):
           match stage:
               case TrainingStage.DO_OPTIMIZER_STEP:
                   if not will_skip:
                       for optimizer in ctx.optimizers:
                           params = (
                               param
                               for group in optimizer.param_groups
                               for param in group["params"]
                           )
                           torch.nn.utils.clip_grad_norm_(params, self.max_norm)
               case _:
                   pass
           return True, ctx.loss


API reference
-------------

.. currentmodule:: nvalchemi.training.hooks

.. autosummary::
   :toctree: generated
   :nosignatures:

   MixedPrecisionHook
   TrainingUpdateHook
   TrainingUpdateOrchestrator
