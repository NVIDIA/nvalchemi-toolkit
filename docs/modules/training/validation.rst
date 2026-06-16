.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _validation-api:

==========
Validation
==========

Validation reuses the training loop's forward pass and loss machinery but runs
it under configurable inference conditions. By default a validation pass calls
the strategy's ``training_fn`` and evaluates the same
:class:`~nvalchemi.training.losses.ComposedLossFunction`, so it reports the
metrics you train against. Both are overridable: set ``validation_fn`` to run a
user-defined validation function, and set ``loss_fn`` to score against a
different metric (for example a plain MAE for monitoring while you train against
a weighted energy/force loss).

What stays fixed is that there is **no backward pass and no optimizer step** — a
validation pass only runs the forward and the loss, then reduces the per-batch
results across ranks into a single summary. The remaining inference semantics are
configuration-driven, not automatic: modules are placed in eval mode when
``set_eval`` is true (the default) and restored afterward, and autograd is
governed by ``grad_mode`` (see :ref:`configuring-validation-gradients`).

Because validation is a first-class part of
:class:`~nvalchemi.training.TrainingStrategy`, you do not register a validation
hook — you attach a :class:`~nvalchemi.training.ValidationConfig` to the
strategy and validation runs automatically. The mechanics live in a reusable
:class:`~nvalchemi.training.ValidationLoop`, which you can also drive yourself
for standalone metric evaluation (see
:ref:`standalone-validation`).

.. seealso::

   - :doc:`hooks` — training lifecycle stages and update hooks, including
     ``AFTER_VALIDATION``.


How validation differs from training
------------------------------------

Both loops move each batch to the device, call the forward function, and
evaluate the same composed loss. From there they diverge:

.. list-table::
   :header-rows: 1
   :widths: 26 32 42

   * - Aspect
     - Training step
     - Validation pass
   * - Backward / optimizer step
     - Yes
     - No — forward + loss only
   * - Module mode
     - ``train()``
     - ``eval()`` by default (``set_eval``), restored afterward
   * - Autograd
     - Always on
     - Driven by ``grad_mode`` (see below)
   * - Weights
     - Live training weights
     - Live, or the EMA / inference slot
   * - Per-batch output
     - Loss for the update
     - Accumulated into a reduced summary
   * - Gradient buffers
     - Updated in place
     - Snapshotted, cleared, restored

Validation snapshots, clears, and restores parameter ``.grad`` buffers around
the pass, so it never corrupts in-flight training gradients even when it runs
with autograd enabled. Module training modes are likewise snapshotted and
restored.


.. _configuring-validation-gradients:

Configuring gradients
----------------------

Some validation losses need autograd at inference time. Force and stress losses,
for example, differentiate energy with respect to positions, so the forward pass
must build a graph even though no optimizer step follows. ``ValidationConfig``
exposes this through ``grad_mode``:

- ``"auto"`` (default) — enable gradients when any loss component reports
  ``requires_eval_grad=True`` (e.g. force/stress terms) and disable them
  otherwise. This usually does the right thing without configuration.
- ``"enabled"`` — always run under ``torch.enable_grad()``.
- ``"disabled"`` — always run under ``torch.no_grad()``.

When gradients are enabled the loop runs each batch under
``torch.enable_grad()``; otherwise it uses ``torch.no_grad()``. Either way the
parameter gradient buffers are restored on exit.


The validation flow
--------------------

A single pass proceeds as:

1. **Setup** — snapshot module training modes and set them to eval (when
   ``set_eval=True``); snapshot and clear parameter gradients (when grad-enabled).
2. **Per batch** — move the batch to the device; clear gradients; run the
   forward + loss under the resolved autograd and autocast contexts; accumulate
   the per-component loss diagnostics; invoke the optional ``batch_callback``.
3. **Reduce** — all-reduce the accumulated totals across ranks and build the
   summary dict (reduced and available on every rank in distributed runs).
4. **Teardown** — restore parameter gradients and module training modes, even
   if the pass raised.


Strategy-owned validation
-------------------------

Assign a :class:`~nvalchemi.training.ValidationConfig` to
``strategy.validation_config`` and validation runs automatically inside
``strategy.run(...)``:

- at a **step cadence** (``every_n_steps``), after the completed optimizer
  step so EMA weights are already current, or
- at an **epoch cadence** (``every_n_epochs``), at the epoch boundary, and
- once **unconditionally at end-of-training** whenever a config is present.

Each pass stores its summary on ``strategy.last_validation`` and fires the
``AFTER_VALIDATION`` hook stage, so loggers can read the live summary before
any metric-driven learning-rate scheduler consumes it.

.. code-block:: python

   from nvalchemi.training import TrainingStrategy, ValidationConfig

   strategy = TrainingStrategy(...)
   strategy.validation_config = ValidationConfig(
       validation_data=val_data,  # a re-iterable container of Batch
       every_n_epochs=1,
   )
   strategy.run(train_loader)

``validation_data`` must be a *re-iterable* container (a ``list``,
``DataLoader``, or ``Dataset``) — the strategy walks it afresh on every pass, so
one-shot generators are rejected at construction time. By default validation
reuses the strategy's ``training_fn`` and ``loss_fn``; set ``validation_fn`` or
``loss_fn`` on the config to override either.


Using regular hooks with validation
------------------------------------

Validation does not bypass the hook system. Validation passes execute inside
``strategy.run(...)``, so every hook you register on the strategy keeps firing
on its normal stages. The dedicated tap-off point is ``AFTER_VALIDATION``,
fired from inside ``TrainingStrategy.validate()`` the moment a summary is
produced — and before any metric-driven scheduler consumes it. Register an
ordinary hook on that stage to log aggregate metrics from ``ctx.validation``:

.. code-block:: python

   from nvalchemi.training import TrainingStage

   class SummaryLogger:
       stage = TrainingStage.AFTER_VALIDATION
       frequency = 1

       def __call__(self, ctx, stage):
           summary = ctx.validation
           if ctx.global_rank == 0 and summary is not None:
               my_tracker.log(val_loss=float(summary["total_loss"]))

   strategy.register_hook(SummaryLogger())

This is also how metric-driven learning-rate scheduling is wired (see
:ref:`metric-driven-schedulers`): the summary is available to consumers on the
same iteration the pass runs.


Inference model slot
--------------------

``TrainingStrategy`` owns an ``inference_model`` slot. Validation reads it via
the config's ``use_ema`` policy; an :class:`~nvalchemi.training.EMAHook`
publishes its averaged module into the slot at ``AFTER_OPTIMIZER_STEP``. The
writer (EMA / SWA / a distillation teacher) and the reader (validation) never
inspect each other — both only know the strategy. An empty slot falls back to
the live training model(s).


.. _metric-driven-schedulers:

Metric-driven schedulers
------------------------

``ReduceLROnPlateau`` and subclasses are metric-driven: they step only at
validation checkpoints, consuming a scalar extracted from the validation
summary via :attr:`OptimizerConfig.scheduler_metric_adapter
<nvalchemi.training.OptimizerConfig>` (a summary-dict key string or a
callable). Time-based schedulers continue to step every optimizer step.


.. _tapping-off-validation-data:

Tapping off per-batch data with ``batch_callback``
--------------------------------------------------

The ``AFTER_VALIDATION`` hook above sees only the reduced *summary*. When you
need the individual batches — to stream predictions to a Zarr store, dump
per-sample diagnostics, or run a custom error analysis — configure a
``batch_callback``. The toolkit ships no output-sink machinery: you bring your
own sink and the loop simply hands you each batch as it goes.

A ``batch_callback`` is any object matching the
:class:`~nvalchemi.training.BatchValidationCallback` protocol. It is invoked
once per validation batch from inside
:meth:`~nvalchemi.training.ValidationLoop.execute`, immediately after that
batch's predictions and loss are computed. The call is keyword-only —
``batch``, ``predictions``, ``loss``, ``batch_count``, ``step_count``, and
``epoch`` — and you own the sink, its buffering, and its I/O:

.. code-block:: python

   from nvalchemi.training import ValidationConfig

   class ZarrBatchSink:
       """Example escape-hatch sink — write predictions to a Zarr store."""

       def __init__(self, store):
           self._store = store

       def __call__(
           self, *, batch, predictions, loss, batch_count, step_count, epoch
       ):
           group = self._store.require_group(f"step_{step_count}")
           group[f"batch_{batch_count}"] = predictions["energy"].cpu().numpy()

   config = ValidationConfig(
       validation_data=val_data,
       batch_callback=ZarrBatchSink(my_zarr_store),
   )

A plain function works too — any callable with the keyword-only signature
satisfies the protocol:

.. code-block:: python

   def log_batch(*, batch, predictions, loss, batch_count, step_count, epoch):
       ...  # write predictions / per-batch loss to your store of choice

   config = ValidationConfig(validation_data=val_data, batch_callback=log_batch)


.. _standalone-validation:

Standalone validation (metric evaluation)
-----------------------------------------

The same :class:`~nvalchemi.training.ValidationLoop` that the strategy drives
can be run on its own — for example, to evaluate a
trained checkpoint against a held-out set and read back the metrics. Standalone
construction takes the dependencies the strategy would otherwise supply: an
explicit ``model`` (or named ``models``), a ``validation_fn``, a loss (directly
or via ``config.loss_fn``), and optionally an ``autocast`` factory and an
explicit ``grad_enabled`` override. It is a context manager — ``execute()`` must
be called inside the ``with`` block — that snapshots and restores training modes
and gradients on exit, even on exception:

.. code-block:: python

   from nvalchemi.training import ValidationConfig, ValidationLoop

   config = ValidationConfig(validation_data=val_data, loss_fn=loss_fn)
   loop = ValidationLoop(
       validation_data=val_data,
       config=config,
       device=device,
       model=model,
       validation_fn=validation_fn,
   )
   with loop as active:
       summary = active.execute()

   print(summary["total_loss"])

The returned ``summary`` is the same dictionary surfaced on
``ctx.validation`` / ``strategy.last_validation`` during integrated training. It
contains ``total_loss``, per-component totals/weights/samples, batch and sample
counts, ``model_source`` (``"ema"`` / ``"mixed"`` / ``"live"``),
``ema_model_keys``, ``precision``, and ``distributed_reduced``. Under
distributed execution the reduced summary is returned on every rank; guard
external side effects such as tracker logging with ``ctx.global_rank == 0``.

.. note::

   If your loss differentiates the model output (force or stress losses), set
   ``grad_mode="enabled"`` on the config or pass ``grad_enabled=True`` so the
   standalone forward builds an autograd graph; ``grad_mode="auto"`` does this
   for you when run through a strategy but standalone callers should be explicit
   when the loss object cannot be introspected.


API reference
-------------

.. currentmodule:: nvalchemi.training

.. autosummary::
   :toctree: generated
   :nosignatures:

   ValidationConfig
   ValidationLoop
   BatchValidationCallback
