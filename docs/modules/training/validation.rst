.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _validation-api:

==========
Validation
==========

Validation is a first-class part of :class:`~nvalchemi.training.TrainingStrategy`.
There is no validation hook: set a :class:`~nvalchemi.training.ValidationConfig`
on the strategy and validation passes run automatically.

.. seealso::

   - :doc:`hooks` — training lifecycle stages and update hooks, including
     ``AFTER_VALIDATION``.


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


Inference model slot
--------------------

``TrainingStrategy`` owns an ``inference_model`` slot. Validation reads it via
the config's ``use_ema`` policy; an :class:`~nvalchemi.training.EMAHook`
publishes its averaged module into the slot at ``AFTER_OPTIMIZER_STEP``. The
writer (EMA / SWA / a distillation teacher) and the reader (validation) never
inspect each other — both only know the strategy. An empty slot falls back to
the live training model(s).


Metric-driven schedulers
------------------------

``ReduceLROnPlateau`` and subclasses are metric-driven: they step only at
validation checkpoints, consuming a scalar extracted from the validation
summary via :attr:`OptimizerConfig.scheduler_metric_adapter
<nvalchemi.training.OptimizerConfig>` (a summary-dict key string or a
callable). Time-based schedulers continue to step every optimizer step.


Per-batch logging
-----------------

Validation does not bundle any output-sink machinery. For epoch-level logging,
register an ``AFTER_VALIDATION`` hook and read the summary from
``ctx.validation``. For per-batch logging (e.g. streaming predictions to disk),
pass a ``batch_callback`` on the config: any object matching the
:class:`~nvalchemi.training.BatchValidationCallback` protocol. It is invoked
once per validation batch with keyword-only arguments ``batch``,
``predictions``, ``loss``, ``batch_count``, ``step_count``, and ``epoch``. No
concrete implementation is provided — define your own logging system:

.. code-block:: python

   from nvalchemi.training import ValidationConfig

   def log_batch(*, batch, predictions, loss, batch_count, step_count, epoch):
       ...  # write predictions / per-batch loss to your store of choice

   config = ValidationConfig(validation_data=val_data, batch_callback=log_batch)


Standalone validation loop
--------------------------

:class:`~nvalchemi.training.ValidationLoop` holds the validation mechanics and
can be run on its own, outside any training loop. It is a context manager: the
``with`` block snapshots training modes and gradients, and restores them on
exit (even on exception). Construct it with an explicit model (or named
``models``), ``validation_fn``, loss, and optional ``autocast`` callable:

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


API reference
-------------

.. currentmodule:: nvalchemi.training

.. autosummary::
   :toctree: generated
   :nosignatures:

   ValidationConfig
   ValidationLoop
   BatchValidationCallback
