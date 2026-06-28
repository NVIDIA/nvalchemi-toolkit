.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _validation-api:

==========
Validation
==========

:class:`~nvalchemi.training.ValidationConfig` configures strategy-owned
validation passes; :class:`~nvalchemi.training.ValidationLoop` is the reusable
loop that the strategy drives, and that you can also run standalone. Neither
performs a backward pass or optimizer step — validation runs the forward and
loss only, then reduces per-batch results across ranks.

.. seealso::

   - :doc:`hooks` — ``AFTER_VALIDATION`` stage and update hooks.
   - :ref:`training_guide` — validation configuration patterns and customization.


Training vs validation
----------------------

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
     - Driven by ``grad_mode``
   * - Weights
     - Live training weights
     - Live, or the EMA / inference slot (``use_ema``)
   * - Per-batch output
     - Loss for the update
     - Accumulated into a reduced summary
   * - Gradient buffers
     - Updated in place
     - Snapshotted, cleared, restored


ValidationConfig
----------------

.. dataclass-table:: nvalchemi.training.ValidationConfig

Assign to ``strategy.validation_config`` to enable strategy-owned validation:

.. code-block:: python

   from nvalchemi.training import TrainingStrategy, ValidationConfig

   strategy = TrainingStrategy(...)
   strategy.validation_config = ValidationConfig(
       validation_data=val_data,
       every_n_epochs=1,
   )
   strategy.run(train_loader)

``validation_data`` must be a re-iterable container (``list``, ``DataLoader``,
``Dataset``); one-shot generators are rejected at construction time.


Standalone validation
---------------------

:class:`~nvalchemi.training.ValidationLoop` is a context manager — call
``execute()`` inside the ``with`` block; training modes and gradient buffers
are snapshotted and restored on exit, even on exception:

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

The returned ``summary`` matches ``ctx.validation`` / ``strategy.last_validation``
during integrated training: ``total_loss``, per-component totals, batch and
sample counts, ``model_source``, ``precision``, and ``distributed_reduced``.


API reference
-------------

.. currentmodule:: nvalchemi.training

.. autosummary::
   :toctree: generated
   :nosignatures:

   ValidationConfig
   ValidationLoop
   BatchValidationCallback
