.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _training-checkpoints:

Training checkpoints
====================

Training checkpoints capture enough state to stop and restart a
:class:`~nvalchemi.training.TrainingStrategy`: model weights, optimizer state,
learning-rate scheduler state, strategy runtime counters, and the serializable
strategy recipe. They are intended for training restarts, not just inference
weight export.

Manual save and restart
-----------------------

Use :meth:`~nvalchemi.training.TrainingStrategy.save_checkpoint` when a script
wants to take a one-off checkpoint at a known point:

.. code-block:: python

   from nvalchemi.training import TrainingStrategy

   strategy = TrainingStrategy(...)
   strategy.run(train_loader)

   checkpoint_index = strategy.save_checkpoint("runs/example/checkpoints")

Reload with :meth:`~nvalchemi.training.TrainingStrategy.load_checkpoint` when
the checkpoint was written from a strategy:

.. code-block:: python

   from nvalchemi.training import TrainingStrategy

   strategy = TrainingStrategy.load_checkpoint(
       "runs/example/checkpoints",
       map_location="cpu",
       training_fn=training_fn,
   )

   strategy.num_steps = 20_000
   strategy.run(train_loader)
   strategy.save_checkpoint("runs/example/checkpoints")

``checkpoint_index=-1`` loads the latest checkpoint recorded in
``manifest.json``. Pass an explicit index to restart from an older point:

.. code-block:: python

   strategy = TrainingStrategy.load_checkpoint(
       "runs/example/checkpoints",
       checkpoint_index=3,
   )

Training functions
------------------

Checkpoint metadata stores the training function only when it can be expressed
as an importable dotted path. If the original strategy used a local function, a
closure, or another non-importable callable, pass ``training_fn=...`` when
loading. Importable functions do not need to be passed again.

Hooks are runtime objects and are intentionally supplied at load time:

.. code-block:: python

   from nvalchemi.training import CheckpointHook, TrainingStrategy

   strategy = TrainingStrategy.load_checkpoint(
       "runs/example/checkpoints",
       hooks=[
           CheckpointHook("runs/example/checkpoints", step_interval=1000),
       ],
   )

Periodic checkpoint hook
------------------------

Use :class:`~nvalchemi.training.hooks.CheckpointHook` for long-running jobs that
should save without custom logic in the training loop:

.. code-block:: python

   from nvalchemi.training import CheckpointHook, TrainingStrategy

   strategy = TrainingStrategy(
       ...,
       hooks=[
           CheckpointHook("runs/example/checkpoints", step_interval=1000),
       ],
   )
   strategy.run(train_loader)

A checkpoint hook owns one cadence policy. Use ``step_interval`` to save every
N completed optimizer steps, or ``epoch_interval`` to save every N completed
epochs. Register separate hooks only when a job intentionally needs separate
checkpoint roots or policies.

By default, ``CheckpointHook`` captures a CPU snapshot on the training thread
and writes that snapshot on a background thread. This avoids racing live model
and optimizer tensors while moving filesystem writes off the main training
path. Pending async writes are flushed when the strategy exits its hook
context.

Lower-level loader
------------------

The module-level :func:`~nvalchemi.training.save_checkpoint` and
:func:`~nvalchemi.training.load_checkpoint` functions remain available when
callers need the full manifest, component dictionaries, validators, model
subsets, or adapter loads. ``TrainingStrategy.load_checkpoint`` deliberately
returns only the restored strategy and rejects component-only checkpoints.

API reference
-------------

.. currentmodule:: nvalchemi.training

.. autosummary::
   :toctree: generated
   :nosignatures:

   TrainingStrategy.save_checkpoint
   TrainingStrategy.load_checkpoint
   save_checkpoint
   load_checkpoint

.. currentmodule:: nvalchemi.training.hooks

.. autosummary::
   :toctree: generated
   :nosignatures:

   CheckpointHook
