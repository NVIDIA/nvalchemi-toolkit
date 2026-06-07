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

.. warning::
   As hooks are runtime objects, checkpointing does not include their state and
   user workflows are responsible for persisting any hook-specific state they
   need across restarts. One option is to use
   :func:`~nvalchemi.training.create_model_spec` to serialize the hook
   specification. Another is to construct the hook from a
   :class:`~pydantic.BaseModel` configuration.

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

Distributed training
--------------------

Distributed checkpointing follows the same file layout as single-process
checkpointing, but only one process should write the shared checkpoint. The
default ``CheckpointHook(rank_zero_only=True)`` uses the
:class:`~nvalchemi.hooks.TrainContext` global rank and saves only on rank 0.
Other ranks continue training and do not write duplicate manifests or state
files.

The usual end-to-end pattern is:

.. code-block:: python

   from nvalchemi.training import CheckpointHook, TrainingStrategy

   checkpoint_dir = "runs/example/checkpoints"

   strategy = TrainingStrategy(
       ...,
       hooks=[
           CheckpointHook(checkpoint_dir, step_interval=1000),
       ],
   )
   strategy.run(train_loader)

On restart, launch the distributed job again and have each process load the
same checkpoint path:

.. code-block:: python

   from nvalchemi.training import CheckpointHook, TrainingStrategy

   checkpoint_dir = "runs/example/checkpoints"

   strategy = TrainingStrategy.load_checkpoint(
       checkpoint_dir,
       map_location=local_device,
       training_fn=training_fn,
       hooks=[
           CheckpointHook(checkpoint_dir, step_interval=1000),
       ],
   )
   strategy.num_steps = 20_000
   strategy.run(train_loader)

``load_checkpoint`` is not rank-zero-only: every process reconstructs its local
strategy, model, optimizer, scheduler, and counters from the shared checkpoint
files. Pass ``map_location`` when the restored process should load onto a
rank-local device instead of the device recorded in the checkpoint metadata.

The checkpoint directory must be visible to every rank before restart. For
periodic hook saves, the async writer is flushed when the strategy exits. For
manual save workflows, users should coordinate their distributed script so only
one rank calls :meth:`~nvalchemi.training.TrainingStrategy.save_checkpoint`,
then ensure all ranks wait until the checkpoint is complete before any rank
tries to reload it.

Current checkpoints store replicated strategy and optimizer state. They are
intended for the training strategies used by this package and do not provide a
separate sharded checkpoint format for distributed optimizers or model shards.
Workflows that shard model or optimizer state outside the strategy checkpoint
must save and restore those sharded states separately.

``DistributedDataParallel`` wrappers are unwrapped before model specs and model
weights are written, so native checkpoints store the underlying model state
without ``module.`` key prefixes. FSDP and FSDP2 require PyTorch Distributed
Checkpoint (DCP) so that each rank can save its shard and reload under a
possibly different topology. Native strategy checkpoints currently reject
FSDP/FSDP2-wrapped models instead of writing incomplete rank-local state. See
the `PyTorch Distributed Checkpoint recipe <https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html>`_
for the DCP workflow.

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
