.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _training-checkpoints:

Training checkpoints
====================

A :class:`~nvalchemi.training.TrainingStrategy` is the runtime object that owns
one training job: models, optimizers, schedulers, dataloaders supplied at run
time, hooks, counters, and the callable used to turn a batch into a loss. The
strategy is the object you run during training; a checkpoint is a serialized
snapshot of enough of that strategy state to resume the same job later.

Training checkpoints therefore contain more than model weights. They include
model weights, optimizer state, learning-rate scheduler state, strategy runtime
counters, checkpointable hook state, and a serializable recipe for rebuilding the
strategy components that can be reconstructed from metadata. They are intended
for training restarts, not just inference weight export.

Manual save and restart
-----------------------

Use :meth:`~nvalchemi.training.TrainingStrategy.save_checkpoint` when a live
strategy should write its current training state:

.. code-block:: python

   from nvalchemi.training import TrainingStrategy

   strategy = TrainingStrategy(...)
   strategy.run(train_loader)

   checkpoint_index = strategy.save_checkpoint("runs/example/checkpoints")

Use :meth:`~nvalchemi.training.TrainingStrategy.load_checkpoint` to reconstruct a
new ``TrainingStrategy`` from one of those checkpoint snapshots. The method name
refers to the checkpoint source; the return value is a strategy ready to keep
training after you provide any runtime-only objects the checkpoint cannot
serialize:

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
as an importable dotted path. Importable functions do not need to be passed again:

.. code-block:: python

   # my_project/train_fns.py
   def supervised_step(model, batch):
       predictions = model(batch)
       return loss_fn(predictions, batch)

   strategy = TrainingStrategy(..., training_fn=supervised_step)
   strategy.save_checkpoint(checkpoint_dir)

   restored = TrainingStrategy.load_checkpoint(checkpoint_dir)

If the original strategy used a local function, a closure, or another
non-importable callable, the checkpoint records that the function must be
provided by the caller. Pass ``training_fn=...`` when loading:

.. code-block:: python

   def build_loss(scale):
       def training_fn(model, batch):
           predictions = model(batch)
           return scale * loss_fn(predictions, batch)

       return training_fn

   restored = TrainingStrategy.load_checkpoint(
       checkpoint_dir,
       training_fn=build_loss(scale=0.5),
   )

Hooks are runtime objects and are intentionally supplied at load time. For
example, pass a new :class:`~nvalchemi.training.hooks.CheckpointHook` when the
restarted job should continue periodic checkpoint writes:

.. code-block:: python

   from nvalchemi.training import CheckpointHook, TrainingStrategy

   strategy = TrainingStrategy.load_checkpoint(
       "runs/example/checkpoints",
       hooks=[
           CheckpointHook("runs/example/checkpoints", step_interval=1000),
       ],
   )

Restartable hook state
----------------------

Hooks are still runtime objects and must be supplied when loading a strategy.
However, hooks that implement :class:`~nvalchemi.hooks.CheckpointableHook` have
their runtime state stored in strategy checkpoints and restored into the
matching hook supplied at load time. This is intended for hooks whose state
changes training semantics, such as :class:`~nvalchemi.training.hooks.EMAHook`
and its averaged weights.

.. code-block:: python

   from nvalchemi.training import CheckpointHook, EMAHook, TrainingStrategy

   checkpoint_dir = "runs/example/checkpoints"

   ema = EMAHook(model_key="main", decay=0.999)
   strategy = TrainingStrategy(
       ...,
       hooks=[
           ema,
           CheckpointHook(checkpoint_dir, step_interval=1000),
       ],
   )
   strategy.run(train_loader)

   restored_ema = EMAHook(model_key="main", decay=0.999)
   restored = TrainingStrategy.load_checkpoint(
       checkpoint_dir,
       hooks=[
           restored_ema,
           CheckpointHook(checkpoint_dir, step_interval=1000),
       ],
   )

When a script already constructs the strategy and its runtime hooks, use
:meth:`~nvalchemi.training.TrainingStrategy.restore_checkpoint` to restore checkpoint state into those
live objects in place instead of reconstructing the strategy from metadata:

.. code-block:: python

   restored = TrainingStrategy(
       ...,
       hooks=[
           restored_ema,
           CheckpointHook(checkpoint_dir, step_interval=1000),
       ],
   )
   restored.restore_checkpoint(checkpoint_dir)

Checkpointable hooks are matched by class occurrence in the runtime hook list,
so load-time hooks should be registered in the same relative order as the hooks
that wrote the checkpoint. Non-checkpointable hook state remains the user's
responsibility. Prefer deriving transient state from restored strategy counters
or rebuilding caches at setup time when possible.

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

Model reconstruction specs
--------------------------

Strategy checkpoints store model weights separately from a small JSON model
spec. A model spec records an importable callable plus JSON-serializable
keyword arguments. For ordinary modules, this callable is usually the class
constructor::

   create_model_spec(torch.nn.Linear, in_features=16, out_features=1)

For models that are created by a factory, adapter, monkey patch, or optimized
conversion pass, the spec can point at that factory instead::

   create_model_spec(
       MACEWrapper.from_checkpoint,
       checkpoint_path="small-0b",
       dtype=torch.float32,
       enable_cueq=True,
   )

During load, the checkpoint layer rebuilds the model from the spec and then
loads the saved training weights. If the factory accepts ``device``, the loader
passes ``map_location`` into the factory so device-sensitive conversions, such
as MACE cuEquivariance conversion, happen directly on the target device.

Models may provide their own reconstruction spec by implementing
``checkpoint_spec()`` and returning a :class:`~nvalchemi.training._spec.BaseSpec`
or ``None``. Returning ``None`` keeps the default constructor-introspection
fallback. This is useful for wrappers whose live module cannot be reconstructed
from its transformed ``__init__`` arguments.

Serialization scope
-------------------

Model specs make the broader rule explicit: a checkpoint stores *data and
references*, never live Python objects. The manifest and every component recipe
are plain JSON, and tensor bundles are reloaded with
``torch.load(..., weights_only=True)``. That keeps checkpoints portable and safe
to reload without ``pickle``, but it also bounds what a checkpoint can carry.

A checkpoint can embed:

- **Reconstruction specs** for models, optimizers, and schedulers — an
  importable dotted path to a constructor or factory plus its keyword
  arguments, as described above.
- **Spec keyword arguments** that are JSON-serializable (strings, numbers,
  booleans, ``None``, lists, and dicts) or one of the registered tensor-aware
  types: :class:`torch.Tensor` (stored as ``{dtype, shape, data}``),
  :class:`torch.dtype`, and :class:`torch.device`. Register additional types
  with ``register_type_serializer`` from ``nvalchemi.training``.
- **Tensor state** — model weights, optimizer state, and scheduler state —
  written as separate weights-only bundles.
- **Strategy runtime counters**, so local/global step, batch, and epoch positions resume.
- **Checkpointable hook state** for hooks implementing
  :class:`~nvalchemi.hooks.CheckpointableHook`.
- **The training function**, but only when it resolves to an importable dotted
  path.

A checkpoint cannot embed the following; supply them from your training script
at load time:

- **Non-importable callables** — lambdas, locally defined or nested functions,
  closures, ``functools.partial`` objects, and bound methods of live instances.
  The training function is the common case: when it is one of these, the
  manifest records that it must be provided, and you pass ``training_fn=...`` to
  :meth:`~nvalchemi.training.TrainingStrategy.load_checkpoint` (see
  `Training functions`_).
- **Spec arguments that are neither JSON-serializable nor registered** —
  arbitrary Python objects, open file handles, or live modules. Reduce them to
  plain configuration, or expose a factory that rebuilds them from serializable
  arguments and point the spec at that factory.
- **Targets with positional-only parameters** — ``create_model_spec`` builds a
  recipe from keyword arguments only.
- **Runtime objects** such as hooks, dataloaders, and datasets. You reconstruct
  hooks in your script and pass them at load time; only checkpointable hook
  *state* is restored into them.

The practical takeaway: anything that cannot be reduced to an importable
reference plus serializable arguments belongs in your training script, and the
checkpoint expects you to re-supply it when you reload the strategy.

MACE checkpoints and cuEquivariance
-----------------------------------

When training starts from an existing MACE checkpoint, construct the wrapper
with :meth:`~nvalchemi.models.mace.MACEWrapper.from_checkpoint` and then let
:class:`TrainingStrategy` save and reload the full restart checkpoint::

   import torch

   from nvalchemi.models.mace import MACEWrapper
   from nvalchemi.training import EMAHook, TrainingStrategy

   model = MACEWrapper.from_checkpoint(
       "small-0b",
       device=torch.device("cuda"),
       dtype=torch.float32,
       enable_cueq=True,
   )

   ema = EMAHook(model_key="main", decay=0.999)
   strategy = TrainingStrategy(
       models=model,
       ...,
       hooks=[ema],
   )
   strategy.run(train_loader)
   strategy.save_checkpoint(checkpoint_dir)

On restart, reload the strategy checkpoint rather than saving or loading the
EMA hook in isolation::

   restored_ema = EMAHook(model_key="main", decay=0.999)
   restored = TrainingStrategy.load_checkpoint(
       checkpoint_dir,
       map_location=torch.device("cuda"),
       hooks=[restored_ema],
       training_fn=training_fn,
   )

The saved model spec calls ``MACEWrapper.from_checkpoint`` again with the
recorded MACE checkpoint and options, then the strategy loader restores model
weights, optimizer state, counters, and checkpointable hook state such as EMA
averages.

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

When a native checkpoint should seed a new fine-tuning job rather than resume
the same run, use
:meth:`~nvalchemi.training.FineTuningStrategy.from_pretrained_checkpoint`
instead. It loads the checkpoint model weights into a fresh fine-tuning strategy
without optimizer state, scheduler state, counters, hooks, or run limits. Source
loss and optimizer/scheduler configuration are reused only when explicitly
requested by that constructor.

API reference
-------------

.. currentmodule:: nvalchemi.training

.. autosummary::
   :toctree: generated
   :nosignatures:

   TrainingStrategy.save_checkpoint
   TrainingStrategy.restore_checkpoint
   TrainingStrategy.load_checkpoint
   save_checkpoint
   load_checkpoint

.. currentmodule:: nvalchemi.training.hooks

.. autosummary::
   :toctree: generated
   :nosignatures:

   CheckpointHook
