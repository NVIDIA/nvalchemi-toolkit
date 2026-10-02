.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _training-strategy-api:

Training strategy API
=====================

Core training-loop classes and helpers.

.. seealso::

   - **Fine-tuning guide**: :ref:`finetuning_guide`
   - **Fine-tuning API**: :ref:`training-finetuning-api`
   - **Losses guide**: :ref:`losses_guide`
   - **Training update hooks**: :ref:`training-update-hooks`


Strategies
----------

.. currentmodule:: nvalchemi.training

.. autosummary::
   :toctree: generated
   :nosignatures:

   TrainingStrategy
   default_training_fn

.. dataclass-table:: nvalchemi.training.TrainingStrategy

``devices`` has length ``1`` or ``len(models)``. A named-model run stages one
batch on ``devices[0]`` and hands it to every model. A per-model list must
therefore name the same device in every entry, and a list naming more than one
distinct device is refused at run time. The per-model form lets a caller list
a device for each model it places; it is not a way to span devices. Entries are
compared after :func:`~nvalchemi.data.resolve_device` fills in the index of an
index-less ``cuda``, which means whichever device the process has made current,
so ``cuda`` and ``cuda:0`` count as one device on the rank whose current device
is ``0`` and as two on every other rank. A single-model run is unaffected. A :class:`~nvalchemi.training.hooks.DDPHook` on the NCCL backend
collapses ``devices`` to this rank's own device before the check runs.


Optimizer helpers
-----------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   OptimizerConfig
   setup_optimizers
   zero_gradients
   step_optimizers
   step_lr_schedulers

.. dataclass-table:: nvalchemi.training.OptimizerConfig


Serialization and checkpoints
-----------------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseSpec
   create_model_spec
   create_model_spec_from_json
   register_type_serializer
   CheckpointManifest
   save_checkpoint
   load_checkpoint


Runtime helpers
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   configure_dataloader
   configure_parallelism
   move_to_devices
   rehome_optimizer_state
   freeze_unconfigured_models
   eval_configured_models
   evaluating


Parallelism helpers
-------------------

A parallelism wrapper takes the place of the model a strategy holds. Code that
needs the model's own surface, such as its ``model_config``, a
:class:`~nvalchemi.models.base.BaseModelMixin` method, or an unwrapped
``state_dict``, has to reach through the wrapper first. :func:`unwrap_model`
returns the module behind the wrapper, or the model itself when nothing wraps
it. It recognizes a wrapper by the ``module`` attribute the wrapper publishes,
not by its class, so a hand-rolled wrapper unwraps exactly as
:class:`~torch.nn.parallel.DistributedDataParallel` does.

.. autosummary::
   :toctree: generated
   :nosignatures:

   unwrap_model


Distributed helpers
-------------------

:func:`~nvalchemi.training.distributed.all_reduce_flags` collects one flag per
rank into a world-sized vector that reads the same everywhere: each rank raises
its own entry and a ``MAX`` all-reduce merges them, on the device
:func:`~nvalchemi.distributed.collective_device` picks for the backend. A
strategy uses it to turn a verdict only one rank can see, such as an empty data
shard, into a refusal every rank raises together, before any rank reaches a
collective its peers would block on.

.. currentmodule:: nvalchemi.training.distributed

.. autosummary::
   :toctree: generated
   :nosignatures:

   all_reduce_flags
