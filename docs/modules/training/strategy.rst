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
batch on ``devices[0]`` and hands it to every model, so a per-model list has to
name the same device throughout and one naming more than one distinct device is
refused at run time; the longer form is a spelling that lets a caller enumerate
the models it places, not a way to span devices. Names are compared as written,
which keeps an index-less ``cuda`` — whichever device the process has made
current — apart from ``cuda:0``. A single-model run is unaffected, and a
:class:`~nvalchemi.training.hooks.DDPHook` on the NCCL backend collapses
``devices`` to this rank's own device before the check runs.


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


Parallelism helpers
-------------------

A parallelism wrapper stands in for the model a strategy holds, so anything
reaching for the model's own surface — its ``model_config``, a
:class:`~nvalchemi.models.base.BaseModelMixin` method, an unwrapped
``state_dict`` — has to step through it first. :func:`unwrap_model` returns the
module behind the wrapper, or the model itself when nothing wraps it, and
recognizes a wrapper by the ``module`` attribute it publishes rather than by its
class, so a hand-rolled wrapper unwraps exactly as
:class:`~torch.nn.parallel.DistributedDataParallel` does.

.. autosummary::
   :toctree: generated
   :nosignatures:

   unwrap_model
