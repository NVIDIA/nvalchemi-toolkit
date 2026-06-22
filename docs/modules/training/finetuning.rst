.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _training-finetuning-api:

Fine-tuning API
===============

Registration-time helpers for adapting pretrained models before optimizer
construction.

.. seealso::

   - **User guide**: :ref:`finetuning_guide`
   - **Training strategy API**: :ref:`training-strategy-api`
   - **Training update hooks**: :ref:`training-update-hooks`


Strategy
--------

.. currentmodule:: nvalchemi.training

.. autosummary::
   :toctree: generated
   :nosignatures:

   FineTuningStrategy
   FineTuningStrategy.from_pretrained_checkpoint
   FineTuningStrategy.load_checkpoint


Checkpoint constructors
~~~~~~~~~~~~~~~~~~~~~~~

Use ``FineTuningStrategy.load_checkpoint(...)`` to resume an interrupted
fine-tuning strategy with its saved optimizer state, scheduler state, counters,
checkpointable hook state, and serialized fine-tuning configuration.

Use ``FineTuningStrategy.from_pretrained_checkpoint(...)`` to start a new
fine-tuning strategy from a model stored in an existing nvalchemi checkpoint.
The complete checkpoint model set is used as initialization. Single-model
checkpoints become a single-model strategy input, while multi-model checkpoints
preserve their named model mapping. Source optimizer classes, optimizer state,
scheduler state, hooks, counters, epoch/step limits, losses, and validation
settings do not carry over unless a future API exposes that reuse as an
explicit opt-in.


Hooks
-----

Fine-tuning hooks run when registered on a training workflow. They do not own
``backward()`` or optimizer-step behavior; use :ref:`training-update-hooks` for
batch-update policies such as mixed precision or gradient clipping.

.. currentmodule:: nvalchemi.training.hooks

.. autosummary::
   :toctree: generated
   :nosignatures:

   ModulePatchHook
   TrainableParameterHook
