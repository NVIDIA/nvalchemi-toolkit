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
   LoRAFineTuningStrategy
   LoRAFineTuningStrategy.save_checkpoint
   LoRAFineTuningStrategy.save_adapter
   LoRAFineTuningStrategy.load_adapter_into_model
   LoRAFineTuningStrategy.merge_model_inplace

Use ``FineTuningStrategy.load_checkpoint(...)`` to resume an interrupted run
with saved optimizer state, scheduler state, counters, and serialized
fine-tuning configuration. Use ``FineTuningStrategy.from_pretrained_checkpoint(...)``
to start a new fine-tuning run whose model weights are initialized from an
existing checkpoint; optimizer state, hooks, and counters do not carry over.
Use ``LoRAFineTuningStrategy`` for adapter-based fine-tuning where the base
model is frozen by default and only LoRA adapters, module patches, and selected
extra parameters are trainable. See :ref:`finetuning_guide` for patterns,
checkpointing, and adapter export examples.


Hooks
-----

Registration-time hooks that adapt the model tree and optimizer parameter set
before training starts. They do not own ``backward()`` or optimizer-step
behavior; use :ref:`training-update-hooks` for batch-update policies.

.. currentmodule:: nvalchemi.training.hooks

.. autosummary::
   :toctree: generated
   :nosignatures:

   ModulePatchHook
   TrainableParameterHook
   LoRAApplyHook

**ModulePatchHook**

.. dataclass-table:: nvalchemi.training.hooks.ModulePatchHook

**TrainableParameterHook**

.. dataclass-table:: nvalchemi.training.hooks.TrainableParameterHook

**LoRAApplyHook**

.. dataclass-table:: nvalchemi.training.hooks.LoRAApplyHook
