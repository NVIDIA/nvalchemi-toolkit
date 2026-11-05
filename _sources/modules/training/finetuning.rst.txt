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

Use ``FineTuningStrategy.load_checkpoint(...)`` to resume an interrupted run
with saved optimizer state, scheduler state, counters, and serialized
fine-tuning configuration. Use ``FineTuningStrategy.from_pretrained_checkpoint(...)``
to start a new fine-tuning run whose model weights are initialized from an
existing checkpoint; optimizer state, hooks, and counters do not carry over.


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
   FineTuningSummaryHook
   BaseFingerprintHook

**ModulePatchHook**

.. dataclass-table:: nvalchemi.training.hooks.ModulePatchHook

**TrainableParameterHook**

.. dataclass-table:: nvalchemi.training.hooks.TrainableParameterHook

**FineTuningSummaryHook**

.. dataclass-table:: nvalchemi.training.hooks.FineTuningSummaryHook

**BaseFingerprintHook**

.. dataclass-table:: nvalchemi.training.hooks.BaseFingerprintHook

**LoRAHook**

.. currentmodule:: nvalchemi.training.peft.lora_hook

.. autosummary::
   :toctree: generated
   :nosignatures:

   LoRAHook

.. dataclass-table:: nvalchemi.training.peft.lora_hook.LoRAHook


PEFT helpers
------------

Configuration and utility APIs for parameter-efficient fine-tuning workflows.
LoRA is the only PEFT method currently supported. Pass
``peft_config=LoRAConfig(...)`` to ``FineTuningStrategy`` to train LoRA adapters
and use ``load_peft_checkpoint_into_model(...)`` to load a trainable-state PEFT
checkpoint into a compatible base model.

.. currentmodule:: nvalchemi.training.peft

.. autosummary::
   :toctree: generated
   :nosignatures:

   PeftMethodRegistration
   register_peft_method
   available_peft_methods
   load_peft_checkpoint_into_model

.. currentmodule:: nvalchemi.training.peft.lora

.. autosummary::
   :toctree: generated
   :nosignatures:

   LoRAConfig
   available_lora_wrappers
