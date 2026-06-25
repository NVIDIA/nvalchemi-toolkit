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
