.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _training-hooks-api:
.. _training-hooks:

========================
Hooks - Training Updates
========================

Training update hooks customize the backward and optimizer-step portions of a
training batch. Register bare update hooks on
:class:`~nvalchemi.training.strategy.TrainingStrategy`; the strategy folds them
into one :class:`~nvalchemi.training.hooks.TrainingUpdateOrchestrator`.

See :ref:`hooks-api` for the underlying protocol and dispatch semantics.

EMA model averaging
-------------------

:class:`~nvalchemi.training.hooks.EMAHook` maintains an
``AveragedModel`` for one model in ``ctx.models``. It updates after successful
optimizer steps and skips updates when an earlier update hook vetoes
``TrainingStage.DO_OPTIMIZER_STEP``.

.. code-block:: python

   from nvalchemi.training.hooks import EMAHook
   from nvalchemi.training.strategy import TrainingStrategy

   ema = EMAHook(model_key="main", decay=0.999)
   strategy = TrainingStrategy(..., hooks=[ema])

Update hook API
---------------

Concrete update hooks subclass
:class:`~nvalchemi.training.hooks.TrainingUpdateHook` and return
``tuple[bool, torch.Tensor | None]`` from ``__call__``. The boolean
participates in any-veto-wins decisions for ``BEFORE_BATCH`` and
``DO_OPTIMIZER_STEP``. The tensor is the loss value threaded through hooks
before the orchestrator calls ``backward()``.

Reference
---------

.. currentmodule:: nvalchemi.training.hooks

.. autosummary::
   :toctree: generated
   :nosignatures:

   EMAHook
   TrainingUpdateHook
   TrainingUpdateOrchestrator
