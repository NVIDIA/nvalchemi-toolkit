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

Mixed precision
---------------

:class:`~nvalchemi.training.hooks.MixedPrecisionHook` enables
``torch.amp.autocast`` for forward and loss computation and uses
``torch.amp.GradScaler`` when ``precision`` is ``torch.float16``. The
``precision`` argument is required so configs must choose one of the supported
policies explicitly:

.. code-block:: python

   import torch

   from nvalchemi.training.hooks import MixedPrecisionHook
   from nvalchemi.training.strategy import TrainingStrategy

   strategy = TrainingStrategy(
       ...,
       hooks=[MixedPrecisionHook(precision=torch.bfloat16)],
   )

``precision`` accepts the dtype objects ``torch.float32``, ``torch.bfloat16``,
and ``torch.float16`` or the canonical strings ``"float32"``, ``"bfloat16"``,
and ``"float16"``.

The policies are:

* ``torch.float32``: autocast is disabled and no scaler is used.
* ``torch.bfloat16``: eligible ops run under bf16 autocast and no scaler is used.
* ``torch.float16``: eligible forward/loss ops run under fp16 autocast, the hook
  scales the loss before backward, unscales gradients immediately before an
  optimizer step proceeds, and lets the scaler skip steps with ``inf`` or
  ``nan`` gradients.

Gradient accumulation
---------------------

With fp16 gradient scaling, accumulated gradients stay scaled until the
effective batch is ready to step. A gradient-accumulation update hook should
veto ``TrainingStage.DO_OPTIMIZER_STEP`` on intermediate microbatches; that
suppresses AMP unscale, scaler step, and scaler update for those batches. When
the accumulation window is complete, the optimizer-step stage proceeds and
``MixedPrecisionHook`` unscales once per optimizer just before stepping.

Update hook API
---------------

Concrete update hooks subclass
:class:`~nvalchemi.training.hooks.TrainingUpdateHook` and return
``tuple[bool, torch.Tensor]`` from ``__call__``. The boolean participates in
any-veto-wins decisions for ``BEFORE_BATCH`` and ``DO_OPTIMIZER_STEP``. The
tensor is the loss value threaded through hooks before the orchestrator calls
``backward()``.

Reference
---------

.. currentmodule:: nvalchemi.training.hooks

.. autosummary::
   :toctree: generated
   :nosignatures:

   EMAHook
   MixedPrecisionHook
   TrainingUpdateHook
   TrainingUpdateOrchestrator
