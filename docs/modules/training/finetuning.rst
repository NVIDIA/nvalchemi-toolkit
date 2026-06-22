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
