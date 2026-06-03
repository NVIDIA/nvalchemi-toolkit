.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _losses-api:

=======================
Losses — Training Terms
=======================

Composable, tensor-first loss functions for MLIP training.

.. seealso::

   - **User guide**: :ref:`losses_guide` — conceptual overview, usage
     patterns, and how to write your own loss term.


Leaf and composition
--------------------

Leaf losses subclass :class:`~nvalchemi.training.BaseLossFunction`;
compositions use :class:`~nvalchemi.training.ComposedLossFunction` and
return a :class:`~nvalchemi.training.ComposedLossOutput`.

.. currentmodule:: nvalchemi.training

.. autosummary::
   :toctree: generated
   :nosignatures:

   BaseLossFunction
   ComposedLossFunction
   ComposedLossOutput
   LossWeightSchedule


Concrete losses
---------------

Built-in leaf losses for common quantum-chemistry targets.

.. autosummary::
   :toctree: generated
   :nosignatures:

   EnergyLoss
   ForceLoss
   StressLoss


Weight schedules
----------------

Pydantic ``frozen`` models satisfying :class:`~nvalchemi.training.LossWeightSchedule`.

.. autosummary::
   :toctree: generated
   :nosignatures:

   ConstantWeight
   LinearWeight
   CosineWeight
   PiecewiseWeight


Reduction helpers
-----------------

Per-graph reduction helpers — scatter reductions (``V ... → B ...``)
and matrix reductions (``B ... m n → B ...``) — importable for use in
custom losses.

.. currentmodule:: nvalchemi.training.losses.reductions

.. autosummary::
   :toctree: generated
   :nosignatures:

   per_graph_sum
   per_graph_mean
   frobenius_mse
