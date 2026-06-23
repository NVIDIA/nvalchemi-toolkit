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

A typical training loss is a composition of tensor-first leaf losses. The
composition routes prediction/target mappings into each leaf, applies the
configured weights, and returns a structured output whose ``total_loss`` is the
scalar used for backpropagation:

.. code-block:: python

   from nvalchemi.training import ComposedLossFunction, EnergyMSELoss, ForceMSELoss

   loss_fn = ComposedLossFunction(
       components=[
           EnergyMSELoss(),
           ForceMSELoss(),
       ],
       weights=[1.0, 10.0],
   )

   out = loss_fn(predictions, targets, batch=batch, step=step, epoch=epoch)
   out["total_loss"].backward()

   for name, value in out["per_component_unweighted"].items():
       logger.info("%s raw loss: %s", name, value.detach())



Dtype alignment
---------------

Leaf losses default to ``dtype_policy="strict"``, which preserves prediction
and target tensors and raises on dtype mismatch during validation. Built-in
leaves also accept ``"prediction_to_target"`` and ``"target_to_prediction"`` to
cast one tensor before validation. ``ComposedLossFunction(dtype_policy=...)``
provides the same policy as a call-time default for strict leaves without
mutating reusable component instances. For compositions built with operator
sugar, set ``loss_fn.dtype_policy`` after construction.

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
   ReductionContext
   ComposedLossFunction
   ComposedLossOutput
   LossWeightSchedule


Concrete losses
---------------

Built-in leaf losses for common quantum-chemistry targets.

.. autosummary::
   :toctree: generated
   :nosignatures:

   EnergyMSELoss
   EnergyMAELoss
   ForceMSELoss
   ForceL2NormLoss
   StressMSELoss


Weight schedules
----------------

Pydantic ``frozen`` models satisfying :class:`~nvalchemi.training.LossWeightSchedule`.
Custom schedules may also satisfy the protocol directly. For strategy checkpoint
round-trips, implement ``to_spec()`` returning a ``BaseSpec``. The built-in
Pydantic schedule base provides this method from ``model_dump()``.

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
