.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _enhanced-sampling-api:

Enhanced sampling API
=====================

The enhanced sampling API adds biases and replica exchange to a dynamics run.
:class:`EnhancedSampling` builds the engine. A bias is an additive potential
that the engine sums against the unmodified model output.

.. seealso::

   :doc:`The enhanced sampling user guide </userguide/enhanced_sampling>` for
   the workflow, choice of bias, and worked examples.

.. currentmodule:: nvalchemi.enhanced_sampling

Engine
------

.. autosummary::
   :toctree: generated
   :nosignatures:

   EnhancedSampling
   ThermodynamicState
   ReplicaExchange

Bias contract and hooks
-----------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   ConservativeBias
   AdaptivePotentialMixin
   BiasHook
   EpochCommitHook
   WalkerIdentityHook

Built-in biases
---------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   HarmonicUmbrellaBias
   UpperWall
   LowerWall
   FlatBottomRestraint
   WellTemperedMetaDynamicsBias
   RMSDMetaDynamicsBias
   AdaptiveBiasingForce

Collective variables
--------------------

.. autosummary::
   :toctree: generated
   :nosignatures:

   pair_distance
   pair_displacement
   periodic_difference
