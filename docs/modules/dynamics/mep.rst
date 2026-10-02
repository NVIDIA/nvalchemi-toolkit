.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _dynamics-mep:

======================
Reaction Paths and NEB
======================

- **User guide**: :ref:`dynamics_mep_guide` --- task-oriented walkthrough
  covering the high-level ``NEB`` strategy and the low-level
  ``FusedStage`` + hooks recipe.

The :mod:`nvalchemi.dynamics.mep` package provides tools for calculating
minimum energy paths, including path construction, nudged elastic band (NEB)
optimization, and hooks for monitoring and controlling the calculation.

Strategy
--------

.. currentmodule:: nvalchemi.dynamics.mep

.. autosummary::
   :toctree: _generated
   :nosignatures:

   NEB
   ClimbingImageConfig

Path construction
------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   interpolate_paths
   IDPPModel
   prepare_idpp_targets
   validate_paths

Position alignment
------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   align_batch_positions

.. autoclass:: PositionAlignment

Spring and method configuration
--------------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   NEBMethod
   TorchNEBMethod
   SpringConfig
   ConstantSpringConfig
   SpringContext

Hooks
-----

.. currentmodule:: nvalchemi.dynamics.mep.hooks

.. autosummary::
   :toctree: _generated
   :nosignatures:

   PathEnergyStatsHook
   PathEnergyStats
   PathDiagnosticsHook
   PathDiagnostics

.. autosummary::
   :toctree: _generated
   :nosignatures:

   NEBForceHook
   ClimbingImageSelectionHook

Fixing atoms during NEB (``endpoint_mode="fixed"`` or ``fixed_atom_indices``)
is enforced by the general-purpose
:class:`~nvalchemi.dynamics.hooks.FreezeAtomsHook`, documented in
:ref:`dynamics-hooks`.
