.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _dynamics-paths:

======================
Reaction Paths and NEB
======================

- **User guide**: :ref:`dynamics_paths_guide` --- task-oriented walkthrough
  covering the high-level ``NEB`` strategy and the low-level
  ``FusedStage`` + hooks recipe.

This page is the API reference for :mod:`nvalchemi.dynamics.paths`.

Strategy
--------

.. currentmodule:: nvalchemi.dynamics.paths

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

Spring and method configuration
--------------------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   NEBMethod
   SpringConfig
   ConstantSpringConfig
   SpringContext

Hooks
-----

.. currentmodule:: nvalchemi.dynamics.paths.hooks

.. autosummary::
   :toctree: _generated
   :nosignatures:

   PathEnergyStatsHook
   PathDiagnosticsHook

.. currentmodule:: nvalchemi.dynamics.paths.neb.hooks

.. autosummary::
   :toctree: _generated
   :nosignatures:

   NEBForceHook
   ClimbingImageSelectionHook

Fixing atoms during NEB (``endpoint_mode="fixed"`` or ``fixed_atom_indices``)
is enforced by the general-purpose
:class:`~nvalchemi.dynamics.hooks.FreezeAtomsHook`, documented in
:ref:`dynamics-hooks`.
