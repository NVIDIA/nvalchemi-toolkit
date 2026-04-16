.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _dynamics-api:

=================
API Reference
=================

Core classes
------------

.. currentmodule:: nvalchemi.dynamics

.. autosummary::
   :toctree: _generated
   :nosignatures:

   BaseDynamics
   DemoDynamics
   FusedStage
   DistributedPipeline

Protocols and enums
-------------------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   DynamicsStage

Convergence
-----------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ConvergenceHook

Hooks
-----

.. currentmodule:: nvalchemi.dynamics.hooks

.. autosummary::
   :toctree: _generated
   :nosignatures:

   ConvergedSnapshotHook
   EnergyDriftMonitorHook
   FreezeAtomsHook
   LoggingHook
   MaxForceClampHook
   NaNDetectorHook
   ProfilerHook
   SnapshotHook

General-purpose hooks (:class:`~nvalchemi.hooks.NeighborListHook`,
:class:`~nvalchemi.hooks.BiasedPotentialHook`,
:class:`~nvalchemi.hooks.WrapPeriodicHook`) and the core hook
protocol are documented in :ref:`hooks-api`.

Data sinks
----------

.. currentmodule:: nvalchemi.dynamics

.. autosummary::
   :toctree: _generated
   :nosignatures:

   DataSink
   GPUBuffer
   HostMemory
   ZarrData

Sampling
--------

.. currentmodule:: nvalchemi.dynamics

.. autosummary::
   :toctree: _generated
   :nosignatures:

   SizeAwareSampler
