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

   Hook
   HookStageEnum

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

   BiasedPotentialHook
   EnergyDriftMonitorHook
   LoggingHook
   MaxForceClampHook
   NaNDetectorHook
   ProfilerHook
   SnapshotHook
   WrapPeriodicHook

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

.. autosummary::
   :toctree: _generated
   :nosignatures:

   SizeAwareSampler
