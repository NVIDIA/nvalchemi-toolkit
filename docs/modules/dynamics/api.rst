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

Dynamics-specific hooks are documented in :ref:`dynamics-hooks`.
General-purpose hooks (:class:`~nvalchemi.hooks.NeighborListHook`,
:class:`~nvalchemi.hooks.BiasedPotentialHook`,
:class:`~nvalchemi.hooks.WrapPeriodicHook`), the shared profiling hooks
(:class:`~nvalchemi.hooks.StageTimingHook`,
:class:`~nvalchemi.hooks.TorchProfilerHook`), and the core hook protocol
are documented in :ref:`hooks-api`.

Data sinks
----------

.. currentmodule:: nvalchemi.dynamics

.. autosummary::
   :toctree: _generated
   :nosignatures:

   DataSink
   ResizableSink
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
   OrderedStructureSampler
   StructureSource
   FitPolicy
   WithinBudget

Benchmarking
------------

:func:`~nvalchemi.dynamics.measure_throughput` times a propagator at steady
state: a discarded warmup window absorbs neighbor-list builds, lazy allocation,
and kernel compilation, then a timed window with the device synchronized at
both ends reports steps per second, atoms per second, and simulated nanoseconds
per day as a :class:`~nvalchemi.dynamics.ThroughputMetrics` record.

.. currentmodule:: nvalchemi.dynamics

.. autosummary::
   :toctree: _generated
   :nosignatures:

   measure_throughput
   ThroughputMetrics
