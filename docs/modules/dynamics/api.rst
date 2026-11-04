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
   DynamicsStrategy
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

Constants
---------

.. autosummary::
   :toctree: _generated
   :nosignatures:

   KB_EV

Integrators
-----------

The algorithms are described in :ref:`dynamics-methods`.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   NVE
   NVTLangevin
   NVTNoseHoover
   NPT
   NPH

Optimizers
----------

``FIRE`` and ``FIREVariableCell`` are deprecated; see :ref:`dynamics-methods`.

.. autosummary::
   :toctree: _generated
   :nosignatures:

   FIRE2
   FIRE2VariableCell
   LBFGS
   LBFGSVariableCell

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
