ALCHEMI Toolkit Examples
========================

This directory contains worked examples organized by complexity tier.
Each example is a self-contained Python script that runs end-to-end
and is compatible with Sphinx-gallery.

Tiers
-----

.. toctree::
   :maxdepth: 1

.. rubric:: Basic

Introduces :class:`~nvalchemi.data.AtomicData`,
:class:`~nvalchemi.data.Batch`, the FIRE geometry optimizer, NVE and NVT
integrators, and the built-in hooks (NeighborListHook, WrapPeriodicHook,
LoggingHook).  Suitable for users coming from ASE.

.. rubric:: Intermediate

Covers multi-stage pipelines (:class:`~nvalchemi.dynamics.FusedStage`),
trajectory I/O with :class:`~nvalchemi.dynamics.ZarrData`, pressure-controlled
NPT dynamics, inflight batching with
:class:`~nvalchemi.dynamics.SizeAwareSampler`, and defensive MD patterns
(NaNDetectorHook, MaxForceClampHook, EnergyDriftMonitorHook, ProfilerHook).

.. rubric:: Advanced

Shows framework extension: writing custom hooks, multi-criteria convergence
with ``custom_op``, biased sampling, wrapping a MACE foundation model, and
subclassing :class:`~nvalchemi.dynamics.base.BaseDynamics` to implement a new
integrator.

.. rubric:: Distributed

Multi-GPU pipelines via :class:`~nvalchemi.dynamics.DistributedPipeline`.
These examples require ``torchrun`` and are **not executed** during the
Sphinx build.  See ``distributed/README.rst`` for run instructions.
