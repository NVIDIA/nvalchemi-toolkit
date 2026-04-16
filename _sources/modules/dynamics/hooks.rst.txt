.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _dynamics-hooks:

================================
Dynamics Hooks — Stages & Usage
================================

This page covers hook behaviour specific to dynamics simulations.
For the general hook protocol, context, and registry see
:ref:`hooks-api`.

.. seealso::

   - **User guide**: :ref:`hooks_guide` — conceptual overview, writing
     custom hooks, and composing hook pipelines.
   - **Core framework**: :ref:`hooks-api` — the ``Hook`` protocol,
     ``HookContext``, and ``HookRegistryMixin``.


DynamicsStage
--------------

:class:`~nvalchemi.dynamics.base.DynamicsStage` enumerates the nine
hook-firing points within a single dynamics step:

.. graphviz::
   :caption: DynamicsStage hook firing points within a single step.

   digraph dynamics_stages {
       rankdir=TB
       compound=true
       fontname="Helvetica"
       node [fontname="Helvetica" fontsize=11 shape=box style="rounded,filled" fillcolor="#dce6f1"]
       edge [fontname="Helvetica" fontsize=10 style=bold]

       BEFORE_STEP [label="BEFORE_STEP" fillcolor="#f9e2ae"]

       subgraph cluster_step {
           label="step body"
           style=rounded
           color="#4a90d9"
           fontcolor="#4a90d9"
           fontname="Helvetica"
           fontsize=12

           BEFORE_PRE_UPDATE  [label="BEFORE_PRE_UPDATE"]
           pre_update         [label="pre_update()" fillcolor="#eeeeee"]
           AFTER_PRE_UPDATE   [label="AFTER_PRE_UPDATE"]

           BEFORE_COMPUTE     [label="BEFORE_COMPUTE"]
           compute            [label="compute()" fillcolor="#eeeeee"]
           AFTER_COMPUTE      [label="AFTER_COMPUTE"]

           BEFORE_POST_UPDATE [label="BEFORE_POST_UPDATE"]
           post_update        [label="post_update()" fillcolor="#eeeeee"]
           AFTER_POST_UPDATE  [label="AFTER_POST_UPDATE"]

           BEFORE_PRE_UPDATE -> pre_update -> AFTER_PRE_UPDATE
           AFTER_PRE_UPDATE -> BEFORE_COMPUTE
           BEFORE_COMPUTE -> compute -> AFTER_COMPUTE
           AFTER_COMPUTE -> BEFORE_POST_UPDATE
           BEFORE_POST_UPDATE -> post_update -> AFTER_POST_UPDATE
       }

       AFTER_STEP  [label="AFTER_STEP" fillcolor="#f9e2ae"]
       ON_CONVERGE [label="ON_CONVERGE\n(if converged)" fillcolor="#f9e2ae"]

       BEFORE_STEP -> BEFORE_PRE_UPDATE [lhead=cluster_step]
       AFTER_POST_UPDATE -> AFTER_STEP [ltail=cluster_step]
       AFTER_STEP -> ON_CONVERGE [style=dashed]
   }

.. list-table:: Dynamics stages reference
   :widths: 30 10 60
   :header-rows: 1

   * - Stage
     - Value
     - When it fires
   * - ``BEFORE_STEP``
     - 0
     - Very start of each step, before any operations.
   * - ``BEFORE_PRE_UPDATE``
     - 1
     - Before the first integrator half-step (positions).
   * - ``AFTER_PRE_UPDATE``
     - 2
     - After positions are updated, before the forward pass.
   * - ``BEFORE_COMPUTE``
     - 3
     - Before the model forward pass.
   * - ``AFTER_COMPUTE``
     - 4
     - After forces/energy are written to the batch.
   * - ``BEFORE_POST_UPDATE``
     - 5
     - Before the second integrator half-step (velocities).
   * - ``AFTER_POST_UPDATE``
     - 6
     - After velocities are updated.
   * - ``AFTER_STEP``
     - 7
     - Very end of the step, after all operations.
   * - ``ON_CONVERGE``
     - 8
     - Only when the convergence hook detects converged samples.


Built-in dynamics hooks
------------------------

The ``nvalchemi.dynamics.hooks`` package ships production-ready hooks
organized into four categories. General-purpose hooks
(:class:`~nvalchemi.hooks.NeighborListHook`,
:class:`~nvalchemi.hooks.BiasedPotentialHook`,
:class:`~nvalchemi.hooks.WrapPeriodicHook`) are documented in
:ref:`hooks-api`.

Observer hooks (read-only, fire at ``AFTER_STEP``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These hooks **do not** modify the batch — they record, log, or
monitor simulation state.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Hook
     - Purpose
   * - :class:`~nvalchemi.dynamics.hooks.LoggingHook`
     - Log scalar observables (energy, fmax, temperature) to
       ``loguru``, CSV, TensorBoard, or a custom backend.
   * - :class:`~nvalchemi.dynamics.hooks.SnapshotHook`
     - Write the full batch state to a
       :class:`~nvalchemi.dynamics.DataSink`
       (``GPUBuffer``, ``HostMemory``, or ``ZarrData``).
   * - :class:`~nvalchemi.dynamics.hooks.ConvergedSnapshotHook`
     - Write only newly converged samples to a
       :class:`~nvalchemi.dynamics.DataSink`. Fires at
       ``ON_CONVERGE``; ideal for persisting optimized structures
       from :class:`~nvalchemi.dynamics.FusedStage` pipelines.
   * - :class:`~nvalchemi.dynamics.hooks.EnergyDriftMonitorHook`
     - Track cumulative energy drift in NVE runs; warn or halt on
       excessive drift.
   * - :class:`~nvalchemi.dynamics.hooks.ProfilerHook`
     - Instrument steps with NVTX ranges and wall-clock timing for
       Nsight Systems profiling. Fires at multiple stages via
       ``_runs_on_stage`` and uses ``plum`` dispatch to support
       dynamics and custom workflows.

Post-compute hooks (modify batch, fire at ``AFTER_COMPUTE``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These hooks modify the batch **after** the model forward pass and
**before** the velocity update.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Hook
     - Purpose
   * - :class:`~nvalchemi.dynamics.hooks.NaNDetectorHook`
     - Detect NaN/Inf in forces and energy; raise with
       diagnostic info (affected graph indices, step count).
   * - :class:`~nvalchemi.dynamics.hooks.MaxForceClampHook`
     - Clamp per-atom force magnitudes to a safe maximum,
       preserving force direction. Prevents numerical explosions.

Constraint hooks (modify batch, fire at ``BEFORE_PRE_UPDATE``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These hooks enforce geometric constraints across integration steps.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Hook
     - Purpose
   * - :class:`~nvalchemi.dynamics.hooks.FreezeAtomsHook`
     - Freeze atoms by category (e.g. substrate, boundary). Snapshots
       positions at ``BEFORE_PRE_UPDATE`` and restores them (with
       zeroed velocities) at ``AFTER_POST_UPDATE``.


Usage examples
--------------

Logging to CSV every 100 steps
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nvalchemi.dynamics.hooks import LoggingHook

   hook = LoggingHook(frequency=100, backend="csv", log_path="md_log.csv")
   dynamics = DemoDynamics(model=model, n_steps=10_000, dt=0.5, hooks=[hook])
   dynamics.run(batch)

Recording trajectories to a data sink
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nvalchemi.dynamics.hooks import SnapshotHook
   from nvalchemi.dynamics import HostMemory

   sink = HostMemory(capacity=10_000)
   hook = SnapshotHook(sink=sink, frequency=10)
   dynamics = DemoDynamics(model=model, n_steps=1_000, dt=0.5, hooks=[hook])
   dynamics.run(batch)   # 100 snapshots
   trajectory = sink.read()

Safety: NaN detection + force clamping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nvalchemi.dynamics.hooks import MaxForceClampHook, NaNDetectorHook

   dynamics = DemoDynamics(
       model=model,
       dt=0.5,
       hooks=[
           # Clamp first, then check — both fire at AFTER_COMPUTE
           # in registration order.
           MaxForceClampHook(max_force=50.0, log_clamps=True),
           NaNDetectorHook(extra_keys=["stress"]),
       ],
   )

Enhanced sampling with a bias potential
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nvalchemi.hooks import BiasedPotentialHook
   from nvalchemi.dynamics.base import DynamicsStage

   def harmonic_restraint(batch):
       """Restrain center of mass to the origin."""
       k = 10.0  # eV/Å²
       com = batch.positions.mean(dim=0, keepdim=True)
       bias_energy = 0.5 * k * (com ** 2).sum().unsqueeze(0).unsqueeze(0)
       bias_forces = -k * com.expand_as(batch.positions) / batch.num_nodes
       return bias_energy, bias_forces

   hook = BiasedPotentialHook(bias_fn=harmonic_restraint, stage=DynamicsStage.AFTER_COMPUTE)
   dynamics = DemoDynamics(model=model, dt=0.5, hooks=[hook])

Profiling with Nsight Systems
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nvalchemi.dynamics.hooks import ProfilerHook

   hook = ProfilerHook(enable_nvtx=True, enable_timer=True, frequency=10)
   dynamics = DemoDynamics(model=model, n_steps=1_000, dt=0.5, hooks=[hook])

   # Run under: nsys profile python my_script.py
   dynamics.run(batch)

NVE energy drift monitoring
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nvalchemi.dynamics.hooks import EnergyDriftMonitorHook

   hook = EnergyDriftMonitorHook(
       threshold=1e-5,
       metric="per_atom_per_step",
       action="raise",    # or "warn" for production
       frequency=100,
   )
   dynamics = DemoDynamics(model=model, dt=0.5, hooks=[hook])


Hooks inside ``FusedStage``
---------------------------

When hooks are registered on sub-stage dynamics inside a
:class:`~nvalchemi.dynamics.FusedStage`, their firing semantics differ
slightly from standalone execution:

**Fired on each sub-stage:**

- ``BEFORE_STEP``, ``AFTER_COMPUTE``, ``BEFORE_PRE_UPDATE``,
  ``AFTER_POST_UPDATE``, ``AFTER_STEP``, ``ON_CONVERGE``

**Not fired on sub-stages** (because the forward pass is shared):

- ``BEFORE_COMPUTE``, ``AFTER_PRE_UPDATE``, ``BEFORE_POST_UPDATE``

This means safety hooks (``NaNDetectorHook``, ``MaxForceClampHook``)
and observer hooks (``LoggingHook``, ``SnapshotHook``) work as expected
inside fused stages, since they fire at ``AFTER_COMPUTE`` or
``AFTER_STEP``.

Hook ordering inside a fused step:

.. graphviz::
   :caption: Hook ordering inside a single ``FusedStage.step()``.

   digraph fused_hook_order {
       rankdir=TB
       compound=true
       fontname="Helvetica"
       node [fontname="Helvetica" fontsize=11 shape=box style="rounded,filled" fillcolor="#dce6f1"]
       edge [fontname="Helvetica" fontsize=10 style=bold]

       subgraph cluster_before {
           label="for each sub-stage"
           style=dashed
           color="#4a90d9"
           fontcolor="#4a90d9"
           fontname="Helvetica"
           fontsize=10
           BEFORE_STEP [label="BEFORE_STEP hooks"]
       }

       compute [label="single compute()" fillcolor="#f9e2ae"]

       subgraph cluster_after_compute {
           label="for each sub-stage"
           style=dashed
           color="#4a90d9"
           fontcolor="#4a90d9"
           fontname="Helvetica"
           fontsize=10
           AFTER_COMPUTE [label="AFTER_COMPUTE hooks"]
       }

       subgraph cluster_update {
           label="for each sub-stage"
           style=dashed
           color="#4a90d9"
           fontcolor="#4a90d9"
           fontname="Helvetica"
           fontsize=10
           BEFORE_PRE [label="BEFORE_PRE_UPDATE hooks"]
           masked     [label="masked_update()\n(if samples match status)" fillcolor="#eeeeee"]
           AFTER_POST [label="AFTER_POST_UPDATE hooks"]
           BEFORE_PRE -> masked -> AFTER_POST
       }

       subgraph cluster_after_step {
           label="for each sub-stage"
           style=dashed
           color="#4a90d9"
           fontcolor="#4a90d9"
           fontname="Helvetica"
           fontsize=10
           AFTER_STEP [label="AFTER_STEP hooks"]
       }

       subgraph cluster_converge {
           label="for each sub-stage"
           style=dashed
           color="#4a90d9"
           fontcolor="#4a90d9"
           fontname="Helvetica"
           fontsize=10
           conv_check  [label="convergence check" fillcolor="#eeeeee"]
           ON_CONVERGE [label="ON_CONVERGE hooks" fillcolor="#f9e2ae"]
           conv_check -> ON_CONVERGE [style=dashed label="if converged"]
       }

       BEFORE_STEP -> compute
       compute -> AFTER_COMPUTE
       AFTER_COMPUTE -> BEFORE_PRE
       AFTER_POST -> AFTER_STEP
       AFTER_STEP -> conv_check
   }
