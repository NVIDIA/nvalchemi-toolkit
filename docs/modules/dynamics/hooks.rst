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
     ``HookContext``/``DynamicsContext``, and ``HookRegistryMixin``.


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

The ``nvalchemi.dynamics.hooks`` package ships production-ready hooks in three
categories. :class:`~nvalchemi.hooks.NeighborListHook`,
:class:`~nvalchemi.hooks.BiasedPotentialHook`, and
:class:`~nvalchemi.hooks.WrapPeriodicHook` are general-purpose hooks documented
in :ref:`hooks-api`.

Observer hooks
~~~~~~~~~~~~~~

Observer hooks fire at ``AFTER_STEP`` and do not modify the batch.

LoggingHook
...........

:class:`~nvalchemi.dynamics.hooks.LoggingHook` writes per-step scalar
observables to a backend. The default scalars are energy (per atom), ``fmax``
(maximum force component across all atoms), temperature (derived from kinetic
energy when velocities are present), and ``converged_fraction`` (fraction of
samples that have met the convergence criterion).

``backend`` selects the output destination:

- ``"loguru"`` (default) — emits a formatted line to the loguru logger. Use
  for live console monitoring during interactive or short runs.
- ``"csv"`` — writes one row per step to ``log_path``. Use when you need
  per-step data for post-run analysis in Python or a spreadsheet.
- ``"tensorboard"`` — writes scalar events to ``log_path`` as a TensorBoard
  event file. Use when comparing scalar trends across experiments.
- A callable ``fn(scalars: dict) -> None`` — routes each snapshot to a custom
  backend, such as W&B or MLflow.

``frequency`` throttles writes to every N steps. For long runs,
``frequency=10`` or higher keeps output manageable without losing trends.

SnapshotHook
............

:class:`~nvalchemi.dynamics.hooks.SnapshotHook` writes the full batch state
— positions, velocities, forces, energy, cell, and atom types — to a
:class:`~nvalchemi.dynamics.DataSink` at a specified frequency.

``sink`` accepts one of three DataSink types:

- :class:`~nvalchemi.dynamics.GPUBuffer` — stores batches in GPU memory. Fastest
  write path; capacity bounded by GPU memory.
- :class:`~nvalchemi.dynamics.HostMemory` — stores in pinned CPU memory.
  Slightly slower; larger capacity and works without GPU.
- :class:`~nvalchemi.dynamics.ZarrData` — streams to disk in Zarr format.
  Unbounded capacity; suitable for long trajectories and persistent storage.

After the run, call ``sink.read()`` to retrieve the accumulated trajectory as a
:class:`~nvalchemi.data.Batch`. Use this hook when you need full atomic-detail
trajectories for analysis, visualization, or continuation from a specific frame.

ConvergedSnapshotHook
.....................

:class:`~nvalchemi.dynamics.hooks.ConvergedSnapshotHook` writes only
newly-converged samples at ``ON_CONVERGE`` — once per sample, exactly when
convergence is detected — rather than periodically. The same DataSink types
apply as for :class:`~nvalchemi.dynamics.hooks.SnapshotHook`.

This hook is designed for :class:`~nvalchemi.dynamics.FusedStage` pipelines
where samples converge at different steps. A periodic snapshot would produce
ragged data or miss samples; this hook captures each sample exactly once. Call
``sink.read()`` after the run to collect all converged structures.

EnergyDriftMonitorHook
......................

:class:`~nvalchemi.dynamics.hooks.EnergyDriftMonitorHook` tracks cumulative
energy drift in NVE (constant-energy) simulations and takes a configurable
action when drift exceeds a threshold.

Key arguments:

- ``threshold`` — allowed drift, in the model's energy output units.
- ``metric`` — how drift is measured. ``"per_atom_per_step"`` normalises by
  system size and simulation length, making the threshold transferable across
  systems and time steps.
- ``action`` — ``"raise"`` (default) halts the run; ``"warn"`` logs and
  continues. Use ``"warn"`` in production, ``"raise"`` during model
  validation.
- ``frequency`` — check every N steps. Checking every step is accurate but
  adds overhead for large batches; ``frequency=100`` is typical.

StageTimingHook and TorchProfilerHook are described in :ref:`hooks-api`.

Post-compute hooks
~~~~~~~~~~~~~~~~~~

Post-compute hooks fire at ``AFTER_COMPUTE``, after forces and energy are
written to the batch but before the velocity update. They may modify the batch.

NaNDetectorHook
...............

:class:`~nvalchemi.dynamics.hooks.NaNDetectorHook` checks energy and forces for
NaN or Inf values after the model forward pass. On detection it raises a
``RuntimeError`` that includes the affected graph indices and the current step
count so the offending sample can be identified.

``extra_keys`` extends the check to additional batch fields beyond energy and
forces. For models that output stress tensors, pass
``extra_keys=["stress"]``.

When used with :class:`~nvalchemi.dynamics.hooks.MaxForceClampHook`, register
the clamping hook first so the detector sees the clamped values and only catches
what clamping did not prevent.

MaxForceClampHook
.................

:class:`~nvalchemi.dynamics.hooks.MaxForceClampHook` rescales per-atom forces
whose magnitude exceeds ``max_force`` back to the threshold, preserving
direction. Energy is not modified.

``max_force`` is in the same units as the model's force output (typically
eV/Å). Set ``log_clamps=True`` to emit a loguru warning each time clamping
occurs, including which atoms were affected — useful during model development
to identify problem configurations.

Clamping prevents numerical blow-up from large forces in high-energy or
poorly-sampled configurations. It is a safety net, not a model fix: if
clamping fires frequently, the model has accuracy problems for those
structures.

Constraint hooks
~~~~~~~~~~~~~~~~

Constraint hooks enforce geometric constraints across integration steps. They
fire at both ``BEFORE_PRE_UPDATE`` (to snapshot positions) and
``AFTER_POST_UPDATE`` (to restore them).

FreezeAtomsHook
...............

:class:`~nvalchemi.dynamics.hooks.FreezeAtomsHook` keeps selected atoms fixed:
it snapshots their positions at ``BEFORE_PRE_UPDATE`` and restores them —
with zeroed velocities — at ``AFTER_POST_UPDATE``. The integrator runs
normally and the positions are overwritten afterward, so no integrator
modification is required.

``categories`` is a string or list of strings matching atom type categories in
the batch (for example, ``"substrate"`` or ``["substrate", "boundary"]``). Only
atoms in the listed categories are frozen; all others evolve freely.

Use this hook for partial-system relaxations (freeze the substrate, relax the
adsorbate), slab calculations (freeze bottom layers), or any configuration
where part of the system must remain rigid.


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

Safety: NaN detection and force clamping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Registration order determines execution order at the same stage. Clamp before
checking so the detector sees the corrected forces:

.. code-block:: python

   from nvalchemi.dynamics.hooks import MaxForceClampHook, NaNDetectorHook

   dynamics = DemoDynamics(
       model=model,
       dt=0.5,
       hooks=[
           MaxForceClampHook(max_force=50.0, log_clamps=True),
           NaNDetectorHook(extra_keys=["stress"]),
       ],
   )


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


API reference
-------------

.. currentmodule:: nvalchemi.dynamics.hooks

.. autosummary::
   :toctree: generated
   :nosignatures:

   LoggingHook
   SnapshotHook
   ConvergedSnapshotHook
   EnergyDriftMonitorHook
   NaNDetectorHook
   MaxForceClampHook
   FreezeAtomsHook
   StageTimingHook
   TorchProfilerHook
