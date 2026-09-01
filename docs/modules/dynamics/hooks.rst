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

:class:`~nvalchemi.dynamics.base.DynamicsStage` enumerates ten lifecycle
hook-firing points: ``ON_ADMISSION`` for batch setup, followed by nine stages
within each dynamics step:

.. graphviz::
   :caption: DynamicsStage lifecycle hook firing points.

   digraph dynamics_stages {
       rankdir=TB
       compound=true
       node [fontsize=11 shape=box style="rounded,filled" fillcolor="#1a1a1a"]
       edge [fontsize=10 style=bold]

       ON_ADMISSION [label="ON_ADMISSION\n(once per admission)" fillcolor="#4a3315"]
       BEFORE_STEP [label="BEFORE_STEP" fillcolor="#4a3315"]

       subgraph cluster_step {
           label="step body"
           style=rounded
           color="#76b900"
           fontcolor="#76b900"
           fontsize=12

           BEFORE_PRE_UPDATE  [label="BEFORE_PRE_UPDATE"]
           pre_update         [label="pre_update()" fillcolor="#1a1a1a"]
           AFTER_PRE_UPDATE   [label="AFTER_PRE_UPDATE"]

           BEFORE_COMPUTE     [label="BEFORE_COMPUTE"]
           compute            [label="compute()" fillcolor="#1a1a1a"]
           AFTER_COMPUTE      [label="AFTER_COMPUTE"]

           BEFORE_POST_UPDATE [label="BEFORE_POST_UPDATE"]
           post_update        [label="post_update()" fillcolor="#1a1a1a"]
           AFTER_POST_UPDATE  [label="AFTER_POST_UPDATE"]

           BEFORE_PRE_UPDATE -> pre_update -> AFTER_PRE_UPDATE
           AFTER_PRE_UPDATE -> BEFORE_COMPUTE
           BEFORE_COMPUTE -> compute -> AFTER_COMPUTE
           AFTER_COMPUTE -> BEFORE_POST_UPDATE
           BEFORE_POST_UPDATE -> post_update -> AFTER_POST_UPDATE
       }

       AFTER_STEP  [label="AFTER_STEP" fillcolor="#4a3315"]
       ON_CONVERGE [label="ON_CONVERGE\n(if converged)" fillcolor="#4a3315"]

       ON_ADMISSION -> BEFORE_STEP
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
   * - ``ON_ADMISSION``
     - -1
     - Once when a batch enters the engine, before force priming and the first step.
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

``ON_ADMISSION`` fires once per run or managed batch replacement, before force
priming. In :class:`~nvalchemi.dynamics.FusedStage`, it runs outside the compiled
``_step_impl``, making it suitable for validation and shape-dependent setup.


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

``backend`` is a required argument that selects the output destination. It must
be one of ``"csv"``, ``"tensorboard"``, or ``"custom"``:

- ``"csv"`` — writes one row per step to ``log_path``. Use when you need
  per-step data for post-run analysis in Python or a spreadsheet.
- ``"tensorboard"`` — writes scalar events to ``log_path`` as a TensorBoard
  event file. Use when comparing scalar trends across experiments.
- ``"custom"`` — routes each snapshot to a custom writer callable passed via the
  separate ``writer_fn`` parameter (signature
  ``fn(step_count, rows) -> None``), such as a W&B or MLflow sink.

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
eV/Å). Clamping is applied in-place to any per-atom force whose magnitude
exceeds the threshold. Frequent clamping during model development is a signal to
identify problem configurations.

Clamping prevents numerical blow-up from large forces in high-energy or
poorly-sampled configurations. It is a safety net, not a model fix: if
clamping fires frequently, the model has accuracy problems for those
structures.

Constraint hooks
~~~~~~~~~~~~~~~~

Constraint hooks enforce geometric constraints across integration steps. They
can span the pre-update, compute, and post-update boundaries to prevent frozen
state from influencing either half of the integrator while still presenting a
constrained geometry to the model.

FreezeAtomsHook
...............

:class:`~nvalchemi.dynamics.hooks.FreezeAtomsHook` keeps selected atoms fixed:
it fires at five stages. At ``BEFORE_PRE_UPDATE`` it snapshots positions and
clears prior forces and velocities; at ``AFTER_PRE_UPDATE`` it restores the
constrained geometry before compute preparation; at ``AFTER_COMPUTE`` it clears
new model forces when ``zero_forces=True``; at ``BEFORE_POST_UPDATE`` it always
clears frozen-atom forces; and at ``AFTER_POST_UPDATE`` it restores positions
and zeroes velocities.

``freeze_category`` is the integer ``batch.atom_categories`` value that marks
frozen atoms. It defaults to :attr:`~nvalchemi._typing.AtomCategory.SPECIAL`.
Only atoms matching that value are frozen; all others evolve freely.

Set ``zero_forces=False`` to expose raw frozen-atom model forces to
``AFTER_COMPUTE`` observers. Those forces are still cleared at
``BEFORE_POST_UPDATE``, before the second integrator update, so they cannot
move the frozen atoms.

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
           MaxForceClampHook(max_force=50.0),
           NaNDetectorHook(extra_keys=["stress"]),
       ],
   )


Hooks inside ``FusedStage``
---------------------------

Hooks may be registered directly on a
:class:`~nvalchemi.dynamics.FusedStage` or on any of its sub-stages. Step,
compute, and integrator update boundaries all fire at both levels. Fused-stage
hooks receive the full batch with the overall active mask (all systems whose
status is below ``exit_status``), while each sub-stage hook receives the same
batch with its status-specific mask. At admission,
the fused-stage ``ON_ADMISSION`` hooks fire first, followed by each sub-stage's
admission hooks in sub-stage order.

At every ``BEFORE_*`` boundary, the fused-stage hooks fire before the sub-stage
hooks. At every ``AFTER_*`` boundary, the sub-stage hooks fire before the
fused-stage hooks. Only ``ON_CONVERGE`` remains sub-stage-only because
convergence is evaluated independently for each sub-stage.

Register a cross-stage constraint once on the fused stage when it should apply
to every active system, regardless of its current sub-stage:

.. code-block:: python

   from nvalchemi.dynamics.hooks import FreezeAtomsHook

   fused = optimizer + md
   fused.register_hook(FreezeAtomsHook())
   fused.run(batch)

Register the hook on an individual sub-stage instead when the constraint should
apply only during that phase.

Hook ordering inside a fused step:

.. graphviz::
   :caption: Hook ordering inside a single ``FusedStage.step()``.

   digraph fused_hook_order {
       rankdir=TB
       ranksep=0.3
       compound=true
       node [fontsize=11 shape=box style="rounded,filled" fillcolor="#1a1a1a"]
       edge [fontsize=10 style=bold]

       fused_on_admission [label="FusedStage ON_ADMISSION hooks\n(outside compiled step)" fillcolor="#4a3315"]
       sub_on_admission [label="each sub-stage ON_ADMISSION hooks\n(in sub-stage order)"]
       fused_before_step [label="FusedStage BEFORE_STEP hooks" fillcolor="#4a3315"]
       fused_before_pre [label="FusedStage BEFORE_PRE_UPDATE hooks" fillcolor="#4a3315"]
       fused_after_pre [label="FusedStage AFTER_PRE_UPDATE hooks" fillcolor="#4a3315"]
       sub_before_step [label="each sub-stage BEFORE_STEP hooks\n(in sub-stage order)"]

       subgraph cluster_pre_update {
           label="for each sub-stage"
           style=dashed
           color="#76b900"
           fontcolor="#76b900"
           fontsize=10
           BEFORE_PRE [label="BEFORE_PRE_UPDATE hooks"]
           pre_update [label="masked_pre_update()" fillcolor="#1a1a1a"]
           AFTER_PRE [label="AFTER_PRE_UPDATE hooks"]
           BEFORE_PRE -> pre_update -> AFTER_PRE
       }

       fused_before_compute [label="FusedStage BEFORE_COMPUTE hooks" fillcolor="#4a3315"]
       sub_before_compute [label="each sub-stage BEFORE_COMPUTE hooks\n(in sub-stage order)"]
       compute [label="single shared compute()" fillcolor="#4a3315"]
       sub_after_compute [label="each sub-stage AFTER_COMPUTE hooks\n(in sub-stage order)"]
       fused_before_post [label="FusedStage BEFORE_POST_UPDATE hooks" fillcolor="#4a3315"]
       fused_after_post [label="FusedStage AFTER_POST_UPDATE hooks" fillcolor="#4a3315"]
       fused_after_compute [label="FusedStage AFTER_COMPUTE hooks" fillcolor="#4a3315"]

       subgraph cluster_post_update {
           label="for each sub-stage"
           style=dashed
           color="#76b900"
           fontcolor="#76b900"
           fontsize=10
           BEFORE_POST [label="BEFORE_POST_UPDATE hooks"]
           post_update [label="masked_post_update()" fillcolor="#1a1a1a"]
           AFTER_POST [label="AFTER_POST_UPDATE hooks"]
           BEFORE_POST -> post_update -> AFTER_POST
       }

       sub_after_step [label="each sub-stage AFTER_STEP hooks\n(in sub-stage order)"]
       fused_after_step [label="FusedStage AFTER_STEP hooks" fillcolor="#4a3315"]

       subgraph cluster_converge {
           label="for each sub-stage"
           style=dashed
           color="#76b900"
           fontcolor="#76b900"
           fontsize=10
           conv_check  [label="convergence evaluation" fillcolor="#1a1a1a"]
           ON_CONVERGE [label="ON_CONVERGE hooks\n(with convergence mask)"]
           conv_check -> ON_CONVERGE [style=dashed]
       }

       fused_on_admission -> sub_on_admission
       sub_on_admission -> fused_before_step
       fused_before_step -> sub_before_step
       sub_before_step -> fused_before_pre
       fused_before_pre -> BEFORE_PRE [lhead=cluster_pre_update]
       AFTER_PRE -> fused_after_pre [ltail=cluster_pre_update]
       fused_after_pre -> fused_before_compute
       fused_before_compute -> sub_before_compute
       sub_before_compute -> compute
       compute -> sub_after_compute
       sub_after_compute -> fused_after_compute
       fused_after_compute -> fused_before_post
       fused_before_post -> BEFORE_POST [lhead=cluster_post_update]
       AFTER_POST -> fused_after_post [ltail=cluster_post_update]
       fused_after_post -> sub_after_step
       sub_after_step -> fused_after_step
       fused_after_step -> conv_check [lhead=cluster_converge]
   }

Initial force priming follows the same nested ``BEFORE_COMPUTE`` and
``AFTER_COMPUTE`` ordering. Safety hooks (``NaNDetectorHook``,
``MaxForceClampHook``) and observer hooks (``LoggingHook``, ``SnapshotHook``)
therefore behave consistently whether they are registered on the fused stage
or on a specific sub-stage.

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

The general-purpose profiling hooks
:class:`~nvalchemi.hooks.StageTimingHook` and
:class:`~nvalchemi.hooks.TorchProfilerHook` also work with dynamics and are
documented in :ref:`hooks-api`.
