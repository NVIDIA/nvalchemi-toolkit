.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _hooks-guide:

==============================
Hooks — Observe & Modify
==============================

Hooks are the primary extensibility mechanism for nvalchemi workflows.
They let you inject custom logic at any stage of an engine's execution
loop—dynamics simulations or custom pipelines—without modifying the
engine itself.

The Hook protocol
-----------------

Any object matching the :class:`~nvalchemi.hooks.Hook` protocol can
be registered with an engine:

.. code-block:: python

   from enum import Enum
   from nvalchemi.hooks import Hook, HookContext
   from nvalchemi.dynamics.base import DynamicsStage

   class MyHook:
       """A minimal custom hook — no inheritance required."""

       stage: Enum
       frequency: int = 1

       def __call__(self, ctx: HookContext, stage: DynamicsStage) -> None:
           print(f"Step {ctx.step_count}: energy = {ctx.batch.energy.mean():.4f}")

Because ``Hook`` is a ``runtime_checkable`` ``Protocol``, you can also
use it as a type hint and check membership with ``isinstance``:

.. code-block:: python

   assert isinstance(MyHook(), Hook)  # True ✓

.. tip::

   **No subclassing required.** The protocol approach means any
   class---or even a frozen ``dataclass``---that provides
   ``frequency``, ``stage``, and ``__call__`` works as a hook.


HookContext
-----------

Every hook receives a :class:`~nvalchemi.hooks.HookContext`, which is a dataclass
that bundles the current workflow state into a single object:

.. list-table:: HookContext fields
   :widths: 20 25 55
   :header-rows: 1

   * - Field
     - Type
     - Description
   * - ``batch``
     - ``Batch``
     - Current batch being processed (all engines).
   * - ``step_count``
     - ``int``
     - Current step number.
   * - ``model``
     - ``BaseModelMixin | None``
     - Model being used (if applicable).
   * - ``converged_mask``
     - ``torch.Tensor | None``
     - Boolean mask of converged samples (dynamics only).
   * - ``global_rank``
     - ``int``
     - Distributed rank of this process.

Each engine overrides ``_build_context(batch)`` to populate the fields
relevant to its workflow.


Task-category specialization
-----------------------------

The hook system supports multiple task categories through stage enums.
Each engine declares which stage types it accepts via ``_stage_type``.

**Dynamics stages** — :class:`~nvalchemi.dynamics.base.DynamicsStage`:

.. code-block:: text

   BEFORE_STEP ─────────────────────────────────────────────────┐
   │                                                            │
   │  BEFORE_PRE_UPDATE → pre_update() → AFTER_PRE_UPDATE      │
   │  BEFORE_COMPUTE    → compute()    → AFTER_COMPUTE         │
   │  BEFORE_POST_UPDATE→ post_update()→ AFTER_POST_UPDATE     │
   │                                                            │
   AFTER_STEP ──────────────────────────────────────────────────┘
   ON_CONVERGE  (fires only when convergence is detected)

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


Registration and execution
--------------------------

Hooks are registered either at construction or via ``register_hook()``:

.. code-block:: python

   from nvalchemi.dynamics import DemoDynamics
   from nvalchemi.dynamics.hooks import LoggingHook, NaNDetectorHook

   # At construction (recommended for most cases)
   dynamics = DemoDynamics(
       model=model,
       dt=0.5,
       hooks=[LoggingHook(backend="csv", log_path="log.csv", frequency=100), NaNDetectorHook()],
   )

   # Or register later
   dynamics.register_hook(MaxForceClampHook(max_force=50.0))

Hooks are dispatched by the :class:`~nvalchemi.hooks.HookRegistryMixin`
machinery. At each stage, **all** registered hooks for that stage fire in
registration order, but only if ``step_count % hook.frequency == 0``.

The dispatch logic for each hook is:

1. If the hook defines ``_runs_on_stage(stage) -> bool``, call it.
2. Otherwise, check ``stage == hook.stage``.
3. If matched, call ``hook(ctx, stage)`` with a fresh
   :class:`~nvalchemi.hooks.HookContext`.

.. note::

   At ``step_count == 0`` all hooks fire (since ``0 % n == 0`` for
   any ``n``), making step 0 a good point for initialization logic.


Built-in hooks reference
------------------------

The ``nvalchemi.dynamics.hooks`` package ships eleven production-ready
hooks organized into four categories.

Pre-compute hooks (modify batch, fire at ``BEFORE_COMPUTE``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These hooks prepare the batch **before** the model forward pass.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Hook
     - Purpose
   * - :class:`~nvalchemi.dynamics.hooks.NeighborListHook`
     - Compute or refresh the neighbor list (``MATRIX`` or ``COO``
       format) with optional Verlet-skin buffering to skip redundant
       rebuilds.

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
   * - :class:`~nvalchemi.dynamics.hooks.BiasedPotentialHook`
     - Add an external bias potential (energy + forces) for
       enhanced sampling: umbrella sampling, metadynamics,
       steered MD, harmonic restraints, wall potentials.
   * - :class:`~nvalchemi.dynamics.hooks.WrapPeriodicHook`
     - Wrap atomic positions back into the unit cell under PBC.
       Fires at ``AFTER_POST_UPDATE``, respects per-system
       ``batch.pbc`` flags.

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

   from nvalchemi.dynamics.hooks import BiasedPotentialHook

   def harmonic_restraint(batch):
       """Restrain center of mass to the origin."""
       k = 10.0  # eV/Å²
       com = batch.positions.mean(dim=0, keepdim=True)
       bias_energy = 0.5 * k * (com ** 2).sum().unsqueeze(0).unsqueeze(0)
       bias_forces = -k * com.expand_as(batch.positions) / batch.num_nodes
       return bias_energy, bias_forces

   hook = BiasedPotentialHook(bias_fn=harmonic_restraint)
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

Custom scalars via LoggingHook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nvalchemi.dynamics.hooks import LoggingHook

   def pressure(ctx, stage):
       """Compute instantaneous pressure from the virial."""
       return compute_pressure(ctx.batch.stress, ctx.batch.cell)

   hook = LoggingHook(
       backend="csv",
       log_path="log.csv",
       frequency=50,
       custom_scalars={"pressure": pressure},
   )

Writing a custom hook from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The built-in hooks cover common needs, but you will often want to write
your own.  Typical reasons include:

- Recording a domain-specific observable (e.g. an RDF, a diffusion
  coefficient, or a custom order parameter).
- Injecting physics that the integrator does not natively support
  (e.g. thermostat rescaling, external fields).
- Bridging to an external system (database writes, message queues,
  dashboard updates).

A hook is **any Python object** that provides three things:

1. A ``stage`` attribute — an ``Enum`` value that tells the engine
   *when* the hook should fire (e.g. ``DynamicsStage.AFTER_STEP``).
2. A ``frequency`` attribute — a positive ``int`` controlling how often
   it fires (``1`` = every step, ``10`` = every tenth step, etc.).
3. A ``__call__(self, ctx: HookContext, stage: Enum) -> None`` method
   that contains the hook's logic.

That's it.  There is no base class to inherit from.  The
:class:`~nvalchemi.hooks.Hook` interface is defined as a
``runtime_checkable`` ``Protocol``, which means Python uses *structural
subtyping*: any object whose methods and attributes
matches the protocol is accepted, regardless of its class hierarchy.
You never *need* to write ``class MyHook(Hook)``; just provide the three
members.

Here is a concrete example---a Berendsen-like velocity rescaling hook:

.. code-block:: python

   from nvalchemi.dynamics.base import DynamicsStage
   from nvalchemi.hooks import HookContext

   class VelocityRescaleHook:
       """Rescale velocities to a target temperature (Berendsen-like)."""

       frequency: int
       stage = DynamicsStage.AFTER_POST_UPDATE

       def __init__(self, target_temp: float, tau: float, frequency: int = 1):
           self.target_temp = target_temp
           self.tau = tau
           self.frequency = frequency

       def __call__(self, ctx: HookContext, stage: DynamicsStage) -> None:
           current_temp = compute_temperature(ctx.batch)
           dt = getattr(ctx.model, 'dt', 1.0)
           scale = (1.0 + (dt / self.tau)
                    * (self.target_temp / current_temp - 1.0)) ** 0.5
           ctx.batch.velocities.mul_(scale)

Because the hook protocol checks structure rather than inheritance, even
a ``dataclass`` or ``NamedTuple`` would work, as long as it exposes the
three required members.  This makes hooks easy to test in isolation —
instantiate one, construct a :class:`~nvalchemi.hooks.HookContext` by
hand, and call it directly without spinning up a full dynamics engine.

Writing a cross-category hook with plum dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes a single hook should work across *multiple* stage enum types---
for example, a profiler that instruments both dynamics and a custom pipeline.
Because the ``stage`` argument to ``__call__`` is a generic ``Enum``,
you can use `plum-dispatch <https://github.com/beartype/plum>`_ to
overload the method for each stage type.  This gives you type-safe
branching: the runtime dispatches to the correct overload based on the
concrete ``Enum`` subclass passed by the engine.

Two additional pieces are needed for a cross-category hook:

- ``_runs_on_stage(self, stage: Enum) -> bool`` — tells the registry
  which stages this hook should fire at.  Without it, the registry only
  checks ``stage == self.stage``, which limits you to a single value.
- Multiple ``@dispatch``-decorated ``__call__`` overloads, one per
  category plus a fallback for unknown enum types.

.. code-block:: python

   from enum import Enum
   from plum import dispatch
   from nvalchemi.dynamics.base import DynamicsStage
   from nvalchemi.hooks import HookContext

   # Example custom stage enum for a hypothetical pipeline
   class MyPipelineStage(Enum):
       BEFORE_PROCESS = 0
       AFTER_PROCESS = 1

   class UniversalProfiler:
       """Hook that behaves differently in dynamics vs custom pipeline."""

       stage = DynamicsStage.BEFORE_STEP  # primary stage
       frequency = 1

       def __init__(self):
           self._stages = {DynamicsStage.BEFORE_STEP, DynamicsStage.AFTER_STEP,
                           MyPipelineStage.BEFORE_PROCESS, MyPipelineStage.AFTER_PROCESS}

       def _runs_on_stage(self, stage: Enum) -> bool:
           return stage in self._stages

       @dispatch
       def __call__(self, ctx: HookContext, stage: DynamicsStage) -> None:
           print(f"[dynamics] {stage.name} at step {ctx.step_count}")

       @dispatch
       def __call__(self, ctx: HookContext, stage: MyPipelineStage) -> None:
           print(f"[pipeline] {stage.name} at step {ctx.step_count}")

       @dispatch
       def __call__(self, ctx: HookContext, stage: Enum) -> None:
           print(f"[custom] {stage.name} at step {ctx.step_count}")

The built-in :class:`~nvalchemi.dynamics.hooks.ProfilerHook` follows
exactly this pattern, using dispatch to annotate NVTX ranges with the
appropriate domain string (``"dynamics"`` or ``"custom"``).


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

.. code-block:: text

   for each sub-stage:
       BEFORE_STEP hooks
   ── single compute() ──
   for each sub-stage:
       AFTER_COMPUTE hooks
   for each sub-stage:
       BEFORE_PRE_UPDATE hooks
       masked_update() (if any samples match status)
       AFTER_POST_UPDATE hooks
   for each sub-stage:
       AFTER_STEP hooks
   for each sub-stage:
       convergence check → ON_CONVERGE hooks
