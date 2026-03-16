.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _hooks-guide:

==============================
Hooks — Observe & Modify
==============================

Hooks are the primary extensibility mechanism for dynamics simulations.
They let you inject custom logic at any stage of the integration step
without modifying the integrator itself.

The Hook protocol
-----------------

Any object matching the :class:`~nvalchemi.dynamics.Hook` protocol can
be registered:

.. code-block:: python

   from nvalchemi.dynamics import Hook, HookStageEnum

   class MyHook:
       """A minimal custom hook — no inheritance required."""

       frequency: int = 1
       stage: HookStageEnum = HookStageEnum.AFTER_STEP

       def __call__(self, batch, dynamics):
           print(f"Step {dynamics.step_count}: energy = {batch.energies.mean():.4f}")

Because ``Hook`` is a ``runtime_checkable`` ``Protocol``, you can also
use it as a type hint and check membership with ``isinstance``:

.. code-block:: python

   assert isinstance(MyHook(), Hook)  # True ✓

.. tip::

   **No subclassing required.** The protocol approach means any
   class---or even a frozen ``dataclass``---that provides
   ``frequency``, ``stage``, and ``__call__`` works as a hook.


Hook stages
-----------

:class:`~nvalchemi.dynamics.HookStageEnum` defines **nine** insertion
points that cover every phase of a dynamics step:

.. code-block:: text

   BEFORE_STEP ─────────────────────────────────────────────────┐
   │                                                            │
   │  BEFORE_PRE_UPDATE → pre_update() → AFTER_PRE_UPDATE      │
   │  BEFORE_COMPUTE    → compute()    → AFTER_COMPUTE         │
   │  BEFORE_POST_UPDATE→ post_update()→ AFTER_POST_UPDATE     │
   │                                                            │
   AFTER_STEP ──────────────────────────────────────────────────┘
   ON_CONVERGE  (fires only when convergence is detected)

.. list-table:: Hook Stages Reference
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
     - After forces/energies are written to the batch.
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
       hooks=[LoggingHook(frequency=100), NaNDetectorHook()],
   )

   # Or register later
   dynamics.register_hook(MaxForceClampHook(max_force=50.0))

Hooks are dispatched by ``BaseDynamics._call_hooks(stage, batch)``.
At each stage, **all** registered hooks for that stage fire in
registration order, but only if ``step_count % hook.frequency == 0``.

.. note::

   At ``step_count == 0`` all hooks fire (since ``0 % n == 0`` for
   any ``n``), making step 0 a good point for initialization logic.


Built-in hooks reference
------------------------

The ``nvalchemi.dynamics.hooks`` package ships eight production-ready
hooks organized into two categories.

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
   * - :class:`~nvalchemi.dynamics.hooks.EnergyDriftMonitorHook`
     - Track cumulative energy drift in NVE runs; warn or halt on
       excessive drift.
   * - :class:`~nvalchemi.dynamics.hooks.ProfilerHook`
     - Instrument steps with NVTX ranges and wall-clock timing for
       Nsight Systems profiling. Fires at ``BEFORE_STEP`` and
       manages the end-of-step counterpart internally.

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
     - Detect NaN/Inf in forces and energies; raise with
       diagnostic info (affected graph indices, step count).
   * - :class:`~nvalchemi.dynamics.hooks.MaxForceClampHook`
     - Clamp per-atom force magnitudes to a safe maximum,
       preserving force direction. Prevents numerical explosions.
   * - :class:`~nvalchemi.dynamics.hooks.BiasedPotentialHook`
     - Add an external bias potential (energies + forces) for
       enhanced sampling: umbrella sampling, metadynamics,
       steered MD, harmonic restraints, wall potentials.
   * - :class:`~nvalchemi.dynamics.hooks.WrapPeriodicHook`
     - Wrap atomic positions back into the unit cell under PBC.
       Fires at ``AFTER_POST_UPDATE``, respects per-system
       ``batch.pbc`` flags.


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
           NaNDetectorHook(extra_keys=["stresses"]),
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

   def pressure(batch, dynamics):
       """Compute instantaneous pressure from the virial."""
       return compute_pressure(batch.stresses, batch.cell)

   hook = LoggingHook(
       frequency=50,
       custom_scalars={"pressure": pressure},
   )

Writing a custom hook from scratch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from nvalchemi.dynamics import HookStageEnum

   class VelocityRescaleHook:
       """Rescale velocities to a target temperature (Berendsen-like)."""

       frequency: int
       stage = HookStageEnum.AFTER_POST_UPDATE

       def __init__(self, target_temp: float, tau: float, frequency: int = 1):
           self.target_temp = target_temp
           self.tau = tau
           self.frequency = frequency

       def __call__(self, batch, dynamics):
           current_temp = compute_temperature(batch)
           scale = (1.0 + (dynamics.dt / self.tau)
                    * (self.target_temp / current_temp - 1.0)) ** 0.5
           batch.velocities.mul_(scale)


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
