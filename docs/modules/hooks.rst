.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _hooks-api:

======================
Hooks — Core Framework
======================

The :mod:`nvalchemi.hooks` package provides the general-purpose hook
system used across all nvalchemi workflows (dynamics, training, custom
pipelines). It defines the protocol, context dataclasses, registry, and a set of
hooks that are useful regardless of the specific engine type.

.. seealso::

   - **User guide**: :ref:`hooks_guide` — conceptual overview and usage
     patterns.
   - **Dynamics hooks**: :ref:`dynamics-hooks` — hooks and stages
     specific to dynamics simulations.
   - **Training update hooks**: :ref:`training-update-hooks` — update-stage
     ownership, veto semantics, and constraints for training hooks.


The Hook protocol
-----------------

:class:`~nvalchemi.hooks.Hook` is a ``runtime_checkable``
:class:`~typing.Protocol`. Any object that exposes the three required
members — ``stage``, ``frequency``, and ``__call__`` — is a valid hook,
with no subclassing required:

.. code-block:: python

   from enum import Enum
   from nvalchemi.hooks import Hook, HookContext

   class MyHook:
       """A minimal custom hook — no inheritance required."""

       stage: Enum
       frequency: int = 1

       def __call__(self, ctx: HookContext, stage: Enum) -> None:
           print(f"graphs={ctx.batch.num_graphs}, stage={stage.name}")

Because ``Hook`` is a ``runtime_checkable`` ``Protocol``, you can also
use it as a type hint and check membership with ``isinstance``:

.. code-block:: python

   assert isinstance(MyHook(), Hook)  # True ✓

CheckpointableHook
------------------

:class:`~nvalchemi.hooks.CheckpointableHook` is an optional second protocol
for hooks that carry restart-critical runtime state. ``Hook`` is required by
every hook; ``CheckpointableHook`` is opt-in — only hooks that implement both
``state_dict()`` and ``load_state_dict()`` participate in checkpoint save and
restore. Developers should meet the checkpointable protocol if the hook has
state that needs to be persisted and restartable.

The two required methods:

- ``state_dict() -> dict`` — return a serializable snapshot of all runtime
  state that must survive a restart: accumulated counters, learned parameters,
  and runtime tensors. Do not include configuration already captured by the
  constructor; those are restored at construction time.
- ``load_state_dict(state: Mapping) -> None`` — restore state from a
  ``state_dict()`` snapshot. Validate critical configuration fields (such as
  decay rate or step frequency) before restoring runtime tensors to catch
  checkpoint/config mismatches early.

The training checkpoint loader calls ``state_dict()`` on every hook that
satisfies this protocol and stores the results alongside model and optimizer
state. On resume via ``TrainingStrategy.load_checkpoint(path, hooks=[...])``,
``load_state_dict()`` is called on each matching hook by class name. Hooks that
do not implement ``CheckpointableHook`` are silently skipped; they restart from
their initial state, which is correct for purely stateless hooks.

``isinstance(hook, CheckpointableHook)`` is ``True`` for any hook that provides
both methods, with no subclassing required.

.. seealso::

   :ref:`training-update-hooks` — a concrete ``CheckpointableHook`` pattern
   with Pydantic fields and private runtime tensors.


Context dataclasses
-------------------

Every hook receives a :class:`~nvalchemi.hooks.HookContext` or a
workflow-specific subclass. The base dataclass contains only fields shared by
all hook-enabled engines; specialized contexts add fields that are meaningful
only for one workflow category.

**HookContext** (base, all engines)

.. dataclass-table:: nvalchemi.hooks.HookContext

**DynamicsContext** (dynamics workflows)

.. dataclass-table:: nvalchemi.hooks.DynamicsContext

**TrainContext** (training workflows)

.. dataclass-table:: nvalchemi.hooks.TrainContext


Registration and dispatch
-------------------------

Hooks are registered either at construction or manually via ``register_hook()``.
The :class:`~nvalchemi.hooks.HookRegistryMixin` provides flat-list
storage and dispatch logic for any engine.

.. code-block:: python

   # At construction (recommended for most cases)
   engine = MyEngine(hooks=[MyHook()])

   # Or register later
   engine.register_hook(AnotherHook())

At each stage, **all** registered hooks for that stage fire in
registration order, but only if ``step_count % hook.frequency == 0``.

The dispatch logic for each hook is:

1. If the hook defines ``_runs_on_stage(stage) -> bool``, call it.
2. Otherwise, check ``stage == hook.stage``.
3. If matched, call ``hook(ctx, stage)`` with a fresh context object.

.. note::

   At ``step_count == 0`` all hooks fire (since ``0 % n == 0`` for
   any ``n``).


Stage enums and multi-stage hooks
-----------------------------------

Each workflow engine fires hooks at named lifecycle points defined by a stage
enum. The two built-in enums are:

- :class:`~nvalchemi.dynamics.base.DynamicsStage` — 9 stages from
  ``BEFORE_STEP`` through ``ON_CONVERGE``. See :ref:`dynamics-hooks`.
- :class:`~nvalchemi.training.TrainingStage` — stages from ``SETUP``
  through ``AFTER_TRAINING``. See :ref:`training-hooks-api`.

Custom pipelines may use any ``Enum`` type. For hooks that fire at more
than one stage, define ``_runs_on_stage(stage) -> bool`` instead of a
single ``stage`` attribute. Hooks that must support multiple enum types
can overload ``__call__`` with plum-dispatch; see :ref:`hooks_guide`.


General-purpose hooks
---------------------

These hooks live in :mod:`nvalchemi.hooks` and work with any engine
that uses the hook system, not just dynamics.

.. list-table::
   :widths: 25 75
   :header-rows: 1

   * - Hook
     - Purpose
   * - :class:`~nvalchemi.hooks.NeighborListHook`
     - Compute or refresh the neighbor list (``MATRIX`` or ``COO``
       format) with optional Verlet-skin buffering to skip redundant
       rebuilds. Fires at ``BEFORE_COMPUTE``.
   * - :class:`~nvalchemi.hooks.BiasedPotentialHook`
     - Add an external bias potential (energy + forces) for enhanced
       sampling: umbrella sampling, metadynamics, steered MD, harmonic
       restraints, wall potentials.
   * - :class:`~nvalchemi.hooks.WrapPeriodicHook`
     - Wrap atomic positions back into the unit cell under PBC.
       Fires at ``AFTER_POST_UPDATE``, respects per-system
       ``batch.pbc`` flags.
   * - :class:`~nvalchemi.hooks.StageTimingHook`
     - Measure elapsed time between hook stages, with optional NVTX ranges, CSV
       output, and console summaries.
   * - :class:`~nvalchemi.hooks.TorchProfilerHook`
     - Capture PyTorch profiler Chrome traces for training and dynamics through
       PhysicsNeMo's profiler wrapper, with rank-specific output directories.


Reporting
---------

:class:`~nvalchemi.hooks.ReportingOrchestrator` is a standard hook that fans
reporting events to a list of reporter objects at a configured stage and
frequency. The ``Reporter`` protocol requires one method:

.. code-block:: python

   def report(ctx: HookContext, stage: Enum, state: ReportingState) -> None: ...

Two optional class attributes control distributed behavior:

- ``rank_zero_only = True`` — the orchestrator does not call the reporter
  on nonzero ranks.
- ``requires_all_ranks = True`` — all ranks participate in a collective
  reduction; only rank zero calls ``report()`` with the merged snapshot.

:func:`~nvalchemi.hooks.collect_scalars` assembles a
:class:`~nvalchemi.hooks.ScalarSnapshot` — a frozen payload of scalar
values, counters, and metadata — from the current hook context. Custom
reporters call it directly; built-in reporters call it internally.

:class:`~nvalchemi.hooks.TensorBoardReporter` and
:class:`~nvalchemi.hooks.RichReporter` are provided implementations.
:class:`~nvalchemi.hooks.RichLayout` and
:class:`~nvalchemi.hooks.BaseRichLayout` control the dashboard surface for
``RichReporter``.

.. seealso::

   :doc:`/userguide/reporting` — setup, layout design, and custom reporters.


API Reference
-------------

Protocol
~~~~~~~~

.. currentmodule:: nvalchemi.hooks

.. autosummary::
   :toctree: generated
   :nosignatures:

   Hook
   CheckpointableHook
   HookContext
   DynamicsContext
   TrainContext
   HookRegistryMixin

General-purpose hooks
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   BiasedPotentialHook
   NeighborListHook
   StageTimingHook
   TorchProfilerHook
   WrapPeriodicHook

Reporting
~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   Reporter
   ReportingOrchestrator
   ReportingState
   ScalarSnapshot
   collect_scalars
   TensorBoardReporter
   RichReporter
   RichLayout
   BaseRichLayout
   TrainingRichLayout
   DynamicsRichLayout
