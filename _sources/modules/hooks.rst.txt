.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _hooks-api:

======================
Hooks — Core Framework
======================

The :mod:`nvalchemi.hooks` package provides the general-purpose hook
system used across all nvalchemi workflows (dynamics, training, custom
pipelines). It defines the protocol, context, registry, and a set of
hooks that are useful regardless of the specific engine type.

.. seealso::

   - **User guide**: :ref:`hooks_guide` — conceptual overview and usage
     patterns.
   - **Dynamics hooks**: :ref:`dynamics-hooks` — hooks and stages
     specific to dynamics simulations.


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

Every hook receives a :class:`~nvalchemi.hooks.HookContext`, a dataclass
that bundles the current workflow state into a single object. Each engine
overrides ``_build_context(batch)`` to populate the fields relevant to
its workflow.

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
   * - ``loss``
     - ``torch.Tensor | None``
     - Current loss value (training only).
   * - ``optimizer``
     - ``torch.optim.Optimizer | None``
     - Optimizer being used (training only).
   * - ``lr_scheduler``
     - ``object | None``
     - Learning rate scheduler (training only).
   * - ``gradients``
     - ``dict[str, torch.Tensor] | None``
     - Parameter gradients (training only).
   * - ``epoch``
     - ``int | None``
     - Current epoch number (training only).
   * - ``global_rank``
     - ``int``
     - Distributed rank of this process.
   * - ``workflow``
     - ``Any``
     - Back-reference to the engine running the hooks.


Registration and dispatch
-------------------------

Hooks are registered either at construction or via ``register_hook()``.
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
3. If matched, call ``hook(ctx, stage)`` with a fresh
   :class:`~nvalchemi.hooks.HookContext`.

.. note::

   At ``step_count == 0`` all hooks fire (since ``0 % n == 0`` for
   any ``n``), making step 0 a good point for initialization logic.


Task-category specialization
-----------------------------

The hook system supports multiple task categories through stage enums.
Each engine declares which stage types it accepts via ``_stage_type``.
For example:

- **Dynamics**: :class:`~nvalchemi.dynamics.base.DynamicsStage` — 9
  stages from ``BEFORE_STEP`` through ``ON_CONVERGE``.
- **Custom pipelines**: Any custom ``Enum`` type — the system accepts
  arbitrary enum types.

For multi-stage hooks, define a ``_runs_on_stage(stage) -> bool``
method. Hooks that need to support multiple enum types can use
`plum-dispatch <https://github.com/wesselb/plum>`_ to overload
``__call__``. See :ref:`hooks_guide` for full examples.


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


API Reference
-------------

Protocol
~~~~~~~~

.. currentmodule:: nvalchemi.hooks

.. autosummary::
   :toctree: generated
   :nosignatures:

   Hook
   HookContext
   HookRegistryMixin

General-purpose hooks
~~~~~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: generated
   :nosignatures:

   BiasedPotentialHook
   NeighborListHook
   WrapPeriodicHook
