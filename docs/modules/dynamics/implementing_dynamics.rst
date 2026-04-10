.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _implementing-dynamics:

=======================================
Implementing Custom Dynamics
=======================================

This guide walks through the developer contract for creating a new
integrator, using :class:`~nvalchemi.dynamics.DemoDynamics` (Velocity
Verlet) as the running example.

The developer contract
----------------------

To implement a new integrator you **must**:

1. Subclass :class:`~nvalchemi.dynamics.BaseDynamics`.
2. Override ``pre_update(batch)`` and/or ``post_update(batch)``.
3. Declare ``__needs_keys__`` and ``__provides_keys__``.

You should **not** override ``step()``, ``compute()``, or ``run()`` —
the base class orchestrates hook firing, model evaluation, output
validation, and convergence checking.

.. code-block:: python

   from nvalchemi.dynamics import BaseDynamics
   from nvalchemi.data import Batch

   class MyIntegrator(BaseDynamics):
       """Minimal skeleton for a custom integrator."""

       # Declare what the model must produce
       __needs_keys__: set[str] = {"forces"}

       # Declare what this integrator writes to the batch
       __provides_keys__: set[str] = {"velocities", "positions"}

       def pre_update(self, batch: Batch) -> None:
           """First half-step: update positions using current state."""
           ...

       def post_update(self, batch: Batch) -> None:
           """Second half-step: update velocities using new forces."""
           ...


``__needs_keys__`` and ``__provides_keys__``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These class-level sets drive **automatic validation**:

* After every ``compute()`` call, ``BaseDynamics._validate_model_outputs``
  checks that every key in ``__needs_keys__`` is present and non-``None``
  in the model outputs. A clear ``RuntimeError`` is raised otherwise.
* ``__provides_keys__`` documents which additional batch fields the
  integrator writes (beyond model outputs like forces and energy).
  The diagnostic helper ``_validate_batch_keys`` can verify them.

When dynamics are composed into a :class:`~nvalchemi.dynamics.FusedStage`,
the fused stage computes the **union** of all sub-stage keys
automatically:

.. code-block:: python

   fused = relax + md  # __needs_keys__ = relax.__needs_keys__ | md.__needs_keys__


Walkthrough: ``DemoDynamics`` (Velocity Verlet)
------------------------------------------------

The full implementation lives in ``nvalchemi/dynamics/demo.py``. Let's
break it down section by section.

Class declaration and keys
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class DemoDynamics(BaseDynamics):
       """Velocity Verlet integrator for molecular dynamics simulations."""

       __needs_keys__: set[str] = {"forces"}
       __provides_keys__: set[str] = {"velocities", "positions"}

       _prev_accelerations: torch.Tensor | None

The integrator requires ``forces`` from the model and writes
``velocities`` and ``positions`` back to the batch. A private
``_prev_accelerations`` cache stores the previous step's accelerations
for the half-step update.

Constructor
~~~~~~~~~~~

.. code-block:: python

   def __init__(
       self,
       model: BaseModelMixin,
       n_steps: int,
       dt: float = 1.0,
       hooks: list[Hook] | None = None,
       convergence_hook: ConvergenceHook | dict | None = None,
       **kwargs: Any,
   ) -> None:
       super().__init__(
           model=model,
           hooks=hooks,
           convergence_hook=convergence_hook,
           n_steps=n_steps,
           **kwargs,              # ← forwards communication kwargs
       )
       self.dt = dt
       self._prev_accelerations = None

The ``**kwargs`` forwarding is essential for cooperative MRO:
``BaseDynamics.__init__`` forwards to ``_CommunicationMixin.__init__``,
which accepts ``prior_rank``, ``next_rank``, ``sinks``,
``max_batch_size``, ``sampler``, etc. By forwarding ``**kwargs``, a
single constructor call configures both the integrator *and* the
communication layer.

Note that ``dt`` is **not** part of the base class — each subclass
that needs a timestep should accept it explicitly and store it as
``self.dt``:

.. code-block:: python

   # Works seamlessly in a pipeline context
   dyn = DemoDynamics(
       model=model,
       dt=0.5,
       max_batch_size=64,
       comm_mode="async_recv",
   )

``pre_update``: position half-step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def pre_update(self, batch: Batch) -> None:
       positions: NodePositions = batch.positions
       velocities: NodeVelocities = batch.velocities
       forces: Forces | None = batch.forces
       masses = batch.atomic_masses.unsqueeze(-1)

       dt = self.dt

       with torch.no_grad():
           if forces is not None and not torch.all(forces == 0):
               accelerations = forces / masses
               self._prev_accelerations = accelerations.clone()
               # x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
               positions.add_(velocities * dt + 0.5 * accelerations * dt * dt)
           else:
               # First step: Euler fallback
               positions.add_(velocities * dt)

Key patterns:

* **In-place tensor ops** (``add_``, ``copy_``) — the batch is
  modified in-place; never reassign ``batch.positions = ...``.
* **``torch.no_grad()`` context** — avoids conflicts when the model
  uses conservative (autograd) forces.
* **Type annotations from** ``nvalchemi._typing`` — ``NodePositions``,
  ``NodeVelocities``, ``Forces`` provide jaxtyping shape documentation.

``post_update``: velocity half-step
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def post_update(self, batch: Batch) -> None:
       velocities: NodeVelocities = batch.velocities
       forces: Forces = batch.forces
       masses = batch.atomic_masses.unsqueeze(-1)

       dt = self.dt

       with torch.no_grad():
           new_accelerations = forces / masses

           if self._prev_accelerations is not None:
               # v(t+dt) = v(t) + 0.5*(a(t) + a(t+dt))*dt
               velocities.add_(
                   0.5 * (self._prev_accelerations + new_accelerations) * dt,
               )
           else:
               # First step: Euler fallback
               velocities.add_(new_accelerations * dt)

At this point, ``forces`` are the **new** forces from ``compute()``,
which ran between ``pre_update`` and ``post_update``. The standard
Velocity Verlet averaging of old and new accelerations gives symplectic,
time-reversible integration.


How ``step()`` orchestrates everything
--------------------------------------

You do **not** override ``step()``. The base class runs this sequence on
every call:

.. code-block:: text

   1.  BEFORE_STEP hooks
   2.  BEFORE_PRE_UPDATE hooks  →  pre_update()  →  AFTER_PRE_UPDATE hooks
   3.  BEFORE_COMPUTE hooks     →  compute()      →  AFTER_COMPUTE hooks
   4.  BEFORE_POST_UPDATE hooks →  post_update()  →  AFTER_POST_UPDATE hooks
   5.  AFTER_STEP hooks
   6.  convergence check  →  ON_CONVERGE hooks (if any samples converged)
   7.  step_count += 1

``compute()`` handles the full model pipeline: forward pass →
``adapt_output()`` → ``_validate_model_outputs()`` → write
forces/energy to batch via ``copy_()``.


``masked_update`` for ``FusedStage`` compatibility
--------------------------------------------------

When your dynamics is composed via ``+`` into a
:class:`~nvalchemi.dynamics.FusedStage`, the fused stage calls
``masked_update(batch, mask)`` instead of ``pre_update`` / ``post_update``
directly. The default implementation in ``BaseDynamics`` is:

.. code-block:: python

   def masked_update(self, batch, mask):
       # Expand graph-level mask → node-level via batch.batch_idx
       node_mask = mask[batch.batch_idx]

       # Snapshot unmasked state
       original_positions = batch.positions.clone()
       original_velocities = batch.velocities.clone() if ... else None

       # Run full updates
       self.pre_update(batch)
       self.post_update(batch)

       # Restore unmasked nodes
       with torch.no_grad():
           batch.positions[~node_mask] = original_positions[~node_mask]
           if original_velocities is not None:
               batch.velocities[~node_mask] = original_velocities[~node_mask]

This means your custom ``pre_update`` / ``post_update`` work correctly
inside a ``FusedStage`` without any modifications. The mask
selectively applies your updates only to samples at the corresponding
status code.


Checklist for a new integrator
------------------------------

.. code-block:: text

   ☐  Subclass BaseDynamics
   ☐  Set __needs_keys__   (e.g. {"forces"})
   ☐  Set __provides_keys__ (e.g. {"velocities", "positions"})
   ☐  Override pre_update(batch)  — first half-step (positions)
   ☐  Override post_update(batch) — second half-step (velocities)
   ☐  Use in-place tensor ops (add_, copy_) — never reassign batch attrs
   ☐  Wrap updates in torch.no_grad() if model is conservative
   ☐  Forward **kwargs in __init__ for communication support
   ☐  Accept and store `dt` (or other integrator-specific params) directly
   ☐  Write tests using DemoModelWrapper fixtures
