.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _convergence-guide:

==============================
Convergence Criteria
==============================

:class:`~nvalchemi.dynamics.ConvergenceHook` provides a composable,
declarative system for deciding when samples have converged. It is the
bridge between dynamics and orchestration: when used standalone it
detects convergence; when used inside a
:class:`~nvalchemi.dynamics.FusedStage` it additionally **migrates**
converged samples to the next stage.

Quick examples
--------------

.. code-block:: python

   from nvalchemi.dynamics import ConvergenceHook

   # Simple fmax convergence (default if you pass nothing)
   hook = ConvergenceHook()  # fmax ≤ 0.05

   # Explicit fmax threshold via convenience constructor
   hook = ConvergenceHook.from_fmax(0.01)

   # Multi-criteria: fmax AND energy_change
   hook = ConvergenceHook(criteria=[
       {"key": "fmax", "threshold": 0.05},
       {"key": "energy_change", "threshold": 1e-6},
   ])

   # With status migration (for FusedStage)
   hook = ConvergenceHook(
       criteria=[{"key": "fmax", "threshold": 0.05}],
       source_status=0,    # check samples with status == 0
       target_status=1,    # promote converged ones to status 1
   )


How convergence criteria work
-----------------------------

Each criterion is a :class:`_ConvergenceCriterion` (internal Pydantic
model) that evaluates one tensor key on the batch:

.. code-block:: python

   {"key": "fmax", "threshold": 0.05}
   {"key": "forces", "threshold": 0.01, "reduce_op": "norm", "reduce_dims": -1}
   {"key": "energy_change", "threshold": 1e-6}

The evaluation pipeline for each criterion:

1. Retrieve ``getattr(batch, key)``
2. If ``custom_op`` is set, delegate entirely to it
3. If ``reduce_op`` is set, apply the reduction (``min``, ``max``,
   ``norm``, ``mean``, ``sum``) along ``reduce_dims``
4. If the tensor is node-level (shape ``(V, ...)``), scatter-reduce
   to graph-level ``(B,)`` via max
5. Compare against ``threshold``: ``reduced ≤ threshold``
6. Return a boolean mask ``(B,)``

``ConvergenceHook`` combines multiple criteria with **AND** semantics:
a sample converges only when **every** criterion is satisfied.

.. code-block:: python

   # Sample converges when BOTH conditions hold:
   #   fmax ≤ 0.05  AND  energy_change ≤ 1e-6
   hook = ConvergenceHook(criteria=[
       {"key": "fmax", "threshold": 0.05},
       {"key": "energy_change", "threshold": 1e-6},
   ])


Criterion specification
-----------------------

Each criterion accepts these parameters:

.. list-table::
   :widths: 20 15 65
   :header-rows: 1

   * - Parameter
     - Type
     - Description
   * - ``key``
     - ``str``
     - Tensor attribute name on the batch (e.g. ``"fmax"``,
       ``"forces"``, ``"energy_change"``).
   * - ``threshold``
     - ``float``
     - Values ≤ this are considered converged.
   * - ``reduce_op``
     - ``str | None``
     - Reduction before graph-level aggregation: ``"min"``,
       ``"max"``, ``"norm"``, ``"mean"``, ``"sum"``, or ``None``.
   * - ``reduce_dims``
     - ``int | list[int]``
     - Dimension(s) to reduce over. Default ``-1``.
   * - ``custom_op``
     - callable
     - Fully custom: receives the raw tensor, returns
       ``Bool[Tensor, "B"]``. Overrides all other parameters.


Common patterns
~~~~~~~~~~~~~~~

**Force-based convergence** (the most common for geometry optimization):

.. code-block:: python

   # fmax is expected to be a graph-level scalar on the batch
   {"key": "fmax", "threshold": 0.05}

**Force norm convergence** (when fmax is not pre-computed):

.. code-block:: python

   # Compute L2 norm per atom, then scatter-max to graph level
   {"key": "forces", "threshold": 0.01, "reduce_op": "norm", "reduce_dims": -1}

**Energy change convergence**:

.. code-block:: python

   {"key": "energy_change", "threshold": 1e-6}

**Custom convergence logic**:

.. code-block:: python

   import torch

   def bond_length_criterion(positions: torch.Tensor) -> torch.Tensor:
       """Converge when all bonds are within target range."""
       # Custom logic returning Bool[Tensor, "B"]
       ...

   {"key": "positions", "threshold": 0.0, "custom_op": bond_length_criterion}


Convenience constructors
------------------------

.. code-block:: python

   # from_fmax — backward-compatible shorthand
   hook = ConvergenceHook.from_fmax(threshold=0.01)

   # With status migration
   hook = ConvergenceHook.from_fmax(
       threshold=0.05,
       source_status=0,
       target_status=1,
   )


Status migration for ``FusedStage``
------------------------------------

When ``source_status`` and ``target_status`` are both set, the
convergence hook **also** updates ``batch.status`` for converged
samples:

.. code-block:: text

   For each converged sample:
       if batch.status[i] == source_status:
           batch.status[i] = target_status

This is the mechanism that drives sample progression through fused
stages. See :ref:`fused-stage-guide` for a full walkthrough.

.. note::

   When composing with ``+``, ``FusedStage.__init__`` **auto-registers**
   convergence hooks between adjacent sub-stages, so you rarely need
   to set ``source_status`` / ``target_status`` manually.

   .. code-block:: python

      # The + operator auto-wires convergence hooks:
      #   sub-stage 0 → ConvergenceHook(source_status=0, target_status=1)
      #   sub-stage 1 → (exit at status 2)
      fused = optimizer + md


Attaching to dynamics
---------------------

There are two ways to use ``ConvergenceHook``:

**As the dynamics convergence detector** (via ``convergence_hook=``):

.. code-block:: python

   dynamics = DemoDynamics(
       model=model,
       dt=0.5,
       convergence_hook=ConvergenceHook.from_fmax(0.01),
   )

   # Or pass a dict — auto-converted to ConvergenceHook
   dynamics = DemoDynamics(
       model=model,
       dt=0.5,
       convergence_hook={"criteria": [{"key": "fmax", "threshold": 0.01}]},
   )

**As a registered hook** (for custom status migration logic):

.. code-block:: python

   hook = ConvergenceHook(
       criteria=[{"key": "fmax", "threshold": 0.05}],
       source_status=0,
       target_status=1,
   )
   dynamics.register_hook(hook)
