.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _fused-stage-guide:

============================================
``FusedStage`` — Single-GPU Orchestration
============================================

:class:`~nvalchemi.dynamics.FusedStage` composes multiple dynamics
sub-stages on a **single GPU**, sharing one ``Batch`` and **one model
forward pass** per step. This eliminates redundant forward passes when
multiple simulation phases (e.g. relaxation → MD) operate on the same
hardware. Within the :class:`~nvalchemi.dynamics.DistributedPipeline`
paradigm, ``FusedStage`` is the mechanism used to allow for more than
one dynamics process to run on a single rank.

The ``+`` operator
------------------

The primary way to build a ``FusedStage`` is with the ``+`` operator:

.. code-block:: python

   from nvalchemi.dynamics import DemoDynamics

   optimizer = DemoDynamics(model=model, dt=0.5)
   md = DemoDynamics(model=model, dt=1.0)

   # Fuse two dynamics → one forward pass per step
   fused = optimizer + md

Chaining is supported for three or more stages:

.. code-block:: python

   stage_a = DemoDynamics(model=model, dt=0.5)
   stage_b = DemoDynamics(model=model, dt=1.0)
   stage_c = DemoDynamics(model=model, dt=2.0)

   # (stage_a + stage_b) returns a FusedStage
   # FusedStage + stage_c appends via FusedStage.__add__
   fused = stage_a + stage_b + stage_c

   print(fused)
   # FusedStage(sub_stages=[0:DemoDynamics, 1:DemoDynamics, 2:DemoDynamics],
   #            entry_status=0, exit_status=3, compiled=False, step_count=0)


How it works: status codes and masked updates
----------------------------------------------

Each sub-stage is assigned a **status code** (auto-assigned starting
from 0 when using ``+``). Every sample in the batch carries a
``batch.status`` tensor that determines which sub-stage processes it:

.. code-block:: text

   ┌────────────────────────────────────────────────────────────┐
   │  Batch with 8 samples                                      │
   │  status: [0, 0, 0, 1, 1, 0, 1, 0]                        │
   │                                                            │
   │  Step:                                                     │
   │  1. compute()  — single forward pass for ALL 8 samples     │
   │  2. sub_stage[0].masked_update(batch, status == 0)         │
   │     → updates samples 0, 1, 2, 5, 7                       │
   │  3. sub_stage[1].masked_update(batch, status == 1)         │
   │     → updates samples 3, 4, 6                             │
   │  4. convergence check per sub-stage                        │
   │     → if sample 2 converges: status[2] = 1 (migrated!)    │
   └────────────────────────────────────────────────────────────┘

The key insight is that **only one forward pass happens** regardless of
how many sub-stages exist. The expensive model evaluation is amortized
across all stages.


Convergence-driven stage migration
-----------------------------------

``FusedStage.__init__`` automatically registers ``ConvergenceHook``
instances between adjacent sub-stages:

.. code-block:: python

   fused = optimizer + md

   # Equivalent to manually doing:
   # optimizer gets: ConvergenceHook(source_status=0, target_status=1)
   # exit_status = 2  (one past the last sub-stage code)

When a sample in sub-stage 0 (optimizer) converges:

1. Its ``batch.status`` is updated from ``0`` → ``1``
2. On the next step, it is processed by sub-stage 1 (MD)
3. When it converges in sub-stage 1, its status becomes ``2``
   (= ``exit_status``)
4. It is graduated (either written to sinks or replaced via inflight
   batching)


Running a ``FusedStage``
-------------------------

``FusedStage.run()`` loops until **all** samples reach ``exit_status``.
Unlike ``BaseDynamics.run()``, the ``n_steps`` attribute (inherited
from ``BaseDynamics``) and any ``n_steps`` argument to ``run()`` are
**unused** — termination is purely convergence-driven.

**Mode 1: external batch (the common case)**

.. code-block:: python

   fused = optimizer + md
   result = fused.run(batch)
   # Returns when all samples have status == exit_status

**Mode 2: inflight batching with a sampler**

.. code-block:: python

   from nvalchemi.dynamics import SizeAwareSampler

   sampler = SizeAwareSampler(
       dataset=my_dataset,
       max_atoms=200,
       max_edges=1000,
       max_batch_size=64,
   )

   fused = optimizer + md
   # Configure the sampler on the fused stage
   fused.sampler = sampler
   fused.refill_frequency = 1

   result = fused.run()  # batch built from sampler automatically

   # result is None when sampler is exhausted and all samples graduated

In inflight mode, graduated samples are **replaced in-place** using
dummy-graph pointer manipulation---no batch reconstruction overhead.


Using a CUDA stream
--------------------

Use the context manager to run all computation on a dedicated CUDA
stream:

.. code-block:: python

   fused = optimizer + md

   with fused:
       fused.run(batch)
       # All GPU ops run on a dedicated stream

The stream is automatically propagated to all sub-stages.


``torch.compile`` support
--------------------------

.. code-block:: python

   fused = FusedStage(
       sub_stages=[(0, optimizer), (1, md)],
       compile_step=True,
       compile_kwargs={"mode": "reduce-overhead", "fullgraph": True},
   )

When ``compile_step=True``, the internal ``_step_impl`` method is
wrapped with ``torch.compile``. This can significantly improve
throughput by fusing GPU kernels across the entire fused step.


Combining with hooks and sinks
-------------------------------

Hooks registered on individual sub-stages are respected inside the
fused step:

.. code-block:: python

   from nvalchemi.dynamics.hooks import LoggingHook, SnapshotHook
   from nvalchemi.dynamics import HostMemory

   sink = HostMemory(capacity=10_000)

   optimizer = DemoDynamics(
       model=model,
       dt=0.5,
       hooks=[LoggingHook(frequency=100)],
   )
   md = DemoDynamics(
       model=model,
       dt=1.0,
       hooks=[SnapshotHook(sink=sink, frequency=10)],
   )

   fused = optimizer + md
   fused.run(batch)

   # LoggingHook fires every 100 steps for all samples
   # SnapshotHook fires every 10 steps and captures the full batch state


Explicit construction
---------------------

For advanced control, construct ``FusedStage`` directly:

.. code-block:: python

   from nvalchemi.dynamics import FusedStage, ConvergenceHook

   optimizer = DemoDynamics(
       model=model,
       dt=0.5,
       convergence_hook=ConvergenceHook(
           criteria=[
               {"key": "fmax", "threshold": 0.05},
               {"key": "energy_change", "threshold": 1e-6},
           ],
           source_status=0,
           target_status=1,
       ),
   )
   md = DemoDynamics(model=model, dt=1.0)

   fused = FusedStage(
       sub_stages=[(0, optimizer), (1, md)],
       entry_status=0,
       exit_status=2,
       compile_step=True,
   )


Summary of syntactic sugars
----------------------------

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Expression
     - Result
   * - ``dyn_a + dyn_b``
     - ``FusedStage`` with sub-stages ``[0: dyn_a, 1: dyn_b]``
   * - ``fused + dyn_c``
     - New ``FusedStage`` with ``dyn_c`` appended at next status code
   * - ``dyn_a + dyn_b + dyn_c``
     - Left-associative: ``(dyn_a + dyn_b) + dyn_c``
   * - ``with fused:``
     - Dedicated CUDA stream for all sub-stages
   * - ``fused.run(batch)``
     - Loop until all samples reach ``exit_status``
   * - ``fused.run()``
     - Inflight mode (requires ``sampler``)
