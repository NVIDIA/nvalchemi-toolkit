.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _distributed-pipeline-guide:

================================================
``DistributedPipeline`` — Multi-GPU Workflows
================================================

:class:`~nvalchemi.dynamics.DistributedPipeline` maps **one dynamics
stage per GPU rank** and coordinates sample graduation between stages
via ``Batch.isend`` / ``Batch.irecv``.  Where ``FusedStage`` shares a
single forward pass on one GPU, ``DistributedPipeline`` lets each rank
run its own model independently---ideal when stages have different
computational profiles or when you need to scale beyond one GPU.

Every stage that participates in inter-rank communication **must**
have a :class:`~nvalchemi.dynamics.BufferConfig`, as this will determine
the on-GPU buffer used for point-to-point sample passing. See
:ref:`buffers-data-flow` for the full buffer lifecycle.


The ``|`` operator
------------------

The primary way to build a ``DistributedPipeline`` is with the ``|``
operator with a series of dynamics:

.. code-block:: python

   from nvalchemi.dynamics import DemoDynamics, BufferConfig

   buffer_config = BufferConfig(
       num_systems=64, num_nodes=2000, num_edges=10000,
   )

   optimizer = DemoDynamics(model=model, dt=0.5, buffer_config=buffer_config)
   md = DemoDynamics(model=model, dt=1.0, buffer_config=buffer_config)

   # Distribute across 2 GPU ranks
   pipeline = optimizer | md

Chaining is fully supported:

.. code-block:: python

   # Three-stage pipeline across 3 ranks
   pipeline = stage_a | stage_b | stage_c

   # Left-associative:
   #   stage_a | stage_b  → DistributedPipeline(stages={0: a, 1: b})
   #   pipeline | stage_c → DistributedPipeline(stages={0: a, 1: b, 2: c})

This creates a pipeline where each successive rank is a consumer of data
from its predecessor.

You can also chain entire pipelines:

.. code-block:: python

   pipe1 = stage_a | stage_b     # ranks 0, 1
   pipe2 = stage_c | stage_d     # ranks 0, 1 (renumbered)

   full = pipe1 | pipe2          # ranks 0, 1, 2, 3 (renumbered)

Finally, and perhaps the most powerful aspect of this abstraction
is the ability to combine with :class:`~nvalchemi.dynamics.FusedStage`,
i.e. run multiple stages on a single GPU within a global context:

.. code-block:: python

   full = (fire2 + annealer) + langevin

This emits a distributed pipeline where the first rank will combine
FIRE2 optimization with an annealing process, and pipe the state to
run Langevin dynamics.


Running a pipeline
------------------

**Context manager (recommended)**

.. code-block:: python

   pipeline = optimizer | md

   with pipeline:
       pipeline.run()

The context manager handles:

1. ``init_distributed()`` — initializes ``torch.distributed`` if not
   already done.
2. ``setup()`` — validates that every communicating stage has a
   ``BufferConfig``, wires ``prior_rank`` / ``next_rank`` between
   adjacent stages, moves the model to the correct device, computes
   recv templates via ``_share_templates()``, and initializes the
   distributed done tensor.
3. ``cleanup()`` — destroys the process group if the pipeline
   initialized it.

**Manual lifecycle**

.. code-block:: python

   pipeline = optimizer | md

   pipeline.init_distributed()
   pipeline.setup()
   pipeline.run()      # loop until all stages report done
   pipeline.cleanup()

**Launching with** ``torchrun``

.. code-block:: bash

   # 2-stage pipeline → launch with 2 ranks
   torchrun --nproc_per_node=2 my_pipeline_script.py


How it works
------------

.. code-block:: text

   Rank 0 (optimizer)              Rank 1 (MD)
   ┌──────────────────┐            ┌──────────────────┐
   │ _prestep_sync     │           │ _prestep_sync     │
   │   (recv from —)   │           │   (recv from 0)   │
   │                   │           │                   │
   │ step(batch)       │           │ step(batch)       │
   │                   │           │                   │
   │ _poststep_sync    │           │ _poststep_sync    │
   │   (send to 1)     │──────────→│   (send to —)     │
   │                   │  isend/   │                   │
   │ _sync_done_flags  │  irecv   │ _sync_done_flags  │
   └──────────────────┘            └──────────────────┘
         ↕ all_reduce(done)               ↕

Each step:

1. **Pre-step sync** (``_prestep_sync_buffers``): Zeros the send
   buffer and posts an ``irecv`` from the prior rank.  In sync mode
   the receive completes inline; in async modes a handle is stored
   for later completion.
2. **Complete pending recv** (``_complete_pending_recv``): Waits on
   the deferred receive, routes data through the recv buffer into
   the active batch via ``_buffer_to_batch``, and drains overflow
   sinks to backfill any available capacity.
3. **Dynamics step**: Each rank runs ``step(active_batch)`` on its
   local stage.
4. **Post-step sync** (``_poststep_sync_buffers``): Converged samples
   are copied into the send buffer via ``_populate_send_buffer``.
   The send buffer is then **unconditionally sent** to the next rank
   — even when empty — so the downstream ``irecv`` always completes
   (deadlock prevention).  On the final rank, converged samples are
   extracted via ``index_select`` and written to sinks.
5. **Termination check**: All ranks synchronize ``done`` flags via
   ``dist.all_reduce(MAX)``; the loop terminates when every rank
   reports done.


Communication modes
--------------------

The ``comm_mode`` parameter on each stage controls the blocking behavior
of inter-rank communication:

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Mode
     - Behavior
   * - ``"sync"``
     - Blocking receive in ``_prestep_sync_buffers``. Simplest and
       most debuggable. Good for small pipelines.
   * - ``"async_recv"``
     - Deferred receive: ``irecv`` is posted in ``_prestep_sync_buffers``
       but ``wait()`` is called later in ``_complete_pending_recv``.
       Allows compute to overlap with communication.
       **Default mode.**
   * - ``"fully_async"``
     - Both send and receive are deferred. Sends from the previous
       step are drained at the start of the next
       ``_prestep_sync_buffers``. Maximum overlap, highest
       throughput.

.. code-block:: python

   buffer_config = BufferConfig(
       num_systems=64, num_nodes=2000, num_edges=10000,
   )
   optimizer = DemoDynamics(
       model=model, dt=0.5,
       comm_mode="fully_async",
       buffer_config=buffer_config,
   )
   md = DemoDynamics(
       model=model, dt=1.0,
       comm_mode="async_recv",
       buffer_config=buffer_config,
   )
   pipeline = optimizer | md


Debug mode
~~~~~~~~~~

``debug_mode=True`` on :class:`~nvalchemi.dynamics.DistributedPipeline`
propagates to every stage and enables per-step ``loguru`` logging of
buffer populations, send/recv completions, and convergence events:

.. code-block:: python

   pipeline = DistributedPipeline(
       stages={0: optimizer, 1: md},
       debug_mode=True,
   )

Synchronized mode (debugging)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For debugging ordering or deadlock issues:

.. code-block:: python

   pipeline = DistributedPipeline(
       stages={0: optimizer, 1: md},
       synchronized=True,    # global barrier after every step
   )

.. warning::

   ``synchronized=True`` inserts a ``dist.barrier()`` after every
   ``step()``, which eliminates all inter-rank pipelining and
   significantly reduces throughput. Use only for debugging.


Inflight batching on the first stage
--------------------------------------

When the first stage has a ``SizeAwareSampler``, it builds the initial
batch from the sampler and replaces graduated samples automatically:

.. code-block:: python

   from nvalchemi.dynamics import SizeAwareSampler, BufferConfig

   buffer_config = BufferConfig(
       num_systems=64, num_nodes=2000, num_edges=10000,
   )
   sampler = SizeAwareSampler(
       dataset=my_dataset,
       max_atoms=200,
       max_edges=1000,
       max_batch_size=64,
   )

   optimizer = DemoDynamics(
       model=model, dt=0.5,
       sampler=sampler,
       refill_frequency=1,
       max_batch_size=64,
       buffer_config=buffer_config,
   )
   md = DemoDynamics(model=model, dt=1.0, buffer_config=buffer_config)

   pipeline = optimizer | md
   with pipeline:
       pipeline.run()

The second rank receives graduated samples via ``irecv`` and
accumulates them in its ``active_batch``.


Data sinks for the final stage
-------------------------------

On the final rank, converged samples are written to
:class:`~nvalchemi.dynamics.DataSink` instances:

.. code-block:: python

   from nvalchemi.dynamics import HostMemory, ZarrData

   md = DemoDynamics(
       model=model,
       dt=1.0,
       sinks=[
           HostMemory(capacity=10_000),          # primary: CPU memory
           ZarrData(store="results.zarr"),        # overflow: disk
       ],
   )

Sinks are tried in priority order; the first non-full sink receives
the data.


Combining ``FusedStage`` and ``DistributedPipeline``
-----------------------------------------------------

The ``+`` and ``|`` operators compose freely. You can fuse stages on
a single GPU and then distribute fused stages across GPUs:

.. code-block:: python

   buffer_config = BufferConfig(
       num_systems=64, num_nodes=2000, num_edges=10000,
   )

   # Rank 0: fused relax → anneal (one GPU, shared forward pass)
   rank0_stage = relax + anneal

   # Rank 1: production MD
   rank1_stage = DemoDynamics(model=model, dt=1.0, buffer_config=buffer_config)

   # Distribute across 2 GPUs
   pipeline = rank0_stage | rank1_stage

   with pipeline:
       pipeline.run()

This gives you the best of both worlds:

- **Rank 0** runs a ``FusedStage`` with two sub-stages, one forward
  pass per step, masked updates for each sub-stage.
- **Rank 1** runs standalone MD, receiving graduated samples from
  rank 0.

You can also compose multiple ``FusedStage`` instances:

.. code-block:: python

   rank0 = stage_a + stage_b     # fused on GPU 0
   rank1 = stage_c + stage_d     # fused on GPU 1
   rank2 = production_md         # standalone on GPU 2

   pipeline = rank0 | rank1 | rank2
   with pipeline:
       pipeline.run()


Summary of syntactic sugars
----------------------------

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Expression
     - Result
   * - ``dyn_a | dyn_b``
     - ``DistributedPipeline(stages={0: dyn_a, 1: dyn_b})``
   * - ``pipe | dyn_c``
     - Appended pipeline with ``dyn_c`` at next rank
   * - ``pipe1 | pipe2``
     - Merged pipeline (stages renumbered contiguously)
   * - ``dyn_a | dyn_b | dyn_c``
     - Left-associative chaining
   * - ``(a + b) | (c + d)``
     - Fused on rank 0, fused on rank 1
   * - ``with pipeline:``
     - Auto init_distributed → setup → cleanup
   * - ``pipeline.run()``
     - Loop until all ranks report done


Full end-to-end example
------------------------

.. code-block:: python

   #!/usr/bin/env python
   """Three-stage distributed pipeline: relax → anneal → production MD.

   Launch with:
       torchrun --nproc_per_node=3 pipeline_example.py
   """
   from nvalchemi.dynamics import (
       BufferConfig,
       DemoDynamics,
       ConvergenceHook,
       HostMemory,
       SizeAwareSampler,
   )
   from nvalchemi.dynamics.hooks import (
       LoggingHook,
       NaNDetectorHook,
       SnapshotHook,
   )

   buffer_config = BufferConfig(
       num_systems=64, num_nodes=2000, num_edges=10000,
   )

   # ── Stage 0: Geometry optimization with inflight batching ──
   sampler = SizeAwareSampler(
       dataset=my_dataset,
       max_atoms=200,
       max_edges=1000,
       max_batch_size=64,
   )
   optimizer = DemoDynamics(
       model=model,
       dt=0.5,
       convergence_hook=ConvergenceHook.from_fmax(0.05),
       hooks=[NaNDetectorHook()],
       sampler=sampler,
       comm_mode="fully_async",
       buffer_config=buffer_config,
   )

   # ── Stage 1: Annealing MD ──
   anneal = DemoDynamics(
       model=model,
       dt=1.0,
       hooks=[LoggingHook(backend="csv", log_path="anneal_log.csv", frequency=100)],
       comm_mode="async_recv",
       buffer_config=buffer_config,
   )

   # ── Stage 2: Production MD with trajectory recording ──
   sink = HostMemory(capacity=100_000)
   production = DemoDynamics(
       model=model,
       dt=2.0,
       hooks=[
           SnapshotHook(sink=sink, frequency=10),
           LoggingHook(backend="csv", log_path="production_log.csv", frequency=100),
       ],
       sinks=[sink],
       comm_mode="async_recv",
       buffer_config=buffer_config,
   )

   # ── Compose and run ──
   pipeline = optimizer | anneal | production

   with pipeline:
       pipeline.run()

   # On rank 2: trajectory = sink.read()
