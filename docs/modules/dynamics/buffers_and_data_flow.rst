.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0

.. _buffers-data-flow:

==============================
Buffers & Data Flow
==============================

The dynamics framework uses a layered buffer architecture to manage
data flow between the active simulation batch, inter-rank
communication, and persistent storage. Understanding this architecture
is essential for optimizing throughput in multi-GPU pipelines and
debugging data routing issues.


The three buffer layers
-----------------------

.. list-table::
   :widths: 20 25 20 35
   :header-rows: 1

   * - Layer
     - Class / Location
     - Storage
     - Purpose
   * - Communication
     - ``send_buffer`` / ``recv_buffer`` on :class:`~nvalchemi.dynamics.base._CommunicationMixin`
     - Pre-allocated :meth:`Batch.empty() <nvalchemi.data.Batch.empty>`
     - Zero-copy inter-rank transfer via ``isend`` / ``irecv``
   * - Overflow sinks
     - :class:`~nvalchemi.dynamics.DataSink` (``GPUBuffer``, ``HostMemory``, ``ZarrData``)
     - Varies
     - Staging when active batch is full
   * - Active batch
     - ``active_batch`` on :class:`~nvalchemi.dynamics.base._CommunicationMixin`
     - Live :class:`~nvalchemi.data.Batch`
     - The working set being integrated

.. code-block:: text

   Dataset/Sampler --> [Active Batch] --> step() --> convergence check
                            ^                             |
                       _recv_to_batch              _poststep_sync_buffers
                            ^                             |
                      [Recv Buffer]                [Send Buffer]
                            ^                             |
                      Batch.irecv                   Batch.isend
                            ^                             |
               --- prior rank --------------- next rank ---
                                                    |
                                             [Overflow Sinks]

Data flows from samplers or upstream ranks into the active batch,
through the dynamics step, and out to downstream ranks or sinks.


Pre-allocated communication buffers
-----------------------------------

Communication buffers are configured via
:class:`~nvalchemi.dynamics.BufferConfig`:

.. code-block:: python

   from nvalchemi.dynamics import DemoDynamics, BufferConfig

   buffer_config = BufferConfig(
       num_systems=64,    # max graphs in buffer
       num_nodes=2000,    # max total atoms
       num_edges=10000,   # max total edges
   )

   dynamics = DemoDynamics(
       model=model,
       dt=1.0,
       buffer_config=buffer_config,
   )

**Lazy initialization.** Buffers are created on the first step via
``_ensure_buffers()``. The first concrete batch serves as a template
for attribute keys, dtypes, and trailing shapes (e.g., hidden
dimensions). This lazy approach is necessary because the attribute
schema is not known until a real batch appears.

**The buffer lifecycle.** Communication buffers follow a
``Batch.empty()`` -> ``put()`` -> ``defrag()`` -> ``zero()`` cycle:

1. :meth:`Batch.empty() <nvalchemi.data.Batch.empty>` allocates storage
   with zero graphs but full capacity.
2. :meth:`Batch.put() <nvalchemi.data.Batch.put>` copies selected
   graphs from a source batch using Warp GPU kernels.
3. :meth:`Batch.defrag() <nvalchemi.data.Batch.defrag>` compacts the
   source batch in-place after extraction.
4. :meth:`Batch.zero() <nvalchemi.data.Batch.zero>` resets occupancy
   while preserving allocated memory.

.. warning::

   :meth:`Batch.put() <nvalchemi.data.Batch.put>` uses Warp GPU kernels
   that only copy **float32** attributes. Integer and other dtypes may
   need separate handling.

.. note::

   Adjacent stages in a :class:`~nvalchemi.dynamics.DistributedPipeline`
   must have identical ``BufferConfig`` values. This is validated
   during ``setup()``.


The communication protocol
--------------------------

:class:`~nvalchemi.dynamics.DistributedPipeline` uses a four-phase step
to coordinate data flow between ranks:

1. **_prestep_sync_buffers()**: Zeros the send buffer, posts ``irecv``
   from the prior rank. In sync mode, the receive completes inline. In
   async modes, the handle is stored for later completion.

2. **_complete_pending_recv()**: Waits on the deferred receive, routes
   data through the recv buffer into the active batch, and drains
   overflow sinks to backfill any available capacity.

3. **step()**: The dynamics integration step (forward pass, pre_update,
   post_update, convergence check).

4. **_poststep_sync_buffers()**: Extracts converged samples into the
   send buffer (subject to back-pressure), sends to the next rank. On
   the final rank, writes to sinks instead.

**Communication modes:**

.. list-table::
   :widths: 20 80
   :header-rows: 1

   * - Mode
     - Behavior
   * - ``"sync"``
     - Blocking receive in ``_prestep_sync_buffers``. Simplest and most
       debuggable. Good for small pipelines.
   * - ``"async_recv"``
     - Deferred receive: ``irecv`` is posted in ``_prestep_sync_buffers``
       but ``wait()`` is called later in ``_complete_pending_recv``.
       Allows compute to overlap with communication. **Default mode.**
   * - ``"fully_async"``
     - Both send and receive are deferred. Sends from the previous step
       are drained at the start of the next ``_prestep_sync_buffers``.
       Maximum overlap, highest throughput.

**Deadlock prevention.** When no samples converge, an empty send buffer
is still sent so the downstream ``irecv`` completes. This ensures the
pipeline does not stall waiting for data that will never arrive.


Back-pressure
-------------

When a pre-allocated ``send_buffer`` has limited capacity:

- Only ``min(converged_count, remaining_capacity)`` samples are
  extracted.
- Excess converged samples remain in the active batch.
- ``step()`` treats them as no-ops: their positions and velocities are
  saved before the integrator and restored after (the ``active_mask``
  logic in :meth:`BaseDynamics.step() <nvalchemi.dynamics.BaseDynamics.step>`).
- An empty buffer is still sent for deadlock prevention.

Without ``BufferConfig``, all converged samples are sent without
capacity constraints (backward compatibility mode).


Data routing helpers
--------------------

The :class:`~nvalchemi.dynamics.base._CommunicationMixin` provides
several helper methods for routing data between buffers:

- **_recv_to_batch(incoming)**: Stages data through the recv buffer
  (if present) into the active batch via ``_buffer_to_batch``, then
  zeros the recv buffer.

- **_buffer_to_batch(incoming)**: Routes incoming data into the active
  batch. Three cases:

  1. No active batch exists: adopt the incoming batch directly.
  2. Room available: append via :meth:`to_data_list()
     <nvalchemi.data.Batch.to_data_list>` + :meth:`from_data_list()
     <nvalchemi.data.Batch.from_data_list>`.
  3. No room: overflow to sinks.

- **_batch_to_buffer(mask)**: Copies graduated samples from the active
  batch into the send buffer via :meth:`put()
  <nvalchemi.data.Batch.put>`, then defrags the active batch.

- **_overflow_to_sinks(batch, mask)**: Writes to the first non-full
  sink in priority order.

- **_drain_sinks_to_batch()**: Pulls samples from sinks back into the
  active batch when room is available.

.. note::

   ``_buffer_to_batch`` uses :meth:`to_data_list()
   <nvalchemi.data.Batch.to_data_list>` + :meth:`from_data_list()
   <nvalchemi.data.Batch.from_data_list>` for combining batches. This
   is O(N) Python-level reconstruction. In high-throughput pipelines,
   this can be a bottleneck compared to the Warp-accelerated
   ``put`` / ``defrag`` path.


Sample lifecycle
----------------

This section traces a sample's journey through three representative
workflows.

**Standalone BaseDynamics.run()**

.. code-block:: text

   Batch passed to run()
         |
         v
   loop for n_steps:
         |
         v
     pre_update --> compute --> post_update --> convergence check
         |
         v
   return batch

The simplest workflow: a batch is passed in, stepped for ``n_steps``
iterations, and returned.


**FusedStage with inflight batching**

.. code-block:: text

   1. sampler.build_initial_batch()
      creates batch with status=0, fmax=inf
         |
         v
   2. Each step:
      compute() --> per-sub-stage masked_update based on batch.status
         |
         v
   3. ConvergenceHook updates batch.status (e.g., 0 --> 1 --> 2)
         |
         v
   4. Every refill_frequency steps: _refill_check()
      - identifies graduated samples (status >= exit_status)
      - writes them to sinks
      - extracts remaining via index_select
      - requests replacements from sampler
      - appends them, rebuilds status/fmax tensors
         |
         v
   5. Terminates when sampler is exhausted and all graduated,
      or all samples reach exit_status

Samples migrate through sub-stages based on convergence, and graduated
samples are continuously replaced from the sampler.


**DistributedPipeline**

.. code-block:: text

   Rank 0 (first, inflight):
     - builds batch from sampler
     - runs step
     - sends converged downstream via _poststep_sync_buffers
     - refills from sampler
         |
         v (isend/irecv)
   Rank 1..N-1 (middle):
     - receives from prior rank via _prestep_sync_buffers
     - _complete_pending_recv routes data to active batch
     - runs step
     - sends converged downstream
         |
         v (isend/irecv)
   Rank N (final):
     - receives from prior rank
     - runs step
     - writes converged to sinks

   All ranks:
     - synchronize done flags via all_reduce(MAX)
     - loop terminates when all report done

Samples flow from the first rank through intermediate ranks to the
final rank, where they are persisted to sinks.


Data sinks
----------

Three :class:`~nvalchemi.dynamics.DataSink` implementations are
available:

- :class:`~nvalchemi.dynamics.GPUBuffer`: Pre-allocates on first write.
  Uses :meth:`Batch.put() <nvalchemi.data.Batch.put>` internally. Has a
  known performance limitation: ``read()`` falls back to
  :meth:`to_data_list() <nvalchemi.data.Batch.to_data_list>` instead
  of :meth:`index_select() <nvalchemi.data.Batch.index_select>` due to
  Warp int32/int64 dtype incompatibility.

- :class:`~nvalchemi.dynamics.HostMemory`: CPU-resident, decomposes
  batches into :class:`~nvalchemi.data.AtomicData` lists.

- :class:`~nvalchemi.dynamics.ZarrData`: Disk-backed, delegates to
  :class:`~nvalchemi.data.AtomicDataZarrWriter`.

Sinks are tried in priority order; the first non-full sink receives
the data.
