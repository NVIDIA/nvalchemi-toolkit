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

   from nvalchemi.dynamics import DemoDynamics
   from nvalchemi.dynamics.base import BufferConfig

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

**The buffer lifecycle.**  Communication buffers (``send_buffer`` and
``recv_buffer``) follow a ``Batch.empty()`` → ``put()`` → ``zero()``
cycle:

1. :meth:`Batch.empty() <nvalchemi.data.Batch.empty>` allocates storage
   with zero graphs but full capacity.
2. :meth:`Batch.put() <nvalchemi.data.Batch.put>` copies selected
   graphs from a source batch using Warp GPU kernels.
3. :meth:`Batch.zero() <nvalchemi.data.Batch.zero>` resets occupancy
   while preserving allocated memory.

After ``put()`` extracts converged graphs into the send buffer, the
*active batch* (the working simulation state) is rebuilt without those
graphs via :meth:`Batch.trim() <nvalchemi.data.Batch.trim>`, which
returns a new :class:`~nvalchemi.data.Batch` with tight storage — no
padding, no trailing buffer slots.

**``trim`` vs ``defrag``.**  :meth:`~nvalchemi.data.Batch.defrag`
compacts data to the front of a pre-allocated buffer **in-place**,
preserving the physical capacity for further ``put`` / ``defrag``
cycles.  :meth:`~nvalchemi.data.Batch.trim` instead creates a
brand-new batch whose tensors are sized to exactly fit the remaining
graphs.  Use ``defrag`` for reusable communication buffers; use
``trim`` for batches that will be consumed directly by a model or
integrator and must have self-consistent tensor shapes.

.. warning::

   :meth:`Batch.put() <nvalchemi.data.Batch.put>` uses Warp GPU kernels
   that only copy **float32** attributes. Integer and other dtypes may
   need separate handling.

.. note::

   Every stage in a :class:`~nvalchemi.dynamics.DistributedPipeline`
   must provide a ``BufferConfig``, and adjacent stages must have
   identical values. Both constraints are validated during
   ``setup()``.


The communication protocol
--------------------------

:class:`~nvalchemi.dynamics.DistributedPipeline` uses a five-phase step
to coordinate data flow between ranks:

1. **_prestep_sync_buffers()**: Zeros the send buffer and posts
   ``irecv`` from the prior rank (using a pre-computed template for
   correct buffer shapes).  In sync mode the receive completes inline;
   in async modes the handle is stored for later completion.

2. **_complete_pending_recv()**: Waits on the deferred receive, routes
   data through the recv buffer into the active batch (stripping buffer
   padding via ``to_data_list`` / ``from_data_list``), and drains
   overflow sinks to backfill any available capacity.

3. **step()**: The dynamics integration step (forward pass, pre_update,
   post_update, convergence check).

4. **_poststep_sync_buffers()**: Extracts converged samples into the
   send buffer via ``_populate_send_buffer`` (subject to back-pressure).
   The send buffer is then **unconditionally sent** to the next rank.
   On the final rank, converged samples are extracted via
   ``_remove_converged_final_stage`` and written to sinks.

5. **_sync_done_flags()**: All ranks synchronize ``done`` flags via
   ``dist.all_reduce(MAX)``; the loop terminates when every rank
   reports done.

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

**Template sharing.**  During ``setup()``, the pipeline computes recv
templates for every stage via ``_share_templates()``.  Since all stages
are available on every rank, this is done locally without inter-rank
communication.  Inflight (first) stages build their initial batch from
the sampler; downstream stages derive templates from their upstream
neighbour.  Templates are used to pre-allocate correctly shaped
``irecv`` buffers.

**Non-blocking receives.**  ``_BatchRecvHandle.wait()`` uses
non-blocking ``dist.irecv`` for all message components (metadata,
segment lengths, and bulk tensor data) and waits on all handles at the
end.  This enables better overlap with computation compared to the
previous blocking ``dist.recv`` approach.

**Deadlock prevention.** When no samples converge, an empty send buffer
is still sent so the downstream ``irecv`` completes.  This ensures the
pipeline does not stall waiting for data that will never arrive.


Back-pressure
-------------

When the ``send_buffer`` capacity is limited:

- Only ``min(converged_count, remaining_capacity)`` samples are
  extracted via ``_populate_send_buffer``.
- The active batch is replaced by a tight copy (via ``trim()``) that
  excludes the graduated samples.
- Excess converged samples that did not fit in the send buffer remain
  in the active batch.
- ``step()`` treats them as no-ops: their positions and velocities are
  saved before the integrator and restored after (the ``active_mask``
  logic in :meth:`BaseDynamics.step() <nvalchemi.dynamics.BaseDynamics.step>`).
- The send buffer is sent **unconditionally** every step for deadlock
  prevention, even when empty.


Data routing helpers
--------------------

The :class:`~nvalchemi.dynamics.base._CommunicationMixin` provides
several helper methods for routing data between buffers:

- **_recv_to_batch(incoming)**: Stages data through the recv buffer
  into the active batch via ``_buffer_to_batch``, then zeros the
  recv buffer.

- **_buffer_to_batch(incoming)**: Routes incoming data into the active
  batch.  All code paths reconstruct the batch via
  :meth:`to_data_list() <nvalchemi.data.Batch.to_data_list>` +
  :meth:`from_data_list() <nvalchemi.data.Batch.from_data_list>` to
  strip buffer padding and produce tight tensors.  Three cases:

  1. No active batch exists: adopt the incoming data directly.
  2. Room available: append to the existing active batch.
  3. No room: overflow to sinks.

- **_batch_to_buffer(mask)**: Copies graduated samples from the active
  batch into the send buffer via :meth:`put()
  <nvalchemi.data.Batch.put>`, then replaces the active batch with a
  tight copy via :meth:`trim() <nvalchemi.data.Batch.trim>` (or sets
  it to ``None`` if every graph was graduated).

- **_populate_send_buffer(converged_indices)**: Creates a boolean mask
  from converged indices and delegates to ``_batch_to_buffer``.  Does
  not send — the caller issues ``isend`` separately.

- **_remove_converged_final_stage(converged_indices)**: On the final
  stage, extracts converged graphs via ``index_select``, removes them
  from the active batch, and writes them to sinks.

- **_manage_send_handle(handle)**: Stores the send handle for later
  draining (``fully_async``) or waits on it immediately (other modes).

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
