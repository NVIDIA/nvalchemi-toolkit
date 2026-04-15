# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Snapshot hook for saving batch state to a data sink.

Provides :class:`SnapshotHook`, which periodically writes the full batch
state to a :class:`~nvalchemi.dynamics.sinks.DataSink` (GPU buffer, host
memory, or Zarr store).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import torch

from nvalchemi.data import Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks._context import HookContext

if TYPE_CHECKING:
    from nvalchemi.dynamics.sinks import DataSink

__all__ = ["SnapshotHook", "ConvergedSnapshotHook"]


class SnapshotHook:
    """Save a snapshot of the active batch to a :class:`DataSink` at a given frequency.

    This hook writes the **full** batch state — positions, velocities,
    forces, energy, and any other tensors present on the
    :class:`~nvalchemi.data.Batch` — to the configured sink every
    ``frequency`` steps.  It is the primary mechanism for recording
    trajectories and creating restart checkpoints during dynamics runs.

    The hook delegates serialization entirely to the
    :class:`~nvalchemi.dynamics.sinks.DataSink` interface, meaning the
    same ``SnapshotHook`` instance works with any backend:

    * :class:`~nvalchemi.dynamics.sinks.GPUBuffer` — pre-allocated
      device memory for high-speed, in-simulation buffering.
    * :class:`~nvalchemi.dynamics.sinks.HostMemory` — CPU-resident
      list-of-:class:`AtomicData` storage, useful for staging before
      disk I/O.
    * :class:`~nvalchemi.dynamics.sinks.ZarrData` — persistent,
      Zarr-backed storage with CSR-style layout for variable-length
      graph data; supports local, in-memory, and remote (S3/GCS)
      stores.

    ``SnapshotHook`` fires at :attr:`~DynamicsStage.AFTER_STEP` — after all integrator
    updates, force clamping, and convergence checks have completed —
    guaranteeing that the snapshot reflects the fully resolved state
    for each recorded step.

    Parameters
    ----------
    sink : DataSink
        The storage backend to write snapshots to.
    frequency : int, optional
        Write a snapshot every ``frequency`` steps. Default ``1``
        (every step).

    Attributes
    ----------
    sink : DataSink
        The storage backend.
    frequency : int
        Snapshot frequency in steps.
    stage : DynamicsStage
        Fixed to ``AFTER_STEP``.

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import SnapshotHook
    >>> from nvalchemi.dynamics.sinks import HostMemory
    >>> sink = HostMemory(capacity=10_000)
    >>> hook = SnapshotHook(sink=sink, frequency=10)
    >>> dynamics = DemoDynamics(model=model, n_steps=1000, dt=0.5, hooks=[hook])
    >>> dynamics.run(batch)  # 100 snapshots written
    >>> trajectory = sink.read()

    Notes
    -----
    * The hook does **not** clone the batch before writing.  Whether
      data is copied depends on the sink implementation (e.g.
      :class:`HostMemory` moves to CPU; :class:`GPUBuffer` copies
      into pre-allocated slots).
    * For long simulations, prefer :class:`ZarrData` to avoid
      accumulating the full trajectory in memory.
    * When used inside a :class:`FusedStage`, the snapshot includes
      samples at all status codes in a single write.
    """

    def __init__(
        self,
        sink: DataSink,
        frequency: int = 1,
        stage: Enum = DynamicsStage.AFTER_STEP,
    ) -> None:
        self.sink = sink
        self.stage = stage
        self.frequency = frequency

    @torch.compiler.disable
    def _write_snapshot(self, batch: Batch) -> None:
        """Write the current batch state to the configured sink."""
        self.sink.write(batch)

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Write the current batch state to the configured sink."""
        self._write_snapshot(ctx.batch)


class ConvergedSnapshotHook:
    """Write only newly converged samples to a :class:`DataSink`.

    Fires at :attr:`~DynamicsStage.ON_CONVERGE` and uses the converged
    sample indices (available via ``dynamics._last_converged``) to build
    a boolean mask passed to :meth:`DataSink.write`.  Only samples that
    just satisfied the convergence criterion are written — samples that
    converged on earlier steps are not re-written.

    This is the recommended hook for persisting optimized structures to
    Zarr in a :class:`FusedStage` pipeline, where multiple relaxations
    run concurrently and structures converge at different steps.

    Parameters
    ----------
    sink : DataSink
        The storage backend to write converged samples to.
        :class:`~nvalchemi.dynamics.sinks.ZarrData` is the typical
        choice for persistent storage.
    frequency : int, optional
        Execute every ``frequency`` steps. Default ``1`` (check every
        step that convergence fires).
    stage : Enum, optional
        The stage at which to run this hook. Default is
        ``DynamicsStage.ON_CONVERGE``.

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import ConvergedSnapshotHook
    >>> from nvalchemi.dynamics.sinks import ZarrData
    >>> sink = ZarrData(store="converged.zarr", capacity=100_000)
    >>> hook = ConvergedSnapshotHook(sink=sink)
    >>> dynamics.register_hook(hook)
    """

    def __init__(
        self,
        sink: DataSink,
        frequency: int = 1,
        stage: Enum = DynamicsStage.ON_CONVERGE,
    ) -> None:
        self.sink = sink
        self.frequency = frequency
        self.stage = stage

    # Neighbor data keys to strip before writing to the sink.
    # Neighbor tensors are ephemeral (rebuilt by NeighborListHook each step)
    # and their K-dimension can change between rebuilds due to adaptive
    # sizing, causing shape mismatches when the sink concatenates snapshots.
    _NEIGHBOR_KEYS = frozenset(
        {
            "neighbor_matrix",
            "num_neighbors",
            "neighbor_matrix_shifts",
            "neighbor_list",
            "neighbor_list_shifts",
        }
    )

    @torch.compiler.disable
    def _write_converged(self, batch: Batch, mask: torch.Tensor | None) -> None:
        """Write converged samples to the configured sink.

        Neighbor data is stripped before writing because its K-dimension
        can vary between adaptive rebuilds, causing shape mismatches when
        the sink later concatenates snapshots into a single Batch.

        A sub-batch is created via :meth:`Batch.index_select` so the
        live ``batch`` object is never mutated — downstream hooks that
        read neighbor data after ``ON_CONVERGE`` see an intact batch.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data.
        mask : torch.Tensor | None
            Boolean mask of converged samples, or None if none converged.
        """
        if mask is None or not mask.any():
            return
        # Build a sub-batch of only converged systems so we never
        # mutate the live batch object.
        indices = torch.nonzero(mask, as_tuple=True)[0]
        _ = batch.batch_ptr  # trigger lazy init for SegmentedLevelStorage
        sub_batch = batch.index_select(indices)
        # Strip ephemeral neighbor data before writing to avoid
        # variable-width concatenation failures in the sink.
        for key in self._NEIGHBOR_KEYS:
            try:
                del sub_batch[key]
            except (KeyError, IndexError):
                pass
        self.sink.write(sub_batch)

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Write converged samples to the configured sink."""
        self._write_converged(ctx.batch, ctx.converged_mask)
