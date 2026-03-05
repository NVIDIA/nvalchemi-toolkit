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

from typing import TYPE_CHECKING

from nvalchemi.dynamics.hooks._base import _ObserverHook

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import BaseDynamics
    from nvalchemi.dynamics.sinks import DataSink

__all__ = ["SnapshotHook"]


class SnapshotHook(_ObserverHook):
    """Save a snapshot of the active batch to a :class:`DataSink` at a given frequency.

    This hook writes the **full** batch state — positions, velocities,
    forces, energies, and any other tensors present on the
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

    Because ``SnapshotHook`` inherits from :class:`_ObserverHook`, it
    fires at :attr:`~HookStageEnum.AFTER_STEP` — after all integrator
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
    stage : HookStageEnum
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

    def __init__(self, sink: DataSink, frequency: int = 1) -> None:
        super().__init__(frequency=frequency)
        self.sink = sink

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Write the current batch state to the configured sink.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data.
        dynamics : BaseDynamics
            The dynamics engine instance (provides ``step_count``
            and model metadata).

        Raises
        ------
        NotImplementedError
            This hook is not yet implemented.
        """
        raise NotImplementedError("SnapshotHook is not yet implemented.")
