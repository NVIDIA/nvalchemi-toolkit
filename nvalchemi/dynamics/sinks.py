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
Data sink abstractions for storing and retrieving batched atomic data.

This module provides storage backends for Batch data used in dynamics
simulations. Implementations include GPU buffers, CPU memory, and
disk-backed Zarr storage.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from torch import distributed as dist

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.backends.zarr import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
    StoreLike,
)


class DataSink(ABC):
    """
    Abstract base class for local storage of Batch data.

    DataSink provides a unified interface for storing and retrieving
    batched atomic data. Implementations can target different storage
    backends such as GPU memory, CPU memory, or disk.

    Attributes
    ----------
    capacity : int
        Maximum number of samples that can be stored.

    Methods
    -------
    write(batch)
        Store a batch of data.
    read()
        Retrieve all stored data as a Batch.
    drain()
        Read all stored data and clear the sink.
    zero()
        Clear all stored data.
    __len__()
        Return the number of samples currently stored.

    Examples
    --------
    >>> sink = HostMemory(capacity=100)
    >>> sink.write(batch)
    >>> len(sink)
    2
    >>> retrieved = sink.read()
    """

    @abstractmethod
    def write(self, batch: Batch, mask: torch.Tensor | None = None) -> None:
        """
        Store a batch of atomic data.

        Parameters
        ----------
        batch : Batch
            The batch of atomic data to store.
        mask : torch.Tensor | None, optional
            Boolean tensor of shape ``(batch.num_graphs,)`` indicating
            which samples to copy (``True`` = copy). If ``None``, all
            samples are copied. Default is ``None``.

        Raises
        ------
        RuntimeError
            If the buffer is full and cannot accept more data.
        """
        ...

    @abstractmethod
    def read(self) -> Batch:
        """
        Retrieve all stored data as a single Batch.

        Returns
        -------
        Batch
            A batch containing all stored atomic data.

        Raises
        ------
        RuntimeError
            If no data has been stored (buffer is empty).
        """
        ...

    @abstractmethod
    def zero(self) -> None:
        """
        Clear all stored data and reset the buffer.

        After calling this method, `len(self)` returns 0.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """
        Return the number of samples currently stored.

        Returns
        -------
        int
            Number of atomic data samples in the buffer.
        """
        ...

    @property
    @abstractmethod
    def capacity(self) -> int:
        """
        Return the maximum storage capacity.

        Returns
        -------
        int
            Maximum number of samples that can be stored.
        """
        ...

    @property
    def is_full(self) -> bool:
        """
        Check if the buffer has reached capacity.

        Returns
        -------
        bool
            True if the buffer is at or over capacity, False otherwise.
        """
        return len(self) >= self.capacity

    def drain(self) -> Batch:
        """
        Read all stored samples and clear the sink.

        This is equivalent to calling :meth:`read` followed by
        :meth:`zero`, but subclasses may override for a more efficient
        atomic operation.

        Returns
        -------
        Batch
            All samples that were stored in the sink.

        Raises
        ------
        RuntimeError
            If the sink is empty.
        """
        batch = self.read()
        self.zero()
        return batch

    @property
    def local_rank(self) -> int:
        """Return the local rank of this data sink."""
        rank = 0
        if dist.is_initialized():
            rank = dist.get_node_local_rank()
        return rank

    @property
    def global_rank(self) -> int:
        """Return the global rank of this data sink."""
        rank = 0
        if dist.is_initialized():
            rank = dist.get_global_rank()
        return rank


class GPUBuffer(DataSink):
    """GPU-resident buffer for storing batched atomic data.

    This buffer lazily pre-allocates a :class:`Batch` with fixed maximum
    sizes for atoms and edges on the first :meth:`write` call.  The
    incoming batch serves as a template for attribute keys and dtypes,
    ensuring all fields are preserved (not just positions and
    atomic_numbers).

    Parameters
    ----------
    capacity : int
        Maximum number of samples (graphs) to store.
    max_atoms : int
        Maximum number of atoms per sample.
    max_edges : int
        Maximum number of edges per sample.
    device : torch.device | str, optional
        CUDA device to store data on. Default is "cuda".

    Attributes
    ----------
    capacity : int
        Maximum storage capacity.
    device : torch.device
        Target CUDA device for stored tensors.

    Examples
    --------
    >>> buffer = GPUBuffer(capacity=100, max_atoms=50, max_edges=200, device="cuda:0")
    >>> buffer.write(batch)
    >>> len(buffer)
    2
    >>> retrieved = buffer.read()
    """

    def __init__(
        self,
        capacity: int,
        max_atoms: int,
        max_edges: int,
        device: torch.device | str = "cuda",
    ) -> None:
        """Initialize the GPU buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of samples (graphs) to store.
        max_atoms : int
            Maximum number of atoms per sample.
        max_edges : int
            Maximum number of edges per sample.
        device : torch.device | str, optional
            CUDA device to store data on. Default is "cuda".

        Raises
        ------
        RuntimeError
            If CUDA is not available or a non-CUDA device is specified.
        """
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPUBuffer requires available CUDA devices:"
                f" found CUDA available: {torch.cuda.is_available()}"
                f" with device count={torch.cuda.device_count()}"
            )
        if isinstance(device, str) and "cuda" not in device:
            raise RuntimeError(f"GPUBuffer requires a CUDA device, got: '{device}'")
        if isinstance(device, torch.device) and "cuda" not in device.type:
            raise RuntimeError(
                f"GPUBuffer requires a CUDA device, got: '{device.type}'"
            )

        self._capacity = capacity
        self._max_atoms = max_atoms
        self._max_edges = max_edges
        self._device = torch.device(device) if isinstance(device, str) else device
        self._buffer: Batch | None = None
        # Pre-allocated masks to avoid allocation in hot path
        self._copied_mask: torch.Tensor | None = None
        self._dest_mask: torch.Tensor | None = None

    def _ensure_buffer(self, template: Batch) -> None:
        """Create the internal Batch buffer on first use.

        Allocates a pre-sized buffer using :meth:`Batch.empty` with
        capacity derived from constructor parameters. The template
        batch provides attribute keys and dtypes.

        Parameters
        ----------
        template : Batch
            A concrete batch to derive attribute keys and dtypes from.
        """
        if self._buffer is not None:
            return
        self._buffer = Batch.empty(
            num_systems=self._capacity,
            num_nodes=self._capacity * self._max_atoms,
            num_edges=self._capacity * self._max_edges,
            template=template,
            device=self._device,
        )
        # Pre-allocate masks for system-level occupancy tracking
        self._dest_mask = torch.zeros(
            self._capacity, dtype=torch.bool, device=self._device
        )
        self._copied_mask = torch.zeros(
            template.num_graphs, dtype=torch.bool, device=self._device
        )
        # Trigger lazy init of _batch_ptr for all groups so zero() can preserve it
        for group in self._buffer._storage.groups.values():
            if hasattr(group, "_lazy_init_batch_ptr"):
                group._lazy_init_batch_ptr()
        # Extend _batch_ptr to capacity + 2 (Batch.empty uses capacity + 1,
        # but compute_put_per_system_fit_mask requires capacity + 2)
        self._restore_batch_ptr_capacity()

    def _restore_batch_ptr_capacity(self) -> None:
        """Restore ``_batch_ptr`` to full pre-allocated capacity after put.

        :meth:`SegmentedLevelStorage.put` trims ``_batch_ptr`` to the number
        of active segments via ``.clone()``, which destroys the headroom
        needed for subsequent appends.  This method re-extends each
        segmented group's ``_batch_ptr`` to ``capacity + 2`` while
        preserving the meaningful prefix written by the Warp kernels.

        The ``+2`` accounts for the formula used by
        :meth:`SegmentedLevelStorage.compute_put_per_system_fit_mask`:
        it requires ``num_dest_segments + n_seg + 2`` entries to
        accommodate segment boundaries plus safety margin.
        """
        if self._buffer is None:
            return
        # Need capacity + 2 to satisfy compute_put_per_system_fit_mask formula
        batch_ptr_cap = self._capacity + 2
        for group in self._buffer._storage.groups.values():
            if not hasattr(group, "segment_lengths"):
                continue  # skip UniformLevelStorage
            bp = group._batch_ptr
            if bp is not None and bp.shape[0] < batch_ptr_cap:
                new_bp = torch.zeros(batch_ptr_cap, dtype=bp.dtype, device=bp.device)
                new_bp[: bp.shape[0]] = bp
                group._batch_ptr = new_bp

    def write(self, batch: Batch, mask: torch.Tensor | None = None) -> None:
        """Store atomic data into the buffer.

        When *mask* is provided, only samples where ``mask[i]`` is ``True``
        are copied into the buffer.  When *mask* is ``None``, all samples
        in *batch* are copied.

        Uses :meth:`Batch.put` for efficient in-place copying without
        tensor allocation.

        This method will set values for ``_copied_mask`` and ``_dest_mask``.

        Parameters
        ----------
        batch : Batch
            The source batch of atomic data.
        mask : torch.Tensor | None, optional
            Boolean tensor of shape ``(batch.num_graphs,)`` indicating
            which samples to copy (``True`` = copy).  If ``None``, all
            samples are copied.

        Raises
        ------
        RuntimeError
            If adding the selected samples would exceed capacity, or if
            a system exceeds the configured max_atoms or max_edges limits.
        ValueError
            If mask length does not match batch.num_graphs.
        """
        num_total = batch.num_graphs or 0
        if num_total == 0:
            return

        # Build mask if not provided
        if mask is None:
            mask = torch.ones(num_total, dtype=torch.bool, device=batch.device)
        else:
            mask = mask.to(device=batch.device, dtype=torch.bool)
            if mask.shape[0] != num_total:
                raise ValueError(
                    f"mask length {mask.shape[0]} != num_graphs {num_total}"
                )

        # Ensure buffer is allocated with full capacity (lazy init on first write)
        self._ensure_buffer(template=batch)

        # Count how many graphs we're trying to write
        num_to_write = int(mask.sum().item())
        if num_to_write == 0:
            return

        # Validate graph capacity
        current_count = len(self)
        if current_count + num_to_write > self._capacity:
            raise RuntimeError(
                f"Buffer is full. Cannot add {num_to_write} samples to buffer "
                f"with {current_count}/{self._capacity} samples."
            )

        # Validate atom capacity for masked graphs
        nodes_per_graph = batch.num_nodes_per_graph
        max_atoms_in_batch = int(nodes_per_graph[mask].max().item())
        if max_atoms_in_batch > self._max_atoms:
            raise RuntimeError(
                f"Atom capacity exceeded: a system has {max_atoms_in_batch} atoms "
                f"but buffer max_atoms={self._max_atoms}"
            )

        # Validate edge capacity for masked graphs (only if edges exist in batch)
        if self._max_edges > 0 and batch.num_edges > 0:
            edges_per_graph = batch.num_edges_per_graph
            max_edges_in_batch = int(edges_per_graph[mask].max().item())
            if max_edges_in_batch > self._max_edges:
                raise RuntimeError(
                    f"Edge capacity exceeded: a system has {max_edges_in_batch} edges "
                    f"but buffer max_edges={self._max_edges}"
                )

        # Allocate fresh _copied_mask
        self._copied_mask = torch.zeros(
            num_total, dtype=torch.bool, device=self._device
        )

        self._buffer.put(
            batch,
            mask,
            copied_mask=self._copied_mask,
            dest_mask=self._dest_mask,
        )

        # Restore _batch_ptr capacity after put() trims it
        self._restore_batch_ptr_capacity()

    def read(self) -> Batch:
        """Retrieve stored (non-padding) data as a single Batch.

        The pre-allocated buffer may have more capacity than stored
        samples.  This method extracts only the filled graphs,
        excluding zero-padded slots.

        Returns
        -------
        Batch
            A batch containing the stored atomic data (no padding).

        Raises
        ------
        RuntimeError
            If the buffer is empty.

        Notes
        -----
        The current implementation uses ``to_data_list`` for extraction,
        which is slower than ``index_select``. Direct ``index_select``
        fails due to Warp dtype incompatibility (int32 batch_ptr vs int64
        expected by Warp kernels). This is a known limitation that could
        be optimized once Warp dtype handling is fixed upstream.
        """
        if self._buffer is None or len(self) == 0:
            raise RuntimeError("Cannot read from empty buffer.")
        if len(self) == self._capacity:
            # Buffer is full — return clone of entire buffer
            return self._buffer.clone()
        # Extract only the filled portion via to_data_list.
        # NOTE: index_select would be faster, but fails due to Warp dtype
        # issues with int32 batch_ptr. See docstring Notes for details.
        data_list = self._buffer.to_data_list()[: len(self)]
        return Batch.from_data_list(data_list, device=self._device)

    def zero(self) -> None:
        """Clear all stored data and reset the buffer.

        Zeros all subtensors within the pre-allocated buffer while
        preserving the data structure and allocated memory.  This avoids
        re-allocation on the next :meth:`write` and keeps the buffer
        shape intact for ``isend``/``irecv`` symmetry.
        """
        # Reset occupancy masks
        if self._dest_mask is not None:
            self._dest_mask.zero_()
        if self._copied_mask is not None:
            self._copied_mask.zero_()
        if self._buffer is None:
            return
        # Delegate buffer reset to Batch.zero() which properly handles
        # both UniformLevelStorage and SegmentedLevelStorage bookkeeping.
        self._buffer.zero()
        # Clear _num_segments cache (not handled by Batch.zero)
        for group in self._buffer._storage.groups.values():
            if hasattr(group, "_num_segments"):
                object.__delattr__(group, "_num_segments")

    def __len__(self) -> int:
        """Return the number of samples currently stored.

        Returns
        -------
        int
            Number of atomic data samples in the buffer.
        """
        if self._buffer is None:
            return 0
        return self._buffer.num_graphs

    @property
    def capacity(self) -> int:
        """Return the maximum storage capacity.

        Returns
        -------
        int
            Maximum number of samples that can be stored.
        """
        return self._capacity

    @property
    def device(self) -> torch.device:
        """Return the storage device.

        Returns
        -------
        torch.device
            Device where data is stored.
        """
        return self._device


class HostMemory(DataSink):
    """
    CPU-resident buffer for storing batched atomic data.

    This buffer ensures all data is stored on CPU memory, regardless
    of the input batch's device. It is useful for staging data before
    disk I/O or for CPU-side processing.

    Parameters
    ----------
    capacity : int
        Maximum number of samples to store.

    Attributes
    ----------
    capacity : int
        Maximum storage capacity.

    Examples
    --------
    >>> host_buffer = HostMemory(capacity=1000)
    >>> host_buffer.write(gpu_batch)  # Data moved to CPU
    >>> cpu_batch = host_buffer.read()
    """

    def __init__(self, capacity: int) -> None:
        """
        Initialize the host memory buffer.

        Parameters
        ----------
        capacity : int
            Maximum number of samples to store.
        """
        self._capacity = capacity
        self._data_list: list[AtomicData] = []
        self._device = torch.device("cpu")

    def write(self, batch: Batch, mask: torch.Tensor | None = None) -> None:
        """
        Store a batch of atomic data on CPU.

        Decomposes the batch into individual AtomicData objects,
        moves them to CPU, and appends to internal storage.

        Parameters
        ----------
        batch : Batch
            The batch of atomic data to store.
        mask : torch.Tensor | None, optional
            Boolean tensor of shape ``(batch.num_graphs,)`` indicating
            which samples to write (``True`` = write). If ``None``, all
            samples are written. Default is ``None``.

        Raises
        ------
        RuntimeError
            If adding the selected samples would exceed capacity.
        ValueError
            If mask length does not match batch.num_graphs.
        """
        num_total = batch.num_graphs or 0
        if num_total == 0:
            return

        # Apply mask to select samples
        if mask is not None:
            mask = mask.to(device=batch.device, dtype=torch.bool)
            if mask.shape[0] != num_total:
                raise ValueError(
                    f"mask length {mask.shape[0]} != num_graphs {num_total}"
                )
            num_selected = int(mask.sum().item())
            if num_selected == 0:
                return
            if num_selected < num_total:
                indices = torch.nonzero(mask, as_tuple=True)[0]
                _ = batch.ptr  # trigger lazy init for SegmentedLevelStorage
                batch = batch.index_select(indices)

        data_list = batch.to_data_list()
        if len(self._data_list) + len(data_list) > self._capacity:
            raise RuntimeError(
                f"Buffer is full. Cannot add {len(data_list)} samples "
                f"to buffer with {len(self._data_list)}/{self._capacity} samples."
            )
        # Move data to CPU before storing
        for data in data_list:
            self._data_list.append(data.to(self._device))

    def read(self) -> Batch:
        """
        Retrieve all stored data as a CPU-resident Batch.

        Returns
        -------
        Batch
            A batch containing all stored atomic data on CPU.

        Raises
        ------
        RuntimeError
            If the buffer is empty.
        """
        if len(self._data_list) == 0:
            raise RuntimeError("Cannot read from empty buffer.")
        return Batch.from_data_list(self._data_list, device=self._device)

    def zero(self) -> None:
        """Clear all stored data and reset the buffer."""
        self._data_list.clear()

    def __len__(self) -> int:
        """
        Return the number of samples currently stored.

        Returns
        -------
        int
            Number of atomic data samples in the buffer.
        """
        return len(self._data_list)

    @property
    def capacity(self) -> int:
        """
        Return the maximum storage capacity.

        Returns
        -------
        int
            Maximum number of samples that can be stored.
        """
        return self._capacity


class ZarrData(DataSink):
    """
    Zarr-backed storage for batched atomic data.

    This sink persists atomic data using the Zarr format, supporting
    both local filesystem and remote/in-memory stores via ``StoreLike``.
    Delegates serialization to :class:`AtomicDataZarrWriter` for
    efficient, amortized I/O with CSR-style pointer arrays.

    Supports any zarr-compatible store: filesystem paths (str or Path),
    zarr Store instances (LocalStore, MemoryStore, FsspecStore for remote
    storage like S3/GCS), StorePath, or dict for in-memory buffers.

    Parameters
    ----------
    store : StoreLike
        Any zarr-compatible store: filesystem path (str or Path), zarr Store
        instance, StorePath, or dict for in-memory buffer storage.
    capacity : int, optional
        Maximum number of samples to store. Default is 1,000,000.

    Attributes
    ----------
    capacity : int
        Maximum storage capacity.
    store : StoreLike
        The backing zarr store.

    Examples
    --------
    >>> zarr_sink = ZarrData("/path/to/store", capacity=100000)
    >>> zarr_sink.write(batch)
    >>> loaded_batch = zarr_sink.read()

    Using an in-memory store:

    >>> zarr_sink = ZarrData({}, capacity=1000)  # dict acts as memory store
    """

    def __init__(self, store: StoreLike, capacity: int = 1_000_000) -> None:
        """
        Initialize the Zarr data sink.

        Parameters
        ----------
        store : StoreLike
            Any zarr-compatible store: filesystem path (str or Path), zarr Store
            instance, StorePath, or dict for in-memory buffer storage.
        capacity : int, optional
            Maximum number of samples to store. Default is 1,000,000.
        """
        self._store: StoreLike = store
        self._capacity = capacity
        self._count = 0
        self._written_once = False
        # Lazily create writer — don't create store until first write
        self._writer: AtomicDataZarrWriter | None = None

    def _get_writer(self) -> AtomicDataZarrWriter:
        """Get or create the AtomicDataZarrWriter instance.

        Returns
        -------
        AtomicDataZarrWriter
            The writer instance for this sink.
        """
        if self._writer is None:
            self._writer = AtomicDataZarrWriter(self._store)
        return self._writer

    def write(self, batch: Batch, mask: torch.Tensor | None = None) -> None:
        """
        Store a batch of atomic data to Zarr.

        Uses :class:`AtomicDataZarrWriter` for efficient bulk writes.
        The first write uses ``write()`` (creates store), subsequent
        writes use ``append()`` (extends existing store).

        Parameters
        ----------
        batch : Batch
            The batch of atomic data to store.
        mask : torch.Tensor | None, optional
            Boolean tensor of shape ``(batch.num_graphs,)`` indicating
            which samples to write (``True`` = write). If ``None``, all
            samples are written. Default is ``None``.

        Raises
        ------
        RuntimeError
            If adding the selected samples would exceed capacity.
        ValueError
            If mask length does not match batch.num_graphs.
        """
        num_total = batch.num_graphs or 0
        if num_total == 0:
            return  # Nothing to write

        # Apply mask to select samples
        if mask is not None:
            mask = mask.to(device=batch.device, dtype=torch.bool)
            if mask.shape[0] != num_total:
                raise ValueError(
                    f"mask length {mask.shape[0]} != num_graphs {num_total}"
                )
            num_selected = int(mask.sum().item())
            if num_selected == 0:
                return
            if num_selected < num_total:
                indices = torch.nonzero(mask, as_tuple=True)[0]
                _ = batch.ptr  # trigger lazy init for SegmentedLevelStorage
                batch = batch.index_select(indices)
            num_graphs = num_selected
        else:
            num_graphs = num_total

        if self._count + num_graphs > self._capacity:
            raise RuntimeError(
                f"Store is full. Cannot add {num_graphs} samples "
                f"to store with {self._count}/{self._capacity} samples."
            )

        writer = self._get_writer()
        if not self._written_once:
            writer.write(batch)
            self._written_once = True
        else:
            writer.append(batch)

        self._count += num_graphs

    def read(self) -> Batch:
        """
        Load all stored data from Zarr as a Batch.

        Delegates to :class:`AtomicDataZarrReader` for efficient reading
        of samples from the CSR-style layout created by
        :class:`AtomicDataZarrWriter`.

        Returns
        -------
        Batch
            A batch containing all stored atomic data.

        Raises
        ------
        RuntimeError
            If the store is empty.
        """
        if self._count == 0:
            raise RuntimeError("Cannot read from empty store.")

        with AtomicDataZarrReader(self._store) as reader:
            # TODO: optimize this by adding index_select/slicing to amortize overhead
            data_list = [AtomicData(**reader[i][0]) for i in range(len(reader))]
            return Batch.from_data_list(data_list)

    def zero(self) -> None:
        """Clear all stored data and reset the store."""
        # Handle different store types for cleanup
        if isinstance(self._store, (str, Path)):
            # Filesystem path — delete directory if it exists
            store_path = Path(self._store)
            if store_path.exists():
                shutil.rmtree(store_path)
        elif isinstance(self._store, dict):
            # In-memory dict store — clear all contents
            self._store.clear()
        # For other Store types (LocalStore, MemoryStore, FsspecStore, etc.),
        # the writer will handle overwriting when opened in write mode.

        # Reset state
        self._writer = AtomicDataZarrWriter(self._store)
        self._count = 0
        self._written_once = False

    def __len__(self) -> int:
        """
        Return the number of samples currently stored.

        Returns
        -------
        int
            Number of atomic data samples in the store.
        """
        return self._count

    @property
    def capacity(self) -> int:
        """
        Return the maximum storage capacity.

        Returns
        -------
        int
            Maximum number of samples that can be stored.
        """
        return self._capacity
