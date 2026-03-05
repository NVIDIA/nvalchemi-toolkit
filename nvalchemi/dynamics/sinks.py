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
from tensordict import TensorDict
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
    def write(self, batch: Batch) -> None:
        """
        Store a batch of atomic data.

        Parameters
        ----------
        batch : Batch
            The batch of atomic data to store.

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
    """
    GPU-resident buffer for storing batched atomic data.

    This buffer pre-allocates a TensorDict with fixed maximum sizes for atoms
    and edges, using CSR-style pointer tracking for variable-length graph data.
    It is CUDA-only and will reject non-CUDA devices.

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
        """
        Initialize the GPU buffer.

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
        # TODO: add CUDA stream context manager
        # Validate CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPUBuffer requires available CUDA devices:"
                f" found CUDA available: {torch.cuda.is_available()}"
                f" with device count={torch.cuda.device_count()}"
            )
        # Validate device is CUDA
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

        # Pre-allocate TensorDict with fixed sizes
        total_atoms = capacity * max_atoms
        total_edges = capacity * max_edges

        data_dict = {
            "positions": torch.empty(
                (total_atoms, 3), dtype=torch.float32, device=self._device
            ),
            "atomic_numbers": torch.empty(
                (total_atoms,), dtype=torch.int64, device=self._device
            ),
            "batch_idx": torch.empty(
                (total_atoms,), dtype=torch.int64, device=self._device
            ),
        }

        if max_edges > 0:
            data_dict["edge_index"] = torch.empty(
                (2, total_edges), dtype=torch.int64, device=self._device
            )

        self._data = TensorDict(data_dict, device=self._device)

        # Initialize occupancy tracking (CSR-style pointers)
        self._count: int = 0
        self._atoms_ptr: list[int] = [0]
        self._edges_ptr: list[int] = [0]

    def write(self, batch: Batch) -> None:
        """
        Store a batch of atomic data into the preallocated buffer.

        Copies batch data into the TensorDict at the appropriate offsets
        using slice assignment.

        Parameters
        ----------
        batch : Batch
            The batch of atomic data to store.

        Raises
        ------
        RuntimeError
            If adding this batch would exceed sample capacity.
        RuntimeError
            If the total atoms would exceed the preallocated atom capacity.
        RuntimeError
            If the total edges would exceed the preallocated edge capacity.
        """
        num_new_graphs = batch.num_graphs or 0
        if num_new_graphs == 0:
            return  # Nothing to write

        if self._count + num_new_graphs > self._capacity:
            raise RuntimeError(
                f"Buffer is full. Cannot add {num_new_graphs} samples "
                f"to buffer with {self._count}/{self._capacity} samples."
            )

        # Get per-graph atom and edge counts
        num_nodes_list = batch.num_nodes_list or []
        num_edges_list = batch.num_edges_list or []

        total_new_atoms = sum(num_nodes_list)
        total_new_edges = sum(num_edges_list) if num_edges_list else 0

        # Current offsets
        current_atom_offset = self._atoms_ptr[-1]
        current_edge_offset = self._edges_ptr[-1]

        # Check capacity constraints
        max_total_atoms = self._capacity * self._max_atoms
        max_total_edges = self._capacity * self._max_edges

        if current_atom_offset + total_new_atoms > max_total_atoms:
            raise RuntimeError(
                f"Atom capacity exceeded. Cannot add {total_new_atoms} atoms "
                f"to buffer with {current_atom_offset}/{max_total_atoms} atoms."
            )

        if (
            total_new_edges > 0
            and current_edge_offset + total_new_edges > max_total_edges
        ):
            raise RuntimeError(
                f"Edge capacity exceeded. Cannot add {total_new_edges} edges "
                f"to buffer with {current_edge_offset}/{max_total_edges} edges."
            )

        # Move batch to device if needed
        batch = batch.to(self._device)

        # Copy positions and atomic_numbers
        atom_end = current_atom_offset + total_new_atoms
        self._data["positions"][current_atom_offset:atom_end] = batch.positions
        self._data["atomic_numbers"][current_atom_offset:atom_end] = (
            batch.atomic_numbers
        )

        # Build batch_idx tensor for the new data
        batch_idx = torch.cat(
            [
                torch.full(
                    (n,), self._count + i, dtype=torch.int64, device=self._device
                )
                for i, n in enumerate(num_nodes_list)
            ]
        )
        self._data["batch_idx"][current_atom_offset:atom_end] = batch_idx

        # Copy edge_index if present
        if (
            total_new_edges > 0
            and hasattr(batch, "edge_index")
            and batch.edge_index is not None
        ):
            edge_end = current_edge_offset + total_new_edges
            # Adjust edge indices by the current atom offset
            adjusted_edge_index = batch.edge_index + current_atom_offset
            self._data["edge_index"][:, current_edge_offset:edge_end] = (
                adjusted_edge_index
            )

        # Update CSR pointers for each graph
        atom_offset = current_atom_offset
        edge_offset = current_edge_offset
        for i in range(num_new_graphs):
            node_count = num_nodes_list[i] if num_nodes_list[i] is not None else 0
            atom_offset += node_count
            self._atoms_ptr.append(atom_offset)

            if num_edges_list:
                edge_count = num_edges_list[i] if num_edges_list[i] is not None else 0
                edge_offset += edge_count
            self._edges_ptr.append(edge_offset)

        self._count += num_new_graphs

    def read(self) -> Batch:
        """
        Retrieve all stored data as a single Batch.

        Reconstructs a Batch from the occupied region of the TensorDict
        using the CSR-style pointer arrays.

        Returns
        -------
        Batch
            A batch containing all stored atomic data.

        Raises
        ------
        RuntimeError
            If the buffer is empty.
        """
        if self._count == 0:
            raise RuntimeError("Cannot read from empty buffer.")

        # Extract the occupied slices
        total_atoms = self._atoms_ptr[-1]
        total_edges = self._edges_ptr[-1]

        positions = self._data["positions"][:total_atoms].clone()
        atomic_numbers = self._data["atomic_numbers"][:total_atoms].clone()
        batch_tensor = self._data["batch_idx"][:total_atoms].clone()

        # Renumber batch_idx to start from 0 (in case buffer was partially cleared)
        # The batch_idx should already be 0-indexed from our write logic

        # Create the ptr tensor (cumulative nodes per graph)
        ptr = torch.tensor(self._atoms_ptr, dtype=torch.int64, device=self._device)

        # Reconstruct num_nodes_list and num_edges_list
        num_nodes_list = [
            self._atoms_ptr[i + 1] - self._atoms_ptr[i] for i in range(self._count)
        ]
        num_edges_list = [
            self._edges_ptr[i + 1] - self._edges_ptr[i] for i in range(self._count)
        ]

        # Build kwargs for Batch
        batch_kwargs: dict = {
            "positions": positions,
            "atomic_numbers": atomic_numbers,
            "batch": batch_tensor,
            "ptr": ptr,
            "device": self._device,
            "num_graphs": self._count,
            "num_nodes_list": num_nodes_list,
            "num_edges_list": num_edges_list,
        }

        # Add edge_index if present
        if total_edges > 0 and "edge_index" in self._data.keys():
            edge_index = self._data["edge_index"][:, :total_edges].clone()
            batch_kwargs["edge_index"] = edge_index

        return Batch(**batch_kwargs)

    def zero(self) -> None:
        """
        Clear all stored data and reset the buffer.

        Resets the occupancy counter and CSR pointers. The preallocated
        memory is retained and will be overwritten on subsequent writes.
        """
        self._count = 0
        self._atoms_ptr = [0]
        self._edges_ptr = [0]

    def __len__(self) -> int:
        """
        Return the number of samples currently stored.

        Returns
        -------
        int
            Number of atomic data samples in the buffer.
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

    @property
    def device(self) -> torch.device:
        """
        Return the storage device.

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

    def write(self, batch: Batch) -> None:
        """
        Store a batch of atomic data on CPU.

        Decomposes the batch into individual AtomicData objects,
        moves them to CPU, and appends to internal storage.

        Parameters
        ----------
        batch : Batch
            The batch of atomic data to store.

        Raises
        ------
        RuntimeError
            If adding this batch would exceed capacity.
        """
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

    def write(self, batch: Batch) -> None:
        """
        Store a batch of atomic data to Zarr.

        Uses :class:`AtomicDataZarrWriter` for efficient bulk writes.
        The first write uses ``write()`` (creates store), subsequent
        writes use ``append()`` (extends existing store).

        Parameters
        ----------
        batch : Batch
            The batch of atomic data to store.

        Raises
        ------
        RuntimeError
            If adding this batch would exceed capacity.
        """
        num_graphs = batch.num_graphs or 0
        if num_graphs == 0:
            return  # Nothing to write

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
