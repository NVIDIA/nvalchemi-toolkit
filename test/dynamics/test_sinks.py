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
Tests for dynamics data sink implementations.

Tests cover the DataSink ABC and concrete implementations: GPUBuffer,
HostMemory, and ZarrData.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.sinks import DataSink, GPUBuffer, HostMemory, ZarrData

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def create_test_batch(num_graphs: int = 2, device: str = "cpu") -> Batch:
    """
    Create a test batch with specified number of graphs.

    Parameters
    ----------
    num_graphs : int
        Number of graphs to include in the batch.
    device : str
        Device to create tensors on.

    Returns
    -------
    Batch
        A batch containing the specified number of AtomicData objects.
    """
    data_list = [
        AtomicData(
            atomic_numbers=torch.tensor([6, 8], dtype=torch.long),
            positions=torch.randn(2, 3),
        )
        for _ in range(num_graphs)
    ]
    return Batch.from_data_list(data_list, device=device)


def create_single_atom_batch(num_graphs: int = 1, device: str = "cpu") -> Batch:
    """
    Create a minimal test batch with single-atom graphs.

    Parameters
    ----------
    num_graphs : int
        Number of single-atom graphs to include.
    device : str
        Device to create tensors on.

    Returns
    -------
    Batch
        A batch containing single-atom AtomicData objects.
    """
    data_list = [
        AtomicData(
            atomic_numbers=torch.tensor([1], dtype=torch.long),
            positions=torch.randn(1, 3),
        )
        for _ in range(num_graphs)
    ]
    return Batch.from_data_list(data_list, device=device)


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------


class TestDataSinkABC:
    """Test suite for the DataSink abstract base class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Verify DataSink cannot be instantiated directly."""
        with pytest.raises(TypeError, match="abstract"):
            DataSink()  # type: ignore[abstract]

    def test_is_full_logic_with_mock_subclass(self) -> None:
        """Verify is_full logic with a mock concrete subclass."""

        class MockSink(DataSink):
            """Mock concrete implementation for testing."""

            def __init__(self, capacity: int) -> None:
                self._capacity = capacity
                self._count = 0

            def write(self, batch: Batch) -> None:
                self._count += batch.num_graphs

            def read(self) -> Batch:
                return create_test_batch(1)

            def zero(self) -> None:
                self._count = 0

            def __len__(self) -> int:
                return self._count

            @property
            def capacity(self) -> int:
                return self._capacity

        sink = MockSink(capacity=5)
        assert not sink.is_full
        assert len(sink) == 0

        # Add samples to fill to capacity
        sink._count = 5
        assert sink.is_full

        # Over capacity should also be full
        sink._count = 10
        assert sink.is_full


# Skip GPUBuffer tests if CUDA is not available
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for GPUBuffer tests"
)


@requires_cuda
class TestGPUBuffer:
    """Test suite for GPUBuffer implementation."""

    def test_initialization(self) -> None:
        """Verify GPUBuffer initializes correctly."""
        buffer = GPUBuffer(capacity=100, max_atoms=10, max_edges=20, device="cuda")
        assert buffer.capacity == 100
        assert buffer.device.type == "cuda"
        assert len(buffer) == 0
        assert not buffer.is_full

    def test_write_read_roundtrip(self) -> None:
        """Verify data can be written and read back correctly."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        batch = create_test_batch(num_graphs=2, device="cuda")

        # Store original positions for comparison
        original_positions = batch.positions.clone()

        buffer.write(batch)
        retrieved = buffer.read()

        # Verify positions match
        assert torch.allclose(retrieved.positions, original_positions)
        assert retrieved.num_graphs == 2

    def test_zero_clears_buffer(self) -> None:
        """Verify zero() clears all stored data."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        batch = create_test_batch(num_graphs=2, device="cuda")

        buffer.write(batch)
        assert len(buffer) == 2

        buffer.zero()
        assert len(buffer) == 0

    def test_is_full_at_capacity(self) -> None:
        """Verify is_full returns True at capacity."""
        buffer = GPUBuffer(capacity=3, max_atoms=10, max_edges=20, device="cuda")

        # Write up to capacity
        batch = create_test_batch(num_graphs=3, device="cuda")
        buffer.write(batch)

        assert buffer.is_full
        assert len(buffer) == 3

    def test_write_when_full_raises(self) -> None:
        """Verify writing to full buffer raises RuntimeError."""
        buffer = GPUBuffer(capacity=2, max_atoms=10, max_edges=20, device="cuda")

        # Fill to capacity
        batch = create_test_batch(num_graphs=2, device="cuda")
        buffer.write(batch)

        # Attempt to add more
        with pytest.raises(RuntimeError, match="Buffer is full"):
            buffer.write(create_test_batch(num_graphs=1, device="cuda"))

    def test_read_when_empty_raises(self) -> None:
        """Verify reading from empty buffer raises RuntimeError."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")

        with pytest.raises(RuntimeError, match="empty"):
            buffer.read()

    def test_len_tracks_samples(self) -> None:
        """Verify __len__ returns correct count after writes."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")

        assert len(buffer) == 0

        buffer.write(create_test_batch(num_graphs=2, device="cuda"))
        assert len(buffer) == 2

        buffer.write(create_test_batch(num_graphs=3, device="cuda"))
        assert len(buffer) == 5

    def test_multiple_writes_accumulate(self) -> None:
        """Verify multiple writes accumulate data."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")

        buffer.write(create_single_atom_batch(num_graphs=2, device="cuda"))
        buffer.write(create_single_atom_batch(num_graphs=3, device="cuda"))

        retrieved = buffer.read()
        assert retrieved.num_graphs == 5

    def test_device_property(self) -> None:
        """Verify device property returns correct device."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        assert buffer.device.type == "cuda"

    def test_cpu_device_rejected(self) -> None:
        """Verify RuntimeError when device='cpu' is specified."""
        # This test doesn't need CUDA to run - it tests the validation logic
        if torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="GPUBuffer requires a CUDA device"):
                GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cpu")
        else:
            # If CUDA is not available, it will raise about CUDA availability first
            with pytest.raises(RuntimeError, match="CUDA"):
                GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cpu")

    def test_max_atoms_exceeded_raises(self) -> None:
        """Verify error when batch has more atoms than capacity allows."""
        # Create buffer with very limited atom capacity
        buffer = GPUBuffer(capacity=2, max_atoms=1, max_edges=0, device="cuda")

        # Create batch with 2 atoms per graph, 2 graphs = 4 atoms total
        # But buffer can only hold 2 * 1 = 2 atoms
        batch = create_test_batch(num_graphs=2, device="cuda")  # 2 atoms per graph

        with pytest.raises(RuntimeError, match="Atom capacity exceeded"):
            buffer.write(batch)


class TestHostMemory:
    """Test suite for HostMemory implementation."""

    def test_initialization(self) -> None:
        """Verify HostMemory initializes correctly."""
        buffer = HostMemory(capacity=100)
        assert buffer.capacity == 100
        assert len(buffer) == 0
        assert not buffer.is_full

    def test_write_read_roundtrip(self) -> None:
        """Verify data can be written and read back correctly."""
        buffer = HostMemory(capacity=10)
        batch = create_test_batch(num_graphs=2)

        # Store original positions
        original_positions = batch.positions.clone()

        buffer.write(batch)
        retrieved = buffer.read()

        # Verify positions match
        assert torch.allclose(retrieved.positions, original_positions)
        assert retrieved.num_graphs == 2

    def test_data_is_on_cpu(self) -> None:
        """Verify all tensors are on CPU after read."""
        buffer = HostMemory(capacity=10)
        batch = create_test_batch(num_graphs=2)

        buffer.write(batch)
        retrieved = buffer.read()

        # Verify all tensors are on CPU
        assert retrieved.positions.device == torch.device("cpu")
        assert retrieved.atomic_numbers.device == torch.device("cpu")

    def test_zero_clears_buffer(self) -> None:
        """Verify zero() clears all stored data."""
        buffer = HostMemory(capacity=10)
        batch = create_test_batch(num_graphs=2)

        buffer.write(batch)
        assert len(buffer) == 2

        buffer.zero()
        assert len(buffer) == 0

    def test_write_when_full_raises(self) -> None:
        """Verify writing to full buffer raises RuntimeError."""
        buffer = HostMemory(capacity=2)

        # Fill to capacity
        batch = create_test_batch(num_graphs=2)
        buffer.write(batch)

        # Attempt to add more
        with pytest.raises(RuntimeError, match="Buffer is full"):
            buffer.write(create_test_batch(num_graphs=1))

    def test_read_when_empty_raises(self) -> None:
        """Verify reading from empty buffer raises RuntimeError."""
        buffer = HostMemory(capacity=10)

        with pytest.raises(RuntimeError, match="empty"):
            buffer.read()

    def test_len_tracks_samples(self) -> None:
        """Verify __len__ returns correct count after writes."""
        buffer = HostMemory(capacity=10)

        assert len(buffer) == 0

        buffer.write(create_test_batch(num_graphs=2))
        assert len(buffer) == 2

        buffer.write(create_test_batch(num_graphs=3))
        assert len(buffer) == 5


class TestZarrData:
    """Test suite for ZarrData implementation."""

    def test_initialization(self, tmp_path: Path) -> None:
        """Verify ZarrData initializes correctly."""
        store_path = tmp_path / "test_store"
        sink = ZarrData(store=store_path, capacity=1000)

        assert sink.capacity == 1000
        assert sink._store == store_path
        assert len(sink) == 0
        assert not sink.is_full

    @pytest.mark.parametrize("num_graphs", [1, 4, 8])
    def test_write_read_roundtrip(self, num_graphs: int, tmp_path: Path) -> None:
        """Verify data can be written and read back correctly."""
        store_path = tmp_path / "test_store"
        sink = ZarrData(store=store_path, capacity=100)

        batch = create_test_batch(num_graphs=num_graphs)
        original_positions = batch.positions.clone()
        original_atomic_numbers = batch.atomic_numbers.clone()

        sink.write(batch)
        retrieved = sink.read()

        # Verify data matches
        assert torch.allclose(retrieved.positions, original_positions)
        assert torch.equal(retrieved.atomic_numbers, original_atomic_numbers)
        assert retrieved.num_graphs == num_graphs

    @pytest.mark.parametrize("num_graphs", [1, 4, 8])
    def test_zero_clears_store(self, num_graphs: int, tmp_path: Path) -> None:
        """Verify zero() clears all stored data."""
        store_path = tmp_path / "test_store"
        sink = ZarrData(store=store_path, capacity=100)

        batch = create_test_batch(num_graphs=num_graphs)
        sink.write(batch)
        assert len(sink) == num_graphs

        sink.zero()
        assert len(sink) == 0

    @pytest.mark.parametrize("num_graphs", [1, 4, 8])
    def test_len_tracks_samples(self, num_graphs: int, tmp_path: Path) -> None:
        """Verify __len__ returns correct count after writes."""
        store_path = tmp_path / "test_store"
        sink = ZarrData(store=store_path, capacity=100)

        assert len(sink) == 0

        sink.write(create_test_batch(num_graphs=num_graphs))
        assert len(sink) == num_graphs

        sink.write(create_test_batch(num_graphs=num_graphs))
        assert len(sink) == num_graphs + num_graphs

    def test_write_when_full_raises(self, tmp_path: Path) -> None:
        """Verify writing to full store raises RuntimeError."""
        store_path = tmp_path / "test_store"
        sink = ZarrData(store=store_path, capacity=2)

        # Fill to capacity
        batch = create_test_batch(num_graphs=2)
        sink.write(batch)

        # Attempt to add more
        with pytest.raises(RuntimeError, match="full"):
            sink.write(create_test_batch(num_graphs=1))

    def test_read_when_empty_raises(self, tmp_path: Path) -> None:
        """Verify reading from empty store raises RuntimeError."""
        store_path = tmp_path / "test_store"
        sink = ZarrData(store=store_path, capacity=100)

        with pytest.raises(RuntimeError, match="empty"):
            sink.read()
