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
from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch, LevelSchema
from nvalchemi.dynamics.sinks import (
    DataSink,
    GPUBuffer,
    HostMemory,
    ResizableSink,
    ZarrData,
)

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


def create_grouped_batch(group_sizes: list[int], device: str = "cpu") -> Batch:
    """Create a grouped batch with the given number of graphs per group."""
    batch = create_test_batch(num_graphs=sum(group_sizes), device=device)
    batch.set_group_layout(
        torch.repeat_interleave(
            torch.arange(len(group_sizes), device=device),
            torch.tensor(group_sizes, device=device),
        )
    )
    return batch


def _make_sink(kind: str, tmp_path: Path, capacity: int = 100) -> DataSink:
    """Create a host or Zarr sink for grouped-write tests."""
    if kind == "host":
        return HostMemory(capacity=capacity)
    return ZarrData(store=tmp_path / "grouped_store", capacity=capacity)


# -----------------------------------------------------------------------------
# Test Classes
# -----------------------------------------------------------------------------


class _GrowingHostMemory(HostMemory):
    """Host-memory sink that can be grown after construction."""

    def resize(self, capacity: int) -> None:
        """Raise the capacity to *capacity*."""
        self._capacity = capacity


class TestResizableSink:
    """Structural check for sinks a consumer may grow."""

    def test_a_sink_with_capacity_and_resize_satisfies_the_protocol(self) -> None:
        """Offering ``capacity`` and ``resize`` is all the protocol asks."""
        sink = _GrowingHostMemory(capacity=1)

        assert isinstance(sink, ResizableSink)
        sink.resize(4)
        assert sink.capacity == 4

    def test_a_fixed_capacity_sink_does_not(self) -> None:
        """A stock DataSink exposes ``capacity`` but no ``resize``."""
        assert not isinstance(HostMemory(capacity=1), ResizableSink)


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

            def write(self, batch: Batch, mask: torch.Tensor | None = None) -> None:
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


class TestDrainMethod:
    """Test suite for the drain() method on DataSink implementations."""

    def test_drain_returns_data_and_clears_host_memory(self) -> None:
        """Verify drain() returns stored data and clears the sink."""
        buffer = HostMemory(capacity=10)
        batch = create_test_batch(num_graphs=3)
        original_positions = batch.positions.clone()

        buffer.write(batch)
        assert len(buffer) == 3

        # Drain should return data and clear
        drained = buffer.drain()
        assert drained.num_graphs == 3
        assert torch.allclose(drained.positions, original_positions)
        assert len(buffer) == 0

    def test_drain_empty_sink_raises(self) -> None:
        """Verify drain() on empty sink raises RuntimeError (from read())."""
        buffer = HostMemory(capacity=10)
        # drain() calls read() which raises on empty
        with pytest.raises(RuntimeError, match="empty"):
            buffer.drain()

    def test_drain_multiple_writes(self) -> None:
        """Verify drain() returns all accumulated data from multiple writes."""
        buffer = HostMemory(capacity=10)
        buffer.write(create_test_batch(num_graphs=2))
        buffer.write(create_test_batch(num_graphs=3))
        assert len(buffer) == 5

        drained = buffer.drain()
        assert drained.num_graphs == 5
        assert len(buffer) == 0

    def test_drain_allows_subsequent_write(self) -> None:
        """Verify drain() allows writing new data afterward."""
        buffer = HostMemory(capacity=10)
        batch = create_test_batch(num_graphs=2)
        buffer.write(batch)

        buffer.drain()
        assert len(buffer) == 0

        # Should be able to write again
        new_batch = create_test_batch(num_graphs=4)
        buffer.write(new_batch)
        assert len(buffer) == 4


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

    def test_write_moves_a_cpu_batch_to_the_buffer_device(self) -> None:
        """A batch handed over on CPU is moved to the buffer rather than crashing."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        cuda_batch = create_test_batch(num_graphs=2, device="cuda")
        cpu_batch = create_test_batch(num_graphs=1, device="cpu")

        buffer.write(cuda_batch)
        buffer.write(cpu_batch)

        retrieved = buffer.read()
        assert retrieved.num_graphs == 3
        assert retrieved.positions.device.type == "cuda"
        torch.testing.assert_close(
            retrieved.positions[cuda_batch.num_nodes :].cpu(), cpu_batch.positions
        )

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

    def test_drain_returns_data_and_clears(self) -> None:
        """Verify drain() returns stored data and clears the buffer."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        batch = create_test_batch(num_graphs=2, device="cuda")
        original_positions = batch.positions.clone()

        buffer.write(batch)
        assert len(buffer) == 2

        drained = buffer.drain()
        assert drained.num_graphs == 2
        assert torch.allclose(drained.positions, original_positions)
        assert len(buffer) == 0

    def test_drain_empty_buffer_raises(self) -> None:
        """Verify drain() on empty buffer raises RuntimeError."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        with pytest.raises(RuntimeError, match="empty"):
            buffer.drain()

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

    def test_zero_preserves_buffer_structure(self) -> None:
        """Verify zero() keeps the pre-allocated buffer intact."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        batch = create_test_batch(num_graphs=2, device="cuda")

        buffer.write(batch)
        assert len(buffer) == 2

        # Store reference to internal buffer before zero
        internal_buffer_before = buffer._buffer
        assert internal_buffer_before is not None

        buffer.zero()
        assert len(buffer) == 0

        # Buffer structure should be preserved (not None)
        assert buffer._buffer is not None
        # Should be the same object (no re-allocation)
        assert buffer._buffer is internal_buffer_before

    def test_zero_zeros_all_tensors(self) -> None:
        """Verify zero() sets all tensor data to zeros."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        batch = create_test_batch(num_graphs=2, device="cuda")

        buffer.write(batch)
        buffer.zero()

        # All tensors in all groups should be zeroed
        for group in buffer._buffer._storage.groups.values():
            for key in group._data.keys():
                tensor = group._data[key]
                assert torch.all(tensor == 0), f"Tensor '{key}' not zeroed after zero()"

    def test_write_after_zero_works(self) -> None:
        """Verify write() works correctly after zero() without re-allocation."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        batch = create_test_batch(num_graphs=2, device="cuda")

        buffer.write(batch)
        buffer.zero()

        # Write new data — should work without re-allocating
        new_batch = create_test_batch(num_graphs=3, device="cuda")
        original_positions = new_batch.positions.clone()
        buffer.write(new_batch)

        assert len(buffer) == 3
        retrieved = buffer.read()
        assert retrieved.num_graphs == 3
        assert torch.allclose(retrieved.positions, original_positions)

    def test_write_with_mask_copies_only_selected(self) -> None:
        """Verify write() with mask only copies selected samples."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        batch = create_test_batch(num_graphs=4, device="cuda")

        # Only select first 2 of 4 graphs
        mask = torch.tensor([True, True, False, False], device="cuda")
        buffer.write(batch, mask=mask)

        # Should have copied exactly 2 samples
        assert len(buffer) == 2

    def test_write_with_all_true_mask(self) -> None:
        """Verify write() with all-True mask behaves like write without mask."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        batch = create_test_batch(num_graphs=3, device="cuda")

        mask = torch.ones(3, dtype=torch.bool, device="cuda")
        buffer.write(batch, mask=mask)

        assert len(buffer) == 3

    def test_write_with_all_false_mask_is_noop(self) -> None:
        """Verify write() with all-False mask doesn't store anything."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        batch = create_test_batch(num_graphs=3, device="cuda")

        mask = torch.zeros(3, dtype=torch.bool, device="cuda")
        buffer.write(batch, mask=mask)

        assert len(buffer) == 0

    def test_write_with_none_mask_copies_all(self) -> None:
        """Verify write() without mask copies all samples (backward compat)."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        batch = create_test_batch(num_graphs=2, device="cuda")

        buffer.write(batch)  # No mask

        assert len(buffer) == 2

    def test_write_mask_length_mismatch_raises(self) -> None:
        """Verify ValueError when mask length doesn't match num_graphs."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        batch = create_test_batch(num_graphs=3, device="cuda")

        # Wrong mask length
        mask = torch.tensor([True, False], device="cuda")
        with pytest.raises(ValueError, match="mask length"):
            buffer.write(batch, mask=mask)

    def test_write_without_mask_no_name_error(self) -> None:
        """Verify maskless write does not crash with NameError on num_total."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        batch = create_test_batch(num_graphs=2, device="cuda")

        # This should not raise NameError
        buffer.write(batch)
        assert len(buffer) == 2

    def test_write_updates_count_and_read_succeeds(self) -> None:
        """Verify write increments _count so read() succeeds."""
        buffer = GPUBuffer(capacity=10, max_atoms=10, max_edges=20, device="cuda")
        batch = create_test_batch(num_graphs=3, device="cuda")

        buffer.write(batch)
        assert len(buffer) == 3
        assert not buffer.is_full

        # read() should succeed (not raise "empty buffer")
        retrieved = buffer.read()
        assert retrieved.num_graphs == 3


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

    def test_write_with_mask(self) -> None:
        """Verify write() with mask only writes selected samples."""
        buffer = HostMemory(capacity=100)
        batch = create_test_batch(num_graphs=4)

        mask = torch.tensor([True, False, True, False])
        buffer.write(batch, mask=mask)

        assert len(buffer) == 2
        retrieved = buffer.read()
        assert retrieved.num_graphs == 2

    def test_write_with_none_mask_writes_all(self) -> None:
        """Verify write() without mask writes all samples."""
        buffer = HostMemory(capacity=100)
        batch = create_test_batch(num_graphs=3)
        buffer.write(batch)
        assert len(buffer) == 3

    def test_write_with_all_false_mask_is_noop(self) -> None:
        """Verify write() with all-False mask doesn't write anything."""
        buffer = HostMemory(capacity=100)
        batch = create_test_batch(num_graphs=3)
        mask = torch.zeros(3, dtype=torch.bool)
        buffer.write(batch, mask=mask)
        assert len(buffer) == 0

    def test_write_mask_length_mismatch_raises(self) -> None:
        """Verify ValueError when mask length doesn't match num_graphs."""
        buffer = HostMemory(capacity=100)
        batch = create_test_batch(num_graphs=3)
        mask = torch.tensor([True, False])
        with pytest.raises(ValueError, match="mask length"):
            buffer.write(batch, mask=mask)


class TestHostMemoryStorage:
    """HostMemory stores bulk CPU copies that stay isolated from callers."""

    def test_write_and_read_do_not_decompose_graphs(self, device: str) -> None:
        """Grouped writes and reads stay bulk instead of rebuilding each graph."""
        sink = HostMemory(capacity=100)
        batch = create_grouped_batch([3, 2], device=device)
        with patch.object(Batch, "get_data", side_effect=AssertionError):
            sink.write(batch)
            sink.write(batch, mask=torch.tensor([0, 0, 0, 1, 1], device=device) > 0)
            result = sink.read()
        assert result.group_idx.tolist() == [0, 0, 0, 1, 1, 2, 2]

    def test_read_keeps_custom_fields_identity_and_connectivity(
        self, device: str
    ) -> None:
        """Custom fields, path identity and edge offsets survive grouped writes."""

        def make(group_idx: list[int], ids: list[int]) -> Batch:
            data_list = [
                AtomicData(
                    atomic_numbers=torch.tensor([6, 8]),
                    positions=torch.randn(2, 3),
                    neighbor_list=torch.tensor([[0, 1], [1, 0]]),
                )
                for _ in group_idx
            ]
            batch = Batch.from_data_list(data_list, device=device)
            batch.set_group_layout(torch.tensor(group_idx, device=device))
            batch.add_key("group_id", [torch.tensor([i]) for i in ids], level="system")
            batch.add_key(
                "status",
                list(torch.arange(batch.num_graphs).float().unbind()),
                level="system",
            )
            return batch

        first = make([0, 0, 1], [7, 7, 3])
        second = make([0, 1, 1], [5, 9, 9])
        sink = HostMemory(capacity=100)
        sink.write(first)
        sink.write(second, mask=torch.tensor([False, True, True], device=device))

        result = sink.read()
        assert result.device == torch.device("cpu")
        assert result.group_idx.tolist() == [0, 0, 1, 2, 2]
        assert result.group_id.flatten().tolist() == [7, 7, 3, 9, 9]
        assert result.status.flatten().tolist() == [0.0, 1.0, 2.0, 1.0, 2.0]
        expected = torch.cat([first.positions, second.positions[2:]]).cpu()
        assert torch.equal(result.positions, expected)
        # Each graph's edges point at its own two atoms after concatenation.
        starts = result.batch_ptr[:-1].repeat_interleave(2)
        assert torch.equal(
            result.neighbor_list,
            torch.tensor([[0, 1], [1, 0]] * result.num_graphs) + starts[:, None],
        )

    def test_stored_data_is_independent_of_source_and_reads(self, device: str) -> None:
        """Later changes to the source batch or a read result do not leak in."""
        sink = HostMemory(capacity=100)
        batch = create_grouped_batch([2, 1], device=device)
        expected = batch.positions.cpu().clone()
        sink.write(batch)

        # A reused source buffer is overwritten after the write.
        batch.positions.zero_()
        batch.group_idx.fill_(5)
        first_read = sink.read()
        first_read.positions.add_(1.0)
        first_read.group_idx.add_(3)

        second_read = sink.read()
        assert torch.equal(second_read.positions, expected)
        assert second_read.group_idx.tolist() == [0, 0, 1]

    def test_rejected_writes_leave_the_sink_unchanged(self) -> None:
        """Partial groups, overflow and mixed modes raise before storing."""
        sink = HostMemory(capacity=5)
        sink.write(create_grouped_batch([2, 1]))
        before = sink.read().positions

        batch = create_grouped_batch([1, 2])
        with pytest.raises(ValueError, match="complete groups"):
            sink.write(batch, mask=torch.tensor([True, True, False]))
        with pytest.raises(RuntimeError, match="Buffer is full"):
            sink.write(create_grouped_batch([3]))
        with pytest.raises(ValueError, match="mix grouped and ungrouped"):
            sink.write(create_test_batch(num_graphs=1))

        assert len(sink) == 3
        assert torch.equal(sink.read().positions, before)
        # Capacity counts graphs, so two more graphs still fit.
        sink.write(create_grouped_batch([2]))
        assert len(sink) == 5
        assert sink.is_full

    def test_read_rejects_a_later_write_missing_a_stored_field(self) -> None:
        """A field the first write stored is never zero-filled or dropped."""
        sink = HostMemory(capacity=10)
        first = create_grouped_batch([2, 1])
        first.add_key("status", [torch.tensor([1.0])] * 3, level="system")
        sink.write(first)
        sink.write(create_grouped_batch([2]))
        assert len(sink) == 5

        for _ in range(2):
            with pytest.raises(
                ValueError,
                match=r"Field 'status' in level 'system' is missing from samples \[3, 4\]",
            ):
                sink.read()

    @pytest.mark.parametrize("grouped", [False, True])
    @pytest.mark.parametrize(
        "padded_first,padded_second", [(True, False), (False, True), (True, True)]
    )
    def test_read_padded_writes(self, device, grouped, padded_first, padded_second):
        """Whole-buffer writes retain capacity; merging excludes unoccupied rows."""
        source = (
            create_grouped_batch([2, 1], device)
            if grouped
            else create_test_batch(3, device)
        )
        writes = []
        for padded in (padded_first, padded_second):
            if padded:
                batch = Batch.empty(
                    num_systems=8,
                    num_nodes=32,
                    num_edges=0,
                    template=source,
                    device=device,
                )
                batch.put(source, torch.ones(3, dtype=torch.bool, device=device))
                writes.append(batch)
            else:
                writes.append(source.clone())
        sink = HostMemory(capacity=6)
        for batch in writes:
            sink.write(batch)
        for stored, original in zip(sink._batches, writes, strict=True):
            assert stored.system_capacity == original.system_capacity
        result = sink.read()
        assert result.num_graphs == 6
        assert result.num_nodes == 12
        assert result.positions.shape[0] == 12
        torch.testing.assert_close(
            result.positions, torch.cat([source.positions.cpu()] * 2)
        )
        if grouped:
            assert result.system_capacity == 6
            assert result.group_idx.tolist() == [0, 0, 1, 2, 2, 3]
        for original in writes:
            assert original.num_graphs == 3

    def test_read_ignores_fields_only_later_writes_carry(self) -> None:
        """The first write's fields stay authoritative for extra incoming fields."""
        first = create_grouped_batch([2, 1])
        second = create_grouped_batch([1, 2])
        second.add_key("status", [torch.tensor([1.0])] * 3, level="system")
        sink = HostMemory(capacity=10)
        sink.write(first)
        sink.write(second)
        result = sink.read()
        assert "status" not in result
        assert "status" in second
        assert result.group_idx.tolist() == [0, 0, 1, 2, 3, 3]

    def test_failed_concatenation_keeps_stored_writes(self) -> None:
        """A failed bulk merge leaves the original cached writes intact."""
        sink = HostMemory(capacity=10)
        first = create_test_batch(1)
        second = create_test_batch(1)
        second.positions = torch.ones(second.num_nodes, 4)
        sink.write(first)
        sink.write(second)
        stored = sink._batches.copy()
        for _ in range(2):
            with pytest.raises(ValueError, match="incompatible trailing shapes"):
                sink.read()
            assert len(sink._batches) == len(stored)
            assert all(
                actual is before for actual, before in zip(sink._batches, stored)
            )
            torch.testing.assert_close(sink._batches[0].positions, first.positions)

    def test_zero_allows_a_different_mode(self) -> None:
        """After zero() the sink accepts ungrouped data and starts empty."""
        sink = HostMemory(capacity=10)
        sink.write(create_grouped_batch([2]))
        sink.zero()
        with pytest.raises(RuntimeError, match="empty"):
            sink.read()
        sink.write(create_test_batch(num_graphs=3))
        result = sink.read()
        assert len(sink) == 3
        assert "group_idx" not in result

    def test_writes_after_a_read_keep_accumulating(self) -> None:
        """A read between writes does not lose, repeat or merge groups."""
        sink = HostMemory(capacity=20)
        batches = [create_grouped_batch(sizes) for sizes in ([2, 1], [1], [3, 1])]
        sink.write(batches[0])
        sink.write(batches[1])
        assert sink.read().group_idx.tolist() == [0, 0, 1, 2]

        sink.write(batches[2])
        result = sink.read()
        assert len(sink) == 8
        assert result.group_idx.tolist() == [0, 0, 1, 2, 3, 3, 3, 4]
        expected = torch.cat([batch.positions for batch in batches])
        assert torch.equal(result.positions, expected)
        assert torch.equal(sink.read().positions, expected)

    def test_read_keeps_data_class_and_custom_levels(self) -> None:
        """Custom AtomicData classes and custom-level fields round-trip."""

        class TaggedData(AtomicData):
            """AtomicData subclass standing in for a user data class."""

        def make(num_graphs: int, offset: float) -> Batch:
            data_list = []
            for i in range(num_graphs):
                data = TaggedData(
                    atomic_numbers=torch.tensor([6, 8]),
                    positions=torch.randn(2, 3),
                )
                data.samples = torch.full((i + 1, 1), offset + i)
                data_list.append(data)
            schema = LevelSchema()
            schema.add_level("draws", segmented=True)
            schema.set("samples", "draws")
            return Batch.from_data_list(data_list, attr_map=schema)

        sink = HostMemory(capacity=10)
        sink.write(make(2, 0.0))
        sink.write(make(1, 10.0))
        result = sink.read()

        assert "draws" in result.level_keys
        assert result.samples.flatten().tolist() == [0.0, 1.0, 1.0, 10.0]
        assert isinstance(result.get_data(2), TaggedData)
        assert result.get_data(1).samples.flatten().tolist() == [1.0, 1.0]


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

    def test_write_with_mask(self, tmp_path: Path) -> None:
        """Verify write() with mask only writes selected samples."""
        store_path = tmp_path / "test_store"
        sink = ZarrData(store=store_path, capacity=100)

        batch = create_test_batch(num_graphs=4)
        mask = torch.tensor([True, False, True, False])
        sink.write(batch, mask=mask)

        assert len(sink) == 2
        retrieved = sink.read()
        assert retrieved.num_graphs == 2

    def test_write_with_none_mask_writes_all(self, tmp_path: Path) -> None:
        """Verify write() without mask writes all samples."""
        store_path = tmp_path / "test_store"
        sink = ZarrData(store=store_path, capacity=100)

        batch = create_test_batch(num_graphs=3)
        sink.write(batch)

        assert len(sink) == 3

    def test_write_with_all_false_mask_is_noop(self, tmp_path: Path) -> None:
        """Verify write() with all-False mask doesn't write anything."""
        store_path = tmp_path / "test_store"
        sink = ZarrData(store=store_path, capacity=100)

        batch = create_test_batch(num_graphs=3)
        mask = torch.zeros(3, dtype=torch.bool)
        sink.write(batch, mask=mask)

        assert len(sink) == 0

    def test_write_mask_length_mismatch_raises(self, tmp_path: Path) -> None:
        """Verify ValueError when mask length doesn't match num_graphs."""
        store_path = tmp_path / "test_store"
        sink = ZarrData(store=store_path, capacity=100)

        batch = create_test_batch(num_graphs=3)
        mask = torch.tensor([True, False])  # Wrong length
        with pytest.raises(ValueError, match="mask length"):
            sink.write(batch, mask=mask)

    def test_zero_restarts_group_labels(self, tmp_path: Path) -> None:
        """Group labels written after zero() start again at zero."""
        sink = ZarrData(store=tmp_path / "test_store", capacity=100)
        sink.write(create_grouped_batch([2, 1]))
        sink.zero()
        sink.write(create_grouped_batch([1, 2]))
        assert sink.read().group_idx.tolist() == [0, 1, 1]


@pytest.mark.parametrize("kind", ["host", "zarr"])
class TestGroupedHostAndZarrSinks:
    """Grouped writes to HostMemory and ZarrData keep groups intact."""

    def test_repeated_writes_keep_groups_distinct(
        self, kind: str, device: str, tmp_path: Path
    ) -> None:
        """Labels from separate writes, including masked ones, never merge."""
        sink = _make_sink(kind, tmp_path)
        first = create_grouped_batch([3, 2], device=device)
        second = create_grouped_batch([2, 1], device=device)
        sink.write(first)
        sink.write(second, mask=torch.tensor([True, True, False], device=device))

        assert len(sink) == 7
        result = sink.read()
        assert result.group_idx.tolist() == [0, 0, 0, 1, 1, 2, 2]
        assert result.group_layout.num_groups == 3
        expected = torch.cat([first.positions, second.positions[:4]]).cpu()
        assert torch.equal(result.positions.cpu(), expected)
        # The caller's labels are not offset in place.
        assert second.group_idx.tolist() == [0, 0, 1]

    def test_partial_group_mask_raises(self, kind: str, tmp_path: Path) -> None:
        """A mask that splits a group is rejected and nothing is written."""
        sink = _make_sink(kind, tmp_path)
        batch = create_grouped_batch([3, 2])
        with pytest.raises(ValueError, match="complete groups"):
            sink.write(batch, mask=torch.tensor([True, True, False, False, False]))
        assert len(sink) == 0

    def test_mixing_grouped_and_ungrouped_writes_raises(
        self, kind: str, tmp_path: Path
    ) -> None:
        """A sink holding grouped data rejects ungrouped writes."""
        sink = _make_sink(kind, tmp_path)
        sink.write(create_grouped_batch([2]))
        with pytest.raises(ValueError, match="mix grouped and ungrouped"):
            sink.write(create_test_batch(num_graphs=1))
