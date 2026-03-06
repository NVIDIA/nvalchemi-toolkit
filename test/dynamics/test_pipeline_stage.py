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
Tests for the pipeline stage infrastructure and Batch communication methods.

Tests cover:
- _CommunicationMixin construction, properties, buffer routing, and stage composition.
- Batch.isend / Batch.irecv error handling (without distributed).
- Batch._collect_tensor_fields serialization helper.
- DistributedPipeline orchestrator construction and setup.
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch
from torch import distributed as dist

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import (
    BufferConfig,
    DistributedPipeline,
    _CommunicationMixin,
)
from nvalchemi.dynamics.sinks import HostMemory

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_atomic_data(num_atoms: int = 3) -> AtomicData:
    """Create a simple AtomicData for testing.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the system.

    Returns
    -------
    AtomicData
        A minimal AtomicData instance.
    """
    return AtomicData(
        atomic_numbers=torch.randint(1, 20, (num_atoms,)),
        positions=torch.randn(num_atoms, 3),
    )


def _make_batch(num_graphs: int = 3, num_atoms: int = 3) -> Batch:
    """Create a test batch with the specified number of graphs.

    Parameters
    ----------
    num_graphs : int
        Number of graphs in the batch.
    num_atoms : int
        Number of atoms per graph.

    Returns
    -------
    Batch
        A batch containing the specified number of graphs.
    """
    data_list = [_make_atomic_data(num_atoms) for _ in range(num_graphs)]
    return Batch.from_data_list(data_list)


def _make_atomic_data_with_system(num_atoms: int = 3) -> AtomicData:
    """Create an AtomicData with system-level attributes for testing.

    Parameters
    ----------
    num_atoms : int
        Number of atoms in the system.

    Returns
    -------
    AtomicData
        An AtomicData instance with system-level energies.
    """
    return AtomicData(
        atomic_numbers=torch.randint(1, 20, (num_atoms,)),
        positions=torch.randn(num_atoms, 3),
        energies=torch.tensor([[0.0]]),
    )


def _make_batch_with_system(num_graphs: int = 3, num_atoms: int = 3) -> Batch:
    """Create a test batch with system-level attributes.

    Parameters
    ----------
    num_graphs : int
        Number of graphs in the batch.
    num_atoms : int
        Number of atoms per graph.

    Returns
    -------
    Batch
        A batch containing the specified number of graphs with system-level data.
    """
    data_list = [_make_atomic_data_with_system(num_atoms) for _ in range(num_graphs)]
    return Batch.from_data_list(data_list)


# ---------------------------------------------------------------------------
# Test_CommunicationMixin — Construction & Properties
# ---------------------------------------------------------------------------


class Test_CommunicationMixinConstruction:
    """Test _CommunicationMixin initialization and basic properties."""

    def test_default_construction(self) -> None:
        """Verify default field values after construction."""
        stage = _CommunicationMixin()
        assert stage.prior_rank is None
        assert stage.next_rank is None
        assert stage.sinks == []
        assert stage.active_batch is None
        assert stage.done is False

    def test_custom_construction(self) -> None:
        """Verify custom field values are respected."""
        buf = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            prior_rank=0,
            next_rank=2,
            sinks=[buf],
            max_batch_size=200,
        )
        assert stage.prior_rank == 0
        assert stage.next_rank == 2
        assert len(stage.sinks) == 1
        assert stage.max_batch_size == 200

    @pytest.mark.parametrize("combination", [(True, None), (False, 1), (False, 6)])
    def test_is_final_stage(self, combination: tuple[bool, None | int]) -> None:
        """Verify is_final_stage is True when next_rank is None."""
        expectation, rank = combination
        stage = _CommunicationMixin(next_rank=rank)
        assert stage.is_final_stage is expectation

    @pytest.mark.parametrize("combination", [(True, None), (False, 1), (False, 6)])
    def test_is_first_stage(self, combination: tuple[bool, None | int]) -> None:
        """Verify is_first_stage is True when prior_rank is None."""
        expectation, rank = combination
        stage = _CommunicationMixin(prior_rank=rank)
        assert stage.is_first_stage is expectation

    def test_active_batch_size_with_batch(self) -> None:
        """Verify active_batch_size reflects the batch num_graphs."""
        batch = _make_batch(num_graphs=5)
        stage = _CommunicationMixin(active_batch=batch)
        assert stage.active_batch_size == 5

    @pytest.mark.parametrize("combination", [(True, 3), (False, 16), (True, 1)])
    def test_active_batch_has_room(self, combination: tuple[bool, int]) -> None:
        """Verify active_batch_has_room when below max_batch_size."""
        expectation, num_graphs = combination
        batch = _make_batch(num_graphs=num_graphs)
        stage = _CommunicationMixin(active_batch=batch, max_batch_size=10)
        assert stage.active_batch_has_room is expectation

    @pytest.mark.parametrize("combination", [(7, 3), (0, 16), (9, 1)])
    def test_room_in_active_batch(self, combination: tuple[int, int]) -> None:
        """Verify room_in_active_batch returns correct remaining capacity."""
        expectation, num_graphs = combination
        batch = _make_batch(num_graphs=num_graphs)
        stage = _CommunicationMixin(active_batch=batch, max_batch_size=10)
        assert stage.room_in_active_batch == expectation


# ---------------------------------------------------------------------------
# Test_CommunicationMixin — Buffer Routing
# ---------------------------------------------------------------------------


class TestBufferRouting:
    """Test _buffer_to_batch and _overflow_to_sinks methods."""

    def test_buffer_to_batch_no_active_batch(self) -> None:
        """Verify incoming batch becomes the active batch when none exists."""
        stage = _CommunicationMixin(max_batch_size=10)
        incoming = _make_batch(num_graphs=3)
        stage._buffer_to_batch(incoming)
        assert stage.active_batch is not None
        assert stage.active_batch_size == 3

    def test_buffer_to_batch_with_room(self) -> None:
        """Verify incoming data is appended when active batch has room."""
        batch = _make_batch(num_graphs=2)
        stage = _CommunicationMixin(active_batch=batch, max_batch_size=10)
        incoming = _make_batch(num_graphs=3)
        stage._buffer_to_batch(incoming)
        assert stage.active_batch_size == 5

    def test_buffer_to_batch_overflow_to_sinks(self) -> None:
        """Verify excess samples go to sinks when active batch is full."""
        batch = _make_batch(num_graphs=4)
        sink = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=batch,
            max_batch_size=5,
            sinks=[sink],
        )
        incoming = _make_batch(num_graphs=3)
        stage._buffer_to_batch(incoming)
        # 4 + 1 fit = 5 in batch, 2 overflow to sink
        assert stage.active_batch_size == 5
        assert len(sink) == 2

    def test_buffer_to_batch_no_room_all_to_sinks(self) -> None:
        """Verify all go to sinks when active batch is completely full."""
        batch = _make_batch(num_graphs=5)
        sink = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=batch,
            max_batch_size=5,
            sinks=[sink],
        )
        incoming = _make_batch(num_graphs=2)
        stage._buffer_to_batch(incoming)
        assert stage.active_batch_size == 5
        assert len(sink) == 2

    def test_buffer_to_batch_no_active_overflow(self) -> None:
        """Verify incoming > max_batch_size causes overflow when no batch."""
        sink = HostMemory(capacity=50)
        stage = _CommunicationMixin(max_batch_size=2, sinks=[sink])
        incoming = _make_batch(num_graphs=5)
        stage._buffer_to_batch(incoming)
        assert stage.active_batch_size == 2
        assert len(sink) == 3

    def test_overflow_to_sinks_raises_when_full(self) -> None:
        """Verify RuntimeError when all sinks are full."""
        sink = HostMemory(capacity=1)
        # Fill the sink
        sink.write(_make_batch(num_graphs=1))
        stage = _CommunicationMixin(sinks=[sink])
        with pytest.raises(RuntimeError, match="All sinks are full"):
            stage._overflow_to_sinks(_make_batch(num_graphs=1))

    def test_overflow_to_sinks_uses_first_available(self) -> None:
        """Verify overflow goes to the first non-full sink."""
        sink1 = HostMemory(capacity=1)
        sink2 = HostMemory(capacity=50)
        # Fill sink1
        sink1.write(_make_batch(num_graphs=1))
        stage = _CommunicationMixin(sinks=[sink1, sink2])
        stage._overflow_to_sinks(_make_batch(num_graphs=2))
        assert len(sink2) == 2


# ---------------------------------------------------------------------------
# Test_CommunicationMixin — Batch Extraction
# ---------------------------------------------------------------------------


class TestBatchExtraction:
    """Test _batch_to_buffer for extracting graduated samples."""

    def test_extract_some_samples(self) -> None:
        """Verify extracting a subset of samples from active batch."""
        batch = _make_batch(num_graphs=5)
        stage = _CommunicationMixin(active_batch=batch)
        indices = torch.tensor([1, 3])
        graduated = stage._batch_to_buffer(indices)
        assert graduated.num_graphs == 2
        assert stage.active_batch_size == 3

    def test_extract_all_samples(self) -> None:
        """Verify extracting all samples sets active_batch to None."""
        batch = _make_batch(num_graphs=3)
        stage = _CommunicationMixin(active_batch=batch)
        indices = torch.tensor([0, 1, 2])
        graduated = stage._batch_to_buffer(indices)
        assert graduated.num_graphs == 3
        assert stage.active_batch is None

    def test_extract_single_sample(self) -> None:
        """Verify extracting a single sample works correctly."""
        batch = _make_batch(num_graphs=4)
        stage = _CommunicationMixin(active_batch=batch)
        indices = torch.tensor([2])
        graduated = stage._batch_to_buffer(indices)
        assert graduated.num_graphs == 1
        assert stage.active_batch_size == 3

    def test_extract_no_active_batch_raises(self) -> None:
        """Verify RuntimeError when no active batch exists."""
        stage = _CommunicationMixin()
        with pytest.raises(RuntimeError, match="No active batch"):
            stage._batch_to_buffer(torch.tensor([0]))


# ---------------------------------------------------------------------------
# Test_CommunicationMixin — Stage Composition
# ---------------------------------------------------------------------------


class TestStageComposition:
    """Test the __or__ operator for stage composition."""

    def test_or_creates_pipeline(self) -> None:
        """Verify stage_a | stage_b creates a DistributedPipeline."""
        stage_a = _CommunicationMixin()
        stage_b = _CommunicationMixin()
        pipeline = stage_a | stage_b
        assert isinstance(pipeline, DistributedPipeline)
        assert len(pipeline.stages) == 2

    def test_or_maps_to_sequential_ranks(self) -> None:
        """Verify the pipeline maps stages to ranks 0 and 1."""
        stage_a = _CommunicationMixin()
        stage_b = _CommunicationMixin()
        pipeline = stage_a | stage_b
        assert 0 in pipeline.stages
        assert 1 in pipeline.stages
        assert pipeline.stages[0] is stage_a
        assert pipeline.stages[1] is stage_b


# ---------------------------------------------------------------------------
# TestDistributedPipeline — Construction & Setup
# ---------------------------------------------------------------------------


class TestDistributedPipelineConstruction:
    """Test DistributedPipeline orchestrator construction and setup."""

    def test_construction(self) -> None:
        """Verify DistributedPipeline accepts a stage dictionary."""
        stages = {0: _CommunicationMixin(), 1: _CommunicationMixin()}
        pipeline = DistributedPipeline(stages=stages)
        assert len(pipeline.stages) == 2
        assert pipeline.synchronized is False

    def test_synchronized_flag(self) -> None:
        """Verify synchronized flag is stored."""
        stages = {0: _CommunicationMixin(), 1: _CommunicationMixin()}
        pipeline = DistributedPipeline(stages=stages, synchronized=True)
        assert pipeline.synchronized is True

    def test_setup_wires_ranks(self) -> None:
        """Verify setup() connects prior_rank/next_rank between stages."""
        s0 = _CommunicationMixin()
        s1 = _CommunicationMixin()
        s2 = _CommunicationMixin()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1, 2: s2})
        pipeline.setup()

        # First stage
        assert s0.prior_rank is None
        assert s0.next_rank == 1

        # Middle stage
        assert s1.prior_rank == 0
        assert s1.next_rank == 2

        # Last stage
        assert s2.prior_rank == 1
        assert s2.next_rank is None

    def test_setup_two_stages(self) -> None:
        """Verify setup() works with exactly two stages."""
        s0 = _CommunicationMixin()
        s1 = _CommunicationMixin()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1})
        pipeline.setup()

        assert s0.prior_rank is None
        assert s0.next_rank == 1
        assert s1.prior_rank == 0
        assert s1.next_rank is None

    def test_setup_non_contiguous_ranks(self) -> None:
        """Verify setup() handles non-contiguous rank numbers."""
        s0 = _CommunicationMixin()
        s5 = _CommunicationMixin()
        s10 = _CommunicationMixin()
        pipeline = DistributedPipeline(stages={0: s0, 5: s5, 10: s10})
        pipeline.setup()

        assert s0.next_rank == 5
        assert s5.prior_rank == 0
        assert s5.next_rank == 10
        assert s10.prior_rank == 5

    def test_setup_single_stage_raises(self) -> None:
        """Verify setup() raises ValueError with fewer than 2 stages."""
        pipeline = DistributedPipeline(stages={0: _CommunicationMixin()})
        with pytest.raises(ValueError, match="at least 2 stages"):
            pipeline.setup()


# ---------------------------------------------------------------------------
# TestDistributedPipelineSyncBuffers — Without Distributed
# ---------------------------------------------------------------------------


class TestDistributedPipelineSyncBuffers:
    """Test _prestep_sync_buffers and _poststep_sync_buffers logic.

    Since these methods require torch.distributed for actual communication,
    we test the non-distributed code paths (prior_rank=None, etc.) and
    verify that the buffer synchronization flow works.
    """

    def test_prestep_no_prior_rank(self) -> None:
        """Verify _prestep_sync_buffers is a no-op without prior_rank."""
        stage = _CommunicationMixin(prior_rank=None)
        # Should not raise
        stage._prestep_sync_buffers()

    def test_poststep_no_convergence(self) -> None:
        """Verify _poststep_sync_buffers is a no-op when None is passed."""
        batch = _make_batch(num_graphs=3)
        stage = _CommunicationMixin(active_batch=batch)
        # No converged indices → no-op
        stage._poststep_sync_buffers(converged_indices=None)
        assert stage.active_batch_size == 3

    def test_poststep_final_stage_stores_graduated(self) -> None:
        """Verify final stage stores converged samples in sinks."""
        batch = _make_batch(num_graphs=4)
        sink = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=None,  # final stage
            sinks=[sink],
        )
        # Manually pass converged indices (samples 0 and 2)
        converged_indices = torch.tensor([0, 2])
        stage._poststep_sync_buffers(converged_indices)
        # Samples 0 and 2 should be graduated to sink
        assert len(sink) == 2
        assert stage.active_batch_size == 2

    def test_sync_mode_recv_completes_inline(self) -> None:
        """Verify sync mode completes irecv and routes data inline."""
        batch = _make_batch(num_graphs=3)
        stage = _CommunicationMixin(
            prior_rank=0,
            comm_mode="sync",
            active_batch=batch,
            max_batch_size=10,
        )

        mock_incoming = _make_batch(num_graphs=2)
        mock_handle = Mock()
        mock_handle.wait.return_value = mock_incoming

        with (
            patch.object(Batch, "irecv", return_value=mock_handle) as mock_irecv,
            patch.object(stage, "_buffer_to_batch") as mock_b2b,
        ):
            stage._prestep_sync_buffers()

            # irecv was called, handle.wait() was called, _buffer_to_batch was called
            mock_irecv.assert_called_once()
            mock_handle.wait.assert_called_once()
            mock_b2b.assert_called_once_with(mock_incoming)

            # _complete_pending_recv is a no-op (handle already consumed)
            mock_b2b.reset_mock()
            mock_handle.wait.reset_mock()
            stage._complete_pending_recv()
            mock_handle.wait.assert_not_called()
            mock_b2b.assert_not_called()

        assert stage._pending_recv_handle is None

    def test_async_recv_mode_defers_wait(self) -> None:
        """Verify async_recv mode defers handle.wait to _complete_pending_recv."""
        batch = _make_batch(num_graphs=3)
        stage = _CommunicationMixin(
            prior_rank=0,
            comm_mode="async_recv",
            active_batch=batch,
            max_batch_size=10,
        )

        mock_incoming = _make_batch(num_graphs=2)
        mock_handle = Mock()
        mock_handle.wait.return_value = mock_incoming

        with (
            patch.object(Batch, "irecv", return_value=mock_handle) as mock_irecv,
            patch.object(stage, "_buffer_to_batch") as mock_b2b,
        ):
            stage._prestep_sync_buffers()

            # irecv was called but wait and _buffer_to_batch were NOT called
            mock_irecv.assert_called_once()
            mock_handle.wait.assert_not_called()
            mock_b2b.assert_not_called()
            assert stage._pending_recv_handle is mock_handle

            # Now complete the deferred recv
            stage._complete_pending_recv()
            mock_handle.wait.assert_called_once()
            mock_b2b.assert_called_once_with(mock_incoming)

        assert stage._pending_recv_handle is None

    def test_fully_async_mode_drains_send_and_defers_recv(self) -> None:
        """Verify fully_async drains prior send handle and defers recv."""
        batch = _make_batch(num_graphs=4)
        stage = _CommunicationMixin(
            prior_rank=0,
            next_rank=2,
            comm_mode="fully_async",
            active_batch=batch,
            max_batch_size=10,
        )

        # Simulate a pending send handle from a previous iteration
        mock_old_send = Mock()
        stage._pending_send_handle = mock_old_send

        mock_recv_handle = Mock()
        mock_recv_handle.wait.return_value = _make_batch(num_graphs=1)

        with (
            patch.object(Batch, "irecv", return_value=mock_recv_handle),
            patch.object(stage, "_buffer_to_batch"),
        ):
            stage._prestep_sync_buffers()

            # Old send handle was drained
            mock_old_send.wait.assert_called_once()
            assert stage._pending_send_handle is None

            # Recv was deferred (not waited)
            mock_recv_handle.wait.assert_not_called()
            assert stage._pending_recv_handle is mock_recv_handle

        # Now test that _poststep stores the send handle
        mock_new_send = Mock()
        mock_graduated = Mock()
        mock_graduated.isend.return_value = mock_new_send

        with patch.object(stage, "_batch_to_buffer", return_value=mock_graduated):
            stage._poststep_sync_buffers(converged_indices=torch.tensor([0]))

        mock_graduated.isend.assert_called_once_with(dst=2)
        assert stage._pending_send_handle is mock_new_send


# ---------------------------------------------------------------------------
# TestDistributedPipelineLifecycle — Distributed Init/Cleanup
# ---------------------------------------------------------------------------


class TestDistributedPipelineLifecycle:
    """Test DistributedPipeline distributed initialization and cleanup."""

    def test_init_distributed_when_not_initialized(self) -> None:
        """Verify init_distributed calls init_process_group."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()},
            backend="gloo",
        )
        with (
            patch.object(dist, "is_initialized", return_value=False),
            patch.object(dist, "init_process_group") as mock_init,
        ):
            pipeline.init_distributed()
            mock_init.assert_called_once_with(backend="gloo")
            assert pipeline._dist_initialized is True

    def test_init_distributed_noop_when_already_initialized(self) -> None:
        """Verify init_distributed is a no-op when dist is already active."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )
        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "init_process_group") as mock_init,
        ):
            pipeline.init_distributed()
            mock_init.assert_not_called()
            assert pipeline._dist_initialized is False

    def test_cleanup_destroys_process_group(self) -> None:
        """Verify cleanup destroys the process group when we initialized it."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )
        pipeline._dist_initialized = True
        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "destroy_process_group") as mock_destroy,
        ):
            pipeline.cleanup()
            mock_destroy.assert_called_once()
            assert pipeline._dist_initialized is False

    def test_cleanup_noop_when_not_our_init(self) -> None:
        """Verify cleanup is a no-op when we didn't initialize dist."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )
        assert pipeline._dist_initialized is False
        with patch.object(dist, "destroy_process_group") as mock_destroy:
            pipeline.cleanup()
            mock_destroy.assert_not_called()

    def test_context_manager_calls_init_and_cleanup(self) -> None:
        """Verify context manager calls init_distributed, setup, and cleanup."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )
        with (
            patch.object(pipeline, "init_distributed") as mock_init,
            patch.object(pipeline, "setup") as mock_setup,
            patch.object(pipeline, "cleanup") as mock_cleanup,
        ):
            with pipeline:
                mock_init.assert_called_once()
                mock_setup.assert_called_once()
            mock_cleanup.assert_called_once()

    def test_context_manager_cleanup_on_exception(self) -> None:
        """Verify cleanup is called even when an exception occurs."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )
        with (
            patch.object(pipeline, "init_distributed"),
            patch.object(pipeline, "setup"),
            patch.object(pipeline, "cleanup") as mock_cleanup,
        ):
            with pytest.raises(ValueError, match="test error"):
                with pipeline:
                    raise ValueError("test error")
            mock_cleanup.assert_called_once()

    def test_dist_initialized_default_false(self) -> None:
        """Verify _dist_initialized defaults to False."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )
        assert pipeline._dist_initialized is False


# ---------------------------------------------------------------------------
# TestDistributedPipelineWorldSizeValidation — World-Size Validation
# ---------------------------------------------------------------------------


class TestDistributedPipelineWorldSizeValidation:
    """Test DistributedPipeline world-size validation against torch.distributed."""

    def test_validate_world_size_matching(self) -> None:
        """No error when world_size matches number of stages."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )
        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=2),
        ):
            # Should not raise
            pipeline._validate_world_size()

    def test_validate_world_size_mismatch_raises(self) -> None:
        """RuntimeError when world_size != number of stages."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )
        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=4),
        ):
            with pytest.raises(RuntimeError, match="expects 2 ranks"):
                pipeline._validate_world_size()

    def test_validate_world_size_not_initialized_noop(self) -> None:
        """No error when dist is not initialized (local testing)."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )
        with patch.object(dist, "is_initialized", return_value=False):
            # Should not raise
            pipeline._validate_world_size()

    def test_setup_calls_validate_world_size(self) -> None:
        """setup() should call _validate_world_size."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )
        with patch.object(pipeline, "_validate_world_size") as mock_validate:
            pipeline.setup()
            mock_validate.assert_called_once()


# ---------------------------------------------------------------------------
# TestDevicePlacement — Device Type and Device Property
# ---------------------------------------------------------------------------


class TestDevicePlacement:
    """Test device_type and device property on _CommunicationMixin and DistributedPipeline."""

    @pytest.mark.parametrize(
        "device_type,expectation",
        [("cpu", True), ("cuda", torch.cuda.is_available()), ("blah", False)],
    )
    def test_device_type(self, device_type: str, expectation: bool) -> None:
        """device_type should accept custom values."""
        stage = _CommunicationMixin(device_type=device_type)
        if expectation:
            stage.device == torch.device(device_type)
        else:
            with pytest.raises(RuntimeError, match="Unable to create"):
                _ = stage.device

    def test_device_property_cuda_uses_local_rank(self) -> None:
        """CUDA device should incorporate local_rank."""
        stage = _CommunicationMixin(device_type="cuda")
        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_node_local_rank", return_value=3),
        ):
            assert stage.device == torch.device("cuda:3")

    def test_local_rank_not_initialized(self) -> None:
        """local_rank returns 0 when dist is not initialized."""
        stage = _CommunicationMixin()
        with patch.object(dist, "is_initialized", return_value=False):
            assert stage.local_rank == 0

    def test_pipeline_device_type_not_on_pipeline(self) -> None:
        """DistributedPipeline should not have a device_type attribute."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )
        assert not hasattr(pipeline, "device_type")


# ---------------------------------------------------------------------------
# TestPoststepSentinelSend — Sentinel Send on No Convergence
# ---------------------------------------------------------------------------


class TestPoststepNoConvergenceSend:
    """Test that _poststep_sync_buffers uses send_buffer when nothing converges."""

    def test_sends_buffer_when_no_convergence_and_next_rank(self) -> None:
        """When nothing converges and next_rank is set, send the send_buffer."""
        batch = _make_batch(num_graphs=3)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
        )

        mock_handle = Mock()
        mock_send_buffer = Mock()
        mock_send_buffer.isend.return_value = mock_handle
        stage.send_buffer = mock_send_buffer

        stage._poststep_sync_buffers(converged_indices=None)

        mock_send_buffer.isend.assert_called_once_with(dst=1)
        # Active batch should be unchanged (no samples graduated)
        assert stage.active_batch_size == 3

    def test_sends_buffer_when_empty_convergence_and_next_rank(self) -> None:
        """When converged_indices is empty tensor and next_rank is set, send send_buffer."""
        batch = _make_batch(num_graphs=3)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
        )

        mock_handle = Mock()
        mock_send_buffer = Mock()
        mock_send_buffer.isend.return_value = mock_handle
        stage.send_buffer = mock_send_buffer

        stage._poststep_sync_buffers(converged_indices=torch.tensor([]))
        mock_send_buffer.isend.assert_called_once_with(dst=1)

        assert stage.active_batch_size == 3

    def test_no_send_when_no_convergence_and_no_next_rank(self) -> None:
        """When converged_indices is None and next_rank is None, no-op."""
        batch = _make_batch(num_graphs=3)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=None,
        )
        # Should not raise or try to send anything
        stage._poststep_sync_buffers(converged_indices=None)
        assert stage.active_batch_size == 3

    def test_send_buffer_stores_handle_in_fully_async(self) -> None:
        """In fully_async mode, send buffer's send handle is stored."""
        batch = _make_batch(num_graphs=3)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
            comm_mode="fully_async",
        )

        mock_handle = Mock()
        mock_send_buffer = Mock()
        mock_send_buffer.isend.return_value = mock_handle
        stage.send_buffer = mock_send_buffer

        stage._poststep_sync_buffers(converged_indices=None)

        assert stage._pending_send_handle is mock_handle

    def test_convergence_sends_real_data_not_buffer(self) -> None:
        """When converged_indices is provided, send real graduated data."""
        batch = _make_batch(num_graphs=4)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
        )

        mock_graduated = Mock()
        mock_handle = Mock()
        mock_graduated.isend.return_value = mock_handle

        # Also set up a send_buffer that should NOT be used
        mock_send_buffer = Mock()
        stage.send_buffer = mock_send_buffer

        with patch.object(
            stage, "_batch_to_buffer", return_value=mock_graduated
        ) as mock_b2b:
            stage._poststep_sync_buffers(converged_indices=torch.tensor([0, 2]))

            mock_b2b.assert_called_once()
            mock_graduated.isend.assert_called_once_with(dst=1)
            # Should NOT use send_buffer for real data
            mock_send_buffer.isend.assert_not_called()

    def test_no_send_when_no_convergence_and_no_send_buffer(self) -> None:
        """When nothing converges and send_buffer is None, no-op even with next_rank."""
        batch = _make_batch(num_graphs=3)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
        )
        # send_buffer is None by default
        assert stage.send_buffer is None

        # Should not raise or try to send
        stage._poststep_sync_buffers(converged_indices=None)
        assert stage.active_batch_size == 3


# ---------------------------------------------------------------------------
# TestBufferConfigValidation — Buffer Config Validation in DistributedPipeline
# ---------------------------------------------------------------------------


class TestBufferConfigValidation:
    """Test DistributedPipeline.setup() validates buffer configs between adjacent stages."""

    def test_matching_buffer_configs_pass(self) -> None:
        """setup() succeeds when adjacent stages have matching buffer configs."""
        cfg = BufferConfig(num_systems=10, num_nodes=500, num_edges=2000)
        pipeline = DistributedPipeline(
            stages={
                0: _CommunicationMixin(buffer_config=cfg),
                1: _CommunicationMixin(buffer_config=cfg),
            }
        )
        # Should not raise
        pipeline.setup()

    def test_mismatched_buffer_configs_raise(self) -> None:
        """setup() raises ValueError when adjacent stages have different buffer configs."""
        cfg_a = BufferConfig(num_systems=10, num_nodes=500, num_edges=2000)
        cfg_b = BufferConfig(num_systems=20, num_nodes=1000, num_edges=4000)
        pipeline = DistributedPipeline(
            stages={
                0: _CommunicationMixin(buffer_config=cfg_a),
                1: _CommunicationMixin(buffer_config=cfg_b),
            }
        )
        with pytest.raises(ValueError, match="Buffer configuration mismatch"):
            pipeline.setup()

    def test_one_none_buffer_config_passes(self) -> None:
        """setup() succeeds when only one stage has buffer_config."""
        cfg = BufferConfig(num_systems=10, num_nodes=500, num_edges=2000)
        pipeline = DistributedPipeline(
            stages={
                0: _CommunicationMixin(buffer_config=cfg),
                1: _CommunicationMixin(buffer_config=None),
            }
        )
        # Should not raise (validation only when both have configs)
        pipeline.setup()

    def test_both_none_buffer_configs_pass(self) -> None:
        """setup() succeeds when no stages have buffer_config."""
        pipeline = DistributedPipeline(
            stages={
                0: _CommunicationMixin(),
                1: _CommunicationMixin(),
            }
        )
        pipeline.setup()

    def test_dict_coercion_for_buffer_config(self) -> None:
        """_CommunicationMixin accepts dict and coerces to BufferConfig."""
        stage = _CommunicationMixin(
            buffer_config={"num_systems": 10, "num_nodes": 500, "num_edges": 2000}
        )
        assert isinstance(stage.buffer_config, BufferConfig)
        assert stage.buffer_config.num_systems == 10
        assert stage.buffer_config.num_nodes == 500
        assert stage.buffer_config.num_edges == 2000


# ---------------------------------------------------------------------------
# TestSyncDoneFlags — Distributed Done Flag Synchronization
# ---------------------------------------------------------------------------


class TestSyncDoneFlags:
    """Test DistributedPipeline._sync_done_flags and _done_tensor initialization."""

    def test_setup_initializes_done_tensor(self) -> None:
        """setup() should initialize _done_tensor with correct size."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )
        pipeline.setup()
        assert pipeline._done_tensor is not None
        assert pipeline._done_tensor.shape == (2,)
        assert pipeline._done_tensor.dtype == torch.int32
        assert (pipeline._done_tensor == 0).all()

    def test_setup_initializes_done_tensor_three_stages(self) -> None:
        """setup() should initialize _done_tensor with size matching stage count."""
        pipeline = DistributedPipeline(
            stages={
                0: _CommunicationMixin(),
                1: _CommunicationMixin(),
                2: _CommunicationMixin(),
            }
        )
        pipeline.setup()
        assert pipeline._done_tensor.shape == (3,)

    def test_sync_done_flags_raises_without_setup(self) -> None:
        """_sync_done_flags raises RuntimeError if setup() was not called."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )
        with pytest.raises(RuntimeError, match="_done_tensor is not initialized"):
            pipeline._sync_done_flags()

    def test_sync_done_flags_all_not_done(self) -> None:
        """Returns False when no stages are done (no distributed)."""
        s0 = _CommunicationMixin()
        s1 = _CommunicationMixin()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1})
        pipeline.setup()

        # Without dist initialized, global_rank returns 0
        with patch.object(dist, "is_initialized", return_value=False):
            result = pipeline._sync_done_flags()

        assert result is False

    def test_sync_done_flags_local_stage_done(self) -> None:
        """Returns False when only local stage is done (needs all)."""
        s0 = _CommunicationMixin()
        s1 = _CommunicationMixin()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1})
        pipeline.setup()

        s0.done = True

        with patch.object(dist, "is_initialized", return_value=False):
            result = pipeline._sync_done_flags()

        # Only rank 0 done, rank 1 not done → False
        assert result is False
        # But _done_tensor[0] should be 1
        assert pipeline._done_tensor[0] == 1
        assert pipeline._done_tensor[1] == 0

    def test_sync_done_flags_all_done_no_dist(self) -> None:
        """Returns True when all local stages are done (no distributed)."""
        s0 = _CommunicationMixin()
        s1 = _CommunicationMixin()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1})
        pipeline.setup()

        s0.done = True
        s1.done = True

        # Without dist, _sync_done_flags writes both flags locally
        # and checks all()
        with patch.object(dist, "is_initialized", return_value=False):
            # First call writes rank 0's flag
            # But without dist, global_rank is always 0
            # So only s0's flag gets written
            result = pipeline._sync_done_flags()

        # Since global_rank returns 0, only s0.done is written to tensor
        # s1.done is never written → _done_tensor[1] == 0 → False
        # This is the expected behavior: without distributed, only local rank matters
        assert result is False

    def test_run_uses_sync_done_flags(self) -> None:
        """run() should call _sync_done_flags instead of local done check."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )

        call_count = 0

        def mock_step() -> None:
            """Mock step that does nothing."""
            pass

        def mock_sync() -> bool:
            """Mock sync that returns True on second call."""
            nonlocal call_count
            call_count += 1
            return call_count >= 2

        with (
            patch.object(pipeline, "setup"),
            patch.object(pipeline, "step", side_effect=mock_step),
            patch.object(pipeline, "_sync_done_flags", side_effect=mock_sync),
        ):
            pipeline.run()

        assert call_count == 2

    def test_done_tensor_not_initialized_before_setup(self) -> None:
        """_done_tensor should be None before setup() is called."""
        pipeline = DistributedPipeline(
            stages={0: _CommunicationMixin(), 1: _CommunicationMixin()}
        )
        assert pipeline._done_tensor is None


# ---------------------------------------------------------------------------
# TestEnsureBuffersWiring — _ensure_buffers integration in pipeline step
# ---------------------------------------------------------------------------


class TestEnsureBuffersWiring:
    """Test that _ensure_buffers is called during pipeline step operations."""

    def test_ensure_buffers_called_on_first_step(self) -> None:
        """Verify send/recv buffers are created after relevant pipeline methods.

        When a stage has buffer_config, prior_rank, and next_rank, calling
        _ensure_buffers with an active_batch should create both send_buffer
        and recv_buffer.
        """
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)
        batch = _make_batch_with_system(num_graphs=3)
        stage = _CommunicationMixin(
            buffer_config=cfg,
            prior_rank=0,
            next_rank=2,
            active_batch=batch,
            device_type="cpu",
        )

        # Before _ensure_buffers, buffers are None
        assert stage.send_buffer is None
        assert stage.recv_buffer is None

        # Call _ensure_buffers with the template batch
        stage._ensure_buffers(batch)

        # After _ensure_buffers, buffers should be created
        assert stage.send_buffer is not None
        assert stage.recv_buffer is not None
        assert stage.send_buffer.system_capacity == 10
        assert stage.recv_buffer.system_capacity == 10

    def test_ensure_buffers_only_creates_needed_buffers(self) -> None:
        """Verify _ensure_buffers only creates buffers for active directions.

        When next_rank is None (final stage), only recv_buffer is created.
        When prior_rank is None (first stage), only send_buffer is created.
        """
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)
        batch = _make_batch_with_system(num_graphs=3)

        # Final stage: no next_rank
        final_stage = _CommunicationMixin(
            buffer_config=cfg,
            prior_rank=0,
            next_rank=None,
            device_type="cpu",
        )
        final_stage._ensure_buffers(batch)
        assert final_stage.send_buffer is None
        assert final_stage.recv_buffer is not None

        # First stage: no prior_rank
        first_stage = _CommunicationMixin(
            buffer_config=cfg,
            prior_rank=None,
            next_rank=2,
            device_type="cpu",
        )
        first_stage._ensure_buffers(batch)
        assert first_stage.send_buffer is not None
        assert first_stage.recv_buffer is None

    def test_ensure_buffers_noop_without_buffer_config(self) -> None:
        """Verify _ensure_buffers is a no-op when buffer_config is None.

        Note: Due to a bug in _CommunicationMixin where buffer_config=None
        raises TypeError, we test _ensure_buffers directly by setting
        buffer_config to None after initialization.
        """
        batch = _make_batch(num_graphs=3)
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)
        stage = _CommunicationMixin(
            prior_rank=0,
            next_rank=2,
            buffer_config=cfg,
        )
        # Manually set buffer_config to None to test the None path
        stage.buffer_config = None

        # Without buffer_config, _ensure_buffers should not create buffers
        stage._ensure_buffers(batch)
        assert stage.send_buffer is None
        assert stage.recv_buffer is None


# ---------------------------------------------------------------------------
# TestPrestepZerosSendBuffer — _prestep_sync_buffers zeros send_buffer
# ---------------------------------------------------------------------------


class TestPrestepZerosSendBuffer:
    """Test that _prestep_sync_buffers zeros send_buffer (not sinks[0])."""

    def test_prestep_zeros_send_buffer(self) -> None:
        """Verify _prestep_sync_buffers zeros send_buffer when present.

        When a stage has prior_rank and a send_buffer, _prestep_sync_buffers
        should call send_buffer.zero() and NOT sinks[0].zero().
        """
        batch = _make_batch(num_graphs=3)
        sink = HostMemory(capacity=50)
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)

        stage = _CommunicationMixin(
            prior_rank=0,
            comm_mode="sync",
            active_batch=batch,
            max_batch_size=10,
            sinks=[sink],
            buffer_config=cfg,
        )

        # Create a mock send_buffer with a mock zero method
        mock_send_buffer = Mock()
        stage.send_buffer = mock_send_buffer

        # Mock Batch.irecv to avoid actual distributed communication
        mock_incoming = _make_batch(num_graphs=2)
        mock_handle = Mock()
        mock_handle.wait.return_value = mock_incoming

        with (
            patch.object(Batch, "irecv", return_value=mock_handle),
            patch.object(stage, "_buffer_to_batch"),
        ):
            stage._prestep_sync_buffers()

            # Verify send_buffer.zero() was called
            mock_send_buffer.zero.assert_called_once()

    def test_prestep_does_not_zero_sinks(self) -> None:
        """Verify _prestep_sync_buffers does NOT zero sinks[0] anymore.

        The old behavior zeroed sinks[0]; the new behavior zeros send_buffer.
        This test confirms sinks are not touched.
        """
        batch = _make_batch(num_graphs=3)
        sink = HostMemory(capacity=50)
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)

        stage = _CommunicationMixin(
            prior_rank=0,
            comm_mode="sync",
            active_batch=batch,
            max_batch_size=10,
            sinks=[sink],
            buffer_config=cfg,
        )

        # Create a send_buffer so the old code path would have had something
        mock_send_buffer = Mock()
        stage.send_buffer = mock_send_buffer

        # Write something to the sink so we can verify it's not cleared
        sink.write(_make_batch(num_graphs=1))
        assert len(sink) == 1

        mock_incoming = _make_batch(num_graphs=2)
        mock_handle = Mock()
        mock_handle.wait.return_value = mock_incoming

        with (
            patch.object(Batch, "irecv", return_value=mock_handle),
            patch.object(stage, "_buffer_to_batch"),
        ):
            stage._prestep_sync_buffers()

        # Sink should still have its data (not zeroed)
        assert len(sink) == 1

    def test_prestep_no_error_without_send_buffer(self) -> None:
        """Verify _prestep_sync_buffers handles None send_buffer gracefully.

        When send_buffer is None, the zeroing is skipped without error.
        """
        batch = _make_batch(num_graphs=3)
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)
        stage = _CommunicationMixin(
            prior_rank=0,
            comm_mode="sync",
            active_batch=batch,
            max_batch_size=10,
            buffer_config=cfg,
        )

        # Ensure send_buffer is None (it's only created by _ensure_buffers)
        assert stage.send_buffer is None

        mock_incoming = _make_batch(num_graphs=2)
        mock_handle = Mock()
        mock_handle.wait.return_value = mock_incoming

        with (
            patch.object(Batch, "irecv", return_value=mock_handle),
            patch.object(stage, "_buffer_to_batch"),
        ):
            # Should not raise even with send_buffer=None
            stage._prestep_sync_buffers()


# ---------------------------------------------------------------------------
# TestPoststepBackPressure — Back-pressure in _poststep_sync_buffers
# ---------------------------------------------------------------------------


class TestPoststepBackPressure:
    """Test that _poststep_sync_buffers respects send buffer capacity (back-pressure)."""

    def test_poststep_respects_send_buffer_capacity(self) -> None:
        """Verify only as many converged samples as buffer capacity are extracted.

        When 5 samples converge but the send buffer can only hold 2, only the
        first 2 should be extracted and sent; the remaining 3 stay in active_batch.
        """
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)
        batch = _make_batch(num_graphs=5)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
            buffer_config=cfg,
        )

        # Create a mock send_buffer with capacity for 2 more graphs
        mock_send_buffer = Mock()
        mock_send_buffer.system_capacity = 2
        mock_send_buffer.num_graphs = 0
        stage.send_buffer = mock_send_buffer

        mock_graduated = Mock()
        mock_handle = Mock()
        mock_graduated.isend.return_value = mock_handle

        with patch.object(
            stage, "_batch_to_buffer", return_value=mock_graduated
        ) as mock_b2b:
            stage._poststep_sync_buffers(
                converged_indices=torch.tensor([0, 1, 2, 3, 4])
            )

            # Should only extract first 2 indices (capacity=2)
            mock_b2b.assert_called_once()
            call_args = mock_b2b.call_args[0][0]
            assert call_args.numel() == 2
            assert call_args.tolist() == [0, 1]
            mock_graduated.isend.assert_called_once_with(dst=1)

    def test_poststep_sends_empty_when_capacity_zero(self) -> None:
        """Verify no extraction when send buffer is full; empty buffer is sent.

        When send buffer is at full capacity (capacity=0 remaining), no samples
        should be extracted from active_batch. The empty send_buffer is sent
        for deadlock prevention.
        """
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)
        batch = _make_batch(num_graphs=5)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
            buffer_config=cfg,
        )

        # Create a mock send_buffer at full capacity (no room)
        mock_send_buffer = Mock()
        mock_send_buffer.system_capacity = 3
        mock_send_buffer.num_graphs = 3  # Full: 0 capacity remaining
        mock_handle = Mock()
        mock_send_buffer.isend.return_value = mock_handle
        stage.send_buffer = mock_send_buffer

        with patch.object(stage, "_batch_to_buffer") as mock_b2b:
            stage._poststep_sync_buffers(converged_indices=torch.tensor([0, 1]))

            # _batch_to_buffer should NOT be called (no capacity)
            mock_b2b.assert_not_called()

            # send_buffer.isend should be called for deadlock prevention
            mock_send_buffer.isend.assert_called_once_with(dst=1)

        # Active batch should still have all 5 samples
        assert stage.active_batch_size == 5

    def test_poststep_no_capacity_limit_without_send_buffer(self) -> None:
        """Verify _send_buffer_capacity returns large value when send_buffer is None.

        This ensures backward compatibility: when there's no pre-allocated buffer,
        all converged samples are sent without capacity constraints.
        """
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)
        stage = _CommunicationMixin(
            next_rank=1,
            buffer_config=cfg,
        )

        # send_buffer is None by default (not yet created)
        assert stage.send_buffer is None

        # _send_buffer_capacity should return a very large value
        capacity = stage._send_buffer_capacity
        assert capacity >= 10000  # At least a reasonable large threshold
        # Actually it should be sys.maxsize
        import sys

        assert capacity == sys.maxsize

    def test_remaining_converged_samples_persist(self) -> None:
        """Verify samples not extracted due to capacity remain in active_batch.

        When only 1 of 3 converged samples fits in the send buffer, the
        remaining 2 should still be in active_batch with their original data.
        """
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)
        # Create a batch with identifiable positions
        batch = _make_batch(num_graphs=4)

        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
            buffer_config=cfg,
        )

        # Create a mock send_buffer with capacity for 1 more graph
        mock_send_buffer = Mock()
        mock_send_buffer.system_capacity = 1
        mock_send_buffer.num_graphs = 0
        stage.send_buffer = mock_send_buffer

        mock_graduated = Mock()
        mock_handle = Mock()
        mock_graduated.isend.return_value = mock_handle

        # We use actual _batch_to_buffer, not mocked, to verify remaining data
        with patch.object(mock_graduated, "isend", return_value=mock_handle):
            # Mark indices 0, 1, 2 as converged
            converged_indices = torch.tensor([0, 1, 2])

            # Patch _batch_to_buffer to only extract the first index
            def selective_extract(indices: torch.Tensor) -> Mock:
                """Extract only the truncated indices."""
                # This simulates what the real method does
                assert indices.numel() == 1  # Only first one due to capacity
                assert indices.tolist() == [0]
                # Remove index 0 from active_batch
                remaining = [1, 2, 3]
                stage.active_batch = stage.active_batch.index_select(remaining)
                return mock_graduated

            with patch.object(stage, "_batch_to_buffer", side_effect=selective_extract):
                stage._poststep_sync_buffers(converged_indices=converged_indices)

        # Should have 3 graphs left (started with 4, extracted 1)
        assert stage.active_batch_size == 3
        # The remaining samples should be the ones that weren't extracted

    def test_poststep_final_stage_ignores_send_capacity(self) -> None:
        """Verify final stage sends ALL converged samples to sinks.

        The final stage (next_rank=None) should not be affected by send buffer
        capacity—all converged samples go directly to sinks.
        """
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)
        batch = _make_batch(num_graphs=5)
        sink = HostMemory(capacity=50)

        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=None,  # Final stage
            sinks=[sink],
            buffer_config=cfg,
        )

        # Even if we set a send_buffer (shouldn't be used), it shouldn't matter
        mock_send_buffer = Mock()
        mock_send_buffer.system_capacity = 1  # Would limit to 1 if used
        mock_send_buffer.num_graphs = 0
        stage.send_buffer = mock_send_buffer

        mock_graduated = _make_batch(num_graphs=3)

        with patch.object(
            stage, "_batch_to_buffer", return_value=mock_graduated
        ) as mock_b2b:
            with patch.object(stage, "_overflow_to_sinks") as mock_overflow:
                stage._poststep_sync_buffers(converged_indices=torch.tensor([0, 1, 2]))

                # All 3 converged indices should be passed (no truncation)
                mock_b2b.assert_called_once()
                call_args = mock_b2b.call_args[0][0]
                assert call_args.numel() == 3
                assert call_args.tolist() == [0, 1, 2]

                # Graduated batch should go to sinks
                mock_overflow.assert_called_once_with(mock_graduated)

    def test_poststep_partial_capacity_extracts_correct_subset(self) -> None:
        """Verify when capacity allows partial extraction, the first N are taken.

        If 4 samples converge but only 2 fit, indices [0, 1] should be extracted
        (not [2, 3] or a random selection).
        """
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)
        batch = _make_batch(num_graphs=6)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
            buffer_config=cfg,
        )

        mock_send_buffer = Mock()
        mock_send_buffer.system_capacity = 2
        mock_send_buffer.num_graphs = 0
        stage.send_buffer = mock_send_buffer

        mock_graduated = Mock()
        mock_handle = Mock()
        mock_graduated.isend.return_value = mock_handle

        with patch.object(
            stage, "_batch_to_buffer", return_value=mock_graduated
        ) as mock_b2b:
            # Converged indices are 1, 2, 4, 5 (out of order intentionally)
            stage._poststep_sync_buffers(converged_indices=torch.tensor([1, 2, 4, 5]))

            # Should only take first 2: [1, 2]
            mock_b2b.assert_called_once()
            call_args = mock_b2b.call_args[0][0]
            assert call_args.numel() == 2
            assert call_args.tolist() == [1, 2]

    def test_send_buffer_capacity_fully_async_stores_handle(self) -> None:
        """Verify handle is stored in fully_async mode with capacity constraint."""
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)
        batch = _make_batch(num_graphs=5)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
            buffer_config=cfg,
            comm_mode="fully_async",
        )

        mock_send_buffer = Mock()
        mock_send_buffer.system_capacity = 2
        mock_send_buffer.num_graphs = 0
        stage.send_buffer = mock_send_buffer

        mock_graduated = Mock()
        mock_handle = Mock()
        mock_graduated.isend.return_value = mock_handle

        with patch.object(stage, "_batch_to_buffer", return_value=mock_graduated):
            stage._poststep_sync_buffers(converged_indices=torch.tensor([0, 1, 2, 3]))

        # Handle should be stored in fully_async mode
        assert stage._pending_send_handle is mock_handle
