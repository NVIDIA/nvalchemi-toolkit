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
    BaseDynamics,
    BufferConfig,
    DistributedPipeline,
    _CommunicationMixin,
)
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

_DEFAULT_CFG = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)

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
        An AtomicData instance with system-level energy.
    """
    return AtomicData(
        atomic_numbers=torch.randint(1, 20, (num_atoms,)),
        positions=torch.randn(num_atoms, 3),
        energy=torch.tensor([[0.0]]),
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
        assert stage.prior_rank == -1
        assert stage.next_rank == -1
        assert stage.sinks == []
        assert stage.active_batch is None
        assert stage.done is False
        assert stage._admission_initialized is False
        assert stage._forces_primed is False

    def test_custom_construction(self) -> None:
        """Verify custom field values are respected."""
        buf = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            prior_rank=0,
            next_rank=2,
            sinks=[buf],
            max_batch_size=200,
            buffer_config=_DEFAULT_CFG,
        )
        assert stage.prior_rank == 0
        assert stage.next_rank == 2
        assert len(stage.sinks) == 1
        assert stage.max_batch_size == 200

    @pytest.mark.parametrize(
        ("expected", "next_rank"), [(True, None), (False, 1), (False, 6)]
    )
    def test_is_final_stage(self, expected: bool, next_rank: int | None) -> None:
        """Check is_final_stage matches next_rank."""
        kwargs: dict = {"next_rank": next_rank}
        if next_rank is not None:
            kwargs["buffer_config"] = _DEFAULT_CFG
        stage = _CommunicationMixin(**kwargs)
        assert stage.is_final_stage is expected

    @pytest.mark.parametrize(
        ("expected", "prior_rank"), [(True, None), (False, 1), (False, 6)]
    )
    def test_is_first_stage(self, expected: bool, prior_rank: int | None) -> None:
        """Check is_first_stage matches prior_rank."""
        kwargs: dict = {"prior_rank": prior_rank}
        if prior_rank is not None:
            kwargs["buffer_config"] = _DEFAULT_CFG
        stage = _CommunicationMixin(**kwargs)
        assert stage.is_first_stage is expected

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
        stage._admission_initialized = True
        stage._forces_primed = True
        incoming = _make_batch(num_graphs=3)
        stage._buffer_to_batch(incoming)
        assert stage.active_batch_size == 5
        assert stage._pending_graph_indices.tolist() == [0, 1, -1, -1, -1]
        assert stage._admission_initialized is False
        assert stage._forces_primed is False

    def test_buffer_to_batch_overflow_to_sinks(self) -> None:
        """Verify excess samples go to overflow sinks when active batch is full."""
        batch = _make_batch(num_graphs=4)
        sink = HostMemory(capacity=50)
        results = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=batch,
            max_batch_size=5,
            overflow_sinks=[sink],
            sinks=[results],
        )
        incoming = _make_batch(num_graphs=3)
        stage._buffer_to_batch(incoming)
        # 4 + 1 fit = 5 in batch, 2 overflow to sink
        assert stage.active_batch_size == 5
        assert len(sink) == 2
        assert len(results) == 0

    def test_buffer_to_batch_no_room_all_to_sinks(self) -> None:
        """Verify all go to overflow sinks when active batch is completely full."""
        batch = _make_batch(num_graphs=5)
        sink = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=batch,
            max_batch_size=5,
            overflow_sinks=[sink],
        )
        stage._admission_initialized = True
        incoming = _make_batch(num_graphs=2)
        stage._buffer_to_batch(incoming)
        assert stage.active_batch_size == 5
        assert len(sink) == 2
        assert stage._admission_initialized
        assert stage._pending_graph_indices is None

    def test_buffer_to_batch_no_active_overflow(self) -> None:
        """Verify incoming > max_batch_size causes overflow when no batch."""
        sink = HostMemory(capacity=50)
        stage = _CommunicationMixin(max_batch_size=2, overflow_sinks=[sink])
        incoming = _make_batch(num_graphs=5)
        stage._buffer_to_batch(incoming)
        assert stage.active_batch_size == 2
        assert len(sink) == 3

    def test_overflow_to_sinks_raises_when_full(self) -> None:
        """Verify RuntimeError when all overflow sinks are full."""
        sink = HostMemory(capacity=1)
        # Fill the sink
        sink.write(_make_batch(num_graphs=1))
        stage = _CommunicationMixin(overflow_sinks=[sink])
        with pytest.raises(RuntimeError, match="All sinks are full"):
            stage._overflow_to_sinks(_make_batch(num_graphs=1))

    def test_overflow_to_sinks_uses_first_available(self) -> None:
        """Skip a partly filled sink without room for the whole write."""
        sink1 = HostMemory(capacity=10)
        sink2 = HostMemory(capacity=50)
        sink1.write(_make_batch(num_graphs=9))
        stage = _CommunicationMixin(overflow_sinks=[sink1, sink2])
        stage._overflow_to_sinks(_make_batch(num_graphs=2))
        assert len(sink1) == 9
        assert len(sink2) == 2


# ---------------------------------------------------------------------------
# Test_CommunicationMixin — Batch Extraction
# ---------------------------------------------------------------------------


class TestBatchExtraction:
    """Test _batch_to_buffer for moving graduated samples into send buffer."""

    @pytest.mark.parametrize("selected", [[1, 3], [0, 1, 2, 3, 4], [2]])
    def test_extract_selected_samples(self, selected: list[int], device: str) -> None:
        """Real buffer copies and removes exactly the selected graphs."""
        batch = _make_batch_with_system(num_graphs=5).to(device)
        batch["system_id"] = torch.arange(5, device=device).unsqueeze(-1)
        stage = _CommunicationMixin(active_batch=batch, device_type=device)
        stage.send_buffer = Batch.empty(
            num_systems=5,
            num_nodes=15,
            num_edges=0,
            template=batch,
            device=device,
        )
        stage._admission_initialized = True
        stage._forces_primed = True
        mask = torch.zeros(5, dtype=torch.bool, device=device)
        mask[selected] = True

        stage._batch_to_buffer(mask, num_requested=len(selected))

        assert (
            stage.send_buffer.system_id[: len(selected)].flatten().tolist() == selected
        )
        remaining = sorted(set(range(5)) - set(selected))
        if remaining:
            assert stage.active_batch.system_id.flatten().tolist() == remaining
            assert stage._pending_graph_indices.tolist() == remaining
        else:
            assert stage.active_batch is None
            assert stage._pending_graph_indices is None
        assert stage._admission_initialized is False
        # Removal keeps retained graphs' forces valid.
        assert stage._forces_primed is True

    @pytest.mark.parametrize("limit", ["atoms", "edges"])
    @pytest.mark.parametrize("selected", [[0, 1, 2], [0, 2], [1]])
    def test_insufficient_capacity_raises_before_trim_or_send(
        self, limit: str, selected: list[int], device: str
    ) -> None:
        """Incomplete copies fail without removing graphs or invalidating state."""
        data = [
            AtomicData(
                positions=torch.full((8, 3), float(i)),
                atomic_numbers=torch.ones(8, dtype=torch.long),
                energy=torch.tensor([[float(i)]]),
                neighbor_list=torch.tensor([[j, (j + 1) % 8] for j in range(8)]),
                shifts=torch.zeros(8, 3),
            )
            for i in range(3)
        ]
        batch = Batch.from_data_list(data, device=device)
        capacity = 10 if len(selected) > 1 else 7
        cfg = BufferConfig(
            num_systems=4,
            num_nodes=capacity if limit == "atoms" else 24,
            num_edges=capacity if limit == "edges" else 24,
        )
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
            buffer_config=cfg,
            device_type=device,
        )
        stage._ensure_buffers(batch)
        stage._admission_initialized = stage._forces_primed = True
        indices = torch.tensor(selected, device=device)
        stage._last_converged = indices
        copied = 1 if len(selected) > 1 else 0

        with patch.object(Batch, "isend") as send:
            with pytest.raises(
                ValueError, match=f"copied {copied} of {len(selected)} requested graphs"
            ) as error:
                stage._poststep_sync_buffers(indices)
            send.assert_not_called()

        assert "Increase BufferConfig capacities" in str(error.value)
        assert f"buffer_config={cfg}" in str(error.value)
        assert stage.active_batch is batch
        assert batch.num_graphs == 3
        assert batch.energy.flatten().tolist() == [0.0, 1.0, 2.0]
        assert stage.send_buffer.num_graphs == copied
        assert stage._admission_initialized and stage._forces_primed
        assert stage._last_converged is indices

    def test_full_buffer_raises_without_removing_source(self) -> None:
        """An occupied buffer must not cause an uncopied graph to be removed."""
        batch = _make_batch_with_system(num_graphs=1)
        stage = _CommunicationMixin(active_batch=batch, device_type="cpu")
        stage.send_buffer = Batch.empty(
            num_systems=2,
            num_nodes=3,
            num_edges=0,
            template=batch,
        )
        mask = torch.ones(1, dtype=torch.bool)
        stage.send_buffer.put(batch, mask)
        stage._admission_initialized = stage._forces_primed = True

        with pytest.raises(ValueError, match="copied 0 of 1 requested graphs"):
            stage._batch_to_buffer(mask)

        assert stage.active_batch is batch
        assert stage.send_buffer.num_graphs == 1
        assert stage._admission_initialized and stage._forces_primed

    def test_extract_no_active_batch_raises(self) -> None:
        """Verify RuntimeError when no active batch exists."""
        stage = _CommunicationMixin()
        stage.send_buffer = Mock()  # send_buffer exists but no active_batch
        with pytest.raises(RuntimeError, match="No active batch"):
            stage._batch_to_buffer(torch.tensor([True, False]))

    def test_extract_no_send_buffer_raises(self) -> None:
        """Verify RuntimeError when no send buffer exists."""
        batch = _make_batch(num_graphs=3)
        stage = _CommunicationMixin(active_batch=batch)
        # send_buffer is None by default
        with pytest.raises(RuntimeError, match="No send buffer"):
            stage._batch_to_buffer(torch.tensor([True, False, False]))


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
        s0 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s1 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s2 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        # setup() accesses local_stage.model, which doesn't exist on _CommunicationMixin
        s0.model = Mock()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1, 2: s2})
        with patch.object(pipeline, "_share_templates"):
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
        s0 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s1 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s0.model = Mock()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1})
        with patch.object(pipeline, "_share_templates"):
            pipeline.setup()

        assert s0.prior_rank is None
        assert s0.next_rank == 1
        assert s1.prior_rank == 0
        assert s1.next_rank is None

    def test_setup_non_contiguous_ranks(self) -> None:
        """Verify setup() handles non-contiguous rank numbers."""
        s0 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s5 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s10 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s0.model = Mock()
        pipeline = DistributedPipeline(stages={0: s0, 5: s5, 10: s10})
        with patch.object(pipeline, "_share_templates"):
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
        stage = _CommunicationMixin(active_batch=batch, next_rank=None)
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
            buffer_config=_DEFAULT_CFG,
        )

        mock_incoming = _make_batch(num_graphs=2)
        mock_handle = Mock()
        mock_handle.wait.return_value = mock_incoming

        with (
            patch.object(Batch, "irecv", return_value=mock_handle) as mock_irecv,
            patch.object(stage, "_recv_to_batch") as mock_r2b,
        ):
            stage._prestep_sync_buffers()

            # irecv was called, handle.wait() was called, _recv_to_batch was called
            mock_irecv.assert_called_once()
            mock_handle.wait.assert_called_once()
            mock_r2b.assert_called_once_with(mock_incoming)

            # _complete_pending_recv is a no-op (handle already consumed)
            mock_r2b.reset_mock()
            mock_handle.wait.reset_mock()
            stage._complete_pending_recv()
            mock_handle.wait.assert_not_called()
            mock_r2b.assert_not_called()

        assert stage._pending_recv_handle is None

    def test_async_recv_mode_defers_wait(self) -> None:
        """Verify async_recv mode defers handle.wait to _complete_pending_recv."""
        batch = _make_batch(num_graphs=3)
        stage = _CommunicationMixin(
            prior_rank=0,
            comm_mode="async_recv",
            active_batch=batch,
            max_batch_size=10,
            buffer_config=_DEFAULT_CFG,
        )

        mock_incoming = _make_batch(num_graphs=2)
        mock_handle = Mock()
        mock_handle.wait.return_value = mock_incoming

        with (
            patch.object(Batch, "irecv", return_value=mock_handle) as mock_irecv,
            patch.object(stage, "_recv_to_batch") as mock_r2b,
        ):
            stage._prestep_sync_buffers()

            # irecv was called but wait and _recv_to_batch were NOT called
            mock_irecv.assert_called_once()
            mock_handle.wait.assert_not_called()
            mock_r2b.assert_not_called()
            assert stage._pending_recv_handle is mock_handle

            # Now complete the deferred recv
            stage._complete_pending_recv()
            mock_handle.wait.assert_called_once()
            mock_r2b.assert_called_once_with(mock_incoming)

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
            buffer_config=_DEFAULT_CFG,
        )

        # Simulate a pending send handle from a previous iteration
        mock_old_send = Mock()
        stage._pending_send_handle = mock_old_send

        mock_recv_handle = Mock()
        mock_recv_handle.wait.return_value = _make_batch(num_graphs=1)

        with (
            patch.object(Batch, "irecv", return_value=mock_recv_handle),
            patch.object(stage, "_recv_to_batch"),
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
        mock_send_buffer = Mock()
        mock_send_buffer.system_capacity = 10  # Plenty of capacity
        mock_send_buffer.num_graphs = 0
        mock_send_buffer.isend.return_value = mock_new_send
        stage.send_buffer = mock_send_buffer

        with patch.object(stage, "_batch_to_buffer"):
            stage._poststep_sync_buffers(converged_indices=torch.tensor([0]))

        mock_send_buffer.isend.assert_called_once_with(dst=2)
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
            patch.object(pipeline, "_share_templates"),
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
            patch.object(pipeline, "_share_templates"),
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
        s0 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s1 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s0.model = Mock()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1})
        with (
            patch.object(pipeline, "_validate_world_size") as mock_validate,
            patch.object(pipeline, "_share_templates"),
        ):
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
        """When nothing converges, clear and send the empty send_buffer."""
        batch = _make_batch(num_graphs=3)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
            buffer_config=_DEFAULT_CFG,
        )

        mock_handle = Mock()
        mock_send_buffer = Mock()
        mock_send_buffer.isend.return_value = mock_handle
        stage.send_buffer = mock_send_buffer

        stage._poststep_sync_buffers(converged_indices=None)

        mock_send_buffer.zero.assert_called_once()
        mock_send_buffer.isend.assert_called_once_with(dst=1)
        # Active batch should be unchanged (no samples graduated)
        assert stage.active_batch_size == 3

    def test_sends_buffer_when_empty_convergence_and_next_rank(self) -> None:
        """When converged_indices is empty tensor and next_rank is set, send send_buffer."""
        batch = _make_batch(num_graphs=3)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
            buffer_config=_DEFAULT_CFG,
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

    def test_poststep_always_sends_buffer(self) -> None:
        """send_buffer.isend is called even when no samples converge."""
        batch = _make_batch(num_graphs=3)
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
            buffer_config=cfg,
        )

        mock_send_buffer = Mock()
        mock_handle = Mock()
        mock_send_buffer.isend.return_value = mock_handle
        stage.send_buffer = mock_send_buffer

        stage._poststep_sync_buffers(converged_indices=None)
        mock_send_buffer.isend.assert_called_once_with(dst=1)
        mock_handle.wait.assert_called_once()

    def test_send_buffer_stores_handle_in_fully_async(self) -> None:
        """In fully_async mode, send buffer's send handle is stored."""
        batch = _make_batch(num_graphs=3)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
            comm_mode="fully_async",
            buffer_config=_DEFAULT_CFG,
        )

        mock_handle = Mock()
        mock_send_buffer = Mock()
        mock_send_buffer.isend.return_value = mock_handle
        stage.send_buffer = mock_send_buffer

        stage._poststep_sync_buffers(converged_indices=None)

        assert stage._pending_send_handle is mock_handle

    def test_fully_async_waits_before_reusing_send_buffer(self) -> None:
        """A pending send completes before its buffer is cleared and reused."""
        stage = _CommunicationMixin(
            active_batch=_make_batch(num_graphs=3),
            next_rank=1,
            comm_mode="fully_async",
            buffer_config=_DEFAULT_CFG,
        )
        prior_handle = Mock()
        next_handle = Mock()
        send_buffer = Mock()
        send_buffer.isend.return_value = next_handle
        stage.send_buffer = send_buffer
        stage._pending_send_handle = prior_handle

        stage._poststep_sync_buffers(converged_indices=None)

        prior_handle.wait.assert_called_once()
        send_buffer.zero.assert_called_once()
        send_buffer.isend.assert_called_once_with(dst=1)
        assert stage._pending_send_handle is next_handle

    def test_convergence_sends_send_buffer(self) -> None:
        """When converged_indices is provided, put into send_buffer and send it."""
        batch = _make_batch(num_graphs=4)
        stage = _CommunicationMixin(
            active_batch=batch,
            next_rank=1,
            buffer_config=_DEFAULT_CFG,
        )

        mock_handle = Mock()

        # Set up a send_buffer that should now be used for sending
        mock_send_buffer = Mock()
        mock_send_buffer.system_capacity = 10  # Plenty of capacity
        mock_send_buffer.num_graphs = 0
        mock_send_buffer.isend.return_value = mock_handle
        stage.send_buffer = mock_send_buffer

        with patch.object(stage, "_batch_to_buffer") as mock_b2b:
            stage._poststep_sync_buffers(converged_indices=torch.tensor([0, 2]))

            # _batch_to_buffer called with a boolean mask
            mock_b2b.assert_called_once()
            call_args = mock_b2b.call_args[0][0]
            assert call_args.dtype == torch.bool
            assert call_args.shape[0] == 4  # num_graphs
            assert call_args[0].item() is True  # index 0
            assert call_args[1].item() is False
            assert call_args[2].item() is True  # index 2
            assert call_args[3].item() is False

            # send_buffer.isend IS now called (new behavior)
            mock_send_buffer.isend.assert_called_once_with(dst=1)


# ---------------------------------------------------------------------------
# TestBufferConfigValidation — Buffer Config Validation in DistributedPipeline
# ---------------------------------------------------------------------------


class TestBufferConfigValidation:
    """Test DistributedPipeline.setup() validates buffer configs between adjacent stages."""

    def test_matching_buffer_configs_pass(self) -> None:
        """setup() succeeds when adjacent stages have matching buffer configs."""
        cfg = BufferConfig(num_systems=10, num_nodes=500, num_edges=2000)
        s0 = _CommunicationMixin(buffer_config=cfg)
        s1 = _CommunicationMixin(buffer_config=cfg)
        s0.model = Mock()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1})
        # Should not raise
        with patch.object(pipeline, "_share_templates"):
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

    def test_one_none_buffer_config_raises(self) -> None:
        """setup() raises when any stage lacks buffer_config."""
        cfg = BufferConfig(num_systems=10, num_nodes=500, num_edges=2000)
        pipeline = DistributedPipeline(
            stages={
                0: _CommunicationMixin(buffer_config=cfg),
                1: _CommunicationMixin(buffer_config=None),
            }
        )
        with pytest.raises(ValueError, match="buffer_config"):
            pipeline.setup()

    def test_both_none_buffer_configs_raises(self) -> None:
        """setup() raises when all stages lack buffer_config."""
        pipeline = DistributedPipeline(
            stages={
                0: _CommunicationMixin(),
                1: _CommunicationMixin(),
            }
        )
        with pytest.raises(ValueError, match="buffer_config"):
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

    def test_buffer_config_required_with_next_rank(self) -> None:
        """ValueError raised when next_rank is set but buffer_config is None."""
        with pytest.raises(ValueError, match="buffer_config"):
            _CommunicationMixin(next_rank=1)

    def test_buffer_config_required_with_prior_rank(self) -> None:
        """ValueError raised when prior_rank is set but buffer_config is None."""
        with pytest.raises(ValueError, match="buffer_config"):
            _CommunicationMixin(prior_rank=0)


# ---------------------------------------------------------------------------
# TestEdgelessBuffers — BufferConfig(num_edges=0) leaves edges untransported
# ---------------------------------------------------------------------------


def _make_batch_with_edges(num_graphs: int = 3, num_atoms: int = 3) -> Batch:
    """Create a batch whose graphs carry a ring ``neighbor_list``."""
    atoms = torch.arange(num_atoms)
    data_list = [
        AtomicData(
            atomic_numbers=torch.randint(1, 20, (num_atoms,)),
            positions=torch.randn(num_atoms, 3),
            neighbor_list=torch.stack([atoms, atoms.roll(-1)], dim=1),
        )
        for _ in range(num_graphs)
    ]
    return Batch.from_data_list(data_list)


def _level_fields(batch: Batch) -> dict[str, list[str]]:
    """Return each materialized level's field names."""
    return {name: sorted(group.keys()) for name, group in batch._storage.groups.items()}


class TestEdgelessBuffers:
    """With num_edges=0, edges never enter buffers or receive templates."""

    _EDGELESS_CFG = BufferConfig(num_systems=10, num_nodes=100, num_edges=0)

    def test_buffers_omit_edges_and_accept_graphs_with_edges(self) -> None:
        """Buffers leave the edges level out, so graphs with edges still fit."""
        batch = _make_batch_with_edges(num_graphs=3)
        stage = _CommunicationMixin(
            prior_rank=0,
            next_rank=2,
            buffer_config=self._EDGELESS_CFG,
            device_type="cpu",
        )

        stage._ensure_buffers(batch)

        for buffer in (stage.send_buffer, stage.recv_buffer):
            assert "edges" not in buffer._storage.groups
        copied = torch.zeros(3, dtype=torch.bool)
        stage.send_buffer.put(
            batch, torch.ones(3, dtype=torch.bool), copied_mask=copied
        )
        assert copied.all()
        assert stage.send_buffer.num_graphs == 3
        assert "edges" in batch._storage.groups

    def test_buffers_keep_edges_when_capacity_is_set(self) -> None:
        batch = _make_batch_with_edges(num_graphs=2)
        stage = _CommunicationMixin(
            prior_rank=0, buffer_config=_DEFAULT_CFG, device_type="cpu"
        )

        stage._ensure_buffers(batch)

        assert "neighbor_list" in stage.recv_buffer._storage.groups["edges"]

    @pytest.mark.parametrize("recv_buffer_exists", [False, True])
    def test_receive_template_matches_sender_buffer(
        self, recv_buffer_exists: bool
    ) -> None:
        """The irecv template has exactly the sender buffer's fields."""
        batch = _make_batch_with_edges(num_graphs=2)
        sender = _CommunicationMixin(
            next_rank=1, buffer_config=self._EDGELESS_CFG, device_type="cpu"
        )
        sender._ensure_buffers(batch)
        receiver = _CommunicationMixin(
            prior_rank=0,
            comm_mode="async_recv",
            buffer_config=self._EDGELESS_CFG,
            device_type="cpu",
        )
        receiver._recv_template = Batch.empty_like(batch)
        if recv_buffer_exists:
            # A stepped receiver carries bookkeeping the sender never sends.
            resident = batch.clone()
            resident["reprime_pending"] = torch.zeros(2, 1, dtype=torch.bool)
            receiver._ensure_buffers(resident)
            assert "reprime_pending" in receiver.recv_buffer

        with patch.object(Batch, "irecv", return_value=Mock()) as mock_irecv:
            receiver._prestep_sync_buffers()

        template = mock_irecv.call_args.kwargs["template"]
        assert template is receiver._recv_template
        assert "edges" not in template._storage.groups
        # Extra receive fields can hang communication waiting for unsent data.
        assert _level_fields(template) == _level_fields(sender.send_buffer)

    def test_buffer_to_batch_keeps_resident_fields_arrivals_lack(self) -> None:
        """Residents keep edges and bookkeeping; arrivals get empty placeholders."""
        residents = _make_batch_with_edges(num_graphs=2)
        residents["reprime_pending"] = torch.ones(2, 1, dtype=torch.bool)
        expected_edges = [residents.get_data(i).neighbor_list for i in range(2)]
        stage = _CommunicationMixin(active_batch=residents, max_batch_size=4)
        sink = HostMemory(capacity=10)
        stage.overflow_sinks = [sink]

        stage._buffer_to_batch(_make_batch(num_graphs=3))

        batch = stage.active_batch
        assert batch.num_graphs == 4
        assert batch.num_edges_per_graph.tolist() == [3, 3, 0, 0]
        for index, edges in enumerate(expected_edges):
            torch.testing.assert_close(batch.get_data(index).neighbor_list, edges)
        assert batch.reprime_pending.flatten().tolist() == [True, True, False, False]
        assert len(sink) == 1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_buffer_to_batch_keeps_active_batch_on_its_device(self) -> None:
        """CPU arrivals, e.g. drained from HostMemory, join a CUDA batch on CUDA."""
        stage = _CommunicationMixin(
            active_batch=_make_batch(num_graphs=2).to("cuda"), max_batch_size=10
        )

        stage._buffer_to_batch(_make_batch(num_graphs=2))

        assert stage.active_batch.num_graphs == 4
        assert stage.active_batch.device.type == "cuda"


# ---------------------------------------------------------------------------
# TestSyncDoneFlags — Distributed Done Flag Synchronization
# ---------------------------------------------------------------------------


class TestSyncDoneFlags:
    """Test DistributedPipeline._sync_done_flags and _done_tensor initialization."""

    def test_setup_initializes_done_tensor(self) -> None:
        """setup() should initialize _done_tensor with correct size."""
        s0 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s1 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s0.model = Mock()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1})
        with patch.object(pipeline, "_share_templates"):
            pipeline.setup()
        assert pipeline._done_tensor is not None
        assert pipeline._done_tensor.shape == (2,)
        assert pipeline._done_tensor.dtype == torch.int32
        assert (pipeline._done_tensor == 0).all()

    def test_setup_initializes_done_tensor_three_stages(self) -> None:
        """setup() should initialize _done_tensor with size matching stage count."""
        s0 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s1 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s2 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s0.model = Mock()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1, 2: s2})
        with patch.object(pipeline, "_share_templates"):
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
        s0 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s1 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s0.model = Mock()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1})
        with patch.object(pipeline, "_share_templates"):
            pipeline.setup()

        # Without dist initialized, global_rank returns 0
        with patch.object(dist, "is_initialized", return_value=False):
            result = pipeline._sync_done_flags()

        assert result is False

    def test_sync_done_flags_local_stage_done(self) -> None:
        """Returns False when only local stage is done (needs all)."""
        s0 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s1 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s0.model = Mock()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1})
        with patch.object(pipeline, "_share_templates"):
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
        s0 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s1 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s0.model = Mock()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1})
        with patch.object(pipeline, "_share_templates"):
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
            patch.object(pipeline, "_share_templates"),
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
            patch.object(stage, "_recv_to_batch"),
        ):
            stage._prestep_sync_buffers()

            # Verify send_buffer.zero() was called
            mock_send_buffer.zero.assert_called_once()

    def test_prestep_does_not_zero_sinks(self) -> None:
        """Verify _prestep_sync_buffers does NOT call sink.zero() directly.

        The old behavior zeroed sinks[0]; the new behavior zeros send_buffer.
        However, the new drain-back behavior will drain sinks into the batch
        if there is room.  This test verifies drain happens vs direct zero.
        """
        batch = _make_batch(num_graphs=3)
        sink = HostMemory(capacity=50)
        cfg = BufferConfig(num_systems=10, num_nodes=100, num_edges=200)

        stage = _CommunicationMixin(
            prior_rank=0,
            comm_mode="sync",
            active_batch=batch,
            max_batch_size=10,
            overflow_sinks=[sink],
            buffer_config=cfg,
        )

        # Create a send_buffer so the old code path would have had something
        mock_send_buffer = Mock()
        stage.send_buffer = mock_send_buffer

        # Write something to the sink so we can verify drain-back behavior
        sink.write(_make_batch(num_graphs=1))
        assert len(sink) == 1

        mock_incoming = _make_batch(num_graphs=2)
        mock_handle = Mock()
        mock_handle.wait.return_value = mock_incoming

        with patch.object(Batch, "irecv", return_value=mock_handle):
            stage._prestep_sync_buffers()

        # Sink should be drained (not just zeroed) - data went into batch
        # Before: 3 graphs, incoming: 2, from sink: 1 -> total 6
        assert len(sink) == 0  # Sink was drained
        assert stage.active_batch_size == 6  # Data moved to batch

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
            patch.object(stage, "_recv_to_batch"),
        ):
            # Should not raise even with send_buffer=None
            stage._prestep_sync_buffers()

    def test_prestep_zeros_recv_buffer(self) -> None:
        """Verify _prestep_sync_buffers zeros recv_buffer when present.

        When a stage has prior_rank and a recv_buffer, _prestep_sync_buffers
        should call recv_buffer.zero() before initiating the receive.
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

        # Create mock buffers
        mock_send_buffer = Mock()
        mock_recv_buffer = Mock()
        stage.send_buffer = mock_send_buffer
        stage.recv_buffer = mock_recv_buffer

        mock_incoming = _make_batch(num_graphs=2)
        mock_handle = Mock()
        mock_handle.wait.return_value = mock_incoming

        with (
            patch.object(Batch, "irecv", return_value=mock_handle),
            patch.object(stage, "_recv_to_batch"),
        ):
            stage._prestep_sync_buffers()

            # Verify both send_buffer.zero() and recv_buffer.zero() were called
            mock_send_buffer.zero.assert_called_once()
            mock_recv_buffer.zero.assert_called_once()


# ---------------------------------------------------------------------------
# TestRecvToBatch — _recv_to_batch staging behavior
# ---------------------------------------------------------------------------


class TestRecvToBatch:
    """Test that _recv_to_batch correctly stages data through recv_buffer."""

    def test_recv_to_batch_uses_recv_buffer(self) -> None:
        """Verify _recv_to_batch copies incoming to recv_buffer, routes, then zeros.

        When recv_buffer is present:
        1. recv_buffer.put(incoming, mask=all_true) is called
        2. _buffer_to_batch(recv_buffer) is called
        3. recv_buffer.zero() is called
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

        # Create a mock recv_buffer
        mock_recv_buffer = Mock()
        stage.recv_buffer = mock_recv_buffer

        incoming = _make_batch(num_graphs=2)

        with patch.object(stage, "_buffer_to_batch") as mock_b2b:
            stage._recv_to_batch(incoming)

            # Verify recv_buffer.put was called with incoming and all-True mask
            mock_recv_buffer.put.assert_called_once()
            call_args = mock_recv_buffer.put.call_args
            assert call_args[0][0] is incoming  # First positional arg is incoming
            mask_arg = call_args[1]["mask"]
            assert mask_arg.dtype == torch.bool
            assert mask_arg.shape == (2,)
            assert mask_arg.all()

            # Verify _buffer_to_batch was called with recv_buffer
            mock_b2b.assert_called_once_with(mock_recv_buffer)

            # Verify recv_buffer.zero() was called after routing
            mock_recv_buffer.zero.assert_called_once()

    def test_recv_to_batch_falls_back_on_empty_incoming(self) -> None:
        """Verify _recv_to_batch routes directly when incoming has 0 graphs.

        When incoming.num_graphs == 0 (sentinel/empty batch), route directly
        via _buffer_to_batch rather than going through recv_buffer.
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

        # Create a mock recv_buffer that should NOT be used
        mock_recv_buffer = Mock()
        stage.recv_buffer = mock_recv_buffer

        # Create a mock empty incoming batch (0 graphs)
        # Note: Batch.from_data_list([]) raises, so we use a Mock
        empty_incoming = Mock()
        empty_incoming.num_graphs = 0

        with patch.object(stage, "_buffer_to_batch") as mock_b2b:
            stage._recv_to_batch(empty_incoming)

            # recv_buffer.put should NOT be called
            mock_recv_buffer.put.assert_not_called()

            # _buffer_to_batch should be called directly with incoming
            mock_b2b.assert_called_once_with(empty_incoming)

            # recv_buffer.zero() should NOT be called
            mock_recv_buffer.zero.assert_not_called()


# ---------------------------------------------------------------------------
# TestPoststepBackPressure — Back-pressure in _poststep_sync_buffers
# ---------------------------------------------------------------------------


class TestPoststepBackPressure:
    """Test that _poststep_sync_buffers respects send buffer capacity (back-pressure)."""

    @pytest.mark.parametrize("comm_mode", ["sync", "async_recv", "fully_async"])
    def test_plain_stage_retries_graduates_without_new_convergence(
        self, comm_mode: str
    ) -> None:
        """Retained graduates leave even when later steps report none."""
        batch = _make_batch_with_system(num_graphs=4)
        batch["status"] = torch.tensor([[0], [2], [3], [0]])
        batch["system_id"] = torch.arange(4).view(-1, 1)
        retained_positions = batch.positions[9:].clone()
        stage = BaseDynamics(
            model=DemoModelWrapper(DemoModel()),
            active_batch=batch,
            exit_status=2,
            prior_rank=None,
            next_rank=1,
            buffer_config=BufferConfig(num_systems=1, num_nodes=3, num_edges=0),
            device_type="cpu",
            comm_mode=comm_mode,
        )
        stage._ensure_buffers(batch)

        with patch.object(Batch, "isend", return_value=Mock()):
            for expected_id, indices in enumerate(
                (torch.tensor([0, 1]), None, torch.empty(0, dtype=torch.long))
            ):
                stage._poststep_sync_buffers(indices)
                assert stage.send_buffer.num_graphs == 1
                assert stage.send_buffer.system_id.view(-1).tolist() == [expected_id]

            stage._poststep_sync_buffers(None)

        assert stage.send_buffer.num_graphs == 0
        assert stage.active_batch.system_id.view(-1).tolist() == [3]
        torch.testing.assert_close(stage.active_batch.positions, retained_positions)

    def test_plain_final_stage_combines_reported_and_retained_graduates(self) -> None:
        """The final stage stores both reported and status-marked graduates once."""
        batch = _make_batch_with_system(num_graphs=3)
        batch["status"] = torch.tensor([[0], [2], [0]])
        batch["system_id"] = torch.arange(3).view(-1, 1)
        sink = HostMemory(capacity=3)
        stage = BaseDynamics(
            model=DemoModelWrapper(DemoModel()),
            active_batch=batch,
            exit_status=2,
            prior_rank=None,
            next_rank=None,
            sinks=[sink],
            device_type="cpu",
        )

        stage._poststep_sync_buffers(torch.tensor([2]))
        stage._poststep_sync_buffers(None)

        assert sink.read().system_id.view(-1).tolist() == [1, 2]
        assert stage.active_batch.system_id.view(-1).tolist() == [0]

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
        mock_handle = Mock()
        mock_send_buffer.isend.return_value = mock_handle
        stage.send_buffer = mock_send_buffer

        with patch.object(stage, "_batch_to_buffer") as mock_b2b:
            stage._poststep_sync_buffers(
                converged_indices=torch.tensor([0, 1, 2, 3, 4])
            )

            # Should only extract first 2 indices (capacity=2) as a boolean mask
            mock_b2b.assert_called_once()
            call_args = mock_b2b.call_args[0][0]
            # New signature: mask is a boolean tensor
            assert call_args.dtype == torch.bool
            assert call_args.shape[0] == 5  # num_graphs in batch
            # Only first 2 should be True (due to capacity truncation)
            assert call_args.sum().item() == 2
            assert call_args[0].item() is True
            assert call_args[1].item() is True
            assert call_args[2].item() is False
            assert call_args[3].item() is False
            assert call_args[4].item() is False

            # send_buffer.isend should now be called (new behavior)
            mock_send_buffer.isend.assert_called_once_with(dst=1)

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
        mock_handle = Mock()
        mock_send_buffer.isend.return_value = mock_handle
        stage.send_buffer = mock_send_buffer

        # Mark indices 0, 1, 2 as converged
        converged_indices = torch.tensor([0, 1, 2])

        # Patch _batch_to_buffer to simulate the new behavior
        def selective_extract(mask: torch.Tensor, *, num_requested: int) -> None:
            """Extract based on boolean mask, simulating defrag."""
            # Due to capacity=1, only index 0 should be True
            assert mask.dtype == torch.bool
            assert num_requested == 1
            assert mask.sum().item() == 1  # Only first one due to capacity
            assert mask[0].item() is True
            # Simulate defrag: remove the graph where mask is True
            remaining = [i for i in range(4) if not mask[i]]
            stage.active_batch = stage.active_batch.index_select(remaining)

        with patch.object(stage, "_batch_to_buffer", side_effect=selective_extract):
            stage._poststep_sync_buffers(converged_indices=converged_indices)

        # Should have 3 graphs left (started with 4, extracted 1)
        assert stage.active_batch_size == 3
        # send_buffer.isend should have been called
        mock_send_buffer.isend.assert_called_once_with(dst=1)

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
        stage._admission_initialized = True
        stage._forces_primed = True

        # Note: final stage does NOT use _batch_to_buffer anymore.
        # It uses inline index_select and _write_to_sinks directly.
        # Even if we set a send_buffer (shouldn't be used), it shouldn't matter
        mock_send_buffer = Mock()
        mock_send_buffer.system_capacity = 1  # Would limit to 1 if used
        mock_send_buffer.num_graphs = 0
        stage.send_buffer = mock_send_buffer

        with patch.object(stage, "_write_to_sinks") as mock_overflow:
            stage._poststep_sync_buffers(converged_indices=torch.tensor([0, 1, 2]))

            # _write_to_sinks should be called with a graduated batch
            mock_overflow.assert_called_once()
            graduated_batch = mock_overflow.call_args[0][0]
            # All 3 converged indices should be in the graduated batch
            assert graduated_batch.num_graphs == 3

        # Active batch should have 2 remaining
        assert stage.active_batch_size == 2
        assert stage._pending_graph_indices.tolist() == [3, 4]
        assert stage._admission_initialized is False
        # Removal keeps retained graphs' forces valid.
        assert stage._forces_primed is True

        # send_buffer.isend should NOT be called (final stage has no next_rank)
        mock_send_buffer.isend.assert_not_called()

    def test_poststep_partial_capacity_extracts_correct_subset(self) -> None:
        """Verify when capacity allows partial extraction, the first N are taken.

        If 4 samples converge but only 2 fit, indices [1, 2] from the converged
        list should have True in the boolean mask (not [4, 5]).
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
        mock_handle = Mock()
        mock_send_buffer.isend.return_value = mock_handle
        stage.send_buffer = mock_send_buffer

        with patch.object(stage, "_batch_to_buffer") as mock_b2b:
            # Converged indices are 1, 2, 4, 5 (out of order intentionally)
            stage._poststep_sync_buffers(converged_indices=torch.tensor([1, 2, 4, 5]))

            # Should only take first 2 from converged list: [1, 2]
            # The mask should be a boolean tensor of size 6 with True at indices 1, 2
            mock_b2b.assert_called_once()
            call_args = mock_b2b.call_args[0][0]
            assert call_args.dtype == torch.bool
            assert call_args.shape[0] == 6  # full batch size
            assert call_args.sum().item() == 2  # only 2 True values
            assert call_args[1].item() is True  # index 1 is True
            assert call_args[2].item() is True  # index 2 is True
            assert call_args[4].item() is False  # index 4 is NOT True (truncated)
            assert call_args[5].item() is False  # index 5 is NOT True (truncated)

            # send_buffer.isend should be called
            mock_send_buffer.isend.assert_called_once_with(dst=1)

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

        mock_handle = Mock()
        mock_send_buffer = Mock()
        mock_send_buffer.system_capacity = 2
        mock_send_buffer.num_graphs = 0
        mock_send_buffer.isend.return_value = mock_handle
        stage.send_buffer = mock_send_buffer

        with patch.object(stage, "_batch_to_buffer"):
            stage._poststep_sync_buffers(converged_indices=torch.tensor([0, 1, 2, 3]))

        # Handle from send_buffer.isend should be stored in fully_async mode
        mock_send_buffer.isend.assert_called_once_with(dst=1)
        assert stage._pending_send_handle is mock_handle


# ---------------------------------------------------------------------------
# Test Overflow Drain-Back
# ---------------------------------------------------------------------------


class TestOverflowDrainBack:
    """Test _drain_sinks_to_batch for pulling overflow back into active batch."""

    def test_drain_pulls_from_sinks_when_room(self) -> None:
        """Verify sink data is pulled into active batch when there's room."""
        # Active batch with 5 graphs, room for 5 more
        batch = _make_batch(num_graphs=5)
        sink = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=batch,
            max_batch_size=10,
            overflow_sinks=[sink],
        )

        # Put 3 graphs into the sink
        overflow_batch = _make_batch(num_graphs=3)
        sink.write(overflow_batch)
        assert len(sink) == 3

        # Drain should pull sink data into active batch
        stage._drain_sinks_to_batch()

        assert stage.active_batch_size == 8  # 5 + 3
        assert len(sink) == 0  # Sink should be empty

    def test_drain_does_not_overfill(self) -> None:
        """Verify drain respects max_batch_size and overflows remainder."""
        # Active batch with 8 graphs, room for 2 more
        batch = _make_batch(num_graphs=8)
        sink = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=batch,
            max_batch_size=10,
            overflow_sinks=[sink],
        )

        # Put 5 graphs into the sink
        overflow_batch = _make_batch(num_graphs=5)
        sink.write(overflow_batch)
        assert len(sink) == 5

        # Drain should take only 2, leaving 3 in overflow
        stage._drain_sinks_to_batch()

        assert stage.active_batch_size == 10  # Full
        assert len(sink) == 3  # 5 - 2 = 3 remaining

    def test_drain_skips_empty_sinks(self) -> None:
        """Verify drain handles empty sinks without error."""
        batch = _make_batch(num_graphs=5)
        sink1 = HostMemory(capacity=50)
        sink2 = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=batch,
            max_batch_size=10,
            overflow_sinks=[sink1, sink2],
        )

        # Both sinks empty
        assert len(sink1) == 0
        assert len(sink2) == 0

        # Should complete without error, batch unchanged
        stage._drain_sinks_to_batch()

        assert stage.active_batch_size == 5

    def test_drain_priority_order(self) -> None:
        """Verify sinks are drained in priority order (first to last)."""
        batch = _make_batch(num_graphs=3)
        sink1 = HostMemory(capacity=50)
        sink2 = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=batch,
            max_batch_size=10,
            overflow_sinks=[sink1, sink2],
        )

        # Put data in both sinks
        sink1.write(_make_batch(num_graphs=2))
        sink2.write(_make_batch(num_graphs=3))
        assert len(sink1) == 2
        assert len(sink2) == 3

        # Drain - should take from sink1 first, then sink2
        stage._drain_sinks_to_batch()

        # Active batch should have all 8 (3 + 2 + 3)
        assert stage.active_batch_size == 8
        assert len(sink1) == 0  # Drained first
        assert len(sink2) == 0  # Then drained second

    def test_drain_stops_when_batch_full(self) -> None:
        """Verify drain stops early when batch becomes full."""
        batch = _make_batch(num_graphs=8)
        sink1 = HostMemory(capacity=50)
        sink2 = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=batch,
            max_batch_size=10,
            overflow_sinks=[sink1, sink2],
        )

        # Put data in both sinks
        sink1.write(_make_batch(num_graphs=3))
        sink2.write(_make_batch(num_graphs=3))

        # Drain - room for 2, should take 2 from sink1
        stage._drain_sinks_to_batch()

        assert stage.active_batch_size == 10  # Full
        assert len(sink1) == 1  # 3 - 2 = 1 remaining (overflowed back)
        # sink2 should not be touched since batch is full
        assert len(sink2) == 3

    def test_drain_with_no_active_batch(self) -> None:
        """Verify drain works when there's no active batch."""
        sink = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=None,
            max_batch_size=10,
            overflow_sinks=[sink],
        )

        # Put data in sink
        overflow_batch = _make_batch(num_graphs=3)
        sink.write(overflow_batch)

        # Drain should create active batch from sink
        stage._drain_sinks_to_batch()

        assert stage.active_batch is not None
        assert stage.active_batch_size == 3
        assert len(sink) == 0

    def test_prestep_drains_after_recv_sync_mode(self) -> None:
        """Verify _prestep_sync_buffers drains sinks in sync mode."""
        batch = _make_batch(num_graphs=5)
        sink = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=batch,
            max_batch_size=10,
            overflow_sinks=[sink],
            prior_rank=0,
            comm_mode="sync",
            buffer_config=_DEFAULT_CFG,
        )

        stage._admission_initialized = True
        # Put data in sink
        sink.write(_make_batch(num_graphs=2))
        assert len(sink) == 2

        # Mock the irecv process
        mock_handle = Mock()
        mock_incoming = _make_batch(num_graphs=1)
        mock_handle.wait.return_value = mock_incoming

        with patch.object(Batch, "irecv", return_value=mock_handle):
            stage._prestep_sync_buffers()

        # Should have drained: 5 original + 1 from recv + 2 from sink = 8
        assert stage.active_batch_size == 8
        assert stage._pending_graph_indices.tolist() == [0, 1, 2, 3, 4, -1, -1, -1]
        assert len(sink) == 0

    def test_prestep_drains_without_prior_rank(self) -> None:
        """Verify _prestep_sync_buffers drains sinks when no prior_rank."""
        batch = _make_batch(num_graphs=5)
        sink = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=batch,
            max_batch_size=10,
            overflow_sinks=[sink],
            prior_rank=None,  # First stage
            comm_mode="sync",
        )

        # Put data in sink
        sink.write(_make_batch(num_graphs=3))
        assert len(sink) == 3

        # Prestep should still drain (no irecv needed)
        stage._prestep_sync_buffers()

        assert stage.active_batch_size == 8
        assert len(sink) == 0

    def test_complete_pending_recv_drains_after_recv(self) -> None:
        """Verify _complete_pending_recv drains sinks in async modes."""
        batch = _make_batch(num_graphs=5)
        sink = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=batch,
            max_batch_size=10,
            overflow_sinks=[sink],
            prior_rank=0,
            comm_mode="async_recv",
            buffer_config=_DEFAULT_CFG,
        )

        # Put data in sink
        sink.write(_make_batch(num_graphs=2))
        assert len(sink) == 2

        # Mock the pending recv handle
        mock_handle = Mock()
        mock_incoming = _make_batch(num_graphs=1)
        mock_handle.wait.return_value = mock_incoming
        stage._pending_recv_handle = mock_handle

        # Complete recv should also drain sinks
        stage._complete_pending_recv()

        # Should have: 5 original + 1 from recv + 2 from sink = 8
        assert stage.active_batch_size == 8
        assert len(sink) == 0
        assert stage._pending_recv_handle is None

    def test_complete_pending_recv_drains_without_pending(self) -> None:
        """Verify _complete_pending_recv drains sinks even without pending recv."""
        batch = _make_batch(num_graphs=5)
        sink = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=batch,
            max_batch_size=10,
            overflow_sinks=[sink],
            comm_mode="async_recv",
        )

        # Put data in sink
        sink.write(_make_batch(num_graphs=2))

        # No pending recv handle
        stage._pending_recv_handle = None

        # Should still drain sinks
        stage._complete_pending_recv()

        assert stage.active_batch_size == 7
        assert len(sink) == 0


# ---------------------------------------------------------------------------
# Test Overflow Storage Kept Separate from Result Sinks
# ---------------------------------------------------------------------------


class TestOverflowTransfer:
    """CPU overflow is selected before arrivals move to the target device."""

    @pytest.mark.parametrize("has_active_batch", [False, True])
    def test_only_admitted_graphs_move_to_device(
        self, device: str, has_active_batch: bool
    ) -> None:
        """Partial admission and repeated refill never transfer the waiting tail."""
        sizes = [2, 5, 3, 4, 6]
        pending = Batch.from_data_list([_make_atomic_data(n) for n in sizes])
        pending.sample_id = torch.arange(5).view(-1, 1)
        sink = HostMemory(capacity=10)
        sink.write(pending)
        resident = None
        if has_active_batch:
            resident = _make_batch(1, num_atoms=2).to(device)
            resident.sample_id = torch.tensor([[99]], device=device)
        stage = _CommunicationMixin(
            active_batch=resident,
            max_batch_size=3,
            overflow_sinks=[sink],
            device_type=device,
        )
        transfers = []
        original_to = Batch.to

        def record_transfer(batch, target, *args, **kwargs):
            if batch.device.type == "cpu" and torch.device(target).type == "cuda":
                transfers.append(batch.sample_id.flatten().tolist())
            return original_to(batch, target, *args, **kwargs)

        admitted = 2 if has_active_batch else 3
        selected = list(range(admitted))
        waiting = list(range(admitted, 5))
        with patch.object(Batch, "to", autospec=True, side_effect=record_transfer):
            stage._drain_sinks_to_batch()

            assert stage.active_batch.device.type == device
            assert stage.active_batch.sample_id.flatten().tolist() == (
                ([99] if has_active_batch else []) + selected
            )
            assert stage.active_batch.num_nodes_list == (
                ([2] if has_active_batch else []) + sizes[:admitted]
            )
            assert len(sink) == len(waiting)
            assert sink.read().device.type == "cpu"
            assert sink.read().sample_id.flatten().tolist() == waiting

            stage.active_batch = None
            stage._drain_sinks_to_batch()

        assert stage.active_batch.device.type == device
        assert stage.active_batch.sample_id.flatten().tolist() == waiting
        assert len(sink) == 0
        assert transfers == ([selected, waiting] if device == "cuda" else [])


class TestOverflowSinksSeparation:
    """Pending arrivals use ``overflow_sinks``; finished results use ``sinks``."""

    def test_drain_reads_only_overflow_sinks(self) -> None:
        """Draining refills the batch from pending storage, never results."""
        results = HostMemory(capacity=50)
        pending = HostMemory(capacity=50)
        results.write(_make_batch(num_graphs=2))
        pending.write(_make_batch(num_graphs=2))
        stage = _CommunicationMixin(
            active_batch=_make_batch(num_graphs=1),
            max_batch_size=10,
            sinks=[results],
            overflow_sinks=[pending],
        )
        stage._drain_sinks_to_batch()

        assert stage.active_batch_size == 3
        assert len(pending) == 0
        assert len(results) == 2

    def test_default_overflow_drains_back_to_stage_device(self, device) -> None:
        """Host-side default overflow returns to the active batch's device."""
        results = HostMemory(capacity=50)
        stage = _CommunicationMixin(
            active_batch=_make_batch(num_graphs=2).to(device),
            max_batch_size=2,
            device_type=device,
            sinks=[results],
        )
        stage._buffer_to_batch(_make_batch(num_graphs=2).to(device))
        assert len(results) == 0
        assert sum(len(sink) for sink in stage.overflow_sinks) == 2
        assert all(sink not in stage.sinks for sink in stage.overflow_sinks)

        # Every resident graph was sent downstream.
        stage.active_batch = None
        stage._drain_sinks_to_batch()

        assert stage.active_batch_size == 2
        assert stage.active_batch.positions.device.type == device
        assert sum(len(sink) for sink in stage.overflow_sinks) == 0
        assert len(results) == 0

    def test_result_write_skips_sink_without_room_for_whole_write(self) -> None:
        """Result writes also choose a sink that can hold every result."""
        nearly_full = HostMemory(capacity=10)
        nearly_full.write(_make_batch(num_graphs=9))
        spare = HostMemory(capacity=10)
        stage = _CommunicationMixin(
            active_batch=_make_batch(num_graphs=3),
            next_rank=None,
            sinks=[nearly_full, spare],
        )
        stage._remove_converged_final_stage(torch.tensor([0, 1]))

        assert len(nearly_full) == 9
        assert len(spare) == 2

    def test_downstream_not_done_while_overflow_pending(self) -> None:
        """A stage with pending overflow keeps running after upstream ends."""
        pending = HostMemory(capacity=50)
        pending.write(_make_batch(num_graphs=2))
        s0 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s1 = _CommunicationMixin(
            buffer_config=_DEFAULT_CFG,
            overflow_sinks=[pending],
        )
        s0.model = Mock()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1})
        with patch.object(pipeline, "_share_templates"):
            pipeline.setup()
        pipeline._done_tensor[0] = 1

        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_rank", return_value=1),
            patch.object(s1, "_prestep_sync_buffers"),
            patch.object(s1, "_complete_pending_recv"),
            patch.object(s1, "_poststep_sync_buffers"),
        ):
            pipeline.step()

        assert s1.active_batch is None
        assert not s1.done

    def test_downstream_done_with_results_stored(self) -> None:
        """Stored results do not keep a finished downstream stage running."""
        results = HostMemory(capacity=50)
        results.write(_make_batch(num_graphs=2))
        s0 = _CommunicationMixin(buffer_config=_DEFAULT_CFG)
        s1 = _CommunicationMixin(buffer_config=_DEFAULT_CFG, sinks=[results])
        s0.model = Mock()
        pipeline = DistributedPipeline(stages={0: s0, 1: s1})
        with patch.object(pipeline, "_share_templates"):
            pipeline.setup()
        pipeline._done_tensor[0] = 1

        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_rank", return_value=1),
            patch.object(s1, "_prestep_sync_buffers"),
            patch.object(s1, "_complete_pending_recv"),
            patch.object(s1, "_poststep_sync_buffers"),
        ):
            pipeline.step()

        assert s1.done
        assert len(results) == 2
