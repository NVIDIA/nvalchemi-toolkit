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
"""Tests for the rank-to-rank hand-off between ``FusedStage`` pipeline stages.

Systems that graduate on one rank carry that rank's ``exit_status``; the
receiving ``FusedStage`` must admit them at its own ``entry_status``, and a
failing rank must not hang the pipeline in teardown.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    FIRE2,
    ConvergenceHook,
    DistributedPipeline,
    FusedStage,
    HostMemory,
    NVTLangevin,
    SizeAwareSampler,
)
from nvalchemi.dynamics.base import BufferConfig
from nvalchemi.dynamics.hooks import SnapshotHook
from nvalchemi.models.demo import DemoModel, DemoModelWrapper
from test.distributed._gloo_harness import run_gloo

_N_SYSTEMS = 4
_LANGEVIN_STEPS = 20


def _make_system(seed: int) -> AtomicData:
    """Build a small rattled molecule with the fields dynamics needs."""
    gen = torch.Generator().manual_seed(seed)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
    ) + 0.15 * torch.randn(3, 3, generator=gen)
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.tensor([8, 1, 1]),
        forces=torch.zeros(3, 3),
        energy=torch.zeros(1, 1),
    )
    data.add_node_property("velocities", torch.zeros(3, 3))
    return data


def _langevin_stage(**kwargs) -> FusedStage:
    dynamics = NVTLangevin(
        model=DemoModelWrapper(DemoModel()),
        dt=0.5,
        temperature=300.0,
        friction=0.01,
        n_steps=_LANGEVIN_STEPS,
        device_type="cpu",
        **kwargs.pop("dynamics_kwargs", {}),
    )
    return FusedStage(sub_stages=[(0, dynamics)], device_type="cpu", **kwargs)


class TestFusedStageOnReceive:
    """``FusedStage._on_receive`` resets sender bookkeeping on arrival."""

    def test_status_and_counters_reset_to_entry_state(self) -> None:
        stage = _langevin_stage(entry_status=0)
        incoming = Batch.from_data_list([_make_system(i) for i in range(3)])
        incoming["status"] = torch.full((3, 1), stage.exit_status, dtype=torch.long)
        incoming["n_steps_counter_0"] = torch.full((3, 1), 7, dtype=torch.long)

        stage._on_receive(incoming)

        assert (incoming.status == stage.entry_status).all()
        assert (incoming.n_steps_counter_0 == 0).all()

    def test_nonzero_entry_status_is_applied(self) -> None:
        stage = _langevin_stage(entry_status=3)
        incoming = Batch.from_data_list([_make_system(0)])
        incoming["status"] = torch.zeros(1, 1, dtype=torch.long)

        stage._on_receive(incoming)

        assert (incoming.status == 3).all()

    def test_batch_without_bookkeeping_is_left_alone(self) -> None:
        stage = _langevin_stage()
        incoming = Batch.from_data_list([_make_system(0)])

        stage._on_receive(incoming)

        assert getattr(incoming, "status", None) is None

    def test_recv_to_batch_admits_systems_as_active(self) -> None:
        stage = _langevin_stage()
        incoming = Batch.from_data_list([_make_system(i) for i in range(2)])
        incoming["status"] = torch.full((2, 1), stage.exit_status, dtype=torch.long)

        stage._recv_to_batch(incoming)

        mask = stage.active_graph_mask(stage.active_batch, stage.exit_status)
        assert mask.all()

    @pytest.mark.parametrize("n_graphs", [1, 2, 3, 4])
    def test_recv_buffer_initializes_missing_bookkeeping(self, n_graphs: int) -> None:
        """Fused fields fit the buffer allocation even on partial receives."""
        stage = _langevin_stage(
            prior_rank=0,
            next_rank=None,
            buffer_config=BufferConfig(num_systems=4, num_nodes=12, num_edges=0),
        )
        incoming = Batch.from_data_list([_make_system(i) for i in range(n_graphs)])
        incoming["status"] = torch.full(
            (n_graphs, 1), stage.exit_status, dtype=torch.long
        )
        incoming["system_id"] = torch.arange(n_graphs).view(-1, 1)
        stage._ensure_buffers(Batch.empty_like(incoming))

        stage._recv_to_batch(incoming)

        batch = stage.active_batch
        assert batch.num_graphs == n_graphs
        assert (batch.status == stage.entry_status).all()
        assert not batch.reprime_pending.any()
        assert (batch.n_steps_counter_0 == 0).all()
        torch.testing.assert_close(batch.system_id, incoming.system_id)
        assert stage.recv_buffer.num_graphs == 0
        assert stage.recv_buffer.reprime_pending.shape == (4, 1)
        assert stage.recv_buffer.n_steps_counter_0.shape == (4, 1)


class TestPipelineExit:
    """``DistributedPipeline.__exit__`` must not block while unwinding an error."""

    @staticmethod
    def _pipeline() -> DistributedPipeline:
        pipeline = DistributedPipeline.__new__(DistributedPipeline)
        pipeline._dist_initialized = True
        return pipeline

    def test_exception_skips_process_group_teardown(self) -> None:
        pipeline = self._pipeline()
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.destroy_process_group") as destroy,
        ):
            result = pipeline.__exit__(RuntimeError, RuntimeError("boom"), None)

        destroy.assert_not_called()
        assert not result  # the exception must propagate

    def test_clean_exit_destroys_process_group(self) -> None:
        pipeline = self._pipeline()
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.destroy_process_group") as destroy,
        ):
            pipeline.__exit__(None, None, None)

        destroy.assert_called_once()


def _fused_pipeline_worker(rank: int, world_size: int, queue) -> None:
    """Run FIRE2 -> Langevin across two gloo ranks and report the sink size."""
    sink = HostMemory(capacity=100)
    dataset = [_make_system(i) for i in range(_N_SYSTEMS)]

    class _Dataset:
        def __len__(self) -> int:
            return len(dataset)

        def __getitem__(self, idx: int):
            d = dataset[idx]
            return d, {"num_atoms": d.num_nodes, "num_edges": d.num_edges}

        def get_metadata(self, idx: int) -> tuple[int, int]:
            d = dataset[idx]
            return d.num_nodes, d.num_edges

    # Every receive is partial, including when all systems graduate together.
    buffer_cfg = BufferConfig(num_systems=_N_SYSTEMS + 1, num_nodes=50, num_edges=0)
    optimizer = FIRE2(
        model=DemoModelWrapper(DemoModel()),
        dt=0.05,
        n_steps=50,
        device_type="cpu",
        convergence_hook=ConvergenceHook(
            criteria=[
                {
                    "key": "forces",
                    "threshold": 0.05,
                    "reduce_op": "norm",
                    "reduce_dims": -1,
                }
            ],
        ),
    )
    stages = {
        0: FusedStage(
            sub_stages=[(0, optimizer)],
            device_type="cpu",
            sampler=SizeAwareSampler(
                dataset=_Dataset(),
                max_atoms=50,
                max_edges=0,
                max_batch_size=_N_SYSTEMS,
            ),
            refill_frequency=1,
            prior_rank=None,
            next_rank=1,
            buffer_config=buffer_cfg,
        ),
        1: _langevin_stage(
            dynamics_kwargs={"hooks": [SnapshotHook(sink=sink, frequency=1)]},
            prior_rank=0,
            next_rank=None,
            buffer_config=buffer_cfg,
        ),
    }

    with DistributedPipeline(stages=stages, backend="gloo") as pipeline:
        pipeline.run()
    queue.put((rank, len(sink)))


@pytest.mark.slow
def test_fused_stage_pipeline_graduates_and_terminates() -> None:
    """Downstream steps each system ``n_steps`` times, then the run ends."""
    results = dict(run_gloo(world_size=2, fn=_fused_pipeline_worker, timeout_sec=120.0))

    assert results[1] == _N_SYSTEMS * _LANGEVIN_STEPS
