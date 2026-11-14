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
    NVE,
    ConvergenceHook,
    DistributedPipeline,
    DynamicsStage,
    FusedStage,
    HostMemory,
    NVTLangevin,
    SizeAwareSampler,
)
from nvalchemi.dynamics.base import BufferConfig
from nvalchemi.dynamics.hooks import SnapshotHook
from nvalchemi.hooks import DynamicsContext
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
        expected_keys = {
            level: set(keys) for level, keys in incoming.level_keys.items()
        }

        stage._recv_to_batch(incoming)

        mask = stage.active_graph_mask(stage.active_batch, stage.exit_status)
        assert mask.all()
        assert stage._recv_template.level_keys == expected_keys

    @pytest.mark.parametrize("n_graphs", [1, 2, 3, 4])
    def test_receive_initializes_missing_bookkeeping(self, n_graphs: int) -> None:
        """Received samples get fresh bookkeeping, including partial batches."""
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
        assert batch.reprime_pending.shape == (n_graphs, 1)
        assert batch.n_steps_counter_0.shape == (n_graphs, 1)


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


class _SystemDataset:
    """``_make_system`` molecules in the form ``SizeAwareSampler`` reads."""

    def __init__(self, num_systems: int) -> None:
        self.systems = [_make_system(i) for i in range(num_systems)]

    def __len__(self) -> int:
        return len(self.systems)

    def __getitem__(self, idx: int) -> tuple[AtomicData, dict[str, int]]:
        data = self.systems[idx]
        return data, {"num_atoms": data.num_nodes, "num_edges": data.num_edges}

    def get_metadata(self, idx: int) -> tuple[int, int]:
        data = self.systems[idx]
        return data.num_nodes, data.num_edges


def _plain_receiver_worker(rank: int, world_size: int, queue) -> None:
    """Relax on a fused stage, then on a plain one; report what moves on rank 1."""
    torch.manual_seed(0)
    model = DemoModelWrapper(DemoModel())
    buffer_cfg = BufferConfig(num_systems=_N_SYSTEMS, num_nodes=50, num_edges=0)
    stages = {
        0: FusedStage(
            sub_stages=[(0, FIRE2(model=model, dt=0.05, n_steps=3, device_type="cpu"))],
            device_type="cpu",
            sampler=SizeAwareSampler(
                dataset=_SystemDataset(_N_SYSTEMS),
                max_atoms=50,
                max_edges=0,
                max_batch_size=_N_SYSTEMS,
            ),
            prior_rank=None,
            next_rank=1,
            buffer_config=buffer_cfg,
        ),
        # A tight fmax keeps every arrival relaxing on the plain stage.
        1: FIRE2(
            model=model,
            dt=0.05,
            convergence_hook=ConvergenceHook.from_fmax(1e-7),
            device_type="cpu",
            prior_rank=0,
            next_rank=None,
            buffer_config=buffer_cfg,
        ),
    }
    first_positions: dict[int, torch.Tensor] = {}
    moved: set[int] = set()
    with DistributedPipeline(stages=stages, backend="gloo") as pipeline:
        stage = pipeline.local_stage
        for _ in range(20):
            pipeline.step()
            batch = stage.active_batch
            if rank == 0 or batch is None:
                continue
            positions = batch.positions.detach().split(
                batch.num_nodes_per_graph.tolist()
            )
            for system_id, pos in zip(batch.system_id.view(-1).tolist(), positions):
                first = first_positions.setdefault(system_id, pos.clone())
                if not torch.equal(first, pos):
                    moved.add(system_id)
    if rank == 1:
        queue.put((sorted(first_positions), sorted(moved)))


def test_plain_stage_steps_systems_sent_by_fused_stage() -> None:
    """Systems that leave a fused stage at its exit status move on a plain stage."""
    [(seen, moved)] = run_gloo(
        world_size=2, fn=_plain_receiver_worker, timeout_sec=90.0
    )

    assert seen == list(range(_N_SYSTEMS))
    assert moved == seen


class _GraduationRecorder:
    """Record the ``system_id`` of every graph that graduates."""

    stage = DynamicsStage.ON_GRADUATE
    frequency = 1

    def __init__(self) -> None:
        self.system_ids: list[int] = []

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        """Append the identifiers of this step's graduates."""
        system_id = ctx.batch.system_id.view(-1)[: ctx.batch.num_graphs]
        self.system_ids.extend(system_id[ctx.graduated_mask].tolist())


def _bursts(every: int) -> ConvergenceHook:
    """Converge every system at once on each ``every``-th evaluation."""
    calls = 0

    def converge_in_bursts(energy: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return torch.full((energy.shape[0],), calls % every == 0)

    return ConvergenceHook(
        criteria=[{"key": "energy", "threshold": 0.0, "custom_op": converge_in_bursts}]
    )


def _atom_capacity_worker(rank: int, world_size: int, queue) -> None:
    """Run three stages whose sends carry one three-atom system at a time."""
    torch.manual_seed(0)
    model = DemoModelWrapper(DemoModel())
    # Room for two systems but only four atoms, so sends copy part of a request.
    config = BufferConfig(num_systems=2, num_nodes=4, num_edges=0)
    recorder = _GraduationRecorder()
    relax = FIRE2(model=model, dt=0.05, device_type="cpu", convergence_hook=_bursts(3))
    stages = {
        0: FusedStage(
            sub_stages=[(0, FIRE2(model=model, dt=0.05, n_steps=2, device_type="cpu"))],
            device_type="cpu",
            sampler=SizeAwareSampler(
                dataset=_SystemDataset(8), max_atoms=12, max_edges=0, max_batch_size=4
            ),
            prior_rank=None,
            next_rank=1,
            buffer_config=config,
        ),
        1: FusedStage(
            sub_stages=[(0, relax)],
            device_type="cpu",
            prior_rank=0,
            next_rank=2,
            buffer_config=config,
        ),
        2: FusedStage(
            sub_stages=[(0, NVE(model=model, dt=0.5, n_steps=2, device_type="cpu"))],
            device_type="cpu",
            hooks=[recorder],
            prior_rank=1,
            next_rank=None,
            buffer_config=config,
        ),
    }
    finished = False
    with DistributedPipeline(stages=stages, backend="gloo") as pipeline:
        for _ in range(200):
            pipeline.step()
            if pipeline._sync_done_flags():
                finished = True
                break
    if rank == 2:
        queue.put((finished, recorder.system_ids))


def test_partial_sends_deliver_every_system_once() -> None:
    """Graduates left by atom-limited sends leave later; the run then ends."""
    [(finished, graduated)] = run_gloo(
        world_size=3, fn=_atom_capacity_worker, timeout_sec=90.0
    )

    assert finished
    assert sorted(graduated) == list(range(8))
