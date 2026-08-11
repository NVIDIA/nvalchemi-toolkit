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
"""Regression tests for convergence handling in ``DomainParallel``."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from _gloo_harness import run_gloo
from torch.distributed import DeviceMesh

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed._core.gather_primitives import mesh_group
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.domain_parallel import DomainParallel
from nvalchemi.dynamics.base import (
    ConvergenceHook,
    DistributedPipeline,
    DynamicsStage,
)
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.models.demo import DemoModel, DemoModelWrapper
from nvalchemi.models.lj import LennardJonesModelWrapper


class _NoOpThermo:
    """Minimal coordinator used to isolate the convergence tail of ``step``."""

    def globalize_dof(self, batch: Batch) -> None:
        pass

    @contextmanager
    def reduce_scope(self) -> Iterator[None]:
        yield

    def broadcast_state(self, batch: Batch) -> None:
        pass


def _stubbed_distributed_step(
    mesh: Any,
    *,
    force_value: float,
    check_convergence: bool = True,
    frequency: int = 1,
    process_group: Any = None,
) -> tuple[DomainParallel, Batch]:
    """Build a DD step whose only live behavior is convergence evaluation."""
    model = DemoModelWrapper(DemoModel())
    inner = DemoDynamics(
        model=model,
        n_steps=10,
        convergence_hook=(
            ConvergenceHook.from_fmax(0.5, frequency=frequency)
            if check_convergence
            else None
        ),
    )
    dp = DomainParallel(
        dynamics=inner,
        config=DomainConfig(cutoff=3.0, skin=0.5, mesh=mesh),
    )

    data = AtomicData(
        atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
        positions=torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
    )
    batch = Batch.from_data_list([data])
    batch.forces = torch.full((2, 3), force_value)

    # Exercise DomainParallel.step's real convergence tail without requiring
    # halo exchange, a model forward, or an integrator update.
    dp._dist_model = object()
    dp._strategy = SimpleNamespace(
        process_group=(process_group if process_group is not None else mesh_group(mesh))
    )
    dp._forces_primed = True
    dp._thermo = _NoOpThermo()
    dp._resolve_pending_migrate = MagicMock(side_effect=lambda current: current)
    dp._call_hooks = MagicMock()
    dp._wrap_owned_positions = MagicMock()
    dp._distributed_compute = MagicMock()
    dp._dispatch_async_migrate_check = MagicMock()
    inner._ensure_state_initialized = MagicMock()
    inner._call_hooks = MagicMock()
    inner.pre_update = MagicMock()
    inner.post_update = MagicMock()

    return dp, batch


def _serialize_convergence(value: Any) -> tuple[str, Any]:
    if isinstance(value, torch.Tensor):
        return "tensor", value.tolist()
    return type(value).__name__, value


def _single_process_domain_parallel() -> DomainParallel:
    model = DemoModelWrapper(DemoModel())
    inner = DemoDynamics(model=model, n_steps=10)
    return DomainParallel(
        dynamics=inner,
        config=DomainConfig(cutoff=3.0, skin=0.5),
    )


def _single_graph_batch() -> Batch:
    data = AtomicData(
        atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
        positions=torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
    )
    return Batch.from_data_list([data])


def _partitioned_convergence_batch() -> Batch:
    """Five atoms with different rank-local force maxima after an x split."""
    dtype = torch.float64
    positions = torch.tensor(
        [
            [2.0, 5.0, 5.0],
            [9.0, 5.0, 5.0],
            [11.0, 5.0, 5.0],
            [13.0, 5.0, 5.0],
            [14.0, 5.0, 5.0],
        ],
        dtype=dtype,
    )
    data = AtomicData(
        atomic_numbers=torch.arange(1, 6, dtype=torch.long),
        positions=positions,
        atomic_masses=torch.ones(5, dtype=dtype),
        forces=torch.zeros_like(positions),
        energy=torch.zeros(1, 1, dtype=dtype),
        cell=torch.diag(torch.tensor([20.0, 10.0, 10.0], dtype=dtype)).unsqueeze(0),
        pbc=torch.zeros(1, 3, dtype=torch.bool),
    )
    data.add_node_property("velocities", torch.zeros_like(positions))
    return Batch.from_data_list([data])


def _all_ranks_converged_worker(
    rank: int,
    world_size: int,
    queue: Any,
) -> None:
    mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("domain",))
    dp, batch = _stubbed_distributed_step(mesh, force_value=0.0)

    _, converged = dp.step(batch)
    queue.put(
        (
            rank,
            _serialize_convergence(converged),
            _serialize_convergence(dp._dynamics._last_converged),
        )
    )


def _ranks_disagree_worker(
    rank: int,
    world_size: int,
    queue: Any,
) -> None:
    mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("domain",))
    force_value = 0.0 if rank == 0 else 1.0
    dp, batch = _stubbed_distributed_step(mesh, force_value=force_value)

    _, converged = dp.step(batch)
    dist.barrier()
    queue.put((rank, _serialize_convergence(converged)))


def _partitioned_run_worker(
    rank: int,
    world_size: int,
    queue: Any,
) -> None:
    mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("domain",))
    convergence = ConvergenceHook.from_fmax(1.0)
    model = LennardJonesModelWrapper(
        epsilon=1.0,
        sigma=1.0,
        cutoff=2.5,
    )
    dynamics = DemoDynamics(
        model=model,
        n_steps=3,
        dt=0.0,
        hooks=[
            NeighborListHook(
                config=model.model_config.neighbor_config,
                skin=0.0,
                stage=DynamicsStage.BEFORE_COMPUTE,
            )
        ],
        convergence_hook=convergence,
    )
    dp = DomainParallel(
        dynamics=dynamics,
        config=DomainConfig(
            cutoff=2.5,
            skin=0.0,
            mesh=mesh,
            mesh_dim="domain",
            grid_dims=(2, 1, 1),
        ),
    )

    # This test targets DD control flow, not NeighborListHook compilation.
    # Running its eager body also avoids concurrent Inductor cache writes from
    # the two spawned CPU workers while preserving the real neighbor-list work.
    eager_neighbor_call = NeighborListHook.__call__.__wrapped__
    with patch.object(NeighborListHook, "__call__", eager_neighbor_call), dp:
        local = dp.partition(_partitioned_convergence_batch() if rank == 0 else None)
        owned = local.atomic_numbers.tolist()

        # The first real DD step has one locally-converged shard and one
        # non-converged shard, so the global result must be non-converged.
        local, global_low = dp.step(local)
        local_low = convergence.evaluate(local)
        full_low = dp.gather(local, dst=0)
        full_low_result = (
            convergence.evaluate(full_low) if full_low is not None else None
        )

        # With dt=0 the positions are unchanged. Raising only the threshold makes
        # the same full system converge, and run() must stop after its first step.
        convergence.criteria[0].threshold = 100.0
        before_run = dp.step_count
        local = dp.run(local, n_steps=3)
        run_steps = dp.step_count - before_run
        global_high = dynamics._last_converged
        full_high = dp.gather(local, dst=0)
        full_high_result = (
            convergence.evaluate(full_high) if full_high is not None else None
        )
        same_positions = (
            torch.equal(full_low.positions, full_high.positions)
            if full_low is not None and full_high is not None
            else None
        )

    queue.put(
        (
            rank,
            owned,
            _serialize_convergence(local_low),
            _serialize_convergence(global_low),
            _serialize_convergence(full_low_result),
            run_steps,
            _serialize_convergence(global_high),
            _serialize_convergence(full_high_result),
            same_positions,
        )
    )


def _grouped_pipeline_convergence_worker(
    rank: int,
    world_size: int,
    queue: Any,
) -> None:
    from torch.distributed import init_device_mesh

    assert world_size == 4
    mesh = init_device_mesh(
        "cpu",
        (2, 2),
        mesh_dim_names=("pipeline", "domain"),
    )
    domain = mesh["domain"]
    pipeline_index = int(mesh["pipeline"].get_local_rank())
    stage, batch = _stubbed_distributed_step(domain, force_value=0.0)

    # Item 5 covers physical partitioning. Here the normal grouped-pipeline
    # driver, handoff, sentinel, and retirement paths remain live while avoiding
    # Gloo's concurrent subgroup all-to-all/P2P progress limitation.
    stage.partition = MagicMock(return_value=batch)
    if pipeline_index == 0 and stage._is_group_lead:
        stage._pending_input = batch

    pipeline = DistributedPipeline(stages={pipeline_index: stage}, mesh=mesh)
    pipeline.run()
    queue.put(
        (
            rank,
            pipeline_index,
            stage.done,
            stage.active_batch is None,
            stage.step_count,
            stage._system_step,
            _serialize_convergence(stage._dynamics._last_converged),
        )
    )


def test_converged_system_zero_remains_an_index_tensor() -> None:
    """``tensor([0])`` means system zero converged; zero is not ``False``."""
    results = sorted(
        run_gloo(world_size=2, fn=_all_ranks_converged_worker),
        key=lambda item: item[0],
    )

    assert results == [
        (0, ("tensor", [0]), ("tensor", [0])),
        (1, ("tensor", [0]), ("tensor", [0])),
    ]


def test_rank_disagreement_returns_no_convergence_without_hanging() -> None:
    """Every rank must enter the same convergence collective."""
    results = sorted(
        run_gloo(
            world_size=2,
            fn=_ranks_disagree_worker,
            timeout_sec=30.0,
        ),
        key=lambda item: item[0],
    )

    assert results == [
        (0, ("NoneType", None)),
        (1, ("NoneType", None)),
    ]


def test_real_partition_run_matches_full_system_convergence() -> None:
    """Real partitioning and model execution match full-system convergence."""
    results = sorted(
        run_gloo(
            world_size=2,
            fn=_partitioned_run_worker,
            timeout_sec=60.0,
        ),
        key=lambda item: item[0],
    )

    assert results == [
        (
            0,
            [1, 2],
            ("tensor", [0]),
            ("NoneType", None),
            ("NoneType", None),
            1,
            ("tensor", [0]),
            ("tensor", [0]),
            True,
        ),
        (
            1,
            [3, 4, 5],
            ("NoneType", None),
            ("NoneType", None),
            ("NoneType", None),
            1,
            ("tensor", [0]),
            ("NoneType", None),
            None,
        ),
    ]


def test_no_convergence_hook_adds_no_collective() -> None:
    """Fixed-step dynamics must not gain a convergence collective."""
    dp, batch = _stubbed_distributed_step(
        MagicMock(),
        force_value=0.0,
        check_convergence=False,
    )

    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "all_reduce") as all_reduce,
    ):
        _, converged = dp.step(batch)

    assert converged is None
    all_reduce.assert_not_called()


def test_frequency_gates_check_and_uses_strategy_process_group() -> None:
    """Only due steps check convergence on the strategy's domain group."""
    process_group = object()
    dp, batch = _stubbed_distributed_step(
        MagicMock(),
        force_value=0.0,
        frequency=2,
        process_group=process_group,
    )
    check = MagicMock(return_value=torch.tensor([0]))
    dp._dynamics._check_convergence = check

    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "all_reduce") as all_reduce,
    ):
        results = [_serialize_convergence(dp.step(batch)[1]) for _ in range(3)]

    assert results == [
        ("tensor", [0]),
        ("NoneType", None),
        ("tensor", [0]),
    ]
    assert check.call_count == 2
    assert all_reduce.call_count == 2
    assert [call.kwargs["group"] for call in all_reduce.call_args_list] == [
        process_group,
        process_group,
    ]


def test_run_stops_after_every_system_has_converged() -> None:
    """A distributed run should stop once all resident systems converge."""
    dp = _single_process_domain_parallel()
    batch = _single_graph_batch()
    step = MagicMock(return_value=(batch, torch.tensor([0])))
    dp.step = step
    dp._dist_model = object()
    dp._forces_primed = True

    result = dp.run(batch, n_steps=5)

    assert result is batch
    assert step.call_count == 1


def test_pipeline_stage_retires_converged_system_zero() -> None:
    """A final pipeline stage must retire system 0 when it converges."""
    stage = _single_process_domain_parallel()
    stage.next_rank = None
    stage.active_batch = _single_graph_batch()
    stage._dd_event = MagicMock()

    stage._poststep_sync_buffers(torch.tensor([0]))

    assert stage.active_batch is None
    assert stage._system_step == 0
    assert "converged" in stage._dd_event.call_args.args[0]


@pytest.mark.skip(
    reason="cross-stage handoff is unreliable under Gloo's single-machine "
    "progress engine; validate grouped pipelines on NCCL"
)
def test_grouped_pipeline_run_retires_converged_system() -> None:
    """The normal grouped driver retires system zero and terminates every rank."""
    results = sorted(
        run_gloo(
            world_size=4,
            fn=_grouped_pipeline_convergence_worker,
            timeout_sec=30.0,
        ),
        key=lambda item: item[0],
    )

    expected = (True, True, 1, 0, ("tensor", [0]))
    assert results == [
        (0, 0, *expected),
        (1, 0, *expected),
        (2, 1, *expected),
        (3, 1, *expected),
    ]
