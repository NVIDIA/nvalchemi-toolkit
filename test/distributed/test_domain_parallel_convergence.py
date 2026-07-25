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
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import torch
import torch.distributed as dist
from _gloo_harness import run_gloo
from torch.distributed import DeviceMesh

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.domain_parallel import DomainParallel
from nvalchemi.dynamics.base import ConvergenceHook
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


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
) -> tuple[DomainParallel, Batch]:
    """Build a DD step whose only live behavior is convergence evaluation."""
    model = DemoModelWrapper(DemoModel())
    inner = DemoDynamics(
        model=model,
        n_steps=10,
        convergence_hook=(
            ConvergenceHook.from_fmax(0.5) if check_convergence else None
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
