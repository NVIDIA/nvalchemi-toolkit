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

"""per_system_reduce primitive tests.

- Single-process smoke (degenerate: world_size=1, no dist init) — verifies
  local scatter-sum math and autograd.
- Multi-rank gloo: forward value matches a centralized reference.
- Multi-rank gloo: adjoint consistency <y, f(x)> == <f.backward(y), x>
  across ranks (the same rigorous autograd adjoint check used for the
  halo exchange primitives).
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.distributed._core.per_system import per_system_reduce
from nvalchemi.distributed._core.shard_tensor import ShardTensor
from nvalchemi.distributed.spec import SPEC_MPNN_HALO


def _mock_config() -> Any:
    cfg = MagicMock()
    cfg.mesh.get_group.return_value = None
    return cfg


def test_single_process_sum_matches_scatter() -> None:
    torch.manual_seed(0)
    local_vals = torch.randn(10, 3, dtype=torch.float64)
    system_index = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0], dtype=torch.long)
    n_systems = 3

    expected = torch.zeros(3, 3, dtype=torch.float64)
    expected.scatter_add_(0, system_index.unsqueeze(-1).expand(-1, 3), local_vals)

    got = per_system_reduce(local_vals, system_index, n_systems, _mock_config())
    torch.testing.assert_close(got, expected)


def test_single_process_autograd() -> None:
    torch.manual_seed(1)
    local_vals = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)
    system_index = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    out = per_system_reduce(local_vals, system_index, 3, _mock_config())
    loss = (out * torch.tensor([[1.0], [2.0], [3.0]])).sum()
    (grad,) = torch.autograd.grad(loss, local_vals)

    # Expected: grad[i] = op_scale[system_index[i]] = broadcast of [1, 1, 2, 2, 3, 3]
    expected = torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0], dtype=torch.float64)[
        :, None
    ].expand(-1, 2)
    torch.testing.assert_close(grad, expected)


def _init_gloo(rank: int, world_size: int, port: str = "29511") -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _worker(rank: int, world_size: int, fn: Any, *args: Any) -> None:
    _init_gloo(rank, world_size)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


class _MeshStub:
    def get_group(self) -> Any:
        return None


def _cfg() -> Any:
    cfg = MagicMock()
    cfg.mesh = _MeshStub()
    return cfg


def _test_forward_value(rank: int, world_size: int) -> None:
    """Each rank has some atoms for each of N systems; after per_system_reduce,
    every rank should see the same per-system sum equal to the global sum."""
    torch.manual_seed(100 + rank)
    n_systems = 3
    n_local = 5

    local_vals = torch.randn(n_local, 2, dtype=torch.float64)
    # Systems assigned uniformly-ish across ranks
    system_index = torch.randint(0, n_systems, (n_local,), dtype=torch.long)

    result = per_system_reduce(local_vals, system_index, n_systems, _cfg())

    # Reference: all ranks gather everyone's (local_vals, system_index), do the
    # scatter centrally, compare.
    all_vals: list[torch.Tensor] = [
        torch.zeros_like(local_vals) for _ in range(world_size)
    ]
    all_sys: list[torch.Tensor] = [
        torch.zeros_like(system_index) for _ in range(world_size)
    ]
    dist.all_gather(all_vals, local_vals)
    dist.all_gather(all_sys, system_index)

    ref = torch.zeros(n_systems, 2, dtype=torch.float64)
    for v, s in zip(all_vals, all_sys, strict=True):
        ref.scatter_add_(0, s.unsqueeze(-1).expand(-1, 2), v)

    torch.testing.assert_close(result, ref, rtol=1e-12, atol=1e-14)


def test_forward_value_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_forward_value), nprocs=2)


def test_forward_value_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_forward_value), nprocs=4)


def _test_adjoint(rank: int, world_size: int) -> None:
    """<y, f(x)>_global == <f.backward(y), x>_global.

    Same rigorous check as the halo-exchange adjoint tests. If this passes
    to 1e-10 tolerance, the backward all_reduce is doing the right thing
    across ranks.
    """
    torch.manual_seed(200 + rank)
    n_systems = 4
    n_local = 8
    feat_dim = 3

    x = torch.randn(n_local, feat_dim, dtype=torch.float64, requires_grad=True)
    system_index = torch.randint(0, n_systems, (n_local,), dtype=torch.long)

    out = per_system_reduce(x, system_index, n_systems, _cfg())

    gen = torch.Generator().manual_seed(999 + rank)
    y = torch.randn(n_systems, feat_dim, dtype=torch.float64, generator=gen)

    local_lhs = (y * out).sum()
    (grad_x,) = torch.autograd.grad(local_lhs, x)
    local_rhs = (grad_x * x.detach()).sum()

    global_lhs = local_lhs.detach().clone()
    global_rhs = local_rhs.detach().clone()
    dist.all_reduce(global_lhs, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_rhs, op=dist.ReduceOp.SUM)

    # Forward replicates across ranks — local_lhs varies per rank because y
    # differs. The ADJOINT identity is on the GLOBAL inner products.
    torch.testing.assert_close(global_lhs, global_rhs, rtol=1e-10, atol=1e-10)


def test_adjoint_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_adjoint), nprocs=2)


def test_adjoint_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_adjoint), nprocs=4)


_SPEC_HALO_WITH_SYSTEMS = SPEC_MPNN_HALO


def _fake_cfg_and_meta(
    n_padded: int, n_owned: int | None = None
) -> tuple[MagicMock, MagicMock]:
    cfg = MagicMock()
    cfg.mesh = _MeshStub()
    meta = MagicMock()
    meta.n_padded = n_padded
    # For single-system single-process tests, n_owned == n_padded is fine
    # (no halo). Handler only slices when src shape > n_owned.
    meta.n_owned = n_owned if n_owned is not None else n_padded
    return cfg, meta


def _test_mol_sum_idiom_routes_through_primitive(rank: int, world_size: int) -> None:
    """Simulates an AIMNet2 / MEGNet idiom:

        out = new_zeros(n_systems, F).scatter_add_(0, system_index, per_atom)

    When halo_context.n_systems matches ``out``'s dim-0, our registered
    handler routes through per_system_reduce, giving the correct global
    sum across all ranks.
    """
    torch.manual_seed(300 + rank)
    n_systems = 3
    n_local = 5
    feat_dim = 4

    per_atom_local = torch.randn(n_local, feat_dim, dtype=torch.float64)
    system_index = torch.randint(0, n_systems, (n_local,), dtype=torch.long)

    cfg, meta = _fake_cfg_and_meta(n_padded=n_local)

    # MLIP idiom: wrap the zero accumulator (which carries the per-system
    # shape) with halo metadata + n_systems, then scatter_add_ into it.
    accumulator = ShardTensor.wrap(
        torch.zeros(n_systems, feat_dim, dtype=torch.float64),
        meta=meta,
        config=cfg,
        n_systems=n_systems,
        spec=_SPEC_HALO_WITH_SYSTEMS,
    )
    result = accumulator.scatter_add_(
        0,
        system_index.unsqueeze(-1).expand(-1, feat_dim),
        per_atom_local,
    )

    assert isinstance(result, ShardTensor)

    # Reference via explicit all_gather + centralized scatter
    all_vals = [torch.zeros_like(per_atom_local) for _ in range(world_size)]
    all_sys = [torch.zeros_like(system_index) for _ in range(world_size)]
    dist.all_gather(all_vals, per_atom_local)
    dist.all_gather(all_sys, system_index)
    ref = torch.zeros(n_systems, feat_dim, dtype=torch.float64)
    for v, s in zip(all_vals, all_sys, strict=True):
        ref.scatter_add_(0, s.unsqueeze(-1).expand(-1, feat_dim), v)

    torch.testing.assert_close(result.unwrap(), ref, rtol=1e-12, atol=1e-14)


def test_mol_sum_idiom_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_mol_sum_idiom_routes_through_primitive), nprocs=2)


def test_mol_sum_idiom_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_mol_sum_idiom_routes_through_primitive), nprocs=4)


def _test_autograd_through_dispatch(rank: int, world_size: int) -> None:
    torch.manual_seed(400 + rank)
    n_systems = 2
    n_local = 6
    feat_dim = 3

    per_atom_local = torch.randn(
        n_local, feat_dim, dtype=torch.float64, requires_grad=True
    )
    system_index = torch.randint(0, n_systems, (n_local,), dtype=torch.long)

    cfg, meta = _fake_cfg_and_meta(n_padded=n_local)

    acc = ShardTensor.wrap(
        torch.zeros(n_systems, feat_dim, dtype=torch.float64),
        meta=meta,
        config=cfg,
        n_systems=n_systems,
        spec=_SPEC_HALO_WITH_SYSTEMS,
    )
    result = acc.scatter_add_(
        0, system_index.unsqueeze(-1).expand(-1, feat_dim), per_atom_local
    )

    # Adjoint-style test: <y, f(x)>_global == <f.backward(y), x>_global.
    gen = torch.Generator().manual_seed(9000 + rank)
    y = torch.randn(n_systems, feat_dim, dtype=torch.float64, generator=gen)

    local_lhs = (y * result).sum()
    (grad_x,) = torch.autograd.grad(local_lhs, per_atom_local)
    local_rhs = (grad_x * per_atom_local.detach()).sum()

    g_lhs = local_lhs.detach().clone()
    g_rhs = local_rhs.detach().clone()
    dist.all_reduce(g_lhs, op=dist.ReduceOp.SUM)
    dist.all_reduce(g_rhs, op=dist.ReduceOp.SUM)

    torch.testing.assert_close(g_lhs, g_rhs, rtol=1e-10, atol=1e-10)


def test_autograd_through_dispatch_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_autograd_through_dispatch), nprocs=2)


def _test_nonzero_accumulator_raises(rank: int, world_size: int) -> None:
    cfg, meta = _fake_cfg_and_meta(n_padded=4)
    # Accumulator pre-seeded with a non-zero value
    acc = ShardTensor.wrap(
        torch.ones(2, 3, dtype=torch.float64),
        meta=meta,
        config=cfg,
        n_systems=2,
        spec=_SPEC_HALO_WITH_SYSTEMS,
    )
    with pytest.raises(RuntimeError, match="zero-initialized"):
        acc.scatter_add_(
            0,
            torch.zeros(1, 3, dtype=torch.long),
            torch.ones(1, 3, dtype=torch.float64),
        )


def test_nonzero_accumulator_raises_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_nonzero_accumulator_raises), nprocs=2)
