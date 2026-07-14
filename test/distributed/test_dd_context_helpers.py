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

"""The ``current_dd_context()`` accessor + the shared context-aware helper
vocabulary (``refresh_neighbors`` / ``scatter_to_owners`` / ``system_sum``).

Two layers of coverage:

* **Single-process (no process group):** the context lifecycle (activate /
  restore / nest / sentinel), the derived properties, and the helper
  fallbacks (identity / plain scatter when not distributed).
* **Multi-rank gloo (CPU):** the helpers reproduce *exactly* the inline halo /
  per-system math the MACE / AIMNet2 / UMA wrappers express, so the wrappers
  can call the shared helper instead of re-implementing it.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.distributed._core.context import (
    NOT_DISTRIBUTED,
    DistributedContext,
    activate_dd_context,
    current_dd_context,
)
from nvalchemi.distributed._core.enums import Scope
from nvalchemi.distributed._core.particle_halo import (
    ParticleHaloConfig,
    halo_forward_exchange,
    halo_reverse_exchange,
    particle_halo_padding,
)
from nvalchemi.distributed._core.per_system import per_system_reduce
from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.helpers import (
    refresh_neighbors,
    scatter_to_owners,
    system_sum,
)
from nvalchemi.distributed.partitioner import SpatialPartitioner

# ======================================================================
# Single-process: context lifecycle
# ======================================================================


def test_sentinel_outside_any_scope() -> None:
    """Outside a DD forward, the accessor returns the inert sentinel."""
    ctx = current_dd_context()
    assert ctx is NOT_DISTRIBUTED
    assert ctx.is_distributed is False
    assert ctx.is_halo is False
    assert ctx.is_sharded is False


def test_activate_sets_and_restores() -> None:
    """``activate_dd_context`` makes a ctx current for the block and
    restores the previous one (the sentinel) on exit."""
    ctx = DistributedContext()
    assert current_dd_context() is NOT_DISTRIBUTED
    with activate_dd_context(ctx) as entered:
        assert entered is ctx
        assert current_dd_context() is ctx
    assert current_dd_context() is NOT_DISTRIBUTED


def test_activate_nesting_restores_outer() -> None:
    """Nested activation restores the outer context, not the sentinel."""
    outer = DistributedContext()
    inner = DistributedContext()
    with activate_dd_context(outer):
        assert current_dd_context() is outer
        with activate_dd_context(inner):
            assert current_dd_context() is inner
        assert current_dd_context() is outer
    assert current_dd_context() is NOT_DISTRIBUTED


def test_activate_restores_on_exception() -> None:
    """The previous context is restored even if the block raises."""
    ctx = DistributedContext()
    with torch.no_grad():  # noqa: SIM117 — explicit nesting for clarity
        try:
            with activate_dd_context(ctx):
                raise ValueError("boom")
        except ValueError:
            pass
    assert current_dd_context() is NOT_DISTRIBUTED


# ======================================================================
# Single-process: derived properties
# ======================================================================


def _halo_ctx(n_owned: int, n_padded: int, world_size: int = 2) -> DistributedContext:
    """A context with a mock halo meta/config (no real exchange needed for
    property checks)."""
    meta = MagicMock()
    meta.n_owned = n_owned
    meta.n_padded = n_padded
    meta.send_sizes = [[0] * world_size for _ in range(world_size)]
    cfg = MagicMock()
    cfg.rank = 1
    return DistributedContext(halo_config=cfg, halo_meta=meta)


def test_halo_properties_derive_from_meta() -> None:
    ctx = _halo_ctx(n_owned=7, n_padded=10, world_size=3)
    assert ctx.is_halo is True
    assert ctx.is_sharded is False
    assert ctx.is_distributed is True
    assert ctx.n_owned == 7
    assert ctx.n_padded == 10
    assert ctx.world_size == 3
    assert ctx.rank == 1
    assert ctx.compiling is False


def test_sharded_properties_derive_from_gather_meta() -> None:
    gm = MagicMock()
    gm.n_owned = 4
    gm.n_global = 12
    ctx = DistributedContext(gather_meta=gm, mesh=None)
    assert ctx.is_sharded is True
    assert ctx.is_halo is False
    assert ctx.is_distributed is True
    assert ctx.n_owned == 4
    # Node-replicate holds the full node set locally, so the padded view is
    # the global count.
    assert ctx.n_padded == 12


def test_sentinel_has_no_counts() -> None:
    assert NOT_DISTRIBUTED.n_owned is None
    assert NOT_DISTRIBUTED.n_padded is None
    assert NOT_DISTRIBUTED.world_size == 1
    assert NOT_DISTRIBUTED.rank == 0


# ======================================================================
# Single-process: helper fallbacks (not distributed -> plain local)
# ======================================================================


def test_refresh_neighbors_identity_when_not_distributed() -> None:
    x = torch.randn(5, 3)
    assert refresh_neighbors(x) is x


def test_scatter_to_owners_identity_when_not_distributed() -> None:
    x = torch.randn(5, 3)
    assert scatter_to_owners(x) is x


def test_system_sum_local_scatter_when_not_distributed() -> None:
    vals = torch.randn(10, 3, dtype=torch.float64)
    idx = torch.tensor([0, 0, 1, 1, 2, 2, 0, 1, 2, 0], dtype=torch.long)
    got = system_sum(vals, idx, 3)
    expected = torch.zeros(3, 3, dtype=torch.float64)
    expected.scatter_add_(0, idx.unsqueeze(-1).expand(-1, 3), vals)
    torch.testing.assert_close(got, expected)


# ======================================================================
# Multi-rank gloo harness (mirrors _core/test_halo_autograd.py)
# ======================================================================


def _patch_all_to_all_for_gloo() -> None:
    import physicsnemo.distributed.utils as pn_utils  # noqa: PLC0415

    def _indexed_all_to_all_v_gloo(tensor, indices, sizes, dim=0, group=None):
        comm_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        x_send = [tensor[idx].contiguous() for idx in indices]
        x_recv = []
        tensor_shape = list(tensor.shape)
        for r in range(comm_size):
            tensor_shape[dim] = sizes[r][rank]
            x_recv.append(
                torch.empty(tensor_shape, dtype=tensor.dtype, device=tensor.device)
            )
        ops = []
        for r in range(comm_size):
            if r == rank:
                x_recv[r].copy_(x_send[r])
            elif x_send[r].numel() > 0 or x_recv[r].numel() > 0:
                if x_send[r].numel() > 0:
                    ops.append(dist.isend(x_send[r], dst=r, group=group))
                if x_recv[r].numel() > 0:
                    ops.append(dist.irecv(x_recv[r], src=r, group=group))
        for op in ops:
            op.wait()
        return torch.cat(x_recv, dim=dim)

    pn_utils.indexed_all_to_all_v_wrapper = _indexed_all_to_all_v_gloo


def _init_gloo(rank: int, world_size: int, port: str = "29541") -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    _patch_all_to_all_for_gloo()


def _worker(rank: int, world_size: int, test_fn: Any, *args: Any) -> None:
    _init_gloo(rank, world_size)
    try:
        test_fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


class _MockMesh:
    def __init__(self, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size

    def get_local_rank(self) -> int:
        return self._rank

    def size(self, dim: int | None = None) -> int:
        return self._world_size

    def get_group(self) -> Any:
        return None


def _cubic_lattice(
    n_side: int = 6, lattice: float = 3.4
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    coords = torch.arange(n_side, dtype=torch.float64) * lattice
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    cell = torch.eye(3, dtype=torch.float64) * (n_side * lattice)
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, cell, pbc


def _build_rank_halo(
    rank: int, world_size: int, ghost_width: float = 5.0
) -> tuple[torch.Tensor, Any, Any]:
    """This rank's owned positions, halo metadata, and halo config."""
    positions, cell, pbc = _cubic_lattice(n_side=6, lattice=3.4)
    mesh = _MockMesh(rank, world_size)
    domain_config = DomainConfig(cutoff=ghost_width, mesh=mesh)
    partitioner = SpatialPartitioner(
        config=domain_config,
        cell_matrix=cell.unsqueeze(0),
        pbc=pbc.unsqueeze(0),
    )
    halo_config = ParticleHaloConfig(
        ghost_width=ghost_width, partitioner=partitioner, mesh=mesh
    )
    rank_assignment = partitioner.assign_atoms_to_ranks(positions)
    local_pos = positions[rank_assignment == rank].contiguous()
    _padded_pos, meta = particle_halo_padding(local_pos, halo_config)
    return local_pos, meta, halo_config


# ======================================================================
# Multi-rank: helper == inline math (behavior preservation)
# ======================================================================


def _test_refresh_neighbors_matches_inline(rank: int, world_size: int) -> None:
    """``refresh_neighbors(x)`` under the live ctx == the inline pattern
    ``halo_forward_exchange(x[:n_owned], meta, cfg)`` with trailing
    dead-padding rows (if any) preserved."""
    _local_pos, meta, cfg = _build_rank_halo(rank, world_size)
    n_owned = int(meta.n_owned)
    n_padded = int(meta.n_padded)
    feat_dim = 3
    gen = torch.Generator().manual_seed(10 + rank)

    ctx = DistributedContext(
        mesh=_MockMesh(rank, world_size), halo_config=cfg, policy=HaloStoragePolicy()
    )
    ctx.halo_meta = meta

    # (a) x exactly n_padded (no caps padding).
    x = torch.randn((n_padded, feat_dim), dtype=torch.float64, generator=gen)
    expected = halo_forward_exchange(x[:n_owned].contiguous(), meta, cfg)
    with activate_dd_context(ctx):
        got = refresh_neighbors(x)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)

    # (b) x with trailing dead-padding rows (the caps case) — preserved as-is.
    n_dead = 4
    x_pad = torch.cat(
        [x, torch.full((n_dead, feat_dim), -7.0, dtype=torch.float64)], dim=0
    )
    inner = halo_forward_exchange(x_pad[:n_owned].contiguous(), meta, cfg)
    expected_pad = torch.cat([inner, x_pad[n_padded:]], dim=0)
    with activate_dd_context(ctx):
        got_pad = refresh_neighbors(x_pad)
    torch.testing.assert_close(got_pad, expected_pad, rtol=0, atol=0)
    # dead rows are untouched
    torch.testing.assert_close(got_pad[n_padded:], x_pad[n_padded:], rtol=0, atol=0)


def test_refresh_neighbors_matches_inline_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_refresh_neighbors_matches_inline), nprocs=2)


def test_refresh_neighbors_matches_inline_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_refresh_neighbors_matches_inline), nprocs=4)


def _test_scatter_to_owners_matches_inline(rank: int, world_size: int) -> None:
    """``scatter_to_owners(out)`` == inline reverse-then-forward (the eager
    analogue of the compiled scatter-correct op)."""
    _local_pos, meta, cfg = _build_rank_halo(rank, world_size)
    n_padded = int(meta.n_padded)
    feat_dim = 2
    gen = torch.Generator().manual_seed(20 + rank)

    ctx = DistributedContext(
        mesh=_MockMesh(rank, world_size), halo_config=cfg, policy=HaloStoragePolicy()
    )
    ctx.halo_meta = meta

    out = torch.randn((n_padded, feat_dim), dtype=torch.float64, generator=gen)
    owned = halo_reverse_exchange(out.contiguous(), meta, cfg)
    expected = halo_forward_exchange(owned, meta, cfg)
    with activate_dd_context(ctx):
        got = scatter_to_owners(out)
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_scatter_to_owners_matches_inline_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_scatter_to_owners_matches_inline), nprocs=2)


def _test_system_sum_owned_matches_global(rank: int, world_size: int) -> None:
    """``system_sum(OWNED)`` == ``per_system_reduce`` over owned rows == the
    centralized global per-system sum across ranks."""
    _local_pos, meta, cfg = _build_rank_halo(rank, world_size)
    n_owned = int(meta.n_owned)
    n_padded = int(meta.n_padded)
    n_systems = 3
    feat_dim = 2
    gen = torch.Generator().manual_seed(30 + rank)

    ctx = DistributedContext(
        mesh=_MockMesh(rank, world_size), halo_config=cfg, policy=HaloStoragePolicy()
    )
    ctx.halo_meta = meta

    # Per-node values over the padded block; only owned rows must count.
    vals = torch.randn((n_padded, feat_dim), dtype=torch.float64, generator=gen)
    idx = torch.randint(0, n_systems, (n_padded,), dtype=torch.long, generator=gen)

    with activate_dd_context(ctx):
        got = system_sum(vals, idx, n_systems, scope=Scope.OWNED)

    # (a) equals the primitive over owned rows.
    prim = per_system_reduce(
        vals[:n_owned].contiguous(), idx[:n_owned].contiguous(), n_systems, cfg
    )
    torch.testing.assert_close(got, prim, rtol=0, atol=0)

    # (b) equals the centralized reference (all_gather owned, scatter centrally).
    all_vals = [
        torch.zeros(n_owned, feat_dim, dtype=torch.float64) for _ in range(world_size)
    ]
    all_idx = [torch.zeros(n_owned, dtype=torch.long) for _ in range(world_size)]
    dist.all_gather(all_vals, vals[:n_owned].contiguous())
    dist.all_gather(all_idx, idx[:n_owned].contiguous())
    ref = torch.zeros(n_systems, feat_dim, dtype=torch.float64)
    for v, s in zip(all_vals, all_idx, strict=True):
        ref.scatter_add_(0, s.unsqueeze(-1).expand(-1, feat_dim), v)
    torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-14)


def test_system_sum_owned_matches_global_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_system_sum_owned_matches_global), nprocs=2)


def test_system_sum_owned_matches_global_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_system_sum_owned_matches_global), nprocs=4)


def _test_system_sum_local_is_per_rank_partial(rank: int, world_size: int) -> None:
    """``system_sum(LOCAL)`` is this rank's owned-only partial with NO
    all-reduce — equal to a local scatter_add over owned rows."""
    _local_pos, meta, cfg = _build_rank_halo(rank, world_size)
    n_owned = int(meta.n_owned)
    n_padded = int(meta.n_padded)
    n_systems = 3
    feat_dim = 2
    gen = torch.Generator().manual_seed(40 + rank)

    ctx = DistributedContext(
        mesh=_MockMesh(rank, world_size), halo_config=cfg, policy=HaloStoragePolicy()
    )
    ctx.halo_meta = meta

    vals = torch.randn((n_padded, feat_dim), dtype=torch.float64, generator=gen)
    idx = torch.randint(0, n_systems, (n_padded,), dtype=torch.long, generator=gen)

    with activate_dd_context(ctx):
        got = system_sum(vals, idx, n_systems, scope=Scope.LOCAL)

    expected = torch.zeros(n_systems, feat_dim, dtype=torch.float64)
    expected.scatter_add_(
        0,
        idx[:n_owned].unsqueeze(-1).expand(-1, feat_dim),
        vals[:n_owned],
    )
    torch.testing.assert_close(got, expected, rtol=0, atol=0)


def test_system_sum_local_is_per_rank_partial_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_system_sum_local_is_per_rank_partial), nprocs=2)
