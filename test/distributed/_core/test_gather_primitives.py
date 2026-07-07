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

"""ShardTensor primitive tests.

Covers:
- Single-process smoke: ``distributed_index_select`` / ``distributed_scatter_add``
  degrade to local ops when no process group is initialized.
- Metadata construction: global_id → (owner, local_idx) routing is correct.
- Multi-rank gloo forward: gathering rows by global id matches a
  single-process reference.
- Multi-rank gloo autograd: (a) adjoint identity, (b) per-rank gradient
  slice matches single-rank reference on a local-loss pattern (DDP).
"""

from __future__ import annotations

import os
import types
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
import torch._dynamo
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from nvalchemi.distributed._core.gather_primitives import (
    ShardRouting,
    _FixedDistributedIndexSelect,
    distributed_all_reduce,
    distributed_index_select,
    distributed_scatter_add,
    funcol_all_to_all_fixed,
    funcol_all_to_all_v_rows,
    funcol_fixed_index_select,
)
from nvalchemi.distributed._core.per_system import per_system_reduce
from test.distributed._gloo_harness import run_gloo


def test_metadata_from_assignment_roundtrip() -> None:
    """Verify owner_rank and local_index together re-create a valid
    partition (counting each rank's atoms and mapping back without
    collisions)."""
    assignment = torch.tensor([0, 1, 0, 1, 0, 2, 2], dtype=torch.long)
    meta = ShardRouting.from_assignment(assignment, rank=1)

    # Rank 1 owns atoms [1, 3] → n_owned=2.
    assert meta.n_owned == 2
    assert meta.n_global == 7

    # Owner table round-trips.
    torch.testing.assert_close(meta.owner_rank, assignment)

    # Local indices are contiguous 0..k-1 per rank (in argsort-stable order).
    expected_local = torch.tensor([0, 0, 1, 1, 2, 0, 1], dtype=torch.long)
    torch.testing.assert_close(meta.local_index, expected_local)


def _mock_config() -> Any:
    cfg = MagicMock()
    cfg.mesh.get_group.return_value = None
    return cfg


def test_index_select_single_process_degenerate() -> None:
    """Without dist init the primitive does a plain local index_select."""
    n_owned = 5
    x = torch.randn(n_owned, 3, dtype=torch.float64, requires_grad=True)
    meta = ShardRouting(
        n_owned=n_owned,
        n_global=n_owned,
        owner_rank=torch.zeros(n_owned, dtype=torch.long),
        local_index=torch.arange(n_owned, dtype=torch.long),
    )
    indices = torch.tensor([0, 2, 4, 1], dtype=torch.long)

    got = distributed_index_select(x, indices, meta, _mock_config())
    torch.testing.assert_close(got, x.index_select(0, indices))


def test_scatter_add_single_process_degenerate() -> None:
    n_owned = 4
    self_t = torch.zeros(n_owned, 2, dtype=torch.float64)
    src = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float64)
    indices = torch.tensor([0, 2, 0], dtype=torch.long)
    meta = ShardRouting(
        n_owned=n_owned,
        n_global=n_owned,
        owner_rank=torch.zeros(n_owned, dtype=torch.long),
        local_index=torch.arange(n_owned, dtype=torch.long),
    )
    out = distributed_scatter_add(self_t, indices, src, meta, _mock_config())
    expected = torch.zeros(n_owned, 2, dtype=torch.float64)
    expected.scatter_add_(0, indices.unsqueeze(-1).expand(-1, 2), src)
    torch.testing.assert_close(out, expected)


def _init_gloo(rank: int, world_size: int, port: str = "29521") -> None:
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


def _test_index_select_forward(rank: int, world_size: int) -> None:
    """Each rank owns a contiguous slice of a global atom space; each
    rank requests a mix of local + remote global indices. Result must
    match a single-process reference (gather all shards and index_select
    centrally).
    """
    per_rank = 3
    n_global = per_rank * world_size
    feat = 4
    # Partition: rank r owns atoms [r * per_rank, (r+1) * per_rank).
    assignment = torch.arange(n_global, dtype=torch.long) // per_rank
    meta = ShardRouting.from_assignment(assignment, rank=rank)

    # Each rank's shard — seeded deterministically so the reference can
    # reconstruct it.
    torch.manual_seed(7)
    all_shards = [
        torch.randn(per_rank, feat, dtype=torch.float64) for _ in range(world_size)
    ]
    my_shard = all_shards[rank].contiguous()

    # Each rank gathers a different set of global indices.
    torch.manual_seed(100 + rank)
    K = 5
    global_indices = torch.randint(0, n_global, (K,), dtype=torch.long)

    got = distributed_index_select(my_shard, global_indices, meta, _cfg())

    # Reference: single-rank concatenation.
    ref_full = torch.cat(all_shards, dim=0)
    expected = ref_full.index_select(0, global_indices)
    torch.testing.assert_close(got, expected)


def test_index_select_forward_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_index_select_forward), nprocs=2)


def test_index_select_forward_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_index_select_forward), nprocs=4)


def _test_scatter_add_forward(rank: int, world_size: int) -> None:
    """Each rank scatter-adds contributions at global indices; the
    combined result (gathered across ranks) must match a single-rank
    scatter-add reference.
    """
    per_rank = 2
    n_global = per_rank * world_size
    feat = 3
    assignment = torch.arange(n_global, dtype=torch.long) // per_rank
    meta = ShardRouting.from_assignment(assignment, rank=rank)

    my_shard = torch.zeros(per_rank, feat, dtype=torch.float64)

    torch.manual_seed(200 + rank)
    K = 6
    global_indices = torch.randint(0, n_global, (K,), dtype=torch.long)
    src = torch.randn(K, feat, dtype=torch.float64)

    # In-place scatter_add; returned tensor is my_shard.
    my_shard = distributed_scatter_add(my_shard, global_indices, src, meta, _cfg())

    # Reference: gather all ranks' (indices, src) and do centralized scatter.
    all_indices = [torch.zeros_like(global_indices) for _ in range(world_size)]
    all_src = [torch.zeros_like(src) for _ in range(world_size)]
    dist.all_gather(all_indices, global_indices)
    dist.all_gather(all_src, src)
    ref_full = torch.zeros(n_global, feat, dtype=torch.float64)
    for r in range(world_size):
        ref_full.scatter_add_(
            0,
            all_indices[r].unsqueeze(-1).expand(-1, feat),
            all_src[r],
        )

    # Compare rank r's owned slice to ref_full[rank*per_rank : (rank+1)*per_rank].
    expected = ref_full[rank * per_rank : (rank + 1) * per_rank]
    torch.testing.assert_close(my_shard, expected)


def _test_scatter_add_fp64_accumulation(rank: int, world_size: int) -> None:
    """fp32 contributions are folded in fp64 (downcast at the end), so small
    terms survive against a large one. All contributions target a single owned
    index; rank 0 adds a large value, every rank adds many small ones. An fp32
    accumulator rounds the small terms off against the large running sum; the
    fp64 fold keeps them.
    """
    per_rank = 2
    n_global = per_rank * world_size
    feat = 1
    assignment = torch.arange(n_global, dtype=torch.long) // per_rank
    meta = ShardRouting.from_assignment(assignment, rank=rank)
    my_shard = torch.zeros(per_rank, feat, dtype=torch.float32)

    K = 64
    global_indices = torch.zeros(K, dtype=torch.long)  # all -> global atom 0 (rank 0)
    src = torch.full((K, feat), 0.1, dtype=torch.float32)
    if rank == 0:
        src[0, 0] = 1.0e7  # large term that swamps fp32 addition of the 0.1s

    my_shard = distributed_scatter_add(my_shard, global_indices, src, meta, _cfg())

    all_src = [torch.zeros_like(src) for _ in range(world_size)]
    dist.all_gather(all_src, src)
    expected0 = torch.stack(all_src).to(torch.float64).sum().to(torch.float32)

    if rank == 0:
        got0 = my_shard[0, 0]
        # fp64 fold: ~1e7 + (world_size*K - 1)*0.1; an fp32 fold collapses to 1e7.
        torch.testing.assert_close(got0, expected0, rtol=0.0, atol=0.5)
        assert got0.item() > 1.0e7 + 1.0, (
            "small fp32 contributions were lost -- fold is not accumulating in fp64"
        )


def test_scatter_add_fp64_accumulation_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_scatter_add_fp64_accumulation), nprocs=2)


def test_scatter_add_forward_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_scatter_add_forward), nprocs=2)


def test_scatter_add_forward_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_scatter_add_forward), nprocs=4)


def _test_index_select_adjoint(rank: int, world_size: int) -> None:
    """<y, f(x)>_global == <f^T(y), x>_global for f = distributed_index_select.
    Random per-rank y, random per-rank x. The adjoint test fixes the
    backward up to a constant and confirms the all_to_all_v routing
    inside backward mirrors the forward.
    """
    per_rank = 3
    n_global = per_rank * world_size
    feat = 3
    assignment = torch.arange(n_global, dtype=torch.long) // per_rank
    meta = ShardRouting.from_assignment(assignment, rank=rank)

    torch.manual_seed(300 + rank)
    x = torch.randn(per_rank, feat, dtype=torch.float64, requires_grad=True)

    torch.manual_seed(400 + rank)
    K = 6
    global_indices = torch.randint(0, n_global, (K,), dtype=torch.long)
    y = torch.randn(K, feat, dtype=torch.float64)

    out = distributed_index_select(x, global_indices, meta, _cfg())
    local_lhs = (y * out).sum()
    (grad_x,) = torch.autograd.grad(local_lhs, x)
    local_rhs = (grad_x * x.detach()).sum()

    global_lhs = local_lhs.detach().clone()
    global_rhs = local_rhs.detach().clone()
    dist.all_reduce(global_lhs, op=dist.ReduceOp.SUM)
    dist.all_reduce(global_rhs, op=dist.ReduceOp.SUM)
    torch.testing.assert_close(global_lhs, global_rhs, rtol=1e-10, atol=1e-10)


def test_index_select_adjoint_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_index_select_adjoint), nprocs=2)


def test_index_select_adjoint_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_index_select_adjoint), nprocs=4)


def _test_index_select_local_loss(rank: int, world_size: int) -> None:
    """Each rank computes a LOCAL scalar loss from its gathered rows.
    Summing local losses across ranks yields the single-rank total loss.
    Per-rank grad_x should match the single-rank reference's slice of
    the global grad.
    """
    per_rank = 2
    n_global = per_rank * world_size
    feat = 2
    assignment = torch.arange(n_global, dtype=torch.long) // per_rank
    meta = ShardRouting.from_assignment(assignment, rank=rank)

    torch.manual_seed(77)
    all_shards = [
        torch.randn(per_rank, feat, dtype=torch.float64) for _ in range(world_size)
    ]
    my_shard = all_shards[rank].clone().requires_grad_(True)

    # Each rank gathers DIFFERENT global indices — produces a different
    # local scalar loss per rank.
    torch.manual_seed(500 + rank)
    K = 4
    global_indices = torch.randint(0, n_global, (K,), dtype=torch.long)
    w = torch.randn(K, feat, dtype=torch.float64)

    out = distributed_index_select(my_shard, global_indices, meta, _cfg())
    local_loss = (w * out).sum()
    (grad_my_shard,) = torch.autograd.grad(local_loss, my_shard)

    # Reference: gather everyone's indices+w, run a single-rank gather
    # on the concatenated shards, compute sum of per-rank losses, take
    # autograd wrt the full shard.
    all_indices = [torch.zeros_like(global_indices) for _ in range(world_size)]
    all_w = [torch.zeros_like(w) for _ in range(world_size)]
    dist.all_gather(all_indices, global_indices)
    dist.all_gather(all_w, w)

    ref_full = torch.cat(all_shards, dim=0).detach().requires_grad_(True)
    total_loss = torch.zeros((), dtype=torch.float64)
    for r in range(world_size):
        out_r = ref_full.index_select(0, all_indices[r])
        total_loss = total_loss + (all_w[r] * out_r).sum()
    (ref_grad_full,) = torch.autograd.grad(total_loss, ref_full)

    # Rank r's shard grad should match ref_grad_full[r*per_rank:(r+1)*per_rank].
    expected = ref_grad_full[rank * per_rank : (rank + 1) * per_rank]
    torch.testing.assert_close(grad_my_shard, expected, rtol=1e-10, atol=1e-10)


def test_index_select_local_loss_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_index_select_local_loss), nprocs=2)


def test_index_select_local_loss_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_index_select_local_loss), nprocs=4)


def _test_scatter_add_adjoint(rank: int, world_size: int) -> None:
    """Scatter-add forward f(src; idx): accumulates into self_t.
    backward grad_src[k] = grad_self_t[idx[k]] (gathered across ranks).

    Adjoint: <y_self, f(zeros, idx, src)> == <grad_src_backward, src>
    (globally). Easier to verify by: run forward with zero accumulator,
    take backward wrt src, compare to a single-rank reference.
    """
    per_rank = 2
    n_global = per_rank * world_size
    feat = 2
    assignment = torch.arange(n_global, dtype=torch.long) // per_rank
    meta = ShardRouting.from_assignment(assignment, rank=rank)

    torch.manual_seed(600 + rank)
    K = 5
    global_indices = torch.randint(0, n_global, (K,), dtype=torch.long)
    src = torch.randn(K, feat, dtype=torch.float64, requires_grad=True)

    self_t = torch.zeros(per_rank, feat, dtype=torch.float64)
    self_t = distributed_scatter_add(self_t, global_indices, src, meta, _cfg())

    # loss = sum of self_t (global implicit via all_reduce of owned slice).
    local_loss = self_t.sum()
    (grad_src,) = torch.autograd.grad(local_loss, src)

    # Ref: concatenate all (indices, src), do central scatter_add, sum
    # all entries. Local-loss across ranks sums to scatter_add.sum().
    # grad_src[r][k] = 1 per element (since loss = sum of all self_t).
    # After gather, grad_src on rank r should be ones — every src row
    # contributes to exactly one accumulator slot, all of which sum to
    # the total loss.
    expected = torch.ones_like(src)
    torch.testing.assert_close(grad_src, expected, rtol=1e-10, atol=1e-10)


def test_scatter_add_adjoint_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_scatter_add_adjoint), nprocs=2)


def test_scatter_add_adjoint_4ranks() -> None:
    mp.spawn(_worker, args=(4, _test_scatter_add_adjoint), nprocs=4)


def _fake_halo_meta_cfg(n_owned: int) -> tuple[Any, Any]:
    halo_meta = MagicMock()
    halo_meta.n_owned = n_owned
    halo_meta.n_padded = n_owned
    cfg = MagicMock()
    cfg.mesh = _MeshStub()
    return halo_meta, cfg


def _test_dispatch_index_select(rank: int, world_size: int) -> None:
    """``t.index_select(0, global_idx)`` auto-dispatches to the distributed
    gather when the tensor carries ``gather_meta``."""
    from nvalchemi.distributed._core.shard_tensor import ShardTensor
    from nvalchemi.distributed._core.spec import DistributionSpec
    from nvalchemi.distributed._core.storage_policy import PlainShard
    from nvalchemi.distributed.spec import MLIPSpec

    per_rank = 3
    n_global = per_rank * world_size
    feat = 4
    assignment = torch.arange(n_global, dtype=torch.long) // per_rank
    gather_meta = ShardRouting.from_assignment(assignment, rank=rank)

    torch.manual_seed(42)
    all_shards = [
        torch.randn(per_rank, feat, dtype=torch.float64) for _ in range(world_size)
    ]

    torch.manual_seed(500 + rank)
    K = 5
    global_indices = torch.randint(0, n_global, (K,), dtype=torch.long)

    _halo_meta, cfg = _fake_halo_meta_cfg(n_owned=per_rank)
    my_shard = ShardTensor.wrap(
        all_shards[rank].contiguous(),
        gather_meta=gather_meta,
        config=cfg,
        spec=MLIPSpec(distribution=DistributionSpec(policy=PlainShard())),
    )
    got = my_shard.index_select(0, global_indices)

    # Should be a ShardTensor wrapping the distributed result.
    assert isinstance(got, ShardTensor)

    ref_full = torch.cat(all_shards, dim=0)
    expected = ref_full.index_select(0, global_indices)
    torch.testing.assert_close(got.unwrap(), expected)


def test_dispatch_index_select_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_dispatch_index_select), nprocs=2)


def _test_dispatch_scatter_add(rank: int, world_size: int) -> None:
    """``zeros.scatter_add_(0, global_idx, src)`` auto-dispatches to the
    distributed scatter when the accumulator carries ``gather_meta``."""
    from nvalchemi.distributed._core.shard_tensor import ShardTensor
    from nvalchemi.distributed._core.spec import DistributionSpec
    from nvalchemi.distributed._core.storage_policy import PlainShard
    from nvalchemi.distributed.spec import MLIPSpec

    per_rank = 2
    n_global = per_rank * world_size
    feat = 3
    assignment = torch.arange(n_global, dtype=torch.long) // per_rank
    gather_meta = ShardRouting.from_assignment(assignment, rank=rank)

    torch.manual_seed(600 + rank)
    K = 4
    global_indices = torch.randint(0, n_global, (K,), dtype=torch.long)
    src = torch.randn(K, feat, dtype=torch.float64)

    _halo_meta, cfg = _fake_halo_meta_cfg(n_owned=per_rank)
    my_shard = ShardTensor.wrap(
        torch.zeros(per_rank, feat, dtype=torch.float64),
        gather_meta=gather_meta,
        config=cfg,
        spec=MLIPSpec(distribution=DistributionSpec(policy=PlainShard())),
    )

    # Expand index to broadcast to src shape, matching the MLIP idiom.
    my_shard.scatter_add_(0, global_indices.unsqueeze(-1).expand(-1, feat), src)

    # Reference: gather everyone's (indices, src) and central scatter.
    all_idx = [torch.zeros_like(global_indices) for _ in range(world_size)]
    all_src = [torch.zeros_like(src) for _ in range(world_size)]
    dist.all_gather(all_idx, global_indices)
    dist.all_gather(all_src, src)
    ref_full = torch.zeros(n_global, feat, dtype=torch.float64)
    for r in range(world_size):
        ref_full.scatter_add_(
            0,
            all_idx[r].unsqueeze(-1).expand(-1, feat),
            all_src[r],
        )
    expected = ref_full[rank * per_rank : (rank + 1) * per_rank]
    torch.testing.assert_close(my_shard.unwrap(), expected)


def test_dispatch_scatter_add_2ranks() -> None:
    mp.spawn(_worker, args=(2, _test_dispatch_scatter_add), nprocs=2)


def _test_dispatch_fallthrough_without_gather_meta(rank: int, world_size: int) -> None:
    """Without ``gather_meta`` on the tensor, ShardTensor falls through
    to plain-tensor behavior — index_select is local, not distributed."""
    from nvalchemi.distributed._core.shard_tensor import ShardTensor

    per_rank = 3
    # Wrap WITHOUT gather_meta — the dispatch predicate returns False and
    # the default __torch_function__ handles the index_select locally.
    my_shard = ShardTensor.wrap(
        torch.arange(per_rank * 2, dtype=torch.float64).reshape(per_rank, 2)
    )

    local_indices = torch.tensor([0, 2, 1], dtype=torch.long)
    got = my_shard.index_select(0, local_indices)

    assert got.shape == (3, 2)
    torch.testing.assert_close(
        got.unwrap(),
        my_shard.unwrap().index_select(0, local_indices),
    )


def test_dispatch_fallthrough_without_gather_meta_2ranks() -> None:
    mp.spawn(
        _worker, args=(2, _test_dispatch_fallthrough_without_gather_meta), nprocs=2
    )


class SyntheticAIMNet2Gather(nn.Module):
    """AIMNet2-shaped reference model with every cross-rank op expressed
    through ``distributed_index_select`` + ``per_system_reduce``.

    When called in single-process mode, uses plain local index_select /
    scatter_add — results are bit-identical to a distributed 2-rank run
    (to float64 precision).
    """

    def __init__(self, hidden: int = 4, num_layers: int = 2) -> None:
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Linear(1, hidden, bias=False)
        self.update_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden)
                )
                for _ in range(num_layers)
            ]
        )
        self.charge_head = nn.ModuleList(
            [nn.Linear(hidden, 1) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1)
        )

    def forward(
        self,
        atomic_numbers_owned: torch.Tensor,
        positions_owned: torch.Tensor,
        edge_src_global: torch.Tensor,
        edge_dst_local: torch.Tensor,
        cutoff: float,
        system_index_owned: torch.Tensor,
        target_total_charge: torch.Tensor,
        n_systems: int,
        meta: ShardRouting | None,
        config: Any,
    ) -> torch.Tensor:
        """Returns owned per-atom energies (shape ``(n_owned,)``).

        Edge convention: each edge points TO a locally-owned destination
        (``edge_dst_local`` in ``[0, n_owned)``) FROM a source atom
        identified by its GLOBAL id (``edge_src_global`` in
        ``[0, n_global)``). This mirrors halo mode's ``dst < n_owned``
        edge filter: each cross-rank edge lives on exactly one rank.
        """
        # Gather source positions cross-rank for autograd-correct edge
        # weights (each source position contributes to backward).
        if meta is not None and dist.is_initialized():
            src_pos = distributed_index_select(
                positions_owned, edge_src_global, meta, config
            )
        else:
            src_pos = positions_owned.index_select(0, edge_src_global)
        dst_pos = positions_owned.index_select(0, edge_dst_local)
        edge_d = (dst_pos - src_pos).norm(dim=-1)
        edge_weight = 0.5 * (torch.cos(torch.pi * edge_d / cutoff) + 1.0)

        x = self.embed(atomic_numbers_owned.to(positions_owned.dtype).unsqueeze(-1))

        for upd, qh in zip(self.update_mlp, self.charge_head):
            # Gather source features from across ranks — the ONE
            # cross-rank op per layer on the feature tensor.
            if meta is not None and dist.is_initialized():
                src_feats = distributed_index_select(x, edge_src_global, meta, config)
            else:
                src_feats = x.index_select(0, edge_src_global)

            msg = src_feats * edge_weight.unsqueeze(-1)

            # Local scatter to destinations (all owned).
            agg = torch.zeros_like(x)
            agg.index_add_(0, edge_dst_local, msg)

            x = x + upd(agg)

            # Per-molecule charge-equilibration residual. Use the
            # distributed primitive only when a ``meta`` tells us the
            # caller is in gather mode; otherwise do a plain local
            # scatter (dist may be initialized for the reference run
            # inside a distributed worker, but that reference run is
            # single-rank semantically and shouldn't all_reduce).
            q_per_atom = qh(x).squeeze(-1)
            if meta is not None and dist.is_initialized():
                total_q = per_system_reduce(
                    q_per_atom, system_index_owned, n_systems, config
                )
            else:
                total_q = torch.zeros(
                    n_systems, dtype=q_per_atom.dtype, device=q_per_atom.device
                )
                total_q.scatter_add_(0, system_index_owned, q_per_atom)
            residual = (target_total_charge - total_q)[system_index_owned]
            x = x + residual.unsqueeze(-1) * 0.1

        per_atom_e = self.readout(x).squeeze(-1)
        return per_atom_e


def _build_edges(
    positions: torch.Tensor, cutoff: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute (src, dst, edge_weight) for all pairs within cutoff."""
    dr = positions.unsqueeze(0) - positions.unsqueeze(1)
    d = dr.norm(dim=-1)
    mask = (d < cutoff) & (d > 1e-8)
    idx = mask.nonzero(as_tuple=False)
    src, dst = idx[:, 0], idx[:, 1]
    edge_d = d[src, dst]
    w = 0.5 * (torch.cos(torch.pi * edge_d / cutoff) + 1.0)
    return src, dst, w


def _run_gather_synthetic(rank: int, world_size: int) -> None:
    """One molecule (8 atoms) split across 2 ranks. Positions chosen so
    that edges span the rank boundary — the gather has real work to do."""
    assert world_size == 2

    cutoff = 5.0
    dtype = torch.float64
    n_atoms = 8
    n_systems = 1

    # Linear chain.
    positions_global = torch.stack(
        [
            0.25 + torch.arange(n_atoms, dtype=dtype) * 1.5,
            torch.zeros(n_atoms, dtype=dtype),
            torch.zeros(n_atoms, dtype=dtype),
        ],
        dim=1,
    ).contiguous()
    atomic_numbers_global = torch.full((n_atoms,), 6.0, dtype=dtype)
    system_index_global = torch.zeros(n_atoms, dtype=torch.long)
    target_total_charge = torch.zeros(1, dtype=dtype)

    # Rank assignment: first half to rank 0, second half to rank 1.
    per_rank = n_atoms // world_size
    assignment = torch.arange(n_atoms, dtype=torch.long) // per_rank
    meta = ShardRouting.from_assignment(assignment, rank=rank)

    # ---- Single-process reference ----
    torch.manual_seed(1234)
    model_ref = SyntheticAIMNet2Gather(hidden=4, num_layers=2).to(dtype=dtype)
    model_ref.eval()

    ref_positions = positions_global.clone().requires_grad_(True)
    ref_src, ref_dst, _ = _build_edges(ref_positions.detach(), cutoff)
    per_atom_ref = model_ref(
        atomic_numbers_global,
        ref_positions,
        ref_src,
        ref_dst,
        cutoff,
        system_index_global,
        target_total_charge,
        n_systems=n_systems,
        meta=None,
        config=_cfg(),
    )
    e_ref_total = per_atom_ref.sum()
    (grad_ref,) = torch.autograd.grad(e_ref_total, ref_positions)
    forces_ref = -grad_ref.detach()

    # ---- Distributed ----
    torch.manual_seed(1234)
    model_dist = SyntheticAIMNet2Gather(hidden=4, num_layers=2).to(dtype=dtype)
    model_dist.eval()

    local_mask = assignment == rank
    local_pos = positions_global[local_mask].clone().requires_grad_(True)
    local_z = atomic_numbers_global[local_mask]
    local_system_idx = system_index_global[local_mask]

    # Build per-rank edges.
    # First build edges on the FULL positions (so edge sources can be
    # atoms on any rank). Then filter to edges with destination on
    # THIS rank (dst locally owned) — the convention "each cross-rank
    # edge lives on one rank".
    full_src, full_dst, _ = _build_edges(positions_global, cutoff)
    dst_assignment = assignment[full_dst]
    my_edge_mask = dst_assignment == rank
    edge_src_global = full_src[my_edge_mask]
    edge_dst_global = full_dst[my_edge_mask]
    # Convert global dst to LOCAL owned index on this rank.
    edge_dst_local = meta.local_index.to(edge_dst_global.device)[edge_dst_global]

    per_atom_dist = model_dist(
        local_z,
        local_pos,
        edge_src_global,
        edge_dst_local,
        cutoff,
        local_system_idx,
        target_total_charge,
        n_systems=n_systems,
        meta=meta,
        config=_cfg(),
    )
    # Local-loss pattern: sum owned atoms' energies on each rank → grad
    # wrt local_pos matches the single-rank reference's slice.
    e_dist_local = per_atom_dist.sum()
    (grad_dist,) = torch.autograd.grad(e_dist_local, local_pos)
    forces_dist = -grad_dist.detach()

    forces_ref_owned = forces_ref[local_mask]
    # Gather mode preserves float64 precision: each gather/scatter is
    # exactly ONE cross-rank op per call, no chain of halo_forward_exchange
    # autograd nodes as in halo mode's index_select dispatch. Compare to
    # the halo path in test_aimnet2_real.py Case 2, which tolerates 1e-2.
    torch.testing.assert_close(forces_dist, forces_ref_owned, rtol=1e-12, atol=1e-14)


def test_gather_synthetic_aimnet2_2ranks() -> None:
    mp.spawn(_worker, args=(2, _run_gather_synthetic), nprocs=2)


def _fixed_a2a_worker(rank: int, world_size: int, queue, cap: int) -> None:
    from torch.distributed.device_mesh import init_device_mesh

    from nvalchemi.distributed._core.gather_primitives import funcol_all_to_all_fixed

    mesh = init_device_mesh("cpu", (world_size,))
    feat = 2
    # Block destined for rank d is filled with the tag (rank*100 + d).
    send = torch.empty(world_size * cap, feat)
    for d in range(world_size):
        send[d * cap : (d + 1) * cap] = float(rank * 100 + d)

    out = funcol_all_to_all_fixed(send, world_size, mesh)

    # Block received from source i must carry tag (i*100 + rank).
    ok = True
    for i in range(world_size):
        expected = float(i * 100 + rank)
        block = out[i * cap : (i + 1) * cap]
        if not torch.allclose(block, torch.full((cap, feat), expected)):
            ok = False
    queue.put((rank, ok))


def test_fixed_a2a_roundtrip_2ranks() -> None:
    results = run_gloo(world_size=2, fn=_fixed_a2a_worker, args=(3,))
    assert len(results) == 2
    assert all(ok for _rank, ok in results), results


def test_fixed_a2a_roundtrip_4ranks() -> None:
    results = run_gloo(world_size=4, fn=_fixed_a2a_worker, args=(2,))
    assert len(results) == 4
    assert all(ok for _rank, ok in results), results


def _fixed_index_select_worker(rank: int, world_size: int, queue) -> None:
    """Compare the fullgraph fixed-size gather against the trusted variable
    ``distributed_index_select`` AND the analytic expectation."""
    import types

    import torch
    from torch.distributed.device_mesh import init_device_mesh

    from nvalchemi.distributed._core.gather_primitives import (
        ShardRouting,
        distributed_index_select,
        funcol_fixed_index_select,
    )

    n_global = 8
    feat = 3
    assignment = torch.arange(n_global) % world_size  # round-robin ownership
    meta = ShardRouting.from_assignment(assignment, rank=rank)
    mesh = init_device_mesh("cpu", (world_size,))
    config = types.SimpleNamespace(mesh=mesh)

    # Owned rows store their own global id (so a gather of index g returns g).
    owned_globals = torch.where(assignment == rank)[0]
    sharded_input = torch.zeros(owned_globals.shape[0], feat)
    for g in owned_globals.tolist():
        sharded_input[int(meta.local_index[g].item())] = float(g)

    global_indices = torch.arange(n_global)  # every rank requests all atoms
    expected = global_indices.float().unsqueeze(1).repeat(1, feat)

    owner = meta.owner_rank[global_indices]
    cap = int(owner.bincount(minlength=world_size).max().item())

    out_fixed = funcol_fixed_index_select(
        sharded_input,
        global_indices,
        meta.owner_rank,
        meta.local_index,
        cap,
        world_size,
        mesh,
    )
    out_var = distributed_index_select(sharded_input, global_indices, meta, config)

    ok = torch.allclose(out_fixed, expected) and torch.allclose(out_fixed, out_var)
    queue.put((rank, ok))


def test_fixed_index_select_matches_variable_2ranks() -> None:
    results = run_gloo(world_size=2, fn=_fixed_index_select_worker)
    assert len(results) == 2
    assert all(ok for _rank, ok in results), results


def test_fixed_index_select_matches_variable_4ranks() -> None:
    results = run_gloo(world_size=4, fn=_fixed_index_select_worker)
    assert len(results) == 4
    assert all(ok for _rank, ok in results), results


def _fixed_autograd_worker(rank: int, world_size: int, queue) -> None:
    """Fixed gather path matches the trusted variable path on BOTH value and
    gradient (validates ``_FixedDistributedIndexSelect`` forward + backward)."""
    import types

    import torch
    from torch.distributed.device_mesh import init_device_mesh

    from nvalchemi.distributed._core.gather_primitives import (
        ShardRouting,
        distributed_index_select,
    )

    n_global, feat = 8, 3
    assignment = torch.arange(n_global) % world_size
    meta = ShardRouting.from_assignment(assignment, rank=rank)
    mesh = init_device_mesh("cpu", (world_size,))
    config = types.SimpleNamespace(mesh=mesh)

    torch.manual_seed(100 + rank)
    base = torch.randn(int((assignment == rank).sum()), feat, dtype=torch.float64)
    gi = torch.arange(n_global)
    cap = int(meta.owner_rank[gi].bincount(minlength=world_size).max().item())

    x_var = base.clone().requires_grad_(True)
    out_var = distributed_index_select(x_var, gi, meta, config)  # variable path
    (g_var,) = torch.autograd.grad(out_var.pow(2).sum(), x_var)

    x_fix = base.clone().requires_grad_(True)
    out_fix = distributed_index_select(x_fix, gi, meta, config, cap=cap)  # fixed path
    (g_fix,) = torch.autograd.grad(out_fix.pow(2).sum(), x_fix)

    ok = torch.allclose(out_var, out_fix) and torch.allclose(g_var, g_fix)
    queue.put((rank, ok))


def test_fixed_index_select_autograd_matches_variable_2ranks() -> None:
    results = run_gloo(world_size=2, fn=_fixed_autograd_worker)
    assert len(results) == 2
    assert all(ok for _rank, ok in results), results


def test_fixed_index_select_autograd_matches_variable_4ranks() -> None:
    results = run_gloo(world_size=4, fn=_fixed_autograd_worker)
    assert len(results) == 4
    assert all(ok for _rank, ok in results), results


def _fixed_scatter_autograd_worker(rank: int, world_size: int, queue) -> None:
    """Fixed scatter-add path matches the variable path on value + gradient
    (validates ``_FixedDistributedScatterAdd`` forward + backward directly)."""
    import types

    import torch
    from torch.distributed.device_mesh import init_device_mesh

    from nvalchemi.distributed._core.gather_primitives import (
        ShardRouting,
        distributed_scatter_add,
    )

    n_global, feat = 8, 3
    assignment = torch.arange(n_global) % world_size
    meta = ShardRouting.from_assignment(assignment, rank=rank)
    mesh = init_device_mesh("cpu", (world_size,))
    config = types.SimpleNamespace(mesh=mesh)
    n_owned = int((assignment == rank).sum())

    torch.manual_seed(200 + rank)
    gi = torch.arange(n_global)
    src_base = torch.randn(n_global, feat, dtype=torch.float64)
    cap = int(meta.owner_rank[gi].bincount(minlength=world_size).max().item())

    s_var = src_base.clone().requires_grad_(True)
    out_var = distributed_scatter_add(
        torch.zeros(n_owned, feat, dtype=torch.float64), gi, s_var, meta, config
    )
    (g_var,) = torch.autograd.grad(out_var.pow(2).sum(), s_var)

    s_fix = src_base.clone().requires_grad_(True)
    out_fix = distributed_scatter_add(
        torch.zeros(n_owned, feat, dtype=torch.float64),
        gi,
        s_fix,
        meta,
        config,
        cap=cap,
    )
    (g_fix,) = torch.autograd.grad(out_fix.pow(2).sum(), s_fix)

    ok = torch.allclose(out_var, out_fix) and torch.allclose(g_var, g_fix)
    queue.put((rank, ok))


def test_fixed_scatter_add_autograd_matches_variable_2ranks() -> None:
    results = run_gloo(world_size=2, fn=_fixed_scatter_autograd_worker)
    assert len(results) == 2
    assert all(ok for _rank, ok in results), results


pytestmark = pytest.mark.usefixtures("_session_gloo_pg")


def _cpu_mesh_config() -> types.SimpleNamespace:
    """Minimal real config: per_system_reduce only reads ``config.mesh``."""
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cpu", (1,))
    return types.SimpleNamespace(mesh=mesh)


def test_per_system_reduce_compiles_fwd_bwd() -> None:
    """``per_system_reduce`` (setup_context + funcol) traces under compile."""
    config = _cpu_mesh_config()
    n_systems = 3
    system_index = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)

    def fn(local_vals: torch.Tensor) -> torch.Tensor:
        out = per_system_reduce(local_vals, system_index, n_systems, config)
        return out.pow(2).sum()

    local_vals = torch.randn(6, 2, dtype=torch.float64, requires_grad=True)

    # Eager reference
    eager_grad = torch.autograd.grad(fn(local_vals), local_vals)[0]

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="aot_eager")
    local_vals_c = local_vals.detach().clone().requires_grad_(True)
    compiled_grad = torch.autograd.grad(compiled(local_vals_c), local_vals_c)[0]

    torch.testing.assert_close(eager_grad, compiled_grad, atol=1e-10, rtol=1e-10)


def test_distributed_all_reduce_compiles_fwd_bwd() -> None:
    """``distributed_all_reduce`` (setup_context + funcol) traces under compile."""
    config = _cpu_mesh_config()

    def fn(x: torch.Tensor) -> torch.Tensor:
        return distributed_all_reduce(x, config).pow(2).sum()

    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    eager_grad = torch.autograd.grad(fn(x), x)[0]

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="aot_eager")
    x_c = x.detach().clone().requires_grad_(True)
    compiled_grad = torch.autograd.grad(compiled(x_c), x_c)[0]

    torch.testing.assert_close(eager_grad, compiled_grad, atol=1e-10, rtol=1e-10)


def test_funcol_all_to_all_v_rows_compiles() -> None:
    """The funcol ``all_to_all_single`` row helper (powering the halo exchange)
    traces under fullgraph compile. Non-autograd by design — the halo
    ``autograd.Function``s supply the adjoint — so this is a forward-only
    traceability probe. World-1: a degenerate (self-only) exchange, but the
    funcol op is still emitted into the AOT graph (the property under test)."""
    mesh = _cpu_mesh_config().mesh

    def fn(x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        return funcol_all_to_all_v_rows(x, [n], [n], mesh)

    x = torch.randn(4, 3, dtype=torch.float64)
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="aot_eager")
    torch.testing.assert_close(compiled(x), fn(x))


def test_funcol_all_to_all_fixed_compiles_fullgraph() -> None:
    """Fixed-size (uniform-split) all_to_all — the fullgraph workaround for the
    data-dependent sharded gather. Unlike the all-to-all-**v** helper, the split
    sizes here are graph constants (from the static leading shape), so this is
    the path that traces under fullgraph for the AIMNet2 gather."""
    mesh = _cpu_mesh_config().mesh
    world_size = 1  # world-1 under the session fixture

    def fn(x: torch.Tensor) -> torch.Tensor:
        return funcol_all_to_all_fixed(x, world_size, mesh).pow(2).sum()

    x = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
    eager = fn(x)
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="aot_eager")
    torch.testing.assert_close(compiled(x), eager)


def test_funcol_fixed_index_select_compiles_fullgraph() -> None:
    """The full static-bucketing distributed gather (partition + fixed-size
    all_to_all) traces under fullgraph — the AIMNet2 sharded-gather workaround.
    Owner/local maps are passed as plain tensors (not via the metadata object)
    so Dynamo never has to trace through a custom container."""
    mesh = _cpu_mesh_config().mesh
    world_size = 1
    n = 6
    meta = ShardRouting.from_assignment(torch.zeros(n, dtype=torch.long), rank=0)
    owner_rank, local_index = meta.owner_rank, meta.local_index
    global_indices = torch.tensor([0, 2, 4, 1, 5])
    cap = n

    def fn(x: torch.Tensor) -> torch.Tensor:
        return funcol_fixed_index_select(
            x, global_indices, owner_rank, local_index, cap, world_size, mesh
        )

    sharded = torch.arange(n, dtype=torch.float64).unsqueeze(1).repeat(1, 3)
    eager = fn(sharded)
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="aot_eager")
    torch.testing.assert_close(compiled(sharded), eager)
    # Sanity: gather of index g returns row g.
    torch.testing.assert_close(
        eager, global_indices.to(torch.float64).unsqueeze(1).repeat(1, 3)
    )


def test_fixed_index_select_function_compiles_fwd_bwd() -> None:
    """The production ``_FixedDistributedIndexSelect`` autograd Function (the
    AIMNet2 fullgraph gather) traces forward + backward under fullgraph compile,
    with gradients matching eager."""
    mesh = _cpu_mesh_config().mesh
    world_size = 1
    n = 6
    meta = ShardRouting.from_assignment(torch.zeros(n, dtype=torch.long), rank=0)
    owner_rank, local_index = meta.owner_rank, meta.local_index
    global_indices = torch.tensor([0, 2, 4, 1, 5, 3])
    cap = n

    def fn(x: torch.Tensor) -> torch.Tensor:
        out = _FixedDistributedIndexSelect.apply(
            x, global_indices, owner_rank, local_index, cap, world_size, mesh
        )
        return out.pow(2).sum()

    base = torch.randn(n, 3, dtype=torch.float64)
    x_e = base.clone().requires_grad_(True)
    eager_grad = torch.autograd.grad(fn(x_e), x_e)[0]

    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="aot_eager")
    x_c = base.clone().requires_grad_(True)
    compiled_grad = torch.autograd.grad(compiled(x_c), x_c)[0]
    torch.testing.assert_close(eager_grad, compiled_grad, atol=1e-10, rtol=1e-10)
