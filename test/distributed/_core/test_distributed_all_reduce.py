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

"""Tests for :func:`distributed_all_reduce`.

Single-rank paths are exercised as regular unit tests; the 2-rank
correctness tests use ``torch.multiprocessing.spawn`` + gloo so they
run on CPU with no GPU requirement. The primitive is the all-reduce
sibling of :func:`per_system_reduce` — used directly by the Ewald
staged bindings and the PME mesh reduction.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.distributed._core.gather_primitives import distributed_all_reduce

# ======================================================================
# Harness
# ======================================================================


@dataclass
class _FakeConfig:
    """Stand-in for ParticleHaloConfig — distributed_all_reduce only
    reads ``config.mesh`` for the process group."""

    mesh: Any = None


class _MockMesh:
    """Gloo default-group mesh wrapper — ``mesh_group()`` calls
    ``get_group()`` which returns ``dist.group.WORLD`` under our gloo
    init_process_group.
    """

    def get_group(self) -> Any:
        return dist.group.WORLD


def _init_gloo(rank: int, world_size: int, port: str = "29611") -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _worker(rank: int, world_size: int, test_fn: Any, *args: Any) -> None:
    _init_gloo(rank, world_size)
    try:
        test_fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


# ======================================================================
# Single-rank (no dist.init) — exercises the no-op branch.
# ======================================================================


class TestSingleRankPath:
    def test_returns_clone(self):
        """Input is not modified in place; output is a distinct tensor."""
        x = torch.tensor([1.0, 2.0, 3.0])
        out = distributed_all_reduce(x, _FakeConfig())
        assert torch.equal(out, x)
        assert out.data_ptr() != x.data_ptr()

    def test_forward_identity_when_no_dist(self):
        """With no process group, output equals input bit-for-bit."""
        x = torch.randn(4, 5, dtype=torch.float64)
        out = distributed_all_reduce(x, _FakeConfig())
        torch.testing.assert_close(out, x, atol=0.0, rtol=0.0)

    def test_backward_passes_grad_through(self):
        """Backward on a sum-reduced output produces ones on the input."""
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        out = distributed_all_reduce(x, _FakeConfig())
        out.sum().backward()
        torch.testing.assert_close(x.grad, torch.ones_like(x))

    def test_only_sum_is_wired(self):
        with pytest.raises(NotImplementedError, match="only SUM"):
            distributed_all_reduce(torch.zeros(3), _FakeConfig(), op=dist.ReduceOp.MAX)

    def test_preserves_dtype_and_shape(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.randn(2, 3, 4, dtype=dtype)
            out = distributed_all_reduce(x, _FakeConfig())
            assert out.dtype == dtype
            assert out.shape == x.shape


# ======================================================================
# 2-rank — the real distributed path.
# ======================================================================


def _test_forward_sums_across_ranks(rank: int, world_size: int) -> None:
    """Each rank contributes a unique tensor; output replicates the sum
    on every rank and every rank gets the same answer."""
    # Rank r contributes (r+1) * [1, 2, 3, 4].
    base = torch.tensor([1.0, 2.0, 3.0, 4.0])
    x = (rank + 1) * base
    config = _FakeConfig(mesh=_MockMesh())

    out = distributed_all_reduce(x, config)

    # Expected: sum over r in [0, world_size) of (r+1)*base = base * Σ(r+1).
    expected_scale = sum(r + 1 for r in range(world_size))
    expected = base * expected_scale
    torch.testing.assert_close(out, expected, atol=1e-6, rtol=0.0)


def _test_forward_does_not_mutate_input(rank: int, world_size: int) -> None:
    """The caller's tensor must not be touched — they may use it again."""
    x = torch.full((5,), float(rank + 1))
    x_before = x.clone()
    config = _FakeConfig(mesh=_MockMesh())

    _ = distributed_all_reduce(x, config)
    torch.testing.assert_close(x, x_before, atol=0.0, rtol=0.0)


def _test_backward_all_reduces_grad(rank: int, world_size: int) -> None:
    """With output replicated across ranks, a unit-grad downstream on
    every rank produces a ``world_size`` gradient after the backward
    all-reduce — identical on every rank."""
    # Build x as a leaf with the rank-varying values directly — using
    # ``ones(...) * (rank+1)`` after requires_grad=True creates a
    # non-leaf and ``.grad`` would stay None.
    x = torch.full((3,), float(rank + 1), requires_grad=True)
    config = _FakeConfig(mesh=_MockMesh())

    out = distributed_all_reduce(x, config)
    # Downstream ones on every rank: each rank contributes ones to the
    # output's incoming grad, so the input grad (after the backward
    # all-reduce) is world_size * ones.
    out.sum().backward()

    expected = torch.full_like(x, float(world_size))
    torch.testing.assert_close(x.grad, expected, atol=1e-6, rtol=0.0)


def _test_backward_differentiates_per_rank_downstream(
    rank: int, world_size: int
) -> None:
    """When the downstream loss depends on rank (each rank weights the
    output differently), the resulting input grad is the all-reduced
    sum of rank-specific gradients — replicated on every rank."""
    # Leaf tensor with rank-shifted values. See comment in
    # _test_backward_all_reduces_grad for why we can't do
    # ``arange(...) + rank`` after requires_grad_.
    x = (torch.arange(3, dtype=torch.float32) + rank).detach().requires_grad_(True)
    config = _FakeConfig(mesh=_MockMesh())

    out = distributed_all_reduce(x, config)
    # Rank-specific scalar loss: rank r contributes r*out.sum().
    loss = float(rank) * out.sum()
    loss.backward()

    # Each rank's contribution to out's grad is r*ones; all-reduced grad
    # on x is Σ_r r * ones = (world_size * (world_size - 1) / 2) * ones.
    expected_scale = sum(range(world_size))
    expected = torch.full_like(x, float(expected_scale))
    torch.testing.assert_close(x.grad, expected, atol=1e-6, rtol=0.0)


def _test_preserves_autograd_graph_through_multiple_stages(
    rank: int, world_size: int
) -> None:
    """Chain two all-reduces — mimicking PME's
    spread → all_reduce → solve_poisson → all_reduce → backward flow.
    Backward should land the correct gradient on the initial leaf."""
    x = torch.full((4,), float(rank + 1), requires_grad=True)
    config = _FakeConfig(mesh=_MockMesh())

    y = distributed_all_reduce(x, config)  # y = (Σr(r+1)) * ones_like(x)
    z = y * 2.0
    out = distributed_all_reduce(z, config)  # out = world_size * z
    out.sum().backward()

    # d(out.sum())/dy at rank r = world_size * 2 (after the second
    # backward all-reduce propagates the upstream grad across ranks,
    # then the first backward all-reduces a world_size*2 vector).
    # Final: grad_x = world_size * 2 (from 2nd all_reduce backward)
    #                 * world_size (from 1st all_reduce backward)
    #       = 2 * world_size**2  on every element.
    expected = torch.full_like(x, 2.0 * world_size * world_size)
    torch.testing.assert_close(x.grad, expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("world_size", [2, 4])
def test_forward_sums_across_ranks(world_size):
    mp.spawn(
        _worker, args=(world_size, _test_forward_sums_across_ranks), nprocs=world_size
    )


@pytest.mark.parametrize("world_size", [2])
def test_forward_does_not_mutate_input(world_size):
    mp.spawn(
        _worker,
        args=(world_size, _test_forward_does_not_mutate_input),
        nprocs=world_size,
    )


@pytest.mark.parametrize("world_size", [2, 4])
def test_backward_all_reduces_grad(world_size):
    mp.spawn(
        _worker, args=(world_size, _test_backward_all_reduces_grad), nprocs=world_size
    )


@pytest.mark.parametrize("world_size", [2, 3])
def test_backward_differentiates_per_rank_downstream(world_size):
    mp.spawn(
        _worker,
        args=(world_size, _test_backward_differentiates_per_rank_downstream),
        nprocs=world_size,
    )


@pytest.mark.parametrize("world_size", [2])
def test_preserves_autograd_graph_through_multiple_stages(world_size):
    mp.spawn(
        _worker,
        args=(world_size, _test_preserves_autograd_graph_through_multiple_stages),
        nprocs=world_size,
    )
