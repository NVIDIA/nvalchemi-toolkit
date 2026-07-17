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

"""Unit tests for :func:`wrap_custom_op`'s ``owned_slice_inputs`` and
``all_reduce_outputs`` extensions.

These cover the primitive independently of Ewald / PME — a minimal
toy ``@torch.library.custom_op`` is wrapped, driven by a hand-rolled
halo-mode :class:`ShardTensor`, and the handler is verified to:

1. slice every input in ``owned_slice_inputs`` to its ``[:n_owned]``
   prefix before the kernel fires,
2. leave other (non-listed) inputs un-sliced,
3. all-reduce every output in ``all_reduce_outputs`` across the domain
   mesh (both forward and backward — the all-reduce primitive is
   autograd-aware).

Runs on CPU via gloo + ``torch.multiprocessing.spawn`` — the same
harness used by :mod:`test_distributed_all_reduce`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import Tensor

from nvalchemi.distributed._core.escape_hatches import wrap_custom_op
from nvalchemi.distributed._core.shard_tensor import ShardTensor

# ======================================================================
# Toy ops defined at module level — @torch.library.custom_op keeps
# the library object alive and registered for the module's lifetime,
# which the short-lived ``Library(...)`` API inside a function would
# not (Python garbage-collects the library → schema disappears).
# ======================================================================


@torch.library.custom_op("nvalchemi_test::identity_and_sum", mutates_args=())
def _identity_and_sum(x: Tensor) -> tuple[Tensor, Tensor]:
    """Return (x copy, scalar sum). Used to verify owned-slice on
    input 0 and all-reduce on output 1."""
    return x.clone(), x.sum().view(1)


@_identity_and_sum.register_fake
def _identity_and_sum_fake(x: Tensor) -> tuple[Tensor, Tensor]:
    return torch.empty_like(x), x.new_empty(1)


@torch.library.custom_op("nvalchemi_test::pair_sum", mutates_args=())
def _pair_sum(a: Tensor, b: Tensor) -> Tensor:
    """Return sum(a) * sum(b). Used to verify that only listed inputs
    get sliced — b is NOT in owned_slice_inputs so every element
    contributes."""
    return (a.sum() * b.sum()).view(1)


@_pair_sum.register_fake
def _pair_sum_fake(a: Tensor, b: Tensor) -> Tensor:
    return a.new_empty(1)


_IDENTITY_AND_SUM_OP = torch.ops.nvalchemi_test.identity_and_sum.default
_PAIR_SUM_OP = torch.ops.nvalchemi_test.pair_sum.default


# ======================================================================
# Harness — gloo + mp.spawn (same as test_distributed_all_reduce.py)
# ======================================================================


def _init_gloo(rank: int, world_size: int, port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def _worker(rank: int, world_size: int, test_fn: Any, *args: Any) -> None:
    _init_gloo(rank, world_size, port="29623")
    try:
        test_fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


class _MockMesh:
    """Gloo default-group mesh wrapper — matches the
    :func:`~nvalchemi.distributed._core.gather_primitives.mesh_group`
    contract."""

    def get_group(self) -> Any:
        return dist.group.WORLD


@dataclass
class _MiniHaloMeta:
    """Minimal :class:`ParticleHaloMetadata` surrogate — wrap_custom_op's
    handler only reads ``n_owned``; the other fields stay empty because
    no halo exchange fires in these tests."""

    n_owned: int
    n_padded: int


def _minimal_config() -> SimpleNamespace:
    """Build the minimal ParticleHaloConfig surrogate: just needs a
    ``mesh`` attribute exposing ``get_group()``."""
    return SimpleNamespace(mesh=_MockMesh())


def _wrap_as_halo_shardtensor(tensor: torch.Tensor, *, n_owned: int) -> ShardTensor:
    """Tag ``tensor`` as a halo-mode ShardTensor with minimal metadata.

    ``_find_source`` requires a non-None ``_spec`` for the tensor to
    be considered the handler's source; any halo-storage spec suffices
    — we use :data:`~nvalchemi.distributed.spec.SPEC_LJ_HALO` because
    it has zero escape-hatch fields and doesn't force any external
    imports.
    """
    # ShardTensor.wrap() routes through upstream's _make_wrapper_subclass and
    # needs a real DeviceMesh — the gloo workers initialize dist, so a 1-rank
    # CPU mesh is constructible here.
    from torch.distributed.device_mesh import DeviceMesh

    from nvalchemi.distributed.spec import SPEC_LJ_HALO

    if dist.is_initialized() and dist.get_world_size() == 1:
        mesh = DeviceMesh("cpu", [0], mesh_dim_names=("dom",))
    else:
        # Multi-rank gloo worker: build a mesh covering all ranks
        rank_list = list(range(dist.get_world_size()))
        mesh = DeviceMesh("cpu", rank_list, mesh_dim_names=("dom",))
    return ShardTensor.wrap(
        tensor,
        mesh=mesh,
        spec=SPEC_LJ_HALO,
        meta=_MiniHaloMeta(n_owned=n_owned, n_padded=tensor.shape[0]),
        config=_minimal_config(),
    )


# ======================================================================
# Workers
# ======================================================================
#
# Handlers are registered fresh inside each worker (via wrap_custom_op),
# cleared between tests via :func:`~nvalchemi.distributed._core.shard_tensor.clear_handlers`
# so runs don't pollute each other. Ops themselves live at module scope
# (see above).


def _reset_handlers_for(op: Any) -> None:
    """Clear any existing handler(s) on *op* so we can re-register cleanly."""
    from nvalchemi.distributed._core.shard_tensor import clear_handlers

    clear_handlers(op)
    packet = getattr(op, "_overloadpacket", None)
    if packet is not None and packet is not op:
        clear_handlers(packet)


def _test_owned_slice_inputs_single(rank: int, world_size: int) -> None:
    """Handler slices the input to [:n_owned] before the kernel fires;
    all_reduce on output 1 gives a globally-summed scalar."""
    op = _IDENTITY_AND_SUM_OP
    _reset_handlers_for(op)
    wrap_custom_op(op, owned_slice_inputs=(0,), all_reduce_outputs=(1,))

    # Each rank contributes a rank-specific tensor of length 6 where
    # [:4] is the "owned" region and [4:] is halo padding. Under the
    # handler the owned slice is [:4], and the sum should be summed
    # across ranks.
    n_owned = 4
    n_padded = 6
    owned_vals = torch.full((n_owned,), float(rank + 1))
    halo_vals = torch.full((n_padded - n_owned,), float(100 * (rank + 1)))
    full = torch.cat([owned_vals, halo_vals])
    sharded = _wrap_as_halo_shardtensor(full, n_owned=n_owned)

    identity_out, sum_out = op(sharded)

    # identity output: since we didn't register halo-correction on
    # output 0, the handler wraps the raw kernel output back as a
    # ShardTensor — its content is the kernel's computation on the
    # sliced input (length 4), NOT the original full (length 6). The
    # slice behavior is the whole point of ``owned_slice_inputs``.
    assert identity_out.shape == (n_owned,), (
        f"rank {rank}: identity output should be sliced to owned "
        f"length {n_owned}, got shape {identity_out.shape}"
    )
    torch.testing.assert_close(
        identity_out.as_subclass(torch.Tensor),
        owned_vals,
        atol=0.0,
        rtol=0.0,
    )

    # Sum output: each rank's local sum is (rank+1) * n_owned; after
    # all-reduce every rank sees Σ_r (r+1) * n_owned.
    expected_scale = sum(r + 1 for r in range(world_size))
    expected_sum = torch.tensor([float(expected_scale * n_owned)])
    torch.testing.assert_close(
        sum_out.as_subclass(torch.Tensor), expected_sum, atol=1e-6, rtol=0.0
    )


def _test_plain_tensor_pass_through(rank: int, world_size: int) -> None:
    """With plain (non-ShardTensor) inputs the handler is a transparent
    pass-through — no slice, no all-reduce."""
    op = _IDENTITY_AND_SUM_OP
    _reset_handlers_for(op)
    wrap_custom_op(op, owned_slice_inputs=(0,), all_reduce_outputs=(1,))

    x = torch.arange(6, dtype=torch.float32)
    identity_out, sum_out = op(x)

    # No slicing — full tensor round-trips unchanged.
    torch.testing.assert_close(identity_out, x, atol=0.0, rtol=0.0)
    # No all-reduce — sum is just this rank's local sum.
    torch.testing.assert_close(sum_out, torch.tensor([15.0]), atol=0.0, rtol=0.0)


def _test_non_listed_inputs_pass_through(rank: int, world_size: int) -> None:
    """Inputs NOT in ``owned_slice_inputs`` are not sliced — verify by
    wrapping an op that takes two tensors and only slicing the first."""
    op = _PAIR_SUM_OP
    _reset_handlers_for(op)
    # Slice only input 0; all-reduce output 0.
    wrap_custom_op(op, owned_slice_inputs=(0,), all_reduce_outputs=(0,))

    n_owned = 3
    a_full = torch.tensor(
        [1.0, 1.0, 1.0, 999.0, 999.0, 999.0]
    )  # slicing drops the 999s
    b_full = torch.tensor(
        [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
    )  # NOT sliced — all 2s contribute
    sharded_a = _wrap_as_halo_shardtensor(a_full, n_owned=n_owned)

    out = op(sharded_a, b_full)

    # Expected per-rank: sum(a[:3]) * sum(b) = 3 * 12 = 36
    # After all_reduce across world_size ranks: world_size * 36
    expected = torch.tensor([float(world_size * 36)])
    torch.testing.assert_close(
        out.as_subclass(torch.Tensor), expected, atol=1e-6, rtol=0.0
    )


# ======================================================================
# Pytest entry points
# ======================================================================
#
# Note on backward coverage: the all-reduce half of this extension
# delegates to
# :func:`~nvalchemi.distributed._core.gather_primitives.distributed_all_reduce`
# which is independently verified (forward + backward, single- and
# multi-rank, multi-stage autograd chain) in
# ``test_distributed_all_reduce.py``. A backward test at this layer
# would need to register an autograd formula on the toy
# ``@torch.library.custom_op`` (via ``register_autograd``) — out of
# scope for a primitive-level handler test. Compose the two
# guarantees to conclude backward correctness through the full
# ``wrap_custom_op(all_reduce_outputs=…)`` path.


@pytest.mark.parametrize("world_size", [2, 3])
def test_owned_slice_inputs_slices_before_kernel(world_size):
    mp.spawn(
        _worker, args=(world_size, _test_owned_slice_inputs_single), nprocs=world_size
    )


@pytest.mark.parametrize("world_size", [1, 2])
def test_plain_tensor_pass_through(world_size):
    mp.spawn(
        _worker, args=(world_size, _test_plain_tensor_pass_through), nprocs=world_size
    )


@pytest.mark.parametrize("world_size", [2])
def test_non_listed_inputs_pass_through(world_size):
    mp.spawn(
        _worker,
        args=(world_size, _test_non_listed_inputs_pass_through),
        nprocs=world_size,
    )
