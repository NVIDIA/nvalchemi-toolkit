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
"""Importable helpers for distributed tests (gloo harness).

These live outside ``conftest.py`` so subfolder test packages can import them
(``from _helpers import ...``) — pytest only auto-applies conftest *fixtures*
to subdirectories, not module-level classes."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

__all__ = ["_MockMesh", "_LocalShardSpec", "_LocalShardTensor", "make_gloo_sharded_batch"]


class _MockMesh:
    """Four-method DeviceMesh stand-in for gloo-harness tests."""

    def __init__(self, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size

    def get_local_rank(self) -> int:
        return self._rank

    def size(self, dim: int | None = None) -> int:
        return self._world_size

    def get_group(self) -> Any:
        return None


class _LocalShardSpec:
    """Minimal stand-in for ``ShardTensorSpec`` exposing the one method
    :class:`DistributedModel`'s sharded path reads:
    ``sharding_shapes()[0]`` — the ordered per-rank local shapes for the
    mesh's dim-0 sharding. ``DistributedModel`` derives ``rank_sizes``
    from ``[s[0] for s in ...]`` to drive ``_all_gather_v_rows``.
    """

    def __init__(self, sizes: list[int], trailing: tuple[int, ...]) -> None:
        self._sizes = sizes
        self._trailing = trailing

    def sharding_shapes(self) -> tuple[list[torch.Size], ...]:
        return ([torch.Size((s, *self._trailing)) for s in self._sizes],)


class _LocalShardTensor:
    """ShardTensor-shaped stand-in supporting the two methods
    :class:`DistributedModel` invokes on sharded atom fields.

    Parameters
    ----------
    local
        This rank's local slice of the sharded tensor.
    sizes
        Per-rank sizes (``len(sizes) == world_size``) so
        :meth:`full_tensor` can reassemble the global view via
        ``dist.all_gather`` with padding + trim.
    """

    def __init__(self, local: torch.Tensor, sizes: list[int]) -> None:
        self._local = local
        self._sizes = sizes
        self._spec = _LocalShardSpec(sizes, tuple(local.shape[1:]))

    def to_local(self) -> torch.Tensor:
        return self._local

    def full_tensor(self) -> torch.Tensor:
        if not dist.is_initialized():
            return self._local.clone()
        world_size = dist.get_world_size()
        max_size = max(self._sizes)
        shape = list(self._local.shape)
        shape[0] = max_size
        padded = torch.zeros(*shape, dtype=self._local.dtype, device=self._local.device)
        padded[: self._local.shape[0]] = self._local
        gathered = [torch.zeros_like(padded) for _ in range(world_size)]
        dist.all_gather(gathered, padded)
        pieces = [gathered[r][: self._sizes[r]] for r in range(world_size)]
        return torch.cat(pieces, dim=0)


def make_gloo_sharded_batch(
    mesh: _MockMesh,
    local_positions: torch.Tensor,
    local_numbers: torch.Tensor,
    local_masses: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    sizes: list[int],
    n_global: int,
):
    """Build a :class:`ShardedBatch` backed by :class:`_LocalShardTensor`.

    Suits gloo-harness tests that want to exercise the
    ``DistributedModel(sharded)`` contract without depending on real
    ``physicsnemo.ShardTensor`` CUDA paths.
    """
    from nvalchemi.distributed.sharded_batch import ShardedBatch

    atom_fields = {
        "positions": _LocalShardTensor(local_positions, sizes),
        "atomic_numbers": _LocalShardTensor(local_numbers, sizes),
        "atomic_masses": _LocalShardTensor(local_masses, sizes),
    }
    return ShardedBatch(
        mesh=mesh,
        atom_fields=atom_fields,
        cell=cell,
        pbc=pbc,
        n_global=n_global,
    )
