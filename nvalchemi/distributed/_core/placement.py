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

"""Placement & routing foundation for distributed row-sharded tensors.

A small, explicit vocabulary for "how is this field distributed across the
mesh." The domain-agnostic routing-data container (rows, ranks, global indices
— no chemistry):

* :class:`ShardRouting` — the global<->local index map for a field stored
  ``Shard(0)`` over a *permuted* row ordering (i.e. ownership is not a
  contiguous block, as with a spatial decomposition).

This is the routing **data**. The per-field *declaration* that selects a
placement and binds its routing + op behavior is a ``StoragePolicy``:
``PlainShard`` carries ``Shard(0)``; ``HaloStoragePolicy`` carries
``Shard(0)`` + a :class:`ShardRouting` + halo metadata.

The module is intentionally free of any partitioning *strategy*: the seam
:meth:`ShardRouting.from_assignment` consumes a rank->row assignment and never
cares how it was computed. The partitioner that produces the assignment
(spatial / contiguous-block / custom) is a higher-level strategy that lives
above this layer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = [
    "ShardRouting",
]


@dataclass
class ShardRouting:
    """Global<->local index map for a row-sharded field.

    Each rank owns a subset of the ``n_global`` rows; the two tables below let
    every rank translate a global row ID into the pair
    ``(owner_rank, local_index_on_owner)``. Both tables are replicated on every
    rank (O(n_global) memory — fine for up to millions of rows). Ownership need
    not be a contiguous block, so this also describes a spatially-decomposed
    field stored ``Shard(0)`` over a permuted ordering.

    Attributes
    ----------
    n_owned : int
        Number of rows owned by THIS rank.
    n_global : int
        Total number of rows in the sharded field (sum of ``n_owned`` across
        ranks).
    owner_rank : torch.Tensor
        Shape ``(n_global,)`` int64. ``owner_rank[g]`` is the rank that owns
        global row ``g``.
    local_index : torch.Tensor
        Shape ``(n_global,)`` int64. ``local_index[g]`` is row ``g``'s position
        within its owner rank's ``n_owned`` rows.
    """

    n_owned: int
    n_global: int
    owner_rank: torch.Tensor
    local_index: torch.Tensor
    # Global system/graph count, set by the harness. Carried here (not on
    # ShardTensor._n_systems -- that corrupts per-system reduce) so the
    # distributed mol_sum adapter can size its scatter-add agnostically.
    n_systems_global: int | None = None

    @classmethod
    def from_assignment(
        cls,
        assignment: torch.Tensor,
        rank: int,
        world_size: int | None = None,
    ) -> "ShardRouting":
        """Build routing from a global ``(n_global,)`` rank assignment.

        ``assignment[g] = r`` means global row ``g`` is owned by rank ``r``. The
        routing uses contiguous-per-rank local indices — i.e. the n-th row
        assigned to rank ``r`` lands at local index ``n-1`` on rank ``r``.

        This is the assignment-agnostic seam: it consumes a rank->row map and
        does not care how the map was produced (spatial decomposition,
        contiguous block, or any custom partitioner).

        ``world_size`` may be passed explicitly to skip a CPU-sync that would
        otherwise be needed to size the per-rank ``first_index`` table. Callers
        that already know it (e.g. from ``dist.get_world_size``) should pass it;
        otherwise it is derived from ``assignment.max()`` with one sync.
        """
        assignment = assignment.long()
        n_global = assignment.shape[0]
        device = assignment.device

        if world_size is None:
            # Single sync to size the per-rank table — far cheaper than a
            # per-row Python ``.item()`` loop.
            world_size = int(assignment.max().item()) + 1 if n_global > 0 else 1

        order = torch.argsort(assignment, stable=True)
        owners_sorted = assignment[order]

        # Vectorised first-index-per-rank: scatter_reduce(amin) on a
        # ``(world_size,)`` table replaces a per-row Python loop. For n rows x 2
        # ranks this drops the call's CUDA-host sync count from ~2*n to a
        # constant; at n=1715 that was ~3400 syncs per forward, dominating the
        # multi-rank step time.
        rng = torch.arange(n_global, device=device, dtype=torch.long)
        first_index_of_rank = torch.full(
            (world_size,), n_global, dtype=torch.long, device=device
        )
        if n_global > 0:
            first_index_of_rank.scatter_reduce_(
                0, owners_sorted, rng, reduce="amin", include_self=True
            )

        within_rank_position = rng - first_index_of_rank[owners_sorted]
        local_index = torch.empty(n_global, dtype=torch.long, device=device)
        if n_global > 0:
            local_index[order] = within_rank_position

        n_owned = int((assignment == rank).sum().item())

        return cls(
            n_owned=n_owned,
            n_global=n_global,
            owner_rank=assignment,
            local_index=local_index,
        )
