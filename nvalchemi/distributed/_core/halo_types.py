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

"""Halo-storage data types — leaf module.

Holds the dataclasses that ``particle_halo`` produces and that other
``_core/`` primitives consume. Lives at the leaf of the import graph
so neither :mod:`nvalchemi.distributed._core.particle_halo` nor
:mod:`nvalchemi.distributed._core.gather_primitives` need a
``TYPE_CHECKING`` guard to refer to each other — both import from here
directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch

__all__ = [
    "ParticleHaloConfig",
    "ParticleHaloMetadata",
    "GNNHaloMarkers",
]


@dataclass
class ParticleHaloConfig:
    """Configuration for particle-based halo exchange.

    Initialized once (from ``DomainConfig`` + ``SpatialPartitioner``)
    and reused across steps.

    Parameters
    ----------
    ghost_width : float
        Halo region width (typically ``cutoff + skin``).
    partitioner : SpatialPartitioner
        Spatial grid partitioner (provides cell, pbc, rank bounds).
        ``Any``-typed here because :class:`SpatialPartitioner` is
        defined in :mod:`nvalchemi.distributed.partitioner`.
    mesh : DeviceMesh
        1D device mesh for communication.
    """

    ghost_width: float
    partitioner: Any  # SpatialPartitioner at runtime
    mesh: Any  # DeviceMesh at runtime

    # Computed in __post_init__
    rank: int = field(init=False)
    neighbor_ranks: list[int] = field(init=False)
    _pbc_images: dict[tuple[int, int], list[torch.Tensor]] = field(
        init=False, repr=False
    )

    def __post_init__(self) -> None:
        try:
            self.rank = self.mesh.get_local_rank()
        except Exception:
            self.rank = 0

        self.neighbor_ranks = [
            r for r in self.partitioner.get_neighbor_ranks(self.rank) if r != self.rank
        ]
        self._pbc_images = _compute_pbc_image_vectors(self.partitioner)

    @property
    def pbc_shifts(
        self,
    ) -> Mapping[tuple[int, int], tuple[torch.Tensor, ...]]:
        """Materialize a compatibility snapshot of current Cartesian shifts.

        The returned mapping and its value sequences are immutable.
        """
        cell_matrix = self.partitioner.cell_matrix
        shifts = {
            key: tuple(
                image.to(device=cell_matrix.device, dtype=cell_matrix.dtype)
                @ cell_matrix
                for image in images
            )
            for key, images in self._pbc_images.items()
        }
        return MappingProxyType(shifts)


def _compute_pbc_image_vectors(
    partitioner: Any,  # SpatialPartitioner at runtime
) -> dict[tuple[int, int], list[torch.Tensor]]:
    """Precompute cell-independent lattice images for neighbor rank pairs.

    Returns ``{(sender, receiver): [image_1, image_2, ...]}`` where each
    ``(3,)`` image contains integer-valued fractional lattice coefficients.
    For a diagonal neighbor crossing D periodic boundaries there are
    ``2^D - 1`` independent images (all non-empty subsets of crossed dims).
    """
    images: dict[tuple[int, int], list[torch.Tensor]] = {}
    cell_matrix = partitioner.cell_matrix
    pbc = partitioner.pbc
    grid = partitioner.rank_grid

    total_ranks = grid[0] * grid[1] * grid[2]
    for sender_rank in range(total_ranks):
        sender_coords = partitioner.rank_to_grid_coords(sender_rank)
        for receiver_rank in partitioner.get_neighbor_ranks(sender_rank):
            receiver_coords = partitioner.rank_to_grid_coords(receiver_rank)

            per_dim_images: list[torch.Tensor] = []
            for dim in range(3):
                if not pbc[dim] or grid[dim] <= 1:
                    continue
                dim_image = torch.zeros(
                    3, device=cell_matrix.device, dtype=cell_matrix.dtype
                )
                if sender_coords[dim] == grid[dim] - 1 and receiver_coords[dim] == 0:
                    dim_image[dim] = -1
                elif sender_coords[dim] == 0 and receiver_coords[dim] == grid[dim] - 1:
                    dim_image[dim] = 1
                else:
                    continue
                per_dim_images.append(dim_image)

            if not per_dim_images:
                continue

            n = len(per_dim_images)
            combo_images: list[torch.Tensor] = []
            for mask in range(1, 1 << n):
                combo = torch.zeros(
                    3, device=cell_matrix.device, dtype=cell_matrix.dtype
                )
                for bit in range(n):
                    if mask & (1 << bit):
                        combo = combo + per_dim_images[bit]
                combo_images.append(combo)

            images[(sender_rank, receiver_rank)] = combo_images

    return images


@dataclass
class GNNHaloMarkers:
    """Routing metadata for autograd-aware feature exchange on a halo layout.

    Mirrors the routing encoded in :attr:`ParticleHaloMetadata.send_indices`
    but indexes into the *owned* tensor directly — PBC-shifted copies
    are collapsed back onto their source owned rows. Feature tensors
    are translation-invariant, so the same owned row's feature is sent
    for every PBC variant a neighbor rank needs.

    Attributes
    ----------
    send_indices_owned : list[torch.Tensor]
        ``send_indices_owned[r]`` gives owned-tensor indices whose
        features should be sent to rank ``r``. Length equals world
        size.
    """

    send_indices_owned: list[torch.Tensor]
    # Compile-path mirrors of send_indices_owned as int[] constants, precomputed
    # eagerly so the halo_forward marshaller can ride them under fake mode —
    # avoids both a fake-tensor .tolist (errors) and a real-Tensor graph constant
    # (inductor lowering rejects it).
    send_idx_flat: list[int] = field(default_factory=list)
    send_idx_lens: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.send_idx_lens:
            self.send_idx_lens = [int(t.numel()) for t in self.send_indices_owned]
        if not self.send_idx_flat:
            self.send_idx_flat = [
                int(v)
                for t in self.send_indices_owned
                for v in t.to(torch.int64).reshape(-1).tolist()
            ]


@dataclass
class ParticleHaloMetadata:
    """Ephemeral metadata from a ghost exchange, used for stripping and
    backward."""

    n_owned: int
    n_padded: int
    send_indices: list[torch.Tensor]
    send_sizes: list[list[int]]
    recv_sizes: list[list[int]]
    gnn_markers: GNNHaloMarkers | None = None
