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

import itertools
from dataclasses import dataclass, field
from typing import Any

import torch
from jaxtyping import Float

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

    # Computed on demand; see the topology properties below.
    rank: int = field(init=False)
    _topology: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            self.rank = self.mesh.get_local_rank()
        except Exception:
            self.rank = 0
        self._topology = {}

    def _current_topology(self) -> dict[str, Any]:
        """Peer list and lattice images for the partitioner's current geometry.

        Derived rather than stored: a barostat contraction narrows every domain,
        which can widen the ghost shell onto ranks further away, and a copy taken
        at construction would keep exchanging with the old peers. Recomputed only
        when the partitioner reports a new topology.
        """
        version = self.partitioner.topology_version
        if self._topology.get("version") != version:
            self._topology = {
                "version": version,
                "neighbor_ranks": [
                    r
                    for r in self.partitioner.get_neighbor_ranks(self.rank)
                    if r != self.rank
                ],
                "pbc_images": _compute_pbc_image_vectors(self.partitioner),
            }
        return self._topology

    @property
    def neighbor_ranks(self) -> list[int]:
        """Ranks this one exchanges ghosts with, for the current geometry."""
        return self._current_topology()["neighbor_ranks"]

    @property
    def _pbc_images(self) -> dict[tuple[int, int], list[torch.Tensor]]:
        """Lattice images offered per rank pair, for the current geometry."""
        return self._current_topology()["pbc_images"]

    @property
    def pbc_shifts(
        self,
    ) -> dict[tuple[int, int], list[Float[torch.Tensor, "3"]]]:
        """Materialize current Cartesian shifts from the cached lattice images."""
        cell_matrix = self.partitioner.cell_matrix
        return {
            key: list(
                (
                    torch.stack(images).to(
                        device=cell_matrix.device,
                        dtype=cell_matrix.dtype,
                    )
                    @ cell_matrix
                ).unbind(0)
            )
            for key, images in self._pbc_images.items()
        }


def _compute_pbc_image_vectors(
    partitioner: Any,  # SpatialPartitioner at runtime
) -> dict[tuple[int, int], list[Float[torch.Tensor, "3"]]]:
    """Precompute cell-independent lattice images for neighbor rank pairs.

    Returns ``{(sender, receiver): [image_1, image_2, ...]}`` where each
    ``(3,)`` image contains integer-valued fractional lattice coefficients.

    A ghost shell wider than one domain reaches the same neighbor from both
    sides of a periodic axis, and the two sides are different lattice images of
    that neighbor holding different atoms. The offsets the shell actually spans
    are therefore enumerated per axis rather than only the outermost pair; an
    offset that needs no wrap is the direct (unshifted) copy the caller handles
    separately, so only wrapping offsets appear here.
    """
    images: dict[tuple[int, int], list[torch.Tensor]] = {}
    cell_matrix = partitioner.cell_matrix
    pbc = partitioner.pbc
    grid = partitioner.rank_grid
    # No fallback: a partitioner without a span cannot describe how far its
    # ghost shell reaches, and silently assuming one domain is the very
    # under-send this function exists to prevent.
    span = partitioner.neighbor_span()

    total_ranks = grid[0] * grid[1] * grid[2]
    for sender_rank in range(total_ranks):
        sender_coords = partitioner.rank_to_grid_coords(sender_rank)
        for receiver_rank in partitioner.get_neighbor_ranks(sender_rank):
            receiver_coords = partitioner.rank_to_grid_coords(receiver_rank)

            # Per axis: which lattice translations reach this receiver at all.
            axis_choices: list[tuple[int, list[int]]] = []
            for dim in range(3):
                if not pbc[dim] or grid[dim] <= 1:
                    continue
                coefficients: set[int] = set()
                for offset in range(-int(span[dim]), int(span[dim]) + 1):
                    stepped = sender_coords[dim] + offset
                    if stepped % grid[dim] != receiver_coords[dim]:
                        continue
                    # Exact: ``stepped - receiver`` is a whole number of grids.
                    crossings = (stepped - receiver_coords[dim]) // grid[dim]
                    if crossings != 0:
                        coefficients.add(-crossings)
                if coefficients:
                    axis_choices.append((dim, sorted(coefficients)))

            if not axis_choices:
                continue

            # A neighbor across several axes needs every combination of their
            # translations; the all-zero one is the direct copy.
            combo_images: list[torch.Tensor] = []
            dims = [dim for dim, _ in axis_choices]
            for choice in itertools.product(*([0, *c] for _, c in axis_choices)):
                if not any(choice):
                    continue
                combo = torch.zeros(
                    3, device=cell_matrix.device, dtype=cell_matrix.dtype
                )
                for dim, coefficient in zip(dims, choice):
                    combo[dim] = coefficient
                combo_images.append(combo)

            if combo_images:
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
