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
"""Lattice images offered to each neighbour rank pair.

A ghost shell wider than one domain reaches the same neighbour from both sides
of a periodic axis, and each side is a different lattice image holding
different atoms. Offering only one image drops in-cutoff pairs silently, so
these pin which images the halo config exposes. Pure geometry — no process
group, no exchange.
"""

from __future__ import annotations

import torch

from nvalchemi.distributed._core.halo_types import _compute_pbc_image_vectors
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.partitioner import SpatialPartitioner


def _partitioner(
    box_x: float, cutoff: float, ranks_x: int, pbc: torch.Tensor | None = None
) -> SpatialPartitioner:
    """Partitioner split ``ranks_x`` ways along x, without a real DeviceMesh."""
    cell = torch.diag(torch.tensor([box_x, 12.0, 12.0], dtype=torch.float64))
    config = DomainConfig(cutoff=cutoff, skin=0.5, grid_dims=(ranks_x, 1, 1))
    part = SpatialPartitioner.__new__(SpatialPartitioner)
    part.config = config
    part.cell_matrix = cell
    part.pbc = torch.ones(3, dtype=torch.bool) if pbc is None else pbc
    part.world_size = ranks_x
    part.cells_per_dim = config.grid_dims
    part.rank_grid = SpatialPartitioner.compute_rank_grid(part.cells_per_dim, ranks_x)
    part._span = part.neighbor_span()
    part._neighbor_ranks = part._compute_all_neighbor_ranks()
    part._inv_cell = torch.linalg.inv(cell)
    return part


def _x_coefficients(images: dict, pair: tuple[int, int]) -> set[int]:
    """Lattice coefficients along x offered for ``pair``."""
    return {int(image[0].item()) for image in images.get(pair, [])}


class TestNarrowGhost:
    """Span 1: only the outermost pair wraps, as before."""

    def setup_method(self):
        # 48 A over 4 ranks -> 12 A domains against a 5.5 A ghost: span 1.
        self.part = _partitioner(box_x=48.0, cutoff=5.0, ranks_x=4)
        self.images = _compute_pbc_image_vectors(self.part)

    def test_span_is_one(self):
        assert self.part.neighbor_span()[0] == 1

    def test_edge_pair_wraps(self):
        assert _x_coefficients(self.images, (3, 0)) == {-1}
        assert _x_coefficients(self.images, (0, 3)) == {1}

    def test_interior_pair_does_not_wrap(self):
        # Rank 1 reaches rank 2 directly; no lattice translation involved.
        assert _x_coefficients(self.images, (1, 2)) == set()


class TestWideGhost:
    """Span 2: a neighbour reachable both ways needs both images."""

    def setup_method(self):
        # 16 A over 4 ranks -> 4 A domains against a 5.5 A ghost: span 2.
        self.part = _partitioner(box_x=16.0, cutoff=5.0, ranks_x=4)
        self.images = _compute_pbc_image_vectors(self.part)

    def test_span_is_two(self):
        assert self.part.neighbor_span()[0] == 2

    def test_neighbour_reached_from_both_sides_offers_the_wrapped_image(self):
        """Rank 1 reaches rank 3 directly at +2 and around the box at -2.

        The wrapped side carries the atoms at the far end of the box; without
        it they never arrive and their interactions are dropped in silence.
        """
        assert _x_coefficients(self.images, (1, 3)) == {1}
        assert _x_coefficients(self.images, (3, 1)) == {-1}

    def test_edge_pair_still_wraps(self):
        assert _x_coefficients(self.images, (3, 0)) == {-1}
        assert _x_coefficients(self.images, (0, 3)) == {1}

    def test_no_image_is_the_identity(self):
        """An all-zero image would duplicate the direct copy."""
        for pair, offered in self.images.items():
            for image in offered:
                assert image.abs().sum() > 0, f"{pair} offers a zero translation"


class TestTwoRankPeriodic:
    """The ordinary 2-rank split, where both directions reach one neighbour."""

    def setup_method(self):
        self.part = _partitioner(box_x=24.0, cutoff=5.0, ranks_x=2)
        self.images = _compute_pbc_image_vectors(self.part)

    def test_each_direction_offers_one_wrap(self):
        assert _x_coefficients(self.images, (0, 1)) == {1}
        assert _x_coefficients(self.images, (1, 0)) == {-1}


class TestNonPeriodicAxis:
    """A non-periodic axis contributes no lattice translation."""

    def test_open_boundary_has_no_images(self):
        part = _partitioner(
            box_x=16.0, cutoff=5.0, ranks_x=4, pbc=torch.zeros(3, dtype=torch.bool)
        )
        assert _compute_pbc_image_vectors(part) == {}


class TestContractionRefreshesHaloTopology:
    """A barostat contraction must reach the live halo config.

    ``update_cell`` widens the partitioner's span when domains narrow. A halo
    config that copied the peer list and lattice images at construction keeps
    exchanging with the old peers, so atoms that only became reachable after the
    contraction are never sent and their interactions are dropped in silence.
    """

    def _config(self, box_x: float):
        from nvalchemi.distributed._core.halo_types import ParticleHaloConfig

        part = _partitioner(box_x=box_x, cutoff=5.0, ranks_x=4)
        return part, ParticleHaloConfig(ghost_width=5.5, partitioner=part, mesh=None)

    def test_contraction_widens_the_offered_images(self):
        part, halo = self._config(box_x=48.0)
        assert part.neighbor_span()[0] == 1
        assert _x_coefficients(halo._pbc_images, (1, 3)) == set()

        # 48 A over 4 ranks gives 12 A domains; 16 A gives 4 A against a 5.5 A
        # ghost, so rank 1 now reaches rank 3 the long way round the box.
        part.update_cell(
            torch.diag(torch.tensor([16.0, 12.0, 12.0], dtype=torch.float64))
        )
        assert part.neighbor_span()[0] == 2
        assert _x_coefficients(halo._pbc_images, (1, 3)) == {1}

    def test_contraction_widens_the_peer_set(self):
        part, halo = self._config(box_x=48.0)
        before = set(halo.neighbor_ranks)
        part.update_cell(
            torch.diag(torch.tensor([16.0, 12.0, 12.0], dtype=torch.float64))
        )
        assert set(halo.neighbor_ranks) >= before

    def test_topology_is_not_recomputed_without_a_change(self):
        """Derived per geometry, not per access."""
        part, halo = self._config(box_x=48.0)
        first = halo._pbc_images
        assert halo._pbc_images is first
