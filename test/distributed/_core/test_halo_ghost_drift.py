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

"""Ghost selection for an owned atom that drifted past a periodic face, on gloo.

``DomainParallel`` does not wrap positions along the split axes, so an owned
atom can sit a little outside its owner's domain on the far side of a periodic
face. Ghost selection must still offer that atom to every receiver whose
cutoff shell contains it. The receivers that share the owner's grid coordinate
on the drift axis get no lattice image on that axis, so they only see the atom
through the direct test, which has to be made against fractions canonicalized
around the owner's domain, not into [0, 1).

Four ranks on a cubic cell give a rank grid split along two axes. Rank 0 owns
two atoms, one 0.1 Å past the periodic face of the first split axis and one
0.1 Å inside it, both within the cutoff of the neighbor that shares rank 0's
coordinate on that axis. That neighbor must list both atoms as ghosts, in the
local mask and in the exchanged halo.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.distributed import DeviceMesh

from nvalchemi.distributed._core.particle_halo import (
    ParticleHaloConfig,
    _identify_ghosts_split,
    particle_halo_padding,
)
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.partitioner import SpatialPartitioner
from test.distributed._gloo_harness import run_gloo

_WORLD = 4
_CELL = 10.0
_CUTOFF = 1.0
_DRIFT = 0.1  # Å past the periodic face


def _grid_coords(partitioner: SpatialPartitioner, rank: int) -> tuple[int, ...]:
    return tuple(int(c) for c in partitioner.rank_to_grid_coords(rank))


def _drift_check(rank: int, world: int, queue: Any) -> None:
    dtype = torch.float64
    cell = torch.eye(3, dtype=dtype) * _CELL
    mesh = DeviceMesh("cpu", list(range(world)), mesh_dim_names=("domain",))
    domain = DomainConfig(cutoff=_CUTOFF, skin=0.0, mesh=mesh)
    partitioner = SpatialPartitioner(
        config=domain,
        cell_matrix=cell.unsqueeze(0),
        pbc=torch.ones(3, dtype=torch.bool).unsqueeze(0),
    )
    config = ParticleHaloConfig(
        ghost_width=domain.effective_ghost_width(),
        partitioner=partitioner,
        mesh=mesh,
    )

    split = [d for d, n in enumerate(partitioner.rank_grid) if n > 1]
    assert len(split) == 2, partitioner.rank_grid
    axis_a, axis_b = split  # drift axis, and the axis the receiver differs on
    owner = 0
    assert _grid_coords(partitioner, owner) == (0, 0, 0)
    want = tuple(1 if d == axis_b else 0 for d in range(3))
    receiver = next(r for r in range(world) if _grid_coords(partitioner, r) == want)

    # Two atoms owned by rank 0, both 0.5 Å from the receiver's face on axis b
    # and within the cutoff of its region on every axis: one drifted 0.1 Å past
    # the periodic face of axis a (fraction -0.01), one 0.1 Å inside it.
    frac = torch.full((2, 3), 0.5, dtype=dtype)
    frac[:, axis_b] = 0.45
    frac[0, axis_a] = -_DRIFT / _CELL
    frac[1, axis_a] = _DRIFT / _CELL
    owned = frac @ cell

    if rank == owner:
        local = owned
    else:
        lo, hi = partitioner.rank_to_cell_bounds(rank)
        centre = torch.tensor(
            [0.5 * (lo[d] + hi[d]) / partitioner.cells_per_dim[d] for d in range(3)],
            dtype=dtype,
        )
        local = (centre @ cell).unsqueeze(0)

    result: dict[str, Any] = {"rank": rank}
    if rank == owner:
        assert receiver in config.neighbor_ranks
        direct, images = _identify_ghosts_split(local, receiver, config)
        selected = direct.clone()
        for mask, _shift in images:
            selected |= mask
        result["selected"] = selected.tolist()

    padded, _meta = particle_halo_padding(local, config)
    ghosts = padded[local.shape[0] :]
    if rank == receiver:
        close = (ghosts - owned[0]).abs().max(dim=1).values < 1e-9
        result["drifted_received"] = bool(close.any())
        close = (ghosts - owned[1]).abs().max(dim=1).values < 1e-9
        result["inside_received"] = bool(close.any())
    queue.put(result)


def test_drifted_owned_atom_is_ghosted_to_same_coordinate_receiver() -> None:
    results = {r["rank"]: r for r in run_gloo(world_size=_WORLD, fn=_drift_check)}
    assert len(results) == _WORLD
    owner = results[0]
    assert owner["selected"][1], "the atom inside the face must be selected"
    assert owner["selected"][0], "the drifted atom was not selected"
    (receiver,) = [r for r in results.values() if "drifted_received" in r]
    assert receiver["inside_received"]
    assert receiver["drifted_received"], "the drifted atom did not arrive"
