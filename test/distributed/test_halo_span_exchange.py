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
"""Halo exchange when the ghost shell is wider than one rank domain.

``SpatialPartitioner.neighbor_span`` widens the neighbour-rank set once
``ghost > box_axis / ranks_on_axis``, so a rank must then receive ghosts from
ranks *two or more* domains away. ``test_spatial_partitioner.py`` gates the
geometry; these tests gate the data movement, which is what actually feeds the
model. The condition is a ratio, so a small box reaches it at world=4 — no
large-scale run required.
"""

from __future__ import annotations

import torch
from _gloo_harness import run_gloo

# 4 ranks along x over a 16 A box -> 4.0 A domains, against a 5.5 A ghost
# (cutoff + skin). span = ceil(5.5 / 4.0) = 2: every rank needs its
# next-but-one neighbour, which the old +/-1 shell never asked for.
_WORLD = 4
_BOX_X = 16.0
_BOX_YZ = 12.0
_CUTOFF = 5.0
_SKIN = 0.5
_N_ATOMS = 96


def _positions() -> torch.Tensor:
    """Deterministic atoms filling the box, spread along the split axis."""
    gen = torch.Generator().manual_seed(0)
    pos = torch.rand(_N_ATOMS, 3, generator=gen, dtype=torch.float64)
    return pos * torch.tensor([_BOX_X, _BOX_YZ, _BOX_YZ], dtype=torch.float64)


def _cell() -> torch.Tensor:
    return torch.diag(
        torch.tensor([_BOX_X, _BOX_YZ, _BOX_YZ], dtype=torch.float64)
    ).unsqueeze(0)


def _reference_neighbor_counts() -> torch.Tensor:
    """Per-atom neighbour count under the minimum-image convention."""
    pos = _positions()
    box = torch.tensor([_BOX_X, _BOX_YZ, _BOX_YZ], dtype=torch.float64)
    delta = pos.unsqueeze(1) - pos.unsqueeze(0)
    delta -= box * torch.round(delta / box)
    dist = delta.norm(dim=-1)
    within = (dist < _CUTOFF) & (dist > 0)
    return within.sum(dim=1)


def _build_batch():
    from nvalchemi.data import AtomicData, Batch

    n = _N_ATOMS
    data = AtomicData(
        positions=_positions(),
        atomic_numbers=torch.full((n,), 18, dtype=torch.long),
        atomic_masses=torch.ones(n, dtype=torch.float64),
        cell=_cell(),
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    return Batch.from_data_list([data])


def _span_worker(rank: int, world_size: int, queue, *_args) -> None:
    """Halo-pad on a span>1 partition; report each owned atom's neighbour count."""
    from torch.distributed.device_mesh import init_device_mesh

    from nvalchemi.distributed._core.particle_halo import ParticleHaloConfig
    from nvalchemi.distributed.config import DomainConfig
    from nvalchemi.distributed.particle_halo import halo_exchange
    from nvalchemi.distributed.partitioner import SpatialPartitioner
    from nvalchemi.distributed.sharded_batch import ShardedBatch

    mesh = init_device_mesh("cpu", (world_size,))
    # Split all four ranks along x so the domain width is box_x / world.
    config = DomainConfig(
        cutoff=_CUTOFF, skin=_SKIN, mesh=mesh, grid_dims=(world_size, 1, 1)
    )
    full = _build_batch() if rank == 0 else None

    partitioner = SpatialPartitioner(
        config, _cell()[0], torch.ones(3, dtype=torch.bool)
    )
    sharded = ShardedBatch.from_batch(full, mesh=mesh, config=config, src=0)
    halo_config = ParticleHaloConfig(
        ghost_width=config.effective_ghost_width(),
        partitioner=partitioner,
        mesh=mesh,
    )
    halo_exchange(sharded, halo_config)

    padded = sharded.padded_batch
    meta = sharded.halo_meta
    n_owned = int(meta.n_owned)
    local = padded.positions.detach()
    box = torch.tensor([_BOX_X, _BOX_YZ, _BOX_YZ], dtype=local.dtype)

    # Neighbour count for each owned atom against everything this rank can see.
    delta = local[:n_owned].unsqueeze(1) - local.unsqueeze(0)
    delta -= box * torch.round(delta / box)
    dist = delta.norm(dim=-1)
    counts = ((dist < _CUTOFF) & (dist > 0)).sum(dim=1)

    # Plain Python only: a tensor on the queue is rebuilt from a shared-memory
    # fd that dies with the worker, so the parent fails to unpickle it.
    queue.put(
        (
            rank,
            {
                "span": list(partitioner.neighbor_span()),
                "n_owned": n_owned,
                "n_padded": int(local.shape[0]),
                "owned_positions": local[:n_owned].tolist(),
                "counts": counts.tolist(),
            },
        )
    )


def test_wide_ghost_halo_delivers_every_neighbor():
    """Every owned atom must see the same neighbours it sees single-process.

    A ghost shell reaching two domains away is only correct if the exchange
    actually pulls from those ranks; a +/-1 shell silently drops the atoms
    beyond it, which shows up here as a short neighbour count.
    """
    results = run_gloo(world_size=_WORLD, fn=_span_worker, timeout_sec=180.0)
    assert len(results) == _WORLD, f"expected {_WORLD} payloads, got {len(results)}"

    ref_counts = _reference_neighbor_counts()
    ref_pos = _positions()

    total_owned = 0
    for rank, payload in sorted(results, key=lambda r: r[0]):
        assert payload["span"][0] >= 2, f"rank {rank}: span collapsed to 1"
        assert 0 < payload["n_owned"] < _N_ATOMS, (
            f"rank {rank}: degenerate partition, owns {payload['n_owned']}"
        )
        total_owned += payload["n_owned"]

        # Map each owned atom back to its global index by position.
        for local_i, pos in enumerate(payload["owned_positions"]):
            match = (
                (ref_pos - torch.tensor(pos, dtype=ref_pos.dtype)).norm(dim=-1).argmin()
            )
            assert int(payload["counts"][local_i]) == int(ref_counts[match]), (
                f"rank {rank} owned atom {local_i} (global {int(match)}) sees "
                f"{int(payload['counts'][local_i])} neighbours, single-process "
                f"sees {int(ref_counts[match])} — the halo is short, so the "
                f"ghost shell did not reach every contributing rank"
            )

    assert total_owned == _N_ATOMS, (
        f"ranks own {total_owned} atoms in total, expected {_N_ATOMS}"
    )
