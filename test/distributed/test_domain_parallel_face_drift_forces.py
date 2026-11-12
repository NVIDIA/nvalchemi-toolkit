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
"""End-to-end DD forces after one NVE step on a crystal with atoms on the faces.

An alpha-quartz supercell has atoms exactly on the periodic faces of its
hexagonal cell (fraction 0). ``DomainParallel`` leaves the split axes
unwrapped, so in the first velocity-Verlet step about half of those atoms move
to a fraction just below 0 while their owner keeps them until the next
migration. The forces of that first step must still agree with a single
process evaluated at the same positions; before the owner-centred ghost
selection, a receiver that shares the owner's grid coordinate on the drift axis
never received such an atom and its forces missed a neighbor.

Four gloo ranks on a 2 x 2 x 1 grid, Lennard-Jones with a 6 A cutoff, float64.
The cell is the one the benchmark SiO2 builder makes (ASE, spacegroup 152) and
the sites agree with it to 2e-4 A. Both put 49 atoms on each of the faces
a = 0 and b = 0, which are the split axes here. The builder's Si z of 0.3333
leaves its c = 0 sites at 0.9999; with the exact 1/3 used here they sit on
c = 0, but c is not split, so that face does not take part.

Each rank writes its owned positions and forces to a file. Passing them through
the result queue would deadlock: a spawned worker cannot exit until its queued
data has been read, and the harness reads the queue only after joining. Each
rank prints its progress to stderr, so a hang shows where it stopped.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch
from _gloo_harness import run_gloo
from torch.distributed import DeviceMesh

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.domain_parallel import DomainParallel
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.integrators.nve import NVE
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.neighbors import compute_neighbors

_WORLD = 4
_GRID = (2, 2, 1)  # two split axes, so some receivers share the owner's coordinate
_REPEATS = (7, 7, 7)
_CUTOFF = 6.0
_EPSILON = 0.01
_SIGMA = 1.5
_DT_FS = 1.0
_TEMPERATURE_K = 300.0
_KB_EV = 8.617333262e-5
# float64 DD and single-process forces differ only by summation order and by
# the lattice wrap of the reference positions, both near 1e-14 eV/A here. One
# missing neighbor at the 6 A cutoff changes a force by about 1e-5 eV/A, and
# the failure this guards against is about 1e-2 eV/A.
_ATOL = 1e-9


def _alpha_quartz(
    repeats: tuple[int, int, int], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Alpha-quartz supercell (P3_121), as in the repository's SiO2 benchmark.

    Returns positions, atomic numbers, masses and the cell. The Si sites
    (x, 0, 1/3), (0, x, 2/3) and (-x, -x, 0) put atoms exactly on the faces
    a = 0, b = 0 and c = 0.
    """
    ops = (
        lambda x, y, z: (x, y, z),
        lambda x, y, z: (-y, x - y, z + 1.0 / 3.0),
        lambda x, y, z: (-x + y, -x, z + 2.0 / 3.0),
        lambda x, y, z: (y, x, -z),
        lambda x, y, z: (x - y, -y, -z + 2.0 / 3.0),
        lambda x, y, z: (-x, -x + y, -z + 1.0 / 3.0),
    )
    basis = (
        (8, 15.999, (0.413, 0.2711, 0.2172)),
        (14, 28.085, (0.4673, 0.0, 1.0 / 3.0)),
    )
    unit: list[tuple[int, float, tuple[float, ...]]] = []
    for z_num, mass, site in basis:
        for op in ops:
            frac = tuple(v % 1.0 for v in op(*site))
            frac = tuple(0.0 if abs(v - 1.0) < 1e-12 else v for v in frac)
            if not any(
                max(abs(a - b) for a, b in zip(frac, f)) < 1e-6 for _, _, f in unit
            ):
                unit.append((z_num, mass, frac))
    assert len(unit) == 9

    a, c = 4.9019, 5.3988
    unit_cell = torch.tensor(
        [[a, 0.0, 0.0], [-0.5 * a, 0.5 * 3.0**0.5 * a, 0.0], [0.0, 0.0, c]],
        dtype=dtype,
    )
    reps = torch.tensor(repeats, dtype=dtype)
    frac_rows, numbers, masses = [], [], []
    for shift in itertools.product(*(range(n) for n in repeats)):
        for z_num, mass, frac in unit:
            frac_rows.append([(f + s) / n for f, s, n in zip(frac, shift, repeats)])
            numbers.append(z_num)
            masses.append(mass)
    cell = unit_cell * reps.unsqueeze(1)
    positions = torch.tensor(frac_rows, dtype=dtype) @ cell
    return (
        positions,
        torch.tensor(numbers, dtype=torch.long),
        torch.tensor(masses, dtype=dtype),
        cell,
    )


def _quartz_batch(positions: torch.Tensor | None = None) -> Batch:
    """The quartz supercell at 300 K; *positions* replaces the crystal sites."""
    dtype = torch.float64
    sites, numbers, masses, cell = _alpha_quartz(_REPEATS, dtype)
    generator = torch.Generator().manual_seed(0)
    v_std = torch.sqrt(_KB_EV * _TEMPERATURE_K / masses).unsqueeze(-1)
    velocities = v_std * torch.randn(sites.shape, dtype=dtype, generator=generator)
    velocities = velocities - velocities.mean(dim=0, keepdim=True)
    data = AtomicData(
        positions=sites if positions is None else positions,
        atomic_numbers=numbers,
        atomic_masses=masses,
        cell=cell.unsqueeze(0),
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    data.add_node_property("velocities", velocities)
    return Batch.from_data_list([data])


def _lj() -> LennardJonesModelWrapper:
    return LennardJonesModelWrapper(epsilon=_EPSILON, sigma=_SIGMA, cutoff=_CUTOFF)


def _progress(rank: int, message: str) -> None:
    print(f"[face-drift rank {rank}] {message}", file=sys.stderr, flush=True)


def _one_nve_step_worker(rank: int, world_size: int, queue: Any, out_dir: str) -> None:
    _progress(rank, "started")
    mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("domain",))
    model = _lj()
    integrator = NVE(
        model=model,
        dt=_DT_FS,
        n_steps=1,
        hooks=[
            NeighborListHook(
                config=model.model_config.neighbor_config,
                skin=0.0,
                stage=DynamicsStage.BEFORE_COMPUTE,
            )
        ],
    )
    dp = DomainParallel(
        dynamics=integrator,
        config=DomainConfig(cutoff=_CUTOFF, skin=0.0, mesh=mesh, grid_dims=_GRID),
    )
    # Eager neighbor build: this test is about halo contents, and four spawned
    # CPU workers compiling the same hook would race on the Inductor cache.
    eager_neighbor_build = NeighborListHook._build.__wrapped__
    with patch.object(NeighborListHook, "_build", eager_neighbor_build), dp:
        local = dp.partition(_quartz_batch() if rank == 0 else None)
        _progress(rank, f"partitioned, {local.positions.shape[0]} owned atoms")
        local, _ = dp.step(local)
        _progress(rank, "stepped")
        torch.save(
            {
                "positions": local.positions.detach().clone(),
                "forces": local.forces.detach().clone(),
            },
            Path(out_dir) / f"rank{rank}.pt",
        )
    _progress(rank, "saved")
    queue.put(rank)


def test_dd_forces_match_single_process_after_one_nve_step_on_quartz(
    tmp_path: Path,
) -> None:
    """Every owned force after step 1 matches a single process at that geometry."""
    ranks = run_gloo(
        world_size=_WORLD,
        fn=_one_nve_step_worker,
        args=(str(tmp_path),),
        timeout_sec=300.0,
    )
    assert sorted(ranks) == list(range(_WORLD))
    shards = [
        torch.load(tmp_path / f"rank{r}.pt", weights_only=True) for r in range(_WORLD)
    ]

    sites, _, _, cell = _alpha_quartz(_REPEATS, torch.float64)
    positions = torch.cat([s["positions"] for s in shards])
    dd_forces = torch.cat([s["forces"] for s in shards])
    assert positions.shape == sites.shape

    # The split axes are not wrapped, so owned atoms that crossed a periodic
    # face in this step sit outside [0, 1). Without them the test is vacuous.
    frac = positions @ torch.linalg.inv(cell)
    outside = ((frac < 0.0) | (frac >= 1.0)).any(dim=1)
    assert int(outside.sum()) > 0, "no owned atom crossed a periodic face"

    # Reference: one process, same positions (wrapped into the cell, which does
    # not change any force under periodic boundaries).
    reference_batch = _quartz_batch((frac - torch.floor(frac)) @ cell)
    model = _lj()
    compute_neighbors(reference_batch, config=model.model_config.neighbor_config)
    reference_forces = model(reference_batch)["forces"].detach()

    diff = (dd_forces - reference_forces).abs().max(dim=1).values
    worst = int(diff.argmax())
    print(
        f"[face-drift] {int(outside.sum())} owned atoms outside [0, 1), "
        f"max |dF| = {float(diff.max()):.3e} eV/A",
        file=sys.stderr,
        flush=True,
    )
    assert float(diff.max()) <= _ATOL, (
        f"DD forces differ from one process by max |dF| = {float(diff.max()):.3e} "
        f"eV/A on {int((diff > _ATOL).sum())} atoms "
        f"({int(outside.sum())} owned atoms outside [0, 1)); worst atom at "
        f"fraction {frac[worst].tolist()}"
    )
