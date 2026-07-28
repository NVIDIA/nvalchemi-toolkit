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
"""NVE energy-conservation regression for periodic neighbor-list wrapping."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVE
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks._utils import kinetic_energy_per_graph
from nvalchemi.hooks import NeighborListHook, WrapPeriodicHook
from nvalchemi.models.lj import LennardJonesModelWrapper

_EPSILON = 0.0104
_SIGMA = 3.40
_CUTOFF = 6.0
_SWITCH_WIDTH = 1.0
_N_SIDE = 4
_N_ATOMS = _N_SIDE**3
_TEMPERATURE = 300.0
_DT_FS = 2.0
_N_STEPS = 1000
_SKIN = 0.5
_KB_EV = 8.617333262e-5
_ARGON_MASS = 39.948
_R_MIN = 2 ** (1 / 6) * _SIGMA
_MAX_DRIFT_EV_PER_ATOM = 1e-5


def _make_argon_batch() -> Batch:
    """Build the deterministic periodic argon lattice from issue 151."""
    coordinates = torch.arange(_N_SIDE, dtype=torch.float32) * _R_MIN
    grid = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    positions = torch.stack([axis.flatten() for axis in grid], dim=-1)

    generator = torch.Generator().manual_seed(42)
    velocities = torch.randn(_N_ATOMS, 3, generator=generator)
    velocities *= (_KB_EV * _TEMPERATURE / _ARGON_MASS) ** 0.5
    velocities -= velocities.mean(dim=0, keepdim=True)

    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((_N_ATOMS,), 18, dtype=torch.long),
        atomic_masses=torch.full((_N_ATOMS,), _ARGON_MASS),
        forces=torch.zeros(_N_ATOMS, 3),
        energy=torch.zeros(1, 1),
        cell=torch.eye(3).unsqueeze(0) * (_N_SIDE * _R_MIN),
        pbc=torch.ones(1, 3, dtype=torch.bool),
        velocities=velocities,
    )
    return Batch.from_data_list([data])


def _total_energy(batch: Batch) -> float:
    """Return total potential plus kinetic energy in eV."""
    kinetic_energy = kinetic_energy_per_graph(
        batch.velocities,
        batch.atomic_masses,
        batch.batch_idx,
        batch.num_graphs,
    )
    return float(batch.energy.sum() + kinetic_energy.sum())


def _run_nve(skin: float) -> float:
    """Run the issue-151 trajectory and return maximum total-energy drift."""
    model = LennardJonesModelWrapper(
        epsilon=_EPSILON,
        sigma=_SIGMA,
        cutoff=_CUTOFF,
        switch_width=_SWITCH_WIDTH,
    )
    nve = NVE(model=model, dt=_DT_FS, n_steps=_N_STEPS)
    nve.register_hook(
        NeighborListHook(
            config=model.model_config.neighbor_config,
            skin=skin,
            stage=DynamicsStage.BEFORE_COMPUTE,
        ),
        stage=DynamicsStage.BEFORE_COMPUTE,
    )
    nve.register_hook(WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE))

    total_energies: list[float] = []

    def record_energy(ctx, stage) -> None:
        total_energies.append(_total_energy(ctx.batch))

    record_energy.stage = DynamicsStage.AFTER_STEP
    record_energy.frequency = 10
    nve.register_hook(record_energy)
    nve.run(_make_argon_batch())

    initial_energy = total_energies[0]
    return max(abs(energy - initial_energy) for energy in total_energies)


@pytest.mark.slow
def test_nve_energy_conserved_with_skin_and_periodic_wrapping() -> None:
    """Verlet-skin wrapping must conserve energy as well as full rebuilds."""
    control_drift = _run_nve(skin=0.0)
    skin_drift = _run_nve(skin=_SKIN)

    assert skin_drift / _N_ATOMS < _MAX_DRIFT_EV_PER_ATOM
    assert skin_drift <= 5 * max(control_drift, 1e-6)
