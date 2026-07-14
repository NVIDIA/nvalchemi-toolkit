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
"""Phase 2.5(b): NVE energy-conservation gate for UMA.

Runs a short NVE trajectory with :class:`~nvalchemi.models.uma.UMAWrapper`
on a bcc Fe 2x2x2 supercell and asserts total energy drift stays below
1 meV/atom over the integration window. Validates that the
conservative-forces path through ``UMAWrapper.forward`` is actually
conservative end-to-end (adapt_input, predict_unit, adapt_output,
NVE integrator) — the single-GPU gate the user requested before
distributed work.

Skipped when the UMA checkpoint cannot be loaded (no HF access).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip(
    "fairchem.core", reason="fairchem-core not installed; skipping UMA tests"
)

from ase.build import bulk  # noqa: E402

from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.dynamics.hooks._utils import kinetic_energy_per_graph  # noqa: E402
from nvalchemi.dynamics.integrators.nve import NVE  # noqa: E402
from nvalchemi.models.uma import UMAWrapper  # noqa: E402

_CKPT = os.environ.get("NVALCHEMI_UMA_CKPT", "uma-s-1p1")
_DEVICE = os.environ.get("NVALCHEMI_UMA_DEVICE", "cuda")

# Drift budget — 1 meV/atom over the whole trajectory.
_DRIFT_THRESHOLD_EV_PER_ATOM = 1e-3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def predict_unit() -> Any:
    from fairchem.core.calculate import pretrained_mlip
    from huggingface_hub.errors import GatedRepoError

    try:
        return pretrained_mlip.get_predict_unit(_CKPT, device=_DEVICE)
    except GatedRepoError as e:
        pytest.skip(f"no HF access to UMA checkpoint {_CKPT}: {e}")
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"could not load UMA checkpoint {_CKPT}: {e}")


@pytest.fixture(scope="module")
def wrapper_omat(predict_unit) -> UMAWrapper:
    return UMAWrapper(predict_unit, task_name="omat")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bcc_fe_batch(device: str | torch.device, seed: int = 42) -> Batch:
    """bcc Fe 2x2x2 (16 atoms) with Maxwell-Boltzmann velocities at 300 K.

    Returns a batch ready for NVE — positions, atomic_numbers,
    atomic_masses, cell, pbc, velocities. Mass is Fe (55.845 amu); kBT
    at 300 K sets the velocity scale.
    """
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)
    n = len(atoms)

    positions = torch.as_tensor(
        np.asarray(atoms.positions), dtype=torch.float32, device=device
    )
    numbers = torch.as_tensor(
        np.asarray(atoms.get_atomic_numbers()), dtype=torch.long, device=device
    )
    masses = torch.full((n,), 55.845, dtype=torch.float32, device=device)
    cell = torch.as_tensor(
        np.asarray(atoms.cell.array), dtype=torch.float32, device=device
    ).unsqueeze(0)
    pbc = torch.ones(1, 3, dtype=torch.bool, device=device)

    # Maxwell-Boltzmann velocities at T=300 K.
    kB = 8.617333262e-5  # eV/K
    T = 300.0
    g = torch.Generator(device="cpu").manual_seed(seed)
    vel = torch.randn(n, 3, generator=g).to(device) * float((kB * T / 55.845) ** 0.5)
    vel -= vel.mean(dim=0)  # zero net momentum

    data = AtomicData(
        positions=positions,
        atomic_numbers=numbers,
        atomic_masses=masses,
        cell=cell,
        pbc=pbc,
        velocities=vel,
        forces=torch.zeros_like(positions),
        energy=torch.zeros(1, 1, device=device, dtype=torch.float32),
    )
    return Batch.from_data_list([data])


def _total_energy(batch: Batch) -> float:
    """Compute total energy (PE + KE) in eV as a python float."""
    pe = batch.energy.squeeze(-1).sum().item()
    ke = kinetic_energy_per_graph(
        batch.velocities,
        batch.atomic_masses,
        batch.batch_idx,
        batch.num_graphs,
    )
    return pe + ke.squeeze(-1).sum().item()


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestNVEStability:
    """NVE drift over a short trajectory must stay below 1 meV/atom."""

    def test_bcc_fe_300k(self, wrapper_omat):
        """1000-step NVE on bcc Fe 2x2x2 at 300 K — drift < 1 meV/atom."""
        n_steps = int(os.environ.get("NVALCHEMI_UMA_NVE_STEPS", 1000))
        dt_fs = float(os.environ.get("NVALCHEMI_UMA_NVE_DT_FS", 0.5))
        stride = max(1, n_steps // 10)

        batch = _make_bcc_fe_batch(_DEVICE)
        n_atoms = batch.num_nodes

        nve = NVE(wrapper_omat, dt=dt_fs)

        trajectory: list[tuple[int, float]] = []

        def _energy_probe(ctx, stage):
            if ctx.step_count % stride == 0 or ctx.step_count == n_steps:
                trajectory.append((ctx.step_count, _total_energy(ctx.batch)))

        # Register as an AFTER_STEP hook so we observe energies after the
        # full velocity-Verlet update at each landmark step.
        from nvalchemi.dynamics.base import DynamicsStage

        _energy_probe.stage = DynamicsStage.AFTER_STEP
        _energy_probe.frequency = 1
        nve.register_hook(_energy_probe)

        nve.run(batch, n_steps=n_steps)

        assert trajectory, "no energy samples recorded"
        e0 = trajectory[0][1]
        e_final = trajectory[-1][1]
        drift_per_atom = abs(e_final - e0) / n_atoms

        # Print for visibility — pytest shows this on failure only.
        print()
        print(
            f"NVE stability ({_CKPT}, bcc Fe 2x2x2, 300 K, {n_steps} steps @ {dt_fs} fs)"
        )
        print(f"  initial E_total = {e0:.6f} eV")
        print(f"  final   E_total = {e_final:.6f} eV")
        print(f"  drift   = {abs(e_final - e0) * 1e3:.4f} meV total")
        print(f"  drift/atom = {drift_per_atom * 1e3:.4f} meV/atom")

        assert drift_per_atom < _DRIFT_THRESHOLD_EV_PER_ATOM, (
            f"NVE drift {drift_per_atom * 1e3:.3f} meV/atom exceeds "
            f"1 meV/atom over {n_steps} steps"
        )
