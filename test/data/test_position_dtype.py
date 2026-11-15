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
"""Opt-in fp64 state (``precision_preserving_keys``) for large / off-origin systems.

float32 coordinates have an ulp of ~6e-5 A at |x| = 1000 A, so ``x += v*dt``
(~1e-3 A per step) is quantised and drifts. naming ``positions`` / ``velocities`` in ``AtomicData.precision_preserving_keys``
keeps them in fp64 (the rest follow ``fp_compute_dtype``); the default is unchanged.
"""

from __future__ import annotations

import pytest
import torch
from ase.build import molecule

from nvalchemi.data import AtomicData, Batch

SHIFT = 1000.0


@pytest.fixture
def fp64_state(monkeypatch):
    """Keep positions / velocities in fp64 (class-level opt-in)."""
    monkeypatch.setattr(
        AtomicData, "precision_preserving_keys", frozenset({"positions", "velocities"})
    )


def _shifted_water():
    atoms = molecule("H2O")
    atoms.positions += SHIFT + 1.0e-5  # sub-ulp(fp32) offset at 1000 A
    return atoms


class TestPreservedState:
    def test_default_unchanged(self):
        data = AtomicData.from_atoms(_shifted_water())
        assert data.positions.dtype == torch.float32
        assert data.velocities.dtype == torch.float32

    def test_off_loses_precision_on_keeps_it(self, monkeypatch):
        atoms = _shifted_water()
        ref = torch.as_tensor(atoms.positions, dtype=torch.float64)
        err32 = (AtomicData.from_atoms(atoms).positions.double() - ref).abs().max()
        monkeypatch.setattr(
            AtomicData, "precision_preserving_keys", frozenset({"positions"})
        )
        err64 = (AtomicData.from_atoms(atoms).positions - ref).abs().max()
        assert err32 > 1.0e-6  # fp32 quantises at ~6e-5 A
        assert err64 == 0.0

    def test_other_fields_use_compute_dtype(self, fp64_state):
        atoms = _shifted_water()
        atoms.set_cell([20.0, 20.0, 20.0])
        atoms.pbc = True
        data = AtomicData.from_atoms(atoms)
        assert data.positions.dtype == torch.float64
        assert data.velocities.dtype == torch.float64
        assert data.cell.dtype == torch.float32
        assert data.atomic_masses.dtype == torch.float64  # integrators need it

    def test_wider_label_is_cast_to_compute_dtype(self, fp64_state):
        data = AtomicData(
            atomic_numbers=torch.tensor([1, 1]),
            positions=torch.zeros(2, 3, dtype=torch.float64),
            forces=torch.zeros(2, 3, dtype=torch.float64),
        )
        assert data.positions.dtype == torch.float64
        assert data.forces.dtype == torch.float32

    def test_dtype_survives_batch_and_device_move(self, fp64_state):
        data = AtomicData.from_atoms(_shifted_water())
        batch = Batch.from_data_list([data, data])
        assert batch.positions.dtype == torch.float64
        assert batch.velocities.dtype == torch.float64
        assert batch.atomic_masses.dtype == torch.float64
        back = batch.to_data_list()[0]
        assert back.positions.dtype == torch.float64
        assert torch.equal(back.positions, data.positions)
        assert data.to("cpu").positions.dtype == torch.float64
        assert batch.to("cpu").positions.dtype == torch.float64

    def test_from_structure(self, fp64_state):
        pytest.importorskip("pymatgen")
        from pymatgen.core import Molecule

        mol = Molecule(["H", "H"], [[SHIFT, 0.0, 0.0], [SHIFT + 0.74 + 1e-5, 0, 0]])
        on = AtomicData.from_structure(mol)
        assert on.positions.dtype == torch.float64
        assert on.positions[1, 0].item() == SHIFT + 0.74 + 1e-5


class TestOffOriginDrift:
    """Free drift under the Velocity Verlet position update, far from the origin."""

    @staticmethod
    def _drift(dtype: torch.dtype, steps: int = 2000) -> float:
        from nvalchemi.dynamics._ops.velocity_verlet import vv_position_update

        n = 64
        gen = torch.Generator().manual_seed(0)
        x0 = torch.rand(n, 3, generator=gen, dtype=torch.float64) + SHIFT
        v0 = (torch.rand(n, 3, generator=gen, dtype=torch.float64) - 0.5) * 0.02
        dt = 0.5
        pos, vel = x0.to(dtype).clone(), v0.to(dtype).clone()
        frc = torch.zeros(n, 3, dtype=dtype)
        mass = torch.ones(n, dtype=dtype)
        dts = torch.full((1,), dt, dtype=dtype)
        batch = torch.zeros(n, dtype=torch.int32)
        for _ in range(steps):
            vv_position_update(pos, vel, frc, mass, dts, batch)
        exact = x0 + v0 * (dt * steps)
        return (pos.double() - exact).abs().max().item()

    def test_fp64_state_removes_the_drift(self):
        err32, err64 = self._drift(torch.float32), self._drift(torch.float64)
        assert err32 > 1.0e-3  # accumulated rounding at |x| ~ 1000 A
        assert err64 < 1.0e-9
        assert err64 < err32 * 1.0e-4


class TestEngineKeepsFp64:
    """The toolkit's own NVE integrator and sinks keep fp64 state at +1000 A."""

    @staticmethod
    def _batch() -> Batch:
        from nvalchemi.data import AtomicData

        gen = torch.Generator().manual_seed(1)
        data = []
        for _ in range(2):
            pos = torch.rand(4, 3, generator=gen, dtype=torch.float64) + SHIFT
            data.append(
                AtomicData(
                    atomic_numbers=torch.tensor([18] * 4, dtype=torch.long),
                    positions=pos,
                    velocities=(
                        torch.rand(4, 3, generator=gen, dtype=torch.float64) - 0.5
                    )
                    * 0.02,
                )
            )
        return Batch.from_data_list(data)

    @pytest.mark.parametrize("integrator", ["nve", "langevin"])
    def test_integrator_steps_keep_fp64(self, fp64_state, integrator):
        from nvalchemi.dynamics.integrators.nve import NVE
        from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        batch = self._batch()
        x0 = batch.positions.clone()
        model = DemoModelWrapper(DemoModel())
        if integrator == "nve":
            dyn = NVE(model=model, dt=0.5)
        else:
            dyn = NVTLangevin(
                model=model, dt=0.5, temperature=300.0, friction=0.1, random_seed=0
            )
        for _ in range(20):
            dyn.step(batch)
        assert batch.positions.dtype == torch.float64
        assert batch.velocities.dtype == torch.float64
        assert torch.isfinite(batch.positions).all()
        assert (batch.positions - x0).abs().max() > 0
        # sub-fp32-ulp information survives: positions are not fp32-representable
        assert not torch.equal(batch.positions, batch.positions.float().double())

    def test_sinks_round_trip_fp64(self, fp64_state):
        from nvalchemi.dynamics.sinks import GPUBuffer, HostMemory

        batch = self._batch()
        sinks = [HostMemory(capacity=4)]
        if torch.cuda.is_available():
            sinks.append(GPUBuffer(capacity=4, max_atoms=16, max_edges=0))
        for sink in sinks:
            sink.write(batch)
            out = sink.read()
            assert out.positions.dtype == torch.float64
            assert out.velocities.dtype == torch.float64
            assert torch.equal(out.positions.cpu(), batch.positions)
