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
"""MD state precision: state dtype = ``positions.dtype`` at the dynamics boundary.

float32 coordinates have an ulp of ~6e-5 A at |x| = 1000 A, so ``x += v*dt`` (~1e-3 A per
step) is quantised and the energy drifts. Building the data with ``dtype=torch.float64``
keeps positions and velocities in fp64; models compute in their own dtype, and
``BaseDynamics.compute`` casts the model's state outputs (forces, stress) to the state dtype.
"""

from __future__ import annotations

import pytest
import torch
from ase.build import bulk, molecule

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.dynamics.integrators.nve import NVE
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

SHIFT = 1000.0
F64 = torch.float64


def _water(shift: float = SHIFT):
    atoms = molecule("H2O")
    atoms.positions += shift + 1.0e-5  # sub-ulp(fp32) offset at 1000 A
    return atoms


def _argon(shift: float = SHIFT):
    atoms = bulk("Ar", "fcc", a=5.26, cubic=True).repeat(4)  # 21 A box, 256 atoms
    atoms.rattle(0.05, seed=3)
    atoms.positions += shift
    return atoms


def _batch(n_graphs: int = 2) -> Batch:
    gen = torch.Generator().manual_seed(1)
    data = [
        AtomicData(
            atomic_numbers=torch.tensor([18] * 4, dtype=torch.long),
            positions=torch.rand(4, 3, generator=gen, dtype=F64) + SHIFT,
            velocities=(torch.rand(4, 3, generator=gen, dtype=F64) - 0.5) * 0.02,
        )
        for _ in range(n_graphs)
    ]
    return Batch.from_data_list(data)


class _Fp32Model(torch.nn.Module, BaseModelMixin):
    """A model that computes in fp32 whatever the state dtype (zero forces/stress)."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}), supports_pbc=True
        )

    @property
    def embedding_shapes(self):
        return {}

    def compute_embeddings(self, data, **kwargs):
        return data

    def forward(self, batch):
        dev, m = batch.positions.device, batch.num_graphs
        return {
            "energy": torch.zeros((m, 1), dtype=F64, device=dev),
            "forces": torch.zeros(
                batch.positions.shape, dtype=torch.float32, device=dev
            ),
            "stress": torch.zeros((m, 3, 3), dtype=torch.float32, device=dev),
        }


class TestBuilders:
    def test_default_fp32_is_lossy_fp64_is_exact(self):
        atoms = _water()
        ref = torch.as_tensor(atoms.positions, dtype=F64)
        d32 = AtomicData.from_atoms(atoms)
        d64 = AtomicData.from_atoms(atoms, dtype=F64)
        assert d32.positions.dtype == torch.float32  # default unchanged
        assert (d32.positions.double() - ref).abs().max() > 1.0e-6
        assert torch.equal(d64.positions, ref)
        assert d64.velocities.dtype == F64
        assert d64.atomic_masses.dtype == F64

    def test_dtype_survives_batch_to_data_list_and_move(self):
        data = AtomicData.from_atoms(_water(), dtype=F64)
        batch = Batch.from_data_list([data, data])
        assert batch.positions.dtype == F64
        assert batch.velocities.dtype == F64
        back = batch.to_data_list()[0]
        assert torch.equal(back.positions, data.positions)
        assert data.to("cpu").positions.dtype == F64
        assert batch.to("cpu").positions.dtype == F64

    def test_from_structure(self):
        pytest.importorskip("pymatgen")
        from pymatgen.core import Molecule

        mol = Molecule(["H", "H"], [[SHIFT, 0.0, 0.0], [SHIFT + 0.74 + 1e-5, 0, 0]])
        on = AtomicData.from_structure(mol, dtype=F64)
        assert on.positions[1, 0].item() == SHIFT + 0.74 + 1e-5


class TestDrift:
    @staticmethod
    def _drift(dtype: torch.dtype, steps: int = 2000) -> float:
        from nvalchemi.dynamics._ops.velocity_verlet import vv_position_update

        n = 64
        gen = torch.Generator().manual_seed(0)
        x0 = torch.rand(n, 3, generator=gen, dtype=F64) + SHIFT
        v0 = (torch.rand(n, 3, generator=gen, dtype=F64) - 0.5) * 0.02
        dt = 0.5
        pos, vel = x0.to(dtype).clone(), v0.to(dtype).clone()
        frc = torch.zeros(n, 3, dtype=dtype)
        mass = torch.ones(n, dtype=dtype)
        dts = torch.full((1,), dt, dtype=dtype)
        batch = torch.zeros(n, dtype=torch.int32)
        for _ in range(steps):
            vv_position_update(pos, vel, frc, mass, dts, batch)
        return (pos.double() - (x0 + v0 * (dt * steps))).abs().max().item()

    def test_fp64_state_removes_the_drift(self):
        err32, err64 = self._drift(torch.float32), self._drift(F64)
        assert err32 > 1.0e-3  # accumulated rounding at |x| ~ 1000 A
        assert err64 < 1.0e-9
        assert err64 < err32 * 1.0e-4


class TestEngine:
    @pytest.mark.parametrize("integrator", ["nve", "langevin"])
    def test_steps_keep_fp64_with_an_fp32_model(self, integrator):
        batch = _batch()
        x0 = batch.positions.clone()
        model = DemoModelWrapper(DemoModel())  # computes in fp32
        if integrator == "nve":
            dyn = NVE(model=model, dt=0.5)
        else:
            dyn = NVTLangevin(
                model=model, dt=0.5, temperature=300.0, friction=0.1, random_seed=0
            )
        for _ in range(20):
            dyn.step(batch)
        assert batch.positions.dtype == F64
        assert batch.velocities.dtype == F64
        assert batch.forces.dtype == F64  # model output cast to the state dtype
        assert (batch.positions - x0).abs().max() > 0
        # sub-fp32-ulp information survives: positions are not fp32-representable
        assert not torch.equal(batch.positions, batch.positions.float().double())

    def test_existing_fp32_forces_buffer_is_promoted(self):
        batch = _batch(1)
        batch["forces"] = torch.zeros(4, 3)  # fp32 buffer supplied with the data
        dyn = NVE(model=DemoModelWrapper(DemoModel()), dt=0.5)
        for _ in range(3):
            dyn.step(batch)
        assert batch.forces.dtype == F64
        assert batch.positions.dtype == F64

    def test_periodic_npt_with_an_fp32_model(self):
        batch = Batch.from_data_list([AtomicData.from_atoms(_argon(), dtype=F64)])
        batch["stress"] = torch.zeros(1, 3, 3)  # fp32 stress buffer
        npt = NPT(
            model=_Fp32Model(),
            dt=1.0,
            temperature=300.0,
            pressure=1.0,
            barostat_time=100.0,
            thermostat_time=100.0,
            pressure_coupling="isotropic",
            chain_length=3,
        )
        for _ in range(5):
            npt.step(batch)
        assert batch.positions.dtype == F64
        assert batch.cell.dtype == F64
        assert batch.forces.dtype == F64
        assert batch.stress.dtype == F64
        assert torch.isfinite(batch.positions).all()
        assert torch.isfinite(batch.cell).all()

    def test_periodic_lj_nve(self):
        if not torch.cuda.is_available():
            pytest.skip("periodic LJ needs the CUDA neighbour kernels")
        from nvalchemi.dynamics.base import DynamicsStage
        from nvalchemi.hooks.neighbor_list import NeighborListHook
        from nvalchemi.models.lj import LennardJonesModelWrapper

        model = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.4, cutoff=8.5).cuda()
        hook = NeighborListHook(
            config=model.model_config.neighbor_config,
            skin=1.0,
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
        dyn = NVE(model=model, dt=1.0, hooks=[hook])
        batch = Batch.from_data_list(
            [AtomicData.from_atoms(_argon(), dtype=F64)], device="cuda"
        )
        for _ in range(10):
            dyn.step(batch)
        assert batch.positions.dtype == F64
        assert batch.velocities.dtype == F64
        assert batch.cell.dtype == F64
        assert torch.isfinite(batch.positions).all()

    def test_sinks_round_trip_fp64(self):
        from nvalchemi.dynamics.sinks import GPUBuffer, HostMemory

        batch = _batch()
        sinks = [HostMemory(capacity=4)]
        if torch.cuda.is_available():
            sinks.append(GPUBuffer(capacity=4, max_atoms=16, max_edges=0))
        for sink in sinks:
            sink.write(batch)
            out = sink.read()
            assert out.positions.dtype == F64
            assert out.velocities.dtype == F64
            assert torch.equal(out.positions.cpu(), batch.positions)
