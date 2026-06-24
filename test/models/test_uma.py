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
"""Tests for UMAWrapper (fairchem-core predict-unit wrapper).

Organised in two tiers:

* **Structural tests** (``Test*`` classes using ``_Mock*`` predict units)
  exercise ``adapt_input`` / ``adapt_output`` / forward composition,
  task-name validation, and model-config correctness — no checkpoint
  needed, fast, always run when ``fairchem-core`` is importable.
* **Checkpoint tests** load a real fairchem checkpoint (default
  ``uma-s-1p1``, override via ``NVALCHEMI_UMA_CKPT`` / ``NVALCHEMI_UMA_DEVICE``)
  and cover forward-equivalence vs ``FAIRChemCalculator``, charged-input
  response, NVE energy conservation (``@slow``), and the turbo /
  ``torch.compile`` device path (``@slow``, CUDA only). They skip
  cleanly when the gated checkpoint cannot be downloaded.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch
from torch import nn

pytest.importorskip(
    "fairchem.core", reason="fairchem-core not installed; skipping UMA tests"
)

from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402
from fairchem.core.datasets.atomic_data import AtomicData as FCAtomicData  # noqa: E402

from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.dynamics.hooks._utils import kinetic_energy_per_graph  # noqa: E402
from nvalchemi.dynamics.integrators.nve import NVE  # noqa: E402
from nvalchemi.models.base import NeighborListFormat  # noqa: E402
from nvalchemi.models.uma import _UMA_TASKS, UMAWrapper  # noqa: E402

_CKPT = os.environ.get("NVALCHEMI_UMA_CKPT", "uma-s-1p1")
_DEVICE = os.environ.get(
    "NVALCHEMI_UMA_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)


# ===========================================================================
# Structural tests — mock predict unit, no checkpoint
# ===========================================================================


class _MockInferenceSettings:
    base_precision_dtype = torch.float32
    external_graph_gen = False


class _MockBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.r_max = torch.tensor(6.0)
        self.sph_feature_size = 16  # (lmax+1)² with lmax=3
        self.sphere_channels = 128
        # A trainable weight so the train/freeze flag is observable in tests.
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, data: FCAtomicData) -> dict:
        n = data.pos.shape[0]
        return {
            "embedding": torch.zeros(
                n,
                self.sph_feature_size,
                self.sphere_channels,
                dtype=data.pos.dtype,
                device=data.pos.device,
            ),
            "batch": data.batch,
        }


class _MockInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _MockBackbone()


class _MockModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = _MockInnerModel()


class _MockPredictUnit:
    """Minimal stand-in for ``fairchem.core.units.mlip_unit.MLIPPredictUnit``.

    Records the ``FCAtomicData`` handed to ``predict`` so tests can
    inspect the tensor-native conversion. Returns a synthetic
    energy/forces dict matching the prediction schema.
    """

    def __init__(self, supported_tasks: list[str] | None = None) -> None:
        if supported_tasks is None:
            supported_tasks = list(_UMA_TASKS)
        self.dataset_to_tasks = {t: [] for t in supported_tasks}
        self.inference_settings = _MockInferenceSettings()
        self.model = _MockModelWrapper()
        self.last_data: FCAtomicData | None = None

    def predict(self, data: FCAtomicData, undo_element_references: bool = True) -> dict:
        self.last_data = data
        n_graphs = data.num_graphs
        n_atoms = data.pos.shape[0]
        # Differentiable energy = sum of per-atom position L2 norms,
        # grouped by graph. Lets autograd tests through (forces != 0).
        norms = data.pos.pow(2).sum(dim=-1).clamp(min=1e-8).sqrt()
        energy = torch.zeros(n_graphs, dtype=data.pos.dtype, device=data.pos.device)
        energy.scatter_add_(0, data.batch, norms)
        return {
            "energy": energy,
            "forces": torch.zeros(
                n_atoms, 3, dtype=data.pos.dtype, device=data.pos.device
            ),
        }


@pytest.fixture
def mock_pu() -> _MockPredictUnit:
    return _MockPredictUnit()


@pytest.fixture
def mock_omol(mock_pu) -> UMAWrapper:
    return UMAWrapper(mock_pu, task_name="omol")


@pytest.fixture
def mock_omat(mock_pu) -> UMAWrapper:
    return UMAWrapper(mock_pu, task_name="omat")


def _make_propane() -> AtomicData:
    """Propane C3H8 — 11 atoms, molecular (no PBC)."""
    positions = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [1.5260, 0.0000, 0.0000],
            [2.0330, 1.4360, 0.0000],
            [-0.5093, 1.0222, 0.0000],
            [-0.5093, -0.5111, 0.8853],
            [-0.5093, -0.5111, -0.8853],
            [2.0319, -0.5111, 0.8853],
            [2.0319, -0.5111, -0.8853],
            [3.1193, 1.4360, 0.0000],
            [1.6763, 1.9471, 0.8853],
            [1.6763, 1.9471, -0.8853],
        ],
        dtype=torch.float32,
    )
    numbers = torch.tensor([6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    return AtomicData(positions=positions, atomic_numbers=numbers)


def _make_periodic_cu() -> AtomicData:
    """Cubic Cu cell (1 atom per cell, a=3.615 Å) — minimal periodic system."""
    a = 3.615
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    numbers = torch.tensor([29], dtype=torch.long)
    cell = (torch.eye(3, dtype=torch.float32) * a).unsqueeze(0)
    pbc = torch.tensor([[True, True, True]])
    return AtomicData(positions=positions, atomic_numbers=numbers, cell=cell, pbc=pbc)


class TestConstruction:
    def test_invalid_task_name_raises(self, mock_pu):
        with pytest.raises(ValueError, match="task_name"):
            UMAWrapper(mock_pu, task_name="not_a_task")

    def test_unsupported_checkpoint_task_raises(self):
        # A predict unit that only ships the omol head.
        pu = _MockPredictUnit(supported_tasks=["omol"])
        with pytest.raises(ValueError, match="does not ship"):
            UMAWrapper(pu, task_name="omat")

    def test_task_stored(self, mock_omol):
        assert mock_omol.task_name == "omol"

    def test_cutoff_from_backbone(self, mock_omol):
        assert math.isclose(mock_omol.cutoff, 6.0, abs_tol=1e-6)

    def test_inference_freezes_weights(self, mock_omol):
        """train=False (default) freezes the underlying weights for inference."""
        params = list(mock_omol.predict_unit.model.parameters())
        assert params and all(not p.requires_grad for p in params)
        assert mock_omol.training is False

    def test_train_keeps_weights_trainable(self, mock_pu):
        """train=True leaves weights trainable/exposed for fine-tuning."""
        w = UMAWrapper(mock_pu, task_name="omol", train=True)
        assert any(p.requires_grad for p in w.predict_unit.model.parameters())
        assert w.training is True


class TestModelConfig:
    def test_omol_active_outputs(self, mock_omol):
        active = mock_omol.model_config.active_outputs
        assert "energy" in active and "forces" in active
        assert "stress" not in active  # molecular — no stress

    def test_omat_active_outputs(self, mock_omat):
        active = mock_omat.model_config.active_outputs
        assert active >= {"energy", "forces", "stress"}

    def test_needs_pbc_by_task(self, mock_omol, mock_omat):
        assert mock_omol.model_config.needs_pbc is False
        assert mock_omat.model_config.needs_pbc is True

    def test_supports_pbc_always(self, mock_omol, mock_omat):
        assert mock_omol.model_config.supports_pbc is True
        assert mock_omat.model_config.supports_pbc is True

    def test_neighbor_config(self, mock_omol):
        nc = mock_omol.model_config.neighbor_config
        assert nc.cutoff == mock_omol.cutoff
        assert nc.format is NeighborListFormat.COO
        assert nc.half_list is False

    def test_autograd_forces(self, mock_omol):
        assert "forces" in mock_omol.model_config.autograd_outputs
        assert "positions" in mock_omol.model_config.autograd_inputs

    def test_autograd_stress_for_periodic(self, mock_omat):
        assert "stress" in mock_omat.model_config.autograd_outputs


class TestAdaptInput:
    def test_molecular_single_system(self, mock_omol):
        batch = Batch.from_data_list([_make_propane()])
        fc = mock_omol.adapt_input(batch)

        assert isinstance(fc, FCAtomicData)
        assert fc.pos.shape == (11, 3)
        assert fc.atomic_numbers.shape == (11,)
        assert fc.natoms.tolist() == [11]
        assert fc.cell.shape == (1, 3, 3)
        assert fc.pbc.shape == (1, 3)
        assert fc.pbc.any().item() is False  # omol — no PBC
        assert fc.edge_index.shape == (2, 0)
        assert fc.nedges.tolist() == [0]
        assert fc.charge.tolist() == [0]  # default for omol
        assert fc.spin.tolist() == [1]  # OMol default multiplicity = singlet
        assert fc.dataset == ["omol"]

    def test_periodic_single_system(self, mock_omat):
        batch = Batch.from_data_list([_make_periodic_cu()])
        fc = mock_omat.adapt_input(batch)
        # Periodic heads ignore spin; default stays 0.
        assert fc.spin.tolist() == [0]

        assert fc.pos.shape == (1, 3)
        assert fc.cell.shape == (1, 3, 3)
        assert fc.pbc.all().item() is True
        assert fc.dataset == ["omat"]

    def test_multi_system_batch(self, mock_omol):
        batch = Batch.from_data_list([_make_propane(), _make_propane()])
        fc = mock_omol.adapt_input(batch)

        assert fc.pos.shape == (22, 3)
        assert fc.natoms.tolist() == [11, 11]
        assert fc.batch.tolist() == [0] * 11 + [1] * 11
        assert len(fc.dataset) == 2
        assert fc.dataset == ["omol", "omol"]

    def test_accepts_atomicdata_directly(self, mock_omol):
        fc = mock_omol.adapt_input(_make_propane())
        assert fc.num_graphs == 1
        assert fc.pos.shape == (11, 3)

    def test_preserves_device(self, mock_pu):
        """adapt_input must keep tensors on data.positions.device."""
        w = UMAWrapper(mock_pu, task_name="omol")
        data = _make_propane()
        # CPU-only test; device preservation is the invariant.
        fc = w.adapt_input(data)
        assert fc.pos.device == data.positions.device
        assert fc.atomic_numbers.device == data.positions.device
        assert fc.batch.device == data.positions.device

    def test_preserves_gradient_flow(self, mock_omol):
        """positions with requires_grad should flow into the FC AtomicData."""
        data = _make_propane()
        data.positions.requires_grad_(True)
        fc = mock_omol.adapt_input(data)
        # Tensor-native: pos is the same storage (dtype conversion is
        # identity for matching dtypes), so requires_grad carries.
        assert fc.pos.requires_grad

    def test_target_dtype_cast(self, mock_pu):
        """When predict_unit declares fp32, input fp64 positions cast down."""
        mock_pu.inference_settings.base_precision_dtype = torch.float32
        w = UMAWrapper(mock_pu, task_name="omol")
        data = AtomicData(
            positions=torch.randn(5, 3, dtype=torch.float64),
            atomic_numbers=torch.tensor([6, 1, 1, 1, 1], dtype=torch.long),
        )
        fc = w.adapt_input(data)
        assert fc.pos.dtype == torch.float32
        assert fc.cell.dtype == torch.float32

    def test_passes_explicit_charge_spin(self, mock_pu):
        w = UMAWrapper(mock_pu, task_name="omol")
        data = _make_propane()
        # Batch-level charge/spin should flow through.
        batch = Batch.from_data_list([data])
        batch.charge = torch.tensor([-1], dtype=torch.long)
        batch.spin = torch.tensor([2], dtype=torch.long)
        fc = w.adapt_input(batch)
        assert fc.charge.tolist() == [-1]
        assert fc.spin.tolist() == [2]

    def test_passes_tags(self, mock_pu):
        """Per-atom atom_categories (fairchem tags, e.g. OC20/ODAC) pass through."""
        w = UMAWrapper(mock_pu, task_name="oc20")
        data = AtomicData(
            positions=torch.zeros(4, 3),
            atomic_numbers=torch.tensor([1, 1, 1, 1], dtype=torch.long),
            atom_categories=torch.tensor([0, 1, 2, 1], dtype=torch.long),
        )
        fc = w.adapt_input(data)
        assert fc.tags.tolist() == [0, 1, 2, 1]

    def test_tags_default_zero(self, mock_omol):
        """Without atom_categories, the adapter fills tags with zeros."""
        fc = mock_omol.adapt_input(_make_propane())
        assert fc.tags.tolist() == [0] * 11


class TestAdaptOutput:
    def test_molecular_energy_forces(self, mock_omol):
        raw = {
            "energy": torch.tensor([1.5]),
            "forces": torch.zeros(5, 3),
        }
        out = mock_omol.adapt_output(raw)
        assert "energy" in out and "forces" in out
        # Ensure per-system 2D shape.
        assert out["energy"].shape == (1, 1)
        assert "stress" not in out

    def test_energy_already_2d(self, mock_omol):
        raw = {
            "energy": torch.tensor([[1.5]]),
            "forces": torch.zeros(5, 3),
        }
        out = mock_omol.adapt_output(raw)
        assert out["energy"].shape == (1, 1)

    def test_periodic_stress_shape(self, mock_omat):
        raw = {
            "energy": torch.tensor([1.5]),
            "forces": torch.zeros(1, 3),
            "stress": torch.zeros(1, 3, 3),
        }
        out = mock_omat.adapt_output(raw)
        assert out["stress"].shape == (1, 3, 3)

    def test_stress_flat_reshape(self, mock_omat):
        """fairchem sometimes returns stress flattened to (B, 9)."""
        raw = {
            "energy": torch.tensor([1.5]),
            "forces": torch.zeros(1, 3),
            "stress": torch.zeros(1, 9),
        }
        out = mock_omat.adapt_output(raw)
        assert out["stress"].shape == (1, 3, 3)


class TestForward:
    def test_composes(self, mock_omol):
        batch = Batch.from_data_list([_make_propane()])
        out = mock_omol(batch)
        assert "energy" in out and "forces" in out
        assert out["energy"].shape == (1, 1)
        assert out["forces"].shape == (11, 3)

    def test_records_input_at_predict(self, mock_omol, mock_pu):
        batch = Batch.from_data_list([_make_propane()])
        mock_omol(batch)
        assert mock_pu.last_data is not None
        assert isinstance(mock_pu.last_data, FCAtomicData)
        assert mock_pu.last_data.pos.shape == (11, 3)

    def test_batched(self, mock_omol):
        batch = Batch.from_data_list([_make_propane(), _make_propane()])
        out = mock_omol(batch)
        assert out["energy"].shape == (2, 1)
        assert out["forces"].shape == (22, 3)


# ===========================================================================
# Checkpoint tests — real fairchem checkpoint (skipped without HF access)
# ===========================================================================


@pytest.fixture(scope="module")
def predict_unit():
    """Load the UMA predict unit once for the module.

    Skips the dependent tests if HF access or download fails.
    """
    from fairchem.core.calculate import pretrained_mlip
    from huggingface_hub.errors import GatedRepoError

    try:
        return pretrained_mlip.get_predict_unit(_CKPT, device=_DEVICE)
    except GatedRepoError as e:
        pytest.skip(f"no HF access to UMA checkpoint {_CKPT}: {e}")
    except Exception as e:  # noqa: BLE001 — top-level guard for CI portability
        pytest.skip(f"could not load UMA checkpoint {_CKPT}: {e}")


@pytest.fixture(scope="module")
def calc_omol(predict_unit):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    return FAIRChemCalculator(predict_unit=predict_unit, task_name="omol")


@pytest.fixture(scope="module")
def calc_omat(predict_unit):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    return FAIRChemCalculator(predict_unit=predict_unit, task_name="omat")


@pytest.fixture(scope="module")
def wrapper_omol(predict_unit) -> UMAWrapper:
    return UMAWrapper(predict_unit, task_name="omol")


@pytest.fixture(scope="module")
def wrapper_omat(predict_unit) -> UMAWrapper:
    return UMAWrapper(predict_unit, task_name="omat")


def _propane_atoms() -> Atoms:
    """Propane C3H8 — OMol test system."""
    positions = np.array(
        [
            [0.0000, 0.0000, 0.0000],
            [1.5260, 0.0000, 0.0000],
            [2.0330, 1.4360, 0.0000],
            [-0.5093, 1.0222, 0.0000],
            [-0.5093, -0.5111, 0.8853],
            [-0.5093, -0.5111, -0.8853],
            [2.0319, -0.5111, 0.8853],
            [2.0319, -0.5111, -0.8853],
            [3.1193, 1.4360, 0.0000],
            [1.6763, 1.9471, 0.8853],
            [1.6763, 1.9471, -0.8853],
        ]
    )
    numbers = [6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]
    atoms = Atoms(numbers=numbers, positions=positions, pbc=False)
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return atoms


def _bcc_fe_atoms() -> Atoms:
    """bcc Fe, 2x2x2 supercell — OMat test system (16 atoms)."""
    return bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)


def _atomicdata_from_ase(atoms: Atoms) -> AtomicData:
    """Convert an ASE ``Atoms`` into our ``AtomicData`` (CPU tensors)."""
    pos = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float32)
    numbers = torch.as_tensor(np.asarray(atoms.get_atomic_numbers()), dtype=torch.long)
    kwargs: dict = {"positions": pos, "atomic_numbers": numbers}
    if np.any(atoms.pbc):
        kwargs["cell"] = torch.as_tensor(
            np.asarray(atoms.cell.array), dtype=torch.float32
        ).unsqueeze(0)
        kwargs["pbc"] = torch.as_tensor(
            np.asarray(atoms.pbc), dtype=torch.bool
        ).reshape(1, 3)
    return AtomicData(**kwargs)


def _bcc_fe_batch(device: str | torch.device, seed: int = 42) -> Batch:
    """bcc Fe 2x2x2 (16 atoms) on *device* with MB velocities at 300 K.

    Carries positions / atomic_numbers / atomic_masses / cell / pbc /
    velocities — ready for NVE and for the GPU-resident turbo forward.
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

    kB = 8.617333262e-5  # eV/K
    g = torch.Generator(device="cpu").manual_seed(seed)
    vel = torch.randn(n, 3, generator=g).to(device) * float(
        (kB * 300.0 / 55.845) ** 0.5
    )
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


# ---------------------------------------------------------------------------
# Forward equivalence vs FAIRChemCalculator
# ---------------------------------------------------------------------------


class TestOMolEquivalence:
    """Propane molecular energy/forces match ``FAIRChemCalculator``."""

    @pytest.fixture(autouse=True)
    def _setup(self, wrapper_omol, calc_omol):
        self.wrapper = wrapper_omol
        self.calc = calc_omol
        self.atoms = _propane_atoms()

    def _reference(self) -> dict[str, np.ndarray]:
        atoms = self.atoms.copy()
        atoms.info = dict(self.atoms.info)
        atoms.calc = self.calc
        return {
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
        }

    def _wrapper_result(self) -> dict[str, np.ndarray]:
        data = _atomicdata_from_ase(self.atoms)
        batch = Batch.from_data_list([data])
        batch.charge = torch.tensor([0], dtype=torch.long)
        batch.spin = torch.tensor([1], dtype=torch.long)
        out = self.wrapper(batch)
        return {
            "energy": float(out["energy"].detach().cpu().numpy().flatten()[0]),
            "forces": out["forces"].detach().cpu().numpy(),
        }

    def test_energy_matches(self):
        ref = self._reference()
        ours = self._wrapper_result()
        # fp32 precision — 1e-4 eV absolute covers round-trip jitter.
        assert np.isclose(ours["energy"], ref["energy"], atol=1e-4, rtol=1e-5), (
            f"energy mismatch: ours={ours['energy']:.6f} "
            f"ref={ref['energy']:.6f} diff={ours['energy'] - ref['energy']:.2e}"
        )

    def test_forces_match(self):
        ref = self._reference()
        ours = self._wrapper_result()
        assert ours["forces"].shape == ref["forces"].shape
        np.testing.assert_allclose(ours["forces"], ref["forces"], atol=1e-4, rtol=1e-4)


class TestOMatEquivalence:
    """bcc Fe 2x2x2 energy/forces/stress match ``FAIRChemCalculator``."""

    @pytest.fixture(autouse=True)
    def _setup(self, wrapper_omat, calc_omat):
        self.wrapper = wrapper_omat
        self.calc = calc_omat
        self.atoms = _bcc_fe_atoms()

    def _reference(self) -> dict[str, np.ndarray]:
        atoms = self.atoms.copy()
        atoms.calc = self.calc
        return {
            "energy": atoms.get_potential_energy(),
            "forces": atoms.get_forces(),
            "stress": atoms.get_stress(voigt=False),
        }

    def _wrapper_result(self) -> dict[str, np.ndarray]:
        data = _atomicdata_from_ase(self.atoms)
        batch = Batch.from_data_list([data])
        out = self.wrapper(batch)
        return {
            "energy": float(out["energy"].detach().cpu().numpy().flatten()[0]),
            "forces": out["forces"].detach().cpu().numpy(),
            "stress": out["stress"].detach().cpu().numpy()[0],
        }

    def test_energy_matches(self):
        ref = self._reference()
        ours = self._wrapper_result()
        assert np.isclose(ours["energy"], ref["energy"], atol=1e-4, rtol=1e-5), (
            f"energy mismatch: ours={ours['energy']:.6f} "
            f"ref={ref['energy']:.6f} diff={ours['energy'] - ref['energy']:.2e}"
        )

    def test_forces_match(self):
        ref = self._reference()
        ours = self._wrapper_result()
        np.testing.assert_allclose(ours["forces"], ref["forces"], atol=1e-4, rtol=1e-4)

    def test_stress_matches(self):
        ref = self._reference()
        ours = self._wrapper_result()
        # Reference is (3, 3); ours is (3, 3) after the adapt_output path.
        np.testing.assert_allclose(
            ours["stress"].reshape(3, 3),
            ref["stress"].reshape(3, 3),
            atol=1e-4,
            rtol=1e-4,
        )


# ---------------------------------------------------------------------------
# Charged inputs — total charge must propagate into the OMol head
# ---------------------------------------------------------------------------


class TestChargedInputs:
    """OMol energies respond to (and correctly use) the total-charge input."""

    def _wrapper_energy(self, wrapper, charge: int, spin: int) -> float:
        batch = Batch.from_data_list([_atomicdata_from_ase(_propane_atoms())])
        batch.charge = torch.tensor([charge], dtype=torch.long)
        batch.spin = torch.tensor([spin], dtype=torch.long)
        return float(wrapper(batch)["energy"].detach().cpu().flatten()[0])

    def test_charge_changes_energy(self, wrapper_omol):
        """Neutral singlet vs anion doublet — charge must alter the energy."""
        e_neutral = self._wrapper_energy(wrapper_omol, charge=0, spin=1)
        e_anion = self._wrapper_energy(wrapper_omol, charge=-1, spin=2)
        assert math.isfinite(e_neutral) and math.isfinite(e_anion)
        assert abs(e_anion - e_neutral) > 1e-3, (
            f"charge had no effect: neutral={e_neutral:.6f} anion={e_anion:.6f}"
        )

    def test_charged_matches_calculator(self, wrapper_omol, calc_omol):
        """A charged wrapper run matches ``FAIRChemCalculator`` with the same
        charge/spin — confirming charge is passed through correctly, not just
        that *something* changed."""
        atoms = _propane_atoms()
        atoms.info["charge"] = -1
        atoms.info["spin"] = 2
        atoms.calc = calc_omol
        ref = atoms.get_potential_energy()
        ours = self._wrapper_energy(wrapper_omol, charge=-1, spin=2)
        assert np.isclose(ours, ref, atol=1e-4, rtol=1e-5), (
            f"charged energy mismatch: ours={ours:.6f} ref={ref:.6f}"
        )


# ---------------------------------------------------------------------------
# NVE energy conservation (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestNVEStability:
    """NVE drift over a short trajectory must stay below 1 meV/atom."""

    def test_bcc_fe_300k(self, wrapper_omat):
        """1000-step NVE on bcc Fe 2x2x2 at 300 K — drift < 1 meV/atom."""
        from nvalchemi.dynamics.base import DynamicsStage

        n_steps = int(os.environ.get("NVALCHEMI_UMA_NVE_STEPS", 1000))
        dt_fs = float(os.environ.get("NVALCHEMI_UMA_NVE_DT_FS", 0.5))
        stride = max(1, n_steps // 10)
        threshold_ev_per_atom = 1e-3

        batch = _bcc_fe_batch(_DEVICE)
        n_atoms = batch.num_nodes
        nve = NVE(wrapper_omat, dt=dt_fs)

        trajectory: list[tuple[int, float]] = []

        def _total_energy(b: Batch) -> float:
            pe = b.energy.squeeze(-1).sum().item()
            ke = kinetic_energy_per_graph(
                b.velocities, b.atomic_masses, b.batch_idx, b.num_graphs
            )
            return pe + ke.squeeze(-1).sum().item()

        def _energy_probe(ctx, stage):
            if ctx.step_count % stride == 0 or ctx.step_count == n_steps:
                trajectory.append((ctx.step_count, _total_energy(ctx.batch)))

        _energy_probe.stage = DynamicsStage.AFTER_STEP
        _energy_probe.frequency = 1
        nve.register_hook(_energy_probe)

        nve.run(batch, n_steps=n_steps)

        assert trajectory, "no energy samples recorded"
        e0, e_final = trajectory[0][1], trajectory[-1][1]
        drift_per_atom = abs(e_final - e0) / n_atoms
        print(
            f"\nNVE stability ({_CKPT}, bcc Fe 2x2x2, 300 K, {n_steps} @ {dt_fs} fs): "
            f"drift {drift_per_atom * 1e3:.4f} meV/atom"
        )
        assert drift_per_atom < threshold_ev_per_atom, (
            f"NVE drift {drift_per_atom * 1e3:.3f} meV/atom exceeds "
            f"1 meV/atom over {n_steps} steps"
        )


# ---------------------------------------------------------------------------
# Turbo / torch.compile device path (slow, CUDA only)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="turbo/compile device-placement path is CUDA-specific",
)
class TestTurboCompile:
    """UMA under ``inference_settings="turbo"`` (compile + MoLE merge).

    Reproduces the original failing scenario — a GPU-resident first
    forward under turbo, which used to crash with a CPU/CUDA device
    mismatch (fairchem's lazy MoLE merge mis-placed the charge/spin
    embeddings). ``UMAWrapper.forward`` routes the one-time lazy-init
    call through CPU input, so the compiled model lands on the GPU.
    """

    @pytest.fixture(scope="class")
    def wrapper_turbo(self) -> UMAWrapper:
        from huggingface_hub.errors import GatedRepoError

        try:
            return UMAWrapper.from_checkpoint(
                _CKPT, task_name="omat", device="cuda", inference_settings="turbo"
            )
        except GatedRepoError as e:
            pytest.skip(f"no HF access to UMA checkpoint {_CKPT}: {e}")
        except Exception as e:  # noqa: BLE001 — top-level guard for CI portability
            pytest.skip(f"could not load UMA checkpoint {_CKPT}: {e}")

    def test_gpu_resident_first_forward(self, wrapper_turbo: UMAWrapper) -> None:
        """A GPU-resident first forward (lazy merge + compile) must run and
        return finite, on-device outputs — the case that used to crash."""
        out = wrapper_turbo(_bcc_fe_batch("cuda"))
        assert out["energy"].shape == (1, 1)
        assert out["forces"].shape == (16, 3)
        assert out["stress"].shape == (1, 3, 3)
        assert torch.isfinite(out["energy"]).all()
        assert torch.isfinite(out["forces"]).all()
        assert out["forces"].device.type == "cuda"

    def test_second_forward_after_init(self, wrapper_turbo: UMAWrapper) -> None:
        """After lazy init, a fresh GPU batch still runs on-device (CPU
        routing applies only to the first forward)."""
        out = wrapper_turbo(_bcc_fe_batch("cuda"))
        assert torch.isfinite(out["energy"]).all()
        assert out["forces"].device.type == "cuda"
