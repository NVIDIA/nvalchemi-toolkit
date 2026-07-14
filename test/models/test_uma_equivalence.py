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
"""Phase 2.5(a) forward-equivalence tests for ``UMAWrapper``.

These tests load a real fairchem checkpoint (default ``uma-s-1p1``,
overridable via ``NVALCHEMI_UMA_CKPT``) and assert that
:class:`nvalchemi.models.uma.UMAWrapper` produces energy / forces /
stress bit-matching ``fairchem.core.calculate.FAIRChemCalculator`` for
the same input structure.

This is the Phase-2 validation gate: if these equivalence checks pass,
the wrapper's tensor-native ``adapt_input`` + ``adapt_output`` path is
wired correctly and matches the official calculator's numerics. Phases
2.5(b) (NVE drift) and 3 (distributed) are downstream of a green run
here.

Skipped when:

* ``fairchem-core`` is not installed.
* The checkpoint cannot be resolved (no HF token, or no access to
  ``facebook/UMA``) — skipping the test with a message rather than
  failing so the file can live in the default suite.
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

from ase import Atoms  # noqa: E402
from ase.build import bulk  # noqa: E402

from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.models.uma import UMAWrapper  # noqa: E402

_CKPT = os.environ.get("NVALCHEMI_UMA_CKPT", "uma-s-1p1")
_DEVICE = os.environ.get("NVALCHEMI_UMA_DEVICE", "cpu")


# ---------------------------------------------------------------------------
# Fixtures — module-scoped so we pay the checkpoint-load cost once per run.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def predict_unit() -> Any:
    """Load the UMA predict unit once for the module.

    Skips the module entirely if HF access or download fails.
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
def fairchem_calc_omol(predict_unit):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    return FAIRChemCalculator(predict_unit=predict_unit, task_name="omol")


@pytest.fixture(scope="module")
def fairchem_calc_omat(predict_unit):
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    return FAIRChemCalculator(predict_unit=predict_unit, task_name="omat")


@pytest.fixture(scope="module")
def wrapper_omol(predict_unit) -> UMAWrapper:
    return UMAWrapper(predict_unit, task_name="omol")


@pytest.fixture(scope="module")
def wrapper_omat(predict_unit) -> UMAWrapper:
    return UMAWrapper(predict_unit, task_name="omat")


# ---------------------------------------------------------------------------
# Structures
# ---------------------------------------------------------------------------


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
    atoms = bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)
    return atoms


def _atomicdata_from_ase(atoms: Atoms) -> AtomicData:
    """Convert an ASE ``Atoms`` into our ``AtomicData``.

    Mirrors what a data loader would produce. PBC/cell are included
    when the source atoms are periodic.
    """
    pos = torch.as_tensor(np.asarray(atoms.positions), dtype=torch.float32)
    numbers = torch.as_tensor(np.asarray(atoms.get_atomic_numbers()), dtype=torch.long)
    kwargs: dict[str, Any] = {"positions": pos, "atomic_numbers": numbers}
    if np.any(atoms.pbc):
        kwargs["cell"] = torch.as_tensor(
            np.asarray(atoms.cell.array), dtype=torch.float32
        ).unsqueeze(0)
        kwargs["pbc"] = torch.as_tensor(
            np.asarray(atoms.pbc), dtype=torch.bool
        ).reshape(1, 3)
    return AtomicData(**kwargs)


# ---------------------------------------------------------------------------
# OMol equivalence
# ---------------------------------------------------------------------------


class TestOMolEquivalence:
    """Propane molecular energy/forces match ``FAIRChemCalculator``."""

    @pytest.fixture(autouse=True)
    def _setup(self, wrapper_omol, fairchem_calc_omol):
        self.wrapper = wrapper_omol
        self.calc = fairchem_calc_omol
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


# ---------------------------------------------------------------------------
# OMat equivalence
# ---------------------------------------------------------------------------


class TestOMatEquivalence:
    """bcc Fe 2x2x2 energy/forces/stress match ``FAIRChemCalculator``."""

    @pytest.fixture(autouse=True)
    def _setup(self, wrapper_omat, fairchem_calc_omat):
        self.wrapper = wrapper_omat
        self.calc = fairchem_calc_omat
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
