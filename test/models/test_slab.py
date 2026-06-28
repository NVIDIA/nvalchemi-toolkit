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
"""Slab-correction tests for electrostatics model wrappers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from test.models.test_ewald import (
    _finite_difference_charge_gradient,
    _make_charged_batch,
    _make_ewald,
)
from test.models.test_pme import _make_pme


def _make_model(method: str, slab_correction: bool) -> Any:
    """Construct an electrostatics wrapper with stable slab-test defaults."""
    if method == "ewald":
        return _make_ewald(cutoff=8.0, slab_correction=slab_correction)
    if method == "pme":
        return _make_pme(
            cutoff=8.0,
            alpha=0.3,
            mesh_dimensions=(16, 16, 16),
            slab_correction=slab_correction,
        )
    raise ValueError(f"Unknown electrostatics method: {method}")


def _add_empty_matrix_neighbors(batch: Batch, max_neighbors: int = 8) -> None:
    """Attach placeholder matrix-neighbor fields for adapt_input-only tests."""
    num_atoms = batch.num_nodes
    object.__setattr__(
        batch,
        "neighbor_matrix",
        torch.full((num_atoms, max_neighbors), num_atoms, dtype=torch.int32),
    )
    object.__setattr__(
        batch, "num_neighbors", torch.zeros(num_atoms, dtype=torch.int32)
    )
    batch._neighbor_list_cutoff = 15.0


def _make_cell(
    pbc: tuple[bool, bool, bool],
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> torch.Tensor:
    """Create a diagonal cell with vacuum along non-periodic slab axes."""
    lengths = torch.tensor(
        [12.0 if periodic else 36.0 for periodic in pbc],
        dtype=dtype,
        device=device,
    )
    return torch.diag(lengths).unsqueeze(0)


def _make_batch_without_pbc() -> Batch:
    """Build a charged periodic-cell batch that intentionally omits pbc."""
    num_atoms = 4
    positions = torch.rand(num_atoms, 3) * 10.0
    atomic_numbers = torch.ones(num_atoms, dtype=torch.long)
    charges = torch.tensor([1.0, -1.0, 1.0, -1.0])
    data = AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        charges=charges,
        forces=torch.zeros_like(positions),
        energy=torch.zeros(1, 1),
        cell=torch.eye(3).unsqueeze(0) * 10.0,
    )
    return Batch.from_data_list([data])


def _make_oriented_slab_batch(
    pbc: tuple[bool, bool, bool],
    dtype: torch.dtype = torch.float64,
    device: str = "cpu",
) -> Batch:
    """Build a small neutral slab with a dipole along the non-periodic axis."""
    nonperiodic_axis = pbc.index(False)
    positions = torch.tensor(
        [
            [2.0, 2.0, 2.0],
            [4.0, 2.0, 2.0],
            [2.0, 4.0, 2.0],
            [4.0, 4.0, 2.0],
        ],
        dtype=dtype,
        device=device,
    )
    positions[1, nonperiodic_axis] += 5.0
    positions[3, nonperiodic_axis] += 5.0
    charges = torch.tensor([1.0, -1.0, 0.5, -0.5], dtype=dtype, device=device)
    atomic_numbers = torch.ones(positions.shape[0], dtype=torch.long, device=device)
    data = AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        charges=charges,
        forces=torch.zeros_like(positions),
        energy=torch.zeros(1, 1, dtype=dtype, device=device),
        cell=_make_cell(pbc, dtype=dtype, device=device),
        pbc=torch.tensor([pbc], dtype=torch.bool, device=device),
    )
    return Batch.from_data_list([data])


def _make_mixed_slab_batch(
    pbc: tuple[bool, bool, bool], dtype: torch.dtype = torch.float64
) -> Batch:
    """Build one 2D-periodic slab system and one fully 3D-periodic system."""
    slab = _make_oriented_slab_batch(pbc, dtype=dtype).get_data(0)
    torch.manual_seed(123)
    periodic = _make_charged_batch(n_atoms=4, box_size=12.0, dtype=dtype).get_data(0)
    return Batch.from_data_list([slab, periodic])


def _build_nl(batch: Batch, model: Any) -> None:
    """Build matrix neighbors for the wrapper under test."""
    from nvalchemi.neighbors import compute_neighbors

    compute_neighbors(batch, config=model.model_config.neighbor_config)


def _run_model(
    batch: Batch, method: str, slab_correction: bool
) -> dict[str, torch.Tensor]:
    """Evaluate an electrostatics wrapper with energy, forces, and stress."""
    model = _make_model(method, slab_correction=slab_correction)
    model.model_config.active_outputs = {"energy", "forces", "stress"}
    _build_nl(batch, model)
    return model(batch)


def _assert_outputs_close(actual: dict, expected: dict) -> None:
    """Assert model energy, forces, and stress are numerically close."""
    torch.testing.assert_close(actual["energy"], expected["energy"])
    torch.testing.assert_close(actual["forces"], expected["forces"])
    torch.testing.assert_close(actual["stress"], expected["stress"])


@pytest.mark.parametrize(
    "model_factory",
    [
        pytest.param(lambda slab: _make_ewald(slab_correction=slab), id="ewald"),
        pytest.param(lambda slab: _make_pme(slab_correction=slab), id="pme"),
    ],
)
class TestSlabAdaptInput:
    """Slab-specific input adaptation tests."""

    def test_pbc_is_omitted_when_slab_correction_disabled(
        self, model_factory: Callable[[bool], object]
    ) -> None:
        """Non-slab wrappers preserve the existing input contract."""
        model = model_factory(False)
        batch = _make_charged_batch()
        _add_empty_matrix_neighbors(batch)

        inputs = model.adapt_input(batch)

        assert "pbc" not in inputs

    def test_pbc_is_collected_when_slab_correction_enabled(
        self, model_factory: Callable[[bool], object]
    ) -> None:
        """Slab wrappers pass periodicity flags through to nvalchemiops."""
        model = model_factory(True)
        batch = _make_charged_batch()
        _add_empty_matrix_neighbors(batch)

        inputs = model.adapt_input(batch)

        assert "pbc" in inputs
        torch.testing.assert_close(inputs["pbc"], batch.pbc)

    def test_slab_correction_requires_pbc_flags(
        self, model_factory: Callable[[bool], object]
    ) -> None:
        """Slab wrappers fail clearly when cell exists but pbc is missing."""
        model = model_factory(True)
        batch = _make_batch_without_pbc()
        _add_empty_matrix_neighbors(batch)

        with pytest.raises(ValueError, match="data.pbc must be present"):
            model.adapt_input(batch)

    def test_slab_correction_rejects_none_pbc(
        self, model_factory: Callable[[bool], object]
    ) -> None:
        """Slab wrappers fail clearly when pbc exists but is None."""
        model = model_factory(True)
        batch = _make_charged_batch()
        _add_empty_matrix_neighbors(batch)
        batch.pbc = None

        with pytest.raises(ValueError, match="data.pbc must be present"):
            model.adapt_input(batch)


class TestElectrostaticsSlabCorrection:
    """End-to-end slab-correction behavior for Ewald and PME wrappers."""

    @pytest.fixture(autouse=True)
    def _require_ops(self):
        pytest.importorskip("nvalchemiops")

    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_ttt_slab_correction_is_noop(self, method: str) -> None:
        """Fully 3D-periodic systems are unchanged by slab_correction=True."""
        torch.manual_seed(42)
        batch = _make_charged_batch(dtype=torch.float64)

        slab_off = _run_model(batch, method, slab_correction=False)
        slab_on = _run_model(batch, method, slab_correction=True)

        _assert_outputs_close(slab_on, slab_off)

    @pytest.mark.parametrize(
        "pbc",
        [
            pytest.param((True, True, False), id="ttf"),
            pytest.param((True, False, True), id="tft"),
            pytest.param((False, True, True), id="ftt"),
        ],
    )
    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_2d_slab_correction_changes_energy(
        self, method: str, pbc: tuple[bool, bool, bool]
    ) -> None:
        """Each slab orientation receives a non-zero slab energy correction."""
        batch = _make_oriented_slab_batch(pbc)

        slab_off = _run_model(batch, method, slab_correction=False)
        slab_on = _run_model(batch, method, slab_correction=True)

        assert not torch.allclose(slab_on["energy"], slab_off["energy"])
        assert not torch.allclose(slab_on["stress"], slab_off["stress"])

    @pytest.mark.parametrize("method", ["ewald", "pme"])
    @pytest.mark.parametrize(
        "pbc",
        [
            pytest.param((True, True, False), id="ttf"),
            pytest.param((True, False, True), id="tft"),
            pytest.param((False, True, True), id="ftt"),
        ],
    )
    def test_mixed_batch_only_corrects_2d_periodic_system(
        self, method: str, pbc: tuple[bool, bool, bool]
    ) -> None:
        """Mixed batches can combine one slab system and one 3D-periodic system."""
        batch = _make_mixed_slab_batch(pbc)

        slab_off = _run_model(batch, method, slab_correction=False)
        slab_on = _run_model(batch, method, slab_correction=True)

        assert not torch.allclose(slab_on["energy"][0], slab_off["energy"][0])
        torch.testing.assert_close(slab_on["energy"][1], slab_off["energy"][1])
        assert not torch.allclose(slab_on["stress"][0], slab_off["stress"][0])
        torch.testing.assert_close(slab_on["stress"][1], slab_off["stress"][1])

        slab_atoms = batch.batch_idx == 0
        periodic_atoms = batch.batch_idx == 1
        assert not torch.allclose(
            slab_on["forces"][slab_atoms], slab_off["forces"][slab_atoms]
        )
        torch.testing.assert_close(
            slab_on["forces"][periodic_atoms], slab_off["forces"][periodic_atoms]
        )

    @pytest.mark.parametrize(
        "pbc",
        [
            pytest.param((True, True, False), id="ttf"),
            pytest.param((True, False, True), id="tft"),
            pytest.param((False, True, True), id="ftt"),
        ],
    )
    @pytest.mark.parametrize("method", ["ewald", "pme"])
    def test_slab_charge_gradient_matches_finite_difference(
        self, method: str, pbc: tuple[bool, bool, bool]
    ) -> None:
        """Slab energy backward recovers finite-difference charge gradients."""
        model = _make_model(method, slab_correction=True)
        batch = _make_oriented_slab_batch(pbc)

        fd_grad = _finite_difference_charge_gradient(model, batch, _build_nl)
        batch.charges = batch.charges.detach().requires_grad_(True)
        out = model(batch)
        out["energy"].sum().backward()

        assert batch.charges.grad is not None
        torch.testing.assert_close(batch.charges.grad, fd_grad, atol=5e-5, rtol=5e-4)
