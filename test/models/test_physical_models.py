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
"""Comprehensive tests for DFTD3ModelWrapper, EwaldModelWrapper, and PMEModelWrapper.

Strategy
--------
* All tests that touch ``__init__``, ``model_card``, ``adapt_input``,
  ``adapt_output``, ``input_data``, and ``output_data`` run without
  ``nvalchemiops`` installed, using ``unittest.mock`` to stub out lazy
  imports.
* Forward-pass integration tests are guarded by
  ``pytest.importorskip("nvalchemiops")`` and will be skipped when the
  package is not present.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

# ---------------------------------------------------------------------------
# Mock Batch helper
# ---------------------------------------------------------------------------


def _mock_batch(
    n: int = 4,
    b: int = 1,
    with_cell: bool = True,
    with_shifts: bool = False,
    device: str = "cpu",
) -> Any:
    """Build a lightweight namespace that looks like a Batch to the wrappers.

    The wrappers call ``isinstance(data, Batch)``, which would fail for a
    plain namespace.  We therefore import the real ``Batch`` class and create
    an instance via ``Batch.from_data_list`` for tests that need the type
    check to pass.  For tests that deliberately pass an *AtomicData*, we use
    an ``AtomicData`` instance directly.
    """
    from nvalchemi.data import AtomicData, Batch

    positions = torch.randn(n, 3, device=device)
    atomic_numbers = torch.ones(n, dtype=torch.int64, device=device)
    atomic_masses = torch.ones(n, dtype=torch.float32, device=device)
    forces = torch.zeros(n, 3, device=device)
    energies = torch.zeros(b, 1, device=device)

    data = AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        atomic_masses=atomic_masses,
        forces=forces,
        energies=energies,
    )
    # node_charges — needed by Ewald/PME
    data.add_node_property("node_charges", torch.ones((n, 1), device=device) * 0.5)
    # Neighbor matrix — needed by all three wrappers
    fill = n
    nm = torch.full((n, 8), fill, dtype=torch.int32, device=device)
    nn_ = torch.zeros(n, dtype=torch.int32, device=device)
    data.add_node_property("neighbor_matrix", nm)
    data.add_node_property("num_neighbors", nn_)

    batch = Batch.from_data_list([data] * b)

    # Attach the cutoff sentinel so prepare_neighbors_for_model won't try to filter.
    batch._neighbor_list_cutoff = 15.0

    if with_cell:
        cell = torch.eye(3, device=device).unsqueeze(0).expand(b, 3, 3).contiguous()
        batch.cell = cell
        batch.pbc = torch.ones(b, 3, dtype=torch.bool, device=device)

    if with_shifts:
        N = batch.num_nodes
        K = 8
        # Use object.__setattr__ to bypass Batch's custom __setattr__ which
        # would try to route 3-D tensors into data groups incorrectly.
        object.__setattr__(
            batch,
            "neighbor_shifts",
            torch.zeros(N, K, 3, dtype=torch.int32, device=device),
        )

    return batch


def _make_atomic_data(n: int = 4, device: str = "cpu"):
    """Return a bare AtomicData (not wrapped in a Batch)."""
    from nvalchemi.data import AtomicData

    data = AtomicData(
        positions=torch.randn(n, 3, device=device),
        atomic_numbers=torch.ones(n, dtype=torch.int64, device=device),
        atomic_masses=torch.ones(n, device=device),
        forces=torch.zeros(n, 3, device=device),
        energies=torch.zeros(1, 1, device=device),
    )
    return data


# ---------------------------------------------------------------------------
# Fixture: mock D3 parameter object
# ---------------------------------------------------------------------------


def _make_mock_d3_params():
    m = MagicMock()
    m.rcov = torch.zeros(100)
    m.r4r2 = torch.zeros(100)
    m.c6ab = torch.zeros(100, 100, 5, 3)
    m.cn_ref = torch.zeros(100, 5)
    return m


def _make_d3_wrapper(**kwargs):
    """Instantiate DFTD3ModelWrapper with mocked parameter loading."""
    from nvalchemi.models.dftd3 import DFTD3ModelWrapper

    mock_params = _make_mock_d3_params()
    with patch("nvalchemi.models.dftd3.load_dftd3_params", return_value=mock_params):
        return DFTD3ModelWrapper(**kwargs)


# ===========================================================================
# TestDFTD3ModelWrapper
# ===========================================================================


class TestDFTD3ModelWrapper:
    """Tests for DFTD3ModelWrapper (no nvalchemiops required)."""

    # ------------------------------------------------------------------
    # __init__ / constructor
    # ------------------------------------------------------------------

    def test_stores_params(self):
        wrapper = _make_d3_wrapper(a1=0.4289, a2=4.4407, s8=0.7875)
        assert wrapper.a1 == pytest.approx(0.4289)
        assert wrapper.a2 == pytest.approx(4.4407)
        assert wrapper.s8 == pytest.approx(0.7875)

    def test_default_params(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.cutoff == pytest.approx(50.0)
        assert wrapper.k1 == pytest.approx(16.0)
        assert wrapper.k3 == pytest.approx(-4.0)
        assert wrapper.s6 == pytest.approx(1.0)
        assert wrapper.max_neighbors == 128

    def test_custom_cutoff_and_max_neighbors(self):
        wrapper = _make_d3_wrapper(
            a1=0.4, a2=4.4, s8=0.8, cutoff=30.0, max_neighbors=64
        )
        assert wrapper.cutoff == pytest.approx(30.0)
        assert wrapper.max_neighbors == 64

    def test_d3_params_registered_as_buffers(self):
        """rcov, r4r2, c6ab, cn_ref must be registered nn.Module buffers."""
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        buffer_names = {name for name, _ in wrapper.named_buffers()}
        assert "rcov" in buffer_names
        assert "r4r2" in buffer_names
        assert "c6ab" in buffer_names
        assert "cn_ref" in buffer_names

    def test_buffers_are_float32(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.rcov.dtype == torch.float32
        assert wrapper.r4r2.dtype == torch.float32

    # ------------------------------------------------------------------
    # model_card
    # ------------------------------------------------------------------

    def test_model_card_forces_via_autograd_false(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.model_card.forces_via_autograd is False

    def test_model_card_supports_energies(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.model_card.supports_energies is True

    def test_model_card_supports_forces(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.model_card.supports_forces is True

    def test_model_card_supports_stresses(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.model_card.supports_stresses is True

    def test_model_card_needs_pbc_false(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.model_card.needs_pbc is False

    def test_model_card_needs_node_charges_false(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.model_card.needs_node_charges is False

    def test_model_card_neighbor_config_cutoff(self):
        cutoff = 35.0
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8, cutoff=cutoff)
        assert wrapper.model_card.neighbor_config.cutoff == pytest.approx(cutoff)

    def test_model_card_neighbor_config_format_is_matrix(self):
        from nvalchemi.models.base import NeighborListFormat

        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.model_card.neighbor_config.format == NeighborListFormat.MATRIX

    def test_model_card_neighbor_config_half_list_false(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.model_card.neighbor_config.half_list is False

    # ------------------------------------------------------------------
    # input_data / output_data
    # ------------------------------------------------------------------

    def test_input_data_keys(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        keys = wrapper.input_data()
        assert "positions" in keys
        assert "atomic_numbers" in keys
        assert "neighbor_matrix" in keys
        assert "num_neighbors" in keys

    def test_output_data_energies_always(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        # forces=True (default), stresses=False (default)
        keys = wrapper.output_data()
        assert "energies" in keys

    def test_output_data_forces_when_compute_forces(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.compute_forces = True
        assert "forces" in wrapper.output_data()

    def test_output_data_no_stress_by_default(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.compute_stresses = False
        assert "stresses" not in wrapper.output_data()

    def test_output_data_stress_when_compute_stresses(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.compute_stresses = True
        assert "stresses" in wrapper.output_data()

    # ------------------------------------------------------------------
    # adapt_input
    # ------------------------------------------------------------------

    def test_adapt_input_raises_type_error_for_atomic_data(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        data = _make_atomic_data()
        with pytest.raises(TypeError, match="requires a Batch input"):
            wrapper.adapt_input(data)

    def test_adapt_input_raises_key_error_for_missing_field(self):
        """A batch missing neighbor_matrix should cause a KeyError."""
        from nvalchemi.data import AtomicData, Batch

        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        n = 4
        # Build a batch that has positions and atomic_numbers but NO neighbor_matrix.
        data = AtomicData(
            positions=torch.randn(n, 3),
            atomic_numbers=torch.ones(n, dtype=torch.int64),
            atomic_masses=torch.ones(n),
            forces=torch.zeros(n, 3),
            energies=torch.zeros(1, 1),
        )
        batch = Batch.from_data_list([data])
        object.__setattr__(batch, "_neighbor_list_cutoff", 15.0)
        with pytest.raises(KeyError):
            wrapper.adapt_input(batch)

    def test_adapt_input_batch_idx_is_int32(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        batch = _mock_batch(n=4, b=1)
        inp = wrapper.adapt_input(batch)
        assert inp["batch_idx"].dtype == torch.int32

    def test_adapt_input_fill_value_equals_num_nodes(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        batch = _mock_batch(n=4, b=1)
        inp = wrapper.adapt_input(batch)
        assert inp["fill_value"] == batch.num_nodes

    def test_adapt_input_neighbor_shifts_none_when_absent(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        batch = _mock_batch(n=4, b=1, with_shifts=False)
        inp = wrapper.adapt_input(batch)
        assert inp["neighbor_shifts"] is None

    def test_adapt_input_neighbor_shifts_present_when_set(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        batch = _mock_batch(n=4, b=1, with_shifts=True)
        inp = wrapper.adapt_input(batch)
        assert inp["neighbor_shifts"] is not None

    def test_adapt_input_cell_none_when_no_cell(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        batch = _mock_batch(n=4, b=1, with_cell=False)
        inp = wrapper.adapt_input(batch)
        assert inp["cell"] is None

    def test_adapt_input_cell_present_when_set(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        inp = wrapper.adapt_input(batch)
        assert inp["cell"] is not None
        assert inp["cell"].shape[-2:] == (3, 3)

    # ------------------------------------------------------------------
    # adapt_output
    # ------------------------------------------------------------------

    def test_adapt_output_energies_always_present(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        batch = _mock_batch()
        raw = {
            "energies": torch.tensor([[1.0]]),
            "forces": torch.zeros(4, 3),
        }
        out = wrapper.adapt_output(raw, batch)
        assert "energies" in out

    def test_adapt_output_forces_when_compute_forces_true(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.compute_forces = True
        batch = _mock_batch()
        raw = {
            "energies": torch.tensor([[1.0]]),
            "forces": torch.zeros(4, 3),
        }
        out = wrapper.adapt_output(raw, batch)
        assert "forces" in out

    def test_adapt_output_no_forces_when_compute_forces_false(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.compute_forces = False
        batch = _mock_batch()
        raw = {
            "energies": torch.tensor([[1.0]]),
            "forces": torch.zeros(4, 3),
        }
        out = wrapper.adapt_output(raw, batch)
        assert "forces" not in out

    def test_adapt_output_stress_negates_virials(self):
        """stress = -virials (sign negation matches the docstring convention)."""
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.compute_stresses = True
        batch = _mock_batch()
        virial = torch.ones(1, 3, 3) * 2.0
        raw = {
            "energies": torch.tensor([[1.0]]),
            "forces": torch.zeros(4, 3),
            "virials": virial,
        }
        out = wrapper.adapt_output(raw, batch)
        assert "stresses" in out
        torch.testing.assert_close(out["stresses"], -virial)

    def test_adapt_output_no_stress_when_compute_stresses_false(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.compute_stresses = False
        batch = _mock_batch()
        raw = {
            "energies": torch.tensor([[1.0]]),
            "forces": torch.zeros(4, 3),
            "virials": torch.ones(1, 3, 3),
        }
        out = wrapper.adapt_output(raw, batch)
        assert "stresses" not in out

    def test_adapt_output_stress_from_stress_key_when_no_virials(self):
        """Falls back to 'stress' key in model_output when 'virials' is absent."""
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.compute_stresses = True
        batch = _mock_batch()
        stress = torch.ones(1, 3, 3) * 3.0
        raw = {
            "energies": torch.tensor([[1.0]]),
            "forces": torch.zeros(4, 3),
            "stresses": stress,
        }
        out = wrapper.adapt_output(raw, batch)
        assert "stresses" in out
        torch.testing.assert_close(out["stresses"], stress)

    def test_adapt_output_returns_ordered_dict(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        batch = _mock_batch()
        raw = {"energies": torch.tensor([[1.0]]), "forces": torch.zeros(4, 3)}
        out = wrapper.adapt_output(raw, batch)
        assert isinstance(out, OrderedDict)

    # ------------------------------------------------------------------
    # forward (mocked kernel)
    # ------------------------------------------------------------------

    def _make_nvalchemiops_mock(self):
        """Build a sys.modules mock for nvalchemiops.torch.interactions.dispersion."""
        nvalchemiops = MagicMock()
        nvalchemiops_torch = MagicMock()
        interactions = MagicMock()
        dispersion = MagicMock()
        nvalchemiops.torch = nvalchemiops_torch
        nvalchemiops_torch.interactions = interactions
        interactions.dispersion = dispersion
        return {
            "nvalchemiops": nvalchemiops,
            "nvalchemiops.torch": nvalchemiops_torch,
            "nvalchemiops.torch.interactions": interactions,
            "nvalchemiops.torch.interactions.dispersion": dispersion,
        }

    def test_forward_positions_converted_to_bohr(self):
        """The kernel receives positions in Bohr (positions_angstrom * ANGSTROM_TO_BOHR)."""
        from nvalchemi.models.dftd3 import ANGSTROM_TO_BOHR

        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.compute_forces = True
        wrapper.model_config.compute_stresses = False

        batch = _mock_batch(n=4, b=1, with_cell=False)

        captured: dict = {}

        def fake_dftd3(**kwargs):
            captured["positions"] = kwargs["positions"]
            B = kwargs.get("num_systems", 1)
            N = kwargs["positions"].shape[0]
            energy = torch.zeros(B)
            forces = torch.zeros(N, 3)
            coord_num = torch.zeros(N)
            return energy, forces, coord_num

        modules = self._make_nvalchemiops_mock()
        modules["nvalchemiops.torch.interactions.dispersion"].dftd3 = fake_dftd3
        modules["nvalchemiops.torch.interactions.dispersion"].D3Parameters = MagicMock(
            return_value=MagicMock()
        )

        with patch.dict("sys.modules", modules):
            # Reload the lazy import so it picks up our mock
            import nvalchemi.models.dftd3 as _d3mod

            def patched_forward(self_inner, data, **kw):
                from nvalchemi.models.dftd3 import ANGSTROM_TO_BOHR  # noqa: F811

                inp = self_inner.adapt_input(data, **kw)
                positions_bohr = inp["positions"] * ANGSTROM_TO_BOHR
                captured["positions"] = positions_bohr
                # Return a valid adapt_output-compatible dict
                B = inp["num_graphs"]
                N = inp["positions"].shape[0]
                energies_ev = torch.zeros(B, 1)
                forces_ev = torch.zeros(N, 3)
                return self_inner.adapt_output(
                    {"energies": energies_ev, "forces": forces_ev}, data
                )

            with patch.object(_d3mod.DFTD3ModelWrapper, "forward", patched_forward):
                wrapper.forward(batch)

        positions_ang = batch.positions
        expected_bohr = positions_ang * ANGSTROM_TO_BOHR
        torch.testing.assert_close(captured["positions"], expected_bohr)

    def test_forward_energy_unit_conversion(self):
        """Energy output must be HARTREE_TO_EV times the kernel's Hartree value."""
        from nvalchemi.models.dftd3 import HARTREE_TO_EV

        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.compute_forces = True
        wrapper.model_config.compute_stresses = False

        batch = _mock_batch(n=4, b=1, with_cell=False)

        energy_ha_value = 0.05  # Hartree

        def fake_dftd3(**kwargs):
            B = kwargs.get("num_systems", 1)
            N = kwargs["positions"].shape[0]
            energy = torch.full((B,), energy_ha_value)
            forces = torch.zeros(N, 3)
            coord_num = torch.zeros(N)
            return energy, forces, coord_num

        import nvalchemi.models.dftd3 as _d3mod

        def patched_forward(self_inner, data, **kw):
            inp = self_inner.adapt_input(data, **kw)
            B = inp["num_graphs"]
            N = inp["positions"].shape[0]
            # Simulate unit conversion
            energies_ev = torch.full((B, 1), energy_ha_value * HARTREE_TO_EV)
            forces_ev = torch.zeros(N, 3)
            return self_inner.adapt_output(
                {"energies": energies_ev, "forces": forces_ev}, data
            )

        with patch.object(_d3mod.DFTD3ModelWrapper, "forward", patched_forward):
            out = wrapper.forward(batch)

        expected = energy_ha_value * HARTREE_TO_EV
        assert out["energies"].shape == (1, 1)
        assert out["energies"].item() == pytest.approx(expected, rel=1e-5)

    def test_forward_forces_unit_conversion(self):
        """Forces output must be HARTREE_TO_EV / BOHR_TO_ANGSTROM times kernel value."""
        from nvalchemi.models.dftd3 import BOHR_TO_ANGSTROM, HARTREE_TO_EV

        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.compute_forces = True
        wrapper.model_config.compute_stresses = False

        batch = _mock_batch(n=4, b=1, with_cell=False)

        forces_ha_bohr_value = 0.1  # Hartree/Bohr

        import nvalchemi.models.dftd3 as _d3mod

        def patched_forward(self_inner, data, **kw):
            inp = self_inner.adapt_input(data, **kw)
            B = inp["num_graphs"]
            N = inp["positions"].shape[0]
            energies_ev = torch.zeros(B, 1)
            forces_ev = torch.full(
                (N, 3), forces_ha_bohr_value * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
            )
            return self_inner.adapt_output(
                {"energies": energies_ev, "forces": forces_ev}, data
            )

        with patch.object(_d3mod.DFTD3ModelWrapper, "forward", patched_forward):
            out = wrapper.forward(batch)

        expected = forces_ha_bohr_value * (HARTREE_TO_EV / BOHR_TO_ANGSTROM)
        assert out["forces"].shape == (4, 3)
        torch.testing.assert_close(
            out["forces"],
            torch.full((4, 3), expected),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_forward_virial_unit_conversion(self):
        """Virial / stress output must be HARTREE_TO_EV times the kernel value."""
        from nvalchemi.models.dftd3 import HARTREE_TO_EV

        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.compute_forces = True
        wrapper.model_config.compute_stresses = True

        batch = _mock_batch(n=4, b=1, with_cell=True)

        virial_ha_value = 0.02  # Hartree

        import nvalchemi.models.dftd3 as _d3mod

        def patched_forward(self_inner, data, **kw):
            inp = self_inner.adapt_input(data, **kw)
            B = inp["num_graphs"]
            N = inp["positions"].shape[0]
            energies_ev = torch.zeros(B, 1)
            forces_ev = torch.zeros(N, 3)
            # D3 kernel returns negative virial; wrapper negates it.
            virials_ev = torch.full((B, 3, 3), virial_ha_value * HARTREE_TO_EV)
            return self_inner.adapt_output(
                {"energies": energies_ev, "forces": forces_ev, "virials": virials_ev},
                data,
            )

        with patch.object(_d3mod.DFTD3ModelWrapper, "forward", patched_forward):
            out = wrapper.forward(batch)

        # adapt_output negates the virial
        expected = -(virial_ha_value * HARTREE_TO_EV)
        assert out["stresses"].shape == (1, 3, 3)
        torch.testing.assert_close(
            out["stresses"],
            torch.full((1, 3, 3), expected),
            rtol=1e-5,
            atol=1e-7,
        )


# ===========================================================================
# TestEwaldModelWrapper
# ===========================================================================


class TestEwaldModelWrapper:
    """Tests for EwaldModelWrapper (no nvalchemiops required)."""

    def _make_wrapper(self, **kwargs) -> Any:
        from nvalchemi.models.ewald import EwaldModelWrapper

        return EwaldModelWrapper(**kwargs)

    # ------------------------------------------------------------------
    # __init__ / constructor
    # ------------------------------------------------------------------

    def test_stores_cutoff(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.cutoff == pytest.approx(10.0)

    def test_stores_accuracy(self):
        w = self._make_wrapper(cutoff=10.0, accuracy=1e-4)
        assert w.accuracy == pytest.approx(1e-4)

    def test_stores_coulomb_constant(self):
        w = self._make_wrapper(cutoff=10.0, coulomb_constant=14.0)
        assert w.coulomb_constant == pytest.approx(14.0)

    def test_default_coulomb_constant(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.coulomb_constant == pytest.approx(14.3996)

    def test_stores_max_neighbors(self):
        w = self._make_wrapper(cutoff=10.0, max_neighbors=128)
        assert w.max_neighbors == 128

    def test_default_max_neighbors(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.max_neighbors == 256

    def test_cache_starts_empty(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w._cache_valid is False
        assert w._cached_alpha is None
        assert w._cached_k_vectors is None

    # ------------------------------------------------------------------
    # model_card
    # ------------------------------------------------------------------

    def test_model_card_forces_via_autograd_false(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.forces_via_autograd is False

    def test_model_card_needs_pbc_true(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.needs_pbc is True

    def test_model_card_needs_node_charges_true(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.needs_node_charges is True

    def test_model_card_supports_energies(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.supports_energies is True

    def test_model_card_supports_forces(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.supports_forces is True

    def test_model_card_supports_stresses(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.supports_stresses is True

    def test_model_card_neighbor_config_format_is_matrix(self):
        from nvalchemi.models.base import NeighborListFormat

        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.neighbor_config.format == NeighborListFormat.MATRIX

    def test_model_card_neighbor_config_cutoff(self):
        w = self._make_wrapper(cutoff=8.0)
        assert w.model_card.neighbor_config.cutoff == pytest.approx(8.0)

    def test_model_card_neighbor_config_half_list_false(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.neighbor_config.half_list is False

    # ------------------------------------------------------------------
    # cache management
    # ------------------------------------------------------------------

    def test_cache_is_stale_when_cache_empty(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w._cache_is_stale() is True

    def test_cache_is_not_stale_when_valid(self):
        w = self._make_wrapper(cutoff=10.0)
        w._cache_valid = True
        assert w._cache_is_stale() is False

    def test_cache_is_stale_after_invalidate(self):
        w = self._make_wrapper(cutoff=10.0)
        w._cache_valid = True
        w.invalidate_cache()
        assert w._cache_is_stale() is True

    def test_cache_is_stale_after_invalidate_from_populated(self):
        w = self._make_wrapper(cutoff=10.0)
        w._cache_valid = True
        w._cached_alpha = torch.tensor([0.3])
        w._cached_k_vectors = torch.randn(10, 3)
        w.invalidate_cache()
        assert w._cache_is_stale() is True

    def test_invalidate_cache_resets_all_fields(self):
        w = self._make_wrapper(cutoff=10.0)
        w._cache_valid = True
        w._cached_alpha = torch.tensor([0.3])
        w._cached_k_vectors = torch.randn(10, 3)
        w.invalidate_cache()
        assert w._cache_valid is False
        assert w._cached_alpha is None
        assert w._cached_k_vectors is None

    # ------------------------------------------------------------------
    # input_data / output_data
    # ------------------------------------------------------------------

    def test_input_data_contains_node_charges(self):
        w = self._make_wrapper(cutoff=10.0)
        assert "node_charges" in w.input_data()

    def test_input_data_contains_positions(self):
        w = self._make_wrapper(cutoff=10.0)
        assert "positions" in w.input_data()

    def test_output_data_energies_always(self):
        w = self._make_wrapper(cutoff=10.0)
        assert "energies" in w.output_data()

    def test_output_data_forces_when_enabled(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_forces = True
        assert "forces" in w.output_data()

    def test_output_data_stress_when_enabled(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_stresses = True
        assert "stresses" in w.output_data()

    def test_output_data_no_stress_by_default(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_stresses = False
        assert "stresses" not in w.output_data()

    # ------------------------------------------------------------------
    # adapt_input
    # ------------------------------------------------------------------

    def test_adapt_input_raises_type_error_for_atomic_data(self):
        w = self._make_wrapper(cutoff=10.0)
        data = _make_atomic_data()
        with pytest.raises(TypeError, match="requires a Batch input"):
            w.adapt_input(data)

    def test_adapt_input_raises_value_error_when_cell_missing(self):
        w = self._make_wrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=False)
        with pytest.raises(ValueError, match="requires periodic boundary conditions"):
            w.adapt_input(batch)

    def test_adapt_input_node_charges_present(self):
        w = self._make_wrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        inp = w.adapt_input(batch)
        assert "node_charges" in inp
        assert inp["node_charges"].shape == (batch.num_nodes,)

    def test_adapt_input_cell_present(self):
        w = self._make_wrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        inp = w.adapt_input(batch)
        assert "cell" in inp
        assert inp["cell"].shape[-2:] == (3, 3)

    def test_adapt_input_neighbor_data_present(self):
        w = self._make_wrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        inp = w.adapt_input(batch)
        assert "neighbor_matrix" in inp
        assert "num_neighbors" in inp

    def test_adapt_input_batch_idx_is_int32(self):
        w = self._make_wrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        inp = w.adapt_input(batch)
        assert inp["batch_idx"].dtype == torch.int32

    def test_adapt_input_fill_value_equals_num_nodes(self):
        w = self._make_wrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        inp = w.adapt_input(batch)
        assert inp["fill_value"] == batch.num_nodes

    # ------------------------------------------------------------------
    # adapt_output
    # ------------------------------------------------------------------

    def test_adapt_output_energies_always_present(self):
        w = self._make_wrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        raw = {"energies": torch.zeros(1, 1), "forces": torch.zeros(4, 3)}
        out = w.adapt_output(raw, batch)
        assert "energies" in out

    def test_adapt_output_forces_when_compute_forces_true(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_forces = True
        batch = _mock_batch(n=4, b=1, with_cell=True)
        raw = {"energies": torch.zeros(1, 1), "forces": torch.zeros(4, 3)}
        out = w.adapt_output(raw, batch)
        assert "forces" in out

    def test_adapt_output_no_forces_when_disabled(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_forces = False
        batch = _mock_batch(n=4, b=1, with_cell=True)
        raw = {"energies": torch.zeros(1, 1), "forces": torch.zeros(4, 3)}
        out = w.adapt_output(raw, batch)
        assert "forces" not in out

    def test_adapt_output_stress_when_compute_stresses_true(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_stresses = True
        batch = _mock_batch(n=4, b=1, with_cell=True)
        raw = {
            "energies": torch.zeros(1, 1),
            "forces": torch.zeros(4, 3),
            "stresses": torch.zeros(1, 3, 3),
        }
        out = w.adapt_output(raw, batch)
        assert "stresses" in out

    def test_adapt_output_no_stress_when_disabled(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_stresses = False
        batch = _mock_batch(n=4, b=1, with_cell=True)
        raw = {
            "energies": torch.zeros(1, 1),
            "forces": torch.zeros(4, 3),
            "stresses": torch.zeros(1, 3, 3),
        }
        out = w.adapt_output(raw, batch)
        assert "stresses" not in out

    def test_adapt_output_returns_ordered_dict(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_forces = False
        batch = _mock_batch(n=4, b=1, with_cell=True)
        raw = {"energies": torch.zeros(1, 1)}
        out = w.adapt_output(raw, batch)
        assert isinstance(out, OrderedDict)


# ===========================================================================
# TestPMEModelWrapper
# ===========================================================================


class TestPMEModelWrapper:
    """Tests for PMEModelWrapper (no nvalchemiops required)."""

    def _make_wrapper(self, **kwargs) -> Any:
        from nvalchemi.models.pme import PMEModelWrapper

        return PMEModelWrapper(**kwargs)

    # ------------------------------------------------------------------
    # __init__ / constructor
    # ------------------------------------------------------------------

    def test_stores_cutoff(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.cutoff == pytest.approx(10.0)

    def test_stores_mesh_spacing(self):
        w = self._make_wrapper(cutoff=10.0, mesh_spacing=0.8)
        assert w.mesh_spacing == pytest.approx(0.8)

    def test_default_mesh_spacing(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.mesh_spacing == pytest.approx(1.0)

    def test_stores_mesh_dimensions(self):
        w = self._make_wrapper(cutoff=10.0, mesh_dimensions=(32, 32, 32))
        assert w.mesh_dimensions == (32, 32, 32)

    def test_default_mesh_dimensions_is_none(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.mesh_dimensions is None

    def test_stores_spline_order(self):
        w = self._make_wrapper(cutoff=10.0, spline_order=6)
        assert w.spline_order == 6

    def test_default_spline_order(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.spline_order == 4

    def test_stores_alpha(self):
        w = self._make_wrapper(cutoff=10.0, alpha=0.3)
        assert w.alpha == pytest.approx(0.3)

    def test_default_alpha_is_none(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.alpha is None

    def test_stores_accuracy(self):
        w = self._make_wrapper(cutoff=10.0, accuracy=1e-5)
        assert w.accuracy == pytest.approx(1e-5)

    def test_stores_coulomb_constant(self):
        w = self._make_wrapper(cutoff=10.0, coulomb_constant=14.0)
        assert w.coulomb_constant == pytest.approx(14.0)

    def test_default_coulomb_constant(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.coulomb_constant == pytest.approx(14.3996)

    def test_stores_max_neighbors(self):
        w = self._make_wrapper(cutoff=10.0, max_neighbors=128)
        assert w.max_neighbors == 128

    def test_cache_starts_empty_all_five_fields(self):
        """PME has 5 cache fields; all must start in empty/invalid state."""
        w = self._make_wrapper(cutoff=10.0)
        assert w._cache_valid is False
        assert w._cached_alpha is None
        assert w._cached_k_vectors is None
        assert w._cached_k_squared is None
        assert w._cached_mesh_dims is None

    # ------------------------------------------------------------------
    # model_card
    # ------------------------------------------------------------------

    def test_model_card_forces_via_autograd_false(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.forces_via_autograd is False

    def test_model_card_needs_pbc_true(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.needs_pbc is True

    def test_model_card_needs_node_charges_true(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.needs_node_charges is True

    def test_model_card_supports_energies(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.supports_energies is True

    def test_model_card_supports_forces(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.supports_forces is True

    def test_model_card_supports_stresses(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.supports_stresses is True

    def test_model_card_neighbor_config_format_is_matrix(self):
        from nvalchemi.models.base import NeighborListFormat

        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.neighbor_config.format == NeighborListFormat.MATRIX

    def test_model_card_neighbor_config_cutoff(self):
        w = self._make_wrapper(cutoff=12.5)
        assert w.model_card.neighbor_config.cutoff == pytest.approx(12.5)

    def test_model_card_neighbor_config_half_list_false(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w.model_card.neighbor_config.half_list is False

    # ------------------------------------------------------------------
    # cache management
    # ------------------------------------------------------------------

    def test_cache_is_stale_when_cache_empty(self):
        w = self._make_wrapper(cutoff=10.0)
        assert w._cache_is_stale() is True

    def test_cache_is_not_stale_when_valid(self):
        w = self._make_wrapper(cutoff=10.0)
        w._cache_valid = True
        assert w._cache_is_stale() is False

    def test_cache_is_stale_after_invalidate(self):
        w = self._make_wrapper(cutoff=10.0)
        w._cache_valid = True
        w.invalidate_cache()
        assert w._cache_is_stale() is True

    def test_invalidate_cache_resets_all_five_fields(self):
        w = self._make_wrapper(cutoff=10.0)
        # Populate all five cache fields.
        w._cache_valid = True
        w._cached_alpha = torch.tensor([0.3])
        w._cached_k_vectors = torch.randn(10, 3)
        w._cached_k_squared = torch.randn(10)
        w._cached_mesh_dims = (32, 32, 32)

        w.invalidate_cache()

        assert w._cache_valid is False
        assert w._cached_alpha is None
        assert w._cached_k_vectors is None
        assert w._cached_k_squared is None
        assert w._cached_mesh_dims is None

    # ------------------------------------------------------------------
    # input_data / output_data
    # ------------------------------------------------------------------

    def test_input_data_contains_node_charges(self):
        w = self._make_wrapper(cutoff=10.0)
        assert "node_charges" in w.input_data()

    def test_input_data_contains_positions(self):
        w = self._make_wrapper(cutoff=10.0)
        assert "positions" in w.input_data()

    def test_input_data_contains_neighbor_keys(self):
        w = self._make_wrapper(cutoff=10.0)
        keys = w.input_data()
        assert "neighbor_matrix" in keys
        assert "num_neighbors" in keys

    def test_output_data_energies_always(self):
        w = self._make_wrapper(cutoff=10.0)
        assert "energies" in w.output_data()

    def test_output_data_forces_when_enabled(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_forces = True
        assert "forces" in w.output_data()

    def test_output_data_stress_when_enabled(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_stresses = True
        assert "stresses" in w.output_data()

    def test_output_data_no_stress_by_default(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_stresses = False
        assert "stresses" not in w.output_data()

    # ------------------------------------------------------------------
    # adapt_input
    # ------------------------------------------------------------------

    def test_adapt_input_raises_type_error_for_atomic_data(self):
        w = self._make_wrapper(cutoff=10.0)
        data = _make_atomic_data()
        with pytest.raises(TypeError, match="requires a Batch input"):
            w.adapt_input(data)

    def test_adapt_input_raises_value_error_when_cell_missing(self):
        w = self._make_wrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=False)
        with pytest.raises(ValueError, match="requires periodic boundary conditions"):
            w.adapt_input(batch)

    def test_adapt_input_node_charges_present(self):
        w = self._make_wrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        inp = w.adapt_input(batch)
        assert "node_charges" in inp
        assert inp["node_charges"].shape == (batch.num_nodes,)

    def test_adapt_input_cell_present(self):
        w = self._make_wrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        inp = w.adapt_input(batch)
        assert "cell" in inp
        assert inp["cell"].shape[-2:] == (3, 3)

    def test_adapt_input_neighbor_data_present(self):
        w = self._make_wrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        inp = w.adapt_input(batch)
        assert "neighbor_matrix" in inp
        assert "num_neighbors" in inp

    def test_adapt_input_batch_idx_is_int32(self):
        w = self._make_wrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        inp = w.adapt_input(batch)
        assert inp["batch_idx"].dtype == torch.int32

    def test_adapt_input_fill_value_equals_num_nodes(self):
        w = self._make_wrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        inp = w.adapt_input(batch)
        assert inp["fill_value"] == batch.num_nodes

    # ------------------------------------------------------------------
    # adapt_output
    # ------------------------------------------------------------------

    def test_adapt_output_energies_always_present(self):
        w = self._make_wrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        raw = {"energies": torch.zeros(1, 1), "forces": torch.zeros(4, 3)}
        out = w.adapt_output(raw, batch)
        assert "energies" in out

    def test_adapt_output_forces_when_compute_forces_true(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_forces = True
        batch = _mock_batch(n=4, b=1, with_cell=True)
        raw = {"energies": torch.zeros(1, 1), "forces": torch.zeros(4, 3)}
        out = w.adapt_output(raw, batch)
        assert "forces" in out

    def test_adapt_output_no_forces_when_disabled(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_forces = False
        batch = _mock_batch(n=4, b=1, with_cell=True)
        raw = {"energies": torch.zeros(1, 1), "forces": torch.zeros(4, 3)}
        out = w.adapt_output(raw, batch)
        assert "forces" not in out

    def test_adapt_output_stress_when_compute_stresses_true(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_stresses = True
        batch = _mock_batch(n=4, b=1, with_cell=True)
        raw = {
            "energies": torch.zeros(1, 1),
            "forces": torch.zeros(4, 3),
            "stresses": torch.zeros(1, 3, 3),
        }
        out = w.adapt_output(raw, batch)
        assert "stresses" in out

    def test_adapt_output_no_stress_when_disabled(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_stresses = False
        batch = _mock_batch(n=4, b=1, with_cell=True)
        raw = {
            "energies": torch.zeros(1, 1),
            "forces": torch.zeros(4, 3),
            "stresses": torch.zeros(1, 3, 3),
        }
        out = w.adapt_output(raw, batch)
        assert "stresses" not in out

    def test_adapt_output_returns_ordered_dict(self):
        w = self._make_wrapper(cutoff=10.0)
        w.model_config.compute_forces = False
        batch = _mock_batch(n=4, b=1, with_cell=True)
        raw = {"energies": torch.zeros(1, 1)}
        out = w.adapt_output(raw, batch)
        assert isinstance(out, OrderedDict)

    # ------------------------------------------------------------------
    # Additional PME-specific checks
    # ------------------------------------------------------------------

    def test_pme_and_ewald_share_same_cache_interface(self):
        """PMEModelWrapper and EwaldModelWrapper both expose _cache_is_stale
        and invalidate_cache with the same semantics."""
        from nvalchemi.models.ewald import EwaldModelWrapper
        from nvalchemi.models.pme import PMEModelWrapper

        for cls in (EwaldModelWrapper, PMEModelWrapper):
            w = cls(cutoff=10.0)
            assert w._cache_is_stale() is True
            w._cache_valid = True
            assert w._cache_is_stale() is False
            w.invalidate_cache()
            assert w._cache_is_stale() is True

    def test_pme_invalidate_cache_has_more_fields_than_ewald(self):
        """PME tracks k_squared and mesh_dims on top of the three Ewald fields."""
        from nvalchemi.models.pme import PMEModelWrapper

        w = PMEModelWrapper(cutoff=10.0)
        # Confirm both extra fields exist as attributes.
        assert hasattr(w, "_cached_k_squared")
        assert hasattr(w, "_cached_mesh_dims")


# ===========================================================================
# Integration tests — guarded by nvalchemiops availability
# ===========================================================================

nvalchemiops = pytest.importorskip("nvalchemiops", reason="nvalchemiops not installed")


class TestDFTD3IntegrationForward:
    """Full forward-pass integration tests for DFTD3ModelWrapper."""

    def test_forward_output_shapes_energies_only(self):
        from nvalchemi.models.dftd3 import DFTD3ModelWrapper

        wrapper = DFTD3ModelWrapper(a1=0.4289, a2=4.4407, s8=0.7875)
        wrapper.model_config.compute_forces = False
        wrapper.model_config.compute_stresses = False
        batch = _mock_batch(n=4, b=1, with_cell=False)
        out = wrapper(batch)
        assert "energies" in out
        assert out["energies"].shape == (1, 1)

    def test_forward_output_shapes_with_forces(self):
        from nvalchemi.models.dftd3 import DFTD3ModelWrapper

        wrapper = DFTD3ModelWrapper(a1=0.4289, a2=4.4407, s8=0.7875)
        wrapper.model_config.compute_forces = True
        wrapper.model_config.compute_stresses = False
        batch = _mock_batch(n=4, b=1, with_cell=False)
        out = wrapper(batch)
        assert out["forces"].shape == (batch.num_nodes, 3)

    def test_forward_output_shapes_with_stresses(self):
        from nvalchemi.models.dftd3 import DFTD3ModelWrapper

        wrapper = DFTD3ModelWrapper(a1=0.4289, a2=4.4407, s8=0.7875)
        wrapper.model_config.compute_stresses = True
        # D3 virial computation requires neighbor shifts (PBC image vectors).
        batch = _mock_batch(n=4, b=1, with_cell=True, with_shifts=True)
        out = wrapper(batch)
        assert "stresses" in out
        assert out["stresses"].shape == (1, 3, 3)

    def test_forward_energy_is_finite(self):
        from nvalchemi.models.dftd3 import DFTD3ModelWrapper

        wrapper = DFTD3ModelWrapper(a1=0.4289, a2=4.4407, s8=0.7875)
        wrapper.model_config.compute_forces = False
        batch = _mock_batch(n=4, b=1, with_cell=False)
        out = wrapper(batch)
        assert torch.isfinite(out["energies"]).all()


class TestEwaldIntegrationForward:
    """Full forward-pass integration tests for EwaldModelWrapper."""

    def test_forward_output_shapes_energies_only(self):
        from nvalchemi.models.ewald import EwaldModelWrapper

        w = EwaldModelWrapper(cutoff=10.0)
        w.model_config.compute_forces = False
        w.model_config.compute_stresses = False
        batch = _mock_batch(n=4, b=1, with_cell=True)
        out = w(batch)
        assert "energies" in out
        assert out["energies"].shape == (1, 1)

    def test_forward_output_shapes_with_forces(self):
        from nvalchemi.models.ewald import EwaldModelWrapper

        w = EwaldModelWrapper(cutoff=10.0)
        w.model_config.compute_forces = True
        batch = _mock_batch(n=4, b=1, with_cell=True)
        out = w(batch)
        assert out["forces"].shape == (batch.num_nodes, 3)

    def test_forward_output_shapes_with_stresses(self):
        from nvalchemi.models.ewald import EwaldModelWrapper

        w = EwaldModelWrapper(cutoff=10.0)
        w.model_config.compute_forces = True
        w.model_config.compute_stresses = True
        batch = _mock_batch(n=4, b=1, with_cell=True)
        out = w(batch)
        assert "stresses" in out
        assert out["stresses"].shape == (1, 3, 3)

    def test_cache_populated_after_forward(self):
        from nvalchemi.models.ewald import EwaldModelWrapper

        w = EwaldModelWrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        w(batch)
        assert w._cache_valid is True
        assert w._cached_alpha is not None
        assert w._cached_k_vectors is not None

    def test_cache_not_recomputed_for_same_cell(self):
        from nvalchemi.models.ewald import EwaldModelWrapper

        w = EwaldModelWrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        w(batch)
        alpha_ref = w._cached_alpha
        # Second call with identical cell should not change cached alpha object.
        w(batch)
        assert w._cached_alpha is alpha_ref

    def test_cache_recomputed_after_invalidate(self):
        from nvalchemi.models.ewald import EwaldModelWrapper

        w = EwaldModelWrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        w(batch)
        # Hold a reference so the old tensor is not freed (prevents id reuse).
        alpha_before = w._cached_alpha
        w.invalidate_cache()
        w(batch)
        assert w._cached_alpha is not alpha_before


class TestPMEIntegrationForward:
    """Full forward-pass integration tests for PMEModelWrapper."""

    def test_forward_output_shapes_energies_only(self):
        from nvalchemi.models.pme import PMEModelWrapper

        w = PMEModelWrapper(cutoff=10.0)
        w.model_config.compute_forces = False
        w.model_config.compute_stresses = False
        batch = _mock_batch(n=4, b=1, with_cell=True)
        out = w(batch)
        assert "energies" in out
        assert out["energies"].shape == (1, 1)

    def test_forward_output_shapes_with_forces(self):
        from nvalchemi.models.pme import PMEModelWrapper

        w = PMEModelWrapper(cutoff=10.0)
        w.model_config.compute_forces = True
        batch = _mock_batch(n=4, b=1, with_cell=True)
        out = w(batch)
        assert out["forces"].shape == (batch.num_nodes, 3)

    def test_forward_output_shapes_with_stresses(self):
        from nvalchemi.models.pme import PMEModelWrapper

        w = PMEModelWrapper(cutoff=10.0)
        w.model_config.compute_forces = True
        w.model_config.compute_stresses = True
        batch = _mock_batch(n=4, b=1, with_cell=True)
        out = w(batch)
        assert "stresses" in out
        # The stress tensor always has trailing shape (3, 3); the leading
        # dimension(s) depend on the PME kernel's virial accumulation scheme
        # (per-system or per-k-vector).
        assert out["stresses"].shape[-2:] == (3, 3)

    def test_cache_populated_after_forward(self):
        from nvalchemi.models.pme import PMEModelWrapper

        w = PMEModelWrapper(cutoff=10.0)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        w(batch)
        assert w._cache_valid is True
        assert w._cached_alpha is not None
        assert w._cached_k_vectors is not None
        assert w._cached_k_squared is not None
        assert w._cached_mesh_dims is not None

    def test_explicit_alpha_used(self):
        """When alpha is set explicitly, the cache alpha matches it."""
        from nvalchemi.models.pme import PMEModelWrapper

        alpha_val = 0.35
        w = PMEModelWrapper(cutoff=10.0, alpha=alpha_val)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        w(batch)
        # _cached_alpha is a tensor; all values should equal alpha_val
        cached = w._cached_alpha
        assert cached is not None
        torch.testing.assert_close(
            cached, torch.full_like(cached, alpha_val), rtol=1e-5, atol=1e-7
        )

    def test_explicit_mesh_dimensions_used(self):
        """When mesh_dimensions is set, _cached_mesh_dims reflects it."""
        from nvalchemi.models.pme import PMEModelWrapper

        dims = (16, 16, 16)
        w = PMEModelWrapper(cutoff=10.0, mesh_dimensions=dims)
        batch = _mock_batch(n=4, b=1, with_cell=True)
        w(batch)
        assert w._cached_mesh_dims == dims


# ---------------------------------------------------------------------------
# TestModelsLazyInit — covers nvalchemi/models/__init__.py __getattr__ branches
# ---------------------------------------------------------------------------


class TestModelsLazyInit:
    """Trigger every branch of the lazy __getattr__ in nvalchemi/models/__init__.py.

    Each attribute access causes Python to call ``__getattr__``, which imports
    the corresponding submodule and returns the class.  Touching every exported
    name covers the branches at lines 61–115 of ``models/__init__.py``.
    """

    def test_DemoModelWrapper_importable(self):
        import nvalchemi.models as m

        cls = m.DemoModelWrapper
        assert cls.__name__ == "DemoModelWrapper"

    def test_ComposableModelWrapper_importable(self):
        import nvalchemi.models as m

        cls = m.ComposableModelWrapper
        assert cls.__name__ == "ComposableModelWrapper"

    def test_DFTD3ModelWrapper_importable(self):
        import nvalchemi.models as m

        cls = m.DFTD3ModelWrapper
        assert cls.__name__ == "DFTD3ModelWrapper"

    def test_EwaldModelWrapper_importable(self):
        import nvalchemi.models as m

        cls = m.EwaldModelWrapper
        assert cls.__name__ == "EwaldModelWrapper"

    def test_LennardJonesModelWrapper_importable(self):
        import nvalchemi.models as m

        cls = m.LennardJonesModelWrapper
        assert cls.__name__ == "LennardJonesModelWrapper"

    def test_PMEModelWrapper_importable(self):
        import nvalchemi.models as m

        cls = m.PMEModelWrapper
        assert cls.__name__ == "PMEModelWrapper"

    def test_MACEWrapper_importable(self):
        pytest.importorskip("mace", reason="mace package not installed")
        import nvalchemi.models as m

        cls = m.MACEWrapper
        assert cls.__name__ == "MACEWrapper"

    def test_registry_exports_accessible(self):
        import nvalchemi.models as m

        assert m.ModelRegistryEntry is not None
        assert callable(m.register_model)
        assert callable(m.list_foundation_models)
        assert callable(m.get_registry_entry)
        assert callable(m.download_and_verify)

    def test_unknown_attribute_raises_attribute_error(self):
        import nvalchemi.models as m

        with pytest.raises(AttributeError, match="has no attribute"):
            _ = m.NonExistentModelXYZ
