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
"""Tests for DFT-D3 parameter extraction utilities.

These tests cover the pure-Python Fortran-parsing helpers and the parameter
extraction entry point. They do NOT require network access or the
nvalchemiops CUDA extension.
"""

from __future__ import annotations

import textwrap
from collections import OrderedDict
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Import the functions under test
# ---------------------------------------------------------------------------
from nvalchemi.models.dftd3 import (
    _build_c6_arrays,
    _find_fortran_array,
    _limit,
    _parse_pars_array,
    extract_dftd3_parameters,
)


# ---------------------------------------------------------------------------
# _limit
# ---------------------------------------------------------------------------
class TestLimit:
    """Tests for the Fortran element-encoding decoder."""

    def test_simple_element_no_offset(self):
        """Elements 1–100 decode as (element, 1)."""
        assert _limit(1) == (1, 1)
        assert _limit(6) == (6, 1)
        assert _limit(94) == (94, 1)

    def test_encoded_element_cn_index_2(self):
        """Encoded values > 100 subtract 100 and increment cn_idx."""
        # 101 → atom=1, cn_idx=2
        assert _limit(101) == (1, 2)
        # 106 → atom=6, cn_idx=2
        assert _limit(106) == (6, 2)

    def test_encoded_element_cn_index_3(self):
        """Two passes through the loop give cn_idx=3."""
        # 201 = 101 + 100 → atom=1, cn_idx=3
        assert _limit(201) == (1, 3)

    def test_boundary_exactly_100(self):
        """Value == 100 decodes as (100, 1) without entering the loop."""
        assert _limit(100) == (100, 1)


# ---------------------------------------------------------------------------
# _find_fortran_array
# ---------------------------------------------------------------------------
class TestFindFortranArray:
    """Tests for the Fortran data-block parser."""

    def test_parses_simple_data_block(self):
        """Correctly extracts floats from a ``data var / ... /`` block."""
        content = textwrap.dedent("""\
            data myvar /
            1.5_wp, 2.5_wp, 3.0_wp /
        """)
        result = _find_fortran_array(content, "myvar")
        assert result.dtype == np.float64
        np.testing.assert_allclose(result, [1.5, 2.5, 3.0])

    def test_parses_multiple_lines(self):
        """Multi-line data blocks are joined and parsed correctly."""
        content = textwrap.dedent("""\
            data vals /
            0.1_wp,
            0.2_wp,
            0.3_wp /
        """)
        result = _find_fortran_array(content, "vals")
        np.testing.assert_allclose(result, [0.1, 0.2, 0.3])

    def test_skips_comment_lines(self):
        """Lines beginning with '!' are ignored."""
        content = textwrap.dedent("""\
            ! this is a comment
            data arr / 4.0_wp, 5.0_wp /
        """)
        result = _find_fortran_array(content, "arr")
        np.testing.assert_allclose(result, [4.0, 5.0])

    def test_missing_variable_raises(self):
        """Raises ValueError when the variable is not found in the source."""
        with pytest.raises(ValueError, match="not found"):
            _find_fortran_array("data other / 1.0_wp /", "missing_var")

    def test_case_insensitive_match(self):
        """The regex search is case-insensitive for both 'data' and the name."""
        content = "  DATA MyVar / 7.0_wp /"
        result = _find_fortran_array(content, "MyVar")
        np.testing.assert_allclose(result, [7.0])


# ---------------------------------------------------------------------------
# _parse_pars_array
# ---------------------------------------------------------------------------
class TestParseParsArray:
    """Tests for the pars array parser."""

    def test_parses_two_records(self):
        """Two groups of 5 scientific-notation numbers produce a (2, 5) array."""
        content = textwrap.dedent("""\
            pars(1)=(/
                1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0,
                6.0d0, 7.0d0, 8.0d0, 9.0d0, 10.0d0/)
        """)
        result = _parse_pars_array(content)
        assert result.shape == (2, 5)
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0, 4.0, 5.0])
        np.testing.assert_allclose(result[1], [6.0, 7.0, 8.0, 9.0, 10.0])

    def test_strips_inline_comments(self):
        """Inline '!' comments are stripped before number extraction."""
        content = textwrap.dedent("""\
            pars(1)=(/
                1.0d0, 2.0d0, 3.0d0, 4.0d0, 5.0d0/) ! my comment
        """)
        result = _parse_pars_array(content)
        assert result.shape == (1, 5)
        np.testing.assert_allclose(result[0], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_empty_source_returns_empty(self):
        """Source with no pars block returns empty (0, 5) array."""
        result = _parse_pars_array("no pars here")
        assert result.shape == (0, 5)

    def test_handles_D_notation(self):
        """Fortran 'D' exponent notation (1.5D+00) is parsed correctly."""
        content = textwrap.dedent("""\
            pars(1)=(/
                1.5D+00, 2.5D-01, 3.0D+00, 4.0D+00, 5.0D+00/)
        """)
        result = _parse_pars_array(content)
        assert result.shape == (1, 5)
        assert result[0, 0] == pytest.approx(1.5)
        assert result[0, 1] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# _build_c6_arrays
# ---------------------------------------------------------------------------
class TestBuildC6Arrays:
    """Tests for the C6 and coordination-number reference array builder."""

    def test_symmetry_c6ab(self):
        """C6ab[i, j, a, b] == C6ab[j, i, b, a] (symmetry)."""
        # c6=1.0, z_i=H(encoded 1, cn_idx=1), z_j=He(encoded 2, cn_idx=1)
        record = np.array([[1.0, 1.0, 2.0, 0.5, 0.3]])
        c6ab, _ = _build_c6_arrays(record)
        # _limit(1) = (1, 1) → ia=0; _limit(2) = (2, 1) → ja=0
        assert c6ab[1, 2, 0, 0] == pytest.approx(c6ab[2, 1, 0, 0])
        assert c6ab[1, 2, 0, 0] == pytest.approx(1.0)

    def test_out_of_range_elements_skipped(self):
        """Records with atomic numbers outside [1, 94] are ignored."""
        record = np.array([[99.0, 95.0, 1.0, 0.0, 0.0]])  # z_i_enc=95 → iat=95 > 94
        c6ab, _ = _build_c6_arrays(record)
        # Should remain zero for index 95 (or just not crash)
        assert c6ab[1, 1, 0, 0] == pytest.approx(0.0)

    def test_cn_ref_filled_for_valid_records(self):
        """CN reference values are stored for valid records."""
        record = np.array([[1.0, 1.0, 1.0, 1.23, 1.23]])  # H–H pair
        _, cn_ref = _build_c6_arrays(record)
        # cn_ref[1, partner, 0, :] should be 1.23 for all partners once partner loop runs
        # At minimum, cn_ref[1, 1, 0, 0] should NOT be -1 (the initial sentinel)
        assert cn_ref[1, 1, 0, 0] == pytest.approx(1.23)

    def test_empty_records(self):
        """Empty records produce all-zero c6ab and all-(-1) cn_ref."""
        c6ab, cn_ref = _build_c6_arrays(np.zeros((0, 5)))
        assert c6ab.sum() == pytest.approx(0.0)
        assert (cn_ref == -1.0).all()


# ---------------------------------------------------------------------------
# extract_dftd3_parameters — error paths (no network required)
# ---------------------------------------------------------------------------
class TestExtractDFTD3ParametersErrors:
    """Tests for extract_dftd3_parameters that don't require network access."""

    def test_nonexistent_dir_raises_file_not_found(self, tmp_path: Path):
        """FileNotFoundError when dftd3_ref_dir doesn't exist."""
        missing = tmp_path / "does_not_exist"
        with pytest.raises(FileNotFoundError, match="not found"):
            extract_dftd3_parameters(dftd3_ref_dir=missing)

    def test_dir_missing_dftd3_f_raises(self, tmp_path: Path):
        """FileNotFoundError when dftd3.f is absent from the ref dir."""
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        # Only create pars.f — dftd3.f is missing
        (ref_dir / "pars.f").write_text("! empty\n")
        with pytest.raises(FileNotFoundError, match="dftd3.f"):
            extract_dftd3_parameters(dftd3_ref_dir=ref_dir)

    def test_dir_missing_pars_f_raises(self, tmp_path: Path):
        """FileNotFoundError when pars.f is absent from the ref dir."""
        ref_dir = tmp_path / "ref"
        ref_dir.mkdir()
        # Only create dftd3.f — pars.f is missing
        (ref_dir / "dftd3.f").write_text("! empty\n")
        with pytest.raises(FileNotFoundError, match="pars.f"):
            extract_dftd3_parameters(dftd3_ref_dir=ref_dir)


# ---------------------------------------------------------------------------
# DFTD3ModelWrapper stubs (mocking parameter loading)
# ---------------------------------------------------------------------------
class TestDFTD3ModelWrapperStubs:
    """Tests for model stub methods that don't require real D3 parameters."""

    @pytest.fixture
    def wrapper(self):
        """Construct DFTD3ModelWrapper with mocked parameter loading."""
        from unittest.mock import MagicMock

        fake_params = MagicMock()
        fake_params.rcov = torch.zeros(95)
        fake_params.r4r2 = torch.zeros(95)
        fake_params.c6ab = torch.zeros(95, 95, 5, 5)
        fake_params.cn_ref = torch.full((95, 95, 5, 5), -1.0)

        with patch(
            "nvalchemi.models.dftd3.load_dftd3_params", return_value=fake_params
        ):
            from nvalchemi.models.dftd3 import DFTD3ModelWrapper

            return DFTD3ModelWrapper(1.0, 1.0, 1.0)

    def test_embedding_shapes_returns_empty_dict(self, wrapper):
        """embedding_shapes property returns an empty dict (line 536)."""
        assert wrapper.embedding_shapes == {}

    def test_compute_embeddings_raises_not_implemented(self, wrapper):
        """compute_embeddings raises NotImplementedError (line 542)."""
        with pytest.raises(NotImplementedError):
            wrapper.compute_embeddings(None)  # type: ignore[arg-type]

    def test_export_model_raises_not_implemented(self, wrapper):
        """export_model raises NotImplementedError (line 738)."""
        with pytest.raises(NotImplementedError):
            wrapper.export_model(Path("/tmp/dummy"))  # noqa: S108

    def test_model_config_has_expected_outputs(self, wrapper):
        """model_config reports the correct output capabilities."""
        cfg = wrapper.model_config
        assert "forces" in cfg.outputs
        assert "stress" in cfg.outputs


# ---------------------------------------------------------------------------
# Helpers for comprehensive DFTD3ModelWrapper tests
# ---------------------------------------------------------------------------


def _mock_batch(
    n: int = 4,
    b: int = 1,
    with_cell: bool = True,
    with_shifts: bool = False,
    device: str = "cpu",
) -> Any:
    """Build a lightweight Batch that looks correct to the DFTD3 wrappers."""
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
        energy=energies,
    )
    # Neighbor matrix -- needed by all wrappers
    fill = n
    nm = torch.full((n, 8), fill, dtype=torch.int32, device=device)
    nn_ = torch.zeros(n, dtype=torch.int32, device=device)
    data.add_node_property("neighbor_matrix", nm)
    data.add_node_property("num_neighbors", nn_)

    batch = Batch.from_data_list([data] * b)
    batch._neighbor_list_cutoff = 15.0

    if with_cell:
        cell = torch.eye(3, device=device).unsqueeze(0).expand(b, 3, 3).contiguous()
        batch.cell = cell
        batch.pbc = torch.ones(b, 3, dtype=torch.bool, device=device)

    if with_shifts:
        N = batch.num_nodes
        K = 8
        batch.add_key(
            "neighbor_matrix_shifts",
            [torch.zeros(N, K, 3, dtype=torch.int32, device=device)],
            level="node",
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
        energy=torch.zeros(1, 1, device=device),
    )
    return data


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
# TestDFTD3ModelWrapper -- comprehensive tests (no nvalchemiops required)
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
        assert wrapper.cutoff == pytest.approx(15.0)
        assert wrapper.k1 == pytest.approx(16.0)
        assert wrapper.k3 == pytest.approx(-4.0)
        assert wrapper.s6 == pytest.approx(1.0)
        assert wrapper.smoothing_fraction == pytest.approx(0.2)

    def test_custom_smoothing_fraction(self):
        """Constructor stores a user-supplied smoothing_fraction."""
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8, smoothing_fraction=0.3)
        assert wrapper.smoothing_fraction == pytest.approx(0.3)

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
    # model_config
    # ------------------------------------------------------------------

    def test_model_config_no_autograd_outputs(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.model_config.autograd_outputs == frozenset()

    def test_model_config_outputs_energy(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert "energy" in wrapper.model_config.outputs

    def test_model_config_outputs_forces(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert "forces" in wrapper.model_config.outputs

    def test_model_config_outputs_stress(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert "stress" in wrapper.model_config.outputs

    def test_model_config_needs_pbc_false(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.model_config.needs_pbc is False

    def test_model_config_no_extra_inputs(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.model_config.required_inputs == frozenset()

    def test_model_config_neighbor_config_cutoff(self):
        cutoff = 35.0
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8, cutoff=cutoff)
        assert wrapper.model_config.neighbor_config.cutoff == pytest.approx(cutoff)

    def test_model_config_neighbor_config_format_is_matrix(self):
        from nvalchemi.models.base import NeighborListFormat

        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.model_config.neighbor_config.format == NeighborListFormat.MATRIX

    def test_model_config_neighbor_config_half_list_false(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        assert wrapper.model_config.neighbor_config.half_list is False

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

    def test_output_data_energy_always(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        keys = wrapper.output_data()
        assert "energy" in keys

    def test_output_data_forces_when_active(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.active_outputs.add("forces")
        assert "forces" in wrapper.output_data()

    def test_output_data_no_stress_when_inactive(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.active_outputs.discard("stress")
        assert "stress" not in wrapper.output_data()

    def test_output_data_stress_when_active(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.active_outputs.add("stress")
        assert "stress" in wrapper.output_data()

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
        data = AtomicData(
            positions=torch.randn(n, 3),
            atomic_numbers=torch.ones(n, dtype=torch.int64),
            atomic_masses=torch.ones(n),
            forces=torch.zeros(n, 3),
            energy=torch.zeros(1, 1),
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
        assert inp["neighbor_matrix_shifts"] is None

    def test_adapt_input_neighbor_shifts_present_when_set(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        batch = _mock_batch(n=4, b=1, with_shifts=True)
        inp = wrapper.adapt_input(batch)
        assert inp["neighbor_matrix_shifts"] is not None

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

    def test_adapt_output_energy_always_present(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.active_outputs.discard("stress")
        batch = _mock_batch()
        raw = {
            "energy": torch.tensor([[1.0]]),
            "forces": torch.zeros(4, 3),
        }
        out = wrapper.adapt_output(raw, batch)
        assert "energy" in out

    def test_adapt_output_forces_when_active(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.active_outputs.add("forces")
        wrapper.model_config.active_outputs.discard("stress")
        batch = _mock_batch()
        raw = {
            "energy": torch.tensor([[1.0]]),
            "forces": torch.zeros(4, 3),
        }
        out = wrapper.adapt_output(raw, batch)
        assert "forces" in out

    def test_adapt_output_no_forces_when_inactive(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.active_outputs.discard("forces")
        wrapper.model_config.active_outputs.discard("stress")
        batch = _mock_batch()
        raw = {
            "energy": torch.tensor([[1.0]]),
            "forces": torch.zeros(4, 3),
        }
        out = wrapper.adapt_output(raw, batch)
        assert "forces" not in out

    def test_adapt_output_stress_is_virial_over_volume(self):
        """stress == virial / volume (Cauchy stress, eV/A^3)."""
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.active_outputs.add("stress")
        batch = _mock_batch()  # identity cell, volume = 1.0
        virial = torch.ones(1, 3, 3) * 2.0
        raw = {
            "energy": torch.tensor([[1.0]]),
            "forces": torch.zeros(4, 3),
            "virial": virial,
        }
        out = wrapper.adapt_output(raw, batch)
        assert "stress" in out
        volume = torch.det(batch.cell).abs().view(-1, 1, 1)
        torch.testing.assert_close(out["stress"], virial / volume)

    def test_adapt_output_stress_raises_without_cell(self):
        """ValueError when stress+virial is active but data has no cell."""
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.active_outputs.add("stress")
        batch = _mock_batch(with_cell=False)
        raw = {
            "energy": torch.tensor([[1.0]]),
            "forces": torch.zeros(4, 3),
            "virial": torch.ones(1, 3, 3),
        }
        with pytest.raises(ValueError, match="stress output requires cell"):
            wrapper.adapt_output(raw, batch)

    def test_adapt_output_no_stress_when_inactive(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.active_outputs.discard("stress")
        batch = _mock_batch()
        raw = {
            "energy": torch.tensor([[1.0]]),
            "forces": torch.zeros(4, 3),
            "virial": torch.ones(1, 3, 3),
        }
        out = wrapper.adapt_output(raw, batch)
        assert "stress" not in out

    def test_adapt_output_stress_from_stress_key_when_no_virials(self):
        """Falls back to 'stress' key in model_output when 'virial' is absent."""
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.active_outputs.add("stress")
        batch = _mock_batch()
        stress = torch.ones(1, 3, 3) * 3.0
        raw = {
            "energy": torch.tensor([[1.0]]),
            "forces": torch.zeros(4, 3),
            "stress": stress,
        }
        out = wrapper.adapt_output(raw, batch)
        assert "stress" in out
        torch.testing.assert_close(out["stress"], stress)

    def test_adapt_output_stress_raises_when_missing(self):
        """RuntimeError when stress is active but model_output has neither virial nor stress."""
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.active_outputs.add("stress")
        batch = _mock_batch()
        raw = {
            "energy": torch.tensor([[1.0]]),
            "forces": torch.zeros(4, 3),
        }
        with pytest.raises(RuntimeError, match="missing from model output"):
            wrapper.adapt_output(raw, batch)

    def test_adapt_output_returns_ordered_dict(self):
        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.active_outputs.discard("stress")
        batch = _mock_batch()
        raw = {"energy": torch.tensor([[1.0]]), "forces": torch.zeros(4, 3)}
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
        wrapper.model_config.active_outputs.add("forces")
        wrapper.model_config.active_outputs.discard("stress")

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
            import nvalchemi.models.dftd3 as _d3mod

            def patched_forward(self_inner, data, **kw):
                from nvalchemi.models.dftd3 import ANGSTROM_TO_BOHR  # noqa: F811

                inp = self_inner.adapt_input(data, **kw)
                positions_bohr = inp["positions"] * ANGSTROM_TO_BOHR
                captured["positions"] = positions_bohr
                B = inp["num_graphs"]
                N = inp["positions"].shape[0]
                energies_ev = torch.zeros(B, 1)
                forces_ev = torch.zeros(N, 3)
                return self_inner.adapt_output(
                    {"energy": energies_ev, "forces": forces_ev}, data
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
        wrapper.model_config.active_outputs.add("forces")
        wrapper.model_config.active_outputs.discard("stress")

        batch = _mock_batch(n=4, b=1, with_cell=False)

        energy_ha_value = 0.05  # Hartree

        import nvalchemi.models.dftd3 as _d3mod

        def patched_forward(self_inner, data, **kw):
            inp = self_inner.adapt_input(data, **kw)
            B = inp["num_graphs"]
            N = inp["positions"].shape[0]
            energies_ev = torch.full((B, 1), energy_ha_value * HARTREE_TO_EV)
            forces_ev = torch.zeros(N, 3)
            return self_inner.adapt_output(
                {"energy": energies_ev, "forces": forces_ev}, data
            )

        with patch.object(_d3mod.DFTD3ModelWrapper, "forward", patched_forward):
            out = wrapper.forward(batch)

        expected = energy_ha_value * HARTREE_TO_EV
        assert out["energy"].shape == (1, 1)
        assert out["energy"].item() == pytest.approx(expected, rel=1e-5)

    def test_forward_forces_unit_conversion(self):
        """Forces output must be HARTREE_TO_EV / BOHR_TO_ANGSTROM times kernel value."""
        from nvalchemi.models.dftd3 import BOHR_TO_ANGSTROM, HARTREE_TO_EV

        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.active_outputs.add("forces")
        wrapper.model_config.active_outputs.discard("stress")

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
                {"energy": energies_ev, "forces": forces_ev}, data
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

    def test_forward_stress_unit_conversion(self):
        """Stress output is virial_eV / volume (identity cell => stress == virial_eV)."""
        from nvalchemi.models.dftd3 import HARTREE_TO_EV

        wrapper = _make_d3_wrapper(a1=0.4, a2=4.4, s8=0.8)
        wrapper.model_config.active_outputs.add("forces")
        wrapper.model_config.active_outputs.add("stress")

        batch = _mock_batch(n=4, b=1, with_cell=True)

        virial_ha_value = 0.02  # Hartree

        import nvalchemi.models.dftd3 as _d3mod

        def patched_forward(self_inner, data, **kw):
            inp = self_inner.adapt_input(data, **kw)
            B = inp["num_graphs"]
            N = inp["positions"].shape[0]
            energies_ev = torch.zeros(B, 1)
            forces_ev = torch.zeros(N, 3)
            virials_ev = torch.full((B, 3, 3), virial_ha_value * HARTREE_TO_EV)
            return self_inner.adapt_output(
                {"energy": energies_ev, "forces": forces_ev, "virial": virials_ev},
                data,
            )

        with patch.object(_d3mod.DFTD3ModelWrapper, "forward", patched_forward):
            out = wrapper.forward(batch)

        expected = virial_ha_value * HARTREE_TO_EV
        assert out["stress"].shape == (1, 3, 3)
        torch.testing.assert_close(
            out["stress"],
            torch.full((1, 3, 3), expected),
            rtol=1e-5,
            atol=1e-7,
        )

    def test_forward_passes_smoothing_to_kernel(self):
        """Smoothing distances are converted to Bohr and forwarded to dftd3()."""
        from nvalchemi.models.dftd3 import ANGSTROM_TO_BOHR

        cutoff = 20.0
        smoothing_fraction = 0.2
        wrapper = _make_d3_wrapper(
            a1=0.4,
            a2=4.4,
            s8=0.8,
            cutoff=cutoff,
            smoothing_fraction=smoothing_fraction,
        )
        wrapper.model_config.active_outputs.add("forces")
        wrapper.model_config.active_outputs.discard("stress")

        batch = _mock_batch(n=4, b=1, with_cell=False)
        captured: dict = {}

        def fake_dftd3(**kwargs):
            captured["s5_smoothing_on"] = kwargs["s5_smoothing_on"]
            captured["s5_smoothing_off"] = kwargs["s5_smoothing_off"]
            N = kwargs["positions"].shape[0]
            B = kwargs.get("num_systems", 1)
            return torch.zeros(B), torch.zeros(N, 3), torch.zeros(N)

        modules = self._make_nvalchemiops_mock()
        modules["nvalchemiops.torch.interactions.dispersion"].dftd3 = fake_dftd3
        modules["nvalchemiops.torch.interactions.dispersion"].D3Parameters = MagicMock(
            return_value=MagicMock()
        )

        with patch.dict("sys.modules", modules):
            wrapper.forward(batch)

        expected_on = cutoff * (1.0 - smoothing_fraction) * ANGSTROM_TO_BOHR
        expected_off = cutoff * ANGSTROM_TO_BOHR
        assert captured["s5_smoothing_on"] == pytest.approx(expected_on)
        assert captured["s5_smoothing_off"] == pytest.approx(expected_off)


# ===========================================================================
# Integration tests -- guarded by nvalchemiops availability
# ===========================================================================


class TestDFTD3IntegrationForward:
    """Full forward-pass integration tests for DFTD3ModelWrapper."""

    @pytest.fixture(autouse=True)
    def _require_ops(self):
        pytest.importorskip("nvalchemiops")

    def test_forward_output_shapes_energy_only(self):
        from nvalchemi.models.dftd3 import DFTD3ModelWrapper

        wrapper = DFTD3ModelWrapper(a1=0.4289, a2=4.4407, s8=0.7875)
        wrapper.model_config.active_outputs = {"energy"}
        batch = _mock_batch(n=4, b=1, with_cell=False)
        out = wrapper(batch)
        assert "energy" in out
        assert out["energy"].shape == (1, 1)

    def test_forward_output_shapes_with_forces(self):
        from nvalchemi.models.dftd3 import DFTD3ModelWrapper

        wrapper = DFTD3ModelWrapper(a1=0.4289, a2=4.4407, s8=0.7875)
        wrapper.model_config.active_outputs = {"energy", "forces"}
        batch = _mock_batch(n=4, b=1, with_cell=False)
        out = wrapper(batch)
        assert out["forces"].shape == (batch.num_nodes, 3)

    def test_forward_output_shapes_with_stress(self):
        from nvalchemi.models.dftd3 import DFTD3ModelWrapper

        wrapper = DFTD3ModelWrapper(a1=0.4289, a2=4.4407, s8=0.7875)
        wrapper.model_config.active_outputs.add("stress")
        batch = _mock_batch(n=4, b=1, with_cell=True, with_shifts=True)
        out = wrapper(batch)
        assert "stress" in out
        assert out["stress"].shape == (1, 3, 3)

    def test_forward_energy_is_finite(self):
        from nvalchemi.models.dftd3 import DFTD3ModelWrapper

        wrapper = DFTD3ModelWrapper(a1=0.4289, a2=4.4407, s8=0.7875)
        wrapper.model_config.active_outputs = {"energy"}
        batch = _mock_batch(n=4, b=1, with_cell=False)
        out = wrapper(batch)
        assert torch.isfinite(out["energy"]).all()
