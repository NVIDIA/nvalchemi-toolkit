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
from pathlib import Path
from unittest.mock import patch

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
