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
"""Tests for LennardJonesModelWrapper (nvalchemi/models/lj.py) and
the private helpers in nvalchemi/models/_ops/lj.py.

Strategy
--------
* All tests that touch ``__init__``, ``model_config``, ``adapt_input``,
  ``adapt_output``, ``input_data``, and ``output_data`` run without
  ``nvalchemiops`` installed, because the forward pass (which calls the
  Warp kernels) is never exercised.
* The ``TestOpsHelpers`` class is guarded by
  ``pytest.importorskip("warp")`` at the class level.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import (
    ModelConfig,
    NeighborListFormat,
)
from nvalchemi.models.lj import LennardJonesModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lj_batch(n_atoms: int = 4, max_neighbors: int = 8) -> Batch:
    """Create a Batch with neighbor_matrix and num_neighbors set manually."""
    positions = torch.randn(n_atoms, 3)
    atomic_numbers = torch.ones(n_atoms, dtype=torch.int64)
    atomic_masses = torch.ones(n_atoms, dtype=torch.float32)

    data = AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        atomic_masses=atomic_masses,
    )
    # Attach neighbor matrix (padded with fill_value = n_atoms)
    nm = torch.full((n_atoms, max_neighbors), n_atoms, dtype=torch.int32)
    nn_ = torch.zeros(n_atoms, dtype=torch.int32)
    data.add_node_property("neighbor_matrix", nm)
    data.add_node_property("num_neighbors", nn_)

    batch = Batch.from_data_list([data])
    return batch


def _make_model(
    epsilon: float = 0.0104,
    sigma: float = 3.40,
    cutoff: float = 8.5,
) -> LennardJonesModelWrapper:
    """Construct a default LennardJonesModelWrapper for testing."""
    return LennardJonesModelWrapper(epsilon=epsilon, sigma=sigma, cutoff=cutoff)


# ---------------------------------------------------------------------------
# TestLennardJonesModelWrapperInit
# ---------------------------------------------------------------------------


class TestLennardJonesModelWrapperInit:
    """Verify constructor stores parameters and initialises buffers correctly."""

    def test_stores_epsilon(self):
        model = LennardJonesModelWrapper(epsilon=0.5, sigma=2.0, cutoff=6.0)
        assert model.epsilon == 0.5

    def test_stores_sigma(self):
        model = LennardJonesModelWrapper(epsilon=0.5, sigma=2.0, cutoff=6.0)
        assert model.sigma == 2.0

    def test_stores_cutoff(self):
        model = LennardJonesModelWrapper(epsilon=0.5, sigma=2.0, cutoff=6.0)
        assert model.cutoff == 6.0

    def test_stores_switch_width_default(self):
        model = LennardJonesModelWrapper(epsilon=0.5, sigma=2.0, cutoff=6.0)
        assert model.switch_width == 0.0

    def test_stores_switch_width_custom(self):
        model = LennardJonesModelWrapper(
            epsilon=0.5, sigma=2.0, cutoff=6.0, switch_width=1.0
        )
        assert model.switch_width == 1.0

    def test_stores_half_list_default(self):
        model = LennardJonesModelWrapper(epsilon=0.5, sigma=2.0, cutoff=6.0)
        assert model.half_list is False

    def test_stores_half_list_custom(self):
        model = LennardJonesModelWrapper(
            epsilon=0.5, sigma=2.0, cutoff=6.0, half_list=True
        )
        assert model.half_list is True

    def test_atomic_energies_buf_is_none(self):
        model = _make_model()
        assert model._atomic_energies_buf is None

    def test_forces_buf_is_none(self):
        model = _make_model()
        assert model._forces_buf is None

    def test_virials_buf_is_none(self):
        model = _make_model()
        assert model._virials_buf is None

    def test_energies_buf_is_none(self):
        model = _make_model()
        assert model._energies_buf is None

    def test_model_config_is_model_config_instance(self):
        model = _make_model()
        assert isinstance(model.model_config, ModelConfig)

    def test_model_config_active_outputs_forces_default_true(self):
        model = _make_model()
        assert "forces" in model.model_config.active_outputs

    def test_model_config_active_outputs_stresses_default_true(self):
        model = _make_model()
        # active_outputs defaults to outputs = {"energy", "forces"}
        assert "stress" not in model.model_config.active_outputs


# ---------------------------------------------------------------------------
# TestModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    """Tests for model_config and embedding_shapes."""

    def test_model_config_returns_model_config(self):
        model = _make_model()
        assert isinstance(model.model_config, ModelConfig)

    def test_forces_not_via_autograd(self):
        model = _make_model()
        assert "forces" not in model.model_config.autograd_outputs

    def test_neighbor_config_is_matrix_format(self):
        model = _make_model()
        assert model.model_config.neighbor_config is not None
        assert model.model_config.neighbor_config.format == NeighborListFormat.MATRIX

    def test_neighbor_config_cutoff_matches_constructor(self):
        model = LennardJonesModelWrapper(epsilon=0.1, sigma=3.0, cutoff=9.0)
        assert model.model_config.neighbor_config.cutoff == 9.0

    def test_supports_stresses_true(self):
        model = _make_model()
        assert "stress" in model.model_config.outputs

    def test_supports_pbc_true(self):
        model = _make_model()
        assert model.model_config.supports_pbc is True

    def test_embedding_shapes_empty_dict(self):
        model = _make_model()
        assert model.embedding_shapes == {}


# ---------------------------------------------------------------------------
# TestEnsureComputeBuffers
# ---------------------------------------------------------------------------


class TestEnsureComputeBuffers:
    """Tests for _ensure_compute_buffers()."""

    def test_allocates_buffers_on_first_call(self):
        model = _make_model()
        assert model._atomic_energies_buf is None
        model._ensure_compute_buffers(
            N=4, B=1, dtype=torch.float32, device=torch.device("cpu")
        )
        assert isinstance(model._atomic_energies_buf, torch.Tensor)
        assert isinstance(model._forces_buf, torch.Tensor)
        assert isinstance(model._virials_buf, torch.Tensor)
        assert isinstance(model._energies_buf, torch.Tensor)

    def test_buffer_shapes_after_first_call(self):
        model = _make_model()
        model._ensure_compute_buffers(
            N=6, B=2, dtype=torch.float32, device=torch.device("cpu")
        )
        assert model._atomic_energies_buf.shape == (6,)
        assert model._forces_buf.shape == (6, 3)
        assert model._virials_buf.shape == (2, 9)
        assert model._energies_buf.shape == (2,)

    def test_no_realloc_when_shapes_unchanged(self):
        model = _make_model()
        model._ensure_compute_buffers(
            N=4, B=1, dtype=torch.float32, device=torch.device("cpu")
        )
        original_ae = model._atomic_energies_buf
        original_f = model._forces_buf
        model._ensure_compute_buffers(
            N=4, B=1, dtype=torch.float32, device=torch.device("cpu")
        )
        assert model._atomic_energies_buf is original_ae
        assert model._forces_buf is original_f

    def test_reallocates_when_N_changes(self):
        model = _make_model()
        model._ensure_compute_buffers(
            N=4, B=1, dtype=torch.float32, device=torch.device("cpu")
        )
        original_ae = model._atomic_energies_buf
        model._ensure_compute_buffers(
            N=8, B=1, dtype=torch.float32, device=torch.device("cpu")
        )
        assert model._atomic_energies_buf is not original_ae
        assert model._atomic_energies_buf.shape == (8,)


# ---------------------------------------------------------------------------
# TestAdaptInput
# ---------------------------------------------------------------------------


class TestAdaptInput:
    """Tests for adapt_input()."""

    def test_raises_type_error_for_atomic_data(self):
        """adapt_input raises TypeError when given an AtomicData rather than a Batch.

        The wrapper iterates over input_data() keys before the isinstance
        guard, so we must supply all required keys (positions, atomic_numbers,
        neighbor_matrix, num_neighbors) to reach the isinstance check.
        """
        model = _make_model()
        n = 4
        k = 8
        atomic_data = AtomicData(
            positions=torch.randn(n, 3),
            atomic_numbers=torch.ones(n, dtype=torch.int64),
        )
        # Attach the neighbor-matrix fields so the key-loop succeeds and the
        # isinstance(data, Batch) branch is reached.
        atomic_data.add_node_property(
            "neighbor_matrix", torch.full((n, k), n, dtype=torch.int32)
        )
        atomic_data.add_node_property(
            "num_neighbors", torch.zeros(n, dtype=torch.int32)
        )
        with pytest.raises(TypeError, match="requires a Batch input"):
            model.adapt_input(atomic_data)

    def test_returns_dict_for_batch(self):
        model = _make_model()
        batch = _make_lj_batch()
        result = model.adapt_input(batch)
        assert isinstance(result, dict)

    def test_positions_in_result(self):
        model = _make_model()
        batch = _make_lj_batch()
        result = model.adapt_input(batch)
        assert "positions" in result
        assert isinstance(result["positions"], torch.Tensor)

    def test_atomic_numbers_in_result(self):
        model = _make_model()
        batch = _make_lj_batch()
        result = model.adapt_input(batch)
        assert "atomic_numbers" in result

    def test_neighbor_matrix_in_result(self):
        model = _make_model()
        batch = _make_lj_batch()
        result = model.adapt_input(batch)
        assert "neighbor_matrix" in result

    def test_num_neighbors_in_result(self):
        model = _make_model()
        batch = _make_lj_batch()
        result = model.adapt_input(batch)
        assert "num_neighbors" in result

    def test_batch_idx_in_result(self):
        model = _make_model()
        batch = _make_lj_batch()
        result = model.adapt_input(batch)
        assert "batch_idx" in result
        assert result["batch_idx"].dtype == torch.int32

    def test_ptr_in_result(self):
        model = _make_model()
        batch = _make_lj_batch()
        result = model.adapt_input(batch)
        assert "ptr" in result

    def test_num_graphs_in_result(self):
        model = _make_model()
        batch = _make_lj_batch()
        result = model.adapt_input(batch)
        assert "num_graphs" in result
        assert result["num_graphs"] == 1

    def test_fill_value_in_result(self):
        model = _make_model()
        batch = _make_lj_batch(n_atoms=4)
        result = model.adapt_input(batch)
        assert "fill_value" in result
        assert result["fill_value"] == 4

    def test_cells_none_when_no_cell_attribute(self):
        model = _make_model()
        batch = _make_lj_batch()
        # Ensure cell is not set
        result = model.adapt_input(batch)
        assert result["cells"] is None

    def test_neighbor_matrix_shifts_none_when_no_neighbor_matrix_shifts(self):
        model = _make_model()
        batch = _make_lj_batch()
        result = model.adapt_input(batch)
        assert result["neighbor_matrix_shifts"] is None

    def test_cells_returned_when_present(self):
        model = _make_model()
        batch = _make_lj_batch()
        cell = torch.eye(3).unsqueeze(0)
        batch.cell = cell
        result = model.adapt_input(batch)
        assert result["cells"] is not None

    def test_neighbor_matrix_shifts_returned_when_present(self):
        model = _make_model()
        batch = _make_lj_batch(n_atoms=4, max_neighbors=8)
        shifts = torch.zeros(4, 8, 3, dtype=torch.int32)
        object.__setattr__(batch, "neighbor_matrix_shifts", shifts)
        result = model.adapt_input(batch)
        assert result["neighbor_matrix_shifts"] is not None


# ---------------------------------------------------------------------------
# TestAdaptOutput
# ---------------------------------------------------------------------------


class TestAdaptOutput:
    """Tests for adapt_output()."""

    def _model_output(
        self, include_virials: bool = False, include_stresses: bool = False
    ):
        output = {
            "energy": torch.tensor([[1.0]]),
            "forces": torch.randn(4, 3),
        }
        if include_virials:
            output["virial"] = torch.randn(1, 3, 3)
        if include_stresses:
            output["stress"] = torch.randn(1, 3, 3)
        return output

    def test_energies_always_in_output(
        self,
    ):
        model = _make_model()
        batch = _make_lj_batch()
        result = model.adapt_output(self._model_output(include_stresses=True), batch)
        assert "energy" in result

    def test_forces_in_output_when_compute_forces_true(self):
        model = _make_model()
        model.model_config.active_outputs = {"energy", "forces"}
        batch = _make_lj_batch()
        result = model.adapt_output(self._model_output(), batch)
        assert "forces" in result

    def test_forces_not_in_output_when_compute_forces_false(self):
        model = _make_model()
        model.model_config.active_outputs = {"energy"}
        batch = _make_lj_batch()
        result = model.adapt_output(self._model_output(), batch)
        assert "forces" not in result

    def test_stresses_not_in_output_when_compute_stresses_false(self):
        model = _make_model()
        model.model_config.active_outputs = {"energy", "forces"}
        batch = _make_lj_batch()
        result = model.adapt_output(self._model_output(include_virials=True), batch)
        assert "stress" not in result

    def test_stresses_equal_virial_over_volume(self):
        model = _make_model()
        model.model_config.active_outputs = {"energy", "forces", "stress"}
        batch = _make_lj_batch()
        # Add cell so volume can be computed.
        batch.cell = torch.eye(3).unsqueeze(0)
        batch.pbc = torch.ones(1, 3, dtype=torch.bool)
        virials = torch.randn(1, 3, 3)
        mo = self._model_output()
        mo["virial"] = virials
        result = model.adapt_output(mo, batch)
        assert "stress" in result
        volume = torch.det(batch.cell).abs().view(-1, 1, 1)
        assert torch.allclose(result["stress"], virials / volume)

    def test_stresses_is_stresses_when_no_virials_key(self):
        model = _make_model()
        model.model_config.active_outputs = {"energy", "forces", "stress"}
        batch = _make_lj_batch()
        stresses = torch.randn(1, 3, 3)
        mo = self._model_output()
        mo["stress"] = stresses
        result = model.adapt_output(mo, batch)
        assert "stress" in result
        assert torch.allclose(result["stress"], stresses)

    def test_adapt_output_stress_raises_without_cell(self):
        model = _make_model()
        model.model_config.active_outputs = {"energy", "forces", "stress"}
        batch = _make_lj_batch()  # no cell
        mo = self._model_output()
        mo["virial"] = torch.randn(1, 3, 3)
        with pytest.raises(ValueError, match="stress output requires cell"):
            model.adapt_output(mo, batch)

    def test_adapt_output_stress_raises_when_missing(self):
        """RuntimeError when stress is active but model_output has neither virial nor stress."""
        model = _make_model()
        model.model_config.active_outputs = {"energy", "forces", "stress"}
        batch = _make_lj_batch()
        batch.cell = torch.eye(3).unsqueeze(0)
        mo = self._model_output()
        with pytest.raises(RuntimeError, match="missing from model output"):
            model.adapt_output(mo, batch)


# ---------------------------------------------------------------------------
# TestOutputData
# ---------------------------------------------------------------------------


class TestOutputData:
    """Tests for output_data()."""

    def test_energies_always_in_output_data(self):
        model = _make_model()
        assert "energy" in model.output_data()

    def test_forces_in_output_data_when_compute_forces_true(self):
        model = _make_model()
        model.model_config.active_outputs = {"energy", "forces"}
        assert "forces" in model.output_data()

    def test_stresses_in_output_data_when_compute_stresses_true(self):
        model = _make_model()
        model.model_config.active_outputs = {"energy", "forces", "stress"}
        assert "stress" in model.output_data()

    def test_stresses_not_in_output_data_when_compute_stresses_false(self):
        model = _make_model()
        model.model_config.active_outputs = {"energy", "forces"}
        assert "stress" not in model.output_data()


# ---------------------------------------------------------------------------
# TestNotImplemented
# ---------------------------------------------------------------------------


class TestNotImplemented:
    """Verify methods that must raise NotImplementedError."""

    def test_compute_embeddings_raises(self):
        model = _make_model()
        batch = _make_lj_batch()
        with pytest.raises(NotImplementedError):
            model.compute_embeddings(batch)

    def test_export_model_raises(self):
        from pathlib import Path

        model = _make_model()
        with pytest.raises(NotImplementedError):
            model.export_model(Path("/tmp/fake.model"))  # noqa: S108


# ---------------------------------------------------------------------------
# TestOpsHelpers
# ---------------------------------------------------------------------------


class TestOpsHelpers:
    """Tests for private helpers in nvalchemi/models/_ops/lj.py.

    Skipped entirely when warp is not available.
    """

    @pytest.fixture(autouse=True)
    def require_warp(self):
        pytest.importorskip("warp")

    def test_vec_type_float32(self):
        import warp as wp

        from nvalchemi.models._ops.lj import _vec_type

        assert _vec_type(torch.float32) is wp.vec3f

    def test_vec_type_float64(self):
        import warp as wp

        from nvalchemi.models._ops.lj import _vec_type

        assert _vec_type(torch.float64) is wp.vec3d

    def test_mat_type_float32(self):
        import warp as wp

        from nvalchemi.models._ops.lj import _mat_type

        assert _mat_type(torch.float32) is wp.mat33f

    def test_mat_type_float64(self):
        import warp as wp

        from nvalchemi.models._ops.lj import _mat_type

        assert _mat_type(torch.float64) is wp.mat33d

    def test_scalar_type_float32(self):
        import warp as wp

        from nvalchemi.models._ops.lj import _scalar_type

        assert _scalar_type(torch.float32) is wp.float32

    def test_scalar_type_float64(self):
        import warp as wp

        from nvalchemi.models._ops.lj import _scalar_type

        assert _scalar_type(torch.float64) is wp.float64

    def test_get_cached_wp_params_returns_dict(self):
        import warp as wp

        from nvalchemi.models._ops.lj import _get_cached_wp_params

        result = _get_cached_wp_params(
            epsilon=1.0,
            sigma=2.5,
            cutoff=8.0,
            switch_width=0.0,
            scl_t=wp.float32,
            wp_dev="cpu",
        )
        assert isinstance(result, dict)
        assert "epsilon" in result
        assert "sigma" in result
        assert "cutoff" in result
        assert "switch" in result

    def test_get_cached_wp_params_caches(self):
        import warp as wp

        from nvalchemi.models._ops.lj import _get_cached_wp_params

        result1 = _get_cached_wp_params(
            epsilon=1.0,
            sigma=2.5,
            cutoff=8.0,
            switch_width=0.0,
            scl_t=wp.float32,
            wp_dev="cpu",
        )
        result2 = _get_cached_wp_params(
            epsilon=1.0,
            sigma=2.5,
            cutoff=8.0,
            switch_width=0.0,
            scl_t=wp.float32,
            wp_dev="cpu",
        )
        assert result1 is result2
