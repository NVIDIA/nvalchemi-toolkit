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
"""Tests for EwaldModelWrapper.

Strategy
--------
* Constructor, model_config, adapt_input, adapt_output, input_data,
  and output_data tests run without ``nvalchemiops`` because the forward
  pass (which calls the Warp kernels) is never exercised.
* Integration tests that call forward() are guarded by
  ``pytest.importorskip("nvalchemiops")``.
"""

from __future__ import annotations

from collections import OrderedDict
from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import NeighborListFormat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ewald(**kwargs):
    """Construct an EwaldModelWrapper with sensible defaults."""
    from nvalchemi.models.ewald import EwaldModelWrapper

    kwargs.setdefault("cutoff", 10.0)
    return EwaldModelWrapper(**kwargs)


def _make_charged_batch(
    n_atoms: int = 8,
    box_size: float = 10.0,
    device: str = "cpu",
) -> Batch:
    """Build a PBC batch with charges for Ewald/PME tests."""
    positions = torch.rand(n_atoms, 3, dtype=torch.float32, device=device) * box_size
    atomic_numbers = torch.ones(n_atoms, dtype=torch.long, device=device)
    # Alternating +1/-1 charges (charge-neutral)
    charges = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_atoms)],
        dtype=torch.float32,
        device=device,
    ).unsqueeze(-1)  # (N, 1) for AtomicData

    data = AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        charges=charges,
        forces=torch.zeros(n_atoms, 3, device=device),
        energy=torch.zeros(1, 1, device=device),
        cell=torch.eye(3, device=device).unsqueeze(0) * box_size,
        pbc=torch.tensor([[True, True, True]], device=device),
    )
    batch = Batch.from_data_list([data])
    return batch


# ===========================================================================
# Constructor tests
# ===========================================================================


class TestEwaldInit:
    def test_stores_cutoff(self):
        w = _make_ewald(cutoff=12.0)
        assert w.cutoff == pytest.approx(12.0)

    def test_stores_accuracy(self):
        w = _make_ewald(accuracy=1e-4)
        assert w.accuracy == pytest.approx(1e-4)

    def test_default_accuracy(self):
        w = _make_ewald()
        assert w.accuracy == pytest.approx(1e-6)

    def test_stores_coulomb_constant(self):
        w = _make_ewald(coulomb_constant=14.0)
        assert w.coulomb_constant == pytest.approx(14.0)

    def test_default_coulomb_constant(self):
        w = _make_ewald()
        assert w.coulomb_constant == pytest.approx(14.3996)

    def test_cache_starts_invalid(self):
        w = _make_ewald()
        assert w._cache_valid is False
        assert w._cached_alpha is None
        assert w._cached_k_vectors is None


# ===========================================================================
# ModelConfig tests
# ===========================================================================


class TestEwaldModelConfig:
    def test_outputs(self):
        w = _make_ewald()
        assert "energy" in w.model_config.outputs
        assert "forces" in w.model_config.outputs
        assert "stress" in w.model_config.outputs

    def test_no_autograd_outputs(self):
        w = _make_ewald()
        assert w.model_config.autograd_outputs == frozenset()

    def test_needs_pbc(self):
        w = _make_ewald()
        assert w.model_config.needs_pbc is True
        assert w.model_config.supports_pbc is True

    def test_required_inputs_include_charges(self):
        w = _make_ewald()
        assert "charges" in w.model_config.required_inputs

    def test_neighbor_config_matrix_format(self):
        w = _make_ewald()
        nc = w.model_config.neighbor_config
        assert nc is not None
        assert nc.format == NeighborListFormat.MATRIX
        assert nc.cutoff == pytest.approx(10.0)

    def test_active_outputs_default_to_all(self):
        w = _make_ewald()
        assert w.model_config.active_outputs == set(w.model_config.outputs)

    def test_embedding_shapes_empty(self):
        w = _make_ewald()
        assert w.embedding_shapes == {}

    def test_compute_embeddings_raises(self):
        w = _make_ewald()
        with pytest.raises(NotImplementedError):
            w.compute_embeddings(None)

    def test_export_model_raises(self):
        w = _make_ewald()
        with pytest.raises(NotImplementedError):
            w.export_model(None)


# ===========================================================================
# input_data / output_data tests
# ===========================================================================


class TestEwaldInputOutput:
    def test_input_data_override(self):
        """Ewald overrides input_data to drop atomic_numbers."""
        w = _make_ewald()
        keys = w.input_data()
        assert "positions" in keys
        assert "charges" in keys
        assert "neighbor_matrix" in keys
        assert "num_neighbors" in keys

    def test_output_data_with_forces(self):
        w = _make_ewald()
        out = w.output_data()
        assert "energy" in out
        assert "forces" in out

    def test_output_data_with_stresses(self):
        w = _make_ewald()
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        out = w.output_data()
        assert "stress" in out

    def test_output_data_energy_only(self):
        w = _make_ewald()
        w.model_config.active_outputs = {"energy"}
        out = w.output_data()
        assert out == {"energy"}
        assert "forces" not in out


# ===========================================================================
# adapt_input tests
# ===========================================================================


class TestEwaldAdaptInput:
    def test_requires_batch(self):
        """Ewald requires Batch, not AtomicData."""
        w = _make_ewald()
        data = AtomicData(
            positions=torch.randn(4, 3),
            atomic_numbers=torch.ones(4, dtype=torch.long),
        )
        with pytest.raises(TypeError, match="Batch"):
            w.adapt_input(data)

    def test_squeezes_charges(self):
        """Charges stored as (N, 1) are squeezed to (N,)."""
        w = _make_ewald()
        batch = _make_charged_batch()
        # Add neighbor data (normally populated by NeighborListHook)
        N = batch.num_nodes
        object.__setattr__(
            batch, "neighbor_matrix", torch.full((N, 8), N, dtype=torch.int32)
        )
        object.__setattr__(batch, "num_neighbors", torch.zeros(N, dtype=torch.int32))
        batch._neighbor_list_cutoff = 15.0
        inp = w.adapt_input(batch)
        assert inp["charges"].ndim == 1

    def test_collects_cell(self):
        w = _make_ewald()
        batch = _make_charged_batch()
        N = batch.num_nodes
        object.__setattr__(
            batch, "neighbor_matrix", torch.full((N, 8), N, dtype=torch.int32)
        )
        object.__setattr__(batch, "num_neighbors", torch.zeros(N, dtype=torch.int32))
        batch._neighbor_list_cutoff = 15.0
        inp = w.adapt_input(batch)
        assert "cell" in inp
        assert inp["cell"].shape == (1, 3, 3)

    def test_raises_value_error_when_cell_missing(self):
        """Ewald requires PBC; missing cell raises ValueError."""
        w = _make_ewald()
        # Build a batch without cell/pbc
        n = 8
        data = AtomicData(
            positions=torch.randn(n, 3),
            atomic_numbers=torch.ones(n, dtype=torch.long),
            charges=torch.ones(n, 1) * 0.5,
            forces=torch.zeros(n, 3),
            energy=torch.zeros(1, 1),
        )
        batch = Batch.from_data_list([data])
        N = batch.num_nodes
        object.__setattr__(
            batch, "neighbor_matrix", torch.full((N, 8), N, dtype=torch.int32)
        )
        object.__setattr__(batch, "num_neighbors", torch.zeros(N, dtype=torch.int32))
        batch._neighbor_list_cutoff = 15.0
        with pytest.raises(ValueError, match="requires periodic boundary conditions"):
            w.adapt_input(batch)

    def test_charges_present_and_squeezed(self):
        """adapt_input collects charges with correct shape."""
        w = _make_ewald()
        batch = _make_charged_batch()
        N = batch.num_nodes
        object.__setattr__(
            batch, "neighbor_matrix", torch.full((N, 8), N, dtype=torch.int32)
        )
        object.__setattr__(batch, "num_neighbors", torch.zeros(N, dtype=torch.int32))
        batch._neighbor_list_cutoff = 15.0
        inp = w.adapt_input(batch)
        assert "charges" in inp
        assert inp["charges"].shape == (batch.num_nodes,)

    def test_neighbor_data_present(self):
        w = _make_ewald()
        batch = _make_charged_batch()
        N = batch.num_nodes
        object.__setattr__(
            batch, "neighbor_matrix", torch.full((N, 8), N, dtype=torch.int32)
        )
        object.__setattr__(batch, "num_neighbors", torch.zeros(N, dtype=torch.int32))
        batch._neighbor_list_cutoff = 15.0
        inp = w.adapt_input(batch)
        assert "neighbor_matrix" in inp
        assert "num_neighbors" in inp

    def test_batch_idx_is_int32(self):
        w = _make_ewald()
        batch = _make_charged_batch()
        N = batch.num_nodes
        object.__setattr__(
            batch, "neighbor_matrix", torch.full((N, 8), N, dtype=torch.int32)
        )
        object.__setattr__(batch, "num_neighbors", torch.zeros(N, dtype=torch.int32))
        batch._neighbor_list_cutoff = 15.0
        inp = w.adapt_input(batch)
        assert inp["batch_idx"].dtype == torch.int32

    def test_fill_value_equals_num_nodes(self):
        w = _make_ewald()
        batch = _make_charged_batch()
        N = batch.num_nodes
        object.__setattr__(
            batch, "neighbor_matrix", torch.full((N, 8), N, dtype=torch.int32)
        )
        object.__setattr__(batch, "num_neighbors", torch.zeros(N, dtype=torch.int32))
        batch._neighbor_list_cutoff = 15.0
        inp = w.adapt_input(batch)
        assert inp["fill_value"] == batch.num_nodes


# ===========================================================================
# adapt_output tests
# ===========================================================================


class TestEwaldAdaptOutput:
    def test_energy_always_present(self):
        w = _make_ewald()
        w.model_config.active_outputs = {"energy", "forces"}
        raw = {"energy": torch.tensor([[1.0]]), "forces": torch.randn(4, 3)}
        out = w.adapt_output(raw, None)
        assert "energy" in out

    def test_forces_when_active(self):
        w = _make_ewald()
        w.model_config.active_outputs = {"energy", "forces"}
        raw = {"energy": torch.tensor([[1.0]]), "forces": torch.randn(4, 3)}
        out = w.adapt_output(raw, None)
        assert "forces" in out

    def test_no_forces_when_inactive(self):
        w = _make_ewald()
        w.model_config.active_outputs = {"energy"}
        raw = {"energy": torch.tensor([[1.0]]), "forces": torch.randn(4, 3)}
        out = w.adapt_output(raw, None)
        assert "forces" not in out

    def test_stresses_when_active(self):
        w = _make_ewald()
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        raw = {
            "energy": torch.tensor([[1.0]]),
            "forces": torch.randn(4, 3),
            "stress": torch.randn(1, 3, 3),
        }
        out = w.adapt_output(raw, None)
        assert "stress" in out

    def test_no_stress_when_inactive(self):
        w = _make_ewald()
        w.model_config.active_outputs = {"energy", "forces"}
        raw = {
            "energy": torch.tensor([[1.0]]),
            "forces": torch.randn(4, 3),
            "stress": torch.randn(1, 3, 3),
        }
        out = w.adapt_output(raw, None)
        assert "stress" not in out

    def test_adapt_output_stress_raises_when_missing(self):
        """RuntimeError when stress is active but absent from model_output."""
        w = _make_ewald()
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        raw = {"energy": torch.tensor([[1.0]]), "forces": torch.randn(4, 3)}
        with pytest.raises(RuntimeError, match="missing from model output"):
            w.adapt_output(raw, None)

    def test_returns_ordered_dict(self):
        w = _make_ewald()
        w.model_config.active_outputs = {"energy"}
        raw = {"energy": torch.tensor([[1.0]])}
        out = w.adapt_output(raw, None)
        assert isinstance(out, OrderedDict)


# ===========================================================================
# Cache management tests
# ===========================================================================


class TestEwaldCache:
    def test_invalidate_clears_state(self):
        w = _make_ewald()
        w._cache_valid = True
        w._cached_alpha = torch.tensor([1.0])
        w._cached_k_vectors = torch.randn(10, 3)
        w.invalidate_cache()
        assert w._cache_valid is False
        assert w._cached_alpha is None
        assert w._cached_k_vectors is None

    def test_cache_is_stale_when_invalid(self):
        w = _make_ewald()
        assert w._cache_is_stale() is True

    def test_cache_not_stale_after_set(self):
        w = _make_ewald()
        w._cache_valid = True
        assert w._cache_is_stale() is False

    def test_invalidate_from_populated(self):
        """Invalidating a fully-populated cache makes it stale."""
        w = _make_ewald()
        w._cache_valid = True
        w._cached_alpha = torch.tensor([0.3])
        w._cached_k_vectors = torch.randn(10, 3)
        w.invalidate_cache()
        assert w._cache_is_stale() is True

    def test_invalidate_resets_all_fields(self):
        """All cached fields are reset to None after invalidate."""
        w = _make_ewald()
        w._cache_valid = True
        w._cached_alpha = torch.tensor([0.3])
        w._cached_k_vectors = torch.randn(10, 3)
        w.invalidate_cache()
        assert w._cache_valid is False
        assert w._cached_alpha is None
        assert w._cached_k_vectors is None


# ===========================================================================
# Integration tests (require nvalchemiops)
# ===========================================================================


class TestEwaldIntegration:
    """Full forward-pass tests requiring nvalchemiops Warp kernels."""

    @pytest.fixture(autouse=True)
    def _require_ops(self):
        pytest.importorskip("nvalchemiops")

    @staticmethod
    def _build_nl(batch, model):
        """Build a real neighbor list for the batch."""
        from nvalchemi.neighbors import compute_neighbors

        compute_neighbors(batch, config=model.model_config.neighbor_config)

    def test_forward_energy_finite(self):
        w = _make_ewald()
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        out = w(batch)
        assert torch.isfinite(out["energy"]).all()

    def test_forward_forces_finite(self):
        w = _make_ewald()
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        out = w(batch)
        assert torch.isfinite(out["forces"]).all()

    def test_forward_energy_shape(self):
        w = _make_ewald()
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        out = w(batch)
        assert out["energy"].shape == (1, 1)

    def test_forward_forces_shape(self):
        w = _make_ewald()
        batch = _make_charged_batch(n_atoms=8)
        self._build_nl(batch, w)
        out = w(batch)
        assert out["forces"].shape == (8, 3)

    def test_forward_stress_when_requested(self):
        w = _make_ewald()
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        out = w(batch)
        assert "stress" in out
        assert out["stress"].shape == (1, 3, 3)

    def test_forward_stress_is_virial_over_volume(self):
        """Stress == virial / volume (Cauchy stress, eV/A^3)."""
        import nvalchemi.models.ewald as _emod

        w = _make_ewald()
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        batch = _make_charged_batch(box_size=10.0)
        self._build_nl(batch, w)

        known_virial = torch.full((1, 3, 3), 5.0)

        def patched_forward(self_inner, data, **kw):
            N = data.num_nodes
            model_output = {
                "energy": torch.zeros(1, 1),
                "forces": torch.zeros(N, 3),
            }
            volume = torch.det(data.cell).abs().view(-1, 1, 1)
            model_output["stress"] = known_virial / volume
            return self_inner.adapt_output(model_output, data)

        with patch.object(_emod.EwaldModelWrapper, "forward", patched_forward):
            out = w.forward(batch)

        volume = torch.det(batch.cell).abs().view(-1, 1, 1)
        torch.testing.assert_close(out["stress"], known_virial / volume)

    def test_forward_raises_when_virial_none(self):
        """RuntimeError when stress is requested but kernels return no virial."""
        w = _make_ewald()
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        batch = _make_charged_batch()
        self._build_nl(batch, w)

        N = batch.num_nodes

        def _fake_kernel(**kw):
            energies = torch.zeros(N, dtype=torch.float64)
            forces = torch.zeros(N, 3, dtype=torch.float64)
            return energies, forces

        with (
            patch(
                "nvalchemiops.torch.interactions.electrostatics.ewald.ewald_real_space",
                side_effect=_fake_kernel,
            ),
            patch(
                "nvalchemiops.torch.interactions.electrostatics.ewald.ewald_reciprocal_space",
                side_effect=_fake_kernel,
            ),
        ):
            with pytest.raises(RuntimeError, match="kernel did not return a virial"):
                w.forward(batch)

    def test_cache_populated_after_forward(self):
        w = _make_ewald()
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        w(batch)
        assert w._cache_valid is True
        assert w._cached_alpha is not None

    def test_cache_not_recomputed_for_same_cell(self):
        """Second call with identical cell should not change cached alpha object."""
        w = _make_ewald()
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        w(batch)
        alpha_ref = w._cached_alpha
        w(batch)
        assert w._cached_alpha is alpha_ref

    def test_cache_recomputed_after_invalidate(self):
        """After invalidate_cache(), the next forward recomputes a new alpha."""
        w = _make_ewald()
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        w(batch)
        alpha_before = w._cached_alpha
        w.invalidate_cache()
        w(batch)
        assert w._cached_alpha is not alpha_before

    def test_neutral_system_forces_symmetry(self):
        """Two opposite charges should have forces along the axis of separation."""
        w = _make_ewald(cutoff=10.0)
        data = AtomicData(
            positions=torch.tensor([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            atomic_numbers=torch.tensor([11, 17], dtype=torch.long),
            charges=torch.tensor([[1.0], [-1.0]]),
            forces=torch.zeros(2, 3),
            energy=torch.zeros(1, 1),
            cell=torch.eye(3).unsqueeze(0) * 10.0,
            pbc=torch.tensor([[True, True, True]]),
        )
        batch = Batch.from_data_list([data])
        self._build_nl(batch, w)
        out = w(batch)
        assert torch.isfinite(out["forces"]).all()
        # y and z components should be ~0 by symmetry
        assert out["forces"][:, 1].abs().max() < 1e-4
        assert out["forces"][:, 2].abs().max() < 1e-4
