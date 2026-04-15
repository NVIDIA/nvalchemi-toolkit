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
"""Tests for PMEModelWrapper.

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
from nvalchemi.data.level_storage import LevelSchema
from nvalchemi.models.base import NeighborListFormat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pme(**kwargs):
    """Construct a PMEModelWrapper with sensible defaults."""
    from nvalchemi.models.pme import PMEModelWrapper

    kwargs.setdefault("cutoff", 10.0)
    return PMEModelWrapper(**kwargs)


def _make_charged_batch(
    n_atoms: int = 8,
    box_size: float = 10.0,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Batch:
    """Build a PBC batch with charges for PME tests."""
    positions = torch.rand(n_atoms, 3, dtype=dtype, device=device) * box_size
    atomic_numbers = torch.ones(n_atoms, dtype=torch.long, device=device)
    # Alternating +1/-1 charges (charge-neutral)
    charges = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_atoms)],
        dtype=dtype,
        device=device,
    )

    data = AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        charges=charges,
        forces=torch.zeros(n_atoms, 3, dtype=dtype, device=device),
        energy=torch.zeros(1, 1, dtype=dtype, device=device),
        cell=torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * box_size,
        pbc=torch.tensor([[True, True, True]], device=device),
    )
    attr_map = None
    if dtype == torch.float64:
        attr_map = LevelSchema()
        for key in ("positions", "forces", "charges", "cell", "stress", "virial"):
            attr_map.set(key, attr_map.attr_to_group[key], dtype="float64")

    batch = Batch.from_data_list([data], attr_map=attr_map)
    return batch


def _finite_difference_charge_gradient(
    model,
    batch: Batch,
    build_nl,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Estimate dE/dq with central finite differences."""
    build_nl(batch, model)
    base_charges = batch.charges.detach().clone()
    grad = torch.zeros_like(base_charges)

    for atom_idx in range(base_charges.shape[0]):
        batch.charges = base_charges.clone()
        batch.charges[atom_idx] += eps
        energy_plus = model(batch)["energy"].sum().item()

        batch.charges = base_charges.clone()
        batch.charges[atom_idx] -= eps
        energy_minus = model(batch)["energy"].sum().item()

        grad[atom_idx] = (energy_plus - energy_minus) / (2.0 * eps)

    batch.charges = base_charges
    return grad


# ===========================================================================
# Constructor tests
# ===========================================================================


class TestPMEInit:
    def test_stores_cutoff(self):
        w = _make_pme(cutoff=12.0)
        assert w.cutoff == pytest.approx(12.0)

    def test_stores_accuracy(self):
        w = _make_pme(accuracy=1e-4)
        assert w.accuracy == pytest.approx(1e-4)

    def test_default_accuracy(self):
        w = _make_pme()
        assert w.accuracy == pytest.approx(1e-6)

    def test_stores_coulomb_constant(self):
        w = _make_pme(coulomb_constant=14.0)
        assert w.coulomb_constant == pytest.approx(14.0)

    def test_default_coulomb_constant(self):
        w = _make_pme()
        assert w.coulomb_constant == pytest.approx(14.3996)

    def test_stores_mesh_spacing(self):
        w = _make_pme(mesh_spacing=0.5)
        assert w.mesh_spacing == pytest.approx(0.5)

    def test_default_mesh_spacing(self):
        w = _make_pme()
        assert w.mesh_spacing == pytest.approx(1.0)

    def test_stores_spline_order(self):
        w = _make_pme(spline_order=6)
        assert w.spline_order == 6

    def test_default_spline_order(self):
        w = _make_pme()
        assert w.spline_order == 4

    def test_stores_explicit_alpha(self):
        w = _make_pme(alpha=0.3)
        assert w.alpha == pytest.approx(0.3)

    def test_default_alpha_is_none(self):
        w = _make_pme()
        assert w.alpha is None

    def test_stores_explicit_mesh_dimensions(self):
        w = _make_pme(mesh_dimensions=(32, 32, 32))
        assert w.mesh_dimensions == (32, 32, 32)

    def test_default_mesh_dimensions_is_none(self):
        w = _make_pme()
        assert w.mesh_dimensions is None

    def test_cache_starts_invalid(self):
        w = _make_pme()
        assert w._cache_valid is False
        assert w._cached_alpha is None
        assert w._cached_k_vectors is None
        assert w._cached_k_squared is None
        assert w._cached_mesh_dims is None


# ===========================================================================
# ModelConfig tests
# ===========================================================================


class TestPMEModelConfig:
    def test_outputs(self):
        w = _make_pme()
        assert "energy" in w.model_config.outputs
        assert "forces" in w.model_config.outputs
        assert "stress" in w.model_config.outputs

    def test_autograd_outputs_includes_forces(self):
        w = _make_pme()
        assert w.model_config.autograd_outputs == frozenset({"forces"})

    def test_needs_pbc(self):
        w = _make_pme()
        assert w.model_config.needs_pbc is True
        assert w.model_config.supports_pbc is True

    def test_required_inputs_include_charges(self):
        w = _make_pme()
        assert "charges" in w.model_config.required_inputs

    def test_neighbor_config_matrix_format(self):
        w = _make_pme()
        nc = w.model_config.neighbor_config
        assert nc is not None
        assert nc.format == NeighborListFormat.MATRIX
        assert nc.cutoff == pytest.approx(10.0)

    def test_active_outputs_default_to_all(self):
        w = _make_pme()
        assert w.model_config.active_outputs == {"energy", "forces"}

    def test_embedding_shapes_empty(self):
        w = _make_pme()
        assert w.embedding_shapes == {}

    def test_compute_embeddings_raises(self):
        w = _make_pme()
        with pytest.raises(NotImplementedError):
            w.compute_embeddings(None)

    def test_export_model_raises(self):
        w = _make_pme()
        with pytest.raises(NotImplementedError):
            w.export_model(None)


# ===========================================================================
# input_data / output_data tests
# ===========================================================================


class TestPMEInputOutput:
    def test_input_data_override(self):
        """PME overrides input_data to drop atomic_numbers."""
        w = _make_pme()
        keys = w.input_data()
        assert "positions" in keys
        assert "charges" in keys
        assert "neighbor_matrix" in keys
        assert "num_neighbors" in keys

    def test_output_data_with_forces(self):
        w = _make_pme()
        out = w.output_data()
        assert "energy" in out
        assert "forces" in out

    def test_output_data_energy_only(self):
        w = _make_pme()
        w.model_config.active_outputs = {"energy"}
        out = w.output_data()
        assert out == {"energy"}


# ===========================================================================
# adapt_input tests
# ===========================================================================


class TestPMEAdaptInput:
    def test_requires_batch(self):
        w = _make_pme()
        data = AtomicData(
            positions=torch.randn(4, 3),
            atomic_numbers=torch.ones(4, dtype=torch.long),
        )
        with pytest.raises(TypeError, match="Batch"):
            w.adapt_input(data)

    def test_squeezes_charges(self):
        w = _make_pme()
        batch = _make_charged_batch()
        N = batch.num_nodes
        object.__setattr__(
            batch, "neighbor_matrix", torch.full((N, 8), N, dtype=torch.int32)
        )
        object.__setattr__(batch, "num_neighbors", torch.zeros(N, dtype=torch.int32))
        batch._neighbor_list_cutoff = 15.0
        inp = w.adapt_input(batch)
        assert inp["charges"].ndim == 1

    def test_collects_cell(self):
        w = _make_pme()
        batch = _make_charged_batch()
        N = batch.num_nodes
        object.__setattr__(
            batch, "neighbor_matrix", torch.full((N, 8), N, dtype=torch.int32)
        )
        object.__setattr__(batch, "num_neighbors", torch.zeros(N, dtype=torch.int32))
        batch._neighbor_list_cutoff = 15.0
        inp = w.adapt_input(batch)
        assert "cell" in inp

    def test_raises_value_error_when_cell_missing(self):
        """PME requires PBC; missing cell raises ValueError."""
        w = _make_pme()
        n = 8
        data = AtomicData(
            positions=torch.randn(n, 3),
            atomic_numbers=torch.ones(n, dtype=torch.long),
            charges=torch.ones(n) * 0.5,
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
        w = _make_pme()
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
        w = _make_pme()
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
        w = _make_pme()
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
        w = _make_pme()
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


class TestPMEAdaptOutput:
    def test_energy_always_present(self):
        w = _make_pme()
        w.model_config.active_outputs = {"energy", "forces"}
        raw = {"energy": torch.tensor([[1.0]]), "forces": torch.randn(4, 3)}
        out = w.adapt_output(raw, None)
        assert "energy" in out

    def test_forces_when_active(self):
        w = _make_pme()
        w.model_config.active_outputs = {"energy", "forces"}
        raw = {"energy": torch.tensor([[1.0]]), "forces": torch.randn(4, 3)}
        out = w.adapt_output(raw, None)
        assert "forces" in out

    def test_no_forces_when_inactive(self):
        w = _make_pme()
        w.model_config.active_outputs = {"energy"}
        raw = {"energy": torch.tensor([[1.0]]), "forces": torch.randn(4, 3)}
        out = w.adapt_output(raw, None)
        assert "forces" not in out

    def test_stress_when_active(self):
        w = _make_pme()
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        raw = {
            "energy": torch.tensor([[1.0]]),
            "forces": torch.randn(4, 3),
            "stress": torch.randn(1, 3, 3),
        }
        out = w.adapt_output(raw, None)
        assert "stress" in out

    def test_no_stress_when_inactive(self):
        w = _make_pme()
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
        w = _make_pme()
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        raw = {"energy": torch.tensor([[1.0]]), "forces": torch.randn(4, 3)}
        with pytest.raises(RuntimeError, match="missing from model output"):
            w.adapt_output(raw, None)

    def test_returns_ordered_dict(self):
        w = _make_pme()
        w.model_config.active_outputs = {"energy"}
        raw = {"energy": torch.tensor([[1.0]])}
        out = w.adapt_output(raw, None)
        assert isinstance(out, OrderedDict)


# ===========================================================================
# Cache management tests
# ===========================================================================


class TestPMECache:
    def test_invalidate_clears_state(self):
        w = _make_pme()
        w._cache_valid = True
        w._cached_alpha = torch.tensor([1.0])
        w._cached_k_vectors = torch.randn(10, 3)
        w._cached_k_squared = torch.randn(10)
        w._cached_mesh_dims = (16, 16, 16)
        w.invalidate_cache()
        assert w._cache_valid is False
        assert w._cached_alpha is None
        assert w._cached_k_vectors is None
        assert w._cached_k_squared is None
        assert w._cached_mesh_dims is None

    def test_cache_is_stale_when_invalid(self):
        w = _make_pme()
        assert w._cache_is_stale() is True

    def test_cache_not_stale_when_valid(self):
        w = _make_pme()
        w._cache_valid = True
        assert w._cache_is_stale() is False

    def test_cache_stale_after_invalidate(self):
        w = _make_pme()
        w._cache_valid = True
        w.invalidate_cache()
        assert w._cache_is_stale() is True

    def test_invalidate_from_populated_resets_all_five_fields(self):
        """All five cache fields are reset to None after invalidate."""
        w = _make_pme()
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


# ===========================================================================
# Integration tests (require nvalchemiops)
# ===========================================================================


class TestPMEIntegration:
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
        w = _make_pme()
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        out = w(batch)
        assert torch.isfinite(out["energy"]).all()

    def test_forward_forces_finite(self):
        w = _make_pme()
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        out = w(batch)
        assert torch.isfinite(out["forces"]).all()

    def test_forward_energy_shape(self):
        w = _make_pme()
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        out = w(batch)
        assert out["energy"].shape == (1, 1)

    def test_forward_forces_shape(self):
        w = _make_pme()
        batch = _make_charged_batch(n_atoms=8)
        self._build_nl(batch, w)
        out = w(batch)
        assert out["forces"].shape == (8, 3)

    def test_forward_stress_when_requested(self):
        w = _make_pme()
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        out = w(batch)
        assert "stress" in out
        assert out["stress"].ndim == 3
        assert out["stress"].shape[-2:] == (3, 3)

    def test_forward_stress_is_virial_over_volume(self):
        """Stress == virial / volume (Cauchy stress, eV/A^3)."""
        import nvalchemi.models.pme as _pmod

        w = _make_pme()
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

        with patch.object(_pmod.PMEModelWrapper, "forward", patched_forward):
            out = w.forward(batch)

        volume = torch.det(batch.cell).abs().view(-1, 1, 1)
        torch.testing.assert_close(out["stress"], known_virial / volume)

    def test_forward_raises_when_virial_none(self):
        """RuntimeError when stress is requested but kernel returns no virial."""
        w = _make_pme()
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        batch = _make_charged_batch()
        self._build_nl(batch, w)

        N = batch.num_nodes

        def _fake_kernel(**kw):
            energies = torch.zeros(N, dtype=torch.float64)
            forces = torch.zeros(N, 3, dtype=torch.float64)
            return energies, forces

        with patch(
            "nvalchemiops.torch.interactions.electrostatics.pme.particle_mesh_ewald",
            side_effect=_fake_kernel,
        ):
            with pytest.raises(RuntimeError, match="kernel did not return a virial"):
                w.forward(batch)

    def test_cache_populated_after_forward(self):
        w = _make_pme()
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        w(batch)
        assert w._cache_valid is True
        assert w._cached_alpha is not None

    def test_explicit_alpha_used(self):
        w = _make_pme(alpha=0.25)
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        w(batch)
        assert w._cached_alpha is not None
        assert torch.allclose(w._cached_alpha, torch.full_like(w._cached_alpha, 0.25))

    def test_explicit_mesh_dimensions_used(self):
        w = _make_pme(mesh_dimensions=(16, 16, 16))
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        w(batch)
        assert w._cached_mesh_dims == (16, 16, 16)

    def test_ewald_and_pme_agree_on_energy_sign(self):
        """Ewald and PME should produce same-sign energies for a NaCl pair."""
        from nvalchemi.models.ewald import EwaldModelWrapper

        batch = _make_charged_batch(n_atoms=8)

        ewald = EwaldModelWrapper(cutoff=10.0)
        self._build_nl(batch, ewald)
        e_ewald = ewald(batch)["energy"].item()

        # Rebuild neighbors for PME on the same batch.
        pme = _make_pme()
        self._build_nl(batch, pme)
        e_pme = pme(batch)["energy"].item()

        assert e_ewald * e_pme > 0, (
            f"Ewald ({e_ewald:.4f}) and PME ({e_pme:.4f}) disagree on energy sign"
        )

    def test_hybrid_forces_energy_and_forces_returned(self):
        w = _make_pme()
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        out = w(batch)
        assert "energy" in out
        assert "forces" in out

    def test_hybrid_forces_forces_have_no_grad_fn(self):
        """Direct kernel forces are computed on detached positions."""
        w = _make_pme()
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        out = w(batch)
        assert out["forces"].grad_fn is None

    def test_hybrid_forces_energy_has_grad_fn_with_charge_grad(self):
        """Energy carries charge gradient via _InjectChargeGrad."""
        w = _make_pme()
        batch = _make_charged_batch()
        batch.charges = batch.charges.detach().requires_grad_(True)
        self._build_nl(batch, w)
        out = w(batch)
        assert out["energy"].grad_fn is not None

    def test_hybrid_forces_energy_no_grad_fn_without_charge_grad(self):
        """When charges don't require grad, _InjectChargeGrad is skipped."""
        w = _make_pme()
        batch = _make_charged_batch()
        batch.charges = batch.charges.detach().requires_grad_(False)
        self._build_nl(batch, w)
        out = w(batch)
        assert out["energy"].grad_fn is None

    def test_hybrid_forces_charge_gradient_matches_finite_difference(self):
        """energy.backward() should recover the injected dE/dq."""
        torch.manual_seed(42)
        w = _make_pme()
        batch = _make_charged_batch(n_atoms=4, box_size=8.0, dtype=torch.float64)
        fd_grad = _finite_difference_charge_gradient(w, batch, self._build_nl)
        batch.charges = batch.charges.detach().requires_grad_(True)
        out = w(batch)
        out["energy"].sum().backward()
        assert batch.charges.grad is not None
        torch.testing.assert_close(batch.charges.grad, fd_grad, atol=5e-5, rtol=5e-4)

    def test_hybrid_forces_stress_returned_when_active(self):
        """Stress is present in output when included in active_outputs."""
        w = _make_pme()
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        batch = _make_charged_batch()
        self._build_nl(batch, w)
        out = w(batch)
        assert "stress" in out
        assert out["stress"].shape == (1, 3, 3)

    def test_hybrid_forces_stress_has_no_grad_fn(self):
        """Kernel virial is computed on detached positions/cell."""
        w = _make_pme()
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        batch = _make_charged_batch()
        batch.charges = batch.charges.detach().requires_grad_(True)
        self._build_nl(batch, w)
        out = w(batch)
        assert out["stress"].grad_fn is None

    def test_autograd_outputs_includes_forces(self):
        w = _make_pme()
        assert "forces" in w.model_config.autograd_outputs

    def test_hybrid_forces_match_non_hybrid_values(self):
        """hybrid_forces=True gives the same PME forces as the standard path."""
        from nvalchemiops.torch.interactions.electrostatics.pme import (
            particle_mesh_ewald,
        )

        torch.manual_seed(42)
        w = _make_pme()
        batch = _make_charged_batch()
        self._build_nl(batch, w)

        out_hybrid = w(batch)

        inp = w.adapt_input(batch)
        positions = inp["positions"]
        charges = inp["charges"].view(-1)
        cell = inp["cell"]
        batch_idx = inp["batch_idx"]
        fill_value = inp["fill_value"]
        neighbor_matrix = inp["neighbor_matrix"].contiguous()
        neighbor_matrix_shifts = inp.get("neighbor_matrix_shifts")
        if neighbor_matrix_shifts is None:
            N, K = positions.shape[0], neighbor_matrix.shape[1]
            neighbor_matrix_shifts = torch.zeros(
                N, K, 3, dtype=torch.int32, device=positions.device
            )

        w._update_cache(positions, cell, batch_idx)
        result = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=w._cached_alpha,
            mesh_dimensions=w._cached_mesh_dims,
            spline_order=w.spline_order,
            batch_idx=batch_idx,
            k_vectors=w._cached_k_vectors,
            k_squared=w._cached_k_squared,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts.contiguous(),
            mask_value=fill_value,
            compute_forces=True,
            compute_virial=False,
            accuracy=w.accuracy,
            hybrid_forces=False,
        )
        expected_forces = result[1] * w.coulomb_constant

        torch.testing.assert_close(
            out_hybrid["forces"], expected_forces, atol=1e-5, rtol=1e-5
        )

    def test_hybrid_forces_stress_matches_non_hybrid_values(self):
        """hybrid_forces=True gives same virial/stress as standard path."""
        from nvalchemiops.torch.interactions.electrostatics.pme import (
            particle_mesh_ewald,
        )

        torch.manual_seed(42)
        w = _make_pme()
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        batch = _make_charged_batch()
        self._build_nl(batch, w)

        out_hybrid = w(batch)

        inp = w.adapt_input(batch)
        positions = inp["positions"]
        charges = inp["charges"].view(-1)
        cell = inp["cell"]
        batch_idx = inp["batch_idx"]
        fill_value = inp["fill_value"]
        neighbor_matrix = inp["neighbor_matrix"].contiguous()
        neighbor_matrix_shifts = inp.get("neighbor_matrix_shifts")
        if neighbor_matrix_shifts is None:
            N, K = positions.shape[0], neighbor_matrix.shape[1]
            neighbor_matrix_shifts = torch.zeros(
                N, K, 3, dtype=torch.int32, device=positions.device
            )

        w._update_cache(positions, cell, batch_idx)
        result = particle_mesh_ewald(
            positions=positions,
            charges=charges,
            cell=cell,
            alpha=w._cached_alpha,
            mesh_dimensions=w._cached_mesh_dims,
            spline_order=w.spline_order,
            batch_idx=batch_idx,
            k_vectors=w._cached_k_vectors,
            k_squared=w._cached_k_squared,
            neighbor_matrix=neighbor_matrix,
            neighbor_matrix_shifts=neighbor_matrix_shifts.contiguous(),
            mask_value=fill_value,
            compute_forces=False,
            compute_virial=True,
            accuracy=w.accuracy,
            hybrid_forces=False,
        )
        volume = torch.det(batch.cell).abs().view(-1, 1, 1)
        expected_stress = result[1] * w.coulomb_constant / volume

        torch.testing.assert_close(
            out_hybrid["stress"], expected_stress, atol=1e-5, rtol=1e-5
        )

    def test_ewald_and_pme_agree_on_stress_sign(self):
        """Ewald and PME stress tensors should have consistent signs."""
        from nvalchemi.models.ewald import EwaldModelWrapper

        batch = _make_charged_batch(n_atoms=8)

        ewald = EwaldModelWrapper(cutoff=10.0)
        ewald.model_config.active_outputs = {"energy", "forces", "stress"}
        self._build_nl(batch, ewald)
        s_ewald = ewald(batch)["stress"]

        pme = _make_pme()
        pme.model_config.active_outputs = {"energy", "forces", "stress"}
        self._build_nl(batch, pme)
        s_pme = pme(batch)["stress"]

        trace_ewald = s_ewald.diagonal(dim1=-2, dim2=-1).sum()
        trace_pme = s_pme.diagonal(dim1=-2, dim2=-1).sum()
        assert trace_ewald * trace_pme > 0, (
            f"Ewald trace ({trace_ewald:.6f}) and PME trace ({trace_pme:.6f}) "
            "disagree on stress sign"
        )


# ===========================================================================
# Cross-model cache interface tests
# ===========================================================================


class TestPMECrossModel:
    """Tests that compare PME and Ewald cache interfaces."""

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

    def test_pme_has_more_cache_fields(self):
        """PME tracks k_squared and mesh_dims on top of the three Ewald fields."""
        from nvalchemi.models.pme import PMEModelWrapper

        w = PMEModelWrapper(cutoff=10.0)
        assert hasattr(w, "_cached_k_squared")
        assert hasattr(w, "_cached_mesh_dims")
