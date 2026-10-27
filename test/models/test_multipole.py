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
"""Tests for the multipole electrostatics wrappers.

Strategy
--------
* Constructor, split derivation, config, and input-validation tests need no
  kernels and run everywhere.
* Integration tests call the ``nvalchemiops`` multipole kernels and are checked
  against the direct-k oracle ``multipole_electrostatic_energy``, finite
  differences, and each other (COO vs MATRIX, batched vs single).
"""

from __future__ import annotations

import math

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.level_storage import LevelSchema
from nvalchemi.models.base import NeighborListFormat
from nvalchemi.models.multipole import (
    MultipoleEwaldModelWrapper,
    MultipolePMEModelWrapper,
    _derive_split,
    _erfcinv,
    _fft_size,
)

BOTH = pytest.mark.parametrize(
    "cls", [MultipoleEwaldModelWrapper, MultipolePMEModelWrapper]
)

# Box and basis chosen so the direct-k oracle is cheap and converged.
L = 10.0
SIGMA = 0.8
CUTOFF = 8.0
ACCURACY = 1e-5


def _make_batch(
    n_graphs: int = 1,
    l_max: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
) -> Batch:
    """Jittered 3x3x3 lattice systems with neutral random moments."""
    gen = torch.Generator().manual_seed(seed)
    grid = torch.stack(
        torch.meshgrid(*(torch.arange(3.0),) * 3, indexing="ij"), dim=-1
    ).reshape(-1, 3)
    schema = LevelSchema()
    for key in ("positions", "cell", "multipole_moments"):
        schema.set(key, schema.attr_to_group.get(key, "atoms"), dtype=str(dtype)[6:])
    data = []
    for _ in range(n_graphs):
        # Draw in float64 so a float32 batch is the same system, cast down.
        pos = grid * (L / 3) + 0.3 * torch.randn(
            27, 3, generator=gen, dtype=torch.float64
        )
        pos = pos.to(dtype)
        mm = 0.5 * torch.randn(
            27, (l_max + 1) ** 2, generator=gen, dtype=torch.float64
        ).to(dtype)
        mm[:, 0] -= mm[:, 0].mean()
        data.append(
            AtomicData(
                positions=pos.to(device),
                atomic_numbers=torch.ones(27, dtype=torch.long, device=device),
                cell=(torch.eye(3, dtype=dtype) * L).unsqueeze(0).to(device),
                pbc=torch.ones(1, 3, dtype=torch.bool, device=device),
                multipole_moments=mm.to(device),
            )
        )
    return Batch.from_data_list(data, attr_map=schema)


def _build_nl(batch: Batch, model) -> None:
    from nvalchemi.neighbors import compute_neighbors

    compute_neighbors(batch, config=model.model_config.neighbor_config)


def _make(cls, **kwargs):
    kwargs.setdefault("sigma", SIGMA)
    kwargs.setdefault("accuracy", ACCURACY)
    return cls(CUTOFF, **kwargs)


# ===========================================================================
# Split derivation
# ===========================================================================


class TestDeriveSplit:
    def test_alpha_and_kcut_satisfy_envelope_relations(self):
        alpha, k_cut = _derive_split(10.0, 0.6, 1e-6, 0)
        sigma_c = math.sqrt(0.6**2 + 1.0 / (4.0 * alpha**2))
        # Real-space tail at the cutoff hits the accuracy.
        assert math.erfc(10.0 / (2.0 * sigma_c)) == pytest.approx(1e-6, rel=1e-6)
        # Reciprocal envelope at k_cut hits it too.
        assert math.exp(-((k_cut * sigma_c) ** 2)) == pytest.approx(1e-6, rel=1e-6)

    def test_higher_l_max_tightens_the_split(self):
        a0, k0 = _derive_split(10.0, 0.6, 1e-6, 0)
        a2, k2 = _derive_split(10.0, 0.6, 1e-6, 2)
        assert a2 > a0
        assert k2 > k0

    def test_explicit_alpha_only_derives_k_cutoff(self):
        alpha, k_cut = _derive_split(10.0, 0.6, 1e-6, 1, alpha=0.4)
        assert alpha == 0.4
        sigma_c = math.sqrt(0.6**2 + 1.0 / (4.0 * 0.4**2))
        assert k_cut == pytest.approx(math.sqrt(-math.log(1e-7)) / sigma_c)

    def test_too_small_cutoff_raises_with_remedy(self):
        with pytest.raises(ValueError, match="too small for sigma"):
            _derive_split(3.0, 1.0, 1e-6, 0)

    def test_minimum_cutoff_boundary(self):
        sigma = 0.5
        bound = 2.0 * sigma * _erfcinv(1e-6)
        _derive_split(bound * 1.01, sigma, 1e-6, 0)
        with pytest.raises(ValueError):
            _derive_split(bound * 0.99, sigma, 1e-6, 0)

    def test_nonpositive_alpha_raises(self):
        with pytest.raises(ValueError, match="alpha must be positive"):
            _derive_split(10.0, 0.6, 1e-6, 0, alpha=0.0)

    def test_independent_of_system_size(self):
        # The derivation takes no atom count or cell: determinism is the point.
        assert _derive_split(9.0, 0.5, 1e-6, 1) == _derive_split(9.0, 0.5, 1e-6, 1)


class TestFFTSize:
    @pytest.mark.parametrize("n", [1, 5, 17, 31, 33, 100, 127])
    def test_result_is_smooth_multiple_of_four_not_below_n(self, n):
        out = _fft_size(n)
        assert out >= n
        assert out % 4 == 0
        m = out
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        assert m == 1


# ===========================================================================
# Construction and config
# ===========================================================================


class TestConstruction:
    @BOTH
    def test_default_neighbor_format_is_matrix(self, cls):
        w = _make(cls)
        assert w.model_config.neighbor_config.format == NeighborListFormat.MATRIX
        assert w.model_config.neighbor_config.half_list is False
        assert w.model_config.neighbor_config.cutoff == CUTOFF

    @BOTH
    def test_coo_format_is_selectable(self, cls):
        w = _make(cls, neighbor_format=NeighborListFormat.COO)
        assert w.model_config.neighbor_config.format == NeighborListFormat.COO
        assert "neighbor_matrix" not in w.input_data()

    @BOTH
    def test_matrix_format_inputs(self, cls):
        keys = _make(cls).input_data()
        assert {"positions", "multipole_moments", "neighbor_matrix"} <= keys
        assert "num_neighbors" in keys

    @BOTH
    def test_model_config(self, cls):
        cfg = _make(cls).model_config
        assert cfg.required_inputs == frozenset({"multipole_moments"})
        assert cfg.autograd_outputs == frozenset({"forces", "stress"})
        assert cfg.needs_pbc and cfg.supports_pbc
        assert cfg.active_outputs == {"energy", "forces"}

    @BOTH
    def test_sigma_is_required(self, cls):
        with pytest.raises(TypeError):
            cls(CUTOFF)

    @BOTH
    def test_nonpositive_sigma_raises(self, cls):
        with pytest.raises(ValueError, match="sigma must be positive"):
            cls(CUTOFF, sigma=0.0)

    @BOTH
    def test_invalid_l_max_raises(self, cls):
        with pytest.raises(ValueError, match="l_max"):
            _make(cls, l_max=3)

    @BOTH
    def test_invalid_neighbor_format_raises(self, cls):
        with pytest.raises(ValueError):
            _make(cls, neighbor_format="nope")

    @BOTH
    def test_bad_cutoff_sigma_fails_at_construction(self, cls):
        with pytest.raises(ValueError, match="too small for sigma"):
            cls(3.0, sigma=1.0)

    @BOTH
    def test_l_max_checked_at_construction(self, cls):
        # l_max=2 is stricter than l_max=0; a marginal cutoff must be caught.
        bound = 2.0 * SIGMA * _erfcinv(ACCURACY / 100.0)
        with pytest.raises(ValueError):
            cls(bound * 0.99, sigma=SIGMA, accuracy=ACCURACY, l_max=2)

    @BOTH
    def test_no_embeddings(self, cls):
        w = _make(cls)
        assert w.embedding_shapes == {}
        with pytest.raises(NotImplementedError):
            w.compute_embeddings(None)

    @BOTH
    def test_export_not_supported(self, cls):
        with pytest.raises(NotImplementedError):
            _make(cls).export_model("x")

    def test_pme_explicit_mesh_is_kept(self):
        w = _make(MultipolePMEModelWrapper, mesh_dimensions=(24, 24, 24))
        cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * L
        assert w._mesh(0, cell) == (24, 24, 24)

    def test_pme_derived_mesh_scales_with_cell_and_is_cached(self):
        w = _make(MultipolePMEModelWrapper)
        small = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0
        big = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 20.0
        m_small = w._mesh(1, small)
        assert w._mesh(1, small) is w._mesh(1, small)  # cached object
        m_big = w._mesh(1, big)  # cell change invalidates
        assert all(b > s for b, s in zip(m_big, m_small))
        w.invalidate_cache()
        assert w._cached_mesh == {}

    def test_pme_mesh_is_fft_friendly(self):
        w = _make(MultipolePMEModelWrapper)
        cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 13.7
        for n in w._mesh(2, cell):
            assert n % 4 == 0


# ===========================================================================
# Input validation (no kernels)
# ===========================================================================


class TestInputValidation:
    @BOTH
    def test_atomic_data_is_rejected(self, cls):
        data = AtomicData(
            positions=torch.zeros(2, 3),
            atomic_numbers=torch.ones(2, dtype=torch.long),
        )
        with pytest.raises(TypeError, match="requires a Batch"):
            _make(cls).adapt_input(data)

    @BOTH
    @pytest.mark.parametrize("width", [1, 4, 9])
    def test_valid_widths_set_l_max(self, cls, width):
        w = _make(cls, neighbor_format=NeighborListFormat.COO)
        batch = _make_batch(l_max={1: 0, 4: 1, 9: 2}[width])
        _build_nl(batch, w)
        assert w.adapt_input(batch)["l_max"] == {1: 0, 4: 1, 9: 2}[width]

    @BOTH
    def test_bad_width_raises(self, cls):
        w = _make(cls)
        with pytest.raises(ValueError, match="width 5"):
            w._check_moments(torch.zeros(27, 5), 27)

    @BOTH
    def test_bad_rank_or_rows_raise(self, cls):
        w = _make(cls)
        with pytest.raises(ValueError, match="must have shape"):
            w._check_moments(torch.zeros(27), 27)
        with pytest.raises(ValueError, match="must have shape"):
            w._check_moments(torch.zeros(26, 4), 27)

    @BOTH
    def test_l_max_mismatch_raises(self, cls):
        w = _make(cls, l_max=1)
        with pytest.raises(ValueError, match="built with l_max=1"):
            w._check_moments(torch.zeros(27, 9), 27)

    @BOTH
    def test_missing_moments_raises(self, cls):
        w = _make(cls, neighbor_format=NeighborListFormat.COO)
        batch = _make_batch()
        _build_nl(batch, w)
        batch._atoms_group._data.pop("multipole_moments", None)
        with pytest.raises((KeyError, AttributeError)):
            w.adapt_input(batch)

    @BOTH
    def test_second_order_derivatives_rejected(self, cls):
        from nvalchemi.models._derivatives import (
            DerivativeNotSupported,
            _DerivativeRequest,
        )

        w = _make(cls)
        req = _DerivativeRequest.__new__(_DerivativeRequest)
        for field in ("operation", "execution", "strategy"):
            object.__setattr__(req, field, "x")
        with pytest.raises(DerivativeNotSupported):
            w._validate_derivative_request(req)


# ===========================================================================
# CSR adaptation (no kernels)
# ===========================================================================


class TestCSR:
    def test_matrix_and_coo_produce_the_same_pair_set(self, device):
        batch = _make_batch(n_graphs=2, device=device)
        w_m = _make(MultipoleEwaldModelWrapper)
        w_c = _make(MultipoleEwaldModelWrapper, neighbor_format=NeighborListFormat.COO)
        _build_nl(batch, w_m)
        idx_m, ptr_m, sh_m = w_m._csr(batch, 54, batch.positions.device)
        _build_nl(batch, w_c)
        idx_c, ptr_c, sh_c = w_c._csr(batch, 54, batch.positions.device)

        def pairs(idx, ptr, sh):
            out = set()
            for i in range(54):
                for e in range(int(ptr[i]), int(ptr[i + 1])):
                    j = int(idx[e])
                    s = tuple(int(v) for v in sh[e])
                    if not (j == i and s == (0, 0, 0)):  # padding self-slots
                        out.add((i, j, s))
            return out

        assert pairs(idx_m, ptr_m, sh_m) == pairs(idx_c, ptr_c, sh_c)

    def test_matrix_csr_has_static_shape(self, device):
        batch = _make_batch(device=device)
        w = _make(MultipoleEwaldModelWrapper)
        _build_nl(batch, w)
        idx, ptr, sh = w._csr(batch, 27, batch.positions.device)
        K = batch.neighbor_matrix.shape[1]
        assert idx.shape == (27 * K,)
        assert sh.shape == (27 * K, 3)
        assert torch.equal(ptr, (torch.arange(28, device=ptr.device) * K).int())
        assert idx.dtype == ptr.dtype == sh.dtype == torch.int32

    def test_padding_slots_are_self_pairs_with_zero_shift(self, device):
        batch = _make_batch(device=device)
        w = _make(MultipoleEwaldModelWrapper)
        _build_nl(batch, w)
        idx, ptr, sh = w._csr(batch, 27, batch.positions.device)
        K = batch.neighbor_matrix.shape[1]
        invalid = (batch.neighbor_matrix >= 27).reshape(-1)
        rows = torch.arange(27, device=idx.device).repeat_interleave(K)
        assert invalid.any()
        assert torch.equal(idx[invalid], rows[invalid].int())
        assert (sh[invalid] == 0).all()


# ===========================================================================
# Integration (kernels)
# ===========================================================================


def _oracle(batch: Batch, sigma: float = SIGMA) -> torch.Tensor:
    """Per-system direct-k energy, converged to ~1e-9 in the envelope."""
    from nvalchemiops.torch.interactions.electrostatics import (
        multipole_electrostatic_energy,
    )

    k_cut = math.sqrt(-math.log(1e-9)) / sigma
    out = []
    for g in range(batch.num_graphs):
        m = batch.batch_idx == g
        out.append(
            multipole_electrostatic_energy(
                batch.positions[m],
                batch.multipole_moments[m],
                batch.cell[g],
                sigma=sigma,
                k_cutoff=k_cut,
            ).sum()
        )
    return torch.stack(out)


class TestIntegration:
    @pytest.fixture(autouse=True)
    def _require_ops(self):
        pytest.importorskip("nvalchemiops")
        if not torch.cuda.is_available():
            pytest.skip("multipole kernels need CUDA")

    @pytest.fixture
    def device(self):
        return "cuda"

    @BOTH
    @pytest.mark.parametrize("l_max", [0, 1, 2])
    def test_energy_matches_direct_k_oracle(self, cls, l_max, device):
        w = _make(cls)
        batch = _make_batch(l_max=l_max, device=device)
        _build_nl(batch, w)
        e = w(batch)["energy"].view(-1)
        ref = _oracle(batch)
        # PME carries a spline floor that does not shrink with accuracy.
        rtol = 5e-4 if cls is MultipolePMEModelWrapper else 2e-4
        torch.testing.assert_close(e, ref, rtol=rtol, atol=0)

    @BOTH
    def test_atomic_energies_sum_to_energy(self, cls, device):
        w = _make(cls)
        w.model_config.active_outputs = {"energy", "atomic_energies"}
        batch = _make_batch(n_graphs=2, device=device)
        _build_nl(batch, w)
        out = w(batch)
        per_sys = torch.zeros(2, dtype=torch.float64, device=device).scatter_add(
            0, batch.batch_idx.long(), out["atomic_energies"]
        )
        torch.testing.assert_close(out["energy"].view(-1), per_sys)
        assert out["atomic_energies"].dtype == torch.float64

    @BOTH
    @pytest.mark.parametrize("l_max", [0, 1, 2])
    def test_coo_and_matrix_agree(self, cls, l_max, device):
        batch = _make_batch(n_graphs=2, l_max=l_max, device=device)
        outs = []
        for fmt in (NeighborListFormat.MATRIX, NeighborListFormat.COO):
            w = _make(cls, neighbor_format=fmt)
            w.model_config.active_outputs = {"energy", "forces", "stress"}
            _build_nl(batch, w)
            outs.append(w(batch))
        for key in ("energy", "forces", "stress"):
            torch.testing.assert_close(
                outs[0][key], outs[1][key], rtol=1e-9, atol=1e-10
            )

    @BOTH
    def test_extra_matrix_padding_changes_nothing(self, cls, device):
        from nvalchemi.neighbors import compute_neighbors

        w = _make(cls)
        batch = _make_batch(device=device)
        compute_neighbors(batch, config=w.model_config.neighbor_config)
        ref = w(batch)["energy"]
        compute_neighbors(
            batch, config=w.model_config.neighbor_config, max_neighbors=2000
        )
        assert batch.neighbor_matrix.shape[1] == 2000
        torch.testing.assert_close(w(batch)["energy"], ref, rtol=1e-12, atol=1e-12)

    @BOTH
    def test_batched_matches_individual(self, cls, device):
        w = _make(cls)
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        both = _make_batch(n_graphs=2, l_max=1, device=device, seed=3)
        _build_nl(both, w)
        out = w(both)
        for g in range(2):
            single = Batch.from_data_list(
                [both.to_data_list()[g]], attr_map=_schema_for(both)
            )
            _build_nl(single, w)
            ref = w(single)
            m = both.batch_idx == g
            torch.testing.assert_close(
                out["energy"][g], ref["energy"][0], rtol=1e-10, atol=1e-10
            )
            torch.testing.assert_close(
                out["forces"][m], ref["forces"], rtol=1e-8, atol=1e-9
            )
            torch.testing.assert_close(
                out["stress"][g], ref["stress"][0], rtol=1e-8, atol=1e-10
            )

    @BOTH
    @pytest.mark.parametrize("l_max", [0, 1, 2])
    def test_forces_match_finite_difference(self, cls, l_max, device):
        w = _make(cls)
        batch = _make_batch(l_max=l_max, device=device)
        _build_nl(batch, w)
        forces = w(batch)["forces"]
        h = 1e-5
        for atom, axis in ((0, 0), (5, 1), (13, 2)):
            e = []
            for sign in (+1, -1):
                b = batch.clone()
                b.positions[atom, axis] += sign * h
                _build_nl(b, w)
                e.append(w(b)["energy"].sum())
            fd = -(e[0] - e[1]) / (2 * h)
            torch.testing.assert_close(forces[atom, axis], fd, rtol=1e-5, atol=1e-6)

    @BOTH
    @pytest.mark.parametrize("l_max", [0, 1, 2])
    def test_stress_matches_finite_difference(self, cls, l_max, device):
        w = _make(cls)
        w.model_config.active_outputs = {"energy", "forces", "stress"}
        batch = _make_batch(l_max=l_max, device=device)
        _build_nl(batch, w)
        stress = w(batch)["stress"][0]
        volume = L**3
        h = 1e-5
        for a, b in ((0, 0), (1, 1), (0, 1)):
            e = []
            for sign in (+1, -1):
                eps = torch.zeros(3, 3, dtype=torch.float64, device=device)
                eps[a, b] += 0.5 * sign * h
                eps[b, a] += 0.5 * sign * h
                strain = torch.eye(3, dtype=torch.float64, device=device) + eps
                s = batch.clone()
                s.positions.copy_(batch.positions @ strain)
                s.cell.copy_(batch.cell @ strain)
                _build_nl(s, w)
                e.append(w(s)["energy"].sum())
            fd = (e[0] - e[1]) / (2 * h) / volume
            torch.testing.assert_close(stress[a, b], fd, rtol=1e-4, atol=1e-7)

    @BOTH
    def test_energy_only_skips_derivatives(self, cls, device):
        w = _make(cls)
        w.model_config.active_outputs = {"energy"}
        batch = _make_batch(device=device)
        _build_nl(batch, w)
        out = w(batch)
        assert set(out) == {"energy"}

    @BOTH
    def test_float32_runs_and_is_close(self, cls, device):
        w = _make(cls)
        b64 = _make_batch(device=device)
        b32 = _make_batch(device=device, dtype=torch.float32)
        _build_nl(b64, w)
        _build_nl(b32, w)
        e64 = w(b64)["energy"]
        e32 = w(b32)["energy"]
        assert e32.dtype == torch.float32
        torch.testing.assert_close(e32.double(), e64, rtol=5e-3, atol=1e-3)

    @BOTH
    def test_moments_receive_gradient(self, cls, device):
        w = _make(cls)
        w.model_config.active_outputs = {"energy"}
        batch = _make_batch(l_max=1, device=device)
        _build_nl(batch, w)
        mm = batch.multipole_moments.clone().requires_grad_(True)
        batch._atoms_group["multipole_moments"] = mm
        w(batch)["energy"].sum().backward()
        assert mm.grad is not None and torch.isfinite(mm.grad).all()
        assert mm.grad.abs().sum() > 0


def _schema_for(batch: Batch) -> LevelSchema:
    schema = LevelSchema()
    dtype = str(batch.positions.dtype)[6:]
    for key in ("positions", "cell", "multipole_moments"):
        schema.set(key, schema.attr_to_group.get(key, "atoms"), dtype=dtype)
    return schema
