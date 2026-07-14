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
"""Single-GPU equivalence: staged Ewald bindings vs. monolithic API.

Gates that ``ewald_compute_partial_structure_factors`` +
``ewald_reciprocal_space_energy_from_structure_factors``, composed
without a cross-rank reduction, reproduce
``nvalchemiops.torch.interactions.electrostatics.ewald.ewald_reciprocal_space``
bit-for-bit (within fp64 round-off). This is the single-GPU proof
that underpins the E0 Phase — once green, the distributed wrapper
(E0.3) can insert an all-reduce between the two stages without
worrying about the underlying math.
"""

from __future__ import annotations

import math

import pytest
import torch

# Warp kernels require CUDA. Skip the whole module on CPU-only systems.
if not torch.cuda.is_available():
    pytest.skip(
        "Ewald warp kernels require CUDA; no GPU available",
        allow_module_level=True,
    )

from nvalchemiops.torch.interactions.electrostatics.ewald import (  # noqa: E402
    ewald_reciprocal_space,
)
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (  # noqa: E402
    generate_k_vectors_ewald_summation,
)
from nvalchemiops.torch.interactions.electrostatics.parameters import (  # noqa: E402
    estimate_ewald_parameters,
)

from nvalchemi.models._ops.electrostatics.ewald import (  # noqa: E402
    ewald_compute_partial_structure_factors,
    ewald_reciprocal_space_from_structure_factors,
)

# ---------------------------------------------------------------------------
# Test systems
# ---------------------------------------------------------------------------


def _nacl_single(
    device: str | torch.device,
    dtype: torch.dtype = torch.float64,
    box: float = 5.64,
    seed: int = 0,
) -> dict:
    """Simple cubic NaCl-like system — 8 atoms alternating +1 / -1 charges."""
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=dtype,
        device=device,
    ) * (box / 2.0)
    g = torch.Generator(device="cpu").manual_seed(seed)
    jitter = torch.randn(8, 3, dtype=dtype, generator=g).to(device) * 0.05
    positions = coords + jitter
    # Alternating +1 / -1 charges → neutral.
    charges = torch.tensor(
        [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
        dtype=dtype,
        device=device,
    )
    cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * box
    return {
        "positions": positions,
        "charges": charges,
        "cell": cell,
        "box": box,
    }


def _nacl_batched(
    device: str | torch.device,
    dtype: torch.dtype = torch.float64,
    n_systems: int = 3,
    seed: int = 0,
) -> dict:
    """Concatenate ``n_systems`` NaCl cells of varying box sizes."""
    positions_list = []
    charges_list = []
    cells = []
    batch_idx_list = []
    for s in range(n_systems):
        box = 5.0 + 0.8 * s
        data = _nacl_single(device=device, dtype=dtype, box=box, seed=seed + s)
        positions_list.append(data["positions"])
        charges_list.append(data["charges"])
        cells.append(data["cell"])
        batch_idx_list.append(
            torch.full(
                (data["positions"].shape[0],),
                s,
                # Stage-2 staged bindings launch Warp kernels that require int32
                # batch indices; the monolithic reference accepts either.
                dtype=torch.int32,
                device=device,
            )
        )
    return {
        "positions": torch.cat(positions_list, dim=0),
        "charges": torch.cat(charges_list, dim=0),
        "cell": torch.cat(cells, dim=0),
        "batch_idx": torch.cat(batch_idx_list, dim=0),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ewald_params_single(box: float, accuracy: float = 1e-6) -> tuple[float, float]:
    """Alpha + reciprocal cutoff for the given box. Returns (alpha, k_cutoff).

    ``estimate_ewald_parameters`` derives the splitting from the cell volume and
    atom count (Kolafa-Perram), so it takes ``(positions, cell)`` rather than a
    real-space cutoff. Reconstruct the representative 8-atom cubic NaCl cell for
    this box and read the estimator's per-system alpha / reciprocal cutoff.
    """
    data = _nacl_single(device="cuda", box=box)
    params = estimate_ewald_parameters(
        data["positions"], data["cell"], batch_idx=None, accuracy=accuracy
    )
    return float(params.alpha.reshape(-1)[0]), float(
        params.reciprocal_space_cutoff.reshape(-1)[0]
    )


def _monolithic_energy_single(data: dict) -> torch.Tensor:
    """Run the monolithic ewald_reciprocal_space for comparison."""
    alpha, kc = _ewald_params_single(box=data["box"])
    alpha_t = torch.tensor(
        [alpha], dtype=data["positions"].dtype, device=data["positions"].device
    )
    k_vectors = generate_k_vectors_ewald_summation(data["cell"], kc).to(
        data["positions"].dtype
    )
    energies = ewald_reciprocal_space(
        positions=data["positions"],
        charges=data["charges"],
        cell=data["cell"],
        k_vectors=k_vectors,
        alpha=alpha_t,
        batch_idx=None,
        compute_forces=False,
        compute_virial=False,
        hybrid_forces=False,
    )
    return energies, alpha_t, k_vectors


def _staged_energy_single(
    data: dict, alpha: torch.Tensor, k_vectors: torch.Tensor
) -> torch.Tensor:
    real_sf, imag_sf, total_charge = ewald_compute_partial_structure_factors(
        positions=data["positions"],
        charges=data["charges"],
        cell=data["cell"],
        k_vectors=k_vectors,
        alpha=alpha,
        batch_idx=None,
    )
    return ewald_reciprocal_space_from_structure_factors(
        positions=data["positions"],
        charges=data["charges"],
        cell=data["cell"],
        k_vectors=k_vectors,
        alpha=alpha,
        real_sf=real_sf,
        imag_sf=imag_sf,
        total_charge=total_charge,
        batch_idx=None,
    )


# ---------------------------------------------------------------------------
# Single-system equivalence
# ---------------------------------------------------------------------------


class TestSingleSystemEquivalence:
    """Stage1 → Stage2 composition matches the monolithic call."""

    def test_energy_matches(self):
        data = _nacl_single(device="cuda")
        e_mono, alpha, k_vectors = _monolithic_energy_single(data)
        e_staged = _staged_energy_single(data, alpha, k_vectors)
        torch.testing.assert_close(
            e_staged,
            e_mono,
            atol=1e-10,
            rtol=1e-10,
            msg=f"max|Δ|={(e_staged - e_mono).abs().max().item():.3e}",
        )

    def test_dtype_preserved(self):
        """Reciprocal energies are always float64 regardless of input dtype."""
        data = _nacl_single(device="cuda", dtype=torch.float32)
        e_mono, alpha, k_vectors = _monolithic_energy_single(data)
        e_staged = _staged_energy_single(data, alpha, k_vectors)
        assert e_staged.dtype == torch.float64
        torch.testing.assert_close(e_staged, e_mono, atol=1e-6, rtol=1e-6)

    def test_partial_structure_factors_shape(self):
        """Stage 1 returns correctly-shaped outputs."""
        data = _nacl_single(device="cuda")
        alpha, kc = _ewald_params_single(box=data["box"])
        alpha_t = torch.tensor([alpha], dtype=torch.float64, device="cuda")
        k_vectors = generate_k_vectors_ewald_summation(data["cell"], kc)
        real_sf, imag_sf, total_charge = ewald_compute_partial_structure_factors(
            positions=data["positions"],
            charges=data["charges"],
            cell=data["cell"],
            k_vectors=k_vectors,
            alpha=alpha_t,
            batch_idx=None,
        )
        n_k = k_vectors.shape[0]
        assert real_sf.shape == (n_k,)
        assert imag_sf.shape == (n_k,)
        assert total_charge.shape == (1,)
        assert real_sf.dtype == torch.float64

    def test_total_charge_matches_sum(self):
        """Stage 1's total_charge output equals the analytical charge sum."""
        data = _nacl_single(device="cuda")
        alpha, kc = _ewald_params_single(box=data["box"])
        alpha_t = torch.tensor([alpha], dtype=torch.float64, device="cuda")
        k_vectors = generate_k_vectors_ewald_summation(data["cell"], kc)
        _, _, total_charge = ewald_compute_partial_structure_factors(
            positions=data["positions"],
            charges=data["charges"],
            cell=data["cell"],
            k_vectors=k_vectors,
            alpha=alpha_t,
        )
        expected = float(data["charges"].sum().item())
        got = float(total_charge.item())
        assert math.isclose(got, expected, abs_tol=1e-10), (
            f"total_charge mismatch: got {got}, expected {expected}"
        )


# ---------------------------------------------------------------------------
# Batched equivalence
# ---------------------------------------------------------------------------


class TestBatchedEquivalence:
    def _run(self, n_systems: int) -> None:
        data = _nacl_batched(device="cuda", n_systems=n_systems)
        # Per-system alpha + k_vectors. All systems share accuracy target;
        # alphas differ because boxes differ.
        alphas = []
        k_list = []
        for s in range(n_systems):
            sub_cell = data["cell"][s : s + 1]
            box_s = float(sub_cell[0, 0, 0].item())
            alpha_s, kc_s = _ewald_params_single(box=box_s)
            alphas.append(alpha_s)
            k_s = generate_k_vectors_ewald_summation(sub_cell, kc_s)  # (1, K, 3)
            k_list.append(k_s)
        alpha = torch.tensor(alphas, dtype=torch.float64, device="cuda")
        # Pad k-vectors to the max K across systems so we can stack to (B, K, 3).
        k_max = max(k.shape[1] for k in k_list)
        k_padded = []
        for k in k_list:
            if k.shape[1] < k_max:
                pad = torch.zeros(
                    1, k_max - k.shape[1], 3, dtype=k.dtype, device="cuda"
                )
                k = torch.cat([k, pad], dim=1)
            k_padded.append(k)
        k_vectors = torch.cat(k_padded, dim=0)  # (B, K_max, 3)

        e_mono = ewald_reciprocal_space(
            positions=data["positions"],
            charges=data["charges"],
            cell=data["cell"],
            k_vectors=k_vectors,
            alpha=alpha,
            batch_idx=data["batch_idx"],
            compute_forces=False,
            compute_virial=False,
            hybrid_forces=False,
        )

        real_sf, imag_sf, total_charge = ewald_compute_partial_structure_factors(
            positions=data["positions"],
            charges=data["charges"],
            cell=data["cell"],
            k_vectors=k_vectors,
            alpha=alpha,
            batch_idx=data["batch_idx"],
        )
        e_staged = ewald_reciprocal_space_from_structure_factors(
            positions=data["positions"],
            charges=data["charges"],
            cell=data["cell"],
            k_vectors=k_vectors,
            alpha=alpha,
            real_sf=real_sf,
            imag_sf=imag_sf,
            total_charge=total_charge,
            batch_idx=data["batch_idx"],
        )

        torch.testing.assert_close(
            e_staged,
            e_mono,
            atol=1e-10,
            rtol=1e-10,
            msg=f"max|Δ|={(e_staged - e_mono).abs().max().item():.3e}",
        )

    def test_b2(self):
        self._run(2)

    def test_b4(self):
        self._run(4)

    def test_structure_factor_shape(self):
        """(B, K) for batched input."""
        data = _nacl_batched(device="cuda", n_systems=2)
        sub_cell = data["cell"][0:1]
        box = float(sub_cell[0, 0, 0].item())
        alpha_val, kc = _ewald_params_single(box=box)
        alpha = torch.full(
            (data["cell"].shape[0],), alpha_val, dtype=torch.float64, device="cuda"
        )
        k = generate_k_vectors_ewald_summation(data["cell"][0:1], kc)
        k_vectors = k.expand(data["cell"].shape[0], -1, -1).contiguous()
        real_sf, _, total_charge = ewald_compute_partial_structure_factors(
            positions=data["positions"],
            charges=data["charges"],
            cell=data["cell"],
            k_vectors=k_vectors,
            alpha=alpha,
            batch_idx=data["batch_idx"],
        )
        assert real_sf.shape == (data["cell"].shape[0], k_vectors.shape[1])
        assert total_charge.shape == (data["cell"].shape[0],)
