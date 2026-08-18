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
"""Unit tests for core enhanced-sampling abstractions.

Covers:

* :class:`~nvalchemi.enhanced_sampling.BiasResult` — shape validation,
  detachment enforcement, stress/virial mutual exclusion.
* :class:`~nvalchemi.enhanced_sampling.BiasPotential` — structural
  Protocol check.
* :class:`~nvalchemi.enhanced_sampling.ConservativeBias` — forces and
  tensile-positive Cauchy stress from autograd; compare both with finite
  differences; stress symmetry; no ``requires_grad`` escape into live
  batch or result; no memory growth across 10 repeated evaluations.
* :func:`~nvalchemi.enhanced_sampling.pair_distance` — nonperiodic and
  Minkowski-reduced triclinic MIC; shared and per-graph atom indices;
  gradients via ``torch.autograd.gradcheck``; compile-stability under
  ``torch.compile`` (fullgraph=True on CPU); unreduced-cell rejection in
  eager mode (check skipped under compile — caller responsibility).
* :func:`~nvalchemi.enhanced_sampling.aggregate_bias_results` — summing,
  None handling, duplicate-key rejection.
* ``torch.compile`` tests: ``pair_distance`` and ``aggregate_bias_results``
  compile with ``fullgraph=True``; ``ConservativeBias.energy()`` compiles
  with ``fullgraph=True``; ``ConservativeBias.evaluate()`` runs under
  ``fullgraph=False`` (graph break at ``requires_grad_()`` is documented).

GPU integration tests are marked ``@pytest.mark.slow`` and are run only
when a CUDA device is available (the ``device`` fixture handles skip).
"""

from __future__ import annotations

import gc

import pytest
import torch
from torch import Tensor

from nvalchemi.data import AtomicData, Batch
from nvalchemi.enhanced_sampling import (
    BiasPotential,
    BiasResult,
    ConservativeBias,
    aggregate_bias_results,
    pair_distance,
)

# ---------------------------------------------------------------------------
# Shared batch-construction helpers
# ---------------------------------------------------------------------------


def _make_nonperiodic_batch(
    n_graphs: int = 2,
    atoms_per_graph: int = 4,
    device: str = "cpu",
    seed: int = 42,
) -> Batch:
    """Return a simple non-periodic Batch with known positions."""
    torch.manual_seed(seed)
    data_list = [
        AtomicData(
            atomic_numbers=torch.tensor([6] * atoms_per_graph, dtype=torch.long),
            positions=torch.randn(atoms_per_graph, 3),
        )
        for _ in range(n_graphs)
    ]
    batch = Batch.from_data_list(data_list).to(device)
    batch["energy"] = torch.zeros(n_graphs, 1, device=device)
    batch["forces"] = torch.zeros(atoms_per_graph * n_graphs, 3, device=device)
    return batch


def _make_cubic_batch(
    n_graphs: int = 2,
    atoms_per_graph: int = 4,
    box: float = 5.0,
    device: str = "cpu",
    seed: int = 42,
) -> Batch:
    """Return a Batch with cubic unit cells and full 3D PBC."""
    torch.manual_seed(seed)
    data_list = []
    for _ in range(n_graphs):
        positions = torch.rand(atoms_per_graph, 3) * box
        # AtomicData expects cell as [1, 3, 3] and pbc as [1, 3]
        cell = torch.eye(3).unsqueeze(0) * box
        pbc = torch.tensor([[True, True, True]])
        data_list.append(
            AtomicData(
                atomic_numbers=torch.tensor([6] * atoms_per_graph, dtype=torch.long),
                positions=positions,
                cell=cell,
                pbc=pbc,
            )
        )
    batch = Batch.from_data_list(data_list).to(device)
    batch["energy"] = torch.zeros(n_graphs, 1, device=device)
    batch["forces"] = torch.zeros(atoms_per_graph * n_graphs, 3, device=device)
    return batch


def _make_triclinic_batch(
    device: str = "cpu",
    seed: int = 0,
) -> Batch:
    """Return a single-graph Batch with a triclinic unit cell."""
    torch.manual_seed(seed)
    # Tilted cell: a = [5,0,0], b = [1,5,0], c = [0.5,0.5,5]
    cell_mat = torch.tensor([[5.0, 0.0, 0.0], [1.0, 5.0, 0.0], [0.5, 0.5, 5.0]])
    # AtomicData expects [1, 3, 3] and [1, 3]
    cell = cell_mat.unsqueeze(0)
    pbc = torch.tensor([[True, True, True]])
    positions = torch.rand(4, 3) @ cell_mat  # Cartesian, inside cell
    data = AtomicData(
        atomic_numbers=torch.tensor([6, 6, 6, 6], dtype=torch.long),
        positions=positions,
        cell=cell,
        pbc=pbc,
    )
    batch = Batch.from_data_list([data]).to(device)
    batch["energy"] = torch.zeros(1, 1, device=device)
    batch["forces"] = torch.zeros(4, 3, device=device)
    return batch


# ===========================================================================
# 1. BiasResult
# ===========================================================================


class TestBiasResult:
    """Tests for the BiasResult dataclass."""

    def test_empty_construction(self) -> None:
        r = BiasResult()
        assert r.energy is None
        assert r.forces is None
        assert r.observables == {}

    def test_detached_tensors_accepted(self) -> None:
        e = torch.tensor([[1.0]]).detach()
        f = torch.zeros(3, 3).detach()
        r = BiasResult(energy=e, forces=f)
        assert r.energy is e

    def test_requires_grad_energy_raises(self) -> None:
        bad = torch.tensor([[1.0]], requires_grad=True)
        with pytest.raises(ValueError, match="energy.*detached"):
            BiasResult(energy=bad)

    def test_requires_grad_forces_raises(self) -> None:
        bad = torch.zeros(3, 3, requires_grad=True)
        with pytest.raises(ValueError, match="forces.*detached"):
            BiasResult(forces=bad)

    def test_grad_fn_raises(self) -> None:
        x = torch.tensor([[1.0]], requires_grad=True)
        y = x * 2.0  # has grad_fn
        with pytest.raises(ValueError, match="energy.*detached"):
            BiasResult(energy=y)

    def test_stress_and_virial_raises(self) -> None:
        s = torch.zeros(1, 3, 3)
        v = torch.zeros(1, 3, 3)
        with pytest.raises(ValueError, match="stress.*virial"):
            BiasResult(stress=s, virial=v)

    def test_observable_requires_grad_raises(self) -> None:
        bad = torch.zeros(3, requires_grad=True)
        with pytest.raises(ValueError, match="observables"):
            BiasResult(observables={"cv": bad})

    def test_frozen_immutability(self) -> None:
        r = BiasResult(energy=torch.zeros(1, 1))
        with pytest.raises((TypeError, AttributeError)):
            r.energy = torch.ones(1, 1)  # type: ignore[misc]

    # --- shape validation ---

    def test_energy_wrong_ndim_raises(self) -> None:
        """energy must be [B, 1]; a flat [B] tensor is rejected."""
        with pytest.raises(ValueError, match="energy.*\\[B, 1\\]"):
            BiasResult(energy=torch.zeros(2))

    def test_energy_wrong_trailing_dim_raises(self) -> None:
        """energy last dim must be 1, not 3."""
        with pytest.raises(ValueError, match="energy.*\\[B, 1\\]"):
            BiasResult(energy=torch.zeros(2, 3))

    def test_forces_wrong_ndim_raises(self) -> None:
        """forces must be [N, 3]; a 1-D tensor is rejected."""
        with pytest.raises(ValueError, match="forces.*\\[N, 3\\]"):
            BiasResult(forces=torch.zeros(9))

    def test_forces_wrong_width_raises(self) -> None:
        """forces last dim must be 3, not 1."""
        with pytest.raises(ValueError, match="forces.*\\[N, 3\\]"):
            BiasResult(forces=torch.zeros(4, 1))

    def test_stress_wrong_shape_raises(self) -> None:
        """stress must be [B, 3, 3]; a [B, 3] tensor is rejected."""
        with pytest.raises(ValueError, match="stress.*\\[B, 3, 3\\]"):
            BiasResult(stress=torch.zeros(2, 3))

    def test_virial_wrong_shape_raises(self) -> None:
        """virial must be [B, 3, 3]."""
        with pytest.raises(ValueError, match="virial.*\\[B, 3, 3\\]"):
            BiasResult(virial=torch.zeros(2, 9))

    def test_state_version_wrong_ndim_raises(self) -> None:
        """state_version must be 1-D."""
        with pytest.raises(ValueError, match="state_version.*\\[B\\]"):
            BiasResult(state_version=torch.zeros(2, 1, dtype=torch.int32))

    def test_state_version_float_dtype_raises(self) -> None:
        """state_version must be an integer dtype."""
        with pytest.raises(ValueError, match="integer dtype"):
            BiasResult(state_version=torch.zeros(2))  # float32

    def test_state_version_integer_accepted(self) -> None:
        """state_version with int64 dtype is accepted."""
        r = BiasResult(state_version=torch.zeros(2, dtype=torch.int64))
        assert r.state_version is not None

    # --- batch-size consistency ---

    def test_batch_size_mismatch_raises(self) -> None:
        """energy [2, 1] and virial [3, 3, 3] have inconsistent B."""
        with pytest.raises(ValueError, match="inconsistent"):
            BiasResult(energy=torch.zeros(2, 1), virial=torch.zeros(3, 3, 3))

    def test_batch_size_consistent_accepted(self) -> None:
        """energy [2, 1] and virial [2, 3, 3] with matching B=2 are accepted."""
        r = BiasResult(energy=torch.zeros(2, 1), virial=torch.zeros(2, 3, 3))
        assert r.energy is not None

    # --- finiteness ---

    def test_energy_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="energy.*NaN or Inf"):
            BiasResult(energy=torch.tensor([[float("nan")]]))

    def test_energy_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="energy.*NaN or Inf"):
            BiasResult(energy=torch.tensor([[float("inf")]]))

    def test_forces_nan_raises(self) -> None:
        bad = torch.zeros(3, 3)
        bad[1, 2] = float("nan")
        with pytest.raises(ValueError, match="forces.*NaN or Inf"):
            BiasResult(forces=bad)

    def test_virial_inf_raises(self) -> None:
        bad = torch.zeros(1, 3, 3)
        bad[0, 0, 0] = float("-inf")
        with pytest.raises(ValueError, match="virial.*NaN or Inf"):
            BiasResult(virial=bad)

    def test_observable_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="observables.*NaN or Inf"):
            BiasResult(observables={"cv": torch.tensor([float("nan")])})

    def test_valid_result_accepted(self) -> None:
        """A fully-populated valid BiasResult passes all checks."""
        r = BiasResult(
            energy=torch.zeros(2, 1),
            forces=torch.zeros(6, 3),
            virial=torch.zeros(2, 3, 3),
            state_version=torch.zeros(2, dtype=torch.int64),
            observables={"bias/a/cv": torch.zeros(2)},
        )
        assert r.energy is not None


# ===========================================================================
# 2. BiasPotential Protocol
# ===========================================================================


class TestBiasPotentialProtocol:
    """Tests for structural protocol membership."""

    def test_structural_satisfaction(self) -> None:
        class MyBias:
            name = "my_bias"

            def evaluate(self, current: Batch) -> BiasResult:
                return BiasResult()

        assert isinstance(MyBias(), BiasPotential)

    def test_missing_name_not_protocol(self) -> None:
        class NotABias:
            def evaluate(self, current: Batch) -> BiasResult:
                return BiasResult()

        assert not isinstance(NotABias(), BiasPotential)

    def test_missing_evaluate_not_protocol(self) -> None:
        class NotABias:
            name = "x"

        assert not isinstance(NotABias(), BiasPotential)


# ===========================================================================
# 3. ConservativeBias — autograd helper
# ===========================================================================


class _QuadraticBias(ConservativeBias):
    """E = 0.5 * k * ||positions||^2 per graph — analytically tractable."""

    def __init__(self, k: float = 1.0) -> None:
        self.name = "quadratic"
        self.k = k

    def energy(self, current: Batch) -> Tensor:
        # Sum of squared positions per graph → [B, 1]
        # batch_ptr gives atom offsets per graph
        ptr = current.batch_ptr
        B = current.num_graphs
        energies = []
        for b in range(B):
            pos_b = current.positions[ptr[b] : ptr[b + 1]]
            energies.append(0.5 * self.k * (pos_b**2).sum())
        return torch.stack(energies).unsqueeze(-1)  # [B, 1]


class _PairDistanceBias(ConservativeBias):
    """E = 0.5 * k * pair_distance^2 — uses the pair_distance CV."""

    def __init__(self, atom_indices: Tensor, k: float = 1.0) -> None:
        self.name = "pair_dist_bias"
        self.atom_indices = atom_indices
        self.k = k

    def energy(self, current: Batch) -> Tensor:
        d = pair_distance(current, self.atom_indices)  # [B, 1]
        return 0.5 * self.k * d**2  # [B, 1]


class _AnisotropicBias(ConservativeBias):
    """E = sum_n (x_n * y_n) per graph — couples distinct Cartesian components.

    Because the energy mixes the x and y components rather than depending
    only on interatomic distances, its derivative with respect to the full
    deformation gradient F is asymmetric.  The derivative with respect to
    the symmetric strain tensor eps is symmetric, which is what the project
    virial/stress convention requires.  A bias like this is what
    distinguishes the two derivatives; a central pair interaction does not.
    """

    def __init__(self) -> None:
        self.name = "anisotropic"

    def energy(self, current: Batch) -> Tensor:
        ptr = current.batch_ptr
        energies = []
        for b in range(current.num_graphs):
            pos_b = current.positions[ptr[b] : ptr[b + 1]]
            energies.append((pos_b[:, 0] * pos_b[:, 1]).sum())
        return torch.stack(energies).unsqueeze(-1)  # [B, 1]


class TestConservativeBias:
    """Tests for ConservativeBias autograd helper."""

    def test_forces_shape(self, device: str) -> None:
        batch = _make_nonperiodic_batch(n_graphs=2, atoms_per_graph=3, device=device)
        bias = _QuadraticBias(k=1.0)
        result = bias.evaluate(batch)
        assert result.forces is not None
        assert result.forces.shape == (6, 3)

    def test_energy_shape(self, device: str) -> None:
        batch = _make_nonperiodic_batch(n_graphs=2, atoms_per_graph=3, device=device)
        bias = _QuadraticBias(k=1.0)
        result = bias.evaluate(batch)
        assert result.energy is not None
        assert result.energy.shape == (2, 1)

    def test_forces_analytical_vs_autograd(self, device: str) -> None:
        """F = -dE/dr; for E = 0.5 * k * ||r||^2, F = -k * r."""
        k = 2.0
        batch = _make_nonperiodic_batch(n_graphs=1, atoms_per_graph=4, device=device)
        bias = _QuadraticBias(k=k)
        result = bias.evaluate(batch)
        expected_forces = -k * batch.positions
        assert result.forces is not None
        assert torch.allclose(result.forces, expected_forces, atol=1e-5)

    def test_forces_finite_difference(self, device: str) -> None:
        """Compare autograd forces to central-difference finite differences."""
        k = 1.0
        eps = 1e-4
        batch = _make_nonperiodic_batch(n_graphs=1, atoms_per_graph=3, device=device)
        bias = _QuadraticBias(k=k)

        pos = batch.positions.clone()  # [N, 3]
        N = pos.shape[0]
        fd_forces = torch.zeros_like(pos)
        for i in range(N):
            for j in range(3):
                pos_plus = pos.clone()
                pos_plus[i, j] += eps
                batch["positions"] = pos_plus
                e_plus = bias.evaluate(batch).energy.sum().item()

                pos_minus = pos.clone()
                pos_minus[i, j] -= eps
                batch["positions"] = pos_minus
                e_minus = bias.evaluate(batch).energy.sum().item()

                fd_forces[i, j] = -(e_plus - e_minus) / (2 * eps)

        batch["positions"] = pos
        result = bias.evaluate(batch)
        assert result.forces is not None
        # float32 finite differences at eps=1e-4 have ~1e-3 cancellation error;
        # use a tolerance that accounts for float32 precision.
        assert torch.allclose(result.forces, fd_forces, atol=5e-3)

    def test_result_fully_detached(self, device: str) -> None:
        """BiasResult tensors must have requires_grad=False and grad_fn=None."""
        batch = _make_nonperiodic_batch(device=device)
        bias = _QuadraticBias()
        result = bias.evaluate(batch)
        for name in ("energy", "forces"):
            t = getattr(result, name)
            if t is not None:
                assert not t.requires_grad, f"{name} has requires_grad=True"
                assert t.grad_fn is None, f"{name} has non-null grad_fn"

    def test_live_batch_positions_not_mutated(self, device: str) -> None:
        """batch.positions must be restored to original tensor after evaluate()."""
        batch = _make_nonperiodic_batch(device=device)
        original_pos = batch.positions
        original_data = original_pos.clone()
        bias = _QuadraticBias()
        bias.evaluate(batch)
        # The tensor object should be restored
        assert batch.positions is original_pos
        # Values should be unchanged
        assert torch.allclose(batch.positions, original_data)

    def test_live_batch_positions_no_grad(self, device: str) -> None:
        """After evaluate(), batch.positions must not have requires_grad=True."""
        batch = _make_nonperiodic_batch(device=device)
        bias = _QuadraticBias()
        bias.evaluate(batch)
        assert not batch.positions.requires_grad
        assert batch.positions.grad_fn is None

    def test_no_memory_growth_repeated_evaluate(self, device: str) -> None:
        """Repeated evaluate() must not grow GPU allocated memory monotonically.

        Warm up 3 calls, then sample allocated memory over 10 calls.  The
        delta between first and last sample must be ≤ 0 (or a small
        tolerance for caching effects).
        """
        batch = _make_nonperiodic_batch(n_graphs=4, atoms_per_graph=8, device=device)
        bias = _QuadraticBias()

        # Warm up
        for _ in range(3):
            bias.evaluate(batch)

        gc.collect()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            mem_start = torch.cuda.memory_allocated()
        else:
            mem_start = 0

        for _ in range(10):
            bias.evaluate(batch)

        if device == "cuda":
            torch.cuda.synchronize()
            mem_end = torch.cuda.memory_allocated()
            # Allow a small tolerance (1 MB) for CUDA caching allocator overhead
            assert mem_end - mem_start <= 1 * 1024 * 1024, (
                f"GPU memory grew by {mem_end - mem_start} bytes across 10 evaluate() calls"
            )

    def test_stress_analytical_across_image_boundary(self, device: str) -> None:
        """Cauchy stress is correct for a pair bias whose MIC vector crosses an image.

        Setup
        -----
        Box: 10 Å cubic (V = 1000 Å³). Atom 0 at [0.5, 0, 0], atom 1 at
        [9.5, 0, 0].  MIC distance = 1 Å (image at x − 10, so
        dr_mic = [−1, 0, 0]).  Bias: E = 0.5 · k · d²  (k = 1 eV/Å²).

        Analytical derivation
        ---------------------
        Under a homogeneous symmetric strain ε both positions and cell deform,
        so the MIC vector deforms with them (the image index [−1, 0, 0] is
        fixed)::

            dr_mic → dr_mic @ (I + ε)
            d²     = |dr_mic|² + 2 · dr_mic · ε · dr_micᵀ + O(ε²)

        dE/dε_kl |_{ε=0} = k · dr_mic[k] · dr_mic[l]
        σ_kl = (dE/dε_kl) / V = k · dr_mic[k] · dr_mic[l] / V

        For dr_mic = [−1, 0, 0], k = 1, V = 1000:
        σ[0,0] = +1e-3 eV/Å³, all other elements = 0.  Positive (tensile) is
        the expected sign: the harmonic restraint pulls the two atoms together.

        This is only correct when positions and cell are strained together.
        A strain leaf applied to the cell alone misses the atomic-position
        contribution and returns the wrong answer.
        """
        k = 1.0
        box = 10.0
        volume = box**3
        # MIC distance = |9.5 - 0.5 - 10| = 1 Å; dr_mic = [-1, 0, 0]
        positions = torch.tensor([[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]])
        cell = torch.eye(3).unsqueeze(0) * box  # [1, 3, 3]
        pbc = torch.tensor([[True, True, True]])

        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=positions,
            cell=cell,
            pbc=pbc,
        )
        batch = Batch.from_data_list([data]).to(device)
        idx = torch.tensor([0, 1], device=device)
        bias = _PairDistanceBias(atom_indices=idx, k=k)
        result = bias.evaluate(batch)

        assert result.stress is not None, "stress should be non-None for periodic batch"
        assert result.stress.shape == (1, 3, 3)
        assert result.virial is None, "ConservativeBias emits stress, not virial"

        # Analytical: σ = k · outer(dr_mic, dr_mic) / V with dr_mic = [−1, 0, 0]
        expected = torch.zeros(3, 3, device=device)
        expected[0, 0] = k / volume
        sigma = result.stress[0]  # [3, 3]
        assert torch.allclose(sigma, expected, atol=1e-8), (
            f"stress = {sigma}, expected {expected}.  Stress may be missing the "
            "atomic-position contribution (strain not applied to both positions "
            "and cell simultaneously)."
        )

    def test_stress_is_symmetric(self, device: str) -> None:
        """Stress must be symmetric: the project strain tensor ε is symmetric.

        A bias whose energy is not a central pair interaction is the case that
        distinguishes a symmetric strain derivative from a raw deformation
        gradient derivative.  ``_AnisotropicBias`` below uses a per-component
        weighting so that dE/dF is asymmetric while dE/dε is not.
        """
        batch = _make_cubic_batch(n_graphs=2, atoms_per_graph=5, device=device)
        bias = _AnisotropicBias()
        result = bias.evaluate(batch)

        assert result.stress is not None
        sigma = result.stress
        assert torch.allclose(sigma, sigma.mT, atol=1e-6), (
            f"stress is not symmetric:\n{sigma}\nvs transpose\n{sigma.mT}"
        )

    def test_no_stress_for_nonperiodic_batch(self, device: str) -> None:
        """A batch with no cell yields forces but no stress."""
        batch = _make_nonperiodic_batch(device=device)
        result = _QuadraticBias().evaluate(batch)
        assert result.forces is not None
        assert result.stress is None
        assert result.virial is None

    def test_supports_stress_false_skips_stress(self, device: str) -> None:
        """``_supports_stress = False`` skips the strain leaf on a periodic batch."""

        class _ForceOnlyBias(_QuadraticBias):
            _supports_stress = False

        batch = _make_cubic_batch(n_graphs=1, atoms_per_graph=4, device=device)
        result = _ForceOnlyBias().evaluate(batch)
        assert result.forces is not None
        assert result.stress is None

    def test_live_batch_cell_restored(self, device: str) -> None:
        """batch.cell must be restored to the original tensor after evaluate()."""
        batch = _make_cubic_batch(n_graphs=2, atoms_per_graph=4, device=device)
        original_cell = batch.cell
        original_data = original_cell.clone()
        _QuadraticBias().evaluate(batch)
        assert batch.cell is original_cell
        assert torch.allclose(batch.cell, original_data)
        assert not batch.cell.requires_grad
        assert batch.cell.grad_fn is None

    def test_stress_finite_difference(self, device: str) -> None:
        """Compare autograd stress to a finite-difference strain derivative.

        Applies a symmetric strain ``ε`` to both positions and cell, and
        checks ``σ_kl ≈ (E(+ε) − E(−ε)) / (2 h V)`` for each component.
        Uses float64 so the central difference is not dominated by
        cancellation error.
        """
        h = 1e-5
        box = 6.0
        volume = box**3
        torch.manual_seed(7)
        positions = torch.rand(5, 3, dtype=torch.float64) * box
        cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * box

        data = AtomicData(
            atomic_numbers=torch.tensor([6] * 5, dtype=torch.long),
            positions=positions,
            cell=cell,
            pbc=torch.tensor([[True, True, True]]),
        )
        batch = Batch.from_data_list([data]).to(device)
        bias = _PairDistanceBias(
            atom_indices=torch.tensor([0, 3], device=device), k=1.5
        )

        result = bias.evaluate(batch)
        assert result.stress is not None

        base_pos = batch.positions.clone()
        base_cell = batch.cell.clone()
        eye = torch.eye(3, dtype=base_pos.dtype, device=base_pos.device)

        fd_stress = torch.zeros(3, 3, dtype=base_pos.dtype, device=base_pos.device)
        for a in range(3):
            for b in range(3):
                # Symmetric strain perturbation in component (a, b).
                eps = torch.zeros(3, 3, dtype=base_pos.dtype, device=base_pos.device)
                eps[a, b] += 0.5
                eps[b, a] += 0.5
                eps = eps * h

                energies = []
                for sign in (+1.0, -1.0):
                    deform = eye + sign * eps
                    batch["positions"] = base_pos @ deform
                    batch["cell"] = base_cell @ deform
                    energies.append(bias.evaluate(batch).energy.sum().item())

                fd_stress[a, b] = (energies[0] - energies[1]) / (2 * h * volume)

        batch["positions"] = base_pos
        batch["cell"] = base_cell

        assert torch.allclose(result.stress[0], fd_stress, atol=1e-7), (
            f"autograd stress\n{result.stress[0]}\ndiffers from finite differences\n"
            f"{fd_stress}"
        )

    def test_evaluate_is_read_only_no_state_change(self, device: str) -> None:
        """Multiple evaluate() calls must leave bias state unchanged."""
        batch = _make_nonperiodic_batch(device=device)
        bias = _QuadraticBias(k=2.5)
        r1 = bias.evaluate(batch)
        r2 = bias.evaluate(batch)
        assert result_close(r1, r2)


def result_close(a: BiasResult, b: BiasResult, atol: float = 1e-6) -> bool:
    """Return True iff all non-None tensor fields of a and b are close."""
    for attr in ("energy", "forces", "virial", "stress"):
        ta, tb = getattr(a, attr), getattr(b, attr)
        if ta is None and tb is None:
            continue
        if ta is None or tb is None:
            return False
        if not torch.allclose(ta, tb, atol=atol):
            return False
    return True


# ===========================================================================
# 4. pair_distance CV
# ===========================================================================


class TestPairDistance:
    """Tests for the pair_distance collective variable."""

    # --- atom_indices shape / dtype validation ---

    def test_atom_indices_float_dtype_raises(self) -> None:
        """Float atom_indices raises ValueError (would silently cast to int)."""
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=torch.zeros(2, 3),
        )
        batch = Batch.from_data_list([data])
        with pytest.raises(ValueError, match="integer dtype"):
            pair_distance(batch, torch.tensor([0.0, 1.0]))

    def test_atom_indices_1d_wrong_length_raises(self) -> None:
        """1-D atom_indices with length != 2 raises ValueError.

        torch.tensor([0]) would silently become [[0, 0]] (self-distance).
        """
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=torch.zeros(2, 3),
        )
        batch = Batch.from_data_list([data])
        with pytest.raises(ValueError, match="exactly 2 elements"):
            pair_distance(batch, torch.tensor([0]))  # length 1

    def test_atom_indices_2d_extra_column_raises(self) -> None:
        """[B, 3] atom_indices raises ValueError (extra column would be silently dropped)."""
        data_list = [
            AtomicData(
                atomic_numbers=torch.tensor([6, 6, 6], dtype=torch.long),
                positions=torch.zeros(3, 3),
            )
        ] * 2
        batch = Batch.from_data_list(data_list)
        with pytest.raises(ValueError, match="second dimension must be exactly 2"):
            pair_distance(batch, torch.tensor([[0, 1, 2], [0, 1, 2]]))

    def test_atom_indices_2d_wrong_batch_size_raises(self) -> None:
        """[B', 2] atom_indices where B' != B raises ValueError."""
        data_list = [
            AtomicData(
                atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
                positions=torch.zeros(2, 3),
            )
        ] * 3  # B=3
        batch = Batch.from_data_list(data_list)
        # supply [2, 2] instead of [3, 2]
        with pytest.raises(ValueError, match="first dimension must equal B"):
            pair_distance(batch, torch.tensor([[0, 1], [0, 1]]))

    def test_atom_indices_3d_raises(self) -> None:
        """3-D atom_indices raises ValueError."""
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=torch.zeros(2, 3),
        )
        batch = Batch.from_data_list([data])
        with pytest.raises(ValueError, match="1-D.*or.*2-D"):
            pair_distance(batch, torch.zeros(1, 2, 1, dtype=torch.long))

    def test_atom_indices_valid_shapes_accepted(self) -> None:
        """Shape [2] and [B, 2] with integer dtype are accepted."""
        data_list = [
            AtomicData(
                atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
                positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            )
        ] * 2
        batch = Batch.from_data_list(data_list)
        # [2] shared
        d1 = pair_distance(batch, torch.tensor([0, 1]))
        assert d1.shape == (2, 1)
        # [B, 2] per-graph
        d2 = pair_distance(batch, torch.tensor([[0, 1], [0, 1]]))
        assert d2.shape == (2, 1)

    # --- bounds checking ---

    def test_out_of_range_shared_index_raises(self) -> None:
        """Shared [2] index that exceeds graph size raises IndexError, not silent wrap."""
        data_list = [
            AtomicData(
                atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
                positions=torch.zeros(2, 3),
            ),
            AtomicData(
                atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
                positions=torch.zeros(2, 3),
            ),
        ]
        batch = Batch.from_data_list(data_list)
        # Local index 5 is valid for neither 2-atom graph.
        idx = torch.tensor([0, 5])
        with pytest.raises(IndexError, match="out of range"):
            pair_distance(batch, idx)

    def test_out_of_range_per_graph_index_raises(self) -> None:
        """Per-graph [B, 2] index out of range for one graph raises IndexError."""
        data_list = [
            AtomicData(
                atomic_numbers=torch.tensor([6, 6, 6], dtype=torch.long),
                positions=torch.zeros(3, 3),
            ),
            AtomicData(
                atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
                positions=torch.zeros(2, 3),
            ),
        ]
        batch = Batch.from_data_list(data_list)
        # Graph 1 has only 2 atoms; local index 2 is out of range.
        idx = torch.tensor([[0, 1], [0, 2]])
        with pytest.raises(IndexError, match="out of range"):
            pair_distance(batch, idx)

    def test_negative_index_raises(self) -> None:
        """Negative atom index raises IndexError."""
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=torch.zeros(2, 3),
        )
        batch = Batch.from_data_list([data])
        idx = torch.tensor([-1, 0])
        with pytest.raises(IndexError, match="negative"):
            pair_distance(batch, idx)

    def test_variable_size_batch_no_silent_cross_graph(self) -> None:
        """Out-of-range index must not silently reference the next graph's atoms.

        Regression for the reported bug: in a variable-size batch, adding
        batch_ptr[b] to an out-of-range local index wraps into graph b+1's
        rows without error.  The bounds check must catch this before any
        indexing occurs.
        """
        # Graph 0: 2 atoms, graph 1: 4 atoms.
        # Without the bounds check, local index 3 on graph 0 would silently
        # resolve to global row 3, which is atom 1 of graph 1.
        data0 = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        )
        data1 = AtomicData(
            atomic_numbers=torch.tensor([6, 6, 6, 6], dtype=torch.long),
            positions=torch.tensor(
                [[10.0, 0.0, 0.0], [11.0, 0.0, 0.0], [12.0, 0.0, 0.0], [13.0, 0.0, 0.0]]
            ),
        )
        batch = Batch.from_data_list([data0, data1])
        # Local index 3 is valid for graph 1 but out of range for graph 0.
        idx = torch.tensor([[0, 3], [0, 1]])
        with pytest.raises(IndexError, match="out of range"):
            pair_distance(batch, idx)

    # --- nonperiodic with explicit cell (pbc=False) ----------------------

    def test_degenerate_cell_with_pbc_false_does_not_raise(self, device: str) -> None:
        """cell=zeros + pbc=False must not raise LinAlgError.

        Regression: the old guard ``has_cell and has_pbc`` entered _apply_mic
        even when all pbc flags were False, hitting torch.linalg.inv on
        whatever cell was present.  A zero cell causes LinAlgError there.
        """
        positions = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        cell = torch.zeros(1, 3, 3)  # degenerate — not invertible
        pbc = torch.tensor([[False, False, False]])
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=positions,
            cell=cell,
            pbc=pbc,
        )
        batch = Batch.from_data_list([data]).to(device)
        idx = torch.tensor([0, 1], device=device)
        # Must not raise; MIC must be skipped; Euclidean distance = 3 Å.
        d = pair_distance(batch, idx)
        assert torch.allclose(d, torch.tensor([[3.0]], device=device), atol=1e-5)

    def test_valid_cell_with_pbc_false_uses_euclidean(self, device: str) -> None:
        """Valid non-degenerate cell + pbc=False returns plain Euclidean distance."""
        positions = torch.tensor([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
        box = 10.0
        cell = torch.eye(3).unsqueeze(0) * box
        pbc = torch.tensor([[False, False, False]])
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=positions,
            cell=cell,
            pbc=pbc,
        )
        batch = Batch.from_data_list([data]).to(device)
        idx = torch.tensor([0, 1], device=device)
        d = pair_distance(batch, idx)
        # MIC would fold to 0.2 Å; Euclidean is 9.8 Å.
        assert torch.allclose(d, torch.tensor([[9.8]], device=device), atol=1e-4)

    # --- nonperiodic ---

    def test_nonperiodic_known_value(self, device: str) -> None:
        """pair_distance = Euclidean distance for nonperiodic systems."""
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=torch.tensor([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]),
        )
        batch = Batch.from_data_list([data]).to(device)
        idx = torch.tensor([0, 1], device=device)
        d = pair_distance(batch, idx)
        assert d.shape == (1, 1)
        assert torch.allclose(d, torch.tensor([[5.0]], device=device), atol=1e-5)

    def test_nonperiodic_batch_of_two(self, device: str) -> None:
        """Shared atom_indices work correctly across multiple graphs."""
        pos0 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        pos1 = torch.tensor([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        d0_ref = 1.0
        d1_ref = 2.0

        data_list = [
            AtomicData(
                atomic_numbers=torch.tensor([6, 6], dtype=torch.long), positions=pos0
            ),
            AtomicData(
                atomic_numbers=torch.tensor([6, 6], dtype=torch.long), positions=pos1
            ),
        ]
        batch = Batch.from_data_list(data_list).to(device)
        idx = torch.tensor([0, 1], device=device)
        d = pair_distance(batch, idx)
        assert d.shape == (2, 1)
        assert torch.allclose(d[0, 0], torch.tensor(d0_ref, device=device), atol=1e-5)
        assert torch.allclose(d[1, 0], torch.tensor(d1_ref, device=device), atol=1e-5)

    def test_per_graph_atom_indices(self, device: str) -> None:
        """[B, 2] atom_indices select different pairs per graph."""
        data_list = [
            AtomicData(
                atomic_numbers=torch.tensor([6, 6, 6], dtype=torch.long),
                positions=torch.tensor(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
                ),
            ),
            AtomicData(
                atomic_numbers=torch.tensor([6, 6, 6], dtype=torch.long),
                positions=torch.tensor(
                    [[0.0, 0.0, 0.0], [0.0, 0.0, 5.0], [2.0, 0.0, 0.0]]
                ),
            ),
        ]
        batch = Batch.from_data_list(data_list).to(device)
        # graph 0: atoms 0-1 → dist 1; graph 1: atoms 0-2 → dist 2
        idx = torch.tensor([[0, 1], [0, 2]], device=device)
        d = pair_distance(batch, idx)
        assert d.shape == (2, 1)
        assert torch.allclose(d[0, 0], torch.tensor(1.0, device=device), atol=1e-5)
        assert torch.allclose(d[1, 0], torch.tensor(2.0, device=device), atol=1e-5)

    # --- cubic periodic ---

    def test_periodic_cubic_mic(self, device: str) -> None:
        """MIC selects the nearest image in a cubic cell."""
        box = 10.0
        # Atom 0 at 0.1, atom 1 at 9.9 → naive dist = 9.8, MIC dist = 0.2
        positions = torch.tensor([[0.1, 0.0, 0.0], [9.9, 0.0, 0.0]])
        cell = torch.eye(3).unsqueeze(0) * box  # [1, 3, 3]
        pbc = torch.tensor([[True, True, True]])  # [1, 3]
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=positions,
            cell=cell,
            pbc=pbc,
        )
        batch = Batch.from_data_list([data]).to(device)
        idx = torch.tensor([0, 1], device=device)
        d = pair_distance(batch, idx)
        assert torch.allclose(d, torch.tensor([[0.2]], device=device), atol=1e-4)

    # --- triclinic MIC ---

    def test_triclinic_mic_known_value(self, device: str) -> None:
        """MIC distance in triclinic cell: verify against manually computed value."""
        # Cell: a=[4,0,0], b=[1,4,0], c=[0,0,4] — [1,3,3]
        cell = torch.tensor([[[4.0, 0.0, 0.0], [1.0, 4.0, 0.0], [0.0, 0.0, 4.0]]])
        pbc = torch.tensor([[True, True, True]])  # [1, 3]
        # Atom i at origin, atom j across boundary (Cartesian [3.5, 0, 0])
        # Fractional: j @ cell^{-1}; round; nearest image is [-0.5*a] away
        pos_i = torch.tensor([[0.0, 0.0, 0.0]])
        pos_j = torch.tensor([[3.5, 0.0, 0.0]])
        positions = torch.cat([pos_i, pos_j], dim=0)
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=positions,
            cell=cell,
            pbc=pbc,
        )
        batch = Batch.from_data_list([data]).to(device)
        idx = torch.tensor([0, 1], device=device)
        d = pair_distance(batch, idx)
        # Naive: 3.5; MIC: |3.5 - 4| = 0.5 (nearest image in a-direction)
        assert torch.allclose(d, torch.tensor([[0.5]], device=device), atol=1e-4)

    def test_unreduced_cell_raises(self, device: str) -> None:
        """Unreduced cell raises ValueError with a clear message.

        Regression for the reported bug: cell ``[[1,0,0],[10,0.1,0],[0,0,10]]``
        with fractional displacement ``[0,0.49,0]`` requires offset ``[−5,0,0]``,
        which lies outside the 27-image search range.  The old code returned
        ≈ 3.9 Å silently; the new code detects the non-reduced cell and raises.
        """
        # Minkowski check: |a1·a2| = 10 > 0.5*min(|a1|²,|a2|²) = 0.5 — fails.
        cell = torch.tensor([[[1.0, 0.0, 0.0], [10.0, 0.1, 0.0], [0.0, 0.0, 10.0]]])
        pbc = torch.tensor([[True, True, True]])
        positions = torch.tensor([[0.0, 0.0, 0.0], [0.0, 4.9, 0.049]])
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=positions,
            cell=cell,
            pbc=pbc,
        )
        batch = Batch.from_data_list([data]).to(device)
        idx = torch.tensor([0, 1], device=device)
        with pytest.raises(ValueError, match="Minkowski"):
            pair_distance(batch, idx)

    def test_triclinic_mic_reduced_skewed_cell_27image_correct(
        self, device: str
    ) -> None:
        """27-image search returns the correct MIC for a Minkowski-reduced skewed cell.

        Cell: ``[[2,0,0],[0.8,2,0],[0,0,10]]`` — satisfies Minkowski conditions
        (``|a0·a1| = 1.6 ≤ 0.5·min(4, 4.64) = 2.0``).

        Fractional displacement ``[0.49, 0.49, 0]``:
          - Componentwise rounding keeps ``[0.49, 0.49, 0]`` → Cartesian ≈ 1.69 Å.
          - Correct MIC (offset ``[−1, 0, 0]``) → Cartesian ≈ 1.16 Å.

        Componentwise rounding alone would return the wrong (longer) image;
        the 27-image search returns the correct one.
        """
        # Verify Minkowski condition holds: |a0·a1| = 1.6 <= 0.5*min(4,4.64) = 2.0 ✓
        cell = torch.tensor([[[2.0, 0.0, 0.0], [0.8, 2.0, 0.0], [0.0, 0.0, 10.0]]])
        pbc = torch.tensor([[True, True, True]])

        # pos_j: fractional [0.49, 0.49, 0]
        # Cartesian = 0.49*[2,0,0] + 0.49*[0.8,2,0] = [1.372, 0.98, 0]
        pos_i = torch.tensor([[0.0, 0.0, 0.0]])
        pos_j = torch.tensor([[1.372, 0.98, 0.0]])
        positions = torch.cat([pos_i, pos_j], dim=0)

        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=positions,
            cell=cell,
            pbc=pbc,
        )
        batch = Batch.from_data_list([data]).to(device)
        idx = torch.tensor([0, 1], device=device)
        d = pair_distance(batch, idx)

        # Componentwise rounding gives ≈ 1.687 Å; correct MIC is ≈ 1.164 Å.
        naive_dist = torch.linalg.vector_norm(pos_j - pos_i).item()
        assert d.item() < naive_dist * 0.8, (
            f"MIC distance {d.item():.4f} Å should be shorter than the naive "
            f"distance {naive_dist:.4f} Å — 27-image search may not be working."
        )
        assert d.item() < 1.20, (
            f"Expected MIC distance ≈ 1.164 Å, got {d.item():.4f} Å."
        )

    # --- gradients ---

    def test_gradient_nonperiodic(self, device: str) -> None:
        """pair_distance gradient w.r.t. positions is correct (finite diff)."""
        torch.manual_seed(7)
        positions = torch.randn(3, 3, device=device, dtype=torch.float64)
        idx = torch.tensor([0, 2], device=device)

        # Use gradcheck with a wrapper that creates a fresh batch
        def _fn(pos: Tensor) -> Tensor:
            batch_local = Batch.from_data_list(
                [
                    AtomicData(
                        atomic_numbers=torch.tensor([6, 6, 6], dtype=torch.long),
                        positions=pos.detach(),
                    )
                ]
            ).to(device)
            batch_local["positions"] = pos  # keep grad-tracking leaf
            return pair_distance(batch_local, idx)

        pos_double = positions.detach().clone().requires_grad_(True)
        torch.autograd.gradcheck(_fn, (pos_double,), eps=1e-4, atol=1e-3, rtol=1e-3)

    def test_gradient_periodic(self, device: str) -> None:
        """pair_distance gradient is finite and non-zero for periodic systems."""
        box = 8.0
        positions = torch.tensor([[1.0, 0.0, 0.0], [6.0, 0.0, 0.0]], device=device)
        cell = torch.eye(3).unsqueeze(0).to(device) * box  # [1, 3, 3]
        pbc = torch.tensor([[True, True, True]])  # [1, 3]
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=positions.cpu(),
            cell=cell.cpu(),
            pbc=pbc,
        )
        batch = Batch.from_data_list([data]).to(device)
        pos_leaf = batch.positions.detach().requires_grad_(True)
        batch["positions"] = pos_leaf
        idx = torch.tensor([0, 1], device=device)
        d = pair_distance(batch, idx)
        d.sum().backward()
        assert pos_leaf.grad is not None
        assert pos_leaf.grad.isfinite().all()
        assert (pos_leaf.grad.abs() > 0).any()

    # --- tests away from half-cell tie ---

    def test_not_at_half_cell_tie(self, device: str) -> None:
        """Distance is computed correctly well away from the MIC discontinuity."""
        box = 10.0
        # Position atom j at 3.0 from atom i (clearly not near 5.0 = box/2)
        positions = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        cell = torch.eye(3).unsqueeze(0) * box  # [1, 3, 3]
        pbc = torch.tensor([[True, True, True]])  # [1, 3]
        data = AtomicData(
            atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
            positions=positions,
            cell=cell,
            pbc=pbc,
        )
        batch = Batch.from_data_list([data]).to(device)
        idx = torch.tensor([0, 1], device=device)
        d = pair_distance(batch, idx)
        assert torch.allclose(d, torch.tensor([[3.0]], device=device), atol=1e-5)


# ===========================================================================
# 5. aggregate_bias_results
# ===========================================================================


class TestAggregateBiasResults:
    """Tests for bias aggregation."""

    def test_empty_list_returns_empty_result(self) -> None:
        r = aggregate_bias_results([])
        assert r.energy is None
        assert r.forces is None

    def test_single_result_passthrough(self) -> None:
        e = torch.tensor([[1.0]])
        f = torch.zeros(3, 3)
        r = aggregate_bias_results([BiasResult(energy=e, forces=f)])
        assert torch.allclose(r.energy, e)
        assert torch.allclose(r.forces, f)

    def test_energy_summed(self) -> None:
        r1 = BiasResult(energy=torch.tensor([[1.0]]))
        r2 = BiasResult(energy=torch.tensor([[2.0]]))
        agg = aggregate_bias_results([r1, r2])
        assert torch.allclose(agg.energy, torch.tensor([[3.0]]))

    def test_forces_summed(self) -> None:
        f1 = torch.ones(4, 3)
        f2 = torch.ones(4, 3) * 2.0
        r1 = BiasResult(forces=f1)
        r2 = BiasResult(forces=f2)
        agg = aggregate_bias_results([r1, r2])
        assert torch.allclose(agg.forces, torch.ones(4, 3) * 3.0)

    def test_none_fields_handled(self) -> None:
        r1 = BiasResult(energy=torch.tensor([[1.0]]))
        r2 = BiasResult(forces=torch.zeros(2, 3))
        agg = aggregate_bias_results([r1, r2])
        assert agg.energy is not None
        assert agg.forces is not None

    def test_virial_summed(self) -> None:
        v1 = torch.ones(1, 3, 3)
        v2 = torch.ones(1, 3, 3) * 2.0
        r1 = BiasResult(virial=v1)
        r2 = BiasResult(virial=v2)
        agg = aggregate_bias_results([r1, r2])
        assert torch.allclose(agg.virial, torch.ones(1, 3, 3) * 3.0)

    def test_mixed_stress_and_virial_raises(self) -> None:
        """Mixing stress from one result and virial from another raises ValueError.

        The error must come from aggregate_bias_results itself (not from
        BiasResult.__post_init__) with a message that identifies which
        result indices contributed each field.
        """
        r_stress = BiasResult(stress=torch.zeros(1, 3, 3))
        r_virial = BiasResult(virial=torch.zeros(1, 3, 3))
        with pytest.raises(ValueError, match="stress.*virial|virial.*stress"):
            aggregate_bias_results([r_stress, r_virial])

    def test_mixed_stress_and_virial_error_identifies_indices(self) -> None:
        """Error message must identify which result indices are responsible."""
        results = [
            BiasResult(energy=torch.zeros(1, 1)),  # index 0 — no cell response
            BiasResult(stress=torch.zeros(1, 3, 3)),  # index 1 — stress
            BiasResult(energy=torch.zeros(1, 1)),  # index 2 — no cell response
            BiasResult(virial=torch.zeros(1, 3, 3)),  # index 3 — virial
        ]
        with pytest.raises(ValueError, match=r"\[1\].*\[3\]|\[3\].*\[1\]"):
            aggregate_bias_results(results)

    def test_all_stress_aggregates_correctly(self) -> None:
        """Multiple stress contributions are summed without raising."""
        r1 = BiasResult(stress=torch.ones(1, 3, 3))
        r2 = BiasResult(stress=torch.ones(1, 3, 3) * 2.0)
        agg = aggregate_bias_results([r1, r2])
        assert agg.stress is not None
        assert agg.virial is None
        assert torch.allclose(agg.stress, torch.ones(1, 3, 3) * 3.0)

    def test_duplicate_observable_key_raises(self) -> None:
        r1 = BiasResult(observables={"bias/a/cv": torch.zeros(1)})
        r2 = BiasResult(observables={"bias/a/cv": torch.ones(1)})
        with pytest.raises(ValueError, match="duplicate observable key"):
            aggregate_bias_results([r1, r2])

    def test_distinct_observable_keys_merged(self) -> None:
        r1 = BiasResult(observables={"bias/a/cv": torch.tensor([1.0])})
        r2 = BiasResult(observables={"bias/b/cv": torch.tensor([2.0])})
        agg = aggregate_bias_results([r1, r2])
        assert "bias/a/cv" in agg.observables
        assert "bias/b/cv" in agg.observables

    def test_different_registration_orders_same_result(self) -> None:
        """Aggregation must be order-independent (commutativity for sum)."""
        e1 = torch.tensor([[1.5]])
        e2 = torch.tensor([[0.5]])
        agg_ab = aggregate_bias_results([BiasResult(energy=e1), BiasResult(energy=e2)])
        agg_ba = aggregate_bias_results([BiasResult(energy=e2), BiasResult(energy=e1)])
        assert torch.allclose(agg_ab.energy, agg_ba.energy)


# ===========================================================================
# 6. torch.compile — fullgraph and graph-break tests
# ===========================================================================


class TestCompile:
    """Verifies what can and cannot be compiled with ``torch.compile``.

    * :func:`pair_distance` — compiles with ``fullgraph=True``.
    * :func:`aggregate_bias_results` — compiles with ``fullgraph=True`` for
      fixed-size input lists.
    * ``ConservativeBias.evaluate()`` — does **not** compile with
      ``fullgraph=True``.  The root cause is
      ``pos_leaf = positions.detach().requires_grad_(True)``:
      ``torch.compile`` does not support ``.requires_grad_()`` mutation.
      **Chosen approach:** compile :meth:`energy` independently; keep
      ``evaluate()`` as an eager orchestration wrapper.
      ``EnhancedSampling(compile_biases=True)`` will compile each bias's
      ``energy()`` override, not ``evaluate()``.

    Tests in this class:

    * ``fullgraph=True`` tests for compile-capable paths (``pair_distance``,
      ``aggregate_bias_results``).
    * ``fullgraph=False`` tests for ``ConservativeBias.evaluate()`` (allow
      graph break; verify correctness and no memory growth).
    * ``fullgraph=True`` test for compiling ``energy()`` only.
    """

    @staticmethod
    def _compile_kw_full(device: str) -> dict:
        """Compile kwargs for fully-compilable paths (fullgraph=True)."""
        kw: dict = {"fullgraph": True}
        if device == "cuda":
            kw["backend"] = "inductor"
        return kw

    @staticmethod
    def _compile_kw_allow_breaks(device: str) -> dict:
        """Compile kwargs allowing graph breaks (for evaluate())."""
        kw: dict = {"fullgraph": False}
        if device == "cuda":
            kw["backend"] = "inductor"
        return kw

    # ------------------------------------------------------------------
    # pair_distance — fully compilable (fullgraph=True)
    # ------------------------------------------------------------------

    def test_pair_distance_compiles_fullgraph(self, device: str) -> None:
        """pair_distance compiles with fullgraph=True (no graph breaks)."""
        batch = _make_nonperiodic_batch(n_graphs=2, atoms_per_graph=3, device=device)
        idx = torch.tensor([0, 1], device=device)

        compiled = torch.compile(pair_distance, **self._compile_kw_full(device))
        for _ in range(3):
            d = compiled(batch, idx)
        assert d.shape == (2, 1)
        assert d.isfinite().all()

    def test_pair_distance_compile_agrees_eager(self, device: str) -> None:
        """Compiled pair_distance matches eager output within tolerance."""
        batch = _make_nonperiodic_batch(n_graphs=2, atoms_per_graph=4, device=device)
        idx = torch.tensor([0, 1], device=device)

        d_eager = pair_distance(batch, idx)
        compiled = torch.compile(pair_distance, **self._compile_kw_full(device))
        d_compiled = compiled(batch, idx)
        assert torch.allclose(d_eager, d_compiled, atol=1e-5)

    def test_pair_distance_periodic_mic_compiles_fullgraph(self, device: str) -> None:
        """pair_distance with periodic MIC compiles with fullgraph=True."""
        # Reset dynamo to avoid recompile_limit from previous compile tests
        # sharing the pair_distance compiled-function cache.
        torch._dynamo.reset()

        batch = _make_cubic_batch(n_graphs=2, atoms_per_graph=3, box=6.0, device=device)
        idx = torch.tensor([0, 1], device=device)

        compiled = torch.compile(pair_distance, **self._compile_kw_full(device))
        for _ in range(5):
            d = compiled(batch, idx)
        assert d.isfinite().all()

    # ------------------------------------------------------------------
    # aggregate_bias_results — fully compilable (fullgraph=True)
    # ------------------------------------------------------------------

    def test_aggregate_compiles_fullgraph(self, device: str) -> None:
        """aggregate_bias_results compiles with fullgraph=True."""
        e1 = torch.ones(2, 1, device=device)
        e2 = torch.ones(2, 1, device=device) * 2.0
        f1 = torch.ones(8, 3, device=device)
        f2 = torch.ones(8, 3, device=device) * 0.5

        def _agg() -> BiasResult:
            return aggregate_bias_results(
                [BiasResult(energy=e1, forces=f1), BiasResult(energy=e2, forces=f2)]
            )

        compiled = torch.compile(_agg, **self._compile_kw_full(device))
        result = compiled()
        assert result.energy is not None
        assert torch.allclose(result.energy, torch.full((2, 1), 3.0, device=device))

    # ------------------------------------------------------------------
    # ConservativeBias.energy() — compilable when subclassed correctly
    # ------------------------------------------------------------------

    def test_conservative_energy_fn_compiles_fullgraph(self, device: str) -> None:
        """ConservativeBias.energy() compiles with fullgraph=True.

        This is the actual compile target when compile_biases=True.
        evaluate() stays eager; energy() is compiled per the fallback.
        """
        batch = _make_nonperiodic_batch(n_graphs=2, atoms_per_graph=4, device=device)
        bias = _QuadraticBias(k=1.0)

        # Simulate the runner compiling energy() not evaluate()
        compiled_energy = torch.compile(bias.energy, **self._compile_kw_full(device))

        # Temporarily inject fresh positions leaf (as evaluate() does eagerly)
        pos_leaf = batch.positions.detach().requires_grad_(True)
        batch["positions"] = pos_leaf
        for _ in range(3):
            e = compiled_energy(batch)
        batch["positions"] = pos_leaf.detach()
        assert e.shape == (2, 1)
        assert e.isfinite().all()

    # ------------------------------------------------------------------
    # ConservativeBias.evaluate() — runs with graph breaks (fullgraph=False)
    # ------------------------------------------------------------------

    def test_conservative_bias_evaluate_runs_correctly(self, device: str) -> None:
        """ConservativeBias.evaluate() produces correct forces (eager mode).

        evaluate() is NOT compiled with fullgraph=True (see spike finding).
        It is the eager orchestration wrapper; energy() is what gets compiled.
        """
        batch = _make_nonperiodic_batch(n_graphs=2, atoms_per_graph=4, device=device)
        bias = _QuadraticBias(k=1.0)
        result = bias.evaluate(batch)
        assert result.forces is not None
        assert result.forces.shape == (8, 3)
        assert result.forces.isfinite().all()

    def test_conservative_bias_compile_allows_graph_break(self, device: str) -> None:
        """ConservativeBias.evaluate() can run under torch.compile(fullgraph=False).

        With fullgraph=False the graph break at requires_grad_() is allowed.
        Output agrees with eager.
        """
        batch = _make_nonperiodic_batch(n_graphs=2, atoms_per_graph=3, device=device)
        bias = _QuadraticBias(k=2.0)

        r_eager = bias.evaluate(batch)
        compiled = torch.compile(bias.evaluate, **self._compile_kw_allow_breaks(device))
        r_compiled = compiled(batch)

        assert r_eager.energy is not None and r_compiled.energy is not None
        assert torch.allclose(r_eager.energy, r_compiled.energy, atol=1e-4)
        assert r_eager.forces is not None and r_compiled.forces is not None
        assert torch.allclose(r_eager.forces, r_compiled.forces, atol=1e-4)

    def test_no_memory_growth_eager_evaluate_10_calls(self, device: str) -> None:
        """Eager evaluate() must not grow GPU memory across 10 calls."""
        batch = _make_nonperiodic_batch(n_graphs=4, atoms_per_graph=8, device=device)
        bias = _QuadraticBias(k=1.0)

        # Warm up
        for _ in range(3):
            bias.evaluate(batch)

        gc.collect()
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            mem_start = torch.cuda.memory_allocated()

        for _ in range(10):
            bias.evaluate(batch)

        if device == "cuda":
            torch.cuda.synchronize()
            mem_end = torch.cuda.memory_allocated()
            assert mem_end - mem_start <= 1 * 1024 * 1024, (
                f"GPU memory grew by {mem_end - mem_start} bytes across 10 evaluate() calls"
            )

    def test_pair_distance_inside_energy_compiles(self, device: str) -> None:
        """pair_distance used as CV inside energy() compiles with fullgraph=True."""
        batch = _make_nonperiodic_batch(n_graphs=2, atoms_per_graph=4, device=device)
        idx = torch.tensor([0, 1], device=device)
        bias = _PairDistanceBias(atom_indices=idx, k=1.0)

        # Compile energy() — the intended compile target
        compiled_energy = torch.compile(bias.energy, **self._compile_kw_full(device))
        pos_leaf = batch.positions.detach().requires_grad_(True)
        batch["positions"] = pos_leaf
        for _ in range(3):
            e = compiled_energy(batch)
        batch["positions"] = pos_leaf.detach()
        assert e.isfinite().all()
