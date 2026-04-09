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
"""Unit tests for ``nvalchemi.dynamics.hooks.bias`` — Tier 1 bias hook.

Covers :class:`BiasedPotentialHook`.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.dynamics.hooks.bias import BiasedPotentialHook
from nvalchemi.hooks import Hook, HookContext
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    n_graphs: int = 2,
    atoms_per_graph: int = 3,
    device: str = "cpu",
) -> Batch:
    data_list = [
        AtomicData(
            atomic_numbers=torch.tensor([6] * atoms_per_graph, dtype=torch.long),
            positions=torch.randn(atoms_per_graph, 3),
        )
        for _ in range(n_graphs)
    ]
    batch = Batch.from_data_list(data_list).to(device)
    total_atoms = n_graphs * atoms_per_graph
    batch["forces"] = torch.randn(total_atoms, 3, device=device)
    batch["energy"] = torch.randn(n_graphs, 1, device=device)
    return batch


def _make_dynamics() -> BaseDynamics:
    return BaseDynamics(DemoModelWrapper(DemoModel()))


def _make_ctx(batch: Batch, dynamics: BaseDynamics) -> HookContext:
    """Build a HookContext from a batch and dynamics instance."""
    converged = dynamics._last_converged
    if converged is not None:
        mask = torch.zeros(
            batch.num_graphs, dtype=torch.bool, device=batch.positions.device
        )
        mask[converged] = True
    else:
        mask = None
    return HookContext(
        batch=batch,
        step_count=dynamics.step_count,
        model=dynamics.model,
        converged_mask=mask,
        global_rank=dynamics.global_rank,
    )


# ===========================================================================
# BiasedPotentialHook
# ===========================================================================


class TestBiasedPotentialHook:
    """Test suite for :class:`BiasedPotentialHook`."""

    def test_constant_bias_adds_correctly(self, device: str) -> None:
        """Verify constant bias is added to forces and energy."""
        batch = _make_batch(device=device)
        dynamics = _make_dynamics()

        forces_before = batch.forces.clone()
        energies_before = batch.energy.clone()

        bias_e = torch.ones_like(batch.energy) * 0.5
        bias_f = torch.ones_like(batch.forces) * 0.1

        hook = BiasedPotentialHook(bias_fn=lambda b: (bias_e, bias_f))
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

        assert torch.allclose(batch.forces, forces_before + 0.1)
        assert torch.allclose(batch.energy, energies_before + 0.5)

    def test_inplace_mutation(self, device: str) -> None:
        """Verify forces and energy are modified in-place."""
        batch = _make_batch(device=device)
        dynamics = _make_dynamics()

        forces_ref = batch.forces
        energies_ref = batch.energy

        def zero_bias(b):
            return torch.zeros_like(b.energy), torch.zeros_like(b.forces)

        hook = BiasedPotentialHook(bias_fn=zero_bias)
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

        assert forces_ref is batch.forces
        assert energies_ref is batch.energy

    def test_not_inplace(self, device: str) -> None:
        """Verify inplace=False replaces tensors instead of mutating."""
        batch = _make_batch(device=device)
        dynamics = _make_dynamics()

        forces_before = batch.forces.clone()
        energies_before = batch.energy.clone()
        forces_ref = batch.forces
        energies_ref = batch.energy

        bias_e = torch.ones_like(batch.energy) * 0.5
        bias_f = torch.ones_like(batch.forces) * 0.1

        hook = BiasedPotentialHook(bias_fn=lambda b: (bias_e, bias_f), inplace=False)
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

        assert torch.allclose(batch.forces, forces_before + 0.1)
        assert torch.allclose(batch.energy, energies_before + 0.5)
        # Original tensors should NOT have been mutated
        assert torch.allclose(forces_ref, forces_before)
        assert torch.allclose(energies_ref, energies_before)

    def test_energy_shape_mismatch_raises(self, device: str) -> None:
        """Verify RuntimeError when bias_energy has wrong shape."""
        batch = _make_batch(n_graphs=2, device=device)
        dynamics = _make_dynamics()

        def bad_energy(b):
            return torch.zeros(1, 1, device=device), torch.zeros_like(b.forces)

        hook = BiasedPotentialHook(bias_fn=bad_energy)
        ctx = _make_ctx(batch, dynamics)
        with pytest.raises(RuntimeError, match="bias_energy shape"):
            hook(ctx, DynamicsStage.AFTER_COMPUTE)

    def test_forces_shape_mismatch_raises(self, device: str) -> None:
        """Verify RuntimeError when bias_forces has wrong shape."""
        batch = _make_batch(device=device)
        dynamics = _make_dynamics()

        def bad_forces(b):
            return torch.zeros_like(b.energy), torch.zeros(1, 3, device=device)

        hook = BiasedPotentialHook(bias_fn=bad_forces)
        ctx = _make_ctx(batch, dynamics)
        with pytest.raises(RuntimeError, match="bias_forces shape"):
            hook(ctx, DynamicsStage.AFTER_COMPUTE)

    def test_zero_bias_is_noop(self, device: str) -> None:
        """Verify zero bias does not change forces or energy."""
        batch = _make_batch(device=device)
        dynamics = _make_dynamics()

        forces_before = batch.forces.clone()
        energies_before = batch.energy.clone()

        def zero_bias(b):
            return torch.zeros_like(b.energy), torch.zeros_like(b.forces)

        hook = BiasedPotentialHook(bias_fn=zero_bias)
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

        assert torch.allclose(batch.forces, forces_before)
        assert torch.allclose(batch.energy, energies_before)

    def test_stage_is_after_compute(self) -> None:
        hook = BiasedPotentialHook(bias_fn=lambda b: (b.energy, b.forces))
        assert hook.stage == DynamicsStage.AFTER_COMPUTE

    def test_default_frequency_is_one(self) -> None:
        hook = BiasedPotentialHook(bias_fn=lambda b: (b.energy, b.forces))
        assert hook.frequency == 1

    def test_custom_frequency(self) -> None:
        hook = BiasedPotentialHook(bias_fn=lambda b: (b.energy, b.forces), frequency=5)
        assert hook.frequency == 5

    def test_is_hook(self) -> None:
        hook = BiasedPotentialHook(bias_fn=lambda b: (b.energy, b.forces))
        assert isinstance(hook, Hook)

    def test_interaction_with_nan_detector(self, device: str) -> None:
        """Verify bias + NaN detector works when bias produces finite values."""
        from nvalchemi.dynamics.hooks.safety import NaNDetectorHook

        batch = _make_batch(device=device)
        dynamics = _make_dynamics()

        def small_bias(b):
            return torch.ones_like(b.energy) * 0.01, torch.ones_like(b.forces) * 0.01

        bias_hook = BiasedPotentialHook(bias_fn=small_bias)
        nan_hook = NaNDetectorHook()
        ctx = _make_ctx(batch, dynamics)

        bias_hook(ctx, DynamicsStage.AFTER_COMPUTE)
        nan_hook(ctx, DynamicsStage.AFTER_COMPUTE)  # should not raise


class TestBiasedPotentialHookCompile:
    """Verify BiasedPotentialHook works under torch.compile."""

    @staticmethod
    def _compile_kwargs(device: str, *, fullgraph: bool = True) -> dict:
        kw: dict = {"fullgraph": fullgraph}
        if device == "cuda":
            kw["backend"] = "cudagraphs"
        return kw

    def test_compiles_fullgraph(self, device: str) -> None:
        """BiasedPotentialHook._apply_bias compiles with fullgraph=True."""
        batch = _make_batch(n_graphs=1, atoms_per_graph=3, device=device)
        batch["forces"] = torch.ones(3, 3, device=device)
        batch["energy"] = torch.ones(1, 1, device=device)

        bias_e = torch.ones(1, 1, device=device) * 0.5
        bias_f = torch.ones(3, 3, device=device) * 0.1

        hook = BiasedPotentialHook(bias_fn=lambda b: (bias_e, bias_f))
        compiled = torch.compile(hook._apply_bias, **self._compile_kwargs(device))
        compiled(batch)

        expected_e = torch.tensor([[1.5]], device=device, dtype=batch.energy.dtype)
        expected_f = torch.ones(3, 3, device=device, dtype=batch.forces.dtype) * 1.1
        assert torch.allclose(batch.energy, expected_e)
        assert torch.allclose(batch.forces, expected_f)
