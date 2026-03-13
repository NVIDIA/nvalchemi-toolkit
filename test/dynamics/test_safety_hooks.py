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
"""Unit tests for ``nvalchemi.dynamics.hooks.safety`` — Tier 0 safety hooks.

Covers :class:`NaNDetectorHook` and :class:`MaxForceClampHook`.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import BaseDynamics, Hook, HookStageEnum
from nvalchemi.dynamics.hooks.safety import MaxForceClampHook, NaNDetectorHook
from nvalchemi.models.demo import DemoModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    n_graphs: int = 2,
    atoms_per_graph: int = 3,
) -> Batch:
    """Create a deterministic test batch.

    Parameters
    ----------
    n_graphs : int
        Number of graphs.
    atoms_per_graph : int
        Atoms per graph (uniform).

    Returns
    -------
    Batch
        A batch with pre-allocated forces and energies.
    """
    data_list = [
        AtomicData(
            atomic_numbers=torch.tensor([6] * atoms_per_graph, dtype=torch.long),
            positions=torch.randn(atoms_per_graph, 3),
        )
        for _ in range(n_graphs)
    ]
    batch = Batch.from_data_list(data_list)
    batch.__dict__["forces"] = torch.randn(batch.num_nodes, 3)
    batch.__dict__["energies"] = torch.randn(batch.num_graphs, 1)
    return batch


def _make_dynamics() -> BaseDynamics:
    """Create a minimal dynamics engine for testing hooks.

    Returns
    -------
    BaseDynamics
        Dynamics with a DemoModelWrapper and step_count=0.
    """
    model = DemoModelWrapper()
    return BaseDynamics(model)


# ===========================================================================
# NaNDetectorHook
# ===========================================================================


class TestNaNDetectorHook:
    """Test suite for :class:`NaNDetectorHook`."""

    def test_no_nan_is_noop(self) -> None:
        """Verify no error when all values are finite."""
        hook = NaNDetectorHook()
        batch = _make_batch()
        dynamics = _make_dynamics()
        # Should not raise
        hook(batch, dynamics)

    def test_nan_in_forces_raises(self) -> None:
        """Verify RuntimeError when forces contain NaN."""
        hook = NaNDetectorHook()
        batch = _make_batch()
        dynamics = _make_dynamics()

        batch.forces[0, 0] = float("nan")

        with pytest.raises(RuntimeError, match="Non-finite values detected"):
            hook(batch, dynamics)

    def test_nan_in_energies_raises(self) -> None:
        """Verify RuntimeError when energies contain NaN."""
        hook = NaNDetectorHook()
        batch = _make_batch()
        dynamics = _make_dynamics()

        batch.energies[0, 0] = float("nan")

        with pytest.raises(RuntimeError, match="Non-finite values detected"):
            hook(batch, dynamics)

    def test_inf_in_forces_raises(self) -> None:
        """Verify RuntimeError when forces contain Inf."""
        hook = NaNDetectorHook()
        batch = _make_batch()
        dynamics = _make_dynamics()

        batch.forces[1, 2] = float("inf")

        with pytest.raises(RuntimeError, match="Non-finite values detected"):
            hook(batch, dynamics)

    def test_negative_inf_raises(self) -> None:
        """Verify RuntimeError when forces contain -Inf."""
        hook = NaNDetectorHook()
        batch = _make_batch()
        dynamics = _make_dynamics()

        batch.forces[0, 0] = float("-inf")

        with pytest.raises(RuntimeError, match="Non-finite values detected"):
            hook(batch, dynamics)

    def test_nan_in_both_reports_both(self) -> None:
        """Verify diagnostic lists both fields when both have NaN."""
        hook = NaNDetectorHook()
        batch = _make_batch()
        dynamics = _make_dynamics()

        batch.forces[0, 0] = float("nan")
        batch.energies[0, 0] = float("nan")

        with pytest.raises(RuntimeError, match="forces.*energies|energies.*forces"):
            hook(batch, dynamics)

    def test_extra_keys_checked(self) -> None:
        """Verify extra_keys tensors are checked."""
        batch = _make_batch()
        dynamics = _make_dynamics()

        # Add a stresses tensor with NaN
        batch.__dict__["stresses"] = torch.randn(batch.num_graphs, 3, 3)
        batch.stresses[0, 0, 0] = float("nan")

        hook = NaNDetectorHook(extra_keys=["stresses"])

        with pytest.raises(RuntimeError, match="stresses"):
            hook(batch, dynamics)

    def test_missing_extra_key_skipped(self) -> None:
        """Verify missing extra_keys are silently skipped."""
        hook = NaNDetectorHook(extra_keys=["nonexistent_field"])
        batch = _make_batch()
        dynamics = _make_dynamics()
        # Should not raise
        hook(batch, dynamics)

    def test_error_message_contains_step_count(self) -> None:
        """Verify the error message includes the step count."""
        hook = NaNDetectorHook()
        batch = _make_batch()
        dynamics = _make_dynamics()
        dynamics.step_count = 42

        batch.forces[0, 0] = float("nan")

        with pytest.raises(RuntimeError, match="step 42"):
            hook(batch, dynamics)

    def test_error_message_contains_graph_indices(self) -> None:
        """Verify the error message includes affected graph indices."""
        hook = NaNDetectorHook()
        batch = _make_batch(n_graphs=3, atoms_per_graph=2)
        dynamics = _make_dynamics()

        # Inject NaN into graph 1 (atoms 2-3)
        batch.forces[2, 0] = float("nan")

        with pytest.raises(RuntimeError, match=r"\[1\]"):
            hook(batch, dynamics)

    def test_multi_graph_nan_reports_correct_graphs(self) -> None:
        """Verify correct graph indices with NaN in multiple graphs."""
        hook = NaNDetectorHook()
        batch = _make_batch(n_graphs=4, atoms_per_graph=2)
        dynamics = _make_dynamics()

        # NaN in graph 0 (atom 0) and graph 3 (atom 7)
        batch.forces[0, 0] = float("nan")
        batch.forces[7, 1] = float("nan")

        with pytest.raises(RuntimeError) as exc_info:
            hook(batch, dynamics)

        msg = str(exc_info.value)
        assert "0" in msg
        assert "3" in msg

    def test_stage_is_after_compute(self) -> None:
        """Verify the hook fires at AFTER_COMPUTE."""
        hook = NaNDetectorHook()
        assert hook.stage == HookStageEnum.AFTER_COMPUTE

    def test_default_frequency_is_one(self) -> None:
        """Verify default frequency is 1."""
        hook = NaNDetectorHook()
        assert hook.frequency == 1

    def test_custom_frequency(self) -> None:
        """Verify custom frequency is stored."""
        hook = NaNDetectorHook(frequency=10)
        assert hook.frequency == 10

    def test_none_forces_skipped(self) -> None:
        """Verify hook works when forces are None."""
        hook = NaNDetectorHook()
        batch = _make_batch()
        dynamics = _make_dynamics()

        # Remove forces
        del batch.__dict__["forces"]
        # NaN in energies should still be caught
        batch.energies[0, 0] = float("nan")

        with pytest.raises(RuntimeError, match="energies"):
            hook(batch, dynamics)

    def test_none_forces_and_energies_is_noop(self) -> None:
        """Verify hook is noop when both forces and energies are None."""
        hook = NaNDetectorHook()
        batch = _make_batch()
        dynamics = _make_dynamics()

        del batch.__dict__["forces"]
        del batch.__dict__["energies"]

        # Should not raise
        hook(batch, dynamics)


# ===========================================================================
# MaxForceClampHook
# ===========================================================================


class TestMaxForceClampHook:
    """Test suite for :class:`MaxForceClampHook`."""

    def test_all_below_threshold_is_noop(self) -> None:
        """Verify forces below threshold are not modified."""
        hook = MaxForceClampHook(max_force=100.0)
        batch = _make_batch()
        dynamics = _make_dynamics()

        # Set small forces
        batch.__dict__["forces"] = torch.ones(batch.num_nodes, 3) * 0.1
        forces_before = batch.forces.clone()

        hook(batch, dynamics)

        assert torch.allclose(batch.forces, forces_before)

    def test_clamps_above_threshold(self) -> None:
        """Verify forces above threshold are clamped to max_force."""
        max_force = 5.0
        hook = MaxForceClampHook(max_force=max_force)
        batch = _make_batch(n_graphs=1, atoms_per_graph=3)
        dynamics = _make_dynamics()

        # Set one large force
        batch.__dict__["forces"] = torch.tensor(
            [
                [30.0, 40.0, 0.0],  # norm 50.0 -> should be clamped
                [1.0, 0.0, 0.0],  # norm 1.0 -> OK
                [0.0, 0.0, 0.0],  # norm 0.0 -> OK
            ]
        )

        hook(batch, dynamics)

        # Atom 0: clamped, norm should be max_force
        norm_0 = torch.linalg.vector_norm(batch.forces[0])
        assert torch.isclose(norm_0, torch.tensor(max_force), atol=1e-5)

        # Atom 1: unchanged
        assert torch.isclose(batch.forces[1, 0], torch.tensor(1.0))

        # Atom 2: unchanged (zero)
        assert torch.allclose(batch.forces[2], torch.zeros(3))

    def test_direction_preserved(self) -> None:
        """Verify clamping preserves force direction."""
        max_force = 1.0
        hook = MaxForceClampHook(max_force=max_force)
        batch = _make_batch(n_graphs=1, atoms_per_graph=1)
        dynamics = _make_dynamics()

        original_force = torch.tensor([[3.0, 4.0, 0.0]])  # norm 5.0
        batch.__dict__["forces"] = original_force.clone()

        hook(batch, dynamics)

        # Direction should be preserved: (3/5, 4/5, 0)
        expected_direction = original_force / torch.linalg.vector_norm(
            original_force, dim=-1, keepdim=True
        )
        actual_direction = batch.forces / torch.linalg.vector_norm(
            batch.forces, dim=-1, keepdim=True
        )
        assert torch.allclose(actual_direction, expected_direction, atol=1e-5)

    def test_zero_force_untouched(self) -> None:
        """Verify atoms with zero force are not modified."""
        hook = MaxForceClampHook(max_force=1.0)
        batch = _make_batch(n_graphs=1, atoms_per_graph=2)
        dynamics = _make_dynamics()

        batch.__dict__["forces"] = torch.tensor(
            [
                [10.0, 0.0, 0.0],  # above threshold
                [0.0, 0.0, 0.0],  # zero force
            ]
        )

        hook(batch, dynamics)

        # Zero force atom should still be zero
        assert torch.allclose(batch.forces[1], torch.zeros(3))

    def test_inplace_mutation(self) -> None:
        """Verify forces are modified in-place (same tensor object)."""
        hook = MaxForceClampHook(max_force=1.0)
        batch = _make_batch(n_graphs=1, atoms_per_graph=1)
        dynamics = _make_dynamics()

        batch.__dict__["forces"] = torch.tensor([[10.0, 0.0, 0.0]])
        forces_ref = batch.forces  # same object

        hook(batch, dynamics)

        # Should be the same tensor object (in-place mul_)
        assert forces_ref is batch.forces

    def test_stage_is_after_compute(self) -> None:
        """Verify the hook fires at AFTER_COMPUTE."""
        hook = MaxForceClampHook(max_force=1.0)
        assert hook.stage == HookStageEnum.AFTER_COMPUTE

    def test_default_frequency_is_one(self) -> None:
        """Verify default frequency is 1."""
        hook = MaxForceClampHook(max_force=1.0)
        assert hook.frequency == 1

    def test_custom_frequency(self) -> None:
        """Verify custom frequency is stored."""
        hook = MaxForceClampHook(max_force=1.0, frequency=5)
        assert hook.frequency == 5

    def test_at_threshold_not_clamped(self) -> None:
        """Verify force exactly at threshold is NOT clamped."""
        max_force = 5.0
        hook = MaxForceClampHook(max_force=max_force)
        batch = _make_batch(n_graphs=1, atoms_per_graph=1)
        dynamics = _make_dynamics()

        # Exactly at threshold
        batch.__dict__["forces"] = torch.tensor([[3.0, 4.0, 0.0]])  # norm 5.0
        forces_before = batch.forces.clone()

        hook(batch, dynamics)

        assert torch.allclose(batch.forces, forces_before)

    def test_multiple_atoms_clamped(self) -> None:
        """Verify multiple atoms are all clamped correctly."""
        max_force = 1.0
        hook = MaxForceClampHook(max_force=max_force)
        batch = _make_batch(n_graphs=1, atoms_per_graph=4)
        dynamics = _make_dynamics()

        batch.__dict__["forces"] = torch.tensor(
            [
                [10.0, 0.0, 0.0],  # norm 10 -> clamp
                [0.0, 20.0, 0.0],  # norm 20 -> clamp
                [0.0, 0.0, 0.5],  # norm 0.5 -> OK
                [3.0, 4.0, 0.0],  # norm 5 -> clamp
            ]
        )

        hook(batch, dynamics)

        # Check clamped atoms
        for i in [0, 1, 3]:
            norm = torch.linalg.vector_norm(batch.forces[i])
            assert torch.isclose(norm, torch.tensor(max_force), atol=1e-5), (
                f"Atom {i} norm {norm:.4f} != {max_force}"
            )

        # Unclamped atom
        assert torch.isclose(batch.forces[2, 2], torch.tensor(0.5))


class TestHookProtocolCompliance:
    """Verify both safety hooks satisfy the Hook protocol."""

    def test_nan_detector_is_hook(self) -> None:
        """Verify NaNDetectorHook has frequency, stage, and __call__."""
        hook = NaNDetectorHook()
        assert isinstance(hook, Hook)

    def test_max_force_clamp_is_hook(self) -> None:
        """Verify MaxForceClampHook has frequency, stage, and __call__."""
        hook = MaxForceClampHook(max_force=1.0)
        assert isinstance(hook, Hook)


# ===========================================================================
# torch.compile smoke tests
# ===========================================================================


class TestSafetyHooksCompile:
    """Verify safety hooks work under torch.compile.

    Uses the ``device`` fixture from ``test/conftest.py`` to run on both
    CPU (inductor backend) and CUDA (cudagraphs backend) when available.
    """

    @staticmethod
    def _compile_kwargs(device: str, *, fullgraph: bool = True) -> dict:
        kw: dict = {"fullgraph": fullgraph}
        if device == "cuda":
            kw["backend"] = "cudagraphs"
        return kw

    def test_max_force_clamp_compiles_fullgraph_noop(self, device: str) -> None:
        """MaxForceClampHook compiles with fullgraph when nothing to clamp."""
        hook = MaxForceClampHook(max_force=100.0)
        batch = _make_batch(n_graphs=1, atoms_per_graph=3)
        dynamics = _make_dynamics()
        batch.__dict__["forces"] = torch.ones(3, 3, device=device) * 0.1
        forces_before = batch.forces.clone()

        compiled_hook = torch.compile(hook, **self._compile_kwargs(device))
        compiled_hook(batch, dynamics)

        assert torch.allclose(batch.forces, forces_before)

    # def test_max_force_clamp_compiles_fullgraph_active(self, device: str) -> None:
    #     """MaxForceClampHook compiles with fullgraph when clamping occurs."""
    #     max_force = 5.0
    #     hook = MaxForceClampHook(max_force=max_force)
    #     batch = _make_batch(n_graphs=1, atoms_per_graph=3)
    #     dynamics = _make_dynamics()
    #     batch.__dict__["forces"] = torch.tensor(
    #         [[30.0, 40.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    #         device=device,
    #     )

    #     compiled_hook = torch.compile(hook, **self._compile_kwargs(device))
    #     compiled_hook(batch, dynamics)

    #     norm_0 = torch.linalg.vector_norm(batch.forces[0])
    #     assert torch.isclose(norm_0, torch.tensor(max_force, device=device), atol=1e-5)
    #     assert torch.isclose(batch.forces[1, 0], torch.tensor(1.0, device=device))
    #     assert torch.allclose(batch.forces[2], torch.zeros(3, device=device))

    def test_nan_detector_compiles_no_nan(self, device: str) -> None:
        """NaNDetectorHook compiles (with graph breaks) on clean data."""
        hook = NaNDetectorHook()
        batch = _make_batch()
        dynamics = _make_dynamics()

        compiled_hook = torch.compile(
            hook, **self._compile_kwargs(device, fullgraph=False)
        )
        compiled_hook(batch, dynamics)

    def test_nan_detector_compiled_still_raises(self, device: str) -> None:
        """NaNDetectorHook still raises under torch.compile when NaN present."""
        hook = NaNDetectorHook()
        batch = _make_batch()
        dynamics = _make_dynamics()
        batch.forces[0, 0] = float("nan")

        compiled_hook = torch.compile(
            hook, **self._compile_kwargs(device, fullgraph=False)
        )
        with pytest.raises(RuntimeError, match="Non-finite values detected"):
            compiled_hook(batch, dynamics)
