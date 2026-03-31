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
"""Unit tests for ``nvalchemi.dynamics.hooks.freeze`` — FreezeAtomsHook.

Covers :class:`FreezeAtomsHook`.
"""

from __future__ import annotations

import torch

from nvalchemi._typing import AtomCategory
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import BaseDynamics, Hook, HookStageEnum
from nvalchemi.dynamics.hooks import FreezeAtomsHook
from nvalchemi.models.demo import DemoPotential

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(
    n_atoms: int = 6,
    n_frozen: int = 2,
    n_graphs: int = 1,
    device: str = "cpu",
) -> Batch:
    """Create a batch where the last n_frozen atoms per graph are frozen.

    Frozen atoms have atom_categories = AtomCategory.SPECIAL.value (-1),
    others have 0.
    """
    data_list = []
    for _ in range(n_graphs):
        # Last n_frozen atoms are SPECIAL (frozen), rest are GAS (0)
        categories = torch.zeros(n_atoms, dtype=torch.long)
        categories[-n_frozen:] = AtomCategory.SPECIAL.value
        data = AtomicData(
            atomic_numbers=torch.tensor([6] * n_atoms, dtype=torch.long),
            positions=torch.randn(n_atoms, 3),
            cell=torch.eye(3).unsqueeze(0) * 10.0,
            pbc=torch.tensor([[True, True, True]]),
            atom_categories=categories,
        )
        data_list.append(data)

    batch = Batch.from_data_list(data_list).to(device)
    total_atoms = n_graphs * n_atoms

    # Set velocities and forces via direct dict assignment
    batch.__dict__["velocities"] = torch.randn(total_atoms, 3, device=device)
    batch.__dict__["forces"] = torch.randn(total_atoms, 3, device=device)
    batch.__dict__["energies"] = torch.zeros(n_graphs, 1, device=device)

    return batch


def _make_dynamics() -> BaseDynamics:
    """Create a minimal BaseDynamics instance for testing."""
    return BaseDynamics(DemoPotential())


def _call_hook_two_stage(
    hook: FreezeAtomsHook,
    batch: Batch,
    dynamics: BaseDynamics,
) -> None:
    """Simulate the two-stage hook execution (snapshot then restore)."""
    dynamics.current_hook_stage = HookStageEnum.BEFORE_PRE_UPDATE
    hook(batch, dynamics)
    dynamics.current_hook_stage = HookStageEnum.AFTER_POST_UPDATE
    hook(batch, dynamics)


# ===========================================================================
# TestFreezeAtomsHook
# ===========================================================================


class TestFreezeAtomsHook:
    """Test suite for :class:`FreezeAtomsHook`."""

    def test_positions_restored(self, device: str) -> None:
        """Verify frozen atom positions are restored after perturbation."""
        batch = _make_batch(n_atoms=6, n_frozen=2, device=device)
        dynamics = _make_dynamics()
        hook = FreezeAtomsHook()

        # Get mask and save original frozen positions
        mask = batch.atom_categories == AtomCategory.SPECIAL.value
        original_frozen_positions = batch.positions[mask].clone()
        original_unfrozen_positions = batch.positions[~mask].clone()

        # Snapshot stage
        dynamics.current_hook_stage = HookStageEnum.BEFORE_PRE_UPDATE
        hook(batch, dynamics)

        # Perturb all positions
        batch.positions.add_(1.0)

        # Restore stage - should restore frozen positions
        dynamics.current_hook_stage = HookStageEnum.AFTER_POST_UPDATE
        hook(batch, dynamics)

        # Frozen positions should be restored to original
        assert torch.allclose(batch.positions[mask], original_frozen_positions)
        # Unfrozen positions should remain perturbed
        assert torch.allclose(batch.positions[~mask], original_unfrozen_positions + 1.0)

    def test_velocities_zeroed(self, device: str) -> None:
        """Verify frozen atom velocities are zeroed."""
        batch = _make_batch(n_atoms=6, n_frozen=2, device=device)
        dynamics = _make_dynamics()
        hook = FreezeAtomsHook()

        mask = batch.atom_categories == AtomCategory.SPECIAL.value
        original_unfrozen_velocities = batch.velocities[~mask].clone()

        _call_hook_two_stage(hook, batch, dynamics)

        # Frozen velocities should be zero
        assert torch.allclose(
            batch.velocities[mask], torch.zeros_like(batch.velocities[mask])
        )
        # Unfrozen velocities should be unchanged
        assert torch.allclose(batch.velocities[~mask], original_unfrozen_velocities)

    def test_forces_zeroed(self, device: str) -> None:
        """Verify frozen atom forces are zeroed by default."""
        batch = _make_batch(n_atoms=6, n_frozen=2, device=device)
        dynamics = _make_dynamics()
        hook = FreezeAtomsHook()

        mask = batch.atom_categories == AtomCategory.SPECIAL.value
        original_unfrozen_forces = batch.forces[~mask].clone()

        _call_hook_two_stage(hook, batch, dynamics)

        # Frozen forces should be zero
        assert torch.allclose(batch.forces[mask], torch.zeros_like(batch.forces[mask]))
        # Unfrozen forces should be unchanged
        assert torch.allclose(batch.forces[~mask], original_unfrozen_forces)

    def test_unfrozen_atoms_unchanged(self, device: str) -> None:
        """Verify hook does NOT modify positions/velocities/forces of non-frozen atoms."""
        batch = _make_batch(n_atoms=6, n_frozen=2, device=device)
        dynamics = _make_dynamics()
        hook = FreezeAtomsHook()

        mask = batch.atom_categories == AtomCategory.SPECIAL.value

        # Save original unfrozen state
        original_unfrozen_positions = batch.positions[~mask].clone()
        original_unfrozen_velocities = batch.velocities[~mask].clone()
        original_unfrozen_forces = batch.forces[~mask].clone()

        _call_hook_two_stage(hook, batch, dynamics)

        # All unfrozen properties should be unchanged
        assert torch.allclose(batch.positions[~mask], original_unfrozen_positions)
        assert torch.allclose(batch.velocities[~mask], original_unfrozen_velocities)
        assert torch.allclose(batch.forces[~mask], original_unfrozen_forces)

    def test_zero_forces_disabled(self, device: str) -> None:
        """Verify forces on frozen atoms are NOT zeroed when zero_forces=False."""
        batch = _make_batch(n_atoms=6, n_frozen=2, device=device)
        dynamics = _make_dynamics()
        hook = FreezeAtomsHook(zero_forces=False)

        mask = batch.atom_categories == AtomCategory.SPECIAL.value
        original_frozen_forces = batch.forces[mask].clone()

        _call_hook_two_stage(hook, batch, dynamics)

        # Forces should NOT be zeroed
        assert torch.allclose(batch.forces[mask], original_frozen_forces)

    def test_no_frozen_atoms_noop(self, device: str) -> None:
        """Verify hook is a no-op when no atoms are frozen."""
        batch = _make_batch(n_atoms=6, n_frozen=0, device=device)
        # Override to ensure no SPECIAL atoms
        batch.__dict__["atom_categories"] = torch.zeros(
            6, dtype=torch.long, device=device
        )

        dynamics = _make_dynamics()
        hook = FreezeAtomsHook()

        original_positions = batch.positions.clone()
        original_velocities = batch.velocities.clone()
        original_forces = batch.forces.clone()

        _call_hook_two_stage(hook, batch, dynamics)

        # Nothing should change
        assert torch.allclose(batch.positions, original_positions)
        assert torch.allclose(batch.velocities, original_velocities)
        assert torch.allclose(batch.forces, original_forces)

    def test_all_frozen_atoms(self, device: str) -> None:
        """Verify all atoms are frozen when all have SPECIAL category."""
        batch = _make_batch(n_atoms=4, n_frozen=4, device=device)
        dynamics = _make_dynamics()
        hook = FreezeAtomsHook()

        original_positions = batch.positions.clone()

        # Snapshot stage
        dynamics.current_hook_stage = HookStageEnum.BEFORE_PRE_UPDATE
        hook(batch, dynamics)

        # Perturb positions
        batch.positions.add_(2.0)

        # Restore stage
        dynamics.current_hook_stage = HookStageEnum.AFTER_POST_UPDATE
        hook(batch, dynamics)

        # All positions should be restored
        assert torch.allclose(batch.positions, original_positions)
        # All velocities should be zeroed
        assert torch.allclose(batch.velocities, torch.zeros_like(batch.velocities))
        # All forces should be zeroed
        assert torch.allclose(batch.forces, torch.zeros_like(batch.forces))

    def test_multi_graph_batch(self, device: str) -> None:
        """Verify correct per-graph behavior with multiple graphs."""
        batch = _make_batch(n_atoms=4, n_frozen=1, n_graphs=3, device=device)
        dynamics = _make_dynamics()
        hook = FreezeAtomsHook()

        mask = batch.atom_categories == AtomCategory.SPECIAL.value
        original_frozen_positions = batch.positions[mask].clone()

        # Should have 3 frozen atoms (1 per graph)
        assert mask.sum() == 3

        # Snapshot stage
        dynamics.current_hook_stage = HookStageEnum.BEFORE_PRE_UPDATE
        hook(batch, dynamics)

        # Perturb all positions
        batch.positions.add_(1.0)

        # Restore stage
        dynamics.current_hook_stage = HookStageEnum.AFTER_POST_UPDATE
        hook(batch, dynamics)

        # All frozen positions should be restored
        assert torch.allclose(batch.positions[mask], original_frozen_positions)

    def test_custom_freeze_category(self, device: str) -> None:
        """Verify freeze_category parameter works with different categories."""
        batch = _make_batch(n_atoms=6, n_frozen=0, device=device)

        # Set 2 atoms to BULK category
        categories = torch.zeros(6, dtype=torch.long, device=device)
        categories[2:4] = AtomCategory.BULK.value
        batch.__dict__["atom_categories"] = categories

        dynamics = _make_dynamics()
        hook = FreezeAtomsHook(freeze_category=AtomCategory.BULK.value)

        mask = batch.atom_categories == AtomCategory.BULK.value
        original_frozen_positions = batch.positions[mask].clone()

        # Snapshot stage
        dynamics.current_hook_stage = HookStageEnum.BEFORE_PRE_UPDATE
        hook(batch, dynamics)

        # Perturb all positions
        batch.positions.add_(1.0)

        # Restore stage
        dynamics.current_hook_stage = HookStageEnum.AFTER_POST_UPDATE
        hook(batch, dynamics)

        # BULK atoms should be restored
        assert torch.allclose(batch.positions[mask], original_frozen_positions)
        # Velocities should be zeroed for BULK atoms
        assert torch.allclose(
            batch.velocities[mask], torch.zeros_like(batch.velocities[mask])
        )

    def test_stages(self) -> None:
        """Verify hook has correct stage and stages attributes."""
        hook = FreezeAtomsHook()
        assert hook.stage == HookStageEnum.BEFORE_PRE_UPDATE
        assert hook.stages == (
            HookStageEnum.BEFORE_PRE_UPDATE,
            HookStageEnum.AFTER_POST_UPDATE,
        )

    def test_frequency_default(self) -> None:
        """Verify default frequency is 1."""
        assert FreezeAtomsHook().frequency == 1

    def test_hook_protocol_compliance(self) -> None:
        """Verify FreezeAtomsHook satisfies Hook protocol."""
        assert isinstance(FreezeAtomsHook(), Hook)


# ===========================================================================
# TestFreezeAtomsHookCompile
# ===========================================================================


class TestFreezeAtomsHookCompile:
    """Verify FreezeAtomsHook works under torch.compile.

    FreezeAtomsHook supports fullgraph=True because it uses torch.where
    for branchless GPU-vectorized operations and avoids data-dependent
    control flow.
    """

    @staticmethod
    def _compile_kwargs(device: str) -> dict:
        # fullgraph=True supported due to branchless torch.where operations
        kw: dict = {"fullgraph": True}
        if device == "cuda":
            kw["backend"] = "cudagraphs"
        return kw

    def test_compile_smoke(self, device: str) -> None:
        """FreezeAtomsHook compiles with fullgraph=True and runs correctly."""
        batch = _make_batch(n_atoms=6, n_frozen=2, device=device)
        dynamics = _make_dynamics()

        mask = batch.atom_categories == AtomCategory.SPECIAL.value
        original_frozen_positions = batch.positions[mask].clone()

        hook = FreezeAtomsHook()

        # Snapshot stage (trivial, no need to compile)
        dynamics.current_hook_stage = HookStageEnum.BEFORE_PRE_UPDATE
        hook(batch, dynamics)

        # Perturb positions
        batch.positions.add_(1.0)

        # Compile and call the restore stage
        dynamics.current_hook_stage = HookStageEnum.AFTER_POST_UPDATE
        compiled = torch.compile(hook, **self._compile_kwargs(device))
        compiled(batch, dynamics)

        # Positions should be restored
        assert torch.allclose(batch.positions[mask], original_frozen_positions)
