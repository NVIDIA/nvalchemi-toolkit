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
"""Unit tests for ``nvalchemi.hooks.periodic`` — Tier 1 periodic hook.

Covers :class:`WrapPeriodicHook`.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVE
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.hooks import Hook, WrapPeriodicHook
from nvalchemi.models.demo import DemoModel, DemoModelWrapper
from test.dynamics.conftest import make_dynamics_context

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_periodic_batch(
    n_graphs: int = 1,
    atoms_per_graph: int = 3,
    cell_size: float = 10.0,
    pbc: tuple[bool, bool, bool] = (True, True, True),
    device: str = "cpu",
) -> Batch:
    """Create a batch with periodic boundary conditions."""
    data_list = []
    for _ in range(n_graphs):
        data = AtomicData(
            atomic_numbers=torch.tensor([6] * atoms_per_graph, dtype=torch.long),
            positions=torch.randn(atoms_per_graph, 3) * cell_size,
            cell=torch.eye(3).unsqueeze(0) * cell_size,
            pbc=torch.tensor([pbc]),
        )
        data_list.append(data)
    batch = Batch.from_data_list(data_list).to(device)
    total_atoms = n_graphs * atoms_per_graph
    batch["forces"] = torch.zeros(total_atoms, 3, device=device)
    batch["energy"] = torch.zeros(n_graphs, 1, device=device)
    # Normalize cell/pbc to (B, 3, 3) and (B, 3) via storage API
    batch["cell"] = (
        torch.eye(3, device=device).unsqueeze(0).expand(n_graphs, -1, -1) * cell_size
    )
    batch["pbc"] = torch.tensor([pbc], device=device).expand(n_graphs, -1)
    return batch


def _make_dynamics() -> BaseDynamics:
    return BaseDynamics(DemoModelWrapper(DemoModel()))


_make_ctx = make_dynamics_context


# ===========================================================================
# WrapPeriodicHook
# ===========================================================================


class TestWrapPeriodicHook:
    """Test suite for :class:`WrapPeriodicHook`."""

    def test_orthorhombic_wrapping(self, device: str) -> None:
        """Verify positions outside [0, L) are wrapped back."""
        batch = _make_periodic_batch(cell_size=10.0, device=device)
        dynamics = _make_dynamics()

        batch["positions"] = torch.tensor(
            [
                [12.0, -3.0, 25.0],
                [5.0, 5.0, 5.0],
                [-1.0, 10.0, 0.0],
            ],
            device=device,
        )

        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_POST_UPDATE)

        assert (batch.positions >= 0.0).all()
        assert (batch.positions < 10.0 + 1e-6).all()

        expected = torch.tensor(
            [[2.0, 7.0, 5.0], [5.0, 5.0, 5.0], [9.0, 0.0, 0.0]],
            device=device,
        )
        assert torch.allclose(batch.positions, expected, atol=1e-5)

    def test_wraps_only_active_graphs(self, device: str) -> None:
        """Leave positions in inactive substages unwrapped."""
        batch = _make_periodic_batch(
            n_graphs=2, atoms_per_graph=2, cell_size=10.0, device=device
        )
        batch["positions"] = torch.tensor(
            [[12.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [15.0, 0.0, 0.0], [-4.0, 0.0, 0.0]],
            device=device,
        )
        ctx = _make_ctx(batch, _make_dynamics())
        ctx.active_graph_mask = torch.tensor([True, False], device=device)

        WrapPeriodicHook()(ctx, DynamicsStage.AFTER_POST_UPDATE)

        assert torch.allclose(
            batch.positions[:2, 0], torch.tensor([2.0, 9.0], device=device)
        )
        assert torch.allclose(
            batch.positions[2:, 0], torch.tensor([15.0, -4.0], device=device)
        )

    def test_already_wrapped_is_idempotent(self, device: str) -> None:
        """Verify wrapping positions already inside the cell is a no-op."""
        batch = _make_periodic_batch(cell_size=10.0, device=device)
        dynamics = _make_dynamics()

        batch["positions"] = torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [5.0, 5.0, 5.0],
                [9.0, 0.5, 7.0],
            ],
            device=device,
        )
        positions_before = batch.positions.clone()

        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_POST_UPDATE)

        assert torch.allclose(batch.positions, positions_before, atol=1e-5)

    def test_partial_pbc_slab(self, device: str) -> None:
        """Verify only periodic dimensions are wrapped (slab: TT F)."""
        batch = _make_periodic_batch(
            cell_size=10.0, pbc=(True, True, False), device=device
        )
        dynamics = _make_dynamics()

        batch["positions"] = torch.tensor(
            [
                [12.0, -3.0, 25.0],
                [5.0, 5.0, -5.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
        )

        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_POST_UPDATE)

        expected = torch.tensor(
            [[2.0, 7.0, 25.0], [5.0, 5.0, -5.0], [0.0, 0.0, 0.0]],
            device=device,
        )
        assert torch.allclose(batch.positions, expected, atol=1e-5)

    def test_non_periodic_is_noop(self, device: str) -> None:
        """Verify no wrapping when PBC is all False."""
        batch = _make_periodic_batch(
            cell_size=10.0, pbc=(False, False, False), device=device
        )
        dynamics = _make_dynamics()

        batch["positions"] = torch.tensor(
            [
                [12.0, -3.0, 25.0],
                [5.0, 5.0, 5.0],
                [-1.0, 100.0, 0.0],
            ],
            device=device,
        )
        positions_before = batch.positions.clone()

        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_POST_UPDATE)

        assert torch.allclose(batch.positions, positions_before)

    @pytest.mark.parametrize(
        "pbc",
        [
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ],
    )
    def test_mixed_pbc_axes(self, device: str, pbc: tuple[bool, bool, bool]) -> None:
        """Wrap periodic axes only; non-periodic axes keep their exact values."""
        batch = _make_periodic_batch(
            atoms_per_graph=2, cell_size=10.0, pbc=pbc, device=device
        )
        # Every coordinate lies outside [0, 10).
        batch["positions"] = torch.tensor(
            [[12.0, -3.0, 25.0], [-1.0, 14.0, -7.0]], device=device
        )
        before = batch.positions.clone()

        WrapPeriodicHook()(
            _make_ctx(batch, _make_dynamics()), DynamicsStage.AFTER_POST_UPDATE
        )

        periodic = torch.tensor(pbc, device=device)
        wrapped = torch.tensor([[2.0, 7.0, 5.0], [9.0, 4.0, 3.0]], device=device)
        assert torch.allclose(
            batch.positions[:, periodic], wrapped[:, periodic], atol=1e-5
        )
        assert torch.equal(batch.positions[:, ~periodic], before[:, ~periodic])

    def test_per_graph_pbc_rows(self, device: str) -> None:
        """Each graph wraps by its own pbc row, with no leak across graphs."""
        batch = _make_periodic_batch(
            n_graphs=3, atoms_per_graph=1, cell_size=10.0, device=device
        )
        batch["pbc"] = torch.tensor(
            [[True, True, True], [True, True, False], [False, False, False]],
            device=device,
        )
        batch["positions"] = torch.tensor([[12.0, -3.0, 25.0]] * 3, device=device)

        WrapPeriodicHook()(
            _make_ctx(batch, _make_dynamics()), DynamicsStage.AFTER_POST_UPDATE
        )

        expected = torch.tensor(
            [[2.0, 7.0, 5.0], [2.0, 7.0, 25.0], [12.0, -3.0, 25.0]], device=device
        )
        assert torch.allclose(batch.positions, expected, atol=1e-5)

    @pytest.mark.parametrize(
        "pbc",
        [
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, False, False),
            (False, True, False),
            (False, False, True),
        ],
    )
    def test_mixed_pbc_triclinic(
        self, device: str, pbc: tuple[bool, bool, bool]
    ) -> None:
        """pbc applies per lattice vector, so only periodic fractions wrap."""
        batch = _make_periodic_batch(
            atoms_per_graph=2, cell_size=10.0, pbc=pbc, device=device
        )
        # Every lattice vector has components along more than one axis.
        cell = torch.tensor(
            [[10.0, 0.0, 0.0], [3.0, 10.0, 0.0], [2.0, 1.0, 10.0]], device=device
        )
        batch["cell"] = cell.unsqueeze(0)
        frac = torch.tensor([[2.3, -0.6, 1.25], [-1.75, 1.5, -0.2]], device=device)
        batch["positions"] = frac @ cell

        WrapPeriodicHook()(
            _make_ctx(batch, _make_dynamics()), DynamicsStage.AFTER_POST_UPDATE
        )

        # Hand-computed: periodic fractions wrap into [0, 1), others stay.
        wrapped = torch.tensor([[0.3, 0.4, 0.25], [0.25, 0.5, 0.8]], device=device)
        periodic = torch.tensor(pbc, device=device)
        expected = torch.where(periodic, wrapped, frac) @ cell
        assert torch.allclose(batch.positions, expected, atol=1e-4)

    def test_mixed_pbc_triclinic_non_periodic_tilt(self, device: str) -> None:
        """A tilted non-periodic lattice vector does not move the atom.

        The atom sits at fractions (0.05, 1.5, 0.2). Only ``b`` is outside
        [0, 1), and ``b`` is not periodic, so nothing moves.  A per-axis
        Cartesian mask moved the atom to fractions (-0.25, 1.5, 0.2).
        """
        batch = _make_periodic_batch(
            atoms_per_graph=1, cell_size=10.0, pbc=(True, False, True), device=device
        )
        batch["cell"] = torch.tensor(
            [[[10.0, 0.0, 0.0], [3.0, 10.0, 0.0], [0.0, 0.0, 10.0]]], device=device
        )
        batch["positions"] = torch.tensor([[5.0, 15.0, 2.0]], device=device)

        WrapPeriodicHook()(
            _make_ctx(batch, _make_dynamics()), DynamicsStage.AFTER_POST_UPDATE
        )

        expected = torch.tensor([[5.0, 15.0, 2.0]], device=device)
        assert torch.allclose(batch.positions, expected, atol=1e-5)

    def test_triclinic_cell(self, device: str) -> None:
        """Verify wrapping with a non-orthorhombic (triclinic) cell."""
        batch = _make_periodic_batch(cell_size=10.0, device=device)
        dynamics = _make_dynamics()

        batch["cell"] = torch.tensor(
            [[[10.0, 0.0, 0.0], [3.0, 10.0, 0.0], [0.0, 0.0, 10.0]]],
            device=device,
        )
        batch["positions"] = torch.tensor(
            [
                [16.5, 5.0, 5.0],
                [5.0, 5.0, 5.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
        )

        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_POST_UPDATE)

        expected = torch.tensor(
            [[6.5, 5.0, 5.0], [5.0, 5.0, 5.0], [0.0, 0.0, 0.0]],
            device=device,
        )
        assert torch.allclose(batch.positions, expected, atol=1e-4)

    def test_inplace_mutation(self, device: str) -> None:
        """Verify positions tensor identity is preserved (copy_)."""
        batch = _make_periodic_batch(cell_size=10.0, device=device)
        dynamics = _make_dynamics()

        batch["positions"] = torch.tensor(
            [
                [12.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            device=device,
        )
        positions_ref = batch.positions

        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_POST_UPDATE)

        assert positions_ref is batch.positions

    def test_multi_graph_batch(self, device: str) -> None:
        """Verify wrapping works with multiple graphs with different cells."""
        batch = _make_periodic_batch(
            n_graphs=2, atoms_per_graph=2, cell_size=10.0, device=device
        )
        dynamics = _make_dynamics()

        batch["cell"] = torch.tensor(
            [
                [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
                [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
            ],
            device=device,
        )

        batch["positions"] = torch.tensor(
            [
                [12.0, 0.0, 0.0],
                [5.0, 5.0, 5.0],
                [7.0, 0.0, 0.0],
                [3.0, 3.0, 3.0],
            ],
            device=device,
        )

        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_POST_UPDATE)

        expected = torch.tensor(
            [[2.0, 0.0, 0.0], [5.0, 5.0, 5.0], [2.0, 0.0, 0.0], [3.0, 3.0, 3.0]],
            device=device,
        )
        assert torch.allclose(batch.positions, expected, atol=1e-5)

    def test_stage_is_after_post_update(self) -> None:
        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        assert hook.stage == DynamicsStage.AFTER_POST_UPDATE

    def test_default_frequency_is_one(self) -> None:
        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        assert hook.frequency == 1

    def test_custom_frequency(self) -> None:
        hook = WrapPeriodicHook(frequency=10, stage=DynamicsStage.AFTER_POST_UPDATE)
        assert hook.frequency == 10

    def test_is_hook(self) -> None:
        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        assert isinstance(hook, Hook)


class TestWrapPeriodicHookDimensionSqueeze:
    """Cover the squeeze branches in WrapPeriodicHook.__call__ (lines 131-134).

    The hook defensively handles system-level tensors that arrive with an
    extra singleton dimension — e.g. ``cell`` of shape ``(B, 1, 3, 3)``
    instead of the expected ``(B, 3, 3)``, and ``pbc`` of shape ``(B, 1, 3)``
    instead of ``(B, 3)``.  These shapes can occur when batching code adds a
    leading dimension for broadcasting.
    """

    def test_4d_cell_is_squeezed_and_wraps_correctly(self, device: str) -> None:
        """cell with shape (B, 1, 3, 3) is squeezed to (B, 3, 3) before wrapping."""
        batch = _make_periodic_batch(cell_size=10.0, device=device)
        dynamics = _make_dynamics()

        # Promote cell to (B, 1, 3, 3)
        batch["cell"] = batch.cell.unsqueeze(1)  # (1, 1, 3, 3)
        batch["positions"] = torch.tensor(
            [[12.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            device=device,
        )

        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_POST_UPDATE)  # must not raise

        # Atom 0 at 12.0 in a 10.0-cell wraps to 2.0
        assert torch.allclose(
            batch.positions[0, 0], torch.tensor(2.0, device=device), atol=1e-5
        )

    def test_3d_pbc_is_squeezed_and_wraps_correctly(self, device: str) -> None:
        """pbc with shape (B, 1, 3) is squeezed to (B, 3) before wrapping."""
        batch = _make_periodic_batch(cell_size=10.0, device=device)
        dynamics = _make_dynamics()

        # Promote pbc to (B, 1, 3)
        batch["pbc"] = batch.pbc.unsqueeze(1)  # (1, 1, 3)
        batch["positions"] = torch.tensor(
            [[12.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            device=device,
        )

        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_POST_UPDATE)

        assert torch.allclose(
            batch.positions[0, 0], torch.tensor(2.0, device=device), atol=1e-5
        )

    def test_both_4d_cell_and_3d_pbc_together(self, device: str) -> None:
        """Both cell (B,1,3,3) and pbc (B,1,3) squeezed in a single call."""
        batch = _make_periodic_batch(cell_size=10.0, device=device)
        dynamics = _make_dynamics()

        batch["cell"] = batch.cell.unsqueeze(1)
        batch["pbc"] = batch.pbc.unsqueeze(1)
        batch["positions"] = torch.tensor(
            [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            device=device,
        )

        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_POST_UPDATE)

        # -1.0 wraps to 9.0 in a 10.0-cell
        assert torch.allclose(
            batch.positions[0, 0], torch.tensor(9.0, device=device), atol=1e-5
        )


class TestWrapPeriodicHookCompile:
    """Verify WrapPeriodicHook works under torch.compile."""

    @staticmethod
    def _compile_kwargs(device: str) -> dict:
        kw: dict = {"fullgraph": True}
        if device == "cuda":
            kw["backend"] = "cudagraphs"
        return kw

    def test_compiles_fullgraph(self, device: str) -> None:
        """WrapPeriodicHook._wrap_positions compiles with fullgraph=True."""
        batch = _make_periodic_batch(cell_size=10.0, device=device)
        batch["positions"] = torch.tensor(
            [[12.0, 0.0, 0.0], [5.0, 5.0, 5.0], [0.0, 0.0, 0.0]],
            device=device,
        )

        hook = WrapPeriodicHook(stage=DynamicsStage.AFTER_POST_UPDATE)
        compiled = torch.compile(hook._wrap_positions, **self._compile_kwargs(device))
        compiled(batch)

        expected = torch.tensor(
            [[2.0, 0.0, 0.0], [5.0, 5.0, 5.0], [0.0, 0.0, 0.0]],
            device=device,
        )
        assert torch.allclose(batch.positions, expected, atol=1e-5)


class TestWrapPeriodicHookInFusedStage:
    """Regression tests for issue #226.

    ``FusedStage.step()`` keeps ``requires_grad`` on ``batch.positions``
    for the whole step, while single-stage engines drop it before hooks
    run.  ``WrapPeriodicHook`` mutated positions in place without
    ``torch.no_grad()`` and raised "a leaf Variable that requires grad is
    being used in an in-place operation" on the first fused step.
    """

    @staticmethod
    def _make_autograd_data(
        device: str, pbc: tuple[bool, bool, bool] = (True, True, True)
    ) -> list[AtomicData]:
        """Two argon graphs whose positions start outside the 5 A cell.

        Non-periodic axes start near -18 A, far outside the cell, so a wrap
        along one of them is visible after a short run.
        """
        offset = torch.tensor([0.0 if p else -20.0 for p in pbc])
        return [
            AtomicData(
                positions=torch.rand(8, 3) * 6 - 1 + offset,
                atomic_numbers=torch.full((8,), 18),
                atomic_masses=torch.full((8,), 39.948),
                cell=(5 * torch.eye(3)).unsqueeze(0),
                pbc=torch.tensor([pbc]),
                forces=torch.zeros(8, 3),
                energy=torch.zeros(1, 1),
            )
            for _ in range(2)
        ]

    def test_fused_stage_wraps_positions(self, device: str) -> None:
        """WrapPeriodicHook runs inside a two-stage FusedStage and wraps.

        The hook lives on the second sub-stage, as in the issue's
        reproducer.  The engine must complete the run without raising,
        and the final positions must lie inside the unit cell.
        """
        model = DemoModelWrapper(DemoModel()).to(device)
        hook = WrapPeriodicHook(frequency=1, stage=DynamicsStage.AFTER_POST_UPDATE)
        fused = NVE(model=model, dt=1.0, n_steps=3) + NVE(
            model=model, dt=1.0, n_steps=5, hooks=[hook]
        )
        batch = Batch.from_data_list(self._make_autograd_data(device)).to(device)

        fused.run(batch, n_steps=10)

        # After the run every graph has graduated (status == exit_status)
        # and the last position-mutating operation was the hook, so the
        # fractional coordinates must lie in [0, 1) on periodic axes.
        cell = torch.linalg.inv(batch.cell)
        frac = torch.einsum("bij,bj->bi", cell[batch.batch_idx], batch.positions)
        pbc_atoms = batch.pbc[batch.batch_idx]
        in_cell = ((frac >= -1e-5) & (frac < 1 + 1e-5)) | ~pbc_atoms
        assert in_cell.all()

    @pytest.mark.parametrize(
        "pbc", [(True, True, False), (False, True, True), (True, False, False)]
    )
    def test_fused_stage_mixed_pbc(
        self, device: str, pbc: tuple[bool, bool, bool]
    ) -> None:
        """Inside a FusedStage, positions require grad and pbc is mixed.

        Periodic axes end inside the cell. Non-periodic axes are not
        wrapped, so they stay far below zero, where they started.
        """
        model = DemoModelWrapper(DemoModel()).to(device)
        hook = WrapPeriodicHook(frequency=1, stage=DynamicsStage.AFTER_POST_UPDATE)
        fused = NVE(model=model, dt=1.0, n_steps=3) + NVE(
            model=model, dt=1.0, n_steps=5, hooks=[hook]
        )
        batch = Batch.from_data_list(self._make_autograd_data(device, pbc)).to(device)

        fused.run(batch, n_steps=10)

        periodic = torch.tensor(pbc, device=device)
        frac = batch.positions / 5.0
        assert ((frac[:, periodic] >= -1e-5) & (frac[:, periodic] < 1 + 1e-5)).all()
        assert (batch.positions[:, ~periodic] < -10.0).all()

    def test_fused_stage_first_step_no_error(self, device: str) -> None:
        """The first fused step must not raise the leaf-Variable error.

        This isolates the failure mode of the issue: a single step with
        the hook on a sub-stage used to raise before any wrapping
        happened.
        """
        model = DemoModelWrapper(DemoModel()).to(device)
        hook = WrapPeriodicHook(frequency=1, stage=DynamicsStage.AFTER_POST_UPDATE)
        fused = NVE(model=model, dt=1.0, n_steps=3) + NVE(
            model=model, dt=1.0, n_steps=5, hooks=[hook]
        )
        batch = Batch.from_data_list(self._make_autograd_data(device)).to(device)

        fused.run(batch, n_steps=1)

        assert batch.positions.isfinite().all()
