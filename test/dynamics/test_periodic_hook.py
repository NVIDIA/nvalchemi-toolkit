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
"""Unit tests for ``nvalchemi.dynamics.hooks.periodic`` — Tier 1 periodic hook.

Covers :class:`WrapPeriodicHook`.
"""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.dynamics.hooks.periodic import WrapPeriodicHook
from nvalchemi.hooks import Hook, HookContext
from nvalchemi.models.demo import DemoModelWrapper

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
    return BaseDynamics(DemoModelWrapper())


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

        hook = WrapPeriodicHook()
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_POST_UPDATE)

        assert (batch.positions >= 0.0).all()
        assert (batch.positions < 10.0 + 1e-6).all()

        expected = torch.tensor(
            [[2.0, 7.0, 5.0], [5.0, 5.0, 5.0], [9.0, 0.0, 0.0]],
            device=device,
        )
        assert torch.allclose(batch.positions, expected, atol=1e-5)

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

        hook = WrapPeriodicHook()
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

        hook = WrapPeriodicHook()
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

        hook = WrapPeriodicHook()
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_POST_UPDATE)

        assert torch.allclose(batch.positions, positions_before)

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

        hook = WrapPeriodicHook()
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

        hook = WrapPeriodicHook()
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

        hook = WrapPeriodicHook()
        ctx = _make_ctx(batch, dynamics)
        hook(ctx, DynamicsStage.AFTER_POST_UPDATE)

        expected = torch.tensor(
            [[2.0, 0.0, 0.0], [5.0, 5.0, 5.0], [2.0, 0.0, 0.0], [3.0, 3.0, 3.0]],
            device=device,
        )
        assert torch.allclose(batch.positions, expected, atol=1e-5)

    def test_stage_is_after_post_update(self) -> None:
        hook = WrapPeriodicHook()
        assert hook.stage == DynamicsStage.AFTER_POST_UPDATE

    def test_default_frequency_is_one(self) -> None:
        hook = WrapPeriodicHook()
        assert hook.frequency == 1

    def test_custom_frequency(self) -> None:
        hook = WrapPeriodicHook(frequency=10)
        assert hook.frequency == 10

    def test_is_hook(self) -> None:
        hook = WrapPeriodicHook()
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

        hook = WrapPeriodicHook()
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

        hook = WrapPeriodicHook()
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

        hook = WrapPeriodicHook()
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

        hook = WrapPeriodicHook()
        compiled = torch.compile(hook._wrap_positions, **self._compile_kwargs(device))
        compiled(batch)

        expected = torch.tensor(
            [[2.0, 0.0, 0.0], [5.0, 5.0, 5.0], [0.0, 0.0, 0.0]],
            device=device,
        )
        assert torch.allclose(batch.positions, expected, atol=1e-5)
