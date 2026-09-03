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

"""Tests for batched NEB dynamics hooks."""

from __future__ import annotations

import importlib
from unittest.mock import Mock

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.dynamics.paths.hooks import PathEnergyStatsHook
from nvalchemi.dynamics.paths.neb import (
    ConstantSpringConfig,
    SpringContext,
)
from nvalchemi.dynamics.paths.neb._ops.modes import (
    CLIMBING_NEB,
    ENDPOINT,
)
from nvalchemi.dynamics.paths.neb.hooks import (
    ClimbingImageSelectionHook,
    NEBForceHook,
)
from nvalchemi.hooks import DynamicsContext


def _bands(energies: list[float], groups: list[int]) -> Batch:
    images = [
        AtomicData(
            atomic_numbers=torch.tensor([1]),
            positions=torch.tensor([[float(i), 0.0, 0.0]]),
            energy=torch.tensor([[energy]]),
            forces=torch.tensor([[float(i + 1), 0.0, 0.0]]),
        )
        for i, energy in enumerate(energies)
    ]
    batch = Batch.from_data_list(images)
    batch.set_group_layout(torch.tensor(groups))
    return batch


def _force_hook(**kwargs: object) -> NEBForceHook:
    """Build an NEB force hook with its shared energy-statistics provider."""
    return NEBForceHook(energy_stats_hook=PathEnergyStatsHook(), **kwargs)


def _selection_hooks(
    selection: str,
) -> tuple[ClimbingImageSelectionHook, NEBForceHook]:
    """Build selection and force hooks sharing one statistics provider."""
    energy_stats_hook = PathEnergyStatsHook()
    return (
        ClimbingImageSelectionHook(
            energy_stats_hook=energy_stats_hook,
            selection=selection,
        ),
        NEBForceHook(energy_stats_hook=energy_stats_hook),
    )


class TestClimbingImageSelection:
    """Exercise the standalone climbing-image selection hook."""

    def _update_selection(
        self,
        hook: ClimbingImageSelectionHook,
        force_hook: NEBForceHook,
        ctx: DynamicsContext,
        *,
        admit: bool = True,
    ) -> None:
        if admit:
            hook.energy_stats_hook(ctx, DynamicsStage.ON_ADMISSION)
            force_hook(ctx, DynamicsStage.ON_ADMISSION)
            hook(ctx, DynamicsStage.ON_ADMISSION)
        hook.energy_stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

    def test_fixed_selection_is_retained(self) -> None:
        batch = _bands([0.0, 2.0, 1.0, 0.0], [0] * 4)
        hook, force_hook = _selection_hooks("fixed")
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.ones(
                batch.num_graphs, dtype=torch.bool, device=batch.device
            ),
        )
        self._update_selection(hook, force_hook, ctx)

        assert torch.where(batch.force_mode == CLIMBING_NEB)[0].tolist() == [1]

        batch.energy.copy_(torch.tensor([[0.0], [1.0], [3.0], [0.0]]))
        ctx.step_count = 1
        self._update_selection(hook, force_hook, ctx, admit=False)

        assert torch.where(batch.force_mode == CLIMBING_NEB)[0].tolist() == [1]

    def test_fixed_selection_replaces_stale_climber_on_initialization(self) -> None:
        """The first fixed selection replaces any pre-existing climber."""
        batch = _bands([0.0, 2.0, 1.0, 0.0], [0] * 4)
        hook, force_hook = _selection_hooks("fixed")
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.ones(
                batch.num_graphs, dtype=torch.bool, device=batch.device
            ),
        )

        hook.energy_stats_hook(ctx, DynamicsStage.ON_ADMISSION)
        force_hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        batch.force_mode[2] = CLIMBING_NEB
        hook.energy_stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

        assert torch.where(batch.force_mode == CLIMBING_NEB)[0].tolist() == [1]

    def test_fixed_selection_initializes_paths_as_they_become_active(self) -> None:
        """Inactive paths remain eligible for fixed selection on later entry."""
        batch = _bands(
            [0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 3.0, 0.0],
            [0] * 4 + [1] * 4,
        )
        hook, force_hook = _selection_hooks("fixed")
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.zeros(batch.num_graphs, dtype=torch.bool),
        )
        self._update_selection(hook, force_hook, ctx)

        assert not torch.any(batch.force_mode == CLIMBING_NEB)

        ctx.active_graph_mask = torch.tensor(
            [True, True, True, True, False, False, False, False]
        )
        ctx.step_count = 1
        self._update_selection(hook, force_hook, ctx, admit=False)

        assert torch.where(batch.force_mode == CLIMBING_NEB)[0].tolist() == [1]

        batch.energy.copy_(
            torch.tensor([[0.0], [1.0], [4.0], [0.0], [0.0], [5.0], [2.0], [0.0]])
        )
        ctx.active_graph_mask = torch.tensor(
            [False, False, False, False, True, True, True, True]
        )
        ctx.step_count = 2
        self._update_selection(hook, force_hook, ctx, admit=False)

        # Path 0 retains its original climber while path 1 initializes later.
        assert torch.where(batch.force_mode == CLIMBING_NEB)[0].tolist() == [1, 5]

    def test_dynamic_selection_tracks_highest_energy_images(self) -> None:
        batch = _bands(
            [0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 3.0, 0.0],
            [0] * 4 + [1] * 4,
        )
        hook, force_hook = _selection_hooks("dynamic")
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.ones(
                batch.num_graphs, dtype=torch.bool, device=batch.device
            ),
        )
        self._update_selection(hook, force_hook, ctx)

        assert torch.where(batch.force_mode == CLIMBING_NEB)[0].tolist() == [1, 6]

        batch.energy.copy_(
            torch.tensor([[0.0], [1.0], [3.0], [0.0], [0.0], [4.0], [2.0], [0.0]])
        )
        ctx.step_count = 1
        self._update_selection(hook, force_hook, ctx, admit=False)

        assert torch.where(batch.force_mode == CLIMBING_NEB)[0].tolist() == [2, 5]

    def test_only_updates_active_paths(self) -> None:
        """Selection preserves a climbing image owned by another stage."""
        batch = _bands(
            [0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 3.0, 0.0],
            [0] * 4 + [1] * 4,
        )
        hook, force_hook = _selection_hooks("dynamic")
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.tensor(
                [True, True, True, True, False, False, False, False]
            ),
        )
        force_hook(ctx, DynamicsStage.ON_ADMISSION)
        batch.force_mode[5] = CLIMBING_NEB
        hook.energy_stats_hook(ctx, DynamicsStage.ON_ADMISSION)
        hook.energy_stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

        assert torch.where(batch.force_mode == CLIMBING_NEB)[0].tolist() == [1, 5]


class TestNEBForceHook:
    """Test field preparation and the existing tensor-op boundary."""

    def test_registration_requires_group_aware_dynamics(self) -> None:
        hook = _force_hook()

        with pytest.raises(ValueError, match="requires dynamics.*by_group=True"):
            hook.on_register(Mock(by_group=False))

    @pytest.mark.parametrize("active_graph_mask_is_none", [False, True])
    def test_masks_physical_forces_and_publishes_effective_forces(
        self, monkeypatch, active_graph_mask_is_none: bool
    ) -> None:
        batch = _bands([0.0, 2.0, 1.0, 0.0], [0] * 4)
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=(
                None
                if active_graph_mask_is_none
                else torch.ones(batch.num_graphs, dtype=torch.bool, device=batch.device)
            ),
            step_count=0,
        )
        hook = _force_hook(spring=0.2)
        selection_hook = ClimbingImageSelectionHook(
            energy_stats_hook=hook.energy_stats_hook
        )
        hook.energy_stats_hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        selection_hook(ctx, DynamicsStage.ON_ADMISSION)
        workspace = hook._workspace
        assert workspace is not None
        assert workspace.image_ptr.dtype == torch.int32
        assert workspace.path_ptr.dtype == torch.int32
        assert workspace.image_path_idx.dtype == torch.int32
        hook.energy_stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
        selection_hook(ctx, DynamicsStage.AFTER_COMPUTE)
        model_forces = batch.forces.clone()
        seen: dict[str, torch.Tensor] = {}

        def fake_neb_forces(**kwargs):
            seen.update(kwargs)
            kwargs["effective_forces"].fill_(7.0)
            kwargs["link_lengths"].fill_(1.0)
            return kwargs["effective_forces"], kwargs["link_lengths"]

        module = importlib.import_module("nvalchemi.dynamics.paths.neb.hooks.neb_force")
        monkeypatch.setattr(module, "neb_forces", fake_neb_forces)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

        expected_physical = model_forces.masked_fill(
            workspace.fixed_node_mask[:, None], 0
        )
        assert torch.equal(batch.physical_forces, expected_physical)
        assert torch.equal(seen["physical_forces"], expected_physical)
        assert torch.all(batch.forces[workspace.fixed_node_mask] == 0)
        assert torch.all(batch.forces[~workspace.fixed_node_mask] == 7.0)
        assert torch.allclose(seen["spring_constants"], torch.full((3,), 0.2))
        assert seen["image_ptr"] is workspace.image_ptr
        assert seen["path_ptr"] is workspace.path_ptr
        assert seen["image_path_idx"] is workspace.image_path_idx
        assert seen["image_force_mode"] is batch.force_mode
        assert seen["image_force_mode"][1] == CLIMBING_NEB
        assert seen["image_force_mode"][[0, 3]].tolist() == [ENDPOINT] * 2

    @pytest.mark.parametrize(
        ("reprime_pending", "expected_velocity"),
        [
            (None, [1.0, 2.0, 3.0, 4.0]),
            ([True, False, True, True], [0.0, 2.0, 3.0, 0.0]),
        ],
    )
    def test_zeros_velocity_only_for_active_reprime_pending_images(
        self,
        monkeypatch,
        reprime_pending: list[bool] | None,
        expected_velocity: list[float],
    ) -> None:
        """Force reprime resets velocity before the first optimizer update."""
        batch = _bands([0.0, 2.0, 1.0, 0.0], [0] * 4)
        batch.add_key(
            "velocities",
            [torch.full((1, 3), float(i + 1)) for i in range(4)],
            level="node",
            overwrite=True,
        )
        if reprime_pending is not None:
            batch.reprime_pending = torch.tensor(reprime_pending).unsqueeze(-1)
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.tensor([True, True, False, True]),
        )
        hook = _force_hook()
        hook.energy_stats_hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        hook.energy_stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)

        def fake_neb_forces(**kwargs):
            kwargs["effective_forces"].zero_()
            kwargs["link_lengths"].zero_()
            return kwargs["effective_forces"], kwargs["link_lengths"]

        module = importlib.import_module("nvalchemi.dynamics.paths.neb.hooks.neb_force")
        monkeypatch.setattr(module, "neb_forces", fake_neb_forces)

        hook(ctx, DynamicsStage.AFTER_COMPUTE)

        expected = torch.tensor(expected_velocity).unsqueeze(-1).expand(-1, 3)
        assert torch.equal(batch.velocities, expected)

    def test_after_compute_compiles_with_cuda_graphs(self, gpu_device: str) -> None:
        """The complete prepared force hook supports full CUDA-graph capture."""
        batch = _bands(
            [0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 3.0, 0.0],
            [0] * 4 + [1] * 4,
        ).to(gpu_device)
        batch.set_group_layout(torch.tensor([0] * 4 + [1] * 4, device=gpu_device))
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.tensor(
                [True] * 4 + [False] * 4,
                dtype=torch.bool,
                device=gpu_device,
            ),
        )
        hook = _force_hook()
        hook.energy_stats_hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        inactive_forces = batch.forces[4:].clone()

        def refresh() -> None:
            hook.energy_stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
            hook(ctx, DynamicsStage.AFTER_COMPUTE)

        compiled = torch.compile(refresh, backend="cudagraphs", fullgraph=True)
        try:
            compiled()
            torch.cuda.synchronize()
            assert torch.isfinite(batch.forces).all()
            assert torch.equal(batch.forces[4:], inactive_forces)
            assert torch.isfinite(batch.forward_link_length).all()
        finally:
            torch.compiler.reset()

    def test_rejects_incomplete_path(self) -> None:
        batch = _bands([0.0, 1.0], [0, 0])
        hook = _force_hook()
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.ones(
                batch.num_graphs, dtype=torch.bool, device=batch.device
            ),
        )

        try:
            hook._prepare_batch(ctx)
        except ValueError as error:
            assert "at least three images" in str(error)
        else:
            raise AssertionError("Expected an incomplete path to be rejected")

    def test_runs_at_preparation_and_force_stages(self) -> None:
        hook = _force_hook()

        assert hook.stage == DynamicsStage.AFTER_COMPUTE
        assert hook.frequency == 1
        assert hook._runs_on_stage(DynamicsStage.ON_ADMISSION)
        assert hook._runs_on_stage(DynamicsStage.BEFORE_PRE_UPDATE)
        assert hook._runs_on_stage(DynamicsStage.AFTER_COMPUTE)
        assert hook._runs_on_stage(DynamicsStage.BEFORE_POST_UPDATE)
        assert not hook._runs_on_stage(DynamicsStage.AFTER_STEP)

    def test_initializes_constant_springs_on_admission(self) -> None:
        batch = _bands([0.0, 0.0, 0.0, 0.0], [0, 0, 0, 0])
        hook = _force_hook(spring=0.2)

        hook(
            DynamicsContext(
                batch=batch,
                active_graph_mask=torch.ones(batch.num_graphs, dtype=torch.bool),
            ),
            DynamicsStage.ON_ADMISSION,
        )

        assert hook._workspace is not None
        assert torch.equal(
            hook._workspace.spring_constants,
            torch.full((3,), 0.2),
        )
        assert isinstance(hook.spring, ConstantSpringConfig)

    def test_dynamic_spring_receives_read_only_context(self) -> None:
        class DynamicSpringConfig:
            refresh = DynamicsStage.AFTER_COMPUTE

            def __init__(self) -> None:
                self.context: SpringContext | None = None

            def resolve(self, context: SpringContext) -> torch.Tensor:
                self.context = context
                return torch.full(
                    (context.num_links,),
                    0.3,
                    dtype=context.positions.dtype,
                    device=context.positions.device,
                )

        batch = _bands([0.0, 1.0, 2.0, 0.0], [0, 0, 0, 0])
        config = DynamicSpringConfig()
        hook = _force_hook(spring=config)
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.ones(batch.num_graphs, dtype=torch.bool),
            step_count=7,
        )

        hook(ctx, DynamicsStage.ON_ADMISSION)
        hook._refresh_spring_constants(ctx, DynamicsStage.AFTER_COMPUTE)

        workspace = hook._workspace
        context = config.context
        assert workspace is not None
        assert context is not None
        assert context.energies.untyped_storage().data_ptr() == (
            batch.energy.untyped_storage().data_ptr()
        )
        assert context.positions is batch.positions
        assert context.physical_forces is batch.physical_forces
        assert context.image_ptr is workspace.image_ptr
        assert context.layout is batch.group_layout
        assert context.cell is workspace.cell
        assert context.inv_cell is workspace.inv_cell
        assert context.pbc is workspace.pbc
        assert context.step_count == 7
        assert context.num_links == 3
        assert torch.equal(workspace.spring_constants, torch.full((3,), 0.3))

    def test_resolves_missing_cell_and_pbc_only_in_workspace(self) -> None:
        batch = _bands([0.0, 0.0, 0.0], [0, 0, 0])
        hook = _force_hook()

        hook(
            DynamicsContext(
                batch=batch,
                active_graph_mask=torch.ones(batch.num_graphs, dtype=torch.bool),
            ),
            DynamicsStage.ON_ADMISSION,
        )

        assert hook._workspace is not None
        assert torch.equal(hook._workspace.cell, torch.eye(3).unsqueeze(0))
        assert torch.equal(hook._workspace.pbc, torch.zeros(1, 3, dtype=torch.bool))
        assert "cell" not in batch
        assert "pbc" not in batch

    def test_uses_representative_cell_and_pbc_for_each_path(self) -> None:
        batch = _bands([0.0] * 6, [0, 0, 0, 1, 1, 1])
        batch.cell = torch.stack([torch.eye(3)] * 3 + [torch.eye(3) * 2] * 3)
        batch.pbc = torch.tensor([[True, True, True]] * 3 + [[True, False, False]] * 3)
        hook = _force_hook()

        hook(
            DynamicsContext(
                batch=batch,
                active_graph_mask=torch.ones(batch.num_graphs, dtype=torch.bool),
            ),
            DynamicsStage.ON_ADMISSION,
        )

        assert hook._workspace is not None
        assert torch.equal(
            hook._workspace.cell, torch.stack([torch.eye(3), torch.eye(3) * 2])
        )
        assert torch.equal(
            hook._workspace.pbc,
            torch.tensor([[True, True, True], [True, False, False]]),
        )

    @pytest.mark.parametrize(
        "stage",
        [
            DynamicsStage.BEFORE_PRE_UPDATE,
            DynamicsStage.BEFORE_POST_UPDATE,
        ],
    )
    def test_freezes_only_constrained_nodes_in_active_images(
        self, stage: DynamicsStage
    ) -> None:
        images = [
            AtomicData(
                atomic_numbers=torch.tensor([1, 1]),
                positions=torch.zeros(2, 3),
                energy=torch.tensor([[0.0]]),
                forces=torch.full((2, 3), float(image + 1)),
            )
            for image in range(3)
        ]
        batch = Batch.from_data_list(images)
        batch.set_group_layout(torch.zeros(3, dtype=torch.long))
        batch.add_key(
            "velocities",
            [torch.full((2, 3), float(image + 1)) for image in range(3)],
            level="node",
            overwrite=True,
        )
        hook = _force_hook(fixed_atom_indices=[1])
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.tensor([False, True, True]),
        )

        hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, stage)

        assert torch.equal(
            batch.forces,
            torch.tensor(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )
        assert torch.equal(batch.velocities, batch.forces)

    @pytest.mark.parametrize("indices", [[-1], [1]])
    def test_rejects_indices_not_present_in_every_image(
        self, indices: list[int]
    ) -> None:
        batch = _bands([0.0, 0.0, 0.0], [0, 0, 0])
        hook = _force_hook(fixed_atom_indices=indices)

        with pytest.raises(ValueError, match="valid for every image"):
            hook(
                DynamicsContext(
                    batch=batch,
                    active_graph_mask=torch.ones(batch.num_graphs, dtype=torch.bool),
                ),
                DynamicsStage.ON_ADMISSION,
            )

    def test_publishes_minimum_image_forward_link_lengths(self) -> None:
        batch = _bands([0.0, 0.0, 0.0], [0, 0, 0])
        cell = torch.eye(3) * 5
        step = torch.tensor([4.0, 0.0, 0.0])
        batch.cell = cell.repeat(3, 1, 1)
        batch.pbc = torch.tensor([[True, False, False]]).repeat(3, 1)
        batch.positions.copy_(torch.stack([torch.zeros(3), step, 2 * step]))

        hook = _force_hook()
        ctx = DynamicsContext(batch=batch)
        hook.energy_stats_hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        hook.energy_stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

        torch.testing.assert_close(
            batch.forward_link_length,
            torch.tensor([1.0, 1.0, 0.0]),
        )
