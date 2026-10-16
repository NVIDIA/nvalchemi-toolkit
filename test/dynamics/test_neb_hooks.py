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
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pytest
import torch
import torch._dynamo as dynamo
import warp as wp

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import DynamicsStage
from nvalchemi.dynamics.hooks import FreezeAtomsHook
from nvalchemi.dynamics.mep import (
    ConstantSpringConfig,
    NEBMethod,
    SpringContext,
)
from nvalchemi.dynamics.mep._geometry import (
    PreparedMIC,
    minimum_image_displacement,
    prepare_batch_mic,
)
from nvalchemi.dynamics.mep.hooks import (
    ClimbingImageSelectionHook,
    NEBForceHook,
    PathEnergyStatsHook,
)
from nvalchemi.dynamics.mep.neb_equations import neb_effective_force_from_gram_stats
from nvalchemi.dynamics.mep.neb_ops.methods import DEFAULT_NEB_METHOD_KEY
from nvalchemi.dynamics.mep.neb_ops.modes import (
    CLIMBING_NEB,
    ENDPOINT,
)
from nvalchemi.hooks import DynamicsContext


@wp.func
def _custom_tangent_weights(energy_prev: Any, energy_curr: Any, energy_next: Any):
    """Use equal weights to exercise importable custom NEB equations."""
    one = type(energy_curr)(1.0)
    return one, one


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

    @pytest.mark.parametrize("selection", ["fixed", "dynamic"])
    def test_selection_survives_path_admission(
        self, device: str, selection: str
    ) -> None:
        """Retained and arriving paths keep fixed choices after first selection."""
        batch = _bands(
            [0.0, 4.0, 1.0, 0.0, 0.0, 3.0, 2.0, 1.0, 0.0], [0] * 4 + [1] * 5
        ).to(device)
        hook, force_hook = _selection_hooks(selection)
        ctx = DynamicsContext(batch=batch)
        self._update_selection(hook, force_hook, ctx)
        assert torch.where(batch.force_mode == CLIMBING_NEB)[0].tolist() == [1, 5]

        current = batch.index_select_groups([1, 0])
        current.energy.copy_(
            torch.tensor(
                [0.0, 1.0, 5.0, 2.0, 0.0, 0.0, 1.0, 6.0, 0.0], device=device
            ).unsqueeze(-1)
        )
        ctx = DynamicsContext(
            batch=current,
            previous_graph_indices=torch.tensor(
                [4, 5, 6, 7, 8, -1, -1, -1, -1], device=device
            ),
            active_graph_mask=torch.tensor([True] * 5 + [False] * 4, device=device),
        )
        # Match the strategy's admission order: statistics, selection, force.
        hook.energy_stats_hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        force_hook(ctx, DynamicsStage.ON_ADMISSION)
        self._update_selection(hook, force_hook, ctx, admit=False)
        resident_climber = 1 if selection == "fixed" else 2
        climbers = torch.where(current.force_mode == CLIMBING_NEB)[0].tolist()
        assert climbers == [resident_climber]

        ctx.active_graph_mask.fill_(True)
        ctx.step_count = 1
        self._update_selection(hook, force_hook, ctx, admit=False)
        climbers = torch.where(current.force_mode == CLIMBING_NEB)[0].tolist()
        assert climbers == [resident_climber, 7]

        # An arrival keeps its fixed choice even if a different image becomes
        # highest in energy; dynamic selection follows the new maximum.
        current.energy[6] = 7.0
        ctx.step_count = 2
        self._update_selection(hook, force_hook, ctx, admit=False)
        arriving_climber = 7 if selection == "fixed" else 6
        climbers = torch.where(current.force_mode == CLIMBING_NEB)[0].tolist()
        assert climbers == [resident_climber, arriving_climber]

        # A fresh admission resets state even when the batch object is reused.
        ctx.previous_graph_indices = torch.full_like(ctx.previous_graph_indices, -1)
        self._update_selection(hook, force_hook, ctx)
        assert torch.where(current.force_mode == CLIMBING_NEB)[0].tolist() == [2, 6]

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

    def test_compiled_fixed_selection_does_not_recompile(self) -> None:
        """Compiled selection keeps its Python fast-path guard unchanged."""
        batch = _bands([0.0, 2.0, 1.0, 0.0], [0] * 4)
        hook, force_hook = _selection_hooks("fixed")
        ctx = DynamicsContext(batch=batch, active_graph_mask=None)

        hook.energy_stats_hook(ctx, DynamicsStage.ON_ADMISSION)
        force_hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        hook.energy_stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)

        def select() -> None:
            hook(ctx, DynamicsStage.AFTER_COMPUTE)

        previous_error_on_recompile = dynamo.config.error_on_recompile
        dynamo.reset()
        dynamo.config.error_on_recompile = True
        compiled = torch.compile(select, backend="eager", fullgraph=True)
        try:
            compiled()
            compiled()
        finally:
            dynamo.config.error_on_recompile = previous_error_on_recompile
            dynamo.reset()

        assert torch.where(batch.force_mode == CLIMBING_NEB)[0].tolist() == [1]
        assert hook._fixed_selection_complete is False

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

    def test_status_code_restricts_fused_selection_to_one_stage(self) -> None:
        """A fused selector only updates paths owned by its configured stage."""
        batch = _bands(
            [0.0, 2.0, 1.0, 0.0, 0.0, 1.0, 3.0, 0.0],
            [0] * 4 + [1] * 4,
        )
        batch.status = torch.tensor([0] * 4 + [1] * 4)
        energy_stats_hook = PathEnergyStatsHook()
        hook = ClimbingImageSelectionHook(
            energy_stats_hook=energy_stats_hook,
            selection="dynamic",
            status_code=1,
        )
        force_hook = NEBForceHook(energy_stats_hook=energy_stats_hook)
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.ones(batch.num_graphs, dtype=torch.bool),
        )

        self._update_selection(hook, force_hook, ctx)

        assert torch.where(batch.force_mode == CLIMBING_NEB)[0].tolist() == [6]


class TestNEBForceHook:
    """Test field preparation and the existing tensor-op boundary."""

    def test_preserves_retained_state_and_initializes_arrivals(
        self, device: str
    ) -> None:
        """Runtime payloads and packed springs follow paths with unequal sizes."""

        class AdmissionSpring:
            refresh = DynamicsStage.ON_ADMISSION

            def __init__(self) -> None:
                self.calls = 0

            def resolve(self, context: SpringContext) -> torch.Tensor:
                self.calls += 1
                return (
                    torch.arange(
                        context.num_links,
                        dtype=context.positions.dtype,
                        device=context.positions.device,
                    )
                    + 100 * self.calls
                )

        images = [
            AtomicData(
                atomic_numbers=torch.ones(atoms, dtype=torch.long),
                positions=torch.zeros(atoms, 3),
            )
            for count, atoms in ((4, 1), (5, 2))
            for _ in range(count)
        ]
        batch = Batch.from_data_list(images).to(device)
        batch.set_group_layout(torch.tensor([0] * 4 + [1] * 5, device=device))
        supplied = torch.arange(batch.num_nodes, device=device) % 2 == 0
        batch.add_key(
            "neb_fixed_node_mask",
            list(supplied.split(batch.num_nodes_list)),
            level="node",
        )
        hook = _force_hook(
            spring=AdmissionSpring(),
            endpoint_mode="relaxed",
        )
        hook(DynamicsContext(batch=batch), DynamicsStage.ON_ADMISSION)
        batch.force_mode[5] = CLIMBING_NEB
        batch.physical_forces.fill_(7)
        batch.forward_link_length.fill_(9)

        current = batch.index_select_groups([1, 0])
        current.neb_fixed_node_mask[10:] = True
        current.force_mode[5:] = CLIMBING_NEB
        ctx = DynamicsContext(
            batch=current,
            previous_graph_indices=torch.tensor(
                [4, 5, 6, 7, 8, -1, -1, -1, -1], device=device
            ),
        )
        hook(ctx, DynamicsStage.ON_ADMISSION)

        assert torch.where(current.force_mode == CLIMBING_NEB)[0].tolist() == [1]
        assert current.force_mode[[0, 4, 5, 8]].tolist() == [ENDPOINT] * 4
        assert current.neb_fixed_node_mask.tolist() == [True, False] * 5 + [True] * 4
        assert torch.all(current.physical_forces[:10] == 7)
        assert torch.all(current.physical_forces[10:] == 0)
        assert current.forward_link_length.tolist() == [9.0] * 5 + [0.0] * 4
        expected_springs = [103.0, 104.0, 105.0, 106.0, 204.0, 205.0, 206.0]
        assert hook._workspace.spring_constants.tolist() == expected_springs

        # No mapping is a fresh admission, including when reusing this batch.
        hook(DynamicsContext(batch=current), DynamicsStage.ON_ADMISSION)
        assert not torch.any(current.force_mode == CLIMBING_NEB)
        assert not torch.any(current.physical_forces)
        assert not torch.any(current.forward_link_length)
        expected_springs = [300.0, 301.0, 302.0, 303.0, 304.0, 305.0, 306.0]
        assert hook._workspace.spring_constants.tolist() == expected_springs

    @pytest.mark.parametrize("endpoint_mode", ["fixed", "relaxed"])
    def test_uses_supplied_node_mask_in_place(
        self, device: str, endpoint_mode: str
    ) -> None:
        """The supplied mask is the only fixed-atom field and includes endpoints."""
        batch = _bands([0.0] * 4, [0] * 4).to(device)
        supplied = torch.tensor([False, True, False, False], device=device)
        batch.add_key(
            "neb_fixed_node_mask",
            list(supplied.split(batch.num_nodes_list)),
            level="node",
        )
        mask = batch.neb_fixed_node_mask
        hook = _force_hook(endpoint_mode=endpoint_mode)
        hook(DynamicsContext(batch=batch), DynamicsStage.ON_ADMISSION)
        expected = (
            [True, True, False, True]
            if endpoint_mode == "fixed"
            else [False, True, False, False]
        )
        assert batch.neb_fixed_node_mask.tolist() == expected
        assert batch.neb_fixed_node_mask is mask
        assert hook._workspace.fixed_node_mask is mask

    @pytest.mark.parametrize("invalid", ["dtype", "shape", "level"])
    def test_rejects_invalid_node_masks(self, invalid: str) -> None:
        """The mask must be a Boolean atom field with one entry per node."""
        batch = _bands([0.0] * 3, [0] * 3)
        values = torch.zeros(
            (3, 2) if invalid == "shape" else (3,),
            dtype=torch.float32 if invalid == "dtype" else torch.bool,
        )
        batch.add_key(
            "neb_fixed_node_mask",
            list(values.split(1)),
            level="system" if invalid == "level" else "node",
        )
        hook = _force_hook()
        with pytest.raises(ValueError, match="boolean node mask"):
            hook(DynamicsContext(batch=batch), DynamicsStage.ON_ADMISSION)

    @pytest.mark.parametrize(
        "workflow",
        [
            SimpleNamespace(sampler=object(), has_neighbor=False),
            SimpleNamespace(sampler=None, has_neighbor=True),
        ],
    )
    def test_streaming_rejects_fixed_atom_indices(self, workflow: object) -> None:
        """Constraints must follow paths when their positions in the batch change."""
        hook = _force_hook(fixed_atom_indices={0: [0]})
        ctx = DynamicsContext(batch=_bands([0.0] * 3, [0] * 3), workflow=workflow)
        with pytest.raises(
            ValueError, match="Streaming NEB requires batch.neb_fixed_node_mask"
        ):
            hook(ctx, DynamicsStage.ON_ADMISSION)

    def test_published_forces_apply_climbing_law(self) -> None:
        """The real NEB operation publishes climbing and regular forces."""
        translations = (0.0, 1.0, 3.0, 4.0)
        energies = (0.0, 2.0, 1.0, 0.0)
        model_forces = (
            [[7.0, 8.0, 0.0], [9.0, 10.0, 0.0]],
            [[1.0, 2.0, 0.0], [3.0, 4.0, 0.0]],
            [[1.0, 5.0, 0.0], [3.0, 6.0, 0.0]],
            [[11.0, 12.0, 0.0], [13.0, 14.0, 0.0]],
        )
        images = [
            AtomicData(
                atomic_numbers=torch.tensor([1, 1]),
                positions=torch.tensor(
                    [[translation, 0.0, 0.0], [translation, 1.0, 0.0]]
                ),
                energy=torch.tensor([[energy]]),
                forces=torch.tensor(forces),
            )
            for translation, energy, forces in zip(
                translations, energies, model_forces, strict=True
            )
        ]
        batch = Batch.from_data_list(images)
        batch.set_group_layout(torch.zeros(4, dtype=torch.long))
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.ones(batch.num_graphs, dtype=torch.bool),
        )
        hook = _force_hook(
            method="improved_tangent",
            spring=ConstantSpringConfig(0.1),
            endpoint_mode="fixed",
            fixed_atom_indices={},
        )

        hook.on_register(Mock(by_group=True))
        hook.energy_stats_hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        batch.force_mode[1] = CLIMBING_NEB
        hook.energy_stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

        expected_climbing = torch.tensor([[-3.0, 2.0, 0.0], [-1.0, 4.0, 0.0]])
        expected_regular = torch.tensor([[-1.1, 5.0, 0.0], [0.9, 6.0, 0.0]])
        endpoint_rows = (batch.batch_idx == 0) | (batch.batch_idx == 3)
        torch.testing.assert_close(
            batch.forces[batch.batch_idx == 1], expected_climbing
        )
        torch.testing.assert_close(batch.forces[batch.batch_idx == 2], expected_regular)
        assert torch.all(batch.forces[endpoint_rows] == 0)

    def test_torch_method_receives_path_inputs_and_publishes_forces(self) -> None:
        """Torch methods own MIC links; the hook publishes both returned tensors."""

        class DoubleForces:
            seen: dict[str, object] | None = None

            def __call__(
                self,
                batch: Batch,
                *,
                spring_constants: torch.Tensor,
                path_energy_ref: torch.Tensor,
                path_energy_max: torch.Tensor,
                mic: PreparedMIC,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                self.seen = {
                    "batch": batch,
                    "spring_constants": spring_constants,
                    "path_energy_ref": path_energy_ref,
                    "path_energy_max": path_energy_max,
                    "mic": mic,
                }
                links = minimum_image_displacement(
                    batch.positions[1:] - batch.positions[:-1],
                    torch.zeros(2, dtype=torch.long),
                    prepared=mic,
                )
                return 2 * batch.physical_forces, torch.linalg.vector_norm(
                    links, dim=-1
                )

        batch = _bands([0.0, 1.0, 0.0], [0, 0, 0])
        batch.positions[:, 0] = torch.tensor([0.0, 4.0, 8.0])
        batch.cell = (torch.eye(3) * 5).repeat(3, 1, 1)
        batch.pbc = torch.tensor([[True, False, False]]).repeat(3, 1)
        method = DoubleForces()
        hook = _force_hook(method=method)
        ctx = DynamicsContext(batch=batch)
        hook.energy_stats_hook(ctx, DynamicsStage.ON_ADMISSION)
        hook(ctx, DynamicsStage.ON_ADMISSION)
        hook.energy_stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)

        with patch("nvalchemi.dynamics.mep.hooks.neb_force.neb_forces") as warp_op:
            hook(ctx, DynamicsStage.AFTER_COMPUTE)
        warp_op.assert_not_called()

        assert hook.method_key is None
        assert method.seen is not None
        assert method.seen["batch"] is batch
        torch.testing.assert_close(
            batch.forward_link_length, torch.tensor([1.0, 1.0, 0.0])
        )
        torch.testing.assert_close(
            minimum_image_displacement(
                batch.positions[1:2] - batch.positions[0:1],
                torch.zeros(1, dtype=torch.long),
                prepared=method.seen["mic"],
            ),
            torch.tensor([[-1.0, 0.0, 0.0]]),
        )
        torch.testing.assert_close(batch.forces[:, 0], torch.tensor([0.0, 4.0, 0.0]))

    def test_prepares_builtin_and_gram_statistic_method_keys(self) -> None:
        builtin = _force_hook()
        gram = _force_hook(
            method=NEBMethod(effective_force_fn=neb_effective_force_from_gram_stats)
        )

        assert builtin.method_key == DEFAULT_NEB_METHOD_KEY
        assert gram.method_key.startswith("gram_stats|")

        batch = _bands([0.0, 2.0, 1.0, 0.0], [0] * 4)
        ctx = DynamicsContext(batch=batch)
        gram.energy_stats_hook(ctx, DynamicsStage.ON_ADMISSION)
        gram(ctx, DynamicsStage.ON_ADMISSION)
        assert gram._workspace is not None
        assert gram._workspace.vector_scratch is None
        gram.energy_stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
        gram(ctx, DynamicsStage.AFTER_COMPUTE)
        assert torch.isfinite(batch.forces).all()

    def test_custom_method_key_restores_equation_functions(self) -> None:
        method = NEBMethod(
            tangent_weights_fn=_custom_tangent_weights,
            effective_force_fn=neb_effective_force_from_gram_stats,
        )

        key = method.to_key()
        restored = NEBMethod.from_key(key)

        assert key.startswith("gram_stats|")
        assert "test_neb_hooks._custom_tangent_weights" in key
        assert restored.tangent_weights_fn is method.tangent_weights_fn
        assert restored.effective_force_fn is method.effective_force_fn
        assert restored.climbing_force_fn is method.climbing_force_fn
        assert _force_hook(method=restored).method_key == key

    def test_rejects_unknown_method_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown NEB method"):
            _force_hook(method="unknown")

    def test_registration_requires_group_aware_dynamics(self) -> None:
        hook = _force_hook()

        with pytest.raises(ValueError, match="requires dynamics.*by_group=True"):
            hook.on_register(Mock(by_group=False))

    @pytest.mark.parametrize(
        "active_graph_mask",
        [None, [True] * 4, [True, True, False, False]],
    )
    def test_masks_physical_forces_and_publishes_effective_forces(
        self, monkeypatch, active_graph_mask: list[bool] | None
    ) -> None:
        batch = _bands([0.0, 2.0, 1.0, 0.0], [0] * 4)
        mask = (
            None
            if active_graph_mask is None
            else torch.tensor(active_graph_mask, dtype=torch.bool, device=batch.device)
        )
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=mask,
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

        module = importlib.import_module("nvalchemi.dynamics.mep.hooks.neb_force")
        monkeypatch.setattr(module, "neb_forces", fake_neb_forces)
        hook(ctx, DynamicsStage.AFTER_COMPUTE)

        active_nodes = (
            torch.ones(batch.num_nodes, dtype=torch.bool, device=batch.device)
            if mask is None
            else mask[batch.batch_idx.long()]
        )
        expected_physical = torch.where(
            active_nodes.unsqueeze(-1),
            model_forces,
            torch.zeros_like(model_forces),
        )
        expected_physical.masked_fill_(
            (active_nodes & workspace.fixed_node_mask).unsqueeze(-1), 0
        )
        expected_effective = torch.full_like(workspace.effective_forces, 7.0)
        expected_effective.masked_fill_(
            (active_nodes & workspace.fixed_node_mask).unsqueeze(-1), 0
        )
        assert torch.equal(batch.physical_forces, expected_physical)
        assert torch.equal(seen["physical_forces"], expected_physical)
        expected_published = torch.where(
            active_nodes.unsqueeze(-1),
            expected_effective,
            model_forces,
        )
        assert torch.equal(workspace.effective_forces, expected_effective)
        assert torch.equal(batch.forces, expected_published)
        assert torch.allclose(seen["spring_constants"], torch.full((3,), 0.2))
        assert seen["image_ptr"] is workspace.image_ptr
        assert seen["path_ptr"] is workspace.path_ptr
        assert seen["image_path_idx"] is workspace.image_path_idx
        assert seen["image_force_mode"] is batch.force_mode
        assert seen["mic_mode"] is workspace.mic.mode
        assert seen["periodic_basis"] is workspace.mic.periodic_basis
        assert seen["cartesian_to_fractional"] is workspace.mic.cartesian_to_fractional
        assert seen["mic_candidate_count"] is workspace.mic.candidate_count
        assert seen["candidate_shifts"] is workspace.mic.candidate_shifts
        assert seen["method"] == DEFAULT_NEB_METHOD_KEY
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

        module = importlib.import_module("nvalchemi.dynamics.mep.hooks.neb_force")
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

        def refresh() -> None:
            hook.energy_stats_hook(ctx, DynamicsStage.AFTER_COMPUTE)
            hook(ctx, DynamicsStage.AFTER_COMPUTE)

        compiled = torch.compile(refresh, backend="cudagraphs", fullgraph=True)
        try:
            compiled()
            torch.cuda.synchronize()
            assert torch.isfinite(batch.forces).all()
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
        assert hook._runs_on_stage(DynamicsStage.AFTER_COMPUTE)
        assert not hook._runs_on_stage(DynamicsStage.BEFORE_PRE_UPDATE)
        assert not hook._runs_on_stage(DynamicsStage.BEFORE_POST_UPDATE)
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

    def test_constant_spring_refresh_is_fixed(self) -> None:
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            ConstantSpringConfig(0.2, refresh=DynamicsStage.AFTER_COMPUTE)

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

    def test_reuses_prepared_mic_geometry(self) -> None:
        """Admission reuses compatible MIC geometry already cached on the batch."""
        batch = _bands([0.0] * 3, [0, 0, 0])
        batch.cell = torch.eye(3).unsqueeze(0).repeat(3, 1, 1)
        batch.pbc = torch.ones((3, 3), dtype=torch.bool)
        prepared = prepare_batch_mic(batch, batch.cell[:1], batch.pbc[:1])
        hook = _force_hook()

        with patch("nvalchemi.dynamics.mep._geometry.prepare_mic") as prepare:
            hook(DynamicsContext(batch=batch), DynamicsStage.ON_ADMISSION)

        prepare.assert_not_called()
        assert hook._workspace is not None
        assert hook._workspace.mic is prepared
        assert batch._mic_data is prepared

    def test_applies_distinct_fixed_atom_indices_per_path(self, device: str) -> None:
        images = [
            AtomicData(
                atomic_numbers=torch.ones(num_atoms, dtype=torch.long),
                positions=torch.zeros(num_atoms, 3),
                energy=torch.tensor([[0.0]]),
                forces=torch.ones(num_atoms, 3),
            )
            for num_atoms in (1, 3, 2)
            for _ in range(3)
        ]
        batch = Batch.from_data_list(images).to(device)
        batch.set_group_layout(torch.tensor([0] * 3 + [1] * 3 + [2] * 3, device=device))
        hook = _force_hook(
            endpoint_mode="relaxed",
            fixed_atom_indices={1: [0, 2], 2: [1]},
        )

        hook(
            DynamicsContext(
                batch=batch,
                active_graph_mask=torch.ones(
                    batch.num_graphs, dtype=torch.bool, device=device
                ),
            ),
            DynamicsStage.ON_ADMISSION,
        )

        assert hook.fixed_atom_indices == {1: (0, 2), 2: (1,)}
        assert hook._workspace is not None
        assert torch.equal(
            hook._workspace.fixed_node_mask,
            torch.tensor(
                [False] * 3 + [True, False, True] * 3 + [False, True] * 3,
                device=device,
            ),
        )

    @pytest.mark.parametrize(
        ("indices", "message"),
        [
            ({"0": [0]}, "keys must be integer path indices"),
            ({0: 0}, "values must be sequences of integers"),
            ({0: [True]}, "values must contain only integers"),
        ],
    )
    def test_rejects_invalid_fixed_atom_mapping(
        self, indices: object, message: str
    ) -> None:
        with pytest.raises(TypeError, match=message):
            _force_hook(fixed_atom_indices=indices)

    def test_rejects_flat_fixed_atom_indices(self) -> None:
        with pytest.raises(TypeError, match="must map path indices"):
            _force_hook(fixed_atom_indices=[1])

    def test_rejects_nonexistent_fixed_atom_path(self) -> None:
        batch = _bands([0.0] * 6, [0, 0, 0, 1, 1, 1])
        hook = _force_hook(fixed_atom_indices={2: [0]})

        with pytest.raises(ValueError, match="existing paths; got 2"):
            hook(
                DynamicsContext(
                    batch=batch,
                    active_graph_mask=torch.ones(batch.num_graphs, dtype=torch.bool),
                ),
                DynamicsStage.ON_ADMISSION,
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
            [torch.full((2, 3), float(image + 11)) for image in range(3)],
            level="node",
            overwrite=True,
        )
        hook = _force_hook(fixed_atom_indices={0: [1]})
        freeze_hook = FreezeAtomsHook(mask_key="neb_fixed_node_mask")
        ctx = DynamicsContext(
            batch=batch,
            active_graph_mask=torch.tensor([False, True, True]),
        )

        hook(ctx, DynamicsStage.ON_ADMISSION)
        freeze_hook(ctx, stage)

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
        assert torch.equal(
            batch.velocities,
            torch.tensor(
                [
                    [11.0, 11.0, 11.0],
                    [11.0, 11.0, 11.0],
                    [12.0, 12.0, 12.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        )

    @pytest.mark.parametrize("indices", [{0: [-1]}, {0: [1]}])
    def test_rejects_indices_not_present_in_every_image(
        self, indices: dict[int, list[int]]
    ) -> None:
        batch = _bands([0.0, 0.0, 0.0], [0, 0, 0])
        hook = _force_hook(fixed_atom_indices=indices)

        with pytest.raises(ValueError, match="valid image-local atom indices"):
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
