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

"""Tests for the nudged elastic band strategy."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    ConvergenceHook,
    DynamicsStage,
    FusedStage,
)
from nvalchemi.dynamics.hooks import LoggingHook
from nvalchemi.dynamics.paths import (
    NEB,
    ClimbingImageConfig,
    ConstantSpringConfig,
    IDPPModel,
    interpolate_paths,
    prepare_idpp_targets,
)
from nvalchemi.dynamics.paths.hooks import (
    PathDiagnosticsHook,
    PathEnergyStatsHook,
)
from nvalchemi.dynamics.paths.neb.hooks import (
    ClimbingImageSelectionHook,
    NEBForceHook,
)
from nvalchemi.hooks import DynamicsContext
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _CompilerFriendlyModel(torch.nn.Module, BaseModelMixin):
    """Minimal analytical model for end-to-end compilation tests."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            neighbor_config=None,
            needs_pbc=False,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no embedding outputs for this test model."""
        return {}

    def compute_embeddings(self, data: Batch) -> Batch:
        """Return the batch unchanged because embeddings are not used."""
        return data

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Compute analytical per-atom energies and forces."""
        positions = batch.positions
        return {
            "energy": positions.square().sum(dim=-1, keepdim=True),
            "forces": -2 * positions,
        }


class _NoOpHook:
    """Minimal observer used to verify strategy hook replacement semantics."""

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __call__(self, ctx: DynamicsContext, stage: DynamicsStage) -> None:
        """Observe a dynamics stage without changing state."""


def _model() -> DemoModelWrapper:
    """Return a lightweight model for strategy construction and execution."""
    return DemoModelWrapper(DemoModel()).eval()


def _bands() -> Batch:
    """Return one valid three-image path with optimizer state fields."""
    images = []
    for x in (0.0, 0.5, 1.0):
        image = AtomicData(
            atomic_numbers=torch.tensor([1]),
            positions=torch.tensor([[x, 0.0, 0.0]]),
            energy=torch.zeros(1, 1),
            forces=torch.zeros(1, 3),
        )
        image.add_node_property("velocities", torch.zeros(1, 3))
        images.append(image)
    bands = Batch.from_data_list(images)
    bands.set_group_layout(torch.zeros(3, dtype=torch.long))
    return bands


def _force_hook(engine: FusedStage) -> NEBForceHook:
    """Return the shared NEB force hook owned by the fused engine."""
    return next(hook for hook in engine.hooks if isinstance(hook, NEBForceHook))


# ---------------------------------------------------------------------------
# NEB configuration
# ---------------------------------------------------------------------------


class TestNEBConfiguration:
    """Validate strategy configuration and optimizer construction."""

    def test_spec_round_trip_preserves_configuration(self) -> None:
        """JSON recipes preserve NEB-specific configuration."""
        strategy = NEB(
            model=_model(),
            spring=ConstantSpringConfig(0.2),
            climbing=ClimbingImageConfig(
                regular_fmax=0.5,
                max_regular_steps=11,
            ),
            n_steps=19,
            fixed_atom_indices={0: [0, 2], 1: [1]},
            diagnostics_log_path="neb.csv",
        )

        spec = json.loads(json.dumps(strategy.to_spec_dict()))

        assert spec["optimizer"] == "nvalchemi.dynamics.optimizers.fire2.FIRE2"
        assert spec["climbing"]["regular_fmax"] == 0.5

        restored = NEB.from_spec_dict(spec, model=strategy.model)

        assert restored.n_steps == 19
        assert restored.spring == ConstantSpringConfig(0.2)
        assert restored.climbing == ClimbingImageConfig(
            regular_fmax=0.5,
            max_regular_steps=11,
        )
        assert restored.fixed_atom_indices == {0: (0, 2), 1: (1,)}
        assert restored.diagnostics_log_path == Path("neb.csv")

    @pytest.mark.parametrize(
        ("indices", "message"),
        [
            ([0], "must map path indices"),
            ({"0": [0]}, "keys must be integer path indices"),
            ({0: 0}, "values must be sequences of integers"),
            ({0: [True]}, "values must contain only integers"),
        ],
    )
    def test_rejects_invalid_fixed_atom_mapping(
        self, indices: object, message: str
    ) -> None:
        with pytest.raises(TypeError, match=message):
            NEB(model=_model(), fixed_atom_indices=indices)

    def test_forwards_fixed_atom_mapping_to_force_hook(self) -> None:
        strategy = NEB(
            model=_model(),
            fixed_atom_indices={0: [0, 2], 1: [1]},
        )

        hook = _force_hook(strategy.build_engine())

        assert strategy.fixed_atom_indices == {0: (0, 2), 1: (1,)}
        assert hook.fixed_atom_indices == strategy.fixed_atom_indices

    def test_fire2_default_kwargs_are_optimizer_specific(self) -> None:
        strategy = NEB(model=_model())

        assert strategy.optimizer_kwargs == {"dt": 0.01}

    def test_diagnostics_log_path_builds_fresh_ordered_hooks(
        self, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "neb.csv"
        strategy = NEB(model=_model(), diagnostics_log_path=log_path)

        first_engine = strategy.build_engine()
        second_engine = strategy.build_engine()
        first_diagnostics = next(
            hook for hook in first_engine.hooks if isinstance(hook, PathDiagnosticsHook)
        )
        first_logger = next(
            hook for hook in first_engine.hooks if isinstance(hook, LoggingHook)
        )
        second_logger = next(
            hook for hook in second_engine.hooks if isinstance(hook, LoggingHook)
        )

        assert first_logger is not second_logger
        assert first_logger.backend == "csv"
        assert first_logger.log_path == log_path
        assert first_logger.by_group is True
        assert first_logger.stage is DynamicsStage.AFTER_STEP
        assert set(first_logger.custom_scalars or {}) == {
            "fmax",
            "energy_barrier",
            "highest_interior_image_idx",
            "path_length",
        }
        assert first_engine.hooks.index(first_diagnostics) < first_engine.hooks.index(
            first_logger
        )

    def test_convergence_hooks_round_trip_through_spec(self) -> None:
        strategy = NEB(
            model=_model(),
            climbing=ClimbingImageConfig(),
        )
        strategy._convergence_hook = ConvergenceHook(
            criteria={"key": "energy", "threshold": 0.2},
            by_group=True,
        )
        strategy._regular_convergence_hook = ConvergenceHook.from_fmax(
            0.5,
            by_group=True,
        )

        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        restored = NEB.from_spec_dict(spec, model=strategy.model)

        assert restored._convergence_hook is not None
        assert restored._convergence_hook.criteria[0].key == "energy"
        assert restored._regular_convergence_hook is not None
        assert restored._regular_convergence_hook.criteria[0].threshold == 0.5

    def test_regular_convergence_hook_overrides_preliminary_stage(self) -> None:
        final = ConvergenceHook(
            criteria={"key": "energy", "threshold": 0.1},
            by_group=True,
        )
        regular = ConvergenceHook.from_fmax(0.5, by_group=True)
        strategy = NEB(
            model=_model(),
            climbing=ClimbingImageConfig(),
        )
        strategy._convergence_hook = final
        strategy._regular_convergence_hook = regular
        engine = strategy.build_engine()

        regular_hook = engine.sub_stages[0][1].convergence_hook
        climbing_hook = engine.sub_stages[1][1].convergence_hook

        assert regular_hook is not regular
        assert climbing_hook is not final
        assert regular_hook.criteria[0].key == "forces"
        assert climbing_hook.criteria[0].key == "energy"

    def test_shared_override_template_is_copied_for_each_stage(self) -> None:
        template = ConvergenceHook.from_fmax(0.1, by_group=True)
        strategy = NEB(model=_model(), climbing=ClimbingImageConfig())
        strategy._regular_convergence_hook = template
        strategy._convergence_hook = template

        engine = strategy.build_engine()
        regular_hook = engine.sub_stages[0][1].convergence_hook
        final_hook = engine.sub_stages[1][1].convergence_hook

        assert regular_hook is not template
        assert final_hook is not template
        assert regular_hook is not final_hook

    def test_regular_neb_builds_one_grouped_stage(self) -> None:
        spring = ConstantSpringConfig(0.2)
        engine = NEB(
            model=_model(),
            spring=spring,
            fmax=0.05,
            n_steps=17,
            optimizer_kwargs={"dt": 0.02, "maxstep": 0.03},
        ).build_engine()

        assert isinstance(engine, FusedStage)
        assert engine.by_group is True
        assert engine.n_steps == 17
        assert engine.reprime_on_entry == frozenset()
        assert len(engine.sub_stages) == 1
        stage = engine.sub_stages[0][1]
        assert stage.by_group is True
        assert stage._dt_init == 0.02
        assert stage.maxstep == 0.03
        assert stage.convergence_hook.by_group is True
        assert stage.convergence_hook.criteria[0].threshold == 0.05
        assert not any(
            isinstance(hook, ClimbingImageSelectionHook) for hook in engine.hooks
        )
        assert not any(isinstance(hook, PathDiagnosticsHook) for hook in engine.hooks)
        assert not any(isinstance(hook, LoggingHook) for hook in engine.hooks)
        assert _force_hook(engine).spring is spring

    def test_after_regular_builds_two_stage_strategy(self) -> None:
        engine = NEB(
            model=_model(),
            fmax=0.05,
            n_steps=500,
            climbing=ClimbingImageConfig(
                selection="dynamic",
                regular_fmax=0.5,
                max_regular_steps=11,
                max_climbing_steps=13,
            ),
        ).build_engine()

        assert len(engine.sub_stages) == 2
        assert engine.n_steps == 500
        assert engine.reprime_on_entry == frozenset({1})
        regular = engine.sub_stages[0][1]
        climbing = engine.sub_stages[1][1]
        assert regular.n_steps == 11
        assert climbing.n_steps == 13
        assert regular.convergence_hook.criteria[0].threshold == 0.5
        assert climbing.convergence_hook.criteria[0].threshold == 0.05
        assert not any(
            isinstance(hook, ClimbingImageSelectionHook) for hook in regular.hooks
        )
        selection = next(
            hook
            for hook in engine.hooks
            if isinstance(hook, ClimbingImageSelectionHook)
        )
        assert selection.selection == "dynamic"
        assert selection.status_code == 1

    def test_grouped_paths_independently_enter_climbing_stage(self) -> None:
        """Only paths satisfying regular_fmax advance into CI."""
        engine = NEB(
            model=_model(),
            fmax=0.05,
            climbing=ClimbingImageConfig(regular_fmax=0.5),
        ).build_engine()
        bands = _bands().index_select([0, 1, 2, 0, 1, 2])
        bands.set_group_layout(torch.tensor([0, 0, 0, 1, 1, 1]))
        bands.status = torch.zeros(6, dtype=torch.long)
        bands.forces = torch.tensor(
            [
                [0.1, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.3, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.6, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ]
        )
        regular = engine.sub_stages[0][1]
        migration = next(
            hook
            for hook in regular.hooks
            if isinstance(hook, ConvergenceHook)
            and hook.source_status == 0
            and hook.target_status == 1
        )

        migration(DynamicsContext(batch=bands), DynamicsStage.AFTER_STEP)

        torch.testing.assert_close(
            bands.status,
            torch.tensor([1, 1, 1, 0, 0, 0]),
        )

    def test_immediate_builds_one_climbing_stage(self) -> None:
        engine = NEB(
            model=_model(),
            climbing=ClimbingImageConfig(
                mode="immediate",
                max_climbing_steps=7,
            ),
        ).build_engine()

        assert len(engine.sub_stages) == 1
        assert engine.sub_stages[0][1].n_steps == 7
        selection = next(
            hook
            for hook in engine.hooks
            if isinstance(hook, ClimbingImageSelectionHook)
        )
        assert selection.selection == "fixed"
        assert selection.status_code == 0


# ---------------------------------------------------------------------------
# NEB run
# ---------------------------------------------------------------------------


class TestNEBRun:
    """Exercise the public run entry point on grouped path batches."""

    def test_shared_path_hooks_match_per_stage_hooks(self) -> None:
        """Shared status-gated path hooks match per-stage hook pipelines."""
        strategy = NEB(
            model=_CompilerFriendlyModel().eval(),
            fmax=1.0e-12,
            climbing=ClimbingImageConfig(
                max_regular_steps=1,
                max_climbing_steps=2,
            ),
        )
        shared_engine: FusedStage = strategy.build_engine()
        per_stage_engine: FusedStage = strategy.build_engine()

        per_stage_engine.hooks = [
            hook
            for hook in per_stage_engine.hooks
            if not isinstance(
                hook,
                (
                    PathEnergyStatsHook,
                    ClimbingImageSelectionHook,
                    NEBForceHook,
                ),
            )
        ]

        regular_energy_stats = PathEnergyStatsHook()
        regular_stage = per_stage_engine.sub_stages[0][1]
        regular_stage.register_hook(regular_energy_stats)
        regular_stage.register_hook(
            NEBForceHook(
                energy_stats_hook=regular_energy_stats,
                spring=strategy.spring,
                method=strategy.method,
                endpoint_mode=strategy.endpoint_mode,
                fixed_atom_indices=strategy.fixed_atom_indices,
            )
        )

        climbing_energy_stats = PathEnergyStatsHook()
        climbing_stage = per_stage_engine.sub_stages[1][1]
        climbing_stage.register_hook(climbing_energy_stats)
        climbing_stage.register_hook(
            ClimbingImageSelectionHook(energy_stats_hook=climbing_energy_stats)
        )
        climbing_stage.register_hook(
            NEBForceHook(
                energy_stats_hook=climbing_energy_stats,
                spring=strategy.spring,
                method=strategy.method,
                endpoint_mode=strategy.endpoint_mode,
                fixed_atom_indices=strategy.fixed_atom_indices,
            )
        )

        shared_batch = _bands()
        per_stage_batch = _bands()
        shared_batch.positions[1, 1] = 0.5
        per_stage_batch.positions[1, 1] = 0.5

        compared_fields = (
            "status",
            "positions",
            "velocities",
            "energy",
            "forces",
            "physical_forces",
            "force_mode",
            "forward_link_length",
            "reprime_pending",
            "n_steps_counter_0",
            "n_steps_counter_1",
        )
        for _ in range(3):
            shared_engine.step(shared_batch)
            per_stage_engine.step(per_stage_batch)
            for field in compared_fields:
                torch.testing.assert_close(
                    getattr(shared_batch, field), getattr(per_stage_batch, field)
                )

        assert shared_batch.status.unique().tolist() == [1]
        assert not shared_batch.reprime_pending.any()

    def test_compile_executes_strategy(self) -> None:
        torch.compiler.reset()
        try:
            model = _CompilerFriendlyModel().eval()
            result = NEB(
                model=model,
                fmax=1.0e9,
                n_steps=2,
                climbing=ClimbingImageConfig(mode="immediate"),
                compile=True,
                compile_kwargs={"backend": "eager"},
            ).run(_bands())
        finally:
            torch.compiler.reset()

        assert torch.all(result.status == 1)

    def test_diagnostics_log_path_executes_hooks_and_writes_csv(
        self, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "neb.csv"
        engine = NEB(
            model=_model(),
            fmax=1.0e9,
            n_steps=2,
            diagnostics_log_path=log_path,
        ).build_engine()
        diagnostics_hook = next(
            hook for hook in engine.hooks if isinstance(hook, PathDiagnosticsHook)
        )

        engine.run(_bands())
        diagnostics = diagnostics_hook.get_diagnostics()

        assert torch.isfinite(diagnostics.fmax).all()
        assert torch.isfinite(diagnostics.energy_barrier).all()
        assert torch.isfinite(diagnostics.path_length).all()
        assert torch.all(diagnostics.highest_interior_image_idx >= 0)

        with log_path.open(newline="") as csv_file:
            rows = list(csv.DictReader(csv_file))
        assert len(rows) == 1
        assert set(rows[0]) == {
            "step",
            "group_idx",
            "status",
            "fmax",
            "energy_barrier",
            "highest_interior_image_idx",
            "path_length",
        }
        assert float(rows[0]["step"]) == 0.0
        assert float(rows[0]["status"]) == 1.0
        assert float(rows[0]["highest_interior_image_idx"]) == 1.0
        assert all(
            torch.isfinite(torch.tensor(float(rows[0][name])))
            for name in ("fmax", "energy_barrier", "path_length")
        )

    @pytest.mark.parametrize(
        ("climbing", "exit_status"),
        [
            (None, 1),
            (ClimbingImageConfig(mode="immediate"), 1),
            (ClimbingImageConfig(), 2),
        ],
    )
    def test_run_returns_completed_input_batch(
        self,
        climbing: ClimbingImageConfig | None,
        exit_status: int,
    ) -> None:
        bands = _bands()
        result = NEB(
            model=_model(),
            fmax=1.0e9,
            climbing=climbing,
            n_steps=5,
        ).run(bands)

        assert result is bands
        assert torch.all(result.status == exit_status)
        assert "physical_forces" in result
        assert "force_mode" in result


# ---------------------------------------------------------------------------
# NEB with IDPP
# ---------------------------------------------------------------------------


class TestNEBwithIDPP:
    """Exercise NEB optimization with the analytic IDPP model."""

    def test_runs_on_prepared_idpp_path(self) -> None:
        """NEB consumes prepared IDPP energies and forces end to end."""
        atomic_numbers = torch.tensor([1, 1, 1])
        initial = Batch.from_data_list(
            [
                AtomicData(
                    atomic_numbers=atomic_numbers,
                    positions=torch.tensor(
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
                    ),
                )
            ]
        )
        final = Batch.from_data_list(
            [
                AtomicData(
                    atomic_numbers=atomic_numbers,
                    positions=torch.tensor(
                        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
                    ),
                )
            ]
        )
        paths = prepare_idpp_targets(interpolate_paths(initial, final, 5))
        model = IDPPModel()
        outputs = model(paths)
        paths.energy = outputs["energy"]
        paths.forces = outputs["forces"]
        paths.velocities = torch.zeros_like(paths.positions)
        assert torch.any(paths.energy[1:-1] > 0)

        endpoint_mask = (paths.batch_idx == 0) | (paths.batch_idx == 4)
        endpoint_positions = paths.positions[endpoint_mask].clone()
        neighbor_list = paths.neighbor_list.clone()
        target_distances = paths.idpp_target_distances.clone()

        assert paths.num_edges_per_graph.tolist() == [3] * 5
        assert model.make_neighbor_hooks() == []

        result = NEB(model=model, fmax=1.0e9, n_steps=2).run(paths)

        assert result is paths
        assert torch.all(result.status == 1)
        assert torch.isfinite(result.energy).all()
        assert torch.isfinite(result.physical_forces).all()
        torch.testing.assert_close(result.neighbor_list, neighbor_list)
        torch.testing.assert_close(result.idpp_target_distances, target_distances)
        torch.testing.assert_close(result.positions[endpoint_mask], endpoint_positions)
