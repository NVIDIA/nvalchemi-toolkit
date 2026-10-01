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
from typing import Any
from unittest.mock import patch

import pytest
import torch
import warp as wp

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    ConvergenceHook,
    DynamicsStage,
    FusedStage,
)
from nvalchemi.dynamics.base import BaseDynamics
from nvalchemi.dynamics.hooks import FreezeAtomsHook, LoggingHook
from nvalchemi.dynamics.mep import (
    NEB,
    ClimbingImageConfig,
    ConstantSpringConfig,
    IDPPModel,
    NEBMethod,
    SpringConfig,
    SpringContext,
    TorchNEBMethod,
    interpolate_paths,
    prepare_idpp_targets,
)
from nvalchemi.dynamics.mep._geometry import PreparedMIC
from nvalchemi.dynamics.mep.hooks import (
    ClimbingImageSelectionHook,
    NEBForceHook,
    PathDiagnosticsHook,
    PathEnergyStatsHook,
)
from nvalchemi.dynamics.mep.neb import _NEB_FIRE2_DEFAULTS
from nvalchemi.dynamics.mep.neb_equations import (
    neb_effective_force,
    neb_effective_force_from_gram_stats,
)
from nvalchemi.hooks import DynamicsContext, NeighborListHook
from nvalchemi.models.base import BaseModelMixin, ModelConfig, NeighborConfig
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@wp.func
def _custom_tangent_weights(energy_prev: Any, energy_curr: Any, energy_next: Any):
    """Provide an importable custom equation for spec round-trip tests."""
    one = type(energy_curr)(1.0)
    return one, one


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


class _CustomOptimizer(BaseDynamics):
    """Minimal optimizer for configuration and serialization tests."""

    __needs_keys__ = {"forces"}
    __provides_keys__ = {"positions"}

    def __init__(
        self, model: BaseModelMixin, learning_rate: float = 0.01, **kwargs: Any
    ) -> None:
        """Forward shared stage settings and store the custom step size."""
        super().__init__(model=model, **kwargs)
        self.learning_rate = learning_rate

    def pre_update(self, batch: Batch) -> None:
        """Provide the required pre-update interface for construction tests."""

    def post_update(self, batch: Batch) -> None:
        """Leave positions unchanged after force evaluation."""


def _model(device: str = "cpu") -> DemoModelWrapper:
    """Return a lightweight model for strategy construction and execution."""
    return DemoModelWrapper(DemoModel()).to(device).eval()


def _bands(device: str = "cpu") -> Batch:
    """Return one valid three-image path with optimizer state fields."""
    images = []
    for x in (0.0, 0.5, 1.0):
        image = AtomicData(
            atomic_numbers=torch.tensor([1], device=device),
            positions=torch.tensor([[x, 0.0, 0.0]], device=device),
            energy=torch.zeros(1, 1, device=device),
            forces=torch.zeros(1, 3, device=device),
        )
        image.add_node_property("velocities", torch.zeros(1, 3, device=device))
        images.append(image)
    bands = Batch.from_data_list(images)
    bands.set_group_layout(torch.zeros(3, dtype=torch.long, device=device))
    return bands


def _force_hook(engine: FusedStage) -> NEBForceHook:
    """Return the shared NEB force hook owned by the fused engine."""
    return next(hook for hook in engine.hooks if isinstance(hook, NEBForceHook))


class _ScaleTorchMethod:
    """Simple importable Torch method for strategy spec tests."""

    def __init__(self, factor: float = 1.0) -> None:
        self.factor = factor

    def __call__(
        self,
        batch: Batch,
        *,
        spring_constants: torch.Tensor,
        path_energy_ref: torch.Tensor,
        path_energy_max: torch.Tensor,
        mic: PreparedMIC,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scale physical forces and return one length per forward link."""
        del path_energy_ref, path_energy_max, mic
        return self.factor * batch.physical_forces, torch.ones_like(spring_constants)


class _EnergyScaledSpring:
    """Importable spring with reconstructible energy-dependent parameters."""

    refresh = DynamicsStage.AFTER_COMPUTE

    def __init__(self, base: float, scale: float) -> None:
        self.base = base
        self.scale = scale

    def resolve(self, context: SpringContext) -> torch.Tensor:
        """Return a spring constant for each link from the current energies."""
        energy_scale = (
            0.0 if context.energies is None else float(context.energies.max())
        )
        return torch.full(
            (context.num_links,),
            self.base + self.scale * energy_scale,
            dtype=context.positions.dtype,
            device=context.positions.device,
        )


def _freeze_hook(engine: FusedStage) -> FreezeAtomsHook:
    """Return the NEB-owned fixed-node constraint hook."""
    return next(
        hook
        for hook in engine.hooks
        if isinstance(hook, FreezeAtomsHook) and hook.mask_key == "neb_fixed_node_mask"
    )


# ---------------------------------------------------------------------------
# NEB configuration
# ---------------------------------------------------------------------------


class TestNEBConfiguration:
    """Validate strategy configuration and optimizer construction."""

    def test_custom_spring_round_trips_constructor_spec(self) -> None:
        """A custom spring keeps its constructor state and refresh policy."""
        spring = _EnergyScaledSpring(base=0.2, scale=0.05)
        strategy = NEB(model=_model(), spring=spring)

        assert isinstance(strategy.spring, SpringConfig)
        assert _force_hook(strategy.build_engine()).spring is spring

        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        restored = NEB.from_spec_dict(spec, model=strategy.model)

        assert spec["spring"]["type"] == "custom"
        assert isinstance(restored.spring, _EnergyScaledSpring)
        assert restored.spring.base == 0.2
        assert restored.spring.scale == 0.05
        assert restored.spring.refresh is DynamicsStage.AFTER_COMPUTE
        assert _force_hook(restored.build_engine()).spring is restored.spring

    def test_rejects_invalid_custom_spring_specs(self) -> None:
        """Malformed recipes and components outside the protocol fail clearly."""
        strategy = NEB(model=_model())
        spec = strategy.to_spec_dict()

        spec["spring"] = {"type": "custom"}
        with pytest.raises(ValueError, match="custom spring spec must contain"):
            NEB.from_spec_dict(spec, model=strategy.model)

        spec["spring"] = {"type": "custom", "spec": None}
        with pytest.raises(ValueError, match="constructor-spec mapping"):
            NEB.from_spec_dict(spec, model=strategy.model)

        method_spec = NEB(
            model=strategy.model, method=_ScaleTorchMethod()
        ).to_spec_dict()
        spec["spring"] = {"type": "custom", "spec": method_spec["method"]["spec"]}
        with pytest.raises(TypeError, match="must build a SpringConfig"):
            NEB.from_spec_dict(spec, model=strategy.model)

    def test_runtime_only_spring_rejects_serialization(self) -> None:
        """A local spring can run but cannot produce an importable recipe."""

        class LocalSpring:
            refresh = DynamicsStage.ON_ADMISSION

            def resolve(self, context: SpringContext) -> torch.Tensor:
                return torch.ones(context.num_links, device=context.positions.device)

        strategy = NEB(model=_model(), spring=LocalSpring())

        with pytest.raises(ValueError, match="Cannot serialize custom spring"):
            strategy.to_spec_dict()

    def test_torch_method_round_trips_constructor_spec(self) -> None:
        """The API accepts a Torch method and restores its constructor state."""
        method = _ScaleTorchMethod(factor=2.0)
        strategy = NEB(model=_model(), method=method)

        assert isinstance(strategy.method, TorchNEBMethod)
        assert _force_hook(strategy.build_engine()).method is method

        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        restored = NEB.from_spec_dict(spec, model=strategy.model)

        assert spec["method"]["type"] == "torch"
        assert isinstance(restored.method, _ScaleTorchMethod)
        assert restored.method.factor == 2.0
        assert _force_hook(restored.build_engine()).method_key is None

    @pytest.mark.parametrize(
        ("effective_force_fn", "kernel_kind"),
        [
            (neb_effective_force, "stored_tangent"),
            (neb_effective_force_from_gram_stats, "gram_stats"),
        ],
    )
    def test_custom_method_spec_restores_equations(
        self, effective_force_fn: wp.Function, kernel_kind: str
    ) -> None:
        method = NEBMethod(
            tangent_weights_fn=_custom_tangent_weights,
            effective_force_fn=effective_force_fn,
        )
        strategy = NEB(model=_model(), method=method)

        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        key = spec["method"]
        restored = NEB.from_spec_dict(spec, model=strategy.model)

        assert key == method.to_key()
        assert key.startswith(f"{kernel_kind}|")
        assert "test_neb._custom_tangent_weights" in key
        assert isinstance(restored.method, NEBMethod)
        assert restored.method.tangent_weights_fn is _custom_tangent_weights
        assert restored.method.effective_force_fn is effective_force_fn
        assert restored.method.climbing_force_fn is method.climbing_force_fn
        assert _force_hook(restored.build_engine()).method_key == key

    def test_rejects_missing_custom_equation_in_spec(self) -> None:
        method = NEBMethod(tangent_weights_fn=_custom_tangent_weights)
        strategy = NEB(model=_model(), method=method)
        spec = strategy.to_spec_dict()
        spec["method"] = spec["method"].replace(
            "._custom_tangent_weights", ".missing_tangent_weights"
        )

        with pytest.raises(ValueError, match="not importable"):
            NEB.from_spec_dict(spec, model=strategy.model)

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
            diagnostics_frequency=7,
        )

        spec = json.loads(json.dumps(strategy.to_spec_dict()))

        assert spec["optimizer"] == "nvalchemi.dynamics.optimizers.fire2.FIRE2"
        assert spec["climbing"]["regular_fmax"] == 0.5
        assert spec["spring"] == {"type": "constant", "value": 0.2}
        assert spec["method"] == "improved_tangent"

        restored = NEB.from_spec_dict(spec, model=strategy.model)

        assert restored.n_steps == 19
        assert restored.spring == ConstantSpringConfig(0.2)
        assert restored.spring.refresh is DynamicsStage.ON_ADMISSION
        assert restored.climbing == ClimbingImageConfig(
            regular_fmax=0.5,
            max_regular_steps=11,
        )
        assert restored.fixed_atom_indices == {0: (0, 2), 1: (1,)}
        assert restored.diagnostics_log_path == Path("neb.csv")
        assert restored.diagnostics_frequency == 7

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

    def test_builds_mask_aware_freeze_hook_for_constraints(self) -> None:
        """Use the NEB-owned node mask for endpoint and user constraints."""
        engine = NEB(
            model=_model(),
            fixed_atom_indices={0: [0]},
        ).build_engine()

        freeze_hook = _freeze_hook(engine)
        assert freeze_hook.mask_key == "neb_fixed_node_mask"
        assert freeze_hook.zero_velocities

    def test_relaxed_unconstrained_neb_omits_freeze_hook(self) -> None:
        """Avoid constraint lifecycle overhead when no nodes are fixed."""
        engine = NEB(
            model=_model(),
            endpoint_mode="relaxed",
        ).build_engine()

        assert not any(
            isinstance(hook, FreezeAtomsHook) and hook.mask_key == "neb_fixed_node_mask"
            for hook in engine.hooks
        )

    def test_fire2_explicit_kwargs_override_neb_defaults(self) -> None:
        kwargs = {"dt": 0.02, "maxstep": 0.03}
        strategy = NEB(
            model=_model(),
            optimizer_kwargs=kwargs,
        )

        assert strategy.optimizer_kwargs["dt"] == 0.02
        assert strategy.optimizer_kwargs["maxstep"] == 0.03
        assert strategy.optimizer_kwargs["dt"] != _NEB_FIRE2_DEFAULTS["dt"]
        assert strategy.optimizer_kwargs["maxstep"] != _NEB_FIRE2_DEFAULTS["maxstep"]
        stage = strategy.build_engine().sub_stages[0][1]
        assert stage._dt_init == 0.02
        assert stage.maxstep == 0.03
        assert stage.delaystep == _NEB_FIRE2_DEFAULTS["delaystep"]
        assert (
            strategy.optimizer_kwargs["delaystep"] == _NEB_FIRE2_DEFAULTS["delaystep"]
        )
        assert kwargs == {"dt": 0.02, "maxstep": 0.03}

    @pytest.mark.parametrize("kwargs", [{}, {"learning_rate": 0.02}])
    def test_custom_optimizer_round_trips_and_receives_kwargs(
        self, kwargs: dict[str, float]
    ) -> None:
        """Custom optimizer classes and parameters survive JSON serialization."""
        strategy = NEB(
            model=_model(),
            optimizer=_CustomOptimizer,
            optimizer_kwargs=kwargs,
        )
        restored = NEB.from_spec_dict(
            json.loads(json.dumps(strategy.to_spec_dict())), model=strategy.model
        )
        engine = restored.build_engine()
        stage = engine.sub_stages[0][1]
        assert isinstance(stage, _CustomOptimizer)
        assert stage.learning_rate == kwargs.get("learning_rate", 0.01)
        assert restored.optimizer_kwargs == kwargs
        assert stage.by_group
        assert not _freeze_hook(engine).zero_velocities

    def test_rejects_optimizer_outside_dynamics_interface(self) -> None:
        """Unrelated classes fail before constructing an engine."""
        with pytest.raises(TypeError, match="BaseDynamics subclass"):
            NEB(model=_model(), optimizer=object)

    def test_default_neighbor_hooks_are_generated_by_model(self) -> None:
        generated = _NoOpHook()
        model = _model()

        with patch.object(
            DemoModelWrapper,
            "make_neighbor_hooks",
            return_value=[generated],
        ) as make_neighbor_hooks:
            engine = NEB(model=model).build_engine()

        make_neighbor_hooks.assert_called_once_with()
        assert engine.hooks[0] is generated

    @pytest.mark.parametrize("neighbor_hooks", [[], [_NoOpHook()]])
    def test_explicit_neighbor_hooks_replace_generated_hooks(
        self, neighbor_hooks: list[_NoOpHook]
    ) -> None:
        model = _model()
        with patch.object(
            DemoModelWrapper,
            "make_neighbor_hooks",
            side_effect=AssertionError("generated hooks must not be used"),
        ) as make_neighbor_hooks:
            engine = NEB(model=model, neighbor_hooks=neighbor_hooks).build_engine()

        make_neighbor_hooks.assert_not_called()
        assert engine.hooks[: len(neighbor_hooks)] == neighbor_hooks

    def test_rejects_invalid_neighbor_hooks(self) -> None:
        with pytest.raises(TypeError, match="neighbor_hooks"):
            NEB(model=_model(), neighbor_hooks=[object()])

    def test_empty_neighbor_hooks_round_trip_preserves_disabled_mode(self) -> None:
        strategy = NEB(model=_model(), neighbor_hooks=[])

        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        restored = NEB.from_spec_dict(spec, model=strategy.model)

        assert restored.neighbor_hooks == []

    def test_rejects_invalid_neighbor_hook_specs(self) -> None:
        spec = NEB(model=_model()).to_spec_dict()
        spec["neighbor_hook_specs"] = {}

        with pytest.raises(ValueError, match="neighbor_hook_specs"):
            NEB.from_spec_dict(spec, model=_model())

    def test_configured_neighbor_list_hook_round_trips(self) -> None:
        hook = NeighborListHook(
            NeighborConfig(cutoff=3.0),
            skin=0.2,
            method="batch_naive_tile",
        )
        strategy = NEB(model=_model(), neighbor_hooks=[hook])

        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        restored = NEB.from_spec_dict(spec, model=strategy.model)

        assert restored.neighbor_hooks is not None
        restored_hook = restored.neighbor_hooks[0]
        assert isinstance(restored_hook, NeighborListHook)
        assert restored_hook.skin == 0.2
        assert restored_hook.method == "batch_naive_tile"

    def test_diagnostics_log_path_builds_fresh_ordered_hooks(
        self, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "neb.csv"
        strategy = NEB(
            model=_model(),
            diagnostics_log_path=log_path,
            diagnostics_frequency=5,
        )

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
        assert first_diagnostics.frequency == 5
        assert first_logger.frequency == first_diagnostics.frequency
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
        final = ConvergenceHook(
            criteria={"key": "energy", "threshold": 0.2},
            by_group=True,
        )
        regular = ConvergenceHook.from_fmax(0.5, by_group=True)
        strategy = NEB(
            model=_model(),
            climbing=ClimbingImageConfig(),
            convergence_hook=final,
            regular_convergence_hook=regular,
        )

        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        assert "convergence_hook" not in spec
        assert "regular_convergence_hook" not in spec
        restored = NEB.from_spec_dict(spec, model=strategy.model)

        assert restored.convergence_hook is not None
        assert restored.convergence_hook.criteria[0].key == "energy"
        assert restored.regular_convergence_hook is not None
        assert restored.regular_convergence_hook.criteria[0].threshold == 0.5

    def test_regular_convergence_hook_overrides_preliminary_stage(self) -> None:
        final = ConvergenceHook(
            criteria={"key": "energy", "threshold": 0.1},
            by_group=True,
        )
        regular = ConvergenceHook.from_fmax(0.5, by_group=True)
        strategy = NEB(
            model=_model(),
            climbing=ClimbingImageConfig(),
            convergence_hook=final,
            regular_convergence_hook=regular,
        )
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
        strategy.regular_convergence_hook = template
        strategy.convergence_hook = template

        engine = strategy.build_engine()
        regular_hook = engine.sub_stages[0][1].convergence_hook
        final_hook = engine.sub_stages[1][1].convergence_hook

        assert regular_hook is not template
        assert final_hook is not template
        assert regular_hook is not final_hook

    def test_regular_neb_builds_one_grouped_stage(self, device: str) -> None:
        spring = ConstantSpringConfig(0.2)
        engine = NEB(
            model=_model(device),
            spring=spring,
            fmax=0.05,
            n_steps=17,
            optimizer_kwargs={"dt": 0.02, "maxstep": 0.03, "device_type": device},
        ).build_engine()

        assert isinstance(engine, FusedStage)
        assert engine.by_group is True
        assert engine.device_type == device
        assert engine.n_steps == 17
        assert engine.reprime_on_entry == frozenset()
        assert len(engine.sub_stages) == 1
        stage = engine.sub_stages[0][1]
        assert stage.by_group is True
        assert stage.device_type == device
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

    def test_after_regular_builds_two_stage_strategy(self, device: str) -> None:
        engine = NEB(
            model=_model(device),
            fmax=0.05,
            n_steps=500,
            climbing=ClimbingImageConfig(
                selection="dynamic",
                regular_fmax=0.5,
                max_regular_steps=11,
                max_climbing_steps=13,
            ),
            optimizer_kwargs={"device_type": device},
        ).build_engine()

        assert engine.device_type == device
        assert len(engine.sub_stages) == 2
        assert engine.n_steps == 500
        assert engine.reprime_on_entry == frozenset({1})
        regular = engine.sub_stages[0][1]
        climbing = engine.sub_stages[1][1]
        assert regular.n_steps == 11
        assert climbing.n_steps == 13
        assert regular.device_type == device
        assert climbing.device_type == device
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

    def test_neb_force_hook_matches_as_fused_or_substage_hook(
        self, device: str
    ) -> None:
        """NEB forces match for fused-level and substage-level registration."""
        strategy = NEB(
            model=_CompilerFriendlyModel().to(device).eval(),
            fmax=1.0e-12,
            climbing=ClimbingImageConfig(
                max_regular_steps=1,
                max_climbing_steps=2,
            ),
            optimizer_kwargs={"device_type": device},
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

        assert any(isinstance(hook, NEBForceHook) for hook in shared_engine.hooks)
        assert not any(
            isinstance(hook, NEBForceHook) for hook in per_stage_engine.hooks
        )
        assert all(
            any(isinstance(hook, NEBForceHook) for hook in stage.hooks)
            for _, stage in per_stage_engine.sub_stages
        )

        shared_batch = _bands(device)
        per_stage_batch = _bands(device)
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
            "neb_fixed_node_mask",
            "reprime_pending",
            "n_steps_counter_0",
            "n_steps_counter_1",
        )
        for _ in range(3):
            shared_engine.step(shared_batch)
            per_stage_engine.step(per_stage_batch)
            for field in compared_fields:
                torch.testing.assert_close(
                    getattr(shared_batch, field),
                    getattr(per_stage_batch, field),
                    msg=lambda message: f"{field} differs: {message}",
                )

        assert shared_batch.status.unique().tolist() == [1]
        assert not shared_batch.reprime_pending.any()

    def test_shared_path_hooks_match_standalone_dynamics(self, device: str) -> None:
        """Shared path hooks match direct use on a standalone optimizer."""
        strategy = NEB(
            model=_CompilerFriendlyModel().to(device).eval(),
            fmax=1.0e-12,
            optimizer_kwargs={"device_type": device},
        )
        shared_engine = strategy.build_engine()
        standalone = strategy.optimizer(
            model=strategy.model,
            n_steps=None,
            by_group=True,
            hooks=[
                *strategy.model.make_neighbor_hooks(),
                *strategy._build_path_hooks(climbing_status=None),
                *strategy.extra_hooks,
            ],
            convergence_hook=strategy._build_convergence_hook(
                fmax=strategy.fmax,
                regular_stage=False,
            ),
            **strategy.optimizer_kwargs,
        )

        shared_batch = _bands(device)
        standalone_batch = _bands(device)
        shared_batch.positions[1, 1] = 0.5
        standalone_batch.positions[1, 1] = 0.5
        shared_batch.status = torch.zeros(
            shared_batch.num_graphs, dtype=torch.long, device=device
        )
        standalone_batch.status = torch.zeros(
            standalone_batch.num_graphs, dtype=torch.long, device=device
        )

        compared_fields = (
            "positions",
            "velocities",
            "energy",
            "forces",
            "physical_forces",
            "force_mode",
            "forward_link_length",
        )
        for _ in range(2):
            shared_engine.step(shared_batch)
            standalone.step(standalone_batch)
            for field in compared_fields:
                torch.testing.assert_close(
                    getattr(shared_batch, field),
                    getattr(standalone_batch, field),
                )

    def test_compile_executes_strategy(self, device: str) -> None:
        torch.compiler.reset()
        try:
            model = _CompilerFriendlyModel().to(device).eval()
            result = NEB(
                model=model,
                fmax=1.0e9,
                n_steps=2,
                climbing=ClimbingImageConfig(mode="immediate"),
                compile=True,
                compile_kwargs={"backend": "eager"},
                optimizer_kwargs={"device_type": device},
            ).run(_bands(device))
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
        device: str,
    ) -> None:
        bands = _bands(device)
        result = NEB(
            model=_model(device),
            fmax=1.0e9,
            climbing=climbing,
            n_steps=5,
            optimizer_kwargs={"device_type": device},
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

    def test_runs_on_prepared_idpp_path(self, device: str) -> None:
        """NEB consumes prepared IDPP energies and forces end to end."""
        atomic_numbers = torch.tensor([1, 1, 1], device=device)
        initial = Batch.from_data_list(
            [
                AtomicData(
                    atomic_numbers=atomic_numbers,
                    positions=torch.tensor(
                        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                        device=device,
                    ),
                )
            ]
        )
        final = Batch.from_data_list(
            [
                AtomicData(
                    atomic_numbers=atomic_numbers,
                    positions=torch.tensor(
                        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
                        device=device,
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

        result = NEB(
            model=model,
            fmax=1.0e9,
            n_steps=2,
            optimizer_kwargs={"device_type": device},
        ).run(paths)

        assert result is paths
        assert torch.all(result.status == 1)
        assert torch.isfinite(result.energy).all()
        assert torch.isfinite(result.physical_forces).all()
        torch.testing.assert_close(result.neighbor_list, neighbor_list)
        torch.testing.assert_close(result.idpp_target_distances, target_distances)
        torch.testing.assert_close(result.positions[endpoint_mask], endpoint_positions)
