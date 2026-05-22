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
"""Tests for TrainingStrategy, OptimizerConfig, and loop helpers."""

from __future__ import annotations

import json
import operator
from collections.abc import Callable, Mapping
from enum import Enum
from typing import Any

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks._context import HookContext, TrainContext
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import (
    ComposedLossFunction,
    EnergyLoss,
    ForceLoss,
    LinearWeight,
    TrainingStage,
)
from nvalchemi.training.optimizers import OptimizerConfig
from nvalchemi.training.strategy import TrainingStrategy, default_training_fn


def demo_training_fn(model: BaseModelMixin, batch: Batch) -> dict[str, torch.Tensor]:
    """Training step: forward pass producing ``predicted_energy`` + ``predicted_forces``.

    Module-level so it can round-trip through
    :meth:`TrainingStrategy.to_spec_dict` (lambdas and nested functions are
    rejected by the serializer).
    """
    return default_training_fn(model, batch)


def dict_demo_training_fn(
    models: dict[str, BaseModelMixin], batch: Batch
) -> dict[str, torch.Tensor]:
    """Distillation-style dict-model training function using all named models."""
    student = demo_training_fn(models["student"], batch)
    teacher = demo_training_fn(models["teacher"], batch)
    assert set(models) == {"student", "teacher"}
    return {
        "predicted_energy": student["predicted_energy"],
        "predicted_forces": teacher["predicted_forces"],
    }


def mapping_annotated_training_fn(
    models: Mapping[str, BaseModelMixin], batch: Batch
) -> dict[str, torch.Tensor]:
    """Mapping-annotated training function for validation tests."""
    return demo_training_fn(models["main"], batch)


def moduledict_annotated_training_fn(
    models: torch.nn.ModuleDict, batch: Batch
) -> dict[str, torch.Tensor]:
    """ModuleDict-annotated training function for validation tests."""
    return demo_training_fn(models["main"], batch)


def single_model_training_fn(
    model: BaseModelMixin, batch: Batch
) -> dict[str, torch.Tensor]:
    """Single-model training function for validation tests."""
    return demo_training_fn(model, batch)


def _make_atomic_data(n_atoms: int = 3, seed: int = 0) -> AtomicData:
    g = torch.Generator().manual_seed(seed)
    positions = torch.randn(n_atoms, 3, generator=g)
    atomic_numbers = torch.randint(1, 10, (n_atoms,), dtype=torch.long, generator=g)
    energy = torch.randn(1, 1, generator=g)
    forces = torch.randn(n_atoms, 3, generator=g)
    return AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        atomic_masses=torch.ones(n_atoms),
        energy=energy,
        forces=forces,
    )


def _make_batch(n_systems: int = 2, n_atoms_each: int = 3, seed: int = 0) -> Batch:
    data_list = [
        _make_atomic_data(n_atoms_each, seed=seed + i) for i in range(n_systems)
    ]
    return Batch.from_data_list(data_list)


def _make_dataset(
    n_batches: int = 3,
    n_systems: int = 2,
    n_atoms_each: int = 3,
    base_seed: int = 100,
) -> list[Batch]:
    return [
        _make_batch(
            n_systems=n_systems,
            n_atoms_each=n_atoms_each,
            seed=base_seed + i * 10,
        )
        for i in range(n_batches)
    ]


def _make_demo_model() -> Any:
    from nvalchemi.models.demo import DemoModel, DemoModelWrapper

    torch.manual_seed(0)
    return DemoModelWrapper(DemoModel(num_atom_types=20, hidden_dim=8))


def _adam_optimizer_configs(
    lr: float = 1e-3,
) -> dict[str, list[OptimizerConfig]]:
    return {
        "main": [
            OptimizerConfig(
                optimizer_cls=torch.optim.Adam,
                optimizer_kwargs={"lr": lr},
            )
        ]
    }


def _baseline_strategy_kwargs(
    models: BaseModelMixin | dict[str, BaseModelMixin] | None = None,
) -> dict[str, Any]:
    if models is None:
        models = _make_demo_model()
    return {
        "models": models,
        "optimizer_configs": OptimizerConfig(optimizer_cls=torch.optim.Adam),
        "num_epochs": 1,
        "training_fn": demo_training_fn,
        "loss_fn": EnergyLoss() + ForceLoss(normalize_by_atom_count=True),
    }


def _make_strategy(**overrides: Any) -> TrainingStrategy:
    kwargs = _baseline_strategy_kwargs()
    kwargs.update(overrides)
    return TrainingStrategy(**kwargs)


class _RecordingHook:
    """Hook object tagged with ``stage``; forwards ``(ctx, stage)`` to ``callback``.

    Stage filtering is done by the hook runner via ``self.stage``; this
    helper just forwards. Recording runs on CPU — callbacks that convert
    tensors via ``float(...)`` are not safe for GPU tensors without an
    explicit ``.cpu()``.
    """

    def __init__(
        self,
        stage: Enum,
        callback: Callable[[HookContext, Enum], None],
    ) -> None:
        self.stage = stage
        self.frequency = 1
        self._callback = callback

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        self._callback(ctx, stage)


_VALIDATOR_REJECTION_CASES: list[tuple[str, dict[str, Any]]] = [
    (
        "models must contain at least one BaseModelMixin",
        {"models": {}, "optimizer_configs": {}},
    ),
    (
        "optimizer_configs must configure at least one model",
        {"optimizer_configs": {}},
    ),
    (
        r"optimizer_configs\['main'\] must contain",
        {"optimizer_configs": {"main": []}},
    ),
    (
        "models must map names",
        {"models": {"main": torch.nn.Linear(1, 1)}, "optimizer_configs": {}},
    ),
    (
        "not present in models",
        {
            "optimizer_configs": {
                "missing": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
            }
        },
    ),
    (
        "devices must have length",
        {"devices": [torch.device("cpu"), torch.device("cpu")]},
    ),
    (
        "devices must contain at least one torch.device",
        {"devices": []},
    ),
    (
        "Exactly one of num_epochs or num_steps",
        {"num_epochs": 1, "num_steps": 1},
    ),
    (
        "Exactly one of num_epochs or num_steps",
        {"num_epochs": None, "num_steps": None},
    ),
    ("greater than or equal to 1", {"num_epochs": -1}),
    ("greater than or equal to 1", {"num_steps": -1, "num_epochs": None}),
    (
        "no attribute",
        {"training_fn": "nvalchemi.training.strategy.not_a_real_fn"},
    ),
]

_DELETE = object()

_FROM_SPEC_REJECTION_CASES: list[tuple[str, Any, str]] = [
    ("optimizer_configs", [], "optimizer_configs"),
    ("optimizer_configs", {"main": [1]}, "optimizer_configs"),
    ("devices", "cpu", "devices"),
    ("loss_fn_spec", [], "loss_fn_spec"),
    ("model_specs", [], "model_specs"),
    ("training_fn", _DELETE, "no training_fn"),
    ("training_fn", 123, "training_fn"),
    ("single_model_input", "yes", "single_model_input"),
]


class TestTrainingStrategyValidators:
    @pytest.mark.parametrize(
        ("match", "overrides"),
        _VALIDATOR_REJECTION_CASES,
        ids=[
            "empty_models",
            "empty_optimizer_configs",
            "empty_per_model_list",
            "invalid_model_value",
            "optimizer_key_missing",
            "devices_wrong_length",
            "devices_empty",
            "both_num_epochs_and_num_steps",
            "neither_num_epochs_nor_num_steps",
            "negative_num_epochs",
            "negative_num_steps",
            "training_fn_bad_dotted_path",
        ],
    )
    def test_construction_rejected(self, match: str, overrides: dict[str, Any]) -> None:
        kwargs = _baseline_strategy_kwargs()
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=match):
            TrainingStrategy(**kwargs)

    def test_training_fn_dotted_string_resolved(self) -> None:
        strat = _make_strategy(training_fn="operator.add")
        assert strat.training_fn is operator.add

    def test_training_fn_required_message_suggests_default(self) -> None:
        kwargs = _baseline_strategy_kwargs()
        del kwargs["training_fn"]
        with pytest.raises(ValueError, match="default_training_fn"):
            TrainingStrategy(**kwargs)

    def test_leaf_loss_fn_normalized_to_composed_loss(self) -> None:
        strategy = _make_strategy(loss_fn=EnergyLoss())
        assert isinstance(strategy.loss_fn, ComposedLossFunction)
        assert len(strategy.loss_fn.components) == 1
        assert isinstance(strategy.loss_fn.components[0], EnergyLoss)

    def test_single_model_rejects_mapping_annotation(self) -> None:
        with pytest.raises(ValueError, match="single-model"):
            _make_strategy(training_fn=mapping_annotated_training_fn)

    def test_single_model_rejects_moduledict_annotation(self) -> None:
        with pytest.raises(ValueError, match="single-model"):
            _make_strategy(training_fn=moduledict_annotated_training_fn)

    def test_dict_models_reject_single_model_annotation(self) -> None:
        with pytest.raises(ValueError, match="models=model"):
            _make_strategy(
                models={"student": _make_demo_model(), "teacher": _make_demo_model()},
                optimizer_configs={
                    "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
                },
                training_fn=single_model_training_fn,
            )

    def test_duplicate_hook_instances_rejected(self) -> None:
        hook = _RecordingHook(TrainingStage.BEFORE_BATCH, lambda ctx, stage: None)
        with pytest.raises(ValueError, match="duplicate hook"):
            _make_strategy(hooks=[hook, hook])


class TestTrainingStrategyRun:
    def test_single_model_training_fn_receives_model_only(self) -> None:
        seen: list[BaseModelMixin] = []

        def _training_fn(
            model: BaseModelMixin, batch: Batch
        ) -> dict[str, torch.Tensor]:
            seen.append(model)
            return demo_training_fn(model, batch)

        strategy = _make_strategy(training_fn=_training_fn)
        strategy.run([_make_batch()])
        assert seen == [strategy.models["main"]]

    def test_dict_model_training_fn_receives_all_models(self) -> None:
        strategy = _make_strategy(
            models={"student": _make_demo_model(), "teacher": _make_demo_model()},
            optimizer_configs={
                "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
            },
            training_fn=dict_demo_training_fn,
        )
        strategy.run([_make_batch()])
        assert strategy.step_count == 1

    def test_dict_model_multi_device_run_raises(self) -> None:
        strategy = _make_strategy(
            models={"student": _make_demo_model(), "teacher": _make_demo_model()},
            optimizer_configs={
                "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
            },
            training_fn=dict_demo_training_fn,
            devices=[torch.device("cpu"), torch.device("cpu")],
        )
        with pytest.raises(
            ValueError, match="Named-model training with multiple devices"
        ):
            strategy.run([_make_batch()])

    def test_moduledict_models_are_accepted_as_named_models(self) -> None:
        strategy = _make_strategy(
            models=torch.nn.ModuleDict(
                {"student": _make_demo_model(), "teacher": _make_demo_model()}
            ),
            optimizer_configs={
                "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
            },
            training_fn=dict_demo_training_fn,
        )
        assert isinstance(strategy.models, dict)
        assert set(strategy.models) == {"student", "teacher"}
        strategy.run([_make_batch()])
        assert strategy.step_count == 1

    def test_omitted_model_is_temporarily_frozen_and_eval(self) -> None:
        teacher = _make_demo_model()
        teacher.eval()
        params = list(teacher.parameters())
        params[0].requires_grad_(False)
        initial_training = teacher.training
        initial_requires_grad = [param.requires_grad for param in params]
        seen_during_run: list[tuple[bool, list[bool]]] = []

        def _training_fn(
            models: dict[str, BaseModelMixin], batch: Batch
        ) -> dict[str, torch.Tensor]:
            seen_during_run.append(
                (
                    models["teacher"].training,
                    [param.requires_grad for param in models["teacher"].parameters()],
                )
            )
            return dict_demo_training_fn(models, batch)

        strategy = _make_strategy(
            models={"student": _make_demo_model(), "teacher": teacher},
            optimizer_configs={
                "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
            },
            training_fn=_training_fn,
        )
        strategy.run([_make_batch()])
        assert strategy.models["student"].training is True
        assert any(
            param.requires_grad for param in strategy.models["student"].parameters()
        )
        assert seen_during_run == [(False, [False] * len(params))]
        assert strategy.models["teacher"].training is initial_training
        assert [param.requires_grad for param in params] == initial_requires_grad

    def test_default_training_fn_opt_in_runs_single_model(self) -> None:
        strategy = _make_strategy(training_fn=default_training_fn)
        strategy.run([_make_batch()])
        assert strategy.step_count == 1

    def test_train_batch_public_api_runs_per_batch_flow_only(self) -> None:
        seen: list[TrainingStage] = []
        strategy = _make_strategy(
            hooks=[
                _RecordingHook(
                    TrainingStage.BEFORE_TRAINING,
                    lambda _ctx, stage: seen.append(stage),
                ),
                _RecordingHook(
                    TrainingStage.BEFORE_BATCH,
                    lambda _ctx, stage: seen.append(stage),
                ),
            ]
        )

        strategy.train_batch(_make_batch())

        assert seen == [TrainingStage.BEFORE_BATCH]
        assert strategy.step_count == 1
        assert strategy._last_batch is not None

    def test_train_batch_reuses_runtime_optimizer_state(self) -> None:
        strategy = _make_strategy()
        strategy.train_batch(_make_batch())
        optimizers = strategy._optimizers
        schedulers = strategy._lr_schedulers

        strategy.train_batch(_make_batch(seed=10))

        assert strategy.step_count == 2
        assert strategy._optimizers is optimizers
        assert strategy._lr_schedulers is schedulers

    def test_two_epoch_loop_updates_counters_and_loss_hooks(self) -> None:
        torch.manual_seed(0)
        after_loss_calls: list[int] = []

        def _record(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            assert ctx.loss is not None
            after_loss_calls.append(ctx.step_count)

        strategy = _make_strategy(
            num_epochs=2,
            hooks=[_RecordingHook(TrainingStage.AFTER_LOSS, _record)],
        )
        dataset = _make_dataset(n_batches=3)
        strategy.run(dataset)

        assert strategy.step_count == 2 * len(dataset)
        assert strategy.epoch == 2
        assert after_loss_calls == list(range(2 * len(dataset)))


_EXPECTED_STAGE_ORDER: tuple[TrainingStage, ...] = (
    TrainingStage.BEFORE_TRAINING,
    TrainingStage.BEFORE_EPOCH,
    TrainingStage.BEFORE_BATCH,
    TrainingStage.BEFORE_FORWARD,
    TrainingStage.AFTER_FORWARD,
    TrainingStage.BEFORE_LOSS,
    TrainingStage.AFTER_LOSS,
    TrainingStage.BEFORE_BACKWARD,
    TrainingStage.AFTER_BACKWARD,
    TrainingStage.BEFORE_OPTIMIZER_STEP,
    TrainingStage.AFTER_OPTIMIZER_STEP,
    TrainingStage.AFTER_BATCH,
    TrainingStage.AFTER_EPOCH,
    TrainingStage.AFTER_TRAINING,
)


# Snapshot shape: (loss_populated, losses_populated, requires_grad).
_LossSnapshot = tuple[bool, bool, bool]


def _snapshot_ctx(ctx: HookContext) -> _LossSnapshot:
    return (
        ctx.loss is not None,
        ctx.losses is not None,
        bool(ctx.loss.requires_grad) if ctx.loss is not None else False,
    )


class TestTrainingStrategyHookOrder:
    def test_strategy_context_manager_nests_without_reentry(self) -> None:
        events: list[str] = []

        class _ContextHook:
            stage = TrainingStage.BEFORE_BATCH
            frequency = 1

            def __enter__(self) -> None:
                events.append("enter")

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                events.append("exit")

            def __call__(self, ctx: HookContext, stage: Enum) -> None:
                pass

        hook = _ContextHook()
        strategy = _make_strategy(hooks=[hook])
        with strategy:
            with strategy:
                assert events == ["enter"]
            assert events == ["enter"]
        assert events == ["enter", "exit"]

    def test_entered_strategy_run_reuses_hook_context(self) -> None:
        events: list[str] = []

        class _ContextHook:
            stage = TrainingStage.BEFORE_BATCH
            frequency = 1

            def __enter__(self) -> None:
                events.append("enter")

            def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                events.append("exit")

            def __call__(self, ctx: HookContext, stage: Enum) -> None:  # noqa: ARG002
                events.append("call")

        hook = _ContextHook()
        strategy = _make_strategy(hooks=[hook])
        with strategy:
            strategy.run([_make_batch()])
        assert events == ["enter", "call", "exit"]

    def test_strategy_context_exposes_named_models(self) -> None:
        seen_keys: list[set[str]] = []

        def _record(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            assert isinstance(ctx, TrainContext)
            seen_keys.append(set(ctx.models))
            assert ctx.model is ctx.models["main"]

        strategy = _make_strategy(
            hooks=[_RecordingHook(TrainingStage.BEFORE_BATCH, _record)]
        )
        strategy.run([_make_batch()])
        assert seen_keys == [{"main"}]

    def test_stage_order_one_batch(self) -> None:
        torch.manual_seed(0)
        log: list[Enum] = []
        hooks = [
            _RecordingHook(stage, lambda ctx, s, _log=log: _log.append(s))  # noqa: ARG005
            for stage in _EXPECTED_STAGE_ORDER
        ]
        strategy = _make_strategy(hooks=hooks)
        strategy.run([_make_batch()])
        assert tuple(log) == _EXPECTED_STAGE_ORDER

    def test_hook_context_loss_lifecycle(self) -> None:
        torch.manual_seed(0)
        tracked_stages = (
            TrainingStage.BEFORE_LOSS,
            TrainingStage.AFTER_LOSS,
            TrainingStage.BEFORE_BACKWARD,
            TrainingStage.AFTER_BACKWARD,
            TrainingStage.BEFORE_OPTIMIZER_STEP,
            TrainingStage.AFTER_BATCH,
        )
        snapshots: dict[TrainingStage, list[_LossSnapshot]] = {
            stage: [] for stage in tracked_stages
        }

        def _record_snapshot(ctx: HookContext, stage: TrainingStage) -> None:
            snapshots[stage].append(_snapshot_ctx(ctx))

        hooks = [_RecordingHook(stage, _record_snapshot) for stage in tracked_stages]
        strategy = _make_strategy(hooks=hooks)
        strategy.run([_make_batch()])

        # Before the loss is computed, loss + losses are both absent.
        assert snapshots[TrainingStage.BEFORE_LOSS] == [(False, False, False)]

        # AFTER_LOSS + BEFORE_BACKWARD: loss is live and requires grad.
        for stage in (TrainingStage.AFTER_LOSS, TrainingStage.BEFORE_BACKWARD):
            assert snapshots[stage] == [(True, True, True)]

        # From AFTER_BACKWARD onward, loss is detached.
        for stage in (
            TrainingStage.AFTER_BACKWARD,
            TrainingStage.BEFORE_OPTIMIZER_STEP,
            TrainingStage.AFTER_BATCH,
        ):
            assert snapshots[stage] == [(True, True, False)]


class TestTrainingStrategySpecRoundTrip:
    def test_roundtrip_preserves_declarative_fields(self) -> None:
        torch.manual_seed(0)
        loss_fn = EnergyLoss(per_atom=True) + ForceLoss(normalize_by_atom_count=False)
        strategy = _make_strategy(
            optimizer_configs={
                "main": [
                    OptimizerConfig(
                        optimizer_cls=torch.optim.Adam,
                        optimizer_kwargs={"lr": 1e-3},
                        scheduler_cls=torch.optim.lr_scheduler.StepLR,
                        scheduler_kwargs={"step_size": 3, "gamma": 0.5},
                    )
                ]
            },
            num_epochs=2,
            loss_fn=loss_fn,
            devices=[torch.device("cpu")],
        )
        spec = strategy.to_spec_dict()
        spec_back = json.loads(json.dumps(spec))

        fresh_model = _make_demo_model()
        restored = TrainingStrategy.from_spec_dict(
            spec_back, models=fresh_model, hooks=[]
        )
        assert restored.num_epochs == 2
        assert restored.num_steps is None
        assert restored.devices == [torch.device("cpu")]
        assert restored.training_fn is demo_training_fn
        assert "main" in spec["model_specs"]
        assert spec["single_model_input"] is True
        restored_cfg = restored.optimizer_configs["main"][0]
        assert restored_cfg.optimizer_cls is torch.optim.Adam
        assert restored_cfg.optimizer_kwargs["lr"] == pytest.approx(1e-3)
        assert restored_cfg.scheduler_cls is torch.optim.lr_scheduler.StepLR
        assert restored_cfg.scheduler_kwargs == {"step_size": 3, "gamma": 0.5}
        assert isinstance(restored.loss_fn, ComposedLossFunction)
        leaves = list(restored.loss_fn.components)
        assert len(leaves) == 2
        assert isinstance(leaves[0], EnergyLoss)
        assert isinstance(leaves[1], ForceLoss)
        assert leaves[0].per_atom is True
        assert leaves[1].normalize_by_atom_count is False

    def test_roundtrip_preserves_loss_weights_and_normalization(self) -> None:
        loss_fn = ComposedLossFunction(
            [
                EnergyLoss(),
                ForceLoss(normalize_by_atom_count=False),
            ],
            weights=[0.25, LinearWeight(start=0.1, end=0.5, num_steps=10)],
            normalize_weights=False,
        )
        strategy = _make_strategy(loss_fn=loss_fn)

        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        restored = TrainingStrategy.from_spec_dict(
            spec, models=_make_demo_model(), hooks=[]
        )

        assert restored.loss_fn.normalize_weights is False
        assert restored.loss_fn._weights[0] == pytest.approx(0.25)
        assert isinstance(restored.loss_fn._weights[1], LinearWeight)
        schedule = restored.loss_fn._weights[1]
        assert schedule.start == pytest.approx(0.1)
        assert schedule.end == pytest.approx(0.5)
        assert schedule.num_steps == 10

    def test_missing_optimizer_configs_key_raises(self) -> None:
        torch.manual_seed(0)
        spec = _make_strategy().to_spec_dict()
        del spec["optimizer_configs"]
        with pytest.raises(ValueError, match="optimizer_configs"):
            TrainingStrategy.from_spec_dict(spec, models=_make_demo_model(), hooks=[])

    @pytest.mark.parametrize(
        ("key", "value", "match"),
        _FROM_SPEC_REJECTION_CASES,
        ids=[
            "optimizer_configs_not_mapping",
            "optimizer_config_entries_not_specs",
            "devices_not_list",
            "loss_fn_spec_not_mapping",
            "model_specs_not_mapping",
            "missing_training_fn",
            "training_fn_not_string",
            "single_model_input_not_bool",
        ],
    )
    def test_from_spec_rejects_malformed_fields(
        self, key: str, value: Any, match: str
    ) -> None:
        spec = _make_strategy().to_spec_dict()
        if value is _DELETE:
            del spec[key]
        else:
            spec[key] = value

        with pytest.raises(ValueError, match=match):
            TrainingStrategy.from_spec_dict(spec, models=_make_demo_model(), hooks=[])

    def test_integer_optimizer_key_migrates_to_main(self) -> None:
        torch.manual_seed(0)
        spec = _make_strategy().to_spec_dict()
        original = spec["optimizer_configs"]["main"]
        spec["optimizer_configs"] = {"0": original}
        restored = TrainingStrategy.from_spec_dict(
            spec, models=_make_demo_model(), hooks=[]
        )
        assert set(restored.optimizer_configs) == {"main"}

    def test_single_model_spec_without_runtime_model_restores_single_call_mode(
        self,
    ) -> None:
        strategy = _make_strategy()
        seen_args: list[BaseModelMixin | dict[str, BaseModelMixin]] = []

        def _record_training_fn(
            model: BaseModelMixin, batch: Batch
        ) -> dict[str, torch.Tensor]:
            seen_args.append(model)
            return default_training_fn(strategy.models["main"], batch)

        restored = TrainingStrategy.from_spec_dict(
            strategy.to_spec_dict(), hooks=[], training_fn=_record_training_fn
        )
        restored.train_batch(_make_batch())
        assert seen_args == [restored.models["main"]]

    def test_single_main_named_spec_restores_named_call_mode(self) -> None:
        strategy = _make_strategy(
            models={"main": _make_demo_model()},
            optimizer_configs=_adam_optimizer_configs(),
            training_fn=mapping_annotated_training_fn,
        )

        spec = strategy.to_spec_dict()
        restored = TrainingStrategy.from_spec_dict(spec, hooks=[])

        assert spec["single_model_input"] is False
        assert restored.single_model_input is False
        restored.run([_make_batch()])
        assert restored.step_count == 1

    def test_model_spec_roundtrip_restores_runnable_demo_model(self) -> None:
        strategy = _make_strategy(training_fn=default_training_fn)
        restored = TrainingStrategy.from_spec_dict(strategy.to_spec_dict(), hooks=[])

        assert restored.models["main"] is not strategy.models["main"]
        restored.run([_make_batch()])

        assert restored.step_count == 1

    def test_runtime_model_override_merges_over_spec_models(self) -> None:
        torch.manual_seed(0)
        spec = _make_strategy(
            models={"main": _make_demo_model(), "teacher": _make_demo_model()},
            optimizer_configs={
                "main": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
            },
            training_fn=dict_demo_training_fn,
        ).to_spec_dict()
        replacement = _make_demo_model()
        restored = TrainingStrategy.from_spec_dict(spec, models=replacement, hooks=[])
        assert restored.models["main"] is replacement
        assert "teacher" in restored.models
        assert restored.single_model_input is False

    @pytest.mark.parametrize("drop_training_fn", [False, True])
    def test_runtime_training_fn_override(self, drop_training_fn: bool) -> None:
        spec = _make_strategy().to_spec_dict()
        if drop_training_fn:
            del spec["training_fn"]
        restored = TrainingStrategy.from_spec_dict(
            spec,
            models=_make_demo_model(),
            hooks=[],
            training_fn=default_training_fn,
        )
        assert restored.training_fn is default_training_fn

    def test_non_importable_training_fn_warns_and_is_omitted(self) -> None:
        strategy = _make_strategy(training_fn=lambda model, batch: {})
        with pytest.warns(UserWarning, match="Omitting non-importable training_fn"):
            spec = strategy.to_spec_dict()
        assert "training_fn" not in spec
