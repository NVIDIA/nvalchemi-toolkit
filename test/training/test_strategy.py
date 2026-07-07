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
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any
from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.hooks._context import HookContext, TrainContext
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import (
    ComposedLossFunction,
    EnergyMSELoss,
    ForceMSELoss,
    LinearWeight,
    TrainingStage,
    create_model_spec,
)
from nvalchemi.training.hooks import TrainingUpdateHook
from nvalchemi.training.optimizers import OptimizerConfig
from nvalchemi.training.strategy import TrainingStrategy, default_training_fn
from test.training.conftest import (
    _build_adam_optimizer_configs,
    _build_baseline_strategy_kwargs,
    _build_batch,
    _build_dataset,
    _build_demo_model,
)


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


def _make_demo_model() -> Any:
    """Return a freshly seeded demo model for local strategy tests."""
    return _build_demo_model()


def _make_batch(n_systems: int = 2, n_atoms_each: int = 3, seed: int = 0) -> Batch:
    """Return a deterministic batch for local strategy tests."""
    return _build_batch(n_systems=n_systems, n_atoms_each=n_atoms_each, seed=seed)


def _make_dataset(
    n_batches: int = 3,
    n_systems: int = 2,
    n_atoms_each: int = 3,
    base_seed: int = 100,
) -> list[Batch]:
    """Return a deterministic dataset for local strategy tests."""
    return _build_dataset(
        n_batches=n_batches,
        n_systems=n_systems,
        n_atoms_each=n_atoms_each,
        base_seed=base_seed,
    )


def _adam_optimizer_configs() -> dict[str, list[OptimizerConfig]]:
    """Return the default Adam optimizer config mapping."""
    return _build_adam_optimizer_configs()


def _make_strategy(**overrides: Any) -> TrainingStrategy:
    """Build a strategy with baseline kwargs plus local overrides."""
    models = overrides.pop("models") if "models" in overrides else None
    kwargs = _build_baseline_strategy_kwargs(models=models)
    kwargs.update(overrides)
    return TrainingStrategy(**kwargs)


class _RecordingLinear(torch.nn.Linear):
    """Linear module that records device-placement calls."""

    def __init__(self) -> None:
        super().__init__(4, 4)
        self.to_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def to(self, *args: Any, **kwargs: Any) -> torch.nn.Module:
        """Record and forward :meth:`torch.nn.Module.to` calls."""
        self.to_calls.append((args, kwargs))
        return super().to(*args, **kwargs)


class _RuntimeWeightSchedule:
    per_epoch = False

    def __init__(self, value: float = 1.0) -> None:
        self.value = float(value)

    def __call__(self, step: int, epoch: int) -> float:  # noqa: ARG002
        return self.value

    def to_spec(self) -> Any:
        return create_model_spec(type(self), value=self.value)


class _MissingToSpecWeightSchedule:
    per_epoch = False

    def __call__(self, step: int, epoch: int) -> float:  # noqa: ARG002
        return 1.0


class _ToSpecWeightSchedule:
    per_epoch = False

    def __init__(self, value: float, *, expose_attr: bool = True) -> None:
        if expose_attr:
            self.value = float(value)
        else:
            self._value = float(value)

    def __call__(self, step: int, epoch: int) -> float:  # noqa: ARG002
        return self.value if hasattr(self, "value") else self._value

    def to_spec(self) -> Any:
        return create_model_spec(type(self), value=self._value, expose_attr=False)


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


class _EveryOtherOptimizerStepHook(TrainingUpdateHook):
    """Veto optimizer steps on alternating batches."""

    priority = 10

    def __init__(self) -> None:
        self.calls = 0
        self.batch_counts: list[int] = []
        self.step_counts: list[int] = []
        self.step_decisions: list[bool] = []
        self.after_skip: list[bool] = []

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, torch.Tensor | None]:
        if stage is TrainingStage.DO_OPTIMIZER_STEP:
            should_step = self.calls % 2 == 1
            self.batch_counts.append(ctx.batch_count)
            self.step_counts.append(ctx.step_count)
            self.step_decisions.append(should_step)
            self.calls += 1
            return should_step, ctx.loss
        if stage is TrainingStage.AFTER_OPTIMIZER_STEP:
            self.after_skip.append(will_skip)
        return True, ctx.loss


class _EpochSampler:
    """Sampler stub that records epochs passed to ``set_epoch``."""

    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


@dataclass
class _DistributedManagerStub:
    """Minimal distributed manager for counter tests."""

    world_size: int
    rank: int = 0


class _RestartableLoader:
    """Re-iterable sized loader with a sampler for restart tests."""

    def __init__(self, batches: list[Batch]) -> None:
        self._batches = batches
        self.sampler = _EpochSampler()

    def __iter__(self) -> Iterator[Batch]:
        return iter(self._batches)

    def __len__(self) -> int:
        return len(self._batches)


class _SeekableRestartableLoader(_RestartableLoader):
    """Restart test loader that can seek within an epoch cheaply."""

    def __init__(self, batches: list[Batch]) -> None:
        super().__init__(batches)
        self.epoch_steps: list[int] = []
        self._epoch_step_start = 0

    def __iter__(self) -> Iterator[Batch]:
        start = self._epoch_step_start
        self._epoch_step_start = 0
        return iter(self._batches[start:])

    def set_epoch_step(self, step: int) -> None:
        self.epoch_steps.append(step)
        self._epoch_step_start = step


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
    ("greater than 0", {"epoch_step_modifier": 0}),
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
            "nonpositive_epoch_step_modifier",
            "training_fn_bad_dotted_path",
        ],
    )
    def test_construction_rejected(
        self,
        match: str,
        overrides: dict[str, Any],
        baseline_strategy_kwargs: dict[str, Any],
    ) -> None:
        kwargs = {**baseline_strategy_kwargs, **overrides}
        with pytest.raises(ValueError, match=match):
            TrainingStrategy(**kwargs)

    def test_training_fn_dotted_string_resolved(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        strat = TrainingStrategy(
            **{**baseline_strategy_kwargs, "training_fn": "operator.add"}
        )
        assert strat.training_fn is operator.add

    def test_training_fn_required_message_suggests_default(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        kwargs = dict(baseline_strategy_kwargs)
        del kwargs["training_fn"]
        with pytest.raises(ValueError, match="default_training_fn"):
            TrainingStrategy(**kwargs)

    def test_leaf_loss_fn_normalized_to_composed_loss(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        strategy = TrainingStrategy(
            **{**baseline_strategy_kwargs, "loss_fn": EnergyMSELoss()}
        )
        assert isinstance(strategy.loss_fn, ComposedLossFunction)
        assert len(strategy.loss_fn.components) == 1
        assert isinstance(strategy.loss_fn.components[0], EnergyMSELoss)

    def test_schedule_without_to_spec_rejected_by_composition(
        self,
        baseline_strategy_kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        with pytest.raises(
            TypeError,
            match=r"weights\[0\] must be a float or LossWeightSchedule",
        ):
            ComposedLossFunction(
                [EnergyMSELoss()], weights=[_MissingToSpecWeightSchedule()]
            )

    def test_nested_schedule_without_to_spec_rejected_by_composition(
        self,
        baseline_strategy_kwargs: dict[str, Any],  # noqa: ARG002
    ) -> None:
        with pytest.raises(
            TypeError,
            match=r"weights\[0\] must be a float or LossWeightSchedule",
        ):
            _ = 0.25 * ComposedLossFunction(
                [EnergyMSELoss()], weights=[_MissingToSpecWeightSchedule()]
            )

    def test_loss_target_assembler_can_route_prediction_targets(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        observed_workflows: list[Any] = []

        def _training_fn(
            model: BaseModelMixin, batch: Batch
        ) -> dict[str, torch.Tensor]:
            outputs = demo_training_fn(model, batch)
            return {
                "student_energy": outputs["predicted_energy"],
                "teacher_energy": (batch.energy + 0.5).detach(),
            }

        def _target_assembler(
            loss_fn: ComposedLossFunction,
            predictions: Mapping[str, torch.Tensor],
            batch: Batch,
            *,
            workflow: Any | None = None,
            target_keys: Sequence[str] | None = None,
            batch_label: str = "Batch",
        ) -> Mapping[str, torch.Tensor]:
            del loss_fn, batch, target_keys, batch_label
            observed_workflows.append(workflow)
            return {"teacher_energy": predictions["teacher_energy"]}

        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "training_fn": _training_fn,
                "loss_fn": EnergyMSELoss(
                    prediction_key="student_energy",
                    target_key="teacher_energy",
                ),
                "loss_target_assembler": _target_assembler,
            }
        )

        strategy.run([_make_batch()])

        assert observed_workflows == [strategy]

    def test_single_model_rejects_mapping_annotation(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError, match="single-model"):
            TrainingStrategy(
                **{
                    **baseline_strategy_kwargs,
                    "training_fn": mapping_annotated_training_fn,
                }
            )

    def test_single_model_rejects_moduledict_annotation(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError, match="single-model"):
            TrainingStrategy(
                **{
                    **baseline_strategy_kwargs,
                    "training_fn": moduledict_annotated_training_fn,
                }
            )

    def test_dict_models_reject_single_model_annotation(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        with pytest.raises(ValueError, match="models=model"):
            TrainingStrategy(
                **{
                    **baseline_strategy_kwargs,
                    "models": {
                        "student": _build_demo_model(),
                        "teacher": _build_demo_model(),
                    },
                    "optimizer_configs": {
                        "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
                    },
                    "training_fn": single_model_training_fn,
                }
            )

    def test_duplicate_hook_instances_rejected(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        hook = _RecordingHook(TrainingStage.BEFORE_BATCH, lambda ctx, stage: None)
        with pytest.raises(ValueError, match="duplicate hook"):
            TrainingStrategy(**{**baseline_strategy_kwargs, "hooks": [hook, hook]})

    def test_epoch_constructor_alias_populates_epoch_count(self) -> None:
        strategy = _make_strategy(epoch=3)
        assert strategy.epoch_count == 3
        assert strategy.epoch == 3


class TestTrainingStrategyRun:
    def test_single_model_training_fn_receives_model_only(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        seen: list[BaseModelMixin] = []

        def _training_fn(
            model: BaseModelMixin, batch: Batch
        ) -> dict[str, torch.Tensor]:
            seen.append(model)
            return demo_training_fn(model, batch)

        strategy = TrainingStrategy(
            **{**baseline_strategy_kwargs, "training_fn": _training_fn}
        )
        strategy.run([batch])
        assert seen == [strategy.models["main"]]

    def test_dict_model_training_fn_receives_all_models(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "models": {
                    "student": _build_demo_model(),
                    "teacher": _build_demo_model(),
                },
                "optimizer_configs": {
                    "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
                },
                "training_fn": dict_demo_training_fn,
            }
        )
        strategy.run([batch])
        assert strategy.step_count == 1

    def test_dict_model_multi_device_run_raises(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "models": {
                    "student": _build_demo_model(),
                    "teacher": _build_demo_model(),
                },
                "optimizer_configs": {
                    "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
                },
                "training_fn": dict_demo_training_fn,
                "devices": [torch.device("cpu"), torch.device("cpu")],
            }
        )
        with pytest.raises(
            ValueError, match="Named-model training with multiple devices"
        ):
            strategy.run([batch])

    def test_moduledict_models_are_accepted_as_named_models(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "models": torch.nn.ModuleDict(
                    {"student": _build_demo_model(), "teacher": _build_demo_model()}
                ),
                "optimizer_configs": {
                    "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
                },
                "training_fn": dict_demo_training_fn,
            }
        )
        assert isinstance(strategy.models, dict)
        assert set(strategy.models) == {"student", "teacher"}
        strategy.run([batch])
        assert strategy.step_count == 1

    def test_omitted_model_is_temporarily_frozen_and_eval(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        teacher = _build_demo_model()
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

        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "models": {"student": _build_demo_model(), "teacher": teacher},
                "optimizer_configs": {
                    "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
                },
                "training_fn": _training_fn,
            }
        )
        strategy.run([batch])
        assert strategy.models["student"].training is True
        assert any(
            param.requires_grad for param in strategy.models["student"].parameters()
        )
        assert seen_during_run == [(False, [False] * len(params))]
        assert strategy.models["teacher"].training is initial_training
        assert [param.requires_grad for param in params] == initial_requires_grad

    def test_default_training_fn_opt_in_runs_single_model(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        strategy = TrainingStrategy(
            **{**baseline_strategy_kwargs, "training_fn": default_training_fn}
        )
        strategy.run([batch])
        assert strategy.step_count == 1

    def test_train_batch_public_api_runs_per_batch_flow_only(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        seen: list[TrainingStage] = []
        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "hooks": [
                    _RecordingHook(
                        TrainingStage.BEFORE_TRAINING,
                        lambda _ctx, stage: seen.append(stage),
                    ),
                    _RecordingHook(
                        TrainingStage.BEFORE_BATCH,
                        lambda _ctx, stage: seen.append(stage),
                    ),
                ],
            }
        )

        strategy.train_batch(batch)

        assert seen == [TrainingStage.BEFORE_BATCH]
        assert strategy.step_count == 1
        assert strategy.batch_count == 1
        assert strategy._last_batch is not None

    def test_train_batch_reuses_runtime_optimizer_state(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        strategy = TrainingStrategy(**baseline_strategy_kwargs)
        strategy.train_batch(batch)
        optimizers = strategy._optimizers
        schedulers = strategy._lr_schedulers

        strategy.train_batch(_build_batch(seed=10))

        assert strategy.step_count == 2
        assert strategy.batch_count == 2
        assert strategy._optimizers is optimizers
        assert strategy._lr_schedulers is schedulers

    def test_two_epoch_loop_updates_counters_and_loss_hooks(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        after_loss_calls: list[int] = []

        def _record(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            assert ctx.loss is not None
            after_loss_calls.append(ctx.step_count)

        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "num_epochs": 2,
                "hooks": [_RecordingHook(TrainingStage.AFTER_LOSS, _record)],
            }
        )
        dataset = _build_dataset(n_batches=3)
        strategy.run(dataset)

        assert strategy.step_count == 2 * len(dataset)
        assert strategy.batch_count == 2 * len(dataset)
        assert strategy.epoch_count == 2
        assert strategy.epoch_step_count == 0
        assert strategy.epoch == strategy.epoch_count
        assert after_loss_calls == list(range(2 * len(dataset)))

    def test_num_steps_recycles_dataloader_until_target(self) -> None:
        torch.manual_seed(0)
        after_loss_calls: list[int] = []

        def _record(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            after_loss_calls.append(ctx.step_count)

        strategy = _make_strategy(
            num_epochs=None,
            num_steps=5,
            hooks=[_RecordingHook(TrainingStage.AFTER_LOSS, _record)],
        )
        strategy.run(_make_dataset(n_batches=2))

        assert strategy.step_count == 5
        assert strategy.batch_count == 5
        assert after_loss_calls == list(range(5))

    def test_num_steps_run_at_target_is_noop(self) -> None:
        calls = 0

        def _training_fn(
            model: BaseModelMixin, batch: Batch
        ) -> dict[str, torch.Tensor]:
            nonlocal calls
            calls += 1
            return demo_training_fn(model, batch)

        strategy = _make_strategy(
            num_epochs=None,
            num_steps=1,
            training_fn=_training_fn,
        )
        dataset = _make_dataset(n_batches=2)

        strategy.run(dataset)
        strategy.run(dataset)

        assert calls == 1
        assert strategy.step_count == 1
        assert strategy.batch_count == 1

    def test_num_epochs_run_at_converted_target_is_noop(self) -> None:
        calls = 0

        def _training_fn(
            model: BaseModelMixin, batch: Batch
        ) -> dict[str, torch.Tensor]:
            nonlocal calls
            calls += 1
            return demo_training_fn(model, batch)

        strategy = _make_strategy(num_epochs=1, training_fn=_training_fn)
        dataset = _make_dataset(n_batches=2)

        strategy.run(dataset)
        strategy.run(dataset)

        assert calls == len(dataset)
        assert strategy.step_count == len(dataset)
        assert strategy.batch_count == len(dataset)

    def test_num_epochs_target_uses_epoch_step_modifier(self) -> None:
        strategy = _make_strategy(num_epochs=2, epoch_step_modifier=0.5)
        strategy.run(_make_dataset(n_batches=3))

        assert strategy.step_count == 3
        assert strategy.batch_count == 3

    def test_num_epochs_target_counts_executed_optimizer_steps(self) -> None:
        hook = _EveryOtherOptimizerStepHook()
        strategy = _make_strategy(
            num_epochs=1,
            epoch_step_modifier=0.5,
            hooks=[hook],
        )

        strategy.run(_make_dataset(n_batches=4))

        assert strategy.step_count == 2
        assert strategy.batch_count == 4
        assert strategy.epoch_count == 1
        assert strategy.epoch_step_count == 0
        assert hook.batch_counts == [0, 1, 2, 3]
        assert hook.step_counts == [0, 0, 1, 1]
        assert hook.step_decisions == [False, True, False, True]
        assert hook.after_skip == [True, False, True, False]

    def test_num_epochs_requires_sized_dataloader(self) -> None:
        strategy = _make_strategy(num_epochs=1)

        with pytest.raises(ValueError, match="num_epochs requires a sized dataloader"):
            strategy.run(iter(_make_dataset(n_batches=1)))

    def test_run_resumes_from_epoch_and_step_count(self) -> None:
        dataset = _make_dataset(n_batches=3)
        loader = _RestartableLoader(dataset)
        batch_index = {
            float(batch.energy.detach().cpu().flatten()[0]): i
            for i, batch in enumerate(dataset)
        }
        seen_batches: list[int] = []

        def _training_fn(
            model: BaseModelMixin, batch: Batch
        ) -> dict[str, torch.Tensor]:
            key = float(batch.energy.detach().cpu().flatten()[0])
            seen_batches.append(batch_index[key])
            return demo_training_fn(model, batch)

        strategy = _make_strategy(
            num_epochs=None,
            num_steps=7,
            step_count=4,
            epoch_count=1,
            training_fn=_training_fn,
        )

        strategy.run(loader)

        assert loader.sampler.epochs == [1, 2]
        assert seen_batches == [1, 2, 0]
        assert strategy.step_count == 7
        assert strategy.batch_count == 7
        assert strategy.epoch_count == 2
        assert strategy.epoch_step_count == 1

    def test_run_resumes_from_explicit_epoch_step_count(self) -> None:
        dataset = _make_dataset(n_batches=3)
        loader = _RestartableLoader(dataset)
        batch_index = {
            float(batch.energy.detach().cpu().flatten()[0]): i
            for i, batch in enumerate(dataset)
        }
        seen_batches: list[int] = []

        def _training_fn(
            model: BaseModelMixin, batch: Batch
        ) -> dict[str, torch.Tensor]:
            key = float(batch.energy.detach().cpu().flatten()[0])
            seen_batches.append(batch_index[key])
            return demo_training_fn(model, batch)

        strategy = _make_strategy(
            num_epochs=None,
            num_steps=7,
            step_count=4,
            epoch_count=1,
            epoch_step_count=1,
            training_fn=_training_fn,
        )

        strategy.run(loader)

        assert loader.sampler.epochs == [1, 2]
        assert seen_batches == [1, 2, 0]
        assert strategy.step_count == 7
        assert strategy.batch_count == 7
        assert strategy.epoch_count == 2
        assert strategy.epoch_step_count == 1

    def test_run_resumes_by_seeking_loader_epoch_step(self) -> None:
        dataset = _make_dataset(n_batches=5)
        loader = _SeekableRestartableLoader(dataset)
        batch_index = {
            float(batch.energy.detach().cpu().flatten()[0]): i
            for i, batch in enumerate(dataset)
        }
        seen_batches: list[int] = []

        def _training_fn(
            model: BaseModelMixin, batch: Batch
        ) -> dict[str, torch.Tensor]:
            key = float(batch.energy.detach().cpu().flatten()[0])
            seen_batches.append(batch_index[key])
            return demo_training_fn(model, batch)

        strategy = _make_strategy(
            num_epochs=None,
            num_steps=6,
            step_count=3,
            batch_count=3,
            epoch_count=0,
            epoch_step_count=3,
            training_fn=_training_fn,
        )

        strategy.run(loader)

        assert loader.sampler.epochs == [0, 1]
        assert loader.epoch_steps == [3]
        assert seen_batches == [3, 4, 0]
        assert strategy.step_count == 6
        assert strategy.batch_count == 6
        assert strategy.epoch_count == 1
        assert strategy.epoch_step_count == 1

    def test_global_step_count_advances_by_world_size_and_reaches_hooks(self) -> None:
        seen: list[tuple[int, int]] = []

        def _record(ctx: HookContext, stage: TrainingStage) -> None:
            assert isinstance(ctx, TrainContext)
            seen.append((ctx.step_count, ctx.global_step_count))

        strategy = _make_strategy(
            num_epochs=None,
            num_steps=2,
            distributed_manager=_DistributedManagerStub(world_size=4),
            hooks=[_RecordingHook(TrainingStage.AFTER_BATCH, _record)],
        )

        strategy.run(_make_dataset(n_batches=2))

        assert strategy.step_count == 2
        assert strategy.global_step_count == 8
        assert seen == [(1, 4), (2, 8)]

    def test_run_rejects_inconsistent_explicit_epoch_step_count(self) -> None:
        strategy = _make_strategy(
            num_epochs=None,
            num_steps=7,
            step_count=4,
            epoch_count=1,
            epoch_step_count=2,
        )

        with pytest.raises(ValueError, match="restart counters are inconsistent"):
            strategy.run(_make_dataset(n_batches=3))


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
    def test_strategy_context_manager_nests_without_reentry(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
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
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "hooks": [hook]})
        with strategy:
            with strategy:
                assert events == ["enter"]
            assert events == ["enter"]
        assert events == ["enter", "exit"]

    def test_entered_strategy_run_reuses_hook_context(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
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
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "hooks": [hook]})
        with strategy:
            strategy.run([batch])
        assert events == ["enter", "call", "exit"]

    def test_strategy_context_exposes_named_models(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        seen_keys: list[set[str]] = []

        def _record(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            assert isinstance(ctx, TrainContext)
            seen_keys.append(set(ctx.models))
            assert ctx.model is ctx.models["main"]

        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "hooks": [_RecordingHook(TrainingStage.BEFORE_BATCH, _record)],
            }
        )
        strategy.run([batch])
        assert seen_keys == [{"main"}]

    def test_stage_order_one_batch(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        log: list[Enum] = []
        hooks = [
            _RecordingHook(stage, lambda ctx, s, _log=log: _log.append(s))  # noqa: ARG005
            for stage in _EXPECTED_STAGE_ORDER
        ]
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "hooks": hooks})
        strategy.run([batch])
        assert tuple(log) == _EXPECTED_STAGE_ORDER

    def test_hook_context_loss_lifecycle(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
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
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "hooks": hooks})
        strategy.run([batch])

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
    def test_roundtrip_preserves_declarative_fields(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        loss_fn = EnergyMSELoss(per_atom=True) + ForceMSELoss(
            normalize_by_atom_count=False
        )
        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "optimizer_configs": {
                    "main": [
                        OptimizerConfig(
                            optimizer_cls=torch.optim.Adam,
                            optimizer_kwargs={"lr": 1e-3},
                            scheduler_cls=torch.optim.lr_scheduler.StepLR,
                            scheduler_kwargs={"step_size": 3, "gamma": 0.5},
                        )
                    ]
                },
                "num_epochs": 2,
                "epoch_step_modifier": 0.5,
                "loss_fn": loss_fn,
                "devices": [torch.device("cpu")],
            }
        )
        spec = strategy.to_spec_dict()
        spec_back = json.loads(json.dumps(spec))

        fresh_model = _build_demo_model()
        restored = TrainingStrategy.from_spec_dict(
            spec_back, models=fresh_model, hooks=[]
        )
        assert restored.num_epochs == 2
        assert restored.num_steps is None
        assert restored.epoch_step_modifier == pytest.approx(0.5)
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
        assert isinstance(leaves[0], EnergyMSELoss)
        assert isinstance(leaves[1], ForceMSELoss)
        assert leaves[0].per_atom is True
        assert leaves[1].normalize_by_atom_count is False

    def test_roundtrip_preserves_composed_loss_dtype_policy_set_after_sugar(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        loss_fn = EnergyMSELoss() + ForceMSELoss(normalize_by_atom_count=False)
        loss_fn.dtype_policy = "prediction_to_target"
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "loss_fn": loss_fn})

        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        restored = TrainingStrategy.from_spec_dict(
            spec, models=_build_demo_model(), hooks=[]
        )

        assert restored.loss_fn.dtype_policy == "prediction_to_target"
        assert all(
            component.dtype_policy == "strict"
            for component in restored.loss_fn.components
        )

    def test_roundtrip_preserves_custom_protocol_loss_schedule(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        loss_fn = ComposedLossFunction(
            [EnergyMSELoss()],
            weights=[_RuntimeWeightSchedule(2.5)],
            normalize_weights=False,
        )
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "loss_fn": loss_fn})

        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        restored = TrainingStrategy.from_spec_dict(
            spec, models=_build_demo_model(), hooks=[]
        )

        weight = restored.loss_fn._weights[0]
        assert isinstance(weight, _RuntimeWeightSchedule)
        assert weight.value == pytest.approx(2.5)

    def test_roundtrip_uses_custom_schedule_to_spec(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        loss_fn = ComposedLossFunction(
            [EnergyMSELoss()],
            weights=[_ToSpecWeightSchedule(3.5, expose_attr=False)],
            normalize_weights=False,
        )
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "loss_fn": loss_fn})

        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        restored = TrainingStrategy.from_spec_dict(
            spec, models=_build_demo_model(), hooks=[]
        )

        weight = restored.loss_fn._weights[0]
        assert isinstance(weight, _ToSpecWeightSchedule)
        assert weight(0, 0) == pytest.approx(3.5)

    def test_roundtrip_preserves_loss_weights_and_normalization(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        loss_fn = ComposedLossFunction(
            [
                EnergyMSELoss(),
                ForceMSELoss(normalize_by_atom_count=False),
            ],
            weights=[0.25, LinearWeight(start=0.1, end=0.5, num_steps=10)],
            normalize_weights=False,
        )
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "loss_fn": loss_fn})

        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        restored = TrainingStrategy.from_spec_dict(
            spec, models=_build_demo_model(), hooks=[]
        )

        assert restored.loss_fn.normalize_weights is False
        assert restored.loss_fn._weights[0] == pytest.approx(0.25)
        assert isinstance(restored.loss_fn._weights[1], LinearWeight)
        schedule = restored.loss_fn._weights[1]
        assert schedule.start == pytest.approx(0.1)
        assert schedule.end == pytest.approx(0.5)
        assert schedule.num_steps == 10

    def test_roundtrip_preserves_scaled_loss_weight_schedule(self) -> None:
        schedule = LinearWeight(start=0.2, end=1.0, num_steps=10)
        loss_fn = 0.25 * ComposedLossFunction([EnergyMSELoss()], weights=[schedule])
        strategy = _make_strategy(loss_fn=loss_fn)

        spec = json.loads(json.dumps(strategy.to_spec_dict()))
        restored = TrainingStrategy.from_spec_dict(
            spec, models=_make_demo_model(), hooks=[]
        )

        weight = restored.loss_fn._weights[0]
        assert weight(0, 0) == pytest.approx(0.25 * schedule(0, 0))
        assert weight(5, 0) == pytest.approx(0.25 * schedule(5, 0))

    def test_missing_optimizer_configs_key_raises(
        self, strategy: TrainingStrategy
    ) -> None:
        spec = strategy.to_spec_dict()
        del spec["optimizer_configs"]
        with pytest.raises(ValueError, match="optimizer_configs"):
            TrainingStrategy.from_spec_dict(spec, models=_build_demo_model(), hooks=[])

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
        self, key: str, value: Any, match: str, strategy: TrainingStrategy
    ) -> None:
        spec = strategy.to_spec_dict()
        if value is _DELETE:
            del spec[key]
        else:
            spec[key] = value

        with pytest.raises(ValueError, match=match):
            TrainingStrategy.from_spec_dict(spec, models=_build_demo_model(), hooks=[])

    def test_integer_optimizer_key_migrates_to_main(
        self, strategy: TrainingStrategy
    ) -> None:
        spec = strategy.to_spec_dict()
        original = spec["optimizer_configs"]["main"]
        spec["optimizer_configs"] = {"0": original}
        restored = TrainingStrategy.from_spec_dict(
            spec, models=_build_demo_model(), hooks=[]
        )
        assert set(restored.optimizer_configs) == {"main"}

    def test_single_model_spec_without_runtime_model_restores_single_call_mode(
        self, strategy: TrainingStrategy, batch: Batch
    ) -> None:
        seen_args: list[BaseModelMixin | dict[str, BaseModelMixin]] = []

        def _record_training_fn(
            model: BaseModelMixin, batch: Batch
        ) -> dict[str, torch.Tensor]:
            seen_args.append(model)
            return default_training_fn(strategy.models["main"], batch)

        restored = TrainingStrategy.from_spec_dict(
            strategy.to_spec_dict(), hooks=[], training_fn=_record_training_fn
        )
        restored.train_batch(batch)
        assert seen_args == [restored.models["main"]]

    def test_single_main_named_spec_restores_named_call_mode(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "models": {"main": _build_demo_model()},
                "optimizer_configs": _build_adam_optimizer_configs(),
                "training_fn": mapping_annotated_training_fn,
            }
        )

        spec = strategy.to_spec_dict()
        restored = TrainingStrategy.from_spec_dict(spec, hooks=[])

        assert spec["single_model_input"] is False
        assert restored.single_model_input is False
        restored.run([batch])
        assert restored.step_count == 1

    def test_model_spec_roundtrip_restores_runnable_demo_model(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        strategy = TrainingStrategy(
            **{**baseline_strategy_kwargs, "training_fn": default_training_fn}
        )
        restored = TrainingStrategy.from_spec_dict(strategy.to_spec_dict(), hooks=[])

        assert restored.models["main"] is not strategy.models["main"]
        restored.run([batch])

        assert restored.step_count == 1

    def test_runtime_model_override_merges_over_spec_models(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        spec = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "models": {
                    "main": _build_demo_model(),
                    "teacher": _build_demo_model(),
                },
                "optimizer_configs": {
                    "main": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
                },
                "training_fn": dict_demo_training_fn,
            }
        ).to_spec_dict()
        replacement = _build_demo_model()
        restored = TrainingStrategy.from_spec_dict(spec, models=replacement, hooks=[])
        assert restored.models["main"] is replacement
        assert "teacher" in restored.models
        assert restored.single_model_input is False

    @pytest.mark.parametrize("drop_training_fn", [False, True])
    def test_runtime_training_fn_override(
        self, drop_training_fn: bool, strategy: TrainingStrategy
    ) -> None:
        spec = strategy.to_spec_dict()
        if drop_training_fn:
            del spec["training_fn"]
        restored = TrainingStrategy.from_spec_dict(
            spec,
            models=_build_demo_model(),
            hooks=[],
            training_fn=default_training_fn,
        )
        assert restored.training_fn is default_training_fn

    def test_non_importable_training_fn_warns_and_is_omitted(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        strategy = TrainingStrategy(
            **{**baseline_strategy_kwargs, "training_fn": lambda model, batch: {}}
        )
        with pytest.warns(UserWarning, match="Omitting non-importable training_fn"):
            spec = strategy.to_spec_dict()
        assert "training_fn" not in spec


class TestValidationCapabilities:
    """Phase A introspection methods on TrainingStrategy."""

    def _make_validation_strategy(self, **overrides: Any) -> TrainingStrategy:
        """Build a strategy with a ValidationConfig attached."""
        from nvalchemi.training._validation import ValidationConfig

        batch = _make_batch()
        vc_kwargs = overrides.pop("validation_config_kwargs", {})
        vc = ValidationConfig(validation_data=[batch], **vc_kwargs)
        return _make_strategy(validation_config=vc, **overrides)

    # -- model resolution (via ValidationLoop.from_training_strategy) --

    def test_model_arg_returns_live_model_when_slot_none(self) -> None:
        """No inference_model slot -> live training model."""
        from nvalchemi.training._validation import ValidationLoop

        strategy = self._make_validation_strategy()
        assert strategy.inference_model is None
        loop = ValidationLoop.from_training_strategy(strategy)
        assert loop._model_arg is strategy.models["main"]
        assert loop._modules == (strategy.models["main"],)
        assert loop._ema_model_keys == ()

    def test_model_arg_returns_slot_when_set_single_model(self) -> None:
        """Setting inference_model returns the slot model."""
        from nvalchemi.training._validation import ValidationLoop

        strategy = self._make_validation_strategy()
        replacement = torch.nn.Linear(4, 4)
        strategy.inference_model = replacement
        loop = ValidationLoop.from_training_strategy(strategy)
        assert loop._model_arg is replacement
        assert loop._modules == (replacement,)
        assert loop._ema_model_keys == ("main",)

    def test_set_inference_model_moves_module_to_primary_device(self) -> None:
        """Publishing inference_model preserves identity and aligns device."""
        strategy = self._make_validation_strategy(devices=[torch.device("cpu")])
        replacement = _RecordingLinear()

        strategy.set_inference_model(replacement)

        assert strategy.inference_model is replacement
        assert replacement.to_calls == [
            ((torch.device("cpu"),), {"non_blocking": True})
        ]

    def test_model_arg_moduledict_slot_named_model(self) -> None:
        """ModuleDict slot overrides matching keys; missing keys fall back."""
        from nvalchemi.training._validation import ValidationConfig, ValidationLoop

        teacher = _build_demo_model()
        student = _build_demo_model()
        strategy = _make_strategy(
            models={"student": student, "teacher": teacher},
            optimizer_configs={
                "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
            },
            training_fn=dict_demo_training_fn,
        )
        strategy.validation_config = ValidationConfig(validation_data=[_make_batch()])
        ema_student = torch.nn.Linear(4, 4)
        strategy.inference_model = torch.nn.ModuleDict({"student": ema_student})
        loop = ValidationLoop.from_training_strategy(strategy)
        assert loop._model_arg["student"] is ema_student
        assert loop._model_arg["teacher"] is teacher
        assert loop._ema_model_keys == ("student",)
        assert ema_student in loop._modules

    def test_model_arg_moduledict_missing_key_falls_back(self) -> None:
        """ModuleDict slot missing 'teacher' key -> live teacher model used."""
        from nvalchemi.training._validation import ValidationConfig, ValidationLoop

        teacher = _build_demo_model()
        student = _build_demo_model()
        strategy = _make_strategy(
            models={"student": student, "teacher": teacher},
            optimizer_configs={
                "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
            },
            training_fn=dict_demo_training_fn,
        )
        strategy.validation_config = ValidationConfig(validation_data=[_make_batch()])
        ema_student = torch.nn.Linear(4, 4)
        strategy.inference_model = torch.nn.ModuleDict({"student": ema_student})
        loop = ValidationLoop.from_training_strategy(strategy)
        assert loop._model_arg["teacher"] is teacher
        assert "teacher" not in loop._ema_model_keys

    def test_model_arg_use_ema_always_empty_slot_raises(self) -> None:
        """use_ema='always' with no inference_model slot raises."""
        from nvalchemi.training._validation import ValidationLoop

        strategy = self._make_validation_strategy(
            validation_config_kwargs={"use_ema": "always"},
        )
        assert strategy.inference_model is None
        with pytest.raises(RuntimeError, match="inference_model slot"):
            ValidationLoop.from_training_strategy(strategy)

    def test_model_arg_use_ema_never_ignores_slot(self) -> None:
        """use_ema='never' ignores the slot even if populated."""
        from nvalchemi.training._validation import ValidationLoop

        strategy = self._make_validation_strategy(
            validation_config_kwargs={"use_ema": "never"},
        )
        replacement = torch.nn.Linear(4, 4)
        strategy.inference_model = replacement
        loop = ValidationLoop.from_training_strategy(strategy)
        assert loop._model_arg is strategy.models["main"]
        assert loop._modules == (strategy.models["main"],)
        assert loop._ema_model_keys == ()

    def test_model_arg_use_ema_always_named_missing_raises(self) -> None:
        """use_ema='always' with named models where slot misses a key raises."""
        from nvalchemi.training._validation import ValidationConfig, ValidationLoop

        teacher = _build_demo_model()
        student = _build_demo_model()
        strategy = _make_strategy(
            models={"student": student, "teacher": teacher},
            optimizer_configs={
                "student": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
            },
            training_fn=dict_demo_training_fn,
        )
        strategy.validation_config = ValidationConfig(
            validation_data=[_make_batch()], use_ema="always"
        )
        strategy.inference_model = torch.nn.ModuleDict(
            {"student": torch.nn.Linear(4, 4)}
        )
        with pytest.raises(RuntimeError, match="missing"):
            ValidationLoop.from_training_strategy(strategy)

    # -- _inference_autocast --

    def test_inference_autocast_no_hook_returns_float32(self) -> None:
        """No MixedPrecisionHook -> (nullcontext, 'float32')."""
        from contextlib import nullcontext

        strategy = self._make_validation_strategy()
        factory, precision = strategy._inference_autocast(torch.device("cpu"))
        assert factory is nullcontext
        assert precision == "float32"

    def test_inference_autocast_with_mixed_precision_hook(self) -> None:
        """MixedPrecisionHook registered -> its autocast + precision label."""
        from nvalchemi.training.hooks.mixed_precision import MixedPrecisionHook

        mp = MixedPrecisionHook(precision=torch.bfloat16)
        strategy = self._make_validation_strategy(hooks=[mp])
        factory, precision = strategy._inference_autocast(torch.device("cpu"))
        assert precision == "bfloat16"
        ctx = factory()
        assert ctx is not None

    def test_inference_autocast_never_ignores_hook(self) -> None:
        """use_mixed_precision='never' ignores registered MixedPrecisionHook."""
        from contextlib import nullcontext

        from nvalchemi.training.hooks.mixed_precision import MixedPrecisionHook

        mp = MixedPrecisionHook(precision=torch.bfloat16)
        strategy = self._make_validation_strategy(
            hooks=[mp],
            validation_config_kwargs={"use_mixed_precision": "never"},
        )
        factory, precision = strategy._inference_autocast(torch.device("cpu"))
        assert factory is nullcontext
        assert precision == "float32"

    def test_inference_autocast_always_no_hook_raises(self) -> None:
        """use_mixed_precision='always' without hook raises."""
        strategy = self._make_validation_strategy(
            validation_config_kwargs={"use_mixed_precision": "always"},
        )
        with pytest.raises(RuntimeError, match="MixedPrecisionHook"):
            strategy._inference_autocast(torch.device("cpu"))

    # -- grad resolution (via ValidationLoop.from_training_strategy) --

    def test_resolve_grad_enabled(self) -> None:
        """grad_mode='enabled' returns True."""
        from nvalchemi.training._validation import ValidationLoop

        strategy = self._make_validation_strategy(
            validation_config_kwargs={"grad_mode": "enabled"},
        )
        loop = ValidationLoop.from_training_strategy(strategy)
        assert loop._grad_enabled is True

    def test_resolve_grad_disabled(self) -> None:
        """grad_mode='disabled' returns False."""
        from nvalchemi.training._validation import ValidationLoop

        strategy = self._make_validation_strategy(
            validation_config_kwargs={"grad_mode": "disabled"},
        )
        loop = ValidationLoop.from_training_strategy(strategy)
        assert loop._grad_enabled is False

    def test_resolve_grad_auto_with_force_loss(self) -> None:
        """grad_mode='auto' with ForceMSELoss (requires_eval_grad=True) returns True."""
        from nvalchemi.training._validation import ValidationLoop

        strategy = self._make_validation_strategy(
            loss_fn=ForceMSELoss(),
        )
        loop = ValidationLoop.from_training_strategy(strategy)
        assert loop._grad_enabled is True

    def test_resolve_grad_auto_with_energy_loss(self) -> None:
        """grad_mode='auto' with EnergyMSELoss (requires_eval_grad=False) returns False."""
        from nvalchemi.training._validation import ValidationLoop

        strategy = self._make_validation_strategy(
            loss_fn=EnergyMSELoss(),
        )
        loop = ValidationLoop.from_training_strategy(strategy)
        assert loop._grad_enabled is False

    def test_resolve_grad_auto_unknown_component_raises(self) -> None:
        """grad_mode='auto' with requires_eval_grad=None raises ValueError."""
        from nvalchemi.training._validation import ValidationLoop
        from nvalchemi.training.losses.composition import BaseLossFunction

        class _AmbiguousLoss(BaseLossFunction):
            requires_eval_grad = None

            def compute_residual(
                self, pred: torch.Tensor, target: torch.Tensor
            ) -> torch.Tensor:
                return pred - target

        strategy = self._make_validation_strategy(
            loss_fn=_AmbiguousLoss(),
        )
        with pytest.raises(ValueError, match="infer whether"):
            ValidationLoop.from_training_strategy(strategy)

    # -- loss resolution (via ValidationLoop.from_training_strategy) --

    def test_resolve_loss_fn_uses_config_loss(self) -> None:
        """When validation_config.loss_fn is set, use it."""
        from nvalchemi.training._validation import ValidationLoop

        val_loss = EnergyMSELoss()
        strategy = self._make_validation_strategy(
            validation_config_kwargs={"loss_fn": val_loss},
        )
        loop = ValidationLoop.from_training_strategy(strategy)
        assert isinstance(loop._loss_fn, ComposedLossFunction)
        assert isinstance(loop._loss_fn.components[0], EnergyMSELoss)

    def test_resolve_loss_fn_falls_back_to_strategy(self) -> None:
        """When validation_config.loss_fn is None, use strategy.loss_fn."""
        from nvalchemi.training._validation import ValidationLoop

        strategy = self._make_validation_strategy()
        loop = ValidationLoop.from_training_strategy(strategy)
        assert loop._loss_fn is not None
        assert len(loop._loss_fn.components) == len(strategy.loss_fn.components)

    # -- last_validation field --

    def test_last_validation_roundtrips(self) -> None:
        """last_validation is None by default and stores assigned values."""
        strategy = _make_strategy()
        assert strategy.last_validation is None
        strategy.last_validation = {"test": 1}
        assert strategy.last_validation == {"test": 1}


class TestValidationSchedule:
    """Phase C: validation checkpoint wiring into run()."""

    @staticmethod
    def _make_schedule_strategy(
        *,
        every_n_epochs: int | None = None,
        every_n_steps: int | None = None,
        num_epochs: int | None = None,
        num_steps: int | None = None,
        hooks: list[Any] | None = None,
    ) -> TrainingStrategy:
        """Build a strategy with a ValidationConfig attached for schedule tests."""
        from nvalchemi.training._validation import ValidationConfig

        overrides: dict[str, Any] = {}
        if num_epochs is not None:
            overrides["num_epochs"] = num_epochs
        if num_steps is not None:
            overrides["num_epochs"] = None
            overrides["num_steps"] = num_steps
        if hooks is not None:
            overrides["hooks"] = hooks
        val_data = [_make_batch()]
        vc = ValidationConfig(
            validation_data=val_data,
            every_n_epochs=every_n_epochs,
            every_n_steps=every_n_steps,
        )
        return _make_strategy(validation_config=vc, **overrides)

    # -- every_n_epochs --

    def test_every_n_epochs_fires_at_correct_boundaries(self) -> None:
        """Validation fires after epochs 1 and 2, plus the end-of-run pass."""
        strategy = self._make_schedule_strategy(every_n_epochs=1, num_epochs=2)
        validate_epochs: list[int] = []
        orig_validate = TrainingStrategy.validate

        def _recording_validate(self_: Any) -> Any:
            validate_epochs.append(self_.epoch_count)
            return orig_validate(self_)

        dataset = _make_dataset(n_batches=2)
        with patch.object(TrainingStrategy, "validate", _recording_validate):
            strategy.run(dataset)
        # Scheduled epochs 1 and 2, then the unconditional end-of-run pass.
        assert validate_epochs == [1, 2, 2]
        assert strategy.last_validation is not None

    def test_every_n_epochs_skips_intermediate(self) -> None:
        """every_n_epochs=2: fires after epoch 2 (not 1), plus end-of-run."""
        strategy = self._make_schedule_strategy(every_n_epochs=2, num_epochs=3)
        validate_epochs: list[int] = []
        orig_validate = TrainingStrategy.validate

        def _recording_validate(self_: Any) -> Any:
            validate_epochs.append(self_.epoch_count)
            return orig_validate(self_)

        dataset = _make_dataset(n_batches=2)
        with patch.object(TrainingStrategy, "validate", _recording_validate):
            strategy.run(dataset)
        # Scheduled at epoch 2 only, then the unconditional end-of-run pass
        # at the final epoch (3).
        assert validate_epochs == [2, 3]

    def test_every_n_epochs_freshness_flag(self) -> None:
        """_validation_checkpoint returns True only after validation-firing epochs."""
        strategy = self._make_schedule_strategy(every_n_epochs=2, num_epochs=2)
        checkpoint_results: list[tuple[int, bool]] = []
        orig_checkpoint = TrainingStrategy._validation_checkpoint

        def _recording_checkpoint(self_: Any, stage: Any) -> Any:
            result = orig_checkpoint(self_, stage)
            if stage is TrainingStage.AFTER_EPOCH:
                checkpoint_results.append((self_.epoch_count, result))
            return result

        dataset = _make_dataset(n_batches=2)
        with patch.object(
            TrainingStrategy, "_validation_checkpoint", _recording_checkpoint
        ):
            strategy.run(dataset)
        # Epoch 1: no validation (2%2!=0), False; epoch 2: validation, True
        assert checkpoint_results == [(1, False), (2, True)]

    # -- every_n_steps --

    def test_every_n_steps_fires_at_correct_steps(self) -> None:
        """every_n_steps=2 fires at step_count 2 and 4."""
        strategy = self._make_schedule_strategy(every_n_steps=2, num_steps=5)
        validate_steps: list[int] = []
        orig_validate = TrainingStrategy.validate

        def _recording_validate(self_: Any) -> Any:
            validate_steps.append(self_.step_count)
            return orig_validate(self_)

        dataset = _make_dataset(n_batches=10)
        with patch.object(TrainingStrategy, "validate", _recording_validate):
            strategy.run(dataset)
        # Scheduled at steps 2 and 4, then the unconditional end-of-run pass
        # at the final step (5).
        assert validate_steps == [2, 4, 5]

    def test_every_n_steps_freshness_toggles(self) -> None:
        """_validation_checkpoint returns True only on step boundaries."""
        strategy = self._make_schedule_strategy(every_n_steps=3, num_steps=4)
        checkpoint_results: list[tuple[int, bool]] = []
        orig_checkpoint = TrainingStrategy._validation_checkpoint

        def _recording_checkpoint(self_: Any, stage: Any) -> Any:
            result = orig_checkpoint(self_, stage)
            if stage is TrainingStage.AFTER_OPTIMIZER_STEP:
                checkpoint_results.append((self_.step_count, result))
            return result

        dataset = _make_dataset(n_batches=10)
        with patch.object(
            TrainingStrategy, "_validation_checkpoint", _recording_checkpoint
        ):
            strategy.run(dataset)
        # step 3 fires validation (True); steps 1, 2, 4 are False
        assert checkpoint_results == [(1, False), (2, False), (3, True), (4, False)]

    # -- ordering: validate() runs AFTER EMA publishes --

    def test_step_cadence_validate_runs_after_ema_publish(self) -> None:
        """validate() reads inference_model AFTER EMA hook publishes at AFTER_OPTIMIZER_STEP."""
        from nvalchemi.training.hooks import EMAHook

        ema = EMAHook(model_key="main", decay=0.0, start_step=0)
        strategy = self._make_schedule_strategy(
            every_n_steps=1, num_steps=1, hooks=[ema]
        )
        inference_model_at_validate: list[torch.nn.Module | None] = []
        orig_validate = TrainingStrategy.validate

        def _recording_validate(self_: Any) -> Any:
            inference_model_at_validate.append(self_.inference_model)
            return orig_validate(self_)

        dataset = _make_dataset(n_batches=2)
        with patch.object(TrainingStrategy, "validate", _recording_validate):
            strategy.run(dataset)
        # Scheduled at step 1, then the unconditional end-of-run pass.
        assert len(inference_model_at_validate) == 2
        # EMA should have published a module before validate was called.
        assert all(model is not None for model in inference_model_at_validate)

    # -- no validation_config --

    def test_no_validation_config_does_nothing(self) -> None:
        """No validation_config: _validation_checkpoint returns False, run() works."""
        strategy = _make_strategy()
        assert strategy.validation_config is None
        dataset = _make_dataset(n_batches=2)
        strategy.run(dataset)
        assert (
            strategy._validation_checkpoint(TrainingStage.AFTER_OPTIMIZER_STEP) is False
        )
        assert strategy.last_validation is None

    # -- last_validation populated --

    def test_last_validation_populated_after_schedule(self) -> None:
        """After scheduled validation, last_validation has data."""
        strategy = self._make_schedule_strategy(every_n_steps=1, num_steps=1)
        dataset = _make_dataset(n_batches=2)
        strategy.run(dataset)
        assert strategy.last_validation is not None
        assert isinstance(strategy.last_validation, dict)

    # -- unconditional end-of-run validation --

    def test_validation_always_runs_at_end_off_boundary(self) -> None:
        """A validation_config always validates at end-of-run, even off boundary."""
        strategy = self._make_schedule_strategy(every_n_steps=3, num_steps=2)
        validate_steps: list[int] = []
        orig_validate = TrainingStrategy.validate

        def _recording_validate(self_: Any) -> Any:
            validate_steps.append(self_.step_count)
            return orig_validate(self_)

        dataset = _make_dataset(n_batches=10)
        with patch.object(TrainingStrategy, "validate", _recording_validate):
            strategy.run(dataset)
        # No in-loop checkpoint fires (step 2 is not a multiple of 3); only the
        # unconditional end-of-run pass at the final step (2) runs.
        assert validate_steps == [2]
        assert strategy.last_validation is not None

    # -- AFTER_VALIDATION hook --

    def test_after_validation_hook_fires_with_live_summary(self) -> None:
        """AFTER_VALIDATION hooks observe the live summary before it is consumed."""
        strategy = self._make_schedule_strategy(every_n_steps=1, num_steps=1)
        observed: list[dict[str, Any] | None] = []

        def _record(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            observed.append(ctx.validation)

        strategy.register_hook(_RecordingHook(TrainingStage.AFTER_VALIDATION, _record))
        dataset = _make_dataset(n_batches=2)
        strategy.run(dataset)

        # Fired at the step-1 checkpoint and the unconditional end-of-run pass.
        assert len(observed) == 2
        assert all(summary is not None for summary in observed)
        assert all("total_loss" in summary for summary in observed)


class TestMetricSchedulerStepping:
    """Phase D: ReduceLROnPlateau steps only at validation checkpoints."""

    @staticmethod
    def _make_metric_strategy(
        *,
        every_n_steps: int | None = None,
        every_n_epochs: int | None = None,
        num_steps: int | None = None,
        num_epochs: int | None = None,
        plateau_patience: int = 1,
        plateau_factor: float = 0.5,
        plateau_lr: float = 0.1,
        add_time_based: bool = False,
    ) -> TrainingStrategy:
        """Build a strategy with a ReduceLROnPlateau scheduler and ValidationConfig."""
        from nvalchemi.training._validation import ValidationConfig

        opt_cfgs: list[OptimizerConfig] = [
            OptimizerConfig(
                optimizer_cls=torch.optim.SGD,
                optimizer_kwargs={"lr": plateau_lr},
                scheduler_cls=torch.optim.lr_scheduler.ReduceLROnPlateau,
                scheduler_kwargs={
                    "patience": plateau_patience,
                    "factor": plateau_factor,
                    "threshold": 0.0,
                },
            ),
        ]
        if add_time_based:
            opt_cfgs.append(
                OptimizerConfig(
                    optimizer_cls=torch.optim.SGD,
                    optimizer_kwargs={"lr": 0.5},
                    scheduler_cls=torch.optim.lr_scheduler.StepLR,
                    scheduler_kwargs={"step_size": 1, "gamma": 0.9},
                ),
            )
        overrides: dict[str, Any] = {
            "optimizer_configs": {"main": opt_cfgs},
        }
        if num_epochs is not None:
            overrides["num_epochs"] = num_epochs
        if num_steps is not None:
            overrides["num_epochs"] = None
            overrides["num_steps"] = num_steps
        val_data = [_make_batch()]
        vc = ValidationConfig(
            validation_data=val_data,
            every_n_steps=every_n_steps,
            every_n_epochs=every_n_epochs,
        )
        return _make_strategy(validation_config=vc, **overrides)

    def test_plateau_steps_at_validation_checkpoints(self) -> None:
        """ReduceLROnPlateau.step() is called at validation checkpoints only."""
        # every_n_steps=1 with 3 steps: checkpoint at steps 1, 2, 3 + end-of-run
        strategy = self._make_metric_strategy(
            every_n_steps=1,
            num_steps=3,
            plateau_patience=0,
            plateau_factor=0.5,
        )
        dataset = _make_dataset(n_batches=5)

        plateau_step_calls: list[int] = []
        orig_checkpoint = TrainingStrategy._validation_checkpoint

        def _recording_checkpoint(self_: Any, stage: Any) -> Any:
            result = orig_checkpoint(self_, stage)
            if result:
                plateau_step_calls.append(self_.step_count)
            return result

        with patch.object(
            TrainingStrategy, "_validation_checkpoint", _recording_checkpoint
        ):
            strategy.run(dataset)

        # Validation checkpoints fire at steps 1, 2, 3
        assert plateau_step_calls == [1, 2, 3]
        # With patience=0 the LR drops on every validation checkpoint
        # where metric doesn't improve. The plateau scheduler was stepped
        # at each checkpoint, so LR should have dropped.
        final_lr = strategy._optimizers[0].param_groups[0]["lr"]
        assert final_lr < 0.1

    def test_plateau_not_stepped_between_checkpoints(self) -> None:
        """Between validation checkpoints, ReduceLROnPlateau is NOT stepped."""
        strategy = self._make_metric_strategy(
            every_n_steps=3,
            num_steps=4,
            plateau_patience=10,
        )
        dataset = _make_dataset(n_batches=10)
        lr_at_each_step: list[float] = []

        orig_train = TrainingStrategy._train_batch_with_optimizers

        def _recording_train(self_: Any, batch: Any, opts: Any, scheds: Any) -> Any:
            result = orig_train(self_, batch, opts, scheds)
            lr_at_each_step.append(opts[0].param_groups[0]["lr"])
            return result

        with patch.object(
            TrainingStrategy, "_train_batch_with_optimizers", _recording_train
        ):
            strategy.run(dataset)

        # LR should be constant at all steps (patience=10 means no drop)
        assert all(lr == pytest.approx(lr_at_each_step[0]) for lr in lr_at_each_step)

    def test_last_validation_consumed_after_checkpoint(self) -> None:
        """last_validation is None after a checkpoint consumes it."""
        strategy = self._make_metric_strategy(
            every_n_steps=1,
            num_steps=2,
            plateau_patience=10,
        )
        dataset = _make_dataset(n_batches=5)

        post_checkpoint_states: list[bool] = []
        orig_checkpoint = TrainingStrategy._validation_checkpoint

        def _recording_checkpoint(self_: Any, stage: Any) -> Any:
            result = orig_checkpoint(self_, stage)
            if result:
                # After _validation_checkpoint, last_validation should be consumed
                post_checkpoint_states.append(self_.last_validation is None)
            return result

        with patch.object(
            TrainingStrategy, "_validation_checkpoint", _recording_checkpoint
        ):
            strategy.run(dataset)

        # Each checkpoint should have consumed last_validation
        assert len(post_checkpoint_states) >= 1
        assert all(post_checkpoint_states)

    def test_time_based_scheduler_step_count_unchanged(self) -> None:
        """A time-based StepLR scheduler steps every optimizer step, unchanged by metric support."""
        strategy = self._make_metric_strategy(
            every_n_steps=2,
            num_steps=4,
            plateau_patience=10,
            add_time_based=True,
        )
        dataset = _make_dataset(n_batches=10)
        strategy.run(dataset)

        # The second optimizer (StepLR with gamma=0.9, step_size=1) should
        # have stepped every optimizer step. After 4 steps: lr = 0.5 * 0.9^4
        steplr_opt = strategy._optimizers[1]
        expected_lr = 0.5 * (0.9**4)
        actual_lr = steplr_opt.param_groups[0]["lr"]
        assert actual_lr == pytest.approx(expected_lr, rel=1e-5)

    def test_plateau_lr_drops_with_constant_loss(self) -> None:
        """E2E: plateau scheduler drops LR when validation loss plateaus."""
        # patience=1, factor=0.5: after 2 consecutive non-improving
        # validations, LR drops. With every_n_steps=1 and num_steps=4,
        # validation fires at steps 1,2,3,4 + end-of-run. The loss is
        # deterministic (same val data, same model), so it plateaus.
        strategy = self._make_metric_strategy(
            every_n_steps=1,
            num_steps=4,
            plateau_patience=1,
            plateau_factor=0.5,
            plateau_lr=0.01,
        )
        dataset = _make_dataset(n_batches=8)
        initial_lr = 0.01
        strategy.run(dataset)

        final_lr = strategy._optimizers[0].param_groups[0]["lr"]
        # With patience=1, the LR should have dropped at least once
        assert final_lr < initial_lr
