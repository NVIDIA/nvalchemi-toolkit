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

from nvalchemi.data import Batch
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import (
    ComposedLossFunction,
    EnergyLoss,
    ForceLoss,
    TrainingStage,
)
from nvalchemi.training.optimizers import OptimizerConfig
from nvalchemi.training.strategy import TrainingStrategy, default_training_fn
from test.training.conftest import _build_dataset, _build_demo_model


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


def single_model_training_fn(
    model: BaseModelMixin, batch: Batch
) -> dict[str, torch.Tensor]:
    """Single-model training function for validation tests."""
    return demo_training_fn(model, batch)


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
        "models must contain",
        {"models": {}, "optimizer_configs": {}},
    ),
    (
        r"optimizer_configs\[main\] must contain",
        {"optimizer_configs": {"main": []}},
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
        "Exactly one of num_epochs or num_steps",
        {"num_epochs": 1, "num_steps": 1},
    ),
    (
        "Exactly one of num_epochs or num_steps",
        {"num_epochs": None, "num_steps": None},
    ),
    ("num_epochs must be positive", {"num_epochs": -1}),
    (
        "no attribute",
        {"training_fn": "nvalchemi.training.strategy.not_a_real_fn"},
    ),
]


class TestTrainingStrategyValidators:
    @pytest.mark.parametrize(
        ("match", "overrides"),
        _VALIDATOR_REJECTION_CASES,
        ids=[
            "empty_models",
            "empty_per_model_list",
            "optimizer_key_missing",
            "devices_wrong_length",
            "both_num_epochs_and_num_steps",
            "neither_num_epochs_nor_num_steps",
            "negative_num_epochs",
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
            **{**baseline_strategy_kwargs, "loss_fn": EnergyLoss()}
        )
        assert isinstance(strategy.loss_fn, ComposedLossFunction)
        assert len(strategy.loss_fn.components) == 1
        assert isinstance(strategy.loss_fn.components[0], EnergyLoss)

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
            ValueError, match="Dict-model training with multiple devices"
        ):
            strategy.run([batch])

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
        loss_fn = EnergyLoss(per_atom=True) + ForceLoss(normalize_by_atom_count=False)
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
        assert restored.devices == [torch.device("cpu")]
        assert restored.training_fn is demo_training_fn
        assert "main" in spec["model_specs"]
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

    def test_missing_optimizer_configs_key_raises(
        self, strategy: TrainingStrategy
    ) -> None:
        spec = strategy.to_spec_dict()
        del spec["optimizer_configs"]
        with pytest.raises(ValueError, match="optimizer_configs"):
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
        restored._train_one_batch(batch, [], [])
        assert seen_args == [restored.models["main"]]

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


class TestHookContextCaching:
    def test_ctx_built_once_per_batch(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        observed_ctx: list[HookContext] = []

        def _record_ctx(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            observed_ctx.append(ctx)

        # Hooks on every in-batch stage; all should receive the SAME ctx
        # object because the strategy caches it for the batch window.
        in_batch_stages = (
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
        )
        hooks = [_RecordingHook(stage, _record_ctx) for stage in in_batch_stages]
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "hooks": hooks})
        strategy.run([batch])
        # Load-bearing invariant: every hook in the same batch window must
        # see the same HookContext instance so in-place mutations by
        # ``_update_hook_snapshot`` (live→detached loss) and by earlier
        # hooks are visible to later hooks within the batch.
        assert len(observed_ctx) == len(in_batch_stages)
        first_id = id(observed_ctx[0])
        assert all(id(ctx) == first_id for ctx in observed_ctx), (
            "All in-batch hooks must observe the same cached HookContext; "
            "mismatched ids indicate the per-batch cache was rebuilt mid-batch."
        )

    def test_ctx_cleared_after_batch(
        self, strategy: TrainingStrategy, batch: Batch
    ) -> None:
        strategy.run([batch])
        assert strategy._ctx is None


class TestHookContextPopulation:
    def test_ctx_workflow_is_strategy(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        seen: list[object] = []

        def _record(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            seen.append(ctx.workflow)

        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "hooks": [_RecordingHook(TrainingStage.BEFORE_BATCH, _record)],
            }
        )
        strategy.run([batch])
        assert seen == [strategy]

    def test_ctx_optimizers_populated(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        captured: list[list[torch.optim.Optimizer]] = []

        def _record(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            captured.append(ctx.optimizers)

        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "hooks": [_RecordingHook(TrainingStage.BEFORE_OPTIMIZER_STEP, _record)],
            }
        )
        strategy.run([batch])
        assert len(captured) == 1
        assert len(captured[0]) == 1
        assert captured[0] is strategy._flat_opts

    def test_ctx_lr_schedulers_populated(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        captured: list[list[object]] = []

        def _record(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            captured.append(ctx.lr_schedulers)

        strategy = TrainingStrategy(
            **{
                **baseline_strategy_kwargs,
                "optimizer_configs": {
                    "main": [
                        OptimizerConfig(
                            optimizer_cls=torch.optim.Adam,
                            scheduler_cls=torch.optim.lr_scheduler.StepLR,
                            scheduler_kwargs={"step_size": 1},
                        )
                    ]
                },
                "hooks": [_RecordingHook(TrainingStage.BEFORE_OPTIMIZER_STEP, _record)],
            }
        )
        strategy.run([batch])
        assert len(captured) == 1
        assert captured[0] is strategy._flat_scheds
        assert isinstance(captured[0][0], torch.optim.lr_scheduler.StepLR)


class TestLiveDetachedLossPreserved:
    def test_before_backward_live_after_backward_detached(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        records: dict[Enum, bool] = {}

        def _record_requires_grad(ctx: HookContext, stage: Enum) -> None:
            # ``grad_fn is None`` is the most robust signal of detachment.
            records[stage] = ctx.loss is not None and ctx.loss.grad_fn is not None

        hooks = [
            _RecordingHook(TrainingStage.BEFORE_BACKWARD, _record_requires_grad),
            _RecordingHook(TrainingStage.AFTER_BACKWARD, _record_requires_grad),
        ]
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "hooks": hooks})
        strategy.run([batch])

        assert records[TrainingStage.BEFORE_BACKWARD] is True
        assert records[TrainingStage.AFTER_BACKWARD] is False


class _RunsOnStageHook:
    """Minimal hook that claims one or more stages via ``_runs_on_stage``.

    Used by DO_ exclusivity and dispatch tests as a stand-in for the
    future ``MixedPrecisionHook`` which will span multiple stages.
    """

    def __init__(
        self,
        claimed: set[TrainingStage],
        callback: Callable[[HookContext, Enum], None] | None = None,
    ) -> None:
        self._claimed = set(claimed)
        self.frequency = 1
        self.stage = None
        self._callback = callback
        self.calls: list[TrainingStage] = []

    def _runs_on_stage(self, stage: Enum) -> bool:
        return stage in self._claimed

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        self.calls.append(stage)
        if self._callback is not None:
            self._callback(ctx, stage)


class TestDOStageExclusivity:
    def test_single_do_backward_hook_allowed(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        hook = _RunsOnStageHook({TrainingStage.DO_BACKWARD})
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "hooks": [hook]})
        assert strategy._has_do_backward_claim is True
        assert strategy._has_do_optimizer_step_claim is False

    def test_two_do_backward_hooks_rejected(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        h1 = _RunsOnStageHook({TrainingStage.DO_BACKWARD})
        h2 = _RunsOnStageHook({TrainingStage.DO_BACKWARD})
        with pytest.raises(ValueError, match="DO_BACKWARD") as exc_info:
            TrainingStrategy(**{**baseline_strategy_kwargs, "hooks": [h1, h2]})
        assert "_RunsOnStageHook" in str(exc_info.value)

    def test_single_do_optimizer_step_hook_allowed(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        hook = _RunsOnStageHook({TrainingStage.DO_OPTIMIZER_STEP})
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "hooks": [hook]})
        assert strategy._has_do_optimizer_step_claim is True
        assert strategy._has_do_backward_claim is False

    def test_two_do_optimizer_step_hooks_rejected(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        h1 = _RunsOnStageHook({TrainingStage.DO_OPTIMIZER_STEP})
        h2 = _RunsOnStageHook({TrainingStage.DO_OPTIMIZER_STEP})
        with pytest.raises(ValueError, match="DO_OPTIMIZER_STEP") as exc_info:
            TrainingStrategy(**{**baseline_strategy_kwargs, "hooks": [h1, h2]})
        assert "_RunsOnStageHook" in str(exc_info.value)

    def test_no_claim_flags_false_by_default(self, strategy: TrainingStrategy) -> None:
        assert strategy._has_do_backward_claim is False
        assert strategy._has_do_optimizer_step_claim is False

    def test_hook_claims_via_stage_field(
        self, baseline_strategy_kwargs: dict[str, Any]
    ) -> None:
        hook = _RecordingHook(TrainingStage.DO_BACKWARD, lambda ctx, stage: None)
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "hooks": [hook]})
        assert strategy._has_do_backward_claim is True


class TestDODispatch:
    def test_default_backward_runs_when_unclaimed(
        self, strategy: TrainingStrategy, batch: Batch
    ) -> None:
        strategy.run([batch])
        # Default backward ran → at least one model parameter has a grad.
        assert any(
            p.grad is not None and torch.any(p.grad != 0)
            for p in strategy.models["main"].parameters()
        )

    def test_default_backward_skipped_when_claimed(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        hook = _RunsOnStageHook({TrainingStage.DO_BACKWARD})
        # Also claim DO_OPTIMIZER_STEP to avoid stepping on uninitialized grads.
        hook._claimed.add(TrainingStage.DO_OPTIMIZER_STEP)
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "hooks": [hook]})
        strategy.run([batch])
        # Hook was invoked for DO_BACKWARD at least once.
        assert TrainingStage.DO_BACKWARD in hook.calls
        # Since the hook did NOT call .backward(), no parameter should have a grad.
        assert all(
            p.grad is None or torch.all(p.grad == 0)
            for p in strategy.models["main"].parameters()
        )

    def test_default_step_runs_when_unclaimed(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        # Baseline: the default optimizer step updates every trainable
        # parameter by roughly ``lr`` (1e-3). We compare snapshots against
        # post-step values; ``embedding.weight`` is excluded because
        # DemoModel mutates it lazily on first forward (unrelated to the
        # optimizer step).
        snapshots: list[tuple[str, torch.Tensor]] = []

        def _snapshot(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            snapshots.extend(
                (name, p.detach().clone())
                for name, p in ctx.models["main"].named_parameters()
            )

        snapshotter = _RecordingHook(TrainingStage.BEFORE_BATCH, _snapshot)
        strategy = TrainingStrategy(
            **{**baseline_strategy_kwargs, "hooks": [snapshotter]}
        )
        strategy.run([batch])
        after = dict(strategy.models["main"].named_parameters())
        changed = [
            name
            for name, before in snapshots
            if name != "model.embedding.weight" and not torch.equal(before, after[name])
        ]
        # With the default step, every non-embedding param should move.
        assert len(changed) == len(snapshots) - 1

    def test_default_step_skipped_when_claimed(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        # When a hook claims DO_OPTIMIZER_STEP and does nothing, no
        # trainable parameter should change value (apart from the lazy
        # ``embedding.weight`` init described in the sibling test).
        claim = _RunsOnStageHook({TrainingStage.DO_OPTIMIZER_STEP})
        snapshots: list[tuple[str, torch.Tensor]] = []

        def _snapshot(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            snapshots.extend(
                (name, p.detach().clone())
                for name, p in ctx.models["main"].named_parameters()
            )

        snapshotter = _RecordingHook(TrainingStage.BEFORE_BATCH, _snapshot)
        strategy = TrainingStrategy(
            **{**baseline_strategy_kwargs, "hooks": [claim, snapshotter]}
        )
        strategy.run([batch])
        after = dict(strategy.models["main"].named_parameters())
        changed = [
            name
            for name, before in snapshots
            if name != "model.embedding.weight" and not torch.equal(before, after[name])
        ]
        assert changed == []
        assert TrainingStage.DO_OPTIMIZER_STEP in claim.calls
