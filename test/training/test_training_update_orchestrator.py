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
"""Tests for ``TrainingUpdateHook`` and ``TrainingUpdateOrchestrator``.

Covers the hook framework defined in ``nvalchemi.training.hooks.update`` and
its integration with :class:`TrainingStrategy` (auto-wrap, conflict
detection, dispatch-driven training-loop suppression). The strategy-level
helpers (``demo_training_fn``, ``_make_demo_model`` etc.) are duplicated
locally rather than imported from ``test_strategy`` to keep these tests
self-contained and immune to pytest collection ordering.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock, patch

import pydantic
import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks._context import TrainContext
from nvalchemi.hooks._protocol import Hook
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    TrainingStage,
)
from nvalchemi.training.hooks import (
    TrainingUpdateHook,
    TrainingUpdateOrchestrator,
)
from nvalchemi.training.hooks.update import (
    _MULTIPLE_ORCHESTRATOR_MSG,
    _check_veto,
    _fold_training_update_hooks,
    _hook_claims_stage,
)
from nvalchemi.training.optimizers import OptimizerConfig
from nvalchemi.training.strategy import TrainingStrategy, default_training_fn

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

_UPDATE_STAGES: tuple[TrainingStage, ...] = (
    TrainingStage.BEFORE_BATCH,
    TrainingStage.DO_BACKWARD,
    TrainingStage.DO_OPTIMIZER_STEP,
    TrainingStage.AFTER_OPTIMIZER_STEP,
)

_NON_UPDATE_STAGES: tuple[TrainingStage, ...] = tuple(
    s for s in TrainingStage if s not in _UPDATE_STAGES
)

_GATED_STAGES: tuple[TrainingStage, ...] = (
    TrainingStage.BEFORE_BATCH,
    TrainingStage.DO_OPTIMIZER_STEP,
)

_DO_STAGES: tuple[TrainingStage, ...] = (
    TrainingStage.DO_BACKWARD,
    TrainingStage.DO_OPTIMIZER_STEP,
)


def _demo_training_fn(model: BaseModelMixin, batch: Batch) -> dict[str, torch.Tensor]:
    return default_training_fn(model, batch)


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
    return Batch.from_data_list(
        [_make_atomic_data(n_atoms_each, seed=seed + i) for i in range(n_systems)]
    )


def _make_demo_model() -> Any:
    from nvalchemi.models.demo import DemoModel, DemoModelWrapper

    torch.manual_seed(0)
    return DemoModelWrapper(DemoModel(num_atom_types=20, hidden_dim=8))


def _baseline_strategy_kwargs() -> dict[str, Any]:
    return {
        "models": _make_demo_model(),
        "optimizer_configs": OptimizerConfig(optimizer_cls=torch.optim.Adam),
        "num_epochs": 1,
        "training_fn": _demo_training_fn,
        "loss_fn": EnergyMSELoss() + ForceMSELoss(normalize_by_atom_count=True),
    }


def _make_strategy(**overrides: Any) -> TrainingStrategy:
    kwargs = _baseline_strategy_kwargs()
    kwargs.update(overrides)
    return TrainingStrategy(**kwargs)


def _make_ctx(loss: torch.Tensor | None = None) -> TrainContext:
    if loss is None:
        loss = torch.tensor(1.0)
    batch = _make_batch()
    return TrainContext(batch=batch, loss=loss)


def _single_orchestrator(strategy: TrainingStrategy) -> TrainingUpdateOrchestrator:
    orchs = [h for h in strategy.hooks if isinstance(h, TrainingUpdateOrchestrator)]
    assert len(orchs) == 1, f"Expected exactly one orchestrator, found {len(orchs)}"
    return orchs[0]


@contextlib.contextmanager
def _patched_update_helpers():  # type: ignore[no-untyped-def]
    """Patch the orchestrator-side and strategy-side training helpers.

    Yields a ``SimpleNamespace`` with attributes:
    ``orch_zero``, ``orch_step``, ``orch_sched`` for the orchestrator path
    (``nvalchemi.training.hooks.update.*``) and ``strategy_zero``,
    ``strategy_step``, ``strategy_sched`` for the strategy default path
    (``nvalchemi.training.strategy.*``).
    """
    with (
        patch("nvalchemi.training.hooks.update.zero_gradients") as orch_zero,
        patch("nvalchemi.training.hooks.update.step_optimizers") as orch_step,
        patch("nvalchemi.training.hooks.update.step_lr_schedulers") as orch_sched,
        patch("nvalchemi.training.strategy.zero_gradients") as strategy_zero,
        patch("nvalchemi.training.strategy.step_optimizers") as strategy_step,
        patch("nvalchemi.training.strategy.step_lr_schedulers") as strategy_sched,
    ):
        yield SimpleNamespace(
            orch_zero=orch_zero,
            orch_step=orch_step,
            orch_sched=orch_sched,
            strategy_zero=strategy_zero,
            strategy_step=strategy_step,
            strategy_sched=strategy_sched,
        )


@contextlib.contextmanager
def _run_strategy_with_patched_helpers(hooks: list[Any]):  # type: ignore[no-untyped-def]
    """Build a strategy from ``hooks``, train one batch, and yield the mock namespace.

    The helper patches both strategy-side and orchestrator-side update helpers
    while ``train_batch`` runs synchronously, then yields the mocks for
    inspection.
    """
    strategy = _make_strategy(hooks=hooks)
    with _patched_update_helpers() as m:
        strategy.train_batch(_make_batch())
        yield m


# ---------------------------------------------------------------------------
# Hook subclasses for tests
# ---------------------------------------------------------------------------


class _RecordingUpdateHook(TrainingUpdateHook):
    def __init__(self, priority: int = 50) -> None:
        self.priority = priority
        self.calls: list[tuple[TrainingStage, bool]] = []

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, torch.Tensor]:
        self.calls.append((stage, will_skip))
        return True, ctx.loss


class _VetoHook(TrainingUpdateHook):
    def __init__(self, veto_stage: TrainingStage, priority: int = 50) -> None:
        self.priority = priority
        self.veto_stage = veto_stage
        self.calls: list[tuple[TrainingStage, bool]] = []

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, torch.Tensor]:
        self.calls.append((stage, will_skip))
        return stage is not self.veto_stage, ctx.loss


class _BadProceedHook(TrainingUpdateHook):
    def __init__(self, proceed: object, priority: int = 50) -> None:
        self.priority = priority
        self._bad_proceed = proceed

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, torch.Tensor]:
        return self._bad_proceed, ctx.loss  # type: ignore[return-value]


class _LossTransformHook(TrainingUpdateHook):
    def __init__(self, factor: float, priority: int = 50) -> None:
        self.priority = priority
        self.factor = factor

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, torch.Tensor]:
        if stage == TrainingStage.DO_BACKWARD:
            return True, ctx.loss * self.factor
        return True, ctx.loss


class _NoneLossHook(TrainingUpdateHook):
    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, None]:
        return True, None


class _GradScalerSetHook(TrainingUpdateHook):
    """Update hook that writes ``ctx.grad_scaler`` on ``DO_BACKWARD``."""

    priority = 10

    def __init__(self, scaler: object) -> None:
        self._scaler = scaler

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, torch.Tensor]:
        if stage == TrainingStage.DO_BACKWARD:
            ctx.grad_scaler = self._scaler
        return True, ctx.loss


class _GradScalerReadHook(TrainingUpdateHook):
    """Update hook that records ``ctx.grad_scaler`` on ``DO_BACKWARD``."""

    priority = 20

    def __init__(self) -> None:
        self.observed: object | None = None

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, torch.Tensor]:
        if stage == TrainingStage.DO_BACKWARD:
            self.observed = ctx.grad_scaler
        return True, ctx.loss


class _LifecycleUpdateHook(TrainingUpdateHook):
    """Update hook that records lifecycle method calls."""

    def __init__(self, name: str, events: list[str], priority: int = 50) -> None:
        self.name = name
        self.events = events
        self.priority = priority

    def __enter__(self) -> None:
        self.events.append(f"enter:{self.name}")

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.events.append(f"exit:{self.name}")


class _CloseOnlyUpdateHook(TrainingUpdateHook):
    """Update hook that records ``close`` calls."""

    def __init__(self, name: str, events: list[str], priority: int = 50) -> None:
        self.name = name
        self.events = events
        self.priority = priority

    def close(self) -> None:
        self.events.append(f"close:{self.name}")


class _AfterOptimizerStepHook(TrainingUpdateHook):
    """Update hook that records ``will_skip`` on ``AFTER_OPTIMIZER_STEP``."""

    def __init__(self, priority: int = 50) -> None:
        self.priority = priority
        self.will_skip_values: list[bool] = []

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, torch.Tensor]:
        if stage == TrainingStage.AFTER_OPTIMIZER_STEP:
            self.will_skip_values.append(will_skip)
        return True, ctx.loss


class _FakeEqHook:
    """Hook-like object whose ``__eq__`` always returns ``True``.

    Used to verify that ``_validate_single_do_claimants`` uses identity
    (``is``) rather than equality when checking whether the candidate hook
    is already in the existing hook list.
    """

    def __init__(self, stage: TrainingStage | None = None) -> None:
        self.stage = stage
        self.frequency = 1

    def __eq__(self, other: object) -> bool:
        return True

    def __hash__(self) -> int:
        return id(self)

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:
        return None


class _StageOnlyHook:
    def __init__(self, stage: TrainingStage) -> None:
        self.stage = stage
        self.frequency = 1

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:
        return None


class _ContextCaptureHook(_StageOnlyHook):
    def __init__(self, stage: TrainingStage) -> None:
        super().__init__(stage)
        self.contexts: list[TrainContext] = []

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:
        self.contexts.append(ctx)


class _RaisingStageHook(_StageOnlyHook):
    def __init__(self, stage: TrainingStage) -> None:
        super().__init__(stage)
        self.enabled = True

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:
        if self.enabled:
            raise RuntimeError("forced hook failure")


class _DoBackwardOwnerHook(_StageOnlyHook):
    def __init__(self) -> None:
        super().__init__(TrainingStage.DO_BACKWARD)
        self.calls = 0

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:
        self.calls += 1
        assert ctx.loss is not None
        ctx.loss.backward()


class _DoOptimizerStepOwnerHook(_StageOnlyHook):
    def __init__(self) -> None:
        super().__init__(TrainingStage.DO_OPTIMIZER_STEP)
        self.calls = 0
        self.contexts: list[TrainContext] = []

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:
        self.calls += 1
        self.contexts.append(ctx)
        for optimizer in ctx.optimizers:
            optimizer.step()
        for scheduler in ctx.lr_schedulers:
            if scheduler is not None:
                scheduler.step()


class _HybridStageRunsOnHook(_StageOnlyHook):
    def _runs_on_stage(self, stage: TrainingStage) -> bool:
        return False


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestTrainingUpdateHookDefaults:
    def test_default_priority_is_fifty(self) -> None:
        assert TrainingUpdateHook.priority == 50

    def test_runs_on_stage_true_for_update_stages(self) -> None:
        hook = TrainingUpdateHook()
        for stage in _UPDATE_STAGES:
            assert hook._runs_on_stage(stage) is True

    def test_runs_on_stage_false_for_non_update_stages(self) -> None:
        hook = TrainingUpdateHook()
        for stage in _NON_UPDATE_STAGES:
            assert hook._runs_on_stage(stage) is False, (
                f"Expected False for {stage.name}, got True."
            )

    def test_default_call_returns_true_and_ctx_loss(self) -> None:
        loss = torch.tensor(3.14)
        ctx = _make_ctx(loss=loss)
        hook = TrainingUpdateHook()
        for stage in _UPDATE_STAGES:
            proceed, returned_loss = hook(ctx, stage, will_skip=False)
            assert proceed is True
            assert returned_loss is loss


class TestAddAlgebra:
    def test_hook_plus_hook_yields_orchestrator(self) -> None:
        a = _RecordingUpdateHook(priority=10)
        b = _RecordingUpdateHook(priority=20)
        result = a + b
        assert isinstance(result, TrainingUpdateOrchestrator)
        assert result._hooks == [a, b]

    def test_hook_plus_orchestrator_yields_flat_orchestrator(self) -> None:
        a = _RecordingUpdateHook(priority=10)
        b = _RecordingUpdateHook(priority=20)
        c = _RecordingUpdateHook(priority=30)
        orch = TrainingUpdateOrchestrator(b, c)
        result = a + orch
        assert isinstance(result, TrainingUpdateOrchestrator)
        assert result._hooks == [a, b, c]
        for inner in result._hooks:
            assert not isinstance(inner, TrainingUpdateOrchestrator)

    def test_orchestrator_plus_hook_yields_flat_orchestrator(self) -> None:
        a = _RecordingUpdateHook(priority=10)
        b = _RecordingUpdateHook(priority=30)
        c = _RecordingUpdateHook(priority=20)
        orch = TrainingUpdateOrchestrator(a, b)
        result = orch + c
        assert isinstance(result, TrainingUpdateOrchestrator)
        assert result._hooks == [a, c, b]

    def test_orchestrator_plus_orchestrator_yields_flat_orchestrator(self) -> None:
        a = _RecordingUpdateHook(priority=10)
        b = _RecordingUpdateHook(priority=40)
        c = _RecordingUpdateHook(priority=20)
        d = _RecordingUpdateHook(priority=30)
        left = TrainingUpdateOrchestrator(a, b)
        right = TrainingUpdateOrchestrator(c, d)
        result = left + right
        assert isinstance(result, TrainingUpdateOrchestrator)
        assert result._hooks == [a, c, d, b]

    def test_constituents_preserve_identity(self) -> None:
        a = _RecordingUpdateHook(priority=10)
        b = _RecordingUpdateHook(priority=20)
        result = a + b
        assert result._hooks[0] is a
        assert result._hooks[1] is b

    def test_hook_plus_int_raises_type_error(self) -> None:
        hook = _RecordingUpdateHook()
        with pytest.raises(TypeError):
            _ = hook + 42  # type: ignore[operator]

    def test_orchestrator_plus_int_raises_type_error(self) -> None:
        orch = TrainingUpdateOrchestrator(_RecordingUpdateHook())
        with pytest.raises(TypeError):
            _ = orch + 42  # type: ignore[operator]

    def test_addition_never_returns_bare_hook(self) -> None:
        a = _RecordingUpdateHook()
        b = _RecordingUpdateHook()
        assert isinstance(a + b, TrainingUpdateOrchestrator)


class TestPriorityOrdering:
    def test_three_hooks_sorted_ascending(self) -> None:
        h_high = _RecordingUpdateHook(priority=30)
        h_low = _RecordingUpdateHook(priority=10)
        h_mid = _RecordingUpdateHook(priority=20)
        orch = TrainingUpdateOrchestrator(h_high, h_low, h_mid)
        ctx = _make_ctx()
        orch(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)
        assert [h.priority for h in orch._hooks] == [10, 20, 30]
        assert orch._hooks == [h_low, h_mid, h_high]
        assert h_low.calls and h_mid.calls and h_high.calls

    def test_stable_sort_preserves_insertion_order_on_ties(self) -> None:
        first = _RecordingUpdateHook(priority=20)
        second = _RecordingUpdateHook(priority=20)
        third = _RecordingUpdateHook(priority=20)
        orch = TrainingUpdateOrchestrator(first, second, third)
        assert orch._hooks == [first, second, third]


class TestUpdateStageDispatch:
    def test_before_batch_calls_zero_gradients_when_proceed(self) -> None:
        hook = _RecordingUpdateHook(priority=10)
        orch = TrainingUpdateOrchestrator(hook)
        ctx = _make_ctx()
        with _patched_update_helpers() as m:
            orch(ctx, TrainingStage.BEFORE_BATCH)
        m.orch_zero.assert_called_once_with(ctx.optimizers)

    def test_before_batch_skips_zero_gradients_on_veto(self) -> None:
        hook = _VetoHook(veto_stage=TrainingStage.BEFORE_BATCH, priority=10)
        orch = TrainingUpdateOrchestrator(hook)
        ctx = _make_ctx()
        with _patched_update_helpers() as m:
            orch(ctx, TrainingStage.BEFORE_BATCH)
        m.orch_zero.assert_not_called()

    def test_do_backward_calls_backward_and_assigns_loss(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0]))
        loss = (param * 3.0).sum()  # dL/dparam = 3 prior to chain
        hook = _LossTransformHook(factor=2.0, priority=10)
        orch = TrainingUpdateOrchestrator(hook)
        ctx = _make_ctx(loss=loss)
        orch(ctx, TrainingStage.DO_BACKWARD)
        assert param.grad is not None
        # Original grad (3.0) scaled by 2.0 = 6.0
        assert param.grad.item() == pytest.approx(6.0)
        # ctx.loss is replaced with the transformed scalar tensor.
        assert ctx.loss is not loss

    def test_do_optimizer_step_calls_step_helpers_when_proceed(self) -> None:
        hook = _RecordingUpdateHook(priority=10)
        orch = TrainingUpdateOrchestrator(hook)
        ctx = _make_ctx()
        with _patched_update_helpers() as m:
            orch(ctx, TrainingStage.DO_OPTIMIZER_STEP)
        m.orch_step.assert_called_once_with(ctx.optimizers)
        m.orch_sched.assert_called_once_with(ctx.lr_schedulers)

    def test_do_optimizer_step_uses_grad_scaler_when_present(self) -> None:
        hook = _RecordingUpdateHook(priority=10)
        orch = TrainingUpdateOrchestrator(hook)
        optimizer = Mock(spec=torch.optim.Optimizer)
        scheduler = Mock()
        scaler = Mock(spec=torch.amp.GradScaler)
        scaler.get_scale.side_effect = [128.0, 128.0]
        ctx = _make_ctx()
        ctx.optimizers = [optimizer]
        ctx.lr_schedulers = [scheduler]
        ctx.grad_scaler = scaler

        with _patched_update_helpers() as m:
            orch(ctx, TrainingStage.DO_OPTIMIZER_STEP)

        m.orch_step.assert_not_called()
        scaler.step.assert_called_once_with(optimizer)
        scaler.update.assert_called_once_with()
        m.orch_sched.assert_called_once_with(ctx.lr_schedulers)
        assert orch.optimizer_step_skipped is False

    def test_do_optimizer_step_skips_schedulers_when_grad_scaler_skips(self) -> None:
        observer = _AfterOptimizerStepHook(priority=20)
        orch = TrainingUpdateOrchestrator(_RecordingUpdateHook(priority=10), observer)
        optimizer = Mock(spec=torch.optim.Optimizer)
        scheduler = Mock()
        scaler = Mock(spec=torch.amp.GradScaler)
        scaler.get_scale.side_effect = [128.0, 64.0]
        ctx = _make_ctx()
        ctx.optimizers = [optimizer]
        ctx.lr_schedulers = [scheduler]
        ctx.grad_scaler = scaler

        with _patched_update_helpers() as m:
            orch(ctx, TrainingStage.DO_OPTIMIZER_STEP)
        orch(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)

        m.orch_step.assert_not_called()
        scaler.step.assert_called_once_with(optimizer)
        scaler.update.assert_called_once_with()
        m.orch_sched.assert_not_called()
        assert orch.optimizer_step_skipped is True
        assert observer.will_skip_values == [True]

    def test_do_optimizer_step_skips_step_helpers_on_veto(self) -> None:
        hook = _VetoHook(veto_stage=TrainingStage.DO_OPTIMIZER_STEP, priority=10)
        orch = TrainingUpdateOrchestrator(hook)
        ctx = _make_ctx()
        with _patched_update_helpers() as m:
            orch(ctx, TrainingStage.DO_OPTIMIZER_STEP)
        m.orch_step.assert_not_called()
        m.orch_sched.assert_not_called()

    def test_after_optimizer_step_iterates_with_will_skip_false(self) -> None:
        h1 = _RecordingUpdateHook(priority=10)
        h2 = _RecordingUpdateHook(priority=20)
        orch = TrainingUpdateOrchestrator(h1, h2)
        ctx = _make_ctx()
        orch(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)
        assert h1.calls == [(TrainingStage.AFTER_OPTIMIZER_STEP, False)]
        assert h2.calls == [(TrainingStage.AFTER_OPTIMIZER_STEP, False)]

    def test_after_optimizer_step_receives_will_skip_false_after_step(self) -> None:
        observer = _AfterOptimizerStepHook(priority=10)
        orch = TrainingUpdateOrchestrator(observer)
        ctx = _make_ctx()
        with _patched_update_helpers():
            orch(ctx, TrainingStage.DO_OPTIMIZER_STEP)
        orch(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)
        assert observer.will_skip_values == [False]

    def test_after_optimizer_step_receives_will_skip_true_after_veto(self) -> None:
        veto = _VetoHook(veto_stage=TrainingStage.DO_OPTIMIZER_STEP, priority=10)
        observer = _AfterOptimizerStepHook(priority=20)
        orch = TrainingUpdateOrchestrator(veto, observer)
        ctx = _make_ctx()
        with _patched_update_helpers():
            orch(ctx, TrainingStage.DO_OPTIMIZER_STEP)
        orch(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)
        assert observer.will_skip_values == [True]

    @pytest.mark.parametrize(
        "stage",
        [TrainingStage.BEFORE_FORWARD, TrainingStage.BEFORE_BACKWARD],
        ids=lambda s: s.name,
    )
    def test_non_update_stage_is_noop(self, stage: TrainingStage) -> None:
        h1 = _RecordingUpdateHook(priority=10)
        h2 = _RecordingUpdateHook(priority=20)
        orch = TrainingUpdateOrchestrator(h1, h2)
        ctx = _make_ctx()
        orch(ctx, stage)
        assert h1.calls == []
        assert h2.calls == []


class TestVetoComposition:
    def test_before_batch_no_short_circuit_all_hooks_called(self) -> None:
        h1 = _RecordingUpdateHook(priority=10)
        h2 = _VetoHook(veto_stage=TrainingStage.BEFORE_BATCH, priority=20)
        h3 = _RecordingUpdateHook(priority=30)
        h4 = _RecordingUpdateHook(priority=40)
        orch = TrainingUpdateOrchestrator(h1, h2, h3, h4)
        ctx = _make_ctx()
        with _patched_update_helpers() as m:
            orch(ctx, TrainingStage.BEFORE_BATCH)
        assert len(h1.calls) == 1
        assert len(h2.calls) == 1
        assert len(h3.calls) == 1
        assert len(h4.calls) == 1
        # Hooks BEFORE the vetoing hook saw will_skip=False.
        assert h1.calls[0] == (TrainingStage.BEFORE_BATCH, False)
        assert h2.calls[0] == (TrainingStage.BEFORE_BATCH, False)
        # Hooks AFTER the vetoing hook saw will_skip=True.
        assert h3.calls[0] == (TrainingStage.BEFORE_BATCH, True)
        assert h4.calls[0] == (TrainingStage.BEFORE_BATCH, True)
        m.orch_zero.assert_not_called()

    def test_do_optimizer_step_veto_suppresses_both_helpers(self) -> None:
        h1 = _RecordingUpdateHook(priority=10)
        h2 = _VetoHook(veto_stage=TrainingStage.DO_OPTIMIZER_STEP, priority=20)
        orch = TrainingUpdateOrchestrator(h1, h2)
        ctx = _make_ctx()
        with _patched_update_helpers() as m:
            orch(ctx, TrainingStage.DO_OPTIMIZER_STEP)
        m.orch_step.assert_not_called()
        m.orch_sched.assert_not_called()

    def test_any_false_among_trues_wins(self) -> None:
        # Five priority-buckets; only the priority-30 hook vetoes BEFORE_BATCH.
        hooks: list[TrainingUpdateHook] = [
            _RecordingUpdateHook(priority=10),
            _RecordingUpdateHook(priority=20),
            _VetoHook(veto_stage=TrainingStage.BEFORE_BATCH, priority=30),
            _RecordingUpdateHook(priority=40),
            _RecordingUpdateHook(priority=50),
        ]
        orch = TrainingUpdateOrchestrator(*hooks)
        ctx = _make_ctx()
        with _patched_update_helpers() as m:
            orch(ctx, TrainingStage.BEFORE_BATCH)
        m.orch_zero.assert_not_called()

    def test_all_true_path_runs_gated_operation(self) -> None:
        hooks = [_RecordingUpdateHook(priority=p) for p in (10, 20, 30)]
        orch = TrainingUpdateOrchestrator(*hooks)
        ctx = _make_ctx()
        with _patched_update_helpers() as m:
            orch(ctx, TrainingStage.BEFORE_BATCH)
        m.orch_zero.assert_called_once_with(ctx.optimizers)


class TestLossChain:
    def test_two_hook_chain_multiplies_loss(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0]))
        x = 5.0
        loss = (param * x).sum()  # base dL/dparam = 5
        hook_lo = _LossTransformHook(factor=0.5, priority=10)
        hook_hi = _LossTransformHook(factor=4.0, priority=20)
        orch = TrainingUpdateOrchestrator(hook_lo, hook_hi)
        ctx = _make_ctx(loss=loss)
        orch(ctx, TrainingStage.DO_BACKWARD)
        assert param.grad is not None
        assert param.grad.item() == pytest.approx(2.0 * x)

    def test_passthrough_hook_preserves_chain(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0]))
        x = 2.0
        loss = (param * x).sum()
        hook_lo = _LossTransformHook(factor=3.0, priority=10)
        # Default __call__ returns (True, ctx.loss) — pass-through.
        hook_passthrough = _RecordingUpdateHook(priority=20)
        orch = TrainingUpdateOrchestrator(hook_lo, hook_passthrough)
        ctx = _make_ctx(loss=loss)
        orch(ctx, TrainingStage.DO_BACKWARD)
        assert param.grad is not None
        assert param.grad.item() == pytest.approx(3.0 * x)

    def test_ctx_loss_replaced_post_chain(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0]))
        original = (param * 1.0).sum()
        orch = TrainingUpdateOrchestrator(
            _LossTransformHook(factor=0.5, priority=10),
            _LossTransformHook(factor=4.0, priority=20),
        )
        ctx = _make_ctx(loss=original)
        orch(ctx, TrainingStage.DO_BACKWARD)
        assert ctx.loss is not original
        # Final scalar value: 1.0 * 0.5 * 4.0 = 2.0.
        assert ctx.loss.item() == pytest.approx(2.0)

    def test_do_backward_rejects_none_loss(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0]))
        loss = (param * 2.0).sum()
        hook = _NoneLossHook()
        orch = TrainingUpdateOrchestrator(hook)
        ctx = _make_ctx(loss=loss)
        with pytest.raises(TypeError, match="DO_BACKWARD"):
            orch(ctx, TrainingStage.DO_BACKWARD)
        assert param.grad is None


class TestStrictBoolValidation:
    """``_check_veto`` rejects non-bool ``proceed`` returns on gated stages."""

    @pytest.mark.parametrize(
        "bad_value",
        [None, 1, 0, "yes", []],
        ids=["none", "int_truthy", "int_zero", "str", "list"],
    )
    @pytest.mark.parametrize("stage", _GATED_STAGES, ids=lambda s: s.name)
    def test_non_bool_proceed_raises_type_error(
        self, bad_value: object, stage: TrainingStage
    ) -> None:
        hook = _BadProceedHook(proceed=bad_value)
        orch = TrainingUpdateOrchestrator(hook)
        ctx = _make_ctx()
        with (
            _patched_update_helpers(),
            pytest.raises(TypeError, match=stage.name) as exc_info,
        ):
            orch(ctx, stage)
        assert "_BadProceedHook" in str(exc_info.value)

    @pytest.mark.parametrize("stage", _GATED_STAGES, ids=lambda s: s.name)
    def test_true_proceed_does_not_raise(self, stage: TrainingStage) -> None:
        hook = _RecordingUpdateHook(priority=10)
        orch = TrainingUpdateOrchestrator(hook)
        ctx = _make_ctx()
        with _patched_update_helpers():
            orch(ctx, stage)  # no raise

    def test_non_gated_stages_skip_veto_validation(self) -> None:
        param = torch.nn.Parameter(torch.tensor([1.0]))
        loss = (param * 2.0).sum()
        # proceed=None must not raise on DO_BACKWARD or AFTER_OPTIMIZER_STEP.
        bad = _BadProceedHook(proceed=None, priority=10)
        orch = TrainingUpdateOrchestrator(bad)
        ctx = _make_ctx(loss=loss)
        orch(ctx, TrainingStage.DO_BACKWARD)  # no raise
        assert param.grad is not None
        orch(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)  # no raise

    def test_check_veto_helper_directly(self) -> None:
        sentinel = object()
        with pytest.raises(TypeError, match="DO_OPTIMIZER_STEP"):
            _check_veto(None, sentinel, TrainingStage.DO_OPTIMIZER_STEP)
        # bool decision passes silently.
        _check_veto(True, sentinel, TrainingStage.BEFORE_BATCH)
        _check_veto(False, sentinel, TrainingStage.BEFORE_BATCH)


class TestOrchestratorConstructor:
    def test_empty_orchestrator_succeeds(self) -> None:
        orch = TrainingUpdateOrchestrator()
        assert orch._hooks == []

    def test_two_hooks_flattened(self) -> None:
        a = _RecordingUpdateHook(priority=20)
        b = _RecordingUpdateHook(priority=10)
        orch = TrainingUpdateOrchestrator(a, b)
        assert orch._hooks == [b, a]

    def test_nested_orchestrator_flattened(self) -> None:
        a = _RecordingUpdateHook(priority=10)
        b = _RecordingUpdateHook(priority=20)
        inner = TrainingUpdateOrchestrator(a, b)
        c = _RecordingUpdateHook(priority=30)
        outer = TrainingUpdateOrchestrator(inner, c)
        assert outer._hooks == [a, b, c]
        for hook in outer._hooks:
            assert not isinstance(hook, TrainingUpdateOrchestrator)

    @pytest.mark.parametrize(
        ("bad_value", "type_name"),
        [(42, "int"), ("a string", "str"), (object(), "object")],
    )
    def test_non_hook_argument_raises_type_error(
        self, bad_value: object, type_name: str
    ) -> None:
        a = _RecordingUpdateHook(priority=10)
        with pytest.raises(TypeError, match="argument 1") as exc_info:
            TrainingUpdateOrchestrator(a, bad_value)  # type: ignore[arg-type]
        assert type_name in str(exc_info.value)
        assert "*hooks" in str(exc_info.value)

    def test_list_instead_of_varargs_raises_type_error(self) -> None:
        a = _RecordingUpdateHook(priority=10)
        b = _RecordingUpdateHook(priority=20)
        with pytest.raises(TypeError, match="argument 0") as exc_info:
            TrainingUpdateOrchestrator([a, b])  # type: ignore[arg-type]
        assert "list" in str(exc_info.value)


class TestOrchestratorLifecycle:
    def test_context_manager_forwards_lifecycle_to_children(self) -> None:
        events: list[str] = []
        first = _LifecycleUpdateHook("first", events, priority=10)
        second = _LifecycleUpdateHook("second", events, priority=20)

        with TrainingUpdateOrchestrator(second, first):
            events.append("inside")

        assert events == [
            "enter:first",
            "enter:second",
            "inside",
            "exit:second",
            "exit:first",
        ]

    def test_context_manager_closes_children_without_exit(self) -> None:
        events: list[str] = []
        first = _CloseOnlyUpdateHook("first", events, priority=10)
        second = _CloseOnlyUpdateHook("second", events, priority=20)

        with TrainingUpdateOrchestrator(first, second):
            events.append("inside")

        assert events == ["inside", "close:second", "close:first"]

    def test_strategy_context_enters_autowrapped_update_hooks(self) -> None:
        events: list[str] = []
        update_hook = _LifecycleUpdateHook("update", events)
        strategy = _make_strategy(hooks=[update_hook])

        with strategy:
            events.append("inside")

        assert events == ["enter:update", "inside", "exit:update"]


class TestRunsOnStage:
    def test_base_hook_claims_only_update_stages(self) -> None:
        hook = TrainingUpdateHook()
        for stage in TrainingStage:
            assert hook._runs_on_stage(stage) is (stage in _UPDATE_STAGES)

    def test_orchestrator_claims_only_update_stages(self) -> None:
        orch = TrainingUpdateOrchestrator(_RecordingUpdateHook())
        for stage in TrainingStage:
            assert orch._runs_on_stage(stage) is (stage in _UPDATE_STAGES)

    def test_hook_claims_stage_helper_matches_runs_on_stage(self) -> None:
        hook = TrainingUpdateHook()
        for stage in TrainingStage:
            assert _hook_claims_stage(hook, stage) is (stage in _UPDATE_STAGES)

    def test_hook_claims_stage_uses_stage_attr_when_no_runs_on_stage(self) -> None:
        hook = _StageOnlyHook(TrainingStage.BEFORE_BACKWARD)
        assert _hook_claims_stage(hook, TrainingStage.BEFORE_BACKWARD) is True
        for stage in TrainingStage:
            if stage is TrainingStage.BEFORE_BACKWARD:
                continue
            assert _hook_claims_stage(hook, stage) is False

    def test_hybrid_hook_runs_on_stage_takes_precedence(self) -> None:
        hook = _HybridStageRunsOnHook(stage=TrainingStage.DO_BACKWARD)
        # Even though stage == DO_BACKWARD, _runs_on_stage returns False.
        assert _hook_claims_stage(hook, TrainingStage.DO_BACKWARD) is False


class TestAutoWrapConstructor:
    def test_single_bare_hook_wrapped_in_orchestrator(self) -> None:
        bare = _RecordingUpdateHook(priority=10)
        strategy = _make_strategy(hooks=[bare])
        assert len(strategy.hooks) == 1
        wrapper = _single_orchestrator(strategy)
        assert wrapper._hooks == [bare]
        assert strategy._has_update_orchestrator is True

    def test_multiple_bare_hooks_folded_into_one_orchestrator(self) -> None:
        a = _RecordingUpdateHook(priority=20)
        b = _RecordingUpdateHook(priority=10)
        strategy = _make_strategy(hooks=[a, b])
        assert len(strategy.hooks) == 1
        wrapper = _single_orchestrator(strategy)
        assert wrapper._hooks == [b, a]

    def test_explicit_orchestrator_kept_as_is(self) -> None:
        bare = _RecordingUpdateHook(priority=10)
        explicit = TrainingUpdateOrchestrator(bare)
        strategy = _make_strategy(hooks=[explicit])
        assert strategy.hooks[0] is explicit
        assert strategy._has_update_orchestrator is True

    def test_explicit_orchestrator_uses_base_frequency_validation(self) -> None:
        explicit = TrainingUpdateOrchestrator(_RecordingUpdateHook(priority=10))
        explicit.frequency = 0
        with pytest.raises(pydantic.ValidationError, match="Hook frequency"):
            _make_strategy(hooks=[explicit])

    def test_explicit_orchestrator_plus_bare_folded(self) -> None:
        a = _RecordingUpdateHook(priority=10)
        b = _RecordingUpdateHook(priority=20)
        explicit = TrainingUpdateOrchestrator(a)
        strategy = _make_strategy(hooks=[explicit, b])
        wrapper = _single_orchestrator(strategy)
        assert len(wrapper._hooks) == 2
        assert any(h is a for h in wrapper._hooks)
        assert any(h is b for h in wrapper._hooks)

    def test_non_update_hooks_preserved_with_orchestrator_inserted(self) -> None:
        non_a = _StageOnlyHook(TrainingStage.AFTER_BATCH)
        non_b = _StageOnlyHook(TrainingStage.BEFORE_FORWARD)
        update_a = _RecordingUpdateHook(priority=10)
        update_b = _RecordingUpdateHook(priority=20)
        strategy = _make_strategy(hooks=[non_a, update_a, non_b, update_b])
        assert strategy.hooks[0] is non_a
        assert isinstance(strategy.hooks[1], TrainingUpdateOrchestrator)
        assert strategy.hooks[2] is non_b
        non_update = [
            h for h in strategy.hooks if not isinstance(h, TrainingUpdateOrchestrator)
        ]
        assert non_update == [non_a, non_b]
        assert strategy.hooks[0] is non_a
        assert isinstance(strategy.hooks[1], TrainingUpdateOrchestrator)
        assert strategy.hooks[2] is non_b
        wrapper = _single_orchestrator(strategy)
        assert len(wrapper._hooks) == 2
        assert any(h is update_a for h in wrapper._hooks)
        assert any(h is update_b for h in wrapper._hooks)

    def test_orchestrator_inserted_at_first_bare_update_hook(self) -> None:
        update = _RecordingUpdateHook(priority=10)
        non_update = _StageOnlyHook(TrainingStage.AFTER_BATCH)
        strategy = _make_strategy(hooks=[update, non_update])
        assert isinstance(strategy.hooks[0], TrainingUpdateOrchestrator)
        assert strategy.hooks[1] is non_update

    def test_no_orchestrator_when_no_update_hooks(self) -> None:
        # Auto-wrap is keyed off ``TrainingUpdateHook`` type, not stage
        # membership; a plain ``Hook``-protocol object on BEFORE_BATCH does
        # not trigger orchestrator creation.
        plain_stage_hook = _StageOnlyHook(TrainingStage.BEFORE_BATCH)
        strategy = _make_strategy(hooks=[plain_stage_hook])
        assert strategy._has_update_orchestrator is False


class TestAutoWrapRegisterHook:
    def test_register_first_bare_hook_creates_orchestrator(self) -> None:
        strategy = _make_strategy(hooks=[])
        bare = _RecordingUpdateHook(priority=10)
        strategy.register_hook(bare)
        wrapper = _single_orchestrator(strategy)
        assert wrapper._hooks == [bare]
        assert strategy._has_update_orchestrator is True

    def test_register_second_bare_hook_merges(self) -> None:
        a = _RecordingUpdateHook(priority=10)
        strategy = _make_strategy(hooks=[a])
        b = _RecordingUpdateHook(priority=20)
        strategy.register_hook(b)
        wrapper = _single_orchestrator(strategy)
        assert len(wrapper._hooks) == 2
        assert any(h is a for h in wrapper._hooks)
        assert any(h is b for h in wrapper._hooks)

    def test_register_non_update_hook_skips_autowrap(self) -> None:
        strategy = _make_strategy(hooks=[])
        # Auto-wrap is keyed off ``TrainingUpdateHook`` type, not stage.
        plain_stage_hook = _StageOnlyHook(TrainingStage.BEFORE_BATCH)
        strategy.register_hook(plain_stage_hook)
        assert strategy._has_update_orchestrator is False
        assert plain_stage_hook in strategy.hooks

    def test_register_update_hook_with_stage_raises_value_error(self) -> None:
        strategy = _make_strategy(hooks=[])
        bare = _RecordingUpdateHook(priority=10)
        with pytest.raises(ValueError, match="stage=.*TrainingUpdateHook"):
            strategy.register_hook(bare, stage=TrainingStage.BEFORE_BATCH)

    def test_register_second_orchestrator_raises_value_error(self) -> None:
        a = _RecordingUpdateHook(priority=10)
        strategy = _make_strategy(hooks=[TrainingUpdateOrchestrator(a)])
        b = _RecordingUpdateHook(priority=20)
        orch_b = TrainingUpdateOrchestrator(b)
        with pytest.raises(ValueError, match="Only one TrainingUpdateOrchestrator"):
            strategy.register_hook(orch_b)

    def test_register_orchestrator_uses_base_frequency_validation(self) -> None:
        strategy = _make_strategy(hooks=[])
        explicit = TrainingUpdateOrchestrator(_RecordingUpdateHook(priority=10))
        explicit.frequency = 0
        with pytest.raises(ValueError, match="Hook frequency"):
            strategy.register_hook(explicit)

    def test_claim_flags_refreshed_after_registration(self) -> None:
        strategy = _make_strategy(hooks=[])
        assert strategy._has_do_backward_claim is False
        assert strategy._has_do_optimizer_step_claim is False
        bare = _RecordingUpdateHook(priority=10)
        strategy.register_hook(bare)
        # Orchestrator claims both DO stages.
        assert strategy._has_do_backward_claim is True
        assert strategy._has_do_optimizer_step_claim is True


class TestTwoOrchestratorRejection:
    def test_constructor_two_orchestrators_raises_validation_error(self) -> None:
        a = _RecordingUpdateHook(priority=10)
        b = _RecordingUpdateHook(priority=20)
        orch_a = TrainingUpdateOrchestrator(a)
        orch_b = TrainingUpdateOrchestrator(b)
        with pytest.raises(pydantic.ValidationError) as exc_info:
            _make_strategy(hooks=[orch_a, orch_b])
        assert "Only one TrainingUpdateOrchestrator" in str(exc_info.value)

    def test_register_hook_two_orchestrators_raises_value_error(self) -> None:
        a = _RecordingUpdateHook(priority=10)
        b = _RecordingUpdateHook(priority=20)
        strategy = _make_strategy(hooks=[TrainingUpdateOrchestrator(a)])
        orch_b = TrainingUpdateOrchestrator(b)
        with pytest.raises(ValueError, match="Only one TrainingUpdateOrchestrator"):
            strategy.register_hook(orch_b)
        # Ensure it is NOT a ValidationError subclass at runtime.
        with pytest.raises(ValueError) as exc_info:
            strategy.register_hook(orch_b)
        assert not isinstance(exc_info.value, pydantic.ValidationError)

    def test_message_references_compose_with_plus(self) -> None:
        assert "+" in _MULTIPLE_ORCHESTRATOR_MSG


class TestDoStageConflict:
    @pytest.mark.parametrize("do_stage", _DO_STAGES, ids=lambda s: s.name)
    def test_orchestrator_plus_non_update_hook_with_do_stage_constructor(
        self, do_stage: TrainingStage
    ) -> None:
        bare = _RecordingUpdateHook(priority=10)
        orch = TrainingUpdateOrchestrator(bare)
        non_update = _StageOnlyHook(do_stage)
        with pytest.raises(pydantic.ValidationError) as exc_info:
            _make_strategy(hooks=[orch, non_update])
        msg = str(exc_info.value)
        assert "At most one hook may claim" in msg
        assert do_stage.name in msg

    @pytest.mark.parametrize("do_stage", _DO_STAGES, ids=lambda s: s.name)
    def test_register_hook_do_stage_collision_raises_value_error(
        self, do_stage: TrainingStage
    ) -> None:
        bare = _RecordingUpdateHook(priority=10)
        strategy = _make_strategy(hooks=[bare])
        non_update = _StageOnlyHook(TrainingStage.AFTER_BATCH)
        with pytest.raises(ValueError, match=do_stage.name):
            strategy.register_hook(non_update, stage=do_stage)

    def test_two_non_update_hooks_with_same_do_stage_rejected(self) -> None:
        h1 = _StageOnlyHook(TrainingStage.DO_BACKWARD)
        h2 = _StageOnlyHook(TrainingStage.DO_BACKWARD)
        with pytest.raises(pydantic.ValidationError) as exc_info:
            _make_strategy(hooks=[h1, h2])
        assert "DO_BACKWARD" in str(exc_info.value)

    def test_fake_eq_hook_counted_only_once_via_identity_check(self) -> None:
        """``_validate_single_do_claimants`` uses ``is`` for the candidate check.

        A hook whose ``__eq__`` returns ``True`` for any comparison should NOT
        be spuriously double-counted when registered once with a DO stage.
        This verifies the identity-vs-equality fix.
        """
        strategy = _make_strategy(hooks=[])
        fake = _FakeEqHook(stage=TrainingStage.DO_BACKWARD)
        # Should succeed: only one claimant of DO_BACKWARD even though
        # ``fake == anything`` is True.
        strategy.register_hook(fake)
        # Use identity, not ``in`` (since fake.__eq__ would return True for
        # any peer hook in strategy.hooks).
        assert any(h is fake for h in strategy.hooks)
        assert strategy._has_do_backward_claim is True


class TestTrainContextGradScaler:
    def test_default_grad_scaler_is_none(self) -> None:
        ctx = _make_ctx()
        assert ctx.grad_scaler is None

    def test_grad_scaler_accepts_mocked_instance(self) -> None:
        scaler = Mock(spec=torch.amp.GradScaler)
        ctx = TrainContext(
            batch=_make_batch(),
            step_count=0,
            grad_scaler=scaler,
        )
        assert ctx.grad_scaler is scaler

    def test_grad_scaler_visible_to_later_hook_in_dispatch(self) -> None:
        scaler = Mock(spec=torch.amp.GradScaler)
        setter = _GradScalerSetHook(scaler)
        reader = _GradScalerReadHook()
        orch = TrainingUpdateOrchestrator(setter, reader)
        param = torch.nn.Parameter(torch.tensor([1.0]))
        loss = (param * 1.0).sum()
        ctx = _make_ctx(loss=loss)
        orch(ctx, TrainingStage.DO_BACKWARD)
        assert reader.observed is scaler


class TestTrainContextLifecycle:
    def test_no_hook_run_does_not_build_train_context(self) -> None:
        strategy = _make_strategy(hooks=[])
        with patch.object(
            strategy,
            "_build_context",
            side_effect=AssertionError("_build_context should not run without hooks"),
        ) as build_context:
            strategy.run([_make_batch()])
        build_context.assert_not_called()
        assert strategy._ctx is None

    def test_context_cache_cleared_after_hook_failure_and_retry(self) -> None:
        capture = _ContextCaptureHook(TrainingStage.BEFORE_BATCH)
        raiser = _RaisingStageHook(TrainingStage.BEFORE_FORWARD)
        strategy = _make_strategy(hooks=[capture, raiser])

        with pytest.raises(RuntimeError, match="forced hook failure"):
            strategy.run([_make_batch()])

        assert strategy._ctx is None
        assert len(capture.contexts) == 1
        failed_ctx = capture.contexts[0]

        raiser.enabled = False
        strategy.run([_make_batch(seed=10)])

        assert strategy._ctx is None
        assert len(capture.contexts) == 2
        assert capture.contexts[1] is not failed_ctx
        assert capture.contexts[1].optimizers is strategy._optimizers


class TestPlainDoStageHooks:
    def test_plain_do_backward_hook_owns_backward(self) -> None:
        hook = _DoBackwardOwnerHook()
        strategy = _make_strategy(hooks=[hook])
        strategy.run([_make_batch()])
        assert hook.calls == 1
        assert strategy.step_count == 1

    def test_plain_do_optimizer_hook_suppresses_default_step_helpers(self) -> None:
        hook = _DoOptimizerStepOwnerHook()
        strategy = _make_strategy(hooks=[hook])
        with _patched_update_helpers() as m:
            strategy.run([_make_batch()])
        assert hook.calls == 1
        assert len(hook.contexts) == 1
        m.strategy_step.assert_not_called()
        m.strategy_sched.assert_not_called()


# ---------------------------------------------------------------------------
# Integration tests: orchestrator vs. strategy default training-loop paths.
#
# These tests run a single ``strategy.run(...)`` per scenario with all six
# helper functions patched (``_patched_update_helpers``) so we can assert
# which path called which helper. We deliberately keep one canonical
# strategy.run() per (path, gating) combination rather than table-driving
# every assertion through a fixture; this preserves stack-trace clarity if
# the strategy's dispatch contract regresses.
# ---------------------------------------------------------------------------


class TestZeroGradSuppression:
    def test_veto_suppresses_both_zero_gradient_paths(self) -> None:
        hook = _VetoHook(veto_stage=TrainingStage.BEFORE_BATCH, priority=10)
        with _run_strategy_with_patched_helpers(hooks=[hook]) as m:
            pass
        m.strategy_zero.assert_not_called()
        m.orch_zero.assert_not_called()

    def test_orchestrator_zero_grad_called_on_proceed(self) -> None:
        hook = _RecordingUpdateHook(priority=10)
        with _run_strategy_with_patched_helpers(hooks=[hook]) as m:
            pass
        m.strategy_zero.assert_not_called()
        m.orch_zero.assert_called_once()

    def test_default_zero_grad_called_when_no_orchestrator(self) -> None:
        with _run_strategy_with_patched_helpers(hooks=[]) as m:
            pass
        m.strategy_zero.assert_called_once()
        m.orch_zero.assert_not_called()


class TestOptimizerStepSuppression:
    def test_veto_suppresses_step_helpers(self) -> None:
        hook = _VetoHook(veto_stage=TrainingStage.DO_OPTIMIZER_STEP, priority=10)
        with _run_strategy_with_patched_helpers(hooks=[hook]) as m:
            pass
        m.orch_step.assert_not_called()
        m.orch_sched.assert_not_called()
        m.strategy_step.assert_not_called()
        m.strategy_sched.assert_not_called()

    def test_orchestrator_step_helpers_called_on_proceed(self) -> None:
        hook = _RecordingUpdateHook(priority=10)
        with _run_strategy_with_patched_helpers(hooks=[hook]) as m:
            pass
        m.orch_step.assert_called_once()
        m.orch_sched.assert_called_once()
        m.strategy_step.assert_not_called()
        m.strategy_sched.assert_not_called()

    def test_default_step_helpers_called_without_orchestrator(self) -> None:
        with _run_strategy_with_patched_helpers(hooks=[]) as m:
            pass
        m.strategy_step.assert_called_once()
        m.strategy_sched.assert_called_once()

    def test_unpatched_orchestrator_steps_optimizer_and_scheduler(self) -> None:
        hook = _RecordingUpdateHook(priority=10)
        strategy = _make_strategy(
            hooks=[hook],
            optimizer_configs=OptimizerConfig(
                optimizer_cls=torch.optim.SGD,
                optimizer_kwargs={"lr": 0.1},
                scheduler_cls=torch.optim.lr_scheduler.StepLR,
                scheduler_kwargs={"step_size": 1, "gamma": 0.5},
            ),
        )
        before = [p.detach().clone() for p in strategy.models["main"].parameters()]

        strategy.run([_make_batch()])

        after = list(strategy.models["main"].parameters())
        assert any(not torch.equal(old, new) for old, new in zip(before, after))
        assert strategy._optimizers[0].param_groups[0]["lr"] == pytest.approx(0.05)
        assert hook.calls[-1] == (TrainingStage.AFTER_OPTIMIZER_STEP, False)


class TestAfterOptimizerStepAlwaysRuns:
    def test_after_optimizer_step_runs_when_step_vetoed(self) -> None:
        hook = _VetoHook(veto_stage=TrainingStage.DO_OPTIMIZER_STEP, priority=10)
        strategy = _make_strategy(hooks=[hook])
        strategy.train_batch(_make_batch())
        seen_stages = {stage for stage, _will_skip in hook.calls}
        assert TrainingStage.AFTER_OPTIMIZER_STEP in seen_stages
        assert (TrainingStage.AFTER_OPTIMIZER_STEP, True) in hook.calls
        # Sanity: DO_OPTIMIZER_STEP was indeed dispatched (so the veto path ran).
        assert TrainingStage.DO_OPTIMIZER_STEP in seen_stages


class TestHookProtocolCompliance:
    def test_orchestrator_satisfies_hook_protocol(self) -> None:
        orch = TrainingUpdateOrchestrator(_RecordingUpdateHook())
        assert isinstance(orch, Hook)

    def test_bare_training_update_hook_does_not_satisfy_protocol(self) -> None:
        """Bare ``TrainingUpdateHook`` lacks ``frequency``/``stage`` so it fails the check.

        The base class intentionally omits ``frequency``/``stage`` because it is
        not directly registry-compatible; the orchestrator is the registry-facing
        wrapper. Auto-wrapping in ``TrainingStrategy`` ensures users do not have
        to confront this distinction.
        """
        bare = TrainingUpdateHook()
        assert not isinstance(bare, Hook)


class TestFoldHelper:
    def test_no_update_hooks_returns_input_list(self) -> None:
        non_a = _StageOnlyHook(TrainingStage.AFTER_BATCH)
        non_b = _StageOnlyHook(TrainingStage.BEFORE_FORWARD)
        result = _fold_training_update_hooks([non_a, non_b])
        assert result == [non_a, non_b]
        assert all(not isinstance(h, TrainingUpdateOrchestrator) for h in result)

    def test_two_orchestrators_raises_value_error(self) -> None:
        a = _RecordingUpdateHook(priority=10)
        b = _RecordingUpdateHook(priority=20)
        orch_a = TrainingUpdateOrchestrator(a)
        orch_b = TrainingUpdateOrchestrator(b)
        with pytest.raises(ValueError, match="Only one TrainingUpdateOrchestrator"):
            _fold_training_update_hooks([orch_a, orch_b])

    def test_single_bare_hook_wrapped_in_orchestrator(self) -> None:
        bare = _RecordingUpdateHook(priority=10)
        result = _fold_training_update_hooks([bare])
        assert len(result) == 1
        assert isinstance(result[0], TrainingUpdateOrchestrator)
        assert result[0]._hooks == [bare]
