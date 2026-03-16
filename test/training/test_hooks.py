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
"""Tests for :class:`TrainingHook` protocol and :class:`TrainingContext`."""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import torch
from tensordict import TensorDict

from nvalchemi.training._hooks import TrainingContext, TrainingHook
from nvalchemi.training._stages import TrainingStageEnum

# ---------------------------------------------------------------------------
# Concrete hook implementations for testing
# ---------------------------------------------------------------------------


class _SingleStageHook:
    """Minimal single-stage hook that records calls."""

    def __init__(self, stage: TrainingStageEnum, frequency: int = 1) -> None:
        self.stage = stage
        self.frequency = frequency
        self.calls: list[tuple[TrainingContext, Any, Any]] = []

    def __call__(self, ctx: TrainingContext, model: Any, trainer: Any) -> None:
        self.calls.append((ctx, model, trainer))


class _MultiStageHook:
    """Hook that registers for multiple stages via ``stages`` frozenset."""

    def __init__(
        self,
        stages: frozenset[TrainingStageEnum],
        frequency: int = 1,
    ) -> None:
        self.stages = stages
        self.frequency = frequency
        # ``stage`` still required by the protocol — pick first sorted member.
        self.stage = min(stages, key=lambda s: s.value)
        self.calls: list[tuple[TrainingContext, Any, Any]] = []

    def __call__(self, ctx: TrainingContext, model: Any, trainer: Any) -> None:
        self.calls.append((ctx, model, trainer))


# ---------------------------------------------------------------------------
# TrainingContext tests
# ---------------------------------------------------------------------------


class TestTrainingContext:
    """Tests for the :class:`TrainingContext` dataclass."""

    def test_default_values(self) -> None:
        """A freshly-created context has sensible defaults."""
        ctx = TrainingContext()
        assert ctx.epoch == 0
        assert ctx.global_step == 0
        assert ctx.batch is None
        assert ctx.model_outputs is None
        assert isinstance(ctx.losses, TensorDict)
        assert ctx.losses.batch_size == torch.Size([])
        assert len(ctx.losses.keys()) == 0
        assert ctx.total_loss is None
        assert ctx.metrics == {}
        assert ctx.extra == {}

    def test_mutability(self) -> None:
        """Context fields can be mutated in-place (not frozen)."""
        ctx = TrainingContext()
        ctx.epoch = 5
        ctx.global_step = 100
        ctx.losses = TensorDict(energy=torch.tensor(0.5), batch_size=[])
        ctx.metrics["lr"] = 1e-3
        ctx.extra["foo"] = "bar"

        assert ctx.epoch == 5
        assert ctx.global_step == 100
        assert "energy" in ctx.losses.keys()
        assert ctx.metrics["lr"] == 1e-3
        assert ctx.extra["foo"] == "bar"

    def test_total_loss_assignment(self) -> None:
        """``total_loss`` can be set to a scalar tensor."""
        ctx = TrainingContext()
        loss = torch.tensor(1.23)
        ctx.total_loss = loss
        assert ctx.total_loss is loss

    def test_independent_default_dicts(self) -> None:
        """Each instance gets its own TensorDict (no shared mutable defaults)."""
        ctx_a = TrainingContext()
        ctx_b = TrainingContext()
        ctx_a.losses.set("x", torch.tensor(1.0))
        assert "x" not in ctx_b.losses.keys()

    def test_stage_counts_default_empty(self) -> None:
        """``stage_counts`` defaults to an empty dict."""
        ctx = TrainingContext()
        assert ctx.stage_counts == {}

    def test_stage_counts_independent_per_instance(self) -> None:
        """Each context gets its own ``stage_counts`` dict."""
        ctx_a = TrainingContext()
        ctx_b = TrainingContext()
        ctx_a.stage_counts[TrainingStageEnum.AFTER_STEP] = 5
        assert TrainingStageEnum.AFTER_STEP not in ctx_b.stage_counts

    def test_stage_counts_increment(self) -> None:
        """``stage_counts`` can be incremented like a counter dict."""
        ctx = TrainingContext()
        stage = TrainingStageEnum.AFTER_STEP
        ctx.stage_counts[stage] = ctx.stage_counts.get(stage, 0) + 1
        ctx.stage_counts[stage] = ctx.stage_counts.get(stage, 0) + 1
        assert ctx.stage_counts[stage] == 2


# ---------------------------------------------------------------------------
# TrainingHook protocol conformance
# ---------------------------------------------------------------------------


class TestTrainingHookProtocol:
    """Verify that concrete hooks satisfy the ``TrainingHook`` protocol."""

    def test_single_stage_hook_is_instance(self) -> None:
        """A single-stage hook satisfies the runtime-checkable protocol."""
        hook = _SingleStageHook(TrainingStageEnum.AFTER_FORWARD)
        assert isinstance(hook, TrainingHook)

    def test_multi_stage_hook_is_instance(self) -> None:
        """A multi-stage hook also satisfies the protocol."""
        hook = _MultiStageHook(
            frozenset(
                {TrainingStageEnum.BEFORE_FORWARD, TrainingStageEnum.AFTER_FORWARD}
            ),
        )
        assert isinstance(hook, TrainingHook)

    def test_non_conforming_object_is_not_instance(self) -> None:
        """An object missing required attributes does not match the protocol."""
        assert not isinstance(object(), TrainingHook)
        assert not isinstance("not a hook", TrainingHook)

    def test_missing_frequency_fails(self) -> None:
        """An object with stage and __call__ but no frequency is not a hook."""

        class _Bad:
            stage = TrainingStageEnum.BEFORE_STEP

            def __call__(
                self, ctx: TrainingContext, model: Any, trainer: Any
            ) -> None: ...

        assert not isinstance(_Bad(), TrainingHook)


# ---------------------------------------------------------------------------
# Hook invocation
# ---------------------------------------------------------------------------


class TestHookInvocation:
    """Test that hooks are callable with the expected signature."""

    def test_single_stage_hook_records_call(self) -> None:
        """Calling a single-stage hook appends to its call log."""
        hook = _SingleStageHook(TrainingStageEnum.AFTER_LOSS)
        ctx = TrainingContext(epoch=1, global_step=42)
        model = Mock()
        trainer = Mock()

        hook(ctx, model, trainer)

        assert len(hook.calls) == 1
        assert hook.calls[0] == (ctx, model, trainer)

    def test_multi_stage_hook_records_call(self) -> None:
        """Multi-stage hooks are invoked the same way as single-stage ones."""
        hook = _MultiStageHook(
            frozenset(
                {TrainingStageEnum.BEFORE_BACKWARD, TrainingStageEnum.AFTER_BACKWARD}
            ),
        )
        ctx = TrainingContext(epoch=0, global_step=10)
        model = Mock()
        trainer = Mock()

        hook(ctx, model, trainer)

        assert len(hook.calls) == 1


# ---------------------------------------------------------------------------
# Multi-stage registration helpers
# ---------------------------------------------------------------------------


class TestMultiStageHookRegistration:
    """Verify the ``stages`` attribute pattern used for multi-stage registration.

    The trainer's ``register_hook`` method (Phase 2) will inspect ``hook.stages``
    analogously to ``BaseDynamics.register_hook``.  These tests validate the
    attribute contract that hooks must fulfil.
    """

    def test_stages_attribute_is_frozenset(self) -> None:
        """Multi-stage hooks expose a ``stages`` frozenset."""
        stages = frozenset(
            {TrainingStageEnum.BEFORE_EPOCH, TrainingStageEnum.AFTER_EPOCH}
        )
        hook = _MultiStageHook(stages)
        assert hook.stages == stages
        assert isinstance(hook.stages, frozenset)

    def test_single_stage_hook_has_no_stages_attribute(self) -> None:
        """Single-stage hooks do not carry a ``stages`` attribute."""
        hook = _SingleStageHook(TrainingStageEnum.AFTER_STEP)
        assert not hasattr(hook, "stages")

    def test_register_hook_simulation(self) -> None:
        """Simulate the register_hook logic from BaseDynamics for training hooks.

        The trainer will mirror this pattern: if ``hook.stages`` exists, register
        the hook under each stage in the frozenset; otherwise use ``hook.stage``.
        """
        hooks: dict[TrainingStageEnum, list[TrainingHook]] = {
            s: [] for s in TrainingStageEnum
        }

        single = _SingleStageHook(TrainingStageEnum.BEFORE_FORWARD)
        multi = _MultiStageHook(
            frozenset(
                {TrainingStageEnum.BEFORE_FORWARD, TrainingStageEnum.AFTER_FORWARD}
            ),
        )

        # --- single-stage registration ---
        hook_stages = getattr(single, "stages", None)
        if hook_stages is not None:
            for s in hook_stages:
                hooks[s].append(single)
        else:
            hooks[single.stage].append(single)

        # --- multi-stage registration ---
        hook_stages = getattr(multi, "stages", None)
        if hook_stages is not None:
            for s in hook_stages:
                hooks[s].append(multi)
        else:
            hooks[multi.stage].append(multi)

        assert single in hooks[TrainingStageEnum.BEFORE_FORWARD]
        assert single not in hooks[TrainingStageEnum.AFTER_FORWARD]

        assert multi in hooks[TrainingStageEnum.BEFORE_FORWARD]
        assert multi in hooks[TrainingStageEnum.AFTER_FORWARD]
