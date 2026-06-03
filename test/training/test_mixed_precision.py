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
"""Tests for :class:`nvalchemi.training.hooks.MixedPrecisionHook`."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from pydantic import ValidationError

from nvalchemi.data import Batch
from nvalchemi.hooks._context import HookContext, TrainContext
from nvalchemi.hooks._protocol import Hook
from nvalchemi.models.base import BaseModelMixin
from nvalchemi.training import EnergyMSELoss, ForceMSELoss
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training.hooks import (
    MixedPrecisionHook,
    TrainingUpdateHook,
    TrainingUpdateOrchestrator,
)
from nvalchemi.training.hooks.mixed_precision import MixedPrecisionHook as _MP
from nvalchemi.training.optimizers import OptimizerConfig
from nvalchemi.training.strategy import TrainingStrategy, default_training_fn
from test.training.conftest import _build_demo_model

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

ALL_PRECISIONS: list[torch.dtype] = [torch.float32, torch.bfloat16, torch.float16]


def _cast_back_training_fn(
    model: BaseModelMixin, batch: Batch
) -> dict[str, torch.Tensor]:
    """Forward the model and cast predictions back to fp32.

    Autocast casts eligible ops to lower precision inside its region, but
    the project's loss terms enforce dtype parity with the ``batch.*``
    targets (which remain fp32). Casting predictions back restores that
    parity while still exercising the autocast-covered forward pass.
    """
    preds = default_training_fn(model, batch)
    return {k: v.to(torch.float32) for k, v in preds.items()}


@pytest.fixture(params=ALL_PRECISIONS, ids=lambda p: str(p).replace("torch.", ""))
def precision(request: pytest.FixtureRequest) -> torch.dtype:
    """Parametrize over the three supported AMP precisions."""
    return request.param


@pytest.fixture
def strategy_factory(
    baseline_strategy_kwargs: dict[str, Any],
) -> Callable[..., TrainingStrategy]:
    """Return a factory that builds a strategy with the cast-back training_fn.

    The factory is kept because eight tests call it with varying ``hooks``
    and ``devices`` kwargs; eliminating it would repeat the same merge
    pattern at each site.
    """

    def _factory(**overrides: Any) -> TrainingStrategy:
        kwargs = {
            **baseline_strategy_kwargs,
            "training_fn": _cast_back_training_fn,
            **overrides,
        }
        return TrainingStrategy(**kwargs)

    return _factory


@pytest.fixture
def mocked_scaler() -> Any:
    """Patch ``torch.amp.GradScaler`` and yield the mock scaler instance.

    The scaler reports a healthy step (no inf, constant scale) and returns
    a ``MagicMock`` from ``scale()`` so ``backward`` on the scaled loss is
    observable by the tests.
    """
    with patch("torch.amp.GradScaler", autospec=True) as scaler_cls:
        scaler = scaler_cls.return_value
        scaler.get_scale.return_value = 65536.0
        scaler._found_inf_per_device.return_value = {
            torch.device("cpu"): torch.tensor(0.0)
        }
        scaler.scale.return_value = MagicMock(name="scaled_loss")
        yield scaler


class _ObserverHook:
    """Observer hook that forwards ``(ctx, stage)`` to ``callback`` at ``stage``.

    Attributes
    ----------
    stage : TrainingStage
        The stage on which the registry should dispatch this hook.
    frequency : int
        Fixed to ``1``.
    """

    def __init__(
        self,
        stage: TrainingStage,
        callback: Callable[[HookContext, Enum], None],
    ) -> None:
        self.stage = stage
        self.frequency = 1
        self._callback = callback

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        self._callback(ctx, stage)


class _ClaimsStagesHook:
    """Observer hook that opts into one or more stages via ``_runs_on_stage``.

    Used by exclusivity tests to collide with :class:`MixedPrecisionHook`
    on the ``DO_BACKWARD`` stage.

    Attributes
    ----------
    stage : None
        Explicitly ``None`` — stage selection is delegated to ``_runs_on_stage``.
    frequency : int
        Fixed to ``1``.
    """

    def __init__(self, claimed: set[TrainingStage]) -> None:
        self._claimed = set(claimed)
        self.stage = None
        self.frequency = 1

    def _runs_on_stage(self, stage: Enum) -> bool:
        return stage in self._claimed

    def __call__(self, ctx: HookContext, stage: Enum) -> None:  # noqa: ARG002
        pass


class _OptimizerStepVetoHook(TrainingUpdateHook):
    """Update hook that vetoes optimizer stepping before AMP unscales grads."""

    priority = 10

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,  # noqa: ARG002
    ) -> tuple[bool, torch.Tensor]:
        return stage is not TrainingStage.DO_OPTIMIZER_STEP, ctx.loss


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Constructor validation and Hook-Protocol attribute defaults."""

    def test_invalid_precision_rejected(self) -> None:
        with pytest.raises(ValidationError):
            MixedPrecisionHook(precision="fp8")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "bad", [torch.float64, "float64"], ids=["float64_dtype", "float64_str"]
    )
    def test_unsupported_dtype_rejected(self, bad: Any) -> None:
        with pytest.raises(ValidationError):
            MixedPrecisionHook(precision=bad)

    def test_precision_accepts_dtype_object(self) -> None:
        assert MixedPrecisionHook(precision=torch.float16).precision == torch.float16

    def test_precision_accepts_canonical_string(self) -> None:
        assert MixedPrecisionHook(precision="bfloat16").precision == torch.bfloat16

    @pytest.mark.parametrize(
        ("alias", "expected"),
        [
            ("fp32", torch.float32),
            ("bf16", torch.bfloat16),
            ("fp16", torch.float16),
        ],
    )
    def test_precision_accepts_common_aliases(
        self, alias: str, expected: torch.dtype
    ) -> None:
        assert MixedPrecisionHook(precision=alias).precision == expected

    def test_unknown_config_key_rejected(self) -> None:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            MixedPrecisionHook(precision="float16", precison="float32")  # type: ignore[call-arg]

    def test_invalid_precision_message_lists_supported_values(self) -> None:
        with pytest.raises(
            ValidationError,
            match="MixedPrecisionHook.precision must be one of "
            r"\(float32, bfloat16, float16\)",
        ):
            MixedPrecisionHook(precision="fp64")  # type: ignore[arg-type]

    def test_is_training_update_hook(self, precision: torch.dtype) -> None:
        hook = MixedPrecisionHook(precision=precision)
        assert isinstance(hook, TrainingUpdateHook)
        assert not isinstance(hook, Hook)
        assert hook.priority == 20

    def test_class_identity_across_module_paths(self) -> None:
        # Both import paths must resolve to the same class.
        assert MixedPrecisionHook is _MP

    def test_strategy_autowraps_into_update_orchestrator(
        self, strategy_factory: Callable[..., TrainingStrategy]
    ) -> None:
        hook = MixedPrecisionHook(precision=torch.float32)
        strategy = strategy_factory(hooks=[hook])
        assert len(strategy.hooks) == 1
        assert isinstance(strategy.hooks[0], TrainingUpdateOrchestrator)
        assert strategy.hooks[0]._hooks == [hook]


# ---------------------------------------------------------------------------
# Stage dispatch
# ---------------------------------------------------------------------------


class TestStageDispatch:
    """Update-hook fallback keeps unhandled stages a silent no-op."""

    def test_unclaimed_stage_is_noop(self) -> None:
        hook = MixedPrecisionHook(precision=torch.float32)
        ctx = Mock(spec=TrainContext)
        ctx.loss = Mock(name="loss")
        proceed, loss = hook(ctx, TrainingStage.AFTER_BATCH, will_skip=False)
        assert proceed is True
        assert loss is ctx.loss
        assert hook._scaler is None
        assert hook._autocast_ctx is None
        assert hook._active is False


# ---------------------------------------------------------------------------
# Core training (precision × device)
# ---------------------------------------------------------------------------


class TestCoreTraining:
    """One-step training with the hook enabled under every precision / device.

    Covers autocast state visibility at ``BEFORE_FORWARD``, clean completion
    on CPU, and the absence of ``MixedPrecisionHook``-originated warnings.
    """

    def test_one_step_completes_cleanly(
        self,
        precision: torch.dtype,
        device: str,
        strategy_factory: Callable[..., TrainingStrategy],
        batch: Batch,
        recwarn: pytest.WarningsRecorder,
    ) -> None:
        mp = MixedPrecisionHook(precision=precision)
        strategy = strategy_factory(hooks=[mp], devices=[torch.device(device)])
        strategy.run([batch])
        assert strategy.step_count == 1
        assert all("MixedPrecisionHook" not in str(w.message) for w in recwarn.list), [
            str(w.message) for w in recwarn.list
        ]

    def test_autocast_state_during_forward(
        self,
        precision: torch.dtype,
        device: str,
        strategy_factory: Callable[..., TrainingStrategy],
        batch: Batch,
    ) -> None:
        records: dict[str, Any] = {}

        def _observe(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            records["enabled"] = torch.is_autocast_enabled(device)
            records["dtype"] = torch.get_autocast_dtype(device)

        mp = MixedPrecisionHook(precision=precision)
        observer = _ObserverHook(TrainingStage.BEFORE_FORWARD, _observe)
        strategy = strategy_factory(
            hooks=[mp, observer], devices=[torch.device(device)]
        )
        strategy.run([batch])
        # fp32 bypasses autocast entirely; low-precision modes enable autocast
        # with the matching dtype.
        expected_enabled = precision != torch.float32
        assert records["enabled"] is expected_enabled
        if expected_enabled:
            assert records["dtype"] == precision

    def test_autocast_disabled_after_backward(
        self,
        precision: torch.dtype,
        device: str,
        strategy_factory: Callable[..., TrainingStrategy],
        batch: Batch,
    ) -> None:
        records: dict[str, bool] = {}

        def _observe(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            records["enabled"] = torch.is_autocast_enabled(device)

        mp = MixedPrecisionHook(precision=precision)
        observer = _ObserverHook(TrainingStage.AFTER_BACKWARD, _observe)
        strategy = strategy_factory(
            hooks=[mp, observer], devices=[torch.device(device)]
        )
        strategy.run([batch])
        assert records["enabled"] is False

    def test_fp32_precision_does_not_create_amp_state(
        self,
        device: str,
        strategy_factory: Callable[..., TrainingStrategy],
        batch: Batch,
    ) -> None:
        records: dict[str, Any] = {}

        mp = MixedPrecisionHook(precision=torch.float32)

        def _observe(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            records["scaler"] = mp._scaler
            records["autocast_ctx"] = mp._autocast_ctx
            records["active"] = mp._active

        observer = _ObserverHook(TrainingStage.BEFORE_FORWARD, _observe)
        strategy = strategy_factory(
            hooks=[mp, observer], devices=[torch.device(device)]
        )
        strategy.run([batch])

        assert records == {"scaler": None, "autocast_ctx": None, "active": False}


# ---------------------------------------------------------------------------
# GradScaler behavior (mocked): call order + multi-optimizer + scheduler gating
# ---------------------------------------------------------------------------


class TestGradScalerBehavior:
    """fp16 drives ``GradScaler`` in canonical order and gates schedulers (reqs 10, 25-27)."""

    def test_scaler_call_order(
        self,
        mocked_scaler: Any,
        strategy_factory: Callable[..., TrainingStrategy],
        batch: Batch,
    ) -> None:
        scaled_loss = mocked_scaler.scale.return_value
        mp = MixedPrecisionHook(precision=torch.float16)
        strategy = strategy_factory(hooks=[mp])
        strategy.run([batch])

        names = [name for name, _, _ in mocked_scaler.method_calls]
        assert scaled_loss.backward.called
        assert names.index("scale") < names.index("unscale_")
        assert names.index("unscale_") < names.index("step")
        assert names.index("step") < names.index("update")
        assert names.count("scale") == 1
        assert names.count("unscale_") == 1
        assert names.count("step") == 1
        assert names.count("update") == 1

    def test_multi_optimizer_unscale_and_step(
        self, mocked_scaler: Any, batch: Batch
    ) -> None:
        model = _build_demo_model()
        params = list(model.parameters())
        half = len(params) // 2
        group_a, group_b = params[:half], params[half:]
        mp_hook = MixedPrecisionHook(precision=torch.float16)
        strategy = TrainingStrategy(
            models=model,
            optimizer_configs=[
                OptimizerConfig(
                    optimizer_cls=torch.optim.Adam,
                    optimizer_kwargs={"lr": 1e-3, "foreach": False},
                ),
            ],
            num_epochs=1,
            training_fn=_cast_back_training_fn,
            loss_fn=EnergyMSELoss() + ForceMSELoss(normalize_by_atom_count=True),
            hooks=[mp_hook],
        )
        # Replace the built optimizer list with two optimizers over disjoint
        # params — more direct than threading multiple configs/models.
        opt_a = torch.optim.Adam(group_a, lr=1e-3)
        opt_b = torch.optim.Adam(group_b, lr=1e-3)
        with strategy:
            strategy._train_batch_with_optimizers(batch, [opt_a, opt_b], [None, None])

        names = [name for name, _, _ in mocked_scaler.method_calls]
        assert names.count("unscale_") == 2
        assert names.count("step") == 2
        assert names.count("update") == 1
        first_step_idx = names.index("step")
        last_unscale_idx = max(i for i, n in enumerate(names) if n == "unscale_")
        assert last_unscale_idx < first_step_idx

    def test_vetoed_optimizer_step_does_not_unscale_or_update_scaler(
        self,
        mocked_scaler: Any,
        batch: Batch,
    ) -> None:
        scaled_loss = mocked_scaler.scale.return_value
        param = torch.nn.Parameter(torch.ones(()))
        opt = torch.optim.SGD([param], lr=1.0)
        workflow = Mock()
        workflow.devices = [torch.device("cpu")]
        ctx = TrainContext(
            batch=batch,
            workflow=workflow,
            loss=param.square(),
            optimizers=[opt],
            lr_schedulers=[],
        )
        mp = MixedPrecisionHook(precision=torch.float16)
        orch = TrainingUpdateOrchestrator(_OptimizerStepVetoHook(), mp)
        with orch:
            orch(ctx, TrainingStage.DO_BACKWARD)
            orch(ctx, TrainingStage.DO_OPTIMIZER_STEP)

        assert scaled_loss.backward.called
        mocked_scaler.unscale_.assert_not_called()
        mocked_scaler.step.assert_not_called()
        mocked_scaler.update.assert_not_called()

    def test_grad_scaler_no_scheduler_fast_path_skips_found_inf_query(
        self, mocked_scaler: Any
    ) -> None:
        param = torch.nn.Parameter(torch.ones(()))
        opt = torch.optim.SGD([param], lr=1.0)
        ctx = TrainContext(
            batch=Mock(spec=Batch),
            optimizers=[opt],
            lr_schedulers=[],
            grad_scaler=mocked_scaler,
        )
        orch = TrainingUpdateOrchestrator()
        orch(ctx, TrainingStage.DO_OPTIMIZER_STEP)

        mocked_scaler.step.assert_called_once_with(opt)
        mocked_scaler.update.assert_called_once()
        mocked_scaler._found_inf_per_device.assert_not_called()

    @pytest.mark.parametrize(
        ("found_inf", "expected_step_called"),
        [(0.0, True), (1.0, False)],
        ids=["no_inf_steps_sched", "found_inf_skips_sched"],
    )
    def test_scheduler_gating(
        self,
        found_inf: float,
        expected_step_called: bool,
        strategy_factory: Callable[..., TrainingStrategy],
        batch: Batch,
    ) -> None:
        with patch("torch.amp.GradScaler", autospec=True) as scaler_cls:
            scaler = scaler_cls.return_value
            scaler.get_scale.return_value = 65536.0
            scaler._found_inf_per_device.return_value = {
                torch.device("cpu"): torch.tensor(found_inf)
            }
            scaler.scale.return_value = MagicMock(name="scaled_loss")

            sched = MagicMock(name="sched")
            mp = MixedPrecisionHook(precision=torch.float16)
            strategy = strategy_factory(hooks=[mp])
            opt = torch.optim.Adam(strategy.models["main"].parameters(), lr=1e-3)
            with strategy:
                strategy._train_batch_with_optimizers(batch, [opt], [sched])

        if expected_step_called:
            sched.step.assert_called_once()
        else:
            sched.step.assert_not_called()


# ---------------------------------------------------------------------------
# Real CUDA end-to-end (no mock) — bf16 / fp16
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestCUDAEndToEnd:
    """Real autocast + real ``GradScaler`` drive a full step without error."""

    @pytest.mark.parametrize(
        "cuda_precision",
        [torch.bfloat16, torch.float16],
        ids=["bf16", "fp16"],
    )
    def test_single_step_runs_cleanly(
        self, cuda_precision: torch.dtype, batch: Batch
    ) -> None:
        device = torch.device("cuda:0")
        model = _build_demo_model()
        mp = MixedPrecisionHook(precision=cuda_precision)
        observed: dict[str, Any] = {}

        def _capture_forward(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            observed["autocast_enabled"] = torch.is_autocast_enabled("cuda")
            observed["autocast_dtype"] = torch.get_autocast_dtype("cuda")

        def _capture_after_step(ctx: HookContext, stage: Enum) -> None:  # noqa: ARG001
            # Capture after the update step; the scaler persists across batches
            # until ``__exit__`` resets it.
            if mp._scaler is not None:
                observed["scale"] = mp._scaler.get_scale()

        forward_hook = _ObserverHook(TrainingStage.BEFORE_FORWARD, _capture_forward)
        after_hook = _ObserverHook(
            TrainingStage.AFTER_OPTIMIZER_STEP, _capture_after_step
        )
        strategy = TrainingStrategy(
            models=model,
            optimizer_configs=OptimizerConfig(
                optimizer_cls=torch.optim.Adam,
                optimizer_kwargs={"lr": 1e-3},
            ),
            num_epochs=1,
            training_fn=_cast_back_training_fn,
            loss_fn=EnergyMSELoss() + ForceMSELoss(normalize_by_atom_count=True),
            devices=[device],
            hooks=[mp, forward_hook, after_hook],
        )
        strategy.run([batch])

        assert strategy.step_count == 1
        assert observed["autocast_enabled"] is True
        assert observed["autocast_dtype"] == cuda_precision
        if cuda_precision == torch.float16:
            assert "scale" in observed
            assert torch.isfinite(torch.tensor(observed["scale"]))

    def test_real_fp16_overflow_skips_optimizer_and_scheduler(self) -> None:
        device = torch.device("cuda:0")
        param = torch.nn.Parameter(torch.ones((), device=device))
        opt = torch.optim.SGD([param], lr=1.0)
        sched = MagicMock(name="sched")
        mp = MixedPrecisionHook(precision=torch.float16)
        orch = TrainingUpdateOrchestrator(mp)
        workflow = Mock()
        workflow.devices = [device]
        ctx = TrainContext(
            batch=Mock(spec=Batch),
            workflow=workflow,
            loss=param * torch.tensor(float("inf"), device=device),
            optimizers=[opt],
            lr_schedulers=[sched],
        )

        try:
            param_before = param.detach().clone()
            orch(ctx, TrainingStage.DO_BACKWARD)
            assert mp._scaler is not None
            scale_before = mp._scaler.get_scale()
            orch(ctx, TrainingStage.DO_OPTIMIZER_STEP)
            torch.testing.assert_close(param.detach(), param_before)
            sched.step.assert_not_called()
            assert mp._scaler.get_scale() < scale_before
        finally:
            orch.__exit__(None, None, None)


# ---------------------------------------------------------------------------
# DO_ stage exclusivity (integration)
# ---------------------------------------------------------------------------


class TestDOStageExclusivity:
    """MixedPrecisionHook composes through the update orchestrator."""

    def test_two_mp_hooks_rejected_to_prevent_double_scaling(
        self, strategy_factory: Callable[..., TrainingStrategy]
    ) -> None:
        first = MixedPrecisionHook(precision=torch.float32)
        second = MixedPrecisionHook(precision=torch.bfloat16)
        with pytest.raises(ValueError, match="MixedPrecisionHook"):
            strategy_factory(hooks=[first, second])

    def test_mp_plus_other_do_backward_claimant_rejected(
        self, strategy_factory: Callable[..., TrainingStrategy]
    ) -> None:
        with pytest.raises(ValueError, match="DO_BACKWARD"):
            strategy_factory(
                hooks=[
                    MixedPrecisionHook(precision=torch.float32),
                    _ClaimsStagesHook({TrainingStage.DO_BACKWARD}),
                ]
            )


# ---------------------------------------------------------------------------
# Live-vs-detached loss contract
# ---------------------------------------------------------------------------


class TestLiveDetachedLossContract:
    """The live-before-backward / detached-after-backward invariant holds (req 22)."""

    def test_loss_graph_state_around_backward(
        self,
        strategy_factory: Callable[..., TrainingStrategy],
        batch: Batch,
    ) -> None:
        records: dict[TrainingStage, bool] = {}

        def _record(ctx: HookContext, stage: TrainingStage) -> None:
            records[stage] = ctx.loss.grad_fn is not None

        hooks = [
            MixedPrecisionHook(precision=torch.float32),
            _ObserverHook(TrainingStage.BEFORE_BACKWARD, _record),
            _ObserverHook(TrainingStage.AFTER_BACKWARD, _record),
        ]
        strategy = strategy_factory(hooks=hooks)
        strategy.run([batch])
        assert records[TrainingStage.BEFORE_BACKWARD] is True
        assert records[TrainingStage.AFTER_BACKWARD] is False


# ---------------------------------------------------------------------------
# zero_grad(set_to_none=True) regression
# ---------------------------------------------------------------------------


class TestZeroGradSetToNone:
    """Regression: optimizers are zeroed with ``set_to_none=True`` (req 28)."""

    def test_zero_grad_called_with_set_to_none_true(
        self, baseline_strategy_kwargs: dict[str, Any], batch: Batch
    ) -> None:
        captured_kwargs: list[dict[str, Any]] = []
        original = torch.optim.Adam.zero_grad

        def _spy(self: torch.optim.Adam, **kwargs: Any) -> None:
            captured_kwargs.append(dict(kwargs))
            original(self, **kwargs)

        mp = MixedPrecisionHook(precision=torch.float32)
        strategy = TrainingStrategy(**{**baseline_strategy_kwargs, "hooks": [mp]})
        with patch.object(torch.optim.Adam, "zero_grad", _spy):
            strategy.run([batch])
        assert captured_kwargs, "zero_grad was never called"
        for kw in captured_kwargs:
            assert kw.get("set_to_none") is True
