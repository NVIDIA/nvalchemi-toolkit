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
"""Tests for :class:`nvalchemi.training.hooks.EMAHook`."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, Mock

import pytest
import torch
from pydantic import ValidationError
from torch import nn
from torch.optim.swa_utils import AveragedModel

from nvalchemi.hooks._context import TrainContext
from nvalchemi.training._stages import TrainingStage
from nvalchemi.training._validation import ValidationConfig
from nvalchemi.training.hooks import EMAHook, TrainingUpdateHook
from nvalchemi.training.strategy import TrainingStrategy
from test.training.conftest import _build_baseline_strategy_kwargs, _build_batch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_linear(
    in_f: int = 4, out_f: int = 4, *, seed: int | None = None
) -> nn.Linear:
    if seed is not None:
        torch.manual_seed(seed)
    return nn.Linear(in_f, out_f)


def _make_ctx(
    models: dict[str, nn.Module],
    step_count: int,
    *,
    optimizers: list[Any] | None = None,
) -> Mock:
    return Mock(
        spec=TrainContext,
        models=models,
        step_count=step_count,
        optimizers=optimizers if optimizers is not None else [],
        loss=None,
    )


def _params_equal(a: nn.Module, b: nn.Module) -> bool:
    pa = list(a.parameters())
    pb = list(b.parameters())
    if len(pa) != len(pb):
        return False
    return all(torch.equal(x, y) for x, y in zip(pa, pb, strict=True))


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _drive(
    hook: EMAHook,
    source: nn.Module,
    *,
    n_calls: int,
    start_step_count: int = 0,
) -> None:
    """Call ``hook`` ``n_calls`` times on ``AFTER_OPTIMIZER_STEP``.

    ``ctx.step_count`` runs from ``start_step_count`` to
    ``start_step_count + n_calls - 1`` inclusive.
    """
    for s in range(start_step_count, start_step_count + n_calls):
        ctx = _make_ctx({"main": source}, step_count=s)
        hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)


def _initialized_hook_and_state(
    *,
    seed: int = 0,
    decay: float = 0.5,
) -> tuple[nn.Module, EMAHook, dict[str, Any]]:
    source = _make_linear(seed=seed)
    hook = EMAHook(model_key="main", decay=decay)
    ctx = _make_ctx({"main": source}, step_count=0)
    hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)
    return source, hook, hook.state_dict()


class _VetoFirstOptimizerStepHook(TrainingUpdateHook):
    """Veto the first optimizer step, then allow later steps."""

    priority = 10

    def __init__(self) -> None:
        self.optimizer_step_calls = 0

    def __call__(
        self,
        ctx: TrainContext,
        stage: TrainingStage,
        will_skip: bool,
    ) -> tuple[bool, torch.Tensor | None]:
        if stage is TrainingStage.DO_OPTIMIZER_STEP:
            self.optimizer_step_calls += 1
            return self.optimizer_step_calls > 1, ctx.loss
        return True, ctx.loss


class _CudaBufferResetOnDeepcopy(nn.Module):
    """Exercise EMA repair for modules whose deepcopy loses buffer placement.

    ``AveragedModel`` constructs EMA state by deep-copying the source
    ``nn.Module``. Some generated or monkey-patched modules can reconstruct
    registered buffers on CPU during that copy even when the live training
    module is on CUDA. This fixture creates that failure mode directly so
    EMA tests verify device repair against a real module copy, not a bare
    tensor dictionary.
    """

    def __init__(
        self,
        parameter_device: torch.device,
        buffer_device: torch.device,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones((), device=parameter_device))
        self.register_buffer("constant", torch.ones((), device=buffer_device))

    def __deepcopy__(self, memo: dict[int, Any]) -> _CudaBufferResetOnDeepcopy:
        clone = type(self)(self.weight.device, torch.device("cpu"))
        with torch.no_grad():
            clone.weight.copy_(self.weight)
        memo[id(self)] = clone
        return clone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor that requires parameter and buffer devices to match."""
        return x * self.weight + self.constant


class _CudaBufferOnlyResetOnDeepcopy(nn.Module):
    """Exercise EMA repair for modules with only registered buffers.

    Not every valid ``nn.Module`` has trainable parameters; some wrappers,
    lookup tables, normalizers, or generated helper modules carry their
    device-sensitive state entirely in buffers. This fixture makes the
    deepcopy path reset that buffer to CPU so tests verify EMA device repair
    does not depend on finding a parameter first.
    """

    def __init__(self, buffer_device: torch.device) -> None:
        super().__init__()
        self.register_buffer("constant", torch.ones((), device=buffer_device))

    def __deepcopy__(self, memo: dict[int, Any]) -> _CudaBufferOnlyResetOnDeepcopy:
        clone = type(self)(torch.device("cpu"))
        memo[id(self)] = clone
        return clone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor that requires the buffer to follow the input device."""
        return x + self.constant


class _MixedDeviceBufferOnDeepcopy(nn.Module):
    """Exercise EMA preservation of intentional mixed-device placement.

    The EMA copy should follow each corresponding source tensor, not collapse
    the whole module onto the first parameter's device. This fixture keeps a
    CUDA parameter beside a CPU buffer to guard monkey-patched or third-party
    modules that intentionally store side tables on host while computing with
    device parameters.
    """

    def __init__(
        self,
        parameter_device: torch.device,
        buffer_device: torch.device,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones((), device=parameter_device))
        self.register_buffer("cpu_table", torch.ones((), device=buffer_device))

    def __deepcopy__(self, memo: dict[int, Any]) -> _MixedDeviceBufferOnDeepcopy:
        clone = type(self)(self.weight.device, torch.device("cpu"))
        with torch.no_grad():
            clone.weight.copy_(self.weight)
        memo[id(self)] = clone
        return clone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a CUDA result while leaving the CPU table as side state."""
        return x * self.weight


class _Float64ResetOnDeepcopy(nn.Module):
    """Exercise EMA repair for modules whose deepcopy resets floating dtype."""

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones((), dtype=dtype))
        self.register_buffer("constant", torch.ones((), dtype=dtype))

    def __deepcopy__(self, memo: dict[int, Any]) -> _Float64ResetOnDeepcopy:
        clone = type(self)(torch.float64)
        with torch.no_grad():
            clone.weight.copy_(self.weight)
            clone.constant.copy_(self.constant)
        memo[id(self)] = clone
        return clone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor that requires parameter and buffer dtypes to match."""
        return x * self.weight + self.constant


def _cpu_averaged_state(state: dict[str, Any]) -> dict[str, Any]:
    averaged_state = {
        key: value.cpu() if torch.is_tensor(value) else value
        for key, value in state["averaged_model_state"].items()
    }
    return {**state, "averaged_model_state": averaged_state}


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------


class TestEMAHookConstruction:
    def test_defaults(self) -> None:
        hook = EMAHook()
        assert hook.model_key == "main"
        assert hook.decay == pytest.approx(0.999)
        assert hook.update_every == 1
        assert hook.start_step == 0
        assert hook.use_buffers is True
        assert hook.num_updates == 0
        assert EMAHook.priority == 50
        assert isinstance(hook, TrainingUpdateHook)
        assert hook._averaged_model is None
        assert hook._pending_averaged_state is None

    @pytest.mark.parametrize(
        ("kwargs", "field"),
        [
            pytest.param({"decay": 1.0}, "decay", id="decay_eq_1_rejected"),
            pytest.param({"decay": -0.1}, "decay", id="decay_negative_rejected"),
            pytest.param(
                {"update_every": 0}, "update_every", id="update_every_zero_rejected"
            ),
            pytest.param(
                {"update_every": -1},
                "update_every",
                id="update_every_negative_rejected",
            ),
            pytest.param(
                {"start_step": -1}, "start_step", id="start_step_negative_rejected"
            ),
            pytest.param({"model_key": ""}, "model_key", id="model_key_empty_rejected"),
            pytest.param(
                {"model_key": "   "}, "model_key", id="model_key_whitespace_rejected"
            ),
            pytest.param(
                {"num_updates": -1}, "num_updates", id="num_updates_negative_rejected"
            ),
        ],
    )
    def test_invalid_field_values_raise(
        self, kwargs: dict[str, Any], field: str
    ) -> None:
        with pytest.raises(ValidationError) as excinfo:
            EMAHook(**kwargs)
        # Confirm the error points at the offending field.
        assert any(field in err["loc"] for err in excinfo.value.errors())

    def test_extra_kwargs_rejected(self) -> None:
        with pytest.raises(ValidationError):
            EMAHook(decya=0.9)


class TestEMAHookBuildOverride:
    """The ``_build_averaged_model`` seam lets a caller inject a copy."""

    def test_default_build_deepcopies_source(self) -> None:
        source = _make_linear(seed=0)
        hook = EMAHook(model_key="main", decay=0.5)
        hook(_make_ctx({"main": source}, step_count=0), TrainingStage.SETUP)
        averaged = hook.get_averaged_model()
        # A fresh deepcopy: distinct object, weights mirrored from source.
        assert averaged.module is not source
        assert _params_equal(averaged.module, source)

    def test_override_adopts_prebuilt_without_deepcopy(self) -> None:
        source = _make_linear(seed=0)
        # A pre-built averaged model with deliberately different weights so
        # an accidental deepcopy of ``source`` would be detectable.
        prebuilt = AveragedModel(
            _make_linear(seed=1), multi_avg_fn=None, use_buffers=True
        )

        class _InjectedEMAHook(EMAHook):
            def _build_averaged_model(self, src: nn.Module) -> AveragedModel:
                return prebuilt

        hook = _InjectedEMAHook(model_key="main", decay=0.5)
        hook(_make_ctx({"main": source}, step_count=0), TrainingStage.SETUP)
        # The hook adopted the injected model verbatim — no deepcopy.
        assert hook.get_averaged_model() is prebuilt


# ---------------------------------------------------------------------------
# Single-model update behavior
# ---------------------------------------------------------------------------


class TestEMAHookSingleModelUpdate:
    def setup_method(self) -> None:
        self.source = _make_linear(seed=0)
        self.source_snapshot = _clone_state(self.source)

    def test_single_call_initializes_and_increments(self) -> None:
        hook = EMAHook(model_key="main", decay=0.5)
        ctx = _make_ctx({"main": self.source}, step_count=0)
        hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)
        assert hook.num_updates == 1
        assert hook._averaged_model is not None
        # Source model untouched (hook is observer-only).
        for k, v in self.source.state_dict().items():
            assert torch.equal(v, self.source_snapshot[k])

    def test_decay_zero_matches_source_after_one_update(self) -> None:
        hook = EMAHook(model_key="main", decay=0.0)
        ctx = _make_ctx({"main": self.source}, step_count=0)
        hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)
        averaged = hook.get_averaged_model().module
        for (n, p_src), p_avg in zip(
            self.source.named_parameters(),
            averaged.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(p_src, p_avg, msg=f"param {n} differs")

    def test_no_storage_sharing_with_source(self) -> None:
        """Mutating source after init must not change averaged params."""
        hook = EMAHook(model_key="main", decay=0.0)
        ctx = _make_ctx({"main": self.source}, step_count=0)
        hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)
        averaged = hook.get_averaged_model().module
        averaged_snapshot = _clone_state(averaged)
        for p_src, p_avg in zip(
            self.source.parameters(), averaged.parameters(), strict=True
        ):
            assert id(p_src) != id(p_avg)
            assert p_src.data_ptr() != p_avg.data_ptr()
        with torch.no_grad():
            for p in self.source.parameters():
                p.add_(100.0)
        for k, v in averaged.state_dict().items():
            assert torch.equal(v, averaged_snapshot[k])

    def test_setup_initializes_without_update(self) -> None:
        hook = EMAHook(model_key="main")
        ctx = _make_ctx({"main": self.source}, step_count=0)

        hook(ctx, TrainingStage.SETUP)

        assert hook.num_updates == 0
        assert hook._averaged_model is not None

    def test_non_update_stages_after_setup_do_not_update(self) -> None:
        hook = EMAHook(model_key="main")
        ctx = _make_ctx({"main": self.source}, step_count=0)
        hook(ctx, TrainingStage.SETUP)
        for stage in TrainingStage:
            if stage in (TrainingStage.SETUP, TrainingStage.AFTER_OPTIMIZER_STEP):
                continue
            hook(ctx, stage)
        assert hook.num_updates == 0

    def test_get_averaged_model_before_init_raises(self) -> None:
        hook = EMAHook(model_key="main")
        with pytest.raises(RuntimeError, match="has not initialized"):
            hook.get_averaged_model()


# ---------------------------------------------------------------------------
# model_key selection across multiple models
# ---------------------------------------------------------------------------


class TestEMAHookModelKeySelection:
    def setup_method(self) -> None:
        # Different shapes so we can assert structural identity.
        self.model_a = _make_linear(in_f=4, out_f=4, seed=0)
        self.model_b = _make_linear(in_f=4, out_f=8, seed=1)
        self.snapshot_a = _clone_state(self.model_a)
        self.snapshot_b = _clone_state(self.model_b)

    def test_selects_only_intended_model(self) -> None:
        hook = EMAHook(model_key="ema_target", decay=0.0)
        ctx = _make_ctx(
            {"main": self.model_a, "ema_target": self.model_b},
            step_count=0,
        )
        hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)

        for k, v in self.model_a.state_dict().items():
            assert torch.equal(v, self.snapshot_a[k])
        for k, v in self.model_b.state_dict().items():
            assert torch.equal(v, self.snapshot_b[k])

        averaged = hook.get_averaged_model().module
        assert averaged.weight.shape == self.model_b.weight.shape
        torch.testing.assert_close(averaged.weight, self.model_b.weight)
        torch.testing.assert_close(averaged.bias, self.model_b.bias)

    def test_unmatched_models_untouched(self) -> None:
        hook = EMAHook(model_key="ema_target", decay=0.5)
        ctx = _make_ctx(
            {"main": self.model_a, "ema_target": self.model_b},
            step_count=0,
        )
        a_param_ids_before = {id(p) for p in self.model_a.parameters()}
        hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)
        a_param_ids_after = {id(p) for p in self.model_a.parameters()}
        assert a_param_ids_before == a_param_ids_after
        for k, v in self.model_a.state_dict().items():
            assert torch.equal(v, self.snapshot_a[k])

    def test_two_hooks_average_independently(self) -> None:
        hook1 = EMAHook(model_key="m1", decay=0.0)
        hook2 = EMAHook(model_key="m2", decay=0.0)
        ctx = _make_ctx({"m1": self.model_a, "m2": self.model_b}, step_count=0)
        hook1(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)
        hook2(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)

        avg1 = hook1.get_averaged_model()
        avg2 = hook2.get_averaged_model()
        assert avg1 is not avg2
        assert avg1.module.weight.shape == self.model_a.weight.shape
        assert avg2.module.weight.shape == self.model_b.weight.shape
        ids1 = {p.data_ptr() for p in avg1.parameters()}
        ids2 = {p.data_ptr() for p in avg2.parameters()}
        assert ids1.isdisjoint(ids2)
        assert hook1.num_updates == 1
        assert hook2.num_updates == 1

    def test_missing_model_key_raises_keyerror(self) -> None:
        hook = EMAHook(model_key="ghost")
        ctx = _make_ctx({"main": self.model_a}, step_count=0)
        with pytest.raises(KeyError) as excinfo:
            hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)
        msg = str(excinfo.value)
        assert "'ghost'" in msg
        assert "['main']" in msg


# ---------------------------------------------------------------------------
# Step filtering: update_every and start_step
# ---------------------------------------------------------------------------


class TestEMAHookStepFiltering:
    def setup_method(self) -> None:
        self.source = _make_linear(seed=0)

    def test_update_every_skips_intermediate_steps(self) -> None:
        # step_count=0..6 => completed=1..7; multiples of 3 are 3, 6 => 2 updates.
        hook = EMAHook(model_key="main", update_every=3)
        _drive(hook, self.source, n_calls=7)
        assert hook.num_updates == 2

    def test_update_every_one_fires_every_step(self) -> None:
        hook = EMAHook(model_key="main", update_every=1)
        _drive(hook, self.source, n_calls=5)
        assert hook.num_updates == 5

    def test_start_step_delays_first_update(self) -> None:
        hook = EMAHook(model_key="main", start_step=5, update_every=1)
        # step_count=0..3 => completed=1..4 < 5: no-op.
        _drive(hook, self.source, n_calls=4)
        assert hook.num_updates == 0
        assert hook._averaged_model is None
        # step_count=4 => completed=5: first update fires.
        _drive(hook, self.source, n_calls=1, start_step_count=4)
        assert hook.num_updates == 1
        assert hook._averaged_model is not None
        # step_count=5..9 => completed=6..10: 5 more updates, total 6.
        _drive(hook, self.source, n_calls=5, start_step_count=5)
        assert hook.num_updates == 6

    def test_global_modulo_with_start_step_and_update_every(self) -> None:
        """``update_every`` is a *global* modulo on completed_step, not relative to start_step."""
        hook = EMAHook(model_key="main", start_step=5, update_every=10)
        # completed=1..15: only completed=10 is eligible.
        _drive(hook, self.source, n_calls=15)
        assert hook.num_updates == 1
        # completed=16..20: completed=20 is the next eligible step.
        _drive(hook, self.source, n_calls=5, start_step_count=15)
        assert hook.num_updates == 2


# ---------------------------------------------------------------------------
# No mutation of grads / optimizer / scaler
# ---------------------------------------------------------------------------


class TestEMAHookSideEffects:
    def test_gradients_and_optimizer_state_untouched(self) -> None:
        source = _make_linear(seed=0)
        x = torch.randn(2, 4)
        target = torch.randn(2, 4)
        loss = ((source(x) - target) ** 2).mean()
        loss.backward()
        grad_snapshots = {
            n: p.grad.detach().clone() for n, p in source.named_parameters()
        }

        optimizer_mock = MagicMock(spec=torch.optim.Optimizer)
        hook = EMAHook(model_key="main")
        ctx = _make_ctx({"main": source}, step_count=0, optimizers=[optimizer_mock])
        hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)

        for n, p in source.named_parameters():
            torch.testing.assert_close(p.grad, grad_snapshots[n])
        # Optimizer mock was never called or method-accessed in any way.
        assert optimizer_mock.method_calls == []
        assert optimizer_mock.mock_calls == []

    def test_amp_autocast_smoke(self) -> None:
        """EMAHook runs without error under torch.amp.autocast (no AMP-API coupling)."""
        source = _make_linear(seed=0)
        hook = EMAHook(model_key="main", decay=0.5)
        ctx = _make_ctx({"main": source}, step_count=0)

        with torch.amp.autocast(device_type="cpu", dtype=torch.bfloat16):
            hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)

        assert hook.num_updates == 1

    def test_skipped_optimizer_step_does_not_update_ema(self) -> None:
        source = _make_linear(seed=0)
        hook = EMAHook(model_key="main", decay=0.5)
        ctx = _make_ctx({"main": source}, step_count=0)

        hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP, will_skip=True)

        assert hook.num_updates == 0
        assert hook._averaged_model is None

    def test_averaged_copy_follows_source_floating_dtypes(self) -> None:
        source = _Float64ResetOnDeepcopy(torch.float32)
        hook = EMAHook(model_key="main", decay=0.5)

        hook(
            _make_ctx({"main": source}, step_count=0),
            TrainingStage.AFTER_OPTIMIZER_STEP,
        )
        with torch.no_grad():
            source.weight.fill_(3.0)
            source.constant.fill_(5.0)
        hook(
            _make_ctx({"main": source}, step_count=1),
            TrainingStage.AFTER_OPTIMIZER_STEP,
        )

        averaged = hook.get_averaged_model().module
        assert averaged.weight.dtype is torch.float32
        assert averaged.constant.dtype is torch.float32
        out = averaged(torch.ones((), dtype=torch.float32))
        assert out.dtype is torch.float32

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_averaged_copy_and_state_restore_follow_source_tensor_devices(
        self,
    ) -> None:
        device = torch.device("cuda:0")
        source = _CudaBufferResetOnDeepcopy(device, device)

        # First prove the lazy AveragedModel construction path repairs the
        # deepcopy artifact: the source buffer is CUDA, but this test module's
        # __deepcopy__ reconstructs the averaged buffer on CPU.
        hook = EMAHook(model_key="main", decay=0.0)
        ctx = _make_ctx({"main": source}, step_count=0)
        ctx.workflow = object()

        hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)

        averaged = hook.get_averaged_model().module
        assert averaged.constant.device == device
        out = averaged(torch.ones((), device=device))
        torch.testing.assert_close(out, torch.tensor(2.0, device=device))

        # Simulate a checkpoint loaded on CPU before EMA has seen the live
        # training model. load_state_dict must stash this as pending state,
        # then first EMA update must build the averaged model and reapply the
        # source tensor devices after loading that CPU state.
        cpu_state = _cpu_averaged_state(hook.state_dict())
        restored = EMAHook(model_key="main", decay=0.0)
        restored.load_state_dict(cpu_state)
        restored_ctx = _make_ctx({"main": source}, step_count=1)
        restored_ctx.workflow = object()

        restored(restored_ctx, TrainingStage.AFTER_OPTIMIZER_STEP)

        restored_averaged = restored.get_averaged_model().module
        assert restored_averaged.constant.device == device
        restored_out = restored_averaged(torch.ones((), device=device))
        torch.testing.assert_close(restored_out, torch.tensor(2.0, device=device))

        # Also cover the already-initialized restore path. This is the branch
        # used when an EMA hook has a live AveragedModel and then receives a
        # checkpoint state whose tensors were materialized on CPU.
        initialized = EMAHook(model_key="main", decay=0.0)
        initialized_ctx = _make_ctx({"main": source}, step_count=0)
        initialized_ctx.workflow = object()
        initialized(initialized_ctx, TrainingStage.AFTER_OPTIMIZER_STEP)

        initialized.load_state_dict(cpu_state)

        initialized_averaged = initialized.get_averaged_model().module
        assert initialized_averaged.constant.device == device
        initialized_out = initialized_averaged(torch.ones((), device=device))
        torch.testing.assert_close(initialized_out, torch.tensor(2.0, device=device))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_buffer_only_averaged_copy_follows_source_device(self) -> None:
        device = torch.device("cuda:0")
        source = _CudaBufferOnlyResetOnDeepcopy(device)
        hook = EMAHook(model_key="main", decay=0.0)
        ctx = _make_ctx({"main": source}, step_count=0)
        ctx.workflow = object()

        hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)

        averaged = hook.get_averaged_model().module
        assert averaged.constant.device == device
        out = averaged(torch.ones((), device=device))
        torch.testing.assert_close(out, torch.tensor(2.0, device=device))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_mixed_device_averaged_copy_preserves_source_buffer_device(
        self,
    ) -> None:
        device = torch.device("cuda:0")
        source = _MixedDeviceBufferOnDeepcopy(device, torch.device("cpu"))
        hook = EMAHook(model_key="main", decay=0.0)
        ctx = _make_ctx({"main": source}, step_count=0)
        ctx.workflow = object()

        hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)

        averaged = hook.get_averaged_model().module
        assert averaged.weight.device == device
        assert averaged.cpu_table.device == torch.device("cpu")
        out = averaged(torch.ones((), device=device))
        torch.testing.assert_close(out, torch.tensor(1.0, device=device))


class TestEMAHookStrategyIntegration:
    def test_strategy_autowrap_updates_after_successful_optimizer_steps(self) -> None:
        ema = EMAHook(model_key="main", decay=0.0)
        veto_first = _VetoFirstOptimizerStepHook()
        strategy = TrainingStrategy(
            **{
                **_build_baseline_strategy_kwargs(),
                "num_epochs": None,
                "num_steps": 1,
                "hooks": [veto_first, ema],
            }
        )

        strategy.run([_build_batch(seed=0), _build_batch(seed=10)])

        assert strategy.batch_count == 2
        assert strategy.step_count == 1
        assert veto_first.optimizer_step_calls == 2
        assert ema.num_updates == 1
        averaged = ema.get_averaged_model().module
        for source_param, averaged_param in zip(
            strategy.models["main"].parameters(),
            averaged.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(averaged_param, source_param)


# ---------------------------------------------------------------------------
# Checkpointing: state_dict / load_state_dict
# ---------------------------------------------------------------------------


class TestEMAHookCheckpoint:
    def test_state_dict_contains_config_and_averaged_state(self) -> None:
        source = _make_linear(seed=0)
        hook = EMAHook(model_key="main", decay=0.5, update_every=2, start_step=1)
        ctx = _make_ctx({"main": source}, step_count=1)  # completed=2
        hook(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)
        state = hook.state_dict()
        assert {
            "model_key",
            "decay",
            "update_every",
            "start_step",
            "use_buffers",
            "num_updates",
        } <= state.keys()
        assert state["num_updates"] == 1
        assert "averaged_model_state" in state
        assert isinstance(state["averaged_model_state"], dict)

    def test_round_trip_num_updates_and_weights(self) -> None:
        source_a, hook_a, _ = _initialized_hook_and_state(seed=0, decay=0.5)
        # Run a second update with perturbed source so EMA != source.
        with torch.no_grad():
            for p in source_a.parameters():
                p.add_(0.5)
        _drive(hook_a, source_a, n_calls=1, start_step_count=1)
        assert hook_a.num_updates == 2
        state = hook_a.state_dict()

        # Build B with same config, init via a call on a different source, then load.
        source_b, hook_b, _ = _initialized_hook_and_state(seed=99, decay=0.5)
        avg_a = hook_a.get_averaged_model().module
        avg_b = hook_b.get_averaged_model().module
        assert not _params_equal(avg_a, avg_b)

        hook_b.load_state_dict(state)
        assert hook_b.num_updates == hook_a.num_updates
        avg_b = hook_b.get_averaged_model().module
        for k in avg_a.state_dict():
            torch.testing.assert_close(avg_b.state_dict()[k], avg_a.state_dict()[k])
        assert hook_b._pending_averaged_state is None

    def test_pending_state_applied_on_first_call(self) -> None:
        """Pending weights must be loaded BEFORE the first update, not after."""
        decay = 0.5
        source_a, hook_a, state_a = _initialized_hook_and_state(seed=0, decay=decay)
        loaded_pending = {
            k: v.detach().clone()
            for k, v in hook_a.get_averaged_model().module.state_dict().items()
        }

        hook_b = EMAHook(model_key="main", decay=decay)
        hook_b.load_state_dict(state_a)
        assert hook_b._averaged_model is None
        assert hook_b._pending_averaged_state is not None

        source_b = _make_linear(seed=99)
        source_b_snapshot = _clone_state(source_b)
        ctx_b = _make_ctx({"main": source_b}, step_count=10_000)
        hook_b(ctx_b, TrainingStage.AFTER_OPTIMIZER_STEP)

        assert hook_b._averaged_model is not None
        assert hook_b._pending_averaged_state is None
        assert hook_b.num_updates == hook_a.num_updates + 1

        # Verify avg = decay * pending + (1 - decay) * source_b on parameters.
        # Buffers may use a different averaging rule when use_buffers=True.
        averaged = hook_b.get_averaged_model().module
        param_keys = {n for n, _ in averaged.named_parameters()}
        avg_state = averaged.state_dict()
        for key in param_keys:
            expected = (
                decay * loaded_pending[key] + (1.0 - decay) * source_b_snapshot[key]
            )
            torch.testing.assert_close(
                avg_state[key], expected, msg=f"EMA formula mismatch on {key!r}"
            )
        # If pending were ignored, the first AveragedModel update would copy
        # source_b verbatim regardless of multi_avg_fn.
        for key in param_keys:
            assert not torch.equal(avg_state[key], source_b_snapshot[key])

    def test_save_before_init_emits_pending_state(self) -> None:
        _, hook_a, state_a = _initialized_hook_and_state(seed=0, decay=0.5)

        hook_b = EMAHook(model_key="main", decay=0.5)
        hook_b.load_state_dict(state_a)
        state_b = hook_b.state_dict()
        assert "averaged_model_state" in state_b
        # Verify by content, not identity.
        emitted = state_b["averaged_model_state"]
        original = state_a["averaged_model_state"]
        assert emitted.keys() == original.keys()
        for k in emitted:
            torch.testing.assert_close(emitted[k], original[k])

    def test_partial_load_preserves_num_updates(self) -> None:
        hook = EMAHook(model_key="main", decay=0.999)
        hook.num_updates = 5
        hook.load_state_dict({"decay": 0.999})
        assert hook.num_updates == 5

    def test_load_clears_averaged_state_when_absent(self) -> None:
        _, hook_a, state_a = _initialized_hook_and_state(seed=0, decay=0.5)

        pending_hook = EMAHook(model_key="main", decay=0.5)
        pending_hook.load_state_dict(state_a)
        assert pending_hook._pending_averaged_state is not None

        # Subsequent load that omits averaged_model_state should clear pending state.
        pending_hook.load_state_dict({"decay": 0.5})
        assert pending_hook._averaged_model is None
        assert pending_hook._pending_averaged_state is None

        _, initialized_hook, _ = _initialized_hook_and_state(seed=99, decay=0.5)
        assert initialized_hook._averaged_model is not None

        initialized_hook.load_state_dict({"decay": 0.5})
        assert initialized_hook._averaged_model is None
        assert initialized_hook._pending_averaged_state is None

    def test_config_conflict_raises_value_error_with_format(self) -> None:
        hook = EMAHook(model_key="main", decay=0.999)
        with pytest.raises(ValueError) as excinfo:
            hook.load_state_dict({"decay": 0.9})
        msg = str(excinfo.value)
        assert "EMAHook checkpoint conflict:" in msg
        assert "decay=0.9" in msg
        assert "constructor decay=0.999" in msg
        assert "construct the hook with matching config" in msg

    def test_config_conflict_on_model_key(self) -> None:
        hook = EMAHook(model_key="main")
        with pytest.raises(ValueError, match="EMAHook checkpoint conflict: model_key="):
            hook.load_state_dict({"model_key": "ema"})

    def test_load_after_live_init_overwrites_weights(self) -> None:
        _, hook_a, state_a = _initialized_hook_and_state(seed=0, decay=0.5)
        _, hook_b, _ = _initialized_hook_and_state(seed=99, decay=0.5)

        avg_a = hook_a.get_averaged_model().module
        avg_b = hook_b.get_averaged_model().module
        assert not _params_equal(avg_a, avg_b)

        hook_b.load_state_dict(state_a)
        avg_b = hook_b.get_averaged_model().module
        for k in avg_a.state_dict():
            torch.testing.assert_close(avg_b.state_dict()[k], avg_a.state_dict()[k])
        assert hook_b._pending_averaged_state is None


# ---------------------------------------------------------------------------
# Inference-model write via set_inference_model (Phase C)
# ---------------------------------------------------------------------------


class TestInferenceModelWrite:
    """EMAHook publishes averaged weights into the strategy inference_model slot."""

    def test_single_model_publishes_bare_module(self) -> None:
        """After eligible AFTER_OPTIMIZER_STEP, strategy.inference_model is a bare Module."""
        ema = EMAHook(model_key="main", decay=0.0)
        strategy = TrainingStrategy(
            **{
                **_build_baseline_strategy_kwargs(),
                "num_epochs": None,
                "num_steps": 1,
                "hooks": [ema],
            }
        )
        assert strategy.inference_model is None
        strategy.run([_build_batch(seed=0)])
        assert strategy.inference_model is not None
        assert isinstance(strategy.inference_model, nn.Module)
        assert not isinstance(strategy.inference_model, nn.ModuleDict)
        averaged_module = ema.get_averaged_model().module
        assert strategy.inference_model is averaged_module

    def test_two_hooks_produce_moduledict(self) -> None:
        """Two EMA hooks with distinct model_keys produce an nn.ModuleDict."""
        model_a = _make_linear(in_f=4, out_f=4, seed=0)
        model_b = _make_linear(in_f=4, out_f=4, seed=1)

        ema_a = EMAHook(model_key="m1", decay=0.0)
        ema_b = EMAHook(model_key="m2", decay=0.0)

        # Use a lightweight workflow stub that has set_inference_model
        # and single_model_input=False, avoiding full strategy construction.
        class _WorkflowStub:
            single_model_input = False
            inference_model: nn.Module | nn.ModuleDict | None = None

            def set_inference_model(
                self, module: nn.Module, *, model_key: str | None = None
            ) -> None:
                if model_key is None or self.single_model_input:
                    self.inference_model = module
                    return
                if not isinstance(self.inference_model, nn.ModuleDict):
                    self.inference_model = nn.ModuleDict()
                self.inference_model[model_key] = module

        workflow = _WorkflowStub()
        ctx = Mock(
            spec=TrainContext,
            models={"m1": model_a, "m2": model_b},
            step_count=0,
            optimizers=[],
            loss=None,
            workflow=workflow,
        )
        ema_a(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)
        ema_b(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)

        assert isinstance(workflow.inference_model, nn.ModuleDict)
        assert "m1" in workflow.inference_model
        assert "m2" in workflow.inference_model
        assert workflow.inference_model["m1"] is ema_a.get_averaged_model().module
        assert workflow.inference_model["m2"] is ema_b.get_averaged_model().module

    def test_setup_publishes_before_start_step_without_update(self) -> None:
        """SETUP publishes an initial EMA model while start_step still gates updates."""
        ema = EMAHook(model_key="main", decay=0.0, start_step=100)
        strategy = TrainingStrategy(
            **{
                **_build_baseline_strategy_kwargs(),
                "num_epochs": None,
                "num_steps": 1,
                "hooks": [ema],
            }
        )
        assert strategy.inference_model is None
        strategy.run([_build_batch(seed=0)])
        # SETUP initializes the inference model; start_step still prevents updates.
        assert ema.num_updates == 0
        assert strategy.inference_model is ema.get_averaged_model().module

    def test_setup_materializes_pending_checkpoint_state(self) -> None:
        """SETUP can publish restored EMA state before another train step."""
        source = _build_baseline_strategy_kwargs()["models"]
        initialized = EMAHook(model_key="main", decay=0.0)
        initialized(
            _make_ctx({"main": source}, step_count=0),
            TrainingStage.AFTER_OPTIMIZER_STEP,
        )
        state = initialized.state_dict()
        restored = EMAHook(model_key="main", decay=0.0)
        restored.load_state_dict(state)

        strategy = TrainingStrategy(
            **{
                **_build_baseline_strategy_kwargs(),
                "num_epochs": None,
                "num_steps": 1,
                "hooks": [restored],
            }
        )
        strategy.validation_config = ValidationConfig(
            validation_data=[_build_batch(seed=1)],
            use_ema="always",
        )

        assert strategy.inference_model is None
        assert restored._averaged_model is None
        assert restored._pending_averaged_state is not None

        # Simulate a restored strategy that has already reached its target;
        # run() still executes SETUP hooks, then returns before another train step.
        strategy.step_count = 1
        strategy.run([_build_batch(seed=1)])

        assert strategy.inference_model is restored.get_averaged_model().module
        assert restored._pending_averaged_state is None

        summary = strategy.validate()

        assert summary is not None
        assert summary["model_source"] == "ema"
        assert restored.num_updates == initialized.num_updates
        assert strategy.inference_model is restored.get_averaged_model().module

    def test_no_crash_without_set_inference_model(self) -> None:
        """EMAHook works when workflow lacks set_inference_model (defensive guard)."""
        ema = EMAHook(model_key="main", decay=0.0)
        source = _make_linear(seed=0)
        ctx = Mock(
            spec=TrainContext,
            models={"main": source},
            step_count=0,
            optimizers=[],
            loss=None,
        )
        # workflow with no set_inference_model attribute
        ctx.workflow = object()
        ema(ctx, TrainingStage.AFTER_OPTIMIZER_STEP)
        assert ema.num_updates == 1
