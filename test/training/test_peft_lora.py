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
"""Tests for LoRA fine-tuning strategy and related hooks."""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from nvalchemi.training import (
    CheckpointHook,
    FineTuningStrategy,
    OptimizerConfig,
    TrainingStrategy,
    create_model_spec,
)
from nvalchemi.training.hooks import (
    BaseFingerprintHook,
    ModulePatchHook,
    TrainableParameterHook,
)
from nvalchemi.training.peft import load_peft_checkpoint_into_model, lora_wrappers
from nvalchemi.training.peft.lora import (
    LoRAConfig,
    is_lora_layer,
    merge_lora_into_model,
)
from nvalchemi.training.peft.lora_hook import LoRAHook
from nvalchemi.training.peft.lora_wrappers import (
    CuEquivarianceLoRALinear,
    E3NNFullyConnectedLoRALayer,
    EquivariantLoRALinear,
    LoRALayer,
)

_PARTIAL_CHECKPOINT_HOOK_WARNING = "Saving a checkpoint with save_trainable_state_only=True stores only optimizer-selected parameters and buffers."

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lora_strategy_kwargs(**kwargs: Any) -> dict[str, Any]:
    """Return generic fine-tuning kwargs with LoRA PEFT config."""
    defaults: dict[str, Any] = {
        "lora_rank": 1,
        "lora_alpha": 1.0,
        "lora_dropout": 0.0,
        "lora_target_patterns": ("main.model.projection",),
        "lora_wrap_mlp": False,
        "lora_wrapper_registrations": None,
        "compute_base_fingerprints": True,
    }
    defaults.update(kwargs)
    return {
        "compute_base_fingerprints": defaults["compute_base_fingerprints"],
        "peft_config": LoRAConfig(
            rank=defaults["lora_rank"],
            alpha=defaults["lora_alpha"],
            lora_dropout=defaults["lora_dropout"],
            lora_target_patterns=defaults["lora_target_patterns"],
            wrap_mlp=defaults["lora_wrap_mlp"],
            wrapper_registrations=defaults["lora_wrapper_registrations"],
        ),
    }


class _FakeLoRAResult:
    """Small stand-in for PhysicsNeMo apply_lora result metadata."""

    base_fingerprint = "fingerprint-ok"
    n_wrapped = 1
    n_trainable = 2
    n_frozen = 1
    trainable_names = ["model.lora_adapter.weight", "model.lora_adapter.bias"]


def _fake_lora_linear() -> nn.Linear:
    """Return a linear layer marked as a fake LoRA layer."""
    layer = nn.Linear(8, 8)
    layer._fake_lora = True
    return layer


class _Recorder:
    """Record generated hook state after registration."""

    frequency = 1
    stage = None

    def __init__(self) -> None:
        self.saw_base_fingerprints = False
        self.saw_aux_projection = False
        self.saw_optimizer_filter = False

    def _runs_on_stage(self, stage: Any) -> bool:  # noqa: ARG002
        return False

    def on_register(self, workflow: Any) -> None:
        self.saw_base_fingerprints = hasattr(
            workflow,
            "_base_fingerprints",
        )
        self.saw_aux_projection = hasattr(
            workflow.models["main"].model, "aux_projection"
        )
        self.saw_optimizer_filter = workflow._optimizer_parameter_names is not None

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        return


class _CustomCheckpointLoRAWrapper(nn.Module, LoRALayer):
    """Importable custom LoRA wrapper used by checkpoint trust-policy tests."""

    def __init__(self, base_layer: nn.Module, **kwargs: Any) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate the wrapped base layer."""
        return self.base_layer(x)


class _CustomCheckpointPatch(nn.Linear):
    """Importable custom module patch used by checkpoint trust-policy tests."""


class _CustomLoRAWrapper(_CustomCheckpointLoRAWrapper):
    """Custom wrapper used by registration tests."""


class _AlternateLoRAWrapper(_CustomCheckpointLoRAWrapper):
    """Alternate wrapper used by conflict-validation tests."""


def _import_real_o3() -> Any:
    """Import real e3nn.o3 with the PyTorch 2.6 safe-global shim."""
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    return pytest.importorskip("e3nn.o3")


def _import_e3nn_assert_equivariant() -> Any:
    """Import e3nn's equivariance assertion with the PyTorch 2.6 safe-global shim."""
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    test_utils = pytest.importorskip("e3nn.util.test")
    return test_utils.assert_equivariant


def _import_real_e3nn_fc_layer() -> type[nn.Module]:
    """Import real e3nn fully connected layer with the PyTorch 2.6 safe-global shim."""
    if hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals([slice])
    fc_module = pytest.importorskip("e3nn.nn._fc")
    return fc_module._Layer


def _import_real_cueq_linear() -> tuple[Any, type[nn.Module]]:
    """Import real cuEquivariance objects needed by cuEq LoRA tests."""
    cue = pytest.importorskip("cuequivariance")
    linear_module = pytest.importorskip("cuequivariance_torch.operations.linear")
    return cue, linear_module.Linear


def _cueq_irreps(spec: str) -> object:
    """Return real O(3) cuEquivariance irreps for tests."""
    cue, _linear_cls = _import_real_cueq_linear()
    return cue.Irreps("O3", spec)


def _make_cueq_linear(
    irreps_in: object,
    irreps_out: object,
    **kwargs: Any,
) -> nn.Module:
    """Return a real cuEq Linear layer with deterministic test defaults."""
    cue, linear_cls = _import_real_cueq_linear()
    defaults = {
        "layout": cue.mul_ir,
        "internal_weights": True,
        "shared_weights": True,
        "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "dtype": torch.float64,
        "method": "naive",
    }
    defaults.update(kwargs)
    return linear_cls(irreps_in, irreps_out, **defaults)


def _install_fake_peft(
    monkeypatch: pytest.MonkeyPatch,
    *,
    current_fingerprint: str = "fingerprint-ok",
    leave_lora_after_merge: bool = False,
    apply_lora_calls: list[tuple[nn.Module, Any]] | None = None,
) -> None:
    """Install a deterministic fake PhysicsNeMo PEFT surface."""

    class _ConfiguredLoRAResult(_FakeLoRAResult):
        base_fingerprint = current_fingerprint

    def fake_apply_lora(
        model: nn.Module,
        config: Any,
        *,
        compute_fingerprint: bool = True,
    ) -> _FakeLoRAResult:
        assert compute_fingerprint is False
        if apply_lora_calls is not None:
            apply_lora_calls.append((model, config))
        layer = nn.Linear(8, 8)
        layer._fake_lora = True
        model.model.lora_adapter = layer
        return _ConfiguredLoRAResult()

    def fake_merge_lora(model: nn.Module) -> nn.Module:
        if hasattr(model.model, "lora_adapter") and not leave_lora_after_merge:
            model.model.lora_adapter._fake_lora = False
        return model

    monkeypatch.setattr("nvalchemi.training.peft._peft.apply_lora", fake_apply_lora)
    monkeypatch.setattr("nvalchemi.training.peft._peft.merge_lora", fake_merge_lora)
    monkeypatch.setattr(
        "nvalchemi.training.peft._peft.is_lora_layer",
        lambda module: bool(getattr(module, "_fake_lora", False)),
    )
    monkeypatch.setattr(
        "nvalchemi.training.peft._peft.compute_base_fingerprint",
        lambda model: current_fingerprint,
    )
    monkeypatch.setattr(
        lora_wrappers,
        "_BUILTIN_LORA_WRAPPER_FACTORIES",
        (),
    )


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


class TestLoRARegistration:
    def test_builtin_method_is_registered_on_lookup(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from nvalchemi.training.peft import lora, registry

        monkeypatch.setattr(registry, "_REGISTRATIONS_BY_METHOD", {})
        monkeypatch.setattr(registry, "_REGISTRATIONS_BY_CONFIG", {})

        registration = registry.get_peft_registration_by_method(lora.LORA_PEFT_METHOD)

        assert registration.config_cls is lora.LoRAConfig


class TestLoRAHook:
    def test_lora_strategy_applies_real_physicsnemo_peft(
        self,
        baseline_strategy_kwargs: dict[str, Any],
    ) -> None:
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                **_lora_strategy_kwargs(),
            }
        )

        projection = strategy.models["main"].model.projection
        registered_trainable_names = strategy.get_registered_trainable_parameter_names()
        lora_parameter_names = registered_trainable_names["lora"]

        assert is_lora_layer(projection)
        assert lora_parameter_names == frozenset(
            {
                "main.model.projection.lora_A",
                "main.model.projection.lora_B",
            }
        )
        assert sorted(name.removeprefix("main.") for name in lora_parameter_names) == [
            "model.projection.lora_A",
            "model.projection.lora_B",
        ]
        assert "main.model.projection.lora_A" in strategy._optimizer_parameter_names
        assert "main.model.projection.lora_B" in strategy._optimizer_parameter_names
        assert (
            "main.model.projection.base_layer.weight"
            not in strategy._optimizer_parameter_names
        )

    def test_lora_target_patterns_record_cross_model_glob_matches(
        self,
        baseline_strategy_kwargs: dict[str, Any],
    ) -> None:
        from test.training.conftest import _build_baseline_strategy_kwargs
        from test.training.test_strategy import dict_demo_training_fn

        model_a = baseline_strategy_kwargs["models"]
        model_b = _build_baseline_strategy_kwargs()["models"]
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                "models": {"modelA": model_a, "modelB": model_b},
                "optimizer_configs": {
                    "modelA": [OptimizerConfig(optimizer_cls=torch.optim.Adam)],
                    "modelB": [OptimizerConfig(optimizer_cls=torch.optim.Adam)],
                },
                "training_fn": dict_demo_training_fn,
                **_lora_strategy_kwargs(
                    lora_target_patterns=(
                        "*.model.*.0",
                        "modelA.model.projection",
                    ),
                ),
            }
        )

        expected_modules = [
            "model.coord_embedding.0",
            "model.joint_mlp.0",
        ]
        model_a_lora_modules = [
            name
            for name, module in strategy.models["modelA"].named_modules()
            if is_lora_layer(module)
        ]
        model_b_lora_modules = [
            name
            for name, module in strategy.models["modelB"].named_modules()
            if is_lora_layer(module)
        ]
        assert model_a_lora_modules == [
            *expected_modules,
            "model.projection",
        ]
        assert model_b_lora_modules == expected_modules
        assert strategy.to_spec_dict()["peft_config"]["lora_target_patterns"] == [
            "*.model.*.0",
            "modelA.model.projection",
        ]

    def test_lora_target_patterns_reject_model_local_names(
        self,
        baseline_strategy_kwargs: dict[str, Any],
    ) -> None:
        with pytest.raises(ValueError, match="did not match any module"):
            FineTuningStrategy(
                **{
                    **baseline_strategy_kwargs,
                    **_lora_strategy_kwargs(
                        lora_target_patterns=("model.projection",),
                    ),
                }
            )


class TestLoRATrainableParameterRegistration:
    def test_lora_trainable_filter_allows_adapters_and_patches_but_not_wrapped_base(
        self,
        monkeypatch: pytest.MonkeyPatch,
        baseline_strategy_kwargs: dict[str, Any],
    ) -> None:
        class _FakeLoRAWrapper(nn.Module):
            def __init__(self, base_layer: nn.Module) -> None:
                super().__init__()
                self.base_layer = base_layer
                self.lora_down = nn.Linear(8, 1, bias=False)
                self.lora_up = nn.Linear(1, 1, bias=False)
                self._fake_lora = True

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.base_layer(x) + self.lora_up(self.lora_down(x))

        class _ProjectionLoRAResult(_FakeLoRAResult):
            trainable_names = [
                "model.projection.lora_down.weight",
                "model.projection.lora_up.weight",
            ]

        def fake_apply_lora(
            model: nn.Module,
            config: Any,  # noqa: ARG001
            *,
            compute_fingerprint: bool = True,
        ) -> _ProjectionLoRAResult:
            assert compute_fingerprint is False
            model.model.projection = _FakeLoRAWrapper(model.model.projection)
            return _ProjectionLoRAResult()

        monkeypatch.setattr("nvalchemi.training.peft._peft.apply_lora", fake_apply_lora)
        monkeypatch.setattr(
            "nvalchemi.training.peft._peft.is_lora_layer",
            lambda module: bool(getattr(module, "_fake_lora", False)),
        )
        monkeypatch.setattr(
            "nvalchemi.training.peft.lora_wrappers._register_lora_wrappers",
            lambda *args, **kwargs: None,
        )
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                **_lora_strategy_kwargs(),
                "module_patches": {"main.model.aux_projection": nn.Linear(8, 1)},
                "trainable_patterns": ("main.model.projection.*",),
            }
        )

        assert strategy._optimizer_parameter_names is not None
        assert {
            "main.model.projection.lora_down.weight",
            "main.model.projection.lora_up.weight",
            "main.model.aux_projection.weight",
            "main.model.aux_projection.bias",
        }.issubset(strategy._optimizer_parameter_names)
        assert "main.model.projection.base_layer.weight" not in (
            strategy._optimizer_parameter_names
        )
        assert "main.model.projection.base_layer.bias" not in (
            strategy._optimizer_parameter_names
        )

    def test_lora_trainable_filter_rejects_stale_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
        baseline_strategy_kwargs: dict[str, Any],
    ) -> None:
        """Reject apply_lora metadata that names parameters absent from the model."""

        class _StaleLoRAResult(_FakeLoRAResult):
            trainable_names = ["model.missing_adapter.weight"]

        def fake_apply_lora(
            model: nn.Module,
            config: Any,
            *,
            compute_fingerprint: bool = True,
        ) -> _StaleLoRAResult:
            # Ensure LoRAHook disables fingerprinting when delegating to apply_lora.
            assert compute_fingerprint is False
            model.model.lora_adapter = _fake_lora_linear()
            return _StaleLoRAResult()

        monkeypatch.setattr("nvalchemi.training.peft._peft.apply_lora", fake_apply_lora)
        monkeypatch.setattr(
            "nvalchemi.training.peft._peft.is_lora_layer",
            lambda module: bool(getattr(module, "_fake_lora", False)),
        )
        monkeypatch.setattr(
            "nvalchemi.training.peft.lora_wrappers._register_lora_wrappers",
            lambda *args, **kwargs: None,
        )

        with pytest.raises(RuntimeError, match="not present"):
            FineTuningStrategy(
                **{**baseline_strategy_kwargs, **_lora_strategy_kwargs()}
            )


class TestLoRAStrategy:
    def test_accepts_flat_peft_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
        baseline_strategy_kwargs: dict[str, Any],
    ) -> None:
        _install_fake_peft(monkeypatch)
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                "peft_config": {
                    "peft_method": "lora",
                    "rank": 1,
                    "alpha": 1.0,
                    "lora_target_patterns": ["main.model.projection"],
                },
            }
        )

        assert isinstance(strategy.peft_config, LoRAConfig)
        assert strategy.peft_config.peft_method == "lora"
        assert strategy.peft_config.rank == 1
        assert strategy.peft_config.lora_target_patterns == ("main.model.projection",)

    def test_lora_strategy_prepends_generated_hooks(
        self,
        monkeypatch: pytest.MonkeyPatch,
        baseline_strategy_kwargs: dict[str, Any],
    ) -> None:
        calls: list[tuple[nn.Module, Any]] = []
        _install_fake_peft(monkeypatch, apply_lora_calls=calls)
        recorder = _Recorder()
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                **_lora_strategy_kwargs(
                    lora_target_patterns=("main.model.projection",),
                    compute_base_fingerprints=True,
                ),
                "module_patches": {"main.model.aux_projection": nn.Linear(8, 1)},
                "trainable_patterns": ("main.model.projection.*",),
                "hooks": [recorder],
            }
        )

        assert isinstance(strategy.hooks[0], BaseFingerprintHook)
        assert isinstance(strategy.hooks[1], LoRAHook)
        assert isinstance(strategy.hooks[2], ModulePatchHook)
        assert isinstance(strategy.hooks[3], TrainableParameterHook)
        assert strategy.hooks[5] is recorder
        assert calls[0][0] is strategy.models["main"]
        assert calls[0][1].target_modules == ["model.projection"]
        assert recorder.saw_base_fingerprints is True
        assert recorder.saw_aux_projection is True
        assert recorder.saw_optimizer_filter is True

    def test_lora_strategy_rejects_freeze_patterns(
        self,
        baseline_strategy_kwargs: dict[str, Any],
    ) -> None:
        with pytest.raises(ValueError, match="does not accept freeze_patterns"):
            FineTuningStrategy(
                **{
                    **baseline_strategy_kwargs,
                    **_lora_strategy_kwargs(),
                    "freeze_patterns": ("main.model.*",),
                }
            )


class TestLoRAStrategyMerge:
    def test_lora_merge_model_inplace_folds_basic_linear_adapter(
        self,
        baseline_strategy_kwargs: dict[str, Any],
    ) -> None:
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                **_lora_strategy_kwargs(),
            }
        )
        projection = strategy.models["main"].model.projection
        x = torch.randn(4, 8)
        base_output = projection.base_layer(x)

        with torch.no_grad():
            projection.lora_B.fill_(0.05)
        adapted_output = projection(x)

        assert not torch.allclose(adapted_output, base_output)

        merge_lora_into_model(strategy.models["main"])

        merged_projection = strategy.models["main"].model.projection
        assert not is_lora_layer(merged_projection)
        torch.testing.assert_close(merged_projection(x), adapted_output)


class TestLoRAStrategySerialization:
    def test_lora_strategy_to_spec_and_checkpoint_dict_include_adapter_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
        baseline_strategy_kwargs: dict[str, Any],
    ) -> None:
        from test.training.conftest import _build_baseline_strategy_kwargs
        from test.training.test_strategy import dict_demo_training_fn

        _install_fake_peft(monkeypatch)
        model_a = baseline_strategy_kwargs["models"]
        model_b = _build_baseline_strategy_kwargs()["models"]
        strategy = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                "models": {"modelA": model_a, "modelB": model_b},
                "optimizer_configs": {
                    "modelA": [OptimizerConfig(optimizer_cls=torch.optim.Adam)]
                },
                "training_fn": dict_demo_training_fn,
                **_lora_strategy_kwargs(
                    lora_target_patterns=("modelA.model.projection",),
                    compute_base_fingerprints=True,
                ),
            }
        )

        spec = strategy.to_spec_dict()

        assert spec["peft_config"]["peft_method"] == "lora"
        assert "peft_method" not in spec
        assert "peft_config_class" not in spec
        assert spec["peft_config"]["lora_target_patterns"] == [
            "modelA.model.projection"
        ]
        assert spec["freeze_patterns"] == []
        assert spec["freeze_mode"] == "requires_grad"
        assert spec["base_model_fingerprints"] == {
            "modelA": "fingerprint-ok",
            "modelB": "fingerprint-ok",
        }
        assert spec["trainable_parameter_summary"]["lora"] == {
            "tensor_count": 2,
            "parameter_count": 72,
        }
        assert "names" not in spec["trainable_parameter_summary"]["lora"]
        checkpoint = strategy.to_checkpoint_dict()
        assert checkpoint["strategy_cls"].endswith(".FineTuningStrategy")
        assert "runtime_state" in checkpoint
        assert checkpoint["base_model_fingerprints"] == {
            "modelA": "fingerprint-ok",
            "modelB": "fingerprint-ok",
        }
        assert strategy._base_fingerprints == {
            "modelA": "fingerprint-ok",
            "modelB": "fingerprint-ok",
        }

    def test_lora_strategy_from_spec_dict_restores_adapter_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
        baseline_strategy_kwargs: dict[str, Any],
    ) -> None:
        from test.training.conftest import _build_baseline_strategy_kwargs

        _install_fake_peft(monkeypatch)
        source = FineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                **_lora_strategy_kwargs(compute_base_fingerprints=True),
                "module_patches": {
                    "main.model.aux_projection": create_model_spec(
                        nn.Linear,
                        in_features=8,
                        out_features=1,
                    )
                },
                "trainable_patterns": ("main.model.projection.*",),
            }
        )

        restored = FineTuningStrategy.from_spec_dict(
            source.to_spec_dict(),
            models=_build_baseline_strategy_kwargs()["models"],
        )

        assert isinstance(restored.hooks[0], BaseFingerprintHook)
        assert isinstance(restored.hooks[1], LoRAHook)
        assert isinstance(restored.hooks[2], ModulePatchHook)
        assert isinstance(restored.hooks[3], TrainableParameterHook)
        assert isinstance(restored.peft_config, LoRAConfig)
        assert restored.peft_config.rank == 1
        assert restored.peft_config.alpha == 1.0
        assert restored.peft_config.lora_target_patterns == ("main.model.projection",)
        assert set(restored.module_patches) == {"main.model.aux_projection"}
        assert restored.trainable_patterns == ("main.model.projection.*",)
        assert restored._optimizer_parameter_names is not None


class TestLoRAStrategyCheckpointsAndLoad:
    @staticmethod
    def _strategy_kwargs() -> dict[str, Any]:
        """Return LoRA strategy kwargs with patch and extra trainable state."""
        from test.training.conftest import _build_baseline_strategy_kwargs

        return {
            **_build_baseline_strategy_kwargs(),
            **_lora_strategy_kwargs(compute_base_fingerprints=True),
            "module_patches": {
                "main.model.aux_projection": create_model_spec(
                    nn.Linear,
                    in_features=8,
                    out_features=1,
                )
            },
            "trainable_patterns": ("main.model.projection.*",),
        }

    @staticmethod
    def _set_checkpoint_state(
        strategy: FineTuningStrategy,
        *,
        step_count: int = 7,
    ) -> None:
        """Fill every partial checkpoint state category with distinct values."""
        with torch.no_grad():
            strategy.models["main"].model.lora_adapter.weight.fill_(3.0)
            strategy.models["main"].model.lora_adapter.bias.fill_(4.0)
            strategy.models["main"].model.aux_projection.weight.fill_(5.0)
            strategy.models["main"].model.aux_projection.bias.fill_(6.0)
            strategy.models["main"].model.projection.weight.fill_(7.0)
            strategy.models["main"].model.projection.bias.fill_(8.0)
        strategy.step_count = step_count

    @staticmethod
    def _assert_checkpoint_state(
        strategy: FineTuningStrategy,
        *,
        step_count: int = 7,
    ) -> None:
        """Assert LoRA, patch, extra trainable, and runtime state were restored."""
        model = strategy.models["main"].model
        assert strategy.step_count == step_count
        assert hasattr(model, "lora_adapter")
        assert hasattr(model, "aux_projection")
        assert torch.equal(
            model.lora_adapter.weight,
            torch.full_like(model.lora_adapter.weight, 3.0),
        )
        assert torch.equal(
            model.lora_adapter.bias,
            torch.full_like(model.lora_adapter.bias, 4.0),
        )
        assert torch.equal(
            model.aux_projection.weight,
            torch.full_like(model.aux_projection.weight, 5.0),
        )
        assert torch.equal(
            model.aux_projection.bias,
            torch.full_like(model.aux_projection.bias, 6.0),
        )
        assert torch.equal(
            model.projection.weight,
            torch.full_like(model.projection.weight, 7.0),
        )
        assert torch.equal(
            model.projection.bias,
            torch.full_like(model.projection.bias, 8.0),
        )

    @staticmethod
    def _checkpoint_weight_keys(tmp_path: Any) -> list[str]:
        """Return sorted model tensor keys from the partial LoRA checkpoint."""
        weights = torch.load(
            tmp_path / "models" / "main" / "checkpoints" / "0.pt",
            weights_only=True,
        )
        return sorted(weights)

    def test_lora_strategy_save_checkpoint_then_load_checkpoint_restores_partial_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        _install_fake_peft(monkeypatch)
        strategy = FineTuningStrategy(**self._strategy_kwargs())
        self._set_checkpoint_state(strategy)

        with pytest.warns(UserWarning, match=_PARTIAL_CHECKPOINT_HOOK_WARNING):
            strategy.save_checkpoint(tmp_path, save_trainable_state_only=True)

        metadata = json.loads(
            (tmp_path / "strategy" / "checkpoints" / "0.json").read_text()
        )
        restored = FineTuningStrategy.load_checkpoint(tmp_path, map_location="cpu")

        assert self._checkpoint_weight_keys(tmp_path) == [
            "model.aux_projection.bias",
            "model.aux_projection.weight",
            "model.lora_adapter.bias",
            "model.lora_adapter.weight",
            "model.projection.bias",
            "model.projection.weight",
        ]
        assert metadata["model_state_load"] == "partial"
        assert metadata["base_model_fingerprints"] == {"main": "fingerprint-ok"}
        assert isinstance(restored, FineTuningStrategy)
        self._assert_checkpoint_state(restored)

    def test_lora_strategy_save_checkpoint_then_restore_checkpoint_restores_partial_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        _install_fake_peft(monkeypatch)
        source = FineTuningStrategy(**self._strategy_kwargs())
        self._set_checkpoint_state(source)
        with pytest.warns(UserWarning, match=_PARTIAL_CHECKPOINT_HOOK_WARNING):
            source.save_checkpoint(tmp_path, save_trainable_state_only=True)

        restored = FineTuningStrategy(**self._strategy_kwargs())
        with torch.no_grad():
            restored.models["main"].model.lora_adapter.weight.fill_(-3.0)
            restored.models["main"].model.lora_adapter.bias.fill_(-4.0)
            restored.models["main"].model.aux_projection.weight.fill_(-5.0)
            restored.models["main"].model.aux_projection.bias.fill_(-6.0)
            restored.models["main"].model.projection.weight.fill_(-7.0)
            restored.models["main"].model.projection.bias.fill_(-8.0)
        loaded = restored.restore_checkpoint(tmp_path, map_location="cpu")

        assert loaded["strategy"] is restored
        self._assert_checkpoint_state(restored)

    def test_lora_checkpoint_hook_writes_partial_checkpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        dataset: list[Any],
        tmp_path: Any,
    ) -> None:
        _install_fake_peft(monkeypatch)
        hook = CheckpointHook(
            tmp_path,
            step_interval=1,
            async_save=False,
            save_trainable_state_only=True,
        )
        strategy = FineTuningStrategy(
            **{
                **self._strategy_kwargs(),
                "hooks": [hook],
                "num_epochs": None,
                "num_steps": 1,
                "optimizer_configs": OptimizerConfig(
                    optimizer_cls=torch.optim.Adam,
                    optimizer_kwargs={"lr": 0.0},
                ),
            }
        )
        self._set_checkpoint_state(strategy, step_count=0)

        with pytest.warns(UserWarning, match=_PARTIAL_CHECKPOINT_HOOK_WARNING):
            strategy.run([dataset[0]])

        metadata = json.loads(
            (tmp_path / "strategy" / "checkpoints" / "0.json").read_text()
        )

        assert hook.last_checkpoint_index == 0
        assert self._checkpoint_weight_keys(tmp_path) == [
            "model.aux_projection.bias",
            "model.aux_projection.weight",
            "model.lora_adapter.bias",
            "model.lora_adapter.weight",
            "model.projection.bias",
            "model.projection.weight",
        ]
        assert metadata["model_state_load"] == "partial"
        assert metadata["base_model_fingerprints"] == {"main": "fingerprint-ok"}
        assert metadata["runtime_state"]["step_count"] == 1

    def test_lora_from_pretrained_checkpoint_starts_fresh_lora_run(
        self,
        monkeypatch: pytest.MonkeyPatch,
        baseline_strategy_kwargs: dict[str, Any],
        tmp_path: Any,
    ) -> None:
        _install_fake_peft(monkeypatch)
        source = TrainingStrategy(**baseline_strategy_kwargs)
        source.save_checkpoint(tmp_path)
        source_state = {
            name: parameter.detach().clone()
            for name, parameter in source.models["main"].named_parameters()
        }

        strategy = FineTuningStrategy.from_pretrained_checkpoint(
            tmp_path,
            optimizer_configs=OptimizerConfig(optimizer_cls=torch.optim.Adam),
            training_fn=baseline_strategy_kwargs["training_fn"],
            loss_fn=baseline_strategy_kwargs["loss_fn"],
            peft_config=LoRAConfig(
                rank=1,
                alpha=1.0,
                lora_target_patterns=("main.model.projection",),
            ),
            compute_base_fingerprints=True,
            num_steps=1,
        )

        assert isinstance(strategy, FineTuningStrategy)
        assert strategy.step_count == 0
        assert strategy.batch_count == 0
        assert strategy.num_epochs is None
        assert strategy.num_steps == 1
        assert isinstance(strategy.hooks[0], BaseFingerprintHook)
        assert isinstance(strategy.hooks[1], LoRAHook)
        assert isinstance(strategy.hooks[2], TrainableParameterHook)
        assert hasattr(strategy.models["main"].model, "lora_adapter")
        loaded_state = dict(strategy.models["main"].named_parameters())
        for name, parameter in source_state.items():
            assert torch.equal(loaded_state[name], parameter)

    def test_lora_strategy_load_checkpoint_rejects_different_base_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        _install_fake_peft(monkeypatch)
        source = FineTuningStrategy(**self._strategy_kwargs())
        with pytest.warns(UserWarning, match=_PARTIAL_CHECKPOINT_HOOK_WARNING):
            source.save_checkpoint(tmp_path, save_trainable_state_only=True)

        metadata_path = tmp_path / "strategy" / "checkpoints" / "0.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["base_model_fingerprints"] = {"main": "different-base-fingerprint"}
        metadata_path.write_text(json.dumps(metadata))

        monkeypatch.setattr(
            "nvalchemi.training.peft._peft.apply_lora",
            Mock(side_effect=AssertionError("apply_lora should not run")),
        )

        with pytest.raises(ValueError, match="base fingerprint mismatch"):
            FineTuningStrategy.load_checkpoint(tmp_path, map_location="cpu")


class TestLoadPeftCheckpointIntoModel:
    @staticmethod
    def _strategy_kwargs() -> dict[str, Any]:
        """Return generic fine-tuning kwargs with LoRA PEFT config."""
        from test.training.conftest import _build_baseline_strategy_kwargs

        return {
            **_build_baseline_strategy_kwargs(),
            "compute_base_fingerprints": True,
            "peft_config": LoRAConfig(
                rank=1,
                alpha=1.0,
                lora_target_patterns=("main.model.projection",),
            ),
        }

    @staticmethod
    def _strategy_metadata_path(root: Any, index: int = 0) -> Any:
        """Return the saved strategy metadata path for ``index``."""
        return root / "strategy" / "checkpoints" / f"{index}.json"

    def test_loads_lora_from_full_strategy_checkpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        from test.training.conftest import _build_baseline_strategy_kwargs

        _install_fake_peft(monkeypatch)
        source = FineTuningStrategy(**self._strategy_kwargs())
        base_model = _build_baseline_strategy_kwargs()["models"]
        with torch.no_grad():
            source.models["main"].model.lora_adapter.weight.fill_(3.0)
            source.models["main"].model.lora_adapter.bias.fill_(4.0)
            source.models["main"].model.projection.weight.fill_(7.0)
        source.save_checkpoint(tmp_path)

        loaded = load_peft_checkpoint_into_model(
            base_model,
            tmp_path,
            merge=False,
        )

        assert loaded is base_model
        assert torch.equal(
            loaded.model.lora_adapter.weight,
            torch.full_like(loaded.model.lora_adapter.weight, 3.0),
        )
        assert torch.equal(
            loaded.model.lora_adapter.bias,
            torch.full_like(loaded.model.lora_adapter.bias, 4.0),
        )
        assert torch.equal(
            loaded.model.projection.weight,
            torch.full_like(loaded.model.projection.weight, 7.0),
        )

    def test_loads_lora_from_trainable_only_strategy_checkpoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        from test.training.conftest import _build_baseline_strategy_kwargs

        _install_fake_peft(monkeypatch)
        source = FineTuningStrategy(**self._strategy_kwargs())
        with torch.no_grad():
            source.models["main"].model.lora_adapter.weight.fill_(5.0)
            source.models["main"].model.lora_adapter.bias.fill_(6.0)
        with pytest.warns(UserWarning, match=_PARTIAL_CHECKPOINT_HOOK_WARNING):
            source.save_checkpoint(tmp_path, save_trainable_state_only=True)

        base_model = _build_baseline_strategy_kwargs()["models"]
        loaded = load_peft_checkpoint_into_model(
            base_model,
            tmp_path,
            merge=False,
        )

        assert loaded is base_model
        assert torch.equal(
            loaded.model.lora_adapter.weight,
            torch.full_like(loaded.model.lora_adapter.weight, 5.0),
        )
        assert torch.equal(
            loaded.model.lora_adapter.bias,
            torch.full_like(loaded.model.lora_adapter.bias, 6.0),
        )

    def test_rejects_non_peft_checkpoint(
        self,
        baseline_strategy_kwargs: dict[str, Any],
        tmp_path: Any,
    ) -> None:
        source = TrainingStrategy(**baseline_strategy_kwargs)
        source.save_checkpoint(tmp_path)

        with pytest.raises(ValueError, match="peft_config"):
            load_peft_checkpoint_into_model(source.models["main"], tmp_path)

    def test_rejects_unsupported_peft_method(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        _install_fake_peft(monkeypatch)
        source = FineTuningStrategy(**self._strategy_kwargs())
        source.save_checkpoint(tmp_path)
        metadata_path = tmp_path / "strategy" / "checkpoints" / "0.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["peft_config"]["peft_method"] = "ia3"
        metadata_path.write_text(json.dumps(metadata))

        with pytest.raises(ValueError, match="Unsupported PEFT method"):
            load_peft_checkpoint_into_model(source.models["main"], tmp_path)

    def test_fingerprint_mismatch_can_warn_and_continue(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        _install_fake_peft(monkeypatch)
        source = FineTuningStrategy(**self._strategy_kwargs())
        source.save_checkpoint(tmp_path)
        metadata_path = tmp_path / "strategy" / "checkpoints" / "0.json"
        metadata = json.loads(metadata_path.read_text())
        metadata["base_model_fingerprints"] = {"main": "different-fingerprint"}
        metadata_path.write_text(json.dumps(metadata))

        with pytest.warns(UserWarning, match="fingerprint mismatch"):
            loaded = load_peft_checkpoint_into_model(
                source.models["main"],
                tmp_path,
                merge=False,
                strict=False,
            )

        assert loaded is source.models["main"]

    def test_rejects_custom_lora_wrapper_without_allowed_import_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        _install_fake_peft(monkeypatch)
        source = FineTuningStrategy(**self._strategy_kwargs())
        source.save_checkpoint(tmp_path)
        metadata_path = self._strategy_metadata_path(tmp_path)
        metadata = json.loads(metadata_path.read_text())
        metadata["peft_config"]["wrapper_registrations"] = [
            [
                "torch.nn.modules.linear.Linear",
                (
                    f"{_CustomCheckpointLoRAWrapper.__module__}."
                    f"{_CustomCheckpointLoRAWrapper.__qualname__}"
                ),
            ]
        ]
        metadata_path.write_text(json.dumps(metadata))

        with pytest.raises(ValueError, match="LoRA wrapper class import path"):
            load_peft_checkpoint_into_model(
                source.models["main"],
                tmp_path,
                merge=False,
            )

    @pytest.mark.parametrize("allow_namespace", [False, True])
    def test_loads_custom_lora_wrapper_with_allowed_import_path_or_namespace(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        allow_namespace: bool,
    ) -> None:
        _install_fake_peft(monkeypatch)
        registrations: list[tuple[type[nn.Module], type[nn.Module]]] = []
        monkeypatch.setattr(
            "nvalchemi.training.peft._peft.register_lora_wrapper",
            lambda layer_cls, wrapper_cls: registrations.append(
                (layer_cls, wrapper_cls)
            ),
        )
        source = FineTuningStrategy(**self._strategy_kwargs())
        source.save_checkpoint(tmp_path)
        metadata_path = self._strategy_metadata_path(tmp_path)
        metadata = json.loads(metadata_path.read_text())
        wrapper_path = (
            f"{_CustomCheckpointLoRAWrapper.__module__}."
            f"{_CustomCheckpointLoRAWrapper.__qualname__}"
        )
        metadata["peft_config"]["wrapper_registrations"] = [
            ["torch.nn.modules.linear.Linear", wrapper_path]
        ]
        metadata_path.write_text(json.dumps(metadata))
        if allow_namespace:
            wrapper_path = f"{_CustomCheckpointLoRAWrapper.__module__}.*"

        loaded = load_peft_checkpoint_into_model(
            source.models["main"],
            tmp_path,
            allowed_import_paths={wrapper_path},
            merge=False,
        )

        assert loaded is source.models["main"]
        assert registrations == [(nn.Linear, _CustomCheckpointLoRAWrapper)]

    def test_rejects_custom_module_patch_without_allowed_import_path(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        from test.training.conftest import _build_baseline_strategy_kwargs

        _install_fake_peft(monkeypatch)
        source = FineTuningStrategy(
            **{
                **self._strategy_kwargs(),
                "module_patches": {
                    "main.model.aux_projection": create_model_spec(
                        _CustomCheckpointPatch,
                        in_features=8,
                        out_features=1,
                    )
                },
            }
        )
        source.save_checkpoint(tmp_path)
        base_model = _build_baseline_strategy_kwargs()["models"]

        with pytest.raises(ValueError, match="module patch"):
            load_peft_checkpoint_into_model(
                base_model,
                tmp_path,
                merge=False,
            )

    @pytest.mark.parametrize("allow_namespace", [False, True])
    def test_loads_custom_module_patch_with_allowed_import_path_or_namespace(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        allow_namespace: bool,
    ) -> None:
        from test.training.conftest import _build_baseline_strategy_kwargs

        _install_fake_peft(monkeypatch)
        source = FineTuningStrategy(
            **{
                **self._strategy_kwargs(),
                "module_patches": {
                    "main.model.aux_projection": create_model_spec(
                        _CustomCheckpointPatch,
                        in_features=8,
                        out_features=1,
                    )
                },
            }
        )
        source.save_checkpoint(tmp_path)
        patch_path = (
            f"{_CustomCheckpointPatch.__module__}.{_CustomCheckpointPatch.__qualname__}"
        )
        base_model = _build_baseline_strategy_kwargs()["models"]
        if allow_namespace:
            patch_path = f"{_CustomCheckpointPatch.__module__}.*"

        loaded = load_peft_checkpoint_into_model(
            base_model,
            tmp_path,
            allowed_import_paths={patch_path},
            merge=False,
        )

        assert loaded is base_model
        assert isinstance(loaded.model.aux_projection, _CustomCheckpointPatch)

    def test_loads_torch_linear_module_patch_by_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        from test.training.conftest import _build_baseline_strategy_kwargs

        _install_fake_peft(monkeypatch)
        source = FineTuningStrategy(
            **{
                **self._strategy_kwargs(),
                "module_patches": {
                    "main.model.aux_projection": create_model_spec(
                        nn.Linear,
                        in_features=8,
                        out_features=1,
                    )
                },
            }
        )
        source.save_checkpoint(tmp_path)
        base_model = _build_baseline_strategy_kwargs()["models"]

        loaded = load_peft_checkpoint_into_model(
            base_model,
            tmp_path,
            merge=False,
        )

        assert loaded is base_model
        assert isinstance(loaded.model.aux_projection, nn.Linear)

    def test_loads_custom_imports_with_trust_remote_code(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
    ) -> None:
        from test.training.conftest import _build_baseline_strategy_kwargs

        _install_fake_peft(monkeypatch)
        monkeypatch.setattr(
            "nvalchemi.training.peft._peft.register_lora_wrapper",
            lambda layer_cls, wrapper_cls: None,
        )
        source = FineTuningStrategy(
            **{
                **self._strategy_kwargs(),
                "module_patches": {
                    "main.model.aux_projection": create_model_spec(
                        _CustomCheckpointPatch,
                        in_features=8,
                        out_features=1,
                    )
                },
            }
        )
        source.save_checkpoint(tmp_path)
        metadata_path = self._strategy_metadata_path(tmp_path)
        metadata = json.loads(metadata_path.read_text())
        metadata["peft_config"]["wrapper_registrations"] = [
            [
                "torch.nn.modules.linear.Linear",
                (
                    f"{_CustomCheckpointLoRAWrapper.__module__}."
                    f"{_CustomCheckpointLoRAWrapper.__qualname__}"
                ),
            ]
        ]
        metadata_path.write_text(json.dumps(metadata))
        base_model = _build_baseline_strategy_kwargs()["models"]

        loaded = load_peft_checkpoint_into_model(
            base_model,
            tmp_path,
            merge=False,
            trust_remote_code=True,
        )

        assert loaded is base_model
        assert isinstance(loaded.model.aux_projection, _CustomCheckpointPatch)


class TestLoRAWrapperRegistrations:
    def test_available_lora_wrappers_warns_for_unavailable_optional_dependency(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def raise_missing_e3nn() -> tuple[
            type[nn.Module], type[nn.Module], object, type[nn.Module]
        ]:
            raise ImportError("Equivariant LoRA requires e3nn.")

        monkeypatch.setattr(lora_wrappers, "_import_e3nn", raise_missing_e3nn)

        with pytest.warns(UserWarning, match="Skipping built-in LoRA wrapper"):
            lora_wrappers.available_lora_wrappers()

    def test_config_rejects_conflicting_wrappers_for_one_layer(self) -> None:
        with pytest.raises(ValueError, match="Multiple LoRA wrappers configured"):
            LoRAConfig(
                lora_target_patterns=("main.model.projection",),
                wrapper_registrations=(
                    (nn.Linear, _CustomLoRAWrapper),
                    (nn.Linear, _AlternateLoRAWrapper),
                ),
            )

    def test_temporary_registration_overrides_and_restores_wrapper(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            lora_wrappers,
            "_BUILTIN_LORA_WRAPPER_FACTORIES",
            (),
        )
        registry = lora_wrappers._peft._LORA_WRAPPERS
        monkeypatch.setitem(registry, nn.Linear, lora_wrappers.LoRALinear)
        previous = dict(registry)

        with pytest.warns(
            UserWarning,
            match="Temporarily overriding LoRA wrapper",
        ):
            with lora_wrappers._temporary_lora_wrapper_registrations(
                ((nn.Linear, _CustomLoRAWrapper),)
            ):
                assert registry[nn.Linear] is _CustomLoRAWrapper

        assert registry == previous


class TestE3NNFullyConnectedLoRALayer:
    def test_e3nn_fully_connected_lora_creates_trainable_adapter_parameters(
        self,
    ) -> None:
        fc_layer_cls = _import_real_e3nn_fc_layer()
        base_layer = fc_layer_cls(3, 4, None, 1, 1)
        wrapper = E3NNFullyConnectedLoRALayer(
            base_layer,
            rank=2,
            alpha=4.0,
        )

        assert wrapper.lora_A.shape == (3, 2)
        assert wrapper.lora_B.shape == (2, 4)
        assert wrapper.lora_A.requires_grad is True
        assert wrapper.lora_B.requires_grad is True
        assert base_layer.weight.requires_grad is False
        assert torch.count_nonzero(wrapper.lora_A) > 0
        assert torch.count_nonzero(wrapper.lora_B) == 0

    def test_e3nn_fully_connected_lora_preserves_initial_inference(
        self,
    ) -> None:
        fc_layer_cls = _import_real_e3nn_fc_layer()
        torch.manual_seed(1)
        base_layer = fc_layer_cls(3, 4, None, 1, 1)
        x = torch.randn(5, 3)
        expected = base_layer(x)

        wrapper = E3NNFullyConnectedLoRALayer(
            base_layer,
            rank=2,
            alpha=4.0,
        )

        torch.testing.assert_close(wrapper(x), expected)

    def test_e3nn_fully_connected_lora_adapter_weights_affect_inference(
        self,
    ) -> None:
        fc_layer_cls = _import_real_e3nn_fc_layer()
        torch.manual_seed(11)
        base_layer = fc_layer_cls(3, 4, None, 1, 1)
        x = torch.randn(5, 3)
        wrapper = E3NNFullyConnectedLoRALayer(
            base_layer,
            rank=2,
            alpha=4.0,
        )
        initial = wrapper(x)

        with torch.no_grad():
            wrapper.lora_B.fill_(0.05)
        after_lora_B = wrapper(x)

        assert not torch.allclose(after_lora_B, initial)

        with torch.no_grad():
            wrapper.lora_A.add_(0.05)

        assert not torch.allclose(wrapper(x), after_lora_B)

    def test_e3nn_fully_connected_lora_rejects_dropout(self) -> None:
        fc_layer_cls = _import_real_e3nn_fc_layer()
        base_layer = fc_layer_cls(3, 4, None, 1, 1)

        with pytest.raises(ValueError, match="does not support nonzero dropout"):
            E3NNFullyConnectedLoRALayer(
                base_layer,
                rank=2,
                alpha=4.0,
                dropout=0.1,
            )

    @pytest.mark.parametrize("activation", [None, torch.relu])
    def test_e3nn_fully_connected_lora_merge_preserves_adapter_inference(
        self,
        activation: Any,
    ) -> None:
        fc_layer_cls = _import_real_e3nn_fc_layer()
        torch.manual_seed(2)
        base_layer = fc_layer_cls(3, 4, activation, 1, 1)
        wrapper = E3NNFullyConnectedLoRALayer(
            base_layer,
            rank=2,
            alpha=4.0,
        )
        with torch.no_grad():
            wrapper.lora_B.normal_(mean=0.0, std=1e-3)
        x = torch.randn(5, 3)
        expected = wrapper(x)

        wrapper.merge_into_base()

        torch.testing.assert_close(base_layer(x), expected)


class TestEquivariantLoRALinear:
    def test_o3_linear_lora_creates_trainable_adapter_parameters(self) -> None:
        o3 = _import_real_o3()
        base_layer = o3.Linear(
            "2x0e + 1x1o",
            "1x0e + 2x1o",
            internal_weights=True,
            shared_weights=True,
            biases=False,
        )

        wrapper = EquivariantLoRALinear(base_layer, rank=3, alpha=6.0)

        assert wrapper.adapter_irreps == o3.Irreps("3x0e + 3x1o")
        assert wrapper.lora_A.requires_grad is True
        assert wrapper.lora_B.requires_grad is True
        assert base_layer.weight.requires_grad is False
        assert torch.count_nonzero(wrapper.lora_A) > 0
        assert torch.count_nonzero(wrapper.lora_B) == 0

    def test_o3_linear_lora_honors_custom_init(self) -> None:
        o3 = _import_real_o3()
        base_layer = o3.Linear(
            "2x0e + 1x1o",
            "1x0e + 2x1o",
            internal_weights=True,
            shared_weights=True,
            biases=False,
        )

        def fill_ones(tensor: torch.Tensor) -> None:
            tensor.fill_(1.0)

        wrapper = EquivariantLoRALinear(base_layer, rank=2, alpha=4.0, init=fill_ones)

        assert torch.equal(wrapper.lora_A, torch.ones_like(wrapper.lora_A))
        assert torch.count_nonzero(wrapper.lora_B) == 0

    def test_o3_linear_lora_preserves_initial_inference(self) -> None:
        o3 = _import_real_o3()
        torch.manual_seed(1)
        base_layer = o3.Linear(
            "2x0e + 1x1o",
            "1x0e + 2x1o",
            internal_weights=True,
            shared_weights=True,
            biases=False,
        ).to(dtype=torch.float64)
        x = torch.randn(5, base_layer.irreps_in.dim, dtype=torch.float64)
        expected = base_layer(x)

        wrapper = EquivariantLoRALinear(base_layer, rank=2, alpha=4.0)

        torch.testing.assert_close(wrapper(x), expected)

    def test_o3_linear_lora_adapter_weights_affect_inference(self) -> None:
        o3 = _import_real_o3()
        torch.manual_seed(12)
        base_layer = o3.Linear(
            "2x0e + 1x1o",
            "1x0e + 2x1o",
            internal_weights=True,
            shared_weights=True,
            biases=False,
        ).to(dtype=torch.float64)
        x = torch.randn(5, base_layer.irreps_in.dim, dtype=torch.float64)
        wrapper = EquivariantLoRALinear(base_layer, rank=2, alpha=4.0)
        initial = wrapper(x)

        with torch.no_grad():
            wrapper.lora_B.fill_(0.05)
        after_lora_B = wrapper(x)

        assert not torch.allclose(after_lora_B, initial)

        with torch.no_grad():
            wrapper.lora_A.add_(0.05)

        assert not torch.allclose(wrapper(x), after_lora_B)

    def test_o3_linear_lora_merge_preserves_adapter_inference(self) -> None:
        o3 = _import_real_o3()
        torch.manual_seed(2)
        base_layer = o3.Linear(
            "2x0e + 1x1o",
            "1x0e + 2x1o",
            internal_weights=True,
            shared_weights=True,
            biases=False,
        ).to(dtype=torch.float64)
        wrapper = EquivariantLoRALinear(base_layer, rank=2, alpha=4.0)
        with torch.no_grad():
            wrapper.lora_B.normal_(mean=0.0, std=1e-3)
        x = torch.randn(5, wrapper.irreps_in.dim, dtype=torch.float64)
        expected = wrapper(x)

        wrapper.merge_into_base()

        torch.testing.assert_close(base_layer(x), expected)

    def test_o3_linear_lora_remains_equivariant(self) -> None:
        o3 = _import_real_o3()
        assert_equivariant = _import_e3nn_assert_equivariant()
        torch.manual_seed(2)
        base_layer = o3.Linear(
            "2x0e + 1x1o",
            "1x0e + 2x1o",
            internal_weights=True,
            shared_weights=True,
            biases=False,
        ).to(dtype=torch.float64)
        wrapper = EquivariantLoRALinear(base_layer, rank=2, alpha=4.0)
        with torch.no_grad():
            wrapper.lora_B.normal_(mean=0.0, std=1e-3)

        x = torch.randn(5, wrapper.irreps_in.dim, dtype=torch.float64)

        assert_equivariant(
            wrapper,
            args_in=[x],
            irreps_in=[wrapper.irreps_in],
            irreps_out=[wrapper.irreps_out],
            tolerance=1e-6,
            ntrials=3,
        )


class TestCuEquivarianceLoRALinear:
    @staticmethod
    def _base_layer() -> nn.Module:
        return _make_cueq_linear(
            _cueq_irreps("2x0e + 1x1o"),
            _cueq_irreps("1x0e + 2x1o"),
        )

    def test_cueq_linear_lora_creates_trainable_adapter_parameters(
        self,
    ) -> None:
        cue, _linear_cls = _import_real_cueq_linear()
        base_layer = self._base_layer()

        wrapper = CuEquivarianceLoRALinear(base_layer, rank=3, alpha=6.0)

        assert wrapper.adapter_irreps == cue.Irreps("O3", "3x0e + 3x1o")
        assert wrapper.lora_A.shape == (1, 9)
        assert wrapper.lora_B.shape == (1, 9)
        assert wrapper.lora_A.requires_grad is True
        assert wrapper.lora_B.requires_grad is True
        assert base_layer.weight.requires_grad is False
        assert torch.count_nonzero(wrapper.lora_A) > 0
        assert torch.count_nonzero(wrapper.lora_B) == 0

    def test_cueq_linear_lora_honors_custom_init(self) -> None:
        base_layer = self._base_layer()

        def fill_ones(tensor: torch.Tensor) -> None:
            tensor.fill_(1.0)

        wrapper = CuEquivarianceLoRALinear(
            base_layer, rank=2, alpha=4.0, init=fill_ones
        )

        assert torch.equal(wrapper.lora_A, torch.ones_like(wrapper.lora_A))
        assert torch.count_nonzero(wrapper.lora_B) == 0

    def test_cueq_linear_lora_adapter_weights_affect_inference(
        self,
    ) -> None:
        if not torch.cuda.is_available():
            pytest.skip("cuEquivariance forward kernels require CUDA.")
        torch.manual_seed(12)
        base_layer = self._base_layer()
        x = torch.randn(
            5,
            base_layer.irreps_in.dim,
            device=base_layer.weight.device,
            dtype=torch.float64,
        )
        expected = base_layer(x)

        wrapper = CuEquivarianceLoRALinear(base_layer, rank=2, alpha=4.0)
        torch.testing.assert_close(wrapper(x), expected)

        with torch.no_grad():
            wrapper.lora_B.fill_(0.05)
        after_lora_B = wrapper(x)

        assert not torch.allclose(after_lora_B, expected)

        with torch.no_grad():
            wrapper.lora_A.add_(0.05)

        assert not torch.allclose(wrapper(x), after_lora_B)

    def test_cueq_linear_lora_merge_preserves_adapter_inference(
        self,
    ) -> None:
        if not torch.cuda.is_available():
            pytest.skip("cuEquivariance forward kernels require CUDA.")
        torch.manual_seed(2)
        base_layer = self._base_layer()
        wrapper = CuEquivarianceLoRALinear(base_layer, rank=2, alpha=4.0)
        with torch.no_grad():
            wrapper.lora_B.normal_(mean=0.0, std=1e-3)
        x = torch.randn(
            5,
            base_layer.irreps_in.dim,
            device=base_layer.weight.device,
            dtype=torch.float64,
        )
        expected = wrapper(x)

        wrapper.merge_into_base()

        torch.testing.assert_close(base_layer(x), expected)

    @pytest.mark.parametrize(
        ("irreps_in", "irreps_out", "kwargs", "expected"),
        [
            ("1x0e", "1x0e", {}, True),
            ("1x0e", "1x0e", {"weight_classes": 2}, False),
            ("1x0e", "1x0e", {"internal_weights": False}, False),
        ],
    )
    def test_cueq_linear_lora_is_compatible(
        self,
        irreps_in: str,
        irreps_out: str,
        kwargs: dict[str, Any],
        expected: bool,
    ) -> None:
        base_layer = _make_cueq_linear(
            _cueq_irreps(irreps_in),
            _cueq_irreps(irreps_out),
            **kwargs,
        )

        assert CuEquivarianceLoRALinear.is_compatible(base_layer) is expected

    def test_cueq_linear_lora_rejects_no_shared_irreps(self) -> None:
        with pytest.raises(ValueError, match="share no common irreps"):
            CuEquivarianceLoRALinear._build_adapter_irreps(
                _cueq_irreps("1x0e"),
                _cueq_irreps("1x1o"),
                rank=1,
            )

    def test_cueq_linear_lora_rejects_dropout(
        self,
    ) -> None:
        with pytest.raises(ValueError, match="does not support nonzero dropout"):
            CuEquivarianceLoRALinear(
                self._base_layer(),
                rank=2,
                alpha=4.0,
                dropout=0.1,
            )
