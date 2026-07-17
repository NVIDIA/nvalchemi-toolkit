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
"""Tests for LoRA PEFT API scaffolding."""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest
import torch
from torch import nn

from nvalchemi.training import (
    EnergyMSELoss,
    LoRAConfig,
    LoRAFineTuningStrategy,
    OptimizerConfig,
    create_model_spec,
)
from nvalchemi.training.hooks import ModulePatchHook
from nvalchemi.training.peft import _peft
from nvalchemi.training.peft import wrappers as lora_wrappers
from nvalchemi.training.peft.hooks import (
    LoRAApplyHook,
    LoRACheckpointHook,
    LoRATrainableParameterHook,
)
from nvalchemi.training.peft.wrappers import E3NNFullyConnectedLoRALayer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lora_config(**kwargs: Any) -> LoRAConfig:
    """Return a minimal real LoRA config for strategy tests."""
    defaults: dict[str, Any] = {
        "rank": 1,
        "alpha": 1.0,
        "target_modules": ["model.lora_adapter"],
    }
    defaults.update(kwargs)
    return LoRAConfig(**defaults)


class _FakeLoRAResult:
    """Small stand-in for PhysicsNeMo apply_lora result metadata."""

    base_fingerprint = "fingerprint-ok"
    n_wrapped = 1
    n_trainable = 2
    n_frozen = 1
    trainable_names = ["model.lora_adapter.weight", "model.lora_adapter.bias"]


def _stash_fake_lora_config(model: nn.Module) -> None:
    """Stash adapter metadata like ``apply_lora`` does."""
    model._lora_config = {
        "rank": 1,
        "alpha": 1.0,
        "lora_dropout": 0.0,
        "extras_trainable": [],
        "init": "default",
    }


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
        self.saw_lora_metadata = False
        self.saw_aux_projection = False
        self.saw_optimizer_filter = False

    def _runs_on_stage(self, stage: Any) -> bool:  # noqa: ARG002
        return False

    def on_register(self, workflow: Any) -> None:
        self.saw_lora_metadata = hasattr(workflow, "_lora_metadata")
        self.saw_aux_projection = hasattr(
            workflow.models["main"].model, "aux_projection"
        )
        self.saw_optimizer_filter = workflow._optimizer_parameter_names is not None

    def __call__(self, ctx: Any, stage: Any) -> None:  # noqa: ARG002
        return


class _DummyE3NNFCLayer(nn.Module):
    """Small e3nn fully connected stand-in with the same weight layout."""

    def __init__(self, shape: tuple[int, ...] = (3, 4)) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight


class _FakeDDP(nn.Module):
    """Small DDP stand-in exposing an underlying ``module``."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.module(*args, **kwargs)


def _fake_import_e3nn() -> tuple[
    type[nn.Module], type[nn.Module], object, type[nn.Module]
]:
    """Return fake e3nn classes for LoRA wrapper tests."""
    return nn.Linear, _DummyE3NNFCLayer, object(), nn.Dropout


def _install_fake_peft(
    monkeypatch: pytest.MonkeyPatch,
    *,
    current_fingerprint: str = "fingerprint-ok",
    leave_lora_after_merge: bool = False,
) -> None:
    """Install a deterministic fake PhysicsNeMo PEFT surface."""

    class _ConfiguredLoRAResult(_FakeLoRAResult):
        base_fingerprint = current_fingerprint

    def fake_apply_lora(model: nn.Module, config: Any) -> _FakeLoRAResult:  # noqa: ARG001
        layer = nn.Linear(8, 8)
        layer._fake_lora = True
        model.model.lora_adapter = layer
        _stash_fake_lora_config(model)
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
        "nvalchemi.training.peft.wrappers.register_builtin_lora_wrappers",
        lambda: None,
    )
    monkeypatch.setattr(
        "nvalchemi.training.peft.lora.register_builtin_lora_wrappers",
        lambda: None,
    )


# ---------------------------------------------------------------------------
# Hook registration, wrapper registration, and trainable parameter selection
# ---------------------------------------------------------------------------


def test_lora_strategy_prepends_generated_hooks(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
) -> None:
    calls: list[tuple[nn.Module, Any]] = []

    def fake_apply_lora(model: nn.Module, config: Any) -> _FakeLoRAResult:
        calls.append((model, config))
        model.model.lora_adapter = _fake_lora_linear()
        _stash_fake_lora_config(model)
        return _FakeLoRAResult()

    monkeypatch.setattr("nvalchemi.training.peft._peft.apply_lora", fake_apply_lora)
    monkeypatch.setattr(
        "nvalchemi.training.peft._peft.is_lora_layer",
        lambda module: bool(getattr(module, "_fake_lora", False)),
    )
    monkeypatch.setattr(
        "nvalchemi.training.peft.wrappers.register_builtin_lora_wrappers",
        lambda: None,
    )
    recorder = _Recorder()
    config = _lora_config()

    strategy = LoRAFineTuningStrategy(
        **{
            **baseline_strategy_kwargs,
            "lora_config": config,
            "module_patches": {"main.model.aux_projection": nn.Linear(8, 1)},
            "trainable_patterns": ("main.model.projection.*",),
            "hooks": [recorder],
        }
    )

    assert isinstance(strategy.hooks[0], LoRAApplyHook)
    assert isinstance(strategy.hooks[1], ModulePatchHook)
    assert isinstance(strategy.hooks[2], LoRATrainableParameterHook)
    assert strategy.hooks[3] is recorder
    assert calls == [(strategy.models["main"], config)]
    assert recorder.saw_lora_metadata is True
    assert recorder.saw_aux_projection is True
    assert recorder.saw_optimizer_filter is True


def test_register_builtin_lora_wrappers_warns_for_unavailable_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[type[Any], type[Any]]] = []

    def raise_missing_e3nn() -> tuple[
        type[nn.Module], type[nn.Module], object, type[nn.Module]
    ]:
        raise ImportError("Equivariant LoRA requires e3nn.")

    monkeypatch.setattr(lora_wrappers, "_BUILTIN_LORA_WRAPPERS_REGISTERED", False)
    monkeypatch.setattr(lora_wrappers, "_import_e3nn", raise_missing_e3nn)
    monkeypatch.setattr(
        lora_wrappers._peft,
        "register_lora_wrapper",
        lambda layer_cls, wrapper_cls: calls.append((layer_cls, wrapper_cls)),
    )

    with pytest.warns(UserWarning, match="Skipping built-in LoRA wrapper"):
        lora_wrappers.register_builtin_lora_wrappers()

    assert calls == []


def test_lora_strategy_applies_real_physicsnemo_peft(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
) -> None:
    monkeypatch.setattr(lora_wrappers, "register_builtin_lora_wrappers", lambda: None)

    strategy = LoRAFineTuningStrategy(
        **{
            **baseline_strategy_kwargs,
            "lora_config": _lora_config(target_modules=["model.projection"]),
        }
    )

    projection = strategy.models["main"].model.projection
    metadata = strategy._lora_metadata["main"]

    assert _peft.is_lora_layer(projection)
    assert metadata["lora_modules"] == ["model.projection"]
    assert metadata["trainable_names"] == [
        "model.projection.lora_A",
        "model.projection.lora_B",
    ]
    assert "main.model.projection.lora_A" in strategy._optimizer_parameter_names
    assert "main.model.projection.lora_B" in strategy._optimizer_parameter_names
    assert (
        "main.model.projection.base_layer.weight"
        not in strategy._optimizer_parameter_names
    )


def test_lora_trainable_filter_allows_adapters_patches_and_extras(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    batch: Any,
) -> None:
    def fake_apply_lora(model: nn.Module, config: Any) -> _FakeLoRAResult:  # noqa: ARG001
        model.model.lora_adapter = _fake_lora_linear()
        _stash_fake_lora_config(model)
        return _FakeLoRAResult()

    monkeypatch.setattr("nvalchemi.training.peft._peft.apply_lora", fake_apply_lora)
    monkeypatch.setattr(
        "nvalchemi.training.peft._peft.is_lora_layer",
        lambda module: bool(getattr(module, "_fake_lora", False)),
    )
    monkeypatch.setattr(
        "nvalchemi.training.peft.wrappers.register_builtin_lora_wrappers",
        lambda: None,
    )
    strategy = LoRAFineTuningStrategy(
        **{
            **baseline_strategy_kwargs,
            "loss_fn": EnergyMSELoss(),
            "lora_config": _lora_config(),
            "module_patches": {"main.model.aux_projection": nn.Linear(8, 1)},
            "trainable_patterns": ("main.model.projection.*",),
        }
    )

    strategy.train_batch(batch)

    optimizer_param_ids = {
        id(param)
        for optimizer in strategy._optimizers
        for group in optimizer.param_groups
        for param in group["params"]
    }
    for name, param in strategy.models["main"].named_parameters():
        qualified = f"main.{name}"
        if (
            "lora_adapter" in qualified
            or qualified.startswith("main.model.aux_projection.")
            or qualified.startswith("main.model.projection.")
        ):
            assert id(param) in optimizer_param_ids
        elif qualified.startswith("main.model.joint_mlp."):
            assert id(param) not in optimizer_param_ids


def test_lora_trainable_filter_rejects_stale_metadata(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
) -> None:
    class _StaleLoRAResult(_FakeLoRAResult):
        trainable_names = ["model.missing_adapter.weight"]

    def fake_apply_lora(model: nn.Module, config: Any) -> _StaleLoRAResult:  # noqa: ARG001
        model.model.lora_adapter = _fake_lora_linear()
        _stash_fake_lora_config(model)
        return _StaleLoRAResult()

    monkeypatch.setattr("nvalchemi.training.peft._peft.apply_lora", fake_apply_lora)
    monkeypatch.setattr(
        "nvalchemi.training.peft._peft.is_lora_layer",
        lambda module: bool(getattr(module, "_fake_lora", False)),
    )
    monkeypatch.setattr(
        "nvalchemi.training.peft.wrappers.register_builtin_lora_wrappers",
        lambda: None,
    )

    with pytest.raises(RuntimeError, match="not present"):
        LoRAFineTuningStrategy(
            **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
        )


def test_lora_trainable_filter_excludes_wrapped_base_from_extras(
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

    def fake_apply_lora(model: nn.Module, config: Any) -> _ProjectionLoRAResult:  # noqa: ARG001
        model.model.projection = _FakeLoRAWrapper(model.model.projection)
        _stash_fake_lora_config(model)
        return _ProjectionLoRAResult()

    monkeypatch.setattr("nvalchemi.training.peft._peft.apply_lora", fake_apply_lora)
    monkeypatch.setattr(
        "nvalchemi.training.peft._peft.is_lora_layer",
        lambda module: bool(getattr(module, "_fake_lora", False)),
    )
    monkeypatch.setattr(
        "nvalchemi.training.peft.wrappers.register_builtin_lora_wrappers",
        lambda: None,
    )

    strategy = LoRAFineTuningStrategy(
        **{
            **baseline_strategy_kwargs,
            "lora_config": _lora_config(),
            "trainable_patterns": ("main.model.projection.*",),
        }
    )

    assert strategy._optimizer_parameter_names is not None
    assert (
        "main.model.projection.lora_down.weight" in strategy._optimizer_parameter_names
    )
    assert "main.model.projection.lora_up.weight" in strategy._optimizer_parameter_names
    assert (
        "main.model.projection.base_layer.weight"
        not in strategy._optimizer_parameter_names
    )
    assert (
        "main.model.projection.base_layer.bias"
        not in strategy._optimizer_parameter_names
    )
    assert "main.model.projection.lora_down.weight" in getattr(
        strategy, "_lora_adapter_parameter_names"
    )
    assert "main.model.projection.base_layer.weight" not in getattr(
        strategy, "_lora_adapter_parameter_names"
    )


def test_lora_strategy_rejects_freeze_patterns(
    baseline_strategy_kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="does not accept freeze_patterns"):
        LoRAFineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                "lora_config": _lora_config(),
                "freeze_patterns": ("main.model.*",),
            }
        )


def test_lora_strategy_rejects_physicsnemo_extras_trainable(
    baseline_strategy_kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="extras_trainable"):
        LoRAFineTuningStrategy(
            **{
                **baseline_strategy_kwargs,
                "lora_config": _lora_config(extras_trainable=["model.readout"]),
            }
        )


# ---------------------------------------------------------------------------
# Strategy serialization and checkpoint behavior
# ---------------------------------------------------------------------------


def test_lora_strategy_serializes_adapter_and_checkpoint_metadata(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
) -> None:
    _install_fake_peft(monkeypatch)
    strategy = LoRAFineTuningStrategy(
        **{
            **baseline_strategy_kwargs,
            "lora_config": _lora_config(),
        }
    )

    spec = strategy.to_spec_dict()

    assert "lora_config" in spec
    assert spec["lora_config"]["target_modules"] == ["model.lora_adapter"]
    assert "lora_modules" not in spec["lora_config"]
    assert "freeze_patterns" not in spec
    assert "freeze_mode" not in spec
    checkpoint = strategy.to_checkpoint_dict()
    assert checkpoint["strategy_cls"].endswith(".LoRAFineTuningStrategy")
    assert "runtime_state" in checkpoint
    assert checkpoint["base_model_fingerprints"] == {"main": "fingerprint-ok"}


def test_lora_strategy_checkpoint_roundtrip_injects_lora_before_weights(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    _install_fake_peft(monkeypatch)
    strategy = LoRAFineTuningStrategy(
        **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
    )
    with torch.no_grad():
        strategy.models["main"].model.lora_adapter.weight.fill_(3.0)
        strategy.models["main"].model.lora_adapter.bias.fill_(4.0)

    strategy.step_count = 7
    strategy.save_checkpoint(tmp_path)

    weights = torch.load(
        tmp_path / "models" / "main" / "checkpoints" / "0.pt",
        weights_only=True,
    )
    metadata = json.loads(
        (tmp_path / "strategy" / "checkpoints" / "0.json").read_text()
    )
    restored = LoRAFineTuningStrategy.load_checkpoint(tmp_path, map_location="cpu")

    assert sorted(weights) == [
        "model.lora_adapter.bias",
        "model.lora_adapter.weight",
    ]
    assert metadata["model_state_load"] == "partial"
    assert metadata["base_model_fingerprints"] == {"main": "fingerprint-ok"}
    assert isinstance(restored, LoRAFineTuningStrategy)
    assert restored.step_count == 7
    assert hasattr(restored.models["main"].model, "lora_adapter")
    assert torch.equal(
        restored.models["main"].model.lora_adapter.weight,
        torch.full_like(restored.models["main"].model.lora_adapter.weight, 3.0),
    )
    assert torch.equal(
        restored.models["main"].model.lora_adapter.bias,
        torch.full_like(restored.models["main"].model.lora_adapter.bias, 4.0),
    )


@pytest.mark.parametrize("checkpoint_entry_point", ["strategy", "hook"])
def test_lora_checkpoint_can_include_base_parameters(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    dataset: list[Any],
    tmp_path: Any,
    checkpoint_entry_point: str,
) -> None:
    _install_fake_peft(monkeypatch)
    hooks = []
    if checkpoint_entry_point == "hook":
        hooks.append(
            LoRACheckpointHook(
                tmp_path,
                step_interval=1,
                async_save=False,
                include_base_parameters=True,
            )
        )
    strategy = LoRAFineTuningStrategy(
        **{
            **baseline_strategy_kwargs,
            "lora_config": _lora_config(),
            "hooks": hooks,
            "num_epochs": None,
            "num_steps": 1,
        }
    )

    if checkpoint_entry_point == "strategy":
        strategy.save_checkpoint(tmp_path, include_base_parameters=True)
    else:
        strategy.run([dataset[0]])

    weights = torch.load(
        tmp_path / "models" / "main" / "checkpoints" / "0.pt",
        weights_only=True,
    )
    metadata = json.loads(
        (tmp_path / "strategy" / "checkpoints" / "0.json").read_text()
    )
    assert "model.projection.weight" in weights
    assert "model.lora_adapter.weight" in weights
    assert "model_state_load" not in metadata


def test_lora_checkpoint_hook_saves_trainable_state_by_default(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    dataset: list[Any],
    tmp_path: Any,
) -> None:
    _install_fake_peft(monkeypatch)
    hook = LoRACheckpointHook(tmp_path, step_interval=1, async_save=False)
    strategy = LoRAFineTuningStrategy(
        **{
            **baseline_strategy_kwargs,
            "lora_config": _lora_config(),
            "hooks": [hook],
            "num_epochs": None,
            "num_steps": 1,
        }
    )

    strategy.run([dataset[0]])

    weights = torch.load(
        tmp_path / "models" / "main" / "checkpoints" / "0.pt",
        weights_only=True,
    )
    metadata = json.loads(
        (tmp_path / "strategy" / "checkpoints" / "0.json").read_text()
    )
    assert hook.last_checkpoint_index == 0
    assert sorted(weights) == [
        "model.lora_adapter.bias",
        "model.lora_adapter.weight",
    ]
    assert metadata["model_state_load"] == "partial"
    assert metadata["model_state_kind"] == "lora_trainable"
    assert metadata["base_model_fingerprints"] == {"main": "fingerprint-ok"}

    restored = LoRAFineTuningStrategy.load_checkpoint(tmp_path, map_location="cpu")
    assert restored.step_count == 1
    assert hasattr(restored.models["main"].model, "lora_adapter")


def test_lora_strategy_load_checkpoint_validates_base_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    _install_fake_peft(monkeypatch)
    strategy = LoRAFineTuningStrategy(
        **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
    )
    strategy.save_checkpoint(tmp_path)

    class _ChangedFingerprintLoRAResult(_FakeLoRAResult):
        base_fingerprint = "changed"

    def fake_apply_lora(model: nn.Module, config: Any) -> _ChangedFingerprintLoRAResult:  # noqa: ARG001
        model.model.lora_adapter = _fake_lora_linear()
        _stash_fake_lora_config(model)
        return _ChangedFingerprintLoRAResult()

    monkeypatch.setattr("nvalchemi.training.peft._peft.apply_lora", fake_apply_lora)

    with pytest.raises(ValueError, match="base fingerprint mismatch"):
        LoRAFineTuningStrategy.load_checkpoint(tmp_path, map_location="cpu")


def test_lora_strategy_from_spec_dict_restores_adapter_metadata(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
) -> None:
    from test.training.conftest import _build_baseline_strategy_kwargs

    _install_fake_peft(monkeypatch)
    source = LoRAFineTuningStrategy(
        **{
            **baseline_strategy_kwargs,
            "lora_config": _lora_config(),
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

    restored = LoRAFineTuningStrategy.from_spec_dict(
        source.to_spec_dict(),
        models=_build_baseline_strategy_kwargs()["models"],
    )

    assert isinstance(restored.hooks[0], LoRAApplyHook)
    assert isinstance(restored.hooks[1], ModulePatchHook)
    assert isinstance(restored.hooks[2], LoRATrainableParameterHook)
    assert isinstance(restored.lora_config, LoRAConfig)
    assert restored.lora_config.target_modules == ["model.lora_adapter"]
    assert set(restored.module_patches) == {"main.model.aux_projection"}
    assert restored.trainable_patterns == ("main.model.projection.*",)
    assert restored._optimizer_parameter_names is not None


# ---------------------------------------------------------------------------
# Adapter export behavior
# ---------------------------------------------------------------------------


def test_lora_save_adapter_writes_alchemi_payload(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    def fake_apply_lora(model: nn.Module, config: Any) -> _FakeLoRAResult:  # noqa: ARG001
        model.model.lora_adapter = _fake_lora_linear()
        _stash_fake_lora_config(model)
        return _FakeLoRAResult()

    monkeypatch.setattr("nvalchemi.training.peft._peft.apply_lora", fake_apply_lora)
    monkeypatch.setattr(
        "nvalchemi.training.peft._peft.is_lora_layer",
        lambda module: bool(getattr(module, "_fake_lora", False)),
    )
    monkeypatch.setattr(
        "nvalchemi.training.peft.wrappers.register_builtin_lora_wrappers",
        lambda: None,
    )
    strategy = LoRAFineTuningStrategy(
        **{
            **baseline_strategy_kwargs,
            "lora_config": _lora_config(),
            "module_patches": {
                "main.model.aux_projection": create_model_spec(
                    nn.Linear,
                    in_features=8,
                    out_features=1,
                )
            },
            "trainable_patterns": ("main.model.*",),
        }
    )

    adapter_root = tmp_path / "adapter"
    adapter_dir = strategy.save_adapter(adapter_root)

    assert adapter_dir == adapter_root / "lora"
    manifest = json.loads((adapter_dir / "manifest.json").read_text())
    strategy_spec = json.loads((adapter_dir / "strategy.json").read_text())
    lora_state = torch.load(
        adapter_dir / "models" / "main" / "lora.pt",
        weights_only=True,
    )
    extras_state = torch.load(
        adapter_dir / "models" / "main" / "extras.pt",
        weights_only=True,
    )
    patches_state = torch.load(
        adapter_dir / "models" / "main" / "patches.pt",
        weights_only=True,
    )

    assert manifest["kind"] == "nvalchemi_lora_adapter"
    assert manifest["models"]["main"]["lora_state"] == "models/main/lora.pt"
    assert manifest["models"]["main"]["extras_state"] == "models/main/extras.pt"
    assert manifest["models"]["main"]["patches_state"] == "models/main/patches.pt"
    assert manifest["models"]["main"]["module_patches"] == ["main.model.aux_projection"]
    assert "module_patches" not in manifest
    assert "module_patches" in strategy_spec
    assert strategy_spec["base_model_fingerprints"] == {"main": "fingerprint-ok"}
    assert "freeze_patterns" not in strategy_spec
    assert "model.lora_adapter.weight" in lora_state
    assert "model.lora_adapter.weight" not in extras_state
    assert "model.projection.weight" in extras_state
    assert "main.model.aux_projection" in patches_state


def test_lora_save_adapter_can_export_ema_inference_model(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    _install_fake_peft(monkeypatch)
    strategy = LoRAFineTuningStrategy(
        **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
    )
    ema_model = copy.deepcopy(strategy.models["main"])
    with torch.no_grad():
        strategy.models["main"].model.lora_adapter.weight.fill_(2.0)
        ema_model.model.lora_adapter.weight.fill_(9.0)
    strategy.inference_model = ema_model

    adapter_dir = strategy.save_adapter(tmp_path / "adapter", use_ema=True)
    lora_state = torch.load(
        adapter_dir / "models" / "main" / "lora.pt",
        weights_only=True,
    )

    assert torch.equal(
        lora_state["model.lora_adapter.weight"],
        torch.full_like(lora_state["model.lora_adapter.weight"], 9.0),
    )


def test_lora_save_adapter_use_ema_requires_inference_model(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    _install_fake_peft(monkeypatch)
    strategy = LoRAFineTuningStrategy(
        **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
    )

    with pytest.raises(RuntimeError, match="inference_model"):
        strategy.save_adapter(tmp_path / "adapter", use_ema=True)


def test_lora_save_adapter_warns_for_existing_adapter_directory(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    _install_fake_peft(monkeypatch)
    strategy = LoRAFineTuningStrategy(
        **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
    )
    adapter_root = tmp_path / "adapter"
    adapter_dir = strategy.save_adapter(adapter_root)
    stale_file = adapter_dir / "stale.txt"
    stale_file.write_text("keep me")

    with pytest.warns(UserWarning, match="already exists and is non-empty"):
        saved_dir = strategy.save_adapter(adapter_root)

    assert saved_dir == adapter_dir
    assert (adapter_dir / "manifest.json").is_file()
    assert (adapter_dir / "strategy.json").is_file()
    assert stale_file.read_text() == "keep me"


def test_lora_save_adapter_rejects_direct_module_patches(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    def fake_apply_lora(model: nn.Module, config: Any) -> _FakeLoRAResult:  # noqa: ARG001
        model.model.lora_adapter = _fake_lora_linear()
        _stash_fake_lora_config(model)
        return _FakeLoRAResult()

    monkeypatch.setattr("nvalchemi.training.peft._peft.apply_lora", fake_apply_lora)
    monkeypatch.setattr(
        "nvalchemi.training.peft._peft.is_lora_layer",
        lambda module: bool(getattr(module, "_fake_lora", False)),
    )
    monkeypatch.setattr(
        "nvalchemi.training.peft.wrappers.register_builtin_lora_wrappers",
        lambda: None,
    )
    strategy = LoRAFineTuningStrategy(
        **{
            **baseline_strategy_kwargs,
            "lora_config": _lora_config(),
            "module_patches": {"main.model.aux_projection": nn.Linear(8, 1)},
        }
    )

    with pytest.raises(TypeError, match="BaseSpec"):
        strategy.save_adapter(tmp_path / "adapter")


def test_lora_save_adapter_rejects_fsdp_wrapped_model(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    _install_fake_peft(monkeypatch)
    strategy = LoRAFineTuningStrategy(
        **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
    )
    monkeypatch.setattr(
        "nvalchemi.training._checkpoint._is_fsdp_wrapped",
        lambda module: True,
    )

    with pytest.raises(NotImplementedError, match="FSDP/FSDP2"):
        strategy.save_adapter(tmp_path / "adapter")


def test_lora_save_adapter_requires_base_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    class _NoFingerprintLoRAResult(_FakeLoRAResult):
        base_fingerprint = ""

    def fake_apply_lora(model: nn.Module, config: Any) -> _NoFingerprintLoRAResult:  # noqa: ARG001
        model.model.lora_adapter = _fake_lora_linear()
        _stash_fake_lora_config(model)
        return _NoFingerprintLoRAResult()

    monkeypatch.setattr("nvalchemi.training.peft._peft.apply_lora", fake_apply_lora)
    monkeypatch.setattr(
        "nvalchemi.training.peft._peft.is_lora_layer",
        lambda module: bool(getattr(module, "_fake_lora", False)),
    )
    monkeypatch.setattr(
        "nvalchemi.training.peft.wrappers.register_builtin_lora_wrappers",
        lambda: None,
    )
    strategy = LoRAFineTuningStrategy(
        **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
    )

    with pytest.raises(ValueError, match="base_model_fingerprint"):
        strategy.save_adapter(tmp_path / "adapter")


def test_lora_save_adapter_unwraps_ddp_model(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    _install_fake_peft(monkeypatch)
    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", _FakeDDP)
    strategy = LoRAFineTuningStrategy(
        **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
    )
    strategy.models["main"] = _FakeDDP(strategy.models["main"])

    adapter_dir = strategy.save_adapter(tmp_path / "adapter")
    lora_state = torch.load(
        adapter_dir / "models" / "main" / "lora.pt",
        weights_only=True,
    )

    assert lora_state
    assert all(not key.startswith("module.") for key in lora_state)


# ---------------------------------------------------------------------------
# Adapter loading behavior
# ---------------------------------------------------------------------------


def test_lora_load_adapter_into_model_applies_saved_adapter(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    from test.training.conftest import _build_baseline_strategy_kwargs

    _install_fake_peft(monkeypatch)
    source = LoRAFineTuningStrategy(
        **{
            **baseline_strategy_kwargs,
            "lora_config": _lora_config(),
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
    with torch.no_grad():
        source.models["main"].model.lora_adapter.weight.fill_(0.25)
        source.models["main"].model.projection.weight.fill_(0.5)
        source.models["main"].model.aux_projection.weight.fill_(0.75)
    adapter_root = tmp_path / "adapter"
    source.save_adapter(adapter_root)

    base_model = _build_baseline_strategy_kwargs()["models"]
    loaded = LoRAFineTuningStrategy.load_adapter_into_model(base_model, adapter_root)

    assert loaded is base_model
    assert hasattr(loaded.model, "lora_adapter")
    assert hasattr(loaded.model, "aux_projection")
    assert torch.allclose(
        loaded.model.lora_adapter.weight,
        source.models["main"].model.lora_adapter.weight,
    )
    assert torch.allclose(
        loaded.model.projection.weight,
        source.models["main"].model.projection.weight,
    )
    assert torch.allclose(
        loaded.model.aux_projection.weight,
        source.models["main"].model.aux_projection.weight,
    )


def test_lora_load_adapter_requires_manifest(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    _install_fake_peft(monkeypatch)
    strategy = LoRAFineTuningStrategy(
        **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
    )

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        LoRAFineTuningStrategy.load_adapter_into_model(
            strategy.models["main"],
            tmp_path,
        )


@pytest.mark.parametrize(
    ("manifest_update", "match"),
    [
        ({"kind": "other"}, "kind"),
        ({"schema_version": 999}, "schema version"),
    ],
)
def test_lora_load_adapter_rejects_bad_manifest_metadata(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
    manifest_update: dict[str, Any],
    match: str,
) -> None:
    _install_fake_peft(monkeypatch)
    source = LoRAFineTuningStrategy(
        **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
    )
    adapter_dir = source.save_adapter(tmp_path / "adapter")
    manifest = json.loads((adapter_dir / "manifest.json").read_text())
    manifest.update(manifest_update)
    (adapter_dir / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=match):
        LoRAFineTuningStrategy.load_adapter_into_model(
            source.models["main"],
            adapter_dir,
        )


def test_lora_load_adapter_validates_base_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    _install_fake_peft(monkeypatch)
    source = LoRAFineTuningStrategy(
        **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
    )
    adapter_dir = source.save_adapter(tmp_path / "adapter")

    _install_fake_peft(monkeypatch, current_fingerprint="different")
    from test.training.conftest import _build_baseline_strategy_kwargs

    strict_target = _build_baseline_strategy_kwargs()["models"]
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        LoRAFineTuningStrategy.load_adapter_into_model(strict_target, adapter_dir)

    relaxed_target = _build_baseline_strategy_kwargs()["models"]
    with pytest.warns(UserWarning, match="fingerprint mismatch"):
        LoRAFineTuningStrategy.load_adapter_into_model(
            relaxed_target,
            adapter_dir,
            strict=False,
        )


def test_lora_load_adapter_rejects_missing_patch_spec(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    _install_fake_peft(monkeypatch)
    source = LoRAFineTuningStrategy(
        **{
            **baseline_strategy_kwargs,
            "lora_config": _lora_config(),
            "module_patches": {
                "main.model.aux_projection": create_model_spec(
                    nn.Linear,
                    in_features=8,
                    out_features=1,
                )
            },
        }
    )
    adapter_dir = source.save_adapter(tmp_path / "adapter")
    strategy = json.loads((adapter_dir / "strategy.json").read_text())
    strategy["module_patches"]["main.model.aux_projection"] = {"direct": True}
    (adapter_dir / "strategy.json").write_text(json.dumps(strategy))

    with pytest.raises(ValueError, match="BaseSpec"):
        LoRAFineTuningStrategy.load_adapter_into_model(
            source.models["main"],
            adapter_dir,
        )


def test_lora_load_adapter_rejects_unexpected_tensor_keys(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    _install_fake_peft(monkeypatch)
    source = LoRAFineTuningStrategy(
        **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
    )
    adapter_dir = source.save_adapter(tmp_path / "adapter")
    state_path = adapter_dir / "models" / "main" / "lora.pt"
    state = torch.load(state_path, weights_only=True)
    state["model.lora_adapter.not_a_parameter"] = torch.ones(1)
    torch.save(state, state_path)

    with pytest.raises(RuntimeError, match="unexpected tensor keys"):
        LoRAFineTuningStrategy.load_adapter_into_model(
            source.models["main"],
            adapter_dir,
        )


# ---------------------------------------------------------------------------
# Merge behavior
# ---------------------------------------------------------------------------


def test_lora_merge_model_inplace_merges_adapters(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
    tmp_path: Any,
) -> None:
    _install_fake_peft(monkeypatch)
    strategy = LoRAFineTuningStrategy(
        **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
    )

    merged = LoRAFineTuningStrategy.merge_model_inplace(strategy.models["main"])
    path = tmp_path / "merged.pt"
    torch.save(merged, path)

    assert path.exists()
    restored = torch.load(path, weights_only=False)
    assert isinstance(restored, type(merged))
    assert getattr(strategy.models["main"].model.lora_adapter, "_fake_lora") is False
    with pytest.raises(RuntimeError, match="after adapters were merged"):
        strategy.to_checkpoint_dict()


def test_lora_merge_model_inplace_rejects_leftover_lora(
    monkeypatch: pytest.MonkeyPatch,
    baseline_strategy_kwargs: dict[str, Any],
) -> None:
    _install_fake_peft(monkeypatch, leave_lora_after_merge=True)
    strategy = LoRAFineTuningStrategy(
        **{**baseline_strategy_kwargs, "lora_config": _lora_config()}
    )

    with pytest.raises(RuntimeError, match="non-mergeable adapter"):
        LoRAFineTuningStrategy.merge_model_inplace(strategy.models["main"])

    with pytest.warns(UserWarning, match="non-mergeable adapter"):
        LoRAFineTuningStrategy.merge_model_inplace(
            strategy.models["main"],
            strict=False,
        )


# ---------------------------------------------------------------------------
# e3nn wrapper behavior
# ---------------------------------------------------------------------------


def test_e3nn_fully_connected_lora_honors_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nvalchemi.training.peft.wrappers._import_e3nn", _fake_import_e3nn
    )

    def init_lora_A(param: torch.Tensor) -> None:
        with torch.no_grad():
            param.fill_(0.25)

    base_layer = _DummyE3NNFCLayer()
    wrapper = E3NNFullyConnectedLoRALayer(
        base_layer,
        rank=2,
        alpha=4.0,
        init=init_lora_A,
    )

    assert torch.allclose(wrapper.lora_A, torch.full_like(wrapper.lora_A, 0.25))
    assert torch.count_nonzero(wrapper.lora_B) == 0
    assert wrapper.scaling == 2.0
    assert base_layer.weight.requires_grad is False


def test_e3nn_fully_connected_lora_is_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nvalchemi.training.peft.wrappers._import_e3nn", _fake_import_e3nn
    )

    assert E3NNFullyConnectedLoRALayer.is_compatible(_DummyE3NNFCLayer())
    assert not E3NNFullyConnectedLoRALayer.is_compatible(_DummyE3NNFCLayer((3,)))
    assert not E3NNFullyConnectedLoRALayer.is_compatible(nn.Linear(3, 4))
