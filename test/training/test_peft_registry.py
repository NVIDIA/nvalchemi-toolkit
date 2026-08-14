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
"""Tests for generic PEFT method registration and dispatch."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import pytest
from torch import nn

from nvalchemi.training.peft.config import (
    PeftConfig,
    build_peft_setup_hooks,
)
from nvalchemi.training.peft.registry import (
    available_peft_methods,
    register_peft_method,
)


class _DemoConfig(PeftConfig):
    peft_method: Literal["demo"] = "demo"
    value: int


class _DemoConfigSubclass(_DemoConfig):
    pass


def _build_peft_setup_hooks(
    config: PeftConfig,
    strategy_data: Mapping[str, Any],  # noqa: ARG001
) -> list[Any]:
    assert isinstance(config, _DemoConfig)
    return [config.value]


def _apply_peft_from_checkpoint_metadata(
    model: nn.Module,
    strategy_metadata: Mapping[str, Any],  # noqa: ARG001
    *,
    model_name: str,  # noqa: ARG001
    import_path_validator: Any = None,  # noqa: ARG001
) -> None:
    model._demo_peft_applied = True


def _merge(model: nn.Module, *, strict: bool = True) -> nn.Module:  # noqa: ARG001
    return model


def test_registered_peft_method_supports_config_round_trip_and_setup_hooks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from nvalchemi.training.peft import registry

    monkeypatch.setattr(
        registry,
        "_REGISTRATIONS_BY_METHOD",
        dict(registry._REGISTRATIONS_BY_METHOD),
    )
    monkeypatch.setattr(
        registry,
        "_REGISTRATIONS_BY_CONFIG",
        dict(registry._REGISTRATIONS_BY_CONFIG),
    )
    register_peft_method(
        "demo",
        config_cls=_DemoConfig,
        build_peft_setup_hooks=_build_peft_setup_hooks,
        apply_peft_from_checkpoint_metadata=_apply_peft_from_checkpoint_metadata,
        merge_peft=_merge,
    )
    assert "demo" in available_peft_methods()

    config = _DemoConfig(value=7)
    config_data = config.to_spec_dict()

    assert config_data == {"peft_method": "demo", "value": 7}
    assert config.peft_method == "demo"
    assert PeftConfig.from_spec_dict(config_data) == config
    assert build_peft_setup_hooks(config, {}) == [7]
    assert build_peft_setup_hooks(_DemoConfigSubclass(value=8), {}) == [8]
