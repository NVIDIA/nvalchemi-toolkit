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
"""Base model fingerprint helpers for PEFT workflows."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from torch import nn

from nvalchemi.hooks._context import HookContext
from nvalchemi.training import _strategy_validation as strategy_validation

__all__ = [
    "BaseFingerprintHook",
    "compute_base_fingerprints",
    "validate_base_fingerprints",
]


def compute_base_fingerprints(models: Mapping[str, nn.Module]) -> dict[str, str]:
    """Return base fingerprints for all named models."""
    from nvalchemi.training.peft import _peft

    return {
        model_name: _peft.compute_base_fingerprint(model)
        for model_name, model in models.items()
    }


def validate_base_fingerprints(
    models: strategy_validation.ModelInput,
    saved_fingerprints: Mapping[str, Any],
) -> None:
    """Raise if saved base fingerprints do not match current models."""
    if not isinstance(saved_fingerprints, Mapping):
        raise ValueError("base_model_fingerprints must be a mapping.")
    normalized_models = strategy_validation._normalize_models(models)
    current_fingerprints = compute_base_fingerprints(normalized_models)
    normalized_saved: dict[str, str] = {}
    for model_name, fingerprint in saved_fingerprints.items():
        if not isinstance(model_name, str) or not isinstance(fingerprint, str):
            raise ValueError(
                "base_model_fingerprints must map model names to fingerprint strings."
            )
        normalized_saved[model_name] = fingerprint
    if normalized_saved != current_fingerprints:
        mismatches = {
            name: (normalized_saved.get(name), current_fingerprints.get(name))
            for name in sorted(set(normalized_saved) | set(current_fingerprints))
            if normalized_saved.get(name) != current_fingerprints.get(name)
        }
        raise ValueError(f"PEFT base fingerprint mismatch: {mismatches!r}.")


class BaseFingerprintHook(BaseModel):
    """Cache model fingerprints before PEFT hooks mutate models."""

    frequency: ClassVar[int] = 1
    stage: ClassVar[None] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _runs_on_stage(self, stage: Enum) -> bool:  # noqa: ARG002
        """Return ``False`` because fingerprinting runs on registration."""
        return False

    def __call__(self, ctx: HookContext, stage: Enum) -> None:  # noqa: ARG002
        """No-op stage hook; fingerprinting is handled by ``on_register``."""
        return

    def on_register(self, workflow: Any) -> None:
        """Store base model fingerprints on ``workflow``."""
        models = getattr(workflow, "models", None)
        if not isinstance(models, Mapping):
            raise TypeError(
                "BaseFingerprintHook requires a workflow with a models mapping."
            )
        register_fingerprints = getattr(
            workflow,
            "register_base_fingerprints",
            None,
        )
        if not callable(register_fingerprints):
            raise TypeError(
                "BaseFingerprintHook requires a workflow with a "
                "register_base_fingerprints(fingerprints) method."
            )
        register_fingerprints(compute_base_fingerprints(models))
