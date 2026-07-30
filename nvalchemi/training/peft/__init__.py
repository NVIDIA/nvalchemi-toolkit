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
"""Parameter-efficient fine-tuning helpers."""

from __future__ import annotations

from nvalchemi.training.peft import lora_wrappers
from nvalchemi.training.peft.fingerprints import (
    BaseFingerprintHook,
    compute_base_fingerprints,
    validate_base_fingerprints,
)
from nvalchemi.training.peft.lora_hook import LoRAHook

__all__ = [
    "BaseFingerprintHook",
    "CuEquivariantLoRALinear",
    "E3NNFullyConnectedLoRALayer",
    "EquivariantLoRALinear",
    "LoRAHook",
    "LoRAConfig",
    "LoRALayer",
    "LoRALinear",
    "LoRAWrappableLayer",
    "LoRAWrapper",
    "LoRAWrapperRegistrations",
    "PeftConfig",
    "available_lora_wrappers",
    "compute_base_fingerprints",
    "is_lora_layer",
    "load_peft_checkpoint_into_model",
    "register_builtin_lora_wrappers",
    "validate_base_fingerprints",
]
if lora_wrappers._TRANSFORMER_ENGINE_LORA_LINEAR is not None:
    __all__.append("TransformerEngineLoRALinear")


def __getattr__(name: str) -> object:
    """Lazily expose PEFT helpers with optional dependency boundaries."""
    if name == "LoRAConfig":
        from nvalchemi.training.peft import lora

        return getattr(lora, name)
    if name == "PeftConfig":
        from nvalchemi.training.peft import config

        return getattr(config, name)
    if name == "load_peft_checkpoint_into_model":
        from nvalchemi.training.peft import loading

        return getattr(loading, name)
    if name == "is_lora_layer":
        from nvalchemi.training.peft import _peft

        return _peft.is_lora_layer
    if name in {
        "CuEquivariantLoRALinear",
        "E3NNFullyConnectedLoRALayer",
        "EquivariantLoRALinear",
        "is_lora_layer",
        "LoRALayer",
        "LoRALinear",
        "LoRAWrappableLayer",
        "LoRAWrapper",
        "LoRAWrapperRegistrations",
        "TransformerEngineLoRALinear",
        "available_lora_wrappers",
        "register_builtin_lora_wrappers",
    }:
        from nvalchemi.training.peft import lora_wrappers

        return getattr(lora_wrappers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
