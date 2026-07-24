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

from nvalchemi.training.peft import wrappers as _lora_wrappers
from nvalchemi.training.peft.lora_hooks import LoRAApplyHook

__all__ = [
    "CuEquivariantLoRALinear",
    "E3NNFullyConnectedLoRALayer",
    "EquivariantLoRALinear",
    "LoRAApplyHook",
    "LoRAFineTuningStrategy",
    "LoRALayer",
    "LoRALinear",
    "LoRAWrappableLayer",
    "LoRAWrapper",
    "LoRAWrapperRegistrations",
    "available_lora_wrappers",
    "register_builtin_lora_wrappers",
]
if _lora_wrappers._TRANSFORMER_ENGINE_LORA_LINEAR is not None:
    __all__.append("TransformerEngineLoRALinear")


def __getattr__(name: str) -> object:
    """Lazily expose PEFT helpers with optional dependency boundaries."""
    if name == "LoRAFineTuningStrategy":
        from nvalchemi.training.peft.lora import LoRAFineTuningStrategy

        exports = {
            "LoRAFineTuningStrategy": LoRAFineTuningStrategy,
        }
        return exports[name]
    if name in {
        "CuEquivariantLoRALinear",
        "E3NNFullyConnectedLoRALayer",
        "EquivariantLoRALinear",
        "LoRALayer",
        "LoRALinear",
        "LoRAWrappableLayer",
        "LoRAWrapper",
        "LoRAWrapperRegistrations",
        "available_lora_wrappers",
        "register_builtin_lora_wrappers",
    }:
        from nvalchemi.training.peft import wrappers

        return getattr(wrappers, name)
    if name == "TransformerEngineLoRALinear":
        from nvalchemi.training.peft import wrappers

        return getattr(wrappers, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
