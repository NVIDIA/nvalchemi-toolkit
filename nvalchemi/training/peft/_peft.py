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
"""Access to PhysicsNeMo PEFT primitives."""

from __future__ import annotations

from physicsnemo.experimental import peft as _physicsnemo_peft
from physicsnemo.experimental.peft import utils as _physicsnemo_peft_utils

PhysicsNeMoLoRAConfig = _physicsnemo_peft.LoRAConfig
LoRAInit = _physicsnemo_peft.config.LoRAInit
LoRALayer = _physicsnemo_peft.LoRALayer
LoRALinear = _physicsnemo_peft.LoRALinear
apply_lora = _physicsnemo_peft.apply_lora
ApplyResult = _physicsnemo_peft.ApplyResult
compute_base_fingerprint = _physicsnemo_peft_utils.compute_base_fingerprint
is_lora_layer = _physicsnemo_peft.is_lora_layer
load_adapter = _physicsnemo_peft.load_adapter
merge_lora = _physicsnemo_peft.merge_lora
register_lora_wrapper = _physicsnemo_peft.register_lora_wrapper
resolve_targets = _physicsnemo_peft.resolve_targets
save_adapter = _physicsnemo_peft.save_adapter
wrappable_types = _physicsnemo_peft.lora.wrappable_types
_LORA_WRAPPERS = _physicsnemo_peft.lora._LORA_WRAPPERS
