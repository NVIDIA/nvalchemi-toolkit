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

from nvalchemi.training.peft.config import PeftConfig
from nvalchemi.training.peft.fingerprints import (
    BaseFingerprintHook,
    compute_base_fingerprints,
    validate_base_fingerprints,
)
from nvalchemi.training.peft.loading import load_peft_checkpoint_into_model
from nvalchemi.training.peft.registry import (
    PeftMethodRegistration,
    available_peft_methods,
    register_peft_method,
)

__all__ = [
    "BaseFingerprintHook",
    "PeftConfig",
    "PeftMethodRegistration",
    "available_peft_methods",
    "compute_base_fingerprints",
    "load_peft_checkpoint_into_model",
    "register_peft_method",
    "validate_base_fingerprints",
]
