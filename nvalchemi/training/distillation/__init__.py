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
"""Knowledge-distillation workflows for ALCHEMI training."""

from __future__ import annotations

from nvalchemi.training.distillation.config import (
    OnPolicyConfig,
    OnPolicySettings,
    ResizableSink,
)
from nvalchemi.training.distillation.hooks import (
    TeacherLabelHook,
    nonfinite_divergence,
)
from nvalchemi.training.distillation.labeling import label_dataset
from nvalchemi.training.distillation.losses import AtomicEnergyMatchingLoss
from nvalchemi.training.distillation.replay import (
    FIFO,
    AdmissionPolicy,
    EvictionPolicy,
    ReplayBuffer,
    ReplayEviction,
    build_mixed_loader,
)
from nvalchemi.training.distillation.scoring import (
    BUILTIN_SIGNALS,
    SUPPORTED_SIGNALS,
    InProcessTeacherScorer,
    NeighborListPolicy,
    SignalLevel,
    TeacherLabels,
    TeacherScorer,
    TeacherSignal,
    scorer_fields,
    signal_fields,
    signal_for_field,
)
from nvalchemi.training.distillation.seeding import (
    FitPolicy,
    InitialStructures,
    InitialStructuresSource,
    WithinBudget,
)
from nvalchemi.training.distillation.strategy import (
    DistillationStrategy,
    default_distillation_fn,
)

__all__ = [
    "FIFO",
    "AdmissionPolicy",
    "AtomicEnergyMatchingLoss",
    "BUILTIN_SIGNALS",
    "DistillationStrategy",
    "EvictionPolicy",
    "FitPolicy",
    "InProcessTeacherScorer",
    "InitialStructures",
    "InitialStructuresSource",
    "NeighborListPolicy",
    "OnPolicyConfig",
    "OnPolicySettings",
    "ReplayBuffer",
    "ReplayEviction",
    "ResizableSink",
    "SUPPORTED_SIGNALS",
    "SignalLevel",
    "TeacherLabelHook",
    "TeacherLabels",
    "TeacherScorer",
    "TeacherSignal",
    "WithinBudget",
    "build_mixed_loader",
    "default_distillation_fn",
    "label_dataset",
    "nonfinite_divergence",
    "scorer_fields",
    "signal_fields",
    "signal_for_field",
]
