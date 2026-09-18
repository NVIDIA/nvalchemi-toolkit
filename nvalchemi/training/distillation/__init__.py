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

from nvalchemi.training.distillation.config import OnPolicyConfig, OnPolicySettings
from nvalchemi.training.distillation.hooks import TeacherLabelHook
from nvalchemi.training.distillation.labeling import label_dataset
from nvalchemi.training.distillation.losses import (
    AtomicEnergyMatchingLoss,
    BoltzmannMatchingLoss,
    EmbeddingMatchingLoss,
    EmbeddingProjector,
    HessianMatchingLoss,
)
from nvalchemi.training.distillation.replay import (
    FIFO,
    AdmissionPolicy,
    EvictionPolicy,
    ReplayBuffer,
    ReplayEviction,
    build_mixed_loader,
)
from nvalchemi.training.distillation.scoring import (
    SUPPORTED_SIGNALS,
    InProcessTeacherScorer,
    SignalLevel,
    TeacherLabels,
    TeacherScorer,
    hessian_vector_product,
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
    embedding_distillation_fn,
    hessian_distillation_fn,
)

__all__ = [
    "FIFO",
    "AdmissionPolicy",
    "AtomicEnergyMatchingLoss",
    "BoltzmannMatchingLoss",
    "DistillationStrategy",
    "EmbeddingMatchingLoss",
    "EmbeddingProjector",
    "EvictionPolicy",
    "FitPolicy",
    "HessianMatchingLoss",
    "InProcessTeacherScorer",
    "InitialStructures",
    "InitialStructuresSource",
    "OnPolicyConfig",
    "OnPolicySettings",
    "ReplayBuffer",
    "ReplayEviction",
    "SUPPORTED_SIGNALS",
    "SignalLevel",
    "TeacherLabelHook",
    "TeacherLabels",
    "TeacherScorer",
    "WithinBudget",
    "build_mixed_loader",
    "default_distillation_fn",
    "embedding_distillation_fn",
    "hessian_distillation_fn",
    "hessian_vector_product",
    "label_dataset",
    "scorer_fields",
    "signal_fields",
    "signal_for_field",
]
