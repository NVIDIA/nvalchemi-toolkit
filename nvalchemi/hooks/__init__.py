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
"""Shared hook infrastructure for nvalchemi workflows."""

from __future__ import annotations

from nvalchemi.hooks._context import DynamicsContext, HookContext, TrainContext
from nvalchemi.hooks._protocol import CheckpointableHook, Hook
from nvalchemi.hooks._registry import HookRegistryMixin
from nvalchemi.hooks.bias import BiasedPotentialHook
from nvalchemi.hooks.neighbor_list import NeighborListHook
from nvalchemi.hooks.periodic import WrapPeriodicHook
from nvalchemi.hooks.physicsnemo_profiling import TorchProfilerHook
from nvalchemi.hooks.reporting import (
    BaseRichLayout,
    DynamicsRichLayout,
    Reporter,
    ReporterMessage,
    ReportingErrorPolicy,
    ReportingOrchestrator,
    ReportingState,
    RichLayout,
    RichReporter,
    ScalarCallback,
    ScalarSnapshot,
    TensorBoardReporter,
    TensorBoardWriter,
    TrainingRichLayout,
    collect_scalars,
    extract_dynamics_scalars,
    extract_loss_scalars,
    extract_optimizer_lr_scalars,
    extract_scalars,
)
from nvalchemi.hooks.stage_timing import StageTimingHook

__all__ = [
    "BaseRichLayout",
    "BiasedPotentialHook",
    "CheckpointableHook",
    "DynamicsContext",
    "DynamicsRichLayout",
    "Hook",
    "HookContext",
    "HookRegistryMixin",
    "NeighborListHook",
    "Reporter",
    "ReporterMessage",
    "ReportingErrorPolicy",
    "ReportingOrchestrator",
    "ReportingState",
    "RichLayout",
    "RichReporter",
    "ScalarCallback",
    "ScalarSnapshot",
    "TensorBoardReporter",
    "TensorBoardWriter",
    "StageTimingHook",
    "TorchProfilerHook",
    "TrainContext",
    "TrainingRichLayout",
    "WrapPeriodicHook",
    "collect_scalars",
    "extract_dynamics_scalars",
    "extract_loss_scalars",
    "extract_optimizer_lr_scalars",
    "extract_scalars",
]
