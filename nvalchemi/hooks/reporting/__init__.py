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
"""Hook-native workflow reporting."""

from __future__ import annotations

from nvalchemi.hooks.reporting._jsonl import JSONLMode, JSONLReporter, RankReduction
from nvalchemi.hooks.reporting._orchestrator import (
    DEFAULT_REPORT_STAGES,
    ReportingErrorPolicy,
    ReportingOrchestrator,
)
from nvalchemi.hooks.reporting._protocol import Reporter
from nvalchemi.hooks.reporting._scalars import (
    ScalarCallback,
    ScalarSnapshot,
    collect_scalars,
    extract_loss_scalars,
    extract_optimizer_lr_scalars,
    extract_scalars,
)
from nvalchemi.hooks.reporting._state import ReporterMessage, ReportingState

__all__ = [
    "DEFAULT_REPORT_STAGES",
    "JSONLMode",
    "JSONLReporter",
    "RankReduction",
    "Reporter",
    "ReporterMessage",
    "ReportingErrorPolicy",
    "ReportingOrchestrator",
    "ReportingState",
    "ScalarCallback",
    "ScalarSnapshot",
    "collect_scalars",
    "extract_loss_scalars",
    "extract_optimizer_lr_scalars",
    "extract_scalars",
]
