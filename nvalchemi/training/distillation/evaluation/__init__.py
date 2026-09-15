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
"""Evaluation and acceptance suite for distilled students."""

from __future__ import annotations

from nvalchemi.training.distillation.evaluation.accuracy import (
    AccuracyMetrics,
    AccuracyQuantity,
    NonConservativeResidual,
    evaluate_accuracy,
    nonconservative_residual,
)
from nvalchemi.training.distillation.evaluation.report import (
    BAR_FAMILIES,
    AcceptanceCheck,
    AcceptanceReport,
    AcceptanceThresholds,
    DrafterMetrics,
    MetricFamily,
    StudentEvaluation,
    StudentVerdict,
    build_acceptance_report,
    measured_bars,
)
from nvalchemi.training.distillation.evaluation.stability import (
    ExtensivityMetrics,
    RadialDistribution,
    RDFComparison,
    StabilityMetrics,
    StabilityMonitor,
    compare_radial_distributions,
    extensivity_error,
    radial_distribution,
    total_momentum,
)
from nvalchemi.training.distillation.evaluation.throughput import (
    ThroughputMetrics,
    measure_throughput,
)

__all__ = [
    "AcceptanceCheck",
    "AcceptanceReport",
    "AcceptanceThresholds",
    "AccuracyMetrics",
    "AccuracyQuantity",
    "BAR_FAMILIES",
    "DrafterMetrics",
    "ExtensivityMetrics",
    "MetricFamily",
    "NonConservativeResidual",
    "RDFComparison",
    "RadialDistribution",
    "StabilityMetrics",
    "StabilityMonitor",
    "StudentEvaluation",
    "StudentVerdict",
    "ThroughputMetrics",
    "build_acceptance_report",
    "compare_radial_distributions",
    "evaluate_accuracy",
    "extensivity_error",
    "measure_throughput",
    "measured_bars",
    "nonconservative_residual",
    "radial_distribution",
    "total_momentum",
]
