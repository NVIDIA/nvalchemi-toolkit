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
"""Steady-state throughput measurement for a model driving dynamics.

The measurement is :func:`nvalchemi.dynamics.measure_throughput`, which lives
in :mod:`nvalchemi.dynamics.benchmark` because it times any propagator; the
evaluation suite re-exports it with its record.
"""

from __future__ import annotations

from nvalchemi.dynamics.benchmark import ThroughputMetrics, measure_throughput

__all__ = ["ThroughputMetrics", "measure_throughput"]
