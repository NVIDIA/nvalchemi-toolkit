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
"""Base class for the measurement records of the evaluation suite.

Every measurement exports with ``to_dict`` and rebuilds with ``from_dict``, so a
sweep can persist each student's results and aggregate them later. The record
base is :class:`nvalchemi._serialization.MeasurementRecord`, re-exported here
so the evaluation suite keeps one import path for it.
"""

from __future__ import annotations

from nvalchemi._serialization import MeasurementRecord

__all__ = ["MeasurementRecord"]
