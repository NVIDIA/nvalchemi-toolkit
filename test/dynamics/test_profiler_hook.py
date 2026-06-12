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
"""Compatibility tests for removed dynamics ProfilerHook imports."""

from __future__ import annotations

import pytest


def test_removed_profiler_hook_import_from_package_points_to_replacements() -> None:
    """Old package-level ProfilerHook imports raise a targeted migration error."""
    with pytest.raises(ImportError, match="TorchProfilerHook.*StageTimingHook"):
        from nvalchemi.dynamics.hooks import ProfilerHook  # noqa: F401


def test_removed_profiler_hook_import_from_module_points_to_replacements() -> None:
    """Old module-level ProfilerHook imports raise a targeted migration error."""
    with pytest.raises(ImportError, match="TorchProfilerHook.*StageTimingHook"):
        from nvalchemi.dynamics.hooks.profiling import ProfilerHook  # noqa: F401


def test_stage_timing_hook_still_imports_from_dynamics_package() -> None:
    """StageTimingHook remains discoverable next to dynamics hooks."""
    from nvalchemi.dynamics.hooks import StageTimingHook
    from nvalchemi.hooks import StageTimingHook as SharedStageTimingHook

    assert StageTimingHook is SharedStageTimingHook
