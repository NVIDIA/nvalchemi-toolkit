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
"""Compatibility shim for removed dynamics timing profiler hooks."""

from __future__ import annotations

from nvalchemi.hooks.stage_timing import StageTimingHook

__all__ = ["StageTimingHook"]

_REMOVED_HOOKS = {"ProfilerHook"}


def __getattr__(name: str) -> object:
    """Raise a targeted import error for removed profiler hook names."""
    if name in _REMOVED_HOOKS:
        raise ImportError(
            f"nvalchemi.dynamics.hooks.profiling.{name} was removed. "
            "Use nvalchemi.dynamics.hooks.TorchProfilerHook for PyTorch traces or nvalchemi.dynamics.hooks.StageTimingHook for per-stage timing instead."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
