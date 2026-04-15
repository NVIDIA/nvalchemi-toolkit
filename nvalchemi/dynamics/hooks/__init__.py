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
"""
Dynamics hooks for observation, safety, and behavior modification.

This sub-package provides concrete hook implementations that plug into the
:class:`~nvalchemi.dynamics.base.BaseDynamics` hook system.  Every class
satisfies the :class:`~nvalchemi.hooks.Hook` protocol and can be
registered with any dynamics engine via
:meth:`~nvalchemi.dynamics.base.BaseDynamics.register_hook`.

Hooks are organized into the following modules:

.. list-table::
   :widths: 20 80

   * - :mod:`snapshot`
     - Save batch state to a :class:`~nvalchemi.dynamics.sinks.DataSink`.
   * - :mod:`logging`
     - Log scalar observables (energy, temperature, fmax, etc.).
   * - :mod:`safety`
     - Numerical safety guards (NaN detection, force clamping).
   * - :mod:`monitors`
     - Long-running diagnostic monitors (energy drift).
   * - :mod:`freeze`
     - Freeze selected atoms by category during dynamics.
   * - :mod:`cell_align`
     - Align periodic cells to upper-triangular form for variable-cell optimization.
   * - :mod:`profiling`
     - Performance profiling and step timing.

All hooks implement the :class:`~nvalchemi.hooks.Hook` protocol and accept
a :class:`~nvalchemi.hooks.HookContext` plus a stage enum in their
``__call__`` method.
"""

from __future__ import annotations

from nvalchemi.dynamics.hooks.cell_align import AlignCellHook
from nvalchemi.dynamics.hooks.freeze import FreezeAtomsHook
from nvalchemi.dynamics.hooks.logging import LoggingHook
from nvalchemi.dynamics.hooks.monitors import EnergyDriftMonitorHook
from nvalchemi.dynamics.hooks.profiling import ProfilerHook
from nvalchemi.dynamics.hooks.safety import MaxForceClampHook, NaNDetectorHook
from nvalchemi.dynamics.hooks.snapshot import ConvergedSnapshotHook, SnapshotHook

__all__ = [
    "AlignCellHook",
    "ConvergedSnapshotHook",
    "EnergyDriftMonitorHook",
    "FreezeAtomsHook",
    "LoggingHook",
    "MaxForceClampHook",
    "NaNDetectorHook",
    "ProfilerHook",
    "SnapshotHook",
]
