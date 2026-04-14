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
"""Dynamics simulation framework for molecular systems."""

from __future__ import annotations

from nvalchemi.dynamics import hooks, integrators, optimizers
from nvalchemi.dynamics._ops.thermostat_utils import initialize_velocities
from nvalchemi.dynamics.base import (
    BaseDynamics,
    ConvergenceHook,
    DistributedPipeline,
    DynamicsStage,
    FusedStage,
    Hook,
)
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.dynamics.integrators import NPH, NPT, NVE, NVTLangevin, NVTNoseHoover
from nvalchemi.dynamics.optimizers import (
    FIRE,
    FIRE2,
    FIRE2VariableCell,
    FIREVariableCell,
)
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import DataSink, GPUBuffer, HostMemory, ZarrData

__all__ = [
    "BaseDynamics",
    "ConvergenceHook",
    "DataSink",
    "DemoDynamics",
    "DistributedPipeline",
    "DynamicsStage",
    "FIRE",
    "FIRE2",
    "FIRE2VariableCell",
    "FIREVariableCell",
    "FusedStage",
    "GPUBuffer",
    "Hook",
    "HostMemory",
    "NPH",
    "NPT",
    "NVE",
    "NVTLangevin",
    "NVTNoseHoover",
    "SizeAwareSampler",
    "ZarrData",
    "hooks",
    "initialize_velocities",
    "integrators",
    "optimizers",
]
