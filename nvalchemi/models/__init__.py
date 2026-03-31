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
from __future__ import annotations

from nvalchemi.models.aimnet2 import AIMNet2Potential
from nvalchemi.models.autograd import EnergyDerivativesStep
from nvalchemi.models.base import ForwardContext, Potential
from nvalchemi.models.composite import CompositeCalculator
from nvalchemi.models.contracts import (
    MLIPPotentialCard,
    NeighborListCard,
    NeighborListProfile,
    NeighborRequirement,
    PipelineContract,
    PotentialCard,
    PotentialProfile,
    StepCard,
    StepProfile,
)
from nvalchemi.models.demo import DemoPotential
from nvalchemi.models.dftd3 import DFTD3Config, DFTD3Potential
from nvalchemi.models.dsf import DSFCoulombConfig, DSFCoulombPotential
from nvalchemi.models.ewald import (
    EwaldCoulombConfig,
    EwaldCoulombPotential,
)
from nvalchemi.models.lj import LennardJonesConfig, LennardJonesPotential
from nvalchemi.models.mace import MACEPotential
from nvalchemi.models.metadata import (
    ATOMIC_CHARGES,
    DISPERSION,
    ELECTROSTATICS,
    IMPLICIT,
    MLIP,
    PAIRWISE,
    REPULSION,
    SHORT_RANGE,
    CheckpointInfo,
    ModelCard,
    PhysicalTerm,
)
from nvalchemi.models.neighbors import (
    AdaptiveNeighborListBuilder,
    AdaptiveNeighborListConfig,
    NeighborListBuilder,
    NeighborListBuilderConfig,
    neighbor_result_key,
)
from nvalchemi.models.pme import PMEConfig, PMEPotential
from nvalchemi.models.registry import (
    DEFAULT_CACHE_DIR,
    KnownArtifactEntry,
    ResolvedArtifact,
    get_known_artifact,
    list_known_artifacts,
    register_known_artifact,
    resolve_known_artifact,
)
from nvalchemi.models.results import CalculatorResults

__all__ = [
    "ATOMIC_CHARGES",
    "AIMNet2Potential",
    "AdaptiveNeighborListBuilder",
    "AdaptiveNeighborListConfig",
    "CalculatorResults",
    "CheckpointInfo",
    "CompositeCalculator",
    "DEFAULT_CACHE_DIR",
    "DFTD3Config",
    "DFTD3Potential",
    "DISPERSION",
    "DemoPotential",
    "DSFCoulombConfig",
    "DSFCoulombPotential",
    "ELECTROSTATICS",
    "EnergyDerivativesStep",
    "EwaldCoulombConfig",
    "EwaldCoulombPotential",
    "ForwardContext",
    "IMPLICIT",
    "KnownArtifactEntry",
    "LennardJonesConfig",
    "LennardJonesPotential",
    "MACEPotential",
    "MLIP",
    "MLIPPotentialCard",
    "ModelCard",
    "NeighborListBuilder",
    "NeighborListBuilderConfig",
    "NeighborListCard",
    "NeighborListProfile",
    "NeighborRequirement",
    "PAIRWISE",
    "PMEConfig",
    "PMEPotential",
    "PhysicalTerm",
    "PipelineContract",
    "Potential",
    "PotentialCard",
    "PotentialProfile",
    "REPULSION",
    "ResolvedArtifact",
    "SHORT_RANGE",
    "StepCard",
    "StepProfile",
    "get_known_artifact",
    "list_known_artifacts",
    "neighbor_result_key",
    "register_known_artifact",
    "resolve_known_artifact",
]
