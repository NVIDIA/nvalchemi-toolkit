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
"""Built-in bias implementations.

Static biases: :class:`HarmonicUmbrellaBias`, :class:`UpperWall`,
:class:`LowerWall`, :class:`FlatBottomRestraint`.

History-dependent biases: :class:`WellTemperedMetaDynamicsBias` (Gaussian
hills along a chosen collective variable) and :class:`RMSDMetaDynamicsBias`
(xTB-style repulsion from retained reference geometries, for when the
interesting coordinates are not known in advance).

Adaptive biasing force is not yet implemented.
"""

from nvalchemi.enhanced_sampling.biases.metadynamics import (
    WellTemperedMetaDynamicsBias,
)
from nvalchemi.enhanced_sampling.biases.rmsd_metad import RMSDMetaDynamicsBias
from nvalchemi.enhanced_sampling.biases.umbrella import HarmonicUmbrellaBias
from nvalchemi.enhanced_sampling.biases.walls import (
    FlatBottomRestraint,
    LowerWall,
    UpperWall,
)

__all__ = [
    "FlatBottomRestraint",
    "HarmonicUmbrellaBias",
    "LowerWall",
    "RMSDMetaDynamicsBias",
    "UpperWall",
    "WellTemperedMetaDynamicsBias",
]
