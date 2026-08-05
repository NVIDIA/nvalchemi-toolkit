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
"""Enhanced-sampling subpackage for nvalchemi-toolkit.

PR 1 (compile spike) public surface
------------------------------------
* :class:`BiasResult` — frozen dataclass; fully-detached bias outputs.
* :class:`BiasPotential` — ``@runtime_checkable`` Protocol; structural
  interface every bias must satisfy.
* :class:`ConservativeBias` — autograd helper; subclass and override
  :meth:`~ConservativeBias.energy` to get forces and virial for free.
* :func:`aggregate_bias_results` — sums a list of ``BiasResult`` objects.
* :func:`pair_distance` — P0 differentiable pair-distance CV; supports
  nonperiodic and general triclinic MIC.

Deferred to later PRs
---------------------
* :class:`EnhancedSampling` runner — PR 2
* :class:`ThermodynamicState`, :class:`ReplicaExchange` — PR 5
* Built-in biases (umbrella, metadynamics, walls, ABF) — PR 2–6
* Zarr checkpoint support — PR 4
"""

from nvalchemi.enhanced_sampling._bias import (
    BiasResult,
    BiasPotential,
    ConservativeBias,
    aggregate_bias_results,
)
from nvalchemi.enhanced_sampling.cv import pair_distance

__all__ = [
    # Core abstractions
    "BiasResult",
    "BiasPotential",
    "ConservativeBias",
    "aggregate_bias_results",
    # Collective variables
    "pair_distance",
]
