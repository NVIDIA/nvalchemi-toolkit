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

Public surface
--------------
* :class:`BiasResult` — frozen dataclass; fully-detached bias outputs.
* :class:`BiasPotential` — ``@runtime_checkable`` Protocol; structural
  interface every bias must satisfy.
* :class:`ConservativeBias` — autograd helper; subclass and override
  :meth:`~ConservativeBias.energy` to get forces and tensile-positive
  Cauchy stress for free.
* :func:`aggregate_bias_results` — sums a list of ``BiasResult`` objects.
* :func:`pair_distance` — differentiable pair-distance CV; supports
  nonperiodic and Minkowski-reduced triclinic MIC.  General triclinic MIC
  (unreduced cells via LLL) is not yet implemented.

Not yet implemented
-------------------
* :class:`EnhancedSampling` runner
* :class:`ThermodynamicState`, :class:`ReplicaExchange`
* Built-in biases (umbrella, metadynamics, walls, ABF)
* Zarr checkpoint support
* General triclinic MIC for unreduced cells

Relationship to ``BiasedPotentialHook``
---------------------------------------
:class:`~nvalchemi.hooks.BiasedPotentialHook` covers the same ground with a
narrower contract and is **deprecated** in favour of this subpackage.  It
still works, and it is not scheduled for removal before the
:class:`EnhancedSampling` runner ships, because until then it is the only
way to actually apply a bias during dynamics.

============================  ==============================  ==========================================
Concern                       ``BiasedPotentialHook``         ``enhanced_sampling``
============================  ==============================  ==========================================
Contract                      ``bias_fn(batch) -> (E, F)``    ``BiasPotential.evaluate -> BiasResult``
Forces                        written by hand                 autograd, from one energy definition
Cell response                 none                            symmetric-strain ``stress``
Composing several biases      in-place, sequential            summed against unmodified model output
Diagnostics                   none                            namespaced ``observables``
Runs in dynamics today        yes                             not until the runner ships
============================  ==============================  ==========================================

Which to use
    Write new biases against :class:`ConservativeBias` (or
    :class:`BiasPotential` directly).  Its cell response is the substantive
    difference: a ``bias_fn`` bias contributes no stress, so under NPT/NPH
    the barostat reads a ``batch.stress`` the bias never touched and the
    cell evolves as if the bias were absent — with no error raised.  Reach
    for the hook only when you need a bias applied during dynamics right
    now and can accept that limitation (NVE/NVT, where no barostat reads
    the stress, is the safe case).

No adapter is provided
    Bridging a :class:`BiasPotential` onto ``bias_fn`` would have to drop
    :attr:`BiasResult.stress` on the floor, since the hook has nowhere to
    put it — reintroducing the exact failure the new API exists to remove.
    A silent adapter would be worse than none.
"""

from nvalchemi.enhanced_sampling._bias import (
    BiasPotential,
    BiasResult,
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
