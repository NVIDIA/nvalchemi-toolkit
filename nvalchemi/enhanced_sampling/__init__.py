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

What a bias is
--------------
A bias is an **additive potential**: a
:class:`~nvalchemi.models.base.BaseModelMixin` that maps a ``Batch`` to
:data:`~nvalchemi._typing.ModelOutputs`, exactly like ``DFTD3ModelWrapper`` or
``LennardJonesModelWrapper``.  There is no bias-specific protocol and no
bias-specific result type: diagnostics ride as ``diagnostics/<key>`` entries
and the producer's state revision as ``state_version``, both general
``ModelOutputs`` conventions.  A bias whose state evolves during sampling adds
the :class:`~nvalchemi.hooks.StatefulHook` lifecycle — ``frequency``,
``stage``, ``read_only``, ``commit`` — through
:class:`AdaptivePotentialMixin`.

Public surface
--------------
* :class:`ConservativeBias` — autograd helper; subclass and override
  :meth:`~ConservativeBias.energy` to get forces and tensile-positive
  Cauchy stress for free.
* :class:`AdaptivePotentialMixin` — the ``StatefulHook`` battery for biases
  whose state evolves during sampling; supplies ``update`` / ``commit`` /
  state versioning.
* :func:`pair_distance` — differentiable pair-distance CV; supports
  nonperiodic and Minkowski-reduced triclinic MIC.  General triclinic MIC
  (unreduced cells via LLL) is not yet implemented.
* :func:`periodic_difference` — CV differences wrapped onto a circle.
* :class:`EnhancedSampling` — the runner: walker identity, force-step
  ordering, exactly-once ``update()``, and force priming.
* :class:`ThermodynamicState`, :class:`ReplicaExchange` — synchronous
  replica exchange; swaps state labels, not coordinates.
* Built-in static biases: :class:`HarmonicUmbrellaBias`,
  :class:`UpperWall`, :class:`LowerWall`, :class:`FlatBottomRestraint`.
* :class:`WellTemperedMetaDynamicsBias` — Gaussian hills along a chosen CV,
  with well-tempered height damping and a free-energy estimator.
* :class:`RMSDMetaDynamicsBias` — xTB-style repulsion from retained
  reference geometries, for exploring when the interesting coordinates are
  not known in advance.  Non-periodic systems only.

* :class:`AdaptiveBiasingForce` — measures and cancels the mean force along
  a pair distance, including the metric correction that a naive Cartesian
  projection omits.  Force-only, so it is excluded from replica exchange.

Two helpers this subpackage relies on live in shared code rather than here,
because neither is specific to sampling: :class:`~nvalchemi.hooks.BiasContext`
(the ``DynamicsContext`` subclass handed to ``update``, placed beside
``TrainContext``) and
:func:`~nvalchemi.models._utils.aggregate_contributions` (the strict
counterpart to ``sum_outputs``).

Not yet implemented
-------------------
* Asynchronous replica exchange
* General triclinic MIC for unreduced cells
* Replica exchange over a temperature ladder combined with a bias whose
  energy depends on the thermodynamic state (per-state umbrella windows,
  per-state metadynamics history); such biases declare
  ``state_dependent_for_exchange`` and the runner rejects the combination
  rather than applying an acceptance rule that does not cover it.

Relationship to ``BiasedPotentialHook``
---------------------------------------
:class:`~nvalchemi.hooks.BiasedPotentialHook` covers the same ground with a
narrower contract and is **deprecated** in favour of this subpackage.  With
:class:`EnhancedSampling` now available, the migration path is complete:
anything the hook can do, this subpackage does, and it carries a cell
response the hook has no slot for.  The hook remains functional so existing
code keeps working; no removal date is set.

============================  ==============================  ==========================================
Concern                       ``BiasedPotentialHook``         ``enhanced_sampling``
============================  ==============================  ==========================================
Contract                      ``bias_fn(batch) -> (E, F)``    ``BaseModelMixin -> ModelOutputs``
Forces                        written by hand                 autograd, from one energy definition
Cell response                 none                            symmetric-strain ``stress``
Composing several biases      in-place, sequential            summed against unmodified model output
Diagnostics                   none                            namespaced ``diagnostics/<key>``
Evolving bias state           closure-held, ad hoc            ``update()`` exactly once per due step
============================  ==============================  ==========================================

Which to use
    :class:`ConservativeBias` (or any ``BaseModelMixin`` returning
    ``ModelOutputs``), run through :class:`EnhancedSampling`, for everything
    new.  The cell response is the substantive difference: a ``bias_fn`` bias
    contributes no stress, so under NPT/NPH the barostat reads a
    ``batch.stress`` the bias never touched and the cell evolves as if the
    bias were absent — with no error raised.  Existing hook-based code is
    correct under NVE and NVT, where nothing reads the stress, and can be
    migrated when convenient rather than urgently.

No adapter is provided
    Bridging a bias onto ``bias_fn`` would have to drop its ``stress`` on the
    floor, since the hook has nowhere to put it — reintroducing the exact
    failure the new API exists to remove.  A silent adapter would be worse
    than none.
"""

from nvalchemi.enhanced_sampling._adaptive import AdaptivePotentialMixin
from nvalchemi.enhanced_sampling._bias import ConservativeBias
from nvalchemi.enhanced_sampling._exchange import (
    ReplicaExchange,
    ThermodynamicState,
)
from nvalchemi.enhanced_sampling._runner import EnhancedSampling
from nvalchemi.enhanced_sampling.biases import (
    AdaptiveBiasingForce,
    FlatBottomRestraint,
    HarmonicUmbrellaBias,
    LowerWall,
    RMSDMetaDynamicsBias,
    UpperWall,
    WellTemperedMetaDynamicsBias,
)
from nvalchemi.enhanced_sampling.cv import (
    pair_displacement,
    pair_distance,
    periodic_difference,
)

__all__ = [
    # Core abstractions
    "ConservativeBias",
    "AdaptivePotentialMixin",
    # Runner
    "EnhancedSampling",
    # Replica exchange
    "ThermodynamicState",
    "ReplicaExchange",
    # Collective variables
    "pair_distance",
    "pair_displacement",
    "periodic_difference",
    # Built-in biases
    "HarmonicUmbrellaBias",
    "UpperWall",
    "LowerWall",
    "FlatBottomRestraint",
    "WellTemperedMetaDynamicsBias",
    "RMSDMetaDynamicsBias",
    "AdaptiveBiasingForce",
]
