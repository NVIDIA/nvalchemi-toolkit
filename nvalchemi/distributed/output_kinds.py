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

"""Output kind classification for distributed consolidation.

Each model output is declared on
:attr:`~nvalchemi.distributed.spec.MLIPSpec.output_kinds`, making
:class:`MLIPSpec` the single source of truth that
:mod:`nvalchemi.distributed.output_consolidation` reads directly
(rather than inferring per-atom vs per-system from tensor shapes).

Output classification combines two axes:

1. **Shape**: per-atom (one row per node, ``n_padded``-aligned) vs
   per-system (one row per graph, ``n_systems``-aligned).
2. **Globalness**: each rank's value is a partial that needs combining
   across the mesh, vs already-globally-correct.

:data:`PER_NODE` and :data:`PER_GRAPH` cover the shape axis; the
:attr:`MLIPSpec.owned_only_outputs` /
:attr:`MLIPSpec.all_reduce_outputs` sets cover the (orthogonal)
globalness axis. :data:`GLOBAL` is the convenience kind for outputs
that are already correct on every rank and pass through untouched
(rare; typically scalar metadata or replicated config tensors).
:data:`UNKNOWN` lets a wrapper omit declarations — the consolidation
falls back to the shape heuristic and logs a warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = ["OutputKind", "OutputSpec", "Reduce"]


class OutputKind(Enum):
    """Per-output classification used by consolidation.

    See module docstring for the design rationale.

    Members
    -------
    PER_NODE
        One row per atom. Halo storage: ``shape[0] == n_padded`` (owned
        + halo rows). Sharded storage: ``shape[0] == n_owned`` (owned
        only). Combine rule depends on
        :attr:`MLIPSpec.owned_only_outputs` /
        :attr:`MLIPSpec.all_reduce_outputs` membership and whether the
        key is in :attr:`ModelConfig.autograd_outputs`.
    PER_GRAPH
        One row per system. ``shape[0] == n_systems``. Combine rule
        depends on autograd / all_reduce membership.
    GLOBAL
        Already globally-correct on every rank; passthrough. Rare —
        typically scalar metadata or replicated config tensors that
        come out of the wrapper unchanged.
    UNKNOWN
        Undeclared default. Consolidation falls back to the shape-based
        heuristic and logs a warning so the wrapper author knows to
        declare. Also accepted for non-tensor output values (which
        always pass through anyway).
    """

    PER_NODE = "per_node"
    PER_GRAPH = "per_graph"
    GLOBAL = "global"
    UNKNOWN = "unknown"


class Reduce(Enum):
    """How an output's per-rank value is combined into the global value.

    Passed inside :class:`OutputSpec`. Mirrors the three consolidation
    branches in :mod:`~nvalchemi.distributed.output_consolidation`.

    Members
    -------
    NONE
        Default per-kind consolidation (e.g. an autograd per-node output is
        halo-reverse-summed to owners; a per-graph output passes through).
    ALL_REDUCE
        Each rank holds a partial; sum across the mesh to the global value.
        (Maps to ``MLIPSpec.all_reduce_outputs``.)
    OWNED_ONLY
        Already globally-correct on every rank; slice/passthrough, no
        cross-rank reduce. (Maps to ``MLIPSpec.owned_only_outputs``.)
    """

    NONE = "none"
    ALL_REDUCE = "all_reduce"
    OWNED_ONLY = "owned_only"


@dataclass(frozen=True)
class OutputSpec:
    """How one named model output is shaped and combined under DD.

    The single per-output declaration that collapses the three parallel sets
    ``output_kinds`` / ``all_reduce_outputs`` / ``owned_only_outputs`` into one
    place::

        outputs={"stress": OutputSpec(kind=OutputKind.PER_GRAPH,
                                      reduce=Reduce.ALL_REDUCE)}

    :class:`MLIPSpec` accepts ``outputs={name: OutputSpec}`` and lowers it onto
    those legacy fields, so consolidation and serialization are unchanged.
    """

    kind: OutputKind = OutputKind.UNKNOWN
    reduce: Reduce = Reduce.NONE
