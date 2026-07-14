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

"""Enumerations for the distributed declaration / helper vocabulary.

Small, type-safe choices that adapter bodies and declarations pass to the
helper functions — never raw strings.
"""

from __future__ import annotations

from enum import Enum

__all__ = ["Scope"]


class Scope(Enum):
    """Which rows a per-system reduction sums, and whether it crosses ranks.

    Passed to :func:`~nvalchemi.distributed.helpers.system_sum`.

    Attributes
    ----------
    OWNED
        Sum this rank's *owned* rows only, then all-reduce across the mesh
        to the true global per-system total (replicated on every rank).
        The default for an energy-like readout.
    LOCAL
        Sum this rank's owned rows into a *per-rank partial* with **no**
        cross-rank all-reduce; the framework's output consolidation
        finishes the global sum. Used where the consolidation step owns
        the reduction (e.g. a per-graph virial).
    """

    OWNED = "owned"
    LOCAL = "local"
