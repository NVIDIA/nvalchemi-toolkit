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
"""Progress and result records returned by crystal packing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from nvalchemi.csp.data import RigidMoleculeASUBatch


class PackingStopReason(str, Enum):
    """Reason a packing call stopped producing candidates."""

    TARGET_REACHED = "target_reached"
    CANDIDATE_BUDGET_EXHAUSTED = "candidate_budget_exhausted"


@dataclass(frozen=True)
class PackingProgress:
    """Cumulative counters and overlap values reported during packing.

    ``generated_count`` and ``accepted_count`` are cumulative. ``active_count``,
    ``converged_count``, and ``expired_count`` describe the current contact
    check. ``replaced_count`` counts converged rows when the target ends the
    call, or distinct converged or expired rows when packing continues. It does
    not count newly sampled rows. ``generated_count`` includes any
    refill completed before the callback, while active and overlap diagnostics
    describe candidates before retirement or refill. ``total_overlap`` is the
    mean of the active candidates' summed positive contact overlaps, in Å.
    ``max_overlap`` is the largest positive contact overlap across those
    candidates, also in Å.
    """

    generated_count: int
    accepted_count: int
    active_count: int
    converged_count: int
    expired_count: int
    replaced_count: int
    iteration: int
    total_overlap: float
    max_overlap: float
    rank: int = 0
    world_size: int = 1


@dataclass(frozen=True)
class PackingResult:
    """Accepted structures in ASU representation and the packing call's
    stopping outcome.

    ``run_id`` identifies the packing call and is shared by the first column of
    ``structures.structure_ids``. In single-rank packing, the second column is
    the accepted row's zero-based ordinal. In a process group of size ``W``, it
    is ``group_rank + W * local_accept_ordinal`` so IDs are unique across ranks.

    Per-structure ``steps``, ``total_overlap``, and ``max_overlap`` diagnostics
    live in ``structures.properties`` so selection and storage preserve them.
    ``steps`` counts applied relaxation updates. ``total_overlap`` sums
    positive contact overlaps and ``max_overlap`` records the largest one;
    both distances are in Å at the acceptance check.

    A gathered result reports the sum of generated candidates across ranks and
    compares its row count with the global target. Rank-local results report
    that rank's counters and local quota.

    Examples
    --------
    An ASU result can be counted without expanding full-cell P1 atoms::

        result = packer.pack(packing_input, num_samples=4)
        if len(result):
            asu_batch = result.structures
    """

    structures: RigidMoleculeASUBatch
    generated_count: int
    stop_reason: PackingStopReason
    run_id: int

    def __len__(self) -> int:
        """Return the number of accepted structures in ASU representation."""
        return self.structures.num_structures
