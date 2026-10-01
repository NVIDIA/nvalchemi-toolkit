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
"""Packing progress, rank reports, and typed result records."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from typing import Generic, Literal, TypeVar

from nvalchemi.csp.data import RigidMoleculeASUBatch
from nvalchemi.data.batch import Batch

__all__ = [
    "OverlapReliefProgress",
    "PackingReport",
    "PackingResult",
    "PackingStopReason",
]

_StructureT = TypeVar("_StructureT", RigidMoleculeASUBatch, Batch)


class PackingStopReason(str, Enum):
    """Summary reason derived from the reports in a packing result."""

    TARGET_REACHED = "target_reached"
    CANDIDATE_BUDGET_EXHAUSTED = "candidate_budget_exhausted"
    SHORTFALL = "shortfall"


@dataclass(frozen=True)
class PackingReport:
    """One rank's requested, accepted, generated, and stopping counts.

    A complete report has accepted_count equal to requested_count and the
    target_reached reason. An incomplete report must use another nonempty
    reason string; implementations may define additional rank-local reasons.
    generated_count is None when the implementation cannot report it.

    Parameters
    ----------
    rank : int
        Nonnegative rank represented by this report.
    requested_count : int
        Accepted structures requested from this rank.
    accepted_count : int
        Structures this rank placed in the result payload.
    stop_reason : str
        target_reached for a complete report or a nonempty reason for a
        shortfall.
    generated_count : int or None, optional
        Number of candidates initialized by this rank, if known.
    """

    rank: int
    requested_count: int
    accepted_count: int
    stop_reason: str
    generated_count: int | None = None

    def __post_init__(self) -> None:
        """Validate counters and keep the complete/incomplete reason coherent."""
        for name in ("rank", "requested_count", "accepted_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be a nonnegative integer")
            value = int(value)
            if value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
            object.__setattr__(self, name, value)
        if self.generated_count is not None:
            generated = self.generated_count
            if isinstance(generated, bool) or not isinstance(generated, Integral):
                raise TypeError("generated_count must be a nonnegative integer or None")
            generated = int(generated)
            if generated < 0:
                raise ValueError(
                    "generated_count must be a nonnegative integer or None"
                )
            object.__setattr__(self, "generated_count", generated)
        if self.accepted_count > self.requested_count:
            raise ValueError("accepted_count must not exceed requested_count")
        reason = self.stop_reason
        if isinstance(reason, PackingStopReason):
            reason = reason.value
        if not isinstance(reason, str):
            raise TypeError("stop_reason must be a nonempty string")
        if not reason:
            raise ValueError("stop_reason must be a nonempty string")
        object.__setattr__(self, "stop_reason", reason)
        if self.complete and reason != PackingStopReason.TARGET_REACHED.value:
            raise ValueError("a complete report must use stop_reason='target_reached'")
        if not self.complete and reason == PackingStopReason.TARGET_REACHED.value:
            raise ValueError(
                "an incomplete report cannot use stop_reason='target_reached'"
            )

    @property
    def complete(self) -> bool:
        """Whether this rank accepted its entire requested quota."""
        return self.accepted_count == self.requested_count


@dataclass(frozen=True)
class OverlapReliefProgress:
    """Cumulative counters and overlap values reported during local packing.

    ``generated_count`` and ``accepted_count`` are cumulative. ``active_count``,
    ``converged_count``, and ``expired_count`` describe the current contact
    check. ``replaced_count`` counts converged rows when the target ends the
    call, or distinct converged or expired rows when packing continues. It does
    not count newly sampled rows. ``generated_count`` includes any refill
    completed before the callback, while active and overlap diagnostics describe
    candidates before retirement or refill. ``total_overlap`` is the mean of
    the active candidates' summed positive contact overlaps, in angstroms.
    ``max_overlap`` is the largest positive contact overlap across those
    candidates, also in angstroms.
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
class PackingResult(Generic[_StructureT]):
    """Typed structures and one or more reports for a packing call.

    The payload may contain compact RigidMoleculeASUBatch rows or materialized
    Toolkit Batch graphs. Its count is checked against the reports at
    construction; report counters remain the stable packing outcome if a
    caller later filters or mutates the payload. A local result has one
    report. A gathered result has rank-ordered reports from zero through
    world_size minus one. This record performs no process-group communication.

    Parameters
    ----------
    structures : RigidMoleculeASUBatch or Batch
        Compact ASU structures or materialized Toolkit graphs.
    run_id : int
        Nonnegative 63-bit identity for this packing call.
    reports : tuple of PackingReport
        Nonempty rank-ordered completion records. Their accepted counts must
        equal the number of structures present at construction.
    scope : local or gathered, default=local
        Whether reports describe one rank or all ranks in rank order.

    Attributes
    ----------
    accepted_count : int
        Number of structures accepted when the result was constructed.
    requested_count : int
        Sum of requested counts across reports.
    complete : bool
        Whether the accepted count reaches the requested count.
    generated_count : int or None
        Sum of generated candidates, or None if any report omits that value.
    stop_reason : PackingStopReason
        Summary of completion or rank-local shortfall reasons.
    """

    structures: _StructureT
    run_id: int
    reports: tuple[PackingReport, ...]
    scope: Literal["local", "gathered"] = "local"

    def __post_init__(self) -> None:
        """Validate the payload identity and report aggregation contract."""
        if isinstance(self.run_id, bool) or not isinstance(self.run_id, Integral):
            raise TypeError("run_id must be an integer")
        run_id = int(self.run_id)
        if not 0 <= run_id < 2**63:
            raise ValueError("run_id must satisfy 0 <= run_id < 2**63")
        object.__setattr__(self, "run_id", run_id)

        if isinstance(self.reports, (str, bytes)):
            raise TypeError(
                "reports must be a nonempty sequence of PackingReport values"
            )
        reports = tuple(self.reports)
        if not reports:
            raise ValueError("reports must contain at least one PackingReport")
        if any(not isinstance(report, PackingReport) for report in reports):
            raise TypeError("reports must contain only PackingReport values")
        ranks = tuple(report.rank for report in reports)
        if any(right <= left for left, right in zip(ranks, ranks[1:], strict=False)):
            raise ValueError("report ranks must be unique and ordered")
        if self.scope not in ("local", "gathered"):
            raise ValueError("scope must be 'local' or 'gathered'")
        if self.scope == "local" and len(reports) != 1:
            raise ValueError("a local result must contain exactly one report")
        if self.scope == "gathered" and ranks != tuple(range(len(reports))):
            raise ValueError(
                "gathered reports must contain ranks 0 through world_size - 1"
            )
        object.__setattr__(self, "reports", reports)

        if isinstance(self.structures, RigidMoleculeASUBatch):
            accepted_count = self.structures.num_structures
        elif isinstance(self.structures, Batch):
            accepted_count = self.structures.num_graphs
        else:
            raise TypeError("structures must be a RigidMoleculeASUBatch or Batch")
        if accepted_count != sum(report.accepted_count for report in reports):
            raise ValueError(
                "reported accepted counts must equal the occupied payload count"
            )

    @property
    def accepted_count(self) -> int:
        """Number of structures accepted at result construction."""
        return sum(report.accepted_count for report in self.reports)

    @property
    def requested_count(self) -> int:
        """Sum of requested counts across the represented rank reports."""
        return sum(report.requested_count for report in self.reports)

    @property
    def complete(self) -> bool:
        """Whether the accepted count reaches the requested count."""
        return self.accepted_count == self.requested_count

    @property
    def generated_count(self) -> int | None:
        """Sum of generated candidates, or ``None`` if any rank is unknown."""
        if any(report.generated_count is None for report in self.reports):
            return None
        return sum(
            report.generated_count
            for report in self.reports
            if report.generated_count is not None
        )

    @property
    def stop_reason(self) -> PackingStopReason:
        """Summarize completion and any rank-local shortfall reasons."""
        if self.complete:
            return PackingStopReason.TARGET_REACHED
        incomplete = [report for report in self.reports if not report.complete]
        if incomplete and all(
            report.stop_reason == PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED.value
            for report in incomplete
        ):
            return PackingStopReason.CANDIDATE_BUDGET_EXHAUSTED
        return PackingStopReason.SHORTFALL

    def __len__(self) -> int:
        """Return the number of structures accepted by the packing call.

        Returns
        -------
        int
            Accepted count recorded in the reports.
        """
        return self.accepted_count
