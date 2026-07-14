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

"""Result types for ``trace_and_validate``. Pure dataclasses, no behaviour."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nvalchemi.distributed.spec import MLIPSpec

__all__ = ["Attempt", "TraceReport"]


@dataclass
class Attempt:
    """One trip through validation with a particular spec."""

    spec: MLIPSpec
    rationale: str
    """Human-readable reason this spec was chosen
    (``"initial inference"`` for the first attempt, otherwise the rule's
    description)."""

    passed: bool
    max_abs_diff: dict[str, float]
    max_rel_diff: dict[str, float]
    handler_counts: dict[str, int]
    """Aggregate count of dispatch-trace records by handler name from
    the multi-rank run."""

    error: str | None = None
    """Set when the spawn worker raised an exception or NaNs were
    found; ``None`` on a clean numerical run."""

    helper_diagnostics: list[Any] = field(default_factory=list)
    """List of :class:`HelperDiagnosis` records — one per
    ``(module, function)`` pair observed in any watched third-party
    package during the run. ``Any``-typed in this annotation only to
    avoid a top-of-file import cycle; concrete type is
    :class:`nvalchemi.distributed._core.helper_diagnosis.HelperDiagnosis`.

    Iterate ``[d for d in attempt.helper_diagnostics if d.suspected_gap]``
    to surface the actionable subset — diagnoses for helpers the
    classifier flagged as likely needing distributed wrapping."""

    halo_completeness: dict[str, Any] | None = None
    """Verdict from
    :func:`~nvalchemi.distributed.validate.halo_diagnostics._check_halo_completeness`
    cross-referencing each rank's halo-padded NL against single-process's NL.
    Populated only for halo-storage specs. ``{'matches': True, ...}`` when each
    rank's owned atoms see the same neighbors as in single-process;
    ``{'matches': False, 'interpretation': ...}`` when halo coverage
    drops edges (the most common cause of "model output disagrees
    with single-process by a few percent under partial halo")."""

    layer_divergence: dict[str, Any] | None = None
    """Verdict from
    :func:`~nvalchemi.distributed.validate.layer_diagnostics.diff_layer_records`.
    Populated when ``trace_and_validate(..., layer_diagnostic=True)``.
    ``first_divergent`` names the first sub-module whose
    ``sum(rank_outputs)`` disagrees with the single-process output;
    that's where the bug is, not where the helper-trace classifier
    might be flagging downstream symptoms."""

    partition_health: dict[str, Any] | None = None
    """Verdict from
    :func:`~nvalchemi.distributed.validate.halo_diagnostics._partition_health`.
    Populated for halo-storage specs. Reports each rank's owned / halo /
    remote atom composition and flags a DEGENERATE partition — one where the
    distributed path isn't meaningfully exercised (a rank with 0 halo atoms =
    no cross-rank dependency; 0 remote atoms = sees every atom, so partition
    geometry is trivial; 0 owned = empty shard). A green validation on a
    degenerate partition is not evidence the spec is correct — surface this so
    the user picks a system/world_size with non-trivial owned + halo + remote
    counts on every rank."""


@dataclass
class TraceReport:
    """End-to-end ``trace_and_validate`` result.

    ``ok`` is the verdict; ``spec`` is the working spec on success (or
    closest variant on failure); ``attempts`` is the per-attempt log;
    ``next_action`` is the actionable one-liner the validator surfaces
    in the report.
    """

    ok: bool
    spec: MLIPSpec
    attempts: list[Attempt] = field(default_factory=list)

    next_action: str = ""
    """Single-line guidance the caller can act on:
    ``"OK — spec at report.spec is ready to use"`` on success;
    ``"investigate <op>"`` plus context on failure."""

    @property
    def fix_applied(self) -> str | None:
        """The rule that worked, if any (``None`` if first attempt
        passed or no rule cleared tolerance)."""
        if not self.ok or len(self.attempts) <= 1:
            return None
        return self.attempts[-1].rationale

    def log_summary(self, logger: Any) -> None:
        """Render this report onto a loguru-style ``logger`` (any object
        with ``.info / .warning / .error / .success`` methods). Surfaces
        every diagnostic field that points at the failure mode when
        validation didn't pass:

        * Worker-side error / traceback (or timeout marker).
        * Per-output absolute and relative diffs vs the single-process
          reference.
        * Dispatch-handler firings observed during the multi-rank run
          (or up to the crash, on RUN_ERROR).
        * Halo-completeness verdict (verified vs. incomplete).
        * Helper-diagnostic gaps from any watched third-party packages.

        On success, also reports any auto-fix that was applied and the
        per-output residual diffs so the caller can see the noise floor.

        This is the canonical "show the user the report" rendering — use
        it instead of building one in your own code so any future
        diagnostic field automatically lands in your output.
        """
        if self.ok:
            logger.success(
                "  validation PASSED in {n} attempt(s).",
                n=len(self.attempts),
            )
            if self.fix_applied is not None:
                logger.info("  auto-fix applied: {f!r}", f=self.fix_applied)
            last = self.attempts[-1] if self.attempts else None
            if last is not None and last.partition_health:
                self._log_partition_health(logger, last.partition_health)
            if last is not None and last.max_abs_diff:
                logger.info(
                    "  per-output abs/rel diffs (vs single-process):\n{d}",
                    d="\n".join(
                        f"    {k}: abs={last.max_abs_diff.get(k, 0.0):.3e}, "
                        f"rel={last.max_rel_diff.get(k, 0.0):.3%}"
                        for k in last.max_abs_diff
                    ),
                )
            return

        logger.warning("  validation did not pass:\n    {n}", n=self.next_action)

        last = self.attempts[-1] if self.attempts else None
        if last is None:
            return

        if last.error:
            logger.error("  worker error/traceback:\n{e}", e=last.error)

        if last.max_abs_diff:
            logger.info(
                "  per-output abs/rel diffs:\n{d}",
                d="\n".join(
                    f"    {k}: abs={last.max_abs_diff.get(k, 0.0):.3e}, "
                    f"rel={last.max_rel_diff.get(k, 0.0):.3%}"
                    for k in last.max_abs_diff
                ),
            )

        if last.handler_counts:
            logger.info(
                "  dispatch-handler firings (multi-rank run):\n{h}",
                h="\n".join(
                    f"    {name}: {count}"
                    for name, count in last.handler_counts.items()
                ),
            )

        if last.partition_health:
            self._log_partition_health(logger, last.partition_health)

        if last.halo_completeness:
            verdict = last.halo_completeness
            if verdict.get("matches"):
                logger.info(
                    "  halo coverage VERIFIED: every owned atom sees the "
                    "single-process neighbour count. Output divergences "
                    "originate elsewhere (combine rule / autograd topology)."
                )
            else:
                logger.warning(
                    "  halo coverage INCOMPLETE: {i}",
                    i=verdict.get("interpretation", "see ``halo_completeness``"),
                )

        flagged = [d for d in last.helper_diagnostics if d.suspected_gap]
        if flagged:
            logger.warning(
                "  helper-diagnostic gaps ({n}):\n{g}",
                n=len(flagged),
                g="\n".join(
                    f"    - {d.module}.{d.function}: {d.suspected_gap.splitlines()[0]}"
                    for d in flagged
                ),
            )

    @staticmethod
    def _log_partition_health(logger: Any, health: dict[str, Any]) -> None:
        """Render the per-rank owned/halo/remote composition + degeneracy
        warning. A degenerate partition means the validation didn't
        meaningfully exercise domain decomposition — surface it loudly."""
        comp = "\n".join(
            f"    rank{r}: owned={d['owned']} halo={d['halo']} remote={d['remote']}"
            for r, d in sorted(health.get("per_rank", {}).items())
        )
        if health.get("degenerate"):
            logger.warning(
                "  partition is DEGENERATE — a green result here is NOT "
                "evidence the spec is correct:\n{w}\n  composition:\n{c}",
                w="\n".join("    - " + m for m in health["degenerate"]),
                c=comp,
            )
        else:
            logger.info("  partition composition (owned/halo/remote):\n{c}", c=comp)
