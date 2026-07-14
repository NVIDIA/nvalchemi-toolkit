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

"""Pattern classifier over :class:`HelperCall` records.

Consumes the per-call summaries captured by :mod:`helper_trace` from
the validator's reference run and from each spawned worker, groups
them by ``(module, function)``, and emits a :class:`HelperDiagnosis`
per group describing:

* What pattern the helper appears to implement (per-system reduction,
  index-gather, scatter-add, sentinel mask, ...).
* Whether the per-rank outputs combine into the reference output the
  way that pattern predicts. (This is the *consistency check* — the
  thing that turns "input/output shapes look like a reduction" into
  "values verify it.")
* Whether the wrapper has already declared this helper wrapped via a
  :class:`PythonAdapter` (or :class:`JitAdapter`) in
  ``spec.distribution.third_party_helpers``.
* If unwrapped and the consistency check passes: a human-readable
  ``suspected_gap`` and ``likely_remedy`` so a wrapper author can act.

Scope
-----
The fully fleshed-out classifier is the per-system reduction — the
canonical "wrapper forgot to wrap a helper" case (e.g. AIMNet2's
``mol_sum``). Other patterns (scatter-add, full-tensor gather, sentinel
mask) have classifier stubs in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nvalchemi.distributed._core.helper_trace import HelperCall

__all__ = ["HelperDiagnosis", "classify"]


@dataclass
class HelperDiagnosis:
    """One classified third-party helper.

    ``suspected_gap`` and ``likely_remedy`` are populated only when the
    classifier is *confident* the helper needs distributed wrapping
    AND the wrapper hasn't declared it wrapped. Both ``None``
    otherwise — including for helpers that look fine (shape-invariant
    elementwise ops) and for helpers that *are* declared wrapped (we
    trust the spec).

    For tuple-/dict-returning helpers (e.g. UMA's
    ``compute_forces_and_stress`` returns ``(forces, stress)``), the
    top-level fields summarize the most-suspicious slot, and
    ``slot_diagnoses`` holds a per-slot verdict for fine-grained
    inspection.
    """

    module: str
    function: str
    n_calls_ref: int
    n_calls_per_rank: dict[int, int] = field(default_factory=dict)
    pattern: str = "unknown"
    pattern_confidence: float = 0.0
    consistency_check: str = ""
    consistency_passed: bool = False
    already_wrapped: bool = False
    suspected_gap: str | None = None
    likely_remedy: str | None = None

    # Per-slot verdicts for tuple / dict returns; empty for single-tensor
    # outputs. Each entry is a dict keyed like the top-level fields above
    # (``slot``, ``pattern``, ``consistency_check``, ``suspected_gap``, ...).
    slot_diagnoses: list[dict[str, Any]] = field(default_factory=list)

    # Free-form notes for the user, populated even when no formal pattern
    # matched, to give a starting point to investigate.
    divergence_notes: list[str] = field(default_factory=list)

    # Representative call data for the report; from the first call
    # (``call_index = 0``) of this (module, function) — identical across ranks
    # and the reference run since model forwards are deterministic.
    representative_input_shapes: dict[str, tuple[int, ...] | None] = field(
        default_factory=dict
    )
    representative_output_shape: tuple[int, ...] | None = None
    ref_output_summary: dict[str, Any] = field(default_factory=dict)
    rank_output_summaries: dict[int, dict[str, Any]] = field(default_factory=dict)


def classify(
    ref_calls: list[HelperCall],
    dist_calls_per_rank: dict[int, list[HelperCall]],
    already_wrapped_fns: set[tuple[str, str]],
) -> list[HelperDiagnosis]:
    """Produce one diagnosis per ``(module, function)`` pair in the trace.

    Parameters
    ----------
    ref_calls : list[HelperCall]
        Records captured during the single-process reference run.
    dist_calls_per_rank : dict[int, list[HelperCall]]
        Records keyed by rank id from the multi-rank run.
    already_wrapped_fns : set[tuple[str, str]]
        ``(module_path, attr_name)`` pairs that
        ``spec.distribution.third_party_helpers`` declares the wrapper covers.
        The diagnosis trusts the spec: if a helper is declared wrapped,
        ``suspected_gap`` stays ``None`` even when its signature would flag.

    Returns
    -------
    list[HelperDiagnosis]
        One diagnosis per observed ``(module, function)`` pair.
    """
    keys: set[tuple[str, str]] = {(c.module, c.function) for c in ref_calls}

    diagnoses: list[HelperDiagnosis] = []
    for module, function in sorted(keys):
        ref_for_fn = [
            c for c in ref_calls if c.module == module and c.function == function
        ]
        per_rank_for_fn: dict[int, list[HelperCall]] = {
            r: [c for c in calls if c.module == module and c.function == function]
            for r, calls in dist_calls_per_rank.items()
        }
        diag = _classify_one(
            module,
            function,
            ref_for_fn,
            per_rank_for_fn,
            already_wrapped=(module, function) in already_wrapped_fns,
        )
        diagnoses.append(diag)
    return diagnoses


def _iter_tensor_slots(
    output_summary: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Yield ``(slot_label, slot_summary)`` for each tensor leaf in a
    helper's output summary.

    * Single tensor: yields one entry with ``slot_label = ""``.
    * Tuple / list: yields one entry per ``[i]`` slot (only those
      whose value is a tensor summary).
    * Dict (e.g. AIMNet's ``data`` bag): yields one entry per
      tensor-valued key.

    Non-tensor leaves (scalars, nested non-tensor types) are skipped —
    the diagnostic is shape-driven and there's nothing to compare for
    those.
    """
    if not isinstance(output_summary, dict):
        return []
    # Top-level tensor: has ``shape``.
    if "shape" in output_summary:
        return [("", output_summary)]
    # Container types record ``type``: "tuple" / "list" / "dict".
    out: list[tuple[str, dict[str, Any]]] = []
    for k, v in output_summary.items():
        if k == "type" or k == "len":
            continue
        if isinstance(v, dict) and "shape" in v:
            out.append((k, v))
    return out


def _slot_summaries_for_rank(
    full_summary: dict[str, Any], slot_label: str
) -> dict[str, Any] | None:
    """Pick the summary for a specific slot from a per-rank output
    summary. Mirrors :func:`_iter_tensor_slots`'s slot-label
    convention."""
    if not isinstance(full_summary, dict):
        return None
    if slot_label == "":
        return full_summary if "shape" in full_summary else None
    return (
        full_summary.get(slot_label)
        if isinstance(full_summary.get(slot_label), dict)
        else None
    )


def _classify_one(
    module: str,
    function: str,
    ref_calls: list[HelperCall],
    per_rank_calls: dict[int, list[HelperCall]],
    *,
    already_wrapped: bool,
) -> HelperDiagnosis:
    """Build one :class:`HelperDiagnosis`. The classifier picks the
    *first* call (``call_index == 0``) as representative; for hot
    helpers called many times per forward, that's enough to determine
    the pattern, and the ``n_calls_*`` fields surface the multiplicity
    in the report.

    For tuple-/dict-returning helpers, classifies each tensor slot
    independently and aggregates the per-slot verdicts. The top-level
    ``pattern`` / ``suspected_gap`` / ``likely_remedy`` fields
    summarize the *most-suspicious* slot (preferring slots with
    non-None ``suspected_gap``, then highest ``pattern_confidence``).
    """
    diag = HelperDiagnosis(
        module=module,
        function=function,
        n_calls_ref=len(ref_calls),
        n_calls_per_rank={r: len(calls) for r, calls in per_rank_calls.items()},
        already_wrapped=already_wrapped,
    )
    if not ref_calls:
        return diag

    ref0 = ref_calls[0]
    diag.representative_input_shapes = {
        k: v.get("shape")
        for k, v in ref0.input_summary.items()
        if isinstance(v, dict) and "shape" in v
    }
    diag.representative_output_shape = ref0.output_summary.get("shape")
    diag.ref_output_summary = dict(ref0.output_summary)
    for rank, calls in per_rank_calls.items():
        if calls:
            diag.rank_output_summaries[rank] = dict(calls[0].output_summary)

    # Iterate tensor slots in the output. Single-tensor returns yield
    # one slot with label ""; tuple/dict returns yield one per slot.
    slots = _iter_tensor_slots(ref0.output_summary)
    if not slots:
        return diag  # Non-tensor return; nothing to classify.

    for slot_label, ref_slot in slots:
        # Build per-rank summaries for this specific slot.
        per_rank_slot_summaries: dict[int, dict[str, Any]] = {}
        for rank, calls in per_rank_calls.items():
            if not calls:
                continue
            slot_for_rank = _slot_summaries_for_rank(
                calls[0].output_summary, slot_label
            )
            if slot_for_rank is not None:
                per_rank_slot_summaries[rank] = slot_for_rank
        slot_verdict = _classify_slot(
            ref0,
            ref_slot,
            per_rank_slot_summaries,
            already_wrapped=already_wrapped,
            slot_label=slot_label,
        )
        diag.slot_diagnoses.append(slot_verdict)
        if slot_verdict.get("divergence_note"):
            diag.divergence_notes.append(slot_verdict["divergence_note"])

    # Pick the most-suspicious slot to surface at the top level:
    # preferring (a) non-None suspected_gap, (b) higher
    # pattern_confidence. Falls through to "unknown" if every slot
    # came back unclassified.
    best = max(
        diag.slot_diagnoses,
        key=lambda s: (
            s["suspected_gap"] is not None,
            s["pattern_confidence"],
        ),
        default=None,
    )
    if best is not None:
        diag.pattern = best["pattern"]
        diag.pattern_confidence = best["pattern_confidence"]
        diag.consistency_check = best["consistency_check"]
        diag.consistency_passed = best["consistency_passed"]
        diag.suspected_gap = best["suspected_gap"]
        diag.likely_remedy = best["likely_remedy"]

    return diag


def _classify_slot(
    ref_call: HelperCall,
    ref_slot: dict[str, Any],
    per_rank_slot_summaries: dict[int, dict[str, Any]],
    *,
    already_wrapped: bool,
    slot_label: str,
) -> dict[str, Any]:
    """Run pattern classifiers on a single tensor slot. Returns a
    dict with the per-slot verdict — slot label, pattern, consistency
    check, suspected gap, likely remedy, and an optional
    ``divergence_note`` (free-form description of how this slot's
    ranks vs. ref behave, even when no formal pattern matches)."""
    verdict: dict[str, Any] = {
        "slot": slot_label,
        "pattern": "unknown",
        "pattern_confidence": 0.0,
        "consistency_check": "",
        "consistency_passed": False,
        "suspected_gap": None,
        "likely_remedy": None,
        "ref_summary": ref_slot,
        "rank_summaries": dict(per_rank_slot_summaries),
        "divergence_note": None,
    }
    for classifier in _PATTERN_CLASSIFIERS:
        result = classifier(
            ref_call,
            ref_slot,
            per_rank_slot_summaries,
            already_wrapped,
            slot_label,
        )
        if result is not None:
            (
                verdict["pattern"],
                verdict["pattern_confidence"],
                verdict["consistency_check"],
                verdict["consistency_passed"],
                verdict["suspected_gap"],
                verdict["likely_remedy"],
            ) = result
            break

    # Always compute a divergence note when ref + per-rank summaries
    # let us — it surfaces in the validator's ``next_action`` even
    # when no formal pattern matched.
    note = _divergence_note(ref_slot, per_rank_slot_summaries, slot_label)
    if note:
        verdict["divergence_note"] = note
    return verdict


def _divergence_note(
    ref_slot: dict[str, Any],
    per_rank: dict[int, dict[str, Any]],
    slot_label: str,
) -> str | None:
    """Free-form one-line description of how this slot's per-rank
    outputs relate to ref. Used in the report's ``next_action`` even
    when no classifier emits a formal verdict — gives the user a
    starting point to investigate.

    Cases:
    * Per-rank values match ref to fp noise → no note (everything's
      fine for this slot).
    * Per-rank values agree with each other but disagree with ref →
      "replicated mismatch".
    * Per-rank values disagree with each other AND with ref →
      "rank-disaggregated divergence".
    * Ref value near zero, per-rank values not → "near-zero ref;
      compare absolutes only".
    """
    ref_sum = ref_slot.get("sum")
    ref_max = ref_slot.get("max_abs")
    if ref_sum is None or ref_max is None:
        return None
    rank_sums: list[tuple[int, float]] = []
    rank_maxes: list[tuple[int, float]] = []
    for r, s in sorted(per_rank.items()):
        rs = s.get("sum")
        rm = s.get("max_abs")
        if rs is None or rm is None:
            continue
        rank_sums.append((r, rs))
        rank_maxes.append((r, rm))
    if not rank_sums:
        return None

    # Near-zero ref handling.
    if abs(ref_max) < 1e-6:
        max_rank_max = max(m for _, m in rank_maxes)
        if max_rank_max > 1e-3:
            label = f"slot {slot_label!r} " if slot_label else ""
            return (
                f"{label}ref output is ~zero (|max|={ref_max:.2e}) but "
                f"per-rank |max| up to {max_rank_max:.2e} — likely "
                "degenerate test case (symmetry-zero forces?), not a "
                "real disagreement"
            )

    # Replicated check: do all ranks agree with each other to fp
    # noise? Tolerance: 1e-4 rel against the largest rank value.
    sums = [s for _, s in rank_sums]
    rank_spread = max(sums) - min(sums)
    biggest = max(abs(s) for s in sums) or 1.0
    ranks_agree = (rank_spread / biggest) < 1e-4

    # Compare ranks to ref. Use any-rank-vs-ref for replicated case,
    # or sum-of-ranks-vs-ref for partition-disaggregated case.
    rank_avg = sum(sums) / len(sums)
    rank_total = sum(sums)
    rel_avg = abs(rank_avg - ref_sum) / max(abs(ref_sum), 1e-30)
    rel_total = abs(rank_total - ref_sum) / max(abs(ref_sum), 1e-30)

    if ranks_agree and rel_avg > 1e-3:
        label = f"slot {slot_label!r} " if slot_label else ""
        return (
            f"{label}per-rank values agree with each other "
            f"(rank-spread {rank_spread:.2e}) but disagree with ref by "
            f"{rel_avg:.2%} — replicated output diverges from "
            "single-process. Likely cause: model computes this output "
            "from local-edge graph; consider whether it should see the "
            "global graph."
        )

    if not ranks_agree and rel_avg > 1e-3:
        label = f"slot {slot_label!r} " if slot_label else ""
        return (
            f"{label}per-rank values disagree with each other "
            f"(rank-spread {rank_spread:.2e}) AND with ref (avg-vs-ref "
            f"{rel_avg:.2%}) — each rank's output reflects its local "
            "subgraph, which differs both across ranks (partial halo) "
            "and from the global single-process graph. The combine "
            f"rule for this output isn't a sum (rel-of-sum {rel_total:.2%}) "
            "or an average; needs dedicated handling."
        )

    return None


# ----------------------------------------------------------------------
# Pattern classifiers — each returns ``None`` or a 6-tuple
# ``(pattern, confidence, consistency_check, passed, suspected_gap,
# likely_remedy)``. Confidence is a soft floor.
# ----------------------------------------------------------------------


def _shape_of_arg0(call: HelperCall) -> tuple[int, ...] | None:
    a0 = call.input_summary.get("arg0")
    if isinstance(a0, dict):
        return a0.get("shape")
    return None


def _classify_per_system_reduction(
    ref_call: HelperCall,
    ref_slot: dict[str, Any],
    per_rank_slot_summaries: dict[int, dict[str, Any]],
    already_wrapped: bool,
    slot_label: str,
) -> tuple[str, float, str, bool, str | None, str | None] | None:
    """Helper slot takes ``(N, *F)`` and produces ``(S, *F)`` with
    ``S < N``.

    Pattern signature: float-tensor input/output, output dim 0 strictly
    less than input dim 0. Consistency check: per-rank ``output.sum``
    values sum to the reference ``output.sum`` (within rel 1e-3 — wide
    enough that fp32 round-off across ~thousands of atoms doesn't
    create false negatives).
    """
    in_shape = _shape_of_arg0(ref_call)
    out_shape = ref_slot.get("shape")
    out_dtype = ref_slot.get("dtype", "")
    out_sum = ref_slot.get("sum")

    if (
        in_shape is None
        or out_shape is None
        or out_sum is None
        or not in_shape
        or not out_shape
        or not out_dtype.startswith("torch.float")
    ):
        return None

    n_in, n_out = in_shape[0], out_shape[0]
    if n_out >= n_in:
        return None

    # Bumped confidence when trailing dims match (clean reduction along
    # leading dim, nothing else fancy).
    confidence = 0.7
    if in_shape[1:] == out_shape[1:]:
        confidence = 0.9

    # Consistency check: do per-rank outputs at call_index=0 sum to ref?
    rank_sums: list[float] = []
    rank_summary_str: list[str] = []
    for rank, slot in sorted(per_rank_slot_summaries.items()):
        s = slot.get("sum")
        if s is None:
            continue
        rank_sums.append(s)
        rank_summary_str.append(f"rank{rank}={s:+.4e}")

    if not rank_sums:
        consistency = "no per-rank output recorded"
        passed = False
    else:
        total = sum(rank_sums)
        rel = abs(total - out_sum) / max(abs(out_sum), 1e-30)
        passed = rel <= 1e-3
        consistency = (
            f"sum across ranks ({' + '.join(rank_summary_str)} = {total:+.4e}) "
            f"vs ref ({out_sum:+.4e}): rel diff {rel:.2e} → "
            f"{'matches' if passed else 'does NOT match'} the per-system "
            "reduction prediction"
        )

    suspected_gap = None
    likely_remedy = None
    if passed and not already_wrapped:
        slot_str = f" (slot {slot_label!r})" if slot_label else ""
        suspected_gap = (
            f"{ref_call.module}.{ref_call.function}{slot_str} has the "
            "shape signature of a per-system reduction "
            f"(input dim0={n_in}, output dim0={n_out}) and the "
            "per-rank outputs sum to the reference output — i.e. each "
            "rank produced a partial that should have been all-reduced "
            "across the mesh. The wrapper hasn't declared this helper "
            "wrapped in spec.distribution.third_party_helpers, so each rank's "
            "forward sees only its local sum."
        )
        likely_remedy = (
            "Declare this helper on your spec's "
            "``distribution.third_party_helpers``:\n"
            f"    PythonAdapter(module_path={ref_call.module!r}, "
            f"attr_name={ref_call.function!r}, "
            "replacement=<your_distributed_replacement>)\n"
            "The replacement should call "
            "nvalchemi.distributed._core.per_system.per_system_reduce — see "
            "_distributed_mol_sum in nvalchemi/models/aimnet2.py for the "
            "reference template (it strips the local padding row and "
            "routes through per_system_reduce, which does local "
            "scatter-add + cross-rank all_reduce)."
        )

    return (
        "per_system_reduction",
        confidence,
        consistency,
        passed,
        suspected_gap,
        likely_remedy,
    )


def _classify_replicated_output(
    ref_call: HelperCall,
    ref_slot: dict[str, Any],
    per_rank_slot_summaries: dict[int, dict[str, Any]],
    already_wrapped: bool,
    slot_label: str,
) -> tuple[str, float, str, bool, str | None, str | None] | None:
    """Slot has the same shape on every rank and the same shape as ref,
    AND every rank computed the SAME value (within fp noise), AND that
    value disagrees with ref by > 1e-3 rel.

    Pattern signature: under full halo coverage every rank's input
    graph is the global graph, so each rank computes the same output.
    If that output also matches ref, the spec is fine and nothing
    flags. If the per-rank value matches across ranks but disagrees
    with ref, the model computed the same wrong value on every rank —
    typically because a downstream consolidation pass over-divided
    (``/world_size``) or because the reference path uses different
    edge-list construction logic from the distributed path.
    """
    out_shape = ref_slot.get("shape")
    out_sum = ref_slot.get("sum")
    out_dtype = ref_slot.get("dtype", "")

    if out_shape is None or out_sum is None or not out_dtype.startswith("torch.float"):
        return None
    if not per_rank_slot_summaries:
        return None
    # Need every rank to have a recorded sum AND a matching shape.
    rank_sums: list[tuple[int, float]] = []
    for r, slot in sorted(per_rank_slot_summaries.items()):
        s = slot.get("sum")
        sh = slot.get("shape")
        if s is None or sh != out_shape:
            return None
        rank_sums.append((r, s))
    if len(rank_sums) < 2:
        return None

    sums = [s for _, s in rank_sums]
    biggest = max(abs(x) for x in sums) or 1.0
    rank_spread = max(sums) - min(sums)
    if (rank_spread / biggest) >= 1e-4:
        # Ranks disagree with each other → not "replicated" pattern;
        # falls through to the rank-disaggregated classifier.
        return None

    rank_value = sums[0]  # all ranks agree to fp noise
    rel = abs(rank_value - out_sum) / max(abs(out_sum), 1e-30)
    if rel <= 1e-3:
        return None  # everything matches; nothing to flag

    confidence = 0.85
    consistency = (
        f"all ranks computed sum={rank_value:+.4e} (rank-spread "
        f"{rank_spread:.2e}), ref sum={out_sum:+.4e} → rel diff "
        f"{rel:.2%}: ranks AGREE with each other but disagree with "
        "single-process"
    )

    slot_str = f" (slot {slot_label!r})" if slot_label else ""
    if already_wrapped:
        # Helper IS declared wrapped, but the value still mismatches —
        # report the pattern but note it's the wrap that's wrong, not
        # the absence of wrap.
        suspected_gap = (
            f"{ref_call.module}.{ref_call.function}{slot_str} is "
            "declared wrapped in spec.distribution.third_party_helpers, but "
            "the wrapped replacement still produces a value that "
            "disagrees with "
            f"single-process by {rel:.2%}. Either the replacement is "
            "buggy, or the gap isn't actually a python-helper-level "
            "problem (could be a custom-op spec issue or a "
            "consolidation-path issue)."
        )
        likely_remedy = (
            "Compare the replacement's per-rank output (computed "
            "correctly) to single-process. If they should match but "
            "don't, the replacement has a bug. If the reference path "
            "uses different inputs than the distributed path (e.g. "
            "different edge-list construction), the disagreement is "
            "upstream of the helper."
        )
    else:
        suspected_gap = (
            f"{ref_call.module}.{ref_call.function}{slot_str}: every rank "
            f"computed the same value ({rank_value:+.4e}) but it "
            f"disagrees with single-process ({out_sum:+.4e}) by {rel:.2%}. "
            "Pattern: replicated output diverges from single-process. "
            "Common causes: (a) downstream consolidation /world_size "
            "divides a value that's already replicated, or (b) the "
            "distributed path constructs the model's input graph "
            "differently from the single-process path."
        )
        likely_remedy = (
            "Inspect ``output_consolidation`` for this output key. If "
            "it falls into the ``autograd_outputs`` /world_size branch "
            "but is in fact replicated by the model itself, add the "
            "key to ``MLIPSpec.owned_only_outputs`` to skip "
            "the divide. If the value really does need different "
            "handling, add a custom branch."
        )

    return (
        "replicated_output_diverges",
        confidence,
        consistency,
        False,
        suspected_gap,
        likely_remedy,
    )


def _classify_rank_disaggregated_divergence(
    ref_call: HelperCall,
    ref_slot: dict[str, Any],
    per_rank_slot_summaries: dict[int, dict[str, Any]],
    already_wrapped: bool,
    slot_label: str,
) -> tuple[str, float, str, bool, str | None, str | None] | None:
    """Per-rank values disagree with each other AND with ref; no clean
    combine rule (sum doesn't recover ref, average doesn't either).

    Pattern signature: partial-halo case where each rank's local
    subgraph differs from the global graph AND from each other rank's,
    so the helper computes a different value per rank, none matching
    ref. Typical cause is a *graph-aware* output (energy normalised by
    cell volume, stress, virial) computed from the rank-local edge
    set rather than the global edge set. **Not fixable by a spec
    change** — needs the distributed forward to expose the global
    graph to this output's computation, which is wrapper / model
    architecture territory.
    """
    out_shape = ref_slot.get("shape")
    out_sum = ref_slot.get("sum")
    out_dtype = ref_slot.get("dtype", "")

    if out_shape is None or out_sum is None or not out_dtype.startswith("torch.float"):
        return None
    if not per_rank_slot_summaries:
        return None
    rank_sums: list[tuple[int, float]] = []
    for r, slot in sorted(per_rank_slot_summaries.items()):
        s = slot.get("sum")
        sh = slot.get("shape")
        if s is None or sh != out_shape:
            return None
        rank_sums.append((r, s))
    if len(rank_sums) < 2:
        return None

    sums = [s for _, s in rank_sums]
    biggest = max(abs(x) for x in sums) or 1.0
    rank_spread = max(sums) - min(sums)
    # Ranks must disagree with each other for this pattern.
    if (rank_spread / biggest) < 1e-4:
        return None

    avg = sum(sums) / len(sums)
    total = sum(sums)
    rel_avg = abs(avg - out_sum) / max(abs(out_sum), 1e-30)
    rel_total = abs(total - out_sum) / max(abs(out_sum), 1e-30)
    rel_min = min(rel_avg, rel_total)

    # If averaging or summing recovers ref to within fp noise, the
    # pattern would be a normal sum-reduction or mean-reduction —
    # those have other classifiers (or a future addition). Only flag
    # when *neither* combine helps.
    if rel_min < 1e-3:
        return None

    # And if the divergence is small to begin with, fp noise across
    # ranks is enough — don't flag.
    if max(rel_avg, rel_total) < 1e-3:
        return None

    confidence = 0.7
    rank_str = ", ".join(f"rank{r}={s:+.4e}" for r, s in rank_sums)
    consistency = (
        f"per-rank sums ({rank_str}) — spread {rank_spread:.2e}; "
        f"ref={out_sum:+.4e}; rel diff: avg-vs-ref {rel_avg:.2%}, "
        f"sum-vs-ref {rel_total:.2%}. Neither combine recovers ref."
    )

    slot_str = f" (slot {slot_label!r})" if slot_label else ""
    suspected_gap = (
        f"{ref_call.module}.{ref_call.function}{slot_str}: each rank "
        "computed a different value, all disagree with single-process. "
        "No simple combine rule (sum/average) recovers the ref. "
        "Pattern: graph-aware output computed from rank-local subgraph "
        "(partial halo coverage). Each rank's output reflects the "
        "edges visible to it, not the global edge set."
    )
    likely_remedy = (
        "This isn't fixable by a ``MLIPSpec`` field change — "
        "the model needs to compute this output from the global edge "
        "set, not the per-rank halo-padded subgraph. Options:\n"
        "  (a) move this output's computation into a wrapper-level "
        "hook that operates on the all-gathered global tensor;\n"
        "  (b) make the model's relevant submodule halo-aware (it "
        "may already be — check whether the cross-rank edges are "
        "actually being delivered to it);\n"
        "  (c) accept the divergence if the output's downstream use "
        "tolerates rank-local approximation (rare)."
    )

    return (
        "rank_disaggregated_divergence",
        confidence,
        consistency,
        False,
        suspected_gap,
        likely_remedy,
    )


_PATTERN_CLASSIFIERS = [
    _classify_per_system_reduction,
    _classify_replicated_output,
    _classify_rank_disaggregated_divergence,
]
