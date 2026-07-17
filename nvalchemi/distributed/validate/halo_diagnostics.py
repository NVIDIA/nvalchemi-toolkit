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

"""Halo-coverage completeness diagnostic.

When a halo-storage wrapper's multi-rank output disagrees with
single-process by a few percent, the most common root cause is missing
edges in each rank's halo-padded neighbor list (i.e. the halo construction
isn't covering all the atoms the rank's owned atoms should see). These
helpers capture per-rank halo NL summaries and cross-reference them
against the single-process NL summary; the verdict is surfaced in the
``TraceReport.next_action`` so the user sees the root cause before
chasing combine-rule symptoms.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["_capture_halo_summary", "_check_halo_completeness"]


def _capture_halo_summary(spec: Any, sharded: Any) -> dict[str, Any]:
    """Per-rank summary used by the halo-completeness diagnostic.

    Returns ``{}`` for non-halo-storage specs (no padded batch). For
    halo storage, returns each owned atom's valid-neighbor count along
    with the owned positions — the launcher matches positions to
    single-process atom IDs to verify halo coverage on a per-atom
    basis.

    Handles both NL formats. For MATRIX, valid slots are
    ``nbmat[:n_owned] < n_padded``. For COO, owned edges are those
    where ``src < n_owned`` and ``dst < n_padded`` (halo atoms are
    in the upper portion of the index range; the sentinel is the
    padded total ``n_padded``).
    """
    from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy

    policy = getattr(getattr(spec, "distribution", None), "policy", None)
    if not isinstance(policy, HaloStoragePolicy):
        return {}
    padded_batch = getattr(sharded, "padded_batch", None)
    halo_meta = getattr(sharded, "halo_meta", None)
    if padded_batch is None or halo_meta is None:
        return {}
    n_owned = int(halo_meta.n_owned)
    owned_positions = padded_batch.positions[:n_owned].detach().cpu().clone()

    nbmat = getattr(padded_batch, "neighbor_matrix", None)
    if nbmat is not None:
        n_padded = int(nbmat.shape[0])
        valid = nbmat[:n_owned] < n_padded
        per_owned_count = valid.sum(dim=1).to(torch.int64).detach().cpu()
        return {
            "n_owned": n_owned,
            "n_padded": n_padded,
            "per_owned_count": per_owned_count,
            "owned_positions": owned_positions,
            "total_owned_valid": int(valid.sum().item()),
            "format": "MATRIX",
        }

    nl = getattr(padded_batch, "neighbor_list", None)
    if nl is not None:
        n_padded = int(padded_batch.positions.shape[0])
        src = nl[:, 0].to(torch.int64)
        dst = nl[:, 1].to(torch.int64)
        valid = (src < n_owned) & (dst < n_padded)
        valid_src = src[valid]
        per_owned_count = (
            torch.bincount(valid_src, minlength=n_owned).to(torch.int64).cpu()
        )
        return {
            "n_owned": n_owned,
            "n_padded": n_padded,
            "per_owned_count": per_owned_count,
            "owned_positions": owned_positions,
            "total_owned_valid": int(valid.sum().item()),
            "format": "COO",
        }

    return {}


def _check_halo_completeness(
    ref_nl_summary: dict[str, Any],
    per_rank_halo_summaries: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    """Cross-reference ref's per-atom NL counts vs each rank's
    per-owned NL counts. Returns a verdict dict, or ``None`` when the
    check doesn't apply (non-halo spec, no NL recorded).

    Each rank's owned atoms are matched to ref atoms by position
    (positions are unique 32-bit triplets in a typical MD batch — no
    realistic collision). For each match, compare ref's neighbor
    count vs rank's neighbor count. Per-rank totals across all owned
    atoms summed across ranks should equal ref's global total. Any
    mismatch is *the* most likely root cause when output diffs are a
    few percent and other diagnoses don't fit.
    """
    if not ref_nl_summary or not per_rank_halo_summaries:
        return None
    ref_pos = ref_nl_summary.get("positions")
    ref_count = ref_nl_summary.get("per_atom_count")
    ref_total = ref_nl_summary.get("total_valid")
    if ref_pos is None or ref_count is None or ref_total is None:
        return None

    # Position → ref-index lookup. Stack as row-major tuples and use
    # a dict; cheaper than nearest-neighbor for the small N we're
    # validating.
    ref_pos_tuples = {
        tuple(round(float(v), 5) for v in row): i for i, row in enumerate(ref_pos)
    }

    per_rank_observed: dict[int, int] = {}
    per_rank_expected: dict[int, int] = {}
    per_rank_mismatches: list[str] = []
    for rank, summary in sorted(per_rank_halo_summaries.items()):
        if not summary:
            continue
        owned_positions = summary.get("owned_positions")
        per_owned_count = summary.get("per_owned_count")
        if owned_positions is None or per_owned_count is None:
            continue
        observed_total = int(summary.get("total_owned_valid", 0))
        expected_total = 0
        atom_mismatches = 0
        for j, row in enumerate(owned_positions):
            key = tuple(round(float(v), 5) for v in row)
            ref_idx = ref_pos_tuples.get(key)
            if ref_idx is None:
                continue
            expected = int(ref_count[ref_idx].item())
            actual = int(per_owned_count[j].item())
            expected_total += expected
            if expected != actual:
                atom_mismatches += 1
        per_rank_observed[rank] = observed_total
        per_rank_expected[rank] = expected_total
        if atom_mismatches:
            per_rank_mismatches.append(
                f"rank{rank}: {atom_mismatches} owned atom(s) saw a "
                f"different neighbor count than single-process"
            )

    # No rank contributed halo data — the check doesn't apply (sharded-storage
    # spec uses global gather, not a halo-padded NL; or no NL was recorded).
    # Return None rather than fabricating a "0 edges observed" mismatch.
    if not per_rank_observed:
        return None

    rank_total_observed = sum(per_rank_observed.values())
    rank_total_expected = sum(per_rank_expected.values())
    matches = (
        rank_total_observed == rank_total_expected
        and rank_total_observed == ref_total
        and not per_rank_mismatches
    )

    verdict = {
        "matches": matches,
        "ref_total_valid_edges": ref_total,
        "rank_total_observed": rank_total_observed,
        "rank_total_expected_from_owned": rank_total_expected,
        "per_rank_observed": per_rank_observed,
        "per_rank_expected": per_rank_expected,
        "atom_level_mismatches": per_rank_mismatches,
    }
    if not matches:
        verdict["interpretation"] = (
            f"halo coverage is INCOMPLETE: across all ranks, owned atoms "
            f"see a total of {rank_total_observed} edges in the halo-padded "
            f"NL, but single-process counts {rank_total_expected} edges for "
            f"those same atoms (global total {ref_total}). "
            f"Each rank's halo-padded forward is missing edges that would "
            f"be visible in single-process — its computation of any "
            f"edge-integrating quantity (energy, stress, forces) for owned "
            f"atoms differs from single-process. **This is the root cause "
            f"of per-system output disagreement under partial halo coverage.** "
            f"Spec-level fixes (consolidation rules, all_reduce_outputs) "
            f"can't recover the missing edges; the halo construction must "
            f"include them."
        )
    return verdict


def _partition_health(
    ref_nl_summary: dict[str, Any],
    per_rank_halo_summaries: dict[int, dict[str, Any]],
) -> dict[str, Any] | None:
    """Report each rank's owned / halo / remote atom composition and flag a
    DEGENERATE partition — one where domain decomposition isn't meaningfully
    exercised, so a passing validation isn't evidence the spec is correct.

    For a halo rank holding ``n_padded = n_owned + n_halo`` rows out of
    ``n_global`` total atoms, ``n_remote = n_global - n_padded`` is the count
    of atoms it neither owns nor borrows. A meaningful test wants all three
    non-trivial on every rank:

    * ``n_halo == 0`` → no cross-rank dependency (the halo path never fires).
    * ``n_remote == 0`` → the rank sees every atom (owned + halo == global), so
      partition geometry is trivial and can't catch remote-atom bugs.
    * ``n_owned == 0`` → empty shard.

    Returns a verdict dict (``per_rank`` composition + ``degenerate`` warnings
    + ``healthy`` bool), or ``None`` when inapplicable (no halo summaries, or
    the global count is unknown).
    """
    if not per_rank_halo_summaries:
        return None
    ref_count = (ref_nl_summary or {}).get("per_atom_count")
    n_global = int(len(ref_count)) if ref_count is not None else None
    if n_global is None:
        return None

    per_rank: dict[int, dict[str, int]] = {}
    degenerate: list[str] = []
    for rank, summary in sorted(per_rank_halo_summaries.items()):
        if not summary:
            continue
        n_owned = int(summary["n_owned"])
        n_padded = int(summary["n_padded"])
        n_halo = n_padded - n_owned
        n_remote = n_global - n_padded
        per_rank[rank] = {"owned": n_owned, "halo": n_halo, "remote": n_remote}
        if n_owned == 0:
            degenerate.append(f"rank{rank}: 0 owned atoms (empty shard)")
        elif n_halo == 0:
            degenerate.append(
                f"rank{rank}: 0 halo atoms — no cross-rank dependency, the "
                f"halo path is never exercised"
            )
        elif n_remote <= 0:
            degenerate.append(
                f"rank{rank}: 0 remote atoms — sees all {n_global} atoms "
                f"(owned+halo), so partition geometry is trivial (full "
                f"coverage); use a larger system or more ranks"
            )

    if not per_rank:
        return None
    return {
        "n_global": n_global,
        "per_rank": per_rank,
        "degenerate": degenerate,
        "healthy": not degenerate,
    }
