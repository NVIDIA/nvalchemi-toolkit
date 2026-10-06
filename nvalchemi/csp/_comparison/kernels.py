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
"""Private Warp kernels for exact and threshold radial scoring."""

from __future__ import annotations

from functools import lru_cache

import warp as wp

_TYPE_PAD = -2147483648
_MAX_MISMATCH = 3.402823466e38


@lru_cache(maxsize=3)
def _make_scoring_kernels(
    has_center_types: bool, typed_neighbors: bool
) -> tuple[wp.Kernel, wp.Kernel]:
    """Build kernels specialized for one valid typing mode."""
    if (has_center_types, typed_neighbors) not in {
        (False, False),
        (True, False),
        (True, True),
    }:
        raise ValueError("typed neighbors require center atom types")

    mode = f"center_{int(has_center_types)}_neighbors_{int(typed_neighbors)}"
    HAS_CENTER_TYPES = wp.constant(bool(has_center_types))
    TYPED_NEIGHBORS = wp.constant(bool(typed_neighbors))

    @wp.func
    def row_log_mismatch(
        distances_a: wp.array(dtype=wp.float32),
        types_a: wp.array(dtype=wp.int32),
        row_a: wp.int64,
        width_a: int,
        distances_b: wp.array(dtype=wp.float32),
        types_b: wp.array(dtype=wp.int32),
        row_b: wp.int64,
        width_b: int,
        log_cutoff: float,
    ) -> float:
        """Return the largest log-distance mismatch for one atom-centered row."""
        worst = float(0.0)
        if TYPED_NEIGHBORS:
            i = int(0)
            j = int(0)
            while i < width_a or j < width_b:
                type_a = wp.int32(_TYPE_PAD)
                type_b = wp.int32(_TYPE_PAD)
                if i < width_a:
                    type_a = types_a[row_a + wp.int64(i)]
                if j < width_b:
                    type_b = types_b[row_b + wp.int64(j)]

                if type_a == type_b:
                    if type_a != wp.int32(_TYPE_PAD):
                        mismatch = wp.abs(
                            distances_a[row_a + wp.int64(i)]
                            - distances_b[row_b + wp.int64(j)]
                        )
                        worst = wp.max(worst, mismatch)
                    i += 1
                    j += 1
                elif type_b == wp.int32(_TYPE_PAD) or (
                    type_a != wp.int32(_TYPE_PAD) and type_a < type_b
                ):
                    mismatch = wp.abs(distances_a[row_a + wp.int64(i)] - log_cutoff)
                    worst = wp.max(worst, mismatch)
                    i += 1
                else:
                    mismatch = wp.abs(distances_b[row_b + wp.int64(j)] - log_cutoff)
                    worst = wp.max(worst, mismatch)
                    j += 1
        else:
            width = wp.max(width_a, width_b)
            for k in range(width):
                value_a = log_cutoff
                value_b = log_cutoff
                if k < width_a:
                    value_a = distances_a[row_a + wp.int64(k)]
                if k < width_b:
                    value_b = distances_b[row_b + wp.int64(k)]
                worst = wp.max(worst, wp.abs(value_a - value_b))
        return worst

    @wp.func
    def row_log_mismatch_within_bound(
        distances_a: wp.array(dtype=wp.float32),
        types_a: wp.array(dtype=wp.int32),
        row_a: wp.int64,
        width_a: int,
        distances_b: wp.array(dtype=wp.float32),
        types_b: wp.array(dtype=wp.int32),
        row_b: wp.int64,
        width_b: int,
        log_cutoff: float,
        log_bound: float,
    ) -> float:
        """Bound row mismatch work, returning a sentinel once the limit is exceeded."""
        worst = float(0.0)
        if TYPED_NEIGHBORS:
            i = int(0)
            j = int(0)
            while i < width_a or j < width_b:
                type_a = wp.int32(_TYPE_PAD)
                type_b = wp.int32(_TYPE_PAD)
                if i < width_a:
                    type_a = types_a[row_a + wp.int64(i)]
                if j < width_b:
                    type_b = types_b[row_b + wp.int64(j)]

                mismatch = float(0.0)
                if type_a == type_b:
                    if type_a != wp.int32(_TYPE_PAD):
                        mismatch = wp.abs(
                            distances_a[row_a + wp.int64(i)]
                            - distances_b[row_b + wp.int64(j)]
                        )
                    i += 1
                    j += 1
                elif type_b == wp.int32(_TYPE_PAD) or (
                    type_a != wp.int32(_TYPE_PAD) and type_a < type_b
                ):
                    mismatch = wp.abs(distances_a[row_a + wp.int64(i)] - log_cutoff)
                    i += 1
                else:
                    mismatch = wp.abs(distances_b[row_b + wp.int64(j)] - log_cutoff)
                    j += 1
                worst = wp.max(worst, mismatch)
                if worst > log_bound:
                    return float(_MAX_MISMATCH)
        else:
            width = wp.max(width_a, width_b)
            for k in range(width):
                value_a = log_cutoff
                value_b = log_cutoff
                if k < width_a:
                    value_a = distances_a[row_a + wp.int64(k)]
                if k < width_b:
                    value_b = distances_b[row_b + wp.int64(k)]
                worst = wp.max(worst, wp.abs(value_a - value_b))
                if worst > log_bound:
                    return float(_MAX_MISMATCH)
        return worst

    @wp.func
    def candidate_log_mismatch(
        distances_a: wp.array(dtype=wp.float32),
        types_a: wp.array(dtype=wp.int32),
        centers_a: wp.array(dtype=wp.int32),
        center_a: wp.int64,
        row_a: wp.int64,
        width_a: int,
        distances_b: wp.array(dtype=wp.float32),
        types_b: wp.array(dtype=wp.int32),
        centers_b: wp.array(dtype=wp.int32),
        center_b: wp.int64,
        row_b: wp.int64,
        width_b: int,
        log_cutoff: float,
    ) -> float:
        """Compare compatible center atoms and return their row mismatch."""
        if HAS_CENTER_TYPES:
            if centers_a[center_a] != centers_b[center_b]:
                return float(_MAX_MISMATCH)
        return row_log_mismatch(
            distances_a,
            types_a,
            row_a,
            width_a,
            distances_b,
            types_b,
            row_b,
            width_b,
            log_cutoff,
        )

    @wp.func
    def candidate_log_mismatch_within_bound(
        distances_a: wp.array(dtype=wp.float32),
        types_a: wp.array(dtype=wp.int32),
        centers_a: wp.array(dtype=wp.int32),
        center_a: wp.int64,
        row_a: wp.int64,
        width_a: int,
        distances_b: wp.array(dtype=wp.float32),
        types_b: wp.array(dtype=wp.int32),
        centers_b: wp.array(dtype=wp.int32),
        center_b: wp.int64,
        row_b: wp.int64,
        width_b: int,
        log_cutoff: float,
        log_bound: float,
    ) -> float:
        """Compare compatible centers with early exit above a log mismatch bound."""
        if HAS_CENTER_TYPES:
            if centers_a[center_a] != centers_b[center_b]:
                return float(_MAX_MISMATCH)
        return row_log_mismatch_within_bound(
            distances_a,
            types_a,
            row_a,
            width_a,
            distances_b,
            types_b,
            row_b,
            width_b,
            log_cutoff,
            log_bound,
        )

    @wp.func
    def directed_log_mismatch(
        distances_source: wp.array(dtype=wp.float32),
        types_source: wp.array(dtype=wp.int32),
        centers_source: wp.array(dtype=wp.int32),
        row_start_source: wp.int64,
        atom_start_source: wp.int64,
        width_source: int,
        atom_count_source: int,
        distances_target: wp.array(dtype=wp.float32),
        types_target: wp.array(dtype=wp.int32),
        centers_target: wp.array(dtype=wp.int32),
        row_start_target: wp.int64,
        atom_start_target: wp.int64,
        width_target: int,
        atom_count_target: int,
        log_cutoff: float,
    ) -> float:
        """Return the directed nearest-row mismatch from source to target."""
        directed = float(0.0)
        for atom_source in range(atom_count_source):
            best = float(_MAX_MISMATCH)
            row_source = row_start_source + wp.int64(atom_source) * wp.int64(
                width_source
            )
            center_source = atom_start_source + wp.int64(atom_source)
            for atom_target in range(atom_count_target):
                row_target = row_start_target + wp.int64(atom_target) * wp.int64(
                    width_target
                )
                center_target = atom_start_target + wp.int64(atom_target)
                mismatch = candidate_log_mismatch(
                    distances_source,
                    types_source,
                    centers_source,
                    center_source,
                    row_source,
                    width_source,
                    distances_target,
                    types_target,
                    centers_target,
                    center_target,
                    row_target,
                    width_target,
                    log_cutoff,
                )
                best = wp.min(best, mismatch)
            directed = wp.max(directed, best)
        return directed

    @wp.kernel(module="unique")
    def score_pairs_kernel(
        distances_a: wp.array(dtype=wp.float32),
        types_a: wp.array(dtype=wp.int32),
        distances_b: wp.array(dtype=wp.float32),
        types_b: wp.array(dtype=wp.int32),
        centers_a: wp.array(dtype=wp.int32),
        centers_b: wp.array(dtype=wp.int32),
        row_offsets_a: wp.array(dtype=wp.int64),
        row_offsets_b: wp.array(dtype=wp.int64),
        atom_offsets_a: wp.array(dtype=wp.int64),
        atom_offsets_b: wp.array(dtype=wp.int64),
        widths_a: wp.array(dtype=wp.int32),
        widths_b: wp.array(dtype=wp.int32),
        atom_counts_a: wp.array(dtype=wp.int32),
        atom_counts_b: wp.array(dtype=wp.int32),
        pairs: wp.array2d(dtype=wp.int32),
        log_cutoff: float,
        log_scores: wp.array(dtype=wp.float32),
    ) -> None:
        """Write the symmetric directed nearest-row log mismatch for each pair."""
        pair = wp.tid()
        structure_a = pairs[pair, 0]
        structure_b = pairs[pair, 1]
        width_a = widths_a[structure_a]
        width_b = widths_b[structure_b]
        row_start_a = row_offsets_a[structure_a]
        row_start_b = row_offsets_b[structure_b]
        atom_start_a = atom_offsets_a[structure_a]
        atom_start_b = atom_offsets_b[structure_b]

        directed_ab = directed_log_mismatch(
            distances_a,
            types_a,
            centers_a,
            row_start_a,
            atom_start_a,
            width_a,
            atom_counts_a[structure_a],
            distances_b,
            types_b,
            centers_b,
            row_start_b,
            atom_start_b,
            width_b,
            atom_counts_b[structure_b],
            log_cutoff,
        )
        directed_ba = directed_log_mismatch(
            distances_b,
            types_b,
            centers_b,
            row_start_b,
            atom_start_b,
            width_b,
            atom_counts_b[structure_b],
            distances_a,
            types_a,
            centers_a,
            row_start_a,
            atom_start_a,
            width_a,
            atom_counts_a[structure_a],
            log_cutoff,
        )
        log_scores[pair] = wp.max(directed_ab, directed_ba)

    @wp.kernel(module="unique")
    def threshold_score_pairs_kernel(
        distances_a: wp.array(dtype=wp.float32),
        types_a: wp.array(dtype=wp.int32),
        distances_b: wp.array(dtype=wp.float32),
        types_b: wp.array(dtype=wp.int32),
        centers_a: wp.array(dtype=wp.int32),
        centers_b: wp.array(dtype=wp.int32),
        row_offsets_a: wp.array(dtype=wp.int64),
        row_offsets_b: wp.array(dtype=wp.int64),
        atom_offsets_a: wp.array(dtype=wp.int64),
        atom_offsets_b: wp.array(dtype=wp.int64),
        widths_a: wp.array(dtype=wp.int32),
        widths_b: wp.array(dtype=wp.int32),
        atom_counts_a: wp.array(dtype=wp.int32),
        atom_counts_b: wp.array(dtype=wp.int32),
        pairs: wp.array2d(dtype=wp.int32),
        log_cutoff: float,
        log_bound: float,
        accept_log_bound: float,
        log_scores: wp.array(dtype=wp.float32),
        outcomes: wp.array(dtype=wp.int32),
    ) -> None:
        """Score pair tiles with threshold rejection and certified-match outcomes.

        Outcome 0 rejects and outcome 2 certifies a match; their ``log_scores``
        placeholders are FP32 max/zero values before ``expm1``. Outcome 1
        returns an exact log score for the caller to compare with the threshold.
        """
        pair = wp.tid()
        structure_a = pairs[pair, 0]
        structure_b = pairs[pair, 1]
        width_a = widths_a[structure_a]
        width_b = widths_b[structure_b]
        row_start_a = row_offsets_a[structure_a]
        row_start_b = row_offsets_b[structure_b]
        atom_start_a = atom_offsets_a[structure_a]
        atom_start_b = atom_offsets_b[structure_b]

        # The inward endpoint can certify a match as soon as every center has
        # one compatible row within it. A wider outward endpoint safely
        # rejects a pair only when some center has no compatible row within it.
        accepted_ab = bool(True)
        for atom_a in range(atom_counts_a[structure_a]):
            found = bool(False)
            row_a = row_start_a + wp.int64(atom_a) * wp.int64(width_a)
            center_a = atom_start_a + wp.int64(atom_a)
            for atom_b in range(atom_counts_b[structure_b]):
                row_b = row_start_b + wp.int64(atom_b) * wp.int64(width_b)
                center_b = atom_start_b + wp.int64(atom_b)
                mismatch = candidate_log_mismatch_within_bound(
                    distances_a,
                    types_a,
                    centers_a,
                    center_a,
                    row_a,
                    width_a,
                    distances_b,
                    types_b,
                    centers_b,
                    center_b,
                    row_b,
                    width_b,
                    log_cutoff,
                    accept_log_bound,
                )
                if mismatch <= accept_log_bound:
                    found = True
                    break
            if not found:
                accepted_ab = False
                break

        accepted_ba = bool(True)
        if accepted_ab:
            for atom_b in range(atom_counts_b[structure_b]):
                found = bool(False)
                row_b = row_start_b + wp.int64(atom_b) * wp.int64(width_b)
                center_b = atom_start_b + wp.int64(atom_b)
                for atom_a in range(atom_counts_a[structure_a]):
                    row_a = row_start_a + wp.int64(atom_a) * wp.int64(width_a)
                    center_a = atom_start_a + wp.int64(atom_a)
                    mismatch = candidate_log_mismatch_within_bound(
                        distances_b,
                        types_b,
                        centers_b,
                        center_b,
                        row_b,
                        width_b,
                        distances_a,
                        types_a,
                        centers_a,
                        center_a,
                        row_a,
                        width_a,
                        log_cutoff,
                        accept_log_bound,
                    )
                    if mismatch <= accept_log_bound:
                        found = True
                        break
                if not found:
                    accepted_ba = False
                    break

        if accepted_ab and accepted_ba:
            log_scores[pair] = float(0.0)
            outcomes[pair] = 2
            return

        rejected = bool(False)
        for atom_a in range(atom_counts_a[structure_a]):
            found = bool(False)
            row_a = row_start_a + wp.int64(atom_a) * wp.int64(width_a)
            center_a = atom_start_a + wp.int64(atom_a)
            for atom_b in range(atom_counts_b[structure_b]):
                row_b = row_start_b + wp.int64(atom_b) * wp.int64(width_b)
                center_b = atom_start_b + wp.int64(atom_b)
                mismatch = candidate_log_mismatch_within_bound(
                    distances_a,
                    types_a,
                    centers_a,
                    center_a,
                    row_a,
                    width_a,
                    distances_b,
                    types_b,
                    centers_b,
                    center_b,
                    row_b,
                    width_b,
                    log_cutoff,
                    log_bound,
                )
                if mismatch <= log_bound:
                    found = True
                    break
            if not found:
                rejected = True
                break
        if not rejected:
            for atom_b in range(atom_counts_b[structure_b]):
                found = bool(False)
                row_b = row_start_b + wp.int64(atom_b) * wp.int64(width_b)
                center_b = atom_start_b + wp.int64(atom_b)
                for atom_a in range(atom_counts_a[structure_a]):
                    row_a = row_start_a + wp.int64(atom_a) * wp.int64(width_a)
                    center_a = atom_start_a + wp.int64(atom_a)
                    mismatch = candidate_log_mismatch_within_bound(
                        distances_b,
                        types_b,
                        centers_b,
                        center_b,
                        row_b,
                        width_b,
                        distances_a,
                        types_a,
                        centers_a,
                        center_a,
                        row_a,
                        width_a,
                        log_cutoff,
                        log_bound,
                    )
                    if mismatch <= log_bound:
                        found = True
                        break
                if not found:
                    rejected = True
                    break
        if rejected:
            log_scores[pair] = float(_MAX_MISMATCH)
            outcomes[pair] = 0
            return

        # This pair lies in the conservative endpoint band. Preserve the exact
        # FP32 score and leave the inclusive threshold decision to Torch.
        directed_ab = directed_log_mismatch(
            distances_a,
            types_a,
            centers_a,
            row_start_a,
            atom_start_a,
            width_a,
            atom_counts_a[structure_a],
            distances_b,
            types_b,
            centers_b,
            row_start_b,
            atom_start_b,
            width_b,
            atom_counts_b[structure_b],
            log_cutoff,
        )
        directed_ba = directed_log_mismatch(
            distances_b,
            types_b,
            centers_b,
            row_start_b,
            atom_start_b,
            width_b,
            atom_counts_b[structure_b],
            distances_a,
            types_a,
            centers_a,
            row_start_a,
            atom_start_a,
            width_a,
            atom_counts_a[structure_a],
            log_cutoff,
        )
        log_scores[pair] = wp.max(directed_ab, directed_ba)
        outcomes[pair] = 1

    score_pairs_kernel.__name__ = f"_score_pairs_{mode}"
    threshold_score_pairs_kernel.__name__ = f"_threshold_score_pairs_{mode}"
    return score_pairs_kernel, threshold_score_pairs_kernel


def _get_scoring_kernels(
    has_center_types: bool, typed_neighbors: bool
) -> tuple[wp.Kernel, wp.Kernel]:
    """Return cached kernels for a public comparison typing mode."""
    mode = (bool(has_center_types), bool(typed_neighbors))
    if mode not in {(False, False), (True, False), (True, True)}:
        raise ValueError("typed neighbors require center atom types")
    return _make_scoring_kernels(*mode)
