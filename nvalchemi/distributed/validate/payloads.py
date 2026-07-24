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

"""Wire-format helpers for the validator's spawn boundary.

* :func:`_batch_to_payload` / :func:`_payload_to_batch` — pickle-friendly
  Batch ↔ tensor-dict conversion. Full :class:`Batch` pickling across
  spawned procs is fragile (custom ``__torch_function__``, lazy fields,
  neighbour-list artefacts) so we ship raw tensor fields and rebuild on
  the worker side.
* :func:`_extract_cutoff` — DomainConfig cutoff extraction for the worker.
* :func:`_diff_outputs` — partition-invariant diff metric for the
  reference-vs-multi-rank comparison.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "_batch_to_payload",
    "_payload_to_batch",
    "_extract_cutoff",
    "_diff_outputs",
]


_BATCH_FIELDS_TO_SHIP: tuple[str, ...] = (
    # Geometry
    "positions",
    "atomic_numbers",
    "atomic_masses",
    "cell",
    "pbc",
    # Per-atom physical quantities (electrostatics models, charge eq.)
    "charges",
    "charge",
    # AIMNet2 / UMA: spin multiplicity (per-system) and tags (per-atom)
    "spin",
    "mult",
    "tags",
)
"""Per-batch tensor fields shipped from launcher to worker.

Each must be a constructor kwarg of :class:`AtomicData` so that
:func:`_payload_to_batch` can rebuild the batch via
``AtomicData(**fields)``. Neighbour-list artefacts (``neighbor_matrix``
etc.) deliberately aren't shipped — the worker rebuilds them with
``_ensure_neighbors`` from the wrapper's ``neighbor_config``,
keeping the wire payload minimal and avoiding the AtomicData-vs-Batch
attachment dance for derived state.
"""


def _batch_to_payload(batch: Any) -> dict[str, Any]:
    """Serialize a :class:`Batch` into a process-portable dict.

    Full :class:`Batch` pickling across spawned procs is fragile
    (custom ``__torch_function__``, lazy fields, neighbour-list
    artefacts) — instead we ship the raw tensor fields and reconstruct
    via :func:`_payload_to_batch` on the worker. Field list lives in
    :data:`_BATCH_FIELDS_TO_SHIP`; extend it when adding wrapper
    families that need new per-batch tensors.
    """
    fields: dict[str, Any] = {}
    for key in _BATCH_FIELDS_TO_SHIP:
        v = getattr(batch, key, None)
        if isinstance(v, torch.Tensor):
            fields[key] = v.detach().cpu().clone()
    return fields


def _payload_to_batch(payload: dict[str, Any], device: str) -> Any:
    """Inverse of :func:`_batch_to_payload`. Reconstruct an
    :class:`AtomicData` + :class:`Batch` on the worker."""
    from nvalchemi.data import AtomicData, Batch  # noqa: PLC0415

    kwargs: dict[str, Any] = {}
    for k, v in payload.items():
        kwargs[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    data = AtomicData(**kwargs)
    return Batch.from_data_list([data], device=device)


def _extract_cutoff(wrapper: Any) -> float:
    """Best-effort cutoff extraction from a wrapper for DomainConfig."""
    nc = getattr(wrapper.model_config, "neighbor_config", None)
    if nc is not None and getattr(nc, "cutoff", None) is not None:
        return float(nc.cutoff)
    return float(getattr(wrapper, "cutoff", 5.0))


def _lexsort_rows(t: torch.Tensor) -> torch.Tensor:
    """Sort the rows of ``t`` lexicographically by their full feature vector.

    Permutation-invariant over the first (atom) dimension while keeping each
    row's vector intact — so two outputs match only if the *set of per-atom
    vectors* agrees, not merely the multiset of scalar values. 1-D inputs
    (per-graph scalars) reduce to a plain value sort.
    """
    if t.dim() <= 1:
        return t.sort().values
    flat = t.reshape(t.shape[0], -1)
    order = torch.arange(flat.shape[0], device=t.device)
    # Stable sort by each column, last column first → lexicographic by row.
    for col in range(flat.shape[1] - 1, -1, -1):
        order = order[torch.argsort(flat[order, col], stable=True)]
    return t[order]


def _diff_outputs(
    ref: dict[str, torch.Tensor], got: dict[str, torch.Tensor]
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-output diff between single-process reference and the
    rank-concatenated multi-rank output.

    The multi-rank value is the concat of every rank's owned slice in
    rank order — same total length as the reference for per-atom
    outputs, but the per-atom *order* may differ when the partitioner
    interleaves atoms across ranks. So shape-equal isn't sufficient
    grounds for element-wise comparison.

    Strategy: report ``min(elem_diff, agg_diff)`` so spec-correct
    outputs pass regardless of partition order.

    1. ``elem_diff`` — element-wise ``(ref - got).abs().max()``. Exact
       when global atom order matches (e.g. ``contiguous_block``
       partition + rank-order concat); meaningless under spatial
       partitioning.
    2. ``agg_diff`` — element-wise diff after **sorting the flattened
       tensors**. Permutation-invariant *and* wrongness-sensitive: a
       systematic per-atom error shifts the sorted distribution, so
       this catches correctness bugs that the previous
       ``max(net, peak_magnitude)`` aggregate hid (forces sum to ~=0
       on both sides by Newton's 3rd regardless of correctness; peak
       magnitude is a single scalar that often agrees by chance).
    """
    abs_diff: dict[str, float] = {}
    rel_diff: dict[str, float] = {}
    for k, v in ref.items():
        gv = got.get(k)
        if gv is None:
            abs_diff[k] = float("inf")
            rel_diff[k] = float("inf")
            continue
        v64 = v.detach().cpu().to(torch.float64)
        gv64 = gv.detach().cpu().to(torch.float64)

        if v64.shape != gv64.shape:
            # Trailing dims differ or first-dim mismatch we can't
            # reconcile (rank slice without concat) — irrecoverable.
            if v64.shape[1:] != gv64.shape[1:]:
                abs_diff[k] = float("inf")
                rel_diff[k] = float("inf")
                continue
            elem_diff = float("inf")
        else:
            elem_diff = float((v64 - gv64).abs().max().item())

        # Permutation-invariant compare that PRESERVES the vector dimensions:
        # sort whole rows lexicographically rather than flattening. Flattening
        # mixed atoms and components together, so a force with values on the
        # wrong axis (or wrong atom) could compare equal; row-sorting keeps each
        # atom's vector intact, so a scrambled component no longer matches.
        # (Full per-atom-ID matching — the strongest check — additionally needs a
        # stable atom id threaded through the gather; tracked as a follow-up.)
        if v64.shape[1:] != gv64.shape[1:]:
            agg_diff = float("inf")
        else:
            ref_sorted = _lexsort_rows(v64)
            got_sorted = _lexsort_rows(gv64)
            if ref_sorted.shape != got_sorted.shape:
                agg_diff = float("inf")
            else:
                agg_diff = float((ref_sorted - got_sorted).abs().max().item())

        worst = min(elem_diff, agg_diff)
        abs_diff[k] = float(worst)
        denom = max(v64.abs().max().item(), 1e-30)
        rel_diff[k] = float(worst) / denom
    return abs_diff, rel_diff
