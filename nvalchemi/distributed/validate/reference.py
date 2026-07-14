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

"""Single-process reference run + neighbor-list summary helpers.

Step 1 of ``trace_and_validate``: run the wrapper single-process and
capture the dispatch trace, watched-helper trace, and a per-atom
neighbor-list summary used downstream by halo-completeness diagnosis.
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from nvalchemi.distributed._core.dispatch_trace import dispatch_trace

__all__ = [
    "_reference_run",
    "_ensure_neighbors",
    "_summarize_neighbor_list",
]


def _reference_run(
    model_factory: Callable[[], Any],
    sample_batch: Any,
    *,
    watched_helper_packages: tuple[str, ...] = (),
    helper_sample_every: int = 8,
    layer_diagnostic: bool = False,
) -> tuple[
    dict[str, torch.Tensor],
    list[dict[str, Any]],
    list,
    dict[str, Any],
    list,
]:
    """Run the wrapper single-process, capture outputs + dispatch
    trace + watched-helper trace + neighbor-list summary +
    (optionally) per-module layer records.

    The neighbor-list summary captures per-atom valid-neighbor counts
    on the global batch — used by ``_check_halo_completeness`` to
    diagnose whether each rank's halo-padded NL covers its owned atoms
    the same way single-process does. **A halo decomposition that drops
    edges is the most common cause of "model output disagrees with
    single-process by a few percent under partial halo coverage" —
    surfacing this upfront beats sending the wrapper author down a
    rabbit hole on consolidation rules.**

    When ``layer_diagnostic`` is true, also installs forward hooks on
    every sub-module of ``wrapper.model`` and returns the recorded
    per-module sums alongside the rest. Workers ship a parallel list;
    the launcher diffs ref vs sum-of-ranks per module to localize the
    first divergent submodule when validation fails.
    """
    from nvalchemi.distributed._core.helper_trace import (  # noqa: PLC0415
        HelperCall,
        helper_trace,
    )
    from nvalchemi.distributed.validate.layer_diagnostics import (  # noqa: PLC0415
        attach_layer_hooks,
        finalize_layer_records,
    )

    helper_records: list[HelperCall] = []
    layer_records: list = []
    with helper_trace(
        watched_helper_packages, sample_every=helper_sample_every
    ) as h_records:
        wrapper = model_factory()
        _ensure_neighbors(sample_batch, wrapper)
        layer_handles: list = []
        if layer_diagnostic and isinstance(wrapper, torch.nn.Module):
            # Hooking the wrapper itself recurses into every nn.Module
            # under it, including nested attribute paths
            # (UMAWrapper.predict_unit.model.module, AIMNet2Wrapper.model,
            # MACEWrapper.model, …) without per-wrapper plumbing.
            layer_handles = attach_layer_hooks(wrapper, layer_records)
        try:
            with dispatch_trace() as records:
                outputs = wrapper(sample_batch)
        finally:
            for h in layer_handles:
                h.remove()
        # Single bulk sync: convert on-device per-module sums to floats.
        if layer_records:
            finalize_layer_records(layer_records)
        helper_records = list(h_records)

    nl_summary = _summarize_neighbor_list(sample_batch)

    captured: dict[str, torch.Tensor] = {
        k: v.detach().clone() for k, v in outputs.items() if isinstance(v, torch.Tensor)
    }
    return captured, list(records), helper_records, nl_summary, layer_records


def _summarize_neighbor_list(batch: Any) -> dict[str, Any]:
    """Capture per-atom valid-neighbor counts from a batch's NL.

    Used by the halo-completeness diagnostic: each rank's halo-padded
    forward should produce the *same* per-atom neighbor count on its
    owned atoms as single-process does on those same atoms. If counts
    differ, halo coverage is missing edges, and any output that
    integrates over edges (energy, stress, force) won't decompose
    cleanly across the partition.

    Handles both NL formats:

    * MATRIX (``batch.neighbor_matrix``, shape ``(N, max_nbrs)``):
      sentinel for unfilled slots is any value ``>= N``.
    * COO (``batch.neighbor_list``, shape ``(E, 2)`` of (src, dst)):
      sentinel for padding edges is ``src >= N`` or ``dst >= N``.

    Returns a dict with ``per_atom_count`` (1-D int64 tensor of shape
    ``(n_atoms,)``) and ``positions`` (cloned, used by the launcher to
    map per-rank owned atoms to their global IDs by position match).
    Returns ``{}`` when no NL is present on the batch.
    """
    n_atoms = int(batch.positions.shape[0])
    positions = batch.positions.detach().cpu().clone()

    nbmat = getattr(batch, "neighbor_matrix", None)
    if nbmat is not None:
        # MATRIX format
        valid = nbmat < n_atoms
        per_atom_count = valid.sum(dim=1).to(torch.int64).detach().cpu()
        return {
            "n_atoms": n_atoms,
            "per_atom_count": per_atom_count,
            "positions": positions,
            "total_valid": int(valid.sum().item()),
            "format": "MATRIX",
        }

    nl = getattr(batch, "neighbor_list", None)
    if nl is not None:
        # COO format: (E, 2). Drop edges that touch padding/sentinel.
        src = nl[:, 0].to(torch.int64)
        dst = nl[:, 1].to(torch.int64)
        valid = (src < n_atoms) & (dst < n_atoms)
        valid_src = src[valid]
        per_atom_count = (
            torch.bincount(valid_src, minlength=n_atoms).to(torch.int64).cpu()
        )
        return {
            "n_atoms": n_atoms,
            "per_atom_count": per_atom_count,
            "positions": positions,
            "total_valid": int(valid.sum().item()),
            "format": "COO",
        }

    return {}


def _ensure_neighbors(batch: Any, wrapper: Any) -> None:
    """Idempotent neighbor-list construction. Skips when the batch
    already has the configured format populated, or when the wrapper
    has no neighbor_config."""
    nc = getattr(wrapper.model_config, "neighbor_config", None)
    if nc is None:
        return
    # Probe for an already-populated NL in the configured format.
    fmt = getattr(nc, "format", None)
    fmt_name = getattr(fmt, "name", str(fmt)) if fmt is not None else ""
    nb_attr = (
        "neighbor_matrix" if fmt_name.upper().startswith("MATRIX") else "neighbor_list"
    )
    if getattr(batch, nb_attr, None) is not None:
        return
    from nvalchemi.neighbors import compute_neighbors  # noqa: PLC0415

    compute_neighbors(batch, config=nc)
