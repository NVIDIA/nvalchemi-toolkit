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
"""In-place variant of ``batch_neighbor_list_needs_rebuild`` for CUDA graph capture.

The upstream :func:`nvalchemiops.torch.neighbors.rebuild_detection.batch_neighbor_list_needs_rebuild`
function allocates its output tensor dynamically and infers ``num_systems``
via ``batch_idx.max().item()``, which forces a CPU–GPU synchronisation.
Both properties prevent the function from being captured inside a
``torch.cuda.CUDAGraph``.

:func:`batch_neighbor_list_rebuild_inplace` is a drop-in replacement that
accepts a pre-allocated ``rebuild_flags`` tensor and writes results into it
in-place, removing all dynamic allocation and CPU synchronisation from the
hot path.
"""

from __future__ import annotations

import torch
import warp as wp
from nvalchemiops.neighbors.rebuild_detection import check_batch_neighbor_list_rebuild
from nvalchemiops.torch.types import get_wp_dtype, get_wp_mat_dtype, get_wp_vec_dtype


@torch.library.custom_op(
    "nvalchemi::_batch_neighbor_list_rebuild_inplace",
    mutates_args=("reference_positions", "rebuild_flags"),
)
def _batch_neighbor_list_rebuild_inplace(
    reference_positions: torch.Tensor,
    current_positions: torch.Tensor,
    batch_idx: torch.Tensor,
    rebuild_flags: torch.Tensor,
    skin_distance_threshold: float,
    update_reference_positions: bool = False,
    cell: torch.Tensor | None = None,
    cell_inv: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
) -> None:
    """Write per-system rebuild flags into a pre-allocated tensor.

    This is a CUDA-graph-safe variant of
    :func:`~nvalchemiops.torch.neighbors.rebuild_detection.batch_neighbor_list_needs_rebuild`.
    The caller is responsible for allocating ``rebuild_flags`` with shape
    ``(num_systems,)`` and dtype ``torch.bool``.  The tensor is zeroed at
    the start of each call (GPU memset, graph-capturable) so the caller does
    not need to reset it between steps.

    Parameters
    ----------
    reference_positions : torch.Tensor, shape ``(N, 3)``
        Positions when the neighbor list was last built.  Updated in-place
        when ``update_reference_positions=True`` and a rebuild is detected.
    current_positions : torch.Tensor, shape ``(N, 3)``
        Current positions to compare against reference.
    batch_idx : torch.Tensor, shape ``(N,)``, dtype int32
        Per-atom system index.
    rebuild_flags : torch.Tensor, shape ``(num_systems,)``, dtype bool
        **Pre-allocated output buffer.**  Zeroed at the start of each call
        then set to ``True`` for every system where any atom exceeded
        ``skin_distance_threshold``.
    skin_distance_threshold : float
        Maximum allowed per-atom displacement before a rebuild is required.
    update_reference_positions : bool, optional
        If ``True``, overwrite ``reference_positions`` with
        ``current_positions`` in-place when a rebuild is detected.
    cell : torch.Tensor or None, shape ``(num_systems, 3, 3)``, optional
        Per-system cell matrices.  Required together with ``cell_inv`` and
        ``pbc`` for minimum-image-convention displacement.
    cell_inv : torch.Tensor or None, shape ``(num_systems, 3, 3)``, optional
        Inverse cell matrices.
    pbc : torch.Tensor or None, shape ``(num_systems, 3)``, dtype bool, optional
        Per-system periodic boundary condition flags.
    """
    # Zero the output buffer in-place (GPU memset — graph-capturable).
    rebuild_flags.zero_()

    if reference_positions.shape != current_positions.shape:
        # Shape mismatch forces a full rebuild of all systems.
        rebuild_flags.fill_(True)
        return

    total_atoms = reference_positions.shape[0]
    if total_atoms == 0:
        return

    wp_dtype = get_wp_dtype(reference_positions.dtype)
    wp_vec_dtype = get_wp_vec_dtype(reference_positions.dtype)

    wp_reference = wp.from_torch(
        reference_positions, dtype=wp_vec_dtype, return_ctype=True
    )
    wp_current = wp.from_torch(current_positions, dtype=wp_vec_dtype, return_ctype=True)
    wp_batch_idx = wp.from_torch(batch_idx, dtype=wp.int32, return_ctype=True)
    wp_rebuild_flags = wp.from_torch(rebuild_flags, dtype=wp.bool, return_ctype=True)

    wp_cell = wp_cell_inv = wp_pbc = None
    if cell is not None and cell_inv is not None and pbc is not None:
        wp_mat_dtype = get_wp_mat_dtype(reference_positions.dtype)
        wp_cell = wp.from_torch(cell, dtype=wp_mat_dtype, return_ctype=True)
        wp_cell_inv = wp.from_torch(cell_inv, dtype=wp_mat_dtype, return_ctype=True)
        wp_pbc = wp.from_torch(pbc, dtype=wp.bool, return_ctype=True)

    check_batch_neighbor_list_rebuild(
        reference_positions=wp_reference,
        current_positions=wp_current,
        batch_idx=wp_batch_idx,
        skin_distance_threshold=skin_distance_threshold,
        rebuild_flags=wp_rebuild_flags,
        wp_dtype=wp_dtype,
        device=str(reference_positions.device),
        update_reference_positions=update_reference_positions,
        cell=wp_cell,
        cell_inv=wp_cell_inv,
        pbc=wp_pbc,
    )


@_batch_neighbor_list_rebuild_inplace.register_fake
def _batch_neighbor_list_rebuild_inplace_fake(
    reference_positions: torch.Tensor,
    current_positions: torch.Tensor,
    batch_idx: torch.Tensor,
    rebuild_flags: torch.Tensor,
    skin_distance_threshold: float,
    update_reference_positions: bool = False,
    cell: torch.Tensor | None = None,
    cell_inv: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
) -> None:
    """Fake implementation for torch.compile / tracing compatibility."""
    pass


def batch_neighbor_list_rebuild_inplace(
    reference_positions: torch.Tensor,
    current_positions: torch.Tensor,
    batch_idx: torch.Tensor,
    rebuild_flags: torch.Tensor,
    skin_distance_threshold: float,
    update_reference_positions: bool = False,
    cell: torch.Tensor | None = None,
    cell_inv: torch.Tensor | None = None,
    pbc: torch.Tensor | None = None,
) -> None:
    """CUDA-graph-safe per-system neighbor-list rebuild detection.

    Thin public wrapper around the registered custom op.  See
    :func:`_batch_neighbor_list_rebuild_inplace` for full parameter
    documentation.

    The caller must pre-allocate ``rebuild_flags`` as
    ``torch.ones(num_systems, dtype=torch.bool, device=device)`` (ones
    ensures a full rebuild on the very first step before reference
    positions are established).  The buffer is zeroed internally at the
    start of each call before the warp kernel writes fresh values.

    Unlike the upstream ``batch_neighbor_list_needs_rebuild``:

    * No ``batch_idx.max().item()`` — no CPU–GPU synchronisation.
    * No ``torch.zeros`` allocation — ``rebuild_flags`` is mutated
      in-place, keeping GPU addresses stable across CUDA graph replays.
    """
    _batch_neighbor_list_rebuild_inplace(
        reference_positions,
        current_positions,
        batch_idx,
        rebuild_flags,
        skin_distance_threshold,
        update_reference_positions,
        cell,
        cell_inv,
        pbc,
    )
