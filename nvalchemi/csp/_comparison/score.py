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
"""Private symmetric batched radial score implementation."""

from __future__ import annotations

import torch
import warp as wp
from nvalchemiops.torch._warp_op_helpers import scoped_warp_stream
from torch import Tensor

from nvalchemi.csp._comparison.kernels import (
    _get_scoring_kernels,
)
from nvalchemi.csp._comparison.store import DescriptorBlock, _log_cutoff_fp32


def threshold_score(threshold: float, device: torch.device) -> Tensor:
    """Convert a score threshold to its FP32 comparison value.

    Parameters
    ----------
    threshold : float
        Fractional score threshold to convert.
    device : torch.device
        Device on which to create the scalar tensor.

    Returns
    -------
    Tensor
        Zero-dimensional FP32 threshold tensor on ``device``.

    Notes
    -----
    The converted value is advanced by one representable FP32 step toward
    positive infinity before threshold scoring.
    """
    raw = torch.tensor(threshold, dtype=torch.float32, device=device)
    return torch.nextafter(
        raw, torch.tensor(float("inf"), dtype=torch.float32, device=device)
    )


def score_descriptor_pairs(
    left: DescriptorBlock,
    right: DescriptorBlock,
    pairs: Tensor,
    cutoff: float,
    has_center_types: bool,
    typed_neighbors: bool,
) -> tuple[Tensor, Tensor]:
    """Score a tile of indexed structure pairs.

    Parameters
    ----------
    left, right : DescriptorBlock
        Ragged descriptors and optional type labels for each side of the
        comparison.
    pairs : Tensor
        Pair indices with shape ``[num_pairs, 2]``. Each row selects one
        structure from ``left`` and one from ``right``.
    cutoff : float
        Neighbor cutoff in the same length unit used to build the descriptors.
    has_center_types : bool
        Whether the selected scoring mode compares center type labels.
    typed_neighbors : bool
        Whether the selected scoring mode compares type-grouped neighbor rows.

    Returns
    -------
    scores : Tensor
        FP32 scores with shape ``[num_pairs]``.
    log_scores : Tensor
        FP32 logarithmic scores used to produce ``scores``.

    Notes
    -----
    One specialized Warp kernel scores the tile on the descriptor device.
    Empty input returns two empty FP32 tensors without launching the kernel.
    """
    if pairs.numel() == 0:
        empty = torch.empty(0, dtype=torch.float32, device=left.distances.device)
        return empty, empty
    device = left.distances.device
    score_kernel, _ = _get_scoring_kernels(has_center_types, typed_neighbors)
    pairs = pairs.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    log_scores = torch.empty((pairs.shape[0],), dtype=torch.float32, device=device)
    with scoped_warp_stream(device):
        wp.launch(
            score_kernel,
            dim=pairs.shape[0],
            inputs=[
                wp.from_torch(left.distances, dtype=wp.float32),
                wp.from_torch(left.neighbor_types, dtype=wp.int32),
                wp.from_torch(right.distances, dtype=wp.float32),
                wp.from_torch(right.neighbor_types, dtype=wp.int32),
                wp.from_torch(left.center_types, dtype=wp.int32),
                wp.from_torch(right.center_types, dtype=wp.int32),
                wp.from_torch(left.row_offsets, dtype=wp.int64),
                wp.from_torch(right.row_offsets, dtype=wp.int64),
                wp.from_torch(left.atom_offsets, dtype=wp.int64),
                wp.from_torch(right.atom_offsets, dtype=wp.int64),
                wp.from_torch(left.widths, dtype=wp.int32),
                wp.from_torch(right.widths, dtype=wp.int32),
                wp.from_torch(left.atom_counts, dtype=wp.int32),
                wp.from_torch(right.atom_counts, dtype=wp.int32),
                wp.from_torch(pairs, dtype=wp.int32),
                _log_cutoff_fp32(cutoff),
                wp.from_torch(log_scores, dtype=wp.float32),
            ],
            device=str(device),
        )
    return torch.expm1(log_scores), log_scores


def threshold_score_descriptor_pairs(
    left: DescriptorBlock,
    right: DescriptorBlock,
    pairs: Tensor,
    cutoff: float,
    log_bound: float,
    accept_log_bound: float,
    has_center_types: bool,
    typed_neighbors: bool,
) -> tuple[Tensor, Tensor]:
    """Classify indexed pairs, scoring only cases inside the endpoint bounds.

    Parameters
    ----------
    left, right : DescriptorBlock
        Ragged descriptors and optional type labels for each side of the
        comparison.
    pairs : Tensor
        Pair indices with shape ``[num_pairs, 2]``.
    cutoff : float
        Neighbor cutoff in the same length unit used to build the descriptors.
    log_bound : float
        Outward logarithmic threshold used for conservative rejection.
    accept_log_bound : float
        Inward logarithmic threshold used to certify a match.
    has_center_types : bool
        Whether the selected scoring mode compares center type labels.
    typed_neighbors : bool
        Whether the selected scoring mode compares type-grouped neighbor rows.

    Returns
    -------
    scores : Tensor
        FP32 scores with shape ``[num_pairs]``. Outcome 1 contains an exact
        score; outcome 0 uses infinity and outcome 2 uses zero as placeholders.
    outcomes : Tensor
        Int32 outcome codes with shape ``[num_pairs]``: 0 rejects, 1 requires
        the caller to compare the exact score with its threshold, and 2
        certifies a match.

    Notes
    -----
    A specialized Warp kernel classifies the pairs on the descriptor device.
    Empty input returns empty tensors without launching the kernel.
    """
    if pairs.numel() == 0:
        empty = torch.empty(0, dtype=torch.float32, device=left.distances.device)
        return empty, torch.empty(0, dtype=torch.int32, device=left.distances.device)
    device = left.distances.device
    _, threshold_kernel = _get_scoring_kernels(has_center_types, typed_neighbors)
    pairs = pairs.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
    log_scores = torch.empty((pairs.shape[0],), dtype=torch.float32, device=device)
    outcomes = torch.empty((pairs.shape[0],), dtype=torch.int32, device=device)
    with scoped_warp_stream(device):
        wp.launch(
            threshold_kernel,
            dim=pairs.shape[0],
            inputs=[
                wp.from_torch(left.distances, dtype=wp.float32),
                wp.from_torch(left.neighbor_types, dtype=wp.int32),
                wp.from_torch(right.distances, dtype=wp.float32),
                wp.from_torch(right.neighbor_types, dtype=wp.int32),
                wp.from_torch(left.center_types, dtype=wp.int32),
                wp.from_torch(right.center_types, dtype=wp.int32),
                wp.from_torch(left.row_offsets, dtype=wp.int64),
                wp.from_torch(right.row_offsets, dtype=wp.int64),
                wp.from_torch(left.atom_offsets, dtype=wp.int64),
                wp.from_torch(right.atom_offsets, dtype=wp.int64),
                wp.from_torch(left.widths, dtype=wp.int32),
                wp.from_torch(right.widths, dtype=wp.int32),
                wp.from_torch(left.atom_counts, dtype=wp.int32),
                wp.from_torch(right.atom_counts, dtype=wp.int32),
                wp.from_torch(pairs, dtype=wp.int32),
                _log_cutoff_fp32(cutoff),
                log_bound,
                accept_log_bound,
                wp.from_torch(log_scores, dtype=wp.float32),
                wp.from_torch(outcomes, dtype=wp.int32),
            ],
            device=str(device),
        )
    return torch.expm1(log_scores), outcomes
