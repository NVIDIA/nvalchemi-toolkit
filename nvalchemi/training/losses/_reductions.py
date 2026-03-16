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
"""Segmented reduction wrappers around ``nvalchemiops``.

Provides ``torch.library.custom_op`` wrappers for segmented sum/mean
operations, plus composite ``segmented_mse`` and ``segmented_mae``
reductions used by the loss system.

These custom-op wrappers create opaque boundaries that
``torch.compile`` treats as single graph nodes, preventing tracing
into the warp interop code.
"""

from __future__ import annotations

from typing import Literal

import torch
import warp as wp
from nvalchemiops.segment_ops import segmented_mean, segmented_sum

# ---------------------------------------------------------------------------
# Custom ops — segmented sum and mean
# ---------------------------------------------------------------------------


@torch.library.custom_op("nvalchemi_training::segmented_sum", mutates_args=())
def _segmented_sum(
    values: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Segmented sum via nvalchemiops warp kernel."""
    out = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    segmented_sum(
        wp.from_torch(values.contiguous()),
        wp.from_torch(idx.to(torch.int32)),
        wp.from_torch(out),
    )
    return out


@_segmented_sum.register_fake
def _(values: torch.Tensor, idx: torch.Tensor, num_segments: int) -> torch.Tensor:
    return torch.empty(num_segments, device=values.device, dtype=values.dtype)


@torch.library.custom_op("nvalchemi_training::segmented_mean", mutates_args=())
def _segmented_mean(
    values: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    """Segmented mean via nvalchemiops warp kernel."""
    sums = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    counts = torch.zeros(num_segments, device=values.device, dtype=torch.int32)
    out = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    segmented_mean(
        wp.from_torch(values.contiguous()),
        wp.from_torch(idx.to(torch.int32)),
        wp.from_torch(sums),
        wp.from_torch(counts),
        wp.from_torch(out),
    )
    return out


@_segmented_mean.register_fake
def _(values: torch.Tensor, idx: torch.Tensor, num_segments: int) -> torch.Tensor:
    return torch.empty(num_segments, device=values.device, dtype=values.dtype)


# ---------------------------------------------------------------------------
# Composite reductions: segmented MSE / MAE
# ---------------------------------------------------------------------------


def segmented_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch_idx: torch.Tensor,
    num_segments: int,
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    """Per-segment mean squared error.

    Computes ``(pred - target) ** 2`` element-wise, then reduces per
    segment using the segmented sum or mean kernel.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values, shape ``(V,)`` (1-D, flattened if needed).
    target : torch.Tensor
        Target values, same shape as *pred*.
    batch_idx : torch.Tensor
        Integer index mapping each element to its segment, shape ``(V,)``.
    num_segments : int
        Number of segments (graphs) in the batch.
    reduction : {"mean", "sum"}
        Per-segment reduction.  ``"mean"`` divides by the number of
        elements in each segment; ``"sum"`` returns the raw sum.

    Returns
    -------
    torch.Tensor
        Shape ``(num_segments,)`` with per-segment MSE or SSE.
    """
    sq_err = (pred - target).square()
    if reduction == "mean":
        return _segmented_mean(sq_err, batch_idx, num_segments)
    return _segmented_sum(sq_err, batch_idx, num_segments)


def segmented_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    batch_idx: torch.Tensor,
    num_segments: int,
    reduction: Literal["mean", "sum"] = "mean",
) -> torch.Tensor:
    """Per-segment mean absolute error.

    Computes ``|pred - target|`` element-wise, then reduces per segment.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted values, shape ``(V,)``.
    target : torch.Tensor
        Target values, same shape as *pred*.
    batch_idx : torch.Tensor
        Integer index mapping each element to its segment, shape ``(V,)``.
    num_segments : int
        Number of segments (graphs) in the batch.
    reduction : {"mean", "sum"}
        Per-segment reduction.

    Returns
    -------
    torch.Tensor
        Shape ``(num_segments,)`` with per-segment MAE or SAE.
    """
    abs_err = (pred - target).abs()
    if reduction == "mean":
        return _segmented_mean(abs_err, batch_idx, num_segments)
    return _segmented_sum(abs_err, batch_idx, num_segments)
