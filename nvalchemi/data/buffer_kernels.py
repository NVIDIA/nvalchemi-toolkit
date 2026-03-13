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
"""
Warp buffer kernels: system-level and segmented copy-masked and coalesce.

This module provides GPU-friendly (Warp) routines to copy selected rows or segments
into a destination buffer and to coalesce (compact) the source in place by removing
copied items. No host synchronization is used in the copy path; coalesce may
perform one sync when the caller trims containers.

Layouts
-------
- **System-level**: One row per system; tensors have shape ``(num_systems, cols)``.
  Use :func:`put_masked_per_system` and :func:`defrag_per_system`.
- **Segmented**: Flat data with segment boundaries in ``batch_ptr`` of shape
  ``(num_systems + 1)``; segment ``i`` is ``data[batch_ptr[i] : batch_ptr[i+1], :]``.
  Use :func:`put_masked_segmented` and :func:`defrag_segmented`.

Kernels are fused where possible to reduce launch overhead. Prefix sums use
``wp.utils.array_scan(..., inclusive=False)``. All code runs on CPU and CUDA.
"""

from __future__ import annotations

from typing import Any

import torch
import warp as wp

wp.config.quiet = True
wp.init()

TORCH_TO_WP: dict[torch.dtype, type] = {
    torch.bool: wp.bool,
    torch.float32: wp.float32,
    torch.float64: wp.float64,
    torch.int32: wp.int32,
    torch.int64: wp.int64,
}

# =============================================================================
# System-level: copy masked and coalesce (one row per system)
# =============================================================================


@wp.kernel
def _count_empty_dest_kernel(
    dest_mask: wp.array(dtype=wp.bool),
    out_count: wp.array(dtype=wp.int32),
) -> None:
    """Count empty dest slots (dest_mask[i] False) via atomic add into out_count[0].

    Parameters
    ----------
    dest_mask : wp.array(dtype=wp.bool), shape (num_dest,)
        True where the dest slot is occupied.
    out_count : wp.array(dtype=wp.int32), shape (1,)
        Output: total number of indices i with dest_mask[i] == False.
        Must be zero-initialized before launch.

    Notes
    -----
    Launch: dim = num_dest.
    """
    i = wp.tid()
    inc = 1 if not dest_mask[i] else 0
    wp.atomic_add(out_count, 0, inc)


@wp.kernel
def _count_masked_and_max_copy_kernel(
    source_scan: wp.array(dtype=wp.int32),
    source_mask: wp.array(dtype=wp.bool),
    num_src: wp.int32,
    count_empty: wp.array(dtype=wp.int32),
    count_masked: wp.array(dtype=wp.int32),
    max_copy: wp.array(dtype=wp.int32),
) -> None:
    """Compute total masked count and max_copy = min(count_masked, count_empty).

    Reads exclusive scan last element and count_empty (from a prior _count_empty_dest_kernel launch).

    Parameters
    ----------
    source_scan : wp.array(dtype=wp.int32), shape (num_src,)
        Exclusive scan of source mask (int).
    source_mask : wp.array(dtype=wp.bool), shape (num_src,)
        True for rows to copy.
    num_src : wp.int32
        Length of source.
    count_empty : wp.array(dtype=wp.int32), shape (1,)
        Input: number of empty dest slots (written by _count_empty_dest_kernel).
    count_masked : wp.array(dtype=wp.int32), shape (1,)
        Output: total number of True in source_mask.
    max_copy : wp.array(dtype=wp.int32), shape (1,)
        Output: min(count_masked, count_empty).

    Notes
    -----
    Launch: dim = 1.
    """
    last_scan = source_scan[num_src - 1]
    last_masked = 1 if source_mask[num_src - 1] else 0
    count_masked[0] = last_scan + last_masked
    max_copy[0] = wp.min(count_masked[0], count_empty[0])


@wp.kernel
def _first_empty_slot_kernel(
    dest_mask: wp.array(dtype=wp.bool),
    out_index: wp.array(dtype=wp.int32),
) -> None:
    """Find the first index where dest_mask is False via atomic_min.

    Parameters
    ----------
    dest_mask : wp.array(dtype=wp.bool), shape (num_dest,)
        True where the dest slot is occupied.
    out_index : wp.array(dtype=wp.int32), shape (1,)
        Input: initialize to num_dest. Output: minimum i with dest_mask[i] == False.

    Notes
    -----
    Launch: dim = num_dest.
    """
    i = wp.tid()
    if not dest_mask[i]:
        wp.atomic_min(out_index, 0, i)


@wp.kernel
def _copy_masked_per_system_kernel(
    source: wp.array2d(dtype=Any),
    source_mask: wp.array(dtype=wp.bool),
    source_scan: wp.array(dtype=wp.int32),
    dest: wp.array2d(dtype=Any),
    dest_mask: wp.array(dtype=wp.bool),
    first_empty_slot: wp.array(dtype=wp.int32),
    source_data_copied: wp.array(dtype=wp.bool),
    max_copy: wp.array(dtype=wp.int32),
) -> None:
    """Copy up to max_copy[0] masked source rows into dest at first empty slot(s).

    Parameters
    ----------
    source : wp.array2d, shape (num_src, cols)
        Source data.
    source_mask : wp.array(dtype=wp.bool), shape (num_src,)
        True for rows to copy.
    source_scan : wp.array(dtype=wp.int32), shape (num_src,)
        Exclusive scan of source mask (int); gives per-row dest offset.
    dest : wp.array2d, shape (num_dest, cols)
        Destination buffer; written at first_empty_slot + source_scan[i].
    dest_mask : wp.array(dtype=wp.bool), shape (num_dest,)
        Output: set True for each dest row written.
    first_empty_slot : wp.array(dtype=wp.int32), shape (1,)
        First dest index to use (from _first_empty_slot_kernel).
    source_data_copied : wp.array(dtype=wp.bool), shape (num_src,)
        Output: True where the row was copied.
    max_copy : wp.array(dtype=wp.int32), shape (1,)
        Maximum number of rows to copy (min of masked count and empty count).

    Notes
    -----
    Launch: dim = num_src.
    """
    i = wp.tid()
    if not source_mask[i] or source_scan[i] >= max_copy[0]:
        return
    dest_idx = first_empty_slot[0] + source_scan[i]
    for c in range(source.shape[1]):
        dest[dest_idx, c] = type(dest[dest_idx, c])(source[i, c])
    dest_mask[dest_idx] = True
    source_data_copied[i] = True


@wp.kernel
def _coalesce_scatter_kernel(
    source: wp.array2d(dtype=Any),
    source_data_copied: wp.array(dtype=wp.bool),
    kept_rank: wp.array(dtype=wp.int32),
    temp: wp.array2d(dtype=Any),
) -> None:
    """Scatter rows where source_data_copied[i] is False into temp at kept_rank[i].

    Parameters
    ----------
    source : wp.array2d, shape (num_src, cols)
        Source data (read-only).
    source_data_copied : wp.array(dtype=wp.bool), shape (num_src,)
        True = row was copied (skip); False = keep.
    kept_rank : wp.array(dtype=wp.int32), shape (num_src,)
        Exclusive scan of (1 - source_data_copied); output index per kept row.
    temp : wp.array2d, shape (num_src, cols)
        Output: compacted rows.

    Notes
    -----
    Launch: dim = num_src.
    """
    i = wp.tid()
    if source_data_copied[i]:
        return
    out_idx = kept_rank[i]
    for c in range(source.shape[1]):
        temp[out_idx, c] = type(temp[out_idx, c])(source[i, c])


@wp.kernel
def _coalesce_copy_back_zero_and_update_mask_kernel(
    source: wp.array2d(dtype=Any),
    temp: wp.array2d(dtype=Any),
    num_kept: wp.array(dtype=wp.int32),
    source_data_copied: wp.array(dtype=wp.bool),
) -> None:
    """Copy temp[:num_kept] to source, zero source[num_kept:], and set source_data_copied[i] = (i >= num_kept).

    Fused to avoid a separate update_mask kernel launch.

    Parameters
    ----------
    source : wp.array2d, shape (num_src, cols)
        Output: compacted data then zeros.
    temp : wp.array2d, shape (num_src, cols)
        Input: scattered kept rows.
    num_kept : wp.array(dtype=wp.int32), shape (1,)
        Number of kept rows.
    source_data_copied : wp.array(dtype=wp.bool), shape (num_src,)
        Output: True for i >= num_kept (logically "copied" / to be dropped next time).

    Notes
    -----
    Launch: dim = num_src.
    """
    i = wp.tid()
    nk = num_kept[0]
    for c in range(source.shape[1]):
        source[i, c] = (
            type(source[i, c])(temp[i, c]) if i < nk else type(source[i, c])(0.0)
        )
    source_data_copied[i] = i >= nk


@wp.kernel
def _mask_to_int_kernel(
    mask: wp.array(dtype=wp.bool), out: wp.array(dtype=wp.int32)
) -> None:
    """Convert bool mask to int32 for scan: out[i] = 1 if mask[i] else 0.

    Parameters
    ----------
    mask : wp.array(dtype=wp.bool)
        Input mask.
    out : wp.array(dtype=wp.int32)
        Output; same length as mask.

    Notes
    -----
    Launch: dim = len(mask).
    """
    i = wp.tid()
    out[i] = wp.int32(1) if mask[i] else wp.int32(0)


@wp.kernel
def _inv_mask_to_int_kernel(
    mask: wp.array(dtype=wp.bool), out: wp.array(dtype=wp.int32)
) -> None:
    """Inverse mask for 'kept': out[i] = 0 if mask[i] else 1 (kept = not copied).

    Parameters
    ----------
    mask : wp.array(dtype=wp.bool)
        True = copied (drop); False = kept.
    out : wp.array(dtype=wp.int32)
        Output; 1 where kept, 0 where copied.

    Notes
    -----
    Launch: dim = len(mask).
    """
    i = wp.tid()
    out[i] = wp.int32(0) if mask[i] else wp.int32(1)


@wp.kernel
def _num_kept_from_scan_kernel(
    kept_rank: wp.array(dtype=wp.int32),
    source_data_copied: wp.array(dtype=wp.bool),
    n: wp.int32,
    num_kept_out: wp.array(dtype=wp.int32),
) -> None:
    """Compute total number of kept rows from exclusive scan and last mask bit.

    Parameters
    ----------
    kept_rank : wp.array(dtype=wp.int32), shape (n,)
        Exclusive scan of kept-int (1 where not copied).
    source_data_copied : wp.array(dtype=wp.bool), shape (n,)
        True = copied.
    n : wp.int32
        Length.
    num_kept_out : wp.array(dtype=wp.int32), shape (1,)
        Output: kept_rank[n-1] + (1 if not source_data_copied[n-1] else 0).

    Notes
    -----
    Launch: dim = 1.
    """
    last_rank = kept_rank[n - 1]
    last_kept = 0 if source_data_copied[n - 1] else 1
    num_kept_out[0] = last_rank + last_kept


########################################################
# Overloading kernels for different data types
########################################################
_copy_masked_per_system_overloads: dict[type, Any] = {}
_coalesce_scatter_overloads: dict[type, Any] = {}
_coalesce_copy_back_zero_and_update_mask_overloads: dict[type, Any] = {}
for _wp_t in [wp.bool, wp.float32, wp.float64, wp.int32, wp.int64]:
    _copy_masked_per_system_overloads[_wp_t] = wp.overload(
        _copy_masked_per_system_kernel,
        [
            wp.array2d(dtype=_wp_t),
            wp.array(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            wp.array2d(dtype=_wp_t),
            wp.array(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.bool),
            wp.array(dtype=wp.int32),
        ],
    )
    _coalesce_scatter_overloads[_wp_t] = wp.overload(
        _coalesce_scatter_kernel,
        [
            wp.array2d(dtype=_wp_t),
            wp.array(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            wp.array2d(dtype=_wp_t),
        ],
    )
    _coalesce_copy_back_zero_and_update_mask_overloads[_wp_t] = wp.overload(
        _coalesce_copy_back_zero_and_update_mask_kernel,
        [
            wp.array2d(dtype=_wp_t),
            wp.array2d(dtype=_wp_t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.bool),
        ],
    )


# =============================================================================
# Segmented: put masked and defrag (batch_ptr layout)
# =============================================================================


@wp.kernel
def _masked_segment_lengths_kernel(
    batch_ptr: wp.array(dtype=wp.int32),
    mask: wp.array(dtype=wp.bool),
    out: wp.array(dtype=wp.int32),
    num_systems: wp.int32,
) -> None:
    """Segment length where mask is True, else zero: out[i] = (batch_ptr[i+1]-batch_ptr[i]) if mask[i] else 0.

    Parameters
    ----------
    batch_ptr : wp.array(dtype=wp.int32), shape (num_systems + 1,)
        Cumulative segment boundaries.
    mask : wp.array(dtype=wp.bool), shape (num_systems,)
        True = include this segment's length.
    out : wp.array(dtype=wp.int32), shape (num_systems,)
        Output: segment length or 0.
    num_systems : wp.int32
        Number of segments.

    Notes
    -----
    Launch: dim = num_systems.
    """
    i = wp.tid()
    if i < num_systems:
        out[i] = (batch_ptr[i + 1] - batch_ptr[i]) if mask[i] else wp.int32(0)


@wp.kernel
def _add_scalar_to_int_kernel(
    scalar: wp.int32,
    arr: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
) -> None:
    """out[0] = scalar + arr[0]. Launch: dim=1."""
    out[0] = scalar + arr[0]


@wp.kernel
def _num_copied_from_mask_scan_kernel(
    mask_scan: wp.array(dtype=wp.int32),
    source_mask: wp.array(dtype=wp.bool),
    num_systems: wp.int32,
    num_copied_out: wp.array(dtype=wp.int32),
) -> None:
    """Compute number of segments with source_mask True from exclusive scan of mask.

    Parameters
    ----------
    mask_scan : wp.array(dtype=wp.int32), shape (num_systems,)
        Exclusive scan of source_mask as int.
    source_mask : wp.array(dtype=wp.bool), shape (num_systems,)
        True = copy this segment.
    num_systems : wp.int32
        Number of segments.
    num_copied_out : wp.array(dtype=wp.int32), shape (1,)
        Output: mask_scan[num_systems-1] + (1 if source_mask[num_systems-1] else 0).

    Notes
    -----
    Launch: dim = 1.
    """
    n = num_systems
    last_scan = mask_scan[n - 1]
    last_masked = 1 if source_mask[n - 1] else 0
    num_copied_out[0] = last_scan + last_masked


@wp.kernel
def _set_new_num_dest_segmented_kernel(
    num_dest_segments: wp.int32,
    fit_scan: wp.array(dtype=wp.int32),
    fits: wp.array(dtype=wp.int32),
    num_systems: wp.int32,
    new_num_dest: wp.array(dtype=wp.int32),
) -> None:
    """new_num_dest[0] = num_dest_segments + fit_scan[num_systems-1] + fits[num_systems-1]. Launch: dim=1."""
    n = num_systems
    new_num_dest[0] = num_dest_segments + fit_scan[n - 1] + fits[n - 1]


@wp.kernel
def _compute_segmented_fits_kernel(
    dest_batch_ptr: wp.array(dtype=wp.int32),
    num_dest_segments: wp.int32,
    copy_dest_offset: wp.array(dtype=wp.int32),
    masked_lengths: wp.array(dtype=wp.int32),
    source_mask: wp.array(dtype=wp.bool),
    dest_capacity: wp.int32,
    fits_out: wp.array(dtype=wp.int32),
    num_systems: wp.int32,
) -> None:
    """fits_out[i] = 1 if segment i is masked and fits in dest, else 0. Launch: dim = num_systems."""
    i = wp.tid()
    if not source_mask[i]:
        fits_out[i] = 0
        return
    base = dest_batch_ptr[num_dest_segments]
    end = base + copy_dest_offset[i] + masked_lengths[i]
    fits_out[i] = 1 if end <= dest_capacity else 0


@wp.kernel
def _compute_segmented_fit_mask_kernel(
    dest_batch_ptr: wp.array(dtype=wp.int32),
    num_dest_segments: wp.int32,
    copy_dest_offset: wp.array(dtype=wp.int32),
    masked_lengths: wp.array(dtype=wp.int32),
    source_mask: wp.array(dtype=wp.bool),
    dest_capacity: wp.int32,
    fit_mask: wp.array(dtype=wp.bool),
    num_systems: wp.int32,
) -> None:
    """Write fit_mask[i] = True iff segment i is masked and fits in dest. Launch: dim = num_systems."""
    i = wp.tid()
    if not source_mask[i]:
        fit_mask[i] = False
        return
    base = dest_batch_ptr[num_dest_segments]
    end = base + copy_dest_offset[i] + masked_lengths[i]
    fit_mask[i] = end <= dest_capacity


@torch.library.custom_op(
    "nvalchemi::compute_put_fit_mask_segmented",
    mutates_args=("fit_mask",),
)
def compute_put_fit_mask_segmented_impl(
    source_batch_ptr: torch.Tensor,
    source_mask: torch.Tensor,
    dest_batch_ptr: torch.Tensor,
    num_dest_segments: int,
    dest_capacity: int,
    fit_mask: torch.Tensor,
) -> None:
    """Kernel path: write fit_mask in place for segmented put (call only when room)."""
    num_systems = source_batch_ptr.shape[0] - 1
    device = str(source_batch_ptr.device)
    wp_source_batch_ptr = wp.from_torch(
        source_batch_ptr.to(torch.int32), dtype=wp.int32, return_ctype=True
    )
    wp_source_mask = wp.from_torch(source_mask, dtype=wp.bool, return_ctype=True)
    wp_dest_batch_ptr = wp.from_torch(
        dest_batch_ptr.to(torch.int32), dtype=wp.int32, return_ctype=True
    )
    wp_fit_mask = wp.from_torch(fit_mask, dtype=wp.bool, return_ctype=True)

    masked_lengths = torch.empty(
        num_systems, device=source_batch_ptr.device, dtype=torch.int32
    )
    wp_masked_lengths = wp.from_torch(masked_lengths, dtype=wp.int32)
    wp.launch(
        _masked_segment_lengths_kernel,
        dim=num_systems,
        inputs=[
            wp_source_batch_ptr,
            wp_source_mask,
            wp_masked_lengths,
            num_systems,
        ],
        device=device,
    )
    copy_dest_offset = torch.empty(
        num_systems, device=source_batch_ptr.device, dtype=torch.int32
    )
    wp_copy_dest_offset = wp.from_torch(copy_dest_offset, dtype=wp.int32)
    wp.utils.array_scan(wp_masked_lengths, wp_copy_dest_offset, inclusive=False)

    wp.launch(
        _compute_segmented_fit_mask_kernel,
        dim=num_systems,
        inputs=[
            wp_dest_batch_ptr,
            num_dest_segments,
            wp_copy_dest_offset,
            wp_masked_lengths,
            wp_source_mask,
            dest_capacity,
            wp_fit_mask,
            num_systems,
        ],
        device=device,
    )


def compute_put_fit_mask_segmented(
    source_batch_ptr: torch.Tensor,
    source_mask: torch.Tensor,
    dest_batch_ptr: torch.Tensor,
    num_dest_segments: int,
    dest_capacity: int,
    fit_mask: torch.Tensor,
) -> None:
    """Compute per-system fit mask for segmented put; write result into fit_mask in place.

    For each segment i, fit_mask[i] is set True iff source_mask[i] is True and
    that segment fits in the destination (data capacity and batch_ptr capacity).
    No data is copied. Used with put_masked_segmented so the caller can combine
    fit masks across levels (e.g. logical_and) before calling put.

    Parameters
    ----------
    source_batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype int32
        Cumulative segment boundaries in source; segment i is
        [source_batch_ptr[i], source_batch_ptr[i+1]).
    source_mask : torch.Tensor, shape (num_systems,), dtype bool
        True for each segment considered for copy.
    dest_batch_ptr : torch.Tensor, shape (num_dest_segments + 1,) or larger, dtype int32
        Destination segment boundaries; next free index is dest_batch_ptr[num_dest_segments].
    num_dest_segments : int
        Current number of segments in dest.
    dest_capacity : int
        Destination data buffer row capacity (e.g. dest.shape[0]).
    fit_mask : torch.Tensor, shape (num_systems,), dtype bool
        Output written in place: True where the segment fits in dest.

    Notes
    -----
    If dest_batch_ptr has fewer than num_dest_segments + num_systems + 2 elements,
    no new segment boundaries can be appended; fit_mask is zeroed (all False).
    All tensors must be on the same device. Launch uses Warp; no host sync.
    """
    num_systems = source_batch_ptr.shape[0] - 1
    if num_systems == 0:
        return
    min_batch_size = num_dest_segments + num_systems + 2
    if dest_batch_ptr.shape[0] < min_batch_size:
        fit_mask.zero_()
        return
    compute_put_fit_mask_segmented_impl(
        source_batch_ptr,
        source_mask,
        dest_batch_ptr,
        num_dest_segments,
        dest_capacity,
        fit_mask,
    )


@wp.kernel
def _compute_uniform_fit_mask_kernel(
    source_mask: wp.array(dtype=wp.bool),
    source_scan: wp.array(dtype=wp.int32),
    max_copy: wp.array(dtype=wp.int32),
    fit_mask: wp.array(dtype=wp.bool),
) -> None:
    """fit_mask[i] = source_mask[i] and (source_scan[i] < max_copy[0]). Launch: dim = len(source_mask)."""
    i = wp.tid()
    fit_mask[i] = source_mask[i] and (source_scan[i] < max_copy[0])


@wp.kernel
def _copy_masked_segmented_kernel(
    source: wp.array2d(dtype=Any),
    source_batch_ptr: wp.array(dtype=wp.int32),
    source_mask: wp.array(dtype=wp.bool),
    copy_dest_offset: wp.array(dtype=wp.int32),
    masked_lengths: wp.array(dtype=wp.int32),
    dest: wp.array2d(dtype=Any),
    dest_batch_ptr: wp.array(dtype=wp.int32),
    source_data_copied: wp.array(dtype=wp.bool),
    num_systems: wp.int32,
    num_dest_segments: wp.int32,
    fit_scan: wp.array(dtype=wp.int32),
    new_num_dest: wp.array(dtype=wp.int32),
    fits: wp.array(dtype=wp.int32),
) -> None:
    """Copy only masked segments that fit; parallel over num_systems. Skip if not fits[i]."""
    i = wp.tid()
    base = dest_batch_ptr[num_dest_segments]
    n_fit = new_num_dest[0] - num_dest_segments
    if not (source_mask[i] and fits[i]):
        return
    start = source_batch_ptr[i]
    end = source_batch_ptr[i + 1]
    dest_start = base + copy_dest_offset[i]
    cols = source.shape[1]
    dd = dest[0, 0]
    for row in range(start, end):
        for c in range(cols):
            dest[dest_start + (row - start), c] = type(dd)(source[row, c])
    source_data_copied[i] = True
    my_end = base + copy_dest_offset[i] + masked_lengths[i]
    dest_batch_ptr[num_dest_segments + 1 + fit_scan[i]] = my_end
    if fit_scan[i] == n_fit - 1:
        dest_batch_ptr[num_dest_segments + 1 + n_fit] = my_end


@wp.kernel
def _coalesce_segmented_scatter_kernel(
    source: wp.array2d(dtype=Any),
    source_batch_ptr: wp.array(dtype=wp.int32),
    source_data_copied: wp.array(dtype=wp.bool),
    kept_rank: wp.array(dtype=wp.int32),
    new_batch_ptr: wp.array(dtype=wp.int32),
    temp: wp.array2d(dtype=Any),
    num_systems: wp.int32,
) -> None:
    """Scatter kept segments into temp at offsets given by new_batch_ptr.

    For each segment i with source_data_copied[i] False, copy segment i to
    temp starting at new_batch_ptr[kept_rank[i]].

    Parameters
    ----------
    source : wp.array2d, shape (total_elems, cols)
        Source data.
    source_batch_ptr : wp.array(dtype=wp.int32), shape (num_systems + 1,)
        Segment boundaries.
    source_data_copied : wp.array(dtype=wp.bool), shape (num_systems,)
        True = segment was copied (skip); False = keep.
    kept_rank : wp.array(dtype=wp.int32), shape (num_systems,)
        Exclusive scan of kept; output segment index for each kept segment.
    new_batch_ptr : wp.array(dtype=wp.int32), shape (num_kept + 1,) or larger
        Start index in temp for each kept segment.
    temp : wp.array2d, shape (total_elems, cols)
        Output buffer.
    num_systems : wp.int32
        Number of segments.

    Notes
    -----
    Launch: dim = num_systems.
    """
    i = wp.tid()
    if source_data_copied[i]:
        return
    start, end = source_batch_ptr[i], source_batch_ptr[i + 1]
    out_start = new_batch_ptr[kept_rank[i]]
    dd = temp[0, 0]
    for row in range(start, end):
        for c in range(source.shape[1]):
            temp[out_start + (row - start), c] = type(dd)(source[row, c])


@wp.kernel
def _scatter_kept_lengths_kernel(
    source_batch_ptr: wp.array(dtype=wp.int32),
    source_data_copied: wp.array(dtype=wp.bool),
    kept_rank: wp.array(dtype=wp.int32),
    kept_lengths: wp.array(dtype=wp.int32),
    num_systems: wp.int32,
) -> None:
    """Write segment length for each kept segment into kept_lengths at kept_rank[i].

    Parameters
    ----------
    source_batch_ptr : wp.array(dtype=wp.int32), shape (num_systems + 1,)
        Segment boundaries.
    source_data_copied : wp.array(dtype=wp.bool), shape (num_systems,)
        True = skip; only threads with False write.
    kept_rank : wp.array(dtype=wp.int32), shape (num_systems,)
        Output index for each kept segment (0..num_kept-1).
    kept_lengths : wp.array(dtype=wp.int32), shape (num_systems,) or (num_kept,)
        Output: length of each kept segment; zero-initialize before launch.
    num_systems : wp.int32
        Number of segments.

    Notes
    -----
    Launch: dim = num_systems.
    """
    i = wp.tid()
    if i < num_systems and not source_data_copied[i]:
        kept_lengths[kept_rank[i]] = source_batch_ptr[i + 1] - source_batch_ptr[i]


@wp.kernel
def _sum_prefix_kernel(
    arr: wp.array(dtype=wp.int32),
    n: wp.array(dtype=wp.int32),
    out: wp.array(dtype=wp.int32),
) -> None:
    """Sum the first n[0] elements of arr into out[0].

    Parameters
    ----------
    arr : wp.array(dtype=wp.int32)
        Input array.
    n : wp.array(dtype=wp.int32), shape (1,)
        Number of elements to sum.
    out : wp.array(dtype=wp.int32), shape (1,)
        Output: sum(arr[0:n[0]]).

    Notes
    -----
    Launch: dim = 1.
    """
    total = wp.int32(0)
    for i in range(n[0]):
        total = total + arr[i]
    out[0] = total


@wp.kernel
def _num_kept_segmented_kernel(
    kept_rank: wp.array(dtype=wp.int32),
    source_data_copied: wp.array(dtype=wp.bool),
    num_systems: wp.int32,
    num_kept_out: wp.array(dtype=wp.int32),
) -> None:
    """Compute number of kept segments from scan and last mask bit.

    Parameters
    ----------
    kept_rank : wp.array(dtype=wp.int32), shape (num_systems,)
        Exclusive scan of kept-int.
    source_data_copied : wp.array(dtype=wp.bool), shape (num_systems,)
        True = copied.
    num_systems : wp.int32
        Number of segments.
    num_kept_out : wp.array(dtype=wp.int32), shape (1,)
        Output: kept_rank[num_systems-1] + (1 if not source_data_copied[num_systems-1] else 0).

    Notes
    -----
    Launch: dim = 1.
    """
    n = num_systems
    last_rank = kept_rank[n - 1]
    last_kept = 0 if source_data_copied[n - 1] else 1
    num_kept_out[0] = last_rank + last_kept


@wp.kernel
def _build_new_batch_ptr_kernel(
    scan: wp.array(dtype=wp.int32),
    num_kept: wp.array(dtype=wp.int32),
    total_kept_elems: wp.array(dtype=wp.int32),
    new_batch_ptr: wp.array(dtype=wp.int32),
    num_systems: wp.int32,
) -> None:
    """Build new_batch_ptr from exclusive scan of kept segment lengths.

    new_batch_ptr[0..num_kept-1] = scan[0..num_kept-1], new_batch_ptr[num_kept] = total_kept_elems.

    Parameters
    ----------
    scan : wp.array(dtype=wp.int32), shape (num_systems,)
        Exclusive scan of kept_lengths.
    num_kept : wp.array(dtype=wp.int32), shape (1,)
        Number of kept segments.
    total_kept_elems : wp.array(dtype=wp.int32), shape (1,)
        Total elements in kept segments.
    new_batch_ptr : wp.array(dtype=wp.int32), shape (num_systems + 1,) or larger
        Output: segment start indices for compacted layout.
    num_systems : wp.int32
        Number of segments.

    Notes
    -----
    Launch: dim = num_systems + 1.
    """
    i = wp.tid()
    nk = num_kept[0]
    if i <= nk:
        new_batch_ptr[i] = total_kept_elems[0] if i == nk else scan[i]


@wp.kernel
def _write_zero_kernel(out: wp.array(dtype=wp.int32)) -> None:
    """Write zero to out[0].

    Parameters
    ----------
    out : wp.array(dtype=wp.int32), shape at least (1,)
        Output; out[0] = 0.

    Notes
    -----
    Launch: dim = 1.
    """
    out[0] = 0


@wp.kernel
def _copy_batch_ptr_prefix_kernel(
    num_kept: wp.array(dtype=wp.int32),
    src: wp.array(dtype=wp.int32),
    dest: wp.array(dtype=wp.int32),
) -> None:
    """Copy src[0 : num_kept[0]+1] into dest in place. No host sync.

    Launch: dim = at least num_kept[0] + 1 (e.g. num_systems + 1).
    """
    i = wp.tid()
    n = num_kept[0] + 1
    if i < n:
        dest[i] = src[i]


@wp.kernel
def _fill_batch_ptr_tail_kernel(
    dest: wp.array(dtype=wp.int32),
    num_kept: wp.array(dtype=wp.int32),
    total_kept_elems: wp.array(dtype=wp.int32),
    num_systems: wp.int32,
) -> None:
    """Fill dest[num_kept[0]+1 : num_systems+1] with total_kept_elems[0].

    Caller can then read total_kept_elems as dest[-1] without host sync.
    Launch: dim = num_systems + 1.
    """
    i = wp.tid()
    nk = num_kept[0]
    if i > nk:
        dest[i] = total_kept_elems[0]


@wp.kernel
def _coalesce_segmented_copy_back_from_total_kernel(
    source: wp.array2d(dtype=Any),
    temp: wp.array2d(dtype=Any),
    total_kept: wp.array(dtype=wp.int32),
) -> None:
    """Copy temp[0:total_kept[0]] into source and zero source[total_kept[0]:].

    Parameters
    ----------
    source : wp.array2d, shape (total_elems, cols)
        Output: compacted data then zeros.
    temp : wp.array2d, shape (total_elems, cols)
        Input: scattered kept data.
    total_kept : wp.array(dtype=wp.int32), shape (1,)
        Number of elements to copy.

    Notes
    -----
    Launch: dim = total_elems.
    """
    i = wp.tid()
    nk = total_kept[0]
    for c in range(source.shape[1]):
        source[i, c] = (
            type(source[0, 0])(temp[i, c]) if i < nk else type(source[0, 0])(0.0)
        )


########################################################
# Overloading kernels for different data types
########################################################
_copy_masked_segmented_overloads: dict[type, Any] = {}
_coalesce_segmented_scatter_overloads: dict[type, Any] = {}
_coalesce_segmented_copy_back_from_total_overloads: dict[type, Any] = {}
for _wp_t in [wp.bool, wp.float32, wp.float64, wp.int32, wp.int64]:
    _copy_masked_segmented_overloads[_wp_t] = wp.overload(
        _copy_masked_segmented_kernel,
        [
            wp.array2d(dtype=_wp_t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array2d(dtype=_wp_t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.bool),
            wp.int32,
            wp.int32,
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
        ],
    )
    _coalesce_segmented_scatter_overloads[_wp_t] = wp.overload(
        _coalesce_segmented_scatter_kernel,
        [
            wp.array2d(dtype=_wp_t),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.bool),
            wp.array(dtype=wp.int32),
            wp.array(dtype=wp.int32),
            wp.array2d(dtype=_wp_t),
            wp.int32,
        ],
    )
    _coalesce_segmented_copy_back_from_total_overloads[_wp_t] = wp.overload(
        _coalesce_segmented_copy_back_from_total_kernel,
        [
            wp.array2d(dtype=_wp_t),
            wp.array2d(dtype=_wp_t),
            wp.array(dtype=wp.int32),
        ],
    )

########################################################


@torch.library.custom_op(
    "nvalchemi::put_masked_per_system",
    mutates_args=(
        "source",
        "source_mask",
        "dest",
        "dest_mask",
        "source_data_copied",
    ),
)
def put_masked_per_system_impl(
    source: torch.Tensor,
    source_mask: torch.Tensor,
    dest: torch.Tensor,
    dest_mask: torch.Tensor,
    source_data_copied: torch.Tensor,
) -> None:
    """Copy masked source rows into dest at the first empty slot(s).

    Rows with source_mask[i] True are copied into dest starting at the first
    index where dest_mask is False. Only as many rows as fit in dest's empty
    slots are copied. All counting and limit computation is done in kernels.
    Supports bool, float32, float64, int32, int64 2D tensors.
    """
    device = str(source.device)
    source_dtype = source.dtype
    wp_dtype = TORCH_TO_WP[source_dtype]

    num_src = source.shape[0]
    num_dest = dest.shape[0]
    if num_src == 0:
        return

    # Create warp arrays
    wp_source = wp.from_torch(source, dtype=wp_dtype, return_ctype=True)
    wp_source_mask = wp.from_torch(source_mask, dtype=wp.bool, return_ctype=True)
    wp_dest = wp.from_torch(dest, dtype=wp_dtype, return_ctype=True)
    wp_dest_mask = wp.from_torch(dest_mask, dtype=wp.bool, return_ctype=True)
    wp_source_data_copied = wp.from_torch(
        source_data_copied, dtype=wp.bool, return_ctype=True
    )

    mask_int = torch.empty(num_src, device=device, dtype=torch.int32)
    wp_mask_int = wp.from_torch(mask_int, dtype=wp.int32)
    source_scan = torch.empty(num_src, device=device, dtype=torch.int32)
    wp_source_scan = wp.from_torch(source_scan, dtype=wp.int32)

    count_0 = torch.zeros(1, device=device, dtype=torch.int32)
    wp_count_0 = wp.from_torch(count_0, dtype=wp.int32, return_ctype=True)
    count_1 = torch.zeros(1, device=device, dtype=torch.int32)
    wp_count_1 = wp.from_torch(count_1, dtype=wp.int32, return_ctype=True)
    count_2 = torch.zeros(1, device=device, dtype=torch.int32)
    wp_count_2 = wp.from_torch(count_2, dtype=wp.int32, return_ctype=True)

    first_empty_slot = torch.full((1,), num_dest, device=device, dtype=torch.int32)
    wp_first_empty_slot = wp.from_torch(
        first_empty_slot, dtype=wp.int32, return_ctype=True
    )

    # Launch kernels
    wp.launch(
        _mask_to_int_kernel,
        dim=num_src,
        inputs=[wp_source_mask, wp_mask_int],
        device=device,
    )
    wp.utils.array_scan(
        wp_mask_int,
        wp_source_scan,
        inclusive=False,
    )
    wp.launch(
        _count_empty_dest_kernel,
        dim=num_dest,
        inputs=[
            wp_dest_mask,
            wp_count_1,
        ],
        device=device,
    )
    wp.launch(
        _count_masked_and_max_copy_kernel,
        dim=1,
        inputs=[
            wp_source_scan,
            wp_source_mask,
            num_src,
            wp_count_1,
            wp_count_0,
            wp_count_2,
        ],
        device=device,
    )

    wp.launch(
        _first_empty_slot_kernel,
        dim=num_dest,
        inputs=[wp_dest_mask, wp_first_empty_slot],
        device=device,
    )
    wp.launch(
        _copy_masked_per_system_overloads[wp_dtype],
        dim=num_src,
        inputs=[
            wp_source,
            wp_source_mask,
            wp_source_scan,
            wp_dest,
            wp_dest_mask,
            wp_first_empty_slot,
            wp_source_data_copied,
            wp_count_2,
        ],
        device=device,
    )


@torch.library.custom_op(
    "nvalchemi::compute_put_fit_mask_per_system",
    mutates_args=("fit_mask",),
)
def compute_put_fit_mask_per_system_impl(
    source_mask: torch.Tensor,
    dest_mask: torch.Tensor,
    fit_mask: torch.Tensor,
) -> None:
    """Kernel path: write fit_mask in place for uniform put (call when num_src > 0)."""
    num_src = source_mask.shape[0]
    num_dest = dest_mask.shape[0]
    device = str(source_mask.device)
    wp_source_mask = wp.from_torch(source_mask, dtype=wp.bool, return_ctype=True)
    wp_dest_mask = wp.from_torch(dest_mask, dtype=wp.bool, return_ctype=True)
    wp_fit_mask = wp.from_torch(fit_mask, dtype=wp.bool, return_ctype=True)

    mask_int = torch.empty(num_src, device=source_mask.device, dtype=torch.int32)
    wp_mask_int = wp.from_torch(mask_int, dtype=wp.int32)
    source_scan = torch.empty(num_src, device=source_mask.device, dtype=torch.int32)
    wp_source_scan = wp.from_torch(source_scan, dtype=wp.int32)
    count_empty = torch.zeros(1, device=source_mask.device, dtype=torch.int32)
    wp_count_empty = wp.from_torch(count_empty, dtype=wp.int32, return_ctype=True)
    count_masked = torch.zeros(1, device=source_mask.device, dtype=torch.int32)
    wp_count_masked = wp.from_torch(count_masked, dtype=wp.int32, return_ctype=True)
    max_copy = torch.zeros(1, device=source_mask.device, dtype=torch.int32)
    wp_max_copy = wp.from_torch(max_copy, dtype=wp.int32, return_ctype=True)

    wp.launch(
        _mask_to_int_kernel,
        dim=num_src,
        inputs=[wp_source_mask, wp_mask_int],
        device=device,
    )
    wp.utils.array_scan(wp_mask_int, wp_source_scan, inclusive=False)
    wp.launch(
        _count_empty_dest_kernel,
        dim=num_dest,
        inputs=[wp_dest_mask, wp_count_empty],
        device=device,
    )
    wp.launch(
        _count_masked_and_max_copy_kernel,
        dim=1,
        inputs=[
            wp_source_scan,
            wp_source_mask,
            num_src,
            wp_count_empty,
            wp_count_masked,
            wp_max_copy,
        ],
        device=device,
    )
    wp.launch(
        _compute_uniform_fit_mask_kernel,
        dim=num_src,
        inputs=[wp_source_mask, wp_source_scan, wp_max_copy, wp_fit_mask],
        device=device,
    )


def compute_put_fit_mask_per_system(
    source_mask: torch.Tensor,
    dest_mask: torch.Tensor,
    fit_mask: torch.Tensor,
) -> None:
    """Compute per-system fit mask for uniform put; write result into fit_mask in place.

    For each row i, fit_mask[i] is set True iff source_mask[i] is True and that row
    would be copied by put_masked_per_system (i.e. it is among the first min(num_masked,
    num_empty) masked rows). No data is copied. Used so the caller can combine fit
    masks across levels (e.g. logical_and) before calling put.

    Parameters
    ----------
    source_mask : torch.Tensor, shape (num_src,), dtype bool
        True for each source row considered for copy.
    dest_mask : torch.Tensor, shape (num_dest,), dtype bool
        True where the destination slot is occupied; False = empty.
    fit_mask : torch.Tensor, shape (num_src,), dtype bool
        Output written in place: True where the row would fit in dest's empty slots.

    Notes
    -----
    Same empty-slot and max_copy logic as put_masked_per_system. All tensors must
    be on the same device. Launch uses Warp; no host sync.
    """
    num_src = source_mask.shape[0]
    if num_src == 0:
        return
    compute_put_fit_mask_per_system_impl(source_mask, dest_mask, fit_mask)


def put_masked_per_system(
    source: torch.Tensor,
    source_mask: torch.Tensor,
    dest: torch.Tensor,
    dest_mask: torch.Tensor,
    source_data_copied: torch.Tensor,
) -> None:
    """Copy masked source rows into dest at the first empty slot(s).

    Rows with source_mask[i] True are copied into dest starting at the first
    index where dest_mask is False. Only as many rows as fit in dest's empty
    slots are copied. All counting and limit computation is done in kernels.
    Supports bool, float32, float64, int32, int64 2D tensors.

    Parameters
    ----------
    source : torch.Tensor, shape (num_src, cols) or (num_src, ...)
        Source data; non-2D is viewed as (num_src, -1).
    source_mask : torch.Tensor, shape (num_src,), dtype bool
        True for each row to copy.
    dest : torch.Tensor, shape (num_dest, cols) or (num_dest, ...)
        Destination buffer; same dtype as source.
    dest_mask : torch.Tensor, shape (num_dest,), dtype bool
        True where dest slot is occupied; updated in place for written slots.
    source_data_copied : torch.Tensor, shape (num_src,), dtype bool
        Output: True for each row that was actually copied.

    Returns
    -------
    None

    Notes
    -----
    If num_src == 0, returns immediately without launching kernels.
    """
    if source.dim() != 2:
        source = source.view(source.shape[0], -1)
    if dest.dim() != 2:
        dest = dest.view(dest.shape[0], -1)
    return put_masked_per_system_impl(
        source, source_mask, dest, dest_mask, source_data_copied
    )


@torch.library.custom_op(
    "nvalchemi::defrag_per_system",
    mutates_args=(
        "source",
        "source_data_copied",
    ),
)
def defrag_per_system_impl(
    source: torch.Tensor,
    source_data_copied: torch.Tensor,
) -> None:
    """Compact source in-place by removing rows where source_data_copied[i] is True."""
    source_dtype = source.dtype
    wp_dtype = TORCH_TO_WP[source_dtype]

    if source.dim() != 2:
        source = source.view(source.shape[0], -1)
    num_src = source.shape[0]
    device = str(source.device)

    # Create warp arrays
    wp_source = wp.from_torch(source, dtype=wp_dtype, return_ctype=True)
    wp_source_data_copied = wp.from_torch(
        source_data_copied, dtype=wp.bool, return_ctype=True
    )
    kept_int = torch.empty(num_src, device=device, dtype=torch.int32)
    wp_kept_int = wp.from_torch(kept_int, dtype=wp.int32)

    kept_rank = torch.empty(num_src, device=device, dtype=torch.int32)
    wp_kept_rank = wp.from_torch(kept_rank, dtype=wp.int32)

    num_kept = torch.empty(1, device=device, dtype=torch.int32)
    wp_num_kept = wp.from_torch(num_kept, dtype=wp.int32)

    temp = torch.empty_like(source, device=device)
    wp_temp = wp.from_torch(temp, dtype=wp_dtype)

    wp.launch(
        _inv_mask_to_int_kernel,
        dim=num_src,
        inputs=[wp_source_data_copied, wp_kept_int],
        device=device,
    )
    wp.utils.array_scan(
        wp_kept_int,
        wp_kept_rank,
        inclusive=False,
    )
    wp.launch(
        _num_kept_from_scan_kernel,
        dim=1,
        inputs=[wp_kept_rank, wp_source_data_copied, num_src, wp_num_kept],
        device=device,
    )

    wp.launch(
        _coalesce_scatter_overloads[wp_dtype],
        dim=num_src,
        inputs=[
            wp_source,
            wp_source_data_copied,
            wp_kept_rank,
            wp_temp,
        ],
        device=device,
    )
    wp.launch(
        _coalesce_copy_back_zero_and_update_mask_overloads[wp_dtype],
        dim=num_src,
        inputs=[
            wp_source,
            wp_temp,
            wp_num_kept,
            wp_source_data_copied,
        ],
        device=device,
    )


def defrag_per_system(
    source: torch.Tensor,
    source_data_copied: torch.Tensor,
) -> None:
    """Compact source in-place by removing rows where source_data_copied[i] is True.

    Rows with source_data_copied[i] False (kept) are moved to the front;
    the rest are zeroed. source_data_copied is updated so that after the call,
    source_data_copied[i] is True for i >= num_kept. Supports float32, float64,
    int32, int64. No host sync inside this routine.

    Parameters
    ----------
    source : torch.Tensor, shape (num_src, cols) or (num_src, ...)
        Data to compact; modified in place. Non-2D is viewed as (num_src, -1).
    source_data_copied : torch.Tensor, shape (num_src,), dtype bool
        True = row was copied (drop); False = keep. Updated in place.

    Returns
    -------
    None

    """
    return defrag_per_system_impl(source, source_data_copied)


########################################################


@torch.library.custom_op(
    "nvalchemi::put_masked_segmented",
    mutates_args=(
        "source",
        "source_batch_ptr",
        "source_mask",
        "dest",
        "dest_batch_ptr",
        "source_data_copied",
        "new_num_dest",
    ),
)
def put_masked_segmented_impl(
    source: torch.Tensor,
    source_batch_ptr: torch.Tensor,
    source_mask: torch.Tensor,
    dest: torch.Tensor,
    dest_batch_ptr: torch.Tensor,
    num_dest_segments: int,
    source_data_copied: torch.Tensor,
    new_num_dest: torch.Tensor,
) -> None:
    """Copy masked segments from source into dest and append segment boundaries to dest_batch_ptr."""
    source_dtype = source.dtype
    device = str(source.device)
    wp_dtype = TORCH_TO_WP[source_dtype]

    num_systems = source_batch_ptr.shape[0] - 1
    if num_systems == 0:
        return None

    wp_source = wp.from_torch(source, dtype=wp_dtype, return_ctype=True)
    wp_source_batch_ptr = wp.from_torch(
        source_batch_ptr.to(torch.int32), dtype=wp.int32, return_ctype=True
    )
    wp_source_mask = wp.from_torch(source_mask, dtype=wp.bool, return_ctype=True)
    wp_dest = wp.from_torch(dest, dtype=wp_dtype, return_ctype=True)
    wp_dest_batch_ptr = wp.from_torch(dest_batch_ptr, dtype=wp.int32, return_ctype=True)
    wp_source_data_copied = wp.from_torch(
        source_data_copied, dtype=wp.bool, return_ctype=True
    )

    masked_lengths = torch.empty(num_systems, device=device, dtype=torch.int32)
    wp_masked_lengths = wp.from_torch(masked_lengths, dtype=wp.int32)
    wp.launch(
        _masked_segment_lengths_kernel,
        dim=num_systems,
        inputs=[
            wp_source_batch_ptr,
            wp_source_mask,
            wp_masked_lengths,
            num_systems,
        ],
        device=device,
    )
    copy_dest_offset = torch.empty(num_systems, device=device, dtype=torch.int32)
    wp_copy_dest_offset = wp.from_torch(copy_dest_offset, dtype=wp.int32)
    wp.utils.array_scan(
        wp_masked_lengths,
        wp_copy_dest_offset,
        inclusive=False,
    )

    dest_capacity = dest.shape[0]
    fits = torch.empty(num_systems, device=device, dtype=torch.int32)
    wp_fits = wp.from_torch(fits, dtype=wp.int32)
    wp.launch(
        _compute_segmented_fits_kernel,
        dim=num_systems,
        inputs=[
            wp_dest_batch_ptr,
            num_dest_segments,
            wp_copy_dest_offset,
            wp_masked_lengths,
            wp_source_mask,
            dest_capacity,
            wp_fits,
            num_systems,
        ],
        device=device,
    )
    fit_scan = torch.empty(num_systems, device=device, dtype=torch.int32)
    wp_fit_scan = wp.from_torch(fit_scan, dtype=wp.int32)
    wp.utils.array_scan(wp_fits, wp_fit_scan, inclusive=False)
    wp_new_num_dest = wp.from_torch(new_num_dest, dtype=wp.int32)
    wp.launch(
        _set_new_num_dest_segmented_kernel,
        dim=1,
        inputs=[
            num_dest_segments,
            wp_fit_scan,
            wp_fits,
            num_systems,
            wp_new_num_dest,
        ],
        device=device,
    )
    wp.launch(
        _copy_masked_segmented_overloads[wp_dtype],
        dim=num_systems,
        inputs=[
            wp_source,
            wp_source_batch_ptr,
            wp_source_mask,
            wp_copy_dest_offset,
            wp_masked_lengths,
            wp_dest,
            wp_dest_batch_ptr,
            wp_source_data_copied,
            num_systems,
            num_dest_segments,
            wp_fit_scan,
            wp_new_num_dest,
            wp_fits,
        ],
        device=device,
    )


def put_masked_segmented(
    source: torch.Tensor,
    source_batch_ptr: torch.Tensor,
    source_mask: torch.Tensor,
    dest: torch.Tensor,
    dest_batch_ptr: torch.Tensor,
    num_dest_segments: int,
    source_data_copied: torch.Tensor,
) -> torch.Tensor | None:
    """Copy masked segments from source into dest and append segment boundaries to dest_batch_ptr.

    Destination is described by dest_batch_ptr and num_dest_segments (same
    layout as source: segment boundaries in a batch_ptr). Data is written
    starting at dest_batch_ptr[num_dest_segments]. New segment end indices
    are written into dest_batch_ptr[num_dest_segments+1 : num_dest_segments+1+num_copied].
    No host sync. Supports bool, float32, float64, int32, int64.

    Parameters
    ----------
    source : torch.Tensor, shape (total_elems, cols) or (total_elems, ...)
        Source data; segment i is source[source_batch_ptr[i]:source_batch_ptr[i+1], :].
    source_batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype int32
        Cumulative segment boundaries in source.
    source_mask : torch.Tensor, shape (num_systems,), dtype bool
        True for each segment to copy.
    dest : torch.Tensor, shape (dest_capacity, cols) or similar
        Destination buffer; same dtype as source.
    dest_batch_ptr : torch.Tensor, shape >= num_dest_segments + num_systems + 2, dtype int32
        Segment boundaries in dest. Read: dest_batch_ptr[num_dest_segments] = next
        free element. Write: new segment end indices appended at
        [num_dest_segments+1, num_dest_segments+1+num_copied_segments].
    num_dest_segments : int
        Current number of segments in dest (dest_batch_ptr has num_dest_segments+1
        valid entries; dest_batch_ptr[num_dest_segments] is the next free index).
    source_data_copied : torch.Tensor, shape (num_systems,), dtype bool
        Output: True for each segment that was copied.

    Returns
    -------
    torch.Tensor or None
        If num_systems > 0: (1,) int32 tensor new_num_dest_segments =
        num_dest_segments + num_copied_segments. If num_systems == 0, None.

    Notes
    -----
    dest_batch_ptr must be pre-allocated with size at least
    num_dest_segments + num_systems + 2 so that new boundaries can be appended.
    """
    if source.dim() != 2:
        source = source.view(source.shape[0], -1)
    if dest.dim() != 2:
        dest = dest.view(dest.shape[0], -1)

    num_systems = source_batch_ptr.shape[0] - 1
    if num_systems == 0:
        return None

    min_dest_batch_size = num_dest_segments + num_systems + 2
    new_num_dest = torch.empty(1, device=source.device, dtype=torch.int32)
    if dest_batch_ptr.shape[0] < min_dest_batch_size:
        raise ValueError(
            f"dest_batch_ptr must have size >= num_dest_segments + num_systems + 2 "
            f"({min_dest_batch_size}); got {dest_batch_ptr.shape[0]}"
        )

    put_masked_segmented_impl(
        source,
        source_batch_ptr,
        source_mask,
        dest,
        dest_batch_ptr,
        num_dest_segments,
        source_data_copied,
        new_num_dest,
    )
    return new_num_dest


@torch.library.custom_op(
    "nvalchemi::defrag_segmented",
    mutates_args=(
        "source",
        "source_batch_ptr",
        "temp",
        "num_kept",
    ),
)
def defrag_segmented_impl(
    source: torch.Tensor,
    source_batch_ptr: torch.Tensor,
    source_data_copied: torch.Tensor,
    temp: torch.Tensor,
    num_kept: torch.Tensor,
) -> None:
    """Defrag source in place by keeping only segments with source_data_copied[i] False."""
    source_dtype = source.dtype
    wp_dtype = TORCH_TO_WP[source_dtype]
    device = str(source.device)
    num_systems = source_batch_ptr.shape[0] - 1
    total_elems = source.shape[0]

    # Kernel writes int32; use caller's tensor in place when possible so write-back is visible.
    bp_for_kernel = (
        source_batch_ptr
        if source_batch_ptr.dtype == torch.int32
        else source_batch_ptr.to(torch.int32)
    )
    wp_source_batch_ptr = wp.from_torch(
        bp_for_kernel, dtype=wp.int32, return_ctype=True
    )

    wp_source = wp.from_torch(source, dtype=wp_dtype, return_ctype=True)
    wp_source_data_copied = wp.from_torch(
        source_data_copied, dtype=wp.bool, return_ctype=True
    )
    wp_temp = wp.from_torch(temp, dtype=wp_dtype)
    wp_num_kept = wp.from_torch(num_kept, dtype=wp.int32)

    if num_systems == 0:
        wp.launch(
            _write_zero_kernel,
            dim=1,
            inputs=[wp_num_kept],
            device=device,
        )
        wp.launch(
            _write_zero_kernel,
            dim=1,
            inputs=[wp_source_batch_ptr],
            device=device,
        )
        return

    kept_int = torch.empty(num_systems, device=device, dtype=torch.int32)
    wp_kept_int = wp.from_torch(kept_int, dtype=wp.int32)
    wp.launch(
        _inv_mask_to_int_kernel,
        dim=num_systems,
        inputs=[wp_source_data_copied, wp_kept_int],
        device=device,
    )
    kept_rank = torch.empty(num_systems, device=device, dtype=torch.int32)
    wp_kept_rank = wp.from_torch(kept_rank, dtype=wp.int32)
    wp.utils.array_scan(
        wp_kept_int,
        wp_kept_rank,
        inclusive=False,
    )
    wp.launch(
        _num_kept_segmented_kernel,
        dim=1,
        inputs=[
            wp_kept_rank,
            wp_source_data_copied,
            num_systems,
            wp_num_kept,
        ],
        device=device,
    )

    kept_lengths = torch.zeros(num_systems, device=device, dtype=torch.int32)
    wp_kept_lengths = wp.from_torch(kept_lengths, dtype=wp.int32)
    wp.launch(
        _scatter_kept_lengths_kernel,
        dim=num_systems,
        inputs=[
            wp_source_batch_ptr,
            wp_source_data_copied,
            wp_kept_rank,
            wp_kept_lengths,
            num_systems,
        ],
        device=device,
    )
    total_kept_elems = torch.empty(1, device=device, dtype=torch.int32)
    wp_total_kept_elems = wp.from_torch(total_kept_elems, dtype=wp.int32)
    wp.launch(
        _sum_prefix_kernel,
        dim=1,
        inputs=[
            wp_kept_lengths,
            wp_num_kept,
            wp_total_kept_elems,
        ],
        device=device,
    )
    new_batch_ptr_scan = torch.empty(num_systems, device=device, dtype=torch.int32)
    wp_new_batch_ptr_scan = wp.from_torch(new_batch_ptr_scan, dtype=wp.int32)
    wp.utils.array_scan(
        wp_kept_lengths,
        wp_new_batch_ptr_scan,
        inclusive=False,
    )
    new_batch_ptr = torch.empty(num_systems + 1, device=device, dtype=torch.int32)
    wp_new_batch_ptr = wp.from_torch(new_batch_ptr, dtype=wp.int32)
    wp.launch(
        _build_new_batch_ptr_kernel,
        dim=num_systems + 1,
        inputs=[
            wp_new_batch_ptr_scan,
            wp_num_kept,
            wp_total_kept_elems,
            wp_new_batch_ptr,
            num_systems,
        ],
        device=device,
    )

    wp.launch(
        _coalesce_segmented_scatter_overloads[wp_dtype],
        dim=num_systems,
        inputs=[
            wp_source,
            wp_source_batch_ptr,
            wp_source_data_copied,
            wp_kept_rank,
            wp_new_batch_ptr,
            wp_temp,
            num_systems,
        ],
        device=device,
    )
    wp.launch(
        _coalesce_segmented_copy_back_from_total_overloads[wp_dtype],
        dim=total_elems,
        inputs=[
            wp_source,
            wp_temp,
            wp_total_kept_elems,
        ],
        device=device,
    )
    wp.launch(
        _copy_batch_ptr_prefix_kernel,
        dim=num_systems + 1,
        inputs=[
            wp_num_kept,
            wp_new_batch_ptr,
            wp_source_batch_ptr,
        ],
        device=device,
    )
    wp.launch(
        _fill_batch_ptr_tail_kernel,
        dim=num_systems + 1,
        inputs=[
            wp_source_batch_ptr,
            wp_num_kept,
            wp_total_kept_elems,
            num_systems,
        ],
        device=device,
    )
    # When caller passed int64 (or other dtype), write the full buffer back into their tensor.
    if bp_for_kernel is not source_batch_ptr:
        source_batch_ptr.copy_(bp_for_kernel)


def defrag_segmented(
    source: torch.Tensor,
    source_batch_ptr: torch.Tensor,
    source_data_copied: torch.Tensor,
) -> torch.Tensor:
    """Defrag source in place by keeping only segments with source_data_copied[i] False.

    Kept segments are moved to the front; the rest of source is zeroed.
    source_batch_ptr is updated in place: the new segment boundaries are
    written to source_batch_ptr[0 : num_kept+1]. Returns the number of kept
    segments. Supports bool, float32, float64, int32, int64.

    Parameters
    ----------
    source : torch.Tensor, shape (total_elems, cols) or (total_elems, ...)
        Data to compact; modified in place. Non-2D is viewed as (total_elems, -1).
    source_batch_ptr : torch.Tensor, shape (num_systems + 1,), dtype int32
        Segment boundaries; read for input layout, overwritten in place with
        new boundaries for the compacted data (valid prefix [0:num_kept+1]).
    source_data_copied : torch.Tensor, shape (num_systems,), dtype bool
        True = segment was copied (drop); False = keep.
    temp : torch.Tensor, optional
        Scratch buffer same shape as source; allocated if None.

    Returns
    -------
    num_kept : torch.Tensor, shape (1,), dtype int32
        Number of kept segments. source_batch_ptr[0:num_kept[0]+1] holds the
        new segment boundaries; total compacted elements = source_batch_ptr[num_kept[0]].

    Notes
    -----
    When num_systems == 0, source_batch_ptr[0] is set to 0 and returns 0.
    Caller may perform one host sync then trim source and source_batch_ptr to
    num_kept.
    """
    if source.dim() != 2:
        source = source.view(source.shape[0], -1)
    temp = torch.empty_like(source)
    num_kept = torch.empty(1, device=source.device, dtype=torch.int32)
    defrag_segmented_impl(source, source_batch_ptr, source_data_copied, temp, num_kept)
    return num_kept
