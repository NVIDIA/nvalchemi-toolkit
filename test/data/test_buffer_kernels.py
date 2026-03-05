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
"""Tests for system-level and segmented buffer kernels (put masked, defrag)."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data.buffer_kernels import (
    compute_put_fit_mask_per_system,
    compute_put_fit_mask_segmented,
    defrag_per_system,
    defrag_segmented,
    put_masked_per_system,
    put_masked_segmented,
)

# Dtypes supported by the buffer kernels (overloaded).
SUPPORTED_DTYPES = [torch.float32, torch.float64, torch.int32, torch.int64]


def _make_source_data(shape: tuple, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Create source tensor with deterministic values (arange) in the given dtype."""
    size = 1
    for s in shape:
        size *= s
    return torch.arange(size, device=device, dtype=dtype).view(shape)


# -----------------------------------------------------------------------------
# System-level: copy with enough room
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_copy_enough_room(device: str, dtype: torch.dtype):
    """All masked source rows fit in dest; all are copied and source_data_copied set."""

    n_src = n_dest = 6
    cols = 2
    source = _make_source_data((n_src, cols), device, dtype)
    source_mask = torch.tensor([True, False, True, False, True, True], device=device)
    source_data_copied = torch.zeros(n_src, dtype=torch.bool, device=device)
    dest = torch.zeros(n_dest, cols, device=device, dtype=dtype)
    dest_mask = torch.zeros(n_dest, dtype=torch.bool, device=device)

    put_masked_per_system(source, source_mask, dest, dest_mask, source_data_copied)

    num_masked = source_mask.sum().item()
    assert source_data_copied.sum().item() == num_masked
    assert dest_mask.sum().item() == num_masked
    expected_dest = source[source_mask].clone()
    actual_dest = dest[dest_mask].reshape(-1, cols)
    torch.testing.assert_close(actual_dest, expected_dest)


# -----------------------------------------------------------------------------
# System-level: not enough room in dest
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_copy_not_enough_room(device: str, dtype: torch.dtype):
    """Dest has only 2 empty slots; only 2 masked rows copied, rest stay in source with mask."""
    n_src = 5
    n_dest = 4
    cols = 2
    source = _make_source_data((n_src, cols), device, dtype)
    source_mask = torch.tensor(
        [True, True, True, False, True], device=device
    )  # 4 masked
    source_data_copied = torch.zeros(n_src, dtype=torch.bool, device=device)
    dest = torch.zeros(n_dest, cols, device=device, dtype=dtype)
    dest_mask = torch.tensor([False, False, True, True], device=device)

    put_masked_per_system(source, source_mask, dest, dest_mask, source_data_copied)

    assert source_data_copied.sum().item() == 2
    assert dest_mask.sum().item() == 4
    assert source_data_copied[0].item() is True
    assert source_data_copied[1].item() is True
    assert source_data_copied[2].item() is False
    assert source_data_copied[4].item() is False
    assert source_mask[2].item() is True
    torch.testing.assert_close(dest[0], source[0])
    torch.testing.assert_close(dest[1], source[1])


# -----------------------------------------------------------------------------
# System-level: shapes (N, 1) and (N, 3, 3)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("shape", [(6, 1), (4, 3, 3)])
def test_copy_shape_view(device: str, dtype: torch.dtype, shape: tuple):
    """Wrappers accept (num_systems, 1) and (num_systems, 3, 3) via .view to 2D."""
    n = shape[0]
    source = _make_source_data(shape, device, dtype)
    source_mask = torch.ones(n, dtype=torch.bool, device=device)
    source_data_copied = torch.zeros(n, dtype=torch.bool, device=device)
    dest = torch.zeros(n, source.numel() // n, device=device, dtype=dtype)
    dest_mask = torch.zeros(n, dtype=torch.bool, device=device)

    put_masked_per_system(source, source_mask, dest, dest_mask, source_data_copied)

    expected = source.reshape(n, -1)
    torch.testing.assert_close(dest, expected)
    assert source_data_copied.all().item()


# -----------------------------------------------------------------------------
# System-level: coalesce leaves tail zeros
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_coalesce_tail_zeros(device: str, dtype: torch.dtype):
    """After coalesce, indices >= num_kept are all zero; no data left in tail."""
    n_src = 6
    cols = 2
    source = _make_source_data((n_src, cols), device, dtype)
    source_mask = torch.tensor([True, False, True, False, True, False], device=device)
    source_data_copied = torch.zeros(n_src, dtype=torch.bool, device=device)
    dest = torch.zeros(n_src, cols, device=device, dtype=dtype)
    dest_mask = torch.zeros(n_src, dtype=torch.bool, device=device)

    put_masked_per_system(source, source_mask, dest, dest_mask, source_data_copied)
    defrag_per_system(source, source_data_copied)

    num_kept = (~source_data_copied).sum().item()
    assert num_kept == (source_mask == False).sum().item()  # noqa: E712
    if num_kept < n_src:
        tail_zeros = torch.zeros(n_src - num_kept, cols, device=device, dtype=dtype)
        torch.testing.assert_close(source[num_kept:], tail_zeros)


# -----------------------------------------------------------------------------
# System-level: no masked rows
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_copy_no_masked_rows(device: str, dtype: torch.dtype):
    """source_mask all False: nothing copied, dest unchanged."""
    n_src = n_dest = 4
    cols = 2
    source = _make_source_data((n_src, cols), device, dtype)
    source_mask = torch.zeros(n_src, dtype=torch.bool, device=device)
    source_data_copied = torch.zeros(n_src, dtype=torch.bool, device=device)
    dest = torch.zeros(n_dest, cols, device=device, dtype=dtype)
    dest_mask = torch.zeros(n_dest, dtype=torch.bool, device=device)

    put_masked_per_system(source, source_mask, dest, dest_mask, source_data_copied)

    assert source_data_copied.sum().item() == 0
    assert dest_mask.sum().item() == 0
    zero = torch.tensor(0, device=device, dtype=dtype)
    assert (dest == zero).all().item()


# -----------------------------------------------------------------------------
# System-level: empty / small
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_copy_empty_source(device: str, dtype: torch.dtype):
    """num_systems = 0: no-op without error."""
    dest = torch.zeros(2, 2, device=device, dtype=dtype)
    dest_mask = torch.zeros(2, dtype=torch.bool, device=device)
    source = torch.zeros(0, 2, device=device, dtype=dtype)
    source_mask = torch.zeros(0, dtype=torch.bool, device=device)
    source_data_copied = torch.zeros(0, dtype=torch.bool, device=device)
    put_masked_per_system(source, source_mask, dest, dest_mask, source_data_copied)
    assert dest_mask.sum().item() == 0


# -----------------------------------------------------------------------------
# Segmented: copy enough room
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_segmented_copy_enough_room(device: str, dtype: torch.dtype):
    """All masked segments fit; dest_batch_ptr gets new segment boundaries appended."""
    # 3 segments: lengths 2, 3, 1 -> batch_ptr [0, 2, 5, 6]
    batch_ptr = torch.tensor([0, 2, 5, 6], device=device, dtype=torch.int32)
    total_elems = 6
    elem_size = 2
    source = _make_source_data((total_elems, elem_size), device, dtype)
    source_mask = torch.tensor([True, False, True], device=device)
    source_data_copied = torch.zeros(3, dtype=torch.bool, device=device)
    dest = torch.zeros(10, elem_size, device=device, dtype=dtype)
    num_dest_segments = 0
    # Need size >= num_dest_segments + num_systems + 2 = 0 + 3 + 2 = 5
    dest_batch_ptr = torch.zeros(5, device=device, dtype=torch.int32)

    new_num_dest = put_masked_segmented(
        source,
        batch_ptr,
        source_mask,
        dest,
        dest_batch_ptr,
        num_dest_segments,
        source_data_copied,
    )

    # Copied segments 0 (len 2) and 2 (len 1) -> 2 new segments, 3 elements
    assert new_num_dest is not None
    assert new_num_dest[0].item() == 2
    assert dest_batch_ptr[0].item() == 0
    assert dest_batch_ptr[1].item() == 2
    assert dest_batch_ptr[2].item() == 3
    assert source_data_copied[0].item() is True
    assert source_data_copied[1].item() is False
    assert source_data_copied[2].item() is True
    torch.testing.assert_close(dest[0:2], source[0:2])
    torch.testing.assert_close(dest[2:3], source[5:6])


# -----------------------------------------------------------------------------
# Segmented: shapes (total_elems, 1) and (total_elems, 3, 3)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
@pytest.mark.parametrize("elem_shape", [(2,), (3, 3)])
def test_segmented_copy_shape_view(device: str, dtype: torch.dtype, elem_shape: tuple):
    """Segmented copy with elem_size from .view (e.g. (N, 1) or (N, 9))."""
    batch_ptr = torch.tensor(
        [0, 1, 3], device=device, dtype=torch.int32
    )  # 2 segments: len 1, 2
    total_elems = 3
    elem_size = 1
    for s in elem_shape:
        elem_size *= s
    source = _make_source_data((total_elems,) + elem_shape, device, dtype)
    source_mask = torch.ones(2, dtype=torch.bool, device=device)
    source_data_copied = torch.zeros(2, dtype=torch.bool, device=device)
    dest = torch.zeros(10, elem_size, device=device, dtype=dtype)
    num_dest_segments = 0
    dest_batch_ptr = torch.zeros(5, device=device, dtype=torch.int32)

    new_num_dest = put_masked_segmented(
        source,
        batch_ptr,
        source_mask,
        dest,
        dest_batch_ptr,
        num_dest_segments,
        source_data_copied,
    )

    assert new_num_dest is not None
    assert new_num_dest[0].item() == 2
    assert dest_batch_ptr[0].item() == 0
    assert dest_batch_ptr[1].item() == 1
    assert dest_batch_ptr[2].item() == 3
    expected = source.reshape(total_elems, -1)
    torch.testing.assert_close(dest[:3], expected)


# -----------------------------------------------------------------------------
# Segmented: coalesce and tail zeros
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_segmented_coalesce_tail_zeros(device: str, dtype: torch.dtype):
    """After defrag_segmented, tail of source is zero; batch_ptr updated in place."""
    batch_ptr = torch.tensor(
        [0, 2, 5, 7], device=device, dtype=torch.int32
    )  # 3 segments
    total_elems = 7
    elem_size = 2
    source = _make_source_data((total_elems, elem_size), device, dtype)
    # Mark segment 1 as copied (to be removed)
    source_data_copied = torch.tensor([False, True, False], device=device)

    num_kept = defrag_segmented(source, batch_ptr, source_data_copied)

    assert num_kept[0].item() == 2
    assert batch_ptr[0].item() == 0
    assert batch_ptr[2].item() == 2 + 2  # len seg0 + len seg2 = 2+2
    # Tail of source should be zero
    kept_elems = int(batch_ptr[2].item())
    if kept_elems < total_elems:
        tail_zeros = torch.zeros(
            total_elems - kept_elems, elem_size, device=device, dtype=dtype
        )
        torch.testing.assert_close(source[kept_elems:], tail_zeros)


# -----------------------------------------------------------------------------
# Segmented: variable segment lengths and empty
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_segmented_variable_lengths(device: str, dtype: torch.dtype):
    """Segments with different lengths; defrag keeps correct layout."""
    # 4 segments: [0, 0, 3, 4, 6] -> lengths 0, 3, 1, 2
    batch_ptr = torch.tensor([0, 0, 3, 4, 6], device=device, dtype=torch.int32)
    total_elems = 6
    elem_size = 1
    source = _make_source_data((total_elems, elem_size), device, dtype)
    source_data_copied = torch.tensor(
        [True, False, True, False], device=device
    )  # keep 1 and 3

    num_kept = defrag_segmented(source, batch_ptr, source_data_copied)

    assert num_kept[0].item() == 2
    assert batch_ptr[0].item() == 0
    assert batch_ptr[1].item() == 3  # first kept segment length 3
    assert batch_ptr[2].item() == 3 + 2  # second kept segment length 2
    expected_01 = torch.tensor([[0], [1], [2]], device=device, dtype=dtype)
    torch.testing.assert_close(source[:3], expected_01)
    expected_45 = torch.tensor([[4], [5]], device=device, dtype=dtype)
    torch.testing.assert_close(source[3:5], expected_45)
    if total_elems > 5:
        torch.testing.assert_close(
            source[5:], torch.zeros(total_elems - 5, 1, device=device, dtype=dtype)
        )


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_segmented_empty_source(device: str, dtype: torch.dtype):
    """num_systems = 0: defrag returns 0 and sets batch_ptr[0] = 0 in place."""
    batch_ptr = torch.tensor([0], device=device, dtype=torch.int32)
    source = torch.zeros(0, 2, device=device, dtype=dtype)
    source_data_copied = torch.zeros(0, dtype=torch.bool, device=device)
    num_kept = defrag_segmented(source, batch_ptr, source_data_copied)
    assert num_kept[0].item() == 0
    assert batch_ptr[0].item() == 0


# -----------------------------------------------------------------------------
# compute_put_fit_mask_per_system (uniform)
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_compute_put_fit_mask_per_system_enough_room(device: str, dtype: torch.dtype):
    """When dest has enough empty slots, all masked rows get fit_mask True."""
    n_src = 5
    n_dest = 10
    source_mask = torch.tensor([True, False, True, True, False], device=device)
    dest_mask = torch.zeros(n_dest, dtype=torch.bool, device=device)
    fit_mask = torch.zeros(n_src, dtype=torch.bool, device=device)

    compute_put_fit_mask_per_system(source_mask, dest_mask, fit_mask)

    # All 3 masked rows fit
    assert fit_mask.sum().item() == 3
    assert fit_mask[0].item() is True
    assert fit_mask[2].item() is True
    assert fit_mask[3].item() is True
    assert fit_mask[1].item() is False
    assert fit_mask[4].item() is False


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_compute_put_fit_mask_per_system_not_enough_room(
    device: str, dtype: torch.dtype
):
    """When dest has only 2 empty slots, only first 2 masked rows get fit_mask True."""
    n_src = 5
    source_mask = torch.tensor(
        [True, True, True, False, True], device=device
    )  # 4 masked
    dest_mask = torch.tensor([False, False, True, True], device=device)  # 2 empty
    fit_mask = torch.zeros(n_src, dtype=torch.bool, device=device)

    compute_put_fit_mask_per_system(source_mask, dest_mask, fit_mask)

    assert fit_mask.sum().item() == 2
    assert fit_mask[0].item() is True
    assert fit_mask[1].item() is True
    assert fit_mask[2].item() is False
    assert fit_mask[4].item() is False


# -----------------------------------------------------------------------------
# compute_put_fit_mask_segmented
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_compute_put_fit_mask_segmented_fits(device: str, dtype: torch.dtype):
    """All masked segments fit in dest; fit_mask True for masked segments that fit."""
    # Source: 3 segments lengths 2, 1, 3
    source_batch_ptr = torch.tensor([0, 2, 3, 6], device=device, dtype=torch.int32)
    source_mask = torch.tensor([True, False, True], device=device)
    num_dest_segments = 1
    dest_capacity = 20
    dest_batch_ptr = torch.tensor(
        [0, 5, 5, 5, 5, 5], device=device, dtype=torch.int32
    )  # 1 segment of 5, room for more
    fit_mask = torch.zeros(3, dtype=torch.bool, device=device)

    compute_put_fit_mask_segmented(
        source_batch_ptr,
        source_mask,
        dest_batch_ptr,
        num_dest_segments,
        dest_capacity,
        fit_mask,
    )

    # Segment 0 (len 2) and 2 (len 3) masked; both fit (base=5, 5+2=7, 5+2+3=8 <= 20)
    assert fit_mask[0].item() is True
    assert fit_mask[1].item() is False
    assert fit_mask[2].item() is True


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_compute_put_fit_mask_segmented_no_batch_ptr_room(
    device: str, dtype: torch.dtype
):
    """When dest_batch_ptr is too small, fit_mask is all False."""
    source_batch_ptr = torch.tensor(
        [0, 2, 3], device=device, dtype=torch.int32
    )  # 2 segments
    source_mask = torch.ones(2, dtype=torch.bool, device=device)
    num_dest_segments = 1
    dest_capacity = 100
    dest_batch_ptr = torch.tensor(
        [0, 10], device=device, dtype=torch.int32
    )  # length 2; need >= 1+2+2=5
    fit_mask = torch.ones(2, dtype=torch.bool, device=device)

    compute_put_fit_mask_segmented(
        source_batch_ptr,
        source_mask,
        dest_batch_ptr,
        num_dest_segments,
        dest_capacity,
        fit_mask,
    )

    # Implementation zeroes fit_mask when dest_batch_ptr has no room for new boundaries
    assert fit_mask.sum().item() == 0


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_compute_put_fit_mask_segmented_no_data_room(device: str, dtype: torch.dtype):
    """When dest data capacity is too small for a segment, that segment gets False."""
    # Source: 2 segments lengths 2 and 5
    source_batch_ptr = torch.tensor([0, 2, 7], device=device, dtype=torch.int32)
    source_mask = torch.tensor([True, True], device=device)
    num_dest_segments = 0
    dest_capacity = 4  # only 4 rows free; segment 0 (2) fits, segment 1 (5) does not
    dest_batch_ptr = torch.tensor(
        [0, 0, 0, 0, 0, 0], device=device, dtype=torch.int32
    )  # length 6 >= 0+2+2
    fit_mask = torch.zeros(2, dtype=torch.bool, device=device)

    compute_put_fit_mask_segmented(
        source_batch_ptr,
        source_mask,
        dest_batch_ptr,
        num_dest_segments,
        dest_capacity,
        fit_mask,
    )

    # Segment 0: 2 elems, fits (0+2<=4). Segment 1: 5 elems, 2+5=7>4
    assert fit_mask[0].item() is True
    assert fit_mask[1].item() is False


# -----------------------------------------------------------------------------
# Buffer API: scenarios that mirror level_storage / batch defrag tests
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_defrag_segmented_two_segments_keep_second(device: str, dtype: torch.dtype):
    """Mirrors test_put_with_copied_mask_out_segmented: 2 segments [1, 2], drop 0 keep 1.

    copied_mask [True, False] → keep segment 1 (2 elements). After defrag: 1 segment,
    batch_ptr = [0, 2], source[0:2] = original segment 1.
    """
    # 2 segments: lengths 1 and 2, total 3 elements
    batch_ptr = torch.tensor([0, 1, 3], device=device, dtype=torch.int32)
    source = _make_source_data((3, 1), device, dtype)  # [[0], [1], [2]]
    # True = was copied (drop), False = keep → keep segment 1 only
    source_data_copied = torch.tensor([True, False], device=device)

    num_kept = defrag_segmented(source, batch_ptr, source_data_copied)

    assert num_kept[0].item() == 1
    assert batch_ptr[0].item() == 0
    assert batch_ptr[1].item() == 2
    # Kept segment 1 was source[1:3] = [[1], [2]]
    expected = torch.tensor([[1], [2]], device=device, dtype=dtype)
    torch.testing.assert_close(source[:2], expected)


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES)
def test_defrag_segmented_three_segments_keep_middle(device: str, dtype: torch.dtype):
    """Mirrors test_defrag_with_copied_mask: 3 segments [2, 3, 1], drop 0 and 2, keep 1.

    copied_mask [True, False, True] → keep segment 1 (3 elements). After defrag: 1 segment,
    batch_ptr = [0, 3], source[0:3] = original segment 1.
    """
    # 3 segments: lengths 2, 3, 1, total 6 elements
    batch_ptr = torch.tensor([0, 2, 5, 6], device=device, dtype=torch.int32)
    source = _make_source_data((6, 1), device, dtype)  # [[0], [1], [2], [3], [4], [5]]
    # True = was copied (drop), False = keep → keep segment 1 only (indices 2,3,4)
    source_data_copied = torch.tensor([True, False, True], device=device)

    num_kept = defrag_segmented(source, batch_ptr, source_data_copied)

    assert num_kept[0].item() == 1
    assert batch_ptr[0].item() == 0
    assert batch_ptr[1].item() == 3
    # Kept segment 1 was source[2:5] = [[2], [3], [4]]
    expected = torch.tensor([[2], [3], [4]], device=device, dtype=dtype)
    torch.testing.assert_close(source[:3], expected)
