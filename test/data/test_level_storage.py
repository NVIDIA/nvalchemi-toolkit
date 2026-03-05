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
"""Comprehensive tests for level_storage module (LevelSchema, *LevelStorage, TensorDict backend)."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data.level_storage import (
    DEFAULT_ATTRIBUTE_MAP,
    DEFAULT_SEGMENTED_GROUPS,
    TORCH_DTYPE_MAP,
    LevelSchema,
    MultiLevelStorage,
    SegmentedLevelStorage,
    UniformLevelStorage,
)


# -----------------------------------------------------------------------------
# LevelSchema
# -----------------------------------------------------------------------------
class TestLevelSchema:
    """Tests for LevelSchema registry."""

    def test_default_construction(self):
        schema = LevelSchema()
        assert "positions" in schema.attr_to_group
        assert schema.attr_to_group["positions"] == "atoms"
        assert schema.group_to_attrs["atoms"] == DEFAULT_ATTRIBUTE_MAP["atoms"]
        assert schema.segmented_groups == DEFAULT_SEGMENTED_GROUPS

    def test_custom_group_to_attrs(self):
        schema = LevelSchema(group_to_attrs={"nodes": {"x"}, "global": {"e"}})
        assert schema.attr_to_group["x"] == "nodes"
        assert schema.attr_to_group["e"] == "global"
        assert schema.group("x") == "nodes"

    def test_set_new_attr(self):
        schema = LevelSchema()
        schema.set("custom_attr", "atoms", dtype="float32", is_segmented=True)
        assert schema.attr_to_group["custom_attr"] == "atoms"
        assert schema.dtype("custom_attr") == "float32"
        assert schema.is_segmented_attr("custom_attr")

    def test_set_is_segmented_false_discards(self):
        """LevelSchema.set with is_segmented=False removes group from segmented_groups."""
        schema = LevelSchema(group_to_attrs={"g": {"a"}}, segmented_groups={"g"})
        assert schema.is_segmented_group("g")
        schema.set("a", "g", is_segmented=False)
        assert not schema.is_segmented_group("g")

    def test_set_with_torch_dtype(self):
        """LevelSchema.set accepts torch.dtype and maps to string."""
        schema = LevelSchema()
        schema.set("x", "atoms", dtype=torch.float32)
        assert schema.dtype("x") == "float32"
        schema.set("y", "atoms", dtype=torch.int64)
        assert schema.dtype("y") == "int64"

    def test_group_raises_for_unknown_attr(self):
        schema = LevelSchema()
        with pytest.raises(KeyError, match="not found"):
            schema.group("nonexistent")

    def test_dtype_raises_for_unknown_attr(self):
        schema = LevelSchema()
        schema.set("x", "atoms")  # no dtype in default map for "x"
        with pytest.raises(KeyError, match="not found in dtype registry"):
            schema.dtype("x")

    def test_is_segmented_group(self):
        schema = LevelSchema()
        assert schema.is_segmented_group("atoms")
        assert schema.is_segmented_group("edges")
        assert not schema.is_segmented_group("system")

    def test_mark_unmark_group_segmented(self):
        schema = LevelSchema(group_to_attrs={"g": {"a"}}, segmented_groups=set())
        assert not schema.is_segmented_group("g")
        schema.mark_group_segmented("g")
        assert schema.is_segmented_group("g")
        schema.unmark_group_segmented("g")
        assert not schema.is_segmented_group("g")
        with pytest.raises(KeyError):
            schema.unmark_group_segmented("g")

    def test_dtypes_must_match_attrs(self):
        with pytest.raises(ValueError, match="dtype keys must match"):
            LevelSchema(
                group_to_attrs={"g": {"a", "b"}},
                dtypes={"a": "float32"},  # missing b
            )

    def test_clone_is_independent(self):
        schema = LevelSchema()
        schema.set("extra", "atoms")
        cloned = schema.clone()
        cloned.set("another", "edges")
        assert "another" not in schema.attr_to_group
        assert "extra" in cloned.attr_to_group


# -----------------------------------------------------------------------------
# UniformLevelStorage
# -----------------------------------------------------------------------------
class TestUniformLevelStorage:
    """Tests for UniformLevelStorage (TensorDict-backed, uniform first dim)."""

    def test_empty_construction(self):
        u = UniformLevelStorage(device="cpu")
        assert len(u) == 0
        assert u._data.is_empty()

    def test_from_dict(self):
        data = {"a": torch.randn(5, 3), "b": torch.randn(5, 2)}
        u = UniformLevelStorage(data=data, device="cpu", validate=True)
        assert len(u) == 5
        assert u["a"].shape == (5, 3)
        assert u["b"].shape == (5, 2)

    def test_inconsistent_first_dim_raises(self):
        data = {"a": torch.randn(5, 3), "b": torch.randn(4, 2)}
        with pytest.raises(ValueError, match="Inconsistent first dimension"):
            UniformLevelStorage(data=data, device="cpu", validate=True)

    def test_select_slice(self):
        u = UniformLevelStorage(
            data={"a": torch.randn(6, 2), "b": torch.randn(6, 1)},
            device="cpu",
            validate=False,
        )
        sub = u[1:4]
        assert isinstance(sub, UniformLevelStorage)
        assert len(sub) == 3
        assert sub["a"].shape == (3, 2)

    def test_select_int(self):
        u = UniformLevelStorage(
            data={"a": torch.randn(4, 2)},
            device="cpu",
            validate=False,
        )
        sub = u[2]
        assert len(sub) == 1
        assert sub["a"].shape == (1, 2)

    def test_select_tensor_index(self):
        u = UniformLevelStorage(
            data={"a": torch.arange(6).float().unsqueeze(1)},
            device="cpu",
            validate=False,
        )
        sub = u[torch.tensor([0, 2, 4])]
        assert len(sub) == 3
        assert sub["a"].squeeze(1).tolist() == [0.0, 2.0, 4.0]

    def test_update_at(self):
        u = UniformLevelStorage(
            data={"a": torch.zeros(4, 2)},
            device="cpu",
            validate=False,
        )
        u.update_at("a", torch.ones(2, 2), slice(1, 3))
        assert u["a"][1].eq(1).all()
        assert u["a"][0].eq(0).all()

    def test_concatenate_in_place(self):
        u = UniformLevelStorage(
            data={"a": torch.randn(2, 3), "b": torch.randn(2, 1)},
            device="cpu",
            validate=False,
        )
        other = {"a": torch.randn(3, 3), "b": torch.randn(3, 1)}
        u.concatenate(other, strict=False)
        assert len(u) == 5
        assert u["a"].shape == (5, 3)

    def test_concatenate_strict_keys_mismatch_raises(self):
        u = UniformLevelStorage(
            data={"a": torch.randn(2, 1)}, device="cpu", validate=False
        )
        with pytest.raises(ValueError, match="Keys mismatch"):
            u.concatenate({"b": torch.randn(2, 1)}, strict=True)

    def test_is_segmented_false(self):
        u = UniformLevelStorage(
            data={"a": torch.randn(2, 1)}, device="cpu", validate=False
        )
        assert u.is_segmented() is False

    def test_keys_values_items(self):
        u = UniformLevelStorage(
            data={"a": torch.randn(2, 1), "b": torch.randn(2, 1)},
            device="cpu",
            validate=False,
        )
        assert set(u.keys()) == {"a", "b"}
        assert len(list(u.values())) == 2
        assert len(list(u.items())) == 2

    def test_get_default(self):
        u = UniformLevelStorage(
            data={"a": torch.randn(2, 1)}, device="cpu", validate=False
        )
        assert u.get("b", None) is None
        assert u.get("a") is not None

    def test_pop(self):
        u = UniformLevelStorage(
            data={"a": torch.randn(2, 1), "b": torch.randn(2, 1)},
            device="cpu",
            validate=False,
        )
        b = u.pop("b")
        assert b.shape == (2, 1)
        assert "b" not in u

    def test_deepcopy_copy_clone(self):
        u = UniformLevelStorage(
            data={"a": torch.randn(2, 1)},
            device="cpu",
            validate=False,
        )
        c = u.deepcopy()
        assert c["a"] is not u["a"]
        c2 = u.copy()
        assert c2["a"] is u["a"]
        c3 = u.clone()
        assert c3["a"] is not u["a"] and c3.attr_map is not u.attr_map

    def test_to_device(self):
        u = UniformLevelStorage(
            data={"a": torch.randn(2, 1)},
            device="cpu",
            validate=False,
        )
        u.to_device("cpu")
        assert u.device.type == "cpu"

    def test_put_and_defrag(self):
        """put copies masked rows from src into self; defrag compacts source."""
        device = "cpu"
        # Source: 4 rows; dest (buffer): 4 rows capacity
        src = UniformLevelStorage(
            data={
                "a": torch.tensor(
                    [[1.0], [2.0], [3.0], [4.0]], device=device, dtype=torch.float32
                ),
            },
            device=device,
            validate=False,
        )
        dest = UniformLevelStorage(
            data={"a": torch.zeros(4, 1, device=device, dtype=torch.float32)},
            device=device,
            validate=False,
        )
        mask = torch.tensor([True, False, True, False], device=device)
        dest.put(src, mask)
        # Rows 0 and 2 copied into dest at first two slots
        assert dest["a"][0].item() == 1.0
        assert dest["a"][1].item() == 3.0
        # copied_mask stored on src for defrag
        copied = getattr(src, "_copied_mask", None)
        assert copied is not None
        assert copied[0].item() is True
        assert copied[1].item() is False
        assert copied[2].item() is True
        assert copied[3].item() is False
        # Defrag src: keep rows 1 and 3, drop 0 and 2
        src.defrag()
        assert len(src) == 2
        assert src["a"][0].item() == 2.0
        assert src["a"][1].item() == 4.0

    def test_put_with_copied_mask_out(self):
        """put with copied_mask provided updates it in place."""
        device = "cpu"
        src = UniformLevelStorage(
            data={
                "a": torch.tensor([[1.0], [2.0]], device=device, dtype=torch.float32)
            },
            device=device,
            validate=False,
        )
        dest = UniformLevelStorage(
            data={"a": torch.zeros(2, 1, device=device, dtype=torch.float32)},
            device=device,
            validate=False,
        )
        mask = torch.tensor([True, True], device=device)
        copied_mask = torch.zeros(2, dtype=torch.bool, device=device)
        dest.put(src, mask, copied_mask=copied_mask)
        assert copied_mask.all()
        assert dest["a"][0].item() == 1.0
        assert dest["a"][1].item() == 2.0
        src.defrag(copied_mask=copied_mask)
        assert len(src) == 0

    def test_put_defrag_fixed_tensor_shapes(self):
        """Data tensors are not expanded or trimmed by put or defrag (fixed storage)."""
        device = "cpu"
        src = UniformLevelStorage(
            data={
                "a": torch.tensor(
                    [[1.0], [2.0], [3.0]], device=device, dtype=torch.float32
                ),
            },
            device=device,
            validate=False,
        )
        dest = UniformLevelStorage(
            data={"a": torch.zeros(4, 1, device=device, dtype=torch.float32)},
            device=device,
            validate=False,
        )
        shape_before = dest._data["a"].shape
        mask = torch.tensor([True, False, True], device=device)
        dest.put(src, mask)
        assert dest._data["a"].shape == shape_before
        copied = getattr(src, "_copied_mask", None)
        assert copied is not None
        src.defrag()
        assert src._data["a"].shape == (3, 1)

    def test_put_partial_copy_only_what_fits_copied_mask(self):
        """When dest has room for only 1 row, put copies 1; copied_mask True only for that row."""
        device = "cpu"
        src = UniformLevelStorage(
            data={
                "a": torch.tensor(
                    [[1.0], [2.0], [3.0]], device=device, dtype=torch.float32
                ),
            },
            device=device,
            validate=False,
        )
        dest = UniformLevelStorage(
            data={"a": torch.zeros(3, 1, device=device, dtype=torch.float32)},
            device=device,
            validate=False,
        )
        dest_mask = torch.tensor([True, True, False], device=device)  # only 1 empty
        mask = torch.tensor([True, True, True], device=device)
        copied_mask = torch.zeros(3, dtype=torch.bool, device=device)
        dest.put(src, mask, copied_mask=copied_mask, dest_mask=dest_mask)
        assert copied_mask.sum().item() == 1
        assert copied_mask[0].item() is True
        assert copied_mask[1].item() is False
        assert copied_mask[2].item() is False
        assert dest["a"][2].item() == 1.0

    def test_compute_put_per_system_fit_mask(self):
        """compute_put_per_system_fit_mask writes fit_mask; put with it copies same set."""
        device = "cpu"
        src = UniformLevelStorage(
            data={
                "a": torch.tensor(
                    [[1.0], [2.0], [3.0], [4.0]], device=device, dtype=torch.float32
                ),
            },
            device=device,
            validate=False,
        )
        dest = UniformLevelStorage(
            data={"a": torch.zeros(4, 1, device=device, dtype=torch.float32)},
            device=device,
            validate=False,
        )
        source_mask = torch.tensor([True, False, True, False], device=device)
        fit_mask = torch.zeros(4, dtype=torch.bool, device=device)
        dest.compute_put_per_system_fit_mask(src, source_mask, None, fit_mask)
        # All 2 masked rows fit in 4 empty slots
        assert fit_mask.sum().item() == 2
        assert fit_mask[0].item() is True
        assert fit_mask[2].item() is True
        assert fit_mask[1].item() is False
        assert fit_mask[3].item() is False
        # put with fit_mask should copy the same rows
        dest.put(src, fit_mask)
        assert dest["a"][0].item() == 1.0
        assert dest["a"][1].item() == 3.0

    def test_compute_put_per_system_fit_mask_not_enough_room(self):
        """compute_put_per_system_fit_mask only True for rows that fit in empty slots."""
        device = "cpu"
        src = UniformLevelStorage(
            data={
                "a": torch.tensor(
                    [[1.0], [2.0], [3.0]], device=device, dtype=torch.float32
                ),
            },
            device=device,
            validate=False,
        )
        dest = UniformLevelStorage(
            data={"a": torch.zeros(3, 1, device=device, dtype=torch.float32)},
            device=device,
            validate=False,
        )
        # Dest has 2 occupied slots (indices 0, 1), so only 1 empty
        dest_mask = torch.tensor([True, True, False], device=device)
        source_mask = torch.tensor([True, True, True], device=device)
        fit_mask = torch.zeros(3, dtype=torch.bool, device=device)
        dest.compute_put_per_system_fit_mask(src, source_mask, dest_mask, fit_mask)
        assert fit_mask.sum().item() == 1
        assert fit_mask[0].item() is True
        assert fit_mask[1].item() is False
        assert fit_mask[2].item() is False


# -----------------------------------------------------------------------------
# SegmentedLevelStorage
# -----------------------------------------------------------------------------
class TestSegmentedLevelStorage:
    """Tests for SegmentedLevelStorage (TensorDict-backed, variable-length segments)."""

    def test_empty_construction(self):
        s = SegmentedLevelStorage(device="cpu")
        assert len(s) == 0
        assert s.num_elements() == 0

    def test_from_dict_with_segment_lengths(self):
        data = {"x": torch.randn(10, 3), "y": torch.randn(10, 1)}
        s = SegmentedLevelStorage(
            data=data,
            segment_lengths=[4, 6],
            device="cpu",
            validate=True,
        )
        assert len(s) == 2
        assert s.num_elements() == 10
        assert s.segment_lengths.tolist() == [4, 6]

    def test_single_segment_inferred(self):
        data = {"x": torch.randn(7, 2)}
        s = SegmentedLevelStorage(data=data, device="cpu", validate=True)
        assert len(s) == 1
        assert s.segment_lengths.item() == 7

    def test_negative_segment_lengths_raises(self):
        data = {"x": torch.randn(10, 2)}
        with pytest.raises(ValueError, match="Segment lengths cannot be negative"):
            SegmentedLevelStorage(
                data=data,
                segment_lengths=[4, -1, 7],
                device="cpu",
                validate=True,
            )

    def test_segment_lengths_sum_mismatch_raises(self):
        """When sum(segment_lengths) != data first dim, validation raises."""
        data = {"x": torch.randn(10, 2)}
        with pytest.raises(ValueError, match="Sum of segment_lengths.*!= data length"):
            SegmentedLevelStorage(
                data=data,
                segment_lengths=[3, 4],
                device="cpu",
                validate=True,
            )

    def test_batch_idx_length_mismatch_raises(self):
        """When batch_idx is provided and length != total_elements, validation raises."""
        data = {"x": torch.randn(5, 2)}
        batch_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
        with pytest.raises(ValueError, match="batch_idx length"):
            SegmentedLevelStorage(
                data=data,
                segment_lengths=[2, 3],
                device="cpu",
                batch_idx=batch_idx,
                validate=True,
            )

    def test_batch_idx_first_element_not_zero_raises(self):
        """Validation raises when batch_idx does not start at 0."""
        data = {"x": torch.randn(5, 2)}
        batch_idx = torch.tensor([1, 1, 1, 2, 2], dtype=torch.int32)
        with pytest.raises(ValueError, match="batch_idx must start at 0"):
            SegmentedLevelStorage(
                data=data,
                segment_lengths=[2, 3],
                device="cpu",
                batch_idx=batch_idx,
                validate=True,
            )

    def test_batch_idx_last_element_wrong_raises(self):
        """Validation raises when batch_idx last element != num_segments - 1."""
        data = {"x": torch.randn(5, 2)}
        # 2 segments (0, 1); last element must be 1. Use last=0 to trigger error.
        batch_idx = torch.tensor([0, 0, 1, 1, 0], dtype=torch.int32)
        with pytest.raises(ValueError, match="batch_idx last element"):
            SegmentedLevelStorage(
                data=data,
                segment_lengths=[2, 3],
                device="cpu",
                batch_idx=batch_idx,
                validate=True,
            )

    def test_batch_ptr_length_wrong_raises(self):
        """Validation raises when batch_ptr length != num_segments + 1."""
        data = {"x": torch.randn(5, 2)}
        batch_ptr = torch.tensor([0, 2], dtype=torch.int32)
        with pytest.raises(ValueError, match="batch_ptr length"):
            SegmentedLevelStorage(
                data=data,
                segment_lengths=[2, 3],
                device="cpu",
                batch_ptr=batch_ptr,
                validate=True,
            )

    def test_batch_ptr_first_not_zero_raises(self):
        """Validation raises when batch_ptr does not start at 0."""
        data = {"x": torch.randn(5, 2)}
        batch_ptr = torch.tensor([1, 2, 5], dtype=torch.int32)
        with pytest.raises(ValueError, match="batch_ptr must start at 0"):
            SegmentedLevelStorage(
                data=data,
                segment_lengths=[2, 3],
                device="cpu",
                batch_ptr=batch_ptr,
                validate=True,
            )

    def test_batch_ptr_last_not_total_raises(self):
        """Validation raises when batch_ptr last element != total_elements."""
        data = {"x": torch.randn(5, 2)}
        batch_ptr = torch.tensor([0, 2, 4], dtype=torch.int32)
        with pytest.raises(ValueError, match="batch_ptr logical end"):
            SegmentedLevelStorage(
                data=data,
                segment_lengths=[2, 3],
                device="cpu",
                batch_ptr=batch_ptr,
                validate=True,
            )

    def test_setitem_length_mismatch_raises(self):
        """_validate_setitem raises when value length != num_elements."""
        s = SegmentedLevelStorage(
            data={"x": torch.randn(5, 2)},
            segment_lengths=[2, 3],
            device="cpu",
            validate=True,
        )
        with pytest.raises(ValueError, match="Length mismatch"):
            s["x"] = torch.randn(3, 2)

    def test_select_slice_partial_expand_idx(self):
        """Select with slice that is not full range hits _expand_idx else branch."""
        s = SegmentedLevelStorage(
            data={"x": torch.randn(10, 2)},
            segment_lengths=[3, 4, 3],
            device="cpu",
            validate=False,
        )
        sub = s[1:3]
        assert len(sub) == 2
        assert sub.num_elements() == 4 + 3

    def test_select_list_index_normalize_segment_index(self):
        """Select with list index uses _normalize_segment_index (list path)."""
        s = SegmentedLevelStorage(
            data={"x": torch.randn(10, 2)},
            segment_lengths=[3, 4, 3],
            device="cpu",
            validate=False,
        )
        sub = s[[0, 2]]
        assert len(sub) == 2
        assert sub.num_elements() == 3 + 3

    def test_select_bool_mask_wrong_length_raises(self):
        """Select with bool mask length != num segments raises."""
        s = SegmentedLevelStorage(
            data={"x": torch.randn(6, 2)},
            segment_lengths=[2, 3, 1],
            device="cpu",
            validate=False,
        )
        with pytest.raises(ValueError, match="Boolean index length"):
            _ = s[torch.tensor([True, False])]

    def test_select_slice(self):
        s = SegmentedLevelStorage(
            data={"x": torch.randn(10, 2), "y": torch.randn(10, 1)},
            segment_lengths=[3, 4, 3],
            device="cpu",
            validate=False,
        )
        sub = s[1:3]
        assert isinstance(sub, SegmentedLevelStorage)
        assert len(sub) == 2
        assert sub.num_elements() == 4 + 3
        assert sub.segment_lengths.tolist() == [4, 3]

    def test_select_int(self):
        s = SegmentedLevelStorage(
            data={"x": torch.randn(10, 2)},
            segment_lengths=[4, 6],
            device="cpu",
            validate=False,
        )
        sub = s[0]
        assert len(sub) == 1
        assert sub.num_elements() == 4

    def test_select_tensor_index(self):
        s = SegmentedLevelStorage(
            data={"x": torch.arange(12).float().unsqueeze(1)},
            segment_lengths=[2, 4, 6],
            device="cpu",
            validate=False,
        )
        sub = s[torch.tensor([0, 2])]
        assert len(sub) == 2
        assert sub.num_elements() == 2 + 6

    def test_batch_ptr_lazy(self):
        s = SegmentedLevelStorage(
            data={"x": torch.randn(5, 1)},
            segment_lengths=[2, 3],
            device="cpu",
            validate=False,
        )
        ptr = s.batch_ptr
        assert ptr.tolist() == [0, 2, 5]
        assert (s.batch_idx[:2] == 0).all()
        assert (s.batch_idx[2:5] == 1).all()

    def test_update_at(self):
        s = SegmentedLevelStorage(
            data={"x": torch.zeros(5, 2)},
            segment_lengths=[2, 3],
            device="cpu",
            validate=False,
        )
        s.update_at("x", torch.ones(2, 2), 0)
        assert s["x"][:2].eq(1).all()
        assert s["x"][2:].eq(0).all()

    def test_concatenate_in_place(self):
        s = SegmentedLevelStorage(
            data={"x": torch.randn(5, 2), "y": torch.randn(5, 1)},
            segment_lengths=[2, 3],
            device="cpu",
            validate=False,
        )
        other = SegmentedLevelStorage(
            data={"x": torch.randn(4, 2), "y": torch.randn(4, 1)},
            segment_lengths=[1, 3],
            device="cpu",
            validate=False,
        )
        s.concatenate(other, strict=True)
        assert len(s) == 4
        assert s.num_elements() == 5 + 4
        assert s.segment_lengths.tolist() == [2, 3, 1, 3]

    def test_is_segmented_true(self):
        s = SegmentedLevelStorage(
            data={"x": torch.randn(3, 1)},
            segment_lengths=[3],
            device="cpu",
            validate=False,
        )
        assert s.is_segmented() is True

    def test_clone_copies_segment_bookkeeping(self):
        s = SegmentedLevelStorage(
            data={"x": torch.randn(4, 1)},
            segment_lengths=[2, 2],
            device="cpu",
            validate=False,
        )
        c = s.clone()
        assert c.segment_lengths is not s.segment_lengths
        assert c.num_elements() == s.num_elements()

    def test_to_device_moves_segment_lengths(self):
        s = SegmentedLevelStorage(
            data={"x": torch.randn(3, 1)},
            segment_lengths=[3],
            device="cpu",
            validate=False,
        )
        s.to_device("cpu")
        assert s.device.type == "cpu"
        assert s.segment_lengths.device.type == "cpu"

    def test_put_and_defrag(self):
        """put copies masked segments from src into self; defrag compacts source."""
        device = "cpu"
        # Source: 2 segments (lengths 2, 3), total 5 elements
        src = SegmentedLevelStorage(
            data={
                "x": torch.tensor(
                    [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
                    device=device,
                    dtype=torch.float32,
                ),
            },
            segment_lengths=[2, 3],
            device=device,
            validate=False,
        )
        # Dest: 1 segment of 10 elements, data has 15 rows (room for 5 more).
        # Pre-allocate batch_ptr capacity so put can append (fixed storage: no growing).
        dest = SegmentedLevelStorage(
            data={"x": torch.zeros(15, 2, device=device, dtype=torch.float32)},
            segment_lengths=[10],
            device=device,
            batch_ptr_capacity=5,  # 1 + 2 segments + 2
            validate=False,
        )
        mask = torch.tensor([True, True], device=device)
        dest.put(src, mask)
        assert len(dest) == 3  # 1 original + 2 appended
        assert dest.num_elements() == 15
        # First segment unchanged (zeros), next two are copied from src
        torch.testing.assert_close(dest["x"][10:12], src["x"][:2])
        torch.testing.assert_close(dest["x"][12:15], src["x"][2:5])
        copied = getattr(src, "_copied_mask", None)
        assert copied is not None
        assert copied.all()
        src.defrag()
        assert len(src) == 0
        assert src.num_elements() == 0

    def test_put_with_copied_mask_out_segmented(self):
        """put with copied_mask provided updates it in place; defrag uses it."""
        device = "cpu"
        src = SegmentedLevelStorage(
            data={
                "x": torch.tensor(
                    [[1.0], [2.0], [3.0]],
                    device=device,
                    dtype=torch.float32,
                ),
            },
            segment_lengths=[1, 2],
            device=device,
            validate=False,
        )
        # Pre-allocate batch_ptr capacity so put can append (fixed storage: no growing).
        dest = SegmentedLevelStorage(
            data={"x": torch.zeros(10, 1, device=device, dtype=torch.float32)},
            segment_lengths=[0],
            device=device,
            batch_ptr_capacity=5,  # 1 + 2 segments + 2
            validate=False,
        )
        mask = torch.tensor([True, False], device=device)  # copy segment 0 only
        copied_mask = torch.zeros(2, dtype=torch.bool, device=device)
        dest.put(src, mask, copied_mask=copied_mask)
        assert copied_mask[0].item() is True
        assert copied_mask[1].item() is False
        assert len(dest) == 2  # 1 initial (length 0) + 1 appended
        assert dest["x"][0].item() == 1.0
        src.defrag(copied_mask=copied_mask)
        assert len(src) == 1
        assert src.num_elements() == 2
        torch.testing.assert_close(
            src["x"][:2], torch.tensor([[2.0], [3.0]], device=device)
        )

    def test_compute_put_per_system_fit_mask(self):
        """compute_put_per_system_fit_mask writes fit_mask; put with it copies same set."""
        device = "cpu"
        src = SegmentedLevelStorage(
            data={
                "x": torch.tensor(
                    [[1.0], [2.0], [3.0], [4.0], [5.0]],
                    device=device,
                    dtype=torch.float32,
                ),
            },
            segment_lengths=[2, 3],
            device=device,
            validate=False,
        )
        dest = SegmentedLevelStorage(
            data={"x": torch.zeros(15, 1, device=device, dtype=torch.float32)},
            segment_lengths=[10],
            device=device,
            batch_ptr_capacity=5,
            validate=False,
        )
        source_mask = torch.tensor([True, True], device=device)
        fit_mask = torch.zeros(2, dtype=torch.bool, device=device)
        dest.compute_put_per_system_fit_mask(src, source_mask, None, fit_mask)
        assert fit_mask.sum().item() == 2
        assert fit_mask[0].item() is True
        assert fit_mask[1].item() is True
        dest.put(src, fit_mask)
        assert len(dest) == 3
        torch.testing.assert_close(dest["x"][10:12], src["x"][:2])
        torch.testing.assert_close(dest["x"][12:15], src["x"][2:5])

    def test_compute_put_per_system_fit_mask_no_batch_ptr_room(self):
        """compute_put_per_system_fit_mask zeros fit_mask when dest has no batch_ptr room."""
        device = "cpu"
        src = SegmentedLevelStorage(
            data={
                "x": torch.tensor([[1.0], [2.0]], device=device, dtype=torch.float32)
            },
            segment_lengths=[2],
            device=device,
            validate=False,
        )
        # dest batch_ptr length 2 only; need >= 1+1+2=4 to append 1 segment
        dest = SegmentedLevelStorage(
            data={"x": torch.zeros(10, 1, device=device, dtype=torch.float32)},
            segment_lengths=[0],
            device=device,
            validate=False,
        )
        source_mask = torch.tensor([True], device=device)
        fit_mask = torch.ones(1, dtype=torch.bool, device=device)
        dest.compute_put_per_system_fit_mask(src, source_mask, None, fit_mask)
        assert fit_mask.sum().item() == 0

    def test_put_defrag_fixed_tensor_shapes(self):
        """Data tensors are not expanded or trimmed by put or defrag (fixed storage)."""
        device = "cpu"
        src = SegmentedLevelStorage(
            data={
                "x": torch.tensor(
                    [[1.0], [2.0], [3.0], [4.0], [5.0]],
                    device=device,
                    dtype=torch.float32,
                ),
            },
            segment_lengths=[2, 3],
            device=device,
            validate=False,
        )
        dest = SegmentedLevelStorage(
            data={"x": torch.zeros(15, 1, device=device, dtype=torch.float32)},
            segment_lengths=[10],
            device=device,
            batch_ptr_capacity=5,
            validate=False,
        )
        shape_before = dest._data["x"].shape
        mask = torch.tensor([True, True], device=device)
        dest.put(src, mask)
        assert dest._data["x"].shape == shape_before
        copied = getattr(src, "_copied_mask", None)
        assert copied is not None
        src.defrag()
        assert src._data["x"].shape == (5, 1)

    def test_put_partial_copy_only_what_fits_copied_mask(self):
        """When dest has room for only 1 segment, put copies 1; copied_mask True only for that segment."""
        device = "cpu"
        src = SegmentedLevelStorage(
            data={
                "x": torch.tensor(
                    [[1.0], [2.0], [3.0], [4.0], [5.0]],
                    device=device,
                    dtype=torch.float32,
                ),
            },
            segment_lengths=[2, 3],
            device=device,
            validate=False,
        )
        dest = SegmentedLevelStorage(
            data={"x": torch.zeros(12, 1, device=device, dtype=torch.float32)},
            segment_lengths=[10],
            device=device,
            batch_ptr_capacity=5,
            validate=False,
        )
        mask = torch.tensor([True, True], device=device)
        copied_mask = torch.zeros(2, dtype=torch.bool, device=device)
        dest.put(src, mask, copied_mask=copied_mask)
        assert copied_mask[0].item() is True
        assert copied_mask[1].item() is False
        assert dest["x"][10].item() == 1.0
        assert dest["x"][11].item() == 2.0
        assert len(dest) == 2


# -----------------------------------------------------------------------------
# MultiLevelStorage
# -----------------------------------------------------------------------------
class TestMultiLevelStorage:
    """Tests for MultiLevelStorage (multi-group container)."""

    def test_empty_construction(self):
        m = MultiLevelStorage(attr_map=LevelSchema())
        assert len(m) == 0

    def test_from_data_factory(self):
        data = {
            "positions": torch.randn(10, 3),
            "atomic_numbers": torch.ones(10, dtype=torch.long),
            "cell": torch.eye(3).unsqueeze(0).expand(2, 3, 3),
            "energies": torch.randn(2),
        }
        # system-level: 2 graphs; atoms segmented as [4, 6]
        schema = LevelSchema()
        m = MultiLevelStorage.from_data(
            data=data,
            attr_map=schema,
            segment_lengths={"atoms": [4, 6]},
            device="cpu",
            validate=True,
        )
        assert "atoms" in m.groups
        assert "system" in m.groups
        assert len(m) == 2
        assert m.num_atoms == 10

    def test_routing_by_attr(self):
        atoms = UniformLevelStorage(
            data={"a": torch.randn(2, 1)},
            device="cpu",
            validate=False,
        )
        system = UniformLevelStorage(
            data={"e": torch.randn(2, 1)},
            device="cpu",
            validate=False,
        )
        schema = LevelSchema(
            group_to_attrs={"atoms": {"a"}, "system": {"e"}},
            segmented_groups=set(),
        )
        m = MultiLevelStorage(
            groups={"atoms": atoms, "system": system},
            attr_map=schema,
            validate=False,
        )
        assert m["a"].shape == (2, 1)
        assert m["e"].shape == (2, 1)
        assert m._group_name_from_attr("a") == "atoms"
        assert m.group_from_attr("e") is system

    def test_select_delegates_to_groups(self):
        atoms = SegmentedLevelStorage(
            data={"x": torch.randn(6, 1)},
            segment_lengths=[2, 4],
            device="cpu",
            validate=False,
        )
        system = UniformLevelStorage(
            data={"e": torch.randn(2, 1)},
            device="cpu",
            validate=False,
        )
        schema = LevelSchema(
            group_to_attrs={"atoms": {"x"}, "system": {"e"}},
            segmented_groups={"atoms"},
        )
        m = MultiLevelStorage(
            groups={"atoms": atoms, "system": system},
            attr_map=schema,
            validate=False,
        )
        sub = m[1]
        assert len(sub) == 1
        assert sub["x"].shape == (4, 1)
        assert sub["e"].shape == (1, 1)

    def test_setitem_creates_group_if_missing(self):
        schema = LevelSchema(group_to_attrs={"system": {"e"}}, segmented_groups=set())
        m = MultiLevelStorage(
            groups={
                "system": UniformLevelStorage(
                    data={"e": torch.randn(2, 1)}, device="cpu", validate=False
                )
            },
            attr_map=schema,
            validate=False,
        )
        m["e"] = torch.randn(2, 1)
        assert m["e"].shape == (2, 1)

    def test_setitem_segmented_group_not_in_batch_raises(self):
        """Setting a key in a segmented group that is not in the batch raises."""
        schema = LevelSchema(
            group_to_attrs={"atoms": {"positions"}, "system": {"e"}},
            segmented_groups={"atoms"},
        )
        m = MultiLevelStorage(
            groups={
                "system": UniformLevelStorage(
                    data={"e": torch.randn(2, 1)}, device="cpu", validate=False
                )
            },
            attr_map=schema,
            validate=False,
        )
        with pytest.raises(ValueError, match="segmented but not found in batch"):
            m["positions"] = torch.randn(5, 3)

    def test_to_device_clone(self):
        atoms = UniformLevelStorage(
            data={"a": torch.randn(2, 1)}, device="cpu", validate=False
        )
        m = MultiLevelStorage(
            groups={"atoms": atoms},
            attr_map=LevelSchema(
                group_to_attrs={"atoms": {"a"}}, segmented_groups=set()
            ),
            validate=False,
        )
        m.to_device("cpu")
        assert m.device.type == "cpu"
        c = m.clone()
        assert c.groups is not m.groups
        assert c["a"] is not m["a"]


# -----------------------------------------------------------------------------
# Constants / dtype mapping
# -----------------------------------------------------------------------------
class TestLevelStorageConstants:
    """Tests for module constants and helpers."""

    def test_torch_dtype_map_roundtrip(self):
        assert TORCH_DTYPE_MAP["float32"] == torch.float32
        assert TORCH_DTYPE_MAP["int64"] == torch.int64

    def test_default_attribute_map_has_expected_groups(self):
        assert "atoms" in DEFAULT_ATTRIBUTE_MAP
        assert "edges" in DEFAULT_ATTRIBUTE_MAP
        assert "system" in DEFAULT_ATTRIBUTE_MAP
        assert "positions" in DEFAULT_ATTRIBUTE_MAP["atoms"]
        assert "edge_index" in DEFAULT_ATTRIBUTE_MAP["edges"]

    def test_default_segmented_groups(self):
        assert DEFAULT_SEGMENTED_GROUPS == {"atoms", "edges"}
