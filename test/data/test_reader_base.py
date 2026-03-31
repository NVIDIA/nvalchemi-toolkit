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
Unit tests for the abstract ``Reader`` base class.

Tests cover field-name inspection, positive/negative indexing, bounds
checking, pin-memory behaviour, metadata injection, iteration with
error wrapping, context-manager protocol, and ``__repr__``.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data.datapipes.backends.base import Reader

# ---------------------------------------------------------------------------
# Concrete helpers
# ---------------------------------------------------------------------------


class MinimalReader(Reader):
    """Minimal concrete Reader implementation for testing."""

    def __init__(
        self,
        data=None,
        *,
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
    ) -> None:
        super().__init__(
            pin_memory=pin_memory,
            include_index_in_metadata=include_index_in_metadata,
        )
        # Default: three samples with a single float tensor each.
        self._data = (
            data
            if data is not None
            else [{"x": torch.tensor([float(i)])} for i in range(3)]
        )

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        return self._data[index]

    def __len__(self) -> int:
        return len(self._data)


class FailingReader(MinimalReader):
    """Reader that raises ``ValueError`` when a specific index is loaded."""

    def __init__(self, fail_on: int = 1) -> None:
        super().__init__()
        self._fail_on = fail_on

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        if index == self._fail_on:
            raise ValueError(f"Intentional failure on index {index}")
        return super()._load_sample(index)


class CloseTrackingReader(MinimalReader):
    """Reader that records whether ``close()`` has been called."""

    def __init__(self) -> None:
        super().__init__()
        self.close_called = False

    def close(self) -> None:
        self.close_called = True


# ---------------------------------------------------------------------------
# TestReaderFieldNames
# ---------------------------------------------------------------------------


class TestReaderFieldNames:
    """Tests for _get_field_names and the field_names property."""

    def test_field_names_empty_reader(self):
        """An empty reader returns an empty list of field names (lines 113-114)."""
        reader = MinimalReader([])
        assert reader.field_names == []

    def test_get_field_names_empty_reader(self):
        """_get_field_names directly returns [] when len is 0."""
        reader = MinimalReader([])
        assert reader._get_field_names() == []

    def test_field_names_nonempty(self):
        """field_names reflects the keys present in the first sample (line 145)."""
        data = [{"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}]
        reader = MinimalReader(data)
        names = reader.field_names
        assert "a" in names
        assert "b" in names

    def test_get_field_names_reflects_first_sample(self):
        """_get_field_names returns the keys of the first sample."""
        data = [{"alpha": torch.zeros(3), "beta": torch.ones(2)}]
        reader = MinimalReader(data)
        names = reader._get_field_names()
        assert set(names) == {"alpha", "beta"}


# ---------------------------------------------------------------------------
# TestReaderGetItem
# ---------------------------------------------------------------------------


class TestReaderGetItem:
    """Tests for Reader.__getitem__ behaviour."""

    def test_positive_index(self):
        """Positive index loads the expected sample."""
        reader = MinimalReader()
        data_dict, _ = reader[0]
        assert torch.allclose(data_dict["x"], torch.tensor([0.0]))

    def test_negative_index(self):
        """Negative index is converted to the equivalent positive index (line 169)."""
        reader = MinimalReader()
        data_last_via_negative, _ = reader[-1]
        data_last_via_positive, _ = reader[len(reader) - 1]
        assert torch.allclose(data_last_via_negative["x"], data_last_via_positive["x"])

    def test_out_of_range_positive_raises_index_error(self):
        """A positive index beyond the dataset length raises IndexError (line 171)."""
        reader = MinimalReader()
        with pytest.raises(IndexError):
            reader[100]

    def test_out_of_range_negative_raises_index_error(self):
        """A negative index beyond the dataset length raises IndexError."""
        reader = MinimalReader()
        with pytest.raises(IndexError):
            reader[-100]

    def test_returns_tuple_of_two(self):
        """__getitem__ returns a 2-tuple of (data_dict, metadata)."""
        reader = MinimalReader()
        result = reader[0]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_data_dict_correct(self):
        """The data dict portion contains the expected tensor value."""
        reader = MinimalReader()
        data_dict, _ = reader[1]
        assert torch.allclose(data_dict["x"], torch.tensor([1.0]))

    def test_include_index_in_metadata_true(self):
        """When include_index_in_metadata=True, metadata contains 'index'."""
        reader = MinimalReader(include_index_in_metadata=True)
        _, metadata = reader[2]
        assert "index" in metadata
        assert metadata["index"] == 2

    def test_include_index_in_metadata_false(self):
        """When include_index_in_metadata=False, 'index' is absent from metadata (lines 177→180)."""
        reader = MinimalReader(include_index_in_metadata=False)
        _, metadata = reader[0]
        assert "index" not in metadata

    def test_pin_memory_false_default(self):
        """By default tensors are not pinned."""
        reader = MinimalReader()
        data_dict, _ = reader[0]
        assert not data_dict["x"].is_pinned()

    def test_pin_memory_true_pins_tensors(self):
        """With pin_memory=True the loaded tensors are in pinned memory (line 181)."""
        reader = MinimalReader(pin_memory=True)
        data_dict, _ = reader[0]
        assert data_dict["x"].is_pinned()


# ---------------------------------------------------------------------------
# TestReaderGetSampleMetadata
# ---------------------------------------------------------------------------


class TestReaderGetSampleMetadata:
    """Tests for the default _get_sample_metadata implementation."""

    def test_default_metadata_is_empty_dict(self):
        """The base implementation returns an empty dict (line 134)."""
        reader = MinimalReader()
        assert reader._get_sample_metadata(0) == {}


# ---------------------------------------------------------------------------
# TestReaderIteration
# ---------------------------------------------------------------------------


class TestReaderIteration:
    """Tests for Reader.__iter__ and its error-wrapping behaviour."""

    def test_iterates_all_samples(self):
        """Iterating over a reader yields every sample exactly once."""
        reader = MinimalReader()
        results = list(reader)
        assert len(results) == len(reader)
        for i, (data_dict, _) in enumerate(results):
            assert torch.allclose(data_dict["x"], torch.tensor([float(i)]))

    def test_iteration_error_raises_runtime_error(self):
        """A failing sample causes __iter__ to raise RuntimeError (lines 198-204)."""
        reader = FailingReader(fail_on=1)
        with pytest.raises(RuntimeError):
            list(reader)

    def test_runtime_error_wraps_original_cause(self):
        """The RuntimeError chains the original exception as its __cause__."""
        reader = FailingReader(fail_on=1)
        with pytest.raises(RuntimeError) as exc_info:
            list(reader)
        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, ValueError)


# ---------------------------------------------------------------------------
# TestReaderContextManager
# ---------------------------------------------------------------------------


class TestReaderContextManager:
    """Tests for the context-manager protocol (__enter__ / __exit__)."""

    def test_enter_returns_self(self):
        """__enter__ returns the reader itself."""
        reader = MinimalReader()
        with reader as r:
            assert r is reader

    def test_exit_calls_close(self):
        """__exit__ invokes close() on the reader."""
        reader = CloseTrackingReader()
        assert not reader.close_called
        with reader:
            pass
        assert reader.close_called


# ---------------------------------------------------------------------------
# TestReaderRepr
# ---------------------------------------------------------------------------


class TestReaderRepr:
    """Tests for Reader.__repr__ (line 247)."""

    def test_repr_contains_class_name(self):
        """repr includes the concrete class name."""
        reader = MinimalReader()
        assert "MinimalReader" in repr(reader)

    def test_repr_contains_len(self):
        """repr contains the dataset length."""
        reader = MinimalReader()
        assert str(len(reader)) in repr(reader)

    def test_repr_contains_pin_memory(self):
        """repr includes the pin_memory flag value."""
        reader_no_pin = MinimalReader(pin_memory=False)
        reader_pin = MinimalReader(pin_memory=True)
        assert "False" in repr(reader_no_pin)
        assert "True" in repr(reader_pin)
