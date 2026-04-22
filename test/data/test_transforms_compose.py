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
"""Unit tests for the :class:`Compose` transform composition helper."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.data.transforms import Compose


def _make_sample(n_atoms: int = 3) -> AtomicData:
    """Build a minimal :class:`AtomicData` with positions and atomic numbers."""
    return AtomicData(
        positions=torch.zeros(n_atoms, 3, dtype=torch.float32),
        atomic_numbers=torch.ones(n_atoms, dtype=torch.long),
    )


def _make_batch() -> Batch:
    """Build a small two-sample :class:`Batch`."""
    return Batch.from_data_list([_make_sample(3), _make_sample(4)])


def _shift_positions(
    data: AtomicData, metadata: dict[str, Any]
) -> tuple[AtomicData, dict[str, Any]]:
    """Return AtomicData with positions incremented by 1.0, metadata unchanged."""
    return (
        AtomicData(
            positions=data.positions + 1.0,
            atomic_numbers=data.atomic_numbers,
        ),
        metadata,
    )


def _append_tag_a(
    data: AtomicData, metadata: dict[str, Any]
) -> tuple[AtomicData, dict[str, Any]]:
    metadata.setdefault("order", []).append("a")
    return data, metadata


def _append_tag_b(
    data: AtomicData, metadata: dict[str, Any]
) -> tuple[AtomicData, dict[str, Any]]:
    metadata.setdefault("order", []).append("b")
    return data, metadata


def _append_tag_c(
    data: AtomicData, metadata: dict[str, Any]
) -> tuple[AtomicData, dict[str, Any]]:
    metadata.setdefault("order", []).append("c")
    return data, metadata


def _set_metadata_key(
    data: AtomicData, metadata: dict[str, Any]
) -> tuple[AtomicData, dict[str, Any]]:
    metadata["touched"] = True
    return data, metadata


def _assert_touched(
    data: AtomicData, metadata: dict[str, Any]
) -> tuple[AtomicData, dict[str, Any]]:
    assert metadata.get("touched") is True
    metadata["observed_touched"] = True
    return data, metadata


def _batch_shift(batch: Batch) -> Batch:
    """Return a new batch whose positions are shifted by 1.0."""
    shifted = [
        AtomicData(
            positions=sample.positions + 1.0,
            atomic_numbers=sample.atomic_numbers,
        )
        for sample in batch.to_data_list()
    ]
    return Batch.from_data_list(shifted)


def _batch_tag_a(batch: Batch) -> Batch:
    batch._compose_tag = getattr(batch, "_compose_tag", "") + "a"  # type: ignore[attr-defined]
    return batch


def _batch_tag_b(batch: Batch) -> Batch:
    batch._compose_tag = getattr(batch, "_compose_tag", "") + "b"  # type: ignore[attr-defined]
    return batch


def _batch_tag_c(batch: Batch) -> Batch:
    batch._compose_tag = getattr(batch, "_compose_tag", "") + "c"  # type: ignore[attr-defined]
    return batch


def _sample_raise(
    data: AtomicData, metadata: dict[str, Any]
) -> tuple[AtomicData, dict[str, Any]]:
    """Sample transform that always raises ValueError('nope')."""
    raise ValueError("nope")


def _batch_raise(batch: Batch) -> Batch:
    """Batch transform that always raises ValueError('nope')."""
    raise ValueError("nope")


class TestCompose:
    """Unit tests for :class:`Compose` across sample and batch arities."""

    def test_empty_sequence_is_identity_sample(self) -> None:
        data = _make_sample()
        metadata: dict[str, Any] = {"k": 1}
        out_data, out_meta = Compose([])(data, metadata)
        assert out_data is data
        assert out_meta is metadata

    def test_empty_sequence_is_identity_batch(self) -> None:
        batch = _make_batch()
        assert Compose([])(batch) is batch

    def test_three_transforms_left_to_right_sample(self) -> None:
        compose = Compose([_append_tag_a, _append_tag_b, _append_tag_c])
        _, metadata = compose(_make_sample(), {})
        assert metadata["order"] == ["a", "b", "c"]

    def test_three_transforms_left_to_right_batch(self) -> None:
        compose = Compose([_batch_tag_a, _batch_tag_b, _batch_tag_c])
        out = compose(_make_batch())
        assert out._compose_tag == "abc"

    def test_single_transform_sample_returns_tuple(self) -> None:
        out_data, out_meta = Compose([_shift_positions])(_make_sample(), {})
        assert torch.allclose(out_data.positions, torch.ones(3, 3))
        assert out_meta == {}

    def test_single_transform_batch_returns_batch(self) -> None:
        batch = _make_batch()
        out = Compose([_batch_shift])(batch)
        # Must be a Batch, not a 1-tuple wrapping one.
        assert isinstance(out, Batch)
        assert torch.allclose(out.positions, torch.ones_like(out.positions))

    def test_metadata_modification_propagates(self) -> None:
        _, metadata = Compose([_set_metadata_key, _assert_touched])(_make_sample(), {})
        assert metadata["touched"] is True
        assert metadata["observed_touched"] is True

    def test_transforms_stored_as_tuple(self) -> None:
        compose = Compose([_append_tag_a, _append_tag_b])
        assert isinstance(compose.transforms, tuple)
        assert len(compose.transforms) == 2

    def test_repr(self) -> None:
        text = repr(Compose([_append_tag_a, _append_tag_b]))
        assert text.startswith("Compose(")
        assert "_append_tag_a" in text
        assert "_append_tag_b" in text

    def test_accepts_any_sequence(self) -> None:
        assert Compose(tuple()).transforms == ()
        assert Compose(iter([_append_tag_a])).transforms == (_append_tag_a,)

    def test_named_functions_stored_not_lambdas(self) -> None:
        """Regression: composing top-level functions keeps them picklable."""
        import pickle

        compose = Compose([_append_tag_a, _append_tag_b])
        # Round-trip through pickle to catch accidental lambda storage.
        roundtripped = pickle.loads(pickle.dumps(compose))  # noqa: S301
        assert isinstance(roundtripped, Compose)

    @pytest.mark.parametrize(
        "bad_entry, bad_type_name",
        [("not a transform", "str"), (42, "int"), (None, "NoneType")],
    )
    def test_non_callable_raises_with_index(
        self, bad_entry: Any, bad_type_name: str
    ) -> None:
        with pytest.raises(TypeError, match=r"index 1") as excinfo:
            Compose([_append_tag_a, bad_entry])
        msg = str(excinfo.value)
        assert "Compose" in msg
        assert bad_type_name in msg

    @pytest.mark.parametrize(
        "transforms, args",
        [
            (
                [_append_tag_a, _sample_raise],
                (_make_sample(), {}),
            ),
            (
                [_batch_tag_a, _batch_raise],
                (_make_batch(),),
            ),
        ],
        ids=["sample", "batch"],
    )
    def test_error_wrapped_with_index(self, transforms: list, args: tuple) -> None:
        compose = Compose(transforms)
        with pytest.raises(RuntimeError, match=r"transform\[1\]") as excinfo:
            compose(*args)
        assert "Compose" in str(excinfo.value)
        assert "ValueError" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert str(excinfo.value.__cause__) == "nope"

    def test_is_batch_flag_set_correctly(self) -> None:
        assert Compose([_append_tag_a]).is_batch is False
        assert Compose([_batch_tag_a]).is_batch is True
        # Empty composition defaults to non-batch but is polymorphic at call time.
        assert Compose([]).is_batch is False

    def test_mixing_sample_and_batch_transforms_rejected(self) -> None:
        with pytest.raises(TypeError, match=r"mix sample and batch") as excinfo:
            Compose([_append_tag_a, _batch_tag_a])
        msg = str(excinfo.value)
        assert "index 0" in msg and "sample" in msg
        assert "index 1" in msg and "batch" in msg

    def test_mixing_batch_and_sample_transforms_rejected(self) -> None:
        with pytest.raises(TypeError, match=r"mix sample and batch"):
            Compose([_batch_tag_a, _append_tag_a])

    def test_typed_first_param_neither_atomicdata_nor_batch_rejected(self) -> None:
        def bad(data: int, metadata: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            return data, metadata

        with pytest.raises(
            TypeError, match=r"expected AtomicData .* or Batch"
        ) as excinfo:
            Compose([bad])
        assert "index 0" in str(excinfo.value)

    def test_typed_first_param_batch_param_but_wrong_type_rejected(self) -> None:
        def bad(b: int) -> int:
            return b

        with pytest.raises(TypeError, match=r"expected AtomicData .* or Batch"):
            Compose([bad])

    def test_untyped_callable_accepted_and_does_not_classify(self) -> None:
        # Fully untyped 2-arg callable is accepted and contributes no kind.
        def s(data, metadata):  # type: ignore[no-untyped-def]
            return data, metadata

        # Fully untyped 1-arg callable is likewise accepted.
        def b(batch):  # type: ignore[no-untyped-def]
            return batch

        # Both compositions are "sample" by default since no typed entry
        # sets the kind.
        assert Compose([s]).is_batch is False
        assert Compose([b]).is_batch is False

    def test_untyped_callable_takes_kind_from_typed_sibling(self) -> None:
        def untyped(batch):  # type: ignore[no-untyped-def]
            return batch

        # A typed batch sibling establishes the kind; the untyped callable
        # is accepted without contributing to the decision.
        compose = Compose([_batch_tag_a, untyped])
        assert compose.is_batch is True

    def test_class_based_callable_accepted(self) -> None:
        """Class instances with annotated __call__ are classified correctly."""

        class BatchOp:
            def __call__(self, batch: Batch) -> Batch:
                return batch

        class SampleOp:
            def __call__(
                self, data: AtomicData, metadata: dict[str, Any]
            ) -> tuple[AtomicData, dict[str, Any]]:
                return data, metadata

        assert Compose([BatchOp()]).is_batch is True
        assert Compose([SampleOp()]).is_batch is False

    def test_class_based_callable_annotation_mismatch_rejected(self) -> None:
        class Bad:
            def __call__(self, x: int) -> int:
                return x

        with pytest.raises(TypeError, match=r"expected AtomicData .* or Batch"):
            Compose([Bad()])

    def test_sample_compose_rejects_batch_call(self) -> None:
        compose = Compose([_append_tag_a])
        with pytest.raises(TypeError, match=r"sample composition requires"):
            compose(_make_batch())

    def test_batch_compose_rejects_sample_call(self) -> None:
        compose = Compose([_batch_tag_a])
        with pytest.raises(TypeError, match=r"batch composition takes a single"):
            compose(_make_sample(), {})
