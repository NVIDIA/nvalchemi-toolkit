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
"""Data collection utilities for splitting readers and constructing loaders."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping, Sequence
from enum import Enum
from itertools import accumulate
from typing import Any, Protocol, runtime_checkable

import torch

from nvalchemi.data.datapipes.backends.base import Reader
from nvalchemi.data.datapipes.dataloader import DataLoader
from nvalchemi.data.datapipes.dataset import Dataset


class Split(str, Enum):
    """Predefined dataset split names.

    Using a ``str`` enum ensures values are usable as dict keys and
    print as human-readable strings.
    """

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SubsetReader(Reader):
    """A reader that exposes a subset of a parent reader's samples.

    Parameters
    ----------
    parent : Reader
        The underlying reader to draw samples from.
    indices : Sequence[int]
        Indices into the parent reader.  All values must be in
        ``[0, len(parent))``.

    Raises
    ------
    ValueError
        If any index is out of range for the parent reader.
    """

    def __init__(self, parent: Reader, indices: Sequence[int]) -> None:
        super().__init__(
            pin_memory=parent.pin_memory,
            include_index_in_metadata=parent.include_index_in_metadata,
        )
        parent_len = len(parent)
        for idx in indices:
            if idx < 0 or idx >= parent_len:
                raise ValueError(
                    f"Index {idx} out of range for parent reader with "
                    f"{parent_len} samples"
                )
        self._parent = parent
        self._indices = list(indices)

    def __len__(self) -> int:
        """Return the number of samples in the subset."""
        return len(self._indices)

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Load a sample from the parent reader at the remapped index."""
        parent_idx = self._indices[index]
        return self._parent._load_sample(parent_idx)

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return metadata from the parent reader at the remapped index."""
        parent_idx = self._indices[index]
        return self._parent._get_sample_metadata(parent_idx)

    def close(self) -> None:
        """Close the parent reader."""
        self._parent.close()


class ConcatReader(Reader):
    """Concatenate multiple readers into a single contiguous reader.

    Samples are numbered contiguously: indices ``[0, len(readers[0]))``
    map to the first reader, ``[len(readers[0]), len(readers[0]) +
    len(readers[1]))`` to the second, and so on.

    Parameters
    ----------
    readers : Sequence[Reader]
        One or more readers to concatenate.  All readers must expose
        the same ``field_names``.

    Raises
    ------
    ValueError
        If *readers* is empty or if the readers expose different field
        names.
    """

    def __init__(self, readers: Sequence[Reader]) -> None:
        if len(readers) == 0:
            raise ValueError("ConcatReader requires at least one reader")

        first = readers[0]
        super().__init__(
            pin_memory=first.pin_memory,
            include_index_in_metadata=first.include_index_in_metadata,
        )

        expected_fields = set(first.field_names)
        for i, r in enumerate(readers[1:], 1):
            actual_fields = set(r.field_names)
            if actual_fields != expected_fields:
                raise ValueError(
                    f"Reader {i} has field names {sorted(actual_fields)} "
                    f"which differ from reader 0's {sorted(expected_fields)}"
                )

        self._readers = list(readers)
        self._offsets = [0] + list(accumulate(len(r) for r in self._readers))

    def __len__(self) -> int:
        """Return the total number of samples across all readers."""
        return self._offsets[-1]

    def _resolve(self, index: int) -> tuple[int, int]:
        """Map a global index to ``(reader_index, local_index)``."""
        reader_idx = bisect_right(self._offsets, index) - 1
        local_idx = index - self._offsets[reader_idx]
        return reader_idx, local_idx

    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Load a sample from the appropriate sub-reader."""
        reader_idx, local_idx = self._resolve(index)
        return self._readers[reader_idx]._load_sample(local_idx)

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return metadata from the sub-reader, enriched with reader_index."""
        reader_idx, local_idx = self._resolve(index)
        meta = self._readers[reader_idx]._get_sample_metadata(local_idx)
        meta["reader_index"] = reader_idx
        return meta

    def close(self) -> None:
        """Close all sub-readers."""
        for r in self._readers:
            r.close()


@runtime_checkable
class SplitStrategy(Protocol):
    """Protocol for dataset splitting strategies.

    A split strategy partitions ``n`` samples into named subsets
    according to the given *ratios*.
    """

    def __call__(
        self, n: int, ratios: dict[Split, float], seed: int
    ) -> dict[Split, list[int]]:
        """Partition indices ``[0, n)`` into named splits.

        Parameters
        ----------
        n : int
            Total number of samples.
        ratios : dict[Split, float]
            Mapping of split names to their fractional sizes.
            Values must sum to at most 1.0.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict[Split, list[int]]
            Mapping of split names to lists of sample indices.
        """
        ...


class RandomSplit:
    """Randomly partition indices using a seeded permutation.

    Ratios are allocated proportionally; any remainder from
    rounding is discarded.

    Raises
    ------
    ValueError
        If ratios sum to more than 1.0.
    """

    def __call__(
        self, n: int, ratios: dict[Split, float], seed: int
    ) -> dict[Split, list[int]]:
        """Partition indices ``[0, n)`` into named splits.

        Parameters
        ----------
        n : int
            Total number of samples.
        ratios : dict[Split, float]
            Mapping of split names to fractional sizes.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict[Split, list[int]]
            Mapping of split names to lists of sample indices.
        """
        total_ratio = sum(ratios.values())
        if total_ratio > 1.0 + 1e-9:
            raise ValueError(f"Ratios sum to {total_ratio:.4f}, which exceeds 1.0")

        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(n, generator=gen).tolist()

        splits: dict[Split, list[int]] = {}
        offset = 0
        for name, ratio in ratios.items():
            count = int(n * ratio)
            splits[name] = perm[offset : offset + count]
            offset += count
        return splits


def _coerce_reader(value: Reader | Sequence[Reader]) -> Reader:
    """Wrap a sequence of readers into a ConcatReader if needed."""
    if isinstance(value, Reader):
        return value
    readers = list(value)
    if len(readers) == 1:
        return readers[0]
    return ConcatReader(readers)


def _validate_split_keys(keys: Any) -> None:
    """Raise ``ValueError`` if any key is not a :class:`Split` member."""
    for k in keys:
        if not isinstance(k, Split):
            valid = ", ".join(s.value for s in Split)
            raise ValueError(
                f"Invalid split key {k!r}. Must be a Split enum member ({valid})."
            )


class DataCollection:
    """Manages train/val/test splits and provides Dataset and DataLoader access.

    There are two usage modes:

    **Unsplit (inference)** — Pass a single reader (or list) with no
    *splits*.  All data is accessible via :meth:`get_reader`,
    :meth:`get_dataset`, and :meth:`get_loader` without specifying a
    split name.

    **Split (training)** — Either provide a dict of ``Split`` →
    reader(s) (pre-split) or a single reader with *splits* ratios and a
    *split_strategy* to partition automatically.

    Parameters
    ----------
    reader : Reader | Sequence[Reader] | Mapping[Split, Reader | Sequence[Reader]]
        A single reader, a list of readers (auto-concatenated), or a
        mapping of :class:`Split` keys to readers or lists of readers.
    splits : dict[Split, float] | None
        Fractional split ratios.  Required when using
        *split_strategy* to partition a single reader.  Ignored when
        *reader* is a mapping.
    split_strategy : SplitStrategy | None
        Strategy used to partition indices.  Must be provided together
        with *splits*; there is no implicit default.
    seed : int
        Random seed passed to the split strategy.
    device : torch.device | str | None
        Target device for Datasets created by :meth:`get_dataset`.
        Accepts a :class:`torch.device`, a device-type string
        (``"cuda"``, ``"cpu"``), or ``None`` (auto-detect).  When a
        bare device-type string is stored (e.g. ``"cuda"``), the actual
        device index is resolved lazily by :class:`Dataset`, which lets
        multi-rank runs pick the correct GPU.

    Raises
    ------
    ValueError
        If *splits* is provided without a *split_strategy*, or if any
        dict key is not a valid :class:`Split` member.
    """

    def __init__(
        self,
        reader: Reader | Sequence[Reader] | Mapping[Split, Reader | Sequence[Reader]],
        splits: dict[Split, float] | None = None,
        split_strategy: SplitStrategy | None = None,
        seed: int = 42,
        device: torch.device | str | None = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device: torch.device | str = device
        self._readers: dict[Split, Reader] | None = None
        self._unsplit_reader: Reader | None = None

        if isinstance(reader, Mapping):
            _validate_split_keys(reader.keys())
            self._readers = {
                Split(name): _coerce_reader(r) for name, r in reader.items()
            }
        else:
            coerced = _coerce_reader(reader)
            if splits is not None:
                _validate_split_keys(splits.keys())
                if split_strategy is None:
                    raise ValueError(
                        "split_strategy must be provided when splits are "
                        "specified. Use RandomSplit() for shuffled "
                        "partitioning."
                    )
                index_map = split_strategy(len(coerced), splits, seed)
                self._readers = {
                    name: SubsetReader(coerced, indices)
                    for name, indices in index_map.items()
                }
            else:
                self._unsplit_reader = coerced

    @property
    def is_split(self) -> bool:
        """Whether this collection has named splits."""
        return self._readers is not None

    @property
    def split_names(self) -> list[Split]:
        """Return the names of available splits.

        Raises
        ------
        RuntimeError
            If the collection is unsplit (no splits were defined).
        """
        if self._readers is None:
            raise RuntimeError(
                "Collection has no named splits. Use get_reader() / "
                "get_dataset() / get_loader() without a split argument."
            )
        return list(self._readers.keys())

    def get_reader(self, split: Split | None = None) -> Reader:
        """Return the :class:`Reader` for the given split.

        Parameters
        ----------
        split : Split | None
            Split name.  Omit for unsplit collections.

        Returns
        -------
        Reader

        Raises
        ------
        KeyError
            If the requested split does not exist.
        ValueError
            If *split* is not a :class:`Split` member.
        RuntimeError
            If *split* is provided on an unsplit collection or omitted
            on a split collection.
        """
        if self._readers is not None:
            if split is None:
                raise RuntimeError(
                    "Collection has named splits; pass a Split member "
                    f"(available: {self.split_names})."
                )
            if not isinstance(split, Split):
                raise ValueError(f"Expected a Split enum member, got {split!r}.")
            if split not in self._readers:
                raise KeyError(
                    f"Split {split.value!r} not in collection. "
                    f"Available: {[s.value for s in self._readers]}."
                )
            return self._readers[split]

        # Unsplit mode
        if split is not None:
            raise RuntimeError(
                "Collection has no named splits. Call get_reader() "
                "without a split argument."
            )
        # Invariant: one of _readers or _unsplit_reader is always set.
        return self._unsplit_reader  # type: ignore[return-value]

    def get_dataset(self, split: Split | None = None, **kwargs: Any) -> Dataset:
        """Build a :class:`Dataset` for the given split.

        Parameters
        ----------
        split : Split | None
            Split name.  Omit for unsplit collections.
        **kwargs
            Forwarded to :class:`Dataset` (e.g. ``num_workers``).

        Returns
        -------
        Dataset
        """
        reader = self.get_reader(split)
        kwargs.setdefault("device", self._device)
        return Dataset(reader, **kwargs)

    def get_loader(self, split: Split | None = None, **kwargs: Any) -> DataLoader:
        """Build a :class:`DataLoader` for the given split.

        Parameters
        ----------
        split : Split | None
            Split name.  Omit for unsplit collections.
        **kwargs
            Forwarded to :class:`DataLoader`.  ``dataset_kwargs`` is
            popped and forwarded to :meth:`get_dataset`.

        Returns
        -------
        DataLoader
        """
        dataset_kwargs = kwargs.pop("dataset_kwargs", {})
        dataset = self.get_dataset(split, **dataset_kwargs)
        return DataLoader(dataset, **kwargs)

    # -- convenience properties / methods for common splits --

    @property
    def train_dataset(self) -> Dataset:
        """Dataset for the ``"train"`` split."""
        return self.get_dataset(Split.TRAIN)

    @property
    def val_dataset(self) -> Dataset:
        """Dataset for the ``"val"`` split."""
        return self.get_dataset(Split.VAL)

    @property
    def test_dataset(self) -> Dataset:
        """Dataset for the ``"test"`` split."""
        return self.get_dataset(Split.TEST)

    def train_loader(self, **kwargs: Any) -> DataLoader:
        """DataLoader for ``"train"`` with shuffle and drop_last defaults."""
        kwargs.setdefault("shuffle", True)
        kwargs.setdefault("drop_last", True)
        return self.get_loader(Split.TRAIN, **kwargs)

    def val_loader(self, **kwargs: Any) -> DataLoader:
        """DataLoader for ``"val"`` with non-shuffled defaults."""
        kwargs.setdefault("shuffle", False)
        kwargs.setdefault("drop_last", False)
        return self.get_loader(Split.VAL, **kwargs)

    def test_loader(self, **kwargs: Any) -> DataLoader:
        """DataLoader for ``"test"`` with non-shuffled defaults."""
        kwargs.setdefault("shuffle", False)
        kwargs.setdefault("drop_last", False)
        return self.get_loader(Split.TEST, **kwargs)
