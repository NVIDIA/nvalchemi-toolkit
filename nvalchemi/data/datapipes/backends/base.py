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
"""Abstract base class for datapipe readers."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any

import torch

logger = logging.getLogger(__name__)


class Reader(ABC):
    """Abstract base class for data readers.

    Readers are intentionally simple and transactional:

    - Load data from a source (file, database, etc.)
    - Return ``(dict[str, torch.Tensor], metadata_dict)`` tuples with CPU tensors
    - No threading, no prefetching, no device transfers

    Subclasses must implement :meth:`_load_sample` and :meth:`__len__`.

    Parameters
    ----------
    pin_memory : bool, default=False
        If True, pin loaded tensors to page-locked memory for faster
        async CPU→GPU transfers.
    include_index_in_metadata : bool, default=True
        If True, automatically add ``"index"`` to each sample's metadata dict.

    Examples
    --------
    >>> class MyReader(Reader):
    ...     def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
    ...         return {"x": torch.randn(3)}
    ...     def __len__(self) -> int:
    ...         return 10
    >>> reader = MyReader()  # doctest: +SKIP
    >>> data, meta = reader[0]  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        pin_memory: bool = False,
        include_index_in_metadata: bool = True,
    ) -> None:
        """Initialize the Reader base class.

        Parameters
        ----------
        pin_memory : bool, default=False
            If True, pin loaded tensors to page-locked memory for faster
            async CPU→GPU transfers.
        include_index_in_metadata : bool, default=True
            If True, automatically add ``"index"`` to each sample's metadata dict.
        """
        self.pin_memory = pin_memory
        self.include_index_in_metadata = include_index_in_metadata

    @abstractmethod
    def _load_sample(self, index: int) -> dict[str, torch.Tensor]:
        """Load raw tensor data for a single sample.

        Parameters
        ----------
        index : int
            Sample index (0-based).

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping of field names to CPU tensors.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of available samples.

        Returns
        -------
        int
            Number of samples.
        """
        raise NotImplementedError

    def _get_field_names(self) -> list[str]:
        """Return field names by inspecting the first sample.

        Returns
        -------
        list[str]
            Field names from the first sample, or empty if reader is empty.
        """
        if len(self) == 0:
            return []
        data = self._load_sample(0)
        return list(data.keys())

    def _get_sample_metadata(self, index: int) -> dict[str, Any]:
        """Return additional metadata for a sample.

        Override in subclasses to provide per-sample metadata such as
        source file paths or physical indices.

        Parameters
        ----------
        index : int
            Sample index.

        Returns
        -------
        dict[str, Any]
            Metadata dictionary.  Empty by default.
        """
        return {}

    @property
    def field_names(self) -> list[str]:
        """Field names available in each sample.

        Returns
        -------
        list[str]
            Field names.
        """
        return self._get_field_names()

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        """Load a sample and its metadata by index.

        Handles negative indexing, bounds checking, optional pin-memory,
        and automatic index injection into metadata.

        Parameters
        ----------
        index : int
            Sample index. Negative values are supported.

        Returns
        -------
        tuple[dict[str, torch.Tensor], dict[str, Any]]
            ``(data_dict, metadata)`` pair with CPU tensors.

        Raises
        ------
        IndexError
            If *index* is out of range.
        """
        if index < 0:
            index = len(self) + index
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for reader with {len(self)} samples"
            )

        data_dict = self._load_sample(index)
        metadata = self._get_sample_metadata(index)
        if self.include_index_in_metadata:
            metadata["index"] = index

        if self.pin_memory:
            data_dict = {k: v.pin_memory() for k, v in data_dict.items()}

        return data_dict, metadata

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], dict[str, Any]]]:
        """Iterate over all samples sequentially.

        Yields
        ------
        tuple[dict[str, torch.Tensor], dict[str, Any]]
            ``(data_dict, metadata)`` for each sample.

        Raises
        ------
        RuntimeError
            If any sample fails to load.
        """
        for i in range(len(self)):
            try:
                yield self[i]
            except Exception as e:
                error_msg = f"Sample {i} failed with exception: {type(e).__name__}: {e}"
                logger.error(error_msg)
                raise RuntimeError(error_msg) from e

    def close(self) -> None:
        """Release resources held by the reader.

        Override in subclasses to close file handles, connections, etc.
        """
        pass

    def __enter__(self) -> Reader:
        """Enter context manager.

        Returns
        -------
        Reader
            This reader instance.
        """
        return self

    def __exit__(
        self, exc_type: type | None, exc_val: BaseException | None, exc_tb: Any
    ) -> None:
        """Exit context manager, calling :meth:`close`.

        Parameters
        ----------
        exc_type : type | None
            Exception type, if any.
        exc_val : BaseException | None
            Exception value, if any.
        exc_tb : Any
            Exception traceback, if any.
        """
        self.close()

    def __repr__(self) -> str:
        """Return a string representation of the reader.

        Returns
        -------
        str
            Human-readable summary.
        """
        return (
            f"{self.__class__.__name__}(len={len(self)}, pin_memory={self.pin_memory})"
        )
