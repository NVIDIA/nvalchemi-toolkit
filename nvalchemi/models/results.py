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
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, MutableMapping
from typing import Any

import torch


class CalculatorResults(MutableMapping[str, Any]):
    """Thin dict-like container for derived calculator outputs.

    Acts as a ``MutableMapping[str, Any]`` backed by a plain ``dict``.
    Supports additive merging: when two steps produce the same key and
    the key is declared as *additive*, the values are summed.
    """

    def __init__(self, initial: Mapping[str, Any] | None = None) -> None:
        """Initialise results from an optional seed mapping.

        Parameters
        ----------
        initial
            Seed key/value pairs copied into the container.
        """

        self._data: dict[str, Any] = dict(initial or {})

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self._data))
        return f"CalculatorResults(keys=[{keys}])"

    def copy(self) -> "CalculatorResults":
        """Return a shallow copy of the current results."""

        return CalculatorResults(self._data.copy())

    def merge(
        self,
        other: Mapping[str, Any] | "CalculatorResults",
        *,
        additive_keys: Iterable[str] = (),
    ) -> None:
        """Merge *other* into this container.

        Parameters
        ----------
        other
            Source mapping whose entries are merged in.
        additive_keys
            Keys whose values are *summed* when already present.
            All other keys use simple overwrite semantics.
        """

        additive = frozenset(additive_keys)
        source = other._data if isinstance(other, CalculatorResults) else dict(other)
        for key, value in source.items():
            if key in additive and key in self._data and value is not None:
                self._data[key] = self._sum_values(self._data[key], value)
            else:
                self._data[key] = value

    @staticmethod
    def _sum_values(left: Any, right: Any) -> Any:
        """Sum two result values with minimal type assumptions."""

        if left is None:
            return right
        if right is None:
            return left
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
            return left + right
        return left + right
