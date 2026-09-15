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
"""Rebuild of the evaluation suite's measurement dataclasses from their exports.

Every measurement here exports with ``to_dict`` and rebuilds with
``from_dict``, so a sweep that evaluates each student in its own job can persist
the results and aggregate them into one acceptance report later. The rebuild is
shared from this module because the measurements are spread across four of them
and a round trip through JSON introduces the same two asymmetries everywhere:
fields the export drops because they were never measured, and tuples that come
back as lists.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any, TypeVar

_Metric = TypeVar("_Metric")


def _as_declared(field: dataclasses.Field, value: Any) -> Any:
    """Return a list value as the tuple its field declares, else the value itself.

    Annotations are strings under postponed evaluation, which is all it takes
    to tell a tuple field from the list a JSON round trip left in its place.
    """
    if isinstance(value, list) and str(field.type).startswith("tuple"):
        return tuple(value)
    return value


def _rebuild(cls: type[_Metric], data: Mapping[str, Any]) -> _Metric:
    """Return an instance of the measurement dataclass *cls* from an export.

    Parameters
    ----------
    cls : type
        Dataclass to rebuild.
    data : Mapping[str, Any]
        Mapping produced by the class's own ``to_dict``. Keys the class does
        not declare are rejected rather than dropped, so an export written by a
        different version fails where it is read instead of rebuilding into a
        silently incomplete object.

    Returns
    -------
    object
        Instance of *cls* equal to the one the export came from.

    Raises
    ------
    ValueError
        If *data* carries a key *cls* does not declare, or omits one of its
        fields that has no default.
    """
    fields = {field.name: field for field in dataclasses.fields(cls)}
    unknown = sorted(set(data) - set(fields))
    if unknown:
        raise ValueError(
            f"{cls.__name__} cannot be rebuilt from a mapping carrying "
            f"{unknown!r}; expected keys from {sorted(fields)!r}."
        )
    missing = sorted(
        name
        for name, field in fields.items()
        if name not in data
        and field.default is dataclasses.MISSING
        and field.default_factory is dataclasses.MISSING
    )
    if missing:
        raise ValueError(
            f"{cls.__name__} cannot be rebuilt from a mapping missing the "
            f"required {missing!r}."
        )
    return cls(**{key: _as_declared(fields[key], value) for key, value in data.items()})
