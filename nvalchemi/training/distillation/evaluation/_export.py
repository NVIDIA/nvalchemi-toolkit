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

Every measurement exports with ``to_dict`` and rebuilds with ``from_dict``, so a
sweep can persist each student's results and aggregate them later. The rebuild
is shared here because a JSON round trip introduces the same asymmetries
everywhere: fields the export dropped as unmeasured, tuples that come back as
lists, and non-finite floats a strict JSON writer had to spell as strings.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any, TypeVar

_Metric = TypeVar("_Metric")

_NONFINITE_TOKENS: dict[str, float] = {
    "nan": math.nan,
    "inf": math.inf,
    "-inf": -math.inf,
}
"""Strings a strict JSON document spells the non-finite floats as, and their values."""


def _json_token(value: float) -> float | str:
    """Return *value*, or the string spelling a strict JSON reader can hold for it.

    ``json.dumps`` writes ``NaN`` and ``Infinity`` as bare tokens that are an
    extension to JSON, so a report carrying an unmeasurable metric would land
    as a file a strict reader rejects. The spelling keeps the reason a bar
    failed visible, where ``null`` would read as a measurement never taken, and
    :func:`_as_declared` reads it back on a float field.
    """
    if math.isfinite(value):
        return value
    if math.isnan(value):
        return "nan"
    return "inf" if value > 0 else "-inf"


def _as_declared(field: dataclasses.Field, value: Any) -> Any:
    """Return *value* as its field declares: a tuple for a list, a float for a spelled one.

    Annotations are strings under postponed evaluation, so the declared type is
    matched by its text. A non-finite spelling is decoded only on a float
    field, so a string field that happens to read ``"nan"`` is left alone.
    """
    declared = str(field.type)
    if isinstance(value, list) and declared.startswith("tuple"):
        return tuple(value)
    if isinstance(value, str) and value in _NONFINITE_TOKENS and "float" in declared:
        return _NONFINITE_TOKENS[value]
    return value


def _rebuild(cls: type[_Metric], data: Mapping[str, Any]) -> _Metric:
    """Return an instance of the measurement dataclass *cls* from an export.

    Keys *cls* does not declare are rejected rather than dropped, so an export
    written by a different version fails where it is read instead of
    rebuilding into a silently incomplete object.

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
