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
"""Private scalar validation helpers for CSP."""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any


def positive_integer(value: Any, *, name: str) -> int:
    """Return a positive integral value as a Python int.

    Parameters
    ----------
    value : Any
        Integral value to validate. Booleans are rejected.
    name : str
        Parameter name included in validation errors.

    Returns
    -------
    int
        The validated value converted to a Python integer.

    Raises
    ------
    TypeError
        If the value is boolean or is not an Integral.
    ValueError
        If the value is not positive.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def nonnegative_int(value: Any, name: str) -> int:
    """Require a nonnegative integer option, excluding booleans.

    Parameters
    ----------
    value : Any
        Value supplied for the option.
    name : str
        Public option name used in validation errors.

    Returns
    -------
    int
        Validated nonnegative integer.

    Raises
    ------
    ValueError
        If ``value`` is boolean, not an integer, or negative.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def finite_nonnegative(value: Any, name: str) -> float:
    """Normalize a finite nonnegative real option to Python float.

    Parameters
    ----------
    value : Any
        Value supplied for the option.
    name : str
        Public option name used in validation errors.

    Returns
    -------
    float
        Validated value normalized to Python float.

    Raises
    ------
    TypeError
        If the value is boolean or is not numeric.
    ValueError
        If the value is negative or nonfinite.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def finite_positive(value: Any, name: str) -> float:
    """Return a finite positive number as a Python float.

    Parameters
    ----------
    value : Any
        Python int or float to validate. Booleans are rejected.
    name : str
        Parameter name included in validation errors.

    Returns
    -------
    float
        The validated value converted to a Python float.

    Raises
    ------
    ValueError
        If the value is boolean, nonnumeric, nonfinite, or not positive.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")
    return float(value)
