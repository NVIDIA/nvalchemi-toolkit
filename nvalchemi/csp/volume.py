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

"""Crystallography Open Database (COD)-derived atomic volume data and
formula-unit volume estimates.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Integral, Real
from typing import TYPE_CHECKING

import torch

from nvalchemi.csp._data_tables import ATOMIC_VOLUMES

if TYPE_CHECKING:
    from torch import Tensor


def get_default_atomic_volumes() -> dict[int, float]:
    """Return a fresh copy of the bundled COD-derived atomic volume table.

    Returns
    -------
    atomic_volumes : dict[int, float]
        Per-atom volume contributions in Å³, keyed by atomic number. The
        returned mapping is independent of library state and may be modified.
    """
    return ATOMIC_VOLUMES.copy()


def _validated_atomic_volumes(atomic_volumes: Mapping[int, float]) -> dict[int, float]:
    """Copy a table of finite positive per-atom volumes keyed by atomic number."""
    if not isinstance(atomic_volumes, Mapping):
        raise TypeError("atomic_volumes must be a mapping")
    validated: dict[int, float] = {}
    for key, value in atomic_volumes.items():
        if isinstance(key, bool) or not isinstance(key, Integral):
            raise TypeError("atomic_volumes keys must be integral atomic numbers")
        number = int(key)
        if not 1 <= number <= 118:
            raise ValueError(
                "atomic_volumes keys must be valid atomic numbers in [1, 118]"
            )
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("atomic_volumes values must be real numbers")
        try:
            volume = float(value)
        except OverflowError as error:
            raise ValueError(
                "atomic_volumes values must be finite and positive"
            ) from error
        if not math.isfinite(volume) or volume <= 0:
            raise ValueError("atomic_volumes values must be finite and positive")
        validated[number] = volume
    return validated


def estimate_formula_unit_volume(
    atomic_numbers: Tensor,
    *,
    atomic_volumes: Mapping[int, float] | None = None,
) -> float:
    """Estimate formula-unit volume by summing per-atom contributions.

    Parameters
    ----------
    atomic_numbers : torch.Tensor, shape ``[A]``
        Nonempty one-dimensional tensor using ``torch.uint8``, ``torch.int8``,
        ``torch.int16``, ``torch.int32``, or ``torch.int64``.
    atomic_volumes : Mapping[int, float], optional
        Complete replacement table in Å³ per atom. When omitted, the bundled
        COD-derived table is used. A custom mapping does not overlay defaults.

    Returns
    -------
    volume : float
        Estimated formula-unit volume in Å³.

    Raises
    ------
    TypeError
        If ``atomic_numbers`` is not a tensor, has an unsupported dtype, or
        ``atomic_volumes`` is not a mapping.
    ValueError
        If the tensor is empty or malformed, a table value is invalid, or one
        or more atomic numbers are missing from the selected table.

    Examples
    --------
    >>> import torch
    >>> estimate_formula_unit_volume(torch.tensor([6, 8]))
    23.78
    """
    if not isinstance(atomic_numbers, torch.Tensor):
        raise TypeError("atomic_numbers must be a torch.Tensor")
    if atomic_numbers.ndim != 1 or atomic_numbers.numel() == 0:
        raise ValueError("atomic_numbers must be a nonempty one-dimensional tensor")
    integral_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
    if atomic_numbers.dtype not in integral_dtypes:
        raise TypeError(
            "atomic_numbers must use torch.uint8, torch.int8, torch.int16, "
            "torch.int32, or torch.int64"
        )
    values = atomic_numbers.detach().to(device="cpu", dtype=torch.int64).tolist()
    if any(not 1 <= number <= 118 for number in values):
        raise ValueError("atomic_numbers must contain valid atomic numbers in [1, 118]")

    table = (
        get_default_atomic_volumes()
        if atomic_volumes is None
        else _validated_atomic_volumes(atomic_volumes)
    )
    missing = sorted(set(values).difference(table))
    if missing:
        raise ValueError(
            f"No atomic volume is available for atomic number(s): {missing}"
        )
    total = math.fsum(table[number] for number in values)
    if not math.isfinite(total):
        raise ValueError("formula-unit volume estimate must be finite")
    return total
