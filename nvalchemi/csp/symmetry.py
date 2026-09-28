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

"""Space-group operations, crystal-system metadata, and sampling helpers.

The source CSP project supplies the pinned standard-setting Hall-number
mapping. Operation tables were regenerated from the spglib 2.7.0 Hall database
using that mapping; the bundled sampling prior is CSD-derived. Operations use
International space-group numbers ``1..230`` and store each fractional
operation as a ``[3, 4]`` matrix ``[R | t]``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from enum import Enum
from functools import lru_cache
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
import torch
from torch import Tensor

from nvalchemi.csp._data_tables import SPACE_GROUP_PROBABILITIES
from nvalchemi.csp._space_group_tables import SG_OPS_IDX, SG_OPS_PTR, SYMM_OPS

Device: TypeAlias = torch.device | str | None
# Each entry is (first space-group number, last space-group number,
# zero-based crystal-system code). The codes follow the CrystalSystem enum below.
_CRYSTAL_SYSTEM_RANGES = (
    (1, 2, 0),
    (3, 15, 1),
    (16, 74, 2),
    (75, 142, 3),
    (143, 167, 4),
    (168, 194, 5),
    (195, 230, 6),
)
# International numbers for chirality-preserving (proper-rotation-only) groups
# queried by :func:`is_sohncke_space_group`.
# fmt: off
_SOHNCKE = frozenset(
    {
        1, 3, 4, 5, 16, 17, 18, 19, 20, 21, 22, 23,
        24, 75, 76, 77, 78, 79, 80, 89, 90, 91, 92, 93,
        94, 95, 96, 97, 98, 143, 144, 145, 146, 149, 150, 151,
        152, 153, 154, 155, 168, 169, 170, 171, 172, 173, 177,
        178, 179, 180, 181, 182, 195, 196, 197, 198, 199,
        207, 208, 209, 210, 211, 212, 213, 214,
    }
)
# fmt: on


class CrystalSystem(str, Enum):
    """The seven crystallographic crystal systems."""

    TRICLINIC = "triclinic"
    MONOCLINIC = "monoclinic"
    ORTHORHOMBIC = "orthorhombic"
    TETRAGONAL = "tetragonal"
    TRIGONAL = "trigonal"
    HEXAGONAL = "hexagonal"
    CUBIC = "cubic"


def _validate_space_group(space_group: int) -> int:
    """Return an International space-group number after range validation."""
    if isinstance(space_group, bool) or not isinstance(space_group, Integral):
        raise TypeError("space_group must be an integer")
    number = int(space_group)
    if not 1 <= number <= 230:
        raise ValueError("space_group must be in [1, 230]")
    return number


def _validate_positive_integer(value: int, name: str) -> int:
    """Return a positive integral value, rejecting booleans."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _validate_table() -> None:
    """Check that bundled operation and pointer tables have consistent shapes."""
    if SYMM_OPS.ndim != 2 or SYMM_OPS.shape[1] != 12:
        raise RuntimeError("space-group operations must have shape [K, 12]")
    if SG_OPS_IDX.ndim != 1 or SG_OPS_PTR.shape != (231,):
        raise RuntimeError("space-group pointer and index tables have invalid shapes")
    if int(SG_OPS_PTR[0]) != 0 or int(SG_OPS_PTR[-1]) != SG_OPS_IDX.size:
        raise RuntimeError(
            "space-group operation pointers do not cover the index table"
        )
    if np.any(np.diff(SG_OPS_PTR) <= 0):
        raise RuntimeError("every space group must have at least one operation")
    if (
        SG_OPS_IDX.size != 4425
        or np.any(SG_OPS_IDX < 0)
        or np.any(SG_OPS_IDX >= len(SYMM_OPS))
    ):
        raise RuntimeError("space-group operation indices are malformed")


_validate_table()


@lru_cache(maxsize=1)
def _default_probabilities() -> dict[int, float]:
    """Return cached, validated default space-group sampling weights."""
    return _validate_probabilities(SPACE_GROUP_PROBABILITIES)


def _validate_probabilities(probabilities: Mapping[int, float]) -> dict[int, float]:
    """Copy and validate finite nonnegative weights keyed by space group."""
    if not isinstance(probabilities, Mapping):
        raise TypeError("probabilities must be a mapping")
    result: dict[int, float] = {}
    for key, value in probabilities.items():
        number = _validate_space_group(key)
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError("probability weights must be real numbers")
        try:
            weight = float(value)
        except OverflowError as error:
            raise ValueError(
                "probability weights must be finite and nonnegative"
            ) from error
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("probability weights must be finite and nonnegative")
        result[number] = weight
    return result


def csd_space_group_probabilities() -> dict[int, float]:
    """Return default space-group sampling weights as a fresh copy.

    These relative weights were derived from the Cambridge Structural
    Database (CSD); they are not probabilities until compatible groups are
    selected and the remaining positive weights are normalized.

    Returns
    -------
    probabilities : dict[int, float]
        Weights for every International space-group number from 1 through 230,
        including entries whose weight is zero. The returned mapping is
        independent of library state.

    Notes
    -----
    The bundled weights use space-group counts from the CCDC report
    `CSD Space Group Statistics: Space Group Number Ordering (2026)
    <https://www.ccdc.cam.ac.uk/media/CSD-Space-Group-Statistics-Space-Group-Number-Ordering-2026.pdf>`_.
    """
    return _default_probabilities().copy()


def get_space_group_operations(
    space_group: int,
    *,
    dtype: torch.dtype = torch.float32,
    device: Device = None,
) -> Tensor:
    """Return standard fractional operations for one space group.

    Parameters
    ----------
    space_group : int
        International space-group number in ``[1, 230]``.
    dtype : torch.dtype, default=torch.float32
        Floating dtype for the returned operation tensor.
    device : torch.device or str, optional
        Device for the returned tensor. The bundled table itself remains on CPU.

    Returns
    -------
    operations : torch.Tensor, shape ``[S, 3, 4]``
        Fractional rotations in ``[..., :3]`` and translations in ``[..., 3]``.
        For row-vector fractional coordinates, ``f_out = f @ R.T + t``.

    Raises
    ------
    TypeError
        If ``space_group`` is not a non-boolean integer or ``dtype`` is not a
        floating Torch dtype.
    ValueError
        If the International number is outside ``[1, 230]``.

    Examples
    --------
    >>> operations = get_space_group_operations(2)  # P-1 space group.
    >>> print(operations)  # Identity, then inversion.
    tensor([[[ 1.,  0.,  0.,  0.],
             [ 0.,  1.,  0.,  0.],
             [ 0.,  0.,  1.,  0.]],
    <BLANKLINE>
            [[-1.,  0.,  0.,  0.],
             [ 0., -1.,  0.,  0.],
             [ 0.,  0., -1.,  0.]]])
    """
    number = _validate_space_group(space_group)
    if not isinstance(dtype, torch.dtype) or not dtype.is_floating_point:
        raise TypeError("dtype must be a floating torch dtype")
    start = int(SG_OPS_PTR[number - 1])
    end = int(SG_OPS_PTR[number])
    flat_rows = SYMM_OPS[SG_OPS_IDX[start:end]]
    operations = torch.empty((end - start, 3, 4), dtype=dtype)
    operations[:, :, :3] = torch.as_tensor(
        flat_rows[:, :9].copy().reshape(-1, 3, 3), dtype=dtype
    )
    operations[:, :, 3] = torch.as_tensor(flat_rows[:, 9:12].copy(), dtype=dtype)
    return operations.to(device=device) if device is not None else operations


def get_space_group_operation_count(space_group: int) -> int:
    """Return the number of standard operations for a space group.

    Parameters
    ----------
    space_group : int
        International space-group number in ``[1, 230]``.

    Returns
    -------
    count : int
        Number of operations in the pinned standard-setting table.

    Raises
    ------
    TypeError
        If ``space_group`` is not a non-boolean integer.
    ValueError
        If the International number is outside ``[1, 230]``.
    """
    number = _validate_space_group(space_group)
    return int(SG_OPS_PTR[number] - SG_OPS_PTR[number - 1])


def get_crystal_system(space_group: int) -> CrystalSystem:
    """Return the crystal system for an International space-group number.

    Parameters
    ----------
    space_group : int
        International space-group number in ``[1, 230]``.

    Returns
    -------
    crystal_system : CrystalSystem
        One of the seven standard crystal systems.

    Raises
    ------
    TypeError
        If ``space_group`` is not a non-boolean integer.
    ValueError
        If the International number is outside ``[1, 230]``.
    """
    number = _validate_space_group(space_group)
    for first, last, index in _CRYSTAL_SYSTEM_RANGES:
        if first <= number <= last:
            return tuple(CrystalSystem)[index]
    raise RuntimeError("valid space group was not assigned to a crystal system")


def is_sohncke_space_group(space_group: int) -> bool:
    """Return whether a group's symmetry preserves molecular handedness.

    Parameters
    ----------
    space_group : int
        International space-group number in ``[1, 230]``.

    Returns
    -------
    is_sohncke : bool
        Whether all symmetry-generated copies preserve handedness.

    Raises
    ------
    TypeError
        If ``space_group`` is not a non-boolean integer.
    ValueError
        If the International number is outside ``[1, 230]``.
    """
    return _validate_space_group(space_group) in _SOHNCKE


def _filter_candidates(
    num_operations: int,
    crystal_system: CrystalSystem | None,
    sohncke_only: bool,
) -> list[int]:
    """List groups matching operation count and optional crystallographic filters."""
    count = _validate_positive_integer(num_operations, "num_operations")
    if crystal_system is not None and not isinstance(crystal_system, CrystalSystem):
        raise TypeError("crystal_system must be a CrystalSystem or None")
    if not isinstance(sohncke_only, bool):
        raise TypeError("sohncke_only must be a bool")
    return [
        number
        for number in range(1, 231)
        if get_space_group_operation_count(number) == count
        and (crystal_system is None or get_crystal_system(number) is crystal_system)
        and (not sohncke_only or number in _SOHNCKE)
    ]


def get_space_group_candidates(
    num_operations: int,
    *,
    crystal_system: CrystalSystem | None = None,
    sohncke_only: bool = False,
) -> Tensor:
    """Return space-group numbers with a requested operation count.

    Parameters
    ----------
    num_operations : int
        Positive required number of symmetry operations.
    crystal_system : CrystalSystem, optional
        Restrict results to one crystal system.
    sohncke_only : bool, default=False
        Keep only groups whose symmetry-generated copies preserve handedness.

    Returns
    -------
    candidates : torch.Tensor, shape ``[K]``, dtype=torch.int32
        Compatible International numbers in ascending order; ``K`` may be zero.

    Raises
    ------
    TypeError
        If ``num_operations`` is not a non-boolean integer, or a filter has
        the wrong type.
    ValueError
        If ``num_operations`` is not positive.
    """
    return torch.tensor(
        _filter_candidates(num_operations, crystal_system, sohncke_only),
        dtype=torch.int32,
    )


def sample_space_groups(
    num_samples: int,
    num_operations: int,
    *,
    probabilities: Mapping[int, float] | None = None,
    crystal_system: CrystalSystem | None = None,
    sohncke_only: bool = False,
    seed: int | None = None,
    device: Device = None,
) -> Tensor:
    """Sample compatible International space-group numbers with replacement.

    A supplied ``probabilities`` mapping replaces the default relative
    sampling weights. Missing group entries have zero weight. After removing
    incompatible groups, the remaining weights determine relative sampling
    probabilities.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw; must be positive.
    num_operations : int
        Required number of symmetry operations.
    probabilities : Mapping[int, float], optional
        Replacement relative weights keyed by group number; values must be
        finite and nonnegative.
    crystal_system : CrystalSystem, optional
        Restrict the compatible groups to one crystal system.
    sohncke_only : bool, default=False
        Use only groups whose symmetry-generated copies preserve molecular
        handedness, such as for chiral molecules.
    seed : int, optional
        Nonnegative seed for a local CPU ``torch.Generator``.
    device : torch.device or str, optional
        Device for the returned int32 tensor.

    Returns
    -------
    space_groups : torch.Tensor, shape ``[num_samples]``, dtype=torch.int32
        Sampled International numbers.

    Raises
    ------
    TypeError
        If the prior, filters, or integer arguments have invalid types.
    ValueError
        If no compatible group has positive weight or ``seed`` is negative.

    Examples
    --------
    >>> groups = sample_space_groups(4, 1, seed=7)
    >>> groups.dtype
    torch.int32
    """
    samples = _validate_positive_integer(num_samples, "num_samples")
    if seed is not None:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise TypeError("seed must be an integer or None")
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be nonnegative")
    candidates = _filter_candidates(num_operations, crystal_system, sohncke_only)
    if probabilities is None:
        prior = _default_probabilities()
    else:
        prior = _validate_probabilities(probabilities)
    weights = np.asarray(
        [prior.get(number, 0.0) for number in candidates], dtype=np.float64
    )
    if weights.size == 0 or not np.isfinite(weights).all() or not np.any(weights > 0):
        raise ValueError(
            "at least one compatible space group must have a positive sampling weight"
        )
    # Scaling first avoids overflow when valid finite weights have a huge sum.
    weights /= weights.max()
    weights /= weights.sum()
    generator = None
    if seed is not None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
    selected = torch.multinomial(
        torch.from_numpy(weights), samples, replacement=True, generator=generator
    )
    result = torch.tensor(candidates, dtype=torch.int32)[selected]
    return result.to(device=device) if device is not None else result
