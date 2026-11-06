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
"""Supercell construction with ASE ``repeat`` semantics for :class:`AtomicData`."""

from __future__ import annotations

import itertools
import numbers
from collections.abc import Collection, Sequence

import torch

from nvalchemi.data.atomic_data import AtomicData

__all__ = [
    "DEFAULT_EXTENSIVE_SYSTEM_KEYS",
    "DEFAULT_INTENSIVE_SYSTEM_KEYS",
    "make_supercell",
]

DEFAULT_EXTENSIVE_SYSTEM_KEYS: frozenset[str] = frozenset(
    {"charge", "dipole", "energy", "virial"}
)
"""System-level fields a k-fold supercell carries k times over."""

DEFAULT_INTENSIVE_SYSTEM_KEYS: frozenset[str] = frozenset({"pbc", "stress"})
"""System-level fields a supercell carries unchanged."""

_DENSE_NEIGHBOR_KEYS: frozenset[str] = frozenset(
    {"neighbor_matrix", "neighbor_matrix_shifts", "num_neighbors"}
)
"""Node-level neighbor tensors whose indices point into the cell they were built on."""


def make_supercell(
    data: AtomicData,
    repeats: Sequence[int],
    *,
    extensive_keys: Collection[str] = DEFAULT_EXTENSIVE_SYSTEM_KEYS,
    intensive_keys: Collection[str] = DEFAULT_INTENSIVE_SYSTEM_KEYS,
    drop_keys: Collection[str] = (),
) -> AtomicData:
    """Return *data* tiled ``repeats`` times along each lattice vector.

    The tiling follows :meth:`ase.Atoms.repeat`. Copies are laid out
    copy-major, with the first lattice vector's index running slowest. Each
    copy is offset by an integer combination of the lattice vectors, and the
    cell is scaled along each axis. Node-level fields are repeated with the
    positions, so every atom of a copy keeps the field value of the site it
    came from. A system-level field is multiplied by the copy count when it is
    in *extensive_keys*, carried unchanged when it is in *intensive_keys*, and
    refused otherwise. An unclassified field is refused because a supercell
    scored under a field that does not scale with it would report the mismatch
    as the model's error. Edge-level fields, the dense neighbor tensors
    (``neighbor_matrix``, ``neighbor_matrix_shifts``, ``num_neighbors``), and
    any field in *drop_keys* are left out for the consumer to rebuild: a
    neighbor index copied into another cell would still point into the first.

    Parameters
    ----------
    data : AtomicData
        Periodic structure to replicate, left unmodified.
    repeats : Sequence[int]
        Three positive replication factors along the lattice vectors. An
        integral float or a one-element tensor is accepted as a count; a
        fractional value is refused rather than truncated.
    extensive_keys : Collection[str], optional
        System-level fields multiplied by the copy count. Default
        :data:`DEFAULT_EXTENSIVE_SYSTEM_KEYS`.
    intensive_keys : Collection[str], optional
        System-level fields carried unchanged. Default
        :data:`DEFAULT_INTENSIVE_SYSTEM_KEYS`.
    drop_keys : Collection[str], optional
        Node- or system-level fields left out of the supercell. Default
        ``()``.

    Returns
    -------
    AtomicData
        The supercell, carrying every field of *data* the tiling defines.

    Raises
    ------
    ValueError
        If *repeats* is not three positive integers, if *data* carries no
        cell, or if a system-level field is in neither *extensive_keys* nor
        *intensive_keys* and not dropped.

    Examples
    --------
    >>> import torch
    >>> from nvalchemi.data import AtomicData
    >>> from nvalchemi.data.transforms import make_supercell
    >>> data = AtomicData(
    ...     positions=torch.zeros(1, 3),
    ...     atomic_numbers=torch.tensor([18]),
    ...     cell=torch.eye(3).unsqueeze(0) * 4.0,
    ...     pbc=torch.tensor([[True, True, True]]),
    ... )
    >>> supercell = make_supercell(data, (2, 1, 1))
    >>> supercell.positions
    tensor([[0., 0., 0.],
            [4., 0., 0.]])
    >>> supercell.cell[0, 0]
    tensor([8., 0., 0.])
    """
    factors = _resolve_repeats(repeats)
    if data.cell is None:
        raise ValueError(
            "A supercell needs a periodic structure; the data carries no cell."
        )
    cell = data.cell.reshape(3, 3)
    scale = torch.tensor(factors, device=cell.device, dtype=cell.dtype)
    offsets = torch.stack(
        [
            torch.tensor(image, device=cell.device, dtype=cell.dtype) @ cell
            for image in itertools.product(*(range(count) for count in factors))
        ]
    )
    copies = len(offsets)
    dropped = set(drop_keys)
    node_keys = set(data.__node_keys__) - dropped - _DENSE_NEIGHBOR_KEYS
    system_keys = set(data.__system_keys__) - dropped - {"cell"}
    fields: dict[str, torch.Tensor] = {
        "positions": (data.positions.unsqueeze(0) + offsets.unsqueeze(1)).reshape(
            -1, 3
        ),
        "cell": (cell * scale.unsqueeze(-1)).unsqueeze(0),
    }
    for key in sorted(node_keys - {"positions"}):
        value = getattr(data, key, None)
        if value is not None:
            fields[key] = value.repeat((copies,) + (1,) * (value.ndim - 1))
    undefined = []
    for key in sorted(system_keys):
        value = getattr(data, key, None)
        if value is None:
            continue
        if key in extensive_keys:
            fields[key] = value * copies
        elif key in intensive_keys:
            fields[key] = value
        else:
            undefined.append(key)
    if undefined:
        raise ValueError(
            f"Replicating a structure carrying the system fields {undefined!r} is "
            "not defined: name each in extensive_keys to multiply it by the copy "
            "count, in intensive_keys to carry it unchanged, or in drop_keys to "
            "leave it out of the supercell."
        )
    declared = set(AtomicData.model_fields)
    supercell = AtomicData(**{key: fields[key] for key in fields if key in declared})
    for key in sorted(set(fields) - declared):
        if key in node_keys:
            supercell.add_node_property(key, fields[key])
        else:
            supercell.add_system_property(key, fields[key])
    return supercell


def _resolve_repeats(repeats: Sequence[int]) -> tuple[int, int, int]:
    """Return *repeats* as three positive ints, refusing a fractional count.

    The check runs before any conversion, so ``(1.5, 1, 1)`` is refused rather
    than tiled as ``(1, 1, 1)``. A one-element tensor is read through
    ``item()`` and an integral float is accepted.
    """
    counts = [
        count.item() if isinstance(count, torch.Tensor) else count for count in repeats
    ]
    integral = all(
        isinstance(count, numbers.Real)
        and not isinstance(count, bool)
        and float(count).is_integer()
        and count >= 1
        for count in counts
    )
    if len(counts) != 3 or not integral:
        raise ValueError(
            f"repeats must be three positive integers; got {list(repeats)!r}."
        )
    return int(counts[0]), int(counts[1]), int(counts[2])
