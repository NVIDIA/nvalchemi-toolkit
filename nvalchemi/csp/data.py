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
"""Owned formula-unit state and rigid-molecule ASU packing results."""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import secrets
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch

_FORMULA_DTYPES: dict[str, torch.dtype] = {
    "conformer_positions": torch.float32,
    "conformer_ptr": torch.int32,
    "molecule_conformer_ptr": torch.int32,
    "molecule_atom_ptr": torch.int32,
    "atomic_numbers": torch.int64,
    "contact_distances": torch.float32,
    "component_index": torch.int32,
}
_FORMULA_STATE_KEYS = {
    "version",
    "conformer_positions",
    "conformer_ptr",
    "molecule_conformer_ptr",
    "molecule_atom_ptr",
    "atomic_numbers",
    "contact_distances",
    "component_index",
    "formula_unit_volume",
    "metadata",
}

_COMPACT_DTYPES: dict[str, torch.dtype] = {
    "structure_molecule_ptr": torch.int32,
    "conformer_indices": torch.int32,
    "rotations": torch.float32,
    "fractional_centers": torch.float32,
    "cells": torch.float32,
    "space_groups": torch.int32,
    "z": torch.int32,
    "z_prime": torch.int32,
    "structure_ids": torch.int64,
}
_ATOM_SOURCE_FIELDS = {
    "csp_source_asu_atom_index",
    "csp_source_molecule_index",
    "csp_source_component_index",
    "csp_source_conformer_index",
    "csp_source_symmetry_operation_index",
}
_SYSTEM_SOURCE_FIELDS = {
    "csp_source_space_group",
    "csp_source_z",
    "csp_source_z_prime",
    "csp_source_structure_id",
}


def _freeze_json(value: Any, path: str = "metadata") -> Any:
    """Copy JSON values into recursively immutable containers."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            copied[key] = _freeze_json(item, f"{path}.{key}")
        return MappingProxyType(copied)
    if isinstance(value, list):
        return tuple(
            _freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value)
        )
    raise TypeError(f"{path} must contain only JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    """Return an independent mutable JSON copy of a frozen value."""
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _copy_formula_tensor(name: str, value: Any) -> Tensor:
    """Validate a formula tensor and return an owned contiguous CPU copy."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    expected_dtype = _FORMULA_DTYPES[name]
    if value.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}, got {value.dtype}")
    return value.detach().to(
        device="cpu", copy=True, memory_format=torch.contiguous_format
    )


def _validate_pointer(name: str, pointer: Tensor, end: int) -> list[int]:
    """Validate a zero-based nondecreasing pointer ending at ``end``."""
    if pointer.ndim != 1 or pointer.numel() < 2:
        raise ValueError(
            f"{name} must be a one-dimensional pointer with at least two entries"
        )
    values = pointer.tolist()
    if values[0] != 0:
        raise ValueError(f"{name} must start at zero")
    if values[-1] != end:
        raise ValueError(f"{name} must end at {end}, got {values[-1]}")
    if any(right < left for left, right in zip(values, values[1:], strict=False)):
        raise ValueError(f"{name} must be nondecreasing")
    return values


def _prepare_formula_input(
    values: Mapping[str, Any], *, center_conformers: bool = True
) -> dict[str, Any]:
    """Validate formula-unit fields and prepare their owned tensor state.

    Conformer coordinates are centered per conformer when ``center_conformers``
    is true. Tensor fields are copied to CPU by the formula-tensor copier.
    """
    required = set(_FORMULA_DTYPES) | {"formula_unit_volume"}
    missing = required - set(values)
    if missing:
        raise ValueError(f"Missing MolecularPackingInput fields: {sorted(missing)}")

    tensors = {
        name: _copy_formula_tensor(name, values[name]) for name in _FORMULA_DTYPES
    }
    positions = tensors["conformer_positions"]
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("conformer_positions must have shape [C_atoms, 3]")
    if not torch.isfinite(positions).all():
        raise ValueError("conformer_positions must contain only finite values")

    atomic_numbers = tensors["atomic_numbers"]
    if atomic_numbers.ndim != 1 or atomic_numbers.numel() == 0:
        raise ValueError("atomic_numbers must be a nonempty one-dimensional tensor")
    if ((atomic_numbers < 1) | (atomic_numbers > 118)).any():
        raise ValueError("atomic_numbers must be in the range [1, 118]")
    num_atoms = int(atomic_numbers.numel())

    atom_ptr = tensors["molecule_atom_ptr"]
    atom_ptr_values = _validate_pointer("molecule_atom_ptr", atom_ptr, num_atoms)
    num_molecules = atom_ptr.numel() - 1
    if num_molecules < 1:
        raise ValueError("A formula unit must contain at least one molecule")
    molecule_atom_counts = [
        right - left
        for left, right in zip(atom_ptr_values, atom_ptr_values[1:], strict=False)
    ]
    if any(count <= 0 for count in molecule_atom_counts):
        raise ValueError("Every formula-unit molecule must contain at least one atom")

    conformer_ptr = tensors["conformer_ptr"]
    conformer_ptr_values = _validate_pointer(
        "conformer_ptr", conformer_ptr, int(positions.shape[0])
    )
    num_conformers = conformer_ptr.numel() - 1
    if num_conformers < 1:
        raise ValueError("Each formula-unit molecule must have at least one conformer")
    conformer_atom_counts = [
        right - left
        for left, right in zip(
            conformer_ptr_values, conformer_ptr_values[1:], strict=False
        )
    ]
    if any(count <= 0 for count in conformer_atom_counts):
        raise ValueError("Every conformer must contain at least one atom")

    molecule_conformer_ptr = tensors["molecule_conformer_ptr"]
    molecule_conformer_values = _validate_pointer(
        "molecule_conformer_ptr", molecule_conformer_ptr, num_conformers
    )
    if molecule_conformer_ptr.numel() != num_molecules + 1:
        raise ValueError(
            "molecule_conformer_ptr must have the same number of entries as "
            "molecule_atom_ptr"
        )
    for molecule_index, (first, stop) in enumerate(
        zip(molecule_conformer_values, molecule_conformer_values[1:], strict=False)
    ):
        if first == stop:
            raise ValueError(
                f"Molecule {molecule_index} must have at least one conformer"
            )
        expected_atoms = molecule_atom_counts[molecule_index]
        if any(count != expected_atoms for count in conformer_atom_counts[first:stop]):
            raise ValueError(
                f"Conformers in molecule {molecule_index}'s pool must each contain "
                f"{expected_atoms} atoms"
            )

    contact_distances = tensors["contact_distances"]
    if contact_distances.shape != (num_atoms, num_atoms):
        raise ValueError(
            f"contact_distances must have shape [{num_atoms}, {num_atoms}]"
        )
    if not torch.isfinite(contact_distances).all() or (contact_distances <= 0).any():
        raise ValueError("contact_distances must contain finite positive values")
    if not torch.allclose(
        contact_distances, contact_distances.T, atol=1.0e-6, rtol=1.0e-5
    ):
        raise ValueError("contact_distances must be symmetric")

    component_index = tensors["component_index"]
    if component_index.shape != (num_molecules,):
        raise ValueError(f"component_index must have shape [{num_molecules}]")
    if (component_index < 0).any():
        raise ValueError("component_index values must be nonnegative")
    component_values = component_index.tolist()
    component_ids = set(component_values)
    if component_ids != set(range(max(component_values) + 1)):
        raise ValueError("component_index IDs must be contiguous from zero")

    raw_volume = values["formula_unit_volume"]
    if isinstance(raw_volume, bool) or not isinstance(raw_volume, numbers.Real):
        raise TypeError("formula_unit_volume must be a real number")
    volume = float(raw_volume)
    if not math.isfinite(volume) or volume <= 0.0:
        raise ValueError("formula_unit_volume must be finite and positive")

    if center_conformers:
        for first, stop in zip(
            conformer_ptr_values, conformer_ptr_values[1:], strict=False
        ):
            positions[first:stop] -= positions[first:stop].mean(dim=0)

    raw_metadata = values.get("metadata")
    if raw_metadata is not None and not isinstance(raw_metadata, Mapping):
        raise TypeError("metadata must be a mapping or None")
    metadata = None if raw_metadata is None else _freeze_json(raw_metadata)
    return {
        **tensors,
        "formula_unit_volume": volume,
        "metadata": metadata,
    }


class MolecularPackingInput(BaseModel):
    """Input data for one ordered molecular formula unit.

    Constructor tensors are copied to contiguous CPU storage. Each conformer is
    centered by its unweighted Cartesian mean. Contact matrix entries are
    intermolecular contact cutoffs used to measure overlap; the
    ``OverlapReliefPacker``
    may accept residual overlap up to ``OverlapReliefConfig.overlap_tolerance``.
    ``formula_unit_volume`` is a positive volume estimate in cubic angstroms.

    Direct in-place mutation of exposed tensors is unsupported. Use
    :meth:`state_dict` for independent CPU copies or :meth:`to` for owned tensor
    storage on another device.

    Parameters
    ----------
    conformer_positions : torch.Tensor, float32 ``[K, 3]``
        Concatenated Cartesian atom coordinates in angstroms. Conformers are
        stored in the order described by ``conformer_ptr`` and centered
        independently by their unweighted Cartesian means at construction.
    conformer_ptr : torch.Tensor, int32 ``[C + 1]``
        Start and end positions of each conformer's atoms in
        ``conformer_positions``. The pointer starts at zero, ends at ``K``, and
        delimits nonempty conformers.
    molecule_conformer_ptr : torch.Tensor, int32 ``[M + 1]``
        Start and end conformer indices for each formula-unit molecule's
        nonempty conformer pool.
    molecule_atom_ptr : torch.Tensor, int32 ``[M + 1]``
        Start and end positions of each molecule's atoms in ``atomic_numbers``.
        Each molecule is nonempty, and every conformer in its pool has the same
        atom count.
    atomic_numbers : torch.Tensor, int64 ``[A]``
        Atomic numbers from 1 through 118 in formula-unit molecule order.
    contact_distances : torch.Tensor, float32 ``[A, A]``
        Symmetric, finite, positive intermolecular contact cutoffs in angstroms
        used to measure overlap between atoms in different molecular copies.
    component_index : torch.Tensor, int32 ``[M]``
        Contiguous nonnegative component IDs, one per formula-unit molecule.
    formula_unit_volume : float
        Positive formula-unit volume estimate in cubic angstroms.
    metadata : Mapping[str, object], optional
        JSON-compatible provenance copied into recursively immutable containers.

    Notes
    -----
    All constructor tensors are detached, copied to contiguous CPU storage, and
    validated once. Pointer tensors describe distinct levels: atom ranges in
    ``molecule_atom_ptr``, conformer ranges in ``molecule_conformer_ptr``, and
    coordinate ranges in ``conformer_ptr``. The formula input owns the
    canonical CPU state; :meth:`to` makes an independent full tensor copy.

    Examples
    --------
    This synthetic two-carbon pair illustrates array layout and centering; it
    is not a chemically prepared molecule.

    >>> import torch
    >>> packing_input = MolecularPackingInput(
    ...     conformer_positions=torch.tensor([[0., 0., 0.], [2., 0., 0.]]),
    ...     conformer_ptr=torch.tensor([0, 2], dtype=torch.int32),
    ...     molecule_conformer_ptr=torch.tensor([0, 1], dtype=torch.int32),
    ...     molecule_atom_ptr=torch.tensor([0, 2], dtype=torch.int32),
    ...     atomic_numbers=torch.tensor([6, 6], dtype=torch.int64),
    ...     contact_distances=torch.ones((2, 2)),
    ...     component_index=torch.tensor([0], dtype=torch.int32),
    ...     formula_unit_volume=25.0,
    ... )
    >>> packing_input.conformer_positions.mean(dim=0)
    tensor([0., 0., 0.])
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="never",
    )

    conformer_positions: Tensor
    conformer_ptr: Tensor
    molecule_conformer_ptr: Tensor
    molecule_atom_ptr: Tensor
    atomic_numbers: Tensor
    contact_distances: Tensor
    component_index: Tensor
    formula_unit_volume: float
    metadata: Mapping[str, Any] | None = None

    @model_validator(mode="before")
    @classmethod
    def _validate_and_own(cls, values: Any) -> dict[str, Any]:
        """Validate formula input mappings and copy existing instances."""
        if isinstance(values, cls):
            values = {
                name: getattr(values, name)
                for name in _FORMULA_STATE_KEYS - {"version"}
            }
            return _prepare_formula_input(values, center_conformers=False)
        if not isinstance(values, Mapping):
            raise TypeError("MolecularPackingInput requires a mapping of tensor fields")
        return _prepare_formula_input(values)

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the ordered formula unit."""
        return int(self.atomic_numbers.numel())

    @property
    def num_molecules(self) -> int:
        """Number of molecules in the ordered formula unit."""
        return int(self.molecule_atom_ptr.numel() - 1)

    @property
    def num_conformers(self) -> int:
        """Number of conformers in all molecule pools."""
        return int(self.conformer_ptr.numel() - 1)

    @property
    def num_components(self) -> int:
        """Number of contiguous component IDs represented in the formula unit."""
        return int(self.component_index.max().item()) + 1

    @property
    def sha256(self) -> str:
        """Return the SHA-256 identity of the exact formula-input representation.

        The established encoding hashes the ordered tensor field names, dtypes,
        shapes, and contiguous bytes, followed by recursively JSON-compatible
        metadata and ``repr(float(formula_unit_volume))``. This identifies the
        exact input representation rather than chemical equivalence. The digest
        is recomputed on every access and is not part of :meth:`state_dict`.

        Reading device-resident tensors copies them to CPU and may synchronize
        the device, so use this property for input identity or startup checks,
        not inside a packing loop. Direct in-place tensor mutation remains
        unsupported.

        Returns
        -------
        str
            Lowercase 64-character SHA-256 hexadecimal digest.
        """
        digest = hashlib.sha256()
        for name in (
            "conformer_positions",
            "conformer_ptr",
            "molecule_conformer_ptr",
            "molecule_atom_ptr",
            "atomic_numbers",
            "contact_distances",
            "component_index",
        ):
            tensor = getattr(self, name).detach().cpu().contiguous()
            digest.update(name.encode())
            digest.update(str(tensor.dtype).encode())
            digest.update(json.dumps(list(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())

        metadata = _thaw_json(self.metadata)
        digest.update(
            json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
        )
        digest.update(repr(float(self.formula_unit_volume)).encode())
        return digest.hexdigest()

    def state_dict(self) -> dict[str, object]:
        """Return version-1 canonical CPU tensors and an independent JSON copy.

        Returns
        -------
        dict[str, object]
            A new mapping containing cloned CPU tensors and mutable JSON metadata.
        """
        return {
            "version": 1,
            **{
                name: getattr(self, name).detach().cpu().clone()
                for name in _FORMULA_DTYPES
            },
            "formula_unit_volume": self.formula_unit_volume,
            "metadata": None if self.metadata is None else _thaw_json(self.metadata),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> MolecularPackingInput:
        """Construct from an independently owned version-1 state mapping.

        Parameters
        ----------
        state : Mapping[str, object]
            Mapping returned by :meth:`state_dict`.

        Returns
        -------
        MolecularPackingInput
            A validated and independently owned formula-unit input.

        Raises
        ------
        ValueError
            If the version, keys, or tensor invariants are invalid.
        """
        if not isinstance(state, Mapping):
            raise TypeError("state must be a mapping")
        if set(state) != _FORMULA_STATE_KEYS:
            raise ValueError(
                "state must contain exactly the version-1 MolecularPackingInput keys"
            )
        version = state["version"]
        if type(version) is not int or version != 1:
            raise ValueError(
                f"Unsupported MolecularPackingInput state version: {version!r}"
            )
        validated = _prepare_formula_input(
            {key: value for key, value in state.items() if key != "version"},
            center_conformers=False,
        )
        return cls._from_validated(**validated)

    def to(self, device: torch.device | str) -> MolecularPackingInput:
        """Return newly owned tensor storage on ``device`` without revalidation.

        Parameters
        ----------
        device : torch.device or str
            Target device for formula tensors.

        Returns
        -------
        MolecularPackingInput
            An independent copy of every input tensor on the requested device,
            including the contact matrix.
        """
        target = torch.device(device)
        tensors = {
            name: getattr(self, name).to(device=target, copy=True)
            for name in _FORMULA_DTYPES
        }
        return self._from_validated(
            **tensors,
            formula_unit_volume=self.formula_unit_volume,
            metadata=self.metadata,
        )

    @classmethod
    def _from_validated(
        cls,
        *,
        conformer_positions: Tensor,
        conformer_ptr: Tensor,
        molecule_conformer_ptr: Tensor,
        molecule_atom_ptr: Tensor,
        atomic_numbers: Tensor,
        contact_distances: Tensor,
        component_index: Tensor,
        formula_unit_volume: float,
        metadata: Mapping[str, Any] | None,
    ) -> MolecularPackingInput:
        """Construct from trusted, already validated canonical state."""
        return cls.model_construct(
            _fields_set=set(_FORMULA_STATE_KEYS - {"version"}),
            conformer_positions=conformer_positions,
            conformer_ptr=conformer_ptr,
            molecule_conformer_ptr=molecule_conformer_ptr,
            molecule_atom_ptr=molecule_atom_ptr,
            atomic_numbers=atomic_numbers,
            contact_distances=contact_distances,
            component_index=component_index,
            formula_unit_volume=formula_unit_volume,
            metadata=metadata,
        )


def _reserved_batch_fields() -> set[str]:
    """Return Toolkit standard and CSP-emitted names forbidden to properties."""
    from nvalchemi.data.atomic_data import AtomicData
    from nvalchemi.data.level_storage import DEFAULT_ATTRIBUTE_MAP

    reserved = set(
        _ATOM_SOURCE_FIELDS | _SYSTEM_SOURCE_FIELDS | {"batch_idx", "batch_ptr"}
    )
    for names in DEFAULT_ATTRIBUTE_MAP.values():
        reserved.update(names)
    reserved.update(AtomicData._default_node_keys)
    reserved.update(AtomicData._default_edge_keys)
    reserved.update(AtomicData._default_system_keys)
    return reserved


def _copy_compact_tensor(name: str, value: Any) -> Tensor:
    """Validate one stored tensor and return an owned contiguous copy."""
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    expected_dtype = _COMPACT_DTYPES[name]
    if value.dtype != expected_dtype:
        raise TypeError(f"{name} must have dtype {expected_dtype}, got {value.dtype}")
    return value.detach().to(
        device=value.device, copy=True, memory_format=torch.contiguous_format
    )


def _prepare_compact(values: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ASU-batch fields and prepare owned tensor state."""
    required = {"packing_input", *(_COMPACT_DTYPES.keys() - {"structure_ids"})}
    missing = required - set(values)
    if missing:
        raise ValueError(f"Missing RigidMoleculeASUBatch fields: {sorted(missing)}")
    packing_input = values["packing_input"]
    if not isinstance(packing_input, MolecularPackingInput):
        raise TypeError("packing_input must be a MolecularPackingInput")

    tensors = {
        name: _copy_compact_tensor(name, values[name])
        for name in _COMPACT_DTYPES
        if name != "structure_ids" and name in values
    }
    if tensors["cells"].ndim != 3 or tensors["cells"].shape[1:] != (3, 3):
        raise ValueError("cells must have shape [P, 3, 3]")
    num_structures = int(tensors["cells"].shape[0])
    if "structure_ids" in values:
        tensors["structure_ids"] = _copy_compact_tensor(
            "structure_ids", values["structure_ids"]
        )
    else:
        run_id = secrets.randbits(63)
        tensors["structure_ids"] = torch.stack(
            (
                torch.full(
                    (num_structures,),
                    run_id,
                    dtype=torch.int64,
                    device=tensors["cells"].device,
                ),
                torch.arange(
                    num_structures, dtype=torch.int64, device=tensors["cells"].device
                ),
            ),
            dim=1,
        )
    if tensors["structure_ids"].shape != (num_structures, 2):
        raise ValueError("structure_ids must have shape [P, 2]")
    if tensors["structure_molecule_ptr"].shape != (num_structures + 1,):
        raise ValueError("structure_molecule_ptr must have shape [P + 1]")
    conformer_indices = tensors["conformer_indices"]
    if conformer_indices.ndim != 1:
        raise ValueError("conformer_indices must have shape [Q]")
    num_molecules = int(conformer_indices.shape[0])
    expected_shapes = {
        "structure_molecule_ptr": (num_structures + 1,),
        "conformer_indices": (num_molecules,),
        "rotations": (num_molecules, 3, 3),
        "fractional_centers": (num_molecules, 3),
        "space_groups": (num_structures,),
        "z": (num_structures,),
        "z_prime": (num_structures,),
    }
    for name, shape in expected_shapes.items():
        if tensors[name].shape != shape:
            raise ValueError(f"{name} must have shape {list(shape)}")

    device = tensors["cells"].device
    for name, tensor in tensors.items():
        if tensor.device != device:
            raise ValueError(
                f"{name} is on {tensor.device}; all compact tensors must share {device}"
            )

    raw_properties = values.get("properties", {})
    if not isinstance(raw_properties, Mapping):
        raise TypeError("properties must be a mapping from names to tensors")
    reserved = _reserved_batch_fields()
    properties: dict[str, Tensor] = {}
    for name, value in raw_properties.items():
        if not isinstance(name, str) or not name:
            raise TypeError("property names must be nonempty strings")
        if name in reserved:
            raise ValueError(
                f"property name {name!r} collides with a Toolkit or CSP field"
            )
        if not isinstance(value, Tensor):
            raise TypeError(f"property {name!r} must be a torch.Tensor")
        if value.ndim < 1 or value.shape[0] != num_structures:
            raise ValueError(
                f"property {name!r} must have leading dimension {num_structures}"
            )
        if value.device != device:
            raise ValueError(
                f"property {name!r} is on {value.device}; expected {device}"
            )
        properties[name] = value.detach().to(
            device=device, copy=True, memory_format=torch.contiguous_format
        )

    return {
        "packing_input": packing_input,
        **tensors,
        "properties": MappingProxyType(properties),
    }


class RigidMoleculeASUBatch(BaseModel):
    """Rigid-molecule ASU representations sharing one molecular formula input.

    Each structure stores independent molecular placements with its cell and
    symmetry metadata, rather than full-cell atom coordinates. Different
    structures may hold different numbers of independent placements.

    ``P`` is the number of structures, ``Z`` is the number of formula units in
    the full cell, and ``Z-prime`` is the number independently placed in the
    asymmetric unit (ASU).

    ASU representation tensors and property values are copied into contiguous
    storage on one device. Direct in-place mutation of exposed tensors is
    unsupported.

    Parameters
    ----------
    packing_input : MolecularPackingInput
        Shared ordered formula-unit state. Public construction retains this
        already validated object by identity.
    structure_molecule_ptr : torch.Tensor, int32 ``[P + 1]``
        Start and end positions of each structure's independent molecules in
        the ASU representation arrays.
    conformer_indices : torch.Tensor, int32 ``[Q]``
        Global conformer-pool indices for each ASU molecule.
    rotations : torch.Tensor, float32 ``[Q, 3, 3]``
        Cartesian molecular rotations.
    fractional_centers : torch.Tensor, float32 ``[Q, 3]``
        Fractional molecular centers.
    cells : torch.Tensor, float32 ``[P, 3, 3]``
        Row-vector lattice cells in angstroms.
    space_groups, z, z_prime : torch.Tensor, int32 ``[P]``
        International space-group numbers, full-cell formula-unit counts
        (``Z``), and independently placed ASU formula-unit counts
        (``Z-prime``), respectively.
    properties : Mapping[str, torch.Tensor], optional
        One value per crystal structure, with leading dimension ``P``, on the
        shared tensor device; trailing dimensions may vary by property. Names
        cannot collide with Toolkit or CSP output fields.
    structure_ids : torch.Tensor, int64 ``[P, 2]``, optional
        ``[run_id, structure_ordinal]`` identifiers. When omitted, a random
        nonnegative run ID and sequential ordinals are generated. Storage
        requires nonnegative identifiers.

    Notes
    -----
    Each structure contains ``z_prime`` independent copies of the formula
    unit in ``packing_input``. Molecules follow the same order as that input
    and select conformers from their respective pools. :meth:`to_batch`
    applies the space-group symmetry to generate the full crystal.

    The unit cell must be compatible with the selected space group, whose
    number of symmetry operations must equal ``z / z_prime``. Fractional
    molecular centers may lie outside ``[0, 1)``; expansion wraps them
    periodically.

    For externally prepared data, call :meth:`check_integrity` before
    expansion to check molecule assignments and symmetry multiplicities.
    This check does not validate molecular orientations or cell geometry.

    ``select`` shares ``packing_input`` and preserves repeated row indices.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
        extra="forbid",
        revalidate_instances="never",
    )

    packing_input: MolecularPackingInput
    structure_molecule_ptr: Tensor
    conformer_indices: Tensor
    rotations: Tensor
    fractional_centers: Tensor
    cells: Tensor
    space_groups: Tensor
    z: Tensor
    z_prime: Tensor
    structure_ids: Tensor
    properties: Mapping[str, Tensor] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _validate_and_own(cls, values: Any) -> dict[str, Any]:
        """Validate ASU-batch mappings and extract fields from existing batches."""
        if isinstance(values, cls):
            return {
                name: getattr(values, name)
                for name in {
                    "packing_input",
                    *_COMPACT_DTYPES,
                    "properties",
                }
            }
        if not isinstance(values, Mapping):
            raise TypeError(
                "RigidMoleculeASUBatch requires a mapping of compact fields"
            )
        return _prepare_compact(values)

    @classmethod
    def _from_validated(
        cls,
        *,
        packing_input: MolecularPackingInput,
        structure_molecule_ptr: Tensor,
        conformer_indices: Tensor,
        rotations: Tensor,
        fractional_centers: Tensor,
        cells: Tensor,
        space_groups: Tensor,
        z: Tensor,
        z_prime: Tensor,
        structure_ids: Tensor,
        properties: Mapping[str, Tensor],
    ) -> RigidMoleculeASUBatch:
        """Construct from trusted, already validated ASU tensors."""
        return cls.model_construct(
            _fields_set={
                "packing_input",
                "structure_molecule_ptr",
                "conformer_indices",
                "rotations",
                "fractional_centers",
                "cells",
                "space_groups",
                "z",
                "z_prime",
                "structure_ids",
                "properties",
            },
            packing_input=packing_input,
            structure_molecule_ptr=structure_molecule_ptr,
            conformer_indices=conformer_indices,
            rotations=rotations,
            fractional_centers=fractional_centers,
            cells=cells,
            space_groups=space_groups,
            z=z,
            z_prime=z_prime,
            structure_ids=structure_ids,
            properties=properties,
        )

    @property
    def num_structures(self) -> int:
        """Number of ASU structures, including zero for an empty batch."""
        return int(self.cells.shape[0])

    def check_integrity(self) -> None:
        """Check compact ASU pointer, multiplicity, symmetry, and pool indices.

        This explicit check copies only the integer metadata needed for these
        relationships to CPU. Calling it for device-resident data may
        synchronize the device. It does not inspect rotations, fractional
        centers, cells, or other geometry; finite coordinates, proper
        rotations, nonsingular positive-volume cells, and cell/symmetry
        compatibility remain caller preconditions. Success therefore does not
        certify scientific geometry.

        The check does not mutate this batch, consume RNG state, or cache a
        validated flag. Constructors, readers, selection, concatenation,
        expansion, and packing do not call it automatically.

        The operation performs O(P + Q) integer-metadata work plus a transfer
        of formula-molecule pool metadata. Its cost is unmeasured.

        Raises
        ------
        ValueError
            If the structure pointer, multiplicity, operation count, or
            conformer-pool membership relationship is invalid.
        """
        from nvalchemi.csp.symmetry import get_space_group_operation_count

        structure_ptr = self.structure_molecule_ptr.detach().cpu().tolist()
        conformer_indices = self.conformer_indices.detach().cpu().tolist()
        space_groups = self.space_groups.detach().cpu().tolist()
        z_values = self.z.detach().cpu().tolist()
        z_prime_values = self.z_prime.detach().cpu().tolist()
        molecule_conformer_ptr = (
            self.packing_input.molecule_conformer_ptr.detach().cpu().tolist()
        )

        num_structures = int(self.cells.shape[0])
        num_asu_molecules = int(self.conformer_indices.numel())
        num_formula_molecules = self.packing_input.num_molecules

        if not structure_ptr or structure_ptr[0] != 0:
            raise ValueError("structure_molecule_ptr must start at zero")
        for structure_index, (left, right) in enumerate(
            zip(structure_ptr, structure_ptr[1:])
        ):
            if right < left:
                raise ValueError(
                    "structure_molecule_ptr must be nondecreasing at "
                    f"structure {structure_index}: {left} > {right}"
                )
        if structure_ptr[-1] != num_asu_molecules:
            raise ValueError(
                "structure_molecule_ptr must end at the conformer_indices length"
            )

        for structure_index in range(num_structures):
            z = int(z_values[structure_index])
            z_prime = int(z_prime_values[structure_index])
            if z <= 0:
                raise ValueError(f"z must be positive for structure {structure_index}")
            if z_prime <= 0:
                raise ValueError(
                    f"z_prime must be positive for structure {structure_index}"
                )
            if z % z_prime:
                raise ValueError(
                    f"z_prime must divide z for structure {structure_index}"
                )

            start = int(structure_ptr[structure_index])
            stop = int(structure_ptr[structure_index + 1])
            expected_count = num_formula_molecules * z_prime
            if stop - start != expected_count:
                raise ValueError(
                    f"structure {structure_index} span has {stop - start} ASU "
                    f"molecules; expected {expected_count} from formula molecules "
                    "and z_prime"
                )

            space_group = int(space_groups[structure_index])
            if not 1 <= space_group <= 230:
                raise ValueError(f"space_groups[{structure_index}] must be in [1, 230]")
            operation_count = get_space_group_operation_count(space_group)
            expected_operation_count = z // z_prime
            if operation_count != expected_operation_count:
                raise ValueError(
                    f"space group {space_group} for structure {structure_index} has "
                    f"{operation_count} operations; expected {expected_operation_count} "
                    "from z / z_prime"
                )

            for row in range(start, stop):
                molecule_index = (row - start) % num_formula_molecules
                pool_start = int(molecule_conformer_ptr[molecule_index])
                pool_stop = int(molecule_conformer_ptr[molecule_index + 1])
                conformer_index = int(conformer_indices[row])
                if not pool_start <= conformer_index < pool_stop:
                    raise ValueError(
                        f"conformer index {conformer_index} at ASU row {row} is "
                        f"outside formula molecule {molecule_index} pool "
                        f"[{pool_start}, {pool_stop})"
                    )

    def select(self, indices: Tensor) -> RigidMoleculeASUBatch:
        """Select ASU rows in requested order, preserving repeated indices.

        Parameters
        ----------
        indices : torch.Tensor, int32 or int64 ``[S]``
            CPU indices or indices on this ASU batch's device.

        Returns
        -------
        RigidMoleculeASUBatch
            A new ASU batch sharing the same formula-unit input object.

        Raises
        ------
        TypeError
            If ``indices`` is not a one-dimensional int32/int64 tensor.
        ValueError
            If indices are on an unrelated device.

        Examples
        --------
        ``[2, 0, 2]`` returns rows in that order, including the repeated last row.
        An empty integer tensor returns a valid zero-structure ASU batch.
        """
        if not isinstance(indices, Tensor) or indices.ndim != 1:
            raise TypeError("indices must be a one-dimensional torch.Tensor")
        if indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("indices must have dtype torch.int32 or torch.int64")
        if indices.device.type != "cpu" and indices.device != self.cells.device:
            raise ValueError("indices must be on CPU or on the compact batch device")
        index = indices.to(device=self.cells.device)

        old_starts = self.structure_molecule_ptr.index_select(0, index)
        old_stops = self.structure_molecule_ptr.index_select(0, index + 1)
        molecule_counts = old_stops.to(torch.int64) - old_starts.to(torch.int64)
        selected_count = (
            int(molecule_counts.sum().item()) if molecule_counts.numel() else 0
        )
        new_ptr64 = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=self.cells.device),
                molecule_counts.cumsum(dim=0),
            )
        )
        if selected_count:
            selected_rows = torch.repeat_interleave(
                torch.arange(
                    index.numel(), device=self.cells.device, dtype=torch.int64
                ),
                molecule_counts,
                output_size=selected_count,
            )
            local_molecule = torch.arange(
                selected_count, device=self.cells.device, dtype=torch.int64
            ) - new_ptr64[:-1].index_select(0, selected_rows)
            molecule_index = (
                old_starts.to(torch.int64).index_select(0, selected_rows)
                + local_molecule
            )
        else:
            molecule_index = torch.empty(0, dtype=torch.int64, device=self.cells.device)

        return self._from_validated(
            packing_input=self.packing_input,
            structure_molecule_ptr=new_ptr64.to(torch.int32),
            conformer_indices=self.conformer_indices.index_select(0, molecule_index),
            rotations=self.rotations.index_select(0, molecule_index),
            fractional_centers=self.fractional_centers.index_select(0, molecule_index),
            cells=self.cells.index_select(0, index),
            space_groups=self.space_groups.index_select(0, index),
            z=self.z.index_select(0, index),
            z_prime=self.z_prime.index_select(0, index),
            structure_ids=self.structure_ids.index_select(0, index),
            properties=MappingProxyType(
                {
                    name: value.index_select(0, index)
                    for name, value in self.properties.items()
                }
            ),
        )

    @classmethod
    def concatenate(
        cls, batches: Sequence[RigidMoleculeASUBatch]
    ) -> RigidMoleculeASUBatch:
        """Concatenate ASU batches with equivalent packing inputs.

        Parameters
        ----------
        batches : Sequence[RigidMoleculeASUBatch]
            Batches to append in sequence order.

        Returns
        -------
        RigidMoleculeASUBatch
            A new ASU batch. Mixed ``z`` and ``z_prime`` values are
            supported.

        Raises
        ------
        ValueError
            If the sequence is empty, formula inputs differ, devices differ, or
            property schemas do not match.
        """
        if not batches:
            raise ValueError("at least one compact batch is required")
        if any(not isinstance(batch, cls) for batch in batches):
            raise TypeError("all entries must be RigidMoleculeASUBatch instances")
        first = batches[0]
        same_input = all(
            batch.packing_input is first.packing_input for batch in batches
        )
        if not same_input and any(
            not _formula_inputs_equal(first.packing_input, batch.packing_input)
            for batch in batches[1:]
        ):
            raise ValueError("all compact batches must have equal packing_input values")
        property_names = set(first.properties)
        for batch in batches:
            if not isinstance(batch, cls):
                raise TypeError("all entries must be RigidMoleculeASUBatch instances")
            if batch.cells.device != first.cells.device:
                raise ValueError("all compact batches must share one device")
            if set(batch.properties) != property_names:
                raise ValueError(
                    "all compact batches must have identical property keys"
                )
            for name in property_names:
                left, right = first.properties[name], batch.properties[name]
                if left.dtype != right.dtype or left.shape[1:] != right.shape[1:]:
                    raise ValueError(
                        f"property {name!r} must have matching dtype and trailing shape"
                    )

        ptr_parts = [first.structure_molecule_ptr]
        molecule_offset = first.conformer_indices.shape[0]
        for batch in batches[1:]:
            ptr_parts.append(batch.structure_molecule_ptr[1:] + molecule_offset)
            molecule_offset += batch.conformer_indices.shape[0]
        props = MappingProxyType(
            {
                name: torch.cat([batch.properties[name] for batch in batches], dim=0)
                for name in property_names
            }
        )
        return cls._from_validated(
            packing_input=first.packing_input,
            structure_molecule_ptr=torch.cat(ptr_parts, dim=0),
            conformer_indices=torch.cat(
                [batch.conformer_indices for batch in batches], dim=0
            ),
            rotations=torch.cat([batch.rotations for batch in batches], dim=0),
            fractional_centers=torch.cat(
                [batch.fractional_centers for batch in batches], dim=0
            ),
            cells=torch.cat([batch.cells for batch in batches], dim=0),
            space_groups=torch.cat([batch.space_groups for batch in batches], dim=0),
            z=torch.cat([batch.z for batch in batches], dim=0),
            z_prime=torch.cat([batch.z_prime for batch in batches], dim=0),
            structure_ids=torch.cat([batch.structure_ids for batch in batches], dim=0),
            properties=props,
        )

    def to(self, device: torch.device | str) -> RigidMoleculeASUBatch:
        """Return newly owned ASU representation and formula tensors on ``device``.

        Parameters
        ----------
        device : torch.device or str
            Target device.

        Returns
        -------
        RigidMoleculeASUBatch
            An independent copy of the ASU fields, properties, and formula
            input on the requested device.
        """
        target = torch.device(device)

        def move(value: Tensor) -> Tensor:
            """Return an independent copy of one tensor on the target device."""
            return value.to(device=target, copy=True)

        return self._from_validated(
            packing_input=self.packing_input.to(target),
            structure_molecule_ptr=move(self.structure_molecule_ptr),
            conformer_indices=move(self.conformer_indices),
            rotations=move(self.rotations),
            fractional_centers=move(self.fractional_centers),
            cells=move(self.cells),
            space_groups=move(self.space_groups),
            z=move(self.z),
            z_prime=move(self.z_prime),
            structure_ids=move(self.structure_ids),
            properties=MappingProxyType(
                {name: move(value) for name, value in self.properties.items()}
            ),
        )

    def to_batch(
        self,
        indices: Tensor | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> Batch:
        """Expand selected ASU rows into a full-cell periodic Toolkit ``Batch``.

        Parameters
        ----------
        indices : torch.Tensor, int32 or int64 ``[S]``, optional
            Ordered ASU row selection. Repeated indices produce repeated
            structures. ``None`` expands every ASU row.
        device : torch.device or str, optional
            Output device. Defaults to this ASU batch's device.

        Returns
        -------
        Batch
            A periodic Toolkit Batch with explicit full-cell atoms and fields
            recording their ASU source.

        Notes
        -----
        Expansion assumes valid index relationships and geometry. It does not
        automatically call :meth:`check_integrity`; the explicit checker covers
        indices only, and geometry remains a caller precondition.

        Cells must be compatible with their space-group operations. Molecular
        centers are wrapped in fractional space before their unwrapped rigid
        atomic displacements are added. Provenance describes source state and
        remains unchanged if a later optimizer changes geometry.

        """
        from nvalchemi.csp._batch import expand_asu_batch

        return expand_asu_batch(self, indices=indices, device=device)


def _formula_inputs_equal(
    left: MolecularPackingInput, right: MolecularPackingInput
) -> bool:
    """Compare formula inputs exactly, moving tensor comparisons to CPU if needed."""
    if left is right:
        return True
    if (
        left.formula_unit_volume != right.formula_unit_volume
        or left.metadata != right.metadata
    ):
        return False
    for key in _FORMULA_DTYPES:
        left_tensor = getattr(left, key)
        right_tensor = getattr(right, key)
        if left_tensor.device != right_tensor.device:
            left_tensor = left_tensor.detach().cpu()
            right_tensor = right_tensor.detach().cpu()
        if not torch.equal(left_tensor, right_tensor):
            return False
    return True
