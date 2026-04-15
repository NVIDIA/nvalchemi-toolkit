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

import numbers
import warnings
from collections.abc import Sequence
from hashlib import blake2s
from typing import TYPE_CHECKING, Annotated, Any, ClassVar

import numpy as np
import periodictable as pt
import torch
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, model_validator

from nvalchemi import OptionalDependency
from nvalchemi import _typing as t
from nvalchemi.data.data import DataMixin  # type: ignore

if TYPE_CHECKING:
    from ase import Atoms
    from pymatgen.core import Molecule, Structure


def _tensor_serialization(tensor: torch.Tensor) -> list[float | int | list]:
    """
    Map a PyTorch tensor to JSON serializable values.

    Parameters
    ----------
    tensor: torch.Tensor
        The tensor to serialize.

    Returns
    -------
    list[float | int] | None
        The serialized tensor, or None if *tensor* is None.
    """
    if tensor is None:
        return None
    return tensor.detach().cpu().tolist()


class AtomicNumberTable:
    """
    Atomic number table
    """

    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self) -> str:
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        """
        Convert index to atomic number
        """
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        """
        Convert atomic number to index
        """
        return self.zs.index(atomic_number)


class AtomicData(BaseModel, DataMixin):
    """Atomic data structure for molecular systems.

    Represents molecular systems as graphs with atomic properties and interactions.
    Uses Pydantic for validation and serialization, with DataMixin for graph functionality.

    Attributes
    ----------
    atomic_numbers : torch.Tensor
        Atomic numbers of each atom [n_nodes]
    positions : torch.Tensor
        Cartesian coordinates [n_nodes, 3]
    atomic_masses : torch.Tensor
        Atomic masses [n_nodes]
    neighbor_list : torch.Tensor
        Neighbor list [n_edges, 2]
    node_attrs : torch.Tensor
        Node attributes [n_nodes, n_node_feats]
    shifts : torch.Tensor
        Cartesian displacement vectors for each edge [n_edges, 3],
        computed as ``neighbor_list_shifts @ cell``.
    neighbor_list_shifts : torch.Tensor
        Integer lattice image indices for periodic edges [n_edges, 3].
    neighbor_matrix : torch.Tensor
        Dense neighbor matrix [n_nodes, max_neighbors]
    neighbor_matrix_shifts : torch.Tensor
        Periodic shifts for the dense neighbor matrix [n_nodes, max_neighbors, 3]
    num_neighbors : torch.Tensor
        Number of valid neighbors per atom [n_nodes]
    cell : torch.Tensor
        Unit cell vectors [3, 3]
    pbc : torch.Tensor
        Periodic boundary conditions [3]
    forces : torch.Tensor
        Atomic forces [n_nodes, 3]
    energy : torch.Tensor
        Total energy [1]
    stress : torch.Tensor
        Stress tensor [1, 3, 3]
    virial : torch.Tensor
        Virial tensor [1, 3, 3]
    dipole : torch.Tensor
        Dipole moment [1, 3]
    charges : torch.Tensor
        Partial atomic charges [n_nodes]
    charge : torch.Tensor
        Total system charge [1]
    info : dict
        Additional information about the system
    """

    # Required fields
    atomic_numbers: Annotated[
        t.AtomicNumbers,
        Field(description="Atomic numbers for each node [n_nodes]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ]
    positions: Annotated[
        t.NodePositions,
        Field(description="Cartesian coordinates for each atom [n_nodes, 3]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ]
    # Optional fields with defaults
    atomic_masses: Annotated[
        t.AtomicMasses | None,
        Field(description="Atomic masses [n_nodes]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    atom_categories: Annotated[
        list[t.AtomCategory] | t.AtomCategories | None,
        Field(
            description="Atom categorical index, based on _typing.AtomCategory Enum [n_nodes]"
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    neighbor_list: Annotated[
        t.NeighborList | None,
        Field(description="Neighbor list [n_edges, 2]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    shifts: Annotated[
        t.PeriodicShifts | None,
        Field(
            description="Cartesian displacement vectors for each edge (neighbor_list_shifts @ cell) [n_edges, 3]"
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    neighbor_list_shifts: Annotated[
        t.NeighborListShifts | None,
        Field(
            description="Integer lattice image indices for periodic edges [n_edges, 3]"
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    neighbor_matrix: Annotated[
        t.NeighborMatrix | None,
        Field(description="Dense neighbor matrix [n_nodes, max_neighbors]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    neighbor_matrix_shifts: Annotated[
        t.NeighborMatrixShifts | None,
        Field(
            description="Periodic shifts for the dense neighbor matrix [n_nodes, max_neighbors, 3]"
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    num_neighbors: Annotated[
        t.NumNeighbors | None,
        Field(description="Number of valid neighbors per atom [n_nodes]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    cell: Annotated[
        t.LatticeVectors | None,
        Field(description="Unit cell vectors [3, 3]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    pbc: Annotated[
        t.Periodicity | None,
        Field(
            description="Boolean tensor indicating periodic boundary conditions along each dimension"
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    forces: Annotated[
        t.Forces | None,
        Field(description="Atomic forces [n_nodes, 3]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    energy: Annotated[
        t.Energy | None,
        Field(description="Total energy [1]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    stress: Annotated[
        t.Stress | None,
        Field(description="Cauchy stress W/V (eV/A^3) [1, 3, 3]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    virial: Annotated[
        t.Virials | None,
        Field(description="Virial tensor [1, 3, 3]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    dipole: Annotated[
        t.Dipole | None,
        Field(description="Dipole moment of the system."),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    charges: Annotated[
        t.NodeCharges | None,
        Field(description="Partial atomic charges [n_nodes]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    charge: Annotated[
        t.GraphCharges | None,
        Field(description="Total system charge [1]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    node_attrs: Annotated[
        t.NodeAttributes | None,
        Field(description="Node attributes [n_nodes, n_node_attrs]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    node_alpha_spins: Annotated[
        t.NodeSpins | None,
        Field(
            description="Alpha spins for each atom, [n_nodes, 1]. Use this field for closed-shell spins."
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    node_beta_spins: Annotated[
        t.NodeSpins | None,
        Field(
            description="Beta spins for each atom, [n_nodes, 1]. For restricted spin, use ``node_alpha_spins`` instead."
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    spin: Annotated[
        t.GraphSpins | None,
        Field(description="Spin or multiplicity value for the system, [1, 1]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    graph_alpha_spins: Annotated[
        t.GraphSpins | None,
        Field(description="Alpha spins for the entire graph, [1, 1]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    node_embeddings: Annotated[
        t.NodeEmbeddings | None,
        Field(description="Embeddings for each node within the batch/graph."),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    edge_embeddings: Annotated[
        t.EdgeEmbeddings | None,
        Field(description="Embeddings for each edge within the batch/graph."),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    graph_embeddings: Annotated[
        t.GraphEmbeddings | None,
        Field(description="Embeddings for the entire graph/graphs within a batch."),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    velocities: Annotated[
        t.NodeVelocities | None,
        Field(description="Atomic velocities [n_nodes, 3], in units set by positions."),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    momenta: Annotated[
        t.NodeMomentum | None,
        Field(description="Atomic momenta [n_nodes, 3], in units set by positions."),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    kinetic_energies: Annotated[
        t.NodeKineticEnergies | None,
        Field(
            description="Per-atom kinetic energies [n_nodes, 1], with the same units as energy."
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    info: dict[str, torch.Tensor] = Field(default_factory=dict)
    # "Node key" means dim(0) == num_nodes; tensors may have any rank.
    _default_node_keys: ClassVar[frozenset[str]] = frozenset(
        {
            "atomic_masses",
            "positions",
            "forces",
            "charges",
            "node_embeddings",
            "atomic_numbers",
            "node_attrs",
            "node_alpha_spins",
            "node_beta_spins",
            "atom_categories",
            "velocities",
            "momenta",
            "kinetic_energies",
            "neighbor_matrix",
            "neighbor_matrix_shifts",
            "num_neighbors",
        }
    )
    _default_edge_keys: ClassVar[frozenset[str]] = frozenset(
        {"shifts", "neighbor_list_shifts", "neighbor_list", "edge_embeddings"}
    )
    _default_system_keys: ClassVar[frozenset[str]] = frozenset(
        {
            "energy",
            "stress",
            "virial",
            "dipole",
            "charge",
            "graph_embeddings",
            "cell",
            "pbc",
            "spin",
        }
    )

    # Pydantic configuration
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="allow"
    )

    def model_post_init(self, __context: Any) -> None:
        """Create per-instance mutable copies of the key sets.

        The class-level defaults are frozen to prevent accidental mutation.
        Each instance gets its own mutable set so that ``add_node_property``
        and friends only affect the instance they are called on.

        Uses ``model_post_init`` rather than ``model_validator`` because
        ``validate_assignment=True`` causes model validators to re-run on
        every ``setattr`` call, which would reset the key sets and lose
        previously added custom keys.
        """
        # Merge defaults with any dynamically-added keys passed during
        # construction (e.g. via model_validate or from_data_list round-trips).
        existing_node = set(getattr(self, "__node_keys__", ()))
        existing_edge = set(getattr(self, "__edge_keys__", ()))
        existing_system = set(getattr(self, "__system_keys__", ()))
        object.__setattr__(
            self, "__node_keys__", set(self._default_node_keys) | existing_node
        )
        object.__setattr__(
            self, "__edge_keys__", set(self._default_edge_keys) | existing_edge
        )
        object.__setattr__(
            self, "__system_keys__", set(self._default_system_keys) | existing_system
        )

    @model_validator(mode="after")
    def check_node_consistency(self) -> AtomicData:
        """Validate that all node-level properties have consistent atom counts.

        This validator runs after all field validators and checks that any node-level
        property that is set has the same number of nodes as atomic_numbers.

        Returns
        -------
        Self
            Returns self if validation passes.

        Raises
        ------
        ValueError
            If any node-level property has an inconsistent number of nodes.
        """
        num_atoms = len(self.atomic_numbers)
        node_keys = self.__dict__.get("__node_keys__", self._default_node_keys)
        for key in node_keys:
            tensor = getattr(self, key, None)
            if isinstance(tensor, torch.Tensor):
                if tensor.size(0) != num_atoms:
                    raise ValueError(
                        f"Inconsistent number of atoms in {key}: "
                        f"expected {num_atoms}, got {tensor.shape[0]}"
                    )
        return self

    @model_validator(mode="after")
    def check_edge_consistency(self) -> AtomicData:
        """Validate that all edge-level properties have consistent atom counts.

        This validator runs after all field validators and checks that any edge-level
        property that is set has the same number of edges as neighbor_list.

        Returns
        -------
        Self
            Returns self if validation passes.

        Raises
        ------
        ValueError
            If any edge-level property has an inconsistent number of edges.
        """
        if not isinstance(self.neighbor_list, torch.Tensor):
            return self
        num_edges = self.neighbor_list.size(0)

        edge_keys = self.__dict__.get("__edge_keys__", self._default_edge_keys)
        for key in edge_keys:
            tensor = getattr(self, key, None)
            if isinstance(tensor, torch.Tensor):
                if tensor.size(0) != num_edges:
                    raise ValueError(
                        f"Inconsistent number of edges in {key}: "
                        f"expected {num_edges}, got {tensor.shape[0]}"
                    )
        return self

    @model_validator(mode="after")
    def check_fp_dtype_consistency(self) -> AtomicData:
        """
        Ensures all floating point tensors are at the same precision
        as the positions tensor.
        """
        dtype = self.positions.dtype
        casted: list[str] = []
        for key in self.model_dump().keys():
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                tensor_dtype = value.dtype
                if tensor_dtype.is_floating_point and tensor_dtype != dtype:
                    # using __dict__ to avoid re-validation
                    self.__dict__[key] = value.to(dtype)
                    casted.append(key)
        if casted:
            casted.sort()
            # Keep the warning attributed to the user's AtomicData(...) call
            # instead of Pydantic's internal validation frames. This may need
            # adjustment if Pydantic's construction stack changes.
            warnings.warn(
                f"AtomicData fields {casted} were cast from their original "
                f"dtypes to {dtype} to match positions. "
                f"Pass tensors with matching dtypes to silence this warning.",
                UserWarning,
                stacklevel=3,
            )
        return self

    @model_validator(mode="after")
    def use_default_masses(self) -> AtomicData:
        """
        If no atomic masses are set, automatically fill in with
        default masses from ``periodictable``.

        Returns
        -------
        Self
            Returns self if validation passes.
        """
        if self.atomic_masses is None:
            masses_list = [pt.elements[int(n)].mass for n in self.atomic_numbers]
            # skip re-validation
            self.__dict__["atomic_masses"] = torch.as_tensor(
                masses_list,
                device=self.atomic_numbers.device,
                dtype=self.positions.dtype,
            )
        return self

    @model_validator(mode="after")
    def use_default_categories(self) -> AtomicData:
        """
        Check to make sure categories for atoms are set.

        In the case that a list is passed, which should be validated by
        ``pydantic``, we will convert it to a tensor.
        """
        if self.atom_categories is None:
            self.__dict__["atom_categories"] = torch.zeros_like(
                self.atomic_numbers, dtype=torch.long
            )
        elif isinstance(self.atom_categories, list):
            if not isinstance(self.atom_categories[0], t.AtomCategory):
                raise ValueError(
                    "Atom categories must be a list of `AtomCategory` enums"
                )
            self.atom_categories = torch.as_tensor(
                [cat.value for cat in self.atom_categories], dtype=torch.long
            )
        return self

    @model_validator(mode="after")
    def use_default_velocities(self) -> AtomicData:
        """
        If no velocities are set, initialize as zeros with proper shape and dtype.

        Returns
        -------
        Self
            Returns self if validation passes.
        """
        if self.velocities is None:
            # skip re-validation
            self.__dict__["velocities"] = torch.zeros_like(self.positions)
        return self

    @model_validator(mode="after")
    def enforce_device_consistency(self) -> AtomicData:
        """
        Enforces all tensors to be on the same device.

        In instances where the devices of atomic numbers and positions are
        different, we will try and promote them to offload over host CPU.
        """
        # we will use atomic numbers and positions as the "ground truth" as
        # they are required fields
        base_devices = list(
            {self.atomic_numbers.device.type, self.positions.device.type}
        )
        # sort the devices to be usable in a match statement
        base_devices = list(sorted(base_devices))
        match base_devices:
            case ["cuda"]:
                target_device = torch.device("cuda")
            case ["mps"]:
                target_device = torch.device("mps")
            case ["cpu", "cuda"]:
                target_device = torch.device("cuda")
            case ["cpu", "mps"]:
                target_device = torch.device("mps")
            # fall back to CPU for all other cases
            case _:
                target_device = torch.device("cpu")

        tensor_devices = [
            value.device.type
            for value in self.model_dump().values()
            if isinstance(value, torch.Tensor)
        ]
        if set(tensor_devices) != {target_device.type}:
            for key in (
                self.__node_keys__
                | self.__edge_keys__
                | self.__system_keys__
                | {"info"}
            ):
                value = getattr(self, key, None)
                if (
                    isinstance(value, torch.Tensor)
                    and value.device.type != target_device.type
                ):
                    # using __dict__ to avoid re-validation
                    self.__dict__[key] = value.to(target_device, non_blocking=False)
        return self

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    @property
    def device(self) -> torch.device:
        """Get the device of the positions tensor."""
        return self.positions.device

    @property
    def dtype(self) -> torch.dtype:
        """Get the dtype of the positions tensor."""
        return self.positions.dtype

    @property
    def node_properties(self) -> dict[str, Any]:
        """Get the node properties of the graph."""
        return self.model_dump(include=self.__node_keys__, exclude_none=True)

    @property
    def edge_properties(self) -> dict[str, Any]:
        """Get the edge properties of the graph."""
        return self.model_dump(include=self.__edge_keys__, exclude_none=True)

    @property
    def system_properties(self) -> dict[str, Any]:
        """Get the system properties of the graph."""
        return self.model_dump(include=self.__system_keys__, exclude_none=True)

    def add_node_property(
        self, key: str, value: torch.Tensor, node_dim: int = 0
    ) -> None:
        """Add a node property to the graph."""
        setattr(self, key, value)
        self.__node_keys__.add(key)

    def add_edge_property(self, key: str, value: Any) -> None:
        """Add an edge property to the graph."""
        setattr(self, key, value)
        self.__edge_keys__.add(key)

    def add_system_property(self, key: str, value: Any) -> None:
        """Add a system property to the graph."""
        setattr(self, key, value)
        self.__system_keys__.add(key)

    @property
    def chemical_hash(self) -> str:
        """Generate a unique hash for the chemical system using the blake2s
        hashing algorithm.

        The hash is unique to a given atomic composition and structure,
        invariant to the ordering of atoms in the data. The hash also
        differentiates between periodic and non-periodic systems, and for
        the former, lattice vectors and directions of periodicity.

        Returns
        -------
        str
            A ``blake2s`` hash string representing the chemical system.

        Notes
        -----
        The hash is generated by:
        1. Sorting atoms by atomic number to ensure invariance to atom ordering
        2. Including atomic numbers and positions of sorted atoms
        3. Including periodic boundary conditions and cell parameters if present
        4. Computing a BLAKE2s hash of the formatted string representation
        """
        atomic_numbers = self.atomic_numbers.cpu().numpy()
        sorted_idx = np.argsort(atomic_numbers)
        atomic_numbers = atomic_numbers[sorted_idx].tolist()
        positions = self.positions.cpu()[sorted_idx].tolist()
        # differentiate between periodic and non-periodic systems
        if self.pbc is not None and self.cell is not None:
            pbc = self.pbc.cpu().tolist()
            cell = self.cell.cpu().tolist()
        else:
            pbc = ""
            cell = ""
        formatted_str = f"{atomic_numbers}\n{positions}\n{pbc}\n{cell}"
        return blake2s(formatted_str.encode("utf-8"), digest_size=32).hexdigest()

    def __eq__(self, other: Any) -> bool:
        """
        Checks if two objects are indeed ``AtomicData``, and if so,
        returns if their chemical hashes are equal.

        Parameters
        ----------
        other : Any
            The object to compare with.

        Returns
        -------
        bool
            True if the chemical hashes are equal, False otherwise.
        """
        if not isinstance(other, AtomicData):
            return False
        return self.chemical_hash == other.chemical_hash

    @classmethod
    @OptionalDependency.ASE.require
    def from_atoms(
        cls,
        atoms: Atoms,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
        virials_key: str = "virials",
        dipole_key: str = "dipole",
        charges_key: str = "charges",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        z_table: AtomicNumberTable | None = None,
    ) -> AtomicData:
        """Create an AtomicData from an ASE-like Atoms object.

        Only fields that are actually present in the input object are
        populated; absent optional fields (energy, forces, stress, virials,
        dipole, charges) remain ``None``.  The input ``atoms`` object is
        **not** mutated.

        The returned ``info`` dict contains only tensor-convertible entries
        from ``atoms.info`` (``np.ndarray``, ``list``, ``int``, ``float``,
        and their numpy equivalents).  ``bool``, ``np.bool_``, strings, and
        other types are dropped.

        Parameters
        ----------
        atoms : ase.Atoms
            An ASE Atoms object.
        energy_key : str
            Key in ``atoms.info`` for total energy.
        forces_key : str
            Key in ``atoms.arrays`` for atomic forces.
        stress_key : str
            Key in ``atoms.info`` for the stress tensor.
        virials_key : str
            Key in ``atoms.info`` for the virial tensor.
        dipole_key : str
            Key in ``atoms.info`` for the dipole moment.
        charges_key : str
            Key in ``atoms.arrays`` for per-atom partial charges.
        device : str | torch.device
            Target device for all output tensors.
        dtype : torch.dtype
            Target floating-point dtype for all output tensors.
        z_table : AtomicNumberTable | None
            Atomic number table used to build one-hot node attributes.

        Returns
        -------
        AtomicData
        """
        # convert device to torch.device
        if isinstance(device, str):
            device = torch.device(device)

        # Get base components from ase.Atoms object
        atomic_numbers = torch.as_tensor(
            atoms.arrays["numbers"], device=device, dtype=torch.int32
        )
        positions = torch.as_tensor(
            atoms.arrays["positions"], device=device, dtype=dtype
        )
        pbc_array = atoms.get_pbc()
        if not pbc_array.any():
            pbc = None
            cell = None
        else:
            cell = torch.as_tensor(
                atoms.get_cell().array.reshape(1, 3, 3),
                device=device,
                dtype=dtype,
            )
            if torch.det(cell.squeeze(0)) <= 0.0:
                raise ValueError(
                    "Cell has undefined (zero) lattice vectors. "
                    "Please set the cell for all directions, "
                    "e.g. using atoms.center(vacuum=10.0)."
                )
            pbc = torch.as_tensor(pbc_array.reshape(1, 3), device=device)

        # Extract optional fields — absent fields remain None instead of
        # being fabricated as zero tensors.
        raw_energy = atoms.info.get(energy_key)
        energy = (
            torch.as_tensor(raw_energy, device=device, dtype=dtype).reshape(1, 1)
            if raw_energy is not None
            else None
        )

        raw_forces = atoms.arrays.get(forces_key)
        forces = (
            torch.as_tensor(raw_forces, device=device, dtype=dtype)
            if raw_forces is not None
            else None
        )

        raw_stress = atoms.info.get(stress_key)
        stress = (
            voigt_to_matrix(
                torch.as_tensor(raw_stress, device=device, dtype=dtype)
            ).unsqueeze(0)
            if raw_stress is not None
            else None
        )

        raw_virials = atoms.info.get(virials_key)
        virials = (
            voigt_to_matrix(
                torch.as_tensor(raw_virials, device=device, dtype=dtype)
            ).unsqueeze(0)
            if raw_virials is not None
            else None
        )

        raw_dipole = atoms.info.get(dipole_key)
        dipole = (
            torch.as_tensor(raw_dipole, device=device, dtype=dtype).reshape(1, 3)
            if raw_dipole is not None
            else None
        )

        raw_charges = atoms.arrays.get(charges_key)
        node_charges = (
            torch.as_tensor(raw_charges, device=device, dtype=dtype)
            if raw_charges is not None
            else None
        )

        # Read raw charge from original atoms.info before building local_info,
        # so it cannot be lost during normalization.
        raw_charge = atoms.info.get("charge")

        # Build local info dict with tensor-convertible entries only.
        # Do not mutate the caller's atoms.info.
        # Skip keys already consumed into dedicated AtomicData fields.
        _consumed_info_keys = {
            energy_key,
            stress_key,
            virials_key,
            dipole_key,
            "charge",
        }
        local_info: dict[str, torch.Tensor] = {}
        for key, value in atoms.info.items():
            if key in _consumed_info_keys:
                continue
            if isinstance(value, (np.ndarray, list)):
                local_info[key] = torch.as_tensor(value, device=device, dtype=dtype)
            elif isinstance(
                value, (int, float, np.integer, np.floating)
            ) and not isinstance(value, (bool, np.bool_)):
                local_info[key] = torch.as_tensor([value], device=device, dtype=dtype)

        # Derive graph-level charge
        if raw_charge is not None:
            if not isinstance(raw_charge, numbers.Integral):
                raise ValueError(
                    f"atoms.info['charge'] must be an integer, "
                    f"got {type(raw_charge).__name__}: {raw_charge}"
                )
            charge = torch.as_tensor([[int(raw_charge)]], device=device, dtype=dtype)
        elif node_charges is not None:
            _charge_f = torch.sum(node_charges)
            _charge = int(_charge_f.round().item())
            if (_charge_f - _charge).abs() >= 1.0e-2:
                raise ValueError(f"Non-integer sum of atomic charges: {_charge_f}")
            charge = torch.as_tensor([[_charge]], device=device, dtype=dtype)
        else:
            charge = None

        node_attrs = None
        if z_table is not None:
            indices = torch.as_tensor(
                atomic_numbers_to_indices(atoms.arrays["numbers"], z_table=z_table),
                device=device,
            )
            node_attrs = to_one_hot(
                indices.unsqueeze(-1),
                num_classes=len(z_table),
            ).to(dtype)

        masses_tensor = torch.from_numpy(atoms.get_masses()).to(device, dtype)
        return cls(
            atomic_masses=masses_tensor,
            atomic_numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=pbc,
            node_attrs=node_attrs,  # type: ignore
            forces=forces,
            energy=energy,
            stress=stress,
            virial=virials,
            dipole=dipole,
            charges=node_charges,
            charge=charge,
            info=local_info,
        )

    @classmethod
    @OptionalDependency.PYMATGEN.require
    def from_structure(
        cls,
        structure: Structure | Molecule,
        energy_key: str = "energy",
        forces_key: str = "forces",
        stress_key: str = "stress",
        virials_key: str = "virials",
        dipole_key: str = "dipole",
        charges_key: str = "charges",
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
        z_table: AtomicNumberTable | None = None,
    ) -> AtomicData:
        """Create an AtomicData from a pymatgen Structure or Molecule.

        Only fields that are actually present in the input are populated;
        absent optional fields (energy, forces, stress, virials, dipole,
        charges) remain ``None``.  The input object is **not** mutated.

        The returned ``info`` dict contains tensor-convertible entries
        from ``structure.properties`` (``np.ndarray``, ``list``, ``int``,
        ``float``, and their numpy equivalents), excluding keys already
        consumed into dedicated fields.  Unsupported types raise
        ``TypeError``.

        Stress and virials accept 3×3 matrices, 6-component Voigt vectors,
        or 9-component flat vectors (see :func:`voigt_to_matrix`).

        Parameters
        ----------
        structure : pymatgen.core.Structure | pymatgen.core.Molecule
            A pymatgen Structure (periodic) or Molecule (non-periodic).
            For Molecule, ``cell`` and ``pbc`` are set to ``None``.
        energy_key : str
            Key in ``structure.properties`` for total energy.
        forces_key : str
            Key in ``structure.site_properties`` for atomic forces.
        stress_key : str
            Key in ``structure.properties`` for the stress tensor.
        virials_key : str
            Key in ``structure.properties`` for the virial tensor.
        dipole_key : str
            Key in ``structure.properties`` for the dipole moment.
        charges_key : str
            Key in ``structure.site_properties`` for per-atom partial charges.
        device : str | torch.device
            Target device for all output tensors.
        dtype : torch.dtype
            Target floating-point dtype for all output tensors.
        z_table : AtomicNumberTable | None
            Atomic number table used to build one-hot node attributes.

        Returns
        -------
        AtomicData
        """
        if isinstance(device, str):
            device = torch.device(device)

        atomic_numbers = torch.as_tensor(
            structure.atomic_numbers, device=device, dtype=torch.int32
        )
        positions = torch.as_tensor(structure.cart_coords, device=device, dtype=dtype)

        # Cell and pbc handling
        if hasattr(structure, "lattice"):
            pbc_tuple = structure.pbc
            if not any(pbc_tuple):
                pbc = None
                cell = None
            else:
                cell = torch.as_tensor(
                    structure.lattice.matrix.copy().reshape(1, 3, 3),
                    device=device,
                    dtype=dtype,
                )
                pbc = torch.as_tensor(pbc_tuple, device=device).reshape(1, 3)
        else:
            pbc = None
            cell = None

        # Extract optional fields from properties (system-level)
        # and site_properties (per-atom).
        raw_energy = structure.properties.get(energy_key)
        energy = (
            torch.as_tensor([[raw_energy]], device=device, dtype=dtype)
            if raw_energy is not None
            else None
        )

        raw_forces = structure.site_properties.get(forces_key)
        forces = (
            torch.as_tensor(raw_forces, device=device, dtype=dtype)
            if raw_forces is not None
            else None
        )

        raw_stress = structure.properties.get(stress_key)
        stress = (
            voigt_to_matrix(
                torch.as_tensor(raw_stress, device=device, dtype=dtype)
            ).unsqueeze(0)
            if raw_stress is not None
            else None
        )

        raw_virials = structure.properties.get(virials_key)
        virials = (
            voigt_to_matrix(
                torch.as_tensor(raw_virials, device=device, dtype=dtype)
            ).unsqueeze(0)
            if raw_virials is not None
            else None
        )

        raw_dipole = structure.properties.get(dipole_key)
        dipole = (
            torch.as_tensor(raw_dipole, device=device, dtype=dtype).reshape(1, 3)
            if raw_dipole is not None
            else None
        )

        raw_charges = structure.site_properties.get(charges_key)
        node_charges = (
            torch.as_tensor(raw_charges, device=device, dtype=dtype)
            if raw_charges is not None
            else None
        )

        # Build local info dict from remaining structure.properties.
        _consumed_props_keys = {
            energy_key,
            stress_key,
            virials_key,
            dipole_key,
        }
        local_info: dict[str, torch.Tensor] = {}
        for key, value in structure.properties.items():
            if key in _consumed_props_keys:
                continue
            if isinstance(value, (np.ndarray, list)):
                local_info[key] = torch.as_tensor(value, device=device, dtype=dtype)
            elif isinstance(
                value, (int, float, np.integer, np.floating)
            ) and not isinstance(value, (bool, np.bool_)):
                local_info[key] = torch.as_tensor([value], device=device, dtype=dtype)
            else:
                raise TypeError(
                    f"Cannot convert structure.properties['{key}'] of type "
                    f"{type(value).__name__} to a tensor."
                )

        # Derive graph-level charge.
        # pymatgen stores charge as float (e.g. 2 → 2.0); round before int cast.
        if structure._charge is not None:
            _charge = structure.charge
            if abs(_charge - round(_charge)) >= 1e-2:
                raise ValueError(f"Structure charge must be an integer, got {_charge}")
            charge = torch.as_tensor(
                [[int(round(_charge))]], device=device, dtype=dtype
            )
        elif node_charges is not None:
            _charge_f = torch.sum(node_charges)
            _charge_i = int(_charge_f.round().item())
            if (_charge_f - _charge_i).abs() >= 1.0e-2:
                raise ValueError(f"Non-integer sum of atomic charges: {_charge_f}")
            charge = torch.as_tensor([[_charge_i]], device=device, dtype=dtype)
        else:
            charge = None

        node_attrs = None
        if z_table is not None:
            indices = torch.as_tensor(
                atomic_numbers_to_indices(
                    list(structure.atomic_numbers), z_table=z_table
                ),
                device=device,
            )
            node_attrs = to_one_hot(
                indices.unsqueeze(-1),
                num_classes=len(z_table),
            ).to(dtype)

        masses = torch.tensor(
            [float(sp.atomic_mass) for sp in structure.species],
            device=device,
            dtype=dtype,
        )

        return cls(
            atomic_masses=masses,
            atomic_numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=pbc,
            node_attrs=node_attrs,  # type: ignore
            forces=forces,
            energy=energy,
            stress=stress,
            virial=virials,
            dipole=dipole,
            charges=node_charges,
            charge=charge,
            info=local_info,
        )

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.atomic_numbers)

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        if self.neighbor_list is None:
            return 0
        return self.neighbor_list.shape[0]


def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """
    Generates one-hot encoding
    """
    shape = indices.shape[:-1] + (num_classes,)
    oh = torch.zeros(shape, device=indices.device).view(shape)

    # scatter_ is the in-place version of scatter
    oh.scatter_(dim=-1, index=indices, value=1)

    return oh.view(*shape)


def voigt_to_matrix(t: torch.Tensor) -> torch.Tensor:
    """
    Convert voigt notation to matrix notation
    """
    if t.shape == (3, 3):
        return t
    if t.shape == (6,):
        return torch.tensor(
            [
                [t[0], t[5], t[4]],
                [t[5], t[1], t[3]],
                [t[4], t[3], t[2]],
            ],
            dtype=t.dtype,
            device=t.device,
        )
    if t.shape == (9,):
        return t.view(3, 3)

    raise ValueError(
        f"Stress tensor must be of shape (6,) or (3, 3), or (9,) but has shape {t.shape}"
    )


def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    """
    Convert atomic numbers to indices
    """
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)
