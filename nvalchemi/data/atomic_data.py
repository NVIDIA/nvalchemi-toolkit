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

from collections.abc import Sequence
from hashlib import blake2s
from typing import Annotated, Any, ClassVar

import numpy as np
import periodictable as pt
import torch
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, model_validator

from nvalchemi import _typing as t
from nvalchemi.data.data import DataMixin  # type: ignore


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

    **Units**: This framework is unit-agnostic — the integrators and neighbor
    list routines work with any self-consistent set of units.  Models, however,
    have units baked into their parameters or training data and must document
    what they expect.  When the automatic mass lookup is used
    (:meth:`use_default_masses`), masses are populated in amu from the
    periodic table, so the implied energy/length units of any model also
    determine the time unit (for eV/Å/amu, 1 natural time unit ≈ 10.18 fs).
    Temperature is always supplied in Kelvin; the thermostat kernels convert
    internally using :math:`k_B = 8.617 \times 10^{-5}` eV/K, so models
    using the built-in thermostats are expected to work in eV.

    **Stress convention**: ``stresses`` and ``virials`` store the **positive
    raw virial** :math:`W = +\sum_{ij} r_{ij} \otimes F_{ij}` in the model's
    energy unit (not divided by volume).  The instantaneous pressure tensor
    :math:`P = (2\,KE + W)/V` is computed by ``compute_pressure_tensor`` in
    the NPT/NPH kernels, where :math:`V` is in the cube of the model's length
    unit.

    Attributes
    ----------
    atomic_numbers : torch.Tensor
        Atomic numbers of each atom [n_nodes]
    positions : torch.Tensor
        Cartesian coordinates [n_nodes, 3] in the model's length unit.
    atomic_masses : torch.Tensor
        Atomic masses [n_nodes]; auto-populated in amu from the periodic
        table if not provided (see :meth:`use_default_masses`).
    edge_index : torch.Tensor
        Edge index [2, n_edges].
    node_attrs : torch.Tensor
        Node attributes [n_nodes, n_node_feats].
    shifts : torch.Tensor
        Physical PBC shift vectors for each edge [n_edges, 3], same length
        unit as positions.
    unit_shifts : torch.Tensor
        Integer PBC image indices for each edge [n_edges, 3]; dimensionless.
    cell : torch.Tensor
        Lattice vectors [3, 3] in the model's length unit; rows are the
        a, b, c lattice vectors.
    pbc : torch.Tensor
        Periodic boundary conditions [3] (bool).
    forces : torch.Tensor
        Atomic forces [n_nodes, 3] in the model's energy / length unit.
    energies : torch.Tensor
        Potential energy [1] in the model's energy unit.
    stresses : torch.Tensor
        Positive raw virial :math:`W = +\sum_{ij} r_{ij} \otimes F_{ij}`
        [1, 3, 3] in the model's energy unit.  Divide by cell volume to
        get the Cauchy stress (energy / length\\ :sup:`3`).
    virials : torch.Tensor
        Positive raw virial [1, 3, 3] in the model's energy unit.
    dipoles : torch.Tensor
        Dipole moment [1, 3]; units depend on the model's charge and length
        convention.
    node_charges : torch.Tensor
        Partial atomic charges [n_nodes]; unit depends on the model's charge
        convention.
    graph_charges : torch.Tensor
        Total system charge [1]; same unit as ``node_charges``.
    velocities : torch.Tensor
        Atomic velocities [n_nodes, 3].  Must be consistent with masses,
        forces, and ``dt`` so that :math:`KE = \tfrac{1}{2}mv^2` is in
        the model's energy unit.
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
        Field(
            description="Cartesian coordinates for each atom [n_nodes, 3] in the model's length unit"
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ]
    # Optional fields with defaults
    atomic_masses: Annotated[
        t.AtomicMasses | None,
        Field(description="Atomic masses [n_nodes] in amu (atomic mass units)"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    atom_categories: Annotated[
        list[t.AtomCategory] | t.AtomCategories | None,
        Field(
            description="Atom categorical index, based on _typing.AtomCategory Enum [n_nodes]"
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    edge_index: Annotated[
        t.EdgeIndex | None,
        Field(description="Edge index [2, n_edges]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    shifts: Annotated[
        t.PeriodicShifts | None,
        Field(description="Shifts for each edge [n_edges, 3]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    unit_shifts: Annotated[
        t.PeriodicUnitShifts | None,
        Field(description="Additional shifts for each edge [n_edges, 3]"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    cell: Annotated[
        t.LatticeVectors | None,
        Field(
            description="Lattice vectors [3, 3] in the model's length unit; rows are the a, b, c vectors"
        ),
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
        Field(
            description="Atomic forces [n_nodes, 3] in the model's energy / length unit"
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    energies: Annotated[
        t.Energy | None,
        Field(description="Potential energy [1] in the model's energy unit"),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    stresses: Annotated[
        t.Stress | None,
        Field(
            description=(
                "Positive raw virial W = +sum r_ij x F_ij [1, 3, 3] in the "
                "model's energy unit. "
                "Divide by cell volume to get the Cauchy stress. "
                "compute_pressure_tensor divides by V internally for NPT/NPH."
            )
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    virials: Annotated[
        t.Virials | None,
        Field(
            description="Positive raw virial tensor [1, 3, 3] in the model's energy unit (W = +sum r_ij x F_ij)"
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    dipoles: Annotated[
        t.Dipole | None,
        Field(
            description="Dipole moment [1, 3]; units depend on the model's charge and length convention"
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    node_charges: Annotated[
        t.NodeCharges | None,
        Field(
            description="Partial atomic charges [n_nodes] in the model's charge unit"
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    graph_charges: Annotated[
        t.GraphCharges | None,
        Field(
            description="Total system charge [1] in the same charge unit as node_charges"
        ),
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

    graph_spins: Annotated[
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
        Field(
            description=(
                "Atomic velocities [n_nodes, 3] in the model's velocity unit. "
                "Must be consistent with masses and forces so that "
                "KE = 0.5 * m * v^2 is in the model's energy unit."
            )
        ),
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
            description="Per-atom kinetic energies [n_nodes, 1], with the same units as energies."
        ),
        PlainSerializer(_tensor_serialization, when_used="json"),
    ] = None

    info: dict[str, torch.Tensor] = Field(default_factory=dict)
    __node_keys__: set[str] = {
        "atomic_masses",
        "positions",
        "forces",
        "positions",
        "node_charges",
        "node_embeddings",
        "atomic_numbers",
        "node_attrs",
        "node_alpha_spins",
        "node_beta_spins",
        "atom_categories",
        "velocities",
        "momenta",
        "kinetic_energies",
    }
    __edge_keys__: set[str] = {"shifts", "unit_shifts", "edge_index", "edge_embeddings"}
    __system_keys__: set[str] = {
        "energies",
        "stresses",
        "virials",
        "dipoles",
        "graph_charges",
        "graph_embeddings",
        "cell",
        "pbc",
        "graph_spins",
    }

    # Pydantic configuration
    model_config: ClassVar[ConfigDict] = ConfigDict(
        arbitrary_types_allowed=True, validate_assignment=True, extra="allow"
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

        for key in self.__node_keys__:
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
        property that is set has the same number of edges as edge_index.

        Returns
        -------
        Self
            Returns self if validation passes.

        Raises
        ------
        ValueError
            If any edge-level property has an inconsistent number of edges.
        """
        # if we don't have a way to reliably determine edge count, skip validation
        if not isinstance(self.edge_index, torch.Tensor):
            return self
        num_edges = self.edge_index.size(1)

        # Dictionary of field name to its first dimension (num atoms)
        for key in self.__edge_keys__:
            tensor = getattr(self, key, None)
            if isinstance(tensor, torch.Tensor):
                dim = 1 if key == "edge_index" else 0
                if tensor.size(dim) != num_edges:
                    raise ValueError(
                        f"Inconsistent number of edges in {key}: "
                        f"expected {num_edges}, got {tensor.shape[dim]}"
                    )
        return self

    @model_validator(mode="after")
    def check_fp_dtype_consistency(self) -> AtomicData:
        """
        Ensures all floating point tensors are at the same precision
        as the positions tensor.
        """
        dtype = self.positions.dtype
        for key in self.model_dump().keys():
            value = getattr(self, key)
            if isinstance(value, torch.Tensor):
                tensor_dtype = value.dtype
                if tensor_dtype.is_floating_point and tensor_dtype != dtype:
                    # using __dict__ to avoid re-validation
                    self.__dict__[key] = value.to(dtype)
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
            masses = [pt.elements[int(n)].mass for n in self.atomic_numbers]
            # skip re-validation
            self.__dict__["atomic_masses"] = torch.as_tensor(
                masses, device=self.atomic_numbers.device, dtype=self.positions.dtype
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
    def from_atoms(
        cls,
        atoms,
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
        """Creates AtomicData from a data structure.

        Parameters
        ----------
        atoms : Any
            The data structure to convert to AtomicData.
        energy_key : str
            The key to get the energy from the data structure.
        forces_key : str
            The key to get the forces from the data structure.
        stress_key : str
            The key to get the stress from the data structure.
        virials_key : str
            The key to get the virials from the data structure.
        dipole_key : str
            The key to get the dipole from the data structure.
        charges_key : str
            The key to get the charges from the data structure.
        device : str | torch.device
            The device to convert the data to.
        dtype : torch.dtype
            The dtype to convert the data to.
        z_table : AtomicNumberTable | None
            The atomic number table to use for the atomic numbers.
        Returns
        -------
        AtomicData
        """
        # convert device to torch.device
        if isinstance(device, str):
            device = torch.device(device)

        # Get base components from ase.Atoms object
        atomic_numbers = torch.as_tensor(
            atoms.arrays["numbers"], device=device, dtype=torch.long
        )
        positions = torch.as_tensor(
            atoms.arrays["positions"], device=device, dtype=dtype
        )
        pbc = torch.as_tensor(atoms.get_pbc().reshape(1, 3), device=device)
        cell = torch.as_tensor(
            np.array(atoms.get_cell(complete=True)).reshape(1, 3, 3),
            device=device,
            dtype=dtype,
        )

        # Get info from the data structure
        energy = torch.as_tensor(
            atoms.info.get(energy_key, [[0.0]]), device=device, dtype=dtype
        )  # eV
        forces = torch.as_tensor(
            atoms.arrays.get(
                forces_key,
                torch.zeros((len(atomic_numbers), 3), device=device, dtype=dtype),
            ),
            device=device,
            dtype=dtype,
        )  # eV / Ang
        stress = torch.as_tensor(
            atoms.info.get(stress_key, torch.zeros((3, 3), device=device, dtype=dtype)),
            device=device,
            dtype=dtype,
        )  # eV / Ang ^ 3
        virials = torch.as_tensor(
            atoms.info.get(
                virials_key, torch.zeros((3, 3), device=device, dtype=dtype)
            ),
            device=device,
            dtype=dtype,
        )
        dipole = torch.as_tensor(
            atoms.info.get(dipole_key, torch.zeros((1, 3), device=device, dtype=dtype)),
            device=device,
            dtype=dtype,
        )  # Debye

        node_charges = torch.as_tensor(
            atoms.arrays.get(
                charges_key,
                torch.zeros((len(atomic_numbers),), device=device, dtype=dtype),
            ),
            device=device,
            dtype=dtype,
        )
        # map tags to AtomCategory enum based off adsorbate construction
        tags = atoms.get_tags()
        # per docs, 0 = adsorbate, and >= 1 are atom layers
        atom_categories = torch.as_tensor(tags)
        atom_categories[atom_categories == 0] = t.AtomCategory.GAS.value
        atom_categories[atom_categories == 1] = t.AtomCategory.SURFACE.value
        atom_categories[atom_categories >= 2] = t.AtomCategory.BULK.value

        # Convert info arrays to tensors
        keys_to_remove = []
        for key, value in atoms.info.items():
            if isinstance(value, (np.ndarray, list)):
                atoms.info[key] = torch.as_tensor(value, device=device, dtype=dtype)
            elif isinstance(value, float):
                atoms.info[key] = torch.as_tensor([value], device=device, dtype=dtype)
            else:
                keys_to_remove.append(key)
        for key in keys_to_remove:
            atoms.info.pop(key)

        # make sure charges meets the expected shape
        if node_charges.ndim == 1:
            node_charges.unsqueeze_(-1)
        # try to get total system charge from atoms.info data
        if "charge" in atoms.info:
            _charge = atoms.info["charge"]
            assert isinstance(  # noqa: S101
                _charge, int
            ), f"Non-integer total charge in atoms.info: {_charge}"
        else:
            _charge_f = torch.sum(node_charges)
            _charge = int(_charge_f.round().item())
            assert (  # noqa: S101
                _charge_f - _charge
            ).abs() < 1.0e-2, f"Non-integer sum of atomic charges: {_charge_f}"
        charge = torch.as_tensor([[_charge]], device=device, dtype=dtype)
        stress = voigt_to_matrix(stress).unsqueeze(0)
        virials = voigt_to_matrix(virials).unsqueeze(0)

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

        masses = torch.from_numpy(atoms.get_masses()).to(device, dtype)
        return cls(
            atomic_masses=masses,
            atomic_numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=pbc,
            node_attrs=node_attrs,  # type: ignore
            forces=forces,
            energies=energy,
            stresses=stress,
            virials=virials,
            dipoles=dipole,
            node_charges=node_charges,
            graph_charges=charge,
            info=atoms.info,
            atom_categories=atom_categories,
        )

    @property
    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        return len(self.atomic_numbers)

    @property
    def num_edges(self) -> int:
        """Return the number of edges in the graph."""
        if self.edge_index is None:
            return 0
        return self.edge_index.shape[1]


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
