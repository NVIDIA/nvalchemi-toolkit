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
"""Optional RDKit preparation for CSP molecular formula inputs.

RDKit-specific calls are guarded by :class:`~nvalchemi.OptionalDependency`.
Importing this module or :mod:`nvalchemi.csp` does not import RDKit.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeAlias

import torch
from pydantic import BaseModel, ConfigDict, model_validator
from torch import Tensor

from nvalchemi._optional import OptionalDependency
from nvalchemi.csp.data import MolecularPackingInput
from nvalchemi.data.batch import Batch

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol

JSONValue: TypeAlias = (
    str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)

__all__ = [
    "JSONValue",
    "RDKitConformerConfig",
    "TopologicalAtomTypeMap",
    "build_contact_distance_matrix",
    "build_molecular_packing_input",
    "generate_conformers_from_smiles",
    "topological_atom_types_from_connectivity",
    "topological_atom_types_from_mol",
]


def topological_atom_types_from_connectivity(
    atomic_numbers: Tensor,
    adjacency: Tensor,
) -> Tensor:
    """Label atoms with equivalent element-and-bond connectivity.

    Parameters
    ----------
    atomic_numbers : torch.Tensor, shape ``[N]``
        Integral atomic numbers in the range 1 through 118.
    adjacency : torch.Tensor, shape ``[N, N]``
        Boolean or binary integer matrix where ``adjacency[i, j]`` is true or 1
        when atoms ``i`` and ``j`` are bonded. It must be symmetric, with false
        or zero on its diagonal.

    Returns
    -------
    torch.Tensor, dtype ``torch.int32``, shape ``[N]``
        One local type ID per atom on the same device as ``atomic_numbers``.
        Interchangeable atoms share an ID; IDs follow the first atom in each
        group.

    Raises
    ------
    TypeError
        If either input is not a tensor or has an unsupported dtype.
    ValueError
        If array shapes, element numbers, or bond connections are invalid.

    Notes
    -----
    These labels restrict which atoms can match during structure comparison.
    Two atoms share a type if the molecule can be relabeled to exchange them
    while preserving every element identity and bond connection. Bond order,
    charge, isotope labels, stereochemistry, and geometry are not considered.
    The check runs on CPU; output stays on the atomic-number tensor's device.

    In the example below, ethanol (CH3-CH2-OH) has six types in atom order:
    the methyl carbon, its three equivalent hydrogens, the methylene carbon,
    its two equivalent hydrogens, the oxygen, and the hydroxyl hydrogen.

    Examples
    --------
    >>> numbers = torch.tensor([6, 1, 1, 1, 6, 1, 1, 8, 1])
    >>> bonds = torch.tensor([[0, 1], [0, 2], [0, 3], [0, 4],
    ...                       [4, 5], [4, 6], [4, 7], [7, 8]])
    >>> adjacency = torch.zeros((9, 9), dtype=torch.bool)
    >>> adjacency[bonds[:, 0], bonds[:, 1]] = True
    >>> adjacency[bonds[:, 1], bonds[:, 0]] = True
    >>> topological_atom_types_from_connectivity(numbers, adjacency)
    tensor([0, 1, 1, 1, 2, 3, 3, 4, 5], dtype=torch.int32)
    """
    numbers, edges = _validated_connectivity_inputs(atomic_numbers, adjacency)
    from nvalchemi.csp._chem._topology import topological_atom_types

    ids = topological_atom_types(numbers, edges)
    return torch.tensor(ids, dtype=torch.int32, device=atomic_numbers.device)


def _validated_connectivity_inputs(
    atomic_numbers: Tensor, adjacency: Tensor
) -> tuple[list[int], list[list[bool]]]:
    """Validate molecular graph tensors and return owned CPU Python lists."""
    if not isinstance(atomic_numbers, Tensor):
        raise TypeError("atomic_numbers must be a torch.Tensor")
    if not isinstance(adjacency, Tensor):
        raise TypeError("adjacency must be a torch.Tensor")
    integer_dtypes = {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint16,
        torch.uint32,
        torch.uint64,
    }
    if atomic_numbers.dtype not in integer_dtypes:
        raise TypeError("atomic_numbers must have an integral dtype")
    if adjacency.dtype != torch.bool and adjacency.dtype not in integer_dtypes:
        raise TypeError("adjacency must have a boolean or integral dtype")
    if atomic_numbers.ndim != 1:
        raise ValueError("atomic_numbers must have shape [N]")
    atom_count = atomic_numbers.shape[0]
    if adjacency.shape != (atom_count, atom_count):
        raise ValueError("adjacency must have shape [N, N]")

    numbers = atomic_numbers.detach().to(device="cpu", dtype=torch.int64)
    edges = adjacency.detach().to(device="cpu")
    if bool(((numbers < 1) | (numbers > 118)).any()):
        raise ValueError("atomic_numbers must be in the range 1 through 118")
    if edges.dtype != torch.bool and bool(((edges != 0) & (edges != 1)).any()):
        raise ValueError("adjacency must contain only binary values")
    edges = edges.to(dtype=torch.bool)
    if bool(torch.diagonal(edges).any()):
        raise ValueError("adjacency must have a zero diagonal")
    if not torch.equal(edges, edges.T):
        raise ValueError("adjacency must be symmetric")

    return numbers.tolist(), edges.tolist()


@OptionalDependency.RDKIT.require
def topological_atom_types_from_mol(mol: Mol) -> Tensor:
    """Group RDKit atoms by their element-and-bond connectivity.

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        Molecule whose explicit atoms and bond presence define the graph.
        Atom and bond annotations other than atomic number and bond presence
        are ignored; implicit hydrogens are not added.

    Returns
    -------
    torch.Tensor, dtype ``torch.int32``, shape ``[N]``
        CPU type IDs in RDKit atom order. IDs are local to this molecule and
        assigned by first atom occurrence; separate molecules need a shared
        type mapping before cross-structure comparison.

    Raises
    ------
    OptionalDependencyError
        If RDKit is unavailable. Install ``nvalchemi-toolkit[rdkit]``.
    TypeError
        If ``mol`` is not an RDKit molecule.

    See Also
    --------
    topological_atom_types_from_connectivity
        The same typing from explicit atomic numbers and bond connections.
    """
    from rdkit import Chem

    if not isinstance(mol, Chem.Mol):
        raise TypeError("mol must be an rdkit.Chem.Mol")
    atom_count = mol.GetNumAtoms()
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    adjacency = [[False] * atom_count for _ in range(atom_count)]
    for bond in mol.GetBonds():
        left = bond.GetBeginAtomIdx()
        right = bond.GetEndAtomIdx()
        adjacency[left][right] = adjacency[right][left] = True

    from nvalchemi.csp._chem._topology import topological_atom_types

    return torch.tensor(
        topological_atom_types(atomic_numbers, adjacency), dtype=torch.int32
    )


class TopologicalAtomTypeMap:
    """Reuse reference molecular topology labels across reordered structures.

    Construct one map from an ordered base formula unit and its complete
    molecular adjacency. The map copies both inputs. For :meth:`for_batch`,
    callers attest that source indices refer to this exact ordered base formula
    unit and that relaxation preserved its molecular connectivity. The method
    checks field lengths and atomic-number agreement, but a ``Batch`` alone
    cannot establish that source indices have the claimed meaning. For
    :meth:`for_connectivity`, each supplied target component is checked against
    a reference component by exact element and bond-preserving atom reordering.

    Parameters
    ----------
    atomic_numbers : torch.Tensor, shape ``[N]``
        Ordered base formula-unit atomic numbers with an integral dtype and
        values from 1 through 118.
    adjacency : torch.Tensor, shape ``[N, N]``
        Boolean or binary integer molecular bond adjacency. It must be
        symmetric and have a zero diagonal; floating-point adjacency is not
        accepted.

    Notes
    -----
    Atoms that can exchange places while preserving every element and bond in
    the complete base formula unit receive the same type ID. Matching
    disconnected reference molecules share corresponding labels.
    ``for_connectivity`` labels each target component that has an exact
    reference match; it does not check component proportions against the base
    formula unit. Bond order, charge, isotope labels, stereochemistry, and
    geometry are not part of the topology contract.

    Examples
    --------
    >>> base_numbers = torch.tensor([6, 6, 8])
    >>> base_edges = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    >>> atom_types = TopologicalAtomTypeMap(base_numbers, base_edges)
    >>> atom_types.for_connectivity(
    ...     torch.tensor([8, 6, 6]),
    ...     torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
    ... )
    tensor([2, 1, 0], dtype=torch.int32)
    """

    __slots__ = ("_atomic_numbers", "_adjacency", "_atom_types", "_components")

    def __init__(self, atomic_numbers: Tensor, adjacency: Tensor) -> None:
        """Validate and snapshot the ordered reference graph."""
        types = topological_atom_types_from_connectivity(atomic_numbers, adjacency)
        numbers = atomic_numbers.detach().to(device="cpu", dtype=torch.int64).tolist()
        edges = adjacency.detach().to(device="cpu", dtype=torch.bool).tolist()
        from nvalchemi.csp._chem._topology import connected_components

        self._atomic_numbers = tuple(int(number) for number in numbers)
        self._adjacency = tuple(tuple(bool(edge) for edge in row) for row in edges)
        self._atom_types = tuple(int(atom_type) for atom_type in types.tolist())
        self._components = tuple(
            tuple(component) for component in connected_components(edges)
        )

    def for_batch(self, batch: Batch) -> Tensor:
        """Map a topology-preserving expanded P1 batch to shared atom types.

        Parameters
        ----------
        batch : nvalchemi.data.batch.Batch
            Batch with atom-level ``atomic_numbers`` and
            ``csp_source_asu_atom_index`` fields. Provenance indices may repeat
            across formula-unit copies and may occur in any atom order.

        Returns
        -------
        torch.Tensor, dtype ``torch.int32``, shape ``[batch.num_nodes]``
            Shared reference atom type for each batch atom, on the batch's
            atomic-number device.

        Raises
        ------
        TypeError
            If ``batch`` is not a ``Batch``, or an atom field has an
            unsupported dtype.
        ValueError
            If atom fields are missing or not tensors, their shapes or lengths
            do not align, an index is negative, a nonempty batch uses an empty
            reference formula, a structure atom count is not a multiple of the
            reference formula size, or a source index has a different atomic
            number.
        """
        if not isinstance(batch, Batch):
            raise TypeError("batch must be a nvalchemi.data.batch.Batch")
        try:
            atomic_numbers = batch.atomic_numbers
        except AttributeError:
            atomic_numbers = None
        try:
            source_indices = batch.csp_source_asu_atom_index
        except AttributeError:
            source_indices = None
        if not isinstance(atomic_numbers, Tensor):
            raise ValueError("batch must contain atomic_numbers")
        if not isinstance(source_indices, Tensor):
            raise ValueError("batch must contain csp_source_asu_atom_index")
        integer_dtypes = {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint16,
            torch.uint32,
            torch.uint64,
        }
        if atomic_numbers.dtype not in integer_dtypes:
            raise TypeError("batch atomic_numbers must have an integral dtype")
        if source_indices.dtype not in integer_dtypes:
            raise TypeError("csp_source_asu_atom_index must have an integral dtype")
        if atomic_numbers.ndim != 1 or source_indices.shape != atomic_numbers.shape:
            raise ValueError("batch atomic_numbers and provenance must align as [N]")
        if batch.num_nodes != atomic_numbers.numel():
            raise ValueError("batch atomic_numbers must align with batch.num_nodes")
        if len(self._atomic_numbers) == 0 and batch.num_nodes:
            raise ValueError("nonempty batch cannot map to an empty reference formula")
        if len(self._atomic_numbers) and any(
            atom_count % len(self._atomic_numbers)
            for atom_count in batch.num_nodes_list
        ):
            raise ValueError(
                "each batch structure atom count must be a multiple of the reference formula"
            )
        indices = source_indices.detach().to(device="cpu").tolist()
        if any(index < 0 for index in indices):
            raise ValueError("csp_source_asu_atom_index values must be nonnegative")
        formula_indices = (
            [index % len(self._atomic_numbers) for index in indices] if indices else []
        )
        actual_numbers = atomic_numbers.detach().to(device="cpu").tolist()
        if any(
            actual != self._atomic_numbers[formula_index]
            for actual, formula_index in zip(
                actual_numbers, formula_indices, strict=True
            )
        ):
            raise ValueError("batch atomic_numbers disagree with source provenance")
        labels = [self._atom_types[index] for index in formula_indices]
        return torch.tensor(labels, dtype=torch.int32, device=atomic_numbers.device)

    def for_connectivity(self, atomic_numbers: Tensor, adjacency: Tensor) -> Tensor:
        """Map complete target graph components onto reference atom types.

        Each connected target component must match a reference component after
        reordering atoms while preserving atomic numbers and bonded pairs.
        Target atom order is preserved in the returned shared labels. Repeated
        and mixed reference components are supported. This method checks that
        each target component is known, but does not check the counts of the
        different components against the reference formula unit.

        Parameters
        ----------
        atomic_numbers : torch.Tensor, shape ``[N]``
            Target atomic numbers in target atom order, with an integral dtype
            and values from 1 through 118.
        adjacency : torch.Tensor, shape ``[N, N]``
            Complete target molecular adjacency with a boolean or binary
            integer dtype, symmetry, and a zero diagonal. Floating-point
            adjacency is not accepted.

        Returns
        -------
        torch.Tensor, dtype ``torch.int32``, shape ``[N]``
            Shared reference atom types in target atom order, on the target
            atomic-number device.

        Raises
        ------
        TypeError
            If either argument is not a tensor or has an unsupported dtype.
        ValueError
            If graph inputs are invalid or any target component has no exact
            reference component match.
        """
        target_numbers, target_edges = _validated_connectivity_inputs(
            atomic_numbers, adjacency
        )
        from nvalchemi.csp._chem._topology import connected_components, find_isomorphism

        result = [-1] * len(target_numbers)
        for target_component in connected_components(target_edges):
            component_numbers = [target_numbers[index] for index in target_component]
            component_edges = [
                [target_edges[left][right] for right in target_component]
                for left in target_component
            ]
            for reference_component in self._components:
                reference_numbers = [
                    self._atomic_numbers[index] for index in reference_component
                ]
                reference_edges = [
                    [self._adjacency[left][right] for right in reference_component]
                    for left in reference_component
                ]
                mapping = find_isomorphism(
                    component_numbers,
                    component_edges,
                    reference_numbers,
                    reference_edges,
                )
                if mapping is not None:
                    for target_local, target_index in enumerate(target_component):
                        reference_index = reference_component[mapping[target_local]]
                        result[target_index] = self._atom_types[reference_index]
                    break
            else:
                raise ValueError(
                    "target graph contains a component with no isomorphic reference component"
                )
        return torch.tensor(result, dtype=torch.int32, device=atomic_numbers.device)


class RDKitConformerConfig(BaseModel):
    """Options for generating 3D conformers with RDKit distance geometry.

    ETKDGv3 is RDKit's distance-geometry method for proposing molecular
    conformations.

    Parameters
    ----------
    num_conformers : int, default=100
        Number of conformers to request.
    random_seed : int or None, default=None
        Nonnegative RDKit random seed. ``None`` leaves RDKit's random seed unset.
    prune_rms_thresh : float, default=-1.0
        Discard a newly generated conformer if its heavy-atom geometry is too
        similar to one already retained. The threshold is RMSD in Å; ``-1``
        disables this pruning.
    embed_force_tol : float, default=0.05
        ETKDG embedding cleanup tolerance passed to RDKit.
    max_embed_attempts : int, default=1000
        Maximum RDKit attempts per requested conformer.
    num_threads : int, default=-1
        Positive values request that many embedding threads. ``0`` asks RDKit
        to use the system's maximum supported thread count; ``-1`` is passed
        through to RDKit unchanged.

    Notes
    -----
    This immutable configuration covers embedding only. It does not request
    a separate MMFF or UFF geometry optimization.
    """

    model_config = ConfigDict(frozen=True, strict=True, extra="forbid")

    num_conformers: int = 100
    random_seed: int | None = None
    prune_rms_thresh: float = -1.0
    embed_force_tol: float = 0.05
    max_embed_attempts: int = 1000
    num_threads: int = -1

    @model_validator(mode="after")
    def _validate_values(self) -> RDKitConformerConfig:
        """Enforce positive generation limits and supported ETKDG option ranges."""
        if self.num_conformers <= 0:
            raise ValueError("num_conformers must be positive")
        if self.random_seed is not None and self.random_seed < 0:
            raise ValueError("random_seed must be nonnegative or None")
        if not math.isfinite(self.prune_rms_thresh) or self.prune_rms_thresh < -1.0:
            raise ValueError("prune_rms_thresh must be finite and at least -1")
        if not math.isfinite(self.embed_force_tol) or self.embed_force_tol <= 0.0:
            raise ValueError("embed_force_tol must be finite and positive")
        if self.max_embed_attempts <= 0:
            raise ValueError("max_embed_attempts must be positive")
        if self.num_threads < -1:
            raise ValueError("num_threads must be at least -1")
        return self


@OptionalDependency.RDKIT.require
def generate_conformers_from_smiles(
    smiles: str,
    *,
    config: RDKitConformerConfig | None = None,
) -> Mol:
    """Generate 3D conformers with explicit hydrogens from one SMILES molecule.

    Parameters
    ----------
    smiles : str
        SMILES for exactly one connected molecule.
    config : RDKitConformerConfig, optional
        RDKit distance-geometry options. Defaults to 100 conformers with unseeded
        sampling, no RMSD pruning, and up to 1000 embedding attempts.

    Returns
    -------
    rdkit.Chem.rdchem.Mol
        A new molecule with explicit hydrogens and all retained 3D conformers.

    Raises
    ------
    OptionalDependencyError
        If RDKit is unavailable. Install ``nvalchemi-toolkit[rdkit]``.
    TypeError
        If ``smiles`` is not a string or ``config`` has the wrong type.
    ValueError
        If the SMILES is invalid or describes multiple disconnected fragments.
    RuntimeError
        If RDKit retains no conformers.

    Notes
    -----
    No separate MMFF or UFF minimization is run. Exact reproducibility across
    RDKit versions, platforms, and thread settings is not guaranteed.

    Examples
    --------
    >>> molecule = generate_conformers_from_smiles(
    ...     "CO", config=RDKitConformerConfig(num_conformers=2, random_seed=7)
    ... )
    >>> molecule.GetNumConformers() > 0
    True
    """
    if config is not None and not isinstance(config, RDKitConformerConfig):
        raise TypeError("config must be an RDKitConformerConfig or None")
    from nvalchemi.csp._chem._conformers import (
        generate_conformers_from_smiles as _generate,
    )

    return _generate(smiles, config or RDKitConformerConfig())


@OptionalDependency.RDKIT.require
def build_contact_distance_matrix(
    molecules: Sequence[Mol],
    *,
    device: torch.device | str = "cpu",
) -> Tensor:
    """Build intermolecular contact cutoffs used to measure overlap.

    Parameters
    ----------
    molecules : sequence of rdkit.Chem.rdchem.Mol
        Nonempty connected components in formula-unit order. Atom order is
        preserved, and no conformers are needed.
    device : torch.device or str, default="cpu"
        Device for the returned matrix; contact rules are evaluated on CPU.

    Returns
    -------
    torch.Tensor, shape ``[A, A]``, dtype ``torch.float32``
        Owned symmetric contact cutoffs in Å. Entry ``[i, j]`` defines overlap
        between atoms ``i`` and ``j`` in different molecular copies; distances
        within one rigid molecule are not measured against this matrix.
        ``CrystalPacker`` may accept residual overlap up to
        ``PackingConfig.overlap_tolerance``.

    Raises
    ------
    OptionalDependencyError
        If RDKit is unavailable. Install ``nvalchemi-toolkit[rdkit]``.
    TypeError
        If ``molecules`` is not a sequence of RDKit molecules.
    ValueError
        If the sequence is empty, a molecule is disconnected, or an element
        has no positive RDKit fallback van der Waals radius.

    Notes
    -----
    Defaults begin with van der Waals radius sums and apply the strongest
    symmetric donor/acceptor reduction, capped at 0.90 Å. The rules include
    hydrogen, halogen, chalcogen, phosphorus, silicon, and boron cases. Known
    element radii are built in; using RDKit's periodic-table fallback emits a
    ``RuntimeWarning``. These experimental chemistry rules and values may
    change in a future release.

    Examples
    --------
    This methanol input has carbon and oxygen atoms; its implicit hydrogens
    have no separate rows in the matrix. The C–O entry applies to atoms in
    different copies, not to the C–O bond inside one methanol molecule.

    >>> from rdkit import Chem
    >>> molecule = Chem.MolFromSmiles("CO")
    >>> contacts = build_contact_distance_matrix([molecule])
    >>> tuple(contacts.shape)
    (2, 2)
    >>> round(float(contacts[0, 1]), 2)
    3.27
    """
    from nvalchemi.csp._chem._contacts import contact_distances

    return contact_distances(molecules).to(device=torch.device(device)).contiguous()


@OptionalDependency.RDKIT.require
def build_molecular_packing_input(
    molecules: Sequence[Mol],
    *,
    contact_distances: Tensor | None = None,
    formula_unit_volume: float | None = None,
    component_index: Tensor | None = None,
    metadata: Mapping[str, JSONValue] | None = None,
) -> MolecularPackingInput:
    """Prepare molecular conformers and contact limits required by the Packer.

    Parameters
    ----------
    molecules : sequence of rdkit.Chem.rdchem.Mol
        One connected formula-unit molecule per entry, in the order retained
        by the resulting input. Every attached conformer must be finite and
        marked 3D. Existing atom, hydrogen, and conformer order is preserved;
        implicit hydrogens are not added.
        :func:`generate_conformers_from_smiles` adds explicit hydrogens before
        embedding.
    contact_distances : torch.Tensor, shape ``[A, A]``, optional
        Custom atom-pair contact cutoffs in Å. Accepted structures may fall
        below these distances by the configured overlap tolerance. When
        omitted, experimental topology-derived contact rules are used.
        Supplied values are copied, converted to float32, checked for
        positivity and symmetry, and symmetrized within tolerance.
    formula_unit_volume : float, optional
        Positive formula-unit volume in Å³. When omitted, the Toolkit atomic
        volume estimate is used.
    component_index : torch.Tensor, int32 ``[F]``, optional
        Contiguous component IDs, one per molecule. Defaults to ``0..F-1``;
        repeated IDs can group molecules that belong to the same component.
    metadata : Mapping[str, JSONValue], optional
        JSON-compatible provenance retained by the formula input.

    Returns
    -------
    MolecularPackingInput
        Owned canonical CPU tensors. The formula-input constructor centers
        each conformer once by its unweighted Cartesian mean.

    Raises
    ------
    OptionalDependencyError
        If RDKit is unavailable. Install ``nvalchemi-toolkit[rdkit]``.
    TypeError
        If the molecule sequence, custom contact matrix, or component IDs have
        an invalid type.
    ValueError
        If a molecule lacks a conformer, coordinates are malformed, or any
        formula-input invariant is violated.

    Notes
    -----
    This helper does not mutate RDKit molecules. Callers that already have
    tensors or use another chemistry toolkit can construct
    :class:`~nvalchemi.csp.data.MolecularPackingInput` directly.

    Examples
    --------
    >>> molecule = generate_conformers_from_smiles(
    ...     "CO", config=RDKitConformerConfig(num_conformers=1, random_seed=7)
    ... )
    >>> packing_input = build_molecular_packing_input([molecule])
    >>> packing_input.num_molecules
    1
    """
    from nvalchemi.csp._chem._input import (
        build_molecular_packing_input as _build,
    )

    return _build(
        molecules,
        contact_distances=contact_distances,
        formula_unit_volume=formula_unit_volume,
        component_index=component_index,
        metadata=metadata,
    )
