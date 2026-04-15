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
"""
Type definitions for molecular and crystal graph representations.

For tensors, we use `jaxtyping` to annotate the types,
particularly to note their data type and shapes.

The notation we will use for shapes is as follows:
- B: Batch size
- V: Number of nodes (vertices)
- E: Number of edges
- H: Hidden feature dimensionality
- A: Number of attributes
- C: Number of centroids
- M: Number of ensemble members
- K: Number of max neighbors
- 3: Number of dimensions for coordinates

The notation for data types is as follows:
- Dimensionalities are assumed to be batch-able; i.e. there are redundant
dimensions for things like charges, spins, and energy. For concatenated
properties like atomic number and masses, they do not have a redundant dimension.
- masses refers to the atomic masses, assumed in amu.
- Coordinates can refer to fractional or Cartesian coordinates.
- Charges can refer to partial or total charges, and can be graph and node level.
"""

from __future__ import annotations

from collections import OrderedDict
from enum import Enum
from typing import TYPE_CHECKING, Protocol, TypeAlias, TypeVar

import torch
from jaxtyping import Bool, Float, Integer, Num

if TYPE_CHECKING:
    from nvalchemi.data import AtomicData, Batch

B: TypeAlias = int
V: TypeAlias = int
E: TypeAlias = int
H: TypeAlias = int
C: TypeAlias = int
M: TypeAlias = int
K: TypeAlias = int  # max neighbors
# used for defining generic protocols
T = TypeVar("T")

# the noqa is a known bug with ruff
Scalar: TypeAlias = Float[torch.Tensor, ""]  # noqa: F722
AtomicNumbers: TypeAlias = Integer[torch.Tensor, "V"]  # noqa: F722
AtomicMasses: TypeAlias = Float[torch.Tensor, "V"]  # noqa: F722
NodePositions: TypeAlias = Float[torch.Tensor, "V 3"]  # noqa: F722
NodeVelocities: TypeAlias = Float[torch.Tensor, "V 3"]  # noqa: F722
NodeMomentum: TypeAlias = Float[torch.Tensor, "V 3"]  # noqa: F722
NodeEmbeddings: TypeAlias = Float[torch.Tensor, "V H"]  # noqa: F722
EdgeEmbeddings: TypeAlias = Float[torch.Tensor, "E H"]  # noqa: F722
GraphEmbeddings: TypeAlias = Float[torch.Tensor, "B H"]  # noqa: F722
NodeAttributes: TypeAlias = Float[torch.Tensor, "V A"]  # noqa: F722
NodeCharges: TypeAlias = Float[torch.Tensor, "V"]  # noqa: F722
GraphCharges: TypeAlias = Float[torch.Tensor, "B 1"]  # noqa: F722
AtomCategories: TypeAlias = Integer[torch.Tensor, "V"]  # noqa: F722
NodeSpins: TypeAlias = Float[torch.Tensor, "V 1"]  # noqa: F722
GraphSpins: TypeAlias = Float[torch.Tensor, "B 1"]  # noqa: F722
# Cartesian displacement vectors: shifts = neighbor_list_shifts @ cell
PeriodicShifts: TypeAlias = Float[torch.Tensor, "E 3"]  # noqa: F722
# Integer lattice image indices; stored as int or float depending on source
NeighborListShifts: TypeAlias = Num[torch.Tensor, "E 3"]  # noqa: F722
LatticeVectors: TypeAlias = Float[torch.Tensor, "B 3 3"]  # noqa: F722
Periodicity: TypeAlias = Bool[torch.Tensor, "B 3"]  # noqa: F722
Forces: TypeAlias = Float[torch.Tensor, "V 3"]  # noqa: F722
Hessian: TypeAlias = Float[torch.Tensor, "V 3 3"]  # noqa: F722
Energy: TypeAlias = Float[torch.Tensor, "B 1"]  # noqa: F722
Stress: TypeAlias = Float[torch.Tensor, "B 3 3"]  # noqa: F722
Virials: TypeAlias = Float[torch.Tensor, "B 3 3"]  # noqa: F722
Dipole: TypeAlias = Float[torch.Tensor, "B 3"]  # noqa: F722
NeighborList: TypeAlias = Integer[torch.Tensor, "E 2"]  # noqa: F722
BatchIndices: TypeAlias = Integer[torch.Tensor, "V"]  # noqa: F722
NumSteps: TypeAlias = Integer[torch.Tensor, "B 1"]  # noqa: F722
Status: TypeAlias = Integer[torch.Tensor, "B 1"]  # noqa: F722
Fmax: TypeAlias = Float[torch.Tensor, "B 1"]  # noqa: F722
ModelOutputs: TypeAlias = OrderedDict[
    str, Energy | Forces | Hessian | Stress | Virials | Dipole | None
]  # noqa: F722
SampleScores: TypeAlias = Float[torch.Tensor, "B 1"]  # noqa: F722
Centroids: TypeAlias = Float[torch.Tensor, "C H"]  # noqa: F722
NodeKineticEnergies: TypeAlias = Float[torch.Tensor, "V 1"]  # noqa: F722
NodeTemperatures: TypeAlias = Float[torch.Tensor, "V 1"]  # noqa: F722
GraphTemperatures: TypeAlias = Float[torch.Tensor, "B 1"]  # noqa: F722
NeighborMatrix: TypeAlias = Integer[torch.Tensor, "V K"]  # noqa: F722
# Integer lattice image indices; stored as int or float depending on source
NeighborMatrixShifts: TypeAlias = Num[torch.Tensor, "V K 3"]  # noqa: F722
NumNeighbors: TypeAlias = Integer[torch.Tensor, "V"]  # noqa: F722

# ensemble variations of above properties
EnsembleEnergies: TypeAlias = Float[torch.Tensor, "M B 1"]  # noqa: F722
EnsembleForces: TypeAlias = Float[torch.Tensor, "M V 3"]  # noqa: F722
EnsembleHessians: TypeAlias = Float[torch.Tensor, "M V 3 3"]  # noqa: F722
EnsembleStresses: TypeAlias = Float[torch.Tensor, "M B 3 3"]  # noqa: F722
EnsembleVirials: TypeAlias = Float[torch.Tensor, "M B 3 3"]  # noqa: F722
EnsembleDipoles: TypeAlias = Float[torch.Tensor, "M B 3"]  # noqa: F722


class AtomCategory(Enum):
    """
    Categorical mapping for atom classifications within a system.

    This can be used to distinguish between different atoms during
    modeling, such as applying different kinds of constraints to
    them during training or inference (e.g. freezing dynamics for
    surface and bulk atoms).

    This Enum should encompass as many modeling types as possible,
    and is not limited to condensed phase modeling.

    The categories are as follows:
    - GAS: Gas phase atoms.
    - LIQUID: Liquid phase, or solvent atoms.
    - GL_INTERFACE: Gas-liquid interface atoms.
    - SURFACE: Surface atoms.
    - GS_INTERFACE: Gas-surface interface atoms.
    - BULK: Bulk atoms; typically those that can be assumed to be non-interacting.
    - SB_INTERFACE: Solid-bulk interface atoms.
    - FRAGMENT: Fragment atoms
    - CLUSTER: Atoms consituting clusters.
    - TERMINAL: Terminal atoms in molecules.
    - CENTRAL: Central atoms in molecules.
    - SPECIAL: Atoms designated to be generically special.

    While the categories are meant to be mapped to their respective
    chemistries, it would also be valid to just treat the Enum as
    distinct types without the mapping (e.g. 0/1 to differentiate
    between arbitrary types). In the binary case, where you just have
    two atom categories, we recommend using 0/-1, with `SPECIAL` atoms
    being used for your particular operation.
    """

    GAS = 0
    LIQUID = 1
    GL_INTERFACE = 2
    SURFACE = 3
    GS_INTERFACE = 4
    BULK = 5
    SB_INTERFACE = 6
    FRAGMENT = 7
    CLUSTER = 8
    TERMINAL = 9
    CENTRAL = 10
    SPECIAL = -1


class AbstractQueue(Protocol[T]):
    """
    Represents a generic queue interface; the requirements
    are that the queue can be used to put and get items of type ``T``.

    We do not require that the queue is thread-safe or process-safe,
    nor the ordering of the items within the queue.
    """

    def put(self, item: T) -> None:
        """
        Add an item to the queue.

        The item can be placed anywhere in the queue by
        the concrete implementation.

        Parameters
        ----------
        item: T
            The item to add to the queue.
        """
        ...

    def get(self) -> T:
        """
        Remove an item from the queue.

        Returns
        -------
        item: T
            The item removed from the queue.
        """
        ...


class EmbeddingModel(Protocol):
    """
    A protocol that defines an abstract interface for retrieving
    graph level embeddings from a model, given some data samples.
    """

    def compute_embeddings(self, samples: AtomicData | Batch) -> GraphEmbeddings:
        """
        Interface that will compute embeddings for a single or batch of samples.

        Parameters
        ----------
        samples: AtomicData | Batch
            The samples to compute the embeddings for.

        Returns
        -------
        graph_embeddings: GraphEmbeddings
            The graph embeddings for the samples.
        """
        ...


class AtomsLike(Protocol):
    """
    Represents the minimum viable data structure that is agnostic to
    batch and unbatched atomic data.

    This is only intended for use when type-hinting, and when the
    concrete cases can be used (e.g. ``AtomicData`` or ``Batch``),
    those should be used instead of this.

    Attributes
    ----------
    atomic_numbers : AtomicNumbers
        1D tensor containing atomic numbers.
    positions : NodePositions
        2D tensor containing atomic positions.
    cell : LatticeVectors
        3D tensor containing lattice parameters for each
        structure within a batch.
    """

    atomic_numbers: AtomicNumbers
    positions: NodePositions
    cell: LatticeVectors | None
    energy: Energy | None
    forces: Forces | None
