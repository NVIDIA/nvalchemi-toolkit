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
"""Shared fixtures and builders for ``test/training/distillation/``.

Extends ``test/training/conftest.py`` — its builders are imported rather
than duplicated, and its autouse seeding fixture applies here too.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any, Literal

import pytest
import torch
from torch import nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes.in_memory_dataset import InMemoryDataset
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
from nvalchemi.models.lj import LennardJonesModelWrapper
from nvalchemi.training.distillation._attach import _attach_teacher_labels
from nvalchemi.training.distillation.scoring import TeacherScorer
from nvalchemi.training.distillation.seeding import FitPolicy
from test.training.conftest import _build_atomic_data, _build_batch, _build_demo_model

_LJ_CUTOFF = 5.0
"""Cutoff of the Lennard-Jones teacher shared by the distillation tests."""

_PAIR_CUTOFF = 4.5
"""Cutoff of the neighbor-list autograd teacher shared by the distillation tests."""

_WIRED_CHARGE = 7.0
"""Per-atom charge the charge-emitting stub teacher writes for every atom."""


_INITIAL_ELEMENT = 1
"""Atomic number tagging every structure an on-policy run generates from."""

_REFERENCE_ELEMENT = 6
"""Atomic number tagging every structure that comes from the reference dataset."""

_ATOMS_PER_SYSTEM = 4
"""Atoms in every synthetic on-policy system, so batches stay small and uniform."""


class _DirectForceModel(nn.Module):
    """Tiny MLP with independent per-atom energy and force heads."""

    def __init__(self, num_atom_types: int = 20, hidden_dim: int = 8) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.trunk = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.SiLU(),
        )
        self.energy_head = nn.Linear(hidden_dim, 1)
        self.force_head = nn.Linear(hidden_dim, 3)

    def features(
        self, atomic_numbers: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Return the per-atom hidden features both heads read."""
        return self.trunk(
            torch.cat([self.embedding(atomic_numbers), positions], dim=-1)
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return energies plus forces predicted directly, not as a gradient."""
        features = self.features(atomic_numbers, positions)
        atomic_energies = self.energy_head(features)
        forces = self.force_head(features)
        if batch_indices is not None:
            num_graphs = int(batch_indices.max().item()) + 1
            energy = torch.zeros(
                (num_graphs, 1),
                device=atomic_energies.device,
                dtype=atomic_energies.dtype,
            )
            energy.scatter_add_(0, batch_indices.unsqueeze(-1), atomic_energies)
        else:
            energy = atomic_energies.sum(dim=0, keepdim=True)
        return {
            "energy": energy,
            "forces": forces,
            "atomic_energies": atomic_energies.squeeze(-1),
        }


class _DirectForceTeacher(nn.Module, BaseModelMixin):
    """Direct-force demo teacher: forces are a head output, not an energy gradient."""

    def __init__(self, model: _DirectForceModel) -> None:
        super().__init__()
        self.model = model
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "atomic_energies"}),
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            neighbor_config=None,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the per-node embedding shape published by this teacher."""
        return {"node_embeddings": (self.model.hidden_dim,)}

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Collect the tensors the underlying model's forward expects, at its dtype.

        Positions are cast to the parameter dtype, which is what lets a
        reduced-precision copy of this model run over an ordinary float32 batch.
        """
        model_inputs = super().adapt_input(data, **kwargs)
        model_inputs["batch_indices"] = (
            data.batch_idx if isinstance(data, Batch) else None
        )
        model_inputs["positions"] = model_inputs["positions"].to(
            next(self.model.parameters()).dtype
        )
        return model_inputs

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Write per-node embeddings onto *data* in place."""
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])
        features = self.model.features(
            data.atomic_numbers,
            data.positions.to(next(self.model.parameters()).dtype),
        )
        atoms_group = data._atoms_group
        if atoms_group is not None:
            atoms_group["node_embeddings"] = features
        else:
            data.node_embeddings = features
        return data

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> OrderedDict:
        """Run the model and adapt its output to the framework format."""
        model_inputs = self.adapt_input(data, **kwargs)
        return self.adapt_output(self.model(**model_inputs), data)


class _PairPotentialModel(nn.Module):
    """Smooth pair potential over a dense neighbor list, with per-species weights."""

    def __init__(self, num_atom_types: int = 20) -> None:
        super().__init__()
        self.weights = nn.Embedding(num_atom_types, 1)

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        positions: torch.Tensor,
        neighbor_matrix: torch.Tensor,
        num_neighbors: torch.Tensor,
        batch_indices: torch.Tensor | None = None,
        compute_forces: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Return the pair energy and, when asked, its gradient-derived forces."""
        neighbors = neighbor_matrix.long().clamp(0, positions.shape[0] - 1)
        live = torch.arange(neighbor_matrix.shape[1]) < num_neighbors.unsqueeze(-1)
        vectors = positions[neighbors] - positions.unsqueeze(1)
        distances = (vectors.pow(2).sum(dim=-1) + 1e-12).sqrt()
        pair_energies = torch.exp(-distances) * live
        atomic_energies = (
            0.5 * self.weights(atomic_numbers) * pair_energies.sum(dim=-1, keepdim=True)
        )
        if batch_indices is not None:
            num_graphs = int(batch_indices.max().item()) + 1
            energy = torch.zeros(
                (num_graphs, 1),
                device=atomic_energies.device,
                dtype=atomic_energies.dtype,
            )
            energy.scatter_add_(0, batch_indices.unsqueeze(-1), atomic_energies)
        else:
            energy = atomic_energies.sum(dim=0, keepdim=True)
        outputs = {"energy": energy, "atomic_energies": atomic_energies.squeeze(-1)}
        if compute_forces:
            outputs["forces"] = -torch.autograd.grad(
                energy,
                inputs=[positions],
                grad_outputs=torch.ones_like(energy),
                create_graph=False,
            )[0]
        return outputs


class _PairPotentialTeacher(nn.Module, BaseModelMixin):
    """Teacher combining autograd forces with a dense neighbor list.

    The quadrant every production teacher occupies: the forward pass consumes a
    neighbor list the scorer has to build, and differentiates the energy through
    the neighbor-gathered edge vectors to get forces.
    """

    def __init__(
        self, model: _PairPotentialModel, cutoff: float = _PAIR_CUTOFF
    ) -> None:
        super().__init__()
        self.model = model
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "atomic_energies"}),
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            neighbor_config=NeighborConfig(
                cutoff=cutoff, format=NeighborListFormat.MATRIX
            ),
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no embedding shapes."""
        return {}

    def compute_embeddings(self, data: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        """Raise, since this teacher produces no embeddings."""
        raise NotImplementedError

    def adapt_input(self, data: AtomicData | Batch, **kwargs: Any) -> dict[str, Any]:
        """Collect the tensors the underlying model's forward expects."""
        model_inputs = super().adapt_input(data, **kwargs)
        model_inputs["batch_indices"] = (
            data.batch_idx if isinstance(data, Batch) else None
        )
        model_inputs["compute_forces"] = "forces" in self.model_config.active_outputs
        return model_inputs

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> OrderedDict:
        """Run the model and adapt its output to the framework format."""
        model_inputs = self.adapt_input(data, **kwargs)
        return self.adapt_output(self.model(**model_inputs), data)


class _ChargeSourceModel(nn.Module, BaseModelMixin):
    """Teacher emitting per-atom charges alongside a flat energy.

    Stands in for a pipeline stage wiring charges into the next one, and for a
    teacher with an output the built-in signal table does not cover.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "charges"}),
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            neighbor_config=None,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no embedding shapes."""
        return {}

    def compute_embeddings(self, data: Any, **kwargs: Any) -> Any:  # noqa: ARG002
        """Raise, since this stage produces no embeddings."""
        raise NotImplementedError

    def forward(self, data: Batch, **kwargs: Any) -> OrderedDict:  # noqa: ARG002
        """Return a zero energy and the charges the next stage consumes."""
        return OrderedDict(
            [
                ("energy", torch.zeros(data.num_graphs, 1)),
                ("charges", torch.full((data.num_nodes,), _WIRED_CHARGE)),
            ]
        )


class _ListSource:
    """Minimal ``InitialStructuresSource`` over a fixed list of structures.

    Serves the whole list as the initial batch, then the remainder through
    ``draw``, and records every shard installed on it.
    """

    def __init__(self, structures: list[AtomicData]) -> None:
        self.structures = structures
        self.shards: list[tuple[int, int]] = []
        self._cursor = 0

    @property
    def exhausted(self) -> bool:
        """Whether every structure has been handed out."""
        return self._cursor >= len(self.structures)

    def shard(self, rank: int, world_size: int) -> None:
        """Record the shard and reopen the cursor."""
        self.shards.append((rank, world_size))
        self._cursor = 0

    def probe(self) -> Batch:
        """Return the first structure as a one-graph batch."""
        return Batch.from_data_list([self.structures[0]])

    def initial_batch(self) -> Batch:
        """Return every structure left as one batch, stamped with clean bookkeeping."""
        batch = Batch.from_data_list(self.structures[self._cursor :])
        batch["status"] = torch.zeros(batch.num_graphs, 1, dtype=torch.long)
        batch["system_id"] = torch.arange(
            self._cursor, self._cursor + batch.num_graphs, dtype=torch.long
        ).unsqueeze(-1)
        self._cursor = len(self.structures)
        return batch

    def draw(
        self,
        *,
        limit: int | None = None,
        fits: FitPolicy | None = None,  # noqa: ARG002
        on_miss: Literal["stop", "skip"] = "stop",  # noqa: ARG002
    ) -> list[AtomicData]:
        """Serve up to *limit* structures, each stamped with its ``system_id``."""
        end = None if limit is None else self._cursor + limit
        served = self.structures[self._cursor : end]
        for offset, data in enumerate(served):
            data.add_system_property(
                "system_id", torch.tensor([[self._cursor + offset]], dtype=torch.long)
            )
        self._cursor += len(served)
        return served

    def state_dict(self) -> dict[str, Any]:
        """Return the cursor."""
        return {"cursor": self._cursor}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Resume at the recorded cursor."""
        self._cursor = int(state["cursor"])


def _build_pair_potential_teacher(
    num_atom_types: int = 20, cutoff: float = _PAIR_CUTOFF, seed: int = 0
) -> _PairPotentialTeacher:
    torch.manual_seed(seed)
    return _PairPotentialTeacher(_PairPotentialModel(num_atom_types), cutoff=cutoff)


def _build_direct_force_model(
    num_atom_types: int = 20, hidden_dim: int = 8, seed: int = 0
) -> _DirectForceModel:
    torch.manual_seed(seed)
    return _DirectForceModel(num_atom_types=num_atom_types, hidden_dim=hidden_dim)


def _build_direct_force_teacher(
    num_atom_types: int = 20, hidden_dim: int = 8, seed: int = 0
) -> _DirectForceTeacher:
    return _DirectForceTeacher(
        _build_direct_force_model(
            num_atom_types=num_atom_types, hidden_dim=hidden_dim, seed=seed
        )
    )


def _build_small_dataset(n_systems: int = 5, base_seed: int = 200) -> InMemoryDataset:
    data_list = [
        _build_atomic_data(n_atoms=2 + index, seed=base_seed + index)
        for index in range(n_systems)
    ]
    return InMemoryDataset(in_memory_batch=Batch.from_data_list(data_list))


def _build_atom_only_dataset(
    n_systems: int = 3, base_seed: int = 400
) -> InMemoryDataset:
    data_list = []
    for index in range(n_systems):
        generator = torch.Generator().manual_seed(base_seed + index)
        n_atoms = 2 + index
        data_list.append(
            AtomicData(
                positions=torch.randn(n_atoms, 3, generator=generator),
                atomic_numbers=torch.randint(
                    1, 10, (n_atoms,), dtype=torch.long, generator=generator
                ),
            )
        )
    return InMemoryDataset(in_memory_batch=Batch.from_data_list(data_list))


def _build_lj_teacher(
    cutoff: float = _LJ_CUTOFF, half_list: bool = False
) -> LennardJonesModelWrapper:
    return LennardJonesModelWrapper(
        epsilon=0.01, sigma=3.4, cutoff=cutoff, half_list=half_list
    )


def _build_periodic_atomic_data(
    n_atoms: int = 6, seed: int = 0, cell_length: float = 8.0
) -> AtomicData:
    generator = torch.Generator().manual_seed(seed)
    return AtomicData(
        positions=torch.rand(n_atoms, 3, generator=generator) * cell_length,
        atomic_numbers=torch.ones(n_atoms, dtype=torch.long),
        atomic_masses=torch.ones(n_atoms),
        cell=torch.eye(3).unsqueeze(0) * cell_length,
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )


def _build_periodic_batch(n_systems: int = 2, n_atoms: int = 6) -> Batch:
    return Batch.from_data_list(
        [_build_periodic_atomic_data(n_atoms, seed=index) for index in range(n_systems)]
    )


def _build_periodic_dataset(
    n_systems: int = 4, base_seed: int = 300
) -> InMemoryDataset:
    data_list = [
        _build_periodic_atomic_data(n_atoms=4 + index, seed=base_seed + index)
        for index in range(n_systems)
    ]
    return InMemoryDataset(in_memory_batch=Batch.from_data_list(data_list))


def _build_propagator_system(
    atomic_number: int, seed: int, *, predictions: bool = True
) -> AtomicData:
    """Return one system tagged by *atomic_number*, carrying the propagator's keys.

    ``predictions=False`` leaves out the ``energy`` and ``forces`` a propagator
    writes and the labeling hook strips again, which is the shape a replay frame
    — and therefore the mixture's reference dataset — has.
    """
    generator = torch.Generator().manual_seed(seed)
    predicted = (
        {"energy": torch.zeros(1, 1), "forces": torch.zeros(_ATOMS_PER_SYSTEM, 3)}
        if predictions
        else {}
    )
    return AtomicData(
        positions=torch.randn(_ATOMS_PER_SYSTEM, 3, generator=generator),
        atomic_numbers=torch.full(
            (_ATOMS_PER_SYSTEM,), atomic_number, dtype=torch.long
        ),
        atomic_masses=torch.ones(_ATOMS_PER_SYSTEM),
        **predicted,
    )


def _build_propagator_batch(
    atomic_number: int, n_systems: int, base_seed: int, *, predictions: bool = True
) -> Batch:
    """Return a batch of *n_systems* systems all tagged by *atomic_number*."""
    return Batch.from_data_list(
        [
            _build_propagator_system(
                atomic_number, base_seed + index, predictions=predictions
            )
            for index in range(n_systems)
        ]
    )


def _build_initial_dataset(n_systems: int = 4, base_seed: int = 500) -> InMemoryDataset:
    """Return the structures the generated trajectories start from."""
    return InMemoryDataset(
        in_memory_batch=_build_propagator_batch(_INITIAL_ELEMENT, n_systems, base_seed)
    )


def _build_reference_dataset(
    scorer: TeacherScorer, n_systems: int = 8, base_seed: int = 700
) -> InMemoryDataset:
    """Return a teacher-labeled reference dataset with the generated frames' schema."""
    frames = _build_propagator_batch(
        _REFERENCE_ELEMENT, n_systems, base_seed, predictions=False
    )
    _attach_teacher_labels(frames, scorer.label(frames))
    return InMemoryDataset(in_memory_batch=frames)


@pytest.fixture
def demo_teacher() -> Any:
    """Return a freshly-seeded autograd-force :class:`DemoModelWrapper` teacher."""
    return _build_demo_model()


@pytest.fixture
def direct_force_teacher() -> _DirectForceTeacher:
    """Return a freshly-seeded direct-force demo teacher."""
    return _build_direct_force_teacher()


@pytest.fixture
def pair_potential_teacher() -> _PairPotentialTeacher:
    """Return a teacher with autograd forces and a dense neighbor list."""
    return _build_pair_potential_teacher()


@pytest.fixture
def small_batch() -> Batch:
    """Return a default :class:`Batch` — 2 systems, 3 atoms each, ``seed=0``."""
    return _build_batch()


@pytest.fixture
def small_dataset() -> InMemoryDataset:
    """Return an :class:`InMemoryDataset` of 5 systems with 2-6 atoms each."""
    return _build_small_dataset()


@pytest.fixture
def atom_only_dataset() -> InMemoryDataset:
    """Return a dataset of 3 systems carrying no system-level field at all."""
    return _build_atom_only_dataset()


@pytest.fixture
def lj_teacher() -> LennardJonesModelWrapper:
    """Return a Lennard-Jones teacher requiring a dense neighbor list."""
    return _build_lj_teacher()


@pytest.fixture
def periodic_batch() -> Batch:
    """Return a periodic :class:`Batch` — 2 systems, 6 atoms each, 8 A cell."""
    return _build_periodic_batch()


@pytest.fixture
def periodic_dataset() -> InMemoryDataset:
    """Return an :class:`InMemoryDataset` of 4 periodic systems with 4-7 atoms each."""
    return _build_periodic_dataset()
