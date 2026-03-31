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

import torch
from torch import nn

from nvalchemi.data import Batch
from nvalchemi.models.base import ForwardContext, Potential
from nvalchemi.models.contracts import PotentialCard
from nvalchemi.models.metadata import ModelCard

__all__ = ["DemoPotential"]


DemoPotentialCard = PotentialCard(
    required_inputs=frozenset({"atomic_numbers", "positions"}),
    optional_inputs=frozenset({"batch"}),
    result_keys=frozenset({"energies", "forces"}),
    default_result_keys=frozenset({"energies", "forces"}),
    additive_result_keys=frozenset({"energies", "forces"}),
    boundary_modes=frozenset({"non_pbc"}),
)


class DemoPotential(Potential):
    """Small direct-output potential used by tests and dynamics demos.

    The implementation is intentionally simple and deterministic enough
    for testing. It consumes atomic numbers and positions directly and
    returns per-system energies together with direct forces derived from
    the scalar energy.
    """

    card = DemoPotentialCard

    def __init__(
        self,
        *,
        num_atom_types: int = 100,
        hidden_dim: int = 64,
        name: str | None = None,
    ) -> None:
        """Initialise the demo potential.

        Parameters
        ----------
        num_atom_types
            Maximum atomic number index handled by the embedding table.
        hidden_dim
            Width of the internal feature representation.
        name
            Optional human-readable step name.
        """

        super().__init__(name=name)
        self.num_atom_types = num_atom_types
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            self.num_atom_types,
            self.hidden_dim,
            padding_idx=0,
            max_norm=1.0,
        )
        self.coord_embedding = nn.Sequential(
            nn.Linear(3, self.hidden_dim, bias=False),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
        )
        self.joint_mlp = nn.Sequential(
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.projection = nn.Linear(self.hidden_dim, 1)

        self.model_card = ModelCard(
            model_family="demo",
            model_name=self.step_name,
        )

    @property
    def dtype(self) -> torch.dtype:
        """Return the parameter dtype used for coordinate processing."""

        return self.projection.weight.dtype

    def compute(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> dict[str, torch.Tensor]:
        """Compute demo energies and direct forces for one batch."""

        atomic_numbers = self.require_input(batch, "atomic_numbers", ctx)
        positions = self.require_input(batch, "positions", ctx)
        batch_indices = self.optional_input(batch, "batch", ctx)

        positions_for_energy = positions.to(self.dtype)
        if "forces" in ctx.outputs:
            positions_for_energy = positions_for_energy.detach().requires_grad_(True)

        atom_z = self.embedding(atomic_numbers)
        coord_z = self.coord_embedding(positions_for_energy)
        embedding = self.joint_mlp(torch.cat([atom_z, coord_z], dim=-1))
        embedding = embedding + atom_z + coord_z
        node_energy = self.projection(embedding)

        if batch_indices is not None:
            num_graphs = batch.num_graphs
            energies = torch.zeros(
                (num_graphs, 1),
                device=node_energy.device,
                dtype=node_energy.dtype,
            )
            energies.scatter_add_(0, batch_indices.unsqueeze(-1), node_energy)
        else:
            energies = node_energy.sum(dim=0, keepdim=True)

        forces = None
        if "forces" in ctx.outputs:
            forces = -torch.autograd.grad(
                energies,
                inputs=[positions_for_energy],
                grad_outputs=torch.ones_like(energies),
                create_graph=False,
                retain_graph=False,
            )[0]

        return self.build_results(ctx, energies=energies, forces=forces)
