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
"""Small demo model used by tests and examples."""

from __future__ import annotations

import torch
from torch import nn

from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.models.utils import (
    build_model_repr,
    collect_nondefault_repr_kwargs,
    initialize_model_repr,
    mapping_get,
)

__all__ = ["DemoModelWrapper"]


_DemoModelConfig = ModelConfig(
    required_inputs=frozenset({"atomic_numbers", "positions"}),
    optional_inputs=frozenset({"batch"}),
    outputs=frozenset({"energies", "forces"}),
    additive_outputs=frozenset({"energies", "forces"}),
    use_autograd=False,
    pbc_mode="non-pbc",
)


class DemoModelWrapper(nn.Module, BaseModelMixin):
    """Small deterministic demo model for tests and debugging.

    This model computes per-system energies and conservative forces directly
    from atomic numbers and Cartesian coordinates. It is not intended for
    production use; it exists as a compact example of the wrapper interface
    and as a stable test fixture for model and dynamics code paths.

    Parameters
    ----------
    num_atom_types : int, optional
        Size of the atomic-number embedding table.
    hidden_dim : int, optional
        Width of the internal embedding and projection layers.
    name : str or None, optional
        Optional stable display name used by the composable runtime.
    """

    spec = _DemoModelConfig

    def __init__(
        self,
        *,
        num_atom_types: int = 100,
        hidden_dim: int = 64,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.num_atom_types = num_atom_types
        self.hidden_dim = hidden_dim
        self._name = name

        self.embedding = nn.Embedding(
            num_atom_types,
            hidden_dim,
            padding_idx=0,
            max_norm=1.0,
        )
        self.coord_embedding = nn.Sequential(
            nn.Linear(3, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.joint_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.projection = nn.Linear(hidden_dim, 1)

        initialize_model_repr(
            self,
            static_kwargs=collect_nondefault_repr_kwargs(
                explicit_values={
                    "num_atom_types": num_atom_types,
                    "hidden_dim": hidden_dim,
                    "name": name,
                },
                defaults={
                    "num_atom_types": 100,
                    "hidden_dim": 64,
                    "name": None,
                },
                order=("num_atom_types", "hidden_dim", "name"),
            ),
            kwarg_order=("num_atom_types", "hidden_dim", "name"),
        )

    def __repr__(self) -> str:
        """Return a compact constructor-style representation."""

        return build_model_repr(self)

    @property
    def dtype(self) -> torch.dtype:
        """Return the internal compute dtype."""

        return self.projection.weight.dtype

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Compute demo energies and conservative forces.

        Parameters
        ----------
        data
            Prepared input mapping containing ``atomic_numbers``,
            ``positions``, and optionally ``batch``.

        Returns
        -------
        dict[str, torch.Tensor]
            Mapping with ``energies`` and ``forces`` tensors.
        """

        atomic_numbers = data["atomic_numbers"]
        positions = data["positions"]
        batch_indices = mapping_get(data, "batch")

        positions_for_energy = positions.to(self.dtype)
        if not positions_for_energy.requires_grad:
            positions_for_energy = positions_for_energy.detach().requires_grad_(True)

        atom_z = self.embedding(atomic_numbers)
        coord_z = self.coord_embedding(positions_for_energy)
        embedding = self.joint_mlp(torch.cat([atom_z, coord_z], dim=-1))
        embedding = embedding + atom_z + coord_z
        node_energy = self.projection(embedding)

        if batch_indices is None:
            energies = node_energy.sum(dim=0, keepdim=True)
        else:
            num_graphs = (
                int(batch_indices.max().item()) + 1 if batch_indices.numel() else 1
            )
            energies = torch.zeros(
                (num_graphs, 1),
                device=node_energy.device,
                dtype=node_energy.dtype,
            )
            energies.scatter_add_(0, batch_indices.long().unsqueeze(-1), node_energy)

        forces = -torch.autograd.grad(
            energies.sum(),
            positions_for_energy,
            create_graph=self.training,
            retain_graph=self.training,
        )[0]

        return {"energies": energies, "forces": forces.to(dtype=positions.dtype)}
