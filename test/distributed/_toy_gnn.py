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
"""Helpers for distributed GNN inference tests.

Provides a minimal message-passing GNN that exercises the exact PyTorch op
surface MACE uses for its interaction blocks — integer indexing (``aten::index``)
for the gather and ``scatter_add_`` for the aggregation — plus utilities for
building a small Argon system and checking forces against finite differences.
"""

from __future__ import annotations

import math
from typing import Callable

import torch
from torch import Tensor, nn


class ToyGNN(nn.Module):
    """Minimal message-passing GNN mirroring MACE's gather/scatter pattern.

    The forward touches exactly the ops with ShardTensor dispatch handlers:
    ``aten::index.Tensor`` (gather), ``aten::zeros_like`` (scatter output
    buffer), ``aten::scatter_add_`` (aggregation), and a final ``.sum()`` for
    total energy.
    """

    def __init__(
        self,
        num_species: int = 20,
        hidden: int = 32,
        num_layers: int = 3,
        r_cut: float = 5.0,
    ) -> None:
        super().__init__()
        self.r_cut = r_cut
        self.embed = nn.Embedding(num_species, hidden)
        self.msg_mlp = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden + 1, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                )
                for _ in range(num_layers)
            ]
        )
        self.update = nn.ModuleList(
            [nn.Linear(hidden, hidden) for _ in range(num_layers)]
        )
        self.readout = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        positions: Tensor,
        atomic_numbers: Tensor,
        edge_index: Tensor,
        edge_vec: Tensor,
    ) -> Tensor:
        edge_len = edge_vec.norm(dim=-1, keepdim=True)
        cutoff_fn = 0.5 * (torch.cos(math.pi * edge_len / self.r_cut) + 1.0)
        cutoff_fn = cutoff_fn * (edge_len < self.r_cut).to(cutoff_fn.dtype)

        x = self.embed(atomic_numbers)
        n, f = x.shape
        dst = edge_index[1].unsqueeze(-1).expand(-1, f)
        for msg_mlp, upd in zip(self.msg_mlp, self.update, strict=True):
            src = x[edge_index[0]]
            msg = msg_mlp(torch.cat([src, edge_len], dim=-1)) * cutoff_fn
            agg = torch.zeros_like(x).scatter_add_(0, dst, msg)
            x = x + upd(agg)
        return self.readout(x).squeeze(-1)


def build_fcc_argon(
    n_per_side: int = 2,
    lattice_const: float = 5.26,
    dtype: torch.dtype = torch.float64,
    device: str | torch.device = "cpu",
) -> tuple[Tensor, Tensor, Tensor]:
    """Build an FCC Argon supercell with 4 * n_per_side**3 atoms.

    Returns ``(positions, cell, atomic_numbers)``. Cell rows are lattice vectors.
    """
    basis = torch.tensor(
        [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
        dtype=dtype,
        device=device,
    )
    grid = torch.arange(n_per_side, dtype=dtype, device=device)
    gx, gy, gz = torch.meshgrid(grid, grid, grid, indexing="ij")
    cell_origin = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    frac = (cell_origin[:, None, :] + basis[None, :, :]).reshape(-1, 3)
    positions = frac * lattice_const
    side = n_per_side * lattice_const
    cell = torch.eye(3, dtype=dtype, device=device) * side
    atomic_numbers = torch.full(
        (positions.shape[0],), 18, dtype=torch.long, device=device
    )
    return positions, cell, atomic_numbers


def brute_force_edges(
    positions: Tensor,
    cutoff: float,
    cell: Tensor,
    pbc: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Build a COO edge list with PBC minimum-image convention.

    Edge selection (the mask) is treated as constant w.r.t. positions so
    ``edge_vec`` is a smooth differentiable function of ``positions`` — this is
    required for finite-difference and autograd forces to agree.
    """
    if pbc is None:
        pbc = torch.ones(3, dtype=torch.bool, device=positions.device)

    with torch.no_grad():
        dr_raw = positions[None, :, :] - positions[:, None, :]
        inv_cell = torch.linalg.inv(cell)
        frac = dr_raw @ inv_cell
        shift_int = -torch.round(frac) * pbc.to(frac.dtype)
        dr = dr_raw + shift_int @ cell
        dist = dr.norm(dim=-1)
        mask = (dist < cutoff) & (dist > 1e-8)

    idx = mask.nonzero(as_tuple=False)
    src, dst = idx[:, 0], idx[:, 1]
    shift = shift_int[src, dst]
    edge_vec = positions[dst] - positions[src] + shift @ cell
    edge_index = torch.stack([src, dst], dim=0)
    return edge_index, edge_vec


def finite_difference_forces(
    energy_fn: Callable[[Tensor], Tensor],
    positions: Tensor,
    h: float = 1e-4,
) -> Tensor:
    """Central-difference forces F_i = -dE/dr_i. O(6N) energy evaluations."""
    forces = torch.zeros_like(positions)
    for i in range(positions.shape[0]):
        for j in range(3):
            pos_p = positions.clone()
            pos_p[i, j] += h
            pos_m = positions.clone()
            pos_m[i, j] -= h
            e_p = energy_fn(pos_p)
            e_m = energy_fn(pos_m)
            forces[i, j] = -(e_p - e_m) / (2.0 * h)
    return forces
