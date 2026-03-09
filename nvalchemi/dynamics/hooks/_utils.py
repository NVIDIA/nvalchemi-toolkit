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
"""Private helper functions shared across hook implementations.

These utilities provide common numerical computations used by multiple
hooks.  Factoring them out avoids code duplication and ensures numerical
consistency between hooks that observe vs. hooks that modify the same
quantities.

This module is **not** part of the public API.
"""

from __future__ import annotations

from typing import Literal

import torch
from jaxtyping import Float

# Boltzmann constant in eV/K (NIST 2018 CODATA value).
KB_EV: float = 8.617333262e-5

# Supported scatter-reduce operations.
ScatterReduce = Literal["amax", "sum", "amin", "mean"]


def scatter_reduce_per_graph(
    values: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
    reduce: ScatterReduce = "amax",
) -> torch.Tensor:
    """Scatter-reduce a 1-D node-level tensor to graph level.

    This is the generic building block for all per-graph reductions.
    Callers are responsible for preparing the 1-D ``values`` tensor
    (e.g. computing norms, kinetic energies, etc.) before calling
    this function.

    Parameters
    ----------
    values : Tensor
        1-D tensor of shape ``(V,)`` with one scalar per node.
    batch_idx : Tensor
        Integer tensor of shape ``(V,)`` mapping each node to its
        graph index.
    num_graphs : int
        Number of graphs in the batch.
    reduce : {"amax", "sum", "amin", "mean"}
        Scatter-reduce operation. Default ``"amax"``.

    Returns
    -------
    Tensor
        1-D tensor of shape ``(B,)`` with per-graph reduced values.
    """
    if reduce == "sum":
        out = torch.zeros(num_graphs, device=values.device, dtype=values.dtype)
        out.scatter_add_(0, batch_idx, values)
        return out

    # For amax, amin, mean — use scatter_reduce_
    fill = {
        "amax": float("-inf"),
        "amin": float("inf"),
        "mean": 0.0,
    }[reduce]
    out = torch.full((num_graphs,), fill, device=values.device, dtype=values.dtype)
    out.scatter_reduce_(0, batch_idx, values, reduce=reduce, include_self=False)
    return out


def kinetic_energy_per_graph(
    velocities: Float[torch.Tensor, "V 3"],
    masses: Float[torch.Tensor, "V ..."],
    batch_idx: torch.Tensor,
    num_graphs: int,
) -> Float[torch.Tensor, "B 1"]:
    """Compute ``0.5 * sum(m_i * ||v_i||^2)`` per graph.

    Parameters
    ----------
    velocities : Float[Tensor, "V 3"]
        Per-atom velocity vectors.
    masses : Float[Tensor, "V ..."]
        Per-atom masses.  May be shape ``(V,)`` or ``(V, 1)``.
    batch_idx : Tensor
        Per-atom graph membership indices of shape ``(V,)``.
    num_graphs : int
        Number of graphs in the batch.

    Returns
    -------
    Float[Tensor, "B 1"]
        Kinetic energy per graph.
    """
    # Ensure masses is (V,) for element-wise multiply
    m = masses.squeeze(-1) if masses.dim() > 1 else masses
    # KE per atom: 0.5 * m * ||v||^2
    ke_per_atom = 0.5 * m * (velocities * velocities).sum(dim=-1)  # (V,)
    # Sum per graph
    ke = scatter_reduce_per_graph(ke_per_atom, batch_idx, num_graphs, reduce="sum")
    return ke.unsqueeze(-1)  # (B, 1)


def temperature_per_graph(
    velocities: Float[torch.Tensor, "V 3"],
    masses: Float[torch.Tensor, "V ..."],
    batch_idx: torch.Tensor,
    num_graphs: int,
    atoms_per_graph: torch.Tensor,
    conversion_factor: float = KB_EV,
) -> Float[torch.Tensor, "B"]:
    """Compute instantaneous kinetic temperature per graph.

    Uses the equipartition theorem with 3N degrees of freedom
    (no constraint correction)::

        T = 2 * KE / (3 * N_atoms * k_B)

    Parameters
    ----------
    velocities : Float[Tensor, "V 3"]
        Per-atom velocity vectors.
    masses : Float[Tensor, "V ..."]
        Per-atom masses.  May be shape ``(V,)`` or ``(V, 1)``.
    batch_idx : Tensor
        Per-atom graph membership indices of shape ``(V,)``.
    num_graphs : int
        Number of graphs in the batch.
    atoms_per_graph : Tensor
        Number of atoms per graph, shape ``(B,)``.
    conversion_factor : float, optional
        Boltzmann coefficient in the correct units; defaults
        to eV

    Returns
    -------
    Float[Tensor, "B"]
        Instantaneous kinetic temperature per graph in Kelvin.
    """
    ke = kinetic_energy_per_graph(velocities, masses, batch_idx, num_graphs).squeeze(
        -1
    )  # (B,)
    n_atoms = atoms_per_graph.float()  # (B,)
    return (2.0 * ke) / (3.0 * n_atoms * KB_EV)


def wrap_positions_into_cell(
    positions: Float[torch.Tensor, "V 3"],
    cell: Float[torch.Tensor, "B 3 3"],
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
) -> Float[torch.Tensor, "V 3"]:
    """Wrap positions into the unit cell using fractional coordinates.

    Respects per-dimension periodicity: only periodic dimensions are
    wrapped.  Non-periodic dimensions are left unchanged.

    TODO: use `nvalchemi-ops` for this

    Parameters
    ----------
    positions : Float[Tensor, "V 3"]
        Per-atom Cartesian positions.
    cell : Float[Tensor, "B 3 3"]
        Lattice vectors as rows, one ``(3, 3)`` matrix per graph.
    pbc : Tensor
        Per-dimension periodicity flags, shape ``(B, 3)``, boolean.
    batch_idx : Tensor
        Per-atom graph membership indices of shape ``(V,)``.

    Returns
    -------
    Float[Tensor, "V 3"]
        Wrapped Cartesian positions.
    """
    # Per-atom cell and PBC lookup
    per_atom_cell = cell[batch_idx]  # (V, 3, 3)
    per_atom_pbc = pbc[batch_idx]  # (V, 3)

    # Fractional coordinates: frac_j = pos_i * inv_cell_ij
    inv_cell = torch.linalg.inv(per_atom_cell)  # (V, 3, 3)
    frac = torch.einsum("vi,vij->vj", positions, inv_cell)  # (V, 3)

    # Wrap only periodic dimensions
    wrapped_frac = torch.where(per_atom_pbc, frac % 1.0, frac)  # (V, 3)

    # Back to Cartesian: pos_i = frac_j * cell_ji
    wrapped_pos = torch.einsum("vj,vji->vi", wrapped_frac, per_atom_cell)  # (V, 3)

    return wrapped_pos
