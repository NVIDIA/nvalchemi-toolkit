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

Where possible, these functions delegate to GPU-optimized kernels in
``nvalchemiops``.  The Python function signatures are preserved for
backward compatibility.

This module is **not** part of the public API.
"""

from __future__ import annotations

from typing import Literal

import torch
import warp as wp
from jaxtyping import Float
from nvalchemiops.dynamics.utils import (
    compute_cell_inverse,
    compute_kinetic_energy,
    wrap_positions_to_cell,
)
from nvalchemiops.segment_ops import (
    segmented_max,
    segmented_mean,
    segmented_min,
    segmented_sum,
)

# Boltzmann constant in eV/K (NIST 2018 CODATA value).
KB_EV: float = 8.617333262e-5

# Supported scatter-reduce operations.
ScatterReduce = Literal["amax", "sum", "amin", "mean"]

# ---------------------------------------------------------------------------
# Custom ops for torch.compile support
# ---------------------------------------------------------------------------
# Wrapping warp kernel calls with @torch.library.custom_op creates an opaque
# boundary that torch.compile treats as a single node — it won't trace into
# the warp interop code (which uses ctypes and breaks fullgraph tracing).


@torch.library.custom_op("nvalchemi_hooks::segmented_sum", mutates_args=())
def _segmented_sum(
    values: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    out = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    segmented_sum(
        wp.from_torch(values.contiguous()),
        wp.from_torch(idx.to(torch.int32)),
        wp.from_torch(out),
    )
    return out


@_segmented_sum.register_fake
def _(values: torch.Tensor, idx: torch.Tensor, num_segments: int) -> torch.Tensor:
    return torch.empty(num_segments, device=values.device, dtype=values.dtype)


@torch.library.custom_op("nvalchemi_hooks::segmented_max", mutates_args=())
def _segmented_max(
    values: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    out = torch.full(
        (num_segments,), float("-inf"), device=values.device, dtype=values.dtype
    )
    segmented_max(
        wp.from_torch(values.contiguous()),
        wp.from_torch(idx.to(torch.int32)),
        wp.from_torch(out),
    )
    return out


@_segmented_max.register_fake
def _(values: torch.Tensor, idx: torch.Tensor, num_segments: int) -> torch.Tensor:
    return torch.empty(num_segments, device=values.device, dtype=values.dtype)


@torch.library.custom_op("nvalchemi_hooks::segmented_min", mutates_args=())
def _segmented_min(
    values: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    out = torch.full(
        (num_segments,), float("inf"), device=values.device, dtype=values.dtype
    )
    segmented_min(
        wp.from_torch(values.contiguous()),
        wp.from_torch(idx.to(torch.int32)),
        wp.from_torch(out),
    )
    return out


@_segmented_min.register_fake
def _(values: torch.Tensor, idx: torch.Tensor, num_segments: int) -> torch.Tensor:
    return torch.empty(num_segments, device=values.device, dtype=values.dtype)


@torch.library.custom_op("nvalchemi_hooks::segmented_mean", mutates_args=())
def _segmented_mean(
    values: torch.Tensor, idx: torch.Tensor, num_segments: int
) -> torch.Tensor:
    sums = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    counts = torch.zeros(num_segments, device=values.device, dtype=torch.int32)
    out = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    segmented_mean(
        wp.from_torch(values.contiguous()),
        wp.from_torch(idx.to(torch.int32)),
        wp.from_torch(sums),
        wp.from_torch(counts),
        wp.from_torch(out),
    )
    return out


@_segmented_mean.register_fake
def _(values: torch.Tensor, idx: torch.Tensor, num_segments: int) -> torch.Tensor:
    return torch.empty(num_segments, device=values.device, dtype=values.dtype)


@torch.library.custom_op("nvalchemi_hooks::compute_kinetic_energy", mutates_args=())
def _compute_ke(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    ke = torch.zeros(num_graphs, device=velocities.device, dtype=velocities.dtype)
    vec_dtype = wp.vec3d if velocities.dtype == torch.float64 else wp.vec3f
    compute_kinetic_energy(
        wp.from_torch(velocities.contiguous(), dtype=vec_dtype),
        wp.from_torch(masses.contiguous()),
        wp.from_torch(ke),
        wp.from_torch(batch_idx.to(torch.int32)),
        num_systems=num_graphs,
    )
    return ke


@_compute_ke.register_fake
def _(
    velocities: torch.Tensor,
    masses: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    return torch.empty(num_graphs, device=velocities.device, dtype=velocities.dtype)


@torch.library.custom_op("nvalchemi_hooks::wrap_positions", mutates_args=())
def _wrap_positions(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    vec_dtype = wp.vec3d if positions.dtype == torch.float64 else wp.vec3f
    mat_dtype = wp.mat33d if positions.dtype == torch.float64 else wp.mat33f

    # Transpose cell from row-convention (nvalchemi) to column-convention (nvalchemiops)
    cell_T = cell.transpose(-2, -1).contiguous()

    # Convert to warp arrays
    num_systems = cell_T.shape[0]
    wp_pos = wp.from_torch(positions.clone().contiguous(), dtype=vec_dtype)
    wp_cell = wp.from_torch(cell_T, dtype=mat_dtype)
    wp_cell_inv = wp.zeros(num_systems, dtype=mat_dtype, device=wp_pos.device)
    wp_batch_idx = wp.from_torch(batch_idx.to(torch.int32))

    compute_cell_inverse(wp_cell, wp_cell_inv)
    wrap_positions_to_cell(wp_pos, wp_cell, wp_cell_inv, wp_batch_idx)

    return wp.to_torch(wp_pos)


@_wrap_positions.register_fake
def _(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(positions)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


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

    Delegates to GPU-optimized segmented reduction kernels in
    ``nvalchemiops.segment_ops``.

    Parameters
    ----------
    values : Tensor
        1-D tensor of shape ``(V,)`` with one scalar per node.
    batch_idx : Tensor
        Integer tensor of shape ``(V,)`` mapping each node to its
        graph index. Must be sorted (guaranteed by torch_geometric-style
        batching).
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
        return _segmented_sum(values, batch_idx, num_graphs)
    if reduce == "amax":
        return _segmented_max(values, batch_idx, num_graphs)
    if reduce == "amin":
        return _segmented_min(values, batch_idx, num_graphs)
    return _segmented_mean(values, batch_idx, num_graphs)


def kinetic_energy_per_graph(
    velocities: Float[torch.Tensor, "V 3"],
    masses: Float[torch.Tensor, "V ..."],
    batch_idx: torch.Tensor,
    num_graphs: int,
) -> Float[torch.Tensor, "B 1"]:
    """Compute ``0.5 * sum(m_i * ||v_i||^2)`` per graph.

    Delegates to ``nvalchemiops.dynamics.utils.compute_kinetic_energy``.

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
    m = masses.squeeze(-1) if masses.dim() > 1 else masses
    ke = _compute_ke(velocities, m, batch_idx, num_graphs)
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

    KE is computed via ``nvalchemiops`` for GPU performance, while the
    temperature formula uses pure PyTorch to preserve exact DOF semantics
    (3N rather than 3N-3).

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
    return (2.0 * ke) / (3.0 * n_atoms * conversion_factor)


def wrap_positions_into_cell(
    positions: Float[torch.Tensor, "V 3"],
    cell: Float[torch.Tensor, "B 3 3"],
    pbc: torch.Tensor,
    batch_idx: torch.Tensor,
) -> Float[torch.Tensor, "V 3"]:
    """Wrap positions into the unit cell using fractional coordinates.

    Respects per-dimension periodicity: only periodic dimensions are
    wrapped.  Non-periodic dimensions are left unchanged.

    This function modifies ``positions`` **in-place** and returns the
    same tensor.  Delegates to ``nvalchemiops.dynamics.utils.wrap_positions_to_cell``
    for GPU-optimized wrapping, then applies per-dimension PBC masking
    in pure PyTorch.

    Parameters
    ----------
    positions : Float[Tensor, "V 3"]
        Per-atom Cartesian positions. Modified in-place.
    cell : Float[Tensor, "B 3 3"]
        Lattice vectors as rows, one ``(3, 3)`` matrix per graph.
    pbc : Tensor
        Per-dimension periodicity flags, shape ``(B, 3)``, boolean.
    batch_idx : Tensor
        Per-atom graph membership indices of shape ``(V,)``.

    Returns
    -------
    Float[Tensor, "V 3"]
        The same ``positions`` tensor (modified in-place).
    """
    original = positions.clone()
    wrapped = _wrap_positions(positions, cell, batch_idx)

    # Restore non-periodic dimensions
    per_atom_pbc = pbc[batch_idx]  # (V, 3)
    positions.copy_(torch.where(per_atom_pbc, wrapped, original))
    return positions
