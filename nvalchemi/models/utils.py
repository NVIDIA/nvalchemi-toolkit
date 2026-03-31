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
"""Utilities for model wrappers: neighbor lists, unit conversion, model helpers."""
from __future__ import annotations

import math
from typing import Any

import torch
from nvalchemiops.torch.neighbors import neighbor_list
from torch import nn

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------

BOHR_TO_ANGSTROM: float = 0.529177210544
ANGSTROM_TO_BOHR: float = 1.0 / BOHR_TO_ANGSTROM
HARTREE_TO_EV: float = 27.211386245981
EV_TO_HARTREE: float = 1.0 / HARTREE_TO_EV

TORCH_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float64": torch.float64,
}


# ---------------------------------------------------------------------------
# Neighbor list resolution
# ---------------------------------------------------------------------------


def resolve_neighbor_list(
    data: Any,
    nblist: AdaptiveNeighborList,
    *,
    cutoff: float,
    positions: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    method: str | None = None,
    return_format: str = "coo",
) -> tuple[torch.Tensor, ...]:
    """Reuse or compute a neighbor list for a model wrapper.

    Parameters
    ----------
    data : Batch
        Input batch.
    nblist : AdaptiveNeighborList
        Neighbor-list builder.
    cutoff : float
        Neighbor cutoff radius.
    positions : torch.Tensor or None
        Optional positions override.
    cell : torch.Tensor or None
        Optional cell override.
    method : str or None
        Optional explicit neighbor-list method override.
    return_format : str
        ``"coo"`` returns ``(edge_index, unit_shifts_or_None)``.
        ``"matrix"`` returns ``(neighbor_matrix, num_neighbors, neighbor_shifts_or_None)``.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Neighbor list tensors depending on ``return_format``.
    """
    if return_format == "coo":
        edge_index = getattr(data, "edge_index", None)
        if edge_index is not None:
            unit_shifts = getattr(data, "unit_shifts", None)
            return (edge_index, unit_shifts)
    elif return_format == "matrix":
        nbmat = getattr(data, "neighbor_matrix", None)
        num_nb = getattr(data, "num_neighbors", None)
        if nbmat is not None and num_nb is not None:
            nb_shifts = getattr(data, "neighbor_shifts", None)
            return (nbmat, num_nb, nb_shifts)

    positions = data.positions if positions is None else positions
    periodic = hasattr(data, "cell") and hasattr(data, "pbc")

    return_neighbor_list = return_format == "coo"

    nl_kwargs: dict[str, Any] = {
        "positions": positions,
        "cutoff": cutoff,
        "batch_idx": data.batch,
        "batch_ptr": data.ptr,
        "return_neighbor_list": return_neighbor_list,
    }
    if periodic:
        nl_kwargs["cell"] = data.cell if cell is None else cell
        nl_kwargs["pbc"] = data.pbc
    if method is not None:
        nl_kwargs["method"] = method

    result = nblist(**nl_kwargs)

    if return_format == "coo":
        # nvalchemiops returns [2, E]; current repo stores [E, 2]
        edge_index = result[0].T if result[0].shape[0] == 2 else result[0]
        unit_shifts = result[2] if len(result) > 2 else None
        return (edge_index, unit_shifts)
    else:
        # matrix mode: (neighbor_matrix, num_neighbors, shifts_or_None)
        return result[0], result[1], result[2] if len(result) > 2 else None


# ---------------------------------------------------------------------------
# AdaptiveNeighborList
# ---------------------------------------------------------------------------


def _round_to_16(n: int) -> int:
    """Round up to the nearest multiple of 16 for warp alignment."""
    return ((n + 15) // 16) * 16


class AdaptiveNeighborList:
    """Neighbor list wrapper with automatic buffer sizing and overflow retry.

    Wraps ``nvalchemiops.torch.neighbors.neighbor_list`` and provides:

    - Initial buffer estimation from ``density * (4/3 * pi * cutoff^3)``
    - Automatic buffer growth (1.5x) on overflow
    - Buffer shrink when utilisation drops below target (matrix mode)
    - Minimum of 16 neighbors (PBC) or ``num_atoms - 1`` (non-PBC)
    - Warp-aligned ``max_neighbors`` (multiples of 16)

    Attributes
    ----------
    cutoff : float
        Interaction cutoff radius (assumed Angstrom).
    density : float
        Estimated neighbor density used for initial buffer sizing.
    target_utilization : float
        Desired buffer utilisation fraction for shrink decisions.
    kwargs : dict[str, Any]
        Extra keyword arguments forwarded to the backend on every call.
    max_neighbors : int
        Current warp-aligned buffer capacity (multiple of 16).

    Parameters
    ----------
    cutoff : float
        Interaction cutoff radius.
    density : float
        Estimated neighbor density for initial buffer sizing.
        Default ``0.2``.
    target_utilization : float
        Shrink buffer if utilisation falls below this fraction.
        Default ``0.75``.
    **kwargs
        Extra keyword arguments forwarded to ``neighbor_list`` on every call.
    """

    def __init__(
        self,
        cutoff: float,
        density: float = 0.2,
        target_utilization: float = 0.75,
        **kwargs: Any,
    ) -> None:
        self.cutoff = cutoff
        self.density = density
        self.target_utilization = target_utilization
        self.kwargs = kwargs
        self.max_neighbors = _round_to_16(
            max(16, int(density * (4.0 / 3.0 * math.pi * cutoff**3)))
        )

    def __call__(self, **kwargs: Any) -> tuple[torch.Tensor, ...]:
        """Compute the neighbor list, retrying on overflow.

        Parameters
        ----------
        **kwargs
            Keyword arguments passed to ``nvalchemiops.torch.neighbors.neighbor_list``.

        Returns
        -------
        tuple[torch.Tensor, ...]
            Results from ``neighbor_list``.
        """
        merged = {**self.kwargs, **kwargs}
        merged.setdefault("cutoff", self.cutoff)
        merged.setdefault("fill_value", -1)

        if "max_neighbors" not in merged:
            merged["max_neighbors"] = self.max_neighbors

        while True:
            try:
                result = neighbor_list(**merged)
                break
            except Exception as e:
                if "overflow" in str(e).lower() or type(e).__name__ == "NeighborOverflowError":
                    self.max_neighbors = _round_to_16(int(self.max_neighbors * 1.5))
                    merged["max_neighbors"] = self.max_neighbors
                else:
                    raise

        if not merged.get("return_neighbor_list", False):
            if isinstance(result, tuple) and len(result) >= 2:
                nbmat = result[0]
                if nbmat.ndim == 2:
                    actual_max = int((nbmat != merged["fill_value"]).sum(dim=-1).max().item())
                    if actual_max > 0 and actual_max < self.max_neighbors * self.target_utilization:
                        self.max_neighbors = _round_to_16(
                            max(16, math.ceil(actual_max / self.target_utilization))
                        )

        return result


# ---------------------------------------------------------------------------
# Model freeze / unfreeze / compile
# ---------------------------------------------------------------------------


def freeze_model(model: nn.Module) -> nn.Module:
    """Freeze all parameters and set eval mode.

    Parameters
    ----------
    model
        Module whose parameters are frozen in-place.

    Returns
    -------
    nn.Module
        The same module with all ``requires_grad`` flags disabled
        and ``eval()`` applied.
    """
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def unfreeze_model(model: nn.Module) -> nn.Module:
    """Unfreeze all parameters and set train mode.

    Parameters
    ----------
    model
        Module whose parameters are unfrozen in-place.

    Returns
    -------
    nn.Module
        The same module with all ``requires_grad`` flags enabled
        and ``train()`` applied.
    """
    model.train()
    for param in model.parameters():
        param.requires_grad_(True)
    return model


def compile_model(model: nn.Module, **compile_kwargs: Any) -> nn.Module:
    """Apply ``torch.compile`` to the model.

    Parameters
    ----------
    model : nn.Module
        Model to compile.
    **compile_kwargs
        Forwarded to ``torch.compile`` (e.g. ``backend``, ``mode``).

    Returns
    -------
    nn.Module
        Compiled model.

    Raises
    ------
    Exception
        Propagates any ``torch.compile`` failure.
    """
    return torch.compile(model, **compile_kwargs)


# ---------------------------------------------------------------------------
# Stress from virial
# ---------------------------------------------------------------------------


def virial_to_stress(
    virial: torch.Tensor,
    cell: torch.Tensor,
) -> torch.Tensor:
    """Convert virial tensor to tensile-positive Cauchy stress.

    ``σ = -W / V`` where ``V = |det(cell)|``.

    The virial ``W`` must follow the ``W = -∂E/∂ε`` convention
    (as returned by the Ewald, PME, and DFT-D3 kernels).  Kernels
    that use the opposite separation-vector convention (e.g. LJ, DSF)
    must negate their virial before calling this function.

    Parameters
    ----------
    virial : torch.Tensor
        Virial tensor in the ``W = -∂E/∂ε`` convention, shape ``(B, 3, 3)``.
    cell : torch.Tensor
        Cell vectors, shape ``(B, 3, 3)``.

    Returns
    -------
    torch.Tensor
        Tensile-positive Cauchy stress tensor, shape ``(B, 3, 3)``.
    """
    volume = torch.abs(torch.linalg.det(cell))
    return -virial / volume.unsqueeze(-1).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Derivative computation infrastructure
# ---------------------------------------------------------------------------


def prepare_stress_scaling(
    positions: torch.Tensor,
    cell: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Set up the affine scaling tensors used for stress via autograd.

    Creates a per-system identity scaling matrix with ``requires_grad``,
    applies it to positions and cell, and returns the transformed tensors
    along with the scaling matrix.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions, shape ``(V, 3)``.
    cell : torch.Tensor
        Cell vectors, shape ``(B, 3, 3)`` or ``(3, 3)``.
    batch_idx : torch.Tensor
        Per-atom system index, shape ``(V,)``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(positions_scaled, cell_scaled, scaling)`` where ``scaling``
        is the gradient target (requires_grad=True).
    """
    scaling = torch.eye(3, dtype=positions.dtype, device=positions.device)
    if cell.ndim == 3:
        scaling = scaling.repeat(cell.shape[0], 1, 1)
    scaling = scaling.requires_grad_(True)

    if scaling.ndim == 3:
        atom_scaling = torch.index_select(scaling, 0, batch_idx)
        positions_scaled = (positions.unsqueeze(1) @ atom_scaling).squeeze(1)
    else:
        positions_scaled = positions @ scaling

    cell_scaled = cell @ scaling if cell.ndim == 2 else torch.bmm(cell, scaling)
    return positions_scaled, cell_scaled, scaling


def compute_derivatives(
    energy: torch.Tensor,
    positions: torch.Tensor,
    scaling: torch.Tensor | None = None,
    cell: torch.Tensor | None = None,
    *,
    compute_forces: bool = True,
    compute_stress: bool = False,
) -> dict[str, torch.Tensor | None]:
    """Compute forces and/or stress from total energy via autograd.

    Parameters
    ----------
    energy : torch.Tensor
        Per-system energies, shape ``(B, 1)`` or ``(B,)``.
    positions : torch.Tensor
        Atomic positions used in forward pass, shape ``(V, 3)``.
        Must have ``requires_grad=True``.
    scaling : torch.Tensor or None
        Affine scaling matrix from :func:`prepare_stress_scaling`.
        Required when ``compute_stress=True``.
    cell : torch.Tensor or None
        Cell vectors, shape ``(B, 3, 3)``.  Required for stress.
    compute_forces : bool
        Whether to compute forces (``-dE/dr``).
    compute_stress : bool
        Whether to compute stress from the scaling gradient.

    Returns
    -------
    dict[str, torch.Tensor | None]
        Keys: ``"forces"``, ``"stresses"``, each present only if requested.
    """
    result: dict[str, torch.Tensor | None] = {}
    tot_energy = energy.sum()

    grad_targets: list[torch.Tensor] = []
    if compute_forces:
        grad_targets.append(positions)
    if compute_stress and scaling is not None:
        grad_targets.append(scaling)

    if not grad_targets:
        return result

    grads = torch.autograd.grad(
        tot_energy, grad_targets, create_graph=False, retain_graph=False
    )

    idx = 0
    if compute_forces:
        result["forces"] = -grads[idx]
        idx += 1

    if compute_stress and scaling is not None and cell is not None:
        dedc = grads[idx]
        volume = torch.abs(torch.linalg.det(cell))
        if volume.ndim == 0:
            result["stresses"] = dedc / volume
        else:
            result["stresses"] = dedc / volume.unsqueeze(-1).unsqueeze(-1)

    return result


# ---------------------------------------------------------------------------
# Per-atom to per-system energy aggregation
# ---------------------------------------------------------------------------


def aggregate_per_system_energy(
    energy: torch.Tensor,
    batch_idx: torch.Tensor,
    num_graphs: int,
) -> torch.Tensor:
    """Aggregate per-atom energy into per-system energy.

    If *energy* is already per-system (shape does not match the number of
    atoms implied by ``batch_idx``), it is returned unchanged.

    Parameters
    ----------
    energy : torch.Tensor
        Energy tensor -- either per-atom ``(V,)`` or per-system ``(B,)``/``(B, 1)``.
    batch_idx : torch.Tensor
        Per-atom system index, shape ``(V,)``.
    num_graphs : int
        Number of systems in the batch.

    Returns
    -------
    torch.Tensor
        Per-system energy, shape ``(B, 1)``.
    """
    flat = energy.squeeze(-1) if energy.ndim > 1 else energy
    if flat.shape[0] != batch_idx.shape[0]:
        return energy if energy.ndim > 1 else energy.unsqueeze(-1)

    # Accumulate in float64 for numerical stability.
    # Energy dtype ambiguity to be resolved by AttributeMap contract.
    per_system = torch.zeros(
        num_graphs,
        dtype=torch.float64,
        device=energy.device,
    )
    per_system = per_system.scatter_add(0, batch_idx.long(), flat.to(torch.float64))
    return per_system.to(energy.dtype).unsqueeze(-1)


# ---------------------------------------------------------------------------
# Shared helpers for direct-force wrappers
# ---------------------------------------------------------------------------
