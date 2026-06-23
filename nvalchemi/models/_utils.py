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
"""Utility functions for model composition.

Standalone building blocks for users who need control beyond what
:class:`~nvalchemi.models.pipeline.PipelineModelWrapper` offers.
These functions are also used internally by the pipeline.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from nvalchemi._typing import (
    BatchIndices,
    Energy,
    Forces,
    LatticeVectors,
    ModelOutputs,
    NodePositions,
    StrainDisplacement,
    Stress,
)

__all__ = [
    "autograd_forces",
    "autograd_forces_and_stresses",
    "autograd_stresses",
    "cell_cache_needs_update",
    "prepare_strain",
    "sum_outputs",
]


def cell_cache_needs_update(
    cell: LatticeVectors,
    cached_cell: LatticeVectors | None,
    rtol: float = 1e-5,
    atol: float | None = None,
) -> bool:
    """Return ``True`` when ``cell`` is incompatible with ``cached_cell``.

    Parameters
    ----------
    cell : torch.Tensor
        Current cell tensor.
    cached_cell : torch.Tensor | None
        Previously cached cell tensor, or ``None`` when no cell has been
        cached yet.
    rtol : float, optional
        Relative tolerance passed to :func:`torch.allclose`.
        Defaults to ``1e-5``.
    atol : float or None, optional
        Absolute tolerance passed to :func:`torch.allclose`.
        When ``None`` (default), uses
        ``max(1e-6, torch.finfo(cell.dtype).eps)``.

    Returns
    -------
    bool
        ``True`` if the cache should be refreshed.
    """
    if atol is None:
        atol = max(1e-6, torch.finfo(cell.dtype).eps)

    if cached_cell is None or cell.shape != cached_cell.shape:
        return True
    # Check device and dtype for compatibility with torch.allclose
    if cell.device != cached_cell.device:
        return True
    if cell.dtype != cached_cell.dtype:
        return True
    if not torch.allclose(cell, cached_cell, rtol=rtol, atol=atol):
        return True
    return False


def autograd_forces(
    energy: Energy,
    positions: NodePositions,
    training: bool = False,
    retain_graph: bool = False,
) -> Forces:
    """Compute forces as ``-dE/dr`` via autograd.

    Parameters
    ----------
    energy : torch.Tensor
        Total energy tensor (must be part of a computation graph that
        includes *positions*).
    positions : torch.Tensor
        Atomic positions with ``requires_grad=True``.
    training : bool, optional
        If ``True``, ``create_graph=True`` is set so that higher-order
        gradients are available (needed for training).
    retain_graph : bool, optional
        If ``True``, the computation graph is retained after the backward
        pass.  Needed when subsequent autograd calls traverse shared
        graph nodes.

    Returns
    -------
    torch.Tensor
        Forces tensor with same shape as *positions*.
    """
    effective_retain = retain_graph or training
    return -torch.autograd.grad(
        energy,
        positions,
        grad_outputs=torch.ones_like(energy),
        create_graph=training,
        retain_graph=effective_retain,
    )[0]


def prepare_strain(
    positions: NodePositions,
    cell: LatticeVectors,
    batch_idx: BatchIndices,
) -> tuple[NodePositions, LatticeVectors, StrainDisplacement]:
    """Set up the affine strain trick for autograd stress computation.

    Creates a per-system 3x3 displacement tensor with
    ``requires_grad=True``, scales positions and cell through the symmetric
    part of it, and returns all three tensors.  After running the model on
    the scaled positions/cell, compute stresses with standard PyTorch
    autograd::

        scaled_pos, scaled_cell, displacement = prepare_strain(
            positions, cell, batch_idx
        )
        energy = model(scaled_pos, scaled_cell, ...)

        # Forces:
        forces = -torch.autograd.grad(
            energy, scaled_pos, torch.ones_like(energy),
            retain_graph=True,
        )[0]

        # Stresses:
        grad = torch.autograd.grad(
            energy, displacement, torch.ones_like(energy),
        )[0]
        volume = torch.det(cell).abs().view(-1, 1, 1)
        stresses = grad.view(B, 3, 3) / volume

    This function is used internally by :class:`PipelineModelWrapper`
    for autograd groups, and is available for users who want to
    implement autograd stresses in their own model wrappers.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions, shape ``[N, 3]``.
    cell : torch.Tensor
        Unit cell, shape ``[B, 3, 3]``.
    batch_idx : torch.Tensor
        Graph index per atom, shape ``[N]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        ``(scaled_positions, scaled_cell, displacement)`` where
        ``displacement`` is ``[B, 3, 3]`` with ``requires_grad=True``.
        The returned tensor is an unconstrained autograd leaf; only its
        symmetric part is applied as strain.
    """
    n_systems = cell.shape[0]
    displacement = torch.zeros(
        n_systems,
        3,
        3,
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (displacement + displacement.mT)
    deformation = (
        torch.eye(3, dtype=positions.dtype, device=positions.device)
        + symmetric_displacement
    )
    # Scale positions: pos'[n] = pos[n] @ deformation[system_of_atom[n]]
    # Index into deformation per-atom, then batch-matmul each atom's row.
    per_atom_deformation = deformation[batch_idx]  # [N, 3, 3]
    scaled_positions = torch.einsum("ni,nij->nj", positions, per_atom_deformation)
    # Scale cell: cell'[b] = cell[b] @ deformation[b]
    scaled_cell = torch.einsum("bij,bjk->bik", cell, deformation)
    return scaled_positions, scaled_cell, displacement


def autograd_stresses(
    energy: Energy,
    displacement: StrainDisplacement,
    cell: LatticeVectors,
    num_graphs: int,
    training: bool = False,
    retain_graph: bool = False,
) -> Stress:
    """Compute tensile-positive Cauchy stress via autograd.

    Returns ``1/V * dE/d(strain)`` in eV/Å³.

    Parameters
    ----------
    energy : torch.Tensor
        Total energy tensor.
    displacement : torch.Tensor
        Displacement tensor (symmetric strain applied to positions).
    cell : torch.Tensor
        Unit cell tensor of shape ``[B, 3, 3]``.
    num_graphs : int
        Number of graphs (systems) in the batch.
    training : bool, optional
        If ``True``, create the computation graph for higher-order gradients.
    retain_graph : bool, optional
        If ``True``, retain the computation graph.

    Returns
    -------
    torch.Tensor
        Cauchy stress tensor of shape ``[B, 3, 3]`` in eV/Å³.
    """
    effective_retain = retain_graph or training
    grad = torch.autograd.grad(
        energy,
        displacement,
        grad_outputs=torch.ones_like(energy),
        create_graph=training,
        retain_graph=effective_retain,
    )[0]
    volume = torch.det(cell).abs().view(-1, 1, 1)
    return grad.view(num_graphs, 3, 3) / volume


def autograd_forces_and_stresses(
    energy: Energy,
    positions: NodePositions,
    displacement: StrainDisplacement,
    cell: LatticeVectors,
    num_graphs: int,
    training: bool = False,
    retain_graph: bool = False,
) -> tuple[Forces, Stress]:
    """Compute forces and tensile-positive Cauchy stress in one autograd call.

    Parameters
    ----------
    energy : torch.Tensor
        Total energy tensor.
    positions : torch.Tensor
        Atomic positions with ``requires_grad=True``.
    displacement : torch.Tensor
        Displacement tensor from :func:`prepare_strain`.
    cell : torch.Tensor
        Original unit cell tensor of shape ``[B, 3, 3]``.
    num_graphs : int
        Number of graphs (systems) in the batch.
    training : bool, optional
        If ``True``, create the computation graph for higher-order gradients.
    retain_graph : bool, optional
        If ``True``, retain the computation graph.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(forces, stress)`` with shapes ``[N, 3]`` and ``[B, 3, 3]``.
    """
    effective_retain = retain_graph or training
    position_grad, displacement_grad = torch.autograd.grad(
        energy,
        (positions, displacement),
        grad_outputs=torch.ones_like(energy),
        create_graph=training,
        retain_graph=effective_retain,
    )
    forces = -position_grad
    volume = torch.det(cell).abs().view(-1, 1, 1)
    stress = displacement_grad.view(num_graphs, 3, 3) / volume
    return forces, stress


def sum_outputs(
    *outputs: ModelOutputs,
    additive_keys: set[str] | None = None,
) -> ModelOutputs:
    """Element-wise sum of :class:`ModelOutputs` on specified keys.

    Keys in *additive_keys* are summed across all *outputs*.
    Non-additive keys use last-write-wins semantics.

    Parameters
    ----------
    *outputs : ModelOutputs
        One or more model output dicts to combine.
    additive_keys : set[str] | None, optional
        Keys whose values should be summed.  Defaults to
        ``{"energy", "forces", "stress"}``.

    Returns
    -------
    ModelOutputs
        Combined output dict.
    """
    additive = additive_keys or {"energy", "forces", "stress"}
    result: ModelOutputs = OrderedDict()
    for out in outputs:
        for key, val in out.items():
            if val is None:
                continue
            if key in additive and key in result and result[key] is not None:
                result[key] = result[key] + val
            else:
                result[key] = val
    return result
