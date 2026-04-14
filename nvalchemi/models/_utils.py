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
    Stress,
)

__all__ = ["autograd_forces", "autograd_stresses", "prepare_strain", "sum_outputs"]


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
) -> tuple[NodePositions, LatticeVectors, torch.Tensor]:
    """Set up the affine strain trick for autograd stress computation.

    Creates a per-system 3x3 displacement tensor with
    ``requires_grad=True``, scales positions and cell through it, and
    returns all three tensors.  After running the model on the scaled
    positions/cell, compute stresses with standard PyTorch autograd::

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
        stresses = -grad.view(B, 3, 3) / volume

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
    symmetric = (
        torch.eye(3, dtype=positions.dtype, device=positions.device) + displacement
    )
    # Scale positions: pos'[n] = pos[n] @ symmetric[system_of_atom[n]]
    # Index into symmetric per-atom, then batch-matmul each atom's row.
    per_atom_symmetric = symmetric[batch_idx]  # [N, 3, 3]
    scaled_positions = torch.einsum("ni,nij->nj", positions, per_atom_symmetric)
    # Scale cell: cell'[b] = cell[b] @ symmetric[b]
    scaled_cell = torch.einsum("bij,bjk->bik", cell, symmetric)
    return scaled_positions, scaled_cell, displacement


def autograd_stresses(
    energy: Energy,
    displacement: torch.Tensor,
    cell: LatticeVectors,
    num_graphs: int,
    training: bool = False,
    retain_graph: bool = False,
) -> Stress:
    """Compute Cauchy stress as ``W/V = -1/V * dE/d(strain)`` via autograd.

    Returns the Cauchy stress tensor in eV/Å³.

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
    return -grad.view(num_graphs, 3, 3) / volume


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
