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

from nvalchemi._typing import ModelOutputs

__all__ = ["autograd_forces", "autograd_stresses", "sum_outputs"]


def autograd_forces(
    energy: torch.Tensor,
    positions: torch.Tensor,
    training: bool = False,
    retain_graph: bool = False,
) -> torch.Tensor:
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


def autograd_stresses(
    energy: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    num_graphs: int,
    training: bool = False,
    retain_graph: bool = False,
) -> torch.Tensor:
    """Compute stresses as ``-1/V * dE/d(strain)`` via autograd.

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
        Stress tensor of shape ``[B, 3, 3]``.
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
        ``{"energies", "forces", "stresses"}``.

    Returns
    -------
    ModelOutputs
        Combined output dict.
    """
    additive = additive_keys or {"energies", "forces", "stresses"}
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
