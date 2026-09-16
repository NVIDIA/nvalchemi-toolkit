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
"""Private geometry utilities shared by reaction-path implementations."""

from __future__ import annotations

import torch
from torch import Tensor


def minimum_image_displacement(
    displacement: Tensor,
    graph_idx: Tensor,
    cell: Tensor | None,
    pbc: Tensor | None,
) -> Tensor:
    """Apply batched minimum-image wrapping to Cartesian displacements.

    Parameters
    ----------
    displacement : Tensor
        Flat Cartesian displacement vectors with shape ``(N, 3)``.
    graph_idx : Tensor
        Graph index for each displacement with shape ``(N,)``.
    cell : Tensor or None
        Cell vectors for every graph with shape ``(B, 3, 3)``. ``None``
        denotes nonperiodic geometry.
    pbc : Tensor or None
        Periodic-boundary flags for every graph with shape ``(B, 3)``.
        ``None`` denotes nonperiodic geometry.

    Returns
    -------
    Tensor
        Minimum-image Cartesian displacements with shape ``(N, 3)``.
    """
    if cell is None or pbc is None:
        return displacement

    cells = cell.reshape(-1, 3, 3)
    pbcs = pbc.reshape(-1, 3)
    periodic_graphs = torch.any(pbcs, dim=-1)
    identity = torch.eye(3, dtype=cells.dtype, device=cells.device).expand_as(cells)
    working_cells = torch.where(periodic_graphs[:, None, None], cells, identity)
    graph_idx = graph_idx.long()
    displacement_cells = working_cells.index_select(0, graph_idx)
    inverse_cells = torch.linalg.inv(working_cells).index_select(0, graph_idx)
    fractional = torch.einsum("ni,nij->nj", displacement, inverse_cells)
    displacement_pbc = pbcs.index_select(0, graph_idx)
    fractional = fractional - torch.where(
        displacement_pbc, torch.floor(fractional + 0.5), 0.0
    )
    return torch.einsum("ni,nij->nj", fractional, displacement_cells)
