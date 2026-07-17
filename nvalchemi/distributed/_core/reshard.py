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
"""Per-element redistribution of ShardTensors based on destination rank.

``reshard_by_destination`` is the particle analogue of grid-based
``redistribute``.  Instead of changing placement strategy, it physically
moves elements between ranks based on a per-element destination map
(e.g., spatial rank assignment for atom migration).

Uses ``indexed_all_to_all_v_wrapper`` internally but returns a proper
``ShardTensor`` with updated ``sharding_shapes``.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def reshard_by_destination(
    tensor: torch.Tensor,
    destinations: torch.Tensor,
    mesh: Any,  # DeviceMesh at runtime
) -> torch.Tensor:
    """Redistribute tensor elements to new ranks based on per-element destinations.

    Unlike ``ShardTensor.redistribute()`` which changes placement strategy,
    this physically moves elements between ranks based on a destination map.
    Returns a plain ``torch.Tensor`` with the received elements.

    Parameters
    ----------
    tensor : torch.Tensor
        Local tensor, shape ``(N_local, ...)``.
    destinations : torch.Tensor
        ``(N_local,)`` int tensor where ``destinations[i]`` is the rank
        that should own element ``i`` after resharding.
    mesh : DeviceMesh
        1D device mesh for communication.

    Returns
    -------
    torch.Tensor
        Received elements, shape ``(N_new_local, ...)``.
    """
    # Single-process is a no-op: every element already lives on the only
    # rank. Gate on the *default* group's world size (not the mesh) so this
    # returns before touching ``mesh`` — both when ``dist`` is uninitialized
    # and when an ambient 1-rank group is up (e.g. a session-scoped gloo PG
    # under pytest). Resharding within a genuine multi-rank world proceeds.
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor

    from physicsnemo.distributed.utils import indexed_all_to_all_v_wrapper

    from nvalchemi.distributed._core.gather_primitives import mesh_group

    group = mesh_group(mesh)
    world_size = dist.get_world_size(group=group)
    device = tensor.device

    # Sort by destination for contiguous sends.
    destinations = destinations.to(torch.int64)
    counts = torch.bincount(destinations, minlength=world_size)
    sorted_idx = torch.argsort(destinations, stable=True)
    offsets = torch.cat(
        [torch.zeros(1, dtype=counts.dtype, device=device), counts.cumsum(0)]
    )

    # Build per-rank send indices.
    send_indices: list[torch.Tensor] = [
        sorted_idx[offsets[r] : offsets[r + 1]] for r in range(world_size)
    ]

    # All-gather send counts → sizes matrix.
    all_counts_list = [torch.zeros_like(counts) for _ in range(world_size)]
    dist.all_gather(all_counts_list, counts, group=group)
    sizes = [c.tolist() for c in all_counts_list]

    # Exchange.
    received = indexed_all_to_all_v_wrapper(
        tensor=tensor,
        indices=send_indices,
        sizes=sizes,
        dim=0,
        group=group,
    )

    return received
