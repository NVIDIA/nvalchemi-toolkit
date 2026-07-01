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

"""In-compile halo-routing holder.

The per-layer neighbor refresh runs inside the compiled region and needs the
halo routing tensors, which vary every step — so they must reach it as graph
inputs, never as constants baked at trace time. The compile bridge publishes the
routing here from its graph inputs; the in-region refresh helper reads it back.
Both happen in the same compiled frame, so Dynamo threads the tensors through.
Outside compile the holder is ``None`` and the helpers take their eager path.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "set_compile_routing",
    "get_compile_routing",
    "clear_compile_routing",
    "compile_routing_active",
    "set_gp_compile_routing",
    "get_gp_compile_routing",
    "clear_gp_compile_routing",
]

# Single-slot holder. ``None`` = not inside a compiled DD region (helpers use
# their eager path). A 5-tuple ``(send_index, recv_dest, recv_real, n_owned,
# world_size)`` = an in-region refresh should use the static halo op wired to it.
_COMPILE_ROUTING: list[Any] = [None]


def set_compile_routing(
    send_index: Any,
    recv_dest: Any,
    recv_real: Any,
    n_owned: Any,
    world_size: int,
) -> None:
    """Publish this step's halo routing for in-region refresh helpers. Call
    inside the compiled region with the routing taken from graph inputs, so the
    values stay fakified. ``world_size`` is the (constant) mesh size."""
    _COMPILE_ROUTING[0] = (send_index, recv_dest, recv_real, n_owned, world_size)


def get_compile_routing() -> Any:
    """Return the published routing tuple, or ``None`` outside a compiled DD
    region (the eager case)."""
    return _COMPILE_ROUTING[0]


def clear_compile_routing() -> None:
    """Reset the holder to ``None``. The bridge calls this after each compiled
    forward so a later eager refresh never reads stale trace-time routing."""
    _COMPILE_ROUTING[0] = None


def compile_routing_active() -> bool:
    """True iff in-compile routing is currently published."""
    return _COMPILE_ROUTING[0] is not None


# Separate single-slot holder for the graph-parallel (node-partition) per-layer
# all-gather. ``None`` = not active. A 6-tuple ``(global_indices, owner_rank,
# local_index, cap, world_size, mesh)`` drives the fullgraph-traceable fixed
# gather. Unlike the halo routing this is STATIC across MD steps (index-based
# partition, no atom migration), so the in-region refresh may read it as
# trace-time constants without recompiling.
_GP_COMPILE_ROUTING: list[Any] = [None]


def set_gp_compile_routing(
    global_indices: Any,
    owner_rank: Any,
    local_index: Any,
    cap: int,
    world_size: int,
    mesh: Any,
) -> None:
    """Publish the graph-parallel all-gather routing for the in-region refresh.

    Set eagerly before the (model-internal) compiled forward; the static routing
    is read back inside the compiled Edgewise via :func:`get_gp_compile_routing`.
    """
    _GP_COMPILE_ROUTING[0] = (
        global_indices, owner_rank, local_index, cap, world_size, mesh,
    )


def get_gp_compile_routing() -> Any:
    """Return the graph-parallel all-gather routing tuple, or ``None``."""
    return _GP_COMPILE_ROUTING[0]


def clear_gp_compile_routing() -> None:
    """Reset the graph-parallel routing holder to ``None`` after the forward."""
    _GP_COMPILE_ROUTING[0] = None
