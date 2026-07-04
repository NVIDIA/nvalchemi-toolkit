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
"""2D-parallel dynamics: pipeline × domain-decomposition building blocks.

A ``DistributedPipeline`` today maps one rank → one stage. To run a *streaming
pipeline whose stages are each domain-decomposed* (proposal-distributed-pipeline-dd.md)
we lay out a 2D ``DeviceMesh`` ``(pipeline, domain)`` and let each pipeline index own
a whole **domain sub-mesh row** running :class:`DomainParallel`. This module holds the
group-aware primitives that layer stays built on:

* :class:`StageGroup` — one pipeline stage occupying a domain sub-mesh.
* :func:`resolve_stage_layout` — this rank's ``(pipeline_index, domain_rank,
  domain_sub_mesh)`` from the 2D mesh.
* :func:`group_all_done` — a stage-group finishes as a **unit** (a MIN-reduce of the
  ``done`` flag over the domain sub-mesh), so no rank of a stage advances or exits
  while its peers are still running (which would deadlock the group's next collective).

Multi-node layout (proposal §0): ``DeviceMesh`` is row-major, so building the mesh
as ``("pipeline", "domain")`` makes ``domain`` the inner/contiguous dim — intra-node
(NVLink) when ``domain_size ≤ gpus_per_node`` — while the ``pipeline`` dim (rare
group→group handoffs) is the cross-node axis. The handoff (Phase 3) runs on the
``pipeline``-dim group, never the global group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:
    from nvalchemi.distributed.domain_parallel import DomainParallel


@dataclass
class StageGroup:
    """One pipeline stage that occupies a domain sub-mesh (a rank-group).

    Parameters
    ----------
    dynamics : DomainParallel
        The stage's dynamics, wrapped in :class:`DomainParallel` and bound to this
        stage's **domain sub-mesh** (``mesh2d["domain"]`` for its pipeline row). The
        group's intra-stage DD (halo/all-gather, globalized thermo/convergence) is
        handled entirely by ``DomainParallel`` over that sub-mesh.
    pipeline_index : int
        This stage's position along the pipeline dimension.
    prior_index, next_index : int | None
        Pipeline neighbors (indices, not ranks) for the group→group handoff. ``None``
        marks the first/last stage. Auto-wired into a linear chain when unset (-1).
    buffer_config : Any
        Fixed-size handoff buffer (full-system sized for gather→lead→scatter), as in
        the 1-rank pipeline. Only the group **lead** (domain-rank 0) uses it.
    """

    dynamics: "DomainParallel"
    pipeline_index: int
    prior_index: int | None = -1
    next_index: int | None = -1
    buffer_config: Any = None


def resolve_stage_layout(mesh: Any) -> tuple[int, int, Any]:
    """This rank's ``(pipeline_index, domain_rank, domain_sub_mesh)``.

    ``mesh`` is the 2D ``DeviceMesh`` with dims ``("pipeline", "domain")``. Slicing
    ``mesh["domain"]`` yields this rank's domain row (its DD group) and
    ``mesh["pipeline"]`` its pipeline column; ``get_local_rank`` on each gives the
    index along that axis. Row-major layout ⇒ global rank =
    ``pipeline_index * domain_size + domain_rank``.
    """
    domain_sub_mesh = mesh["domain"]
    domain_rank = int(domain_sub_mesh.get_local_rank())
    pipeline_index = int(mesh["pipeline"].get_local_rank())
    return pipeline_index, domain_rank, domain_sub_mesh


def group_all_done(local_done: bool, domain_sub_mesh: Any) -> bool:
    """Return whether the whole stage-group is done (MIN-reduce over the domain row).

    A stage-group must finish as a unit: only when **every** rank of the group is
    done does the group stop. A per-rank ``done`` would let one rank exit while its
    peers still expect it in the group's next collective, deadlocking the group.
    Single-process / no domain group → passthrough.
    """
    from nvalchemi.distributed._core.gather_primitives import mesh_group

    if not dist.is_initialized() or domain_sub_mesh is None:
        return local_done
    group = mesh_group(domain_sub_mesh)
    if dist.get_world_size(group=group) == 1:
        return local_done
    flag = torch.tensor([1 if local_done else 0], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    return bool(flag.item())


def handoff_forward(
    full_batch: Any, sender_index: int, receiver_index: int, mesh: Any
) -> Any:
    """Hand a full system from one stage-group's lead to the next stage-group's lead.

    The cross-stage hop is **per-system** (a system advances a stage only when its
    stage finishes it), so it runs point-to-point between the two group **leads**
    (domain-rank 0 of each stage) over the ``pipeline``-dim group — never the global
    group, so on multi-node it routes over IB between the paired leads (proposal §0).
    A full, mesh-consistent :class:`Batch` (produced by the sender's
    ``DomainParallel.gather(dst=lead)``) is shipped as a picklable object; the
    receiver then re-scatters it with ``DomainParallel.partition`` inside its group.

    Only lead ranks participate; non-leads return ``None`` (they receive their owned
    block from the receiver's ``partition``). On the send lead ``full_batch`` is the
    gathered system; returns ``None``. On the recv lead returns the received system.

    Parameters
    ----------
    full_batch : Batch | None
        The gathered full system on the sender's lead; ignored elsewhere.
    sender_index, receiver_index : int
        Pipeline indices of the sending / receiving stage-groups.
    mesh : DeviceMesh
        The 2D ``(pipeline, domain)`` mesh.
    """
    from nvalchemi.distributed._core.gather_primitives import mesh_group

    if not dist.is_initialized():
        return full_batch  # single-process: hand straight through

    if int(mesh["domain"].get_local_rank()) != 0:
        return None  # non-lead ranks sit out the cross-stage hop

    # ``mesh.mesh`` is the (n_pipeline, domain_size) global-rank layout; column 0 is
    # the stage leads. The lead ranks share a ``pipeline``-dim group.
    layout = mesh.mesh
    lead_send = int(layout[sender_index, 0])
    lead_recv = int(layout[receiver_index, 0])
    group = mesh_group(mesh["pipeline"])
    me = dist.get_rank()
    if me == lead_send:
        dist.send_object_list([full_batch], dst=lead_recv, group=group)
        return None
    if me == lead_recv:
        box: list[Any] = [None]
        dist.recv_object_list(box, src=lead_send, group=group)
        return box[0]
    return None


def wire_stage_chain(stages: dict[int, StageGroup]) -> None:
    """Auto-wire ``prior_index``/``next_index`` into a linear pipeline chain.

    Mirrors ``DistributedPipeline.setup``'s rank-chain wiring, but over pipeline
    indices (each index is a stage-group, not a single rank). Explicit values (not
    ``-1``) are left untouched so branched topologies can be declared.
    """
    order = sorted(stages.keys())
    for i, idx in enumerate(order):
        st = stages[idx]
        if st.prior_index == -1:
            st.prior_index = order[i - 1] if i > 0 else None
        if st.next_index == -1:
            st.next_index = order[i + 1] if i < len(order) - 1 else None
