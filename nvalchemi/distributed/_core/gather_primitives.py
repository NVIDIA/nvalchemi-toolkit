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
"""Gather-mode primitives: on-demand cross-rank ``index_select`` and
``scatter_add`` routed by GLOBAL atom IDs.

A sharded tensor stores only its rank's ``n_owned`` rows. Global-index
ops (``index_select``, ``scatter_add_``) route requests to the owning
rank via ``all_to_all_v`` exchanges. Elementwise ops (``a + b``,
``mlp(a)``) stay local — there are no halo rows to keep in sync.

These primitives are used by :class:`nvalchemi.distributed._core.shard_tensor.ShardTensor`
when its spec declares ``gather="distributed"`` or ``scatter="distributed"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.distributed._functional_collectives as funcol

from nvalchemi.distributed._core.placement import ShardRouting

if TYPE_CHECKING:
    from nvalchemi.distributed._core.halo_types import ParticleHaloConfig


__all__ = [
    "ShardRouting",
    "distributed_all_reduce",
    "distributed_index_select",
    "distributed_scatter_add",
    "mesh_group",
]


def mesh_group(mesh: Any) -> Any:
    """Return the default ``ProcessGroup`` for *mesh*.

    Accepts either a real ``DeviceMesh`` or a test-harness mock; returns
    ``None`` for both "no distribution configured" and "mesh present but not
    group-capable".

    Parameters
    ----------
    mesh : DeviceMesh or object or None
        The device mesh to resolve a group from.

    Returns
    -------
    ProcessGroup or None
        The mesh's default group, or ``None`` when the mesh is missing or does
        not expose ``get_group``.
    """
    if mesh is None:
        return None
    get_group = getattr(mesh, "get_group", None)
    if get_group is None:
        return None
    return get_group()


def funcol_group(mesh: Any) -> Any:
    """Resolve a concrete ``ProcessGroup`` for functional-collective calls.

    Functional collectives (``torch.distributed._functional_collectives``)
    require an explicit group and reject ``None`` — unlike ``dist.all_reduce``,
    where ``group=None`` means the default world group. This bridges the gap:
    return :func:`mesh_group` when available, else the default world group
    (the same group ``dist.*(group=None)`` resolved to). Callers must guard on
    ``dist.is_initialized()`` (a default group only exists once initialised).

    Eager-only: the ``getattr`` + ``_get_default_group`` resolution is not
    AOT-traceable. Inside ``autograd.Function`` forwards that run under
    ``torch.compile``, use :func:`funcol_all_reduce` instead.
    """
    group = mesh_group(mesh)
    if group is None:
        group = dist.distributed_c10d._get_default_group()
    return group


def _funcol_group_arg(mesh: Any) -> Any:
    """Group argument for functional collectives, in the form Dynamo traces.

    Prefers the ``(DeviceMesh, 0)`` spec — the form Dynamo special-cases (and
    the one physicsnemo's own compiled collectives use). Real distributed
    inference always carries a real ``DeviceMesh``, so the compiled path takes
    this branch. Falls back to the resolved ``ProcessGroup`` only for eager
    test harnesses that pass a lightweight (non-``DeviceMesh``) mesh; the
    ``isinstance`` is a compile-time constant, so under ``torch.compile`` only
    the traceable branch survives.
    """
    from torch.distributed.device_mesh import DeviceMesh  # noqa: PLC0415

    if isinstance(mesh, DeviceMesh):
        return (mesh, 0)
    return funcol_group(mesh)


def funcol_all_reduce(tensor: Any, mesh: Any, op: str = "sum") -> Any:
    """AOT-traceable functional ``all_reduce`` over *mesh*'s dim-0 group.

    ``wait_tensor`` materialises the async result explicitly (traceable).
    """
    return funcol.wait_tensor(funcol.all_reduce(tensor, op, _funcol_group_arg(mesh)))


def funcol_all_to_all_fixed(send_rows: Any, world_size: int, mesh: Any) -> Any:
    """Fixed-size (uniform-split) ``all_to_all`` — the ``fullgraph`` workaround
    for data-dependent all-to-all-v.

    ``send_rows`` has a leading dim of exactly ``world_size * cap`` rows; the
    block ``[r*cap:(r+1)*cap]`` is destined for rank ``r``. Returns the same
    shape, where block ``[i*cap:(i+1)*cap]`` was received from rank ``i``.

    Because every rank sends/receives an identical ``cap`` rows per peer, the
    split sizes are **graph constants** (derived from the static leading shape,
    not a runtime count exchange), so this traces under ``fullgraph=True`` where
    :func:`funcol_all_to_all_v_rows` cannot. Callers pad request/response buffers
    to ``cap`` and mask the padding — trading comms volume (``cap`` vs the true
    per-peer count) for a static graph.
    """
    n_rows = send_rows.shape[0]
    trailing = tuple(send_rows.shape[1:])
    row_size = 1
    for d in trailing:
        row_size *= d
    flat = send_rows.contiguous().reshape(-1)
    per_rank = (n_rows // world_size) * row_size
    splits = [per_rank] * world_size
    recv = funcol.wait_tensor(
        funcol.all_to_all_single(flat, splits, splits, _funcol_group_arg(mesh))
    )
    return recv.reshape((n_rows,) + trailing)


def funcol_fixed_index_select(
    sharded_input: Any,
    global_indices: Any,
    owner_rank: Any,
    local_index: Any,
    cap: int,
    world_size: int,
    mesh: Any,
) -> Any:
    """``fullgraph``-traceable distributed index_select via fixed-size all_to_all.

    Forward gather only (no autograd; the production
    :class:`_DistributedIndexSelect` supplies the adjoint). Replaces the
    data-dependent partition + all-to-all-v with **static** ops so the whole
    gather traces under ``fullgraph=True``:

    1. ``owner`` / ``local_idx`` of each requested global index (tensor gathers).
    2. within-owner slot via one-hot cumsum (no boolean masking → static shapes).
    3. scatter local indices into a ``(world_size * cap)`` request buffer.
    4. fixed-size all_to_all the requests, gather owned rows (masked padding),
       fixed-size all_to_all the rows back, then ``index_select`` into the output.

    ``cap`` must be ``>= max over peers of the per-peer request count`` — compute
    it once eagerly from the cached counts and pass it as a graph constant (it is
    stable across MD steps; Dynamo recompiles on the rare growth). Slots that
    would exceed ``cap`` are dropped, so an undersized ``cap`` silently loses
    requests — callers must size it from the true max.
    """
    device = global_indices.device
    owner = owner_rank.to(device)[global_indices]  # (K,)
    local_idx = local_index.to(device)[global_indices]  # (K,)
    flat_safe, in_range = _fixed_bucket_slots(owner, cap, world_size)

    n_slots = world_size * cap
    trailing = tuple(sharded_input.shape[1:])
    req_buf = torch.zeros(n_slots, dtype=torch.long, device=device)
    req_buf = req_buf.scatter(0, flat_safe, torch.where(in_range, local_idx, 0))
    valid = torch.zeros(n_slots, dtype=sharded_input.dtype, device=device)
    valid = valid.scatter(0, flat_safe, in_range.to(sharded_input.dtype))

    # Exchange requests: recv_*[i*cap:(i+1)*cap] is rank i's requests to me.
    recv_idx = funcol_all_to_all_fixed(req_buf, world_size, mesh).long()
    recv_valid = funcol_all_to_all_fixed(valid, world_size, mesh)

    n_owned = sharded_input.shape[0]
    safe = recv_idx.clamp(min=0, max=max(n_owned - 1, 0))
    recv_rows = sharded_input.index_select(0, safe)  # (n_slots, *F)
    recv_rows = recv_rows * recv_valid.reshape((n_slots,) + (1,) * len(trailing))

    rows_back = funcol_all_to_all_fixed(recv_rows, world_size, mesh)  # (n_slots, *F)
    return rows_back.index_select(0, flat_safe)  # (K, *F)


def _fixed_bucket_slots(owner: Any, cap: int, world_size: int) -> tuple[Any, Any]:
    """Per-request destination slot in a ``(world_size * cap)`` buffer.

    ``owner`` is ``(K,)`` rank-of-each-request. Returns ``(flat_safe, in_range)``
    where ``flat_safe[k] = owner[k]*cap + slot[k]`` (slot = within-owner position
    via one-hot cumsum) clamped into range, and ``in_range`` flags requests that
    fit under ``cap``. Fully static-shape (no boolean indexing) → fullgraph-safe.
    """
    onehot = torch.nn.functional.one_hot(owner, world_size)  # (K, world_size)
    slot = onehot.cumsum(0).gather(1, owner.unsqueeze(1)).squeeze(1) - 1  # (K,)
    in_range = slot < cap
    flat = owner * cap + slot
    flat_safe = torch.where(in_range, flat, torch.zeros_like(flat))
    return flat_safe, in_range


def funcol_fixed_scatter_add(
    values: Any,
    global_indices: Any,
    owner_rank: Any,
    local_index: Any,
    cap: int,
    world_size: int,
    mesh: Any,
    n_owned: int,
) -> Any:
    """``fullgraph``-traceable distributed scatter-add via fixed-size all_to_all.

    Mirror of :func:`funcol_fixed_index_select` (it is the gather's adjoint, and
    vice versa). Scatter-adds ``values`` (``(K, *F)``) at GLOBAL ``global_indices``
    into a ``(n_owned, *F)`` accumulator for THIS rank's owned rows, summing
    contributions routed from every rank. ``cap`` must be ``>=`` the true max
    per-peer count (see :func:`funcol_fixed_index_select`)."""
    device = global_indices.device
    owner = owner_rank.to(device)[global_indices]
    local_idx = local_index.to(device)[global_indices]
    flat_safe, in_range = _fixed_bucket_slots(owner, cap, world_size)

    n_slots = world_size * cap
    trailing = tuple(values.shape[1:])
    idx_buf = torch.zeros(n_slots, dtype=torch.long, device=device).scatter(
        0, flat_safe, torch.where(in_range, local_idx, 0)
    )
    valid = torch.zeros(n_slots, dtype=values.dtype, device=device).scatter(
        0, flat_safe, in_range.to(values.dtype)
    )
    val_buf = torch.zeros((n_slots,) + trailing, dtype=values.dtype, device=device)
    mask = in_range.reshape((-1,) + (1,) * len(trailing)).to(values.dtype)
    val_buf = val_buf.index_copy(0, flat_safe, values * mask)

    recv_idx = funcol_all_to_all_fixed(idx_buf, world_size, mesh).long()
    recv_valid = funcol_all_to_all_fixed(valid, world_size, mesh)
    recv_val = funcol_all_to_all_fixed(val_buf, world_size, mesh)

    safe = recv_idx.clamp(min=0, max=max(n_owned - 1, 0))
    contrib = recv_val * recv_valid.reshape((n_slots,) + (1,) * len(trailing))
    # fp64 accumulation when inputs are fp32 (owned rows fold many cross-rank
    # contributions; atomic-add order is GPU-nondeterministic) -> downcast at
    # end. Matches per_system_reduce_op and the halo folds.
    _acc_dt = torch.float64 if values.dtype == torch.float32 else values.dtype
    acc = torch.zeros((n_owned,) + trailing, dtype=_acc_dt, device=device)
    acc = acc.index_add(0, safe, contrib.to(_acc_dt))
    return acc.to(values.dtype) if _acc_dt != values.dtype else acc


def funcol_all_to_all_v_rows(
    send_rows: Any,
    send_counts: list[int],
    recv_counts: list[int],
    mesh: Any,
) -> Any:
    """AOT-traceable ``all_to_all_v`` for a row-major tensor (rows = dim 0).

    Functional analogue of :func:`_all_to_all_v_rows`: flattens to 1-D, scales
    the per-rank row counts to element counts, runs ``funcol.all_to_all_single``
    (non-autograd — callers that need gradients provide their own adjoint), and
    reshapes back. ``send_counts`` / ``recv_counts`` must be plain ``int`` lists
    (graph constants under compile), not runtime tensors.
    """
    trailing = tuple(send_rows.shape[1:])
    row_size = 1
    for d in trailing:
        row_size *= d
    flat_send = send_rows.contiguous().reshape(-1)
    send_flat = [c * row_size for c in send_counts]
    recv_flat = [c * row_size for c in recv_counts]
    flat_recv = funcol.wait_tensor(
        funcol.all_to_all_single(
            flat_send, recv_flat, send_flat, _funcol_group_arg(mesh)
        )
    )
    total_recv = sum(recv_counts)
    return flat_recv.reshape((total_recv,) + trailing)


# ======================================================================
# Collective helpers (plain-tensor, no autograd).
# ======================================================================


def _all_to_all_v_1d(
    send_tensor: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: Any,
) -> torch.Tensor:
    """all_to_all_v on a 1-D tensor. Wraps ``dist.all_to_all_single`` or
    isend/irecv if the backend lacks native all_to_all_v.
    """
    _bump_collective_count("_all_to_all_v_1d")
    total_recv = sum(recv_counts)
    recv_tensor = torch.empty(
        total_recv, dtype=send_tensor.dtype, device=send_tensor.device
    )
    try:
        dist.all_to_all_single(
            recv_tensor,
            send_tensor.contiguous(),
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=group,
        )
    except (RuntimeError, NotImplementedError):
        _isend_irecv_v_1d(send_tensor, send_counts, recv_tensor, recv_counts, group)
    return recv_tensor


def _isend_irecv_v_1d(
    send_tensor: torch.Tensor,
    send_counts: list[int],
    recv_tensor: torch.Tensor,
    recv_counts: list[int],
    group: Any,
) -> None:
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    # Gloo's TCP transport cannot isend/irecv CUDA device memory ("Bad
    # address" from writev) — unlike its higher-level collectives, p2p ops
    # expose the raw transport with no auto host-staging. Stage cuda->cpu for
    # the wire and copy back into the (cuda) recv buffer after. This path is
    # the gloo fallback only; NCCL uses ``all_to_all_single`` and never gets
    # here, so production cuda runs are unaffected.
    on_cuda = send_tensor.is_cuda
    send_buf = send_tensor.cpu() if on_cuda else send_tensor
    recv_buf = torch.empty_like(recv_tensor, device="cpu") if on_cuda else recv_tensor

    send_offsets = [0]
    for c in send_counts:
        send_offsets.append(send_offsets[-1] + c)
    recv_offsets = [0]
    for c in recv_counts:
        recv_offsets.append(recv_offsets[-1] + c)

    ops = []
    for r in range(world_size):
        send_slice = send_buf[send_offsets[r] : send_offsets[r + 1]].contiguous()
        recv_slice = recv_buf[recv_offsets[r] : recv_offsets[r + 1]]
        if r == rank:
            recv_slice.copy_(send_slice)
        else:
            if send_slice.numel() > 0:
                ops.append(dist.isend(send_slice, dst=r, group=group))
            if recv_slice.numel() > 0:
                ops.append(dist.irecv(recv_slice, src=r, group=group))
    for op in ops:
        op.wait()

    if on_cuda:
        recv_tensor.copy_(recv_buf)


def _all_to_all_v_rows(
    send_rows: torch.Tensor,
    send_counts: list[int],
    recv_counts: list[int],
    group: Any,
) -> torch.Tensor:
    """all_to_all_v for a 2D-or-more-D row-major tensor (rows = dim 0)."""
    _bump_collective_count("_all_to_all_v_rows")
    if send_rows.ndim == 1:
        return _all_to_all_v_1d(send_rows, send_counts, recv_counts, group)

    trailing = tuple(send_rows.shape[1:])
    row_size = 1
    for d in trailing:
        row_size *= d
    flat_send_1d = send_rows.contiguous().reshape(-1)
    send_counts_flat = [c * row_size for c in send_counts]
    recv_counts_flat = [c * row_size for c in recv_counts]
    flat_recv = _all_to_all_v_1d(
        flat_send_1d, send_counts_flat, recv_counts_flat, group
    )
    total_recv = sum(recv_counts)
    return flat_recv.reshape((total_recv,) + trailing)


def _all_gather_v_rows(
    local_rows: torch.Tensor,
    rank_sizes: list[int],
    group: Any,
) -> torch.Tensor:
    """All-gather a row-sharded tensor with potentially uneven per-rank
    row counts. Returns ``(sum(rank_sizes), *F)`` on every rank.

    Why a custom helper. ``physicsnemo.domain_parallel.full_tensor`` lowers
    to ``dist.all_gather`` with per-rank output buffers sized from the
    ShardTensor's spec; under NCCL ``dist.all_gather`` further lowers to
    per-rank ``broadcast_oop`` ops, which require source and dest buffers
    to have the same number of elements. Uneven shards (e.g. 312/313 for
    methane@625 atoms split across 2 ranks) crash with
    ``Tensor input and output of _broadcast_oop must have the same number
    of elements``. This helper sidesteps that by padding every rank's
    contribution to ``max(rank_sizes)`` before ``all_gather``, then
    stripping per-rank padding after.

    Each contribution is contiguous on its own rank; the gathered output
    concatenates per-rank slices in rank order.
    """
    _bump_collective_count("_all_gather_v_rows")
    world_size = dist.get_world_size(group=group)
    if world_size != len(rank_sizes):
        raise ValueError(
            f"rank_sizes has {len(rank_sizes)} entries, expected {world_size}"
        )

    trailing = tuple(local_rows.shape[1:])
    max_size = max(rank_sizes) if rank_sizes else 0
    rank = dist.get_rank(group=group)
    my_size = rank_sizes[rank]

    if local_rows.shape[0] != my_size:
        raise ValueError(
            f"rank {rank}: local rows {local_rows.shape[0]} != declared "
            f"{my_size}; sharding spec disagrees with actual tensor."
        )

    # Pad to max_size so every rank ships a same-shape buffer.
    if my_size < max_size:
        pad = torch.zeros(
            (max_size - my_size,) + trailing,
            dtype=local_rows.dtype,
            device=local_rows.device,
        )
        send_buf = torch.cat([local_rows.contiguous(), pad], dim=0)
    else:
        send_buf = local_rows.contiguous()

    out_bufs = [
        torch.empty(
            (max_size,) + trailing,
            dtype=local_rows.dtype,
            device=local_rows.device,
        )
        for _ in range(world_size)
    ]
    dist.all_gather(out_bufs, send_buf, group=group)

    # Strip per-rank padding and concatenate.
    return torch.cat([out_bufs[r][: rank_sizes[r]] for r in range(world_size)], dim=0)


_COLLECTIVE_COUNTS: dict[str, int] = {}


def _bump_collective_count(key: str) -> None:
    """Diagnostic counter, gated by ``NVALCHEMI_COUNT_COLLECTIVES=1``.

    AIMNet2 message-passing layers fire many small collectives per forward; use
    this to quantify the count distribution before designing a batching fix.
    """
    import os as _os  # noqa: PLC0415

    if _os.environ.get("NVALCHEMI_COUNT_COLLECTIVES"):
        _COLLECTIVE_COUNTS[key] = _COLLECTIVE_COUNTS.get(key, 0) + 1


def dump_collective_counts(label: str = "") -> None:
    """Print and reset ``_COLLECTIVE_COUNTS``. No-op when env var unset.

    Call from worker after a forward to surface the per-rank count
    distribution. Safe to call multiple times.
    """
    import os as _os  # noqa: PLC0415

    if not _os.environ.get("NVALCHEMI_COUNT_COLLECTIVES"):
        return

    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = -1
    print(
        f"[collective-count rank {rank}] {label} {dict(_COLLECTIVE_COUNTS)}",
        flush=True,
    )
    _COLLECTIVE_COUNTS.clear()


# Per-forward cache for ``_exchange_counts`` (key:
# ``(tuple(my_send_counts), id(group))``). AIMNet2's K MP layers reuse
# the same partition, so caching skips K-1 all_gathers per call site.
# Cleared by ``DistributedModel.__exit__`` so every forward starts cold.
_EXCHANGE_COUNTS_CACHE: dict[tuple[Any, int], list[int]] = {}


def _clear_exchange_counts_cache() -> None:
    """Drop the per-forward exchange-counts cache."""
    _EXCHANGE_COUNTS_CACHE.clear()


def _exchange_counts(
    my_send_counts: list[int],
    group: Any,
) -> list[int]:
    """Each rank reports how many items it will send to each other rank;
    learn how many each rank will receive from each other rank.

    Places the count tensors on a device the group's backend supports:
    CUDA when the backend is NCCL (NCCL has no CPU support — calling
    ``all_gather_into_tensor`` with CPU tensors over an NCCL group
    raises ``RuntimeError: No backend type associated with device type
    cpu``), else CPU.

    Cached by ``(tuple(my_send_counts), id(group))`` — repeated calls
    with the same partition skip the all_gather. See
    ``_EXCHANGE_COUNTS_CACHE``.
    """
    # NOTE: do NOT cache on a rank-local key here. ``_exchange_counts``
    # issues an ``all_gather`` (a collective); gating it on a per-rank cache
    # key (e.g. ``my_send_counts``) lets ranks disagree on hit vs miss on an
    # unbalanced partition -> one rank runs the all_gather while another
    # skips it -> collective desync. The exchange is a tiny world^2 int64
    # all_gather; always run it so every rank participates in lockstep.
    _bump_collective_count("_exchange_counts/all_gather_into_tensor")
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)

    backend = dist.get_backend(group)
    if backend == dist.Backend.NCCL and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    else:
        device = torch.device("cpu")

    my_counts = torch.tensor(my_send_counts, dtype=torch.long, device=device)
    all_counts_flat = torch.empty(
        world_size * world_size, dtype=torch.long, device=device
    )
    dist.all_gather_into_tensor(all_counts_flat, my_counts, group=group)
    all_counts = all_counts_flat.view(world_size, world_size)
    result = [int(all_counts[j, rank].item()) for j in range(world_size)]
    return list(result)


# ======================================================================
# Partition-by-owner helper
# ======================================================================


def _partition_by_owner(
    global_indices: torch.Tensor,
    meta: ShardRouting,
    world_size: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Group ``global_indices`` by their owner rank.

    Returns
    -------
    local_indices_per_rank : list of tensors
        ``local_indices_per_rank[r]`` is a 1-D int64 tensor of LOCAL
        indices on rank r that this rank requests.
    original_positions_per_rank : list of tensors
        ``original_positions_per_rank[r][i]`` is the position of the i-th
        request-to-r in the original ``global_indices`` ordering.
    """
    owner = meta.owner_rank.to(global_indices.device)[global_indices]
    local_idx = meta.local_index.to(global_indices.device)[global_indices]

    local_indices_per_rank: list[torch.Tensor] = []
    original_positions_per_rank: list[torch.Tensor] = []
    for r in range(world_size):
        mask = owner == r
        local_indices_per_rank.append(local_idx[mask].contiguous())
        original_positions_per_rank.append(torch.where(mask)[0])
    return local_indices_per_rank, original_positions_per_rank


# ======================================================================
# Autograd-aware primitives
# ======================================================================


class _DistributedIndexSelect(torch.autograd.Function):
    """Gather rows at GLOBAL indices from a sharded tensor."""

    @staticmethod
    def forward(
        ctx: Any,
        sharded_input: torch.Tensor,
        global_indices: torch.Tensor,
        meta: ShardRouting,
        config: "ParticleHaloConfig",
    ) -> torch.Tensor:
        if not dist.is_initialized():
            out = sharded_input.index_select(0, global_indices)
            ctx.save_for_backward(global_indices)
            ctx.meta = meta
            ctx.config = config
            ctx.world_size = 1
            return out

        group = mesh_group(config.mesh)
        world_size = dist.get_world_size(group=group)

        my_requests, original_positions = _partition_by_owner(
            global_indices, meta, world_size
        )
        my_send_counts = [t.shape[0] for t in my_requests]
        recv_counts = _exchange_counts(my_send_counts, group=group)
        send_idx_cat = (
            torch.cat(my_requests, dim=0)
            if any(my_send_counts)
            else torch.empty(0, dtype=torch.long, device=sharded_input.device)
        )
        recv_idx_cat = _all_to_all_v_1d(
            send_idx_cat, my_send_counts, recv_counts, group
        )

        offset = 0
        rows_to_send: list[torch.Tensor] = []
        for _r, n_recv in enumerate(recv_counts):
            chunk_idx = recv_idx_cat[offset : offset + n_recv]
            if n_recv > 0:
                rows = sharded_input.index_select(0, chunk_idx)
            else:
                rows = sharded_input.new_zeros((0,) + tuple(sharded_input.shape[1:]))
            rows_to_send.append(rows)
            offset += n_recv

        rows_send_cat = (
            torch.cat(rows_to_send, dim=0)
            if rows_to_send
            else sharded_input.new_zeros((0,) + tuple(sharded_input.shape[1:]))
        )
        rows_recv_cat = _all_to_all_v_rows(
            rows_send_cat, recv_counts, my_send_counts, group
        )

        K = global_indices.shape[0]
        output = sharded_input.new_empty((K,) + tuple(sharded_input.shape[1:]))
        offset = 0
        for r, n in enumerate(my_send_counts):
            if n > 0:
                output[original_positions[r]] = rows_recv_cat[offset : offset + n]
            offset += n

        ctx.save_for_backward(global_indices)
        ctx.meta = meta
        ctx.config = config
        ctx.world_size = world_size
        ctx.input_shape0 = sharded_input.shape[0]
        return output

    @staticmethod
    def backward(
        ctx: Any,
        grad_output: torch.Tensor,
    ) -> tuple[Any, ...]:
        (global_indices,) = ctx.saved_tensors
        meta = ctx.meta
        config = ctx.config

        # ``sharded_input`` may carry trailing non-owned rows (e.g. aimnet's
        # local padding atom); preserve the input's actual first-dim size
        # so autograd's shape check is satisfied.
        grad_input = torch.zeros(
            (ctx.input_shape0,) + tuple(grad_output.shape[1:]),
            dtype=grad_output.dtype,
            device=grad_output.device,
        )
        # scatter only into the owned slice ([:n_owned]); any trailing
        # rows receive no gradient — they never contributed to any
        # global index.
        owned_view = grad_input[: meta.n_owned]
        owned_view = distributed_scatter_add(
            owned_view, global_indices, grad_output, meta, config
        )
        grad_input = torch.cat([owned_view, grad_input[meta.n_owned :]], dim=0)
        return grad_input, None, None, None


def _fp64_index_add_(
    acc: torch.Tensor, index: torch.Tensor, values: torch.Tensor
) -> None:
    """In-place ``acc.index_add_(0, index, values)`` that accumulates in fp64
    when ``acc`` is fp32 (downcast back in place), so summing many contributions
    into one row does not drift. Matches the fp64 convention of
    ``per_system_reduce_op`` and the halo folds.
    """
    if acc.dtype == torch.float32:
        acc.copy_(acc.double().index_add(0, index, values.double()).to(torch.float32))
    else:
        acc.index_add_(0, index, values)


class _DistributedScatterAdd(torch.autograd.Function):
    """Scatter-add rows of ``src`` at GLOBAL ``indices`` into a sharded
    accumulator."""

    @staticmethod
    def forward(
        ctx: Any,
        self_t: torch.Tensor,
        global_indices: torch.Tensor,
        src: torch.Tensor,
        meta: ShardRouting,
        config: "ParticleHaloConfig",
    ) -> torch.Tensor:
        if not dist.is_initialized():
            _fp64_index_add_(self_t, global_indices, src)
            ctx.save_for_backward(global_indices)
            ctx.meta = meta
            ctx.config = config
            return self_t

        group = mesh_group(config.mesh)
        world_size = dist.get_world_size(group=group)

        local_indices_per_rank, original_positions_per_rank = _partition_by_owner(
            global_indices, meta, world_size
        )
        my_send_counts = [t.shape[0] for t in local_indices_per_rank]
        recv_counts = _exchange_counts(my_send_counts, group=group)

        send_idx_cat = (
            torch.cat(local_indices_per_rank, dim=0)
            if any(my_send_counts)
            else torch.empty(0, dtype=torch.long, device=src.device)
        )
        recv_idx_cat = _all_to_all_v_1d(
            send_idx_cat, my_send_counts, recv_counts, group
        )

        src_reordered = (
            torch.cat(
                [src[original_positions_per_rank[r]] for r in range(world_size)],
                dim=0,
            )
            if any(my_send_counts)
            else src.new_zeros((0,) + tuple(src.shape[1:]))
        )
        src_recv = _all_to_all_v_rows(src_reordered, my_send_counts, recv_counts, group)

        if recv_idx_cat.numel() > 0:
            _fp64_index_add_(self_t, recv_idx_cat, src_recv)

        ctx.save_for_backward(global_indices)
        ctx.meta = meta
        ctx.config = config
        return self_t

    @staticmethod
    def backward(
        ctx: Any,
        grad_self_t: torch.Tensor,
    ) -> tuple[Any, ...]:
        (global_indices,) = ctx.saved_tensors
        meta = ctx.meta
        config = ctx.config

        grad_self_out = grad_self_t
        grad_src = distributed_index_select(grad_self_t, global_indices, meta, config)
        return grad_self_out, None, grad_src, None, None


class _GatherToReplicate(torch.autograd.Function):
    """All-gather row-sharded owned rows to the full tensor on every rank.

    Forward replicates ``local_rows`` (this rank's owned rows) into the global
    ``(sum(rank_sizes), *F)`` tensor. Because the result is consumed
    independently on every rank, the gradient w.r.t. a rank's owned rows is the
    sum across ranks of their grad contributions to those rows — an
    ``all_reduce`` of the incoming gradient sliced to this rank's owned range.
    Adjoint of the per-layer node-feature replicate in the graph-parallel path.
    """

    @staticmethod
    def forward(
        ctx: Any,
        local_rows: torch.Tensor,
        rank_sizes: list[int],
        group: Any,
    ) -> torch.Tensor:
        ctx.rank_sizes = list(rank_sizes)
        ctx.group = group
        single = (
            group is None
            or not dist.is_initialized()
            or dist.get_world_size(group=group) == 1
        )
        ctx.rank = 0 if single else dist.get_rank(group=group)
        if single:
            return local_rows
        return _all_gather_v_rows(local_rows, list(rank_sizes), group)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        sizes = ctx.rank_sizes
        if ctx.group is None or not dist.is_initialized() or len(sizes) == 1:
            return grad_output, None, None
        grad = grad_output.contiguous().clone()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        start = sum(sizes[: ctx.rank])
        return grad[start : start + sizes[ctx.rank]], None, None


def gather_to_replicate(
    local_rows: torch.Tensor, rank_sizes: list[int], group: Any
) -> torch.Tensor:
    """Replicate row-sharded ``local_rows`` to the full ``(sum(rank_sizes), *F)``
    tensor on every rank (autograd-aware; backward sums gradients to owners)."""
    return _GatherToReplicate.apply(local_rows, rank_sizes, group)


def fixed_gather_to_replicate(
    owned_rows: torch.Tensor,
    global_indices: torch.Tensor,
    owner_rank: torch.Tensor,
    local_index: torch.Tensor,
    cap: int,
    world_size: int,
    mesh: Any,
) -> torch.Tensor:
    """``fullgraph``-traceable all-gather of owned node rows to the full tensor.

    The graph-parallel (node-partition) analog of :func:`gather_to_replicate`
    for the compiled forward: fetch every node (``global_indices = arange(N)``)
    from its owner via the fixed-size all-to-all gather, so the whole replicate
    traces inside the model's compiled region (no graph break). Autograd-aware —
    the adjoint is the reduce-scatter-sum (:func:`funcol_fixed_scatter_add`),
    routing each node's feature gradient to its owner exactly once. ``cap`` is the
    max owned-row count over ranks (a graph constant); the node-partition routing
    is static across MD steps, so it never recompiles at fixed ``N``."""
    return _FixedDistributedIndexSelect.apply(
        owned_rows,
        global_indices,
        owner_rank,
        local_index,
        cap,
        world_size,
        mesh,
    )


class _FixedDistributedIndexSelect(torch.autograd.Function):
    """``fullgraph``-compilable gather (fixed-size all_to_all).

    setup_context + the static-bucketing :func:`funcol_fixed_index_select`; its
    adjoint is :func:`funcol_fixed_scatter_add` (the two are mutual adjoints).
    ``owner_rank`` / ``local_index`` are plain tensors (not a metadata object) so
    Dynamo never traces a custom container; ``cap`` / ``world_size`` are graph
    constants supplied by the caller (precomputed eagerly)."""

    @staticmethod
    def forward(  # type: ignore[override]
        sharded_input: torch.Tensor,
        global_indices: torch.Tensor,
        owner_rank: torch.Tensor,
        local_index: torch.Tensor,
        cap: int,
        world_size: int,
        mesh: Any,
    ) -> torch.Tensor:
        return funcol_fixed_index_select(
            sharded_input,
            global_indices,
            owner_rank,
            local_index,
            cap,
            world_size,
            mesh,
        )

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        (
            sharded_input,
            global_indices,
            owner_rank,
            local_index,
            cap,
            world_size,
            mesh,
        ) = inputs
        ctx.save_for_backward(global_indices, owner_rank, local_index)
        ctx.cap = cap
        ctx.world_size = world_size
        ctx.mesh = mesh
        ctx.n_owned = sharded_input.shape[0]

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[Any, ...]:
        global_indices, owner_rank, local_index = ctx.saved_tensors
        grad_input = funcol_fixed_scatter_add(
            grad_output.contiguous(),
            global_indices,
            owner_rank,
            local_index,
            ctx.cap,
            ctx.world_size,
            ctx.mesh,
            ctx.n_owned,
        )
        # (sharded_input, global_indices, owner_rank, local_index, cap, world_size, mesh)
        return grad_input, None, None, None, None, None, None


class _FixedDistributedScatterAdd(torch.autograd.Function):
    """``fullgraph``-compilable scatter-add (fixed-size all_to_all).

    Out-of-place ``self_t + scatter_add(src @ global_indices)``. Adjoint of
    :class:`_FixedDistributedIndexSelect`: grad w.r.t. ``self_t`` is identity,
    grad w.r.t. ``src`` is the gather of ``grad_out`` at ``global_indices``."""

    @staticmethod
    def forward(  # type: ignore[override]
        self_t: torch.Tensor,
        global_indices: torch.Tensor,
        src: torch.Tensor,
        owner_rank: torch.Tensor,
        local_index: torch.Tensor,
        cap: int,
        world_size: int,
        mesh: Any,
    ) -> torch.Tensor:
        contrib = funcol_fixed_scatter_add(
            src,
            global_indices,
            owner_rank,
            local_index,
            cap,
            world_size,
            mesh,
            self_t.shape[0],
        )
        return self_t + contrib

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        (
            _self_t,
            global_indices,
            _src,
            owner_rank,
            local_index,
            cap,
            world_size,
            mesh,
        ) = inputs
        ctx.save_for_backward(global_indices, owner_rank, local_index)
        ctx.cap = cap
        ctx.world_size = world_size
        ctx.mesh = mesh

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        global_indices, owner_rank, local_index = ctx.saved_tensors
        grad_src = funcol_fixed_index_select(
            grad_out.contiguous(),
            global_indices,
            owner_rank,
            local_index,
            ctx.cap,
            ctx.world_size,
            ctx.mesh,
        )
        # (self_t, global_indices, src, owner_rank, local_index, cap, world_size, mesh)
        return grad_out, None, grad_src, None, None, None, None, None


def _fixed_cap(global_indices: Any, owner_rank: Any, world_size: int) -> int:
    """Max per-peer request count (>= cap) computed at runtime from real tensors.
    Runs inside the opaque custom op, so the data-dependent ``.item()`` never
    reaches the trace."""
    if global_indices.numel() == 0:
        local_cap = 1
    else:
        owner = owner_rank.to(global_indices.device)[global_indices]
        local_cap = int(torch.bincount(owner, minlength=world_size).max().item())
    # funcol_all_to_all_fixed pads to world*cap and splits the buffer UNIFORMLY
    # (numel/world), so ``cap`` MUST be identical on every rank. A per-rank-local
    # max desyncs on uneven partitions (rank0 314 vs rank1 313 -> uniform split
    # 314 vs 313 -> all_to_all size mismatch -> hang). All-reduce MAX so every
    # rank agrees on one cap. Runs inside the opaque custom op (balanced across
    # ranks; .item() never reaches the trace).
    if world_size > 1 and dist.is_initialized():
        _capt = torch.tensor(
            [local_cap], device=global_indices.device, dtype=torch.int64
        )
        dist.all_reduce(_capt, op=dist.ReduceOp.MAX)
        local_cap = int(_capt.item())
    return max(local_cap, 1)


@torch.library.custom_op("nvalchemi::distributed_index_select", mutates_args=())
def distributed_index_select_op(
    sharded_input: torch.Tensor,
    global_indices: torch.Tensor,
    owner_rank: torch.Tensor,
    local_index: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Dispatcher-visible distributed gather (default group). Compile-safe +
    autograd-correct analogue of :class:`_FixedDistributedIndexSelect`."""
    cap = _fixed_cap(global_indices, owner_rank, world_size)
    return funcol_fixed_index_select(
        sharded_input, global_indices, owner_rank, local_index, cap, world_size, None
    )


@distributed_index_select_op.register_fake
def _distributed_index_select_fake(
    sharded_input, global_indices, owner_rank, local_index, world_size
):
    return sharded_input.new_empty(
        (global_indices.shape[0],) + tuple(sharded_input.shape[1:])
    )


def _dis_isel_setup(ctx, inputs, output):  # type: ignore[no-untyped-def]
    sharded_input, global_indices, owner_rank, local_index, world_size = inputs
    ctx.save_for_backward(global_indices, owner_rank, local_index)
    ctx.world_size = world_size
    ctx.n_owned = sharded_input.shape[0]


def _dis_isel_backward(ctx, grad_out):  # type: ignore[no-untyped-def]
    global_indices, owner_rank, local_index = ctx.saved_tensors
    cap = _fixed_cap(global_indices, owner_rank, ctx.world_size)
    grad_input = funcol_fixed_scatter_add(
        grad_out.contiguous(),
        global_indices,
        owner_rank,
        local_index,
        cap,
        ctx.world_size,
        None,
        ctx.n_owned,
    )
    return grad_input, None, None, None, None


distributed_index_select_op.register_autograd(
    _dis_isel_backward, setup_context=_dis_isel_setup
)


@torch.library.custom_op("nvalchemi::distributed_scatter_add", mutates_args=())
def distributed_scatter_add_op(
    self_t: torch.Tensor,
    global_indices: torch.Tensor,
    src: torch.Tensor,
    owner_rank: torch.Tensor,
    local_index: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Dispatcher-visible distributed scatter-add (default group). Out-of-place
    ``self_t + scatter_add(src @ global_indices)``; adjoint of the gather."""
    cap = _fixed_cap(global_indices, owner_rank, world_size)
    contrib = funcol_fixed_scatter_add(
        src,
        global_indices,
        owner_rank,
        local_index,
        cap,
        world_size,
        None,
        self_t.shape[0],
    )
    return self_t + contrib


@distributed_scatter_add_op.register_fake
def _distributed_scatter_add_fake(
    self_t, global_indices, src, owner_rank, local_index, world_size
):
    return torch.empty_like(self_t)


def _dis_sadd_setup(ctx, inputs, output):  # type: ignore[no-untyped-def]
    self_t, global_indices, src, owner_rank, local_index, world_size = inputs
    ctx.save_for_backward(global_indices, owner_rank, local_index)
    ctx.world_size = world_size


def _dis_sadd_backward(ctx, grad_out):  # type: ignore[no-untyped-def]
    global_indices, owner_rank, local_index = ctx.saved_tensors
    cap = _fixed_cap(global_indices, owner_rank, ctx.world_size)
    grad_src = funcol_fixed_index_select(
        grad_out.contiguous(),
        global_indices,
        owner_rank,
        local_index,
        cap,
        ctx.world_size,
        None,
    )
    # (self_t, global_indices, src, owner_rank, local_index, world_size)
    return grad_out, None, grad_src, None, None, None


distributed_scatter_add_op.register_autograd(
    _dis_sadd_backward, setup_context=_dis_sadd_setup
)


def distributed_index_select(
    sharded_input: torch.Tensor,
    global_indices: torch.Tensor,
    meta: ShardRouting,
    config: "ParticleHaloConfig",
    cap: int | None = None,
) -> torch.Tensor:
    """Gather rows from a sharded tensor by GLOBAL atom IDs.

    Parameters
    ----------
    sharded_input : Tensor
        ``(n_owned, *F)`` owned rows on this rank.
    global_indices : Tensor
        ``(K,)`` int64. Global atom IDs to gather. Indices belonging to
        other ranks are fetched via on-demand ``all_to_all_v``.
    meta : ShardRouting
        Routing table for global ↔ (owner, local_idx).
    config : ParticleHaloConfig
        Provides the process group.

    Returns
    -------
    Tensor
        ``(K, *F)`` tensor with the requested rows in the order they
        appear in ``global_indices``.

    Notes
    -----
    ``cap=None`` (default) uses the variable-count all-to-all-v path (eager).
    Passing ``cap`` (the max per-peer request count, precomputed eagerly)
    selects the fixed-size, ``fullgraph``-compilable path
    (:class:`_FixedDistributedIndexSelect`). ``cap`` must be ``>=`` the true max
    or requests are silently dropped.
    """
    if cap is None:
        return _DistributedIndexSelect.apply(
            sharded_input, global_indices, meta, config
        )
    world_size = (
        dist.get_world_size(group=mesh_group(config.mesh))
        if dist.is_initialized()
        else 1
    )
    return _FixedDistributedIndexSelect.apply(
        sharded_input,
        global_indices,
        meta.owner_rank,
        meta.local_index,
        cap,
        world_size,
        config.mesh,
    )


def distributed_scatter_add(
    self_t: torch.Tensor,
    global_indices: torch.Tensor,
    src: torch.Tensor,
    meta: ShardRouting,
    config: "ParticleHaloConfig",
    cap: int | None = None,
) -> torch.Tensor:
    """Scatter-add ``src`` rows at GLOBAL ``indices`` into a sharded
    accumulator. ``cap`` selects the fixed-size fullgraph path (see
    :func:`distributed_index_select`)."""
    if cap is None:
        return _DistributedScatterAdd.apply(self_t, global_indices, src, meta, config)
    world_size = (
        dist.get_world_size(group=mesh_group(config.mesh))
        if dist.is_initialized()
        else 1
    )
    return _FixedDistributedScatterAdd.apply(
        self_t,
        global_indices,
        src,
        meta.owner_rank,
        meta.local_index,
        cap,
        world_size,
        config.mesh,
    )


# ----------------------------------------------------------------------
# distributed_all_reduce — autograd-aware SUM all_reduce. Sibling of
# per_system_reduce minus the scatter step (input is already at the
# output shape). Used by PME's partial charge mesh, Ewald's partial
# structure factors, etc. SUM only.
# ----------------------------------------------------------------------


class _DistributedAllReduceSum(torch.autograd.Function):
    """Forward: ``dist.all_reduce(SUM)`` on a clone of the input.

    Backward: because the output is replicated across every rank, a
    downstream consumer on any rank contributes to this call's input
    gradient. Summing incoming gradients across the mesh gives the
    correct per-rank gradient; this is exactly an all_reduce(SUM) on the
    incoming ``grad_out``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        tensor: torch.Tensor,
        config: "ParticleHaloConfig",
    ) -> torch.Tensor:
        # Separate forward + setup_context (no ctx) for AOT-traceability.
        if dist.is_initialized():
            return funcol_all_reduce(tensor.contiguous(), config.mesh)
        return tensor.contiguous().clone()

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        _tensor, config = inputs
        ctx.config = config

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        if dist.is_initialized():
            grad = funcol_all_reduce(grad_out.contiguous(), ctx.config.mesh)
        else:
            grad = grad_out.contiguous().clone()
        # (tensor, config)
        return grad, None


def distributed_all_reduce(
    tensor: torch.Tensor,
    config: "ParticleHaloConfig",
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> torch.Tensor:
    """Autograd-aware SUM all-reduce across ``config.mesh``.

    Use when every rank holds a partial contribution at the output shape
    and you need the globally summed value replicated on every rank. The
    input is a regular ``torch.Tensor`` (not a :class:`ShardTensor`);
    shape is preserved.

    Parameters
    ----------
    tensor : Tensor
        Per-rank partial contribution. Not modified in place — a clone
        is reduced so the caller's tensor is safe to reuse.
    config : ParticleHaloConfig
        Carries the mesh / process group to reduce across. When
        ``dist.is_initialized()`` is ``False`` or the mesh has no
        process group, the call is a no-op copy (single-rank semantics).
    op : ReduceOp, default SUM
        Only ``SUM`` is wired; raise otherwise.

    Returns
    -------
    Tensor
        Same shape/dtype/device as ``tensor``, summed across the mesh
        and replicated on every rank.

    Notes
    -----
    * Backward is also a sum-all-reduce on the incoming gradient —
      symmetric with forward, matching the existing per_system_reduce
      pattern.
    * For per-system reductions where the input is per-atom and needs a
      scatter-add first, use :func:`per_system_reduce` instead — it
      composes the local scatter with the all-reduce in one primitive.
    """
    if op is not dist.ReduceOp.SUM:
        raise NotImplementedError(
            f"distributed_all_reduce op={op} not implemented; only SUM is "
            "currently wired."
        )
    return _DistributedAllReduceSum.apply(tensor, config)
