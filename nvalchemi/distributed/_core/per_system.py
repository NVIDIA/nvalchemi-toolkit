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

"""Distributed segmented (per-group) reduction primitive.

Reduces a sharded per-row tensor into per-group values, where each row
carries an integer group index and the group's rows may be split across
mesh ranks. Needed by any model that does a global per-group reduction
inside the forward pass (rather than only at the output). In MLIP usage a
"group" is a molecular system and a "row" is an atom; concrete cases:

  - AIMNet2: ``mol_sum`` every message-passing pass (charge conservation)
  - UMA OMOL: ``balance_channels_batched`` every layer
  - MEGNet / M3GNet: ``readout_nodes`` mean-pool injected back into per-atom
  - Ewald: structure factor reduction per system per k-point

Signature::

    per_system_reduce(local_vals, system_index, n_systems, config, op=SUM)

Forward: local scatter_sum per group + ``all_reduce`` across the domain
mesh. Backward: all_reduce the incoming gradient (adjoint of a sum-based
all_reduce is itself), then ``index_select`` back to per-row grads.

Only ``ReduceOp.SUM`` is implemented; the MEAN / MAX / MIN extensions are
sketched at the bottom of this file.
"""

from __future__ import annotations

import os as _os
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from nvalchemi.distributed._core.gather_primitives import funcol_all_reduce

if TYPE_CHECKING:
    from nvalchemi.distributed._core.particle_halo import ParticleHaloConfig


def _expand_system_index_like(
    system_index: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Broadcast ``system_index`` (1-D, length n) to match ``values`` (n, *F)
    along all trailing dims — required by ``scatter_add_`` which wants the
    index tensor to broadcast to ``src``'s shape."""
    while system_index.ndim < values.ndim:
        system_index = system_index.unsqueeze(-1)
    return system_index.expand_as(values)


class _PerSystemReduceSum(torch.autograd.Function):
    """Forward: local per-system scatter-add + all_reduce(SUM).

    Backward: all_reduce(SUM) the incoming grad, then index_select back to
    per-atom grads. The all_reduce in backward is required because the
    forward output is *replicated* across ranks (after all_reduce) — so
    a downstream consumer on any rank contributes to *every* rank's
    local_vals gradient, not just the locally-producing rank's.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        local_vals: torch.Tensor,
        system_index: torch.Tensor,
        n_systems: int,
        config: "ParticleHaloConfig",
    ) -> torch.Tensor:
        # Split ``forward`` / ``setup_context`` (no ``ctx`` arg) so AOTAutograd
        # can trace the backward graph under ``torch.compile``.
        # fp32 atomic-add scatter is GPU-nondeterministic (commit order varies
        # with partition), so accumulate in fp64 and downcast — bounds the
        # per-accumulator error at ~1e-15 instead of fp32's ~1e-7 and makes
        # single-rank vs multi-rank reductions match to fp64 noise.
        input_dtype = local_vals.dtype
        accum_dtype = torch.float64 if input_dtype == torch.float32 else input_dtype

        out_shape = (n_systems,) + tuple(local_vals.shape[1:])
        acc = torch.zeros(out_shape, dtype=accum_dtype, device=local_vals.device)
        expanded_idx = _expand_system_index_like(system_index.long(), local_vals)
        src = local_vals.to(accum_dtype) if accum_dtype != input_dtype else local_vals
        acc.scatter_add_(0, expanded_idx, src)

        debug = _os.environ.get("NVALCHEMI_REDUCE_DEBUG")
        if debug:
            rank = dist.get_rank() if dist.is_initialized() else 0
            local_sum = acc.detach().sum().item()
            print(
                f"[reduce-debug rank {rank}] _PerSystemReduceSum.forward  "
                f"local_vals.shape={tuple(local_vals.shape)} n_sys={n_systems}  "
                f"acc.sum(rank-local, post-scatter)={local_sum:+.6e}",
                flush=True,
            )

        # Functional collective (not ``dist.all_reduce``) so the AOT-captured
        # graph holds a traceable ``funcol`` op rather than a raw ProcessGroup
        # reference. We pass the resolved ``ProcessGroup`` (``mesh_group`` —
        # the same group ``dist.all_reduce`` used) rather than a ``(mesh, 0)``
        # tuple, because ``config.mesh`` is not always a real torch
        # ``DeviceMesh`` (test harnesses use a lightweight mesh). ``wait_tensor``
        # materialises the async result explicitly (traceable).
        if dist.is_initialized():
            acc = funcol_all_reduce(acc, config.mesh)

        if debug:
            global_sum = acc.detach().sum().item()
            print(
                f"[reduce-debug rank {rank}] _PerSystemReduceSum.forward  "
                f"acc.sum(post-all_reduce, replicated)={global_sum:+.6e}",
                flush=True,
            )

        out = acc.to(input_dtype) if accum_dtype != input_dtype else acc
        return out

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        local_vals, system_index, n_systems, config = inputs
        ctx.save_for_backward(system_index)
        ctx.config = config
        ctx.n_systems = n_systems

    @staticmethod
    def backward(ctx: Any, grad_out: torch.Tensor) -> tuple[Any, ...]:
        (system_index,) = ctx.saved_tensors

        if dist.is_initialized():
            # Functional all_reduce — no in-place mutation, so no clone needed.
            grad_out = funcol_all_reduce(grad_out.contiguous(), ctx.config.mesh)

        grad_local = grad_out.index_select(0, system_index.long())
        # (local_vals, system_index, n_systems, config)
        return grad_local, None, None, None


# ======================================================================
# Compile-path / dispatch-level custom op.
# ======================================================================
#
# Dispatcher-visible analogue of :class:`_PerSystemReduceSum`. A
# ``torch.library.custom_op`` (opaque to fake mode; eager body runs at runtime
# with the default process group) with ``register_autograd`` so the cross-rank
# ``all_reduce`` adjoint is captured even when the op is reached BELOW autograd
# (the ``__torch_dispatch__`` path used under ``torch.compile``).
# Single-domain-mesh-dim topology (default group). ``n_systems`` rides as an int
# constant; everything else as tensors.


@torch.library.custom_op("nvalchemi::per_system_reduce", mutates_args=())
def per_system_reduce_op(
    local_vals: torch.Tensor,
    system_index: torch.Tensor,
    n_systems: int,
) -> torch.Tensor:
    """Local per-system scatter-add + SUM ``all_reduce`` over the default group.

    ``local_vals`` is the owned-only per-atom values; ``system_index`` maps each
    owned atom to its system id. Returns ``(n_systems, *F)`` replicated on every
    rank. fp32 inputs accumulate in fp64 (atomic-add order is GPU-nondeterministic)
    then downcast — matching :class:`_PerSystemReduceSum`.
    """
    in_dt = local_vals.dtype
    acc_dt = torch.float64 if in_dt == torch.float32 else in_dt
    acc = torch.zeros(
        (n_systems,) + tuple(local_vals.shape[1:]), dtype=acc_dt, device=local_vals.device
    )
    expanded_idx = _expand_system_index_like(system_index.long(), local_vals)
    acc.scatter_add_(0, expanded_idx, local_vals.to(acc_dt))
    if dist.is_initialized():
        acc = funcol_all_reduce(acc, None)
    return acc.to(in_dt) if acc_dt != in_dt else acc


@per_system_reduce_op.register_fake
def _per_system_reduce_fake(local_vals, system_index, n_systems):
    return local_vals.new_empty((n_systems,) + tuple(local_vals.shape[1:]))


def _per_system_reduce_setup_context(ctx, inputs, output):  # type: ignore[no-untyped-def]
    local_vals, system_index, _n_systems = inputs
    ctx.save_for_backward(system_index)


def _per_system_reduce_backward(ctx, grad_out):  # type: ignore[no-untyped-def]
    # Output is replicated (post all_reduce), so the grad is all_reduced (the
    # adjoint of the replicating all_reduce is itself), then index_selected back
    # to per-owned-atom rows by the system index.
    (system_index,) = ctx.saved_tensors
    if dist.is_initialized():
        grad_out = funcol_all_reduce(grad_out.contiguous(), None)
    grad_local = grad_out.index_select(0, system_index.long())
    return grad_local, None, None


per_system_reduce_op.register_autograd(
    _per_system_reduce_backward, setup_context=_per_system_reduce_setup_context
)


def per_system_reduce(
    local_vals: torch.Tensor,
    system_index: torch.Tensor,
    n_systems: int,
    config: "ParticleHaloConfig",
    op: dist.ReduceOp = dist.ReduceOp.SUM,
) -> torch.Tensor:
    """Distributed per-system reduction.

    Each rank contributes ``local_vals`` for the atoms it owns; these are
    aggregated per system via ``scatter_add_``, then an ``all_reduce``
    across the mesh combines contributions from all ranks that own atoms
    of the same system. The result has shape ``(n_systems, *F)`` and is
    replicated on every rank.

    Parameters
    ----------
    local_vals : Tensor
        Shape ``(n_owned, *F)``. Per-atom values on this rank.
    system_index : Tensor
        Shape ``(n_owned,)`` integer. The system each atom belongs to;
        values in ``[0, n_systems)``.
    n_systems : int
        Total number of systems in the distributed batch (globally known).
    config : ParticleHaloConfig
        Halo config, for mesh / process group.
    op : ReduceOp, default SUM
        Only SUM is wired today. MEAN / MAX / MIN would require an
        atom-count-per-system broadcast (MEAN) or per-rank local-reduce
        composition (MAX / MIN) — deliberately deferred until a concrete
        caller needs them.

    Returns
    -------
    Tensor
        Shape ``(n_systems, *F)``, same on every rank.
    """
    if op is not dist.ReduceOp.SUM:
        raise NotImplementedError(
            f"per_system_reduce op={op} not implemented; only SUM is currently wired."
        )
    return _PerSystemReduceSum.apply(local_vals, system_index, n_systems, config)


__all__ = ["per_system_reduce"]


# Future extensions (sketch): ``per_system_mean`` would divide by a
# system_count obtained via per_system_reduce of ones; ``per_system_max``
# would use scatter_reduce_(amax) + all_reduce(MAX), routing backward
# grad to the global argmax atom (saved as (idx, rank) pair).
