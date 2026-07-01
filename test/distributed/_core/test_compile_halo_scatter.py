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

"""Halo-correction scatter under ``torch.compile``.

The MACE message path (``agg.scatter_add_(0, idx, msg)`` on a halo ShardTensor)
must, under ``torch.compile``, fold cross-rank contributions written into
borrowed halo rows back into their owners — identically to the eager
``__torch_function__`` handler (``_halo_scatter_correction``).

Under compile the eager handler is bypassed (it manually constructs
ShardTensors, which Dynamo cannot trace); the scatter routes through
``__torch_dispatch__``, which re-applies the halo reverse+forward via the
``nvalchemi::halo_scatter_correct`` custom op (opaque to fake mode; marker
indices ride as a flat tensor constant). This test asserts the compiled forward
AND backward match eager, and that the correction is non-trivial (cross-rank
work actually happens, so the eager==compiled match is not vacuous).

2-rank gloo + ``torch.multiprocessing.spawn`` so it runs on CPU without GPUs.
"""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import DeviceMesh


def _patch_all_to_all_for_gloo() -> None:
    """physicsnemo's indexed_all_to_all_v_wrapper uses list-form dist.all_to_all
    (gloo-unsupported) during the halo-metadata build; swap an isend/irecv
    equivalent. The funcol halo-exchange path itself uses all_to_all_single,
    which gloo supports."""
    import physicsnemo.distributed.utils as pn_utils

    def _gloo(tensor: Any, indices: Any, sizes: Any, dim: int = 0, group: Any = None):
        comm = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)
        x_send = [tensor[idx].contiguous() for idx in indices]
        x_recv = []
        shape = list(tensor.shape)
        for r in range(comm):
            shape[dim] = sizes[r][rank]
            x_recv.append(torch.empty(shape, dtype=tensor.dtype, device=tensor.device))
        ops = []
        for r in range(comm):
            if r == rank:
                x_recv[r].copy_(x_send[r])
            else:
                if x_send[r].numel() > 0:
                    ops.append(dist.isend(x_send[r], dst=r, group=group))
                if x_recv[r].numel() > 0:
                    ops.append(dist.irecv(x_recv[r], src=r, group=group))
        for op in ops:
            op.wait()
        return torch.cat(x_recv, dim=dim)

    pn_utils.indexed_all_to_all_v_wrapper = _gloo


def _build_rank_halo(
    mesh: Any,
    rank: int,
    world_size: int,
    ghost_width: float = 5.0,
    device: Any = "cpu",
    dtype: Any = torch.float64,
    return_padded: bool = False,
):
    from nvalchemi.distributed._core.halo_types import ParticleHaloConfig
    from nvalchemi.distributed._core.particle_halo import particle_halo_padding
    from nvalchemi.distributed.partitioner import DomainConfig, SpatialPartitioner

    n_side, lattice = 6, 3.4
    coords = torch.arange(n_side, dtype=dtype, device=device) * lattice
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    cell = torch.eye(3, dtype=dtype, device=device) * (n_side * lattice)
    pbc = torch.ones(3, dtype=torch.bool, device=device)

    domain_config = DomainConfig(cutoff=ghost_width, mesh=mesh)
    partitioner = SpatialPartitioner(
        config=domain_config, cell_matrix=cell.unsqueeze(0), pbc=pbc.unsqueeze(0)
    )
    halo_config = ParticleHaloConfig(
        ghost_width=ghost_width, partitioner=partitioner, mesh=mesh
    )
    assignment = partitioner.assign_atoms_to_ranks(positions)
    local_pos = positions[assignment == rank].contiguous()
    padded, meta = particle_halo_padding(local_pos, halo_config)
    if return_padded:
        return meta, halo_config, padded
    return meta, halo_config


def _worker_compile_halo_scatter(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29679"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        import nvalchemi.distributed  # noqa: F401
        from nvalchemi.distributed._core.shard_tensor import ShardTensor
        from nvalchemi.distributed.spec import SPEC_MPNN_HALO

        _patch_all_to_all_for_gloo()
        mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("dom",))
        meta, config = _build_rank_halo(mesh, rank, world_size)
        n_padded, n_owned, feat = meta.n_padded, meta.n_owned, 4
        assert n_padded > n_owned, "degenerate partition: no halo rows to exercise"

        torch.manual_seed(100 + rank)
        src0 = torch.randn(n_padded, feat, dtype=torch.float64)
        idx = torch.arange(n_padded).unsqueeze(-1).expand(-1, feat).contiguous()

        def make_agg() -> Any:
            with mesh:
                return ShardTensor.wrap(
                    torch.zeros(n_padded, feat, dtype=torch.float64),
                    meta=meta,
                    config=config,
                    spec=SPEC_MPNN_HALO,
                )

        def fn(agg: Any, src: torch.Tensor) -> Any:
            return agg.scatter_add_(0, idx, src)

        # Eager reference (the __torch_function__ halo handler).
        src_e = src0.clone().requires_grad_(True)
        with mesh:
            out_e = fn(make_agg(), src_e)
            energy_e = out_e.sum()
        (grad_e,) = torch.autograd.grad(energy_e, src_e)
        out_e_local = out_e.unwrap()

        # Compiled (routes scatter through __torch_dispatch__ + custom op).
        torch._dynamo.reset()
        src_c = src0.clone().requires_grad_(True)
        cf = torch.compile(fn, backend="eager", fullgraph=True)
        with mesh:
            out_c = cf(make_agg(), src_c)
            energy_c = out_c.sum()
        (grad_c,) = torch.autograd.grad(energy_c, src_c)
        out_c_local = out_c.unwrap()

        # Compiled == eager, forward AND backward.
        torch.testing.assert_close(out_c_local, out_e_local, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(grad_c, grad_e, rtol=1e-9, atol=1e-9)

        # The match must not be vacuous: the halo correction has to do real
        # cross-rank work, so the corrected output must differ from a pure-local
        # scatter and equal a manually halo-corrected reference.
        from nvalchemi.distributed._core.particle_halo import (
            halo_forward_exchange,
            halo_reverse_exchange,
        )

        pure_local = torch.zeros(n_padded, feat, dtype=torch.float64)
        pure_local.scatter_add_(0, idx, src0)
        manual = halo_forward_exchange(
            halo_reverse_exchange(pure_local, meta, config), meta, config
        )
        assert not torch.allclose(out_e_local, pure_local), (
            "halo correction is a no-op here — test would pass vacuously"
        )
        torch.testing.assert_close(out_e_local, manual, rtol=1e-9, atol=1e-9)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_gloo_available(), reason="gloo backend required"
)
def _worker_inductor_marker_lowering(rank: int, world_size: int) -> None:
    """Forward-only: the halo-correction scatter must LOWER under inductor. The
    custom op's marker indices ride as int[] constants; a real-Tensor marker
    would raise "convert all Tensors to FakeTensors" during inductor's
    fake-prop of the op node. backend="eager" does not lower, so this is the
    only guard for the int[] marker encoding."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29682"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    try:
        import nvalchemi.distributed  # noqa: F401
        from nvalchemi.distributed._core.shard_tensor import ShardTensor
        from nvalchemi.distributed.spec import SPEC_MPNN_HALO

        _patch_all_to_all_for_gloo()
        mesh = DeviceMesh("cpu", list(range(world_size)), mesh_dim_names=("dom",))
        meta, config = _build_rank_halo(mesh, rank, world_size)
        n_padded, feat = meta.n_padded, 4
        assert meta.n_padded > meta.n_owned, "degenerate partition: no halo rows"

        torch.manual_seed(100 + rank)
        src = torch.randn(n_padded, feat, dtype=torch.float64)  # forward-only: no grad
        idx = torch.arange(n_padded).unsqueeze(-1).expand(-1, feat).contiguous()

        def make_agg() -> Any:
            with mesh:
                return ShardTensor.wrap(
                    torch.zeros(n_padded, feat, dtype=torch.float64),
                    meta=meta,
                    config=config,
                    spec=SPEC_MPNN_HALO,
                )

        def fn(agg: Any, s: torch.Tensor) -> Any:
            return agg.scatter_add_(0, idx, s)

        with mesh:
            out_e = fn(make_agg(), src).unwrap()
        torch._dynamo.reset()
        cf = torch.compile(fn, backend="inductor", fullgraph=True)
        with mesh:
            out_c = cf(make_agg(), src).unwrap()  # must lower without raising
        torch.testing.assert_close(out_c, out_e, rtol=1e-9, atol=1e-9)
    finally:
        dist.destroy_process_group()


def test_compile_halo_scatter_2ranks() -> None:
    mp.spawn(_worker_compile_halo_scatter, args=(2,), nprocs=2)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="gloo backend required")
def test_compile_halo_scatter_inductor_lowering_2ranks() -> None:
    """Regression for the int[] custom-op marker encoding under inductor."""
    mp.spawn(_worker_inductor_marker_lowering, args=(2,), nprocs=2)


# ----------------------------------------------------------------------
# The compile-refresh graph pass on the NO-SUBCLASS path.
# ----------------------------------------------------------------------
#
# The test above routes a ShardTensor scatter through ``__torch_dispatch__`` under
# compile. This exercises the *other* path: a model whose ShardTensor was bridged
# to PLAIN tensors (the no-subclass bridge), so the halo correction is NOT applied
# by dispatch and must be re-inserted by the ``make_dd_halo_backend`` graph pass.
# It verifies end-to-end, on a non-degenerate 2-rank partition with the real
# ``halo_scatter_correct_static`` op + real routing, that the auto-inserted
# correction reproduces the halo-corrected reference (``halo_forward(halo_reverse
# (...))``) in forward AND backward — the same result the eager dispatch handler
# produces.


def _worker_pass_halo_refresh(rank: int, world_size: int) -> None:
    # NCCL + CUDA: the inserted ``halo_scatter_correct_static`` op uses
    # ``funcol_all_to_all_fixed`` (NCCL), unlike the gloo-patched eager exchange,
    # so this runs on real GPUs.
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29681"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        import nvalchemi.distributed  # noqa: F401
        from nvalchemi.distributed._core.particle_halo import (
            build_halo_meta_tensors,
            halo_forward_exchange,
            halo_reverse_exchange,
        )
        from nvalchemi.distributed.compile_refresh import (
            keep_routing_live,
            make_dd_halo_backend,
        )

        device = torch.device(f"cuda:{rank}")
        dtype = torch.float32
        mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("dom",))
        meta, config = _build_rank_halo(
            mesh, rank, world_size, device=device, dtype=dtype
        )
        n_padded, n_owned, feat = meta.n_padded, meta.n_owned, 4
        assert n_padded > n_owned, "degenerate partition: no halo rows to exercise"

        torch.manual_seed(100 + rank)
        # One "message" per padded row, scattered to a per-row receiver carried by
        # an edge_index input (self-edges keep the reference math simple; the point
        # is the node-scatter + halo correction, not the edge semantics).
        src0 = torch.randn(n_padded, feat, dtype=dtype, device=device)
        recv = torch.arange(n_padded, device=device)
        edge_index = torch.stack([recv, recv], dim=0)

        # Reference: pure-local scatter + the eager halo correction.
        pure = torch.zeros(n_padded, feat, dtype=dtype, device=device)
        pure.scatter_add_(0, recv.unsqueeze(-1).expand(-1, feat), src0)
        manual = halo_forward_exchange(
            halo_reverse_exchange(pure, meta, config), meta, config
        )
        assert not torch.allclose(manual, pure), (
            "halo correction is a no-op here — test would pass vacuously"
        )

        # Routing tensors the pass wires the inserted op to.
        max_send = max((max(r) for r in meta.send_sizes), default=0)
        si, rd, rr, no = build_halo_meta_tensors(
            meta, rank, max_send, n_padded, device
        )

        def fn(agg0, edge_index, src, _halo_si, _halo_rd, _halo_rr, _halo_no):
            # Anchor routing as live graph inputs (no-subclass bridge's job), then
            # a plain node-scatter keyed on edge_index. The pass inserts
            # halo_scatter_correct_static on the scatter's output.
            agg0 = keep_routing_live(agg0, _halo_si, _halo_rd, _halo_rr, _halo_no)
            r = edge_index[1].unsqueeze(-1).expand(-1, feat)
            return agg0.scatter_add(0, r, src)

        torch._dynamo.reset()
        backend = make_dd_halo_backend(world_size, "aot_eager")
        cf = torch.compile(fn, backend=backend, fullgraph=True)
        src_c = src0.clone().requires_grad_(True)
        agg0 = torch.zeros(n_padded, feat, dtype=dtype, device=device)
        out_c = cf(agg0, edge_index, src_c, si, rd, rr, no)
        (grad_c,) = torch.autograd.grad(out_c.sum(), src_c)

        # Forward: the auto-inserted correction reproduces the reference.
        torch.testing.assert_close(out_c, manual, rtol=1e-5, atol=1e-5)
        # Backward: gradient flows through the inserted op (its registered adjoint).
        assert grad_c is not None and torch.isfinite(grad_c).all()
        assert grad_c.abs().sum() > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need 2+ CUDA GPUs (static halo op uses NCCL funcol)",
)
def test_compile_pass_halo_refresh_2ranks() -> None:
    """The compile-refresh pass auto-inserts the real halo correction on a
    no-subclass plain scatter and reproduces the eager dispatch-corrected result
    (forward + backward) on a non-degenerate 2-rank partition."""
    mp.spawn(_worker_pass_halo_refresh, args=(2,), nprocs=2)


# --------------------------------------------------------------------------
# The pass on a REAL multi-layer MPNN, end-to-end.
#
# The test above exercises the mechanism on a single synthetic scatter. This
# exercises it on the actual target: a pure-PyTorch message-passing model (the
# *kind* example-04's BPModel is — gather senders -> linear -> scatter_add to
# receivers -> nonlinearity), with the pass placing the refresh automatically
# (the author writes NO distributed code). Two things the single-scatter test
# could not show:
#
#  * **It is load-bearing for OWNED outputs.** A 1-layer model (BPModel itself)
#    doesn't need the ghost refresh — owned features are complete after the local
#    scatter and the energy reduce is owned-only, so a stale ghost never reaches
#    an owned output. Only ≥2 layers make it bite: a layer-1 GHOST feature is a
#    sender into a layer-2 OWNED receiver, so a stale layer-1 ghost corrupts the
#    layer-2 owned output. This model is 2-layer and compares OWNED rows.
#  * **fullgraph is NOT required.** A clean pure-PyTorch model has no graph
#    breaks, so ``fullgraph=False`` still yields one graph with routing + both
#    scatters; the pass inserts at both. (The pass runs per-fragment regardless.)
#
# Reference = the eager dispatch correction (``halo_forward(halo_reverse)`` after
# each scatter) applied by hand; sensitivity = the same model with NO correction,
# whose owned rows must DIVERGE (else the test proves nothing).


class _TwoLayerMPNN(torch.nn.Module):
    """gather senders -> Linear -> scatter_add to receivers -> SiLU, twice.
    ``refresh`` (when given) is applied to each scatter output — this is the
    seam the eager ShardTensor dispatch fills automatically and the compile pass
    inserts; left as ``None`` under compile (the pass inserts it)."""

    def __init__(self, feat: int) -> None:
        super().__init__()
        self.lin1 = torch.nn.Linear(feat, feat, bias=False)
        self.lin2 = torch.nn.Linear(feat, feat, bias=False)

    def forward(self, x, edge_index, refresh=None):  # noqa: ANN001
        send, recv = edge_index[0], edge_index[1]
        ridx = recv.unsqueeze(-1).expand(-1, x.shape[-1])
        h = torch.zeros_like(x).scatter_add(0, ridx, self.lin1(x[send]))
        if refresh is not None:
            h = refresh(h)
        h = torch.nn.functional.silu(h)
        out = torch.zeros_like(x).scatter_add(0, ridx, self.lin2(h[send]))
        if refresh is not None:
            out = refresh(out)
        return out


def _worker_pass_two_layer_mpnn(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29682"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        import nvalchemi.distributed  # noqa: F401
        from nvalchemi.distributed._core.particle_halo import (
            build_halo_meta_tensors,
            halo_forward_exchange,
            halo_reverse_exchange,
        )
        from nvalchemi.distributed.compile_refresh import (
            keep_routing_live,
            make_dd_halo_backend,
        )

        device = torch.device(f"cuda:{rank}")
        dtype = torch.float32
        feat = 8
        mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("dom",))
        meta, config, padded = _build_rank_halo(
            mesh, rank, world_size, device=device, dtype=dtype, return_padded=True
        )
        n_padded, n_owned = meta.n_padded, meta.n_owned
        assert n_padded > n_owned, "degenerate partition: no halo rows to exercise"

        # Real neighbor graph on the padded (owned+ghost) cluster: open-boundary
        # radius graph (the halo already gathered the explicit ghost copies, so no
        # PBC here). cutoff == ghost_width -> every owned atom's neighbors are present.
        d = torch.cdist(padded, padded)
        m = (d > 1e-6) & (d < 5.0)
        edge_index = m.nonzero(as_tuple=False).T.contiguous()  # (2,E): (sender,recv)

        torch.manual_seed(0)  # identical (replicated) weights on every rank
        model = _TwoLayerMPNN(feat).to(device=device, dtype=dtype)

        # Entry-consistent features: ghost rows mirror their owner (what the
        # boundary atomic-number/embedding halo exchange produces on entry).
        x0 = halo_forward_exchange(
            torch.randn(n_padded, feat, dtype=dtype, device=device), meta, config
        )

        def _correct(t):  # the eager dispatch correction
            return halo_forward_exchange(
                halo_reverse_exchange(t, meta, config), meta, config
            )

        ref = model(x0, edge_index, refresh=_correct)        # eager, corrected
        nocorr = model(x0, edge_index, refresh=None)         # eager, NO correction
        # Load-bearing: a stale layer-1 ghost must change layer-2 OWNED outputs.
        owned_gap = (ref[:n_owned] - nocorr[:n_owned]).abs().max()
        assert owned_gap > 1e-4, (
            f"correction is a no-op on owned rows ({owned_gap:.2e}); the test would "
            "pass vacuously (degenerate partition or too-shallow model)"
        )

        # Compiled: the author's model with NO refresh; the pass inserts it. No
        # fullgraph (a clean model is one graph anyway). Routing rides as graph
        # inputs named to match the pass (``_halo_*``).
        max_send = max((max(r) for r in meta.send_sizes), default=0)
        si, rd, rr, no = build_halo_meta_tensors(meta, rank, max_send, n_padded, device)

        def fn(x, edge_index, _halo_si, _halo_rd, _halo_rr, _halo_no):
            x = keep_routing_live(x, _halo_si, _halo_rd, _halo_rr, _halo_no)
            return model(x, edge_index)  # refresh=None -> pass auto-inserts

        torch._dynamo.reset()
        backend = make_dd_halo_backend(world_size, "aot_eager", strict=False)
        cf = torch.compile(fn, backend=backend, fullgraph=False)
        xc = x0.clone().requires_grad_(True)
        out_c = cf(xc, edge_index, si, rd, rr, no)
        (grad_c,) = torch.autograd.grad(out_c[:n_owned].sum(), xc)

        # The auto-placed refresh reproduces the eager-corrected OWNED outputs...
        torch.testing.assert_close(
            out_c[:n_owned], ref[:n_owned], rtol=1e-4, atol=1e-4
        )
        # ...and NOT the uncorrected ones (so the pass genuinely inserted it).
        assert not torch.allclose(
            out_c[:n_owned], nocorr[:n_owned], rtol=1e-4, atol=1e-4
        )
        # Backward flows through the inserted op's registered adjoint.
        assert grad_c is not None and torch.isfinite(grad_c).all()
        assert grad_c.abs().sum() > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need 2+ CUDA GPUs (static halo op uses NCCL funcol)",
)
def test_compile_pass_two_layer_mpnn_owned_equivalence_2ranks() -> None:
    """A real 2-layer pure-PyTorch MPNN with NO author DD code, compiled under DD
    (``fullgraph=False``) — the compile-refresh pass auto-places the halo refresh
    so each rank's OWNED outputs match the eager dispatch-corrected reference, on
    a non-degenerate partition. Verified load-bearing: owned rows diverge from the
    uncorrected model."""
    mp.spawn(_worker_pass_two_layer_mpnn, args=(2,), nprocs=2)


# --------------------------------------------------------------------------
# The same proof, but through the framework-owned bridge (HaloCompileBridge)
# instead of an inline torch.compile. Confirms the reusable scaffolding
# (plain-ify + thread routing + pass backend + cached compile) reproduces the
# owned-output equivalence — the shared path the wrappers use.


def _worker_bridge_two_layer(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29684"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        import nvalchemi.distributed  # noqa: F401
        from nvalchemi.distributed._core.particle_halo import (
            build_halo_meta_tensors,
            halo_forward_exchange,
            halo_reverse_exchange,
        )
        from nvalchemi.distributed.compile_bridge import HaloCompileBridge

        device = torch.device(f"cuda:{rank}")
        dtype = torch.float32
        feat = 8
        mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("dom",))
        meta, config, padded = _build_rank_halo(
            mesh, rank, world_size, device=device, dtype=dtype, return_padded=True
        )
        n_padded, n_owned = meta.n_padded, meta.n_owned
        assert n_padded > n_owned, "degenerate partition: no halo rows to exercise"

        d = torch.cdist(padded, padded)
        m = (d > 1e-6) & (d < 5.0)
        edge_index = m.nonzero(as_tuple=False).T.contiguous()

        torch.manual_seed(0)
        model = _TwoLayerMPNN(feat).to(device=device, dtype=dtype)
        x0 = halo_forward_exchange(
            torch.randn(n_padded, feat, dtype=dtype, device=device), meta, config
        )

        def _correct(t):
            return halo_forward_exchange(
                halo_reverse_exchange(t, meta, config), meta, config
            )

        ref = model(x0, edge_index, refresh=_correct)
        nocorr = model(x0, edge_index, refresh=None)
        assert (ref[:n_owned] - nocorr[:n_owned]).abs().max() > 1e-4

        max_send = max((max(r) for r in meta.send_sizes), default=0)
        routing = build_halo_meta_tensors(meta, rank, max_send, n_padded, device)

        # The framework bridge: author supplies only the forward signature
        # adapter; the bridge plain-ifies, threads routing, compiles with the
        # pass. No inline keep_routing_live / make_dd_halo_backend / torch.compile.
        bridge = HaloCompileBridge(
            lambda mi: model(mi["positions"], mi["edge_index"]),
            world_size=world_size,
            refresh="pass",
            inner_backend="aot_eager",
            anchor_key="positions",
        )
        xc = x0.clone().requires_grad_(True)
        out_c = bridge({"positions": xc, "edge_index": edge_index}, routing)
        (grad_c,) = torch.autograd.grad(out_c[:n_owned].sum(), xc)

        torch.testing.assert_close(out_c[:n_owned], ref[:n_owned], rtol=1e-4, atol=1e-4)
        assert not torch.allclose(
            out_c[:n_owned], nocorr[:n_owned], rtol=1e-4, atol=1e-4
        )
        assert grad_c is not None and torch.isfinite(grad_c).all()
        assert grad_c.abs().sum() > 0
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need 2+ CUDA GPUs (static halo op uses NCCL funcol)",
)
def test_halo_compile_bridge_two_layer_owned_equivalence_2ranks() -> None:
    """The framework-owned ``HaloCompileBridge`` reproduces the inline pass
    result — a 2-layer MPNN's OWNED outputs match the eager-corrected reference
    (and diverge from uncorrected), with the bridge owning all the no-subclass
    scaffolding."""
    mp.spawn(_worker_bridge_two_layer, args=(2,), nprocs=2)


# --------------------------------------------------------------------------
# The compile-routing HOLDER path. The model self-refreshes by calling the
# compile-aware ``scatter_to_owners`` helper directly (refresh is NOT
# auto-inserted by the pass and NOT a closure). The bridge (refresh="self")
# publishes the step's routing to the framework holder *inside* the compiled
# region from the graph-input ``_halo_*`` tensors; the in-region
# ``scatter_to_owners`` reads the holder and emits the fixed-shape static op
# wired to those inputs.


def _worker_holder_self_refresh_two_layer(rank: int, world_size: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29686"
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    try:
        import nvalchemi.distributed  # noqa: F401
        from nvalchemi.distributed._core.compile_routing import get_compile_routing
        from nvalchemi.distributed._core.particle_halo import (
            build_halo_meta_tensors,
            halo_forward_exchange,
            halo_reverse_exchange,
        )
        from nvalchemi.distributed.compile_bridge import HaloCompileBridge
        from nvalchemi.distributed.helpers import scatter_to_owners

        device = torch.device(f"cuda:{rank}")
        dtype = torch.float32
        feat = 8
        mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("dom",))
        meta, config, padded = _build_rank_halo(
            mesh, rank, world_size, device=device, dtype=dtype, return_padded=True
        )
        n_padded, n_owned = meta.n_padded, meta.n_owned
        assert n_padded > n_owned, "degenerate partition: no halo rows to exercise"

        d = torch.cdist(padded, padded)
        m = (d > 1e-6) & (d < 5.0)
        edge_index = m.nonzero(as_tuple=False).T.contiguous()

        torch.manual_seed(0)
        model = _TwoLayerMPNN(feat).to(device=device, dtype=dtype)
        x0 = halo_forward_exchange(
            torch.randn(n_padded, feat, dtype=dtype, device=device), meta, config
        )

        def _correct(t):
            return halo_forward_exchange(
                halo_reverse_exchange(t, meta, config), meta, config
            )

        ref = model(x0, edge_index, refresh=_correct)
        nocorr = model(x0, edge_index, refresh=None)
        assert (ref[:n_owned] - nocorr[:n_owned]).abs().max() > 1e-4

        max_send = max((max(r) for r in meta.send_sizes), default=0)
        si, rd, rr, no = build_halo_meta_tensors(
            meta, rank, max_send, n_padded, device
        )

        # The model self-refreshes via the compile-aware helper: inside the
        # compiled region it reads the holder the bridge publishes and emits the
        # static op. No closure, no per-model hook — the framework helper +
        # holder do it. (Eager, the SAME ``scatter_to_owners`` would take its
        # context path; here it is inside compile, so the holder path fires.)
        bridge = HaloCompileBridge(
            lambda mi: model(
                mi["positions"], mi["edge_index"], refresh=scatter_to_owners
            ),
            world_size=world_size,
            refresh="self",
            inner_backend="aot_eager",
        )
        xc = x0.clone().requires_grad_(True)
        inputs = {
            "positions": xc,
            "edge_index": edge_index,
            "_halo_si": si,
            "_halo_rd": rd,
            "_halo_rr": rr,
            "_halo_no": no,
        }
        out_c = bridge(inputs)
        (grad_c,) = torch.autograd.grad(out_c[:n_owned].sum(), xc)

        torch.testing.assert_close(
            out_c[:n_owned], ref[:n_owned], rtol=1e-4, atol=1e-4
        )
        assert not torch.allclose(
            out_c[:n_owned], nocorr[:n_owned], rtol=1e-4, atol=1e-4
        )
        assert grad_c is not None and torch.isfinite(grad_c).all()
        assert grad_c.abs().sum() > 0
        # The bridge cleared the holder after the call — a subsequent eager
        # refresh must not see trace-time routing.
        assert get_compile_routing() is None
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="Need 2+ CUDA GPUs (static halo op uses NCCL funcol)",
)
def test_halo_compile_bridge_holder_self_refresh_2ranks() -> None:
    """The compile-routing holder threads the step's routing from the bridge's
    graph inputs to an in-region ``scatter_to_owners`` call, so a model that
    self-refreshes via the framework helper matches the eager dispatch-corrected
    OWNED outputs (and diverges from uncorrected). This is the generalized,
    framework-owned threading of routing through a closure cell that the declared
    refresh adapter rides."""
    mp.spawn(_worker_holder_self_refresh_two_layer, args=(2,), nprocs=2)
