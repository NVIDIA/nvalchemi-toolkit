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
"""Non-cueq MACE + ``torch.compile`` DD gate: force equivalence (via the
per-interaction halo hook) AND zero steady-state recompiles.

The non-cueq compiled-DD path is a *structurally different* compiled path from
the cueq gates (``test_compile_recompile_gate.py``,
``test_mace_cueq_compile_gate.py``): non-cueq MACE runs e3nn's
``@torch.jit.script`` ``ScriptModule`` layers, which Dynamo cannot trace
(``UnspecializedNNModuleVariable wrapped around ScriptModules unsupported``), so
the forward fragments into ~20 graphs. The halo ghost refresh therefore CANNOT
use the compile-refresh graph pass: with the forward fragmented, each
interaction's message ``scatter_add`` lands in its own tiny graph whose only
inputs are ``edge_index`` + node features (no routing) — routing threaded at the
top frame can't cross the ScriptModule breaks into those fragments, so the pass
inserts zero corrections (measured). The refresh must instead ride
:func:`~nvalchemi.models.mace._install_mace_halo_fix`, the hook wrapped *inside*
each ``interaction.forward`` that corrects at the interaction *output* (back in
the wrapper frame, where routing is a live closure cell) — sidestepping the
fragmentation entirely.

This gate guards exactly that hook:

* **Equivalence** (a fixed *jittered* geometry — a perfect lattice has ~zero net
  force, which would make the comparison vacuous): an *eager*-DD reference (same
  ``DistributedModel`` path, no compile) vs the compiled DD forward on the same
  partition — each rank's owned forces must match. The cell is **elongated**
  along the partition axis so the partition is genuinely non-degenerate
  (``atoms`` cap < total atoms): a cubic 2-rank box has a ghost layer spanning
  the half-slab, so every rank sees every atom and the refresh is a near-no-op.
  Here a disabled/broken refresh leaves stale ghosts that perturb owned forces by
  ~3e-5 — far above the tol and the ~1e-12 fp64 compile noise the working hook
  leaves. Verified sensitive: the assertion FAILS under
  ``NVALCHEMI_MACE_NO_REFRESH=1`` (refresh off) and passes with it on.
* **Recompiles** (jittered MD loop): after warmup the number of unique compiled
  graphs (``torch._dynamo.utils.counters["stats"]["unique_graphs"]``) must not
  grow. The graph count is large (per-fragment) but the framework COO
  fixed-shape caps must hold it *stable* across steps.

Requires 2+ CUDA GPUs + ``mace-torch`` installed. cuequivariance is NOT used.
"""

from __future__ import annotations

import os
import warnings

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig

WORLD_SIZE = 2
WARMUP_STEPS = 8
STEADY_STEPS = 8
JITTER = 0.06  # Å per-step RMS displacement (within cutoff+skin headroom)

_skip = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE,
    reason=f"Need {WORLD_SIZE}+ CUDA GPUs",
)


def _init_pg(rank: int, world_size: int, port: str) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = port
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


def _worker(rank: int, world_size: int, port: str, fn, *args) -> None:
    _init_pg(rank, world_size, port)
    try:
        fn(rank, world_size, *args)
    finally:
        dist.destroy_process_group()


def _build_pbc_argon(nx: int = 12, nyz: int = 3, dtype: torch.dtype = torch.float64):
    """Argon on an ELONGATED cell — long along x (the axis the 2-rank domain
    partition splits), short across y/z. This is deliberate: a *cubic* 2-rank
    box stays degenerate (the cutoff+skin ghost layer spans the half-slab, so
    every rank sees every atom and the halo correction is a near-no-op — the
    gate then can't tell a working refresh from a broken one). An elongated cell
    makes the per-rank slab (box_x/2) much wider than the ghost layer, so a real
    fraction of atoms are NOT visible to each rank (``atoms`` cap < total). Then
    stale ghosts genuinely perturb owned forces, and the equivalence assertion
    actually exercises the per-interaction halo hook. ``nyz`` ≥ 3 keeps
    the transverse cell ≥ 2·cutoff (minimum-image sane)."""
    spacing = 2 ** (1.0 / 6.0) * 3.40 * 1.05  # ~4.007 Å
    cx = torch.arange(nx, dtype=dtype) * spacing
    ct = torch.arange(nyz, dtype=dtype) * spacing
    gx, gy, gz = torch.meshgrid(cx, ct, ct, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]
    box = torch.tensor([nx, nyz, nyz], dtype=dtype) * spacing  # per-axis lengths
    atomic_numbers = torch.full((n,), 18, dtype=torch.long)
    cell = torch.diag(box)
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, cell, pbc, box


def _nocueq_gate_worker(rank: int, world_size: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nvalchemi.models.mace import MACEWrapper

    from torch._dynamo.utils import counters
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.sharded_batch import ShardedBatch

    dtype = torch.float64
    device = torch.device(f"cuda:{rank}")

    positions0, atomic_numbers, cell, pbc, box = _build_pbc_argon()
    positions0 = positions0.to(device=device, dtype=dtype)
    cell_d = cell.to(device=device, dtype=dtype).unsqueeze(0)
    pbc_d = pbc.to(device).unsqueeze(0)
    box = box.to(device=device, dtype=dtype)  # per-axis lengths for PBC wrap

    def _batch(pos):
        if rank != 0:
            return None
        data = AtomicData(
            atomic_numbers=atomic_numbers.to(device),
            positions=pos.clone(),
            cell=cell_d,
            pbc=pbc_d,
        )
        return Batch.from_data_list([data])

    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))

    # A perfect argon lattice has ~zero net force (symmetry), which would make
    # the equivalence comparison vacuous (0 == 0 regardless of refresh). Use a
    # fixed deterministic jitter so the reference geometry has real,
    # O(0.1 eV/Å) forces — then a stale ghost actually perturbs them.
    _rg = torch.Generator(device="cpu").manual_seed(3)
    ref_disp = (
        0.08 * torch.randn(positions0.shape, dtype=torch.float64, generator=_rg)
    ).to(device=device, dtype=dtype)
    positions_ref = positions0 + ref_disp
    positions_ref = positions_ref - torch.floor(positions_ref / box) * box

    # ---- eager-DD reference (same DistributedModel path, NO compile) at the
    # jittered reference geometry. Same partition -> owned forces align. ----
    eager = MACEWrapper.from_checkpoint(
        "small", device=device, dtype=dtype, enable_cueq=False
    )
    cfg = DomainConfig(cutoff=float(eager.cutoff), skin=0.5, mesh=mesh)
    with DistributedModel(eager, cfg) as eager_model:
        f_eager_owned = (
            eager_model(ShardedBatch.from_batch(_batch(positions_ref), mesh, cfg, 0))[
                "forces"
            ]
            .detach()
            .double()
            .cpu()
        )
    del eager, eager_model

    # ---- compiled DD model (non-cueq -> graph breaks -> hook-based refresh).
    # The wrapper is eager (compile_model is the single-process model-compile
    # lever); DD-compile is requested on DistributedModel, which owns the compiled
    # energy-autograd forward. ----
    wrapper = MACEWrapper.from_checkpoint(
        "small", device=device, dtype=dtype, enable_cueq=False
    )
    _cp = wrapper.distribution_spec().compile
    assert _cp is not None and _cp.forces_via_autograd, (
        "non-cueq MACE must declare the framework energy-autograd force strategy "
        "(CompilePolicy.force_strategy) that drives the COO caps path; the "
        "wrapper's spec carries NO compile switch — DD-compile is owned by "
        "DistributedModel(compile=True), not the wrapper"
    )
    assert "atomic_energies" in wrapper.model_config.outputs, (
        "MACE must expose atomic_energies (per-node energy) as a normal output — "
        "the lever the framework uses to drive the energy-only DD forward"
    )

    gen = torch.Generator(device="cpu").manual_seed(20)
    graphs_after_warmup = [0]

    with DistributedModel(wrapper, cfg, compile=True) as dist_model:
        for step in range(WARMUP_STEPS + STEADY_STEPS):
            if step == 0:
                pos = positions_ref  # same jittered geom as the eager reference
            else:
                disp = JITTER * torch.randn(
                    positions0.shape, dtype=torch.float64, generator=gen
                ).to(device=device, dtype=dtype)
                pos = positions0 + disp
                pos = pos - torch.floor(pos / box) * box
            sharded = ShardedBatch.from_batch(_batch(pos), mesh, cfg, 0)
            out = dist_model(sharded)
            f_owned = out["forces"].detach().double()

            if step == 0:
                # Equivalence at the jittered reference geom: compiled DD (the
                # per-interaction halo hook) == eager DD on the same partition.
                # The system is non-degenerate (``atoms`` cap < total atoms), so
                # a missing/broken refresh leaves stale ghosts that perturb owned
                # forces by ~2.5e-5 (measured) — far above this tol, and far
                # above the ~1e-12 fp64 compile-reorder noise the hook leaves. So
                # this assertion genuinely exercises the refresh (verified: it
                # fails under NVALCHEMI_MACE_NO_REFRESH=1).
                _d = (f_owned.cpu() - f_eager_owned).abs()
                print(
                    f"[r{rank}] STEP0 max_abs_diff={_d.max().item():.3e} "
                    f"fmax={f_eager_owned.abs().max().item():.3e}",
                    flush=True,
                )
                torch.testing.assert_close(
                    f_owned.cpu(), f_eager_owned, rtol=1e-6, atol=1e-7,
                    msg=f"rank {rank}: compiled non-cueq DD forces != eager DD",
                )
            _ = f_owned.sum().item()
            n_graphs = counters["stats"].get("unique_graphs", 0)
            if step == WARMUP_STEPS - 1:
                graphs_after_warmup[0] = n_graphs
            print(
                f"[r{rank}] step {step:02d} unique_graphs={n_graphs} "
                f"caps={getattr(dist_model, '_cap_state', {})}",
                flush=True,
            )

    final_graphs = counters["stats"].get("unique_graphs", 0)
    new_recompiles = final_graphs - graphs_after_warmup[0]
    assert new_recompiles == 0, (
        f"rank {rank}: {new_recompiles} recompile(s) during {STEADY_STEPS} "
        f"steady-state steps (unique_graphs {graphs_after_warmup[0]} -> "
        f"{final_graphs}); caps={getattr(dist_model, '_cap_state', {})}. The "
        "framework COO fixed-shape padding is no longer holding the (fragmented) "
        "non-cueq compiled graph stable."
    )


@_skip
def test_mace_nocueq_compile_dd_equivalence_and_zero_recompiles_2ranks():
    """Non-cueq MACE halo + compile under DD: owned forces match an eager-DD
    reference (guarding the closure-cell interaction hook that survives the
    e3nn ScriptModule graph breaks), and a jittered MD loop produces zero
    steady-state recompiles."""
    pytest.importorskip("mace", reason="mace-torch not installed")
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29574", _nocueq_gate_worker),
        nprocs=WORLD_SIZE,
    )
