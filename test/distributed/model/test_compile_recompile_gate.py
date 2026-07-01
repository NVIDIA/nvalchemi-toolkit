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
"""Steady-state recompile gate for compiled domain-decomposed MD.

The fixed-shape padding mechanism (``DistributedModel`` caps via
:func:`~nvalchemi.distributed.graph_padder.resolve_cap` +
``ShardedBatch.pad_padded_view_to_caps`` / per-model padders) exists for
exactly one reason: under ``torch.compile`` a DD model's graph must keep a
**stable shape** across MD steps, or every step that changes the owned+ghost
atom / edge count triggers a recompile (and, under DD, an asymmetric-recompile
NCCL desync). This file gates the recompile COUNT so a caps/padding change that
re-introduces recompiles is caught by the suite.

The gate runs a real compiled DD forward in a multi-step MD-like loop with
per-step position jitter (which fluctuates the halo ghost count and the edge
count exactly as an integrator would), then asserts that after a warmup — during
which the grow-on-overflow caps settle and the graph compiles — the number of
unique compiled graphs (``torch._dynamo.utils.counters["stats"]["unique_graphs"]``)
does **not** increase across the steady-state steps.

Exercises the *framework* caps path (``DistributedModel._call_halo_storage``
gated on the wrapper's ``_compiled_energy_only`` flag → COO ``edge_index``
padding), which MACE drives. cueq is the working compiled MACE path on GPU.

Requires:
* 2+ CUDA GPUs.
* ``cuequivariance`` / ``cuequivariance_torch`` + ``mace-torch`` installed.
"""

from __future__ import annotations

import os

# Serialize cueq's first-call JIT compile across ranks (see the cueq compile
# gate for the full rationale / upstream issue).
os.environ.setdefault("CUEQUIVARIANCE_OPS_PARALLEL_COMPILE", "0")

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
JITTER = 0.08  # Å per-step RMS displacement (well within cutoff+skin headroom)

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


def _build_pbc_argon(n_per_side: int = 4, dtype: torch.dtype = torch.float64):
    spacing = 2 ** (1.0 / 6.0) * 3.40 * 1.05  # ~4.007 Å
    coords = torch.arange(n_per_side, dtype=dtype) * spacing
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]
    box = n_per_side * spacing
    atomic_numbers = torch.full((n,), 18, dtype=torch.long)
    masses = torch.full((n,), 39.948, dtype=dtype)
    cell = torch.eye(3, dtype=dtype) * box
    pbc = torch.ones(3, dtype=torch.bool)
    return positions, atomic_numbers, masses, cell, pbc, box


def _recompile_gate_worker(rank: int, world_size: int) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nvalchemi.models.mace import MACEWrapper

    from torch._dynamo.utils import counters
    from torch.distributed import DeviceMesh

    from nvalchemi.distributed.distributed_model import DistributedModel
    from nvalchemi.distributed.sharded_batch import ShardedBatch

    dtype = torch.float32
    device = torch.device(f"cuda:{rank}")

    positions0, atomic_numbers, masses, cell, pbc, box = _build_pbc_argon(n_per_side=4)
    positions0 = positions0.to(device=device, dtype=dtype)
    cell_d = cell.to(device=device, dtype=dtype).unsqueeze(0)
    pbc_d = pbc.to(device).unsqueeze(0)

    # Eager wrapper (compile_model is the single-process model-compile lever);
    # DD-compile is requested on DistributedModel below.
    wrapper = MACEWrapper.from_checkpoint(
        "small", device=device, dtype=dtype, enable_cueq=True
    )
    _cp = wrapper.distribution_spec().compile
    assert _cp is not None and _cp.forces_via_autograd, (
        "this gate must exercise the framework caps path — MACE must declare a "
        "framework energy-autograd CompilePolicy (force_strategy); the spec "
        "carries no compile switch — DistributedModel(compile=True) owns it"
    )
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    cfg = DomainConfig(cutoff=float(wrapper.cutoff), skin=0.5, mesh=mesh)

    # Deterministic per-step jitter, computed only on rank 0 (which owns the
    # full batch); the scatter makes every rank see the same geometry.
    gen = torch.Generator(device="cpu").manual_seed(1234)

    def _batch_for_step() -> Batch | None:
        if rank != 0:
            return None
        disp = JITTER * torch.randn(
            positions0.shape, dtype=torch.float64, generator=gen
        ).to(device=device, dtype=dtype)
        pos = positions0 + disp
        pos = pos - torch.floor(pos / box) * box  # wrap into the cell
        data = AtomicData(
            atomic_numbers=atomic_numbers.to(device),
            positions=pos.clone(),
            atomic_masses=masses.to(device=device, dtype=dtype),
            cell=cell_d,
            pbc=pbc_d,
        )
        return Batch.from_data_list([data])

    graphs_after_warmup = [0]

    with DistributedModel(wrapper, cfg, compile=True) as dist_model:
        for step in range(WARMUP_STEPS + STEADY_STEPS):
            sharded = ShardedBatch.from_batch(
                batch=_batch_for_step(), mesh=mesh, config=cfg, src=0
            )
            out = dist_model(sharded)
            # Touch forces so the autograd graph is exercised every step.
            _ = out["forces"].sum().item()
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
        f"{final_graphs}); caps={getattr(dist_model, '_cap_state', {})}. "
        "Fixed-shape padding is no longer holding the compiled graph stable."
    )


@_skip
def test_compile_dd_zero_steady_state_recompiles_2ranks():
    """A compiled DD MACE forward run as a jittered MD loop must compile
    its graph during warmup and then hold it: zero new unique graphs across
    the steady-state steps. Guards the fixed-shape padding mechanism that the
    GraphPadder/caps consolidation refactors."""
    pytest.importorskip("mace", reason="mace-torch not installed")
    pytest.importorskip("cuequivariance", reason="cuequivariance not installed")
    mp.spawn(
        _worker,
        args=(WORLD_SIZE, "29572", _recompile_gate_worker),
        nprocs=WORLD_SIZE,
    )
