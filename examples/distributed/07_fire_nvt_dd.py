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

"""
FIRE relaxation → NVT MD, both domain-decomposed on one mesh
============================================================

A two-stage workflow — geometry relaxation with FIRE, then NVT Langevin
molecular dynamics — where **each stage runs the model under**
:class:`~nvalchemi.distributed.DomainParallel` **across the same set of GPUs**.

This is deliberately *not* a :class:`~nvalchemi.distributed.DistributedPipeline`.
A pipeline maps one rank per stage (stage A on GPU 0, stage B on GPU 1, …) and
streams systems through the stages; it cannot host a *domain-decomposed* stage,
because a DD stage needs a whole sub-mesh of ranks cooperating on one system.
Here both stages want the full 2-GPU mesh, and they run **sequentially in time**
on it: FIRE relaxes the batch across both GPUs, then the relaxed configuration is
handed to an NVT integrator that runs across the same two GPUs.

The seam between the stages is deliberately simple and framework-native:

#. ``DomainParallel(FIRE).partition(full_batch)`` scatters the system, ``run``
   relaxes it, and ``gather`` reconstructs the full relaxed batch on rank 0.
#. That relaxed batch (with velocities zeroed and re-sampled to the target
   temperature) is fed to ``DomainParallel(NVT).partition`` for the MD leg.

Both legs are correct under DD without any distributed-aware code in the model or
the integrators: the FIRE velocity mixing is globalized by the dynamics
coordinator (it reduces the ``v·f`` / ``v·v`` / ``f·f`` power/norm scalars over
the mesh), exactly as NVT's kinetic-energy / DOF thermostat coupling is.

System: alpha-quartz SiO2 supercell, periodic on all axes.

.. note::

    Requires 2 GPUs. Run with::

        torchrun --nproc_per_node=2 examples/distributed/07_fire_nvt_dd.py

    For MACE + cuEquivariance across ranks, set the JIT-race guard::

        CUEQUIVARIANCE_OPS_PARALLEL_COMPILE=0 \\
            torchrun --nproc_per_node=2 \\
            examples/distributed/07_fire_nvt_dd.py

Outputs a relaxed-then-NVT trajectory at ``./fire_nvt_dd_trajectory.xyz``
(rank 0 only).
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from loguru import logger

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed import DomainConfig, DomainParallel, HookScope
from nvalchemi.dynamics import HostMemory, NVTLangevin
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import SnapshotHook
from nvalchemi.dynamics.optimizers.fire import FIRE
from nvalchemi.hooks import NeighborListHook

# Skip the heavy distributed launch during the Sphinx-Gallery docs build (it has
# no torchrun environment), mirroring the other distributed examples.
_DOCS_BUILD = os.environ.get("NVALCHEMI_SPHINX_BUILD") == "1"
_DISTRIBUTED_ENV = "RANK" in os.environ and "WORLD_SIZE" in os.environ

# Reuse the SiO2 supercell builder shared across the distributed examples.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmark" / "distributed")
)
from _benchmark_common import build_sio2_supercell  # noqa: E402

# ----------------------------------------------------------------------
# System construction (rank 0 — DomainParallel scatters from there)
# ----------------------------------------------------------------------


def build_initial_batch(
    repeats: tuple[int, int, int], dtype: torch.dtype, device: torch.device
) -> Batch:
    # Perturb the lattice a little (seed=1) so FIRE has something to relax.
    pos, numbers, masses, cell, _velocities = build_sio2_supercell(
        repeats=repeats, dtype=dtype, seed=1
    )
    data = AtomicData(
        positions=pos.to(device),
        atomic_numbers=numbers.to(device),
        atomic_masses=masses.to(device),
        cell=cell.to(device).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]], device=device),
    )
    # FIRE integrates a fictitious velocity from rest.
    data.add_node_property("velocities", torch.zeros_like(pos).to(device))
    return Batch.from_data_list([data], device=device)


def sample_maxwell_boltzmann(
    batch: Batch, temperature_k: float, seed: int
) -> None:
    """Draw velocities from a Maxwell-Boltzmann distribution at
    ``temperature_k`` and zero the net linear momentum, in place."""
    from nvalchemi.dynamics.hooks._utils import KB_EV

    gen = torch.Generator(device="cpu").manual_seed(seed)
    masses = batch.atomic_masses.detach().cpu()
    sigma = torch.sqrt(KB_EV * temperature_k / masses).unsqueeze(-1)
    vel = torch.randn(masses.shape[0], 3, generator=gen) * sigma
    vel = vel - vel.mean(dim=0, keepdim=True)
    batch.velocities = vel.to(batch.positions.device, batch.positions.dtype)


# ----------------------------------------------------------------------
# Trajectory persistence (rank 0 only)
# ----------------------------------------------------------------------


def write_trajectory_xyz(sink: HostMemory, path: Path) -> int:
    """Decode the :class:`HostMemory` sink into per-frame :class:`ase.Atoms`
    and write an extxyz trajectory. Returns the number of frames written."""
    from ase import Atoms
    from ase.io import write as ase_write

    trajectory_batch = sink.read()
    n_frames = trajectory_batch.num_graphs
    if path.exists():
        path.unlink()
    for frame in range(n_frames):
        single = trajectory_batch.index_select(torch.tensor([frame]))
        cell = single.cell
        if cell.dim() == 3:
            cell = cell.squeeze(0)
        atoms = Atoms(
            numbers=single.atomic_numbers.detach().cpu().numpy(),
            positions=single.positions.detach().cpu().numpy(),
            cell=cell.detach().cpu().numpy(),
            pbc=True,
        )
        atoms.info["frame"] = frame
        ase_write(str(path), atoms, format="extxyz", append=True)
    return n_frames


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FIRE relax → NVT MD, both under DomainParallel on one mesh."
    )
    parser.add_argument("--checkpoint", default="medium-0b2")
    parser.add_argument("--repeats", type=int, nargs=3, default=[3, 3, 3])
    parser.add_argument("--fire-steps", type=int, default=100)
    parser.add_argument("--nvt-steps", type=int, default=200)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument("--fire-dt", type=float, default=1.0)
    parser.add_argument("--nvt-dt-fs", type=float, default=0.5)
    parser.add_argument("--friction", type=float, default=0.01)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument(
        "--output-xyz", type=Path, default=Path("fire_nvt_dd_trajectory.xyz")
    )
    parser.add_argument(
        "--backend", default="nccl", help="Use 'gloo' on CPU-only envs."
    )
    args = parser.parse_args()

    if _DOCS_BUILD or not _DISTRIBUTED_ENV:
        logger.info(
            "Not running under torchrun — skipping the distributed run. "
            "Launch with: torchrun --nproc_per_node=2 "
            "examples/distributed/07_fire_nvt_dd.py"
        )
        return

    # ----- Process group + device -----
    dist.init_process_group(backend=args.backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if args.backend == "gloo":
        from nvalchemi.distributed.validate.worker import (
            _patch_physicsnemo_all_to_all_for_gloo,
        )

        _patch_physicsnemo_all_to_all_for_gloo()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        device_index = local_rank % torch.cuda.device_count()
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")

    if rank == 0:
        logger.info(
            "FIRE→NVT DD: world_size={ws} device={dev} checkpoint={ckpt} "
            "repeats={r} fire_steps={fs} nvt_steps={ns} T={T}K",
            ws=world_size, dev=device, ckpt=args.checkpoint,
            r=tuple(args.repeats), fs=args.fire_steps, ns=args.nvt_steps,
            T=args.temperature_k,
        )

    # ----- One DeviceMesh shared by BOTH DD stages -----
    from torch.distributed.device_mesh import DeviceMesh

    backend_override = (("gloo", None),) if args.backend == "gloo" else None
    mesh = DeviceMesh(
        device.type,
        list(range(world_size)),
        mesh_dim_names=("domain",),
        backend_override=backend_override,
    )

    # ----- Model (shared by both stages) -----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nvalchemi.models.mace import MACEWrapper

    dtype = torch.float32
    wrapper = MACEWrapper.from_checkpoint(
        args.checkpoint, dtype=dtype, device=device
    ).eval()
    domain_cfg = DomainConfig(cutoff=float(wrapper.cutoff), skin=0.5, mesh=mesh)

    def _nl_hook() -> NeighborListHook:
        # Fresh hook per stage — each integrator owns its own BEFORE_COMPUTE NL.
        return NeighborListHook(
            wrapper.model_config.neighbor_config,
            skin=0.5,
            stage=DynamicsStage.BEFORE_COMPUTE,
        )

    # ==================================================================
    # STAGE 1 — FIRE relaxation across the full mesh
    # ==================================================================
    fire = FIRE(
        model=wrapper,
        dt=args.fire_dt,
        hooks=[_nl_hook()],
        n_steps=args.fire_steps,
    )
    fire_dd = DomainParallel(dynamics=fire, config=domain_cfg, n_steps=args.fire_steps)

    initial_batch = (
        build_initial_batch(tuple(args.repeats), dtype=dtype, device=device)
        if rank == 0
        else None
    )
    fire_owned = fire_dd.partition(initial_batch)
    if rank == 0:
        logger.info("Stage 1 (FIRE): relaxing across {ws} ranks…", ws=world_size)
    fire_dd.run(fire_owned)

    # Reconstruct the full relaxed system on rank 0; the coordinator globalizes
    # the FIRE power/norm reductions, so the relaxed config is mesh-consistent.
    relaxed_full = fire_dd.gather(fire_owned, dst=0)
    fire_dd.close()

    # ==================================================================
    # STAGE 2 — NVT Langevin MD on the SAME mesh, seeded from the relaxed batch
    # ==================================================================
    if rank == 0:
        # Give the relaxed atoms a thermal velocity distribution for the MD leg.
        sample_maxwell_boltzmann(relaxed_full, args.temperature_k, seed=0)
        nvt_initial = relaxed_full
    else:
        nvt_initial = None

    n_frames_expected = (args.nvt_steps // args.snapshot_every) + 1
    trajectory_sink = HostMemory(capacity=n_frames_expected)
    snapshot_hook = SnapshotHook(sink=trajectory_sink, frequency=args.snapshot_every)
    # RANK_ZERO: gather the FULL system onto rank 0 so each frame has every atom.
    snapshot_hook.scope = HookScope.RANK_ZERO

    nvt = NVTLangevin(
        model=wrapper,
        dt=args.nvt_dt_fs,
        temperature=args.temperature_k,
        friction=args.friction,
        hooks=[_nl_hook()],
        n_steps=args.nvt_steps,
    )
    nvt_dd = DomainParallel(
        dynamics=nvt,
        config=domain_cfg,
        n_steps=args.nvt_steps,
        hooks=[snapshot_hook],
    )

    nvt_owned = nvt_dd.partition(nvt_initial)
    if rank == 0:
        logger.info("Stage 2 (NVT): running MD across {ws} ranks…", ws=world_size)
    nvt_dd.run(nvt_owned)

    # ----- Persist + cleanup -----
    if rank == 0:
        n_frames = write_trajectory_xyz(trajectory_sink, args.output_xyz)
        logger.info(
            "Done. FIRE→NVT complete; wrote {f} xyz frames to {p}.",
            f=n_frames, p=args.output_xyz,
        )

    nvt_dd.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
