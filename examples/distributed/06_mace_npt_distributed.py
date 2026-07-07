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
MACE NPT: domain-decomposed constant-pressure MD with a barostat
=================================================================

The constant-pressure sibling of example 03. Load a MACE checkpoint, run
a short :class:`~nvalchemi.dynamics.NPT` trajectory across multiple ranks
under :class:`~nvalchemi.distributed.DomainParallel`, and record the
trajectory — including the **evolving cell** — to an xyz file from rank 0.

Why NPT-under-DD needs more than NVT: a barostat and a thermostat both
couple to *global* thermodynamic quantities — the total kinetic energy,
the total degrees of freedom, and the full-system pressure tensor — none
of which any single rank can see from its owned subdomain alone. The
framework handles this transparently: :class:`~nvalchemi.dynamics.NPT`
declares ``__dd_thermo_kind__ = "npt"``, and on ``partition()``
:class:`~nvalchemi.distributed.DomainParallel` installs a dynamics
coordinator that

* all-reduces the per-rank kinetic energy and kinetic-pressure tensor into
  mesh-global values (the consolidated virial is already global),
* replaces the integrator's per-shard degrees-of-freedom with the global
  count (and rescales the Nosé–Hoover chain masses accordingly), and
* keeps the replicated barostat state and the cell **byte-identical**
  across ranks by broadcasting them from rank 0 each step.

So the wrapper and the integrator stay ensemble-correct; the only user
change from example 03 is asking the model for ``stress`` (the barostat
needs the virial) and swapping ``NVTLangevin`` for ``NPT``.

System: alpha-quartz SiO2 supercell, isotropic barostat at zero external
pressure — the box relaxes toward its equilibrium volume while the
thermostat holds temperature.

.. note::

    Run with::

        torchrun --nproc_per_node=2 examples/distributed/06_mace_npt_distributed.py

    For multi-GPU MACE+cuEquivariance, set the env var below to avoid a
    JIT-compilation race across ranks::

        CUEQUIVARIANCE_OPS_PARALLEL_COMPILE=0 \\
            torchrun --nproc_per_node=N \\
            examples/distributed/06_mace_npt_distributed.py

Output xyz file at ``./mace_npt_trajectory.xyz`` (rank 0 only); each frame
carries the cell at that step, so the volume relaxation is visible in
OVITO/VMD.
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
from nvalchemi.dynamics import NPT, HostMemory
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import SnapshotHook
from nvalchemi.hooks import NeighborListHook

# Skip the heavy distributed launch during the Sphinx-Gallery docs build (it has
# no torchrun environment), mirroring examples 01-03.
_DOCS_BUILD = os.environ.get("NVALCHEMI_SPHINX_BUILD") == "1"
_DISTRIBUTED_ENV = "RANK" in os.environ and "WORLD_SIZE" in os.environ

# Reuse the SiO2 supercell builder from the benchmark suite (see example 03).
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
    pos, numbers, masses, cell, velocities = build_sio2_supercell(
        repeats=repeats, dtype=dtype, seed=0
    )
    data = AtomicData(
        positions=pos.to(device),
        atomic_numbers=numbers.to(device),
        atomic_masses=masses.to(device),
        cell=cell.to(device).unsqueeze(0),
        pbc=torch.tensor([[True, True, True]], device=device),
    )
    data.add_node_property("velocities", velocities.to(device))
    # NPT reads a stress tensor each step; pre-allocate so the field exists
    # before the first forward (the model overwrites it).
    data["stress"] = torch.zeros(1, 3, 3, dtype=dtype, device=device)
    return Batch.from_data_list([data], device=device)


def cell_volume(batch: Batch) -> float:
    """Volume (Å³) of a single-system batch's (replicated) cell."""
    cell = batch.cell
    if cell.dim() == 3:
        cell = cell[0]
    return float(torch.linalg.det(cell).abs())


# ----------------------------------------------------------------------
# Trajectory persistence (rank 0 only)
# ----------------------------------------------------------------------


def write_trajectory_xyz(sink: HostMemory, path: Path) -> int:
    """Decode the :class:`HostMemory` sink into per-frame
    :class:`ase.Atoms` (with the step's cell) and write an extxyz
    trajectory. Returns the number of frames written.
    """
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
        description="MACE NPT (barostat) under DomainParallel."
    )
    parser.add_argument(
        "--checkpoint",
        default="medium-0b2",
        help="MACE foundation model checkpoint name. "
        "Default fetches MACE-MP-0b2 from HuggingFace.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        nargs=3,
        default=[3, 3, 3],
        help="SiO2 unit-cell repeats along (a, b, c). 3x3x3 → 243 atoms.",
    )
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument(
        "--pressure",
        type=float,
        default=0.0,
        help="Target external pressure in eV/Å³ (1 bar ≈ 6.32e-7). "
        "Default 0 → relax toward the equilibrium volume.",
    )
    parser.add_argument(
        "--dt-fs", type=float, default=1.0, help="MD timestep in femtoseconds."
    )
    parser.add_argument(
        "--barostat-time-fs",
        type=float,
        default=1000.0,
        help="Barostat coupling time τ_P (fs). Larger = gentler cell motion.",
    )
    parser.add_argument(
        "--thermostat-time-fs",
        type=float,
        default=100.0,
        help="Thermostat coupling time τ_T (fs).",
    )
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=10,
        help="Persist a frame to the trajectory sink every N steps.",
    )
    parser.add_argument(
        "--output-xyz",
        type=Path,
        default=Path("mace_npt_trajectory.xyz"),
        help="xyz file path (rank 0 only).",
    )
    parser.add_argument(
        "--backend",
        default="nccl",
        help="Process-group backend. Use 'gloo' on CPU-only envs.",
    )
    args = parser.parse_args()

    # Docs build / no torchrun: no process group to join, so skip the launch.
    if _DOCS_BUILD or not _DISTRIBUTED_ENV:
        logger.info(
            "Not running under torchrun — skipping the distributed run. "
            "Launch with: torchrun --nproc_per_node=N "
            "examples/distributed/06_mace_npt_distributed.py"
        )
        return

    # ----- Process group setup (identical to example 03) -----
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
            "MACE NPT distributed: world_size={ws} device={dev} "
            "checkpoint={ckpt} repeats={r} n_steps={n} T={T}K "
            "P={P} eV/Å³ dt={dt}fs",
            ws=world_size,
            dev=device,
            ckpt=args.checkpoint,
            r=tuple(args.repeats),
            n=args.n_steps,
            T=args.temperature_k,
            P=args.pressure,
            dt=args.dt_fs,
        )

    # ----- DeviceMesh for DomainParallel (identical to example 03) -----
    from torch.distributed.device_mesh import DeviceMesh

    backend_override = (("gloo", None),) if args.backend == "gloo" else None
    mesh = DeviceMesh(
        device.type,
        list(range(world_size)),
        mesh_dim_names=("domain",),
        backend_override=backend_override,
    )

    # ----- Load MACE wrapper -----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nvalchemi.models.mace import MACEWrapper

    dtype = torch.float32
    wrapper = MACEWrapper.from_checkpoint(
        args.checkpoint, dtype=dtype, device=device
    ).eval()
    # The barostat needs the virial → ask MACE for stress (example 03 only
    # needed energy + forces). Everything else is unchanged.
    wrapper.set_config("active_outputs", {"energy", "forces", "stress"})
    if rank == 0:
        logger.info("MACE wrapper ready: cutoff={c} Å", c=wrapper.cutoff)

    # ----- Domain config -----
    domain_cfg = DomainConfig(cutoff=float(wrapper.cutoff), skin=0.5, mesh=mesh)

    # ----- Hooks -----
    nl_hook = NeighborListHook(
        wrapper.model_config.neighbor_config,
        skin=0.5,
        stage=DynamicsStage.BEFORE_COMPUTE,
    )
    n_frames_expected = (args.n_steps // args.snapshot_every) + 1
    trajectory_sink = HostMemory(capacity=n_frames_expected)
    snapshot_hook = SnapshotHook(
        sink=trajectory_sink,
        frequency=args.snapshot_every,
    )
    # RANK_ZERO scope: gather the full system (with the current cell) onto
    # rank 0 so each frame is the whole box, not one rank's shard.
    snapshot_hook.scope = HookScope.RANK_ZERO

    # ----- Inner integrator: NPT -----
    # NPT couples a Nosé–Hoover thermostat chain to the particle velocities
    # and a barostat to the cell. Under DomainParallel the dynamics
    # coordinator globalises the kinetic energy / DOF / pressure tensor and
    # broadcasts the replicated barostat state + cell each step (see the
    # module docstring), so this stays ensemble-correct across ranks.
    integrator = NPT(
        model=wrapper,
        dt=args.dt_fs,
        temperature=args.temperature_k,
        pressure=args.pressure,
        barostat_time=args.barostat_time_fs,
        thermostat_time=args.thermostat_time_fs,
        pressure_coupling="isotropic",
        chain_length=3,
        hooks=[nl_hook],
        n_steps=args.n_steps,
    )

    # ----- DomainParallel wrapping (SnapshotHook on the outer, as in 03) -----
    dynamics = DomainParallel(
        dynamics=integrator,
        config=domain_cfg,
        n_steps=args.n_steps,
        hooks=[snapshot_hook],
    )

    # ----- Build the initial batch on rank 0 and partition -----
    initial_batch = (
        build_initial_batch(tuple(args.repeats), dtype=dtype, device=device)
        if rank == 0
        else None
    )
    v0 = cell_volume(initial_batch) if rank == 0 else None
    owned_batch = dynamics.partition(initial_batch)
    if rank == 0:
        logger.info(
            "Partitioned: n_owned (rank 0) = {n} of {tot} atoms; V0 = {v:.2f} Å³",
            n=int(owned_batch.positions.shape[0]),
            tot=int(initial_batch.positions.shape[0]),
            v=v0,
        )

    # ----- Run the trajectory -----
    final_batch = dynamics.run(owned_batch)

    # ----- Report volume relaxation + persist trajectory + cleanup -----
    # The cell is replicated (broadcast from rank 0 each step), so any rank's
    # final cell is the global cell; report it from rank 0.
    if rank == 0:
        v1 = cell_volume(final_batch)
        logger.info(
            "Volume: {v0:.2f} → {v1:.2f} Å³ ({pct:+.2f}%) over {n} steps",
            v0=v0,
            v1=v1,
            pct=100.0 * (v1 - v0) / v0,
            n=args.n_steps,
        )
        n_frames = write_trajectory_xyz(trajectory_sink, args.output_xyz)
        logger.info("Done. Wrote {f} xyz frames to {p}.", f=n_frames, p=args.output_xyz)

    dynamics.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
