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
MACE NVT Langevin: domain-decomposed MD with xyz snapshot logging
==================================================================

End-to-end distributed MD: load a MACE foundation-model checkpoint, run
a short :class:`~nvalchemi.dynamics.NVTLangevin` trajectory across
multiple ranks under :class:`~nvalchemi.distributed.DomainParallel`, and
record the trajectory to an xyz file from rank 0.

The example is the canonical distributed pattern in miniature:

* The wrapper is stock — :class:`~nvalchemi.models.mace.MACEWrapper`
  with no distributed-aware code at the user layer.
* :class:`~nvalchemi.hooks.NeighborListHook` rebuilds the neighbour
  list each step on the halo-padded batch (the framework arranges
  halo padding before the hook fires).
* :class:`~nvalchemi.dynamics.hooks.SnapshotHook` writes the per-step
  state into a :class:`~nvalchemi.dynamics.HostMemory` sink — the
  rank-0 launcher post-processes that into an xyz file with ASE.
* :meth:`~nvalchemi.distributed.DomainParallel.run` is the single
  entry point for the trajectory loop. No hand-rolled per-step
  callbacks; the hook system observes/persists state.

System: alpha-quartz SiO2 (Si + 2 O × N) supercell at 300 K. Periodic
along all three axes; the spatial partitioner splits along the largest
box dimensions to minimise halo transfer.

.. note::

    Run with::

        torchrun --nproc_per_node=2 examples/distributed/03_mace_nvt_distributed.py

    For multi-GPU MACE+cuEquivariance, set the env var below to avoid a
    JIT-compilation race across ranks::

        CUEQUIVARIANCE_OPS_PARALLEL_COMPILE=0 \\
            torchrun --nproc_per_node=N \\
            examples/distributed/03_mace_nvt_distributed.py

Output xyz file at ``./mace_nvt_trajectory.xyz`` (rank 0 only). Reads
cleanly in OVITO and VMD.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import torch
from loguru import logger

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed import DomainConfig, DomainParallel, HookScope
from nvalchemi.dynamics import HostMemory, NVTLangevin
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import SnapshotHook
from nvalchemi.hooks import NeighborListHook

# Skip the heavy distributed launch during the Sphinx-Gallery docs build (it has
# no torchrun environment), mirroring examples 01 and 02.
_DOCS_BUILD = os.environ.get("NVALCHEMI_SPHINX_BUILD") == "1"
_DISTRIBUTED_ENV = "RANK" in os.environ and "WORLD_SIZE" in os.environ

# Reuse the SiO2 supercell builder from the benchmark suite — one canonical
# periodic test system across the distributed examples. The shared helper lives
# under ``benchmark/distributed`` (repo_root/benchmark/distributed), so add THAT
# to the path, not the example's own directory.
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
    return Batch.from_data_list([data], device=device)


# ----------------------------------------------------------------------
# Trajectory persistence (rank 0 only)
# ----------------------------------------------------------------------


def write_trajectory_xyz(sink: HostMemory, path: Path) -> int:
    """Decode the :class:`HostMemory` sink into per-frame
    :class:`ase.Atoms` and write an extxyz trajectory.

    Returns the number of frames written.
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
        description="MACE NVT Langevin under DomainParallel."
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
        "--dt-fs", type=float, default=0.5, help="MD timestep in femtoseconds."
    )
    parser.add_argument(
        "--friction",
        type=float,
        default=0.01,
        help="Langevin friction coefficient in 1/fs.",
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
        default=Path("mace_nvt_trajectory.xyz"),
        help="xyz file path (rank 0 only).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Model / simulation dtype.",
    )
    args = parser.parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    # Docs build / no torchrun: there is no process group to join, so skip the
    # launch instead of failing in init_process_group (guard matches examples
    # 01 and 02).
    if _DOCS_BUILD or not _DISTRIBUTED_ENV:
        logger.info(
            "Not running under torchrun — skipping the distributed run. "
            "Launch with: torchrun --nproc_per_node=N "
            "examples/distributed/03_mace_nvt_distributed.py"
        )
        return

    # ----- Distributed bootstrap via PhysicsNeMo's DistributedManager -----
    # ``initialize()`` reads the ``torchrun`` env (RANK / WORLD_SIZE / LOCAL_RANK),
    # inits the process group, and binds this rank's device; ``initialize_mesh``
    # builds the 1-D ``("domain",)`` DeviceMesh DomainParallel decomposes over.
    from nvalchemi.distributed import DistributedManager

    DistributedManager.initialize()
    dm = DistributedManager()
    rank, world_size, device = dm.rank, dm.world_size, torch.device(dm.device)
    mesh = dm.initialize_mesh(mesh_shape=(world_size,), mesh_dim_names=("domain",))

    if rank == 0:
        logger.info(
            "MACE NVT distributed: world_size={ws} device={dev} "
            "checkpoint={ckpt} repeats={r} n_steps={n} T={T}K dt={dt}fs",
            ws=world_size,
            dev=device,
            ckpt=args.checkpoint,
            r=tuple(args.repeats),
            n=args.n_steps,
            T=args.temperature_k,
            dt=args.dt_fs,
        )

    # ----- Load MACE wrapper -----
    # Suppress mace-torch's chatty deprecation warnings at import.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nvalchemi.models.mace import MACEWrapper

    wrapper = MACEWrapper.from_checkpoint(
        args.checkpoint, dtype=dtype, device=device
    ).eval()
    if rank == 0:
        logger.info("MACE wrapper ready: cutoff={c} Å", c=wrapper.cutoff)

    # ----- Domain config -----
    # ``cutoff = wrapper.cutoff`` so the partitioner ghost-region width
    # matches the model's interaction range.
    domain_cfg = DomainConfig(cutoff=float(wrapper.cutoff), skin=0.5, mesh=mesh)

    # ----- Hooks -----
    # NeighborListHook runs at BEFORE_COMPUTE — the framework's
    # halo-exchange machinery has already produced the (owned + halo)
    # padded batch by this point, so the hook builds a NL on the
    # padded view that the model consumes verbatim.
    nl_hook = NeighborListHook(
        wrapper.model_config.neighbor_config,
        skin=0.5,
        stage=DynamicsStage.BEFORE_COMPUTE,
    )

    # SnapshotHook fires at AFTER_STEP and writes the resolved batch
    # state to a DataSink. We use HostMemory for tutorial simplicity:
    # cheap, in-memory, and we read it from rank 0 at the end of the
    # run to write the xyz trajectory. For longer runs swap in
    # ZarrData for incremental disk persistence.
    n_frames_expected = (args.n_steps // args.snapshot_every) + 1
    trajectory_sink = HostMemory(capacity=n_frames_expected)
    snapshot_hook = SnapshotHook(
        sink=trajectory_sink,
        frequency=args.snapshot_every,
    )
    # RANK_ZERO scope: DomainParallel gathers the FULL system onto rank 0 and
    # runs the hook only there, so the trajectory contains every atom. Without a
    # scope the hook defaults to LOCAL and each rank would record only its own
    # owned shard (rank 0's snapshot would be a fraction of the system).
    snapshot_hook.scope = HookScope.RANK_ZERO

    # ----- Inner integrator -----
    # NVTLangevin owns the per-graph thermostat state and the
    # velocity-Verlet update. ``NeighborListHook`` lives on the inner
    # because it must fire at ``BEFORE_COMPUTE`` (the padded-batch view
    # is only assembled inside ``DomainParallel._distributed_compute``).
    integrator = NVTLangevin(
        model=wrapper,
        dt=args.dt_fs,
        temperature=args.temperature_k,
        friction=args.friction,
        hooks=[nl_hook],
        n_steps=args.n_steps,
    )

    # ----- DomainParallel wrapping -----
    # Wraps the integrator with halo exchange + per-rank dispatch.
    #
    # ``SnapshotHook`` (AFTER_STEP) must live on the **outer**
    # ``DomainParallel`` — its ``step()`` only fires AFTER_STEP on the
    # outer hook chain, after atom migration has resolved. Inner
    # AFTER_STEP would never fire.
    # ``with dynamics:`` releases the adapter's setup on exit (delegates to
    # ``close()``), so teardown is exception-safe; the process-group lifecycle
    # stays at launcher scope (``DistributedManager.cleanup()`` below).
    with DomainParallel(
        dynamics=integrator,
        config=domain_cfg,
        n_steps=args.n_steps,
        hooks=[snapshot_hook],
    ) as dynamics:
        # ----- Build the initial batch on rank 0 -----
        # DomainParallel.partition() requires the full batch on rank 0 and
        # ``None`` elsewhere; it scatters each rank's owned subdomain.
        initial_batch = (
            build_initial_batch(tuple(args.repeats), dtype=dtype, device=device)
            if rank == 0
            else None
        )
        owned_batch = dynamics.partition(initial_batch)
        if rank == 0:
            logger.info(
                "Partitioned: n_owned (rank 0) = {n} of {tot} global atoms",
                n=int(owned_batch.positions.shape[0]),
                tot=int(initial_batch.positions.shape[0]),
            )

        # ----- Run the trajectory -----
        # ``run`` is the canonical entry point: halo exchange → forward →
        # consolidate → integrator update → atom migration, then closes hooks.
        # SnapshotHook writes every ``snapshot_every`` steps into the sink.
        dynamics.run(owned_batch)

        # ----- Persist trajectory (rank 0 holds the gathered frames) -----
        if rank == 0:
            n_frames = write_trajectory_xyz(trajectory_sink, args.output_xyz)
            logger.info(
                "Done. Wrote {f} xyz frames to {p}.", f=n_frames, p=args.output_xyz
            )

    # Process-group teardown stays at launcher scope.
    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
