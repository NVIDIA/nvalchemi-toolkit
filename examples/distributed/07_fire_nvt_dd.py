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
2-D-parallel dynamics: FIRE → NVT, each stage domain-decomposed
==============================================================

A two-stage streaming pipeline — FIRE relaxation then NVT Langevin MD — where
**each stage is itself domain-decomposed** across a group of GPUs. This is the
2-D generalization of :ref:`01_distributed_pipeline`: that example maps one rank
per stage; here each stage is a whole **domain sub-mesh** cooperating on one large
system, and the two stages form the pipeline dimension.

.. rubric:: Topology

.. graphviz::
   :caption: FIRE (domain group {0,1}) → NVT (domain group {2,3}) on a 2×2 mesh.

   digraph topology {
       rankdir=LR
       fontname="Helvetica"
       node [fontname="Helvetica" fontsize=11 shape=box style="rounded,filled" fillcolor="#dce6f1"]
       edge [fontname="Helvetica" fontsize=10]

       subgraph cluster_fire {
           label="Stage 0 — FIRE (DomainParallel)"; style=dashed; color="#7f8c8d"
           r0 [label="Rank 0\\ndomain-lead"]
           r1 [label="Rank 1"]
       }
       subgraph cluster_nvt {
           label="Stage 1 — NVT (DomainParallel)"; style=dashed; color="#7f8c8d"
           r2 [label="Rank 2\\ndomain-lead" fillcolor="#f9e2ae"]
           r3 [label="Rank 3" fillcolor="#f9e2ae"]
       }

       r0 -> r1 [dir=both style=dashed label="halo"]
       r2 -> r3 [dir=both style=dashed label="halo"]
       r0 -> r2 [style=bold color="#c0392b" penwidth=2 label="hand off\\n(lead→lead)"]
   }

The **domain** dimension is per-step and bandwidth-heavy (the halo exchange runs
every MD step) — keep it intra-node (NVLink). The **pipeline** dimension is
latency-tolerant (a system hands off only when it finishes a stage) — it may span
nodes over IB. ``DeviceMesh`` is row-major, so ``("pipeline", "domain")`` puts the
domain ranks contiguous (same node when ``domain_size ≤ gpus_per_node``); the
lead→lead handoff then rides the pipeline axis.

The whole thing is expressed with the *same* pieces as single-GPU dynamics: a
stage is just ``DomainParallel(dynamics)`` — the same wrap used for standalone
domain decomposition — handed to ``DistributedPipeline(stages, mesh=mesh2d)``.
``DomainParallel`` overrides the pipeline's communication seam so the group lead
performs the cross-stage handoff and the group scatters/gathers to its sub-mesh;
no distributed-aware code leaks into the model or the integrators.

System: alpha-quartz SiO2 supercell, periodic on all axes.

.. note::

    Requires 4 GPUs (2 pipeline stages × 2 domain ranks). Run with::

        torchrun --nproc_per_node=4 examples/distributed/07_fire_nvt_dd.py

    For MACE + cuEquivariance across ranks, set the JIT-race guard::

        CUEQUIVARIANCE_OPS_PARALLEL_COMPILE=0 \\
            torchrun --nproc_per_node=4 \\
            examples/distributed/07_fire_nvt_dd.py

Outputs the NVT trajectory to ``./fire_nvt_dd_trajectory.xyz`` (NVT domain-lead).
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
from nvalchemi.dynamics import DistributedPipeline, HostMemory, NVTLangevin
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks import LoggingHook, SnapshotHook
from nvalchemi.dynamics.optimizers.fire import FIRE
from nvalchemi.hooks import NeighborListHook

# Distributed examples are launcher-only: Sphinx sets this during docs builds
# (no torchrun env), torchrun sets rank/world-size during real launches.
_DOCS_BUILD = os.environ.get("NVALCHEMI_SPHINX_BUILD") == "1"
_DISTRIBUTED_ENV = "RANK" in os.environ and "WORLD_SIZE" in os.environ

# Reuse the SiO2 supercell builder shared across the distributed examples.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[2] / "benchmark" / "distributed")
)
from _benchmark_common import build_sio2_supercell  # noqa: E402


def build_initial_batch(
    repeats: tuple[int, int, int], dtype: torch.dtype, device: torch.device
) -> Batch:
    """A perturbed SiO2 supercell (seed=1) so FIRE has something to relax."""
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
    data.add_node_property("velocities", torch.zeros_like(pos).to(device))
    return Batch.from_data_list([data], device=device)


def write_trajectory_xyz(sink: HostMemory, path: Path) -> int:
    """Decode a :class:`HostMemory` sink into an extxyz trajectory (lead only)."""
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


def make_step_trace_hook(
    *,
    rank: int,
    gpu: int,
    pipeline_index: int,
    domain_rank: int,
    stage_name: str,
    frequency: int,
) -> LoggingHook:
    """A :class:`~nvalchemi.dynamics.hooks.LoggingHook` that streams *where each
    system is* to the console every ``frequency`` steps: for this rank's owned
    shard it logs the step, energy, max force, and temperature, tagged with the
    rank/GPU/stage so you can watch every group make progress in place. (Pair with
    ``--verbose`` — which also enables the framework's GPU/stage hand-off trace.)"""
    tag = f"rank {rank} · gpu {gpu} · {stage_name} (pipe {pipeline_index}/dom {domain_rank})"

    def _writer(step: int, rows: list[dict[str, float]]) -> None:
        for row in rows:
            fields = []
            if "energy" in row:
                fields.append(f"E={row['energy']:.4f} eV")
            if "fmax" in row:
                fields.append(f"fmax={row['fmax']:.4f} eV/Å")
            if "temperature" in row:
                fields.append(f"T={row['temperature']:.1f} K")
            logger.info(
                "[step {s:>5} | {tag}] owned-shard: {f}",
                s=int(row.get("step", step)),
                tag=tag,
                f="  ".join(fields),
            )

    return LoggingHook(
        backend="custom",
        writer_fn=_writer,
        frequency=frequency,
        stage=DynamicsStage.AFTER_STEP,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FIRE → NVT as a 2-D-parallel (pipeline × domain) pipeline."
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
        "--verbose",
        action="store_true",
        help="Trace where each system is at every step (per-rank/GPU/stage state) "
        "and log every GPU/stage hand-off (enables the pipeline's debug_mode).",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Step interval for the per-system state trace under --verbose.",
    )
    args = parser.parse_args()

    if _DOCS_BUILD or not _DISTRIBUTED_ENV:
        logger.info(
            "Not running under torchrun — skipping. Launch with: torchrun "
            "--nproc_per_node=4 examples/distributed/07_fire_nvt_dd.py"
        )
        return

    # ----- Distributed bootstrap: 2-D (pipeline, domain) mesh -----
    # 2 pipeline stages × (world/2) domain ranks. DistributedManager owns init +
    # device binding; ``initialize_mesh`` builds the 2-D (pipeline, domain) mesh.
    from nvalchemi.distributed import DistributedManager

    DistributedManager.initialize()
    dm = DistributedManager()
    rank, world_size, device = dm.rank, dm.world_size, torch.device(dm.device)
    n_pipeline = 2
    if world_size < 4 or world_size % n_pipeline != 0:
        raise RuntimeError(
            f"world_size {world_size} must be an even number >= 4 (2 pipeline "
            "stages × >=2 domain ranks); launch with e.g. --nproc_per_node=4."
        )
    domain_size = world_size // n_pipeline
    mesh = dm.initialize_mesh(
        mesh_shape=(n_pipeline, domain_size),
        mesh_dim_names=("pipeline", "domain"),
    )
    pipeline_index = int(mesh["pipeline"].get_local_rank())
    is_domain_lead = int(mesh["domain"].get_local_rank()) == 0

    if rank == 0:
        logger.info(
            "FIRE→NVT 2-D DD: world={ws} mesh=(pipeline={p}, domain={d}) "
            "ckpt={c} repeats={r} fire={fs} nvt={ns} T={T}K",
            ws=world_size,
            p=n_pipeline,
            d=domain_size,
            c=args.checkpoint,
            r=tuple(args.repeats),
            fs=args.fire_steps,
            ns=args.nvt_steps,
            T=args.temperature_k,
        )

    # ----- Model (one instance per rank; both stages use the same checkpoint) -----
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from nvalchemi.models.mace import MACEWrapper

    dtype = torch.float32
    wrapper = MACEWrapper.from_checkpoint(
        args.checkpoint, dtype=dtype, device=device
    ).eval()
    # Each stage's DomainParallel is bound to its domain sub-mesh row.
    domain_cfg = DomainConfig(
        cutoff=float(wrapper.cutoff), skin=0.5, mesh=mesh["domain"]
    )

    def _nl_hook() -> NeighborListHook:
        return NeighborListHook(
            wrapper.model_config.neighbor_config,
            skin=0.5,
            stage=DynamicsStage.BEFORE_COMPUTE,
        )

    # Per-step "where is my system" console trace (owned-shard view), tagged with
    # rank/GPU/stage. Only under --verbose; None otherwise.
    domain_rank = int(mesh["domain"].get_local_rank())
    stage_name = "FIRE" if pipeline_index == 0 else "NVT"
    trace_hook = (
        make_step_trace_hook(
            rank=rank,
            gpu=(device.index if device.type == "cuda" else 0),
            pipeline_index=pipeline_index,
            domain_rank=domain_rank,
            stage_name=stage_name,
            frequency=args.log_every,
        )
        if args.verbose
        else None
    )

    # ----- Build ONLY this rank's stage, keyed by its pipeline index -----
    # A domain-decomposed stage is just DomainParallel(dynamics); the pipeline mesh
    # drives lead resolution, the lead→lead handoff, and per-group completion.
    if pipeline_index == 0:
        fire = FIRE(model=wrapper, dt=args.fire_dt, hooks=[_nl_hook()])
        outer_hooks = [trace_hook] if trace_hook is not None else []
        stage: DomainParallel = DomainParallel(
            dynamics=fire,
            config=domain_cfg,
            n_steps=args.fire_steps,
            hooks=outer_hooks,
        )
        # The first stage's domain-lead seeds the system; the group scatters it.
        if is_domain_lead:
            stage._pending_input = build_initial_batch(
                tuple(args.repeats), dtype=dtype, device=device
            )
        trajectory_sink = None
    else:
        # NVT production leg. A RANK_ZERO snapshot hook gathers the full system onto
        # the domain-lead each frame so the trajectory has every atom. (The relaxed
        # structure arrives with FIRE's fictitious velocities; the Langevin
        # thermostat equilibrates it to the target temperature.)
        n_frames = (args.nvt_steps // args.snapshot_every) + 1
        trajectory_sink = HostMemory(capacity=n_frames)
        snapshot_hook = SnapshotHook(
            sink=trajectory_sink, frequency=args.snapshot_every
        )
        snapshot_hook.scope = HookScope.RANK_ZERO
        nvt = NVTLangevin(
            model=wrapper,
            dt=args.nvt_dt_fs,
            temperature=args.temperature_k,
            friction=args.friction,
            hooks=[_nl_hook()],
        )
        outer_hooks = [snapshot_hook]
        if trace_hook is not None:
            outer_hooks.append(trace_hook)
        stage = DomainParallel(
            dynamics=nvt,
            config=domain_cfg,
            n_steps=args.nvt_steps,
            hooks=outer_hooks,
        )

    # ----- Drive the 2-D pipeline: FIRE group relaxes → hands off → NVT group runs -----
    # debug_mode surfaces the per-group step flow + every GPU/stage hand-off (the
    # DomainParallel comm seam logs when a system is seeded, received, handed off,
    # or retired) — the "when does each system change GPUs/stages" trace.
    pipeline = DistributedPipeline(
        stages={pipeline_index: stage}, mesh=mesh, debug_mode=args.verbose
    )
    if rank == 0:
        logger.info("Running FIRE→NVT across the 2-D mesh…")
    with pipeline:
        pipeline.run()
    if trace_hook is not None:
        trace_hook.close()

    # ----- Persist the NVT trajectory (its domain-lead) -----
    if pipeline_index == 1 and is_domain_lead and trajectory_sink is not None:
        n = write_trajectory_xyz(trajectory_sink, args.output_xyz)
        logger.info("Done. Wrote {n} NVT frames to {p}.", n=n, p=args.output_xyz)

    stage.close()
    # Process-group teardown stays at launcher scope.
    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
