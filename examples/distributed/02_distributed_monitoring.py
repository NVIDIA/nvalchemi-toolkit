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
Monitoring a Distributed Pipeline: Per-Rank Logging and Profiling
=================================================================

Building on :doc:`01_distributed_pipeline`, this example adds
comprehensive observability to the FIRE → NVTLangevin topology:

* :class:`~nvalchemi.dynamics.hooks.LoggingHook` on each rank, writing
  per-graph scalars (energy, fmax, temperature) to rank-specific CSV files.
* :class:`~nvalchemi.dynamics.hooks.ProfilerHook` on each rank, recording
  step timings to rank-specific CSV files.
* :class:`~nvalchemi.dynamics.ZarrData` sink on the Langevin ranks
  instead of HostMemory — completed trajectories are persisted to disk.
* Post-run collation: rank 0 reads and aggregates the per-rank log files
  to produce a unified summary.

.. note::

    This example requires 4 GPUs.  Run with::

        torchrun --nproc_per_node=4 examples/distributed/02_distributed_monitoring.py

    For CPU-only testing, change ``backend="nccl"`` to ``backend="gloo"``.
"""

from __future__ import annotations

import csv
import logging
import os

import torch
import torch.distributed as dist
from ase.build import molecule
from loguru import logger

from nvalchemi.data import AtomicData
from nvalchemi.dynamics import (
    FIRE,
    ConvergenceHook,
    DistributedPipeline,
    NVTLangevin,
    SizeAwareSampler,
    ZarrData,
)
from nvalchemi.dynamics.base import BufferConfig, DynamicsStage
from nvalchemi.dynamics.hooks import ConvergedSnapshotHook, LoggingHook, ProfilerHook
from nvalchemi.hooks import HookContext
from nvalchemi.models.demo import DemoModelWrapper

logging.basicConfig(level=logging.INFO)

# When run outside ``torchrun`` (e.g. during a Sphinx docs build), the
# distributed environment variables ``RANK`` and ``WORLD_SIZE`` are absent.
# We detect this and skip the pipeline launch so the example renders in
# the gallery without requiring multiple GPUs.
_DISTRIBUTED_ENV = "RANK" in os.environ and "WORLD_SIZE" in os.environ

# ---------------------------------------------------------------------------
# Helpers (identical to 01_distributed_pipeline)
# ---------------------------------------------------------------------------


def atoms_to_data(atoms) -> AtomicData:
    """Convert an ASE Atoms object to AtomicData with dynamics fields."""
    data = AtomicData.from_atoms(atoms)
    n = data.num_nodes
    data.forces = torch.zeros(n, 3)
    data.energy = torch.zeros(1, 1)
    data.add_node_property("velocities", torch.zeros(n, 3))
    return data


class InMemoryDataset:
    """Minimal dataset wrapper for ``SizeAwareSampler``."""

    def __init__(self, data_list: list[AtomicData]) -> None:
        self._data = data_list

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[AtomicData, dict]:
        d = self._data[idx]
        return d, {"num_atoms": d.num_nodes, "num_edges": d.num_edges}

    def get_metadata(self, idx: int) -> tuple[int, int]:
        """Get the metadata associated with idx"""
        d = self._data[idx]
        return d.num_nodes, d.num_edges


class DownstreamDoneHook:
    """Set ``stage.done = True`` after *patience* consecutive idle steps.

    Because :class:`~nvalchemi.hooks.HookContext` does not carry a
    reference to the dynamics engine, the ``dynamics`` attribute must
    be set after the engine is constructed (see ``make_langevin``).
    """

    stage = DynamicsStage.AFTER_STEP
    frequency = 1

    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self._idle_steps = 0
        self.dynamics: object | None = None

    def __call__(self, ctx: HookContext, stage_: DynamicsStage) -> None:
        if ctx.batch.num_graphs == 0:
            self._idle_steps += 1
        else:
            self._idle_steps = 0
        if self._idle_steps >= self.patience and self.dynamics is not None:
            self.dynamics.done = True


def build_dataset() -> list[AtomicData]:
    """Create a handful of small rattled molecules."""
    names = ["H2O", "CH4", "NH3", "H2O", "CH4", "NH3", "H2O", "CH4"]
    data_list = []
    for name in names:
        atoms = molecule(name)
        atoms.rattle(stdev=0.15)
        data_list.append(atoms_to_data(atoms))
    return data_list


# %%
# Per-rank logging and profiling hooks
# -----------------------------------------
# Each rank writes its observability data to separate files named after
# the rank, avoiding any filesystem contention between processes.
#
# :class:`~nvalchemi.dynamics.hooks.LoggingHook` records per-graph scalar
# observables (energy, fmax, temperature) at a configurable step interval.
# :class:`~nvalchemi.dynamics.hooks.ProfilerHook` records wall-clock time
# between hook stages, optionally with NVTX annotations for Nsight Systems.
#
# Both hooks are created *inside* the stage factory functions so that the
# rank number — obtained via ``dist.get_rank()`` after the process group is
# initialised — is available at construction time.  The hooks are used as
# context managers via ``with`` to guarantee that background I/O threads are
# flushed and file handles are closed when the pipeline exits.


# %%
# ZarrData sink for persistent storage
# ------------------------------------------
# The downstream Langevin ranks use :class:`~nvalchemi.dynamics.ZarrData`
# instead of ``HostMemory`` so that completed trajectories are written to
# disk and survive the process.  Each rank writes to a separate Zarr store
# at path ``trajectories_rank{rank}.zarr``.  The ``ZarrData`` constructor
# accepts any ``StoreLike`` — a filesystem path string is the simplest form.
# Capacity is set generously; ``ZarrData`` will raise if the limit is
# exceeded before the run ends.


# %%
# Stage construction with monitoring
# ----------------------------------------
# The stage factories are extended from example 01 to accept and attach
# ``LoggingHook`` and ``ProfilerHook`` instances.  The hooks are passed in
# as context managers so the ``with`` block in ``main()`` can flush and close
# them after ``pipeline.run()`` returns.


def make_fire(
    model: DemoModelWrapper,
    rank: int,
    logging_hook: LoggingHook,
    profiler_hook: ProfilerHook,
    **kwargs,
) -> FIRE:
    """Create a FIRE optimiser stage with logging and profiling."""
    return FIRE(
        model=model,
        dt=1.0,
        n_steps=50,
        hooks=[logging_hook, profiler_hook],
        convergence_hook=ConvergenceHook(
            criteria=[
                {
                    "key": "forces",
                    "threshold": 0.05,
                    "reduce_op": "norm",
                    "reduce_dims": -1,
                }
            ],
        ),
        **kwargs,
    )


def make_langevin(
    model: DemoModelWrapper,
    sink: ZarrData,
    rank: int,
    logging_hook: LoggingHook,
    profiler_hook: ProfilerHook,
    **kwargs,
) -> NVTLangevin:
    """Create an NVTLangevin MD stage with snapshot, logging, and profiling hooks."""
    done_hook = DownstreamDoneHook(patience=10)
    stage = NVTLangevin(
        model=model,
        dt=0.5,
        temperature=300.0,
        friction=0.01,
        n_steps=20,
        hooks=[
            ConvergedSnapshotHook(sink=sink, frequency=1),
            done_hook,
            logging_hook,
            profiler_hook,
        ],
        convergence_hook=ConvergenceHook(
            criteria=[
                {
                    "key": "forces",
                    "threshold": 0.01,
                    "reduce_op": "norm",
                    "reduce_dims": -1,
                }
            ],
        ),
        **kwargs,
    )
    done_hook.dynamics = stage
    return stage


# %%
# Running the pipeline
# -------------------------
# The pipeline setup is identical to example 01; the only differences are
# the sink type (ZarrData vs HostMemory) and the additional hooks.  All
# hook context managers are entered before calling ``pipeline.run()`` so
# that CUDA streams and thread pool executors are live throughout the run.


def main() -> None:
    """Launch two monitored FIRE -> Langevin pipelines on 4 GPUs."""
    model = DemoModelWrapper()

    # Dataset (only used by ranks 0 and 2)
    all_data = build_dataset()
    mid = len(all_data) // 2
    dataset_a = InMemoryDataset(all_data[:mid])
    dataset_b = InMemoryDataset(all_data[mid:])

    sampler_a = SizeAwareSampler(
        dataset=dataset_a,
        max_atoms=50,
        max_edges=0,
        max_batch_size=4,
    )
    sampler_b = SizeAwareSampler(
        dataset=dataset_b,
        max_atoms=50,
        max_edges=0,
        max_batch_size=4,
    )

    buffer_cfg = BufferConfig(num_systems=4, num_nodes=50, num_edges=0)

    if not _DISTRIBUTED_ENV:
        logger.info(
            "Not running under torchrun — skipping pipeline launch. "
            "Run with: torchrun --nproc_per_node=4 "
            "examples/distributed/02_distributed_monitoring.py",
        )
        return

    # Zarr sinks — one per downstream rank, each at a rank-specific path.
    # Created on all ranks for code symmetry; only ranks 1 and 3 write to them.
    sink_a = ZarrData(store="trajectories_rank1.zarr", capacity=1000)
    sink_b = ZarrData(store="trajectories_rank3.zarr", capacity=1000)

    # Hooks — created before pipeline initialisation so that rank is known
    # after dist.init_process_group fires inside DistributedPipeline.__enter__.
    # We defer rank resolution to inside the ``with pipeline:`` block by
    # building stages lazily after the context manager is entered.

    # DistributedPipeline.__enter__ calls dist.init_process_group, so we
    # build the stages dict inside the ``with`` block to ensure dist is live.
    pipeline_kwargs = dict(backend="nccl", debug_mode=True)

    # Build stages inside the context manager so dist.get_rank() is valid.
    # We create a thin wrapper that defers stage construction.
    class _DeferredMain:
        """Helper that constructs stages after the process group is ready."""

        def run(self) -> None:
            rank = dist.get_rank() if dist.is_initialized() else 0

            # Per-rank hook instances — each rank writes to its own files.
            fire_logger = LoggingHook(
                backend="csv",
                log_path=f"fire_rank{rank}_log.csv",
                frequency=10,
            )
            fire_profiler = ProfilerHook(
                "step",
                log_path=f"fire_profile_rank{rank}.csv",
            )
            langevin_logger = LoggingHook(
                backend="csv",
                log_path=f"langevin_rank{rank}_log.csv",
                frequency=5,
            )
            langevin_profiler = ProfilerHook(
                "step",
                log_path=f"langevin_profile_rank{rank}.csv",
            )

            stages = {
                0: make_fire(
                    model,
                    rank=0,
                    logging_hook=fire_logger,
                    profiler_hook=fire_profiler,
                    sampler=sampler_a,
                    refill_frequency=1,
                    prior_rank=None,
                    next_rank=1,
                    buffer_config=buffer_cfg,
                ),
                1: make_langevin(
                    model,
                    sink=sink_a,
                    rank=1,
                    logging_hook=langevin_logger,
                    profiler_hook=langevin_profiler,
                    prior_rank=0,
                    next_rank=None,
                    buffer_config=buffer_cfg,
                ),
                2: make_fire(
                    model,
                    rank=2,
                    logging_hook=fire_logger,
                    profiler_hook=fire_profiler,
                    sampler=sampler_b,
                    refill_frequency=1,
                    prior_rank=None,
                    next_rank=3,
                    buffer_config=buffer_cfg,
                ),
                3: make_langevin(
                    model,
                    sink=sink_b,
                    rank=3,
                    logging_hook=langevin_logger,
                    profiler_hook=langevin_profiler,
                    prior_rank=2,
                    next_rank=None,
                    buffer_config=buffer_cfg,
                ),
            }

            pipeline = DistributedPipeline(stages=stages, **pipeline_kwargs)
            with fire_logger, langevin_logger:
                with pipeline:
                    pipeline.run()

            return rank, fire_logger, langevin_logger, fire_profiler, langevin_profiler

    deferred = _DeferredMain()
    rank, fire_logger, langevin_logger, fire_profiler, langevin_profiler = (
        deferred.run()
    )

    # %%
    # Post-run analysis (rank 0 only)
    # -------------------------------------
    # After all ranks finish, rank 0 reads the per-rank CSV log files,
    # computes summary statistics, and prints a collated table.  This uses
    # only Python's built-in ``csv`` module — no pandas dependency.
    #
    # The log files follow the schema written by LoggingHook: each row has
    # at minimum the columns ``step``, ``graph_idx``, ``energy``, and
    # ``fmax``.  We aggregate the mean step time per rank from the profiler
    # CSV files as well.
    #
    # A barrier ensures all ranks have finished writing their CSV files
    # before rank 0 reads them.
    dist.barrier()

    if rank == 0:
        _print_post_run_summary(num_ranks=4)

    # %%
    # Collecting trajectories from Zarr
    # ----------------------------------------
    # Downstream ranks (1 and 3) wrote completed trajectories to Zarr stores.
    # After the pipeline finishes, any rank can open those stores and inspect
    # their contents.  Here rank 0 reports basic statistics for both stores.

    if rank == 0:
        for store_path, label in [
            ("trajectories_rank1.zarr", "sink_a"),
            ("trajectories_rank3.zarr", "sink_b"),
        ]:
            _report_zarr_stats(store_path, label)


def _print_post_run_summary(num_ranks: int) -> None:
    """Read per-rank CSV logs and print a collated summary on rank 0.

    Parameters
    ----------
    num_ranks : int
        Total number of ranks in the pipeline.
    """
    print("\n=== Post-run summary (rank 0) ===")
    print(
        f"{'Rank':<6} {'Role':<12} {'Log rows':<12} {'Mean energy':<16} {'Mean fmax':<12}"
    )
    print("-" * 60)

    role_map = {0: "FIRE", 1: "Langevin", 2: "FIRE", 3: "Langevin"}
    log_prefix = {0: "fire", 1: "langevin", 2: "fire", 3: "langevin"}

    for r in range(num_ranks):
        log_path = f"{log_prefix[r]}_rank{r}_log.csv"
        try:
            with open(log_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                print(f"{r:<6} {role_map[r]:<12} {'0':<12} {'n/a':<16} {'n/a':<12}")
                continue
            energy = [float(row["energy"]) for row in rows if row.get("energy")]
            fmaxes = [float(row["fmax"]) for row in rows if row.get("fmax")]
            mean_energy = sum(energy) / len(energy) if energy else float("nan")
            mean_fmax = sum(fmaxes) / len(fmaxes) if fmaxes else float("nan")
            print(
                f"{r:<6} {role_map[r]:<12} {len(rows):<12} "
                f"{mean_energy:<16.4f} {mean_fmax:<12.4f}"
            )
        except FileNotFoundError:
            print(f"{r:<6} {role_map[r]:<12} {'(no log)':<12} {'n/a':<16} {'n/a':<12}")

    print()
    print(f"{'Rank':<6} {'Profile rows':<16} {'Mean step time (s)':<22}")
    print("-" * 44)
    for r in range(num_ranks):
        profile_path = f"profile_rank{r}.csv"
        try:
            with open(profile_path, newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            if not rows:
                print(f"{r:<6} {'0':<16} {'n/a':<22}")
                continue
            # ProfilerHook CSV rows contain a "delta_s" column for elapsed time.
            deltas = [float(row["delta_s"]) for row in rows if row.get("delta_s")]
            mean_delta = sum(deltas) / len(deltas) if deltas else float("nan")
            print(f"{r:<6} {len(rows):<16} {mean_delta:<22.6f}")
        except FileNotFoundError:
            print(f"{r:<6} {'(no profile)':<16} {'n/a':<22}")

    print("=================================\n")


def _report_zarr_stats(store_path: str, label: str) -> None:
    """Open a ZarrData store and report basic trajectory statistics.

    Parameters
    ----------
    store_path : str
        Filesystem path to the Zarr store.
    label : str
        Human-readable label for the store (e.g. ``"sink_a"``).
    """
    sink = ZarrData(store=store_path)
    n = len(sink)
    if n == 0:
        logger.info(f"{label} ({store_path}): 0 samples written")
        return
    batch = sink.read()
    logger.info(
        f"{label} ({store_path}): {n} samples, "
        f"mean atoms per system = {batch.num_nodes / batch.num_graphs:.1f}"
    )


if __name__ == "__main__":
    main()
