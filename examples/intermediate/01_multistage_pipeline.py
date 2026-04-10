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
Multi-Stage Dynamics Pipelines with FusedStage
===============================================

This example demonstrates how to compose multiple dynamics stages into a
single GPU-resident pipeline using :class:`~nvalchemi.dynamics.FusedStage`.
A typical MD workflow chains geometry relaxation (FIRE) into equilibration
(NVT) and then production sampling (NVE).  Running all three stages on a
shared batch avoids repeated CPU–GPU data transfers between stages.

Topics covered:

* **Part 1** — Standalone FIRE relaxation as a baseline.
* **Part 2** — Three-stage ``FIRE → NVT → NVE`` pipeline with
  :class:`~nvalchemi.dynamics.hooks.LoggingHook` writing CSV output for each
  stage and a custom ``StageTransitionLogger`` firing on convergence.
* **Part 3** — Inflight batching (Mode 2): :class:`~nvalchemi.dynamics.SizeAwareSampler`
  feeds a dataset into :class:`~nvalchemi.dynamics.FusedStage` so that
  graduated systems are replaced on the fly, keeping the GPU fully occupied.
* **Part 4** — ``register_fused_hook`` attaches a hook that sees the
  combined batch at every step regardless of which sub-stage is active,
  enabling global status inspection.

The sampler, sink, and hook interfaces are all pluggable — users can
supply custom implementations for any of them.  See the advanced examples
for custom hook and integrator patterns.
"""

import logging
import time
from collections import defaultdict

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE, NVE, NVTLangevin, SizeAwareSampler
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage, FusedStage
from nvalchemi.dynamics.hooks import LoggingHook
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.hooks import HookContext
from nvalchemi.models.demo import DemoModelWrapper

logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

FIRE_FMAX_THRESHOLD = 0.05  # eV/Å — force convergence criterion
EQUIL_STEPS_PER_SYSTEM = 10  # NVT equilibration steps (per-system budget)
PROD_STEPS_PER_SYSTEM = 20  # NVE production steps (per-system budget)
TEMPERATURE = 300.0  # Kelvin

# ---------------------------------------------------------------------------
# Shared model
# ---------------------------------------------------------------------------

torch.manual_seed(42)
model = DemoModelWrapper()
model.eval()


# ---------------------------------------------------------------------------
# Helper: create a random molecular system
# ---------------------------------------------------------------------------


def _make_system(n_atoms: int, seed: int) -> AtomicData:
    """Create a random AtomicData with ``n_atoms`` atoms and a fixed seed.

    Parameters
    ----------
    n_atoms : int
        Number of atoms in the system.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    AtomicData
        Randomly initialised system with zero forces and velocities.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    data = AtomicData(
        positions=torch.randn(n_atoms, 3, generator=g),
        atomic_numbers=torch.randint(1, 10, (n_atoms,), dtype=torch.long, generator=g),
        atomic_masses=torch.ones(n_atoms),
        forces=torch.zeros(n_atoms, 3),
        energy=torch.zeros(1, 1),
    )
    data.add_node_property("velocities", torch.zeros(n_atoms, 3))
    return data


# ---------------------------------------------------------------------------
# Custom hook: log every convergence event (FIRE → NVT transition)
# ---------------------------------------------------------------------------


class StageTransitionLogger:
    """Log convergence events with wall-clock timing and status distribution.

    Parameters
    ----------
    label : str
        Human-readable label printed with every message.
    frequency : int
        Print only every ``frequency``-th convergence event.
    """

    stage = DynamicsStage.ON_CONVERGE

    def __init__(self, label: str = "transition", frequency: int = 1) -> None:
        self.label = label
        self.frequency = frequency
        self._t0 = time.monotonic()
        self._n_transitions = 0

    def __call__(self, ctx: HookContext, stage_: DynamicsStage) -> None:
        self._n_transitions += 1
        if self._n_transitions % self.frequency != 0:
            return
        batch = ctx.batch
        elapsed = time.monotonic() - self._t0
        if batch.status is not None:
            dist: dict[int, int] = defaultdict(int)
            for s in batch.status.squeeze(-1).tolist():
                dist[int(s)] += 1
            dist_str = "  ".join(f"status={k}:{v}" for k, v in sorted(dist.items()))
        else:
            dist_str = "status=unknown"
        logging.info(
            "[%s] CONVERGE at step=%d  elapsed=%.2fs  %s",
            self.label,
            ctx.step_count,
            elapsed,
            dist_str,
        )

    @property
    def n_transitions(self) -> int:
        """Number of convergence events seen so far."""
        return self._n_transitions


# %%
# Part 1 — Standalone FIRE relaxation
# -------------------------------------
# The simplest use case: relax a batch of three structures until the maximum
# force component falls below ``FIRE_FMAX_THRESHOLD``.  The ``run()`` method
# returns the relaxed batch; ``n_steps=150`` is the hard step-count ceiling.

data_list_single = [_make_system(n, seed) for n, seed in [(4, 1), (5, 2), (6, 3)]]
batch_single = Batch.from_data_list(data_list_single)

fire_standalone = FIRE(
    model=model,
    dt=0.1,
    n_steps=150,
    convergence_hook=ConvergenceHook.from_forces(FIRE_FMAX_THRESHOLD),
)
batch_single = fire_standalone.run(batch_single)
logging.info(
    "Part 1 done: %d systems relaxed in %d steps",
    batch_single.num_graphs,
    fire_standalone.step_count,
)

# %%
# Part 2 — Three-stage FIRE + NVT + NVE FusedStage
# --------------------------------------------------
# The ``+`` operator on two dynamics objects returns a :class:`FusedStage`
# that migrates each system from stage 0 (FIRE) to stage 1 (NVT) when its
# convergence criterion fires, then to stage 2 (NVE) after ``n_steps``.
# Systems that finish all stages are removed from the batch.
#
# :class:`~nvalchemi.dynamics.hooks.LoggingHook` writes per-step scalar
# observables (energy, temperature, fmax, ...) to a CSV file. Attach one to
# each sub-stage to get separate logs.

fire_logger = LoggingHook(backend="csv", log_path="fire_log.csv", frequency=10)
fire_transition_logger = StageTransitionLogger(label="FIRE->NVT", frequency=1)

equil_logger = LoggingHook(backend="csv", log_path="nvt_log.csv", frequency=5)

fire_stage = FIRE(
    model=model,
    dt=0.1,
    convergence_hook=ConvergenceHook.from_forces(FIRE_FMAX_THRESHOLD),
    hooks=[fire_logger, fire_transition_logger],
)
equil_stage = NVTLangevin(
    model=model,
    dt=0.5,
    temperature=TEMPERATURE,
    friction=0.1,
    random_seed=7,
    n_steps=EQUIL_STEPS_PER_SYSTEM,
    hooks=[equil_logger],
)
prod_stage = NVE(model=model, dt=0.5, n_steps=PROD_STEPS_PER_SYSTEM)

# Compose: FIRE -> NVT -> NVE.  ``n_steps=500`` is the total step budget for
# the fused run; systems migrate automatically between stages as they converge
# or exhaust their per-stage step allocation.
fused = fire_stage + equil_stage + prod_stage

data_list_fused = [
    _make_system(n, seed) for n, seed in [(4, 10), (6, 11), (5, 12), (4, 13)]
]
batch_fused = Batch.from_data_list(data_list_fused)
# LoggingHook must be used as a context manager so its background I/O thread
# is started before the run and flushed/joined after — without the ``with``
# block, log data written near the end of the run may be dropped.
with fire_logger, equil_logger:
    batch_fused = fused.run(batch_fused, n_steps=500)

logging.info(
    "Part 2 done: FIRE->NVT->NVE pipeline completed, %d systems remain",
    batch_fused.num_graphs if batch_fused is not None else 0,
)

# %%
# Part 3 — Inflight batching (Mode 2)
# ------------------------------------
# When the dataset is too large to process at once, pass ``batch=None`` to
# :meth:`FusedStage.run`.  The :class:`~nvalchemi.dynamics.SizeAwareSampler`
# bin-packs samples into the batch budget; as systems graduate they are
# replaced on the fly with new ones from the dataset.
#
# The dataset must implement three methods:
#
# * ``__len__()`` — total number of samples.
# * ``__getitem__(idx) -> (AtomicData, dict)`` — load one sample.
# * ``get_metadata(idx) -> (num_atoms, num_edges)`` — size hints for
#   bin-packing **without** loading the full sample (avoids I/O overhead).


class SimpleDataset:
    """Minimal in-memory dataset of random molecular systems.

    Parameters
    ----------
    n_samples : int
        Total number of systems in the dataset.
    atoms_per_sample : int
        Atom count for every system (uniform for simplicity).
    seed : int
        Base RNG seed; sample ``idx`` uses ``seed + idx``.
    """

    def __init__(self, n_samples: int, atoms_per_sample: int, seed: int = 0) -> None:
        self.n_samples = n_samples
        self.atoms_per_sample = atoms_per_sample
        self.base_seed = seed

    def __len__(self) -> int:
        return self.n_samples

    def get_metadata(self, idx: int) -> tuple[int, int]:
        """Return ``(num_atoms, num_edges)`` without loading the sample."""
        return self.atoms_per_sample, 0

    def __getitem__(self, idx: int) -> tuple[AtomicData, dict]:
        g = torch.Generator()
        g.manual_seed(self.base_seed + idx)
        data = AtomicData(
            positions=torch.randn(self.atoms_per_sample, 3, generator=g),
            atomic_numbers=torch.randint(
                1, 10, (self.atoms_per_sample,), dtype=torch.long, generator=g
            ),
            atomic_masses=torch.ones(self.atoms_per_sample),
            forces=torch.zeros(self.atoms_per_sample, 3),
            energy=torch.zeros(1, 1),
        )
        data.add_node_property("velocities", torch.zeros(self.atoms_per_sample, 3))
        return data, {}


dataset = SimpleDataset(n_samples=20, atoms_per_sample=4, seed=100)

# SizeAwareSampler enforces per-batch atom and system limits.  ``max_atoms=16``
# means at most 16 atoms total in the live batch, so ~4 systems of 4 atoms
# each.  ``max_edges=None`` disables edge-count gating (no edge list here).
sampler = SizeAwareSampler(dataset, max_atoms=16, max_edges=None, max_batch_size=4)

trajectory_sink = HostMemory(capacity=20)

stage0_inflight = FIRE(
    model=model,
    dt=0.1,
    convergence_hook=ConvergenceHook.from_forces(threshold=10.0),
)
stage1_inflight = NVTLangevin(
    model=model,
    dt=0.5,
    temperature=TEMPERATURE,
    friction=0.1,
    random_seed=77,
    n_steps=EQUIL_STEPS_PER_SYSTEM,
)


# ``init_fn`` is called when the initial batch is built; useful for logging
# or mutation before the first dynamics step.
def _inflight_init(batch: Batch) -> None:
    if getattr(batch, "system_id", None) is not None:
        ids = batch.system_id.squeeze(-1).tolist()
        logging.info("init_fn: initial batch system_ids = %s", ids)


fused_inflight = FusedStage(
    sub_stages=[(0, stage0_inflight), (1, stage1_inflight)],
    sampler=sampler,
    sinks=[trajectory_sink],
    refill_frequency=5,
    init_fn=_inflight_init,
)

# ``batch=None`` triggers inflight mode.  The run finishes when the sampler
# is exhausted and all remaining systems have graduated.
result_inflight = fused_inflight.run(batch=None, n_steps=300)

logging.info("Part 3 done: trajectory_sink contains %d snapshots", len(trajectory_sink))

# %%
# Part 4 — Global status monitoring via register_fused_hook
# ----------------------------------------------------------
# :meth:`FusedStage.register_fused_hook` attaches a hook that fires at the
# requested stage on the **combined** batch (all active systems from all
# sub-stages concatenated).  This is the right place to monitor global
# metrics like the status distribution or the mean energy.


class StatusSnapshotHook:
    """Print per-step status distribution and mean energy.

    Parameters
    ----------
    frequency : int
        Print every ``frequency`` steps.
    max_steps : int
        Stop printing after ``max_steps`` outputs to avoid log spam.
    """

    stage = DynamicsStage.AFTER_STEP

    def __init__(self, frequency: int = 2, max_steps: int = 10) -> None:
        self.frequency = frequency
        self._print_count = 0
        self._max_steps = max_steps

    def __call__(self, ctx: HookContext, stage_: DynamicsStage) -> None:
        if self._print_count >= self._max_steps:
            return
        if ctx.step_count % self.frequency != 0:
            return
        batch = ctx.batch
        # NOTE: .cpu() transfers synchronize the GPU.  This hook is limited to
        # ``max_steps`` invocations and is used here for illustration only.
        # For production monitoring prefer LoggingHook which uses async I/O.
        if batch.status is not None:
            dist: dict[int, int] = defaultdict(int)
            for s in batch.status.squeeze(-1).cpu().tolist():
                dist[int(s)] += 1
            dist_str = " | ".join(f"s{k}:{v}" for k, v in sorted(dist.items()))
        else:
            dist_str = "no status"
        e_str = ""
        if batch.energy is not None:
            e_mean = batch.energy.squeeze(-1).mean().cpu().item()
            e_str = f"  E_mean={e_mean:.4f}"
        logging.info("step=%3d  [%s]%s", ctx.step_count, dist_str, e_str)
        self._print_count += 1


snapshot_data = [_make_system(5, 20), _make_system(5, 21)]
snapshot_batch = Batch.from_data_list(snapshot_data)

fire_inspect = FIRE(
    model=model,
    dt=0.1,
    n_steps=30,
    convergence_hook=ConvergenceHook.from_forces(FIRE_FMAX_THRESHOLD),
)
nvt_inspect = NVTLangevin(
    model=model,
    dt=0.5,
    temperature=TEMPERATURE,
    friction=0.1,
    random_seed=99,
    n_steps=10,
)

fused_inspect = fire_inspect + nvt_inspect

snapshot_hook = StatusSnapshotHook(frequency=2, max_steps=12)
fused_inspect.register_fused_hook(snapshot_hook)

snapshot_batch = fused_inspect.run(snapshot_batch, n_steps=60)

logging.info("Part 4 done: status hook fired %d times", snapshot_hook._print_count)
