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
Three-Stage Pipeline: FIRE → NVT Equilibration → NVE Production
================================================================

This example builds a fully-featured three-stage simulation pipeline and
demonstrates every major capability of the :mod:`nvalchemi.dynamics`
framework in one runnable script.

**Pipeline stages**

* **Stage 0 — FIRE relaxation** (:class:`~nvalchemi.dynamics.optimizers.FIRE`):
  Systems start here.  A :class:`~nvalchemi.dynamics.base.ConvergenceHook`
  monitors fmax; once fmax < 0.05 a/u the system migrates to stage 1.
* **Stage 1 — NVT equilibration** (:class:`~nvalchemi.dynamics.integrators.NVTLangevin`):
  Systems are thermalized at the target temperature.  The sub-stage's
  ``n_steps`` attribute controls how many steps each system spends here
  before auto-migrating to stage 2.
* **Stage 2 — NVE production** (:class:`~nvalchemi.dynamics.integrators.NVE`):
  Microcanonical production run.  A second ``n_steps`` budget migrates
  systems to the exit status (3) after ``PROD_STEPS_PER_SYSTEM`` steps,
  writing them to a :class:`~nvalchemi.dynamics.sinks.HostMemory` sink.

All three stages share **one** model forward pass per fused step via a
:class:`~nvalchemi.dynamics.base.FusedStage`.  The three sub-stages are
composed with the ``+`` operator::

    fused = fire_stage + equil_stage + prod_stage

**Features demonstrated**

1. Three-stage :class:`~nvalchemi.dynamics.base.FusedStage` composition
   via chained ``+`` (``dyn_a + dyn_b + dyn_c``).
2. **Custom hook classes** (not inline lambdas):

   * :class:`EnergyTracker` — maintains a running per-system energy
     mean and logs it at configurable frequency.
   * :class:`StageTransitionLogger` — fires ``ON_CONVERGE`` on a
     sub-stage and prints timing and step information.

3. :meth:`~nvalchemi.dynamics.base.ConvergenceHook.from_forces` factory
   for force-norm-based FIRE → NVT migration.
4. :class:`~nvalchemi.dynamics.sinks.HostMemory` sink to collect
   completed production trajectories.
5. **Inflight batching** via :class:`~nvalchemi.dynamics.sampler.SizeAwareSampler`:
   graduated samples are replaced on-the-fly from a dataset, keeping
   the batch saturated.  The sampler stamps a monotonic ``system_id``
   on every sample so systems can be tracked across inflight refills.
6. ``batch.status`` inspection throughout the run using
   :meth:`~nvalchemi.dynamics.base.FusedStage.register_fused_hook`.
7. ``n_steps`` per sub-stage for automatic step-budget migration,
   replacing the need for a custom step-counting hook.
8. ``init_fn=`` parameter to :class:`~nvalchemi.dynamics.base.FusedStage`
   for pre-step batch initialization in Mode 2 inflight runs.

.. note::

    :class:`~nvalchemi.models.demo.DemoModelWrapper` is used throughout.
    It computes per-atom energies and conservative forces via autograd but
    does not produce stresses, so variable-cell integrators are not shown.
"""

from __future__ import annotations

import time
from collections import defaultdict

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE, NVE, NVTLangevin
from nvalchemi.dynamics.base import (
    BaseDynamics,
    ConvergenceHook,
    FusedStage,
    HookStageEnum,
)
from nvalchemi.dynamics.sampler import SizeAwareSampler
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.models.demo import DemoModelWrapper

# %%
# Configuration constants
# -----------------------
# Centralise all tuneable parameters here so they are easy to adjust.

# Fmax threshold for FIRE → NVT transition (arbitrary units)
FIRE_FMAX_THRESHOLD = 0.05

# Steps each system must spend in the NVT equilibration stage before moving on.
# In a real workflow this would be thousands of steps; kept small here for speed.
EQUIL_STEPS_PER_SYSTEM = 10

# Steps each system must spend in the NVE production stage.
PROD_STEPS_PER_SYSTEM = 20

# Target temperature for the NVT Langevin thermostat (Kelvin, demo units)
TEMPERATURE = 300.0

# %%
# Setup — model and system builder
# ---------------------------------
# :class:`~nvalchemi.models.demo.DemoModelWrapper` computes per-atom energies
# and conservative forces via :func:`torch.autograd.grad`.

torch.manual_seed(42)
model = DemoModelWrapper()
model.eval()


def _make_system(n_atoms: int, seed: int) -> AtomicData:
    """Build a small :class:`~nvalchemi.data.AtomicData` system.

    All integrators need ``positions``, ``atomic_numbers``, ``atomic_masses``,
    ``velocities`` (node-level), and zero-placeholder ``forces`` /
    ``energies`` that are overwritten in-place by ``compute()`` every step.

    Parameters
    ----------
    n_atoms : int
        Number of atoms in the system.
    seed : int
        Random seed for reproducible initial geometry.

    Returns
    -------
    AtomicData
        A ready-to-use system for dynamics.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    data = AtomicData(
        positions=torch.randn(n_atoms, 3, generator=g),
        atomic_numbers=torch.randint(1, 10, (n_atoms,), dtype=torch.long, generator=g),
        atomic_masses=torch.ones(n_atoms),
        forces=torch.zeros(n_atoms, 3),
        energies=torch.zeros(1, 1),
    )
    # Velocities is not a standard AtomicData field; add it as a node property.
    data.add_node_property("velocities", torch.zeros(n_atoms, 3))
    return data


# %%
# Custom Hook 1 — EnergyTracker
# ------------------------------
# This hook tracks the running per-system mean energy over the simulation.
# It fires at ``AFTER_COMPUTE`` so energies are always fresh from the model.
# On every ``frequency`` steps it prints a summary table.
#
# Implements the :class:`~nvalchemi.dynamics.base.Hook` protocol:
# must have ``stage``, ``frequency``, and ``__call__(batch, dynamics)``.


class EnergyTracker:
    """Track and report running per-system mean energy.

    Accumulates per-system energy samples after each compute call and
    reports the running mean every ``frequency`` steps.

    Parameters
    ----------
    frequency : int
        How often to print the energy summary.  A value of 5 prints
        every 5th step (i.e. at steps 0, 5, 10, …).
    label : str
        A short label printed in the header line for readability.

    Attributes
    ----------
    stage : HookStageEnum
        Fixed to :attr:`~HookStageEnum.AFTER_COMPUTE`.
    frequency : int
        Report frequency.
    label : str
        Display label.
    _sums : dict[int, float]
        Accumulated energy sum per system index (within the current batch).
    _counts : dict[int, int]
        Number of energy samples accumulated per system.
    """

    stage = HookStageEnum.AFTER_COMPUTE

    def __init__(self, frequency: int = 5, label: str = "energy") -> None:
        self.frequency = frequency
        self.label = label
        # Accumulated statistics.  Keys are 0-based system indices within
        # the batch at the time of hook firing (indices are NOT stable across
        # inflight replacements; treat them as slot indices only).
        self._sums: dict[int, float] = defaultdict(float)
        self._counts: dict[int, int] = defaultdict(int)

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Accumulate per-system energies and print a summary periodically.

        Parameters
        ----------
        batch : Batch
            The current batch; ``batch.energies`` is expected.
        dynamics : BaseDynamics
            The dynamics engine, used to access ``step_count``.
        """
        if batch.energies is None:
            return

        # energies shape may be (num_graphs, 1) or (num_graphs, 1, 1); flatten to 1-D
        energies = batch.energies.detach().cpu().view(-1)
        for i, e in enumerate(energies.tolist()):
            self._sums[i] += e
            self._counts[i] += 1

        if dynamics.step_count % self.frequency == 0:
            n_systems = batch.num_graphs
            means = [self._sums[i] / max(self._counts[i], 1) for i in range(n_systems)]
            status = (
                batch.status.squeeze(-1).tolist()
                if batch.status is not None
                else ["?"] * n_systems
            )
            print(
                f"  [{self.label}] step={dynamics.step_count:4d}"
                f"  n_systems={n_systems}"
                f"  mean_E={[f'{e:.3f}' for e in means]}"
                f"  status={status}"
            )

    @property
    def total_samples(self) -> int:
        """Total number of (system, step) pairs accumulated."""
        return sum(self._counts.values())


# %%
# Custom Hook 2 — StageTransitionLogger
# ---------------------------------------
# This hook fires at ``ON_CONVERGE`` (inside the FIRE sub-stage) and logs
# the step number plus wall-clock time whenever systems migrate from the
# FIRE stage to the NVT equilibration stage.
#
# ``ON_CONVERGE`` fires **after** the :class:`~nvalchemi.dynamics.base.ConvergenceHook`
# has already updated ``batch.status``; we can therefore count how many
# systems are now at ``target_status``.


class StageTransitionLogger:
    """Log system counts and timing when a stage transition is triggered.

    Fires at :attr:`~HookStageEnum.ON_CONVERGE` and reports the step,
    elapsed wall-clock time since construction, and a breakdown of how
    many systems are currently at each status code.

    Parameters
    ----------
    label : str
        Short description of the transition being logged
        (e.g. ``"FIRE→NVT"``).
    frequency : int
        How often to emit output.  Default 1 (every trigger).

    Attributes
    ----------
    stage : HookStageEnum
        Fixed to :attr:`~HookStageEnum.ON_CONVERGE`.
    frequency : int
        Emit frequency.
    label : str
        Display label.
    _t0 : float
        Wall-clock time at hook construction.
    _n_transitions : int
        Cumulative count of convergence events logged.
    """

    stage = HookStageEnum.ON_CONVERGE

    def __init__(self, label: str = "transition", frequency: int = 1) -> None:
        self.frequency = frequency
        self.label = label
        self._t0 = time.monotonic()
        self._n_transitions = 0

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Print a status-distribution summary at each convergence event.

        Parameters
        ----------
        batch : Batch
            The current batch (``batch.status`` is read).
        dynamics : BaseDynamics
            The dynamics engine; ``step_count`` identifies the step.
        """
        self._n_transitions += 1
        if self._n_transitions % self.frequency != 0:
            return

        elapsed = time.monotonic() - self._t0

        # Compute status distribution
        if batch.status is not None:
            status_vec = batch.status.squeeze(-1).tolist()
            dist: dict[int, int] = defaultdict(int)
            for s in status_vec:
                dist[int(s)] += 1
            dist_str = "  ".join(f"status={k}:{v}" for k, v in sorted(dist.items()))
        else:
            dist_str = "status=unknown"

        print(
            f"  [{self.label}] CONVERGE at step={dynamics.step_count:4d}"
            f"  elapsed={elapsed:.2f}s"
            f"  {dist_str}"
        )

    @property
    def n_transitions(self) -> int:
        """Number of convergence events logged so far."""
        return self._n_transitions


# %%
# Part 1 — Standalone FIRE relaxation (baseline)
# -----------------------------------------------
# First we relax a small batch with plain FIRE to establish a baseline
# and confirm that the model + FIRE integrator work correctly in
# isolation before composing the three-stage pipeline.
#
# :meth:`~nvalchemi.dynamics.base.ConvergenceHook.from_forces` is a
# convenience factory that constructs the force-norm criterion directly.

print("=" * 60)
print("Part 1: Standalone FIRE Relaxation")
print("=" * 60)

data_list_single = [_make_system(n, seed) for n, seed in [(4, 1), (5, 2), (6, 3)]]
batch_single = Batch.from_data_list(data_list_single)
print(f"Batch: {batch_single.num_graphs} systems, {batch_single.num_nodes} atoms total")

fire_standalone = FIRE(
    model=model,
    dt=0.1,
    n_steps=150,
    # ConvergenceHook.from_forces() reads the 'forces' key, computes the
    # per-atom vector norm, scatter-maxes to per-system fmax, and marks
    # systems with fmax <= threshold as converged.
    convergence_hook=ConvergenceHook.from_forces(FIRE_FMAX_THRESHOLD),
)

batch_single = fire_standalone.run(batch_single)
print(f"Standalone FIRE completed {fire_standalone.step_count} steps.\n")


# %%
# Part 2 — Three-Stage FusedStage Pipeline
# -----------------------------------------
# Create FIRE + NVT Langevin + NVE and fuse them with the ``+`` operator.
#
# How ``+`` composes three stages
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# * ``fire + equil`` creates ``FusedStage([(0, fire), (1, equil)], exit_status=2)``
#   and auto-registers a ``ConvergenceHook(0→1)`` on ``fire`` that checks the
#   same force-norm criterion as ``fire.convergence_hook``.
# * ``(fire + equil) + prod`` creates a NEW ``FusedStage([(0, fire), (1, equil), (2, prod)],
#   exit_status=3)``.
#
# Step-budget migration via ``n_steps``
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Rather than writing a custom step-counting hook, set ``n_steps`` on each
# sub-stage.  :class:`~nvalchemi.dynamics.base.FusedStage` auto-registers a
# per-system step counter and migrates systems to the next stage when they
# accumulate ``n_steps`` steps in their current stage.  This replaces the
# old ``StepCountMigrationHook`` workaround entirely.

print("=" * 60)
print("Part 2: Three-Stage FusedStage Pipeline")
print("=" * 60)

# ---- Stage 0: FIRE ----
# ConvergenceHook.from_forces() drives both standalone FIRE convergence detection
# AND the auto-registered 0→1 migration hook that FusedStage creates.
# FusedStage inherits the criteria from fire_stage.convergence_hook, so the
# same force-norm threshold governs the inter-stage migration.
fire_stage = FIRE(
    model=model,
    dt=0.1,
    convergence_hook=ConvergenceHook.from_forces(FIRE_FMAX_THRESHOLD),
    device_type="cpu",
)

# ---- Stage 1: NVT Langevin equilibration ----
# n_steps=EQUIL_STEPS_PER_SYSTEM tells FusedStage to auto-migrate each system
# from status 1 → 2 after it has accumulated that many steps in this stage.
# No manual step-counting hook is needed.
equil_stage = NVTLangevin(
    model=model,
    dt=0.5,
    temperature=TEMPERATURE,
    friction=0.1,
    random_seed=7,
    n_steps=EQUIL_STEPS_PER_SYSTEM,
    device_type="cpu",
)

# ---- Stage 2: NVE production ----
# n_steps=PROD_STEPS_PER_SYSTEM auto-migrates each system from status 2 → 3
# (exit) after accumulating that many steps.
prod_stage = NVE(
    model=model,
    dt=0.5,
    n_steps=PROD_STEPS_PER_SYSTEM,
    device_type="cpu",
)

# --- Register custom hooks on the sub-stages BEFORE fusing ---

# EnergyTracker on the FIRE sub-stage: print energy every 10 steps
fire_energy_tracker = EnergyTracker(frequency=10, label="FIRE-energy")
fire_stage.register_hook(fire_energy_tracker)

# StageTransitionLogger on FIRE: fires ON_CONVERGE when fmax threshold met
fire_transition_logger = StageTransitionLogger(label="FIRE→NVT", frequency=1)
fire_stage.register_hook(fire_transition_logger)

# EnergyTracker on the NVT sub-stage: print energy every 5 steps
equil_energy_tracker = EnergyTracker(frequency=5, label="NVT-energy")
equil_stage.register_hook(equil_energy_tracker)

# ---- Fuse stages with the ``+`` operator ----
# fire_stage + equil_stage creates FusedStage([(0, fire), (1, equil)], exit_status=2)
# and auto-registers ConvergenceHook(0→1) on fire_stage.
# Then (fire + equil) + prod_stage creates FusedStage([(0,fire),(1,equil),(2,prod)],
# exit_status=3).  n_steps budgets on equil and prod drive 1→2 and 2→3 migration.
fused = fire_stage + equil_stage + prod_stage
print(f"Created: {fused}")
print(f"  exit_status = {fused.exit_status}  (0=FIRE, 1=NVT, 2=NVE, 3=done)\n")

# ---- Build the initial batch ----
# All systems start at status = 0 (FIRE stage).
# _ensure_bookkeeping_fields (called inside FusedStage.run) auto-initializes
# the 'status' field, so no manual batch["status"] = ... is needed here.
data_list_fused = [
    _make_system(n, seed) for n, seed in [(4, 10), (6, 11), (5, 12), (4, 13)]
]
batch_fused = Batch.from_data_list(data_list_fused)

print(
    f"Initial batch: {batch_fused.num_graphs} systems  "
    f"{batch_fused.num_nodes} atoms total\n"
)

# ---- Run the fused pipeline ----
# FusedStage.run() loops until all_complete() (all systems reach exit_status=3)
# or n_steps is exhausted.  We pass n_steps as a safety bound.
MAX_FUSED_STEPS = 500

print("Running fused 3-stage pipeline …")
batch_fused = fused.run(batch_fused, n_steps=MAX_FUSED_STEPS)

status_final = batch_fused.status.squeeze(-1).tolist()
print(f"\nFused pipeline complete after {fused.step_count} steps.")
print(f"  Final status: {status_final}  (3 = exited pipeline)")
print(
    f"  EnergyTracker (FIRE stage) accumulated {fire_energy_tracker.total_samples} samples"
)
print(
    f"  EnergyTracker (NVT stage) accumulated {equil_energy_tracker.total_samples} samples"
)
print(
    f"  StageTransitionLogger (FIRE→NVT) saw {fire_transition_logger.n_transitions} ON_CONVERGE events\n"
)


# %%
# Part 3 — Inflight Batching with a Sink
# ----------------------------------------
# This section shows Mode 2 of :meth:`~nvalchemi.dynamics.base.FusedStage.run`:
# no external batch is passed; instead a :class:`~nvalchemi.dynamics.sampler.SizeAwareSampler`
# manages the dataset, and :class:`~nvalchemi.dynamics.sinks.HostMemory` stores
# graduated systems.
#
# SizeAwareSampler interface
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# The sampler wraps a dataset that exposes:
#   * ``__len__``
#   * ``__getitem__(idx) -> (AtomicData, dict)``
#   * ``get_metadata(idx) -> (num_atoms, num_edges)``
#
# :meth:`~nvalchemi.dynamics.sampler.SizeAwareSampler.build_initial_batch` greedily
# packs the first batch subject to atom/edge/size constraints.
# Every sample — both at build time and when replacing graduated systems — is
# stamped with a monotonic ``system_id`` integer so systems can be tracked
# across inflight refills without relying on batch slot positions.
#
# Mode 2 with ``init_fn``
# ~~~~~~~~~~~~~~~~~~~~~~~
# Pass ``batch=None`` to :meth:`~nvalchemi.dynamics.base.FusedStage.run` and
# supply an ``init_fn`` to the :class:`~nvalchemi.dynamics.base.FusedStage`
# constructor.  ``init_fn`` receives the sampler-built batch before the first
# step, making it easy to add fields (velocities, etc.) that the sampler does
# not know about.
#
# Inflight mode terminates when the sampler is exhausted AND all current systems
# have graduated, returning ``None``.

print("=" * 60)
print("Part 3: Inflight Batching")
print("=" * 60)


class SimpleDataset:
    """Minimal dataset for the inflight batching demo.

    Each sample is a random single-stage system with a fixed atom count.

    Parameters
    ----------
    n_samples : int
        Number of samples in the dataset.
    atoms_per_sample : int
        Number of atoms per system.
    seed : int
        Base random seed; each sample offsets by its index.

    Notes
    -----
    Implements the three-method interface required by
    :class:`~nvalchemi.dynamics.sampler.SizeAwareSampler`:
    ``__len__``, ``__getitem__``, and ``get_metadata``.
    No edges are used (``num_edges=0``) because
    :class:`~nvalchemi.models.demo.DemoModelWrapper` does not need a
    neighbor list.
    """

    def __init__(self, n_samples: int, atoms_per_sample: int, seed: int = 0) -> None:
        self.n_samples = n_samples
        self.atoms_per_sample = atoms_per_sample
        self.base_seed = seed

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.n_samples

    def get_metadata(self, idx: int) -> tuple[int, int]:
        """Return ``(num_atoms, num_edges)`` for sample ``idx``.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple[int, int]
            Atom and edge count.
        """
        return self.atoms_per_sample, 0

    def __getitem__(self, idx: int) -> tuple[AtomicData, dict]:
        """Load and return sample ``idx`` as an :class:`~nvalchemi.data.AtomicData`.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple[AtomicData, dict]
            The sample and an empty metadata dict.
        """
        g = torch.Generator()
        g.manual_seed(self.base_seed + idx)
        data = AtomicData(
            positions=torch.randn(self.atoms_per_sample, 3, generator=g),
            atomic_numbers=torch.randint(
                1, 10, (self.atoms_per_sample,), dtype=torch.long, generator=g
            ),
            atomic_masses=torch.ones(self.atoms_per_sample),
            forces=torch.zeros(self.atoms_per_sample, 3),
            energies=torch.zeros(1, 1),
        )
        data.add_node_property("velocities", torch.zeros(self.atoms_per_sample, 3))
        return data, {}


# Build a dataset with 20 systems; batch capacity = 4
N_DATASET = 20
BATCH_CAPACITY = 4
ATOMS_PER_SYSTEM = 4

dataset = SimpleDataset(
    n_samples=N_DATASET, atoms_per_sample=ATOMS_PER_SYSTEM, seed=100
)
sampler = SizeAwareSampler(
    dataset,
    max_atoms=BATCH_CAPACITY * ATOMS_PER_SYSTEM,
    max_edges=None,
    max_batch_size=BATCH_CAPACITY,
)

print(
    f"Dataset: {len(dataset)} systems  |  "
    f"batch capacity: {BATCH_CAPACITY}  |  "
    f"atoms/system: {ATOMS_PER_SYSTEM}"
)
print(f"Inflight mode: {sampler is not None}\n")

# ---- Completed-trajectory sink ----
# HostMemory stores each completed system as an AtomicData.
trajectory_sink = HostMemory(capacity=N_DATASET)

# ---- Build a two-stage inflight pipeline with FIRE + NVTLangevin ----
# Use the real FIRE and NVTLangevin integrators — no BaseDynamics no-op workaround
# needed.  n_steps on each sub-stage controls how many steps each system spends
# there before auto-migrating to the next stage.
#
# For the inflight FIRE stage we use a permissive force threshold (10.0) so that
# all systems from the DemoModelWrapper converge quickly in this demonstration.
stage0_inflight = FIRE(
    model=model,
    dt=0.1,
    convergence_hook=ConvergenceHook.from_forces(threshold=10.0),
    device_type="cpu",
)

stage1_inflight = NVTLangevin(
    model=model,
    dt=0.5,
    temperature=TEMPERATURE,
    friction=0.1,
    random_seed=77,
    n_steps=EQUIL_STEPS_PER_SYSTEM,  # Auto-migrate after N steps
    device_type="cpu",
)


# ``init_fn`` is called once on the sampler-built batch before the first step.
# Use it to populate any fields that the sampler does not know about.
# The sampler already adds forces, energies, velocities (via __getitem__),
# so this example uses init_fn to demonstrate the API.
def _inflight_init(batch: Batch) -> None:
    """Demonstrate init_fn: print initial system_ids stamped by the sampler."""
    if hasattr(batch, "system_id") and batch.system_id is not None:
        ids = batch.system_id.squeeze(-1).tolist()
        print(f"  init_fn: initial batch system_ids = {ids}")


# Build the FusedStage in Mode 2 (batch=None): pass sampler, sinks, and init_fn.
# FusedStage.run(batch=None) calls sampler.build_initial_batch() internally,
# then calls init_fn on the result before the first step.
fused_inflight = FusedStage(
    sub_stages=[(0, stage0_inflight), (1, stage1_inflight)],
    sampler=sampler,
    sinks=[trajectory_sink],
    refill_frequency=5,  # Check for graduated samples every 5 steps
    init_fn=_inflight_init,
    device_type="cpu",
)
print(f"Created inflight FusedStage: {fused_inflight}")
print(f"  inflight_mode = {fused_inflight.inflight_mode}  (sampler attached)")

# ---- Run in Mode 2 (batch=None) ----
# FusedStage builds the initial batch internally via the sampler,
# calls init_fn, and loops until the sampler is exhausted and all
# remaining systems have graduated.
MAX_INFLIGHT_STEPS = 300

print("\nRunning inflight batching pipeline (Mode 2, batch=None) …")
result_inflight = fused_inflight.run(batch=None, n_steps=MAX_INFLIGHT_STEPS)

print("\nInflight pipeline complete.")
print(f"  Fused steps executed: {fused_inflight.step_count}")
print(f"  Trajectory sink contains: {len(trajectory_sink)} completed systems")
print(f"  Sampler exhausted: {sampler.exhausted}")

if result_inflight is not None:
    final_status_inflight = result_inflight.status.squeeze(-1).tolist()
    print(f"  Remaining batch status: {final_status_inflight}")
else:
    print("  run() returned None — sampler exhausted and all systems graduated.")

# ---- Inspect collected trajectories and system_ids ----
# The sampler stamps each sample with a monotonic system_id integer.
# This allows tracking systems across inflight refills independent of
# their slot position in the batch.
if len(trajectory_sink) > 0:
    graduated = trajectory_sink.read()
    print(f"\nCollected {len(trajectory_sink)} completed systems from sink.")
    if hasattr(graduated, "system_id") and graduated.system_id is not None:
        system_ids = graduated.system_id.squeeze(-1).tolist()
        print(f"  Graduated system_ids: {system_ids}")
    if hasattr(graduated, "status") and graduated.status is not None:
        print(f"  Status values: {graduated.status.squeeze(-1).tolist()}")


# %%
# Part 4 — Status Inspection Throughout the Run
# -----------------------------------------------
# This section shows how to inspect ``batch.status`` at each fused step
# using :meth:`~nvalchemi.dynamics.base.FusedStage.register_fused_hook`.
#
# Unlike hooks registered on individual sub-stages (which only observe the
# sub-batched slice), a **fused hook** sees the **complete** batch at
# ``BEFORE_STEP`` and ``AFTER_STEP`` on every fused step, regardless of which
# sub-stage had active systems.  This is the correct way to monitor global
# pipeline state.

print("\n" + "=" * 60)
print("Part 4: Step-by-step Status Inspection via register_fused_hook")
print("=" * 60)


class StatusSnapshotHook:
    """Print per-step status distribution for diagnostic purposes.

    Parameters
    ----------
    frequency : int
        How often to print.  Default 2 (every other step).
    max_steps : int
        Stop printing after this many prints to avoid terminal flood.

    Attributes
    ----------
    stage : HookStageEnum
        Fixed to :attr:`~HookStageEnum.AFTER_STEP`.
    frequency : int
        Print frequency.
    """

    stage = HookStageEnum.AFTER_STEP

    def __init__(self, frequency: int = 2, max_steps: int = 10) -> None:
        self.frequency = frequency
        self._print_count = 0
        self._max_steps = max_steps

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Print a one-line status summary.

        Parameters
        ----------
        batch : Batch
            The current batch.
        dynamics : BaseDynamics
            The dynamics engine.
        """
        if self._print_count >= self._max_steps:
            return
        if dynamics.step_count % self.frequency != 0:
            return

        if batch.status is not None:
            status = batch.status.squeeze(-1).tolist()
            dist: dict[int, int] = defaultdict(int)
            for s in status:
                dist[int(s)] += 1
            dist_str = " | ".join(f"s{k}:{v}" for k, v in sorted(dist.items()))
        else:
            dist_str = "no status"

        # Show energy mean if available
        e_str = ""
        if batch.energies is not None:
            e_mean = batch.energies.squeeze(-1).mean().item()
            e_str = f"  E_mean={e_mean:.4f}"

        print(f"  step={dynamics.step_count:3d}  [{dist_str}]{e_str}")
        self._print_count += 1


# Short inspection run with 2 systems and verbose status output
snapshot_data = [_make_system(5, 20), _make_system(5, 21)]
snapshot_batch = Batch.from_data_list(snapshot_data)
# No manual batch["status"] init needed: _ensure_bookkeeping_fields
# auto-initializes status=0 for all systems before the first step.

fire_inspect = FIRE(
    model=model,
    dt=0.1,
    n_steps=30,
    convergence_hook=ConvergenceHook.from_forces(FIRE_FMAX_THRESHOLD),
    device_type="cpu",
)

nvt_inspect = NVTLangevin(
    model=model,
    dt=0.5,
    temperature=TEMPERATURE,
    friction=0.1,
    random_seed=99,
    n_steps=10,  # Auto-migrate status 1 → 2 after 10 steps
    device_type="cpu",
)

fused_inspect = fire_inspect + nvt_inspect

# Register the snapshot hook at the FusedStage level via register_fused_hook().
# It fires on the full batch after every fused step, regardless of which
# sub-stage was active this step.
snapshot_hook = StatusSnapshotHook(frequency=2, max_steps=12)
fused_inspect.register_fused_hook(snapshot_hook)

print("Running short inspection run (status printed every 2 steps) …\n")
snapshot_batch = fused_inspect.run(snapshot_batch, n_steps=60)
print(
    f"\nInspection run done. final status={snapshot_batch.status.squeeze(-1).tolist()}"
)


# %%
# Remaining Limitations
# =====================
#
# The following items are non-goals or genuinely unresolved at this time.
#
# 1. **Three-stage composition silently creates two FusedStage instances.**
#    ``dyn_a + dyn_b + dyn_c`` first creates ``FusedStage([0,1])`` and then
#    creates a new ``FusedStage([0,1,2])``.  Hooks registered on the
#    intermediate FusedStage (if any) are silently discarded.  The docstring
#    for ``__add__`` mentions this but it is easy to miss.
#
# 2. **``FusedStage.run()`` vs ``BaseDynamics.run()`` return-type contract differs.**
#    ``BaseDynamics.run()`` always returns a ``Batch``.  ``FusedStage.run()``
#    can return ``None`` (when the sampler is exhausted).  Callers must check
#    for ``None`` even though the type signature says ``Batch | None`` only on
#    ``FusedStage``.
#
# 3. **``torch.compile`` full compatibility is a non-goal for now.**
#    ``FusedStage`` supports ``compile_step=True`` / ``.compile()`` on the inner
#    ``_step_impl`` kernel, but graph breaks from Python-level hook dispatch
#    and status masking limit end-to-end compilation of the full run loop.
