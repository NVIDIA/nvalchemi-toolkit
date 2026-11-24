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
UMA Hybrid SGC-NPT Simulation
==============================

This input script runs independent periodic Au-Pt walkers through alternating
semi-grand-canonical Monte Carlo (SGC) and isotropic NPT molecular-dynamics
blocks. The walkers share one batched UMA model evaluation but remain separate
Markov chains.

Install the dedicated UMA environment and authenticate with Hugging Face::

    UV_PROJECT_ENVIRONMENT=.venv-uma uv sync --extra uma-cu12 --extra ase --no-group build
    # or, with pip:  pip install 'nvalchemi-toolkit[uma-cu12,ase]'
    hf auth login  # or export HF_TOKEN

This is a short API example, not a converged phase-diagram calculation. The
optional CUDA pre-run selects a productive batch width for this exact model,
structure, and MC-MD protocol. Production work still requires equilibrated,
independently seeded replicas.
"""

from __future__ import annotations

import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import torch
from ase import Atoms
from ase.build import bulk

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.hybrid import HybridMCMD
from nvalchemi.mc import SGC
from nvalchemi.models.uma import UMAWrapper
from nvalchemi.scheduling import (
    CampaignScheduler,
    CampaignSpec,
    FinalStateStore,
    RunSpec,
    SimulationBatchPlanner,
)

# %%
# Simulation input
# ----------------
# Each quantity below is an explicit input to the calculation. Batch width is
# the number of independent trajectories, not the number of atoms in one cell.

CHECKPOINT = "uma-s-1p2"
TASK = "omat"
# SGC changes the composition every step, so no MoLE merge, and no compile
# since MD changes the geometry; energy-only MC is enabled in HybridMCMD below.
INFERENCE_SETTINGS = (
    "compile=false,merge_mole=false,tf32=true,activation_checkpointing=false"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ASE crystal template. ``bulk`` supports common elemental structures such as
# ``fcc``, ``bcc``, ``sc``, ``diamond``, and ``hcp``. Use ``cubic=False`` for
# structures without a conventional cubic cell. Replace ``build_ase_structure``
# with ``ase.io.read`` to begin from a supplied cell.
TEMPLATE_SYMBOL = "Au"
CRYSTAL_STRUCTURE = "fcc"
LATTICE_A_ANG = 4.00
SUPERCELL_REPEATS = (2, 2, 2)  # 32 atoms for conventional FCC
CONVENTIONAL_CELL = True
PT_FRACTION = 0.50
PRESSURE_EV_PER_A3 = 1.01325 / 1.602176634e6  # 1 atmosphere

# Campaign definition: bidirectional high-temperature delta-mu scans followed
# by cooling continuation at fixed delta_mu. Each run has its own temperature
# and reservoir; the batched SGC and NPT stages consume them as per-graph tensors.
REFERENCE_TEMPERATURE_K = 3000.0
COOLING_TEMPERATURES_K = (3000.0, 2800.0, 2600.0)
REFERENCE_DELTA_MU_EV = (-0.10, 0.0, 0.10)
# A fresh directory per run of this example. For production use a persistent
# path: completed runs found there are skipped, so an interrupted campaign resumes.
CHECKPOINT_DIRECTORY = Path(tempfile.mkdtemp(prefix="uma_hybrid_sgc_npt_"))

MC_STEPS_PER_BLOCK = 10
MD_STEPS_PER_BLOCK = 20
N_BLOCKS_PER_RUN = 5
DT_FS = 1.0
THERMOSTAT_TIME_FS = 100.0
BAROSTAT_TIME_FS = 1000.0
SEED = 20260826

# GPU batch-width selection. Keep a manual fallback for CPU runs, debugging,
# or a previously benchmarked production width. The CUDA profile uses the same
# structure, model, NPT settings, and MC/MD block lengths as production.
AUTO_SELECT_BATCH_WIDTH = True
MANUAL_BATCH_WIDTH = 4
BATCH_WIDTH_CANDIDATES = (1, 2, 4, 8, 16)
BATCH_MEMORY_FRACTION = 0.85
BATCH_THROUGHPUT_FRACTION = 0.95
PROFILE_WARMUP_BLOCKS = 1
PROFILE_MEASURED_BLOCKS = 2


def build_ase_structure(
    symbol: str,
    crystal_structure: str,
    lattice_a: float,
    repeats: tuple[int, int, int],
    *,
    cubic: bool,
) -> Atoms:
    """Build a periodic ASE crystal template for one independent run."""
    unit_cell = bulk(
        symbol,
        crystalstructure=crystal_structure,
        a=lattice_a,
        cubic=cubic,
    )
    return unit_cell * repeats


def _walker(
    template: Atoms,
    run: RunSpec,
    seed: int,
    parent_state: AtomicData | None,
) -> AtomicData:
    """Create a fresh state or restore the parent state for continuation."""
    if parent_state is not None:
        return parent_state.to(DEVICE)

    data = AtomicData.from_atoms(template, device=DEVICE)
    n_atoms = data.num_nodes
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    numbers = torch.full_like(data.atomic_numbers, 79)
    pt_fraction = float(run.metadata.get("pt_fraction", PT_FRACTION))
    pt_count = round(pt_fraction * n_atoms)
    numbers[torch.randperm(n_atoms, device=DEVICE, generator=generator)[:pt_count]] = 78
    data.atomic_numbers = numbers
    data.atomic_masses = None
    data.use_default_masses()
    velocity_std = torch.sqrt(
        torch.as_tensor(8.617333262145e-5 * run.temperature_k, device=DEVICE)
        / data.atomic_masses
    )
    data.velocities = (
        torch.randn((n_atoms, 3), device=DEVICE, generator=generator)
        * velocity_std[:, None]
    )
    data.velocities -= data.velocities.mean(dim=0, keepdim=True)
    data.forces = torch.zeros_like(data.positions)
    data.energy = torch.zeros(1, 1, device=DEVICE)
    data.stress = torch.zeros(1, 3, 3, device=DEVICE)
    return data


def _make_batch(
    runs: tuple[RunSpec, ...],
    parent_states: tuple[AtomicData | None, ...],
) -> Batch:
    """Build a batch of fresh states or restored continuation states."""
    if len(runs) != len(parent_states):
        raise ValueError("runs and parent_states must have equal lengths")
    template = build_ase_structure(
        TEMPLATE_SYMBOL,
        CRYSTAL_STRUCTURE,
        LATTICE_A_ANG,
        SUPERCELL_REPEATS,
        cubic=CONVENTIONAL_CELL,
    )
    return Batch.from_data_list(
        [
            _walker(template, run, SEED + index, parent_state)
            for index, (run, parent_state) in enumerate(zip(runs, parent_states))
        ]
    )


# %%
# Build the model, then determine the batch width
# -----------------------------------------------
# The checkpoint is gated on Hugging Face; without access the example exits
# early with the setup hint.

try:
    model = UMAWrapper.from_checkpoint(
        CHECKPOINT,
        task_name=TASK,
        device=str(DEVICE),
        inference_settings=INFERENCE_SETTINGS,
    )
except Exception as exc:  # noqa: BLE001
    sys.exit(
        f"Could not load UMA ({exc}). Install 'nvalchemi-toolkit[uma]', request "
        "access to the gated 'facebook/UMA' repo, and authenticate via "
        "'hf auth login' or HF_TOKEN."
    )

# %%
# Assemble a workload factory and select a production batch width
# ---------------------------------------------------------------
# The same UMA object is deliberately passed to both stages. This is required
# by ``HybridMCMD`` so every trial energy and MD force is supplied by one model.


def make_workload(
    runs: tuple[RunSpec, ...],
    parent_states: tuple[AtomicData | None, ...],
    device: torch.device,
) -> tuple[HybridMCMD, Batch]:
    """Return one per-graph thermodynamic MC-MD workload for compatible runs."""
    if device != DEVICE:
        raise ValueError(f"Expected profiling on {DEVICE}, received {device}")
    if not runs:
        raise ValueError("a workload requires at least one run")
    species = runs[0].species
    if any(run.species != species for run in runs):
        raise ValueError("all runs in a workload must share reservoir species")
    temperatures = torch.tensor([run.temperature_k for run in runs], device=DEVICE)
    pressures = torch.tensor([run.pressure_ev_per_a3 for run in runs], device=DEVICE)
    chemical_potentials = {
        number: torch.tensor(
            [run.chemical_potentials_ev[number] for run in runs],
            device=DEVICE,
        )
        for number in species
    }
    sgc = SGC(
        model=model,
        temperature=temperatures,
        species=species,
        chemical_potentials=chemical_potentials,
        random_seed=SEED,
    )
    npt = NPT(
        model=model,
        dt=DT_FS,
        temperature=temperatures,
        pressure=pressures,
        thermostat_time=THERMOSTAT_TIME_FS,
        barostat_time=BAROSTAT_TIME_FS,
        pressure_coupling="isotropic",
    )
    return (
        HybridMCMD(
            mc=sgc,
            md=npt,
            mc_steps=MC_STEPS_PER_BLOCK,
            md_steps=MD_STEPS_PER_BLOCK,
            mc_energy_only=True,
        ),
        _make_batch(runs, parent_states),
    )


def _reference_scan(
    direction: str, delta_mu_values: tuple[float, ...]
) -> tuple[RunSpec, ...]:
    """Build one high-temperature delta-mu continuation path."""
    parent_id = None
    endpoint_pt_fraction = 0.02 if direction == "up" else 0.98
    runs: list[RunSpec] = []
    for index, delta_mu in enumerate(delta_mu_values):
        run = RunSpec(
            run_id=f"reference.{direction}.mu_{index}",
            temperature_k=REFERENCE_TEMPERATURE_K,
            pressure_ev_per_a3=PRESSURE_EV_PER_A3,
            chemical_potentials_ev={79: 0.0, 78: delta_mu},
            species=(79, 78),
            parent_id=parent_id,
            metadata={
                "delta_mu_ev": delta_mu,
                "direction": direction,
                "role": "high_temperature_reference",
                "pt_fraction": endpoint_pt_fraction
                if parent_id is None
                else PT_FRACTION,
            },
        )
        runs.append(run)
        parent_id = run.run_id
    return tuple(runs)


upward_reference = _reference_scan("up", REFERENCE_DELTA_MU_EV)
downward_reference = _reference_scan("down", tuple(reversed(REFERENCE_DELTA_MU_EV)))
reference_runs = upward_reference + downward_reference
campaign = CampaignSpec.cooling_from_reference(
    reference_runs,
    COOLING_TEMPERATURES_K,
    name="aupt_cooling",
    start_after=(upward_reference[-1].run_id, downward_reference[-1].run_id),
)
campaign_scheduler = CampaignScheduler(
    campaign,
    FinalStateStore(CHECKPOINT_DIRECTORY),
)


def profile_workload(width: int, device: torch.device) -> tuple[HybridMCMD, Batch]:
    """Build representative high-temperature states for a capacity profile."""
    runs = tuple(
        replace(
            reference_runs[index % len(reference_runs)],
            run_id=f"profile.{index}",
        )
        for index in range(width)
    )
    return make_workload(runs, (None,) * width, device)


batch_width = MANUAL_BATCH_WIDTH
if AUTO_SELECT_BATCH_WIDTH and DEVICE.type == "cuda":
    planner = SimulationBatchPlanner(
        memory_fraction=BATCH_MEMORY_FRACTION,
        throughput_fraction=BATCH_THROUGHPUT_FRACTION,
    )
    measurements = planner.profile(
        profile_workload,
        BATCH_WIDTH_CANDIDATES,
        device=DEVICE,
        warmup_blocks=PROFILE_WARMUP_BLOCKS,
        measured_blocks=PROFILE_MEASURED_BLOCKS,
    )
    batch_width = planner.recommend_width(
        measurements,
        total_memory_bytes=torch.cuda.get_device_properties(DEVICE).total_memory,
    )
    print(f"Selected batch width {batch_width} from {measurements}")
elif AUTO_SELECT_BATCH_WIDTH:
    print(f"CUDA unavailable; using MANUAL_BATCH_WIDTH={MANUAL_BATCH_WIDTH}")

# %%
# Run the dependency-aware campaign and checkpoint every final state
# ------------------------------------------------------------------

while ready := campaign_scheduler.ready_batches(batch_width):
    for runs in ready:
        parents = tuple(
            campaign_scheduler.parent_state(run, device=DEVICE) for run in runs
        )
        scheduler, batch = make_workload(runs, parents, DEVICE)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        start = time.perf_counter()
        result = scheduler.run(batch, n_blocks=N_BLOCKS_PER_RUN)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize(DEVICE)
        for run, final_state in zip(runs, result.to_data_list()):
            campaign_scheduler.complete(run.run_id, final_state)
        print(
            f"Completed {[run.run_id for run in runs]} in "
            f"{time.perf_counter() - start:.2f} s; "
            f"SGC acceptance={scheduler.mc.stats.acceptance:.3f}"
        )

print(
    f"Campaign {campaign.name!r} complete: "
    f"{len(campaign_scheduler.completed_ids)} final states in {CHECKPOINT_DIRECTORY}"
)
