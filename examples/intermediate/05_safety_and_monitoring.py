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
Defensive MD: Safety Hooks and Performance Monitoring
=====================================================

Numerical instabilities are a fact of life in machine-learning molecular
dynamics.  ML potentials are trained on a finite region of configuration
space; geometries outside that region can produce enormous forces, NaN
gradients, or energy drift that silently corrupts a long trajectory.

This example demonstrates four hooks that make simulations more robust:

* :class:`~nvalchemi.dynamics.hooks.NaNDetectorHook` — raises
  :class:`RuntimeError` immediately when ``forces`` or ``energy`` contain
  non-finite values (NaN or Inf).  Prevents corrupted state from propagating.
* :class:`~nvalchemi.dynamics.hooks.MaxForceClampHook` — rescales atom force
  vectors whose L2 norm exceeds a threshold, preventing integration blow-ups
  from bad initial geometries or extrapolation errors.
* :class:`~nvalchemi.dynamics.hooks.EnergyDriftMonitorHook` — monitors the
  total energy drift in NVE simulations and warns or raises when it exceeds
  a threshold.  Drift indicates timestep or potential issues.
* :class:`~nvalchemi.dynamics.hooks.ProfilerHook` — records wall-clock time
  per hook stage and writes a CSV timing log.  Essential for identifying
  bottlenecks in the simulation loop.

**Recommended registration order** (when using multiple safety hooks):

1. ``MaxForceClampHook`` — clamp forces first so downstream checks see
   bounded values.
2. ``NaNDetectorHook`` — detect any NaN that slipped through clamping.
3. ``EnergyDriftMonitorHook`` — monitor cumulative drift (``AFTER_STEP``).
4. ``ProfilerHook`` — spans all stages, so register last or use ``stages="all"``.
"""

import logging
import tempfile
from pathlib import Path

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVE, NVTLangevin
from nvalchemi.dynamics.hooks import (
    EnergyDriftMonitorHook,
    MaxForceClampHook,
    NaNDetectorHook,
    NeighborListHook,
    ProfilerHook,
    WrapPeriodicHook,
)
from nvalchemi.models.demo import DemoModelWrapper
from nvalchemi.models.lj import LennardJonesModelWrapper

logging.basicConfig(level=logging.INFO)

# %%
# Shared helpers
# ---------------

torch.manual_seed(0)


def _demo_system(n_atoms: int, seed: int) -> AtomicData:
    """Create a random system compatible with DemoModelWrapper."""
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


def _lj_system_bad(n_atoms: int, seed: int, box: float = 10.0) -> AtomicData:
    """Create a periodic LJ system with atoms dangerously close together.

    Placing atoms at identical or very close positions will produce huge
    (or NaN) forces from the LJ r^-12 repulsion term.  This is a useful
    test case for safety hooks.

    Parameters
    ----------
    n_atoms : int
        Number of atoms.
    seed : int
        RNG seed.
    box : float
        Cubic box side length in Å.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    # Uniform random positions in [0, box) — may have very short interatomic
    # distances that produce enormous LJ repulsion forces.
    positions = torch.rand(n_atoms, 3, generator=g) * box
    cell = torch.eye(3).unsqueeze(0) * box
    mass_ar = 39.948
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n_atoms,), 18, dtype=torch.long),
        atomic_masses=torch.full((n_atoms,), mass_ar),
        forces=torch.zeros(n_atoms, 3),
        energy=torch.zeros(1, 1),
        cell=cell,
        pbc=torch.tensor([[True, True, True]]),
    )
    data.add_node_property(
        "velocities",
        torch.randn(n_atoms, 3, generator=g) * 0.01,
    )
    return data


# %%
# NaNDetectorHook — catching non-finite values early
# ----------------------------------------------------
# A pathological configuration (atoms at identical positions) causes the
# LJ potential to diverge.  :class:`~nvalchemi.dynamics.hooks.NaNDetectorHook`
# detects this on the first step and raises :class:`RuntimeError` with a
# diagnostic message listing the affected fields and graph indices.
#
# Without this hook, NaN forces would silently propagate through velocity
# updates, poisoning the entire trajectory.

lj_model_nan = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5)
lj_model_nan.eval()

# Place two atoms at nearly identical positions — guaranteed LJ blow-up.
bad_data = AtomicData(
    positions=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.001]], dtype=torch.float32),
    atomic_numbers=torch.tensor([18, 18], dtype=torch.long),
    atomic_masses=torch.full((2,), 39.948),
    forces=torch.zeros(2, 3),
    energy=torch.zeros(1, 1),
    cell=torch.eye(3).unsqueeze(0) * 20.0,
    pbc=torch.tensor([[True, True, True]]),
)
bad_data.add_node_property("velocities", torch.zeros(2, 3))
bad_batch = Batch.from_data_list([bad_data])

nan_hook = NaNDetectorHook()
nl_hook_nan = NeighborListHook(lj_model_nan.model_card.neighbor_config)

nvt_nan = NVTLangevin(
    model=lj_model_nan,
    dt=1.0,
    temperature=50.0,
    friction=0.1,
    random_seed=1,
    n_steps=5,
    hooks=[nl_hook_nan, nan_hook],
)

logging.info("Running NaNDetectorHook demo (expect RuntimeError)...")
try:
    nvt_nan.run(bad_batch)
    logging.info("No NaN detected (unexpected for this configuration).")
except RuntimeError as exc:
    logging.info("NaNDetectorHook correctly caught: %s", str(exc)[:120])

# %%
# MaxForceClampHook — surviving bad initial geometries
# ------------------------------------------------------
# :class:`~nvalchemi.dynamics.hooks.MaxForceClampHook` rescales any force
# vector whose L2 norm exceeds ``max_force``, preserving the direction but
# bounding the magnitude.  This lets the integrator take a step even when
# the potential energy surface produces pathologically large gradients.
#
# Register ``MaxForceClampHook`` *before* ``NaNDetectorHook`` so that forces
# are bounded before the NaN check fires (both run at ``AFTER_COMPUTE``).

lj_model_clamp = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5)
lj_model_clamp.eval()

# Atoms randomly placed in a 5 Å box — some pairs will be very close.
clamp_data = _lj_system_bad(n_atoms=8, seed=77, box=5.0)
clamp_batch = Batch.from_data_list([clamp_data])

clamp_hook = MaxForceClampHook(max_force=10.0)  # eV/Å
nl_hook_clamp = NeighborListHook(lj_model_clamp.model_card.neighbor_config)
wrap_hook_clamp = WrapPeriodicHook()

nvt_clamp = NVTLangevin(
    model=lj_model_clamp,
    dt=0.5,
    temperature=50.0,
    friction=0.1,
    random_seed=2,
    n_steps=20,
    hooks=[nl_hook_clamp, clamp_hook, wrap_hook_clamp],
)

logging.info("Running MaxForceClampHook demo (20 steps with force clamping)...")
clamp_batch = nvt_clamp.run(clamp_batch)
final_fmax = clamp_batch.forces.norm(dim=-1).max().item()
logging.info(
    "Completed %d steps. Final fmax=%.4f eV/Å (clamped at 10.0 eV/Å per step).",
    nvt_clamp.step_count,
    final_fmax,
)

# %%
# EnergyDriftMonitorHook — NVE energy conservation
# --------------------------------------------------
# In a well-integrated NVE (microcanonical) simulation the total energy
# (kinetic + potential) should be conserved.  Drift exceeding a threshold
# indicates an overly large timestep or a non-smooth potential.
#
# ``metric="per_atom_per_step"`` normalises the drift by atom count and step
# number, making it comparable across systems of different sizes and lengths.
# ``action="warn"`` emits a log warning rather than stopping the simulation.

demo_model = DemoModelWrapper()
demo_model.eval()

# Provide a system with non-zero initial velocities for kinetic energy.
nve_data = _demo_system(n_atoms=6, seed=10)
KB_EV = 8.617333e-5
kT = 200.0 * KB_EV
g = torch.Generator()
g.manual_seed(11)
nve_data["velocities"] = torch.randn(6, 3, generator=g) * (kT / 1.0) ** 0.5
nve_batch = Batch.from_data_list([nve_data])

drift_hook = EnergyDriftMonitorHook(
    threshold=1e-3,
    metric="per_atom_per_step",
    action="warn",
    frequency=10,
    include_kinetic=True,
)

nve = NVE(
    model=demo_model,
    dt=0.5,
    n_steps=100,
    hooks=[drift_hook],
)

logging.info("Running EnergyDriftMonitorHook demo (100 NVE steps)...")
nve_batch = nve.run(nve_batch)
logging.info(
    "NVE run complete. Reference energy captured; drift monitored every 10 steps."
)

# %%
# ProfilerHook — timing the simulation loop
# ------------------------------------------
# :class:`~nvalchemi.dynamics.hooks.ProfilerHook` records wall-clock time at
# each hook stage.  ``stages="step"`` instruments only ``BEFORE_STEP`` and
# ``AFTER_STEP``, giving a clean per-step elapsed time without the overhead
# of timing every sub-stage.
#
# ``log_path`` writes a CSV with columns: ``rank, step, stage, t_since_init_s,
# delta_s``.  ``show_console=True`` additionally prints a formatted table via
# loguru at each ``console_frequency``-th profiled step.

profiler_out = Path(tempfile.mkdtemp()) / "profile.csv"

profiler_hook = ProfilerHook(
    profiled_stages="step",
    timer_backend="auto",
    log_path=str(profiler_out),
    show_console=True,
    console_frequency=10,
    frequency=1,
)

prof_data = _demo_system(n_atoms=8, seed=20)
prof_batch = Batch.from_data_list([prof_data])

nvt_prof = NVTLangevin(
    model=demo_model,
    dt=0.5,
    temperature=300.0,
    friction=0.1,
    random_seed=3,
    n_steps=50,
    hooks=[profiler_hook],
)

logging.info("Running ProfilerHook demo (50 NVT steps)...")
prof_batch = nvt_prof.run(prof_batch)
profiler_hook.close()

summary = profiler_hook.summary()
for transition, stats in summary.items():
    logging.info(
        "  %s: mean=%.3f ms  std=%.3f ms  n=%d",
        transition,
        stats["mean_s"] * 1000,
        stats["std_s"] * 1000,
        int(stats["n_samples"]),
    )

logging.info("Profile CSV written to: %s", profiler_out)

# %%
# Defensive setup pattern — all four hooks together
# --------------------------------------------------
# When running an unfamiliar potential on new structures, combine all four
# hooks.  Registration order matters:
#
# 1. ``NeighborListHook`` (``BEFORE_COMPUTE``) — must run before model.
# 2. ``MaxForceClampHook`` (``AFTER_COMPUTE``) — clamp before NaN check.
# 3. ``NaNDetectorHook`` (``AFTER_COMPUTE``) — detect remaining bad values.
# 4. ``EnergyDriftMonitorHook`` (``AFTER_STEP``) — cumulative drift check.
# 5. ``ProfilerHook`` (all stages) — spans the full step loop.
#
# This ordering ensures each hook sees the most up-to-date (and safest)
# state when it fires.

logging.info("=== Defensive setup pattern example ===")

demo_model2 = DemoModelWrapper()
demo_model2.eval()

safe_data = _demo_system(n_atoms=5, seed=99)
safe_batch = Batch.from_data_list([safe_data])

clamp = MaxForceClampHook(max_force=50.0)
nan_check = NaNDetectorHook(frequency=1)
drift_check = EnergyDriftMonitorHook(
    threshold=1.0,  # generous for demo purposes
    metric="absolute",
    action="warn",
    frequency=5,
)
profiler = ProfilerHook(profiled_stages="step", timer_backend="auto", frequency=1)

safe_nvt = NVTLangevin(
    model=demo_model2,
    dt=0.5,
    temperature=300.0,
    friction=0.1,
    random_seed=42,
    n_steps=30,
    # Hooks fire in registration order within the same stage.
    hooks=[clamp, nan_check, drift_check, profiler],
)

safe_batch = safe_nvt.run(safe_batch)
profiler.close()
logging.info(
    "Defensive run complete: %d steps, no exceptions raised.",
    safe_nvt.step_count,
)
