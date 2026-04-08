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
FIRE Geometry Optimization with Lennard-Jones Argon
====================================================

This example demonstrates geometry optimization using the
:class:`~nvalchemi.dynamics.optimizers.FIRE` algorithm on a batch of argon
clusters described by the Lennard-Jones potential.

**FIRE** (Fast Inertial Relaxation Engine) is a damped molecular-dynamics
optimizer.  Rather than computing gradients of a surrogate function, it
propagates atoms with Newtonian dynamics but periodically damps the velocities
toward the net force direction.  FIRE often converges faster than steepest descent near minima while remaining
robust far from equilibrium, making it practical for batched relaxation
campaigns where many systems must be processed in an ML-potential workflow.

A :class:`~nvalchemi.dynamics.hooks.NeighborListHook` is registered on the
optimizer so that the dense neighbor matrix is recomputed (or read from a
Verlet skin cache) before every model forward pass.  Without this hook the
model would always see a stale neighbor list.

A :class:`~nvalchemi.dynamics.base.ConvergenceHook` monitors the maximum
per-atom force norm for every system in the batch.  Once a system's fmax
falls below the threshold it is marked as converged and excluded from
subsequent steps, so the optimizer stops automatically rather than running
the full ``n_steps``.

A :class:`~nvalchemi.dynamics.hooks.LoggingHook` records per-system energy
and fmax to a CSV file at a configurable interval, using a background thread
and a CUDA side stream so logging does not stall the GPU pipeline.
"""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE
from nvalchemi.dynamics.base import ConvergenceHook
from nvalchemi.dynamics.hooks import LoggingHook, NeighborListHook
from nvalchemi.models.lj import LennardJonesModelWrapper

# %%
# LJ model and argon parameters
# ------------------------------
# The Lennard-Jones potential describes the interaction between a pair of
# neutral atoms.  The standard argon parameters are:
#
# * ``epsilon = 0.0104 eV``  — depth of the potential well
# * ``sigma   = 3.40 Å``     — distance at which the pair potential is zero
# * ``cutoff  = 8.5 Å``      — truncation radius (≈ 2.5 σ, standard for Ar)
#
# The equilibrium pair distance is r_min = 2^(1/6) σ ≈ 3.82 Å, where the
# force is zero and the energy is −ε.  Starting atoms near r_min gives a
# stable configuration that FIRE can relax in just a few hundred steps.
#
# ``max_neighbors=32`` is generous for small clusters; ``skin=0.5`` means the
# neighbor list is only rebuilt when any atom moves more than 0.25 Å since
# the last rebuild.

LJ_EPSILON = 0.0104  # eV
LJ_SIGMA = 3.40  # Å
LJ_CUTOFF = 8.5  # Å
MAX_NEIGHBORS = 32

model = LennardJonesModelWrapper(
    epsilon=LJ_EPSILON,
    sigma=LJ_SIGMA,
    cutoff=LJ_CUTOFF,
    max_neighbors=MAX_NEIGHBORS,
)

neighbor_hook = NeighborListHook(model.model_card.neighbor_config)


# %%
# System builder — simple cubic lattice
# --------------------------------------
# Atoms are placed on a simple cubic lattice with spacing ``a`` (Å).
# A spacing slightly above the LJ equilibrium distance r_min ≈ 2^(1/6)·σ ≈
# 3.82 Å gives a stable starting configuration that FIRE can quickly relax.
# Atomic number 18 = Argon; masses are auto-filled from the periodic table.

_R_MIN = 2 ** (1 / 6) * LJ_SIGMA  # ≈ 3.82 Å


def _cubic_lattice(n_per_side: int, spacing: float) -> torch.Tensor:
    """Return positions for an n³ simple-cubic lattice (Å)."""
    coords = torch.arange(n_per_side, dtype=torch.float32) * spacing
    # meshgrid produces three (n,n,n) grids; stack into (n³, 3)
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    return torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)


def _make_system(n_per_side: int, spacing: float = _R_MIN * 1.05) -> AtomicData:
    """Build an Argon cluster on a simple cubic lattice.

    Parameters
    ----------
    n_per_side : int
        Number of atoms along each lattice edge; total atoms = n_per_side³.
    spacing : float
        Nearest-neighbour distance (Å).  Defaults to 1.05 × r_min.
    """
    n_atoms = n_per_side**3
    positions = _cubic_lattice(n_per_side, spacing)
    # Add small random perturbations so FIRE has something to relax.
    torch.manual_seed(n_per_side)
    positions = positions + 0.05 * torch.randn_like(positions)

    return AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n_atoms,), 18, dtype=torch.long),  # Argon
        forces=torch.zeros(n_atoms, 3),
        energy=torch.zeros(1, 1),
        velocities=torch.zeros(n_atoms, 3),
    )


# %%
# Understanding Convergence
# --------------------------
# The :class:`~nvalchemi.dynamics.base.ConvergenceHook` evaluates a list of
# scalar criteria after each step.  Each criterion specifies:
#
# * ``key``         — the batch attribute to inspect (e.g. ``"forces"``)
# * ``threshold``   — the value below which convergence is declared
# * ``reduce_op``   — how to reduce the raw tensor (``"norm"`` → per-atom
#   Euclidean norm, then max across atoms in each system)
# * ``reduce_dims`` — the axis along which to compute the norm before taking
#   the per-system max
#
# A system is marked converged when **all** criteria are satisfied.  Converged
# systems are removed from the active batch at the start of the next step, so
# the optimizer does not waste compute on them.  The run terminates early once
# every system has converged or ``n_steps`` is reached.

# %%
# FIRE Geometry Optimization
# ---------------------------
# Build a batch of two 2×2×2 (8-atom) Argon clusters and relax with FIRE.
#
# The NeighborListHook fires at BEFORE_COMPUTE and writes ``neighbor_matrix``
# and ``num_neighbors`` into the batch atoms group before each model evaluation.

print("=== FIRE Geometry Optimization ===")

# Two identical lattice sizes; different spacings give different starting energy.
data_list_opt = [
    _make_system(2, spacing=_R_MIN * 1.05),
    _make_system(2, spacing=_R_MIN * 1.20),
]
batch_opt = Batch.from_data_list(data_list_opt)
print(f"Batch: {batch_opt.num_graphs} systems, {batch_opt.num_nodes} atoms total\n")

fire_opt = FIRE(
    model=model,
    dt=0.5,
    n_steps=300,
    convergence_hook=ConvergenceHook(
        criteria=[
            {
                "key": "forces",
                "threshold": 0.001,
                "reduce_op": "norm",
                "reduce_dims": -1,
            }
        ]
    ),
)
fire_opt.register_hook(neighbor_hook)

# LoggingHook records energy, fmax, and status per graph to a CSV file.
# The context manager starts the background I/O thread and flushes on exit.
with LoggingHook(
    backend="csv", log_path="02_geometry_optimization_fire_log.csv", frequency=5
) as log_hook:
    fire_opt.register_hook(log_hook)
    batch_opt = fire_opt.run(batch_opt)

print(
    f"\nCompleted {fire_opt.step_count} FIRE steps. Log: 02_geometry_optimization_fire_log.csv"
)

# %%
# Final energy per system
# --------------------------
# After the run, ``batch_opt.energy`` holds the per-system potential energy
# as output by the last model forward pass.  For a well-relaxed small cluster
# the total energy should be negative, with each atom contributing roughly
# −½ z ε where z is its coordination number.

final_energy = batch_opt.energy.squeeze(-1).cpu().tolist()
force_norms = batch_opt.forces.norm(dim=-1)
fmax_final = torch.zeros(batch_opt.num_graphs, device=batch_opt.device)
fmax_final.scatter_reduce_(
    0, batch_opt.batch_idx, force_norms, reduce="amax", include_self=True
)
fmax_list = fmax_final.cpu().tolist()

print("\nRelaxed system summary:")
for i in range(batch_opt.num_graphs):
    print(f"  sys{i}: E={final_energy[i]:+.6f} eV  fmax={fmax_list[i]:.6f} eV/Å")
