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
Batched Umbrella Sampling with EnhancedSampling
===============================================

Umbrella sampling computes a free-energy profile along a reaction coordinate
by running several simulations, each restrained to a different value of that
coordinate.  The restraints keep the system in regions it would otherwise
never visit; the resulting biased histograms are recombined afterwards with
WHAM or MBAR.

The usual cost is that every window is a separate simulation.  Here all
windows are **rows of one batch**: each graph carries a
``thermodynamic_state_id`` selecting its own window center, so a single
:class:`~nvalchemi.enhanced_sampling.HarmonicUmbrellaBias` serves all of them
in one batched GPU force evaluation.

This example restrains the distance between two argon atoms in a
Lennard-Jones cluster across five windows, and adds an
:class:`~nvalchemi.enhanced_sampling.UpperWall` to stop any window wandering
off to dissociation.

Key concepts demonstrated
-------------------------
* A collective variable as a plain callable over
  :func:`~nvalchemi.enhanced_sampling.pair_distance`.
* Per-window centers selected by ``thermodynamic_state_id``.
* Composing two biases; they are summed against the *same* unmodified model
  output, so neither observes the other's forces.
* Reading per-bias diagnostics from ``sampling.last_outputs``.

Applications
------------
* Free-energy profiles along a bond, a distance, or a coordination number.
* Potential of mean force for ion transport or ligand unbinding.
* Restrained sampling to generate training data in a targeted region.
"""

from __future__ import annotations

import logging
import os

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.enhanced_sampling import (
    EnhancedSampling,
    HarmonicUmbrellaBias,
    UpperWall,
    pair_distance,
)
from nvalchemi.models.lj import LennardJonesModelWrapper

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Keep the doc build fast; a real run needs orders of magnitude more steps.
N_STEPS = 20 if os.environ.get("NVALCHEMI_SPHINX_BUILD") else 200

# %%
# Build one window per restraint center
# -------------------------------------
# Every window is an independent copy of the same cluster.  The output
# buffers (``forces``, ``energy``) must exist up front: dynamics writes model
# outputs back in place rather than creating the fields.

WINDOW_CENTERS = [3.0, 3.5, 4.0, 4.5, 5.0]  # angstrom
N_ATOMS = 8

torch.manual_seed(0)


def make_cluster() -> AtomicData:
    """Return one argon cluster with the buffers dynamics writes into.

    Atoms sit on a 2x2x2 cube at the LJ minimum separation.  Random
    positions would place some pairs well inside sigma = 3.4 A, where the
    repulsive wall is steep enough that the first step diverges — the
    restraint is then blamed for what is really an overlapping start.
    """
    spacing = 4.0  # ~2^(1/6) * sigma, near the LJ minimum
    grid = torch.tensor(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    )
    positions = grid[:N_ATOMS] * spacing
    # Atoms 0 and 1 are a cube edge apart, so the CV starts at `spacing`.
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((N_ATOMS,), 18, dtype=torch.long),
        atomic_masses=torch.full((N_ATOMS,), 39.948),
        forces=torch.zeros(N_ATOMS, 3),
        energy=torch.zeros(1, 1),
    )
    data.add_node_property("velocities", torch.zeros(N_ATOMS, 3))
    return data


batch = Batch.from_data_list([make_cluster() for _ in WINDOW_CENTERS]).to(DEVICE)

# Each graph gets its own window. This is the field HarmonicUmbrellaBias
# indexes to pick a center, and it is what makes one batch five windows.
batch["thermodynamic_state_id"] = torch.arange(len(WINDOW_CENTERS), device=DEVICE)

# %%
# Define the collective variable
# ------------------------------
# A CV is any differentiable ``cv(batch) -> Tensor[B, D]``.  No base class, no
# registration — a closure over :func:`pair_distance` is a complete CV.

ATOM_PAIR = torch.tensor([0, 1], device=DEVICE)


def bond_distance(atoms: Batch) -> torch.Tensor:
    """Distance between atoms 0 and 1, shape ``[B, 1]``."""
    return pair_distance(atoms, ATOM_PAIR)


# %%
# Build the biases
# ----------------
# ``centers`` has shape ``[S, D]`` — one row per window, one column per CV
# dimension.  ``stiffness`` broadcasts across windows here, but could equally
# be per-window with shape ``[S, D, D]``.

umbrella = HarmonicUmbrellaBias(
    cv=bond_distance,
    centers=torch.tensor(WINDOW_CENTERS).unsqueeze(-1),  # [5, 1]
    stiffness=5.0,  # eV/A^2
    name="umbrella",
)

# A one-sided wall stops a window that escapes its basin from running away to
# dissociation. It contributes exactly zero energy and force while the
# distance stays below the threshold.
wall = UpperWall(
    cv=bond_distance,
    threshold=8.0,
    stiffness=20.0,
    name="dissociation_wall",
)

# %%
# Run biased dynamics
# -------------------
# The runner registers one internal hook on the dynamics and otherwise leaves
# it alone: the model, the integrator, and the thermostat are unchanged.

model = LennardJonesModelWrapper(sigma=3.4, epsilon=0.0104, cutoff=8.5).to(DEVICE)
dynamics = NVTLangevin(model=model, dt=0.5, temperature=120.0, friction=0.05)

# A cutoff model needs its neighbour list rebuilt at BEFORE_COMPUTE. The
# runner fires that stage during priming too, so the first force evaluation
# is as valid as every later one.
for hook in model.make_neighbor_hooks():
    dynamics.register_hook(hook)

sampling = EnhancedSampling(
    dynamics=dynamics,
    biases={"umbrella": umbrella, "dissociation_wall": wall},
)

logger.info("Initial CV per window: %s", bond_distance(batch).flatten().tolist())

batch = sampling.run(batch, n_steps=N_STEPS)

final_cv = bond_distance(batch).flatten().tolist()
logger.info("Target centers:        %s", WINDOW_CENTERS)
logger.info("Final CV per window:   %s", [round(v, 3) for v in final_cv])

# %%
# Read the diagnostics
# --------------------
# ``last_outputs`` separates the physical model contribution from each bias
# and from the total.  These are the tensors a WHAM/MBAR post-processing step
# needs: the CV value, the physical energy, and the bias energy per window.
#
# Free-energy reconstruction is deliberately **not** built in — the raw
# per-window data is returned for ``pymbar`` or an equivalent tool.

for key in sorted(sampling.last_outputs):
    value = sampling.last_outputs[key]
    logger.info("  %-28s shape=%s", key, tuple(value.shape))

umbrella_energy = sampling.last_outputs["bias/umbrella/energy"].flatten()
logger.info(
    "Umbrella energy per window (eV): %s", [round(float(v), 4) for v in umbrella_energy]
)

# The wall should be dormant: every window is well inside 8 A.
wall_energy = sampling.last_outputs["bias/dissociation_wall/energy"].flatten()
logger.info(
    "Wall energy per window (eV):     %s", [round(float(v), 4) for v in wall_energy]
)
