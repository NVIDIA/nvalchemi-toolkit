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
Multiple-Walker Well-Tempered Metadynamics
==========================================

Metadynamics fills the free-energy basin the system is sitting in with
Gaussian hills, until the barrier out of it is no longer a barrier.  Where
umbrella sampling needs you to know in advance which values of the reaction
coordinate to restrain to, metadynamics discovers them: it goes wherever it
has not already been.

The *well-tempered* variant shrinks each new hill in proportion to the bias
already accumulated at that point,

.. math:: h_t = h_0 \\exp\\!\\left(-\\frac{V(s_t)}{k_B T (\\gamma - 1)}\\right)

so the sum converges instead of filling forever, and the converged bias can
be turned back into a free-energy profile.

This example runs the **multiple-walker** scheme: several walkers explore the
same collective variable at once, all depositing into one shared hill
history, so a basin is filled roughly ``B`` times faster.  Because the
walkers are rows of a single batch, that costs one batched GPU force
evaluation per step, not ``B`` separate simulations.

Key concepts demonstrated
-------------------------
* Depositing hills on a schedule with ``update_frequency``, driven by
  :class:`~nvalchemi.enhanced_sampling.EnhancedSampling`.
* Shared multi-walker history, and how it differs from private history.
* Choosing a storage policy, and why ``preallocated`` raises rather than
  silently discarding hills.
* Recovering a free-energy profile with
  :meth:`~nvalchemi.enhanced_sampling.WellTemperedMetaDynamicsBias.free_energy`.

Applications
------------
* Free-energy profiles when the relevant windows are not known in advance.
* Escaping metastable states that plain MD would stay trapped in.
* Barrier crossing for conformational change, dissociation, or diffusion.
"""

from __future__ import annotations

import logging
import os

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.enhanced_sampling import (
    EnhancedSampling,
    UpperWall,
    WellTemperedMetaDynamicsBias,
    pair_distance,
)
from nvalchemi.models.lj import LennardJonesModelWrapper

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Keep the doc build fast; a converged run needs orders of magnitude more.
N_STEPS = 40 if os.environ.get("NVALCHEMI_SPHINX_BUILD") else 600

# %%
# Build the walkers
# -----------------
# Four independent copies of the same argon cluster.  They are not windows:
# no walker is restrained anywhere, and they differ only in the random
# velocities the thermostat gives them.

N_WALKERS = 4
N_ATOMS = 8
TEMPERATURE = 120.0  # K

torch.manual_seed(0)


def make_cluster() -> AtomicData:
    """Return one argon cluster with the buffers dynamics writes into.

    Atoms sit on a 2x2x2 cube near the LJ minimum separation; random
    positions would place some pairs inside sigma, where the repulsive wall
    is steep enough that the first step diverges.
    """
    spacing = 4.0
    grid = torch.tensor(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    )
    data = AtomicData(
        positions=grid[:N_ATOMS] * spacing,
        atomic_numbers=torch.full((N_ATOMS,), 18, dtype=torch.long),
        atomic_masses=torch.full((N_ATOMS,), 39.948),
        forces=torch.zeros(N_ATOMS, 3),
        energy=torch.zeros(1, 1),
    )
    data.add_node_property("velocities", torch.zeros(N_ATOMS, 3))
    return data


batch = Batch.from_data_list([make_cluster() for _ in range(N_WALKERS)]).to(DEVICE)

# %%
# Define the collective variable
# ------------------------------
# The same plain-callable contract as umbrella sampling: any differentiable
# ``cv(batch) -> Tensor[B, D]``.

ATOM_PAIR = torch.tensor([0, 1], device=DEVICE)


def bond_distance(atoms: Batch) -> torch.Tensor:
    """Distance between atoms 0 and 1, shape ``[B, 1]``."""
    return pair_distance(atoms, ATOM_PAIR)


# %%
# Configure the bias
# ------------------
# Three choices carry most of the physics:
#
# ``sigma``
#     Hill width, in CV units.  Roughly the resolution of the resulting free
#     energy: too wide smears out real features, too narrow needs far more
#     hills to fill anything.
#
# ``bias_factor`` (gamma)
#     Sets how high the bias is allowed to climb — the effective temperature
#     of the CV is ``gamma * T``.  Larger explores further and converges
#     slower.  Passing ``None`` gives standard, non-converging metadynamics.
#
# ``update_frequency``
#     Steps between depositions.  Depositing faster than the system
#     decorrelates biases the estimate; the usual choice is a few hundred
#     steps.
#
# ``storage="preallocated"`` keeps the hill tensors a fixed shape for the
# whole run, which is what keeps ``energy()`` compilable without retracing.
# It **raises** when ``max_hills`` is exhausted rather than dropping hills:
# silently discarding them would change the physics of a converging run with
# no error to show for it.

metad = WellTemperedMetaDynamicsBias(
    cv=bond_distance,
    height=0.005,  # eV
    sigma=0.25,  # angstrom
    temperature=TEMPERATURE,
    bias_factor=8.0,
    update_frequency=25,
    storage="preallocated",
    max_hills=512,
    history="shared",  # multiple-walker: every walker feels every hill
    name="metad",
)

# Metadynamics pushes outward by construction, so an unbounded CV eventually
# dissociates the pair. A wall bounds the explored region without touching
# the interior, where it contributes exactly zero.
wall = UpperWall(cv=bond_distance, threshold=9.0, stiffness=20.0, name="wall")

# %%
# Run
# ---
# The runner calls ``update()`` exactly once per due step, at ``AFTER_STEP``
# so each hill marks the configuration the walker actually reached.  A
# deposition bumps the bias state version, and the runner re-primes forces in
# response, so a new hill is felt on the very next step rather than one step
# late.

model = LennardJonesModelWrapper(sigma=3.4, epsilon=0.0104, cutoff=8.5).to(DEVICE)
dynamics = NVTLangevin(model=model, dt=0.5, temperature=TEMPERATURE, friction=0.05)

for hook in model.make_neighbor_hooks():
    dynamics.register_hook(hook)

sampling = EnhancedSampling(dynamics=dynamics, biases={"metad": metad, "wall": wall})

logger.info(
    "Initial CV per walker: %s",
    [round(v, 3) for v in bond_distance(batch).flatten().tolist()],
)

batch = sampling.run(batch, n_steps=N_STEPS)

logger.info(
    "Final CV per walker:   %s",
    [round(v, 3) for v in bond_distance(batch).flatten().tolist()],
)
logger.info(
    "Depositions: %d, hills stored: %d of %d",
    int(metad.deposits),
    int(metad.hill_count),
    metad.capacity,
)

# %%
# Hill heights decay
# ------------------
# This is the well-tempered signature. As the bias accumulates, each new hill
# is shorter than the last; a run whose recent hills are still near ``h_0``
# has not begun to converge.

heights = metad.hill_heights[: int(metad.hill_count)]
logger.info("First hill height: %.6f eV", float(heights[0]))
logger.info("Last hill height:  %.6f eV", float(heights[-1]))

# %%
# Recover the free-energy profile
# -------------------------------
# ``F(s) = -(gamma / (gamma - 1)) V(s)``, up to an additive constant. The
# profile is only meaningful where hills were actually deposited; the flat
# regions beyond simply mean the walkers never went there.

grid = torch.linspace(3.0, 8.0, 26, device=DEVICE).unsqueeze(-1)
free_energy = metad.free_energy(grid)
free_energy = free_energy - free_energy.min()

logger.info("Free-energy profile (eV, shifted to zero minimum):")
for value, energy in zip(grid.flatten().tolist(), free_energy.tolist(), strict=True):
    logger.info("  d = %.2f A   F = %.4f eV", value, energy)

# %%
# Shared versus private history
# -----------------------------
# ``history="shared"`` is what makes this *multiple-walker* metadynamics
# rather than four independent runs: one history, filled four times as fast.
#
# The alternatives are ``"walker"``, which gives each walker its own private
# history and so runs ``B`` genuinely independent simulations in one batch,
# and ``"state"``, which keys the history by ``thermodynamic_state_id`` for a
# replica-exchange ladder.
#
# Under ``"shared"`` every hill is unowned, which is what the ``-1`` here
# records.

logger.info("Hill owners (-1 means shared): %s", metad.hill_owner[:4].tolist())

# %%
# Diagnostics
# -----------
# As with every bias, the runner separates the physical contribution from
# each bias and from the total.

for key in sorted(sampling.last_outputs):
    logger.info("  %-28s shape=%s", key, tuple(sampling.last_outputs[key].shape))
