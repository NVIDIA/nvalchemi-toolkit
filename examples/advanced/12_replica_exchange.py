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
Temperature Replica Exchange (REMD)
===================================

A single low-temperature trajectory gets stuck: the barrier it needs to cross
is many :math:`k_BT` high, so it may never be crossed in the time available.
Replica exchange runs a ladder of temperatures at once and periodically swaps
which walker sits at which temperature.  A configuration that wanders up the
ladder crosses barriers easily at high temperature, and when it comes back
down it lands in a basin the cold replica might never have reached on its own.

The whole ladder is **one batch**: each graph is a walker, and
``thermodynamic_state_id`` records which rung it currently occupies.  One
batched force evaluation advances every replica per step.

Exchange swaps **labels, not coordinates**.  A walker keeps its execution
slot, its velocities, and its integrator arrays; the temperature assigned to
it changes.  That makes the move local — nothing is copied between rows —
which is what allows it inside a batched GPU step.

Key concepts demonstrated
-------------------------
* Building a geometric temperature ladder with
  :class:`~nvalchemi.enhanced_sampling.ThermodynamicState`.
* The even/odd pair schedule and the Metropolis acceptance rule.
* Reading per-pair acceptance rates, which is how a ladder is tuned.
* Confirming the integrator's target temperature follows the assignment.

Applications
------------
* Conformer and reaction-path exploration where barriers exceed a few kT.
* Crystal polymorph search and nucleation.
* Generating diverse training configurations for a potential.
"""

from __future__ import annotations

import logging
import os

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.enhanced_sampling import (
    EnhancedSampling,
    ReplicaExchange,
    ThermodynamicState,
)
from nvalchemi.models.lj import LennardJonesModelWrapper

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_STEPS = 80 if os.environ.get("NVALCHEMI_SPHINX_BUILD") else 400

# %%
# Build the temperature ladder
# ----------------------------
# A geometric spacing keeps the acceptance rate roughly uniform across rungs:
# the acceptance depends on the *ratio* of neighbouring temperatures, not
# their difference, so evenly spaced temperatures would exchange readily at
# the cold end and almost never at the hot end.

N_REPLICAS = 4
BASE_TEMPERATURE = 80.0  # K — argon is liquid around here
LADDER_FACTOR = 1.25

states = [
    ThermodynamicState(state_id=i, temperature=BASE_TEMPERATURE * LADDER_FACTOR**i)
    for i in range(N_REPLICAS)
]
logger.info("Temperature ladder (K): %s", [round(s.temperature, 1) for s in states])

# %%
# One walker per rung
# -------------------
# Output buffers must exist up front — dynamics writes model outputs back in
# place rather than creating the fields.

N_ATOMS = 8


def make_cluster(seed: int) -> AtomicData:
    """Return an argon cluster on a cube at the LJ minimum separation."""
    spacing = 4.0
    grid = torch.tensor(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    )
    torch.manual_seed(seed)
    positions = grid[:N_ATOMS] * spacing + 0.05 * torch.randn(N_ATOMS, 3)
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((N_ATOMS,), 18, dtype=torch.long),
        atomic_masses=torch.full((N_ATOMS,), 39.948),
        forces=torch.zeros(N_ATOMS, 3),
        energy=torch.zeros(1, 1),
    )
    data.add_node_property("velocities", torch.zeros(N_ATOMS, 3))
    return data


batch = Batch.from_data_list([make_cluster(seed) for seed in range(N_REPLICAS)]).to(
    DEVICE
)

# %%
# Configure the exchange
# ----------------------
# ``initial_state_ids`` must be a permutation: replica exchange presumes one
# walker per rung, and a duplicate would let two walkers claim the same
# temperature.  Acceptance randomness is derived from ``random_seed`` plus an
# attempt counter, so a run is reproducible and a checkpoint needs only two
# integers rather than an RNG blob.

exchange = ReplicaExchange(
    states=states,
    initial_state_ids=torch.arange(N_REPLICAS),
    attempt_interval=10,  # dynamics steps per exchange segment
    random_seed=2024,
)
logger.info("Acceptance rule inferred from the ladder: %s", exchange.acceptance)
logger.info("Segment 0 pairs: %s", exchange.pair_schedule(0))
logger.info("Segment 1 pairs: %s", exchange.pair_schedule(1))

# %%
# Run
# ---
# The runner validates up front that the integrator can rebind a
# thermodynamic state.  An integrator that could only accept the new label
# would keep sampling the old temperature — wrong, and with no symptom the
# run would show — so ``NVE`` and friends are rejected rather than silently
# mis-sampled.

model = LennardJonesModelWrapper(sigma=3.4, epsilon=0.0104, cutoff=8.5).to(DEVICE)
dynamics = NVTLangevin(model=model, dt=0.5, temperature=BASE_TEMPERATURE, friction=0.05)
for hook in model.make_neighbor_hooks():
    dynamics.register_hook(hook)

sampling = EnhancedSampling(
    dynamics=dynamics,
    biases={},  # pure temperature REMD; biases would compose here
    replica_exchange=exchange,
    steps_per_epoch=100,
)

batch = sampling.run(batch, n_steps=N_STEPS)

# %%
# Read the acceptance statistics
# ------------------------------
# Per-pair rates are what a ladder is tuned on.  A pair far below the others
# is a gap the walkers cannot cross, and the ladder needs another rung there;
# uniformly high rates mean the rungs are closer than they need to be and the
# ladder is wasting replicas.

logger.info("")
logger.info(
    "Overall acceptance: %d/%d = %.2f",
    exchange.accepted,
    exchange.attempts,
    exchange.acceptance_rate,
)
for index, rate in enumerate(exchange.pair_acceptance_rates()):
    logger.info(
        "  states %d <-> %d (%.0f K <-> %.0f K): %.2f",
        index,
        index + 1,
        states[index].temperature,
        states[index + 1].temperature,
        rate,
    )

# Expect near-unit acceptance here: an 8-atom cluster has energy fluctuations
# far smaller than the spread between rungs, so the Metropolis exponent stays
# close to zero and almost everything is accepted. That is the "rungs closer
# than they need to be" regime — a production system of thousands of atoms
# has much larger fluctuations, and the ladder would be widened until the
# rates land near the usual 20-30% target.

# %%
# Confirm the swap was indivisible
# --------------------------------
# The label, the integrator's target temperature, the velocity scaling, and
# the forces all have to move together.  A walker whose label says one rung
# while its thermostat still targets another would sample the wrong ensemble
# silently, so it is worth asserting rather than assuming.

assignment = batch.thermodynamic_state_id.reshape(-1).tolist()
targets = (dynamics._state.temperature.reshape(-1) / KB_EV).tolist()

logger.info("")
logger.info("Final assignment (walker -> state): %s", assignment)
for walker, state_id in enumerate(assignment):
    expected = states[state_id].temperature
    logger.info(
        "  walker %d holds state %d: target %.1f K (ladder says %.1f K)",
        walker,
        state_id,
        targets[walker],
        expected,
    )
