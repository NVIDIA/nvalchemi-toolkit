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
Adaptive Biasing Force Along a Pair Distance
============================================

Umbrella sampling restrains, and metadynamics fills.  ABF does neither: it
*measures* the mean force along the collective variable in each bin, and
applies its negative.  Once a bin is well sampled the residual force along
the CV averages to zero, so the walker diffuses across the coordinate
instead of being held or pushed.

The payoff is that the accumulated quantity already **is** the free-energy
gradient:

.. math::

    \\frac{\\partial A}{\\partial r} = \\left\\langle
        -\\frac{(\\mathbf{F}_j - \\mathbf{F}_i)\\cdot\\hat{\\mathbf{u}}}{2}
        - \\frac{2 k_B T}{r} \\right\\rangle_r

Integrating it gives the PMF directly — no hills to deconvolve, no
histograms to reweight.

That second term is the **metric correction**, and it is the part worth
understanding before using this method.  This example demonstrates it
numerically against a case with a known answer.

Key concepts demonstrated
-------------------------
* The mean-force estimator, checked against an analytic PMF.
* Why a naive Cartesian force projection is wrong, shown by measuring it.
* The minimum-sample threshold and force ramp.
* Force-only biases: no energy, and therefore no replica exchange.

Applications
------------
* Potentials of mean force for bond breaking, ion pairing, or unbinding.
* Flattening a known coordinate so that orthogonal degrees of freedom relax.
* Free-energy profiles where hill deposition would be too slow to converge.
"""

from __future__ import annotations

import logging
import math
import os

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.enhanced_sampling import (
    AdaptiveBiasingForce,
    BiasResult,
    EnhancedSampling,
)
from nvalchemi.models.lj import LennardJonesModelWrapper

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_STEPS = 40 if os.environ.get("NVALCHEMI_SPHINX_BUILD") else 1500

TEMPERATURE = 120.0  # K
KT = KB_EV * TEMPERATURE

# %%
# The metric correction, measured
# -------------------------------
# Take two particles that do not interact at all.  Their Cartesian force is
# zero, so a naive projection reports a mean force of zero and therefore a
# *flat* free-energy profile.
#
# The true profile is not flat.  The number of ways to place two particles a
# distance ``r`` apart grows as the surface of a sphere, ``4 pi r^2``, so
# ``A(r) = -2 k_B T ln r`` — purely entropic, and it drives the pair apart.
# Missing it is not noise; it is a smooth, plausible, wrong answer.
#
# ABF includes the corresponding ``-2 k_B T / r`` term. Here is the
# difference, on a system whose answer we know exactly.

FREE_PAIR = torch.tensor([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
free_batch = Batch.from_data_list(
    [
        AtomicData(
            positions=FREE_PAIR,
            atomic_numbers=torch.ones(2, dtype=torch.long),
            forces=torch.zeros(2, 3),  # non-interacting
        )
    ]
).to(DEVICE)

demo = AdaptiveBiasingForce(
    atom_indices=torch.tensor([0, 1]),
    temperature=TEMPERATURE,
    cv_range=(2.0, 5.0),
    n_bins=30,
    min_samples=0,
    full_samples=0,
    name="demo",
).to(DEVICE)
demo.update(free_batch, BiasResult())

measured = float(demo.mean_force()[int(demo.bin_index(torch.tensor([3.0]))[0])])
logger.info("Two non-interacting particles at r = 3.0 A:")
logger.info("  naive projection      dA/dr = %+.6f eV/A  (flat PMF — wrong)", 0.0)
logger.info("  ABF with correction   dA/dr = %+.6f eV/A", measured)
logger.info("  analytic  -2 kT / r         = %+.6f eV/A", -2 * KT / 3.0)

# %%
# Build the system
# ----------------
# An argon dimer inside a small cluster.  The CV is the distance between
# atoms 0 and 1; the rest of the cluster is the "environment" whose
# rearrangement the PMF integrates over.

N_ATOMS = 6
N_WALKERS = 4

torch.manual_seed(0)


def make_cluster() -> AtomicData:
    """Return one argon cluster with the buffers dynamics writes into."""
    spacing = 3.9
    grid = torch.tensor(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    )
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


batch = Batch.from_data_list([make_cluster() for _ in range(N_WALKERS)]).to(DEVICE)

# %%
# Configure the bias
# ------------------
# ``min_samples`` / ``full_samples``
#     A mean force from three samples is noise, and applying it would drive
#     the walker on the strength of that noise.  Below ``min_samples`` a bin
#     applies nothing; between the two the applied fraction ramps linearly to
#     one, so no bin ever switches on with a jump.
#
# ``cv_range``
#     Outside it, no force is applied and no sample is recorded.  Choose it
#     to cover the region you want the profile over — ABF will not tell you
#     about coordinates it was never allowed to bin.
#
# ``max_force``
#     A bin visited once at an awkward geometry can hold a large estimate.
#     The cap bounds what that can do to the trajectory while the average
#     settles.

abf = AdaptiveBiasingForce(
    atom_indices=torch.tensor([0, 1]),
    temperature=TEMPERATURE,
    cv_range=(3.0, 6.5),  # angstrom
    n_bins=28,
    min_samples=20,
    full_samples=60,
    max_force=0.5,  # eV/A
    update_frequency=1,  # every uncorrelated sample helps
    name="abf",
)

# %%
# Run
# ---
# The runner hands ABF its observation at ``AFTER_COMPUTE``, where
# ``batch.forces`` still holds the **unbiased** physical force.  This is not
# a detail: an estimator shown its own output converges to whatever it had
# already decided, and the resulting profile looks perfectly smooth.

model = LennardJonesModelWrapper(sigma=3.4, epsilon=0.0104, cutoff=8.5).to(DEVICE)
dynamics = NVTLangevin(model=model, dt=0.5, temperature=TEMPERATURE, friction=0.05)

for hook in model.make_neighbor_hooks():
    dynamics.register_hook(hook)

sampling = EnhancedSampling(dynamics=dynamics, biases={"abf": abf})
batch = sampling.run(batch, n_steps=N_STEPS)

logger.info("")
logger.info(
    "Samples: %d across %d of %d bins",
    int(abf.bin_counts.sum()),
    int((abf.bin_counts > 0).sum()),
    abf.n_bins,
)
logger.info(
    "Bins past the threshold (applying force): %d",
    int((abf.bin_counts > abf.min_samples).sum()),
)

# %%
# ABF adds force but no energy
# ----------------------------
# ``last_outputs`` carries no ``bias/abf/energy`` key, and the total energy
# equals the physical energy exactly. That is not an omission — the applied
# force is genuinely not the gradient of any function the bias holds, which
# is what "non-conservative" means here.

outputs = sampling.last_outputs
logger.info("")
logger.info("Diagnostics: %s", sorted(k for k in outputs if k.startswith("bias/abf/")))
logger.info("Has a bias energy: %s", "bias/abf/energy" in outputs)
logger.info(
    "total/energy == physical/energy: %s",
    bool(torch.allclose(outputs["total/energy"], outputs["physical/energy"])),
)
logger.info(
    "Applied bias force magnitude: %.3e eV/A",
    float(outputs["bias/abf/forces"].abs().max()),
)

# %%
# The free-energy profile
# -----------------------
# Integrating the mean force gives the PMF directly. Bins that were never
# visited come back as ``nan`` rather than zero — a bin with no samples has
# no estimate, and zero is a perfectly plausible free energy that would hide
# that.
#
# A short demo run will not have sampled contiguously, and ``free_energy()``
# **raises** on an interior gap rather than integrating across it: the
# profile beyond a hole would be wrong by an unknown constant.

try:
    profile = abf.free_energy()
    centers = abf.bin_centers
    logger.info("")
    logger.info("Free-energy profile (eV, shifted to zero minimum):")
    for r, value in zip(centers.tolist(), profile.tolist(), strict=True):
        if not math.isnan(value):
            logger.info("  r = %.2f A   A = %+.4f eV", r, value)

    # A free check that the estimator is not fooling itself: the PMF minimum
    # should sit near the Lennard-Jones minimum, 2^(1/6) * sigma.
    sampled = ~torch.isnan(profile)
    minimum = float(centers[sampled][profile[sampled].argmin()])
    logger.info(
        "PMF minimum at %.2f A; Lennard-Jones minimum 2^(1/6) sigma = %.2f A",
        minimum,
        2 ** (1 / 6) * 3.4,
    )
except RuntimeError as error:
    logger.info("")
    logger.info("free_energy() declined, as it should on a short run:")
    logger.info("  %s", error)

# %%
# Why ABF cannot join a replica-exchange ladder
# ---------------------------------------------
# The Metropolis acceptance rule needs each bias's energy evaluated under
# *both* states being swapped. A force-only bias has no such energy, so the
# rule cannot be formed. Rather than silently drop the bias from the
# exponent — which would break detailed balance with nothing to show for it
# — the combination is refused at construction.

logger.info("")
logger.info("supplies_exchange_energy: %s", abf.supplies_exchange_energy)
logger.info("A ReplicaExchange ladder will refuse this bias at construction.")
