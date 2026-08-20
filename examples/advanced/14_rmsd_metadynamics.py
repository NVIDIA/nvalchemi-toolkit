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
xTB-Style RMSD Metadynamics for Structure Exploration
=====================================================

Every method so far has needed a collective variable: a scalar you already
know matters.  That is the assumption RMSD metadynamics drops.  Its history
is a set of retained *structures*, and the bias pushes the system away from
all of them at once:

.. math::

    V(x, t) = \\sum_r f_r(t)\\, k_\\mathrm{push}
              \\exp\\!\\left(-\\alpha\\, \\mathrm{RMSD}(x, x_r)^2\\right)

Nothing here names a reaction coordinate.  The "coordinate" is the whole
geometry, compared after optimal translation and rotation, which is what
makes this the method of choice for conformer and isomer searching — the
scheme xTB/CREST uses.

The trade is explicit: there is no free energy at the end.  This is a
structure *generator*, and :class:`RMSDMetaDynamicsBias` deliberately has no
``free_energy`` method to suggest otherwise.

This example explores isomers of an 8-atom Lennard-Jones cluster, a classic
benchmark with several distinct minima separated by real barriers.

Key concepts demonstrated
-------------------------
* Biasing without choosing a collective variable.
* Optimal-alignment RMSD, and why the bias exerts no net force.
* FIFO retention, and why it is the natural policy here rather than a
  compromise.
* Selecting a subset of atoms to compare.

Applications
------------
* Conformer generation for a flexible molecule.
* Isomer and polymorph searching.
* Escaping a minimum when you cannot say in advance what direction "out" is.
"""

from __future__ import annotations

import logging
import os

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.enhanced_sampling import EnhancedSampling, RMSDMetaDynamicsBias
from nvalchemi.models.lj import LennardJonesModelWrapper

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_STEPS = 40 if os.environ.get("NVALCHEMI_SPHINX_BUILD") else 800

N_WALKERS = 2
N_ATOMS = 8
TEMPERATURE = 40.0  # K — cold enough that plain MD stays near one basin

torch.manual_seed(0)

# %%
# Build the cluster
# -----------------
# A non-periodic molecular batch. This bias **rejects** a periodic one:
# Cartesian RMSD against a stored reference is not defined under periodic
# boundary conditions, because an atom crossing a cell face is physically
# unmoved but Cartesian-displaced by a lattice vector. Rather than return a
# plausible wrong number, ``evaluate()`` raises.
#
# Periodicity is read from ``batch.pbc``, not from the presence of a cell. A
# bounding box with ``pbc=False`` — a boxed or solvated molecule — is fine;
# this cluster simply carries no cell at all.


def make_cluster() -> AtomicData:
    """Return one argon cluster with the buffers dynamics writes into."""
    spacing = 3.8
    grid = torch.tensor(
        [[x, y, z] for x in (0.0, 1.0) for y in (0.0, 1.0) for z in (0.0, 1.0)]
    )
    positions = grid[:N_ATOMS] * spacing
    positions = positions + 0.05 * torch.randn_like(positions)
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
# ``k_push``
#     How hard to push away from a retained structure, in eV.  It sets the
#     barrier height the bias can overcome.
#
# ``alpha``
#     Kernel width in ``A^-2``.  It sets how *different* a structure has to
#     be before it stops feeling the reference, and it is the parameter most
#     worth thinking about, because it must match the RMSD scale your system
#     actually explores.  The kernel only has usable gradient where
#     ``alpha * RMSD^2`` is of order one: pick ``alpha`` far too small and
#     ``exp(-alpha * RMSD^2)`` sits at ~1 for every structure, so the bias is
#     nearly constant and exerts almost no force at all.  A rigid cluster
#     moving 0.3 A wants ``alpha`` around 10; a floppy molecule sampling
#     1 A wants ``alpha`` around 1.
#
# ``storage="fifo"``
#     The default here, and unlike well-tempered metadynamics it is not a
#     compromise.  With no free energy to reconstruct, discarding the oldest
#     reference is a deliberate choice: it keeps the system pushing outward
#     instead of accumulating an ever-stiffer cage that eventually freezes
#     it.  This is the xTB-compatible policy.
#
# ``ramp_depositions``
#     A new reference lands at RMSD zero — exactly where the system is
#     standing.  Without a ramp it would switch on at full amplitude at the
#     worst possible moment, so the default ramps it in over one deposition.

rmsd_bias = RMSDMetaDynamicsBias(
    k_push=0.08,  # eV
    alpha=10.0,  # A^-2, matched to the ~0.3 A scale this cluster explores
    update_frequency=40,
    storage="fifo",
    max_references=24,
    ramp_depositions=1,
    name="rmsd",
)

# %%
# Comparing a subset of atoms
# ---------------------------
# Passing ``atom_indices`` restricts the comparison to chosen atoms, given as
# *per-graph local* indices.  For a molecule the usual choice is heavy atoms
# only: methyl hydrogens spinning freely generate RMSD that says nothing
# about the conformer.  Correspondence is fixed — atom ``i`` is always
# compared with atom ``i`` of the reference, with no permutation search — so
# two structures identical up to relabelling count as distinct.
#
# Every atom is compared here, which is what ``atom_indices=None`` means.

# %%
# Run
# ---

model = LennardJonesModelWrapper(sigma=3.4, epsilon=0.0104, cutoff=8.5).to(DEVICE)
dynamics = NVTLangevin(model=model, dt=0.5, temperature=TEMPERATURE, friction=0.02)

for hook in model.make_neighbor_hooks():
    dynamics.register_hook(hook)

sampling = EnhancedSampling(dynamics=dynamics, biases={"rmsd": rmsd_bias})

initial = sampling.prime_forces(batch)
logger.info(
    "Initial potential energy: %s",
    [round(v, 4) for v in sampling.last_outputs["physical/energy"].flatten().tolist()],
)

batch = sampling.run(batch, n_steps=N_STEPS, prime=False)

logger.info(
    "Depositions: %d, references retained: %d of %d (written: %d)",
    int(rmsd_bias.deposits),
    int(rmsd_bias.reference_count),
    rmsd_bias.capacity,
    int(rmsd_bias.references_written),
)
logger.info(
    "Final potential energy:   %s",
    [round(v, 4) for v in sampling.last_outputs["physical/energy"].flatten().tolist()],
)

# %%
# The retained structures are distinct
# ------------------------------------
# The point of the bias is that what it collects are genuinely different
# geometries, not the same one re-recorded.  Pairwise RMSD between the
# retained references shows that directly.

from nvalchemi.enhanced_sampling.biases.rmsd_metad import _squared_rmsd


def spread(structures: torch.Tensor) -> tuple[float, float]:
    """Return the mean and max pairwise RMSD over a set of structures."""
    pairwise = _squared_rmsd(structures, structures).clamp(min=0.0).sqrt()
    mask = ~torch.eye(pairwise.shape[0], dtype=torch.bool, device=pairwise.device)
    off_diagonal = pairwise[mask]
    return float(off_diagonal.mean()), float(off_diagonal.max())


references = rmsd_bias.reference_coords[: int(rmsd_bias.reference_count)]
biased_mean, biased_max = spread(references)
logger.info("Pairwise RMSD between retained structures (A):")
logger.info("  mean %.3f   max %.3f", biased_mean, biased_max)

# %%
# The comparison that matters
# ---------------------------
# Structural spread on its own proves nothing: a hot enough thermostat
# produces scatter without visiting anything new.  Repeating the identical
# run with the bias removed separates the two.

control_batch = Batch.from_data_list([make_cluster() for _ in range(N_WALKERS)]).to(
    DEVICE
)
control_model = LennardJonesModelWrapper(sigma=3.4, epsilon=0.0104, cutoff=8.5).to(
    DEVICE
)
control_dynamics = NVTLangevin(
    model=control_model, dt=0.5, temperature=TEMPERATURE, friction=0.02
)
for hook in control_model.make_neighbor_hooks():
    control_dynamics.register_hook(hook)

# A "bias" that only records where the unbiased trajectory went. Same
# deposition schedule, no k_push acting on the dynamics.
recorder = RMSDMetaDynamicsBias(
    k_push=1e-12,
    alpha=10.0,
    update_frequency=40,
    max_references=24,
    name="rmsd",
)
control = EnhancedSampling(dynamics=control_dynamics, biases={"rmsd": recorder})
control.run(control_batch, n_steps=N_STEPS)

control_refs = recorder.reference_coords[: int(recorder.reference_count)]
control_mean, control_max = spread(control_refs)
logger.info("Unbiased control, same schedule and temperature:")
logger.info("  mean %.3f   max %.3f", control_mean, control_max)
logger.info("Bias widened the explored set by %.1fx", biased_mean / control_mean)

# %%
# The bias cannot move the cluster bodily
# ---------------------------------------
# RMSD is measured after optimal translation and rotation, so the energy is
# invariant to rigid motion — and therefore the bias forces sum to zero.  A
# bias that failed this would slowly translate the system, and the drift
# would look like physics.

bias_forces = sampling.last_outputs["bias/rmsd/forces"]
net = bias_forces.reshape(N_WALKERS, N_ATOMS, 3).sum(dim=1)
logger.info("Net bias force per walker: %.2e eV/A", float(net.abs().max()))

# %%
# No free energy here
# -------------------
# Well-tempered metadynamics converges to a bias that *is* a free-energy
# estimate.  This method does not: the references are discarded as the FIFO
# ring wraps, and the kernel is not a probability model of anything.  What it
# produces is a set of structures worth optimising or re-scoring with a
# higher level of theory.
#
# ``WellTemperedMetaDynamicsBias.free_energy`` refuses under ``"fifo"`` for
# exactly this reason, and ``RMSDMetaDynamicsBias`` has no such method at all.

logger.info("Retained %d structures for downstream optimisation.", references.shape[0])
