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
Long-Range Electrostatics with Ewald Summation
===============================================

Lennard-Jones and other short-range potentials truncate interactions at a
cutoff radius, which works well for neutral systems with rapidly decaying
forces.  **Ionic systems** — salts, molten metals, charged proteins — require
long-range Coulomb interactions that decay as 1/r and cannot be safely
truncated without severe artefacts.

The **Ewald summation** method handles this by splitting the Coulomb sum into:

* A **real-space** term — short-range, erfc-damped, evaluated over a neighbor
  list exactly as for LJ.
* A **reciprocal-space** term — smooth structure-factor sum in Fourier space
  that captures the long-range tail.

Together they reproduce the exact infinite-lattice Coulomb energy for a
periodic system.

This example:

* Builds a minimal NaCl rock-salt crystal (2×2×2 supercell, 64 atoms) with
  correct formal charges +1 e (Na) and −1 e (Cl).
* Computes energy and forces with :class:`~nvalchemi.models.ewald.EwaldModelWrapper`.
* Compares the total Ewald energy to the analytical Madelung energy.

.. note::

    For molecular-dynamics simulations of ionic systems you need a short-range
    repulsion term alongside the Ewald electrostatics.  Pure Coulomb has no
    equilibrium — without repulsion ions accelerate toward each other without
    bound.  See **Example 07** (Additive Model Composition) for the complete
    ``LennardJones + Ewald`` setup used in production MD runs.

Key concepts demonstrated
--------------------------
* Constructing an :class:`~nvalchemi.data.AtomicData` with ``charges``
  (shape ``[N, 1]``, elementary charge units).
* Instantiating :class:`~nvalchemi.models.ewald.EwaldModelWrapper` with a
  real-space cutoff and auto-estimated Ewald parameters.
* Computing energy and forces for a single snapshot via a direct ``model(batch)``
  call.
* Using :meth:`~nvalchemi.models.ewald.EwaldModelWrapper.invalidate_cache` when
  the simulation cell changes between evaluations.
"""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.hooks import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.ewald import EwaldModelWrapper

# %%
# Ewald model setup
# ------------------
# :class:`~nvalchemi.models.ewald.EwaldModelWrapper` only needs a real-space
# cutoff.  The Ewald splitting parameter α and reciprocal-space cutoff are
# estimated automatically from the ``accuracy`` target (default 1e-6) each
# time the simulation cell changes.  For a fixed-cell run the cache is
# computed once and reused.

CUTOFF = 10.0  # Å — real-space cutoff
model = EwaldModelWrapper(cutoff=CUTOFF, accuracy=1e-6)
# active_outputs defaults to {"energy", "forces"} — no change needed here.

# %%
# Building a NaCl rock-salt supercell
# -------------------------------------
# The NaCl rock-salt lattice has two interpenetrating FCC sub-lattices.
# We build a 2×2×2 conventional cubic supercell (64 atoms).
#
# **Important**: Na and Cl positions are collected separately, then
# concatenated, so that ``atomic_numbers`` and ``charges`` align
# correctly with ``positions``.  Interleaving the two species within the
# same image ordering would silently mis-assign charges.
#
# Lattice constant a₀ = 5.64 Å (experimental value).
# Formal charges: Na → +1 e, Cl → −1 e.
# The system is overall charge-neutral (32 Na + 32 Cl).

A0 = 5.64  # Å — NaCl lattice constant
SUPERCELL = 2  # repeat units along each axis
M_NA = 22.990  # amu
M_CL = 35.453  # amu

cell_size = SUPERCELL * A0  # 11.28 Å per side

# Fractional coordinates of the two sub-lattices within one conventional cell.
# Na occupies the FCC vertices; Cl is offset by (0.5, 0, 0) etc.
_na_basis = torch.tensor(
    [[0.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]],
    dtype=torch.float32,
)  # 4 Na per unit cell
_cl_basis = torch.tensor(
    [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5], [0.5, 0.5, 0.5]],
    dtype=torch.float32,
)  # 4 Cl per unit cell

# Supercell image offsets (in unit-cell units).
offsets = torch.tensor(
    [
        [ix, iy, iz]
        for ix in range(SUPERCELL)
        for iy in range(SUPERCELL)
        for iz in range(SUPERCELL)
    ],
    dtype=torch.float32,
)  # (8, 3)

# Build Na positions: (n_images, n_basis, 3) → (N_Na, 3).
# Dividing by SUPERCELL converts unit-cell fractional to supercell fractional;
# multiplying by cell_size converts to Cartesian Å.
na_positions = ((_na_basis.unsqueeze(0) + offsets.unsqueeze(1)) / SUPERCELL).reshape(
    -1, 3
) * cell_size
cl_positions = ((_cl_basis.unsqueeze(0) + offsets.unsqueeze(1)) / SUPERCELL).reshape(
    -1, 3
) * cell_size

positions = torch.cat([na_positions, cl_positions], dim=0)  # (64, 3)
n_na, n_cl = na_positions.shape[0], cl_positions.shape[0]  # 32, 32
n_atoms = n_na + n_cl  # 64

# Atomic numbers and formal charges — correctly aligned with positions.
atomic_numbers = torch.cat(
    [
        torch.full((n_na,), 11, dtype=torch.long),
        torch.full((n_cl,), 17, dtype=torch.long),
    ]
)
charges = torch.cat(
    [
        torch.ones(
            n_na,
        ),
        -torch.ones(
            n_cl,
        ),
    ]
)  # (N,)

cell = torch.eye(3).unsqueeze(0) * cell_size  # (1, 3, 3)

data = AtomicData(
    positions=positions,
    atomic_numbers=atomic_numbers,
    charges=charges,  # (N, 1) — required shape for AtomicData
    forces=torch.zeros(n_atoms, 3),
    energy=torch.zeros(1, 1),
    cell=cell,
    pbc=torch.tensor([[True, True, True]]),
)
batch = Batch.from_data_list([data])

print(
    f"System: {n_na} Na + {n_cl} Cl, box={cell_size:.2f} Å, "
    f"nearest Na–Cl distance={A0 / 2:.3f} Å"
)

# %%
# Building the neighbor list and evaluating energy + forces
# ----------------------------------------------------------
# For a one-shot energy evaluation, build the neighbor list manually using
# :class:`~nvalchemi.hooks.NeighborListHook` outside the dynamics loop.

nl_hook = NeighborListHook(
    model.model_config.neighbor_config, stage=DynamicsStage.BEFORE_COMPUTE
)
# Create a minimal HookContext for one-time neighbor list build
ctx = HookContext(
    batch=batch,
    step_count=0,
    model=model,
    converged_mask=None,
    global_rank=0,
)
nl_hook(
    ctx, DynamicsStage.BEFORE_COMPUTE
)  # populates batch.neighbor_matrix / batch.num_neighbors

result = model(batch)

energy_eV = result["energy"].item()
forces = result["forces"]  # (N, 3) eV/Å

print(f"\nEwald energy: {energy_eV:.4f} eV")
print(f"Max force magnitude: {forces.norm(dim=-1).max().item():.4f} eV/Å")

# %%
# Comparison with the analytical Madelung energy
# ------------------------------------------------
# For an infinite NaCl rock-salt crystal the total Coulomb energy per
# formula unit is:
#
#   E₀ = −A * k_e * q² / r_{Na–Cl}
#
# where A = 1.7476 is the Madelung constant for the rock-salt structure,
# k_e = 14.3996 eV·Å/e², q = 1 e, and r_{Na–Cl} = a₀/2.
#
# For a finite supercell the Ewald result converges to the thermodynamic
# limit to within the accuracy set in the model (1e-6 here).

MADELUNG_NaCL = 1.7476
K_E = 14.3996  # eV·Å/e²
r_nn = A0 / 2  # nearest-neighbour Na–Cl distance

n_formula_units = n_na  # 32 (one per Na–Cl pair)
E_madelung = -MADELUNG_NaCL * K_E * 1.0**2 / r_nn * n_formula_units

print(f"\nAnalytical Madelung energy ({n_formula_units} f.u.): {E_madelung:.4f} eV")
print(f"Ewald energy:                                    {energy_eV:.4f} eV")
print(
    f"Relative error:                                  {abs(energy_eV - E_madelung) / abs(E_madelung):.2e}"
)

# %%
# Forces should vanish at the perfect-crystal geometry
# -----------------------------------------------------
# At the ideal rock-salt lattice sites every atom sits at a potential minimum
# by symmetry, so the net force on each atom is zero.  Finite cell and
# numerical accuracy leave a small residual.

mean_force = forces.norm(dim=-1).mean().item()
print(
    f"\nMean force magnitude at ideal geometry: {mean_force:.4e} eV/Å  (should be ≈0)"
)

# %%
# Evaluating with a perturbed geometry
# --------------------------------------
# Displace atoms by a small random amount and re-evaluate to show non-zero forces.
# Call :meth:`~nvalchemi.models.ewald.EwaldModelWrapper.invalidate_cache` if the
# cell changes between evaluations (here the cell is unchanged, so not required).

torch.manual_seed(7)
perturbed_positions = positions + 0.05 * torch.randn_like(positions)

data_pert = AtomicData(
    positions=perturbed_positions,
    atomic_numbers=atomic_numbers,
    charges=charges,
    forces=torch.zeros(n_atoms, 3),
    energy=torch.zeros(1, 1),
    cell=cell,
    pbc=torch.tensor([[True, True, True]]),
)
batch_pert = Batch.from_data_list([data_pert])
# Create HookContext for perturbed batch
ctx_pert = HookContext(
    batch=batch_pert,
    step_count=0,
    model=model,
    converged_mask=None,
    global_rank=0,
)
nl_hook(ctx_pert, DynamicsStage.BEFORE_COMPUTE)

result_pert = model(batch_pert)

print("\nPerturbed geometry:")
print(f"  Ewald energy:         {result_pert['energy'].item():.4f} eV")
print(
    f"  Max force magnitude:  {result_pert['forces'].norm(dim=-1).max().item():.4f} eV/Å"
)
