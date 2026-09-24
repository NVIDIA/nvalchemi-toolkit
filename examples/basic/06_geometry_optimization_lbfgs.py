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
L-BFGS Geometry Optimization with Lennard-Jones Argon
======================================================

:class:`~nvalchemi.dynamics.optimizers.LBFGS` is a drop-in alternative to
:class:`~nvalchemi.dynamics.optimizers.FIRE2`: same constructor shape, same
hooks, same convergence criterion, one force evaluation per step.  It builds a
quasi-Newton direction from the last ``history_size`` position/force
differences, so it usually needs far fewer steps.

This example relaxes the same batch of argon clusters with both optimizers and
compares the number of force evaluations.
"""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE2, LBFGS, ConvergenceHook, DynamicsStage
from nvalchemi.models.lj import LennardJonesModelWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# Model and systems
# -----------------
# Lennard-Jones argon, as in the FIRE example.  Three clusters of different
# sizes are batched together; ragged batches need no special handling.

model = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5).to(device)
R_MIN = 2 ** (1 / 6) * 3.40  # LJ equilibrium pair distance, ≈ 3.82 Å


def make_cluster(n_per_side: int, seed: int) -> AtomicData:
    """Perturbed simple-cubic argon cluster with ``n_per_side**3`` atoms."""
    coords = torch.arange(n_per_side, dtype=torch.float32) * R_MIN * 1.1
    grid = torch.stack(torch.meshgrid(coords, coords, coords, indexing="ij"), -1)
    positions = grid.reshape(-1, 3)
    positions += 0.1 * torch.randn(
        positions.shape, generator=torch.Generator().manual_seed(seed)
    )
    n = len(positions)
    return AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n,), 18, dtype=torch.long),
        forces=torch.zeros(n, 3),
        energy=torch.zeros(1, 1),
        velocities=torch.zeros(n, 3),  # read by FIRE2 only
    )


def make_batch() -> Batch:
    """Batch of 8-, 27- and 64-atom clusters."""
    return Batch.from_data_list(
        [make_cluster(2, 0), make_cluster(3, 1), make_cluster(4, 2)]
    ).to(device)


# %%
# Relax with FIRE2, then with L-BFGS
# ----------------------------------
# Only the class and its hyperparameters change.  ``convergence_hook`` stops
# the run once every system's fmax is below the threshold.


def relax(optimizer) -> Batch:
    """Attach neighbor-list hooks and run *optimizer* on a fresh batch."""
    for hook in model.make_neighbor_hooks():
        optimizer.register_hook(hook, stage=DynamicsStage.BEFORE_COMPUTE)
    return optimizer.run(make_batch())


def fmax(batch: Batch) -> float:
    """Largest per-atom force norm in *batch*."""
    return batch.forces.norm(dim=-1).max().item()


converged = ConvergenceHook.from_fmax(1e-3)

# FIRE2 timestep tuned for LJ argon (the default tmax=0.08 is much slower here).
fire2 = FIRE2(model=model, dt=0.5, tmax=1.0, n_steps=2000, convergence_hook=converged)
fire2_batch = relax(fire2)

lbfgs = LBFGS(
    model=model,
    history_size=6,  # stored curvature pairs
    maxstep=0.2,  # largest displacement per step (Å)
    n_steps=2000,
    convergence_hook=converged,
)
lbfgs_batch = relax(lbfgs)

print(f"FIRE2 : {fire2.step_count:4d} steps, fmax {fmax(fire2_batch):.1e} eV/Å")
print(f"L-BFGS: {lbfgs.step_count:4d} steps, fmax {fmax(lbfgs_batch):.1e} eV/Å")

# %%
# Final energies
# --------------
# Each system is relaxed to a local minimum.  Small clusters are floppy, so
# the two optimizers can settle into different (nearby) minima.

for i, (e_fire2, e_lbfgs) in enumerate(
    zip(
        fire2_batch.energy.squeeze(-1).tolist(), lbfgs_batch.energy.squeeze(-1).tolist()
    )
):
    print(f"sys{i}: E(FIRE2) = {e_fire2:+.5f} eV   E(L-BFGS) = {e_lbfgs:+.5f} eV")
