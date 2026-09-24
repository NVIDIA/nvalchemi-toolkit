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
Variable-Cell L-BFGS: Relaxing FCC Argon
========================================

:class:`~nvalchemi.dynamics.optimizers.LBFGSVariableCell` relaxes atomic
positions and the simulation cell together, driven by the model's stress.  It
is used exactly like
:class:`~nvalchemi.dynamics.optimizers.FIRE2VariableCell`:

* the model must return tensile-positive ``stress``;
* cells must be aligned, so install
  :class:`~nvalchemi.dynamics.hooks.AlignCellHook` (``frequency=1``).

Two strained, sheared FCC argon crystals are relaxed with both optimizers;
each should recover the Lennard-Jones equilibrium lattice constant.
"""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    ConvergenceHook,
    DynamicsStage,
    FIRE2VariableCell,
    LBFGSVariableCell,
)
from nvalchemi.dynamics.hooks import AlignCellHook
from nvalchemi.models.lj import LennardJonesModelWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

# %%
# LJ model with stress
# --------------------
# Stress is opt-in on the LJ wrapper.

model = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5).to(device)
model.set_config("active_outputs", {"energy", "forces", "stress"})

# %%
# Strained FCC crystals
# ---------------------
# A 2x2x2 conventional FCC supercell (32 atoms), stretched and sheared away from
# equilibrium.  The sheared cell is not aligned; ``AlignCellHook`` rotates it
# into the aligned (lower-triangular) frame before the first step.


def make_crystal(a: float, shear: float, seed: int) -> AtomicData:
    """Perturbed 32-atom FCC argon cell with lattice constant *a* and a shear."""
    basis = torch.tensor(
        [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]], dtype=dtype
    )
    shifts = torch.tensor(
        [[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=dtype
    )
    frac = ((basis[None] + shifts[:, None]) / 2).reshape(-1, 3)
    cell = 2 * a * torch.eye(3, dtype=dtype)
    cell[0, 1] = shear * 2 * a  # skew: an entry above the diagonal
    positions = frac @ cell.T
    positions += 0.05 * torch.randn(
        positions.shape, generator=torch.Generator().manual_seed(seed), dtype=dtype
    )
    n = len(positions)
    return AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n,), 18, dtype=torch.long),
        cell=cell.unsqueeze(0),
        pbc=torch.tensor([[True, True, True]]),
        forces=torch.zeros(n, 3, dtype=dtype),
        energy=torch.zeros(1, 1, dtype=dtype),
        stress=torch.zeros(1, 3, 3, dtype=dtype),  # filled by the model
        velocities=torch.zeros(n, 3, dtype=dtype),  # read by FIRE2 only
    )


def make_batch() -> Batch:
    """Batch of a stretched and a compressed, sheared crystal."""
    return Batch.from_data_list(
        [make_crystal(5.6, 0.05, 0), make_crystal(5.0, 0.10, 1)]
    ).to(device)


# %%
# Convergence on forces and stress
# --------------------------------
# A system is converged when its fmax *and* its stress norm are small.

converged = ConvergenceHook(
    criteria=[
        {"key": "forces", "threshold": 1e-3, "reduce_op": "norm", "reduce_dims": -1},
        {
            "key": "stress",
            "threshold": 1e-5,
            "reduce_op": "norm",
            "reduce_dims": [-2, -1],
        },
    ]
)


def relax(optimizer) -> Batch:
    """Attach neighbor-list hooks and run *optimizer* on a fresh batch."""
    for hook in model.make_neighbor_hooks():
        optimizer.register_hook(hook, stage=DynamicsStage.BEFORE_COMPUTE)
    return optimizer.run(make_batch())


# %%
# Relax with FIRE2VariableCell, then LBFGSVariableCell
# ----------------------------------------------------
# ``cell_force_scale`` (both classes, default 1.0) scales how far the cell moves
# per step relative to the atoms; the default is used here.

fire2 = FIRE2VariableCell(
    model=model,
    dt=0.5,
    tmax=1.0,
    n_steps=3000,
    hooks=[AlignCellHook()],
    convergence_hook=converged,
)
fire2_batch = relax(fire2)

lbfgs = LBFGSVariableCell(
    model=model,
    n_steps=3000,
    hooks=[AlignCellHook()],
    convergence_hook=converged,
)
lbfgs_batch = relax(lbfgs)

# %%
# Results
# -------
# The LJ FCC equilibrium lattice constant with this cutoff is about 5.27 Å.


def lattice_constant(batch: Batch) -> list[float]:
    """Cubic lattice constant per system from the cell volume."""
    # 32-atom cell = 2x2x2 conventional cells.
    return (torch.linalg.det(batch.cell).abs() ** (1 / 3) / 2).tolist()


for name, opt, batch in (
    ("FIRE2VariableCell", fire2, fire2_batch),
    ("LBFGSVariableCell", lbfgs, lbfgs_batch),
):
    a = ", ".join(f"{x:.4f}" for x in lattice_constant(batch))
    print(f"{name}: {opt.step_count:4d} steps, a = [{a}] Å")
