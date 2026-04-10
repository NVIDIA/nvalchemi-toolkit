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
Multi-Criteria Convergence with Custom Operators
=================================================

:class:`~nvalchemi.dynamics.base.ConvergenceHook` detects when geometry
optimisation or MD has reached a desired stopping condition and removes
converged systems from the active batch so subsequent steps spend compute only
on unconverged systems.

A convergence hook holds a list of **criteria**, each of which evaluates a
tensor attribute of the batch and returns a per-system boolean.  A system is
declared converged when **every** criterion is satisfied simultaneously.  This
AND-logic prevents false positives: a system with near-zero forces but still
large energy fluctuations is not yet converged.

Key concepts demonstrated
--------------------------
* ``ConvergenceHook.from_forces(threshold)`` — the simplest one-liner.
* Multi-criteria convergence combining force norm **and** a second criterion.
* ``custom_op`` — a callable that receives the raw tensor and returns a ``[B]``
  bool mask, used here to implement an energy-change criterion.
* Combining force-norm and energy-change criteria in a FIRE optimisation.
"""

from __future__ import annotations

import logging

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE
from nvalchemi.dynamics.base import ConvergenceHook, DynamicsStage
from nvalchemi.dynamics.hooks import NeighborListHook
from nvalchemi.hooks import HookContext
from nvalchemi.models.lj import LennardJonesModelWrapper

logging.basicConfig(level=logging.INFO)

# %%
# LJ model and helper
# --------------------

LJ_EPSILON = 0.0104
LJ_SIGMA = 3.40
LJ_CUTOFF = 8.5
_R_MIN = 2 ** (1 / 6) * LJ_SIGMA

model = LennardJonesModelWrapper(
    epsilon=LJ_EPSILON,
    sigma=LJ_SIGMA,
    cutoff=LJ_CUTOFF,
    max_neighbors=32,
)


def _make_cluster(
    n_per_side: int = 2, spacing_factor: float = 1.05, seed: int = 0
) -> AtomicData:
    n = n_per_side**3
    spacing = _R_MIN * spacing_factor
    coords = torch.arange(n_per_side, dtype=torch.float32) * spacing
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    torch.manual_seed(seed)
    positions = positions + 0.05 * torch.randn_like(positions)
    return AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n,), 18, dtype=torch.long),
        forces=torch.zeros(n, 3),
        energy=torch.zeros(1, 1),
        velocities=torch.zeros(n, 3),
    )


# %%
# Built-in convergence: from_forces factory
# -------------------------------------------
# ``ConvergenceHook.from_forces(threshold)`` is the standard one-liner for
# geometry optimisation.  It reads the ``"forces"`` key from the batch,
# computes the per-atom Euclidean norm, takes the max over atoms within each
# graph (scatter reduce), and declares convergence when that max norm ≤
# *threshold*.

simple_hook = ConvergenceHook.from_forces(threshold=0.05)
print("Simple hook:", simple_hook)

# %%
# Multi-criteria convergence
# ---------------------------
# Pass a list of criterion dicts.  Each dict maps to a
# :class:`~nvalchemi.dynamics.base._ConvergenceCriterion`.
#
# ``reduce_op="norm"`` computes the per-atom vector norm along ``reduce_dims=-1``
# (last axis = Cartesian), yielding a ``[N]`` scalar per atom.
# The criterion then scatter-reduces to graph level via max before comparing to
# ``threshold``.
#
# You can add as many criteria as you like; all must be satisfied.

dual_hook = ConvergenceHook(
    criteria=[
        # Force criterion: max per-atom force norm ≤ 0.05 eV/Å.
        {
            "key": "forces",
            "threshold": 0.05,
            "reduce_op": "norm",
            "reduce_dims": -1,
        },
        # Energy criterion: per-system energy ≤ -0.1 eV (cluster is bound).
        # This guards against spurious convergence at high energy.
        {
            "key": "energy",
            "threshold": -0.1,
        },
    ]
)
print("Dual hook:", dual_hook)

# %%
# Custom operator: energy-change criterion
# -----------------------------------------
# Built-in ``reduce_op`` values cover many cases, but sometimes you need
# arbitrary logic.  ``custom_op`` receives the raw tensor for that key
# (whatever shape the batch stores it in) and must return a ``[B]`` bool
# tensor.
#
# Here we implement a **relative energy-change** criterion: a system is
# converged when :math:`|\Delta E / E| < \varepsilon` between consecutive steps.  This requires
# state — we track the previous energy in a closure.
#
# The ``custom_op`` callable is called with the full ``batch.energy``
# tensor of shape ``[B, 1]``.

prev_energy: dict[str, torch.Tensor] = {}  # mutable state accessible via closure


def energy_change_criterion(energy: torch.Tensor) -> torch.Tensor:
    """Return True for systems whose relative energy change is < 1e-4.

    Parameters
    ----------
    energy : torch.Tensor
        Shape ``[B, 1]`` — per-system total energy in eV.

    Returns
    -------
    torch.Tensor
        Shape ``[B]`` boolean — True where |ΔE/E| < 1e-4.
    """
    e = energy.squeeze(-1)  # [B]
    if "last" not in prev_energy:
        # First call: cannot compute delta, treat all as unconverged.
        prev_energy["last"] = e.detach().clone()
        return torch.zeros(e.shape[0], dtype=torch.bool, device=e.device)

    delta = (e - prev_energy["last"]).abs()
    denom = prev_energy["last"].abs().clamp(min=1e-12)
    rel_change = delta / denom
    prev_energy["last"] = e.detach().clone()
    return rel_change < 1e-4


custom_hook = ConvergenceHook(
    criteria=[
        {
            "key": "energy",
            "threshold": 0.0,  # threshold is ignored when custom_op is set
            "custom_op": energy_change_criterion,
        }
    ]
)
print("Custom hook:", custom_hook)

# %%
# Practical example: dual force + energy-change convergence
# ----------------------------------------------------------
# Combine a force-norm criterion with the energy-change criterion so that
# FIRE stops only when the optimizer has truly converged — both forces are
# small AND the energy is stable.
#
# The energy-change guard prevents early exit when the optimizer happens to
# take a near-zero force step during a large momentum phase.

print("\n=== FIRE with dual force+energy-change convergence ===")

# Reset the shared closure state for a clean run.
prev_energy.clear()

data_list = [
    _make_cluster(2, spacing_factor=1.05, seed=0),
    _make_cluster(2, spacing_factor=1.20, seed=1),
]
batch = Batch.from_data_list(data_list)
print(f"Batch: {batch.num_graphs} systems, {batch.num_nodes} atoms total\n")

dual_custom_hook = ConvergenceHook(
    criteria=[
        {
            "key": "forces",
            "threshold": 0.01,
            "reduce_op": "norm",
            "reduce_dims": -1,
        },
        {
            "key": "energy",
            "threshold": 0.0,
            "custom_op": energy_change_criterion,
        },
    ]
)

fire = FIRE(
    model=model,
    dt=0.5,
    n_steps=500,
    convergence_hook=dual_custom_hook,
)
fire.register_hook(NeighborListHook(model.model_card.neighbor_config))


class _LogHook:
    """Log energy and fmax every 50 steps."""

    stage = DynamicsStage.AFTER_STEP
    frequency = 50

    def __call__(self, ctx: HookContext, stage_: DynamicsStage) -> None:
        batch = ctx.batch
        step = ctx.step_count + 1
        energy = batch.energy.squeeze(-1)
        fmax_per_sys = torch.zeros(batch.num_graphs, device=batch.device)
        fmax_per_sys.scatter_reduce_(
            0,
            batch.batch_idx,
            batch.forces.norm(dim=-1),
            reduce="amax",
            include_self=True,
        )
        rows = [
            f"  sys{i}: E={energy[i].item():+.5f} eV  fmax={fmax_per_sys[i].item():.5f} eV/Å"
            for i in range(batch.num_graphs)
        ]
        print(f"[step {step:4d}]\n" + "\n".join(rows))


fire.register_hook(_LogHook())
batch = fire.run(batch)
print(f"\nCompleted {fire.step_count} FIRE steps (dual convergence).")

final_energy = batch.energy.squeeze(-1)
for i in range(batch.num_graphs):
    print(f"  sys{i}: final E = {final_energy[i].item():+.6f} eV")
