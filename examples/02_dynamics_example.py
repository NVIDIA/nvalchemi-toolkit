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
FIRE Optimization and FusedStage (FIRE + Langevin): LJ Dynamics Demo
=====================================================================

This example walks through two use cases of the :mod:`nvalchemi.dynamics`
framework using the Warp-accelerated Lennard-Jones potential via
:class:`~nvalchemi.models.lj.LennardJonesModelWrapper` (argon parameters).

A :class:`~nvalchemi.dynamics.hooks.NeighborListHook` is registered on each
dynamics object so that the dense neighbor matrix is recomputed (or read from
a Verlet cache) before every model forward pass.

**Part 1** — :class:`~nvalchemi.dynamics.optimizers.FIRE` geometry optimization.
A :class:`~nvalchemi.dynamics.base.ConvergenceHook` detects convergence
(fmax < 0.001 eV/Å) and fires an ``ON_CONVERGE`` hook; an ``AFTER_STEP`` hook
logs progress every N steps.

**Part 2** — A :class:`~nvalchemi.dynamics.base.FusedStage` that shares one
model forward pass across FIRE (status 0) and NVT Langevin MD (status 1).
The ``+`` operator composes the two sub-stages and auto-registers a
:class:`~nvalchemi.dynamics.base.ConvergenceHook` that migrates relaxed systems
from status 0 → 1.

.. note::

    LJ forces are computed analytically by the Warp kernel rather than via
    autograd, so :attr:`~nvalchemi.models.base.ModelCard.forces_are_conservative`
    is ``False``.  This has no effect on FIRE or NVT Langevin, which only
    consume the force values written to ``batch.forces``.
"""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE, HookStageEnum, NVTLangevin
from nvalchemi.dynamics.hooks import NeighborListHook
from nvalchemi.models.lj import LennardJonesModelWrapper

# %%
# Setup — LJ model and neighbor-list hook factory
# -------------------------------------------------
# Argon LJ parameters:
#   epsilon = 0.0104  eV   (potential well depth)
#   sigma   = 3.40    Å    (zero-crossing distance)
#   cutoff  = 8.5     Å    (≈ 2.5 σ, standard choice for argon)
#
# ``max_neighbors=32`` is generous for small clusters; tune upward for dense
# periodic systems.  ``skin=0.5`` means the list is only rebuilt when any atom
# has moved more than 0.25 Å since the last build.

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
# Progress logger — energy and fmax every N steps
# -------------------------------------------------


class _ProgressHook:
    """AFTER_STEP hook that logs per-system energy and max atomic force."""

    stage = HookStageEnum.AFTER_STEP

    def __init__(self, label: str, frequency: int = 50) -> None:
        self.label = label
        self.frequency = frequency

    def __call__(self, batch: Batch, dynamics) -> None:
        step = dynamics.step_count + 1  # step_count increments after hooks fire
        energies = batch.energies.squeeze(-1)  # (B,)
        force_norms = batch.forces.norm(dim=-1)  # (N,)
        fmax = torch.zeros(batch.num_graphs, device=batch.device)
        fmax.scatter_reduce_(
            0, batch.batch, force_norms, reduce="amax", include_self=True
        )

        has_status = getattr(batch, "status", None) is not None
        rows = []
        for i in range(batch.num_graphs):
            line = f"  sys{i}: E={energies[i].item():+.4f} eV  fmax={fmax[i].item():.4f} eV/Å"
            if has_status:
                line += f"  status={int(batch.status.view(-1)[i].item())}"
            rows.append(line)
        print(f"[{self.label}] step {step:4d}\n" + "\n".join(rows))


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
        energies=torch.zeros(1, 1),
        velocities=torch.zeros(n_atoms, 3),
    )


# %%
# Part 1: FIRE Geometry Optimization
# ------------------------------------
# Build a batch of two 2×2×2 (8-atom) Argon clusters and relax with FIRE.
#
# The NeighborListHook fires at BEFORE_COMPUTE and writes ``neighbor_matrix``
# and ``num_neighbors`` into the batch atoms group before each model evaluation.

print("=== Part 1: FIRE Geometry Optimization ===")

# Two identical lattice sizes; different spacings give different starting energies.
data_list_opt = [
    _make_system(2, spacing=_R_MIN * 1.05),
    _make_system(2, spacing=_R_MIN * 1.20),
]
batch_opt = Batch.from_data_list(data_list_opt)
print(f"Batch: {batch_opt.num_graphs} systems, {batch_opt.num_nodes} atoms total\n")

from nvalchemi.dynamics.base import ConvergenceHook

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
fire_opt.register_hook(_ProgressHook("FIRE-opt", frequency=50))

batch_opt = fire_opt.run(batch_opt)
print(f"\nCompleted {fire_opt.step_count} FIRE steps.")

# %%
# Part 2: FusedStage — FIRE Relaxation → NVT Langevin MD
# --------------------------------------------------------
# A :class:`~nvalchemi.dynamics.base.FusedStage` composes sub-stages that
# share **one** model forward pass per step.  Each system carries a ``status``
# field that routes it to the corresponding sub-stage:
#
# * **status = 0** → processed by FIRE (relaxation phase)
# * **status = 1** → processed by NVTLangevin (MD sampling phase)
#
# One NeighborListHook is registered on the FusedStage so the list is built
# once per fused step (shared across sub-stages).

print("\n\n=== Part 2: FusedStage — FIRE + NVTLangevin ===")

data_list_fused = [
    _make_system(2, spacing=_R_MIN * 1.05),
    _make_system(2, spacing=_R_MIN * 1.30),
    _make_system(2, spacing=_R_MIN * 0.95),  # compressed — needs more relaxation
]
batch_fused = Batch.from_data_list(data_list_fused)
batch_fused["status"] = torch.zeros(batch_fused.num_graphs, 1, dtype=torch.long)

fire_stage = FIRE(
    model=model,
    dt=0.5,
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
    n_steps=300,
)

langevin_stage = NVTLangevin(
    model=model,
    dt=0.5,
    temperature=50.0,  # K — below Ar boiling point (~87 K) so cluster stays bound
    friction=0.1,
    random_seed=42,
    n_steps=300,
)

# ``fire_stage + langevin_stage`` builds the FusedStage and auto-registers
# a ConvergenceHook that migrates systems from status 0 → 1.
fused = fire_stage + langevin_stage
print(f"Created: {fused}\n")

# Register the neighbor-list hook on the fused stage so it fires once per
# fused step, before the shared model forward pass.
fused.register_hook(neighbor_hook)
fused.register_hook(_ProgressHook("FusedStage", frequency=100))

batch_fused = fused.run(batch_fused, n_steps=450)

status_final = batch_fused.status.squeeze(-1).tolist()
print(f"\nFinal status: {status_final}  (0=FIRE, 1=Langevin)")
print(f"FusedStage total steps: {fused.step_count}")
