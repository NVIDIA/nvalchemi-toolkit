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
NVT MD with MACE (with LJ Fallback)
=====================================

MACE (Message-passing Atomic Cluster Expansion) is an equivariant
message-passing neural network potential with competitive accuracy on bulk and
molecular systems within its training distribution.  The MACE-MP family of
foundation models is pre-trained on the Materials Project and provides
reasonable zero-shot predictions across a broad range of compositions, though
accuracy should be validated for the specific system of interest before
production use — particularly for chemistries or structures not represented
in the Materials Project training set.

This example shows how to plug a MACE model into an NVT simulation using
:class:`~nvalchemi.models.mace.MACEWrapper`.  If MACE is not installed (or no
checkpoint path is provided), the example falls back to the Lennard-Jones
potential so that it can run in CI without any ML-potential dependency.

Key concepts demonstrated
--------------------------
* Loading a MACE model via ``MACEWrapper.from_checkpoint``.
* Reading ``model.model_config.neighbor_config`` to wire a
   :class:`~nvalchemi.hooks.NeighborListHook` automatically —
  the same code works for LJ (MATRIX format) and MACE (COO format).
* Model-agnostic temperature and energy observation.

Setting up MACE
---------------
Install the optional MACE dependency::

    pip install 'nvalchemi-toolkit[mace]'

Then point the ``MACE_MODEL_PATH`` environment variable to a MACE-MP
checkpoint, e.g. the medium model::

    export MACE_MODEL_PATH=medium-0b2   # named checkpoint (auto-downloads)
    # or a local path:
    export MACE_MODEL_PATH=/path/to/mace_mp_medium.pt

When ``MACE_MODEL_PATH`` is unset the example uses the LJ potential.
"""

from __future__ import annotations

import logging
import math
import os

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.dynamics.base import DynamicsStage

# KB_EV and kinetic_energy_per_graph are internal helpers used by the built-in
# integrators.  A stable public re-export may be added in a future release.
from nvalchemi.dynamics.hooks._utils import KB_EV, kinetic_energy_per_graph
from nvalchemi.hooks import NeighborListHook

logging.basicConfig(level=logging.INFO)

# %%
# Model selection: MACE or LJ fallback
# --------------------------------------
# The ``MACE_MODEL_PATH`` environment variable selects the model.  If it is
# unset (or MACE is not installed), we fall back to the LJ potential so the
# example works in CI without any MACE dependency.
#
# Both code paths produce a ``model`` object that satisfies the
# :class:`~nvalchemi.models.base.BaseModelMixin` interface, so all simulation
# code below is model-agnostic.

MACE_MODEL_PATH = os.environ.get("MACE_MODEL_PATH", "")
USE_MACE = False

if MACE_MODEL_PATH:
    try:
        from nvalchemi.models.mace import MACEWrapper

        model = MACEWrapper.from_checkpoint(
            checkpoint_path=MACE_MODEL_PATH,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        print(f"Using MACE model from {MACE_MODEL_PATH}")
        USE_MACE = True
    except Exception as exc:
        print(f"Failed to load MACE ({exc}). Falling back to LJ.")

if not USE_MACE:
    from nvalchemi.models.lj import LennardJonesModelWrapper

    model = LennardJonesModelWrapper(  # type: ignore[assignment]
        epsilon=0.0104,  # eV — argon ε
        sigma=3.40,  # Å  — argon σ
        cutoff=8.5,  # Å
    )
    print("Using LJ model (set MACE_MODEL_PATH=/path/to/model.pt to use MACE)")

# %%
# Neighbor list: automatic wiring via ModelConfig
# ------------------------------------------------
# :attr:`~nvalchemi.models.base.ModelConfig.neighbor_config` encodes the cutoff,
# list format (COO or MATRIX), and whether to use a half-list.
# :class:`~nvalchemi.hooks.NeighborListHook` reads this automatically.
#
# If ``neighbor_config`` is ``None`` (e.g. a demo model that does its own
# neighbour search), no hook is needed.

neighbor_hook = NeighborListHook(
    model.model_config.neighbor_config,
    stage=DynamicsStage.BEFORE_COMPUTE,
)

# %%
# Building the system
# --------------------
# When MACE is active we use a small water cluster (3 molecules, 9 atoms).
# With LJ we use an 8-atom argon cluster.  Both result in the same simulation
# code below.

_R_MIN_AR = 2 ** (1 / 6) * 3.40  # ≈ 3.82 Å


def _make_argon_cluster(n_per_side: int = 2, seed: int = 0) -> AtomicData:
    """8-atom argon cluster on a cubic lattice."""
    n = n_per_side**3
    spacing = _R_MIN_AR * 1.05
    coords = torch.arange(n_per_side, dtype=torch.float32) * spacing
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    torch.manual_seed(seed)
    positions = positions + 0.05 * torch.randn_like(positions)
    # Maxwell-Boltzmann at 300 K: v_std = sqrt(kB * T / m), m_Ar = 39.948 amu
    _v_std = math.sqrt(KB_EV * 300.0 / 39.948)
    return AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n,), 18, dtype=torch.long),  # Ar Z=18
        forces=torch.zeros(n, 3),
        energy=torch.zeros(1, 1),
        velocities=_v_std * torch.randn(n, 3),
    )


def _make_water_cluster(n_molecules: int = 3, seed: int = 0) -> AtomicData:
    """Small water cluster (H₂O): n_molecules × {O, H, H} atoms.

    Geometry: O at the origin of each molecule; H atoms at ±104.5° / 0.96 Å.
    Molecules are spaced 3.5 Å apart along the x-axis.
    """
    torch.manual_seed(seed)
    positions_list = []
    atomic_numbers_list = []

    o_h_bond = 0.96  # Å
    half_angle = math.radians(104.5 / 2)

    for i in range(n_molecules):
        ox = float(i) * 3.5
        # Oxygen
        o_pos = torch.tensor([ox, 0.0, 0.0])
        # Two hydrogens in the xz-plane
        h1_pos = o_pos + o_h_bond * torch.tensor(
            [math.sin(half_angle), 0.0, math.cos(half_angle)]
        )
        h2_pos = o_pos + o_h_bond * torch.tensor(
            [-math.sin(half_angle), 0.0, math.cos(half_angle)]
        )
        positions_list.extend([o_pos, h1_pos, h2_pos])
        atomic_numbers_list.extend([8, 1, 1])  # O=8, H=1

    positions = torch.stack(positions_list) + 0.02 * torch.randn(len(positions_list), 3)
    n_atoms = len(atomic_numbers_list)
    # Maxwell-Boltzmann at 300 K per species: O m=15.999 amu, H m=1.008 amu
    _masses = torch.tensor([15.999 if z == 8 else 1.008 for z in atomic_numbers_list])
    _v_stds = (KB_EV * 300.0 / _masses).sqrt().unsqueeze(-1)  # (n_atoms, 1)
    return AtomicData(
        positions=positions.float(),
        atomic_numbers=torch.tensor(atomic_numbers_list, dtype=torch.long),
        forces=torch.zeros(n_atoms, 3),
        energy=torch.zeros(1, 1),
        velocities=(_v_stds * torch.randn(n_atoms, 3)).float(),
    )


if USE_MACE:
    data = _make_water_cluster(n_molecules=3)
    system_label = "water cluster (9 atoms, 3 H₂O)"
else:
    data = _make_argon_cluster(n_per_side=2, seed=0)
    system_label = "argon cluster (8 atoms)"

sim_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch = Batch.from_data_list([data], device=sim_device)
print(f"\nSystem: {system_label}  →  {batch.num_nodes} atoms on {sim_device}")

# %%
# NVT MD simulation
# ------------------
# 100 steps of NVTLangevin at 300 K.  The neighbor hook (if any) fires at
# BEFORE_COMPUTE to rebuild the neighbor list before each forward pass.

nvt = NVTLangevin(
    model=model,
    dt=0.1,  # fs
    temperature=300.0,
    friction=0.5,
    n_steps=100,
    random_seed=99,
)

if neighbor_hook is not None:
    nvt.register_hook(neighbor_hook)

print("\nRunning 100 NVT steps …")
batch = nvt.run(batch)
print(f"Run complete: {nvt.step_count} steps")

# %%
# Model-agnostic observation
# ---------------------------
# Temperature and mean energy are computed the same way regardless of whether
# MACE or LJ provided the forces.

# Kinetic energy per graph (eV) → temperature.
# 2 KE = 3 N k_B T  →  T = 2 KE / (3 N k_B)
ke_per_graph = kinetic_energy_per_graph(
    velocities=batch.velocities,
    masses=batch.atomic_masses,
    batch_idx=batch.batch_idx,
    num_graphs=batch.num_graphs,
)  # [B, 1]

n_atoms_per_graph = batch.num_nodes_per_graph.float()  # [B]
T_inst = (2.0 * ke_per_graph.squeeze(-1)) / (3.0 * n_atoms_per_graph * KB_EV)

mean_E = batch.energy.squeeze(-1).mean().item()
mean_T = T_inst.mean().item()

print("\nFinal observables (model-agnostic):")
print(f"  Mean energy   : {mean_E:+.4f} eV")
print(f"  Inst. temp.   : {mean_T:.1f} K  (target = 300 K)")
print(f"  Model used    : {'MACE' if USE_MACE else 'Lennard-Jones (fallback)'}")
