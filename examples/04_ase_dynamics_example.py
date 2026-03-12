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
Real-World Dynamics with ASE-Built Structures
==============================================

This example builds on :doc:`02_dynamics_example` but replaces synthetic data
with real atomic systems constructed via the `ASE <https://wiki.fysik.dtu.dk/ase/>`_
build module.

* **Part 1** — FIRE geometry optimization of rattled molecules (H2O, CH4, CH3CH2OH).
* **Part 2** — FusedStage (FIRE + NVT Langevin) on a Cu(111) slab with a CO
  adsorbate — a classic surface-science system.  Slab atoms are frozen with
  :class:`~nvalchemi.dynamics.hooks.FreezeAtomsHook`.

.. note::

    :class:`~nvalchemi.models.demo.DemoModelWrapper` is used throughout.
    It produces fictitious energies/forces, so the trajectories are *not*
    physically meaningful.  Replace it with a trained MLIP for real science.
"""

from __future__ import annotations

from pathlib import Path

import torch
from ase import Atoms
from ase.build import fcc111, molecule
from ase.io import write

from nvalchemi._typing import AtomCategory
from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE, NVTLangevin
from nvalchemi.dynamics.base import ConvergenceHook
from nvalchemi.dynamics.hooks import FreezeAtomsHook
from nvalchemi.models.demo import DemoModelWrapper

OUTPUT_DIR = Path("04_ase_dynamics_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# %%
# Setup — model
# -------------
torch.manual_seed(0)
model = DemoModelWrapper()
model.eval()


# %%
# Helper — ASE Atoms to AtomicData
# ---------------------------------
# :meth:`AtomicData.from_atoms` handles positions, atomic_numbers, cell, and
# pbc.  Integrators also need ``forces``, ``energies``, and ``velocities``
# pre-allocated so ``compute()`` can write into them.


def atoms_to_data(atoms) -> AtomicData:
    """Convert an ASE Atoms object to AtomicData with dynamics fields."""
    data = AtomicData.from_atoms(atoms)
    n = data.num_nodes
    data.forces = torch.zeros(n, 3)
    data.energies = torch.zeros(1, 1)
    data.add_node_property("velocities", torch.zeros(n, 3))
    return data


def data_to_atoms(data: AtomicData) -> Atoms:
    """Convert AtomicData back to ASE Atoms for visualization / I/O."""
    atoms = Atoms(
        numbers=data.atomic_numbers.cpu().numpy(),
        positions=data.positions.detach().cpu().numpy(),
    )
    if data.cell is not None:
        atoms.cell = data.cell.squeeze(0).detach().cpu().numpy()
    if data.pbc is not None:
        atoms.pbc = data.pbc.squeeze(0).cpu().numpy()
    return atoms


def batch_to_atoms_list(batch: Batch) -> list[Atoms]:
    """Convert every graph in a Batch to a list of ASE Atoms."""
    return [data_to_atoms(d) for d in batch.to_data_list()]


# %%
# Part 1: FIRE Geometry Optimization of Rattled Molecules
# --------------------------------------------------------
# Build three molecules from the ASE G2 database, rattle them slightly so
# the optimizer has work to do, and relax them in a single batched FIRE run.

print("=== Part 1: FIRE Optimization — Rattled Molecules ===")

molecules = []
for name, seed in [("H2O", 1), ("CH4", 2), ("CH3CH2OH", 3)]:
    mol = molecule(name)
    mol.rattle(stdev=0.15, seed=seed)
    mol.center(vacuum=5.0)
    molecules.append(mol)
    print(f"  {name}: {len(mol)} atoms, rattled")

write(OUTPUT_DIR / "molecules_initial.xyz", molecules)
print(
    f"  Wrote {len(molecules)} initial structures -> {OUTPUT_DIR}/molecules_initial.xyz"
)

data_list_opt = [atoms_to_data(mol) for mol in molecules]
batch_opt = Batch.from_data_list(data_list_opt)
print(f"\nBatch: {batch_opt.num_graphs} systems, {batch_opt.num_nodes} atoms total\n")

fire_opt = FIRE(
    model=model,
    dt=0.1,
    n_steps=200,
    convergence_hook=ConvergenceHook(
        criteria=[
            {"key": "forces", "threshold": 0.05, "reduce_op": "norm", "reduce_dims": -1}
        ]
    ),
)

batch_opt = fire_opt.run(batch_opt)
print(f"\nCompleted {fire_opt.step_count} FIRE steps.")

relaxed_molecules = batch_to_atoms_list(batch_opt)
write(OUTPUT_DIR / "molecules_relaxed.xyz", relaxed_molecules)
print(
    f"Wrote {len(relaxed_molecules)} relaxed structures -> {OUTPUT_DIR}/molecules_relaxed.xyz"
)

# %%
# Part 2: FusedStage — Surface + Adsorbate System with Frozen Slab
# ------------------------------------------------------------------
# Build a Cu(111) slab with a CO molecule adsorbed on top.  This is a
# textbook surface-science setup: relax the adsorbate geometry with FIRE
# (status 0), then run NVT Langevin MD at 300 K (status 1).
#
# The copper slab atoms are frozen using :class:`~nvalchemi.dynamics.hooks.FreezeAtomsHook`,
# which preserves their positions across dynamics steps while the CO
# adsorbate atoms move freely.  Atoms to freeze are identified via
# :attr:`~nvalchemi.data.AtomicData.atom_categories`: slab atoms are
# marked with :attr:`AtomCategory.SPECIAL` (the default freeze target).
#
# We create three copies with different random rattles to form a batch.

print("\n\n=== Part 2: FusedStage — Cu(111) + CO Adsorbate ===")

slab_base = fcc111("Cu", size=(2, 2, 3), vacuum=10.0)

co = molecule("CO")

adsorbate_systems = []
for seed in [10, 11, 12]:
    slab = slab_base.copy()

    # Place CO above the top Cu layer (on-top site)
    top_z = slab.positions[:, 2].max()
    co_copy = co.copy()
    co_copy.translate([slab.cell[0, 0] / 2, slab.cell[1, 1] / 3, top_z + 1.8])

    system = slab + co_copy  # combine slab and adsorbate
    system.rattle(stdev=0.05, seed=seed)
    adsorbate_systems.append(system)
    print(
        f"  System (seed={seed}): {len(system)} atoms "
        f"({len(slab)} slab + {len(co)} adsorbate)"
    )

write(OUTPUT_DIR / "cu111_co_initial.xyz", adsorbate_systems)
print(
    f"  Wrote {len(adsorbate_systems)} initial slab+CO structures "
    f"-> {OUTPUT_DIR}/cu111_co_initial.xyz"
)

data_list_fused = [atoms_to_data(sys) for sys in adsorbate_systems]
batch_fused = Batch.from_data_list(data_list_fused)

# Mark slab atoms for freezing.  ``from_atoms`` maps ASE tags to
# AtomCategory: tag 0 (adsorbate) -> GAS, tag 1 -> SURFACE, tag >= 2 -> BULK.
# Override SURFACE and BULK to SPECIAL so FreezeAtomsHook freezes them.
cats = batch_fused.atom_categories
slab_mask = cats != AtomCategory.GAS.value
cats[slab_mask] = AtomCategory.SPECIAL.value

n_frozen = int(slab_mask.sum().item())
n_free = int((~slab_mask).sum().item())
print(f"  Frozen (slab): {n_frozen} atoms, Free (adsorbate): {n_free} atoms")

# All systems start in the FIRE stage (status = 0).
batch_fused["status"] = torch.zeros(batch_fused.num_graphs, 1, dtype=torch.long)

print(
    f"\nBatch: {batch_fused.num_graphs} systems, {batch_fused.num_nodes} atoms total\n"
)

# FreezeAtomsHook: keeps slab positions fixed, zeros their velocities and forces.
# Both stages share the same hook instance so the snapshot/restore logic is
# consistent across the FIRE -> Langevin transition.
freeze_hook = FreezeAtomsHook()

# FIRE sub-stage (relaxation) — only adsorbate atoms relax
fire_stage = FIRE(
    model=model,
    dt=0.1,
    convergence_hook=ConvergenceHook(
        criteria=[
            {"key": "forces", "threshold": 0.05, "reduce_op": "norm", "reduce_dims": -1}
        ]
    ),
    hooks=[freeze_hook],
    n_steps=200,
)

# NVT Langevin sub-stage (MD at 300 K) — slab remains frozen
langevin_stage = NVTLangevin(
    model=model,
    dt=0.5,
    temperature=300.0,
    friction=0.1,
    random_seed=42,
    hooks=[freeze_hook],
    n_steps=200,
)

# Compose: status 0 → FIRE, status 1 → Langevin
fused = fire_stage + langevin_stage
print(f"Created: {fused}\n")

n_fused_steps = 450
batch_fused = fused.run(batch_fused, n_steps=n_fused_steps)

status_final = batch_fused.status.squeeze(-1).tolist()
print(f"\nFinal status: {status_final}  (0=FIRE, 1=Langevin)")
print(f"FusedStage total steps: {fused.step_count}")

final_systems = batch_to_atoms_list(batch_fused)
write(OUTPUT_DIR / "cu111_co_final.xyz", final_systems)
print(f"Wrote {len(final_systems)} final structures -> {OUTPUT_DIR}/cu111_co_final.xyz")

print(f"\nAll output written to {OUTPUT_DIR.resolve()}/")
print("  Visualize with: ase gui 04_ase_dynamics_output/cu111_co_initial.xyz")
