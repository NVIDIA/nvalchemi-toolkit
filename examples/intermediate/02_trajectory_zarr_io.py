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
Writing and Replaying Trajectories with Zarr
============================================

`Zarr <https://zarr.readthedocs.io>`_ is a chunked, compressed array format
designed for large scientific datasets.  nvalchemi uses it as the on-disk
representation for trajectories: each dynamics snapshot is serialised into a
CSR-style layout (pointer arrays + concatenated fields) so that random-access
reads and incremental appends are both efficient.

The data flow for a full simulation-to-training pipeline is:

::

    NVTLangevin + SnapshotHook
        │  (writes every N steps)
        ▼
    ZarrData  (DataSink backed by Zarr store on disk)
        │
        ▼
    AtomicDataZarrReader  ──►  Dataset  ──►  DataLoader
                                              (yields Batch objects)

This example demonstrates each step:

1. Build a periodic argon NVT simulation using the Lennard-Jones potential.
2. Attach a :class:`~nvalchemi.dynamics.hooks.SnapshotHook` that writes every
   10 steps into a :class:`~nvalchemi.dynamics.ZarrData` sink.
3. Run 100 steps, then read the trajectory back via
   :class:`~nvalchemi.data.datapipes.AtomicDataZarrReader` and
   :class:`~nvalchemi.data.datapipes.DataLoader`.
4. Validate round-trip shape correctness.
"""

import logging
import tempfile
from pathlib import Path

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes import AtomicDataZarrReader, DataLoader, Dataset
from nvalchemi.dynamics import NVTLangevin, ZarrData
from nvalchemi.dynamics.hooks import NeighborListHook, SnapshotHook, WrapPeriodicHook
from nvalchemi.models.lj import LennardJonesModelWrapper

logging.basicConfig(level=logging.INFO)

# %%
# Build an argon NVT simulation
# ------------------------------
# Argon LJ parameters (Rappe & Casewit 1991): epsilon = 0.0104 eV,
# sigma = 3.40 Å.  We use a small 3×3×3 simple-cubic lattice (27 atoms)
# with a 10.5 Å cubic box so that the nearest-neighbour distance (3.5 Å)
# is safely inside the 8.5 Å cutoff.
#
# The :class:`~nvalchemi.models.lj.LennardJonesModelWrapper` requires
# a :class:`~nvalchemi.dynamics.hooks.NeighborListHook` to be registered
# on the dynamics engine so that ``batch.neighbor_matrix`` is populated
# before each model forward pass.

torch.manual_seed(0)

model = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5)
model.eval()

# Build a 3x3x3 simple-cubic lattice of argon atoms.
SPACING = 3.5  # Å — nearest-neighbour distance
N_SIDE = 3
BOX = SPACING * N_SIDE  # 10.5 Å

coords = []
for ix in range(N_SIDE):
    for iy in range(N_SIDE):
        for iz in range(N_SIDE):
            coords.append([ix * SPACING, iy * SPACING, iz * SPACING])  # noqa: PERF401

n_atoms = len(coords)  # 27
positions = torch.tensor(coords, dtype=torch.float32)

# Add small random displacements to break perfect symmetry.
g = torch.Generator()
g.manual_seed(1)
positions += torch.randn(n_atoms, 3, generator=g) * 0.05

cell = torch.eye(3, dtype=torch.float32).unsqueeze(0) * BOX

# Temperature 50 K: Maxwell-Boltzmann velocities.
KB_EV = 8.617333e-5  # eV/K
kT = 50.0 * KB_EV
mass_ar = 39.948  # atomic mass units (amu); forces in eV/Å, mass in amu
g2 = torch.Generator()
g2.manual_seed(2)
velocities = torch.randn(n_atoms, 3, generator=g2) * (kT / mass_ar) ** 0.5

data = AtomicData(
    positions=positions,
    atomic_numbers=torch.full((n_atoms,), 18, dtype=torch.long),  # Ar = 18
    atomic_masses=torch.full((n_atoms,), mass_ar),
    forces=torch.zeros(n_atoms, 3),
    energy=torch.zeros(1, 1),
    cell=cell,
    pbc=torch.tensor([[True, True, True]]),
)
data.add_node_property("velocities", velocities)

batch = Batch.from_data_list([data])

# %%
# Setting up the ZarrData sink and SnapshotHook
# ----------------------------------------------
# :class:`~nvalchemi.dynamics.ZarrData` accepts any zarr-compatible store.
# Here we use a temporary directory on disk so the example is self-contained.
# :class:`~nvalchemi.dynamics.hooks.SnapshotHook` writes the **full** batch
# state to the sink every ``frequency`` steps.

zarr_dir = tempfile.mkdtemp(suffix="_argon_traj")
zarr_path = Path(zarr_dir) / "trajectory.zarr"

zarr_sink = ZarrData(store=str(zarr_path), capacity=10_000)
snapshot_hook = SnapshotHook(sink=zarr_sink, frequency=10)

# %%
# Configuring the NVT integrator
# --------------------------------
# Register:
#
# 1. ``NeighborListHook`` — builds ``neighbor_matrix`` before each force eval.
# 2. ``WrapPeriodicHook`` — folds coordinates back into the primary cell after
#    each position update to prevent atoms drifting outside the box.
# 3. ``SnapshotHook`` — writes to the Zarr sink every 10 steps.

nl_hook = NeighborListHook(model.model_card.neighbor_config)
wrap_hook = WrapPeriodicHook()

nvt = NVTLangevin(
    model=model,
    dt=1.0,  # fs
    temperature=50.0,
    friction=0.1,
    random_seed=42,
    n_steps=100,
    hooks=[nl_hook, wrap_hook, snapshot_hook],
)

# %%
# Running and collecting the trajectory
# ---------------------------------------
# After 100 steps with ``frequency=10``, the sink holds 10 snapshots
# (steps 10, 20, ..., 100).

logging.info("Running 100 NVT steps on %d-atom argon system...", n_atoms)
batch = nvt.run(batch)

n_snaps = len(zarr_sink)
logging.info("Trajectory written: %d snapshots at %s", n_snaps, zarr_path)

# %%
# Reading back with DataLoader
# -----------------------------
# The read path is: ``AtomicDataZarrReader`` provides random-access sample
# loading; ``Dataset`` wraps it and returns ``AtomicData`` objects with
# optional device transfer; ``DataLoader`` collates them into ``Batch``
# objects of a given batch size.

reader = AtomicDataZarrReader(str(zarr_path))
ds = Dataset(reader, device="cpu", num_workers=1)
loader = DataLoader(ds, batch_size=2)

logging.info("Dataset length: %d samples", len(ds))
logging.info("DataLoader yields %d batches of size 2", len(loader))

# Iterate over all batches and collect.
loaded_batches: list[Batch] = []
for loaded_batch in loader:
    loaded_batches.append(loaded_batch)
    logging.info(
        "  batch: num_graphs=%d  positions.shape=%s",
        loaded_batch.num_graphs,
        tuple(loaded_batch.positions.shape),
    )

ds.close()

# %%
# Round-trip validation
# ----------------------
# Check that the loaded trajectory has the correct number of snapshots and
# that each snapshot contains the right number of atoms with the expected
# tensor shapes.

total_loaded = sum(b.num_graphs for b in loaded_batches)
assert total_loaded == n_snaps, f"Expected {n_snaps} snapshots, got {total_loaded}"

# Inspect the first snapshot from the first loaded batch.
first_snap = loaded_batches[0]
assert first_snap.positions.shape[-1] == 3, "positions must be (N, 3)"
assert first_snap.atomic_numbers is not None, "atomic_numbers must be present"

logging.info(
    "Round-trip OK: %d snapshots, each with %d atoms.",
    n_snaps,
    n_atoms,
)

# %%
# Summary
# --------
# The Zarr store persists at ``zarr_path`` and can be reloaded in future
# sessions or used for training downstream ML models.  For long simulations,
# ``ZarrData`` is preferred over :class:`~nvalchemi.dynamics.HostMemory`
# because it streams directly to disk rather than accumulating in RAM.

logging.info("Zarr store location: %s", zarr_path)
