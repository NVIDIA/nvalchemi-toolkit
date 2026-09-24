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
L-BFGS in a FusedStage Pipeline
===============================

:class:`~nvalchemi.dynamics.optimizers.LBFGS` composes into a
:class:`~nvalchemi.dynamics.FusedStage` exactly like FIRE2: relax each system
with L-BFGS, then hand it to MD once it converges.  All stages share one
batch and one model evaluation per step.

* **Part 1** — ``LBFGS + NVTLangevin`` on a fixed batch.
* **Part 2** — the same pipeline with inflight batching: finished systems
  graduate and new ones are admitted mid-run.

L-BFGS keeps a per-atom curvature history.  The pipeline carries it through
both cases: systems owned by another stage leave it untouched, survivors keep
it across refills, and newly admitted systems start fresh.
"""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import (
    LBFGS,
    ConvergenceHook,
    FusedStage,
    NVTLangevin,
    SizeAwareSampler,
)
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.models.lj import LennardJonesModelWrapper

device = "cuda" if torch.cuda.is_available() else "cpu"

# %%
# Model and systems
# -----------------
# Lennard-Jones argon clusters of different sizes, perturbed off-lattice.

model = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5).to(device)
R_MIN = 2 ** (1 / 6) * 3.40  # ≈ 3.82 Å


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
        velocities=torch.zeros(n, 3),
    )


def attach_neighbor_lists(fused: FusedStage) -> FusedStage:
    """Register the model's neighbor-list hooks on the fused stage."""
    for hook in model.make_neighbor_hooks():
        fused.register_hook(hook)
    return fused


# %%
# Part 1 — LBFGS → NVT
# --------------------
# ``+`` builds a :class:`FusedStage`.  A system moves from L-BFGS to NVT when
# its ``convergence_hook`` fires and graduates after ``n_steps`` of NVT.

relax = LBFGS(
    model=model, maxstep=0.2, convergence_hook=ConvergenceHook.from_fmax(1e-3)
)
nvt = NVTLangevin(
    model=model, dt=1.0, temperature=30.0, friction=0.01, random_seed=0, n_steps=50
)
pipeline = attach_neighbor_lists(relax + nvt)

batch = Batch.from_data_list(
    [make_cluster(2, 0), make_cluster(3, 1), make_cluster(3, 2)]
).to(device)
batch = pipeline.run(batch, n_steps=400)
# Each system hands off to NVT as soon as it converges; L-BFGS state is
# per system, so ``iteration`` counts that system's own L-BFGS steps.
print(f"Part 1: pipeline finished after {pipeline.step_count} steps")
print(f"        L-BFGS steps per system: {relax._state.iteration.tolist()}")

# %%
# Part 2 — Inflight batching
# --------------------------
# A :class:`~nvalchemi.dynamics.SizeAwareSampler` keeps the live batch under
# an atom budget.  As systems finish NVT they are written to the sink and
# replaced from the dataset; L-BFGS state is resized to match automatically.


class ClusterDataset:
    """Argon clusters of 8, 27 or 64 atoms."""

    def __init__(self, n_samples: int) -> None:
        self.sizes = [2, 3, 4] * (n_samples // 3) + [2] * (n_samples % 3)

    def __len__(self) -> int:
        return len(self.sizes)

    def get_metadata(self, idx: int) -> tuple[int, int]:
        """Return ``(num_atoms, num_edges)`` without building the sample."""
        return self.sizes[idx] ** 3, 0

    def __getitem__(self, idx: int) -> tuple[AtomicData, dict]:
        return make_cluster(self.sizes[idx], seed=100 + idx), {}


dataset = ClusterDataset(n_samples=9)
sampler = SizeAwareSampler(dataset, max_atoms=128, max_edges=None, max_batch_size=4)
sink = HostMemory(capacity=len(dataset))

inflight = attach_neighbor_lists(
    FusedStage(
        sub_stages=[
            (
                0,
                LBFGS(
                    model=model,
                    convergence_hook=ConvergenceHook.from_fmax(1e-3),
                ),
            ),
            (
                1,
                NVTLangevin(
                    model=model,
                    dt=1.0,
                    temperature=30.0,
                    friction=0.01,
                    random_seed=1,
                    n_steps=20,
                ),
            ),
        ],
        sampler=sampler,
        sinks=[sink],
        refill_frequency=5,
    )
)
inflight.run(batch=None, n_steps=2000)
print(f"Part 2: {len(sink)} of {len(dataset)} systems relaxed and equilibrated")
