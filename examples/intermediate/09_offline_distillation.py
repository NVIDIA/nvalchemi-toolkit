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
Offline Knowledge Distillation
==============================

This example distills a small student potential from a frozen foundation
teacher, offline: the teacher scores the dataset once,
:func:`~nvalchemi.training.distillation.label_dataset` writes its signals into a
Zarr store as ordinary ``teacher_*`` fields, and training then streams that
store through the normal reader/dataset/loader path with no teacher forward
passes at all.

The objective composes three terms — total energy, forces, and the teacher's
per-atom energy decomposition — which is where distillation differs from
supervised training: the first two are built-in loss terms with their
``target_key`` pointed at a teacher field, while
:class:`~nvalchemi.training.distillation.PerAtomEnergyMatchingLoss` matches a
quantity no reference dataset carries.

The teacher here is a direct-force model: it predicts forces from their own
head rather than as the negative gradient of its energy. Distillation treats
that as first class, because every teacher signal is detached before the
student ever sees it.

Everything runs on CPU in a few seconds with a fixed seed, so the numbers below
are reproducible.
"""

from __future__ import annotations

import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes import (
    AtomicDataZarrReader,
    DataLoader,
    Dataset,
    InMemoryDataset,
)
from nvalchemi.hooks import TrainContext
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStage,
)
from nvalchemi.training.distillation import (
    DistillationStrategy,
    InProcessTeacherScorer,
    PerAtomEnergyMatchingLoss,
    label_dataset,
)

# %%
# Configure a small gallery run
# -----------------------------
# Sphinx-gallery examples should execute quickly and deterministically, so the
# configuration is a handful of constants rather than command-line arguments.
# Scaling this to a real workflow means growing the dataset and step count and
# swapping the toy potential below for a wrapped MLIP.

NUM_SYSTEMS = 48
NUM_ATOMS = 6
HIDDEN_DIM = 16
BATCH_SIZE = 4
NUM_STEPS = 60
LEARNING_RATE = 5.0e-3
TEACHER_SEED = 11
STUDENT_SEED = 22
DATA_SEED = 33
SIGNALS = ["energy", "forces", "node_energies"]


# %%
# A potential with a per-atom energy head
# ---------------------------------------
# Both the teacher and the student are instances of this toy potential, seeded
# differently. What matters for distillation is its declared contract:
# ``outputs`` advertises ``atomic_energies`` alongside ``energy`` and
# ``forces``, which is what lets the teacher serve the ``node_energies`` signal
# and the student produce the matching ``predicted_atomic_energies``. An empty
# ``autograd_outputs`` marks it as a direct-force model.


class PerAtomPotential(torch.nn.Module, BaseModelMixin):
    """Toy potential with independent per-atom energy and force heads."""

    def __init__(self, *, hidden_dim: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.embedding = torch.nn.Embedding(16, hidden_dim)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + 3, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
        )
        self.energy_head = torch.nn.Linear(hidden_dim, 1)
        self.force_head = torch.nn.Linear(hidden_dim, 3)
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "atomic_energies"}),
            autograd_outputs=frozenset(),
            autograd_inputs=frozenset(),
            neighbor_config=None,
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no named embeddings for this toy potential."""
        return {}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        """Return ``data`` unchanged because the toy potential has no embeddings."""
        return data

    def forward(self, data: AtomicData | Batch, **kwargs: Any) -> OrderedDict:
        """Predict total energy, per-atom energies, and direct forces."""
        features = self.trunk(
            torch.cat([self.embedding(data.atomic_numbers), data.positions], dim=-1)
        )
        atomic_energies = self.energy_head(features)
        batch_idx = data.batch_idx if isinstance(data, Batch) else None
        if batch_idx is None:
            energy = atomic_energies.sum(dim=0, keepdim=True)
        else:
            energy = torch.zeros(
                (data.num_graphs, 1),
                dtype=atomic_energies.dtype,
                device=atomic_energies.device,
            ).scatter_add_(0, batch_idx.unsqueeze(-1), atomic_energies)
        return self.adapt_output(
            {
                "energy": energy,
                "forces": self.force_head(features),
                "atomic_energies": atomic_energies.squeeze(-1),
            },
            data,
        )


# %%
# Build a handful of systems
# --------------------------
# The source dataset carries no reference labels at all: in pure distillation
# the teacher is the label source, so positions and atomic numbers are enough.


def build_dataset(num_systems: int, num_atoms: int, seed: int) -> InMemoryDataset:
    """Return an in-memory dataset of deterministic random systems."""
    generator = torch.Generator().manual_seed(seed)
    systems = [
        AtomicData(
            positions=torch.randn(num_atoms, 3, generator=generator),
            atomic_numbers=torch.randint(
                1, 9, (num_atoms,), dtype=torch.long, generator=generator
            ),
        )
        for _ in range(num_systems)
    ]
    return InMemoryDataset(in_memory_batch=Batch.from_data_list(systems))


source_dataset = build_dataset(NUM_SYSTEMS, NUM_ATOMS, DATA_SEED)
print(f"Source dataset: {len(source_dataset)} systems, {NUM_ATOMS} atoms each")

# %%
# Label the dataset with the frozen teacher
# -----------------------------------------
# :class:`~nvalchemi.training.distillation.InProcessTeacherScorer` narrows the
# teacher to exactly the outputs the requested signals need, detaches
# everything it returns, and leaves each scored batch as it found it.
# :func:`~nvalchemi.training.distillation.label_dataset` walks the dataset once
# and writes the source fields plus the teacher fields into a Zarr store; the
# run is resumable, so a long labeling job can be interrupted and continued.

teacher = PerAtomPotential(hidden_dim=HIDDEN_DIM, seed=TEACHER_SEED)
scorer = InProcessTeacherScorer(teacher, SIGNALS)

store = Path(tempfile.mkdtemp(suffix="_distillation")) / "labeled.zarr"
num_labeled = label_dataset(source_dataset, scorer, store, batch_size=8)
print(f"Labeled {num_labeled} systems into {store.name}")

reader = AtomicDataZarrReader(store)
print("Stored fields:", ", ".join(sorted(reader.field_levels)))

# %%
# Stream the labeled store
# ------------------------
# Nothing about the consumption path is distillation-specific: the teacher
# fields arrive as ordinary batch attributes at the levels they were written
# at, so the reader, dataset, and loader are the ones any training run uses.

labeled_dataset = Dataset(reader=reader, device="cpu")
loader = DataLoader(labeled_dataset, batch_size=BATCH_SIZE, use_streams=False)

# %%
# Compose the distillation objective
# ----------------------------------
# Each term names the teacher field it reads. The per-atom term is weighted
# below the extensive quantities on purpose: matching the teacher's internal
# energy decomposition is a regularizer, while the total energy and the forces
# are what the student is ultimately judged on. Composed weights are ratios
# rather than coefficients — ``ComposedLossFunction`` renormalizes them to sum
# to one, so the three terms below run at 1/2.2, 1/2.2, and 0.2/2.2. Build the
# composition explicitly with ``normalize_weights=False`` to keep literal
# weights.

loss_fn = (
    EnergyMSELoss(target_key="teacher_energy")
    + ForceMSELoss(target_key="teacher_forces", normalize_by_atom_count=True)
    + 0.2 * PerAtomEnergyMatchingLoss()
)

# %%
# Distill the student
# -------------------
# The teacher is passed to the strategy but left out of ``optimizer_configs``,
# which freezes it for the whole run. Because the batches are pre-labeled, the
# teacher is never called during training; it is still validated at
# construction, so a signal it cannot produce fails immediately instead of on
# the first batch. The hook below records the loss the strategy backpropagates
# so the run can be summarized at the end.


class LossTrace:
    """Collect the total loss of every completed training batch."""

    frequency = 1
    stage = TrainingStage.AFTER_BATCH

    def __init__(self) -> None:
        self.losses: list[float] = []

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:
        """Append the loss of the batch that just finished."""
        self.losses.append(float(ctx.loss))


trace = LossTrace()
strategy = DistillationStrategy(
    models={
        "student": PerAtomPotential(hidden_dim=HIDDEN_DIM, seed=STUDENT_SEED),
        "teacher": teacher,
    },
    optimizer_configs={
        "student": [
            OptimizerConfig(
                optimizer_cls=torch.optim.Adam,
                optimizer_kwargs={"lr": LEARNING_RATE},
            )
        ]
    },
    loss_fn=loss_fn,
    num_steps=NUM_STEPS,
    hooks=[trace],
)
print("Teacher signals:", ", ".join(sorted(strategy.teacher_scorer.signals)))

strategy.run(loader)

# %%
# Read the result
# ---------------
# Averaging the first and last passes over the loader smooths out per-batch
# variation and shows the trend the run is judged on.

batches_per_epoch = len(loader)
first = sum(trace.losses[:batches_per_epoch]) / batches_per_epoch
last = sum(trace.losses[-batches_per_epoch:]) / batches_per_epoch
print(f"Completed {strategy.step_count} steps over {strategy.epoch_count} epochs")
print(f"Mean loss, first pass: {first:.4f}")
print(f"Mean loss, last pass:  {last:.4f}")
print(f"Reduction: {100.0 * (1.0 - last / first):.1f}%")
