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
On-Policy Knowledge Distillation
================================

Offline distillation (see :doc:`09_offline_distillation`) trains a student on
whatever structures the dataset happens to hold. On-policy distillation trains
it on the structures the student itself visits. The student's own propagator
generates frames, the frozen teacher labels them, and the labeled frames
accumulate in a *replay buffer*. Every training batch is a *mixture* of
replay-buffer frames and samples from a *reference dataset*, a fixed
teacher-labeled dataset.

Setting the strategy's ``on_policy`` field turns
:meth:`~nvalchemi.training.distillation.DistillationStrategy.run` into a loop
of *segments*. Each segment generates frames, labels them, and trains on the
mixture. The caller passes no dataloader, because each segment builds its own.

``replay_ratio`` is the central setting. It is the fraction of every training
batch drawn from the replay buffer: ``1.0`` trains on generated frames alone,
while a lower value keeps part of each batch on the reference dataset. The
reference dataset has to be teacher-labeled and carry the same fields a
generated frame does, which is what
:func:`~nvalchemi.training.distillation.label_dataset` produces below.

The teacher in this example predicts forces from its own head rather than as
the negative gradient of its energy, while the student's forces *are* that
gradient. Distilling a non-conservative teacher into a conservative student is
a supported path. Every teacher signal is detached before the student sees
it, so how a force was produced never reaches the objective.

Everything runs in a few seconds with fixed seeds. ``DEVICE`` below picks the
device; on CPU the printed numbers are exactly reproducible.
"""

from __future__ import annotations

import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset, InMemoryDataset
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.hooks import TrainContext
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.models.demo import DemoModel, DemoModelWrapper
from nvalchemi.training import (
    EnergyMSELoss,
    ForceMSELoss,
    OptimizerConfig,
    TrainingStage,
)
from nvalchemi.training.distillation import (
    DistillationStrategy,
    InitialStructures,
    InProcessTeacherScorer,
    OnPolicyConfig,
    label_dataset,
)

# %%
# Configure the run
# -----------------
# The run is three segments long. Each segment propagates ``GENERATION_STEPS``
# steps and then takes ``TRAINING_STEPS_PER_SEGMENT`` optimizer steps on the
# mixture, until ``NUM_STEPS`` is reached. ``DEVICE`` selects where everything
# runs; set it to ``torch.device("cuda")`` for a GPU.

DEVICE = torch.device("cpu")
NUM_INITIAL = 4
NUM_REFERENCE = 8
NUM_ATOMS = 4
HIDDEN_DIM = 8
NUM_STEPS = 12
TRAINING_STEPS_PER_SEGMENT = 4
GENERATION_STEPS = 5
LABEL_FREQUENCY = 1
BATCH_SIZE = 4
REPLAY_RATIO = 0.5
REPLAY_CAPACITY = 256
LEARNING_RATE = 1.0e-2
TEACHER_SEED = 11
STUDENT_SEED = 22
INITIAL_ELEMENT = 1
REFERENCE_ELEMENT = 6
SIGNALS = ["energy", "forces"]


# %%
# A direct-force teacher
# ----------------------
# The teacher's ``model_config`` lists ``energy`` and ``forces`` in
# ``outputs``. Its empty ``autograd_outputs`` says the forces come from a head,
# not from an energy gradient.
# :class:`~nvalchemi.training.distillation.InProcessTeacherScorer` reads
# ``autograd_outputs`` to decide whether scoring runs under
# ``torch.enable_grad()`` or ``torch.no_grad()``. An empty set is right for
# this teacher. A conservative teacher has to declare ``forces`` there, or its
# gradient is never computed. The distillation pipeline does not restrict or
# gate on force conservativeness: every teacher signal is detached, so the
# teacher stays out of the student's autograd graph either way.


class DirectForceTeacher(torch.nn.Module, BaseModelMixin):
    """Toy potential whose forces come from a head, not from an energy gradient."""

    def __init__(self, *, hidden_dim: int, seed: int) -> None:
        super().__init__()
        torch.manual_seed(seed)
        # Checkpoint spec generation reads constructor arguments off same-named
        # attributes, so a teacher that drops them cannot be checkpointed.
        self.hidden_dim = hidden_dim
        self.seed = seed
        self.embedding = torch.nn.Embedding(16, hidden_dim)
        self.trunk = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim + 3, hidden_dim),
            torch.nn.SiLU(),
        )
        self.energy_head = torch.nn.Linear(hidden_dim, 1)
        self.force_head = torch.nn.Linear(hidden_dim, 3)
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
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
        """Predict a total energy and per-atom forces in one pass."""
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
            {"energy": energy, "forces": self.force_head(features)}, data
        )


teacher = DirectForceTeacher(hidden_dim=HIDDEN_DIM, seed=TEACHER_SEED).to(DEVICE)
torch.manual_seed(STUDENT_SEED)
student = DemoModelWrapper(DemoModel(num_atom_types=16, hidden_dim=HIDDEN_DIM)).to(
    DEVICE
)
print("Teacher autograd outputs:", sorted(teacher.model_config.autograd_outputs))
print("Student autograd outputs:", sorted(student.model_config.autograd_outputs))

# %%
# Initial structures and a teacher-labeled reference dataset
# ----------------------------------------------------------
# Build two sets of structures, tagged by atomic number so the mixture is easy
# to read later. The *initial structures* are the bare geometries the
# trajectories start from; the source they go behind is covered under
# :ref:`initial structures <distillation-initial-structures>`. The reference
# dataset is labeled by the same teacher, which is what makes it admissible and
# keeps the dtypes in step; see :ref:`distillation-reference-dataset`.


def build_systems(element: int, num_systems: int, seed: int) -> Batch:
    """Return deterministic random systems tagged by *element*, geometry only."""
    generator = torch.Generator().manual_seed(seed)
    return Batch.from_data_list(
        [
            AtomicData(
                positions=torch.randn(NUM_ATOMS, 3, generator=generator),
                atomic_numbers=torch.full((NUM_ATOMS,), element, dtype=torch.long),
                atomic_masses=torch.ones(NUM_ATOMS),
            )
            for _ in range(num_systems)
        ]
    )


initial_structures = InitialStructures(
    InMemoryDataset(
        in_memory_batch=build_systems(INITIAL_ELEMENT, NUM_INITIAL, 500).to(DEVICE)
    )
)
scorer = InProcessTeacherScorer(teacher, SIGNALS)

store = Path(tempfile.mkdtemp(suffix="_on_policy")) / "reference.zarr"
label_dataset(
    InMemoryDataset(
        in_memory_batch=build_systems(REFERENCE_ELEMENT, NUM_REFERENCE, 700)
    ),
    scorer,
    store,
    batch_size=4,
    device=DEVICE,
)
reference_dataset = Dataset(reader=AtomicDataZarrReader(store), device=DEVICE)
print(
    f"Initial structures: {len(initial_structures)}, "
    f"reference structures: {len(reference_dataset)}"
)
print("Reference fields:", ", ".join(sorted(reference_dataset.field_names)))

# %%
# Configure the segment loop
# --------------------------
# The propagator holds the very module the optimizer updates, which is what
# makes the generated data on-policy; the strategy checks that identity at
# construction. ``label_frequency`` is the throughput setting, since the teacher
# is the expensive model; see :ref:`the cadence <distillation-cadence>`.

on_policy = OnPolicyConfig(
    dynamics=NVTLangevin(
        student, dt=0.5, temperature=300.0, friction=0.01, random_seed=7
    ),
    teacher_scorer=scorer,
    initial_structures=initial_structures,
    replay_ratio=REPLAY_RATIO,
    training_steps_per_segment=TRAINING_STEPS_PER_SEGMENT,
    batch_size=BATCH_SIZE,
    generation_steps=GENERATION_STEPS,
    label_frequency=LABEL_FREQUENCY,
    replay_capacity=REPLAY_CAPACITY,
)

# %%
# Run the loop
# ------------
# The objective reads teacher fields only, because that is all a generated
# frame carries. The run is sized in optimizer steps rather than epochs: every
# segment builds its own loader, so there is no fixed epoch to convert.

loss_fn = EnergyMSELoss(target_key="teacher_energy") + ForceMSELoss(
    target_key="teacher_forces", normalize_by_atom_count=True
)


class MixtureTrace:
    """Record the loss and the source composition of every training batch."""

    frequency = 1
    stage = TrainingStage.AFTER_BATCH

    def __init__(self) -> None:
        self.losses: list[float] = []
        self.compositions: list[tuple[int, int]] = []

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:
        """Append the loss and the (generated, reference) counts of this batch."""
        self.losses.append(float(ctx.loss))
        tags = [
            int(ctx.batch.atomic_numbers[ctx.batch.batch_idx == index][0])
            for index in range(ctx.batch.num_graphs)
        ]
        self.compositions.append(
            (tags.count(INITIAL_ELEMENT), tags.count(REFERENCE_ELEMENT))
        )


trace = MixtureTrace()
strategy = DistillationStrategy(
    models={"student": student, "teacher": teacher},
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
    on_policy=on_policy,
    reference_dataset=reference_dataset,
    devices=[DEVICE],
    hooks=[trace],
)
print("Teacher signals:", ", ".join(sorted(strategy.teacher_scorer.signals)))

strategy.run()

# %%
# Inspect the replay buffer
# -------------------------
# The buffer outlives the run: a second ``run()`` with a raised ``NUM_STEPS``
# appends to these frames. Its schema is the contract the reference dataset had
# to meet: the structure plus the ``teacher_*`` labels, never the propagator's
# own ``energy`` and ``forces``. See :ref:`distillation-reference-dataset`.

buffer = strategy.replay_buffer
print(f"Replay buffer: {len(buffer)} frames")
print("Buffer schema:", ", ".join(sorted(buffer.schema)))

# %%
# Read the result
# ---------------
# Every batch has exactly the composition ``replay_ratio`` sets, not just on
# average. The loss is compared segment to segment, because each segment
# trains on a buffer the previous one grew.

first = sum(trace.losses[:TRAINING_STEPS_PER_SEGMENT]) / TRAINING_STEPS_PER_SEGMENT
last = sum(trace.losses[-TRAINING_STEPS_PER_SEGMENT:]) / TRAINING_STEPS_PER_SEGMENT
print("Batch compositions (generated, reference):", sorted(set(trace.compositions)))
print(f"Completed {strategy.step_count} steps over {strategy.epoch_count} segments")
print(f"Mean loss, first segment: {first:.4f}")
print(f"Mean loss, last segment:  {last:.4f}")
print(f"Reduction: {100.0 * (1.0 - last / first):.1f}%")
