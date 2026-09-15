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
it on the structures the student itself visits: the student's own propagator
generates frames, the frozen teacher labels them, they accumulate in a replay
buffer, and every training batch mixes that buffer with a teacher-labeled
anchor dataset.

The loop is one field on the strategy. Setting ``on_policy`` turns
:meth:`~nvalchemi.training.distillation.DistillationStrategy.run` into a
sequence of generate-label-train segments that need no dataloader from the
caller, because each segment builds its own.

``replay_ratio`` is the knob the run turns on. It is the fraction of every
training batch drawn from generated frames, so ``1.0`` trains on generated data
alone while a lower value keeps each batch anchored in a fixed dataset. The
anchor has to be teacher-labeled and carry the same fields a generated frame
does, which is what :func:`~nvalchemi.training.distillation.label_dataset`
produces below.

The teacher here predicts forces from their own head rather than as the
negative gradient of its energy, and the student's forces *are* that gradient.
Distilling a non-conservative teacher into a conservative student is a
supported path: every teacher signal is detached before the student sees it, so
how a force was produced never reaches the objective.

Everything runs on CPU in a few seconds with fixed seeds, so the numbers below
are reproducible.
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
    InProcessTeacherScorer,
    OnPolicyConfig,
    SeedSource,
    label_dataset,
)

# %%
# Configure a small gallery run
# -----------------------------
# The run is three segments long: each one propagates ``SEGMENT_STEPS`` steps
# and then spends ``STEPS_PER_SEGMENT`` optimizer steps on the mixture, until
# ``NUM_STEPS`` is reached. Scaling this to a real workflow means growing every
# count and swapping the demo potentials for wrapped MLIPs.

NUM_SEEDS = 4
NUM_ANCHORS = 8
NUM_ATOMS = 4
HIDDEN_DIM = 8
NUM_STEPS = 12
STEPS_PER_SEGMENT = 4
SEGMENT_STEPS = 5
LABEL_FREQUENCY = 1
BATCH_SIZE = 4
REPLAY_RATIO = 0.5
REPLAY_CAPACITY = 256
LEARNING_RATE = 1.0e-2
TEACHER_SEED = 11
STUDENT_SEED = 22
SEED_ELEMENT = 1
ANCHOR_ELEMENT = 6
SIGNALS = ["energy", "forces"]


# %%
# A direct-force teacher
# ----------------------
# The teacher's declared contract is what matters here: ``outputs`` advertises
# ``energy`` and ``forces``, and an empty ``autograd_outputs`` marks the forces
# as a head output rather than a gradient. That flag is load-bearing:
# :class:`~nvalchemi.training.distillation.InProcessTeacherScorer` reads it to
# decide whether the labeling forward pass runs under ``enable_grad`` or
# ``no_grad``, so an empty set is right for this teacher and a conservative one
# has to declare ``forces`` there or its gradient is never computed. What
# nothing in the distillation path does is *gate* on conservativeness: every
# teacher signal is detached, so the teacher stays out of the student's autograd
# graph either way.


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


teacher = DirectForceTeacher(hidden_dim=HIDDEN_DIM, seed=TEACHER_SEED)
torch.manual_seed(STUDENT_SEED)
student = DemoModelWrapper(DemoModel(num_atom_types=16, hidden_dim=HIDDEN_DIM))
print("Teacher autograd outputs:", sorted(teacher.model_config.autograd_outputs))
print("Student autograd outputs:", sorted(student.model_config.autograd_outputs))

# %%
# Seed structures and a teacher-labeled anchor
# --------------------------------------------
# Two sets of structures, tagged by atomic number so the mixture is legible
# later. The seeds are what the trajectories start from — they carry the
# ``energy`` and ``forces`` the integrator reads before it has computed any —
# and the anchor is the fixed dataset every batch is partly drawn from.
#
# The seed dataset goes behind a
# :class:`~nvalchemi.training.distillation.SeedSource`, the cursor the initial
# batch and any later backfill both draw from. Left unbudgeted, as here, it
# seeds every row it owns as one batch, so ``NUM_SEEDS`` *is* the number of
# trajectories this run generates from. The source also stamps the batch's
# ``status`` and ``system_id``, and one row is loaded when ``OnPolicyConfig`` is
# built to check the seeds against what the propagator reads — a missing
# ``forces`` is a construction error rather than an ``AttributeError`` on the
# first step.
#
# The anchor is labeled with the same teacher, through
# :func:`~nvalchemi.training.distillation.label_dataset`, and written to a Zarr
# store. That is a requirement rather than a convenience: a mixed batch keeps
# only the fields both sources hold. Both halves of it are checked at
# construction — an anchor carrying no ``teacher_*`` labels is rejected, and so
# is one carrying them *alongside* reference ``energy`` and ``forces``, which is
# the shape labeling an existing reference set leaves behind. That is why the
# anchor structures below are built with ``predictions=False``. Only the full
# field, level, and dtype comparison against real frames waits for the first
# segment's mixed loader; labeling the anchor with the very scorer that drives
# generation is what keeps the dtypes in step. The store is opened on the device
# this run trains on, because both mixture sources are collated into one batch
# before the strategy moves it.


def build_systems(
    element: int, num_systems: int, seed: int, *, predictions: bool = False
) -> Batch:
    """Return deterministic random systems tagged by *element*.

    ``predictions=True`` adds the ``energy`` and ``forces`` an integrator reads
    on its first step, before it has computed any. Anchor structures leave them
    out, because that is the shape a stored frame has. A propagator that opens
    on more than forces — NPT, NPH, or a variable-cell optimizer — needs its
    seeds zero-filled with ``stress`` as well.
    """
    generator = torch.Generator().manual_seed(seed)
    predicted = (
        {"energy": torch.zeros(1, 1), "forces": torch.zeros(NUM_ATOMS, 3)}
        if predictions
        else {}
    )
    return Batch.from_data_list(
        [
            AtomicData(
                positions=torch.randn(NUM_ATOMS, 3, generator=generator),
                atomic_numbers=torch.full((NUM_ATOMS,), element, dtype=torch.long),
                atomic_masses=torch.ones(NUM_ATOMS),
                **predicted,
            )
            for _ in range(num_systems)
        ]
    )


seeds = SeedSource(
    InMemoryDataset(
        in_memory_batch=build_systems(SEED_ELEMENT, NUM_SEEDS, 500, predictions=True)
    )
)
scorer = InProcessTeacherScorer(teacher, SIGNALS)

store = Path(tempfile.mkdtemp(suffix="_on_policy")) / "anchor.zarr"
label_dataset(
    InMemoryDataset(in_memory_batch=build_systems(ANCHOR_ELEMENT, NUM_ANCHORS, 700)),
    scorer,
    store,
    batch_size=4,
)
reference_dataset = Dataset(reader=AtomicDataZarrReader(store), device="cpu")
print(f"Seed structures: {len(seeds)}, anchor structures: {len(reference_dataset)}")
print("Anchor fields:", ", ".join(sorted(reference_dataset.field_names)))

# %%
# Configure the segment loop
# --------------------------
# The propagator holds the very module the optimizer updates, which is what
# makes the generated data on-policy: each segment propagates the weights the
# previous one trained, and the strategy checks that identity at construction.
# ``label_frequency`` is the throughput knob — the teacher is the expensive
# model, and labeling every step is affordable only at this scale. At ``1``
# every frame is stored; above it, a segment stores its last frame per
# trajectory plus one every ``label_frequency`` steps, and the dispatch that
# would land right after a labeled frame is skipped so no segment boundary is
# paid for twice.

on_policy = OnPolicyConfig(
    dynamics=NVTLangevin(
        student, dt=0.5, temperature=300.0, friction=0.01, random_seed=7
    ),
    teacher_scorer=scorer,
    seeds=seeds,
    replay_ratio=REPLAY_RATIO,
    steps_per_segment=STEPS_PER_SEGMENT,
    batch_size=BATCH_SIZE,
    segment_steps=SEGMENT_STEPS,
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
        """Append the loss and the (generated, anchor) counts of this batch."""
        self.losses.append(float(ctx.loss))
        tags = [
            int(ctx.batch.atomic_numbers[ctx.batch.batch_idx == index][0])
            for index in range(ctx.batch.num_graphs)
        ]
        self.compositions.append((tags.count(SEED_ELEMENT), tags.count(ANCHOR_ELEMENT)))


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
    hooks=[trace],
)
print("Teacher signals:", ", ".join(sorted(strategy.teacher_scorer.signals)))

strategy.run()

# %%
# Inspect the replay buffer
# -------------------------
# The buffer stays reachable after the run, and it outlives it: a second
# ``run()`` on this strategy with a raised ``NUM_STEPS`` appends to these frames
# rather than regenerating them. Its schema is the contract the
# anchor had to meet: the structure, whatever travels with it, and the
# ``teacher_*`` labels — with none of the ``energy`` and ``forces`` the
# propagator wrote on the live frame, which the labeling hook strips on the way
# in so a stored frame never carries a self-label under a reference target's
# name.

buffer = strategy.replay_buffer
print(f"Replay buffer: {len(buffer)} frames")
print("Buffer schema:", ", ".join(sorted(buffer.schema)))

# %%
# Read the result
# ---------------
# The composition is exact per batch rather than an average, and the loss is
# compared segment to segment because each segment trains on a buffer the
# previous one grew.

first = sum(trace.losses[:STEPS_PER_SEGMENT]) / STEPS_PER_SEGMENT
last = sum(trace.losses[-STEPS_PER_SEGMENT:]) / STEPS_PER_SEGMENT
print("Batch compositions (generated, anchor):", sorted(set(trace.compositions)))
print(f"Completed {strategy.step_count} steps over {strategy.epoch_count} segments")
print(f"Mean loss, first segment: {first:.4f}")
print(f"Mean loss, last segment:  {last:.4f}")
print(f"Reduction: {100.0 * (1.0 - last / first):.1f}%")
