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
Processing Large Datasets with Inflight Batching
=================================================

**The problem**: a production relaxation campaign may require running
thousands of structures through FIRE → NVT stages.  Loading all of them into
GPU memory at once is impossible; processing one at a time wastes GPU
throughput.

**The solution**: *inflight batching* (Mode 2 in :class:`~nvalchemi.dynamics.FusedStage`).
A fixed-size *live batch* occupies the GPU at all times.  Whenever a system
graduates (converges or exhausts its step budget), it is evicted and a new
sample from the dataset takes its slot.  The
:class:`~nvalchemi.dynamics.SizeAwareSampler` handles bin-packing so the
replacement fits within the memory envelope of the slot it replaces.

Lifecycle of a system:

::

    Dataset → SizeAwareSampler.request_replacement()
        │
        ▼
    Live batch (GPU) — stage 0 (FIRE relaxation)
        │ converges
        ▼
    Live batch (GPU) — stage 1 (NVT equilibration)
        │ n_steps exhausted
        ▼
    ConvergedSnapshotHook → HostMemory sink
        │
        ▼
    Slot freed → next sample loaded from dataset

Key concept: **system_id**.  Each sample loaded from the dataset receives a
monotonically-increasing integer ``system_id`` stamped by the sampler as a
graph-level tensor.  This lets downstream code track individual trajectories
across refill events.

**Important caveat**: when the standard :class:`~nvalchemi.dynamics.sinks.HostMemory`
sink unbatches data via ``Batch.to_data_list()``, the reconstructed ``AtomicData``
objects lose the knowledge that ``system_id`` is a system-level property (because
``__system_keys__`` is reset to defaults during ``model_post_init``).  When the
sink later re-batches via ``Batch.from_data_list()``, ``system_id`` gets dropped.

This example defines a custom sink (``HostMemoryWithSystemId``) that re-registers
``system_id`` as a system property after unbatching, ensuring it survives the
round-trip through the sink.

This example uses :class:`~nvalchemi.models.demo.DemoModelWrapper` (a small
neural network) so no neighbor list is needed.  For a real LJ or MACE model,
add :class:`~nvalchemi.dynamics.hooks.NeighborListHook` to each sub-stage.
"""

import logging

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import FIRE, NVTLangevin, SizeAwareSampler
from nvalchemi.dynamics.base import ConvergenceHook, FusedStage
from nvalchemi.dynamics.hooks import ConvergedSnapshotHook
from nvalchemi.dynamics.sinks import HostMemory
from nvalchemi.models.demo import DemoModelWrapper


class HostMemoryWithSystemId(HostMemory):
    """HostMemory subclass that preserves ``system_id`` across unbatch/rebatch."""

    def write(self, batch: Batch, mask: torch.Tensor | None = None) -> None:
        """Store a batch, preserving ``system_id`` as a system property."""
        num_total = batch.num_graphs or 0
        if num_total == 0:
            return

        # Apply mask to select samples (same logic as HostMemory).
        if mask is not None:
            mask = mask.to(device=batch.device, dtype=torch.bool)
            if mask.shape[0] != num_total:
                raise ValueError(
                    f"mask length {mask.shape[0]} != num_graphs {num_total}"
                )
            num_selected = int(mask.sum().item())
            if num_selected == 0:
                return
            if num_selected < num_total:
                indices = torch.nonzero(mask, as_tuple=True)[0]
                _ = batch.batch_ptr  # trigger lazy init for SegmentedLevelStorage
                batch = batch.index_select(indices)

        # Extract system_id before unbatching; the AtomicData reconstruction loses
        # the __system_keys__ registration.
        system_ids = batch["system_id"].detach().to("cpu")

        data_list = batch.to_data_list()
        if len(self._data_list) + len(data_list) > self._capacity:
            raise RuntimeError(
                f"Buffer is full. Cannot add {len(data_list)} samples "
                f"to buffer with {len(self._data_list)}/{self._capacity} samples."
            )

        for i, data in enumerate(data_list):
            data_cpu = data.to(self._device)
            data_cpu.add_system_property("system_id", system_ids[i : i + 1])
            self._data_list.append(data_cpu)


logging.basicConfig(level=logging.INFO)

# %%
# The dataset interface
# ----------------------
# :class:`SizeAwareSampler` requires a dataset that implements exactly three
# methods:
#
# * ``__len__()`` — total number of samples.
# * ``__getitem__(idx) -> (AtomicData, dict)`` — load one sample by index.
#   The ``dict`` carries arbitrary per-sample metadata (empty is fine).
# * ``get_metadata(idx) -> (num_atoms, num_edges)`` — return atom/edge counts
#   **without** constructing the full ``AtomicData`` object.
#
# The ``get_metadata`` method exists for efficiency: the sampler pre-scans the
# entire dataset at construction time to build size-aware bins.  If loading
# each sample were necessary just for bin-packing, large datasets would be
# prohibitively slow to initialise.


class MixedSizeDataset:
    """A dataset of random molecular systems with varying atom counts.

    Systems have between ``min_atoms`` and ``max_atoms`` atoms, assigned
    by cycling through a range.  This produces a realistic distribution of
    sizes that exercises the bin-packing logic.

    Parameters
    ----------
    n_samples : int
        Total number of systems.
    min_atoms : int
        Minimum atom count.
    max_atoms : int
        Maximum atom count.
    seed : int
        Base RNG seed.
    """

    def __init__(
        self,
        n_samples: int,
        min_atoms: int = 4,
        max_atoms: int = 6,
        seed: int = 0,
    ) -> None:
        self.n_samples = n_samples
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.base_seed = seed
        # Pre-assign atom counts to each sample index.
        span = max_atoms - min_atoms + 1
        self._atom_counts = [min_atoms + (i % span) for i in range(n_samples)]

    def __len__(self) -> int:
        return self.n_samples

    def get_metadata(self, idx: int) -> tuple[int, int]:
        """Return ``(num_atoms, num_edges)`` without loading the sample.

        The sampler calls this for **every** sample at construction time.
        It must be cheap — no I/O, no tensor allocation.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple[int, int]
            ``(num_atoms, num_edges)``; num_edges=0 for models without edge
            lists.
        """
        return self._atom_counts[idx], 0

    def __getitem__(self, idx: int) -> tuple[AtomicData, dict]:
        """Load one sample.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple[AtomicData, dict]
            The AtomicData and an (empty) metadata dict.
        """
        n = self._atom_counts[idx]
        g = torch.Generator()
        g.manual_seed(self.base_seed + idx)
        data = AtomicData(
            positions=torch.randn(n, 3, generator=g),
            atomic_numbers=torch.randint(1, 10, (n,), dtype=torch.long, generator=g),
            atomic_masses=torch.ones(n),
            forces=torch.zeros(n, 3),
            energy=torch.zeros(1, 1),
        )
        data.add_node_property("velocities", torch.zeros(n, 3))
        return data, {}


# %%
# SizeAwareSampler
# -----------------
# The sampler consumes the dataset lazily.  It builds atom-count bins at
# construction (calling ``get_metadata`` on every sample) and then serves
# replacements via ``request_replacement(num_atoms, num_edges)``, which
# finds an unconsumed sample small enough to fit in the vacated slot.
#
# ``max_atoms=24`` allows at most 4–6 systems of 4–6 atoms each in the
# live batch.  ``max_batch_size=6`` caps the graph count independently.
# ``max_edges=None`` disables the edge constraint (DemoModelWrapper does
# not use a neighbor list).

dataset = MixedSizeDataset(n_samples=30, min_atoms=4, max_atoms=6, seed=200)

sampler = SizeAwareSampler(
    dataset,
    max_atoms=24,
    max_edges=None,
    max_batch_size=6,
)
logging.info(
    "Sampler created: %d samples, bins: %s",
    len(sampler),
    sorted(sampler._bins.keys()),
)

# %%
# Building the pipeline
# ----------------------
# Two sub-stages:
#
# * Stage 0 — FIRE geometry relaxation (convergence at fmax < 0.5 eV/Å;
#   deliberately loose so most systems converge quickly in this demo).
# * Stage 1 — NVT equilibration for 20 steps.
#
# :class:`~nvalchemi.dynamics.hooks.ConvergedSnapshotHook` fires at
# ``ON_CONVERGE``.  When a system exits stage 1 (its step budget is
# exhausted and it graduates), the hook writes only that system's data to
# the :class:`~nvalchemi.dynamics.sinks.HostMemory` sink.  This is the
# recommended pattern for collecting results in inflight runs without
# writing intermediate states.

torch.manual_seed(42)
model = DemoModelWrapper()
model.eval()

results_sink = HostMemoryWithSystemId(capacity=30)
converged_hook = ConvergedSnapshotHook(sink=results_sink)

fire_stage = FIRE(
    model=model,
    dt=0.1,
    convergence_hook=ConvergenceHook.from_forces(threshold=0.5),
)
nvt_stage = NVTLangevin(
    model=model,
    dt=0.5,
    temperature=300.0,
    friction=0.1,
    random_seed=11,
    n_steps=20,
    hooks=[converged_hook],
)

fused = FusedStage(
    sub_stages=[(0, fire_stage), (1, nvt_stage)],
    sampler=sampler,
    sinks=[results_sink],
    refill_frequency=5,
)

# %%
# Running in inflight mode (batch=None)
# ---------------------------------------
# Passing ``batch=None`` tells :class:`FusedStage` to call
# ``sampler.build_initial_batch()`` to create the first live batch, then
# run until the sampler is exhausted **and** all remaining systems have
# graduated through all stages.
#
# ``n_steps=500`` is the maximum total step budget.  If the dataset is fully
# processed before this limit, the run terminates early and returns ``None``.

logging.info("Starting inflight batching run...")
result = fused.run(batch=None, n_steps=500)

if result is None:
    logging.info("All %d systems processed — sampler exhausted.", dataset.n_samples)
else:
    logging.info(
        "Step budget exhausted with %d systems still active.", result.num_graphs
    )

# %%
# Inspecting results
# -------------------
# The :class:`~nvalchemi.dynamics.sinks.HostMemory` sink accumulates
# graduated systems on CPU.  Drain it to retrieve a single ``Batch``
# containing all collected structures.

n_collected = len(results_sink)
logging.info("Results sink contains %d systems.", n_collected)

if n_collected > 0:
    results_batch = results_sink.drain()
    system_ids = results_batch["system_id"].squeeze(-1).tolist()
    n_sentinel = sum(1 for sid in system_ids if sid < 0)
    if n_sentinel > 0:
        logging.warning(
            "%d system_id(s) are sentinel values (<0); data may be incomplete.",
            n_sentinel,
        )
    logging.info("Collected system_ids (first 10): %s", system_ids[:10])
    logging.info(
        "Results batch: num_graphs=%d, num_nodes=%d",
        results_batch.num_graphs,
        results_batch.num_nodes,
    )
else:
    logging.info("No results collected (step budget may have been too small).")
