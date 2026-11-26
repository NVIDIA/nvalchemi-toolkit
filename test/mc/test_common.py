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
"""Behaviour every MC sampler shares: graduated graphs, device-resident steps."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.mc import SGC, VCSGC, BaseMonteCarlo, Kawasaki, NonlocalSwap
from nvalchemi.mc.base import _draw_from_mask, _swap_pairs
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _chain(numbers: list[int]) -> AtomicData:
    """One open chain of atoms 1 A apart along x."""
    return AtomicData(
        atomic_numbers=torch.tensor(numbers, dtype=torch.long),
        positions=torch.tensor([[float(i), 0.0, 0.0] for i in range(len(numbers))]),
    )


_SAMPLERS: dict[str, Callable[[DemoModelWrapper], BaseMonteCarlo]] = {
    "sgc": lambda model: SGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: 0.1},
    ),
    "vcsgc": lambda model: VCSGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        kappa=1.0,
        target_concentration=0.5,
    ),
    "kawasaki": lambda model: Kawasaki(model=model, temperature=1000.0, cutoff=1.5),
    "swap": lambda model: NonlocalSwap(model=model, temperature=1000.0),
}


def test_draw_from_mask_picks_one_selectable_row_per_segment() -> None:
    """Each segment draws within itself; an empty segment returns its start row."""
    selectable = torch.tensor([0, 1, 1, 0, 0, 0, 1])
    starts = torch.tensor([0, 3, 5])
    counts = torch.tensor([2, 0, 1])

    low = _draw_from_mask(selectable, starts, counts, torch.zeros(3))
    high = _draw_from_mask(selectable, starts, counts, torch.full((3,), 0.999))

    assert low.tolist() == [1, 3, 6]
    assert high.tolist() == [2, 3, 6]


def test_swap_pairs_only_touches_flagged_pairs() -> None:
    """Unflagged pairs are rewritten with their own values."""
    numbers = torch.tensor([1, 2, 3, 4])
    _swap_pairs(
        numbers, torch.tensor([0, 2]), torch.tensor([1, 3]), torch.tensor([True, False])
    )
    assert numbers.tolist() == [2, 1, 3, 4]


@pytest.mark.parametrize("name", sorted(_SAMPLERS))
def test_graduated_graph_is_neither_mutated_nor_counted(name: str) -> None:
    """A graph at ``status >= exit_status`` gets no proposals and no statistics."""
    batch = Batch.from_data_list([_chain([1, 1, 2, 2]), _chain([1, 2, 1, 2])])
    batch.energy = torch.zeros(batch.num_graphs, 1)
    batch.status = torch.tensor([[0], [1]], dtype=torch.long)
    sampler = _SAMPLERS[name](DemoModelWrapper(DemoModel()))

    sampler.run(batch, n_steps=5)

    second_graph = batch.atomic_numbers[batch.batch_ptr[1] : batch.batch_ptr[2]]
    assert second_graph.tolist() == [1, 2, 1, 2]
    assert sampler.stats.attempted == 5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="sync detection needs CUDA")
@pytest.mark.parametrize("name", sorted(_SAMPLERS))
def test_proposal_and_acceptance_do_not_sync_the_device(name: str) -> None:
    """A device-to-host sync per MC step stalls the queue ahead of the next model call."""
    batch = Batch.from_data_list([_chain([1, 1, 2, 2]), _chain([1, 2, 1, 2])]).to(
        "cuda"
    )
    batch.energy = torch.zeros(batch.num_graphs, 1, device="cuda")
    sampler = _SAMPLERS[name](DemoModelWrapper(DemoModel()).to("cuda"))
    # Warm-up: batch validation, device caches and the proposal graph may sync once.
    sampler.run(batch, n_steps=1)
    torch.cuda.synchronize()

    previous = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        sampler.pre_update(batch)
        sampler.post_update(batch)
    finally:
        torch.cuda.set_sync_debug_mode(previous)

    assert sampler.stats.attempted == 4


@pytest.mark.parametrize("restore_into", ["fresh", "advanced"])
@pytest.mark.parametrize("name", list(_SAMPLERS))
def test_restart_continues_the_random_sequence(name: str, restore_into: str) -> None:
    """A run restored from ``state_dict`` matches an uninterrupted one step for step."""

    def fresh_batch() -> Batch:
        batch = Batch.from_data_list([_chain([1, 2, 1, 2, 1, 2]) for _ in range(3)])
        batch.energy = torch.zeros(batch.num_graphs, 1)
        return batch

    model = DemoModelWrapper(DemoModel())
    reference = fresh_batch()
    uninterrupted = _SAMPLERS[name](model)
    uninterrupted.run(reference, n_steps=20)

    batch = fresh_batch()
    first = _SAMPLERS[name](model)
    first.run(batch, n_steps=10)
    state = first.state_dict()
    assert "generator_state" in state
    if restore_into == "advanced":
        first.run(fresh_batch(), n_steps=5)  # advance it past the checkpoint
        resumed = first
    else:
        resumed = _SAMPLERS[name](model)
    resumed.load_state_dict(state)
    # Re-saving before any step keeps the restored state.
    resaved = resumed.state_dict()
    assert torch.equal(resaved["generator_state"], state["generator_state"])
    assert torch.equal(resaved["mass_table"], state["mass_table"])
    resumed.run(batch, n_steps=10)

    assert torch.equal(batch.atomic_numbers, reference.atomic_numbers)
    assert resumed.stats == uninterrupted.stats
