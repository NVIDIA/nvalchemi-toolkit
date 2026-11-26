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
"""Unit tests for canonical non-local swap Monte Carlo."""

from __future__ import annotations

import itertools
from collections import Counter
from typing import Any

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.mc import NonlocalSwap
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _chain(numbers: list[int], *, separation: float = 1.0) -> AtomicData:
    """One open chain of atoms spaced `separation` apart along x."""
    return AtomicData(
        atomic_numbers=torch.tensor(numbers, dtype=torch.long),
        positions=torch.tensor(
            [[index * separation, 0.0, 0.0] for index in range(len(numbers))]
        ),
    )


def _sampler(**kwargs: Any) -> NonlocalSwap:
    return NonlocalSwap(
        model=DemoModelWrapper(DemoModel()), temperature=1000.0, random_seed=7, **kwargs
    )


def _batch(*graphs: list[int]) -> Batch:
    batch = Batch.from_data_list([_chain(numbers) for numbers in graphs])
    batch.energy = torch.zeros(batch.num_graphs, 1)
    return batch


def test_every_proposal_swaps_an_unlike_pair_within_its_graph() -> None:
    """Each active graph exchanges two unlike atoms of its own; no model call is wasted."""
    batch = _batch([1, 1, 1, 2], [2, 1, 2, 2, 2], [1, 2])
    sampler = _sampler()
    ptr = batch.batch_ptr.tolist()

    for _ in range(50):
        before = batch.atomic_numbers.clone()
        sampler.pre_update(batch)
        assert bool(sampler._proposal_swapped.all())
        for graph in range(batch.num_graphs):
            first = int(sampler._proposal_first[graph])
            second = int(sampler._proposal_second[graph])
            assert ptr[graph] <= first < ptr[graph + 1]
            assert ptr[graph] <= second < ptr[graph + 1]
            assert before[first] != before[second]
        sampler._restore_rejected(batch, torch.ones(batch.num_graphs, dtype=torch.bool))
        assert torch.equal(batch.atomic_numbers, before)


def test_unlike_pairs_are_drawn_uniformly_at_any_distance() -> None:
    """In a binary every unlike pair has the same proposal probability, near or far."""
    numbers = [1, 1, 1, 2, 2]
    batch = _batch(numbers)
    sampler = _sampler()
    draws = 6000
    pairs: Counter[tuple[int, int]] = Counter()
    for _ in range(draws):
        sampler.pre_update(batch)
        pair = sorted(
            (int(sampler._proposal_first[0]), int(sampler._proposal_second[0]))
        )
        pairs[(pair[0], pair[1])] += 1
        sampler._restore_rejected(batch, torch.ones(1, dtype=torch.bool))

    unlike = {
        (i, j) for i in range(5) for j in range(i + 1, 5) if numbers[i] != numbers[j]
    }
    assert (
        set(pairs) == unlike
    )  # all 6 unlike pairs, including the 3-bond-long (0, 3), (0, 4)
    expected = draws / len(unlike)
    for count in pairs.values():
        assert count == pytest.approx(expected, rel=0.1)


def test_single_species_graph_has_no_legal_move() -> None:
    """A graph with one species is skipped instead of proposing a no-op."""
    batch = _batch([1, 1, 1], [1, 2])
    sampler = _sampler()

    sampler.run(batch, n_steps=3)

    assert batch.atomic_numbers[:3].tolist() == [1, 1, 1]
    assert sampler.stats.attempted == 3  # only the mixed graph proposes


def test_composition_and_masses_follow_over_many_steps() -> None:
    """A longer batched run never changes the species counts, and masses move with species."""
    graphs = ([1, 2, 1, 2, 1, 2, 1, 1], [2, 2, 2, 1])
    batch = _batch(*graphs)
    batch.atomic_masses = torch.where(batch.atomic_numbers == 1, 1.0, 4.0).to(
        batch.positions.dtype
    )
    sampler = _sampler()

    sampler.run(batch, n_steps=40)

    ptr = batch.batch_ptr.tolist()
    for graph, numbers in enumerate(graphs):
        assert Counter(
            batch.atomic_numbers[ptr[graph] : ptr[graph + 1]].tolist()
        ) == Counter(numbers)
    expected_masses = torch.where(batch.atomic_numbers == 1, 1.0, 4.0)
    assert torch.equal(
        batch.atomic_masses.reshape(-1), expected_masses.to(batch.atomic_masses.dtype)
    )
    assert sampler.stats.attempted == 80


def test_samples_the_canonical_boltzmann_distribution() -> None:
    """Exact check of detailed balance: 2 B atoms on 5 sites, every arrangement enumerated."""
    torch.manual_seed(0)
    model = DemoModelWrapper(DemoModel())
    temperature = 1000.0
    positions = torch.tensor(
        [[index * 1.0, 0.3 * (index % 2), 0.0] for index in range(5)]
    )
    arrangements = list(itertools.combinations(range(5), 2))

    def graph(sites: tuple[int, ...]) -> AtomicData:
        numbers = torch.ones(5, dtype=torch.long)
        numbers[list(sites)] = 2
        return AtomicData(atomic_numbers=numbers, positions=positions.clone())

    exact = Batch.from_data_list([graph(sites) for sites in arrangements])
    energies = model(exact)["energy"].detach().reshape(-1).double()
    weights = torch.exp(-(energies - energies.min()) / (KB_EV * temperature))
    expected = (weights / weights.sum()).tolist()

    replicas, steps, burn_in = 16, 1200, 100
    batch = Batch.from_data_list([graph(arrangements[0]) for _ in range(replicas)])
    batch.energy = torch.zeros(replicas, 1)
    sampler = NonlocalSwap(model=model, temperature=temperature, random_seed=11)
    counts: Counter[tuple[int, ...]] = Counter()
    for step in range(steps):
        sampler.run(batch, n_steps=1)
        if step < burn_in:
            continue
        numbers = batch.atomic_numbers.reshape(replicas, 5)
        for row in numbers.tolist():
            counts[tuple(i for i, z in enumerate(row) if z == 2)] += 1

    total = sum(counts.values())
    for sites, probability in zip(arrangements, expected):
        assert counts[sites] / total == pytest.approx(probability, abs=0.015)
