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
"""Unit tests for semi-grand-canonical Monte Carlo."""

from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.mc import SGC
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _batch(numbers: list[list[int]]) -> Batch:
    """Create one small graph per supplied sequence of atomic numbers."""
    data = [
        AtomicData(
            atomic_numbers=torch.tensor(values, dtype=torch.long),
            positions=torch.zeros(len(values), 3),
        )
        for values in numbers
    ]
    batch = Batch.from_data_list(data)
    batch.energy = torch.zeros(batch.num_graphs, 1)
    return batch


def _sampler(chemical_potentials: dict[int, float]) -> SGC:
    """Construct a deterministic test sampler around the demo model."""
    return SGC(
        model=DemoModelWrapper(DemoModel()),
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials=chemical_potentials,
        random_seed=7,
    )


def test_higher_chemical_potential_favours_insertion() -> None:
    """A large positive mu for species 2 accepts the 1-to-2 transmutation."""
    batch = _batch([[1]])
    sampler = _sampler({1: 0.0, 2: 1.0e6})

    sampler.run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [2]
    assert batch.mc_accepted.tolist() == [[True]]
    assert sampler.stats.attempted == 1
    assert sampler.stats.accepted == 1


def test_unfavourable_transmutation_restores_the_accepted_type() -> None:
    """A large negative mu for species 2 rejects and restores the old type."""
    batch = _batch([[1]])
    sampler = _sampler({1: 0.0, 2: -1.0e6})

    sampler.run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [1]
    assert batch.mc_accepted.tolist() == [[False]]
    assert sampler.stats.attempted == 1
    assert sampler.stats.accepted == 0


def test_per_graph_chemical_potentials_sample_different_reservoirs() -> None:
    """One SGC batch can carry a distinct reservoir for every graph."""
    batch = _batch([[1], [1]])
    sampler = SGC(
        model=DemoModelWrapper(DemoModel()),
        temperature=torch.tensor([1000.0, 1000.0]),
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: torch.tensor([1.0e6, -1.0e6])},
        random_seed=7,
    )

    sampler.run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [2, 1]
    assert batch.mc_accepted.tolist() == [[True], [False]]


def test_per_graph_chemical_potential_shape_is_checked() -> None:
    """A reservoir tensor must match the number of active graphs exactly."""
    batch = _batch([[1], [1]])
    sampler = SGC(
        model=DemoModelWrapper(DemoModel()),
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: torch.tensor([0.0])},
    )

    with pytest.raises(ValueError, match="chemical-potential tensors"):
        sampler.run(batch, n_steps=1)


def test_unequal_per_graph_chemical_potentials_rejected_at_construction() -> None:
    """Per-graph reservoirs of different lengths cannot describe one batch."""
    with pytest.raises(ValueError, match="equal lengths"):
        _sampler({1: torch.tensor([0.0, 0.1]), 2: torch.tensor([0.0, 0.1, 0.2])})


def test_species_and_potentials_are_copied_to_device_once() -> None:
    """Repeated steps reuse the cached device tensors instead of re-uploading."""
    batch = _batch([[1, 2], [2, 1]])
    sampler = _sampler({1: 0.0, 2: torch.tensor([0.1, -0.1])})

    sampler.run(batch, n_steps=1)
    cached = dict(sampler._device_parameters)
    sampler.run(batch, n_steps=3)

    assert set(cached) == set(sampler._device_parameters)
    assert all(
        sampler._device_parameters[key] is value for key, value in cached.items()
    )
    torch.testing.assert_close(
        sampler._chemical_potentials_for(batch),
        torch.tensor([[0.0, 0.1], [0.0, -0.1]]),
    )


def test_new_batch_gets_a_fresh_energy_baseline() -> None:
    """A batch replacing a freed one (possibly at the same id) is re-initialised."""
    sampler = _sampler({1: 0.0, 2: 0.0})
    sampler.run(_batch([[1, 2]]), n_steps=1)

    with patch.object(
        sampler, "_initialize_energy", wraps=sampler._initialize_energy
    ) as init:
        sampler.run(_batch([[2, 2]]), n_steps=1)

    init.assert_called_once()


def test_rejects_missing_reservoir_species() -> None:
    """The starting composition must be representable by the reservoir."""
    batch = _batch([[3]])
    sampler = _sampler({1: 0.0, 2: 0.0})

    with pytest.raises(ValueError, match="outside the configured species"):
        sampler.run(batch, n_steps=1)


def test_step_follows_the_dynamics_hook_lifecycle() -> None:
    """Admission fires once; per-step hooks see the step's active-graph mask."""
    from nvalchemi.dynamics.base import DynamicsStage

    seen: list[tuple[str, list[bool] | None]] = []

    class Recorder:
        frequency = 1

        def __init__(self, stage: DynamicsStage) -> None:
            self.stage = stage

        def __call__(self, ctx, stage) -> None:
            mask = ctx.active_graph_mask
            seen.append((stage.name, None if mask is None else mask.tolist()))

    batch = _batch([[1], [1]])
    batch.status = torch.tensor([[0], [1]], dtype=torch.long)
    sampler = SGC(
        model=DemoModelWrapper(DemoModel()),
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: 0.0},
        exit_status=1,
        hooks=[
            Recorder(DynamicsStage.ON_ADMISSION),
            Recorder(DynamicsStage.BEFORE_STEP),
        ],
    )

    sampler.run(batch, n_steps=2)

    assert seen == [
        ("ON_ADMISSION", [True, False]),
        ("BEFORE_STEP", [True, False]),
        ("BEFORE_STEP", [True, False]),
    ]
