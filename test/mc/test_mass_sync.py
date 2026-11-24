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
"""Atomic masses follow species through MC moves."""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.atomic_data import _default_mass_table
from nvalchemi.mc import SGC, Kawasaki
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _one_atom(number: int) -> Batch:
    batch = Batch.from_data_list(
        [
            AtomicData(
                atomic_numbers=torch.tensor([number], dtype=torch.long),
                positions=torch.zeros(1, 3),
            )
        ]
    )
    batch.energy = torch.zeros(1, 1)
    batch.forces = torch.zeros(1, 3)
    return batch


def _sgc(model: DemoModelWrapper, mu_2: float) -> SGC:
    """H/He SGC; a huge +/- mu_2 forces acceptance/rejection of H -> He."""
    return SGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: mu_2},
        random_seed=4,
    )


def _default_mass(number: int, like: torch.Tensor) -> torch.Tensor:
    return _default_mass_table().to(dtype=like.dtype)[number]


def test_accepted_transmutation_takes_the_new_species_mass() -> None:
    batch = _one_atom(1)
    _sgc(DemoModelWrapper(DemoModel()), mu_2=1.0e6).run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [2]
    assert torch.isclose(batch.atomic_masses[0], _default_mass(2, batch.atomic_masses))


def test_rejected_transmutation_leaves_the_mass_alone() -> None:
    batch = _one_atom(1)
    before = batch.atomic_masses.clone()
    _sgc(DemoModelWrapper(DemoModel()), mu_2=-1.0e6).run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [1]
    assert torch.equal(batch.atomic_masses, before)


def test_kawasaki_swap_carries_custom_per_species_masses() -> None:
    """A deuterium mass for H must move with the H atom, not reset to 1.008."""
    custom = {1: 2.014, 2: 4.0026}
    data = AtomicData(
        atomic_numbers=torch.tensor([1, 2], dtype=torch.long),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        atomic_masses=torch.tensor([custom[1], custom[2]]),
    )
    batch = Batch.from_data_list([data])
    batch.energy = torch.zeros(1, 1)
    sampler = Kawasaki(
        model=DemoModelWrapper(DemoModel()),
        temperature=1.0e6,
        cutoff=2.0,
        random_seed=7,
    )

    sampler.run(batch, n_steps=5)

    for number, mass in zip(
        batch.atomic_numbers.tolist(), batch.atomic_masses.tolist()
    ):
        assert abs(mass - custom[number]) < 1e-5, (number, mass)
