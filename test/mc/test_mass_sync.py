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

from pathlib import Path

import torch

from nvalchemi._checkpoint import load_checkpoint, save_checkpoint
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


def test_custom_mass_does_not_cross_into_another_graph() -> None:
    """Graph 0's deuterium mass for H must not reach H created in graph 1."""
    deuterium = AtomicData(
        atomic_numbers=torch.tensor([1], dtype=torch.long),
        positions=torch.zeros(1, 3),
        atomic_masses=torch.tensor([2.014]),
    )
    helium = AtomicData(
        atomic_numbers=torch.tensor([2], dtype=torch.long), positions=torch.zeros(1, 3)
    )
    batch = Batch.from_data_list([deuterium, helium])
    batch.energy = torch.zeros(2, 1)
    # mu_2 = -1e6: He -> H always accepted, H -> He always rejected.
    _sgc(DemoModelWrapper(DemoModel()), mu_2=-1.0e6).run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [1, 1]
    assert abs(batch.atomic_masses[0].item() - 2.014) < 1e-5
    assert torch.isclose(batch.atomic_masses[1], _default_mass(1, batch.atomic_masses))


def test_checkpoint_keeps_the_mass_of_a_species_that_left_the_graph(
    tmp_path: Path,
) -> None:
    """A restart must remember graph 0's deuterium mass after its H is gone."""
    batch = Batch.from_data_list(
        [
            AtomicData(
                atomic_numbers=torch.tensor([1], dtype=torch.long),
                positions=torch.zeros(1, 3),
                atomic_masses=torch.tensor([2.014]),
            )
        ]
    )
    batch.energy = torch.zeros(1, 1)
    model = DemoModelWrapper(DemoModel())
    to_helium = _sgc(model, mu_2=1.0e6)
    to_helium.run(batch, n_steps=1)
    assert batch.atomic_numbers.tolist() == [2]
    save_checkpoint(tmp_path / "mc", {"mc": to_helium})

    # Load and re-save before any step: the restored lookup must survive.
    reloaded = _sgc(model, mu_2=-1.0e6)
    load_checkpoint(tmp_path / "mc", {"mc": reloaded})
    save_checkpoint(tmp_path / "mc_again", {"mc": reloaded})

    to_hydrogen = _sgc(model, mu_2=-1.0e6)
    load_checkpoint(tmp_path / "mc_again", {"mc": to_hydrogen})
    to_hydrogen.run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [1]
    assert abs(batch.atomic_masses[0].item() - 2.014) < 1e-5
    assert to_hydrogen.stats.attempted == 2


def test_transmutation_rescales_velocity_at_fixed_kinetic_energy() -> None:
    """H -> He keeps 0.5 m v^2: v scales by sqrt(m_H / m_He)."""
    batch = _one_atom(1)
    batch.velocities = torch.tensor([[1.0, -2.0, 0.5]])
    before_v = batch.velocities.clone()
    before_m = batch.atomic_masses.clone()
    _sgc(DemoModelWrapper(DemoModel()), mu_2=1.0e6).run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [2]
    scale = torch.sqrt(before_m / batch.atomic_masses)
    torch.testing.assert_close(batch.velocities, before_v * scale[:, None])
    torch.testing.assert_close(
        (batch.atomic_masses[:, None] * batch.velocities**2).sum(),
        (before_m[:, None] * before_v**2).sum(),
    )


def test_rejected_transmutation_leaves_velocity_alone() -> None:
    batch = _one_atom(1)
    batch.velocities = torch.tensor([[1.0, -2.0, 0.5]])
    before = batch.velocities.clone()
    _sgc(DemoModelWrapper(DemoModel()), mu_2=-1.0e6).run(batch, n_steps=1)

    assert torch.equal(batch.velocities, before)


def test_swap_keeps_every_atoms_kinetic_energy() -> None:
    """An H-He swap rescales both atoms, so each keeps its kinetic energy."""
    data = AtomicData(
        atomic_numbers=torch.tensor([1, 2], dtype=torch.long),
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        velocities=torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.0]]),
    )
    batch = Batch.from_data_list([data])
    batch.energy = torch.zeros(1, 1)
    kinetic = (batch.atomic_masses[:, None] * batch.velocities**2).sum(-1)
    Kawasaki(
        model=DemoModelWrapper(DemoModel()),
        temperature=1.0e6,
        cutoff=2.0,
        random_seed=7,
    ).run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [2, 1]
    torch.testing.assert_close(
        (batch.atomic_masses[:, None] * batch.velocities**2).sum(-1), kinetic
    )
