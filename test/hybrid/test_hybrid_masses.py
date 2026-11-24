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
"""MD blocks of a hybrid run integrate transmuted atoms with their new mass."""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.data.atomic_data import _default_mass_table
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.hybrid import HybridMCMD
from nvalchemi.mc import SGC
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def test_hybrid_md_integrates_the_transmuted_atom_with_its_new_mass() -> None:
    """No caller-side mass refresh: the MD block already sees the right mass."""
    model = DemoModelWrapper(DemoModel())
    batch = Batch.from_data_list(
        [AtomicData(atomic_numbers=torch.tensor([1]), positions=torch.zeros(1, 3))]
    )
    batch.energy = torch.zeros(1, 1)
    batch.forces = torch.zeros(1, 3)
    # A huge mu for He forces acceptance of the H -> He transmutation.
    sgc = SGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: 1.0e6},
        random_seed=4,
    )
    md = DemoDynamics(model=model, n_steps=None, dt=0.01)
    seen: list[float] = []
    run = md.run

    def spy(b, n_steps):
        seen.append(float(b.atomic_masses[0]))
        return run(b, n_steps=n_steps)

    md.run = spy
    HybridMCMD(mc=sgc, md=md, mc_steps=1, md_steps=1).run(batch, n_blocks=1)

    assert batch.atomic_numbers.tolist() == [2]
    assert abs(seen[0] - float(_default_mass_table()[2])) < 1e-5
