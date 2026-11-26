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
"""Monte Carlo sub-stages inside a ``FusedStage`` (``mc + md``)."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import FusedStage
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.mc import SGC, NonlocalSwap
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _batch() -> Batch:
    """Four one-atom graphs: statuses 0 (MC) and 1 (MD), two each."""
    data = [
        AtomicData(
            atomic_numbers=torch.tensor([1], dtype=torch.long),
            positions=torch.tensor([[float(i), 0.0, 0.0]]),
        )
        for i in range(4)
    ]
    batch = Batch.from_data_list(data)
    batch.forces = torch.zeros(batch.num_nodes, 3)
    batch.energy = torch.zeros(batch.num_graphs, 1)
    batch.velocities = torch.ones(batch.num_nodes, 3)
    batch.fmax = torch.ones(batch.num_graphs, 1)
    batch.status = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    return batch


def _fused(mu_2: float) -> tuple[FusedStage, SGC]:
    model = DemoModelWrapper(DemoModel())
    sgc = SGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: mu_2},
        random_seed=7,
    )
    return sgc + DemoDynamics(model=model, n_steps=100, dt=1.0), sgc


@pytest.mark.parametrize(("mu_2", "mc_types"), [(1.0e6, [2, 2]), (-1.0e6, [1, 1])])
def test_mc_moves_only_its_graphs_and_energies_stay_current(mu_2, mc_types) -> None:
    """Accepted and rejected MC graphs, and the MD graphs, all end with their own energy."""
    fused, sgc = _fused(mu_2)
    batch = _batch()
    md_positions = batch.positions[2:].clone()

    for _ in range(3):
        fused.step(batch)

    assert batch.atomic_numbers.tolist() == [*mc_types, 1, 1]
    assert not torch.allclose(batch.positions[2:], md_positions)  # MD still moves
    assert sgc.stats.attempted == 6  # two MC graphs x three steps
    carried = batch.energy.reshape(-1).clone()
    torch.testing.assert_close(carried, fused.compute(batch)["energy"].reshape(-1))


@pytest.mark.parametrize(("mu_2", "accepted"), [(-1.0e6, False), (1.0e6, True)])
def test_graphs_leaving_mc_use_forces_of_their_current_state(mu_2, accepted) -> None:
    """A rejected trial's forces are not carried into MD; an accepted one's are.

    With ``n_steps=1`` the MC graphs make one trial, then migrate to MD. After a
    rejection they skip one MD update while the shared compute re-primes them;
    after an acceptance the trial forces describe the current state and MD
    integrates with them straight away.
    """
    model = DemoModelWrapper(DemoModel())
    sgc = SGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: mu_2},
        random_seed=7,
        n_steps=1,
    )
    fused = sgc + DemoDynamics(model=model, n_steps=100, dt=1.0)
    batch = _batch()

    fused.step(batch)  # one MC trial on graphs 0 and 1, then migration to MD

    assert batch.status.view(-1).tolist() == [1, 1, 1, 1]
    assert batch.mc_accepted.view(-1)[:2].tolist() == [accepted, accepted]
    assert batch.reprime_pending.view(-1)[:2].tolist() == [not accepted] * 2
    carried = batch.forces[:2].clone()
    current = model(batch.clone())["forces"][:2].detach()
    # The rejected trial's forces belong to the undone species.
    assert torch.allclose(carried, current) is accepted

    positions = batch.positions[:2].clone()
    fused.step(batch)

    # Re-primed graphs sit out one update; accepted ones are integrated at once.
    assert torch.equal(batch.positions[:2], positions) is not accepted
    torch.testing.assert_close(
        batch.forces[:2], model(batch.clone())["forces"][:2].detach()
    )


def test_mc_without_a_legal_move_leaves_the_stage_mask_alone() -> None:
    """Narrowing the step's mask to graphs with a legal move is the sampler's own business.

    A single-species graph has no swap; ``NonlocalSwap`` drops it from its
    step, which must not clear it from the mask the ``FusedStage`` passed in.
    """
    batch = Batch.from_data_list(
        [
            AtomicData(
                atomic_numbers=torch.tensor(numbers, dtype=torch.long),
                positions=torch.arange(6.0).reshape(2, 3),
            )
            for numbers in ([1, 1], [1, 2])
        ]
    )
    batch.energy = torch.zeros(batch.num_graphs, 1)
    sampler = NonlocalSwap(
        model=DemoModelWrapper(DemoModel()), temperature=1000.0, random_seed=7
    )
    mask = torch.ones(batch.num_graphs, dtype=torch.bool)

    sampler._masked_pre_update(batch, mask)

    assert mask.tolist() == [True, True]
    assert sampler._active.tolist() == [False, True]
