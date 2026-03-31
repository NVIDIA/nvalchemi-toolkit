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
from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import (
    AIMNet2Potential,
    CompositeCalculator,
    EnergyDerivativesStep,
    Potential,
    PotentialCard,
)

_AUTOGRAD_ENERGY_CARD = PotentialCard(
    required_inputs=frozenset({"positions"}),
    result_keys=frozenset({"energies"}),
    default_result_keys=frozenset({"energies"}),
    additive_result_keys=frozenset({"energies"}),
    gradient_setup_targets=frozenset({"positions"}),
)


class _AutogradEnergyPotential(Potential):
    """Tiny energy-only potential used to smoke-test the root API."""

    card = _AUTOGRAD_ENERGY_CARD

    def __init__(self) -> None:
        super().__init__(name="autograd_energy")

    def compute(self, batch: Batch, ctx) -> dict[str, torch.Tensor]:
        """Return a simple per-system energy from positions."""

        positions = self.require_input(batch, "positions", ctx)
        per_atom = positions.pow(2).sum(dim=-1)
        energies = torch.zeros(
            batch.num_graphs, 1, dtype=positions.dtype, device=positions.device
        )
        energies.index_add_(0, batch.batch.long(), per_atom.unsqueeze(-1))
        return self.build_results(ctx, energies=energies)


def _make_batch() -> Batch:
    """Return a small non-periodic batch."""

    data = [
        AtomicData(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float32
            ),
            atomic_numbers=torch.tensor([8, 1], dtype=torch.long),
        )
    ]
    return Batch.from_data_list(data)


def test_root_composite_derives_forces() -> None:
    """The root API should expose the composite autograd architecture."""

    batch = _make_batch()
    calculator = CompositeCalculator(
        _AutogradEnergyPotential(),
        EnergyDerivativesStep(),
        outputs={"energies", "forces"},
    )

    results = calculator(batch)

    assert tuple(results["energies"].shape) == (batch.num_graphs, 1)
    assert tuple(results["forces"].shape) == tuple(batch.positions.shape)


def test_aimnet2_card_still_advertises_internal_neighbors() -> None:
    """AIMNet2 should remain an internal-neighbor MLIP in the root package."""

    assert AIMNet2Potential.card.neighbor_requirement.source == "internal"
