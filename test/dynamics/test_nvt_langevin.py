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
"""Unit tests for :class:`~nvalchemi.dynamics.NVTLangevin`.

The integrator reads each atom's timestep, temperature, and friction through
the atom's system id, so the ids it hands the kernels must be those of the
batch it is stepping, whatever batch it stepped before.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics import NVTLangevin
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _make_batch(
    n_graphs: int, atoms_per_graph: int, device: str, seed: int = 0
) -> Batch:
    """Return a batch at rest with the fields the integrator reads and writes."""
    generator = torch.Generator().manual_seed(seed)
    data_list = []
    for _ in range(n_graphs):
        data = AtomicData(
            positions=torch.randn(atoms_per_graph, 3, generator=generator),
            atomic_numbers=torch.full((atoms_per_graph,), 6, dtype=torch.long),
            atomic_masses=torch.ones(atoms_per_graph),
            forces=torch.zeros(atoms_per_graph, 3),
            energy=torch.zeros(1, 1),
        )
        data.add_node_property("velocities", torch.zeros(atoms_per_graph, 3))
        data_list.append(data)
    return Batch.from_data_list(data_list).to(device)


def _langevin(
    model: DemoModelWrapper, temperature: float | torch.Tensor, random_seed: int
) -> NVTLangevin:
    """Return a Langevin integrator over *model*."""
    return NVTLangevin(
        model=model,
        dt=0.5,
        temperature=temperature,
        friction=0.1,
        random_seed=random_seed,
    )


class TestBatchReplacement:
    """The system ids follow the batch the integrator is handed.

    The on-policy segment loop replaces the propagated batch at every segment
    boundary with a fresh draw filling the same atom budget and resizes the
    per-system state through ``_sync_state_to_batch``. Ids kept from the
    previous batch would then send atoms to another system's parameters, or
    past the end of the state when the new batch holds fewer systems.
    """

    @pytest.mark.parametrize(
        ("before", "after"),
        [((3, 4), (2, 6)), ((2, 6), (3, 4))],
        ids=["fewer-graphs", "more-graphs"],
    )
    def test_a_replaced_batch_steps_like_a_fresh_integrator(
        self, device: str, before: tuple[int, int], after: tuple[int, int]
    ) -> None:
        """Same atom count, different systems: the step matches a fresh integrator's."""
        temperature = torch.tensor([300.0, 600.0, 900.0][: after[0]], device=device)
        model = DemoModelWrapper(DemoModel()).to(device)
        swapped = _langevin(model, temperature=300.0, random_seed=7)
        swapped.step(_make_batch(*before, device))

        replaced = _make_batch(*after, device, seed=1)
        swapped.compute(replaced)
        swapped._sync_state_to_batch(
            torch.empty(0, dtype=torch.long, device=device),
            replaced.num_graphs,
            replaced,
        )
        swapped.apply_per_system_params({"temperature": temperature}, replaced)
        swapped.step(replaced)

        # The noise stream is seeded by ``random_seed + step_count``.
        fresh = _langevin(model, temperature=temperature, random_seed=8)
        expected = _make_batch(*after, device, seed=1)
        fresh.step(expected)

        assert bool(torch.isfinite(replaced.positions).all())
        torch.testing.assert_close(replaced.positions, expected.positions)
        torch.testing.assert_close(replaced.velocities, expected.velocities)
