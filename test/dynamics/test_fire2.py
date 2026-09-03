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
"""Tests for FIRE2 optimizer state and integration behavior."""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.optimizers.fire2 import FIRE2
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


def _make_model() -> DemoModelWrapper:
    """Return the demo model wrapper used by FIRE2 tests."""
    return DemoModelWrapper(DemoModel())


def _make_batch(
    n_systems: int,
    *,
    n_atoms_each: int = 4,
) -> Batch:
    """Return a minimal batch suitable for FIRE2 tests."""
    data_list = []
    for seed in range(n_systems):
        generator = torch.Generator().manual_seed(seed)
        data_kwargs = {
            "positions": torch.randn(n_atoms_each, 3, generator=generator),
            "atomic_numbers": torch.randint(
                1,
                10,
                (n_atoms_each,),
                dtype=torch.long,
                generator=generator,
            ),
            "atomic_masses": torch.ones(n_atoms_each),
            "forces": torch.zeros(n_atoms_each, 3),
            "energy": torch.zeros(1, 1),
        }
        data = AtomicData(**data_kwargs)
        data.add_node_property("velocities", torch.zeros(n_atoms_each, 3))
        data_list.append(data)
    return Batch.from_data_list(data_list)


class TestGroupedFIRE2:
    """FIRE2 maps grouped graphs onto group-sized optimizer state."""

    @staticmethod
    def _grouped_batch() -> Batch:
        batch = _make_batch(4, n_atoms_each=2)
        batch.set_group_layout(torch.tensor([0, 0, 1, 1]))
        return batch

    def test_state_and_update_index_forwarding(self, monkeypatch):
        from nvalchemi.dynamics.optimizers import fire2 as fire2_module

        batch = self._grouped_batch()
        original_batch_idx = batch.batch_idx.clone()
        dynamics = FIRE2(model=_make_model(), dt=0.05, by_group=True)
        dynamics._ensure_state_initialized(batch)
        captured = {}

        def fake_fire2_step(*args, **kwargs):
            captured["update_idx"] = args[3].clone()

        monkeypatch.setattr(fire2_module, "fire2_step_coord", fake_fire2_step)
        dynamics.pre_update(batch)

        assert dynamics._state.num_graphs == 2
        assert dynamics._state.dt.shape == (2,)
        torch.testing.assert_close(
            captured["update_idx"],
            torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32),
        )
        torch.testing.assert_close(batch.batch_idx, original_batch_idx)
