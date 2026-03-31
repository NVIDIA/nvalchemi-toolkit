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

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import (
    CompositeCalculator,
    EnergyDerivativesStep,
    MACEPotential,
    NeighborListBuilder,
)


class _MockIrrepsOut:
    """Minimal mock for the MACE hidden-dimension path."""

    dim = 8


class _MockLinear:
    """Minimal mock for the MACE hidden-dimension path."""

    irreps_out = _MockIrrepsOut()


class _MockProduct:
    """Minimal mock for the MACE hidden-dimension path."""

    linear = _MockLinear()


class _MockAtomicEnergies(torch.nn.Module):
    """Minimal mock for the MACE atomic baseline block."""

    def __init__(self, values: list[float]) -> None:
        super().__init__()
        self.register_buffer(
            "atomic_energies", torch.tensor(values, dtype=torch.float64)
        )

    def forward(self, node_attrs: torch.Tensor) -> torch.Tensor:
        """Return per-atom baseline energies from one-hot node attributes."""

        return torch.matmul(
            node_attrs.to(dtype=self.atomic_energies.dtype),
            torch.atleast_2d(self.atomic_energies).T,
        )


class MockMACEModel(torch.nn.Module):
    """Minimal MACE-like model for root package tests."""

    def __init__(self, *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.atomic_numbers = torch.tensor([1, 8], dtype=torch.long)
        self.r_max = torch.tensor(5.0)
        self.products = [_MockProduct()]
        self.atomic_energies_fn = _MockAtomicEnergies([1.5, 10.0])
        self._param = torch.nn.Linear(1, 1, bias=False).to(dtype=dtype)

    def forward(
        self, data_dict: dict[str, torch.Tensor], **kwargs: object
    ) -> dict[str, torch.Tensor]:
        """Return one per-system scalar energy."""

        del kwargs
        positions = data_dict["positions"]
        batch = data_dict["batch"].long()
        num_graphs = int(batch.max().item()) + 1 if positions.shape[0] > 0 else 1
        energy_dtype = torch.promote_types(
            positions.dtype,
            self.atomic_energies_fn.atomic_energies.dtype,
        )
        norms = (
            positions.pow(2).sum(dim=-1).clamp(min=1e-8).sqrt().to(dtype=energy_dtype)
        )
        energy = torch.zeros(num_graphs, dtype=energy_dtype, device=positions.device)
        energy.scatter_add_(0, batch, norms)
        baseline = (
            self.atomic_energies_fn(data_dict["node_attrs"])
            .squeeze(-1)
            .to(device=positions.device)
        )
        energy.scatter_add_(0, batch, baseline)
        return {"energy": energy}


def _make_batch() -> Batch:
    """Build a small non-periodic batch."""

    data = [
        AtomicData(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
                dtype=torch.float32,
            ),
            atomic_numbers=torch.tensor([8, 1, 1], dtype=torch.long),
        )
    ]
    return Batch.from_data_list(data)


@pytest.fixture
def mock_model() -> MockMACEModel:
    """Return a minimal MACE-like model."""

    return MockMACEModel()


def test_mace_root_api_neighbor_contract_and_builder_config(
    mock_model: MockMACEModel,
) -> None:
    """MACE should expose an external neighbor contract through the root package."""

    potential = MACEPotential(mock_model, neighbor_list_name="short_range")

    assert potential.profile.neighbor_requirement.source == "external"
    assert potential.profile.neighbor_requirement.name == "short_range"
    assert potential.profile.neighbor_requirement.format == "coo"
    assert potential.profile.neighbor_requirement.cutoff == pytest.approx(5.0)

    config = potential.neighbor_list_builder_config(cutoff=6.0, reuse_if_available=True)
    assert config is not None
    assert config.neighbor_list_name == "short_range"
    assert config.cutoff == pytest.approx(6.0)
    assert config.reuse_if_available is True


def test_mace_root_api_composite_derives_forces(
    mock_model: MockMACEModel,
) -> None:
    """MACE should participate in the root composite autograd flow."""

    batch = _make_batch()
    potential = MACEPotential(mock_model, neighbor_list_name="short_range")
    neighbor = NeighborListBuilder(potential.neighbor_list_builder_config())
    calculator = CompositeCalculator(
        neighbor,
        potential,
        EnergyDerivativesStep(),
        outputs={"energies", "forces"},
    )

    results = calculator(batch)

    assert tuple(results["energies"].shape) == (batch.num_graphs, 1)
    assert tuple(results["forces"].shape) == tuple(batch.positions.shape)
