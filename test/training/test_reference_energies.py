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
"""Tests for composition-based reference-energy utilities."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.training.reference_energies import (
    fit_atomic_reference_energies,
)


def _structure(numbers: list[int], energy: float) -> AtomicData:
    """Build a minimal graph with a scalar energy target."""
    return AtomicData(
        positions=torch.zeros(len(numbers), 3, dtype=torch.float64),
        atomic_numbers=torch.tensor(numbers, dtype=torch.long),
        energy=torch.tensor([[energy]], dtype=torch.float64),
    )


def test_fit_atomic_reference_energies_infers_elements() -> None:
    dataset = [
        Batch.from_data_list(
            [
                _structure([1, 1], 2.0),
                _structure([6], 10.0),
                _structure([1, 6], 11.0),
            ]
        )
    ]

    fit = fit_atomic_reference_energies(dataset)

    assert fit.atomic_numbers == (1, 6)
    assert fit.reference_energies == pytest.approx({1: 1.0, 6: 10.0})
    assert fit.intercept == 0.0
    assert fit.num_structures == 3
    assert fit.residual_sum_squares == pytest.approx(0.0)


def test_fit_atomic_reference_energies_supports_intercept() -> None:
    dataset = [
        Batch.from_data_list(
            [
                _structure([1], 4.0),
                _structure([6], 8.0),
                _structure([1, 6], 10.0),
            ]
        )
    ]

    fit = fit_atomic_reference_energies(dataset, fit_intercept=True)

    assert fit.reference_energies == pytest.approx({1: 2.0, 6: 6.0})
    assert fit.intercept == pytest.approx(2.0)
    assert fit.rank == 3


def test_fit_atomic_reference_energies_uses_batch_baseline() -> None:
    batch = Batch.from_data_list(
        [
            _structure([1, 1], 7.0),
            _structure([6], 15.0),
            _structure([1, 6], 16.0),
        ]
    )

    def baseline_energy_fn(batch: Batch) -> torch.Tensor:
        return torch.full((int(batch.num_graphs),), 5.0)

    fit = fit_atomic_reference_energies(
        [batch],
        atomic_numbers=[1, 6],
        baseline_energy_fn=baseline_energy_fn,
    )

    assert fit.reference_energies == pytest.approx({1: 1.0, 6: 10.0})


def test_fit_atomic_reference_energies_rejects_unexpected_subset_element() -> None:
    dataset = [Batch.from_data_list([_structure([1, 8], 3.0)])]

    with pytest.raises(ValueError, match="outside the fitted subset"):
        fit_atomic_reference_energies(dataset, atomic_numbers=[1])


def test_fit_atomic_reference_energies_requires_batches() -> None:
    with pytest.raises(TypeError, match="expects Batch items"):
        fit_atomic_reference_energies([_structure([1], 1.0)])
