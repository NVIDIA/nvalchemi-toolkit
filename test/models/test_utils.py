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
"""Tests for the autograd helpers in :mod:`nvalchemi.models._utils`."""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import hessian_vector_product
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

_FD_STEP = 1e-4
"""Displacement of the central difference the product is checked against."""


def _make_batch() -> Batch:
    """Return a two-graph batch of three and two atoms."""
    torch.manual_seed(0)
    return Batch.from_data_list(
        [
            AtomicData(
                positions=torch.randn(3, 3), atomic_numbers=torch.tensor([6, 6, 8])
            ),
            AtomicData(
                positions=torch.randn(2, 3), atomic_numbers=torch.tensor([1, 1])
            ),
        ]
    )


def _make_energy_model() -> DemoModelWrapper:
    """Return a demo model narrowed to its energy, so no forces consume the graph."""
    torch.manual_seed(0)
    model = DemoModelWrapper(DemoModel())
    model.set_config("active_outputs", {"energy"})
    return model


def _energy_gradient(
    model: DemoModelWrapper, batch: Batch, positions: torch.Tensor
) -> torch.Tensor:
    """Return the gradient of *model*'s energy at *positions*."""
    batch.positions = positions.clone().requires_grad_(True)
    energy = model(batch)["energy"]
    return torch.autograd.grad(energy.sum(), batch.positions)[0]


class TestHessianVectorProduct:
    """The double-backward estimator of an energy Hessian's product with a probe."""

    def test_product_matches_a_finite_difference_of_the_energy_gradient(self) -> None:
        """Autograd curvature reproduces a central difference of the energy gradient."""
        model, batch = _make_energy_model(), _make_batch()
        origin = batch.positions.detach().clone()
        probe = torch.randn_like(origin)
        batch.positions = origin.clone().requires_grad_(True)
        product = hessian_vector_product(model(batch)["energy"], batch.positions, probe)
        forward = _energy_gradient(model, batch, origin + _FD_STEP * probe)
        backward = _energy_gradient(model, batch, origin - _FD_STEP * probe)
        reference = (forward - backward) / (2.0 * _FD_STEP)
        torch.testing.assert_close(product, reference, atol=1e-3, rtol=1e-2)

    def test_product_is_block_diagonal_over_graphs(self) -> None:
        """A probe on one graph's atoms leaves the other graph's product zero."""
        model, batch = _make_energy_model(), _make_batch()
        batch.positions = batch.positions.detach().clone().requires_grad_(True)
        probe = torch.ones_like(batch.positions)
        probe[batch.batch_idx == 1] = 0.0
        product = hessian_vector_product(model(batch)["energy"], batch.positions, probe)
        assert bool((product[batch.batch_idx == 1] == 0.0).all())
        assert bool((product[batch.batch_idx == 0] != 0.0).any())

    def test_product_of_a_quadratic_energy_is_the_constant_hessian(self) -> None:
        """For ``E = c |r|^2 / 2`` the product is ``c v``, exactly."""
        positions = torch.randn(4, 3, requires_grad=True)
        probe = torch.randn(4, 3)
        energy = (3.0 * positions.pow(2).sum()).reshape(1, 1) / 2.0
        product = hessian_vector_product(energy, positions, probe)
        torch.testing.assert_close(product, 3.0 * probe)

    def test_linear_energy_gives_a_zero_product(self) -> None:
        """A position-independent gradient is a zero Hessian, not a missing graph."""
        positions = torch.randn(4, 3, requires_grad=True)
        energy = (3.0 * positions).sum().reshape(1, 1)
        product = hessian_vector_product(energy, positions, torch.ones(4, 3))
        torch.testing.assert_close(product, torch.zeros(4, 3))

    def test_zero_product_stays_attached_when_a_graph_is_requested(self) -> None:
        """A model whose curvature vanishes still gives a loss a graph to backpropagate."""
        positions = torch.randn(4, 3, requires_grad=True)
        weight = torch.tensor(3.0, requires_grad=True)
        energy = (weight * positions).sum().reshape(1, 1)
        product = hessian_vector_product(
            energy, positions, torch.ones(4, 3), create_graph=True
        )
        product.pow(2).sum().backward()
        assert product.requires_grad
        torch.testing.assert_close(weight.grad, torch.zeros_like(weight))

    def test_detached_energy_is_reported_as_a_missing_graph(self) -> None:
        """Differentiating an energy with no graph names what the estimator needs."""
        positions = torch.randn(4, 3, requires_grad=True)
        with pytest.raises(RuntimeError, match="twice differentiable"):
            hessian_vector_product(torch.zeros(1, 1), positions, torch.randn(4, 3))

    def test_created_graph_keeps_the_product_differentiable(self) -> None:
        """``create_graph=True`` is what lets a loss backpropagate through it."""
        positions = torch.randn(4, 3, requires_grad=True)
        weight = torch.tensor(2.0, requires_grad=True)
        energy = (weight * positions.pow(2).sum()).reshape(1, 1)
        product = hessian_vector_product(
            energy, positions, torch.ones(4, 3), create_graph=True
        )
        product.sum().backward()
        assert weight.grad is not None
