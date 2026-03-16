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
"""Tests for :class:`LossComponent` and :class:`CompositeLoss`.

All tests run on CPU using :class:`MockBatch` from the test conftest,
with the segmented reduction path monkey-patched to use a pure-torch
scatter-add fallback.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from unittest.mock import patch

import pytest
import torch
from torch import Tensor

from nvalchemi.training.losses._base import CompositeLoss, LossComponent
from test.training.conftest import _make_batch

# ---------------------------------------------------------------------------
# Pure-torch fallback for _segmented_sum (avoids GPU/warp dependency)
# ---------------------------------------------------------------------------


def _cpu_segmented_sum(values: Tensor, idx: Tensor, num_segments: int) -> Tensor:
    out = torch.zeros(num_segments, device=values.device, dtype=values.dtype)
    out.scatter_add_(0, idx.long(), values)
    return out


_PATCH_TARGET = "nvalchemi.training.losses._base._segmented_sum"


# ---------------------------------------------------------------------------
# Concrete identity-error subclass for testing base logic
# ---------------------------------------------------------------------------


class _IdentityLoss(LossComponent):
    """Returns elementwise error equal to the squared difference."""

    def elementwise_error(self, pred: Tensor, target: Tensor) -> Tensor:
        return (pred - target).square()


# ---------------------------------------------------------------------------
# LossComponent tests
# ---------------------------------------------------------------------------


class TestLossComponent:
    """Verify LossComponent reduction and normalisation paths."""

    def test_abstract_elementwise_error(self) -> None:
        """Base class raises NotImplementedError."""
        base = LossComponent(name="base", pred_key="x", target_key="x")
        with pytest.raises(NotImplementedError):
            base.elementwise_error(torch.zeros(1), torch.zeros(1))

    def test_system_level_mean_reduction(self, two_graph_batch) -> None:
        """System-level loss with mean reduction over batch dim."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = _IdentityLoss(
            name="test",
            pred_key="energies",
            target_key="energies",
            level="system",
            reduction="mean",
            normalize_by_atoms=False,
        )
        with patch(_PATCH_TARGET, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # energies: pred [11,19], target [10,20] -> error [1,1] -> (B,1)
        # reshape(B,-1).sum(-1) -> [1, 1], mean -> 1.0
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(1.0))

    def test_system_level_sum_reduction(self, two_graph_batch) -> None:
        """System-level loss with sum reduction."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = _IdentityLoss(
            name="test",
            pred_key="energies",
            target_key="energies",
            level="system",
            reduction="sum",
            normalize_by_atoms=False,
        )
        with patch(_PATCH_TARGET, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # [1, 1] sum -> 2.0
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(2.0))

    def test_system_level_normalize_by_atoms(self, two_graph_batch) -> None:
        """System-level loss with atom-count normalization."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = _IdentityLoss(
            name="test",
            pred_key="energies",
            target_key="energies",
            level="system",
            reduction="mean",
            normalize_by_atoms=True,
        )
        with patch(_PATCH_TARGET, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # per_graph [1, 1], atoms [3, 2] -> normalized [1/3, 1/2]
        # mean -> (1/3 + 1/2) / 2 = 5/12
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(5.0 / 12.0))

    def test_node_level_reduction(self, two_graph_batch) -> None:
        """Node-level loss uses segmented reduction."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = _IdentityLoss(
            name="test",
            pred_key="forces",
            target_key="forces",
            level="node",
            reduction="mean",
            normalize_by_atoms=False,
        )
        with patch(_PATCH_TARGET, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # forces diff^2: each is 0.25 per component where diff=0.5, else 0
        # node 0: (0.5)^2 + 0 + 0 = 0.25
        # node 1: 0 + 0.25 + 0 = 0.25
        # node 2: 0 + 0 + 0.25 = 0.25
        # node 3: 0 + 0.25 + 0 = 0.25
        # node 4: 0 + 0 + 0.25 = 0.25
        # per_graph (scatter_add): seg 0 = 0.75, seg 1 = 0.50
        # mean -> (0.75 + 0.50) / 2 = 0.625
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(0.625))

    def test_weight_applied(self, two_graph_batch) -> None:
        """Component weight is multiplied into the result."""
        batch = two_graph_batch
        outputs = batch._outputs

        loss = _IdentityLoss(
            name="test",
            pred_key="energies",
            target_key="energies",
            level="system",
            reduction="mean",
            normalize_by_atoms=False,
            weight=3.0,
        )
        with patch(_PATCH_TARGET, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        # unweighted = 1.0, weight = 3.0 -> 3.0
        assert result is not None
        torch.testing.assert_close(result, torch.tensor(3.0))

    def test_missing_pred_key_returns_none(self, two_graph_batch) -> None:
        """Returns None when prediction key is absent."""
        batch = two_graph_batch
        outputs: OrderedDict = OrderedDict()

        loss = _IdentityLoss(name="test", pred_key="energies", target_key="energies")
        with patch(_PATCH_TARGET, side_effect=_cpu_segmented_sum):
            assert loss(batch, outputs) is None

    def test_missing_target_key_returns_none(self, two_graph_batch) -> None:
        """Returns None when target key is absent from batch."""
        batch = two_graph_batch
        outputs = OrderedDict(energies=torch.tensor([[1.0], [2.0]]))

        loss = _IdentityLoss(name="test", pred_key="energies", target_key="nonexistent")
        with patch(_PATCH_TARGET, side_effect=_cpu_segmented_sum):
            assert loss(batch, outputs) is None

    def test_system_level_on_device(self, device) -> None:
        """System-level loss runs correctly on the parametrized device."""
        energy_target = torch.tensor([[10.0], [20.0]], device=device)
        energy_pred = torch.tensor([[11.0], [19.0]], device=device)
        batch = _make_batch([3, 2], device=device, data={"energies": energy_target})
        outputs = OrderedDict(energies=energy_pred)

        loss = _IdentityLoss(
            name="test",
            pred_key="energies",
            target_key="energies",
            level="system",
            reduction="mean",
        )
        with patch(_PATCH_TARGET, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        assert result is not None
        assert str(result.device).startswith(device)
        torch.testing.assert_close(result, torch.tensor(1.0, device=device))

    def test_node_level_on_device(self, device) -> None:
        """Node-level loss runs correctly on the parametrized device."""
        forces_target = torch.zeros(5, 3, device=device)
        forces_pred = torch.full((5, 3), 0.5, device=device)
        batch = _make_batch([3, 2], device=device, data={"forces": forces_target})
        outputs = OrderedDict(forces=forces_pred)

        loss = _IdentityLoss(
            name="test",
            pred_key="forces",
            target_key="forces",
            level="node",
            reduction="mean",
        )
        with patch(_PATCH_TARGET, side_effect=_cpu_segmented_sum):
            result = loss(batch, outputs)

        assert result is not None
        assert str(result.device).startswith(device)


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------


class TestLossArithmetic:
    """Test operator overloading on LossComponent and CompositeLoss."""

    def test_add_two_components(self) -> None:
        """``a + b`` produces a CompositeLoss with two terms."""
        a = _IdentityLoss(name="a", pred_key="x", target_key="x")
        b = _IdentityLoss(name="b", pred_key="y", target_key="y")
        composite = a + b
        assert isinstance(composite, CompositeLoss)
        assert len(composite.terms) == 2

    def test_mul_produces_composite(self) -> None:
        """``0.5 * a`` produces a CompositeLoss with coeff=0.5."""
        a = _IdentityLoss(name="a", pred_key="x", target_key="x")
        composite = 0.5 * a
        assert isinstance(composite, CompositeLoss)
        assert len(composite.terms) == 1
        assert composite.terms[0][1] == 0.5

    def test_rmul_matches_mul(self) -> None:
        """``a * 2`` is the same as ``2 * a``."""
        a = _IdentityLoss(name="a", pred_key="x", target_key="x")
        c1 = a * 2.0
        c2 = 2.0 * a
        assert c1.terms[0][1] == c2.terms[0][1] == 2.0

    def test_sum_builtin(self) -> None:
        """``sum([a, b, c])`` produces a CompositeLoss with 3 terms."""
        a = _IdentityLoss(name="a", pred_key="x", target_key="x")
        b = _IdentityLoss(name="b", pred_key="y", target_key="y")
        c = _IdentityLoss(name="c", pred_key="z", target_key="z")
        composite = sum([a, b, c])
        assert isinstance(composite, CompositeLoss)
        assert len(composite.terms) == 3

    def test_composite_add_component(self) -> None:
        """``CompositeLoss + LossComponent`` extends terms."""
        a = _IdentityLoss(name="a", pred_key="x", target_key="x")
        b = _IdentityLoss(name="b", pred_key="y", target_key="y")
        c = _IdentityLoss(name="c", pred_key="z", target_key="z")
        composite = (a + b) + c
        assert isinstance(composite, CompositeLoss)
        assert len(composite.terms) == 3

    def test_composite_add_composite(self) -> None:
        """``CompositeLoss + CompositeLoss`` merges terms."""
        a = _IdentityLoss(name="a", pred_key="x", target_key="x")
        b = _IdentityLoss(name="b", pred_key="y", target_key="y")
        c = _IdentityLoss(name="c", pred_key="z", target_key="z")
        d = _IdentityLoss(name="d", pred_key="w", target_key="w")
        composite = (a + b) + (c + d)
        assert isinstance(composite, CompositeLoss)
        assert len(composite.terms) == 4

    def test_weighted_sum(self) -> None:
        """``0.5 * a + 1.0 * b`` creates a 2-term composite with correct coefficients."""
        a = _IdentityLoss(name="a", pred_key="x", target_key="x")
        b = _IdentityLoss(name="b", pred_key="y", target_key="y")
        composite = 0.5 * a + 1.0 * b
        assert isinstance(composite, CompositeLoss)
        coeffs = [c for _, c in composite.terms]
        assert coeffs == [0.5, 1.0]


# ---------------------------------------------------------------------------
# CompositeLoss forward
# ---------------------------------------------------------------------------


class TestCompositeLoss:
    """Tests for CompositeLoss.forward()."""

    def test_forward_aggregates(self, two_graph_batch) -> None:
        """Composite sums weighted component losses."""
        batch = two_graph_batch
        outputs = batch._outputs

        energy = _IdentityLoss(
            name="energy",
            pred_key="energies",
            target_key="energies",
            level="system",
            reduction="mean",
        )
        force = _IdentityLoss(
            name="force",
            pred_key="forces",
            target_key="forces",
            level="node",
            reduction="mean",
        )
        composite = CompositeLoss(terms=[(energy, 1.0), (force, 10.0)])

        with patch(_PATCH_TARGET, side_effect=_cpu_segmented_sum):
            total, per_term = composite(batch, outputs)

        assert "energy" in per_term
        assert "force" in per_term
        assert per_term["energy"] is not None
        assert per_term["force"] is not None
        expected = per_term["energy"] * 1.0 + per_term["force"] * 10.0
        torch.testing.assert_close(total, expected)

    def test_missing_key_excluded_from_total(self, two_graph_batch) -> None:
        """Terms with missing keys return None and are excluded from the total."""
        batch = two_graph_batch
        outputs = batch._outputs

        energy = _IdentityLoss(
            name="energy",
            pred_key="energies",
            target_key="energies",
            level="system",
            reduction="mean",
        )
        missing = _IdentityLoss(
            name="missing",
            pred_key="nonexistent",
            target_key="nonexistent",
        )
        composite = CompositeLoss(terms=[(energy, 1.0), (missing, 1.0)])

        with patch(_PATCH_TARGET, side_effect=_cpu_segmented_sum):
            total, per_term = composite(batch, outputs)

        assert per_term["missing"] is None
        assert per_term["energy"] is not None
        torch.testing.assert_close(total, per_term["energy"])

    def test_all_terms_missing_returns_zero_with_warning(self, two_graph_batch) -> None:
        """When all terms skip, returns zero loss and emits a warning."""
        batch = two_graph_batch
        outputs: OrderedDict = OrderedDict()

        m1 = _IdentityLoss(name="m1", pred_key="a", target_key="a")
        m2 = _IdentityLoss(name="m2", pred_key="b", target_key="b")
        composite = CompositeLoss(terms=[(m1, 1.0), (m2, 1.0)])

        with patch(_PATCH_TARGET, side_effect=_cpu_segmented_sum):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                total, per_term = composite(batch, outputs)

        assert total.item() == 0.0
        assert per_term["m1"] is None
        assert per_term["m2"] is None
        assert any("all terms returned None" in str(x.message) for x in w)

    def test_registers_submodules(self) -> None:
        """Components are registered as nn.Module children."""
        a = _IdentityLoss(name="a", pred_key="x", target_key="x")
        b = _IdentityLoss(name="b", pred_key="y", target_key="y")
        composite = CompositeLoss(terms=[(a, 1.0), (b, 2.0)])
        children = list(composite._components)
        assert a in children
        assert b in children
