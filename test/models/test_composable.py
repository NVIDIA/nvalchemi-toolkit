# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from torch import nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import ComposableModelWrapper, DemoModelWrapper
from nvalchemi.models.base import BaseModelMixin, ModelConfig


def _sum_per_graph(values: torch.Tensor, batch_idx: torch.Tensor) -> torch.Tensor:
    """Sum node values into per-graph shape ``[B, 1]``."""

    num_graphs = int(batch_idx.max().item()) + 1 if batch_idx.numel() else 1
    result = torch.zeros(
        (num_graphs, 1),
        dtype=values.dtype,
        device=values.device,
    )
    result.scatter_add_(0, batch_idx.unsqueeze(-1), values)
    return result


class _ChargeModel(BaseModelMixin, nn.Module):
    """Autograd-connected model that emits charges and energy."""

    spec = ModelConfig(
        required_inputs=frozenset({"positions", "atomic_numbers"}),
        optional_inputs=frozenset({"batch"}),
        outputs=frozenset({"energies", "charges"}),
        additive_outputs=frozenset({"energies"}),
        use_autograd=True,
    )

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        positions = data["positions"]
        batch_idx = data.get(
            "batch",
            torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device),
        ).long()
        charges = positions.sum(dim=-1, keepdim=True)
        energies = _sum_per_graph(
            (positions.pow(2).sum(dim=-1, keepdim=True) * 0.25), batch_idx
        )
        return {"energies": energies, "charges": charges}


class _ElectroModel(BaseModelMixin, nn.Module):
    """Autograd-connected downstream model that consumes node charges."""

    spec = ModelConfig(
        required_inputs=frozenset({"positions", "node_charges"}),
        optional_inputs=frozenset({"batch"}),
        outputs=frozenset({"energies"}),
        additive_outputs=frozenset({"energies"}),
        use_autograd=True,
    )

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        positions = data["positions"]
        charges = data["node_charges"]
        batch_idx = data.get(
            "batch",
            torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device),
        ).long()
        node_energy = charges * positions[:, :1]
        return {"energies": _sum_per_graph(node_energy, batch_idx)}


class _DirectCorrectionModel(BaseModelMixin, nn.Module):
    """Direct-output correction model for hybrid composition tests."""

    spec = ModelConfig(
        required_inputs=frozenset({"positions"}),
        optional_inputs=frozenset({"batch"}),
        outputs=frozenset({"energies", "forces"}),
        additive_outputs=frozenset({"energies", "forces"}),
        use_autograd=False,
    )

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        positions = data["positions"]
        batch_idx = data.get(
            "batch",
            torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device),
        ).long()
        node_energy = positions[:, 1:2].abs() * 0.1
        return {
            "energies": _sum_per_graph(node_energy, batch_idx),
            "forces": torch.full_like(positions, 0.05),
        }


def _make_batch() -> Batch:
    """Build a small two-system batch for composable tests."""

    data_list = [
        AtomicData(
            atomic_numbers=torch.tensor([6, 1], dtype=torch.long),
            positions=torch.tensor(
                [[0.2, 0.1, 0.0], [0.0, -0.3, 0.4]], dtype=torch.float32
            ),
        ),
        AtomicData(
            atomic_numbers=torch.tensor([8, 1], dtype=torch.long),
            positions=torch.tensor(
                [[0.5, 0.2, -0.1], [-0.2, 0.4, 0.0]], dtype=torch.float32
            ),
        ),
    ]
    return Batch.from_data_list(data_list)


def test_additive_composition_with_plus_operator() -> None:
    """Two direct models should compose additively through ``+``."""

    batch = _make_batch()
    model_a = DemoModelWrapper(hidden_dim=8)
    model_b = DemoModelWrapper(hidden_dim=8)

    composed = model_a + model_b

    assert isinstance(composed, ComposableModelWrapper)
    expected_a = model_a(batch)
    expected_b = model_b(batch)
    actual = composed(batch, compute={"energies", "forces"})

    assert torch.allclose(
        actual["energies"], expected_a["energies"] + expected_b["energies"]
    )
    assert torch.allclose(actual["forces"], expected_a["forces"] + expected_b["forces"])


def test_wire_output_supports_dependent_autograd_chain() -> None:
    """Dependent composition should work with ``wire_output``."""

    batch = _make_batch()
    aimnet = _ChargeModel()
    ewald = _ElectroModel()

    group = aimnet + ewald
    group.wire_output(aimnet, ewald, {"node_charges": "charges"})

    outputs = group(batch, compute={"energies", "forces"})

    assert outputs["energies"].shape == (batch.num_graphs, 1)
    assert outputs["forces"].shape == batch.positions.shape
    assert torch.isfinite(outputs["forces"]).all()


def test_nested_composition_preserves_wiring_for_hybrid_case() -> None:
    """``(aimnet + ewald) + d3`` should preserve the dependent chain."""

    batch = _make_batch()
    aimnet = _ChargeModel()
    ewald = _ElectroModel()
    d3 = _DirectCorrectionModel()

    group = aimnet + ewald
    group.wire_output(aimnet, ewald, {"node_charges": "charges"})
    calc = group + d3

    outputs = calc(batch, compute={"energies", "forces"})

    assert len(calc.models) == 3
    assert outputs["energies"].shape == (batch.num_graphs, 1)
    assert outputs["forces"].shape == batch.positions.shape
    assert torch.isfinite(outputs["forces"]).all()
