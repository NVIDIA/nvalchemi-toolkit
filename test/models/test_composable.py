# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import torch
from torch import nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import ComposableModelWrapper, DemoModelWrapper
from nvalchemi.models.base import BaseModelMixin, ModelConfig, NeighborConfig


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
    """Autograd-connected model that emits canonical node charges and energy."""

    spec = ModelConfig(
        required_inputs=frozenset({"positions", "atomic_numbers"}),
        optional_inputs=frozenset({"batch"}),
        outputs=frozenset({"energies", "node_charges"}),
        additive_outputs=frozenset({"energies"}),
        use_autograd=True,
        autograd_outputs=frozenset({"node_charges"}),
    )

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        positions = data["positions"]
        batch_idx = data.get(
            "batch",
            torch.zeros(positions.shape[0], dtype=torch.long, device=positions.device),
        ).long()
        node_charges = positions.sum(dim=-1, keepdim=True)
        energies = _sum_per_graph(
            (positions.pow(2).sum(dim=-1, keepdim=True) * 0.25), batch_idx
        )
        return {"energies": energies, "node_charges": node_charges}

    def __repr__(self) -> str:
        return "AIMNet2Wrapper(model='aimnet2')"


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


class _AutogradExternalModel(BaseModelMixin, nn.Module):
    """Autograd model with an external neighbor requirement for repr tests."""

    spec = ModelConfig(
        required_inputs=frozenset(
            {"positions", "atomic_numbers", "edge_index", "unit_shifts"}
        ),
        optional_inputs=frozenset({"cell", "pbc", "batch"}),
        outputs=frozenset({"energies"}),
        additive_outputs=frozenset({"energies"}),
        use_autograd=True,
        neighbor_config=NeighborConfig(
            source="external",
            cutoff=6.0,
            format="coo",
            half_list=False,
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._name = "mace"

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "MACEWrapper(model='medium-0b2')"


class _DirectExternalModel(BaseModelMixin, nn.Module):
    """Direct model with an external neighbor requirement for repr tests."""

    spec = ModelConfig(
        required_inputs=frozenset({"positions", "atomic_numbers", "neighbor_matrix"}),
        optional_inputs=frozenset(
            {"batch", "cell", "pbc", "neighbor_shifts", "num_neighbors"}
        ),
        outputs=frozenset({"energies", "forces"}),
        optional_outputs={"stresses": frozenset({"cell", "pbc"})},
        additive_outputs=frozenset({"energies", "forces", "stresses"}),
        use_autograd=False,
        neighbor_config=NeighborConfig(
            source="external",
            cutoff=15.0,
            format="matrix",
            half_list=False,
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._name = "dftd3"

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "DFTD3ModelWrapper(functional='pbe')"


class _ElectroExternalModel(BaseModelMixin, nn.Module):
    """Electrostatics-like external-neighbor consumer for repr tests."""

    spec = ModelConfig(
        required_inputs=frozenset({"positions", "node_charges", "neighbor_matrix"}),
        optional_inputs=frozenset({"cell", "pbc", "neighbor_shifts", "num_neighbors"}),
        outputs=frozenset({"energies"}),
        optional_outputs={"stresses": frozenset({"cell", "pbc"})},
        additive_outputs=frozenset({"energies", "stresses"}),
        use_autograd=True,
        neighbor_config=NeighborConfig(
            source="external",
            cutoff=12.0,
            format="matrix",
            half_list=False,
        ),
    )

    def __init__(self) -> None:
        super().__init__()
        self._name = "pme"

    def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def __repr__(self) -> str:
        return "PMEModelWrapper(cutoff=12.0)"


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


def test_canonical_outputs_support_dependent_autograd_chain_without_wiring() -> None:
    """Canonical outputs should satisfy dependent autograd chains directly."""

    batch = _make_batch()
    aimnet = _ChargeModel()
    ewald = _ElectroModel()

    group = aimnet + ewald

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
    calc = group + d3

    outputs = calc(batch, compute={"energies", "forces"})

    assert len(calc.models) == 3
    assert outputs["energies"].shape == (batch.num_graphs, 1)
    assert outputs["forces"].shape == batch.positions.shape
    assert torch.isfinite(outputs["forces"]).all()


def test_wire_output_supports_explicit_renaming() -> None:
    """Explicit wiring should still support non-canonical rename cases."""

    class _RenamedChargeModel(BaseModelMixin, nn.Module):
        spec = ModelConfig(
            required_inputs=frozenset({"positions", "atomic_numbers"}),
            optional_inputs=frozenset({"batch"}),
            outputs=frozenset({"energies", "charges"}),
            additive_outputs=frozenset({"energies"}),
            use_autograd=True,
            autograd_outputs=frozenset({"charges"}),
        )

        def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            positions = data["positions"]
            batch_idx = data.get(
                "batch",
                torch.zeros(
                    positions.shape[0],
                    dtype=torch.long,
                    device=positions.device,
                ),
            ).long()
            charges = positions.sum(dim=-1, keepdim=True)
            energies = _sum_per_graph(
                (positions.pow(2).sum(dim=-1, keepdim=True) * 0.25), batch_idx
            )
            return {"energies": energies, "charges": charges}

    batch = _make_batch()
    source = _RenamedChargeModel()
    target = _ElectroModel()

    group = source + target
    group.wire_output(source, target, {"node_charges": "charges"})

    outputs = group(batch, compute={"energies", "forces"})

    assert outputs["energies"].shape == (batch.num_graphs, 1)
    assert outputs["forces"].shape == batch.positions.shape
    assert torch.isfinite(outputs["forces"]).all()


def test_single_model_composable_repr_is_zero_indexed_and_side_effect_free() -> None:
    """Single-model repr should start from zero and avoid pipeline caching."""

    calc = ComposableModelWrapper(DemoModelWrapper())

    assert calc._compiled_pipeline is None
    rendered = repr(calc)

    assert "[0] DemoModelWrapper()" in rendered
    assert "inputs: atomic_numbers, positions, batch?" in rendered
    assert "outputs: energies, forces" in rendered
    assert calc._compiled_pipeline is None


def test_repr_shows_external_neighbor_and_derivative_steps_for_hybrid_pipeline() -> (
    None
):
    """Hybrid repr should show external builders and the derivative boundary."""

    calc = ComposableModelWrapper(_AutogradExternalModel(), _DirectExternalModel())

    rendered = repr(calc)

    assert "[0] NeighborListBuilder(cutoff=6.0, format='matrix')" in rendered
    assert "[1] MACEWrapper(model='medium-0b2')" in rendered
    assert "[2] DerivativeStep(forces=True, stresses=True)" in rendered
    assert "grad disabled" in rendered
    assert "[3] NeighborListBuilder(cutoff=15.0, format='matrix')" in rendered
    assert "[4] DFTD3ModelWrapper(functional='pbe')" in rendered


def test_repr_shows_canonical_charge_flow_without_wiring() -> None:
    """Canonical charge flow should not require or display an explicit wire."""

    calc = ComposableModelWrapper(_ChargeModel(), _ElectroExternalModel())

    rendered = repr(calc)

    assert "[0] AIMNet2Wrapper(model='aimnet2')" in rendered
    assert "node_charges*" in rendered
    assert "wires:" not in rendered
    assert "[1] NeighborListBuilder(cutoff=12.0, format='matrix')" in rendered
    assert "[2] PMEModelWrapper(cutoff=12.0)" in rendered


def test_repr_shows_explicit_rename_wire_for_noncanonical_source() -> None:
    """Explicit rename wiring should appear in the composite repr."""

    class _RenamedChargeModel(BaseModelMixin, nn.Module):
        spec = ModelConfig(
            required_inputs=frozenset({"positions", "atomic_numbers"}),
            optional_inputs=frozenset({"cell", "pbc"}),
            outputs=frozenset({"energies", "charges"}),
            additive_outputs=frozenset({"energies"}),
            use_autograd=True,
            autograd_outputs=frozenset({"charges"}),
        )

        def __init__(self) -> None:
            super().__init__()
            self._name = "charge_net"

        def forward(self, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            raise NotImplementedError

        def __repr__(self) -> str:
            return "ChargeNetWrapper()"

    source = _RenamedChargeModel()
    target = _ElectroExternalModel()
    calc = ComposableModelWrapper(source, target)
    calc.wire_output(source, target, {"node_charges": "charges"})

    rendered = repr(calc)

    assert "[0] ChargeNetWrapper()" in rendered
    assert "[2] PMEModelWrapper(cutoff=12.0)" in rendered
    assert "wires: node_charges <- charge_net.charges" in rendered
