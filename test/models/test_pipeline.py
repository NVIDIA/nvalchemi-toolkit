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
"""Comprehensive tests for PipelineModelWrapper composition patterns.

Tests all composition cases from the proposal:
- Independent sum
- Dependent chain with autograd forces
- Feeder model
- Force correction
- Three-model hybrid
- Fan-out (auto-wired and with wire)
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from torch import nn

from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelCard,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
from nvalchemi.models.pipeline import (
    PipelineGroup,
    PipelineModelWrapper,
    PipelineStep,
)

# ---------------------------------------------------------------------------
# Mock models for pipeline composition tests
# ---------------------------------------------------------------------------


class MockEnergyForceModel(nn.Module, BaseModelMixin):
    """Mock model that returns fixed energies and forces (analytical)."""

    def __init__(self, energy: float = 1.0, force_val: float = 0.5) -> None:
        super().__init__()
        self._energy = energy
        self._force_val = force_val
        self.model_config = ModelConfig(compute={"energies", "forces"})
        self._card = ModelCard(
            outputs={"energies", "forces"},
            autograd_outputs=set(),
            needs_pbc=False,
        )

    @property
    def model_card(self) -> ModelCard:
        return self._card

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        B = data.num_graphs if isinstance(data, Batch) else 1
        N = data.positions.shape[0]
        return OrderedDict(
            energies=torch.full((B, 1), self._energy, dtype=data.positions.dtype),
            forces=torch.full((N, 3), self._force_val, dtype=data.positions.dtype),
        )


class MockAutogradEnergyModel(nn.Module, BaseModelMixin):
    """Mock model that returns energies computed from positions (autograd-capable)."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self._scale = scale
        self.model_config = ModelConfig(compute={"energies"})
        self._card = ModelCard(
            outputs={"energies"},
            autograd_outputs={"forces"},
            autograd_inputs={"positions"},
            needs_pbc=False,
        )

    @property
    def model_card(self) -> ModelCard:
        return self._card

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        positions = data.positions
        B = data.num_graphs if isinstance(data, Batch) else 1
        batch = (
            data.batch
            if isinstance(data, Batch)
            else torch.zeros(positions.shape[0], dtype=torch.long)
        )
        per_atom = self._scale * (positions**2).sum(dim=-1)
        energies = torch.zeros(B, 1, dtype=positions.dtype, device=positions.device)
        energies.scatter_add_(0, batch.unsqueeze(-1), per_atom.unsqueeze(-1))
        return OrderedDict(energies=energies)


class MockChargeEnergyModel(nn.Module, BaseModelMixin):
    """Mock model that outputs charges and energies (position-dependent for autograd)."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(compute={"energies", "charges"})
        self._card = ModelCard(
            outputs={"energies", "charges"},
            autograd_outputs=set(),
            needs_pbc=False,
        )

    @property
    def model_card(self) -> ModelCard:
        return self._card

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        positions = data.positions
        B = data.num_graphs if isinstance(data, Batch) else 1
        N = positions.shape[0]
        batch = (
            data.batch if isinstance(data, Batch) else torch.zeros(N, dtype=torch.long)
        )
        # Position-dependent energy so autograd can differentiate
        per_atom = (positions**2).sum(dim=-1)
        energies = torch.zeros(B, 1, dtype=positions.dtype, device=positions.device)
        energies.scatter_add_(0, batch.unsqueeze(-1), per_atom.unsqueeze(-1))
        return OrderedDict(
            energies=energies,
            charges=torch.ones(N, dtype=positions.dtype) * 0.5,
        )


class MockChargeOnlyModel(nn.Module, BaseModelMixin):
    """Mock model that only outputs charges (feeder)."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(compute={"charges"})
        self._card = ModelCard(
            outputs={"charges"},
            autograd_outputs=set(),
            needs_pbc=False,
        )

    @property
    def model_card(self) -> ModelCard:
        return self._card

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        N = data.positions.shape[0]
        return OrderedDict(
            charges=torch.ones(N, dtype=data.positions.dtype) * 0.3,
        )


class MockElectrostaticsModel(nn.Module, BaseModelMixin):
    """Mock model that takes node_charges as input and outputs energies."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(compute={"energies"})
        self._card = ModelCard(
            outputs={"energies"},
            inputs={"node_charges"},
            autograd_outputs=set(),
            needs_pbc=False,
        )

    @property
    def model_card(self) -> ModelCard:
        return self._card

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        B = data.num_graphs if isinstance(data, Batch) else 1
        charges = getattr(data, "node_charges", None)
        if charges is None:
            raise RuntimeError("node_charges not found on data")
        # Position-dependent energy for autograd differentiation
        batch = (
            data.batch
            if isinstance(data, Batch)
            else torch.zeros(charges.shape[0], dtype=torch.long)
        )
        per_atom = charges * (data.positions**2).sum(dim=-1)
        energies = torch.zeros(
            B, 1, dtype=data.positions.dtype, device=data.positions.device
        )
        energies.scatter_add_(0, batch.unsqueeze(-1), per_atom.unsqueeze(-1))
        return OrderedDict(energies=energies)


class MockForceOnlyModel(nn.Module, BaseModelMixin):
    """Mock model that only outputs forces (force corrector)."""

    def __init__(self, force_val: float = 0.1) -> None:
        super().__init__()
        self._force_val = force_val
        self.model_config = ModelConfig(compute={"forces"})
        self._card = ModelCard(
            outputs={"forces"},
            autograd_outputs=set(),
            needs_pbc=False,
        )

    @property
    def model_card(self) -> ModelCard:
        return self._card

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        N = data.positions.shape[0]
        return OrderedDict(
            forces=torch.full((N, 3), self._force_val, dtype=data.positions.dtype),
        )


class MockMultiOutputModel(nn.Module, BaseModelMixin):
    """Mock model that outputs energies + node_charges + node_spin."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            compute={"energies", "node_charges", "node_spin"}
        )
        self._card = ModelCard(
            outputs={"energies", "node_charges", "node_spin"},
            autograd_outputs=set(),
            needs_pbc=False,
        )

    @property
    def model_card(self) -> ModelCard:
        return self._card

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        B = data.num_graphs if isinstance(data, Batch) else 1
        N = data.positions.shape[0]
        return OrderedDict(
            energies=torch.ones(B, 1, dtype=data.positions.dtype),
            node_charges=torch.ones(N, dtype=data.positions.dtype) * 0.5,
            node_spin=torch.ones(N, dtype=data.positions.dtype) * 0.1,
        )


class MockSpinModel(nn.Module, BaseModelMixin):
    """Mock model that takes node_spin as input and outputs energies."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(compute={"energies"})
        self._card = ModelCard(
            outputs={"energies"},
            inputs={"node_spin"},
            autograd_outputs=set(),
            needs_pbc=False,
        )

    @property
    def model_card(self) -> ModelCard:
        return self._card

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        B = data.num_graphs if isinstance(data, Batch) else 1
        spin = getattr(data, "node_spin", None)
        if spin is None:
            raise RuntimeError("node_spin not found on data")
        batch = (
            data.batch
            if isinstance(data, Batch)
            else torch.zeros(spin.shape[0], dtype=torch.long)
        )
        per_atom = spin**2
        energies = torch.zeros(
            B, 1, dtype=data.positions.dtype, device=data.positions.device
        )
        energies.scatter_add_(0, batch.unsqueeze(-1), per_atom.unsqueeze(-1))
        return OrderedDict(energies=energies)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_batch():
    """A minimal 2-system batch."""
    data1 = AtomicData(
        positions=torch.randn(3, 3),
        atomic_numbers=torch.tensor([6, 6, 8]),
        forces=torch.zeros(3, 3),
        energies=torch.zeros(1, 1),
    )
    data2 = AtomicData(
        positions=torch.randn(2, 3),
        atomic_numbers=torch.tensor([1, 1]),
        forces=torch.zeros(2, 3),
        energies=torch.zeros(1, 1),
    )
    return Batch.from_data_list([data1, data2])


# ===========================================================================
# PipelineStep / PipelineGroup tests
# ===========================================================================


class TestPipelineStep:
    def test_default_wire(self):
        m = MockEnergyForceModel()
        step = PipelineStep(model=m)
        assert step.wire == {}

    def test_custom_wire(self):
        m = MockChargeEnergyModel()
        step = PipelineStep(model=m, wire={"charges": "node_charges"})
        assert step.wire == {"charges": "node_charges"}


class TestPipelineGroup:
    def test_default_forces_direct(self):
        group = PipelineGroup(steps=[MockEnergyForceModel()])
        assert group.forces == "direct"

    def test_autograd_forces(self):
        group = PipelineGroup(
            steps=[MockAutogradEnergyModel()],
            forces="autograd",
        )
        assert group.forces == "autograd"


# ===========================================================================
# PipelineModelWrapper composition cases
# ===========================================================================


class TestPipelineConstruction:
    def test_bare_model_normalization(self):
        """Bare models are normalized to PipelineStep."""
        m = MockEnergyForceModel()
        pipe = PipelineModelWrapper(groups=[PipelineGroup(steps=[m])])
        assert len(pipe.groups) == 1
        assert isinstance(pipe.groups[0].steps[0], PipelineStep)

    def test_model_card_synthesis(self):
        a = MockEnergyForceModel()
        b = MockForceOnlyModel()
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[a]),
                PipelineGroup(steps=[b]),
            ]
        )
        card = pipe.model_card
        assert "energies" in card.outputs
        assert "forces" in card.outputs

    def test_not_implemented_methods(self):
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[MockEnergyForceModel()])]
        )
        with pytest.raises(NotImplementedError):
            pipe.compute_embeddings(None)
        with pytest.raises(NotImplementedError):
            pipe.export_model(None)


class TestPipelineIndependentSum:
    """Case 1: Two models predicting energies+forces; pipeline sums both."""

    def test_energies_summed(self, simple_batch):
        a = MockEnergyForceModel(energy=1.0, force_val=0.5)
        b = MockEnergyForceModel(energy=2.0, force_val=0.3)
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[a]),
                PipelineGroup(steps=[b]),
            ]
        )
        out = pipe(simple_batch)
        dtype = simple_batch.positions.dtype
        torch.testing.assert_close(
            out["energies"],
            torch.full((2, 1), 3.0, dtype=dtype),
        )

    def test_forces_summed(self, simple_batch):
        a = MockEnergyForceModel(energy=1.0, force_val=0.5)
        b = MockEnergyForceModel(energy=2.0, force_val=0.3)
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[a]),
                PipelineGroup(steps=[b]),
            ]
        )
        out = pipe(simple_batch)
        dtype = simple_batch.positions.dtype
        torch.testing.assert_close(
            out["forces"],
            torch.full((5, 3), 0.8, dtype=dtype),
        )


class TestPipelineAutogradGroup:
    """Case 2: Autograd group computes forces via shared differentiation."""

    def test_autograd_forces_nonzero(self, simple_batch):
        a = MockAutogradEnergyModel(scale=1.0)
        b = MockAutogradEnergyModel(scale=2.0)
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[a, b], forces="autograd"),
            ]
        )
        out = pipe(simple_batch)
        assert out["forces"].abs().sum() > 0
        assert out["energies"] is not None

    def test_autograd_disables_sub_model_forces(self, simple_batch):
        """Autograd group removes forces from sub-model compute."""
        m = MockAutogradEnergyModel()
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[m], forces="autograd"),
            ]
        )
        # After construction, sub-model should not have forces in compute
        assert "forces" not in pipe.groups[0].steps[0].model.model_config.compute


class TestPipelineDependentAutograd:
    """Case 2b: A predicts charges+energy, B uses charges for energy.
    Forces backprop through both via autograd."""

    def test_wired_charges(self, simple_batch):
        a = MockChargeEnergyModel()
        b = MockElectrostaticsModel()
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[
                        PipelineStep(a, wire={"charges": "node_charges"}),
                        b,
                    ],
                    forces="autograd",
                ),
            ]
        )
        out = pipe(simple_batch)
        assert out["energies"] is not None
        # Forces should be non-zero (autograd through position -> charges -> energy)
        # Note: charges are constant (0.5) regardless of positions,
        # so only model A's energy contributes to forces
        assert out["forces"] is not None


class TestPipelineFeederAutograd:
    """Case 3: A only predicts charges, B uses them for energy."""

    def test_feeder_produces_energy(self, simple_batch):
        a = MockChargeOnlyModel()
        b = MockElectrostaticsModel()
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[
                        PipelineStep(a, wire={"charges": "node_charges"}),
                        b,
                    ],
                    forces="autograd",
                ),
            ]
        )
        out = pipe(simple_batch)
        assert out["energies"] is not None


class TestPipelineForceCorrection:
    """Case 4: A predicts energies+forces, B adds force correction."""

    def test_force_correction_summed(self, simple_batch):
        a = MockEnergyForceModel(energy=1.0, force_val=0.5)
        b = MockForceOnlyModel(force_val=0.1)
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[a]),
                PipelineGroup(steps=[b]),
            ]
        )
        out = pipe(simple_batch)
        dtype = simple_batch.positions.dtype
        # Forces = A.forces + B.forces = 0.5 + 0.1 = 0.6
        torch.testing.assert_close(
            out["forces"],
            torch.full((5, 3), 0.6, dtype=dtype),
        )
        # Energies = A.energies only
        torch.testing.assert_close(
            out["energies"],
            torch.full((2, 1), 1.0, dtype=dtype),
        )


class TestPipelineThreeModelHybrid:
    """Case 5: autograd group + direct group."""

    def test_hybrid_forces(self, simple_batch):
        autograd_model = MockAutogradEnergyModel(scale=1.0)
        direct_model = MockEnergyForceModel(energy=0.5, force_val=0.1)
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[autograd_model], forces="autograd"),
                PipelineGroup(steps=[direct_model], forces="direct"),
            ]
        )
        out = pipe(simple_batch)
        # Total energy = autograd_energy + 0.5
        assert out["energies"] is not None
        # Forces = autograd(-dE/dr) + 0.1
        assert out["forces"] is not None
        assert out["forces"].abs().sum() > 0


class TestPipelineFanoutAutoWired:
    """Case 6: A outputs node_charges + node_spin; B and C consume them."""

    def test_auto_wired_fanout(self, simple_batch):
        a = MockMultiOutputModel()
        b = MockElectrostaticsModel()
        c = MockSpinModel()
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[a, b, c], forces="direct"),
            ]
        )
        out = pipe(simple_batch)
        assert out["energies"] is not None


class TestPipelineModelCardSynthesis:
    """Tests for synthesized model card from sub-models."""

    def test_max_cutoff_neighbor_config(self):
        class _SmallCutoff(MockEnergyForceModel):
            @property
            def model_card(self):
                return ModelCard(
                    outputs={"energies", "forces"},
                    needs_pbc=False,
                    neighbor_config=NeighborConfig(cutoff=5.0),
                )

        class _LargeCutoff(MockEnergyForceModel):
            @property
            def model_card(self):
                return ModelCard(
                    outputs={"energies", "forces"},
                    needs_pbc=False,
                    neighbor_config=NeighborConfig(cutoff=10.0),
                )

        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[_SmallCutoff(), _LargeCutoff()]),
            ]
        )
        assert pipe.model_card.neighbor_config.cutoff == 10.0

    def test_matrix_format_preferred(self):
        class _CooModel(MockEnergyForceModel):
            @property
            def model_card(self):
                return ModelCard(
                    outputs={"energies", "forces"},
                    needs_pbc=False,
                    neighbor_config=NeighborConfig(
                        cutoff=5.0, format=NeighborListFormat.COO
                    ),
                )

        class _MatrixModel(MockEnergyForceModel):
            @property
            def model_card(self):
                return ModelCard(
                    outputs={"energies", "forces"},
                    needs_pbc=False,
                    neighbor_config=NeighborConfig(
                        cutoff=5.0,
                        format=NeighborListFormat.MATRIX,
                        max_neighbors=64,
                    ),
                )

        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[_CooModel(), _MatrixModel()]),
            ]
        )
        assert pipe.model_card.neighbor_config.format == NeighborListFormat.MATRIX

    def test_needs_pbc_any(self):
        class _PbcModel(MockEnergyForceModel):
            @property
            def model_card(self):
                return ModelCard(
                    outputs={"energies", "forces"},
                    needs_pbc=True,
                    supports_pbc=True,
                )

        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[MockEnergyForceModel(), _PbcModel()]),
            ]
        )
        assert pipe.model_card.needs_pbc is True

    def test_half_list_mismatch_raises(self):
        class _HalfList(MockEnergyForceModel):
            @property
            def model_card(self):
                return ModelCard(
                    outputs={"energies", "forces"},
                    needs_pbc=False,
                    neighbor_config=NeighborConfig(cutoff=5.0, half_list=True),
                )

        class _FullList(MockEnergyForceModel):
            @property
            def model_card(self):
                return ModelCard(
                    outputs={"energies", "forces"},
                    needs_pbc=False,
                    neighbor_config=NeighborConfig(cutoff=5.0, half_list=False),
                )

        with pytest.raises(ValueError, match="half_list"):
            PipelineModelWrapper(
                groups=[
                    PipelineGroup(steps=[_HalfList(), _FullList()]),
                ]
            )


class TestPipelineNeighborHooks:
    """Tests for make_neighbor_hooks."""

    def test_no_hooks_without_neighbor_config(self):
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[MockEnergyForceModel()]),
            ]
        )
        hooks = pipe.make_neighbor_hooks()
        assert hooks == []

    def test_single_hook_with_neighbor_config(self):
        class _NLModel(MockEnergyForceModel):
            @property
            def model_card(self):
                return ModelCard(
                    outputs={"energies", "forces"},
                    needs_pbc=False,
                    neighbor_config=NeighborConfig(cutoff=5.0),
                )

        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[_NLModel()]),
            ]
        )
        hooks = pipe.make_neighbor_hooks()
        assert len(hooks) == 1
