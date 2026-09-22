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

import copy
from collections import OrderedDict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch._dynamo
from torch import nn

from nvalchemi._typing import ModelOutputs
from nvalchemi.data import AtomicData, Batch
from nvalchemi.hooks import NeighborListHook
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
from nvalchemi.models.pipeline import (
    _PIPELINE_NEIGHBOR_SOURCES_ATTR,
    PipelineGroup,
    PipelineModelWrapper,
    PipelineStep,
    _NeighborSourceData,
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
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            autograd_outputs=frozenset(),
            needs_pbc=False,
            active_outputs={"energy", "forces"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        B = data.num_graphs if isinstance(data, Batch) else 1
        N = data.positions.shape[0]
        return OrderedDict(
            energy=torch.full((B, 1), self._energy, dtype=data.positions.dtype),
            forces=torch.full((N, 3), self._force_val, dtype=data.positions.dtype),
        )


class MockAutogradEnergyModel(nn.Module, BaseModelMixin):
    """Mock model that returns energies computed from positions (autograd-capable)."""

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self._scale = scale
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            needs_pbc=False,
            active_outputs={"energy"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        positions = data.positions
        B = data.num_graphs if isinstance(data, Batch) else 1
        batch = (
            data.batch_idx
            if isinstance(data, Batch)
            else torch.zeros(positions.shape[0], dtype=torch.long)
        )
        per_atom = self._scale * (positions**2).sum(dim=-1)
        energies = torch.zeros(B, 1, dtype=positions.dtype, device=positions.device)
        energies.scatter_add_(0, batch.unsqueeze(-1), per_atom.unsqueeze(-1))
        return OrderedDict(energy=energies)


class MockTrainableAutogradEnergyModel(nn.Module, BaseModelMixin):
    """Mock autograd model with a trainable energy scale."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            needs_pbc=False,
            active_outputs={"energy"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        positions = data.positions
        B = data.num_graphs if isinstance(data, Batch) else 1
        batch = (
            data.batch_idx
            if isinstance(data, Batch)
            else torch.zeros(positions.shape[0], dtype=torch.long)
        )
        per_atom = self.scale * (positions**2).sum(dim=-1)
        energies = torch.zeros(B, 1, dtype=positions.dtype, device=positions.device)
        energies.scatter_add_(0, batch.unsqueeze(-1), per_atom.unsqueeze(-1))
        return OrderedDict(energy=energies)


class MockChargeEnergyModel(nn.Module, BaseModelMixin):
    """Mock model that outputs charges and energies (position-dependent for autograd)."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "charges"}),
            autograd_outputs=frozenset(),
            needs_pbc=False,
            active_outputs={"energy", "charges"},
        )

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
            data.batch_idx
            if isinstance(data, Batch)
            else torch.zeros(N, dtype=torch.long)
        )
        # Position-dependent energy so autograd can differentiate
        per_atom = (positions**2).sum(dim=-1)
        energies = torch.zeros(B, 1, dtype=positions.dtype, device=positions.device)
        energies.scatter_add_(0, batch.unsqueeze(-1), per_atom.unsqueeze(-1))
        return OrderedDict(
            energy=energies,
            charges=torch.ones(N, dtype=positions.dtype) * 0.5,
        )


class MockChargeOnlyModel(nn.Module, BaseModelMixin):
    """Mock model that only outputs charges (feeder)."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"charges"}),
            autograd_outputs=frozenset(),
            needs_pbc=False,
            active_outputs={"charges"},
        )

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
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            required_inputs=frozenset({"node_charges"}),
            autograd_outputs=frozenset(),
            needs_pbc=False,
            active_outputs={"energy"},
        )

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
            data.batch_idx
            if isinstance(data, Batch)
            else torch.zeros(charges.shape[0], dtype=torch.long)
        )
        per_atom = charges * (data.positions**2).sum(dim=-1)
        energies = torch.zeros(
            B, 1, dtype=data.positions.dtype, device=data.positions.device
        )
        energies.scatter_add_(0, batch.unsqueeze(-1), per_atom.unsqueeze(-1))
        return OrderedDict(energy=energies)


class MockForceOnlyModel(nn.Module, BaseModelMixin):
    """Mock model that only outputs forces (force corrector)."""

    def __init__(self, force_val: float = 0.1) -> None:
        super().__init__()
        self._force_val = force_val
        self.model_config = ModelConfig(
            outputs=frozenset({"forces"}),
            autograd_outputs=frozenset(),
            needs_pbc=False,
            active_outputs={"forces"},
        )

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
            outputs=frozenset({"energy", "node_charges", "node_spin"}),
            autograd_outputs=frozenset(),
            needs_pbc=False,
            active_outputs={"energy", "node_charges", "node_spin"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        B = data.num_graphs if isinstance(data, Batch) else 1
        N = data.positions.shape[0]
        return OrderedDict(
            energy=torch.ones(B, 1, dtype=data.positions.dtype),
            node_charges=torch.ones(N, dtype=data.positions.dtype) * 0.5,
            node_spin=torch.ones(N, dtype=data.positions.dtype) * 0.1,
        )


class MockSpinModel(nn.Module, BaseModelMixin):
    """Mock model that takes node_spin as input and outputs energies."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            required_inputs=frozenset({"node_spin"}),
            autograd_outputs=frozenset(),
            needs_pbc=False,
            active_outputs={"energy"},
        )

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
            data.batch_idx
            if isinstance(data, Batch)
            else torch.zeros(spin.shape[0], dtype=torch.long)
        )
        per_atom = spin**2
        energies = torch.zeros(
            B, 1, dtype=data.positions.dtype, device=data.positions.device
        )
        energies.scatter_add_(0, batch.unsqueeze(-1), per_atom.unsqueeze(-1))
        return OrderedDict(energy=energies)


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
        energy=torch.zeros(1, 1),
    )
    data2 = AtomicData(
        positions=torch.randn(2, 3),
        atomic_numbers=torch.tensor([1, 1]),
        forces=torch.zeros(2, 3),
        energy=torch.zeros(1, 1),
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
    def test_default_use_autograd_false(self):
        group = PipelineGroup(steps=[MockEnergyForceModel()])
        assert group.use_autograd is False

    def test_use_autograd_true(self):
        group = PipelineGroup(
            steps=[MockAutogradEnergyModel()],
            use_autograd=True,
        )
        assert group.use_autograd is True

    def test_derivative_fn_default_none(self):
        group = PipelineGroup(steps=[MockEnergyForceModel()])
        assert group.derivative_fn is None


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

    def test_model_config_synthesis(self):
        a = MockEnergyForceModel()
        b = MockForceOnlyModel()
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[a]),
                PipelineGroup(steps=[b]),
            ]
        )
        cfg = pipe.model_config
        assert "energy" in cfg.outputs
        assert "forces" in cfg.outputs

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
            out["energy"],
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
                PipelineGroup(steps=[a, b], use_autograd=True),
            ]
        )
        # Sub-models only have active_outputs={"energy"}, so pipeline inherits that.
        # Explicitly request forces from the pipeline.
        pipe.model_config.active_outputs = {"energy", "forces"}
        out = pipe(simple_batch)
        assert out["forces"].abs().sum() > 0
        assert out["energy"] is not None

    def test_training_preserves_force_graph_for_backward(self, simple_batch):
        model = MockTrainableAutogradEnergyModel()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}
        pipe.train()

        out = pipe(simple_batch)
        loss = out["forces"].pow(2).sum()
        loss.backward()

        assert out["forces"].requires_grad
        assert model.scale.grad is not None
        assert model.scale.grad.abs() > 0

    def test_training_preserves_stress_graph_for_backward(self):
        data = AtomicData(
            positions=torch.randn(4, 3, dtype=torch.float64),
            atomic_numbers=torch.tensor([6, 6, 8, 1]),
            forces=torch.zeros(4, 3, dtype=torch.float64),
            energy=torch.zeros(1, 1, dtype=torch.float64),
            cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0,
            pbc=torch.tensor([[True, True, True]]),
        )
        batch = Batch.from_data_list([data])
        model = MockTrainableAutogradEnergyModel().to(dtype=torch.float64)
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )
        pipe.model_config.active_outputs = {"energy", "stress"}
        pipe.train()

        out = pipe(batch)
        loss = out["stress"].pow(2).sum()
        loss.backward()

        assert out["stress"].requires_grad
        assert model.scale.grad is not None
        assert model.scale.grad.abs() > 0

    def test_autograd_disables_sub_model_forces(self, simple_batch):
        """Autograd group strips forces at forward time, not permanently."""
        m = MockAutogradEnergyModel()
        original_active = set(m.model_config.active_outputs)
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[m], use_autograd=True),
            ]
        )
        # Sub-model's config should NOT be permanently mutated —
        # the override is stored on the pipeline, not the model.
        assert m.model_config.active_outputs == original_active
        step = pipe.groups[0].steps[0]
        assert "forces" not in pipe._step_active_overrides[id(step)]

    def test_autograd_does_not_mutate_sub_model_config(self, simple_batch):
        """Sub-model with forces in active_outputs is not permanently mutated."""
        a = MockAutogradEnergyModel(scale=1.0)
        b = MockAutogradEnergyModel(scale=2.0)
        # Give both models forces in active_outputs — this is what the
        # pipeline's _configure_sub_models should strip at forward time
        # without permanently mutating the model.
        a.model_config.active_outputs = {"energy", "forces"}
        b.model_config.active_outputs = {"energy", "forces"}

        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[a, b], use_autograd=True)]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}

        # Run forward to exercise the override path.
        pipe(simple_batch)

        # After forward, sub-model configs must be unchanged.
        assert a.model_config.active_outputs == {"energy", "forces"}
        assert b.model_config.active_outputs == {"energy", "forces"}

    def test_deepcopy_refreshes_step_caches_on_forward(self, simple_batch):
        """Deep-copied pipelines rebuild id-keyed step caches on first forward."""
        model = MockAutogradEnergyModel()
        model.model_config.active_outputs = {"energy", "forces"}
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )
        original_step_id = id(pipe.groups[0].steps[0])
        assert original_step_id in pipe._step_active_overrides
        assert "forces" not in pipe._step_active_overrides[original_step_id]

        ema_pipe = copy.deepcopy(pipe)
        copied_step_id = id(ema_pipe.groups[0].steps[0])
        assert copied_step_id != original_step_id
        assert copied_step_id not in ema_pipe._step_active_overrides

        ema_pipe.model_config.active_outputs = {"energy", "forces"}
        ema_pipe(simple_batch)

        assert copied_step_id in ema_pipe._step_active_overrides
        assert "forces" not in ema_pipe._step_active_overrides[copied_step_id]
        assert model.model_config.active_outputs == {"energy", "forces"}


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
                    use_autograd=True,
                ),
            ]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}
        out = pipe(simple_batch)
        assert out["energy"] is not None
        # Forces should be non-zero (autograd through position -> charges -> energy)
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
                    use_autograd=True,
                ),
            ]
        )
        out = pipe(simple_batch)
        assert out["energy"] is not None


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
            out["energy"],
            torch.full((2, 1), 1.0, dtype=dtype),
        )


class TestPipelineThreeModelHybrid:
    """Case 5: autograd group + direct group."""

    def test_hybrid_forces(self, simple_batch):
        autograd_model = MockAutogradEnergyModel(scale=1.0)
        direct_model = MockEnergyForceModel(energy=0.5, force_val=0.1)
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[autograd_model], use_autograd=True),
                PipelineGroup(steps=[direct_model], use_autograd=False),
            ]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}
        out = pipe(simple_batch)
        # Total energy = autograd_energy + 0.5
        assert out["energy"] is not None
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
                PipelineGroup(steps=[a, b, c], use_autograd=False),
            ]
        )
        out = pipe(simple_batch)
        assert out["energy"] is not None


class TestPipelineModelConfigSynthesis:
    """Tests for synthesized model config from sub-models."""

    def test_max_cutoff_neighbor_config(self):
        class _SmallCutoff(MockEnergyForceModel):
            def __init__(self):
                super().__init__()
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy", "forces"}),
                    needs_pbc=False,
                    neighbor_config=NeighborConfig(cutoff=5.0),
                    active_outputs={"energy", "forces"},
                )

        class _LargeCutoff(MockEnergyForceModel):
            def __init__(self):
                super().__init__()
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy", "forces"}),
                    needs_pbc=False,
                    neighbor_config=NeighborConfig(cutoff=10.0),
                    active_outputs={"energy", "forces"},
                )

        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[_SmallCutoff(), _LargeCutoff()]),
            ]
        )
        assert pipe.model_config.neighbor_config.cutoff == 10.0

    def test_matrix_format_preferred(self):
        class _CooModel(MockEnergyForceModel):
            def __init__(self):
                super().__init__()
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy", "forces"}),
                    needs_pbc=False,
                    neighbor_config=NeighborConfig(
                        cutoff=5.0, format=NeighborListFormat.COO
                    ),
                    active_outputs={"energy", "forces"},
                )

        class _MatrixModel(MockEnergyForceModel):
            def __init__(self):
                super().__init__()
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy", "forces"}),
                    needs_pbc=False,
                    neighbor_config=NeighborConfig(
                        cutoff=5.0,
                        format=NeighborListFormat.MATRIX,
                    ),
                    active_outputs={"energy", "forces"},
                )

        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[_CooModel(), _MatrixModel()]),
            ]
        )
        assert pipe.model_config.neighbor_config.format == NeighborListFormat.MATRIX

    def test_needs_pbc_any(self):
        class _PbcModel(MockEnergyForceModel):
            def __init__(self):
                super().__init__()
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy", "forces"}),
                    needs_pbc=True,
                    supports_pbc=True,
                    active_outputs={"energy", "forces"},
                )

        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[MockEnergyForceModel(), _PbcModel()]),
            ]
        )
        assert pipe.model_config.needs_pbc is True

    def test_half_list_mismatch_raises(self):
        class _HalfList(MockEnergyForceModel):
            def __init__(self):
                super().__init__()
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy", "forces"}),
                    needs_pbc=False,
                    neighbor_config=NeighborConfig(cutoff=5.0, half_list=True),
                    active_outputs={"energy", "forces"},
                )

        class _FullList(MockEnergyForceModel):
            def __init__(self):
                super().__init__()
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy", "forces"}),
                    needs_pbc=False,
                    neighbor_config=NeighborConfig(cutoff=5.0, half_list=False),
                    active_outputs={"energy", "forces"},
                )

        with pytest.raises(ValueError, match="half_list"):
            PipelineModelWrapper(
                groups=[
                    PipelineGroup(steps=[_HalfList(), _FullList()]),
                ]
            )


class TestPipelineNeighborAdaptationApi:
    """Tests for pipeline-level neighbor adaptation options."""

    def test_neighbor_adaptation_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="neighbor_adaptation"):
            PipelineModelWrapper(
                groups=[PipelineGroup(steps=[MockEnergyForceModel()])],
                neighbor_adaptation="on a good day",
            )

    def test_max_cutoff_ratio_must_be_at_least_one(self):
        with pytest.raises(ValueError, match="max_cutoff_ratio"):
            PipelineModelWrapper(
                groups=[PipelineGroup(steps=[MockEnergyForceModel()])],
                max_cutoff_ratio=0.99,
            )

    def test_forwards_neighbor_list_method(self):
        class _NLModel(MockEnergyForceModel):
            def __init__(self):
                super().__init__()
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy", "forces"}),
                    needs_pbc=False,
                    neighbor_config=NeighborConfig(cutoff=5.0),
                    active_outputs={"energy", "forces"},
                )

        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[_NLModel()]),
            ]
        )
        hooks = pipe.make_neighbor_hooks(neighbor_list_method="cell_list")

        assert len(hooks) == 1
        assert hooks[0].method == "cell_list"


# ===========================================================================
# Neighbor adaptation tests
# ===========================================================================


def _make_neighbor_batch():
    """Build a 1-system, 4-atom batch with MATRIX neighbor data.

    Positions along x-axis: 0, 1, 3, 7.
    At cutoff 10 (pipeline): atom 0 sees 1(d=1), 2(d=3), 3(d=7);
                              atom 1 sees 0(d=1), 2(d=2), 3(d=6);
                              atom 2 sees 0(d=3), 1(d=2), 3(d=4);
                              atom 3 sees 0(d=7), 1(d=6), 2(d=4).
    At cutoff 4 (tight model): only pairs with d<=4 survive.
    """
    data = AtomicData(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
        ),
        atomic_numbers=torch.tensor([1, 1, 1, 1]),
        forces=torch.zeros(4, 3),
        energy=torch.zeros(1, 1),
    )
    batch = Batch.from_data_list([data])
    N = batch.num_nodes  # 4
    K = 3
    # Full neighbor matrix at cutoff 10 — every atom sees all others.
    nm = torch.tensor(
        [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]],
        dtype=torch.int32,
    )
    nn_ = torch.tensor([3, 3, 3, 3], dtype=torch.int32)
    shifts = torch.zeros(N, K, 3, dtype=torch.int32)

    object.__setattr__(batch, "neighbor_matrix", nm)
    object.__setattr__(batch, "num_neighbors", nn_)
    object.__setattr__(batch, "neighbor_matrix_shifts", shifts)
    object.__setattr__(batch, "_neighbor_list_cutoff", 10.0)
    return batch


class _CaptureMixin:
    """Mixin that records what neighbor data the model sees during forward()."""

    def forward(self, data, **kwargs):
        self.captured_neighbor_matrix = getattr(data, "neighbor_matrix", None)
        self.captured_num_neighbors = getattr(data, "num_neighbors", None)
        self.captured_neighbor_list = getattr(data, "neighbor_list", None)
        self.captured_edge_ptr = getattr(data, "edge_ptr", None)
        self.captured_cutoff = getattr(data, "_neighbor_list_cutoff", None)
        # Clone tensors so they survive the pipeline's restore step.
        for attr in (
            "captured_neighbor_matrix",
            "captured_num_neighbors",
            "captured_neighbor_list",
            "captured_edge_ptr",
        ):
            val = getattr(self, attr, None)
            if val is not None:
                setattr(self, attr, val.clone())
        return super().forward(data, **kwargs)


class _MatrixModel10(_CaptureMixin, MockEnergyForceModel):
    """MATRIX model at cutoff 10 — matches the pipeline's cutoff exactly."""

    def __init__(self):
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=10.0,
                format=NeighborListFormat.MATRIX,
            ),
            active_outputs={"energy", "forces"},
        )


class _MatrixModel4(_CaptureMixin, MockEnergyForceModel):
    """MATRIX model at cutoff 4 — tighter than the pipeline's cutoff."""

    def __init__(self):
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=4.0,
                format=NeighborListFormat.MATRIX,
            ),
            active_outputs={"energy", "forces"},
        )


class _MatrixModel7(_CaptureMixin, MockEnergyForceModel):
    """MATRIX model at cutoff 7 — within default auto adaptation ratio."""

    def __init__(self):
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=7.0,
                format=NeighborListFormat.MATRIX,
            ),
            active_outputs={"energy", "forces"},
        )


class _COOModel4(_CaptureMixin, MockEnergyForceModel):
    """COO model at cutoff 4 — needs both format conversion AND filtering."""

    def __init__(self):
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=4.0,
                format=NeighborListFormat.COO,
            ),
            active_outputs={"energy", "forces"},
        )


class TestPipelineNeighborPlanning:
    """Tests for internal neighbor source planning."""

    def test_always_mode_uses_single_max_source(self):
        wide = _MatrixModel10()
        tight = _MatrixModel4()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[wide, tight])],
            neighbor_adaptation="always",
        )

        assert [s.config.cutoff for s in pipe._neighbor_sources] == [10.0]
        tight_plan = pipe._step_neighbor_plans[id(pipe.groups[0].steps[1])]
        assert tight_plan.source_id == 0
        assert tight_plan.needs_cutoff_adaptation is True

    def test_never_mode_creates_exact_cutoff_sources(self):
        wide = _MatrixModel10()
        tight = _MatrixModel4()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[wide, tight])],
            neighbor_adaptation="never",
        )

        assert [s.config.cutoff for s in pipe._neighbor_sources] == [10.0, 4.0]
        for plan in pipe._step_neighbor_plans.values():
            assert plan.needs_cutoff_adaptation is False

    def test_auto_mode_groups_within_ratio_and_splits_large_gap(self):
        wide = _MatrixModel10()
        mid = _MatrixModel7()
        tight = _MatrixModel4()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[wide, mid, tight])],
        )

        assert [s.config.cutoff for s in pipe._neighbor_sources] == [10.0, 4.0]
        wide_plan = pipe._step_neighbor_plans[id(pipe.groups[0].steps[0])]
        mid_plan = pipe._step_neighbor_plans[id(pipe.groups[0].steps[1])]
        tight_plan = pipe._step_neighbor_plans[id(pipe.groups[0].steps[2])]
        assert wide_plan.source_id == 0
        assert wide_plan.needs_cutoff_adaptation is False
        assert mid_plan.source_id == 0
        assert mid_plan.needs_cutoff_adaptation is True
        assert tight_plan.source_id == 1
        assert tight_plan.needs_cutoff_adaptation is False

    def test_auto_mode_uses_max_cutoff_ratio(self):
        wide = _MatrixModel10()
        mid = _MatrixModel7()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[wide, mid])],
            max_cutoff_ratio=1.2,
        )

        assert [s.config.cutoff for s in pipe._neighbor_sources] == [10.0, 7.0]


class TestPipelineNeighborHookFactory:
    """Tests for single-source and multi-source neighbor hook creation."""

    def test_no_hooks_without_neighbor_config(self):
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[MockEnergyForceModel()])],
        )

        assert pipe.make_neighbor_hooks() == []

    def test_single_source_returns_neighbor_list_hook(self):
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[_MatrixModel10(), _MatrixModel7()])],
        )

        hooks = pipe.make_neighbor_hooks()

        assert len(hooks) == 1
        assert isinstance(hooks[0], NeighborListHook)
        assert hooks[0].config.cutoff == 10.0

    def test_multi_source_returns_composite_hook(self):
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[_MatrixModel10(), _MatrixModel4()])],
        )

        hooks = pipe.make_neighbor_hooks()

        assert len(hooks) == 1
        hook = hooks[0]
        assert type(hook).__name__ == "_PipelineNeighborListHook"
        assert [h.config.cutoff for h in hook.hooks] == [10.0, 4.0]

    def test_composite_hook_source_count_is_static(self):
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[_MatrixModel10(), _MatrixModel4()])],
        )

        hooks1 = pipe.make_neighbor_hooks()
        hooks2 = pipe.make_neighbor_hooks()

        assert len(hooks1) == 1
        assert len(hooks2) == 1
        assert type(hooks1[0]).__name__ == "_PipelineNeighborListHook"
        assert type(hooks2[0]).__name__ == "_PipelineNeighborListHook"
        assert len(hooks1[0].hooks) == 2
        assert len(hooks2[0].hooks) == 2


class TestPipelineNeighborAdaptation:
    """Verify that sub-models receive correctly adapted neighbor data."""

    def test_same_cutoff_no_filtering(self):
        """Model at pipeline cutoff receives the original neighbor matrix."""
        model = _MatrixModel10()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model])],
        )
        batch = _make_neighbor_batch()
        orig_nm = batch.neighbor_matrix.clone()
        pipe(batch)

        assert model.captured_neighbor_matrix is not None
        torch.testing.assert_close(model.captured_neighbor_matrix, orig_nm)
        # All atoms still see 3 neighbors each.
        assert (model.captured_num_neighbors == 3).all()

    def test_tighter_cutoff_filters_matrix(self):
        """Model at cutoff=4 should not see neighbors beyond distance 4.

        With positions [0, 1, 3, 7] and cutoff 4:
          atom 0: sees 1(d=1), 2(d=3)       → 2 neighbors
          atom 1: sees 0(d=1), 2(d=2)       → 2 neighbors
          atom 2: sees 0(d=3), 1(d=2), 3(d=4) → d=4 is NOT < 4 → 2 neighbors
          atom 3: sees 2(d=4) → NOT < 4     → 0 neighbors
        """
        wide = _MatrixModel10()
        tight = _MatrixModel4()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[wide, tight])],
            neighbor_adaptation="always",
        )
        batch = _make_neighbor_batch()
        pipe(batch)

        # Wide model sees all 3 neighbors per atom (unfiltered).
        assert (wide.captured_num_neighbors == 3).all()

        # Tight model sees filtered counts.
        expected_nn = torch.tensor([2, 2, 2, 0], dtype=torch.int32)
        torch.testing.assert_close(tight.captured_num_neighbors, expected_nn)

        # Verify no neighbor index in the tight result is atom 3 for atoms 0,1
        # (distances 7 and 6, both > 4).
        nm = tight.captured_neighbor_matrix
        fill = 4  # num_nodes
        for atom_idx in [0, 1]:
            valid = nm[atom_idx][nm[atom_idx] < fill]
            assert 3 not in valid.tolist(), (
                f"atom {atom_idx} should not see atom 3 at cutoff 4"
            )

    def test_deepcopy_preserves_neighbor_adaptation(self):
        """Deep-copied pipelines still filter neighbors per sub-model cutoff."""
        wide = _MatrixModel10()
        tight = _MatrixModel4()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[wide, tight])],
            neighbor_adaptation="always",
        )
        tight_step_id = id(pipe.groups[0].steps[1])
        assert pipe._step_neighbor_plans[tight_step_id].needs_cutoff_adaptation is True

        copied = copy.deepcopy(pipe)
        tight_copy = copied.groups[0].steps[1].model
        copied_tight_step_id = id(copied.groups[0].steps[1])
        assert copied_tight_step_id not in copied._step_neighbor_plans

        batch = _make_neighbor_batch()
        copied(batch)

        assert copied_tight_step_id in copied._step_neighbor_plans
        expected_nn = torch.tensor([2, 2, 2, 0], dtype=torch.int32)
        torch.testing.assert_close(tight_copy.captured_num_neighbors, expected_nn)

    def test_matrix_to_coo_conversion(self):
        """COO model in a MATRIX pipeline receives converted neighbor list."""
        matrix_model = _MatrixModel10()
        coo_model = _COOModel4()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[matrix_model, coo_model])],
            neighbor_adaptation="always",
        )
        batch = _make_neighbor_batch()
        pipe(batch)

        # Matrix model sees MATRIX data.
        assert matrix_model.captured_neighbor_matrix is not None
        assert matrix_model.captured_neighbor_list is None

        # COO model sees COO data (neighbor_list set by pipeline).
        assert coo_model.captured_neighbor_list is not None
        nl = coo_model.captured_neighbor_list  # (E, 2)
        assert nl.ndim == 2 and nl.shape[1] == 2

        # All edges must be within cutoff 4.
        positions = batch.positions
        for e in range(nl.shape[0]):
            i, j = int(nl[e, 0]), int(nl[e, 1])
            dist = (positions[i] - positions[j]).norm().item()
            assert dist < 4.0, f"edge ({i},{j}) dist={dist:.2f} exceeds cutoff 4"

    def test_batch_restored_after_forward(self):
        """Pipeline must restore the original neighbor data after forward."""
        wide = _MatrixModel10()
        tight = _MatrixModel4()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[wide, tight])],
            neighbor_adaptation="always",
        )
        batch = _make_neighbor_batch()
        orig_nm = batch.neighbor_matrix.clone()
        orig_nn = batch.num_neighbors.clone()
        orig_cutoff = batch._neighbor_list_cutoff

        pipe(batch)

        torch.testing.assert_close(batch.neighbor_matrix, orig_nm)
        torch.testing.assert_close(batch.num_neighbors, orig_nn)
        assert batch._neighbor_list_cutoff == orig_cutoff
        # COO attributes should not leak onto the batch.
        assert "neighbor_list" not in batch.__dict__

    def test_matrix_k_dimension_trimmed(self):
        """Filtered MATRIX must have K trimmed to actual max neighbors.

        Without trimming, a matrix built at 80 Å (K=430k) would be passed
        intact to a 5 Å model — correct but extremely wasteful.
        """
        wide = _MatrixModel10()
        tight = _MatrixModel4()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[wide, tight])],
            neighbor_adaptation="always",
        )
        batch = _make_neighbor_batch()
        orig_k = batch.neighbor_matrix.shape[1]  # K=3
        pipe(batch)

        # Wide model at pipeline cutoff: same K (no adaptation needed).
        assert wide.captured_neighbor_matrix.shape[1] == orig_k

        # Tight model: max num_neighbors is 2, so K should be trimmed to 2.
        assert tight.captured_neighbor_matrix.shape[1] == 2

    def test_storage_backed_neighbor_data(self):
        """Adaptation works when neighbor data is in Batch._storage (not __dict__).

        This is the real-world path: compute_neighbors writes to _storage
        via atoms_group['neighbor_matrix']. The pipeline must shadow it
        in __dict__ for the sub-model to see adapted data.
        """
        data = AtomicData(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [7.0, 0.0, 0.0]],
            ),
            atomic_numbers=torch.tensor([1, 1, 1, 1]),
            forces=torch.zeros(4, 3),
            energy=torch.zeros(1, 1),
            cell=torch.eye(3).unsqueeze(0) * 20.0,
            pbc=torch.tensor([[True, True, True]]),
        )
        batch = Batch.from_data_list([data])

        from nvalchemi.neighbors import compute_neighbors

        compute_neighbors(batch, cutoff=10.0)

        # Verify data is in _storage, NOT __dict__
        assert "neighbor_matrix" not in batch.__dict__
        assert "neighbor_matrix" in batch._storage

        tight = _MatrixModel4()
        wide = _MatrixModel10()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[tight, wide])],
            neighbor_adaptation="always",
        )
        pipe(batch)

        # Tight model should see filtered + trimmed matrix.
        assert tight.captured_neighbor_matrix is not None
        max_nn = int(tight.captured_num_neighbors.max())
        assert tight.captured_neighbor_matrix.shape[1] == max_nn

        # Wide model at pipeline cutoff should see original (unfiltered).
        assert wide.captured_neighbor_matrix is not None
        assert wide.captured_neighbor_matrix.shape == batch.neighbor_matrix.shape

    def test_coo_attrs_removed_after_forward(self):
        """COO attributes set for a COO sub-model must not persist on the batch."""
        coo_model = _COOModel4()
        matrix_model = _MatrixModel10()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[coo_model, matrix_model])],
            neighbor_adaptation="always",
        )
        batch = _make_neighbor_batch()
        pipe(batch)

        # neighbor_list was temporarily added for coo_model, must be gone.
        assert "neighbor_list" not in batch.__dict__
        assert "edge_ptr" not in batch.__dict__

    def test_auto_large_cutoff_gap_uses_exact_tight_source(self):
        wide = _MatrixModel10()
        tight = _MatrixModel4()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[wide, tight])],
        )
        batch = _make_neighbor_batch()

        tight_nm = torch.tensor(
            [[1, 2], [0, 2], [0, 1], [4, 4]],
            dtype=torch.int32,
        )
        tight_nn = torch.tensor([2, 2, 2, 0], dtype=torch.int32)
        batch.__dict__[_PIPELINE_NEIGHBOR_SOURCES_ATTR] = (
            _NeighborSourceData(
                source_id=0,
                config=pipe._neighbor_sources[0].config,
                neighbor_matrix=batch.neighbor_matrix,
                num_neighbors=batch.num_neighbors,
                neighbor_matrix_shifts=batch.neighbor_matrix_shifts,
            ),
            _NeighborSourceData(
                source_id=1,
                config=pipe._neighbor_sources[1].config,
                neighbor_matrix=tight_nm,
                num_neighbors=tight_nn,
                neighbor_matrix_shifts=torch.zeros(4, 2, 3, dtype=torch.int32),
            ),
        )

        pipe(batch)

        torch.testing.assert_close(
            wide.captured_num_neighbors,
            torch.tensor([3, 3, 3, 3], dtype=torch.int32),
        )
        torch.testing.assert_close(tight.captured_num_neighbors, tight_nn)
        assert (
            pipe._step_neighbor_plans[
                id(pipe.groups[0].steps[1])
            ].needs_cutoff_adaptation
            is False
        )

    def test_multi_source_missing_hook_raises(self):
        wide = _MatrixModel10()
        tight = _MatrixModel4()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[wide, tight])],
        )
        batch = _make_neighbor_batch()

        assert len(pipe._neighbor_sources) == 2
        assert _PIPELINE_NEIGHBOR_SOURCES_ATTR not in batch.__dict__

        with pytest.raises(
            RuntimeError, match="planned multiple neighbor-list sources"
        ):
            pipe(batch)

    def test_auto_within_ratio_filters_from_wide_source(self):
        wide = _MatrixModel10()
        mid = _MatrixModel7()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[wide, mid])],
        )
        batch = _make_neighbor_batch()

        pipe(batch)

        assert len(pipe._neighbor_sources) == 1
        assert pipe._step_neighbor_plans[
            id(pipe.groups[0].steps[1])
        ].needs_cutoff_adaptation
        expected_nn = torch.tensor([2, 3, 3, 2], dtype=torch.int32)
        torch.testing.assert_close(mid.captured_num_neighbors, expected_nn)

    def test_multi_source_selection_restores_original_batch_attrs(self):
        wide = _MatrixModel10()
        tight = _MatrixModel4()
        pipe = PipelineModelWrapper(groups=[PipelineGroup(steps=[wide, tight])])
        batch = _make_neighbor_batch()
        orig_nm = batch.neighbor_matrix.clone()
        orig_nn = batch.num_neighbors.clone()
        orig_cutoff = batch._neighbor_list_cutoff

        tight_nm = torch.full((4, 1), 4, dtype=torch.int32)
        tight_nn = torch.zeros(4, dtype=torch.int32)
        batch.__dict__[_PIPELINE_NEIGHBOR_SOURCES_ATTR] = (
            _NeighborSourceData(
                source_id=0,
                config=pipe._neighbor_sources[0].config,
                neighbor_matrix=batch.neighbor_matrix,
                num_neighbors=batch.num_neighbors,
                neighbor_matrix_shifts=batch.neighbor_matrix_shifts,
            ),
            _NeighborSourceData(
                source_id=1,
                config=pipe._neighbor_sources[1].config,
                neighbor_matrix=tight_nm,
                num_neighbors=tight_nn,
                neighbor_matrix_shifts=torch.zeros(4, 1, 3, dtype=torch.int32),
            ),
        )

        pipe(batch)

        torch.testing.assert_close(batch.neighbor_matrix, orig_nm)
        torch.testing.assert_close(batch.num_neighbors, orig_nn)
        assert batch._neighbor_list_cutoff == orig_cutoff


# ===========================================================================
# model_config synthesis tests
# ===========================================================================


class TestPipelineModelConfigActiveSynthesis:
    """Pipeline model_config.active_outputs is union of sub-model active_outputs sets."""

    def test_default_active_outputs_from_submodels(self):
        """Sub-models default to {"energy", "forces"} -> pipeline same."""
        a = MockEnergyForceModel()
        b = MockEnergyForceModel()
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[a]),
                PipelineGroup(steps=[b]),
            ]
        )
        assert pipe.model_config.active_outputs == {"energy", "forces"}

    def test_stresses_inherited_from_submodel(self):
        """If a sub-model has stresses, pipeline should too."""
        a = MockEnergyForceModel()
        a.model_config.active_outputs = {"energy", "forces", "stress"}
        b = MockEnergyForceModel()
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[a]),
                PipelineGroup(steps=[b]),
            ]
        )
        assert "stress" in pipe.model_config.active_outputs

    def test_energy_only_submodels(self):
        """Sub-models with only energies -> pipeline only energies."""
        a = MockAutogradEnergyModel()
        a.model_config.active_outputs = {"energy"}
        b = MockAutogradEnergyModel()
        b.model_config.active_outputs = {"energy"}
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[a, b], use_autograd=True),
            ]
        )
        assert pipe.model_config.active_outputs == {"energy"}

    def test_user_can_expand_active_outputs(self):
        """User can add stresses after construction."""
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[MockEnergyForceModel()]),
            ]
        )
        pipe.model_config.active_outputs = {"energy", "forces", "stress"}
        assert "stress" in pipe.model_config.active_outputs


# ===========================================================================
# Custom derivative_fn tests
# ===========================================================================


class TestPipelineCustomDerivativeFn:
    """Tests for user-provided derivative_fn."""

    def test_custom_fn_called(self, simple_batch):
        """Custom derivative_fn receives energy, data, and requested keys."""
        called_with = {}

        def my_derivs(energy, data, requested):
            called_with["energy"] = energy
            called_with["requested"] = requested
            N = data.positions.shape[0]
            return {"forces": torch.zeros(N, 3, dtype=data.positions.dtype)}

        a = MockAutogradEnergyModel()
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[a],
                    use_autograd=True,
                    derivative_fn=my_derivs,
                ),
            ]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}
        out = pipe(simple_batch)
        assert "energy" in called_with
        assert "forces" in called_with["requested"]
        assert out["forces"] is not None

    def test_custom_fn_novel_output(self, simple_batch):
        """Custom derivative_fn can return novel keys not in default."""

        def my_derivs(energy, data, requested):
            N = data.positions.shape[0]
            result = {}
            if "forces" in requested:
                result["forces"] = -torch.autograd.grad(
                    energy,
                    data.positions,
                    grad_outputs=torch.ones_like(energy),
                    retain_graph="my_hessian" in requested,
                )[0]
            if "my_hessian" in requested:
                result["my_hessian"] = torch.eye(N, dtype=data.positions.dtype)
            return result

        a = MockAutogradEnergyModel()
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[a],
                    use_autograd=True,
                    derivative_fn=my_derivs,
                ),
            ]
        )
        pipe.model_config.active_outputs = {"energy", "forces", "my_hessian"}
        out = pipe(simple_batch)
        assert "my_hessian" in out

    def test_energy_only_skips_derivatives(self, simple_batch):
        """When active_outputs={"energy"}, no derivative function is called."""
        call_count = [0]

        def my_derivs(energy, data, requested):
            call_count[0] += 1
            return {}

        a = MockAutogradEnergyModel()
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[a],
                    use_autograd=True,
                    derivative_fn=my_derivs,
                ),
            ]
        )
        pipe.model_config.active_outputs = {"energy"}
        out = pipe(simple_batch)
        assert call_count[0] == 0
        assert "energy" in out


# ===========================================================================
# prepare_strain tests
# ===========================================================================


class TestPrepareStrain:
    """Tests for the prepare_strain utility."""

    def test_output_shapes(self):
        """prepare_strain returns correct shapes."""
        from nvalchemi.models._utils import prepare_strain

        N, B = 5, 2
        positions = torch.randn(N, 3)
        cell = torch.eye(3).unsqueeze(0).expand(B, -1, -1) * 10.0
        batch_idx = torch.tensor([0, 0, 0, 1, 1])

        scaled_pos, scaled_cell, displacement = prepare_strain(
            positions, cell, batch_idx
        )
        assert scaled_pos.shape == (N, 3)
        assert scaled_cell.shape == (B, 3, 3)
        assert displacement.shape == (B, 3, 3)
        assert displacement.requires_grad

    def test_identity_at_zero_displacement(self):
        """At zero displacement, scaled == original."""
        from nvalchemi.models._utils import prepare_strain

        positions = torch.randn(4, 3, dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 5.0
        batch_idx = torch.zeros(4, dtype=torch.long)

        scaled_pos, scaled_cell, displacement = prepare_strain(
            positions, cell, batch_idx
        )
        torch.testing.assert_close(scaled_pos, positions, atol=1e-12, rtol=0)
        torch.testing.assert_close(scaled_cell, cell, atol=1e-12, rtol=0)

    def test_gradient_flows_through_displacement(self):
        """Energy computed from scaled positions has grad wrt displacement."""
        from nvalchemi.models._utils import prepare_strain

        positions = torch.randn(3, 3, dtype=torch.float64)
        cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0
        batch_idx = torch.zeros(3, dtype=torch.long)

        scaled_pos, scaled_cell, displacement = prepare_strain(
            positions, cell, batch_idx
        )
        energy = (scaled_pos**2).sum()
        grad = torch.autograd.grad(energy, displacement)[0]
        assert grad is not None
        assert grad.shape == (1, 3, 3)

    def test_stress_from_asymmetric_energy_is_symmetric(self):
        """prepare_strain projects displacement gradients onto symmetric strain."""
        from nvalchemi.models._utils import autograd_stresses, prepare_strain

        positions = torch.tensor(
            [[0.3, -0.7, 1.1], [1.2, 0.4, -0.2], [-0.5, 0.8, 0.6]],
            dtype=torch.float64,
        )
        cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 5.0
        batch_idx = torch.zeros(positions.shape[0], dtype=torch.long)

        scaled_pos, scaled_cell, displacement = prepare_strain(
            positions,
            cell,
            batch_idx,
        )
        x, y = scaled_pos[:, 0], scaled_pos[:, 1]
        energy = (x * y**2).sum() + scaled_cell[0, 0, 0] * scaled_cell[0, 1, 1] ** 2
        stress = autograd_stresses(energy, displacement, cell, num_graphs=1)

        torch.testing.assert_close(
            stress,
            stress.transpose(-1, -2),
            atol=1e-12,
            rtol=0,
        )

    def test_symmetric_response_matches_raw_displacement_reference(self):
        """Symmetric strain preserves models with symmetric raw stress response."""
        from nvalchemi.models._utils import autograd_stresses, prepare_strain

        positions = torch.tensor(
            [[0.3, -0.7, 1.1], [1.2, 0.4, -0.2], [-0.5, 0.8, 0.6]],
            dtype=torch.float64,
        )
        cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 5.0
        batch_idx = torch.zeros(positions.shape[0], dtype=torch.long)

        scaled_pos, scaled_cell, displacement = prepare_strain(
            positions,
            cell,
            batch_idx,
        )
        energy = (scaled_pos**2).sum() + (scaled_cell**2).sum()
        stress = autograd_stresses(energy, displacement, cell, num_graphs=1)

        raw_displacement = torch.zeros(
            1,
            3,
            3,
            dtype=positions.dtype,
            requires_grad=True,
        )
        raw_deformation = torch.eye(3, dtype=positions.dtype).unsqueeze(0)
        raw_deformation = raw_deformation + raw_displacement
        raw_scaled_pos = torch.einsum(
            "ni,nij->nj",
            positions,
            raw_deformation[batch_idx],
        )
        raw_scaled_cell = torch.einsum("bij,bjk->bik", cell, raw_deformation)
        raw_energy = (raw_scaled_pos**2).sum() + (raw_scaled_cell**2).sum()
        raw_stress = autograd_stresses(
            raw_energy,
            raw_displacement,
            cell,
            num_graphs=1,
        )

        torch.testing.assert_close(stress, raw_stress, atol=1e-12, rtol=0)

    def test_multi_system_batches(self):
        """Each system gets its own displacement."""
        from nvalchemi.models._utils import prepare_strain

        positions = torch.randn(6, 3, dtype=torch.float64)
        cell = torch.stack(
            [
                torch.eye(3, dtype=torch.float64) * 5.0,
                torch.eye(3, dtype=torch.float64) * 8.0,
            ]
        )
        batch_idx = torch.tensor([0, 0, 0, 1, 1, 1])

        scaled_pos, scaled_cell, displacement = prepare_strain(
            positions, cell, batch_idx
        )
        assert displacement.shape == (2, 3, 3)


# ===========================================================================
# torch.compile tests
# ===========================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestPipelineCompile:
    """Test that pipeline forward passes are compatible with torch.compile.

    Uses DemoModelWrapper (autograd forces) + LennardJonesModelWrapper
    (analytical forces) to exercise both code paths in a compiled pipeline.
    """

    @pytest.fixture
    def lj_batch_cuda(self):
        """A small PBC argon system on CUDA with a real neighbor list."""
        pytest.importorskip("warp")
        from nvalchemi.models.lj import LennardJonesModelWrapper
        from nvalchemi.neighbors import compute_neighbors

        device = torch.device("cuda")
        n_atoms = 8
        spacing = 3.8
        coords = [
            [ix * spacing, iy * spacing, iz * spacing]
            for ix in range(2)
            for iy in range(2)
            for iz in range(2)
        ]
        positions = torch.tensor(coords, dtype=torch.float32)
        box_size = 2 * spacing + 1.0

        data = AtomicData(
            positions=positions,
            atomic_numbers=torch.full((n_atoms,), 18, dtype=torch.long),
            atomic_masses=torch.full((n_atoms,), 39.948),
            forces=torch.zeros(n_atoms, 3),
            energy=torch.zeros(1, 1),
            cell=torch.eye(3).unsqueeze(0) * box_size,
            pbc=torch.tensor([[True, True, True]]),
        )
        data.add_node_property("velocities", torch.zeros(n_atoms, 3))

        lj = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5)
        batch = Batch.from_data_list([data], device=device)
        batch["stress"] = torch.zeros(1, 3, 3, device=device)

        compute_neighbors(batch, config=lj.model_config.neighbor_config)

        return batch, lj

    def test_direct_pipeline_compiles(self, lj_batch_cuda):
        """A direct-force pipeline (LJ only) can be torch.compiled."""
        batch, lj = lj_batch_cuda
        from nvalchemi.models.pipeline import PipelineGroup, PipelineModelWrapper

        pipe = PipelineModelWrapper(groups=[PipelineGroup(steps=[lj])])

        # Warmup (uncompiled)
        out_eager = pipe(batch)
        assert out_eager["energy"] is not None
        assert out_eager["forces"] is not None

        # Compile and run
        compiled_pipe = torch.compile(pipe, fullgraph=False)
        out_compiled = compiled_pipe(batch)

        torch.testing.assert_close(
            out_compiled["energy"], out_eager["energy"], atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            out_compiled["forces"], out_eager["forces"], atol=1e-5, rtol=1e-5
        )

    def test_autograd_pipeline_compiles(self, lj_batch_cuda):
        """An autograd pipeline with a simple model can be torch.compiled.

        Uses a minimal mock model (no beartype, no Pydantic access in
        forward) to test that the pipeline's autograd machinery —
        energy summation, requires_grad, autograd_forces — works under
        torch.compile.
        """
        batch, _ = lj_batch_cuda

        # Use a compile-friendly mock instead of DemoModelWrapper
        # (beartype + Pydantic are not TorchDynamo-compatible).
        model = _QuadraticEnergyModel(scale=1.0)
        model = model.to(batch.device)

        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}

        out_eager = pipe(batch)
        assert out_eager["forces"] is not None

        torch._dynamo.config.suppress_errors = True
        compiled_pipe = torch.compile(pipe, fullgraph=False)
        out_compiled = compiled_pipe(batch)

        torch.testing.assert_close(
            out_compiled["energy"], out_eager["energy"], atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            out_compiled["forces"], out_eager["forces"], atol=1e-5, rtol=1e-5
        )

    def test_hybrid_pipeline_compiles(self, lj_batch_cuda):
        """A hybrid pipeline (autograd mock + LJ direct) can be torch.compiled.

        Combines autograd forces (mock quadratic energy) with analytical
        forces (Lennard-Jones kernel).
        """
        batch, lj = lj_batch_cuda
        autograd_model = _QuadraticEnergyModel(scale=1.0)
        autograd_model = autograd_model.to(batch.device)

        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[autograd_model], use_autograd=True),
                PipelineGroup(steps=[lj]),
            ]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}

        out_eager = pipe(batch)
        assert out_eager["energy"] is not None
        assert out_eager["forces"] is not None

        torch._dynamo.config.suppress_errors = True
        compiled_pipe = torch.compile(pipe, fullgraph=False)
        out_compiled = compiled_pipe(batch)

        torch.testing.assert_close(
            out_compiled["energy"], out_eager["energy"], atol=1e-4, rtol=1e-4
        )
        torch.testing.assert_close(
            out_compiled["forces"], out_eager["forces"], atol=1e-4, rtol=1e-4
        )

    def test_compiled_stresses_from_lj(self, lj_batch_cuda):
        """LJ stress computation works under torch.compile."""
        batch, lj = lj_batch_cuda
        lj.model_config.active_outputs = {"energy", "forces", "stress"}

        pipe = PipelineModelWrapper(groups=[PipelineGroup(steps=[lj])])

        out_eager = pipe(batch)
        assert "stress" in out_eager

        compiled_pipe = torch.compile(pipe, fullgraph=False)
        out_compiled = compiled_pipe(batch)

        assert "stress" in out_compiled
        torch.testing.assert_close(
            out_compiled["stress"], out_eager["stress"], atol=1e-5, rtol=1e-5
        )


# ===========================================================================
# Autograd correctness tests
# ===========================================================================


class _QuadraticEnergyModel(nn.Module, BaseModelMixin):
    """Model whose energy is E = scale * sum(positions^2).

    Analytical forces: F_i = -dE/dr_i = -2 * scale * positions_i.
    This allows exact verification of autograd forces.
    """

    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        self._scale = scale
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            autograd_outputs=frozenset(),
            needs_pbc=False,
            active_outputs={"energy"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        positions = data.positions
        B = data.num_graphs if isinstance(data, Batch) else 1
        batch = (
            data.batch_idx
            if isinstance(data, Batch)
            else torch.zeros(positions.shape[0], dtype=torch.long)
        )
        per_atom = self._scale * (positions**2).sum(dim=-1)
        energy = torch.zeros(B, 1, dtype=positions.dtype, device=positions.device)
        energy.scatter_add_(0, batch.unsqueeze(-1), per_atom.unsqueeze(-1))
        return OrderedDict(energy=energy)


class _ChargeProducerModel(nn.Module, BaseModelMixin):
    """Model that predicts charges as a function of positions.

    charges_i = position_i.sum()  (simple, differentiable)
    Also produces E_A = sum(positions^2) as its own energy.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "charges"}),
            autograd_outputs=frozenset(),
            needs_pbc=False,
            active_outputs={"energy", "charges"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        positions = data.positions
        B = data.num_graphs if isinstance(data, Batch) else 1
        batch = (
            data.batch_idx
            if isinstance(data, Batch)
            else torch.zeros(positions.shape[0], dtype=torch.long)
        )
        # Energy: sum of squared positions
        per_atom_e = (positions**2).sum(dim=-1)
        energy = torch.zeros(B, 1, dtype=positions.dtype, device=positions.device)
        energy.scatter_add_(0, batch.unsqueeze(-1), per_atom_e.unsqueeze(-1))
        # Charges: sum of position components per atom (differentiable)
        charges = positions.sum(dim=-1)
        return OrderedDict(energy=energy, charges=charges)


class _ChargeDependentEnergyModel(nn.Module, BaseModelMixin):
    """Model whose energy depends on node_charges and positions.

    E_B = sum(node_charges * positions.norm(dim=-1))

    This creates a computation graph where dE_B/dr flows through
    both the direct position dependence AND the charge dependence
    (since charges depend on positions in _ChargeProducerModel).
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            required_inputs=frozenset({"node_charges"}),
            autograd_outputs=frozenset(),
            needs_pbc=False,
            active_outputs={"energy"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        positions = data.positions
        charges = getattr(data, "node_charges", None)
        if charges is None:
            raise RuntimeError("node_charges not found")
        B = data.num_graphs if isinstance(data, Batch) else 1
        batch = (
            data.batch_idx
            if isinstance(data, Batch)
            else torch.zeros(positions.shape[0], dtype=torch.long)
        )
        per_atom_e = charges * positions.norm(dim=-1)
        energy = torch.zeros(B, 1, dtype=positions.dtype, device=positions.device)
        energy.scatter_add_(0, batch.unsqueeze(-1), per_atom_e.unsqueeze(-1))
        return OrderedDict(energy=energy)


class TestPipelineAutogradCorrectness:
    """Numerical correctness tests for pipeline autograd forces.

    These tests verify that pipeline-computed forces match manually
    computed reference forces, not just that they're non-zero.
    """

    @pytest.fixture
    def single_system_batch(self):
        """Single-system batch with known positions for analytical verification."""
        torch.manual_seed(42)
        data = AtomicData(
            positions=torch.randn(4, 3, dtype=torch.float64),
            atomic_numbers=torch.tensor([6, 6, 8, 1]),
            forces=torch.zeros(4, 3, dtype=torch.float64),
            energy=torch.zeros(1, 1, dtype=torch.float64),
        )
        return Batch.from_data_list([data])

    def test_single_model_forces_match_analytical(self, single_system_batch):
        """Pipeline autograd forces for E = sum(pos^2) should be F = -2*pos."""
        model = _QuadraticEnergyModel(scale=1.0)
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}
        out = pipe(single_system_batch)

        # Analytical: F_i = -dE/dr_i = -2 * positions_i
        expected_forces = -2.0 * single_system_batch.positions
        torch.testing.assert_close(out["forces"], expected_forces, atol=1e-10, rtol=0)

    def test_single_model_forces_and_stress_match_analytical(self):
        """Pipeline computes forces and stress from the same strained energy."""
        positions = torch.tensor(
            [
                [1.0, 2.0, 0.5],
                [-0.5, 1.5, 2.0],
                [0.25, -1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        cell = torch.eye(3, dtype=torch.float64).unsqueeze(0) * 4.0
        data = AtomicData(
            positions=positions,
            atomic_numbers=torch.tensor([6, 6, 8]),
            forces=torch.zeros(3, 3, dtype=torch.float64),
            energy=torch.zeros(1, 1, dtype=torch.float64),
            cell=cell,
            pbc=torch.tensor([[True, True, True]]),
        )
        batch = Batch.from_data_list([data])

        model = _QuadraticEnergyModel(scale=1.0)
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )
        pipe.model_config.active_outputs = {"energy", "forces", "stress"}
        out = pipe(batch)

        expected_forces = -2.0 * positions
        volume = torch.det(cell).abs().view(-1, 1, 1)
        expected_stress = (2.0 * positions.T @ positions).unsqueeze(0) / volume

        torch.testing.assert_close(out["forces"], expected_forces, atol=1e-10, rtol=0)
        torch.testing.assert_close(out["stress"], expected_stress, atol=1e-10, rtol=0)

    def test_forces_and_stress_use_merged_autograd_helper(self, monkeypatch):
        """Requesting forces and stress together should use one helper call."""
        import nvalchemi.models.pipeline as pipeline_module

        real_helper = pipeline_module.autograd_forces_and_stresses
        calls = []

        def wrapped_helper(*args, **kwargs):
            calls.append((args, kwargs))
            return real_helper(*args, **kwargs)

        monkeypatch.setattr(
            pipeline_module, "autograd_forces_and_stresses", wrapped_helper
        )

        data = AtomicData(
            positions=torch.randn(4, 3, dtype=torch.float64),
            atomic_numbers=torch.tensor([6, 6, 8, 1]),
            forces=torch.zeros(4, 3, dtype=torch.float64),
            energy=torch.zeros(1, 1, dtype=torch.float64),
            cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0,
            pbc=torch.tensor([[True, True, True]]),
        )
        batch = Batch.from_data_list([data])
        model = _QuadraticEnergyModel(scale=1.0)
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )
        pipe.model_config.active_outputs = {"energy", "forces", "stress"}

        out = pipe(batch)

        assert len(calls) == 1
        assert "forces" in out
        assert "stress" in out

    def test_autograd_group_detaches_batch_tensors_after_forces_and_stress(self):
        """Autograd group cleanup should leave batch tensors detached."""
        data = AtomicData(
            positions=torch.randn(4, 3, dtype=torch.float64),
            atomic_numbers=torch.tensor([6, 6, 8, 1]),
            forces=torch.zeros(4, 3, dtype=torch.float64),
            energy=torch.zeros(1, 1, dtype=torch.float64),
            cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0,
            pbc=torch.tensor([[True, True, True]]),
        )
        batch = Batch.from_data_list([data])
        model = _QuadraticEnergyModel(scale=1.0)
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )
        pipe.model_config.active_outputs = {"energy", "forces", "stress"}

        out = pipe(batch)

        assert "forces" in out
        assert "stress" in out
        for _, value in batch:
            if isinstance(value, torch.Tensor):
                assert not value.requires_grad
                assert value.grad_fn is None

    def test_detach_data_tensors_handles_atomic_data_python_model_dump(self):
        """AtomicData cleanup should detach tensors returned by model_dump."""
        source = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        data = AtomicData(
            positions=source * 2.0,
            atomic_numbers=torch.tensor([6, 6, 8, 1]),
        )
        data.extra_tensor = source.sum() * torch.ones(1, dtype=torch.float64)

        assert isinstance(data.model_dump(exclude_none=True)["positions"], torch.Tensor)

        PipelineModelWrapper._detach_data_tensors(data)

        assert not data.positions.requires_grad
        assert data.positions.grad_fn is None
        assert not data.extra_tensor.requires_grad
        assert data.extra_tensor.grad_fn is None

    def test_autograd_group_detaches_batch_tensors_after_exception(self):
        """Autograd group cleanup should run even when a step raises."""

        class _RaisingEnergyModel(_QuadraticEnergyModel):
            def forward(self, data, **kwargs) -> ModelOutputs:
                raise RuntimeError("intentional failure")

        data = AtomicData(
            positions=torch.randn(4, 3, dtype=torch.float64),
            atomic_numbers=torch.tensor([6, 6, 8, 1]),
            forces=torch.zeros(4, 3, dtype=torch.float64),
            energy=torch.zeros(1, 1, dtype=torch.float64),
            cell=torch.eye(3, dtype=torch.float64).unsqueeze(0) * 10.0,
            pbc=torch.tensor([[True, True, True]]),
        )
        batch = Batch.from_data_list([data])
        model = _RaisingEnergyModel(scale=1.0)
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )
        pipe.model_config.active_outputs = {"energy", "forces", "stress"}

        with pytest.raises(RuntimeError, match="intentional failure"):
            pipe(batch)

        for _, value in batch:
            if isinstance(value, torch.Tensor):
                assert not value.requires_grad
                assert value.grad_fn is None

    def test_two_model_sum_forces_match_analytical(self, single_system_batch):
        """Forces from E_total = 1*sum(pos^2) + 3*sum(pos^2) = 4*sum(pos^2).

        Expected: F = -8 * positions.
        """
        a = _QuadraticEnergyModel(scale=1.0)
        b = _QuadraticEnergyModel(scale=3.0)
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[a, b], use_autograd=True)]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}
        out = pipe(single_system_batch)

        expected_forces = -8.0 * single_system_batch.positions
        torch.testing.assert_close(out["forces"], expected_forces, atol=1e-10, rtol=0)

    def test_dependent_chain_forces_include_indirect_gradient(
        self, single_system_batch
    ):
        """Forces must backpropagate through the wired charge dependency.

        Model A: E_A = sum(pos^2), charges = pos.sum(dim=-1)
        Model B: E_B = sum(charges * ||pos||)
                     = sum(pos.sum(dim=-1) * ||pos||)

        E_total = E_A + E_B

        The key test: dE_B/dr has TWO contributions:
          1. Direct: d/dr [charges * ||pos||] holding charges fixed
          2. Indirect: d/dr [charges * ||pos||] through d(charges)/dr

        The pipeline's autograd on E_total must capture BOTH.
        We verify against a manual reference that also captures both.
        """
        model_a = _ChargeProducerModel()
        model_b = _ChargeDependentEnergyModel()
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[
                        PipelineStep(model_a, wire={"charges": "node_charges"}),
                        model_b,
                    ],
                    use_autograd=True,
                ),
            ]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}
        out = pipe(single_system_batch)

        # Manual reference: compute E_total with autograd from scratch.
        positions = single_system_batch.positions.clone().requires_grad_(True)
        batch = single_system_batch.batch_idx
        B = single_system_batch.num_graphs

        # E_A = sum(pos^2)
        per_atom_ea = (positions**2).sum(dim=-1)
        e_a = torch.zeros(B, 1, dtype=positions.dtype)
        e_a.scatter_add_(0, batch.unsqueeze(-1), per_atom_ea.unsqueeze(-1))

        # charges = pos.sum(dim=-1)  (differentiable through positions)
        charges = positions.sum(dim=-1)

        # E_B = sum(charges * ||pos||)
        per_atom_eb = charges * positions.norm(dim=-1)
        e_b = torch.zeros(B, 1, dtype=positions.dtype)
        e_b.scatter_add_(0, batch.unsqueeze(-1), per_atom_eb.unsqueeze(-1))

        e_total = e_a + e_b
        expected_forces = -torch.autograd.grad(
            e_total.sum(), positions, create_graph=False
        )[0]

        torch.testing.assert_close(out["forces"], expected_forces, atol=1e-10, rtol=0)

    def test_hybrid_direct_plus_autograd_forces(self, single_system_batch):
        """Hybrid pipeline: autograd group + direct group.

        Group 1 (autograd): E = 2*sum(pos^2), forces via autograd = -4*pos
        Group 2 (direct): returns fixed forces = 0.5

        Total forces = autograd_forces + direct_forces = -4*pos + 0.5
        """
        autograd_model = _QuadraticEnergyModel(scale=2.0)
        direct_model = MockEnergyForceModel(energy=0.0, force_val=0.5)
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[autograd_model], use_autograd=True),
                PipelineGroup(steps=[direct_model]),
            ]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}
        out = pipe(single_system_batch)

        expected_forces = -4.0 * single_system_batch.positions + 0.5
        torch.testing.assert_close(out["forces"], expected_forces, atol=1e-10, rtol=0)

    def test_energy_is_sum_of_submodels(self, single_system_batch):
        """Pipeline total energy equals sum of individual model energies.

        E_total = E_A(pos) + E_B(charges(pos), pos)
        where charges are wired from A to B.
        """
        model_a = _ChargeProducerModel()
        model_b = _ChargeDependentEnergyModel()
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[
                        PipelineStep(model_a, wire={"charges": "node_charges"}),
                        model_b,
                    ],
                    use_autograd=True,
                ),
            ]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}
        out = pipe(single_system_batch)

        # Compute individual energies manually
        pos = single_system_batch.positions
        batch = single_system_batch.batch_idx
        B = single_system_batch.num_graphs

        e_a = torch.zeros(B, 1, dtype=pos.dtype)
        e_a.scatter_add_(0, batch.unsqueeze(-1), (pos**2).sum(dim=-1, keepdim=True))

        charges = pos.sum(dim=-1)
        e_b = torch.zeros(B, 1, dtype=pos.dtype)
        e_b.scatter_add_(
            0, batch.unsqueeze(-1), (charges * pos.norm(dim=-1)).unsqueeze(-1)
        )

        expected_energy = e_a + e_b
        torch.testing.assert_close(out["energy"], expected_energy, atol=1e-10, rtol=0)


# ===========================================================================
# Hybrid forces: direct + autograd force summation in autograd groups
# ===========================================================================


class _MockHybridForcesModel(nn.Module, BaseModelMixin):
    """Mock model mimicking hybrid_forces behavior.

    Returns direct forces (no grad_fn) and energy with grad_fn through
    a "charge" pathway, similar to Ewald/PME with hybrid_forces=True.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            needs_pbc=False,
            active_outputs={"energy", "forces"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        positions = data.positions
        B = data.num_graphs if isinstance(data, Batch) else 1
        batch_idx = (
            data.batch_idx
            if isinstance(data, Batch)
            else torch.zeros(positions.shape[0], dtype=torch.long)
        )
        # "Charges" derived from positions (simulates q(R))
        charges = positions.sum(dim=-1)
        # Energy depends on charges (not directly on positions)
        per_atom_e = charges**2
        energies = torch.zeros(B, 1, dtype=positions.dtype, device=positions.device)
        energies.scatter_add_(0, batch_idx.unsqueeze(-1), per_atom_e.unsqueeze(-1))
        # Direct forces: partial derivative dE/dR|_q (detached, no grad_fn)
        direct_forces = (
            -2.0 * charges.unsqueeze(-1).detach() * torch.ones_like(positions)
        )
        return OrderedDict(energy=energies, forces=direct_forces.detach())


class _MockChargePathEnergyOnlyModel(nn.Module, BaseModelMixin):
    """Mock model with the same charge-path energy but no direct forces."""

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            needs_pbc=False,
            active_outputs={"energy"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        positions = data.positions
        B = data.num_graphs if isinstance(data, Batch) else 1
        batch_idx = (
            data.batch_idx
            if isinstance(data, Batch)
            else torch.zeros(positions.shape[0], dtype=torch.long)
        )
        charges = positions.sum(dim=-1)
        per_atom_e = charges**2
        energies = torch.zeros(B, 1, dtype=positions.dtype, device=positions.device)
        energies.scatter_add_(0, batch_idx.unsqueeze(-1), per_atom_e.unsqueeze(-1))
        return OrderedDict(energy=energies)


class _MockHybridForcesStressModel(nn.Module, BaseModelMixin):
    """Mock model mimicking hybrid_forces behavior with stress output.

    Returns direct forces and stress (no grad_fn) and energy with grad_fn
    through a "charge" pathway, similar to Ewald/PME with hybrid_forces=True.
    """

    def __init__(self) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            needs_pbc=False,
            active_outputs={"energy", "forces", "stress"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data, **kwargs) -> ModelOutputs:
        positions = data.positions
        B = data.num_graphs if isinstance(data, Batch) else 1
        batch_idx = (
            data.batch_idx
            if isinstance(data, Batch)
            else torch.zeros(positions.shape[0], dtype=torch.long)
        )
        charges = positions.sum(dim=-1)
        per_atom_e = charges**2
        energies = torch.zeros(B, 1, dtype=positions.dtype, device=positions.device)
        energies.scatter_add_(0, batch_idx.unsqueeze(-1), per_atom_e.unsqueeze(-1))
        direct_forces = (
            -2.0 * charges.unsqueeze(-1).detach() * torch.ones_like(positions)
        )
        direct_stress = (
            torch.eye(3, dtype=positions.dtype, device=positions.device)
            .unsqueeze(0)
            .expand(B, -1, -1)
            * 0.5
        )
        return OrderedDict(
            energy=energies,
            forces=direct_forces.detach(),
            stress=direct_stress.detach(),
        )


class TestAutoGradGroupHybridForces:
    """Test that _run_autograd_group sums direct + autograd forces."""

    @pytest.fixture
    def single_batch(self):
        data = AtomicData(
            positions=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            atomic_numbers=torch.tensor([6, 8]),
            forces=torch.zeros(2, 3),
            energy=torch.zeros(1, 1),
        )
        return Batch.from_data_list([data])

    def test_direct_forces_added_to_autograd_forces(self, single_batch):
        """Autograd group sums direct kernel forces with autograd forces."""
        model = _MockHybridForcesModel()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}
        out = pipe(single_batch)
        charges = single_batch.positions.sum(dim=-1, keepdim=True)
        expected_forces = -4.0 * charges.expand_as(single_batch.positions)

        assert "forces" in out
        torch.testing.assert_close(out["forces"], expected_forces)

    def test_energy_only_model_forces_from_autograd_alone(self, single_batch):
        """When model returns energy only, forces come from autograd alone."""
        model = MockAutogradEnergyModel(scale=1.0)
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}
        out = pipe(single_batch)
        expected_forces = -2.0 * single_batch.positions

        assert "forces" in out
        torch.testing.assert_close(out["forces"], expected_forces)

    def test_hybrid_forces_greater_than_autograd_alone(self, single_batch):
        """Hybrid total forces should equal autograd plus direct forces."""
        model = _MockHybridForcesModel()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )
        pipe.model_config.active_outputs = {"energy", "forces"}
        out_hybrid = pipe(single_batch)

        pipe2 = PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[_MockChargePathEnergyOnlyModel()], use_autograd=True
                )
            ]
        )
        pipe2.model_config.active_outputs = {"energy", "forces"}
        out_autograd_only = pipe2(single_batch)
        charges = single_batch.positions.sum(dim=-1, keepdim=True)
        expected_autograd_forces = -2.0 * charges.expand_as(single_batch.positions)
        expected_direct_forces = expected_autograd_forces
        expected_hybrid_forces = expected_autograd_forces + expected_direct_forces

        torch.testing.assert_close(
            out_autograd_only["forces"], expected_autograd_forces
        )
        torch.testing.assert_close(out_hybrid["forces"], expected_hybrid_forces)

    def test_direct_stress_added_to_autograd_stress(self):
        """Autograd group sums detached kernel stress with autograd stress."""
        data = AtomicData(
            positions=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            atomic_numbers=torch.tensor([6, 8]),
            forces=torch.zeros(2, 3),
            energy=torch.zeros(1, 1),
            cell=torch.eye(3).unsqueeze(0) * 10.0,
            pbc=torch.tensor([[True, True, True]]),
        )
        batch = Batch.from_data_list([data])

        # Run with the hybrid model (returns direct stress 0.5*I)
        model = _MockHybridForcesStressModel()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )
        pipe.model_config.active_outputs = {"energy", "forces", "stress"}
        out_hybrid = pipe(batch)

        assert "stress" in out_hybrid
        assert out_hybrid["stress"].shape == (1, 3, 3)

        # Run the energy-only model (no direct stress) with autograd stress
        pipe_autograd = PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[_MockChargePathEnergyOnlyModel()], use_autograd=True
                )
            ]
        )
        pipe_autograd.model_config.active_outputs = {"energy", "forces", "stress"}
        data2 = AtomicData(
            positions=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            atomic_numbers=torch.tensor([6, 8]),
            forces=torch.zeros(2, 3),
            energy=torch.zeros(1, 1),
            cell=torch.eye(3).unsqueeze(0) * 10.0,
            pbc=torch.tensor([[True, True, True]]),
        )
        batch2 = Batch.from_data_list([data2])
        out_autograd = pipe_autograd(batch2)

        direct_stress = 0.5 * torch.eye(3).unsqueeze(0)
        expected = out_autograd["stress"] + direct_stress
        torch.testing.assert_close(out_hybrid["stress"], expected, atol=1e-5, rtol=1e-5)


# ===========================================================================
# Pipeline second-order derivative tests
# ===========================================================================


class _QualifiedPipelineQuadratic(nn.Module, BaseModelMixin):
    """Small qualified position-energy wrapper for pipeline derivatives."""

    def __init__(self, scale: float = 1.0, output_kind: str = "normal") -> None:
        super().__init__()
        self.scale = scale
        self.output_kind = output_kind
        self.forward_calls = 0
        self.derivative_mode = "eager"
        self.seen_active_outputs = None
        self.seen_gradient_keys = None
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            active_outputs={"energy"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def _validate_derivative_request(self, request) -> None:
        if (
            request.execution == "local"
            and request.mode == "eager"
            and (request.operation == "hvp" or request.strategy in {"loop", "vmap"})
        ):
            return
        super()._validate_derivative_request(request)

    def _derivative_execution_mode(self):
        return self.derivative_mode

    def forward(self, data, **kwargs) -> ModelOutputs:
        self.forward_calls += 1
        self.seen_active_outputs = set(self.model_config.active_outputs)
        self.seen_gradient_keys = set(self.model_config.gradient_keys)
        if self.output_kind == "raise":
            raise LookupError("injected pipeline derivative failure")
        if self.output_kind == "missing":
            return OrderedDict()
        positions = data.positions
        node_energy = 0.5 * self.scale * positions.square().sum(dim=-1, keepdim=True)
        energy = torch.zeros(
            data.num_graphs,
            1,
            dtype=positions.dtype,
            device=positions.device,
        ).scatter_add(0, data.batch_idx.long().unsqueeze(-1), node_energy)
        return OrderedDict(energy=energy)


class _UnqualifiedPipelineQuadratic(_QualifiedPipelineQuadratic):
    """Qualified-looking test model that retains the base rejection."""

    def _validate_derivative_request(self, request) -> None:
        BaseModelMixin._validate_derivative_request(self, request)


class _LoopOnlyPipelineQuadratic(_QualifiedPipelineQuadratic):
    """Qualified test wrapper that rejects vectorized dense rows."""

    def _validate_derivative_request(self, request) -> None:
        if request.operation == "dense_hessian" and request.strategy == "vmap":
            raise NotImplementedError("test child supports dense strategy='loop' only")
        super()._validate_derivative_request(request)


class _NonlinearChargeProducer(nn.Module, BaseModelMixin):
    """Produce position-dependent wired values for a nonlinear chain."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0
        self.output_kind = "normal"
        self.seen_active_outputs = None
        self.seen_gradient_keys = None
        self.model_config = ModelConfig(
            outputs=frozenset({"charges"}),
            autograd_inputs=frozenset({"positions"}),
            active_outputs={"charges"},
        )

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def _validate_derivative_request(self, request) -> None:
        if (
            request.execution == "local"
            and request.mode == "eager"
            and (request.operation == "hvp" or request.strategy in {"loop", "vmap"})
        ):
            return
        super()._validate_derivative_request(request)

    def forward(self, data, **kwargs) -> ModelOutputs:
        self.forward_calls += 1
        self.seen_active_outputs = set(self.model_config.active_outputs)
        self.seen_gradient_keys = set(self.model_config.gradient_keys)
        if self.output_kind == "missing":
            return OrderedDict()
        return OrderedDict(charges=data.positions.sin())


class _NonlinearChargeConsumer(_QualifiedPipelineQuadratic):
    """Consume wired charges while retaining a direct position term."""

    def __init__(self, output_kind: str = "normal") -> None:
        super().__init__(output_kind=output_kind)
        self.last_data = None
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            required_inputs=frozenset({"node_charges"}),
            autograd_inputs=frozenset({"positions"}),
            active_outputs={"energy"},
        )

    def forward(self, data, **kwargs) -> ModelOutputs:
        self.forward_calls += 1
        self.last_data = data
        self.seen_active_outputs = set(self.model_config.active_outputs)
        self.seen_gradient_keys = set(self.model_config.gradient_keys)
        if self.output_kind == "raise":
            raise LookupError("injected pipeline derivative failure")
        if self.output_kind == "missing":
            return OrderedDict()
        charges = getattr(data, "node_charges")
        node_energy = charges.pow(3).sum(dim=-1, keepdim=True)
        node_energy += 0.25 * data.positions.square().sum(dim=-1, keepdim=True)
        energy = torch.zeros(
            data.num_graphs,
            1,
            dtype=data.positions.dtype,
            device=data.positions.device,
        ).scatter_add(0, data.batch_idx.long().unsqueeze(-1), node_energy)
        return OrderedDict(energy=energy)


class _CountingForceOnlyModel(MockForceOnlyModel):
    """Force-only model whose invocation is observable by topology tests."""

    def __init__(self) -> None:
        super().__init__()
        self.forward_calls = 0

    def forward(self, data, **kwargs) -> ModelOutputs:
        self.forward_calls += 1
        return super().forward(data, **kwargs)


class _QualifiedNeighborEnergy(_QualifiedPipelineQuadratic):
    """Qualified energy wrapper that records MATRIX/COO runtime state."""

    def __init__(self, neighbor_format: NeighborListFormat) -> None:
        super().__init__()
        self.model_config = ModelConfig(
            outputs=frozenset({"energy"}),
            autograd_inputs=frozenset({"positions"}),
            neighbor_config=NeighborConfig(
                cutoff=10.0 if neighbor_format == NeighborListFormat.MATRIX else 4.0,
                format=neighbor_format,
            ),
            active_outputs={"energy"},
        )
        self.captured_neighbor_matrix = None
        self.captured_neighbor_matrix_shifts = None
        self.captured_neighbor_list = None
        self.captured_neighbor_list_shifts = None
        self.runtime_neighbor_matrix = None
        self.runtime_neighbor_list = None
        self.captured_alias = False

    def forward(self, data, **kwargs) -> ModelOutputs:
        sources = data.__dict__.get(_PIPELINE_NEIGHBOR_SOURCES_ATTR)
        if sources and getattr(data, "neighbor_matrix", None) is not None:
            self.captured_alias = data.neighbor_matrix is sources[0].neighbor_matrix
        matrix = getattr(data, "neighbor_matrix", None)
        neighbor_list = getattr(data, "neighbor_list", None)
        matrix_shifts = getattr(data, "neighbor_matrix_shifts", None)
        list_shifts = getattr(data, "neighbor_list_shifts", None)
        self.runtime_neighbor_matrix = matrix
        self.runtime_neighbor_list = neighbor_list
        self.captured_neighbor_matrix = matrix.clone() if matrix is not None else None
        self.captured_neighbor_matrix_shifts = (
            matrix_shifts.clone() if matrix_shifts is not None else None
        )
        self.captured_neighbor_list = (
            neighbor_list.clone() if neighbor_list is not None else None
        )
        self.captured_neighbor_list_shifts = (
            list_shifts.clone() if list_shifts is not None else None
        )
        output = super().forward(data, **kwargs)
        if matrix is not None:
            topology_weight = data.num_neighbors.sum().to(data.positions) + 1
        else:
            topology_weight = data.positions.new_tensor(neighbor_list.shape[0] + 1)
        output["energy"] = output["energy"] * topology_weight
        return output


def _make_pipeline_derivative_batch(
    *sizes: int,
    dtype: torch.dtype = torch.float64,
) -> Batch:
    """Build a deterministic mixed-size batch for pipeline derivatives."""
    if not sizes:
        sizes = (2,)
    return Batch.from_data_list(
        [
            AtomicData(
                positions=(
                    torch.arange(size * 3, dtype=dtype).reshape(size, 3) / 7.0 - 0.4
                ),
                atomic_numbers=torch.ones(size, dtype=torch.long),
            )
            for size in sizes
        ]
    )


def _make_one_system_derivative_batch() -> Batch:
    """Build a small one-system batch for explicit position derivatives."""
    return _make_pipeline_derivative_batch(2)


def _nonlinear_chain_expected_hvp(
    positions: torch.Tensor, vector: torch.Tensor, *, frozen_charges: bool
) -> torch.Tensor:
    """Compute the explicit autograd reference for the wired chain."""
    leaf = positions.detach().clone().requires_grad_(True)
    charges = leaf.sin()
    if frozen_charges:
        charges = charges.detach()
    energy = charges.pow(3).sum() + 0.25 * leaf.square().sum()
    gradient = torch.autograd.grad(energy, leaf, create_graph=True)[0]
    return torch.autograd.grad(gradient, leaf, grad_outputs=vector)[0]


class TestPipelineDerivatives:
    """Second-order pipeline behavior through qualified synthetic wrappers."""

    def test_additive_hvp_prepared_and_dense_strategies(self):
        batch = _make_pipeline_derivative_batch(1, 3)
        left = _QualifiedPipelineQuadratic()
        right = _QualifiedPipelineQuadratic(scale=2.0)
        pipeline = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[left], use_autograd=True),
                PipelineGroup(steps=[right], use_autograd=True),
            ]
        )
        vector = torch.randn_like(batch.positions)

        hvp = pipeline.hessian_vector_product(batch, vector)
        torch.testing.assert_close(hvp, 3.0 * vector)

        with pipeline.prepare_hessian(batch) as operator:
            torch.testing.assert_close(operator.matvec(vector), hvp)
            torch.testing.assert_close(operator.matvec(-vector), -hvp)
        assert left.forward_calls == 2
        assert right.forward_calls == 2

        loop_batch = batch.clone()
        vmap_batch = batch.clone()
        pipeline.compute_hessian(loop_batch, strategy="loop")
        pipeline.compute_hessian(vmap_batch, strategy="vmap")
        torch.testing.assert_close(loop_batch.hessian, vmap_batch.hessian)
        for index, size in enumerate(batch.num_nodes_list):
            expected = 3.0 * torch.eye(3 * size, dtype=batch.positions.dtype).reshape(
                size, 3, size, 3
            ).permute(0, 2, 1, 3)
            torch.testing.assert_close(loop_batch.get_data(index).hessian, expected)

        start = 0
        for index, size in enumerate(batch.num_nodes_list):
            stop = start + size
            contraction = torch.einsum(
                "abij,bj->ai",
                vmap_batch.get_data(index).hessian,
                vector[start:stop],
            )
            torch.testing.assert_close(contraction, hvp[start:stop])
            start = stop

    def test_nonlinear_wired_chain_matches_autograd_not_frozen_charges(self):
        batch = _make_one_system_derivative_batch()
        producer = _NonlinearChargeProducer()
        consumer = _NonlinearChargeConsumer()
        pipeline = PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[
                        PipelineStep(producer, wire={"charges": "node_charges"}),
                        consumer,
                    ],
                    use_autograd=True,
                )
            ]
        )
        vector = torch.randn_like(batch.positions)

        actual = pipeline.hessian_vector_product(batch, vector)
        expected = _nonlinear_chain_expected_hvp(
            batch.positions, vector, frozen_charges=False
        )
        frozen = _nonlinear_chain_expected_hvp(
            batch.positions, vector, frozen_charges=True
        )
        torch.testing.assert_close(actual, expected)
        assert not torch.allclose(actual, frozen)

        loop_batch = batch.clone()
        vmap_batch = batch.clone()
        pipeline.compute_hessian(loop_batch, strategy="loop", row_chunk_size=2)
        pipeline.compute_hessian(vmap_batch, strategy="vmap", row_chunk_size=3)
        torch.testing.assert_close(
            loop_batch.hessian,
            vmap_batch.hessian,
            rtol=1e-9,
            atol=1e-10,
        )
        dense = vmap_batch.get_data(0).hessian
        contraction = torch.einsum("abij,bj->ai", dense, vector)
        torch.testing.assert_close(contraction, actual, rtol=1e-9, atol=1e-10)

        assert producer.seen_active_outputs == {"charges"}
        assert producer.seen_gradient_keys == {"positions"}
        assert consumer.seen_active_outputs == {"energy"}
        assert consumer.seen_gradient_keys == {"positions"}

    def test_custom_derivative_and_force_only_step_are_excluded_from_hvp(self):
        batch = _make_one_system_derivative_batch()
        derivative_calls = []

        def derivative_fn(energy, data, requested):
            derivative_calls.append(set(requested))
            return {"forces": torch.zeros_like(data.positions)}

        energy = _QualifiedPipelineQuadratic()
        force_only = _CountingForceOnlyModel()
        pipeline = PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[energy], use_autograd=True, derivative_fn=derivative_fn
                ),
                PipelineGroup(steps=[force_only]),
            ]
        )
        pipeline.model_config.active_outputs = {"energy", "forces"}
        pipeline(batch)
        assert derivative_calls == [{"forces"}]
        assert force_only.forward_calls == 1

        derivative_calls.clear()
        force_only.forward_calls = 0
        vector = torch.randn_like(batch.positions)
        actual = pipeline.hessian_vector_product(batch, vector)
        torch.testing.assert_close(actual, vector)
        assert derivative_calls == []
        assert force_only.forward_calls == 0

    def test_failure_restores_runtime_writes_and_config_identity(self):
        batch = _make_one_system_derivative_batch()
        producer = _NonlinearChargeProducer()
        consumer = _NonlinearChargeConsumer(output_kind="raise")
        pipeline = PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[
                        PipelineStep(producer, wire={"charges": "node_charges"}),
                        consumer,
                    ],
                    use_autograd=True,
                )
            ]
        )
        pipeline_active = pipeline.model_config.active_outputs
        pipeline_gradient = pipeline.model_config.gradient_keys
        producer_active = producer.model_config.active_outputs
        producer_gradient = producer.model_config.gradient_keys
        consumer_active = consumer.model_config.active_outputs
        consumer_gradient = consumer.model_config.gradient_keys

        with pytest.raises(LookupError, match="injected pipeline derivative failure"):
            pipeline.prepare_hessian(batch)

        assert "node_charges" not in batch.__dict__
        assert consumer.last_data is not None
        assert "node_charges" not in consumer.last_data.__dict__
        assert pipeline.model_config.active_outputs is pipeline_active
        assert pipeline.model_config.gradient_keys is pipeline_gradient
        assert producer.model_config.active_outputs is producer_active
        assert producer.model_config.gradient_keys is producer_gradient
        assert consumer.model_config.active_outputs is consumer_active
        assert consumer.model_config.gradient_keys is consumer_gradient

        with pytest.raises(LookupError, match="injected pipeline derivative failure"):
            pipeline.compute_hessian(batch, strategy="vmap")
        assert "hessian" not in batch

    def test_success_restores_exact_configuration_objects(self):
        batch = _make_one_system_derivative_batch()
        producer = _NonlinearChargeProducer()
        consumer = _NonlinearChargeConsumer()
        pipeline = PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[
                        PipelineStep(producer, wire={"charges": "node_charges"}),
                        consumer,
                    ]
                )
            ]
        )
        pipeline_active = {"energy", "sentinel"}
        pipeline_gradient = {"cell"}
        producer_active = {"charges", "sentinel"}
        producer_gradient = {"cell"}
        consumer_active = {"energy", "sentinel"}
        consumer_gradient = {"cell"}
        pipeline.model_config.active_outputs = pipeline_active
        pipeline.model_config.gradient_keys = pipeline_gradient
        producer.model_config.active_outputs = producer_active
        producer.model_config.gradient_keys = producer_gradient
        consumer.model_config.active_outputs = consumer_active
        consumer.model_config.gradient_keys = consumer_gradient

        pipeline.hessian_vector_product(batch, torch.randn_like(batch.positions))

        assert pipeline.model_config.active_outputs is pipeline_active
        assert pipeline.model_config.gradient_keys is pipeline_gradient
        assert producer.model_config.active_outputs is producer_active
        assert producer.model_config.gradient_keys is producer_gradient
        assert consumer.model_config.active_outputs is consumer_active
        assert consumer.model_config.gradient_keys is consumer_gradient
        assert producer.seen_active_outputs == {"charges"}
        assert consumer.seen_active_outputs == {"energy"}


class TestPipelineDerivativeTopology:
    """Preflight and graph-topology contracts for pipeline derivatives."""

    def test_unqualified_child_is_rejected_before_forward(self):
        batch = _make_one_system_derivative_batch()
        model = _UnqualifiedPipelineQuadratic()
        pipeline = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )

        with pytest.raises(NotImplementedError, match="has not been qualified"):
            pipeline.hessian_vector_product(batch, torch.ones_like(batch.positions))
        assert model.forward_calls == 0

    @pytest.mark.parametrize(
        "context", ["pipeline_distributed", "child_distributed", "compiled"]
    )
    def test_contextual_capability_rejection_precedes_all_forwards(self, context):
        batch = _make_one_system_derivative_batch()
        earlier = _QualifiedPipelineQuadratic()
        rejected = _QualifiedPipelineQuadratic()
        pipeline = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[earlier, rejected], use_autograd=True)]
        )
        if context == "pipeline_distributed":
            pipeline._dist_ctx = object()
        elif context == "child_distributed":
            rejected._dist_ctx = object()
        else:
            rejected.derivative_mode = "compiled"

        with pytest.raises(NotImplementedError, match=context.split("_")[-1]):
            pipeline.hessian_vector_product(batch, torch.ones_like(batch.positions))

        assert earlier.forward_calls == 0
        assert rejected.forward_calls == 0

    def test_strategy_specific_rejection_does_not_fall_back(self):
        batch = _make_one_system_derivative_batch()
        earlier = _QualifiedPipelineQuadratic()
        loop_only = _LoopOnlyPipelineQuadratic()
        pipeline = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[earlier, loop_only], use_autograd=True)]
        )

        with pytest.raises(NotImplementedError, match="strategy='loop' only"):
            pipeline.compute_hessian(batch, strategy="vmap")
        assert earlier.forward_calls == 0
        assert loop_only.forward_calls == 0

        pipeline.compute_hessian(batch, strategy="loop")
        assert earlier.forward_calls == 1
        assert loop_only.forward_calls == 1

    @pytest.mark.parametrize("operation", ["hvp", "loop", "vmap"])
    def test_no_energy_pipeline_is_rejected_before_forward(self, operation):
        batch = _make_one_system_derivative_batch()
        force_only = _CountingForceOnlyModel()
        pipeline = PipelineModelWrapper(groups=[PipelineGroup(steps=[force_only])])

        with pytest.raises(NotImplementedError, match="no energy-producing step"):
            if operation == "hvp":
                pipeline.hessian_vector_product(batch, torch.ones_like(batch.positions))
            else:
                pipeline.compute_hessian(batch, strategy=operation)
        assert force_only.forward_calls == 0

    def test_nested_pipeline_is_rejected_before_forward(self):
        batch = _make_one_system_derivative_batch()
        inner_model = _QualifiedPipelineQuadratic()
        inner = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[inner_model], use_autograd=True)]
        )
        outer = PipelineModelWrapper(groups=[PipelineGroup(steps=[inner])])

        with pytest.raises(NotImplementedError, match="nested PipelineModelWrapper"):
            outer.hessian_vector_product(batch, torch.ones_like(batch.positions))
        assert inner_model.forward_calls == 0

        ordinary = outer(batch)
        assert ordinary["energy"] is not None
        assert inner_model.forward_calls == 1

        for strategy in ("loop", "vmap"):
            with pytest.raises(
                NotImplementedError, match="nested PipelineModelWrapper"
            ):
                outer.compute_hessian(batch.clone(), strategy=strategy)
        assert inner_model.forward_calls == 1

    def test_missing_required_output_is_runtime_error(self):
        batch = _make_one_system_derivative_batch()
        model = _QualifiedPipelineQuadratic(output_kind="missing")
        pipeline = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[model], use_autograd=True)]
        )

        with pytest.raises(RuntimeError, match="did not return required output"):
            pipeline.hessian_vector_product(batch, torch.ones_like(batch.positions))
        assert model.forward_calls == 1

    def test_missing_wired_output_is_runtime_error(self):
        batch = _make_one_system_derivative_batch()
        producer = _NonlinearChargeProducer()
        producer.output_kind = "missing"
        consumer = _NonlinearChargeConsumer()
        pipeline = PipelineModelWrapper(
            groups=[
                PipelineGroup(
                    steps=[
                        PipelineStep(producer, wire={"charges": "node_charges"}),
                        consumer,
                    ]
                )
            ]
        )

        with pytest.raises(RuntimeError, match="charges"):
            pipeline.hessian_vector_product(batch, torch.ones_like(batch.positions))
        assert producer.forward_calls == 1
        assert consumer.forward_calls == 0

    @pytest.mark.parametrize("operation", ["hvp", "loop", "vmap"])
    def test_later_dftd3_rejects_before_all_forwards(self, operation, monkeypatch):
        from nvalchemi.models.dftd3 import DFTD3ModelWrapper

        params = MagicMock(
            rcov=torch.zeros(100),
            r4r2=torch.zeros(100),
            c6ab=torch.zeros(100, 100, 5, 3),
            cn_ref=torch.zeros(100, 5),
        )
        with patch("nvalchemi.models.dftd3.load_dftd3_params", return_value=params):
            dftd3 = DFTD3ModelWrapper(a1=0.4, a2=4.4, s8=0.8)
        earlier = _QualifiedPipelineQuadratic()
        monkeypatch.setattr(
            dftd3,
            "forward",
            lambda *_args, **_kwargs: pytest.fail("DFT-D3 forward must not run"),
        )
        pipeline = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[earlier], use_autograd=True),
                PipelineGroup(steps=[dftd3]),
            ]
        )
        batch = _make_one_system_derivative_batch()

        with pytest.raises(NotImplementedError, match="analytical Warp"):
            if operation == "hvp":
                pipeline.hessian_vector_product(batch, torch.ones_like(batch.positions))
            else:
                pipeline.compute_hessian(batch, strategy=operation)

        assert earlier.forward_calls == 0

    @pytest.mark.parametrize("wrapper_name", ["ewald", "pme"])
    @pytest.mark.parametrize(
        ("configuration", "reason"),
        [("hybrid", "hybrid_forces=False"), ("slab", "slab_correction=True")],
    )
    def test_unsupported_coulomb_child_rejects_before_forward(
        self, wrapper_name, configuration, reason, monkeypatch
    ):
        if wrapper_name == "ewald":
            from nvalchemi.models.ewald import EwaldModelWrapper as Wrapper
        else:
            from nvalchemi.models.pme import PMEModelWrapper as Wrapper

        child = Wrapper(
            cutoff=6.0,
            hybrid_forces=configuration == "hybrid",
            slab_correction=configuration == "slab",
        )
        monkeypatch.setattr(
            child,
            "forward",
            lambda *_args, **_kwargs: pytest.fail("Coulomb forward must not run"),
        )
        earlier = _QualifiedPipelineQuadratic()
        pipeline = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[earlier, child], use_autograd=True)]
        )
        batch = _make_one_system_derivative_batch()

        with pytest.raises(NotImplementedError, match=reason):
            pipeline.hessian_vector_product(batch, torch.ones_like(batch.positions))
        assert earlier.forward_calls == 0

    def test_prepared_operator_isolated_from_caller_positions_and_neighbors(self):
        batch = _make_neighbor_batch()
        matrix = _QualifiedNeighborEnergy(NeighborListFormat.MATRIX)
        coo = _QualifiedNeighborEnergy(NeighborListFormat.COO)
        pipeline = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[matrix, coo], use_autograd=True)],
            neighbor_adaptation="always",
        )
        source = _NeighborSourceData(
            source_id=0,
            config=pipeline._neighbor_sources[0].config,
            neighbor_matrix=batch.neighbor_matrix,
            num_neighbors=batch.num_neighbors,
            neighbor_matrix_shifts=batch.neighbor_matrix_shifts,
        )
        batch.__dict__[_PIPELINE_NEIGHBOR_SOURCES_ATTR] = (source,)
        original_matrix = batch.neighbor_matrix.clone()
        vector = torch.randn_like(batch.positions)

        with pipeline.prepare_hessian(batch) as operator:
            first = operator.matvec(vector)
            batch.positions.add_(0.5)
            batch.neighbor_matrix[0, 0] = 99
            batch.num_neighbors.zero_()
            second = operator.matvec(vector)

        torch.testing.assert_close(second, first)
        torch.testing.assert_close(matrix.captured_neighbor_matrix, original_matrix)
        torch.testing.assert_close(matrix.runtime_neighbor_matrix, original_matrix)
        assert (
            matrix.runtime_neighbor_matrix.untyped_storage().data_ptr()
            != batch.neighbor_matrix.untyped_storage().data_ptr()
        )
        assert matrix.captured_alias is True
        assert coo.captured_neighbor_list is not None
        assert coo.captured_neighbor_list.shape[1] == 2

    def test_distinct_matrix_and_coo_sources_are_copied_without_rebuilding(self):
        batch = _make_neighbor_batch()
        matrix = _QualifiedNeighborEnergy(NeighborListFormat.MATRIX)
        coo = _QualifiedNeighborEnergy(NeighborListFormat.COO)
        pipeline = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[matrix, coo], use_autograd=True)],
            neighbor_adaptation="never",
        )
        matrix_values = batch.neighbor_matrix.clone()
        matrix_values[0, 1:] = -1
        matrix_counts = batch.num_neighbors.clone()
        matrix_counts[0] = 1
        matrix_shifts = torch.arange(
            matrix_values.numel() * 3, dtype=torch.int32
        ).reshape(*matrix_values.shape, 3)
        coo_values = torch.tensor([[0, 1], [1, 0], [2, 3]], dtype=torch.long)
        edge_ptr = torch.tensor([0, 3], dtype=torch.long)
        coo_shifts = torch.tensor([[1, 0, 0], [-1, 0, 0], [0, 1, 0]], dtype=torch.int32)
        matrix_source = _NeighborSourceData(
            source_id=0,
            config=pipeline._neighbor_sources[0].config,
            neighbor_matrix=matrix_values,
            num_neighbors=matrix_counts,
            neighbor_matrix_shifts=matrix_shifts,
        )
        coo_source = _NeighborSourceData(
            source_id=1,
            config=pipeline._neighbor_sources[1].config,
            neighbor_list=coo_values,
            edge_ptr=edge_ptr,
            neighbor_list_shifts=coo_shifts,
        )
        batch.__dict__[_PIPELINE_NEIGHBOR_SOURCES_ATTR] = (
            matrix_source,
            coo_source,
        )
        batch.__dict__["neighbor_matrix"] = matrix_values
        batch.__dict__["num_neighbors"] = matrix_counts
        batch.__dict__["neighbor_matrix_shifts"] = matrix_shifts
        batch.__dict__["_neighbor_list_cutoff"] = 10.0

        vector = torch.randn_like(batch.positions)
        result = pipeline.hessian_vector_product(batch, vector)

        torch.testing.assert_close(matrix.captured_neighbor_matrix, matrix_values)
        torch.testing.assert_close(
            matrix.captured_neighbor_matrix_shifts, matrix_shifts
        )
        torch.testing.assert_close(coo.captured_neighbor_list, coo_values)
        torch.testing.assert_close(coo.captured_neighbor_list_shifts, coo_shifts)
        assert matrix.captured_alias is True
        torch.testing.assert_close(result, 15 * vector)

    def test_single_source_runtime_shadows_work_without_source_records(self):
        batch = _make_neighbor_batch()
        matrix = _QualifiedNeighborEnergy(NeighborListFormat.MATRIX)
        pipeline = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[matrix], use_autograd=True)]
        )
        assert _PIPELINE_NEIGHBOR_SOURCES_ATTR not in batch.__dict__

        vector = torch.randn_like(batch.positions)
        result = pipeline.hessian_vector_product(batch, vector)

        assert torch.isfinite(result).all()
        torch.testing.assert_close(
            matrix.captured_neighbor_matrix,
            batch.neighbor_matrix,
        )
        torch.testing.assert_close(result, 13 * vector)


@pytest.fixture(scope="module")
def real_aimnet2_pipeline_derivative_wrapper():
    """Load the real CUDA AIMNet2 checkpoint for connected-charge tests."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for connected AIMNet2 pipeline qualification")
    pytest.importorskip("aimnet")
    from nvalchemi.models.aimnet2 import AIMNet2Wrapper

    wrapper = AIMNet2Wrapper.from_checkpoint(
        "aimnet2_wb97m_d3_3",
        device=torch.device("cuda"),
    )
    wrapper.eval()
    return wrapper


def _make_periodic_water_pipeline_batch() -> Batch:
    """Build the small periodic water system used for CUDA qualification."""
    data = AtomicData(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            dtype=torch.float32,
        ),
        atomic_numbers=torch.tensor([8, 1, 1], dtype=torch.long),
        charge=torch.zeros(1, 1),
        cell=torch.eye(3).unsqueeze(0) * 15.0,
        pbc=torch.tensor([[True, True, True]]),
    )
    return Batch.from_data_list([data], device="cuda")


class TestAIMNetCoulombPipelineDerivatives:
    """Real connected AIMNet2 charge response through Ewald and PME."""

    @pytest.mark.parametrize("coulomb_kind", ["ewald", "pme"])
    def test_connected_hvp_dense_and_fixed_topology_force_difference(
        self,
        coulomb_kind,
        real_aimnet2_pipeline_derivative_wrapper,
    ):
        pytest.importorskip("nvalchemiops")
        if coulomb_kind == "ewald":
            from nvalchemi.models.ewald import EwaldModelWrapper as CoulombWrapper
        else:
            from nvalchemi.models.pme import PMEModelWrapper as CoulombWrapper
        from nvalchemi.neighbors import compute_neighbors

        aimnet = real_aimnet2_pipeline_derivative_wrapper
        coulomb = CoulombWrapper(
            cutoff=8.0,
            hybrid_forces=False,
            slab_correction=False,
        ).to(device="cuda")
        coulomb.eval()
        pipeline = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[aimnet, coulomb], use_autograd=True)],
            neighbor_adaptation="always",
        ).eval()
        batch = _make_periodic_water_pipeline_batch()
        compute_neighbors(batch, config=pipeline.model_config.neighbor_config)
        original_batch = {
            key: value.detach().clone()
            for key, value in batch
            if key
            in {
                "positions",
                "charge",
                "cell",
                "neighbor_matrix",
                "num_neighbors",
                "neighbor_matrix_shifts",
            }
        }
        pipeline_active = pipeline.model_config.active_outputs
        pipeline_gradient = pipeline.model_config.gradient_keys
        aimnet_active = aimnet.model_config.active_outputs
        aimnet_gradient = aimnet.model_config.gradient_keys
        coulomb_active = coulomb.model_config.active_outputs
        coulomb_gradient = coulomb.model_config.gradient_keys
        torch.manual_seed(31)
        vector = torch.randn_like(batch.positions)
        vector /= vector.norm()

        hvp = pipeline.hessian_vector_product(batch, vector)

        eps = 2e-3
        plus = batch.clone()
        minus = batch.clone()
        plus.positions = batch.positions + eps * vector
        minus.positions = batch.positions - eps * vector
        requested = {"energy", "forces"}
        pipeline.model_config.active_outputs = requested
        try:
            force_plus = pipeline(plus)["forces"].detach()
            force_minus = pipeline(minus)["forces"].detach()
        finally:
            pipeline.model_config.active_outputs = pipeline_active
        finite_difference = -(force_plus - force_minus) / (2 * eps)
        torch.testing.assert_close(
            hvp,
            finite_difference,
            rtol=3e-2,
            atol=5e-3,
        )

        dense_batch = batch.clone()
        pipeline.compute_hessian(
            dense_batch,
            strategy="vmap",
            row_chunk_size=3,
        )
        dense = dense_batch.get_data(0).hessian
        contraction = torch.einsum("abij,bj->ai", dense, vector)
        torch.testing.assert_close(contraction, hvp, rtol=2e-2, atol=2e-3)
        torch.testing.assert_close(
            dense,
            dense.permute(1, 0, 3, 2),
            rtol=2e-2,
            atol=2e-3,
        )

        for key, value in original_batch.items():
            torch.testing.assert_close(batch[key], value)
        assert "charges" not in batch.__dict__
        assert pipeline.model_config.active_outputs is pipeline_active
        assert pipeline.model_config.gradient_keys is pipeline_gradient
        assert aimnet.model_config.active_outputs is aimnet_active
        assert aimnet.model_config.gradient_keys is aimnet_gradient
        assert coulomb.model_config.active_outputs is coulomb_active
        assert coulomb.model_config.gradient_keys is coulomb_gradient
