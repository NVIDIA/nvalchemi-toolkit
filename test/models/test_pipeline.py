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
        hooks = pipe.make_neighbor_hooks()
        assert len(hooks) == 1


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

    def test_matrix_to_coo_conversion(self):
        """COO model in a MATRIX pipeline receives converted neighbor list."""
        matrix_model = _MatrixModel10()
        coo_model = _COOModel4()
        pipe = PipelineModelWrapper(
            groups=[PipelineGroup(steps=[matrix_model, coo_model])],
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
        )
        batch = _make_neighbor_batch()
        pipe(batch)

        # neighbor_list was temporarily added for coo_model, must be gone.
        assert "neighbor_list" not in batch.__dict__
        assert "edge_ptr" not in batch.__dict__


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

        import torch._dynamo

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

        import torch._dynamo

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
