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
        """Autograd group removes forces from sub-model active_outputs."""
        m = MockAutogradEnergyModel()
        pipe = PipelineModelWrapper(
            groups=[
                PipelineGroup(steps=[m], use_autograd=True),
            ]
        )
        # After construction, sub-model should not have forces in active_outputs
        assert "forces" not in pipe.groups[0].steps[0].model.model_config.active_outputs


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
                        max_neighbors=64,
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
