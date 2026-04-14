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
"""Comprehensive tests for ModelConfig, BaseModelMixin, and _utils.py.

Target: >=85% coverage on nvalchemi/models/base.py.
"""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch
from pydantic import ValidationError

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models._utils import autograd_forces, autograd_stresses, sum_outputs
from nvalchemi.models.base import (
    BaseModelMixin,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
from nvalchemi.models.demo import DemoModel, DemoModelWrapper

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_batch():
    """A minimal 2-system batch for testing."""
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


@pytest.fixture
def demo_model():
    """A DemoModelWrapper instance with default config."""
    return DemoModelWrapper(DemoModel())


# ===========================================================================
# ModelConfig tests
# ===========================================================================


class TestModelConfig:
    """Tests for the unified ModelConfig with frozen capability + mutable runtime fields."""

    def test_default_outputs(self):
        cfg = ModelConfig(needs_pbc=False)
        assert cfg.outputs == frozenset({"energy"})
        assert cfg.autograd_outputs == frozenset()
        assert cfg.autograd_inputs == frozenset({"positions"})
        assert cfg.required_inputs == frozenset()

    def test_custom_outputs(self):
        cfg = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress", "charges"}),
            autograd_outputs=frozenset({"forces", "stress"}),
            needs_pbc=False,
        )
        assert "charges" in cfg.outputs
        assert "forces" in cfg.autograd_outputs

    def test_frozen_immutability(self):
        """Capability fields use frozenset, so in-place mutation is not possible."""
        cfg = ModelConfig(needs_pbc=False)
        with pytest.raises(AttributeError):
            cfg.outputs.add("new_key")  # frozenset has no .add()

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            ModelConfig(needs_pbc=False, unknown_field=True)

    def test_needs_neighborlist_true(self):
        cfg = ModelConfig(
            needs_pbc=False,
            neighbor_config=NeighborConfig(cutoff=5.0),
        )
        assert cfg.needs_neighborlist is True

    def test_needs_neighborlist_false(self):
        cfg = ModelConfig(needs_pbc=False, neighbor_config=None)
        assert cfg.needs_neighborlist is False

    def test_json_serialization_roundtrip(self):
        cfg = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            autograd_outputs=frozenset({"forces"}),
            required_inputs=frozenset({"pbc"}),
            supports_pbc=True,
            needs_pbc=True,
            neighbor_config=NeighborConfig(cutoff=5.0, format=NeighborListFormat.COO),
        )
        json_str = cfg.model_dump_json()
        restored = ModelConfig.model_validate_json(json_str)
        assert restored.outputs == cfg.outputs
        assert restored.autograd_outputs == cfg.autograd_outputs
        assert restored.required_inputs == cfg.required_inputs
        assert restored.supports_pbc == cfg.supports_pbc
        assert restored.needs_pbc == cfg.needs_pbc
        assert restored.neighbor_config.cutoff == cfg.neighbor_config.cutoff

    def test_supports_pbc_defaults(self):
        cfg = ModelConfig(needs_pbc=False)
        assert cfg.supports_pbc is False

    def test_autograd_inputs_default(self):
        cfg = ModelConfig(needs_pbc=False)
        assert cfg.autograd_inputs == frozenset({"positions"})

    def test_autograd_inputs_custom(self):
        cfg = ModelConfig(
            needs_pbc=False,
            autograd_inputs=frozenset({"positions", "displacement"}),
        )
        assert "displacement" in cfg.autograd_inputs

    def test_defaults(self):
        config = ModelConfig()
        assert config.active_outputs == {"energy"}
        assert config.gradient_keys == set()

    def test_custom_active_outputs(self):
        config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            active_outputs={"energy", "forces", "stress"},
        )
        assert "stress" in config.active_outputs

    def test_mutable_active_outputs(self):
        config = ModelConfig()
        config.active_outputs = {"energy"}
        assert config.active_outputs == {"energy"}

    def test_gradient_keys(self):
        config = ModelConfig(gradient_keys={"positions", "cell"})
        assert "cell" in config.gradient_keys

    def test_empty_active_outputs(self):
        config = ModelConfig(active_outputs=set())
        assert config.active_outputs == set()

    def test_novel_property(self):
        """String-based active_outputs allows novel property names without schema changes."""
        config = ModelConfig(
            outputs=frozenset({"energy", "magnetic_moment"}),
            active_outputs={"energy", "magnetic_moment"},
        )
        assert "magnetic_moment" in config.active_outputs


# ===========================================================================
# NeighborConfig tests
# ===========================================================================


class TestNeighborConfig:
    def test_coo_format(self):
        nc = NeighborConfig(cutoff=5.0, format=NeighborListFormat.COO)
        assert nc.format == NeighborListFormat.COO

    def test_matrix_format(self):
        nc = NeighborConfig(
            cutoff=10.0,
            format=NeighborListFormat.MATRIX,
        )
        assert nc.format == NeighborListFormat.MATRIX

    def test_defaults(self):
        nc = NeighborConfig(cutoff=3.0)
        assert nc.format == NeighborListFormat.COO
        assert nc.half_list is False
        assert nc.skin == 0.0


# ===========================================================================
# NeighborListFormat tests
# ===========================================================================


class TestNeighborListFormat:
    def test_coo_value(self):
        assert NeighborListFormat.COO == "coo"

    def test_matrix_value(self):
        assert NeighborListFormat.MATRIX == "matrix"


# ===========================================================================
# BaseModelMixin tests (via DemoModelWrapper)
# ===========================================================================


class TestBaseModelMixinInputData:
    """Tests for BaseModelMixin.input_data()."""

    def test_basic_input_keys(self, demo_model):
        keys = demo_model.input_data()
        assert "positions" in keys
        assert "atomic_numbers" in keys

    def test_coo_neighbor_adds_edge_index(self):
        """When neighbor_config is COO, input_data includes edge_index."""

        class _CooModel(DemoModelWrapper):
            def __init__(self):
                super().__init__(DemoModel())
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy", "forces"}),
                    autograd_outputs=frozenset({"forces"}),
                    autograd_inputs=frozenset({"positions"}),
                    neighbor_config=NeighborConfig(
                        cutoff=5.0, format=NeighborListFormat.COO
                    ),
                    needs_pbc=False,
                )

        model = _CooModel()
        keys = model.input_data()
        assert "neighbor_list" in keys

    def test_matrix_neighbor_adds_keys(self):
        """When neighbor_config is MATRIX, input_data includes neighbor_matrix and num_neighbors."""

        class _MatrixModel(DemoModelWrapper):
            def __init__(self):
                super().__init__(DemoModel())
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy", "forces"}),
                    autograd_outputs=frozenset({"forces"}),
                    autograd_inputs=frozenset({"positions"}),
                    neighbor_config=NeighborConfig(
                        cutoff=5.0,
                        format=NeighborListFormat.MATRIX,
                    ),
                    needs_pbc=False,
                )

        model = _MatrixModel()
        keys = model.input_data()
        assert "neighbor_matrix" in keys
        assert "num_neighbors" in keys

    def test_needs_pbc_adds_pbc(self):
        class _PbcModel(DemoModelWrapper):
            def __init__(self):
                super().__init__(DemoModel())
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy"}),
                    needs_pbc=True,
                )

        model = _PbcModel()
        keys = model.input_data()
        assert "pbc" in keys

    def test_extra_inputs_from_config(self):
        class _ChargeModel(DemoModelWrapper):
            def __init__(self):
                super().__init__(DemoModel())
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy"}),
                    required_inputs=frozenset({"node_charges"}),
                    needs_pbc=False,
                )

        model = _ChargeModel()
        keys = model.input_data()
        assert "node_charges" in keys


class TestBaseModelMixinOutputData:
    """Tests for BaseModelMixin.output_data()."""

    def test_output_data_intersection(self, demo_model):
        """output_data() returns intersection of active_outputs and outputs."""
        demo_model.model_config.active_outputs = {"energy", "forces"}
        out = demo_model.output_data()
        assert out == {"energy", "forces"}

    def test_unsupported_key_warns(self, demo_model):
        """Requesting a key not in outputs warns."""
        demo_model.model_config.active_outputs = {"energy", "forces", "hessian"}
        with pytest.warns(UserWarning, match="hessian"):
            out = demo_model.output_data()
        assert "hessian" not in out

    def test_empty_active_outputs_returns_empty(self, demo_model):
        demo_model.model_config.active_outputs = set()
        out = demo_model.output_data()
        assert out == set()

    def test_novel_key_supported(self):
        """Novel keys in both outputs and active_outputs pass through."""

        class _NovelModel(DemoModelWrapper):
            def __init__(self):
                super().__init__(DemoModel())
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy", "magnetic_moment"}),
                    needs_pbc=False,
                    active_outputs={"energy", "magnetic_moment"},
                )

        model = _NovelModel()
        out = model.output_data()
        assert "magnetic_moment" in out


class TestBaseModelMixinAdaptInput:
    """Tests for BaseModelMixin.adapt_input()."""

    def test_enables_grad_for_autograd_outputs(self, demo_model, simple_batch):
        """When autograd outputs are requested, positions gets requires_grad."""
        demo_model.model_config.active_outputs = {"energy", "forces"}
        inp = demo_model.adapt_input(simple_batch)
        assert inp["positions"].requires_grad

    def test_no_grad_when_no_autograd_outputs(self, simple_batch):
        """When no autograd outputs are requested, positions stays without grad."""

        class _NoAutograd(DemoModelWrapper):
            def __init__(self):
                super().__init__(DemoModel())
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy"}),
                    autograd_outputs=frozenset(),
                    autograd_inputs=frozenset({"positions"}),
                    needs_pbc=False,
                    active_outputs={"energy"},
                )

        model = _NoAutograd()
        model.adapt_input(simple_batch)
        # Positions should not have grad enabled when no autograd output is requested
        assert not simple_batch.positions.requires_grad

    def test_gradient_keys_explicit(self, demo_model, simple_batch):
        """Explicit gradient_keys enables grad on those keys."""
        demo_model.model_config.active_outputs = {"energy"}
        demo_model.model_config.gradient_keys = {"positions"}
        inp = demo_model.adapt_input(simple_batch)
        assert inp["positions"].requires_grad

    def test_missing_key_raises(self, demo_model, simple_batch):
        """Missing required key raises KeyError."""

        class _NeedsMissing(DemoModelWrapper):
            def __init__(self):
                super().__init__(DemoModel())
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy"}),
                    required_inputs=frozenset({"nonexistent_key"}),
                    needs_pbc=False,
                    active_outputs={"energy"},
                )

        model = _NeedsMissing()
        with pytest.raises(KeyError, match="nonexistent_key"):
            model.adapt_input(simple_batch)

    def test_non_tensor_grad_key_raises(self, demo_model, simple_batch):
        """Non-tensor key with gradient requested raises TypeError."""
        # Monkeypatch a non-tensor attribute
        simple_batch.some_str = "not_a_tensor"
        demo_model.model_config.active_outputs = {"energy"}
        demo_model.model_config.gradient_keys = {"some_str"}
        with pytest.raises(TypeError, match="not a tensor"):
            demo_model.adapt_input(simple_batch)

    def test_collects_all_input_keys(self, demo_model, simple_batch):
        inp = demo_model.adapt_input(simple_batch)
        assert "positions" in inp
        assert "atomic_numbers" in inp


class TestBaseModelMixinAdaptOutput:
    """Tests for BaseModelMixin.adapt_output()."""

    def test_populates_from_dict(self, demo_model):
        demo_model.model_config.active_outputs = {"energy", "forces"}
        raw = {
            "energy": torch.tensor([[1.0]]),
            "forces": torch.randn(3, 3),
        }
        out = demo_model.adapt_output(raw, None)
        assert out["energy"] is not None
        assert out["forces"] is not None

    def test_unsqueeze_1d_energies(self):
        """Base adapt_output unsqueezes 1D energies to [B, 1]."""

        class _SimpleModel(DemoModelWrapper):
            def adapt_output(self, model_output, data):
                # Use only the base implementation (skip DemoModelWrapper override)
                return BaseModelMixin.adapt_output(self, model_output, data)

        model = _SimpleModel(DemoModel())
        model.model_config.active_outputs = {"energy"}
        raw = {"energy": torch.tensor([1.0])}
        out = model.adapt_output(raw, None)
        assert out["energy"].ndim == 2

    def test_missing_key_is_none(self):
        """Base adapt_output leaves missing keys as None."""

        class _SimpleModel(DemoModelWrapper):
            def adapt_output(self, model_output, data):
                return BaseModelMixin.adapt_output(self, model_output, data)

        model = _SimpleModel(DemoModel())
        model.model_config.active_outputs = {"energy", "forces"}
        raw = {"energy": torch.tensor([[1.0]])}
        out = model.adapt_output(raw, None)
        assert out["forces"] is None

    def test_non_dict_output(self):
        """Base adapt_output returns all None for non-dict output."""

        class _SimpleModel(DemoModelWrapper):
            def adapt_output(self, model_output, data):
                return BaseModelMixin.adapt_output(self, model_output, data)

        model = _SimpleModel(DemoModel())
        model.model_config.active_outputs = {"energy"}
        out = model.adapt_output("not_a_dict", None)
        assert out["energy"] is None


class TestBaseModelMixinAddOperator:
    """Tests for BaseModelMixin.__add__ (+ operator)."""

    def test_plus_returns_pipeline(self, demo_model):
        from nvalchemi.models.pipeline import PipelineModelWrapper

        other = DemoModelWrapper(DemoModel())
        combined = demo_model + other
        assert isinstance(combined, PipelineModelWrapper)

    def test_plus_creates_two_direct_groups(self, demo_model):
        other = DemoModelWrapper(DemoModel())
        combined = demo_model + other
        assert len(combined.groups) == 2
        assert combined.groups[0].use_autograd is False
        assert combined.groups[1].use_autograd is False

    def test_plus_chains_three_models(self, demo_model):
        """a + b + c flattens into 3 groups (not nested)."""
        b = DemoModelWrapper(DemoModel())
        c = DemoModelWrapper(DemoModel())
        combined = demo_model + b + c
        assert len(combined.groups) == 3

    def test_plus_sums_outputs(self, demo_model, simple_batch):
        other = DemoModelWrapper(DemoModel())
        combined = demo_model + other
        out = combined(simple_batch)
        assert out["energy"] is not None
        assert out["forces"] is not None

    def test_plus_model_config_synthesis(self, demo_model):
        other = DemoModelWrapper(DemoModel())
        combined = demo_model + other
        cfg = combined.model_config
        assert "energy" in cfg.outputs
        assert "forces" in cfg.outputs


class TestBaseModelMixinMakeNeighborHooks:
    """Tests for BaseModelMixin.make_neighbor_hooks()."""

    def test_no_hooks_without_neighbor_config(self, demo_model):
        hooks = demo_model.make_neighbor_hooks()
        assert hooks == []

    def test_hooks_with_neighbor_config(self):
        class _NLModel(DemoModelWrapper):
            def __init__(self):
                super().__init__(DemoModel())
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy"}),
                    neighbor_config=NeighborConfig(cutoff=5.0),
                    needs_pbc=False,
                )

        model = _NLModel()
        hooks = model.make_neighbor_hooks()
        assert len(hooks) == 1


class TestBaseModelMixinExportModel:
    def test_add_output_head_raises(self, demo_model):
        with pytest.raises(NotImplementedError):
            BaseModelMixin.add_output_head(demo_model, "test")


# ===========================================================================
# DemoModelWrapper-specific tests
# ===========================================================================


class TestDemoModelWrapper:
    """Tests for DemoModelWrapper with the new schema."""

    def test_model_config_outputs(self, demo_model):
        cfg = demo_model.model_config
        assert cfg.outputs == frozenset({"energy", "forces"})
        assert cfg.autograd_outputs == frozenset({"forces"})
        assert cfg.neighbor_config is None
        assert cfg.needs_pbc is False

    def test_default_active_outputs(self, demo_model):
        assert "energy" in demo_model.model_config.active_outputs
        assert "forces" in demo_model.model_config.active_outputs

    def test_forward_energies_and_forces(self, demo_model, simple_batch):
        out = demo_model(simple_batch)
        assert "energy" in out
        assert "forces" in out
        assert out["energy"].shape == (2, 1)
        assert out["forces"].shape == (5, 3)

    def test_forward_energy_only(self, simple_batch):
        model = DemoModelWrapper(DemoModel())
        model.model_config.active_outputs = {"energy"}
        out = model(simple_batch)
        assert "energy" in out

    def test_embedding_shapes(self, demo_model):
        shapes = demo_model.embedding_shapes
        assert "node_embeddings" in shapes
        assert "graph_embedding" in shapes

    def test_compute_embeddings_single(self, demo_model):
        """Test compute_embeddings on a single AtomicData."""
        data = AtomicData(
            positions=torch.randn(3, 3),
            atomic_numbers=torch.tensor([6, 6, 8]),
        )
        result = demo_model.compute_embeddings(data)
        assert hasattr(result, "node_embeddings")
        assert hasattr(result, "graph_embeddings")

    def test_export_model(self, demo_model, tmp_path):
        path = tmp_path / "demo.pt"
        demo_model.export_model(path)
        assert path.exists()


# ===========================================================================
# _utils.py tests
# ===========================================================================


class TestAutogradForces:
    """Tests for autograd_forces utility."""

    def test_basic_forces(self):
        positions = torch.randn(5, 3, requires_grad=True)
        energy = (positions**2).sum()
        forces = autograd_forces(energy, positions)
        assert forces.shape == (5, 3)
        # Forces = -gradient = -2 * positions
        torch.testing.assert_close(forces, -2 * positions)

    def test_training_creates_graph(self):
        positions = torch.randn(3, 3, requires_grad=True)
        energy = (positions**2).sum()
        forces = autograd_forces(energy, positions, training=True)
        # Should be able to compute grad of forces (higher-order)
        loss = forces.sum()
        loss.backward()
        assert positions.grad is not None

    def test_retain_graph(self):
        positions = torch.randn(3, 3, requires_grad=True)
        energy = (positions**2).sum()
        # First call with retain_graph
        forces1 = autograd_forces(energy, positions, retain_graph=True)
        # Second call should work because graph is retained
        forces2 = autograd_forces(energy, positions)
        torch.testing.assert_close(forces1, forces2)


class TestAutogradStresses:
    """Tests for autograd_stresses utility."""

    def test_basic_stresses(self):
        displacement = torch.randn(1, 3, 3, requires_grad=True)
        cell = torch.eye(3).unsqueeze(0) * 10.0  # 10 A cube
        energy = (displacement**2).sum()
        stresses = autograd_stresses(energy, displacement, cell, num_graphs=1)
        assert stresses.shape == (1, 3, 3)

    def test_multiple_systems(self):
        displacement = torch.randn(3, 3, 3, requires_grad=True)
        cell = torch.eye(3).unsqueeze(0).expand(3, -1, -1) * 10.0
        energy = (displacement**2).sum()
        stresses = autograd_stresses(energy, displacement, cell, num_graphs=3)
        assert stresses.shape == (3, 3, 3)


class TestSumOutputs:
    """Tests for sum_outputs utility."""

    def test_sum_additive_keys(self):
        a = OrderedDict(
            energy=torch.tensor([[1.0]]),
            forces=torch.tensor([[1.0, 0.0, 0.0]]),
        )
        b = OrderedDict(
            energy=torch.tensor([[2.0]]),
            forces=torch.tensor([[0.0, 1.0, 0.0]]),
        )
        result = sum_outputs(a, b)
        torch.testing.assert_close(result["energy"], torch.tensor([[3.0]]))
        torch.testing.assert_close(result["forces"], torch.tensor([[1.0, 1.0, 0.0]]))

    def test_none_values_skipped(self):
        a = OrderedDict(energy=torch.tensor([[1.0]]), forces=None)
        b = OrderedDict(energy=torch.tensor([[2.0]]), forces=torch.randn(3, 3))
        result = sum_outputs(a, b)
        assert result["energy"].item() == 3.0
        assert result["forces"] is not None

    def test_non_additive_last_wins(self):
        a = OrderedDict(charges=torch.tensor([1.0]))
        b = OrderedDict(charges=torch.tensor([2.0]))
        result = sum_outputs(a, b)
        assert result["charges"].item() == 2.0

    def test_custom_additive_keys(self):
        a = OrderedDict(charges=torch.tensor([1.0]))
        b = OrderedDict(charges=torch.tensor([2.0]))
        result = sum_outputs(a, b, additive_keys={"charges"})
        assert result["charges"].item() == 3.0

    def test_empty_outputs(self):
        result = sum_outputs()
        assert len(result) == 0

    def test_single_output(self):
        a = OrderedDict(energy=torch.tensor([[1.0]]))
        result = sum_outputs(a)
        assert result["energy"].item() == 1.0
