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

import gc
import subprocess
import sys
import weakref
from collections import OrderedDict

import pytest
import torch
from pydantic import ValidationError

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import HessianOperator
from nvalchemi.models._derivatives import _DerivativeRequest
from nvalchemi.models._utils import (
    autograd_forces,
    autograd_forces_and_stresses,
    autograd_stresses,
    cell_cache_needs_update,
    prepare_strain,
    sum_outputs,
)
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


class _QuadraticDerivativeWrapperBase(torch.nn.Module, BaseModelMixin):
    """Small analytical wrapper used to exercise derivative graph preparation."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.model_config = ModelConfig(outputs=frozenset({"energy"}))
        self.forward_calls = 0
        self.seen_requests: list[_DerivativeRequest] = []
        self.observed_active_outputs: set[str] | None = None
        self.observed_gradient_keys: set[str] | None = None
        self.output_kind = "valid"
        self.working_batch_ref = None

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def compute_embeddings(self, data, **kwargs):
        raise NotImplementedError

    def forward(self, data: Batch):
        self.forward_calls += 1
        self.working_batch_ref = weakref.ref(data)
        self.observed_active_outputs = self.model_config.active_outputs
        self.observed_gradient_keys = self.model_config.gradient_keys

        if self.output_kind == "raise":
            raise LookupError("injected derivative forward failure")
        if self.output_kind == "missing":
            return {}
        if self.output_kind == "non_tensor":
            return {"energy": 1.0}

        if self.output_kind == "disconnected":
            return {"energy": self.scale.square().expand(data.num_graphs, 1)}
        if self.output_kind == "coupled":
            energies = []
            start = 0
            for atom_count in data.num_nodes_list:
                stop = start + atom_count
                coordinates = data.positions[start:stop].reshape(-1)
                matrix = _coupled_hessian_matrix(
                    atom_count,
                    dtype=data.positions.dtype,
                    device=data.positions.device,
                )
                energies.append(0.5 * self.scale * coordinates @ matrix @ coordinates)
                start = stop
            energy = (
                torch.stack(energies).reshape(-1, 1)
                if energies
                else data.positions.sum().expand(0, 1)
            )
        elif self.output_kind == "second_derivative_raise":
            energies = []
            start = 0
            for atom_count in data.num_nodes_list:
                stop = start + atom_count
                energies.append(
                    self.scale
                    * _RaiseOnSecondDerivative.apply(data.positions[start:stop])
                )
                start = stop
            energy = torch.stack(energies).reshape(-1, 1)
        elif self.output_kind == "linear":
            node_energy = self.scale * data.positions.sum(dim=-1, keepdim=True)
        else:
            node_energy = self.scale * data.positions.square().sum(dim=-1, keepdim=True)
        if self.output_kind not in {"coupled", "second_derivative_raise"}:
            energy = torch.zeros(
                data.num_graphs,
                1,
                dtype=data.positions.dtype,
                device=data.positions.device,
            ).scatter_add(0, data.batch_idx.long().unsqueeze(-1), node_energy)

        if self.output_kind == "wrong_shape":
            energy = energy.squeeze(-1)
        elif self.output_kind == "wrong_device":
            energy = torch.empty(
                data.num_graphs,
                1,
                device="meta",
                requires_grad=True,
            )
        elif self.output_kind == "non_floating":
            energy = torch.zeros(
                data.num_graphs,
                1,
                dtype=torch.int64,
                device=data.positions.device,
            )
        elif self.output_kind == "detached":
            energy = energy.detach()
        return {"energy": energy}


class _QualifiedQuadraticDerivativeWrapper(_QuadraticDerivativeWrapperBase):
    """Test wrapper accepting supported local derivative requests."""

    def _validate_derivative_request(self, request: _DerivativeRequest) -> None:
        self.seen_requests.append(request)
        supported = request.execution == "local" and (
            (request.operation == "hvp" and request.strategy is None)
            or (
                request.operation == "dense_hessian"
                and request.strategy in {"loop", "vmap"}
            )
        )
        if supported:
            return
        strategy = request.strategy if request.strategy is not None else "none"
        raise NotImplementedError(
            f"{type(self).__name__} does not support derivative operation "
            f"'{request.operation}' for execution='{request.execution}', "
            f"strategy='{strategy}': test capability rejection"
        )


class _LoopOnlyQuadraticDerivativeWrapper(_QualifiedQuadraticDerivativeWrapper):
    """Test wrapper that deliberately leaves vectorized dense rows unqualified."""

    def _validate_derivative_request(self, request: _DerivativeRequest) -> None:
        if request.operation == "dense_hessian" and request.strategy == "vmap":
            self.seen_requests.append(request)
            raise NotImplementedError(
                "test wrapper supports dense strategy='loop' only"
            )
        super()._validate_derivative_request(request)


class _RaiseOnSecondDerivative(torch.autograd.Function):
    """Quadratic energy whose first gradient exists but second derivative fails."""

    @staticmethod
    def forward(ctx, positions):
        ctx.save_for_backward(positions)
        return positions.square().sum()

    @staticmethod
    def backward(ctx, grad_output):
        (positions,) = ctx.saved_tensors
        return 2 * _RaiseDuringBackward.apply(positions) * grad_output


class _RaiseDuringBackward(torch.autograd.Function):
    """Identity used to inject a failure during dense-row differentiation."""

    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, grad_output):
        raise LookupError("injected dense row failure")


def _coupled_hessian_matrix(
    atom_count: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Return a symmetric matrix with cross-atom and cross-axis entries."""
    dimension = 3 * atom_count
    if dimension == 0:
        return torch.empty((0, 0), dtype=dtype, device=device)
    values = torch.arange(
        1,
        dimension * dimension + 1,
        dtype=dtype,
        device=device,
    ).reshape(dimension, dimension)
    return (values + values.T) / (dimension * dimension) + torch.eye(
        dimension,
        dtype=dtype,
        device=device,
    )


def _make_derivative_batch(
    *sizes: int,
    dtype: torch.dtype = torch.float32,
) -> Batch:
    """Build a mixed-size batch for derivative API tests."""
    return Batch.from_data_list(
        [
            AtomicData(
                positions=torch.randn(size, 3, dtype=dtype),
                atomic_numbers=torch.ones(size, dtype=torch.long),
            )
            for size in sizes
        ]
    )


# ===========================================================================
# Cell cache helper tests
# ===========================================================================


class TestCellCacheNeedsUpdate:
    """Tests for cached-cell compatibility checks."""

    def test_updates_when_cached_cell_missing(self):
        """A missing cached cell should force parameter recomputation."""
        cell = torch.eye(3).expand(32, 3, 3)
        assert cell_cache_needs_update(cell, None) is True

    def test_reuses_same_shape_identical_cell(self):
        """Identical cells with matching metadata should keep cache valid."""
        cell = torch.eye(3).expand(32, 3, 3).clone()
        cached_cell = cell.clone()
        assert cell_cache_needs_update(cell, cached_cell) is False

    def test_updates_when_batch_shape_changes(self):
        """Validation batch-size changes should not reach ``torch.allclose``."""
        cached_cell = torch.eye(3).expand(32, 3, 3).clone()
        cell = torch.eye(3).expand(64, 3, 3)
        assert cell_cache_needs_update(cell, cached_cell) is True

    def test_updates_when_cell_values_change(self):
        """Same-shaped but different cells should invalidate cache."""
        cached_cell = torch.eye(3).expand(32, 3, 3).clone()
        cell = (torch.eye(3) * 2.0).expand(32, 3, 3)
        assert cell_cache_needs_update(cell, cached_cell) is True

    def test_updates_when_dtype_changes(self):
        """Dtype changes should invalidate before comparing values."""
        cached_cell = torch.eye(3, dtype=torch.float32).expand(32, 3, 3).clone()
        cell = torch.eye(3, dtype=torch.float64).expand(32, 3, 3)
        assert cell_cache_needs_update(cell, cached_cell) is True


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
# Derivative graph preparation
# ===========================================================================


class TestDerivativeGraphPreparation:
    """Tests for independent, graph-connected derivative inputs."""

    def test_yields_independent_batch_position_leaf_and_connected_energy(
        self, simple_batch
    ):
        model = _QualifiedQuadraticDerivativeWrapper()
        original_positions = simple_batch.positions.clone()
        original_numbers = simple_batch.atomic_numbers.clone()
        original_schema = simple_batch.get_level_schema()

        with model._prepare_derivative_graph(simple_batch, operation="hvp") as graph:
            assert graph.data is not simple_batch
            assert graph.positions is graph.data.positions
            assert graph.positions.is_leaf
            assert graph.positions.requires_grad
            assert graph.positions.data_ptr() != simple_batch.positions.data_ptr()
            assert graph.energy.shape == (simple_batch.num_graphs, 1)
            gradient = torch.autograd.grad(
                graph.energy.sum(), graph.positions, create_graph=True
            )[0]
            torch.testing.assert_close(gradient, 2 * graph.positions)

            assert (
                graph.data.get_level_schema().level_names == original_schema.level_names
            )
            assert (
                graph.data.get_level_schema().attr_to_group
                == original_schema.attr_to_group
            )
            graph.data.add_level("working_only", segmented=False)
            graph.data.atomic_numbers[0] = 99
            with torch.no_grad():
                graph.data.positions[0, 0] = 123.0

        torch.testing.assert_close(simple_batch.positions, original_positions)
        torch.testing.assert_close(simple_batch.atomic_numbers, original_numbers)
        assert "working_only" not in simple_batch.get_level_schema().level_names

    @pytest.mark.parametrize("with_history", [False, True])
    def test_preserves_caller_autograd_state(self, with_history):
        positions = torch.randn(3, 3, requires_grad=with_history)
        if with_history:
            positions = positions * 2.0
        data = AtomicData(
            positions=positions,
            atomic_numbers=torch.tensor([1, 6, 8]),
        )
        batch = Batch.from_data_list([data])
        original_positions = batch.positions.clone()
        original_requires_grad = batch.positions.requires_grad
        original_grad_fn_type = type(batch.positions.grad_fn)
        model = _QualifiedQuadraticDerivativeWrapper()

        with model._prepare_derivative_graph(batch, operation="hvp") as graph:
            assert graph.positions.is_leaf
            assert graph.positions.requires_grad

        assert batch.positions.requires_grad is original_requires_grad
        assert type(batch.positions.grad_fn) is original_grad_fn_type
        torch.testing.assert_close(batch.positions, original_positions)

    def test_preserves_supplied_neighbor_topology(self):
        data = AtomicData(
            positions=torch.randn(3, 3),
            atomic_numbers=torch.tensor([1, 6, 8]),
            neighbor_list=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        )
        batch = Batch.from_data_list([data])
        original_neighbors = batch.neighbor_list.clone()
        model = _QualifiedQuadraticDerivativeWrapper()

        with model._prepare_derivative_graph(batch, operation="hvp") as graph:
            torch.testing.assert_close(graph.data.neighbor_list, original_neighbors)
            assert graph.data.neighbor_list.data_ptr() != batch.neighbor_list.data_ptr()
            graph.data.neighbor_list[0, 0] = 2

        torch.testing.assert_close(batch.neighbor_list, original_neighbors)

    @pytest.mark.parametrize("training", [False, True])
    def test_restores_runtime_config_before_yield_and_preserves_mode(
        self, simple_batch, training
    ):
        model = _QualifiedQuadraticDerivativeWrapper()
        model.train(training)
        active_outputs = {"energy", "custom"}
        gradient_keys = {"cell"}
        model.model_config.active_outputs = active_outputs
        model.model_config.gradient_keys = gradient_keys

        with model._prepare_derivative_graph(simple_batch, operation="hvp") as graph:
            assert graph.energy.requires_grad
            assert model.observed_active_outputs == {"energy"}
            assert model.observed_gradient_keys == {"positions"}
            assert model.model_config.active_outputs is active_outputs
            assert model.model_config.gradient_keys is gradient_keys
            assert model.training is training

        assert model.model_config.active_outputs is active_outputs
        assert model.model_config.gradient_keys is gradient_keys
        assert model.training is training

    def test_forward_failure_restores_state_and_propagates(self, simple_batch):
        model = _QualifiedQuadraticDerivativeWrapper()
        model.output_kind = "raise"
        model.eval()
        active_outputs = {"custom"}
        gradient_keys = {"cell"}
        model.model_config.active_outputs = active_outputs
        model.model_config.gradient_keys = gradient_keys
        original_positions = simple_batch.positions.clone()

        with pytest.raises(LookupError, match="injected derivative forward failure"):
            with model._prepare_derivative_graph(simple_batch, operation="hvp"):
                pass

        assert model.model_config.active_outputs is active_outputs
        assert model.model_config.gradient_keys is gradient_keys
        assert model.training is False
        torch.testing.assert_close(simple_batch.positions, original_positions)

    @pytest.mark.parametrize("requires_grad", [False, True])
    def test_preserves_parameter_autograd_state(self, simple_batch, requires_grad):
        model = _QualifiedQuadraticDerivativeWrapper()
        model.scale.requires_grad_(requires_grad)
        original_grad = torch.tensor(7.0)
        model.scale.grad = original_grad

        with model._prepare_derivative_graph(simple_batch, operation="hvp") as graph:
            assert graph.energy.requires_grad

        assert model.scale.requires_grad is requires_grad
        assert model.scale.grad is original_grad

    def test_prepares_inside_no_grad_and_restores_state_before_yield(
        self, simple_batch
    ):
        model = _QualifiedQuadraticDerivativeWrapper()

        with torch.no_grad():
            assert not torch.is_grad_enabled()
            with model._prepare_derivative_graph(
                simple_batch, operation="hvp"
            ) as graph:
                assert not torch.is_grad_enabled()
                assert graph.energy.requires_grad
            assert not torch.is_grad_enabled()

    def test_prepares_inside_inference_mode_and_restores_state_before_yield(
        self, simple_batch
    ):
        model = _QualifiedQuadraticDerivativeWrapper()

        with torch.inference_mode():
            assert torch.is_inference_mode_enabled()
            with model._prepare_derivative_graph(
                simple_batch, operation="hvp"
            ) as graph:
                assert torch.is_inference_mode_enabled()
                assert not graph.positions.is_inference()
                assert graph.energy.requires_grad
            assert torch.is_inference_mode_enabled()

    def test_context_does_not_leave_graph_state_on_wrapper(self, simple_batch):
        model = _QualifiedQuadraticDerivativeWrapper()

        with model._prepare_derivative_graph(simple_batch, operation="hvp") as graph:
            working_ref = weakref.ref(graph.data)
            assert working_ref() is graph.data

        del graph
        gc.collect()
        assert working_ref() is None
        assert not hasattr(model, "_derivative_graph")

    @pytest.mark.parametrize(
        ("output_kind", "message"),
        [
            ("missing", "did not return energy"),
            ("non_tensor", "must be a torch.Tensor"),
            ("wrong_shape", "must have shape"),
            ("wrong_device", "must be on the positions device"),
            ("non_floating", "must have a floating-point dtype"),
            ("detached", "must retain a graph"),
        ],
    )
    def test_rejects_invalid_energy_contract(self, simple_batch, output_kind, message):
        model = _QualifiedQuadraticDerivativeWrapper()
        model.output_kind = output_kind
        active_outputs = model.model_config.active_outputs
        gradient_keys = model.model_config.gradient_keys

        with pytest.raises(RuntimeError, match=message):
            with model._prepare_derivative_graph(simple_batch, operation="hvp"):
                pass

        assert model.model_config.active_outputs is active_outputs
        assert model.model_config.gradient_keys is gradient_keys

    def test_private_imports_do_not_load_optional_model_packages(self):
        code = """
import sys
import nvalchemi.models.base
import nvalchemi.models._derivatives
for name in ('aimnet', 'mace', 'fairchem'):
    assert name not in sys.modules, name
"""
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )


class TestDerivativeCapability:
    """Tests for contextual derivative capability preflight."""

    def test_unqualified_wrapper_fails_before_forward(self, simple_batch):
        model = _QuadraticDerivativeWrapperBase()

        with pytest.raises(
            NotImplementedError,
            match=(
                "_QuadraticDerivativeWrapperBase.*operation 'hvp'.*"
                "execution='local'.*strategy='none'"
            ),
        ):
            with model._prepare_derivative_graph(simple_batch, operation="hvp"):
                pass

        assert model.forward_calls == 0

    @pytest.mark.parametrize(
        ("operation", "strategy"),
        [
            ("hvp", None),
            ("dense_hessian", "loop"),
            ("dense_hessian", "vmap"),
        ],
    )
    def test_supported_local_requests_reach_forward(
        self, simple_batch, operation, strategy
    ):
        model = _QualifiedQuadraticDerivativeWrapper()

        with model._prepare_derivative_graph(
            simple_batch,
            operation=operation,
            strategy=strategy,
        ):
            pass

        assert model.forward_calls == 1
        request = model.seen_requests[-1]
        assert request.operation == operation
        assert request.strategy == strategy
        assert request.execution == "local"

    def test_distributed_request_fails_before_forward(self, simple_batch):
        model = _QualifiedQuadraticDerivativeWrapper()
        model._dist_ctx = object()

        with pytest.raises(NotImplementedError, match="execution='distributed'"):
            with model._prepare_derivative_graph(simple_batch, operation="hvp"):
                pass

        assert model.forward_calls == 0

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            (
                {
                    "operation": "unknown",
                    "execution": "local",
                    "strategy": None,
                },
                "operation must be",
            ),
            (
                {
                    "operation": "hvp",
                    "execution": "unknown",
                    "strategy": None,
                },
                "execution must be",
            ),
            (
                {
                    "operation": "hvp",
                    "execution": "local",
                    "strategy": "unknown",
                },
                "strategy must be",
            ),
            (
                {
                    "operation": "hvp",
                    "execution": "local",
                    "strategy": "loop",
                },
                "HVP requests must not specify",
            ),
            (
                {
                    "operation": "dense_hessian",
                    "execution": "local",
                    "strategy": None,
                },
                "Dense-Hessian requests must specify",
            ),
        ],
    )
    def test_request_validation_rejects_malformed_context(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            _DerivativeRequest(**kwargs)

    def test_malformed_request_fails_before_forward(self, simple_batch):
        model = _QualifiedQuadraticDerivativeWrapper()

        with pytest.raises(ValueError, match="HVP requests must not specify"):
            model._prepare_derivative_graph(
                simple_batch,
                operation="hvp",
                strategy="loop",
            )

        assert model.seen_requests == []
        assert model.forward_calls == 0

    def test_non_batch_input_fails_before_capability_or_forward(self):
        model = _QualifiedQuadraticDerivativeWrapper()

        with pytest.raises(TypeError, match="batch must be a Batch"):
            model._prepare_derivative_graph(object(), operation="hvp")

        assert model.seen_requests == []
        assert model.forward_calls == 0


# ===========================================================================
# Hessian-vector products
# ===========================================================================


class TestHessianVectorProduct:
    """Tests for the one-shot matrix-free Hessian API."""

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_quadratic_hvp_matches_analytical_result(self, dtype):
        batch = _make_derivative_batch(2, 3, dtype=dtype)
        model = _QualifiedQuadraticDerivativeWrapper().to(dtype=dtype)
        vector = torch.randn_like(batch.positions)

        result = model.hessian_vector_product(batch, vector)

        torch.testing.assert_close(result, 2 * vector)
        assert result.shape == batch.positions.shape
        assert result.dtype == batch.positions.dtype
        assert result.device == batch.positions.device
        assert not result.requires_grad
        assert result.grad_fn is None
        assert model.forward_calls == 1

    def test_hvp_matches_finite_difference_force_jacobian(self):
        batch = _make_derivative_batch(2, 3, dtype=torch.float64)
        model = _QualifiedQuadraticDerivativeWrapper().to(dtype=torch.float64)
        vector = torch.randn_like(batch.positions)
        epsilon = 1e-6

        product = model.hessian_vector_product(batch, vector)

        def evaluate_forces(positions):
            working = batch.clone()
            position_leaf = positions.detach().clone().requires_grad_(True)
            working["positions"] = position_leaf
            with torch.enable_grad():
                energy = model(working)["energy"]
                return -torch.autograd.grad(energy.sum(), position_leaf)[0]

        force_plus = evaluate_forces(batch.positions + epsilon * vector)
        force_minus = evaluate_forces(batch.positions - epsilon * vector)
        force_jacobian_vector = (force_plus - force_minus) / (2 * epsilon)

        torch.testing.assert_close(
            force_jacobian_vector,
            -product,
            rtol=1e-8,
            atol=1e-8,
        )

    def test_zero_vector_and_linear_energy_have_zero_hvp(self):
        batch = _make_derivative_batch(1, 2)
        model = _QualifiedQuadraticDerivativeWrapper()

        quadratic = model.hessian_vector_product(
            batch,
            torch.zeros_like(batch.positions),
        )
        model.output_kind = "linear"
        linear = model.hessian_vector_product(
            batch,
            torch.randn_like(batch.positions),
        )

        torch.testing.assert_close(quadratic, torch.zeros_like(batch.positions))
        torch.testing.assert_close(linear, torch.zeros_like(batch.positions))

    def test_mixed_systems_have_no_cross_system_response(self):
        batch = _make_derivative_batch(2, 3)
        model = _QualifiedQuadraticDerivativeWrapper()
        vector = torch.zeros_like(batch.positions)
        vector[:2] = torch.randn(2, 3)

        result = model.hessian_vector_product(batch, vector)

        torch.testing.assert_close(result[:2], 2 * vector[:2])
        torch.testing.assert_close(result[2:], torch.zeros_like(result[2:]))

    def test_accepts_noncontiguous_vector(self):
        batch = _make_derivative_batch(2, 3)
        model = _QualifiedQuadraticDerivativeWrapper()
        vector = torch.randn(3, batch.positions.shape[0]).T
        assert not vector.is_contiguous()

        result = model.hessian_vector_product(batch, vector)

        torch.testing.assert_close(result, 2 * vector)

    @pytest.mark.parametrize(
        ("case", "error", "message"),
        [
            ("non_tensor", TypeError, "must be a torch.Tensor"),
            ("integer", TypeError, "floating-point dtype"),
            ("wrong_shape", ValueError, "same shape"),
            ("wrong_layout", ValueError, "same shape"),
            ("wrong_dtype", ValueError, "same dtype"),
            ("wrong_device", ValueError, "same device"),
        ],
    )
    def test_invalid_vectors_fail_before_forward(self, case, error, message):
        batch = _make_derivative_batch(2, 3)
        model = _QualifiedQuadraticDerivativeWrapper()
        if case == "non_tensor":
            vector = object()
        elif case == "integer":
            vector = torch.ones_like(batch.positions, dtype=torch.int64)
        elif case == "wrong_shape":
            vector = torch.ones(batch.positions.shape[0] - 1, 3)
        elif case == "wrong_layout":
            vector = torch.ones(3, batch.positions.shape[0])
        elif case == "wrong_dtype":
            vector = torch.ones_like(batch.positions, dtype=torch.float64)
        else:
            vector = torch.ones_like(batch.positions, device="meta")

        with pytest.raises(error, match=message):
            model.hessian_vector_product(batch, vector)

        assert model.seen_requests == []
        assert model.forward_calls == 0

    def test_preserves_caller_vector_batch_and_parameter_state(self):
        positions = torch.randn(3, 3, requires_grad=True) * 2
        batch = Batch.from_data_list(
            [
                AtomicData(
                    positions=positions,
                    atomic_numbers=torch.ones(3, dtype=torch.long),
                )
            ]
        )
        vector = torch.randn_like(batch.positions, requires_grad=True) * 3
        model = _QualifiedQuadraticDerivativeWrapper()
        model.eval()
        parameter_grad = torch.tensor(7.0)
        model.scale.grad = parameter_grad
        active_outputs = {"energy", "custom"}
        gradient_keys = {"cell"}
        model.model_config.active_outputs = active_outputs
        model.model_config.gradient_keys = gradient_keys
        original_positions = batch.positions.clone()
        original_vector = vector.clone()
        positions_grad_fn = type(batch.positions.grad_fn)
        vector_grad_fn = type(vector.grad_fn)

        result = model.hessian_vector_product(batch, vector)

        torch.testing.assert_close(result, 2 * vector)
        torch.testing.assert_close(batch.positions, original_positions)
        torch.testing.assert_close(vector, original_vector)
        assert type(batch.positions.grad_fn) is positions_grad_fn
        assert batch.positions.requires_grad
        assert type(vector.grad_fn) is vector_grad_fn
        assert vector.requires_grad
        assert model.scale.grad is parameter_grad
        assert model.model_config.active_outputs is active_outputs
        assert model.model_config.gradient_keys is gradient_keys
        assert model.training is False

    @pytest.mark.parametrize("outer_mode", ["no_grad", "inference"])
    def test_hvp_restores_outer_autograd_mode(self, outer_mode):
        batch = _make_derivative_batch(2)
        model = _QualifiedQuadraticDerivativeWrapper()
        vector = torch.randn_like(batch.positions)
        context = torch.no_grad() if outer_mode == "no_grad" else torch.inference_mode()

        with context:
            expected_grad = torch.is_grad_enabled()
            expected_inference = torch.is_inference_mode_enabled()
            result = model.hessian_vector_product(batch, vector)
            assert torch.is_grad_enabled() is expected_grad
            assert torch.is_inference_mode_enabled() is expected_inference
            assert not result.is_inference()

        torch.testing.assert_close(result, 2 * vector)

    @pytest.mark.parametrize("context_kind", ["base", "distributed"])
    def test_capability_rejection_occurs_before_forward(
        self, simple_batch, context_kind
    ):
        if context_kind == "base":
            model = _QuadraticDerivativeWrapperBase()
        else:
            model = _QualifiedQuadraticDerivativeWrapper()
        if context_kind == "distributed":
            model._dist_ctx = object()

        with pytest.raises(NotImplementedError):
            model.hessian_vector_product(
                simple_batch,
                torch.ones_like(simple_batch.positions),
            )

        assert model.forward_calls == 0

    def test_flat_pipeline_sums_qualified_hvps(self, simple_batch):
        left = _QualifiedQuadraticDerivativeWrapper()
        right = _QualifiedQuadraticDerivativeWrapper()
        pipeline = left + right
        vector = torch.ones_like(simple_batch.positions)

        result = pipeline.hessian_vector_product(simple_batch, vector)

        torch.testing.assert_close(result, 4 * vector)
        assert left.forward_calls == 1
        assert right.forward_calls == 1


class TestHessianOperator:
    """Tests for retained-graph HVP execution and cleanup."""

    def test_is_immediately_active_and_reuses_one_forward(self, simple_batch):
        model = _QualifiedQuadraticDerivativeWrapper()
        first = torch.randn_like(simple_batch.positions)
        second = torch.randn_like(simple_batch.positions)

        operator = model.prepare_hessian(simple_batch)
        assert isinstance(operator, HessianOperator)
        assert model.forward_calls == 1
        assert operator.__enter__() is operator
        assert operator.__enter__() is operator

        first_result = operator.matvec(first)
        second_result = operator.matvec(second)

        torch.testing.assert_close(first_result, 2 * first)
        torch.testing.assert_close(second_result, 2 * second)
        assert model.forward_calls == 1
        operator.close()

    def test_prepared_results_match_independent_one_shot_calls(self, simple_batch):
        model = _QualifiedQuadraticDerivativeWrapper()
        first = torch.randn_like(simple_batch.positions)
        second = torch.randn_like(simple_batch.positions)

        with model.prepare_hessian(simple_batch) as operator:
            prepared_first = operator.matvec(first)
            prepared_second = operator.matvec(second)
        assert model.forward_calls == 1

        one_shot_first = model.hessian_vector_product(simple_batch, first)
        one_shot_second = model.hessian_vector_product(simple_batch, second)

        torch.testing.assert_close(prepared_first, one_shot_first)
        torch.testing.assert_close(prepared_second, one_shot_second)
        assert model.forward_calls == 3

    def test_close_is_idempotent_and_terminal(self, simple_batch):
        model = _QualifiedQuadraticDerivativeWrapper()
        operator = model.prepare_hessian(simple_batch)
        operator.close()
        operator.close()

        with pytest.raises(RuntimeError, match="HessianOperator is closed"):
            operator.matvec(torch.ones_like(simple_batch.positions))
        with pytest.raises(RuntimeError, match="HessianOperator is closed"):
            operator.__enter__()

    def test_context_exit_closes_after_success_and_exception(self, simple_batch):
        model = _QualifiedQuadraticDerivativeWrapper()
        operator = model.prepare_hessian(simple_batch)
        with operator:
            operator.matvec(torch.ones_like(simple_batch.positions))
        with pytest.raises(RuntimeError, match="HessianOperator is closed"):
            operator.matvec(torch.ones_like(simple_batch.positions))

        failing = model.prepare_hessian(simple_batch)
        with pytest.raises(LookupError, match="body failure"):
            with failing:
                raise LookupError("body failure")
        with pytest.raises(RuntimeError, match="HessianOperator is closed"):
            failing.__enter__()

    @pytest.mark.parametrize("close_kind", ["explicit", "context", "exception"])
    def test_close_releases_private_working_batch(self, simple_batch, close_kind):
        model = _QualifiedQuadraticDerivativeWrapper()
        operator = model.prepare_hessian(simple_batch)
        working_ref = model.working_batch_ref
        assert working_ref is not None
        assert working_ref() is not None

        if close_kind == "explicit":
            operator.close()
        elif close_kind == "context":
            with operator:
                pass
        else:
            with pytest.raises(LookupError, match="body failure"):
                with operator:
                    raise LookupError("body failure")

        gc.collect()
        assert working_ref() is None

    def test_invalid_matvec_leaves_operator_active(self, simple_batch):
        model = _QualifiedQuadraticDerivativeWrapper()
        operator = model.prepare_hessian(simple_batch)

        with pytest.raises(ValueError, match="same shape"):
            operator.matvec(torch.ones(1, 3))

        result = operator.matvec(torch.ones_like(simple_batch.positions))
        torch.testing.assert_close(
            result,
            2 * torch.ones_like(simple_batch.positions),
        )
        assert model.forward_calls == 1
        operator.close()

    def test_disconnected_energy_fails_preparation_and_releases_graph(
        self, simple_batch
    ):
        model = _QualifiedQuadraticDerivativeWrapper()
        model.output_kind = "disconnected"
        active_outputs = model.model_config.active_outputs
        gradient_keys = model.model_config.gradient_keys

        with pytest.raises(RuntimeError, match="not connected to the position leaf"):
            model.prepare_hessian(simple_batch)

        working_ref = model.working_batch_ref
        assert working_ref is not None
        gc.collect()
        assert working_ref() is None
        assert model.model_config.active_outputs is active_outputs
        assert model.model_config.gradient_keys is gradient_keys

    def test_forward_failure_propagates_without_operator_state(self, simple_batch):
        model = _QualifiedQuadraticDerivativeWrapper()
        model.output_kind = "raise"

        with pytest.raises(LookupError, match="injected derivative forward failure"):
            model.prepare_hessian(simple_batch)

        assert model.working_batch_ref is not None
        gc.collect()
        assert model.working_batch_ref() is None
        assert not hasattr(model, "_hessian_operator")

    @pytest.mark.parametrize("outer_mode", ["no_grad", "inference"])
    def test_operator_restores_outer_autograd_mode(self, outer_mode):
        batch = _make_derivative_batch(2)
        model = _QualifiedQuadraticDerivativeWrapper()
        vector = torch.randn_like(batch.positions)
        context = torch.no_grad() if outer_mode == "no_grad" else torch.inference_mode()

        with context:
            expected_grad = torch.is_grad_enabled()
            expected_inference = torch.is_inference_mode_enabled()
            operator = model.prepare_hessian(batch)
            assert torch.is_grad_enabled() is expected_grad
            assert torch.is_inference_mode_enabled() is expected_inference
            result = operator.matvec(vector)
            assert torch.is_grad_enabled() is expected_grad
            assert torch.is_inference_mode_enabled() is expected_inference
            operator.close()

        torch.testing.assert_close(result, 2 * vector)

    def test_public_import_does_not_load_optional_model_packages(self):
        code = """
import sys
from nvalchemi.models import HessianOperator
assert HessianOperator.__name__ == 'HessianOperator'
for name in ('aimnet', 'mace', 'fairchem'):
    assert name not in sys.modules, name
"""
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            check=True,
            capture_output=True,
            text=True,
        )


# ===========================================================================
# Dense Hessian materialization
# ===========================================================================


class TestDenseHessian:
    """Tests for canonical in-place dense Hessian materialization."""

    @staticmethod
    def _expected_blocks(batch: Batch) -> list[torch.Tensor]:
        return [
            _coupled_hessian_matrix(
                atom_count,
                dtype=batch.positions.dtype,
                device=batch.positions.device,
            )
            .reshape(atom_count, 3, atom_count, 3)
            .permute(0, 2, 1, 3)
            .contiguous()
            for atom_count in batch.num_nodes_list
        ]

    @staticmethod
    def _snapshot(batch: Batch):
        schema = batch.get_level_schema()
        schema_state = (
            schema.level_names,
            schema.level_kinds.copy(),
            schema.product_parents.copy(),
            {name: attrs.copy() for name, attrs in schema.group_to_attrs.items()},
            schema.dtypes.copy(),
        )
        tensors = {key: value.clone() for key, value in batch}
        return schema_state, tensors

    @staticmethod
    def _assert_snapshot(batch: Batch, snapshot) -> None:
        schema_state, tensors = snapshot
        schema = batch.get_level_schema()
        assert (
            schema.level_names,
            schema.level_kinds,
            schema.product_parents,
            schema.group_to_attrs,
            schema.dtypes,
        ) == schema_state
        assert {key for key, _ in batch} == set(tensors)
        for key, value in tensors.items():
            torch.testing.assert_close(batch[key], value)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize("strategy", ["loop", "vmap"])
    def test_materializes_analytical_mixed_system_blocks(self, dtype, strategy):
        batch = _make_derivative_batch(1, 3, dtype=dtype)
        model = _QualifiedQuadraticDerivativeWrapper().to(dtype=dtype)
        model.output_kind = "coupled"
        expected = self._expected_blocks(batch)

        result = model.compute_hessian(batch, strategy=strategy)

        assert result is batch
        assert model.forward_calls == 1
        assert batch.get_level_schema().product_parents["atom_atom"] == (
            "atoms",
            "atoms",
        )
        assert batch.get_level_schema().attr_to_group["hessian"] == "atom_atom"
        assert batch.hessian.shape == (10, 3, 3)
        assert batch.level_ptr("atom_atom").tolist() == [0, 1, 10]
        assert not batch.hessian.requires_grad
        assert batch.hessian.grad_fn is None
        for index, expected_block in enumerate(expected):
            actual = batch.get_data(index).hessian
            torch.testing.assert_close(actual, expected_block)
            torch.testing.assert_close(actual, actual.permute(1, 0, 3, 2))

    @pytest.mark.parametrize("row_chunk_size", [None, 1, 4, 64])
    def test_chunk_sizes_and_strategies_agree(self, row_chunk_size):
        source = _make_derivative_batch(2, 3, dtype=torch.float64)
        loop_batch = source.clone()
        vmap_batch = source.clone()
        loop_model = _QualifiedQuadraticDerivativeWrapper().to(dtype=torch.float64)
        vmap_model = _QualifiedQuadraticDerivativeWrapper().to(dtype=torch.float64)
        loop_model.output_kind = "coupled"
        vmap_model.output_kind = "coupled"

        loop_model.compute_hessian(
            loop_batch,
            strategy="loop",
            row_chunk_size=row_chunk_size,
        )
        vmap_model.compute_hessian(
            vmap_batch,
            strategy="vmap",
            row_chunk_size=row_chunk_size,
        )

        torch.testing.assert_close(loop_batch.hessian, vmap_batch.hessian)
        assert loop_model.forward_calls == 1
        assert vmap_model.forward_calls == 1

    def test_stored_block_contraction_matches_hvp(self):
        batch = _make_derivative_batch(2, 3, dtype=torch.float64)
        model = _QualifiedQuadraticDerivativeWrapper().to(dtype=torch.float64)
        model.output_kind = "coupled"
        vector = torch.randn_like(batch.positions)

        product = model.hessian_vector_product(batch, vector)
        model.compute_hessian(batch, strategy="vmap", row_chunk_size=4)

        start = 0
        for index, atom_count in enumerate(batch.num_nodes_list):
            stop = start + atom_count
            block_product = torch.einsum(
                "abij,bj->ai",
                batch.get_data(index).hessian,
                vector[start:stop],
            )
            torch.testing.assert_close(block_product, product[start:stop])
            start = stop

    def test_linear_energy_and_empty_systems_produce_zero_blocks(self):
        batch = _make_derivative_batch(0, 2)
        model = _QualifiedQuadraticDerivativeWrapper()
        model.output_kind = "linear"

        model.compute_hessian(batch, strategy="vmap")

        assert batch.get_data(0).hessian.shape == (0, 0, 3, 3)
        torch.testing.assert_close(
            batch.get_data(1).hessian,
            torch.zeros(2, 2, 3, 3),
        )

    def test_zero_graph_batch_materializes_empty_canonical_field(self):
        batch = Batch.empty(num_systems=0, num_nodes=0, num_edges=0)
        model = _QualifiedQuadraticDerivativeWrapper()
        model.output_kind = "coupled"

        result = model.compute_hessian(batch)

        assert result is batch
        assert batch.hessian.shape == (0, 3, 3)
        assert batch.level_ptr("atom_atom").tolist() == [0]

    def test_repeated_calls_replace_hessian_and_preserve_other_fields(self):
        batch = _make_derivative_batch(2, 1)
        batch.add_key(
            "marker",
            [torch.arange(2), torch.arange(1)],
            level="atoms",
        )
        marker = batch.marker.clone()
        model = _QualifiedQuadraticDerivativeWrapper()

        model.compute_hessian(batch, strategy="loop")
        first = batch.hessian.clone()
        model.scale.data.fill_(2)
        model.compute_hessian(batch, strategy="vmap")

        torch.testing.assert_close(batch.hessian, 2 * first)
        torch.testing.assert_close(batch.marker, marker)
        assert model.forward_calls == 2

    def test_explicit_clone_provides_nonmutating_use(self):
        batch = _make_derivative_batch(2, 3)
        original_schema = batch.get_level_schema()
        result = batch.clone()
        model = _QualifiedQuadraticDerivativeWrapper()

        returned = model.compute_hessian(result)

        assert returned is result
        assert "hessian" not in batch
        assert "atom_atom" not in original_schema.level_names
        assert "hessian" in result

    @pytest.mark.parametrize(
        ("strategy", "row_chunk_size", "error", "message"),
        [
            ("unknown", None, ValueError, "strategy must be"),
            ("loop", True, TypeError, "positive integer or None"),
            ("loop", 1.5, TypeError, "positive integer or None"),
            ("vmap", 0, ValueError, "must be positive"),
            ("vmap", -1, ValueError, "must be positive"),
        ],
    )
    def test_invalid_arguments_fail_before_capability_or_forward(
        self,
        strategy,
        row_chunk_size,
        error,
        message,
    ):
        batch = _make_derivative_batch(2)
        model = _QualifiedQuadraticDerivativeWrapper()
        snapshot = self._snapshot(batch)

        with pytest.raises(error, match=message):
            model.compute_hessian(
                batch,
                strategy=strategy,
                row_chunk_size=row_chunk_size,
            )

        assert model.seen_requests == []
        assert model.forward_calls == 0
        self._assert_snapshot(batch, snapshot)

    def test_invalid_batch_layout_fails_before_forward(self):
        model = _QualifiedQuadraticDerivativeWrapper()
        batch = _make_derivative_batch(2)
        batch["positions"] = torch.zeros(2, 4)

        with pytest.raises(ValueError, match=r"shape \[total_atoms, 3\]"):
            model.compute_hessian(batch)

        assert model.forward_calls == 0

    @pytest.mark.parametrize("case", ["atomic_rank", "capacity"])
    def test_invalid_packed_lengths_fail_before_forward(self, case):
        model = _QualifiedQuadraticDerivativeWrapper()
        if case == "capacity":
            batch = Batch.empty(num_systems=2, num_nodes=4, num_edges=0)
        else:
            batch = _make_derivative_batch(2)
            if case == "atomic_rank":
                batch["atomic_numbers"] = torch.ones((2, 1), dtype=torch.long)

        with pytest.raises(ValueError):
            model.compute_hessian(batch)

        assert model.forward_calls == 0

    def test_non_batch_fails_before_capability_validation(self):
        model = _QualifiedQuadraticDerivativeWrapper()

        with pytest.raises(TypeError, match="batch must be a Batch"):
            model.compute_hessian(object())

        assert model.seen_requests == []
        assert model.forward_calls == 0

    @pytest.mark.parametrize(
        "conflict",
        ["level", "parents", "owner", "dtype", "shape"],
    )
    def test_incompatible_canonical_storage_fails_before_forward(self, conflict):
        batch = _make_derivative_batch(2, 1)
        if conflict == "level":
            batch.add_level("atom_atom", segmented=True)
        elif conflict == "parents":
            batch.add_product_level("atom_atom", left="edges", right="atoms")
        elif conflict == "owner":
            batch.add_key(
                "hessian",
                [torch.zeros(2, 3, 3), torch.zeros(1, 3, 3)],
                level="atoms",
            )
        else:
            batch.add_product_level("atom_atom", left="atoms", right="atoms")
            if conflict == "dtype":
                values = [
                    torch.zeros(2, 2, 3, 3, dtype=torch.float64),
                    torch.zeros(1, 1, 3, 3, dtype=torch.float64),
                ]
            else:
                values = [torch.zeros(2, 2, 2, 3), torch.zeros(1, 1, 2, 3)]
            batch.add_key("hessian", values, level="atom_atom")
        snapshot = self._snapshot(batch)
        model = _QualifiedQuadraticDerivativeWrapper()

        with pytest.raises(ValueError):
            model.compute_hessian(batch)

        assert model.forward_calls == 0
        self._assert_snapshot(batch, snapshot)

    @pytest.mark.parametrize(
        ("output_kind", "strategy", "error", "message"),
        [
            (
                "raise",
                "loop",
                LookupError,
                "injected derivative forward failure",
            ),
            (
                "disconnected",
                "loop",
                RuntimeError,
                "not connected to the position leaf",
            ),
            (
                "second_derivative_raise",
                "loop",
                LookupError,
                "injected dense row failure",
            ),
            (
                "second_derivative_raise",
                "vmap",
                LookupError,
                "injected dense row failure",
            ),
        ],
    )
    def test_derivative_failures_leave_batch_unchanged(
        self,
        output_kind,
        strategy,
        error,
        message,
    ):
        batch = _make_derivative_batch(2, 1)
        snapshot = self._snapshot(batch)
        model = _QualifiedQuadraticDerivativeWrapper()
        model.output_kind = output_kind

        with pytest.raises(error, match=message):
            model.compute_hessian(batch, strategy=strategy, row_chunk_size=1)

        self._assert_snapshot(batch, snapshot)

    def test_preserves_caller_and_model_runtime_state(self):
        positions = torch.randn(3, 3, requires_grad=True) * 2
        batch = Batch.from_data_list(
            [
                AtomicData(
                    positions=positions,
                    atomic_numbers=torch.ones(3, dtype=torch.long),
                )
            ]
        )
        model = _QualifiedQuadraticDerivativeWrapper()
        model.eval()
        parameter_grad = torch.tensor(7.0)
        model.scale.grad = parameter_grad
        active_outputs = {"energy", "custom"}
        gradient_keys = {"cell"}
        model.model_config.active_outputs = active_outputs
        model.model_config.gradient_keys = gradient_keys
        original_positions = batch.positions.clone()
        position_grad_fn = type(batch.positions.grad_fn)

        result = model.compute_hessian(batch)

        assert result is batch
        torch.testing.assert_close(batch.positions, original_positions)
        assert type(batch.positions.grad_fn) is position_grad_fn
        assert batch.positions.requires_grad
        assert model.scale.grad is parameter_grad
        assert model.model_config.active_outputs is active_outputs
        assert model.model_config.gradient_keys is gradient_keys
        assert model.training is False

    @pytest.mark.parametrize("outer_mode", ["no_grad", "inference"])
    def test_restores_outer_autograd_mode(self, outer_mode):
        batch = _make_derivative_batch(2)
        model = _QualifiedQuadraticDerivativeWrapper()
        context = torch.no_grad() if outer_mode == "no_grad" else torch.inference_mode()

        with context:
            expected_grad = torch.is_grad_enabled()
            expected_inference = torch.is_inference_mode_enabled()
            model.compute_hessian(batch)
            assert torch.is_grad_enabled() is expected_grad
            assert torch.is_inference_mode_enabled() is expected_inference
            assert not batch.hessian.is_inference()

    @pytest.mark.parametrize("context_kind", ["base", "distributed"])
    def test_capability_rejection_occurs_before_forward(self, context_kind):
        batch = _make_derivative_batch(2)
        if context_kind == "base":
            model = _QuadraticDerivativeWrapperBase()
        else:
            model = _QualifiedQuadraticDerivativeWrapper()
        if context_kind == "distributed":
            model._dist_ctx = object()
        snapshot = self._snapshot(batch)

        with pytest.raises(NotImplementedError):
            model.compute_hessian(batch)

        assert model.forward_calls == 0
        self._assert_snapshot(batch, snapshot)

    def test_vmap_rejection_does_not_fall_back_to_loop(self):
        batch = _make_derivative_batch(2)
        model = _LoopOnlyQuadraticDerivativeWrapper()

        with pytest.raises(NotImplementedError, match="strategy='loop' only"):
            model.compute_hessian(batch, strategy="vmap")

        assert model.forward_calls == 0
        model.compute_hessian(batch, strategy="loop")
        assert model.forward_calls == 1

    def test_flat_pipeline_materializes_sum_of_qualified_hessians(self):
        batch = _make_derivative_batch(2)
        left = _QualifiedQuadraticDerivativeWrapper()
        right = _QualifiedQuadraticDerivativeWrapper()
        pipeline = left + right

        result = pipeline.compute_hessian(batch)

        assert result is batch
        expected = 4 * torch.eye(6).reshape(2, 3, 2, 3).permute(0, 2, 1, 3)
        torch.testing.assert_close(batch.get_data(0).hessian, expected)
        assert left.forward_calls == 1
        assert right.forward_calls == 1


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

    def test_output_key_order_is_sorted_deterministic(self):
        """The output dict must be seeded in a
        rank-independent key order. ``output_data()`` returns a set, whose
        iteration order is randomized per process (str hashing), so
        ``adapt_output`` must sort. Under DomainParallel, consolidation issues
        one collective per key in dict order — a per-rank order desyncs the NCCL
        schedule and deadlocks (ranks issue ALLREDUCE vs ALLTOALL at the same
        seq)."""

        class _MultiOutModel(DemoModelWrapper):
            def __init__(self):
                super().__init__(DemoModel())
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy", "forces", "stress"}),
                    needs_pbc=False,
                    active_outputs={"stress", "energy", "forces"},
                )

            def adapt_output(self, model_output, data):
                return BaseModelMixin.adapt_output(self, model_output, data)

        model = _MultiOutModel()
        raw = {
            "energy": torch.tensor([[1.0]]),
            "forces": torch.randn(3, 3),
            "stress": torch.randn(1, 3, 3),
        }
        out = model.adapt_output(raw, None)
        keys = list(out.keys())
        assert keys == sorted(keys)
        assert keys == ["energy", "forces", "stress"]

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

    def test_hooks_forward_neighbor_list_method(self):
        class _NLModel(DemoModelWrapper):
            def __init__(self):
                super().__init__(DemoModel())
                self.model_config = ModelConfig(
                    outputs=frozenset({"energy"}),
                    neighbor_config=NeighborConfig(cutoff=5.0),
                    needs_pbc=False,
                )

        model = _NLModel()
        hooks = model.make_neighbor_hooks(neighbor_list_method="batch_naive_tile")

        assert len(hooks) == 1
        assert hooks[0].method == "batch_naive_tile"


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

    def test_compute_embeddings_on_multi_graph_batch(
        self, demo_model, simple_batch
    ) -> None:
        """A batch gets its node embeddings in the atoms group, one row per atom."""
        hidden_dim = demo_model.embedding_shapes["node_embeddings"][-1]

        result = demo_model.compute_embeddings(simple_batch)

        assert result.node_embeddings.shape == (5, hidden_dim)
        assert result.graph_embeddings.shape == (2, hidden_dim)
        assert "node_embeddings" in result._atoms_group

    def test_graph_embeddings_pool_every_feature(
        self, demo_model, simple_batch
    ) -> None:
        """Every graph-embedding feature is the sum of its graph's node rows."""
        hidden_dim = demo_model.embedding_shapes["node_embeddings"][-1]
        assert hidden_dim > 1

        result = demo_model.compute_embeddings(simple_batch)

        expected = torch.stack(
            [
                result.node_embeddings[:3].sum(dim=0),
                result.node_embeddings[3:].sum(dim=0),
            ]
        )
        torch.testing.assert_close(result.graph_embeddings, expected)

    def test_node_embeddings_stay_at_node_level_after_reassignment(
        self, demo_model, simple_batch
    ) -> None:
        """A public reassignment of the written key routes back to the atoms group."""
        hidden_dim = demo_model.embedding_shapes["node_embeddings"][-1]
        result = demo_model.compute_embeddings(simple_batch)

        replacement = torch.ones(result.num_nodes, hidden_dim)
        result.node_embeddings = replacement

        torch.testing.assert_close(result.node_embeddings, replacement)
        assert [d.node_embeddings.shape[0] for d in result.to_data_list()] == [3, 2]

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

    def test_tensile_positive_sign(self):
        displacement = torch.zeros(1, 3, 3, requires_grad=True)
        cell = torch.eye(3).unsqueeze(0)
        energy = 2.0 * displacement[0, 0, 0]
        stresses = autograd_stresses(energy, displacement, cell, num_graphs=1)
        expected = torch.zeros(1, 3, 3)
        expected[0, 0, 0] = 2.0
        torch.testing.assert_close(stresses, expected)

    def test_multiple_systems(self):
        displacement = torch.randn(3, 3, 3, requires_grad=True)
        cell = torch.eye(3).unsqueeze(0).expand(3, -1, -1) * 10.0
        energy = (displacement**2).sum()
        stresses = autograd_stresses(energy, displacement, cell, num_graphs=3)
        assert stresses.shape == (3, 3, 3)


class TestAutogradForcesAndStresses:
    """Tests for merged force and stress autograd utility."""

    def test_matches_separate_autograd_calls(self):
        positions = torch.randn(4, 3, dtype=torch.float64, requires_grad=True)
        cell = torch.stack(
            [
                torch.eye(3, dtype=torch.float64) * 5.0,
                torch.eye(3, dtype=torch.float64) * 8.0,
            ]
        )
        batch_idx = torch.tensor([0, 0, 1, 1])
        scaled_pos, _, displacement = prepare_strain(positions, cell, batch_idx)
        energy = (scaled_pos**2).sum()

        forces, stresses = autograd_forces_and_stresses(
            energy,
            scaled_pos,
            displacement,
            cell,
            num_graphs=2,
        )

        positions_ref = positions.detach().clone().requires_grad_(True)
        scaled_ref, _, displacement_ref = prepare_strain(positions_ref, cell, batch_idx)
        energy_ref = (scaled_ref**2).sum()
        expected_forces = autograd_forces(energy_ref, scaled_ref, retain_graph=True)
        expected_stresses = autograd_stresses(
            energy_ref, displacement_ref, cell, num_graphs=2
        )

        torch.testing.assert_close(forces, expected_forces)
        torch.testing.assert_close(stresses, expected_stresses)

    def test_uses_one_autograd_call(self, monkeypatch):
        real_grad = torch.autograd.grad
        calls = []

        def wrapped_grad(outputs, inputs, *args, **kwargs):
            calls.append(inputs)
            return real_grad(outputs, inputs, *args, **kwargs)

        monkeypatch.setattr(torch.autograd, "grad", wrapped_grad)

        positions = torch.randn(3, 3, requires_grad=True)
        cell = torch.eye(3).unsqueeze(0) * 10.0
        batch_idx = torch.zeros(3, dtype=torch.long)
        scaled_pos, _, displacement = prepare_strain(positions, cell, batch_idx)
        energy = (scaled_pos**2).sum()

        autograd_forces_and_stresses(
            energy,
            scaled_pos,
            displacement,
            cell,
            num_graphs=1,
        )

        assert len(calls) == 1
        assert calls[0][0] is scaled_pos
        assert calls[0][1] is displacement

    def test_retain_graph_allows_later_autograd_call(self):
        positions = torch.randn(3, 3, requires_grad=True)
        cell = torch.eye(3).unsqueeze(0) * 10.0
        batch_idx = torch.zeros(3, dtype=torch.long)
        scaled_pos, _, displacement = prepare_strain(positions, cell, batch_idx)
        energy = (scaled_pos**2).sum()

        autograd_forces_and_stresses(
            energy,
            scaled_pos,
            displacement,
            cell,
            num_graphs=1,
            retain_graph=True,
        )
        forces = autograd_forces(energy, scaled_pos)

        assert forces.shape == scaled_pos.shape


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
