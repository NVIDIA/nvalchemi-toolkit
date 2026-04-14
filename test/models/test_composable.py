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
"""Tests for PipelineModelWrapper -- migration from the old ComposableModelWrapper patterns.

These tests verify that the composition patterns previously covered by
ComposableModelWrapper work correctly under PipelineModelWrapper.  Tests
that are already covered in ``test_pipeline.py`` (autograd groups, wiring,
fan-out, etc.) are not duplicated here; this file focuses on the basic
additive-sum patterns, model-config synthesis, output shapes, and edge cases
that the old composable test suite exercised.
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
# Mock model helpers
# ---------------------------------------------------------------------------


class _SimpleModel(nn.Module, BaseModelMixin):
    """Returns fixed energies and forces (analytical, no autograd)."""

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


class _StressModel(nn.Module, BaseModelMixin):
    """Returns energies, forces, and zero stresses."""

    def __init__(self, energy: float = 1.0, force_val: float = 0.5) -> None:
        super().__init__()
        self._energy = energy
        self._force_val = force_val
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            autograd_outputs=frozenset(),
            needs_pbc=False,
            active_outputs={"energy", "forces", "stress"},
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
            stress=torch.zeros(B, 3, 3, dtype=data.positions.dtype),
        )


class _PbcModel(_SimpleModel):
    """Declares needs_pbc=True."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            autograd_outputs=frozenset(),
            needs_pbc=True,
            supports_pbc=True,
            active_outputs={"energy", "forces"},
        )


class _ForceOnlyModel(nn.Module, BaseModelMixin):
    """Returns only forces (no energies)."""

    def __init__(self, force_val: float = 0.5) -> None:
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


class _CooNeighborModel(_SimpleModel):
    """Model with a COO neighbor config (cutoff=3.0)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            autograd_outputs=frozenset(),
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=3.0, format=NeighborListFormat.COO, half_list=False
            ),
            active_outputs={"energy", "forces"},
        )


class _MatrixNeighborModel(_StressModel):
    """Model with a MATRIX neighbor config (cutoff=5.0)."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            autograd_outputs=frozenset(),
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=5.0,
                format=NeighborListFormat.MATRIX,
                half_list=False,
            ),
            active_outputs={"energy", "forces", "stress"},
        )


class _HalfListCooModel(_SimpleModel):
    """Model with a COO neighbor config and half_list=True."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            autograd_outputs=frozenset(),
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=3.0, format=NeighborListFormat.COO, half_list=True
            ),
            active_outputs={"energy", "forces"},
        )


# ---------------------------------------------------------------------------
# Batch factory helper
# ---------------------------------------------------------------------------


def _make_atomic_data(n_atoms: int = 4, seed: int = 0) -> AtomicData:
    """Return a minimal AtomicData suitable for model forward tests."""
    g = torch.Generator()
    g.manual_seed(seed)
    data = AtomicData(
        positions=torch.randn(n_atoms, 3, generator=g),
        atomic_numbers=torch.randint(1, 10, (n_atoms,), dtype=torch.long, generator=g),
        atomic_masses=torch.ones(n_atoms),
        forces=torch.zeros(n_atoms, 3),
        energy=torch.zeros(1, 1),
    )
    data.add_node_property("velocities", torch.zeros(n_atoms, 3))
    return data


def _make_batch(n_systems: int = 2, n_atoms_each: int = 4, seed: int = 0) -> Batch:
    data_list = [
        _make_atomic_data(n_atoms_each, seed=seed + i) for i in range(n_systems)
    ]
    return Batch.from_data_list(data_list)


def _make_pipeline(*models: BaseModelMixin) -> PipelineModelWrapper:
    """Build a pipeline with each model in its own direct-force group."""
    groups = [PipelineGroup(steps=[m]) for m in models]
    return PipelineModelWrapper(groups=groups)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_batch():
    return _make_batch(n_systems=2, n_atoms_each=4)


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Construction of PipelineModelWrapper from separate groups."""

    def test_two_model_pipeline(self):
        pipe = _make_pipeline(_SimpleModel(), _SimpleModel())
        assert len(pipe.groups) == 2

    def test_three_model_pipeline(self):
        a, b, c = _SimpleModel(), _SimpleModel(), _SimpleModel()
        pipe = _make_pipeline(a, b, c)
        assert len(pipe.groups) == 3

    def test_steps_are_normalized_to_pipeline_step(self):
        m = _SimpleModel()
        pipe = PipelineModelWrapper(groups=[PipelineGroup(steps=[m])])
        assert isinstance(pipe.groups[0].steps[0], PipelineStep)

    def test_single_model_pipeline(self):
        pipe = _make_pipeline(_SimpleModel())
        assert len(pipe.groups) == 1


# ---------------------------------------------------------------------------
# TestModelConfigSynthesis
# ---------------------------------------------------------------------------


class TestModelConfigSynthesis:
    """Synthesised ModelConfig must aggregate sub-model capabilities."""

    def test_outputs_union(self):
        """Pipeline outputs are the union of all sub-model outputs."""
        a = _SimpleModel()  # {energies, forces}
        b = _StressModel()  # {energies, forces, stresses}
        pipe = _make_pipeline(a, b)
        cfg = pipe.model_config
        assert "energy" in cfg.outputs
        assert "forces" in cfg.outputs
        assert "stress" in cfg.outputs

    def test_needs_pbc_is_any(self):
        """needs_pbc: True if any sub-model needs PBC."""
        pbc = _PbcModel()
        no_pbc = _SimpleModel()
        pipe = _make_pipeline(pbc, no_pbc)
        assert pipe.model_config.needs_pbc is True

    def test_needs_pbc_false_when_none_need_it(self):
        pipe = _make_pipeline(_SimpleModel(), _SimpleModel())
        assert pipe.model_config.needs_pbc is False

    def test_neighbor_config_is_none_when_no_sub_models_have_it(self):
        pipe = _make_pipeline(_SimpleModel(), _SimpleModel())
        assert pipe.model_config.neighbor_config is None

    def test_neighbor_config_cutoff_is_max(self):
        coo = _CooNeighborModel()  # cutoff=3.0
        matrix = _MatrixNeighborModel()  # cutoff=5.0
        pipe = _make_pipeline(coo, matrix)
        nc = pipe.model_config.neighbor_config
        assert nc is not None
        assert nc.cutoff == 5.0

    def test_neighbor_config_format_is_matrix_if_any_matrix(self):
        """MATRIX wins if any sub-model uses MATRIX format."""
        coo = _CooNeighborModel()
        matrix = _MatrixNeighborModel()
        pipe = _make_pipeline(coo, matrix)
        nc = pipe.model_config.neighbor_config
        assert nc is not None
        assert nc.format == NeighborListFormat.MATRIX

    def test_neighbor_config_format_is_coo_when_all_coo(self):
        a = _CooNeighborModel()
        b = _CooNeighborModel()
        pipe = _make_pipeline(a, b)
        nc = pipe.model_config.neighbor_config
        assert nc is not None
        assert nc.format == NeighborListFormat.COO

    def test_half_list_mismatch_raises_value_error(self):
        """Composing a half_list model with a full-list model raises ValueError."""
        a = _HalfListCooModel()
        b = _CooNeighborModel()
        with pytest.raises(ValueError, match="half_list"):
            _make_pipeline(a, b)

    def test_neighbor_config_none_when_only_no_neighbor_models(self):
        pipe = _make_pipeline(_SimpleModel(), _StressModel())
        assert pipe.model_config.neighbor_config is None


# ---------------------------------------------------------------------------
# TestForwardPass
# ---------------------------------------------------------------------------


class TestForwardPass:
    """Forward-pass behaviour: summation and output shapes."""

    def test_energies_are_summed(self, simple_batch):
        """Energies from both models must be summed element-wise."""
        a = _SimpleModel(energy=1.0)
        b = _SimpleModel(energy=2.0)
        pipe = _make_pipeline(a, b)
        result = pipe(simple_batch)
        torch.testing.assert_close(
            result["energy"],
            torch.full((2, 1), 3.0),
        )

    def test_forces_are_summed(self, simple_batch):
        """Forces from both models must be summed element-wise."""
        a = _SimpleModel(force_val=0.5)
        b = _SimpleModel(force_val=0.3)
        pipe = _make_pipeline(a, b)
        result = pipe(simple_batch)
        N = simple_batch.positions.shape[0]
        torch.testing.assert_close(
            result["forces"],
            torch.full((N, 3), 0.8),
        )

    def test_stresses_are_summed_when_both_produce_them(self, simple_batch):
        """Stresses from both stress-supporting models must be summed."""
        a = _StressModel(energy=1.0, force_val=0.5)
        b = _StressModel(energy=2.0, force_val=0.3)
        pipe = _make_pipeline(a, b)
        result = pipe(simple_batch)
        assert "stress" in result
        M = simple_batch.num_graphs
        # Both produce zero stresses, so sum is also zero
        torch.testing.assert_close(
            result["stress"],
            torch.zeros(M, 3, 3),
        )

    def test_energies_shape_is_batch_size_by_one(self, simple_batch):
        pipe = _make_pipeline(_SimpleModel(), _SimpleModel())
        result = pipe(simple_batch)
        M = simple_batch.num_graphs
        assert result["energy"].shape == (M, 1)

    def test_forces_shape_matches_n_atoms(self, simple_batch):
        pipe = _make_pipeline(_SimpleModel(), _SimpleModel())
        result = pipe(simple_batch)
        N = simple_batch.positions.shape[0]
        assert result["forces"].shape == (N, 3)

    def test_stresses_shape_is_batch_size_by_3_by_3(self, simple_batch):
        pipe = _make_pipeline(_StressModel(), _StressModel())
        result = pipe(simple_batch)
        M = simple_batch.num_graphs
        assert result["stress"].shape == (M, 3, 3)

    def test_three_model_energies_summed(self, simple_batch):
        """Energies from three models must all be summed."""
        a = _SimpleModel(energy=1.0)
        b = _SimpleModel(energy=2.0)
        c = _SimpleModel(energy=3.0)
        pipe = _make_pipeline(a, b, c)
        result = pipe(simple_batch)
        torch.testing.assert_close(
            result["energy"],
            torch.full((2, 1), 6.0),
        )

    def test_forward_with_single_system_batch(self):
        """Forward pass works correctly with a single-system batch."""
        pipe = _make_pipeline(_SimpleModel(), _SimpleModel())
        batch = _make_batch(n_systems=1, n_atoms_each=5)
        result = pipe(batch)
        assert result["energy"].shape == (1, 1)
        assert result["forces"].shape == (5, 3)

    def test_forward_with_multi_system_batch(self):
        """Forward pass works correctly with a multi-system batch."""
        M = 4
        pipe = _make_pipeline(_SimpleModel(), _SimpleModel())
        batch = _make_batch(n_systems=M, n_atoms_each=3)
        result = pipe(batch)
        assert result["energy"].shape == (M, 1)

    def test_force_correction_pattern(self, simple_batch):
        """A force-only model adds a correction on top of a full model."""
        a = _SimpleModel(energy=1.0, force_val=0.5)
        b = _ForceOnlyModel(force_val=0.1)
        pipe = _make_pipeline(a, b)
        result = pipe(simple_batch)
        N = simple_batch.positions.shape[0]
        # Forces = 0.5 + 0.1 = 0.6
        torch.testing.assert_close(
            result["forces"],
            torch.full((N, 3), 0.6),
        )
        # Energies come only from model a
        torch.testing.assert_close(
            result["energy"],
            torch.full((2, 1), 1.0),
        )


# ---------------------------------------------------------------------------
# TestNotImplementedMethods
# ---------------------------------------------------------------------------


class TestNotImplementedMethods:
    """Methods that are not meaningful for pipeline-composed models."""

    def test_compute_embeddings_raises(self, simple_batch):
        pipe = _make_pipeline(_SimpleModel(), _SimpleModel())
        with pytest.raises(NotImplementedError):
            pipe.compute_embeddings(simple_batch)

    def test_export_model_raises(self, tmp_path):
        pipe = _make_pipeline(_SimpleModel(), _SimpleModel())
        with pytest.raises(NotImplementedError):
            pipe.export_model(tmp_path / "model.pt")

    def test_embedding_shapes_returns_empty_dict(self):
        pipe = _make_pipeline(_SimpleModel(), _SimpleModel())
        assert pipe.embedding_shapes == {}
