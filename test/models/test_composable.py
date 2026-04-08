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
"""Comprehensive tests for ComposableModelWrapper."""

from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import (
    ModelCard,
    ModelConfig,
    NeighborConfig,
    NeighborListFormat,
)
from nvalchemi.models.composable import ComposableModelWrapper
from nvalchemi.models.demo import DemoModelWrapper

# ---------------------------------------------------------------------------
# Mock model helpers
# ---------------------------------------------------------------------------


class _StressModel(DemoModelWrapper):
    """DemoModelWrapper subclass that also reports zero stress."""

    @property
    def model_card(self) -> ModelCard:
        base = super().model_card
        return ModelCard(
            forces_via_autograd=base.forces_via_autograd,
            supports_energies=True,
            supports_forces=True,
            supports_stresses=True,
            needs_pbc=False,
            neighbor_config=None,
        )

    def adapt_output(self, model_output, data):
        M = data.num_graphs if hasattr(data, "num_graphs") else 1
        return OrderedDict(
            [
                ("energy", model_output["energy"]),
                ("forces", model_output["forces"]),
                (
                    "stress",
                    torch.zeros(
                        M,
                        3,
                        3,
                        device=data.positions.device,
                        dtype=data.positions.dtype,
                    ),
                ),
            ]
        )


class _PbcModel(DemoModelWrapper):
    """DemoModelWrapper subclass that declares needs_pbc=True."""

    @property
    def model_card(self) -> ModelCard:
        base = super().model_card
        return ModelCard(
            forces_via_autograd=base.forces_via_autograd,
            supports_energies=base.supports_energies,
            supports_forces=base.supports_forces,
            supports_stresses=False,
            needs_pbc=True,
            neighbor_config=None,
        )


class _NoEnergyModel(DemoModelWrapper):
    """DemoModelWrapper subclass that declares supports_energies=False."""

    @property
    def model_card(self) -> ModelCard:
        base = super().model_card
        return ModelCard(
            forces_via_autograd=base.forces_via_autograd,
            supports_energies=False,
            supports_forces=base.supports_forces,
            supports_stresses=False,
            needs_pbc=False,
            neighbor_config=None,
        )

    def adapt_output(self, model_output, data):
        return OrderedDict([("forces", model_output["forces"])])

    def forward(self, data, **kwargs):
        model_outputs = DemoModelWrapper.forward.__wrapped__(self, data, **kwargs)
        # Strip energy from output
        return OrderedDict([("forces", model_outputs["forces"])])


class _CooNeighborModel(DemoModelWrapper):
    """Model with a COO neighbor config (cutoff=3.0)."""

    @property
    def model_card(self) -> ModelCard:
        base = super().model_card
        return ModelCard(
            forces_via_autograd=base.forces_via_autograd,
            supports_energies=base.supports_energies,
            supports_forces=base.supports_forces,
            supports_stresses=False,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=3.0,
                format=NeighborListFormat.COO,
                half_list=False,
            ),
        )


class _MatrixNeighborModel(DemoModelWrapper):
    """Model with a MATRIX neighbor config (cutoff=5.0, max_neighbors=64)."""

    @property
    def model_card(self) -> ModelCard:
        base = super().model_card
        return ModelCard(
            forces_via_autograd=base.forces_via_autograd,
            supports_energies=base.supports_energies,
            supports_forces=base.supports_forces,
            supports_stresses=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=5.0,
                format=NeighborListFormat.MATRIX,
                half_list=False,
                max_neighbors=64,
            ),
        )

    def adapt_output(self, model_output, data):
        M = data.num_graphs if hasattr(data, "num_graphs") else 1
        return OrderedDict(
            [
                ("energy", model_output["energy"]),
                ("forces", model_output["forces"]),
                (
                    "stress",
                    torch.zeros(
                        M,
                        3,
                        3,
                        device=data.positions.device,
                        dtype=data.positions.dtype,
                    ),
                ),
            ]
        )


class _HalfListCooNeighborModel(DemoModelWrapper):
    """Model with a COO neighbor config and half_list=True (cannot be composed)."""

    @property
    def model_card(self) -> ModelCard:
        base = super().model_card
        return ModelCard(
            forces_via_autograd=base.forces_via_autograd,
            supports_energies=base.supports_energies,
            supports_forces=base.supports_forces,
            supports_stresses=False,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=3.0,
                format=NeighborListFormat.COO,
                half_list=True,
            ),
        )


class _AutoGradModel(DemoModelWrapper):
    """DemoModelWrapper subclass that declares forces_via_autograd=True."""

    @property
    def model_card(self) -> ModelCard:
        base = super().model_card
        return ModelCard(
            forces_via_autograd=True,
            supports_energies=base.supports_energies,
            supports_forces=base.supports_forces,
            supports_stresses=False,
            needs_pbc=False,
            neighbor_config=None,
        )


class _NonAutoGradModel(DemoModelWrapper):
    """DemoModelWrapper subclass that declares forces_via_autograd=False."""

    @property
    def model_card(self) -> ModelCard:
        base = super().model_card
        return ModelCard(
            forces_via_autograd=False,
            supports_energies=base.supports_energies,
            supports_forces=base.supports_forces,
            supports_stresses=False,
            needs_pbc=False,
            neighbor_config=None,
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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def demo_model_a():
    return DemoModelWrapper()


@pytest.fixture
def demo_model_b():
    return DemoModelWrapper()


@pytest.fixture
def demo_model_c():
    return DemoModelWrapper()


@pytest.fixture
def stress_model():
    return _StressModel()


@pytest.fixture
def simple_batch():
    return _make_batch(n_systems=2, n_atoms_each=4)


# ---------------------------------------------------------------------------
# TestConstruction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Construction and flattening of ComposableModelWrapper."""

    def test_direct_construction_two_models(self, demo_model_a, demo_model_b):
        wrapper = ComposableModelWrapper(demo_model_a, demo_model_b)
        assert len(wrapper.models) == 2

    def test_add_operator_produces_additive_wrapper(self, demo_model_a, demo_model_b):
        wrapper = demo_model_a + demo_model_b
        assert isinstance(wrapper, ComposableModelWrapper)

    def test_add_operator_equivalent_to_direct_construction(
        self, demo_model_a, demo_model_b
    ):
        via_add = demo_model_a + demo_model_b
        via_direct = ComposableModelWrapper(demo_model_a, demo_model_b)
        assert len(via_add.models) == len(via_direct.models) == 2

    def test_binary_composition_has_two_models(self, demo_model_a, demo_model_b):
        wrapper = demo_model_a + demo_model_b
        assert len(wrapper.models) == 2

    def test_nesting_left_is_flattened(self, demo_model_a, demo_model_b, demo_model_c):
        """(model_a + model_b) + model_c flattens to 3 models."""
        inner = demo_model_a + demo_model_b
        outer = inner + demo_model_c
        assert isinstance(outer, ComposableModelWrapper)
        assert len(outer.models) == 3
        # The outer should not contain another ComposableModelWrapper
        for m in outer.models:
            assert not isinstance(m, ComposableModelWrapper)

    def test_nesting_right_is_flattened(self, demo_model_a, demo_model_b, demo_model_c):
        """model_a + (model_b + model_c) flattens to 3 models."""
        inner = demo_model_b + demo_model_c
        outer = demo_model_a + inner
        assert isinstance(outer, ComposableModelWrapper)
        assert len(outer.models) == 3
        for m in outer.models:
            assert not isinstance(m, ComposableModelWrapper)

    def test_deep_nesting_is_fully_flattened(self):
        """Three levels of nesting produce a single flat wrapper."""
        models = [DemoModelWrapper() for _ in range(4)]
        # Build ((a + b) + c) + d
        wrapper = ((models[0] + models[1]) + models[2]) + models[3]
        assert len(wrapper.models) == 4
        for m in wrapper.models:
            assert not isinstance(m, ComposableModelWrapper)

    def test_models_stored_as_module_list(self, demo_model_a, demo_model_b):
        """wrapper.models must be an nn.ModuleList so parameters are registered."""
        import torch.nn as nn

        wrapper = ComposableModelWrapper(demo_model_a, demo_model_b)
        assert isinstance(wrapper.models, nn.ModuleList)

    def test_three_model_direct_construction(self):
        """Direct construction with three models."""
        a, b, c = DemoModelWrapper(), DemoModelWrapper(), DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b, c)
        assert len(wrapper.models) == 3


# ---------------------------------------------------------------------------
# TestModelConfigPropagation
# ---------------------------------------------------------------------------


class TestModelConfigPropagation:
    """model_config property must propagate to all sub-models."""

    def test_attribute_set_propagates_to_sub_models(self, demo_model_a, demo_model_b):
        wrapper = ComposableModelWrapper(demo_model_a, demo_model_b)
        wrapper.model_config.compute_stresses = True
        # Both sub-models should see the same config object
        for m in wrapper.models:
            assert m.model_config.compute_stresses is True

    def test_full_config_replacement_propagates(self, demo_model_a, demo_model_b):
        wrapper = ComposableModelWrapper(demo_model_a, demo_model_b)
        new_config = ModelConfig(compute_stresses=True)
        wrapper.model_config = new_config
        for m in wrapper.models:
            assert m.model_config.compute_stresses is True

    def test_config_propagates_to_all_three_models(self):
        models = [DemoModelWrapper() for _ in range(3)]
        wrapper = ComposableModelWrapper(*models)
        wrapper.model_config.compute_stresses = True
        for m in wrapper.models:
            assert m.model_config.compute_stresses is True

    def test_wrapper_config_getter_returns_current_config(
        self, demo_model_a, demo_model_b
    ):
        wrapper = ComposableModelWrapper(demo_model_a, demo_model_b)
        config = ModelConfig(compute_stresses=True)
        wrapper.model_config = config
        assert wrapper.model_config.compute_stresses is True

    def test_compute_forces_propagates(self, demo_model_a, demo_model_b):
        wrapper = ComposableModelWrapper(demo_model_a, demo_model_b)
        wrapper.model_config.compute_forces = False
        for m in wrapper.models:
            assert m.model_config.compute_forces is False

    def test_sub_model_config_is_same_object_after_set(
        self, demo_model_a, demo_model_b
    ):
        """After setting wrapper.model_config, sub-models share the exact same config."""
        wrapper = ComposableModelWrapper(demo_model_a, demo_model_b)
        new_config = ModelConfig()
        wrapper.model_config = new_config
        for m in wrapper.models:
            assert m.model_config is new_config


# ---------------------------------------------------------------------------
# TestModelCardSynthesis
# ---------------------------------------------------------------------------


class TestModelCardSynthesis:
    """Synthesised ModelCard must aggregate sub-model capabilities correctly."""

    def test_forces_via_autograd_is_any(self):
        """forces_via_autograd: True if any sub-model has it True."""
        autograd = _AutoGradModel()  # forces_via_autograd=True
        non_autograd = _NonAutoGradModel()  # forces_via_autograd=False
        wrapper = ComposableModelWrapper(autograd, non_autograd)
        assert wrapper.model_card.forces_via_autograd is True

    def test_forces_via_autograd_false_when_all_false(self):
        a = _NonAutoGradModel()
        b = _NonAutoGradModel()
        wrapper = ComposableModelWrapper(a, b)
        assert wrapper.model_card.forces_via_autograd is False

    def test_two_autograd_models_raises_not_implemented(self):
        """Composing two autograd-forces models is not yet supported."""
        a = _AutoGradModel()  # forces_via_autograd=True
        b = _AutoGradModel()
        with pytest.raises(NotImplementedError):
            ComposableModelWrapper(a, b)

    def test_supports_energies_is_all(self):
        """supports_energies: True only when all sub-models support it."""
        with_energy = DemoModelWrapper()  # supports_energies=True
        no_energy = _NoEnergyModel()  # supports_energies=False
        wrapper = ComposableModelWrapper(with_energy, no_energy)
        assert wrapper.model_card.supports_energies is False

    def test_supports_energies_true_when_all_support(self):
        a = DemoModelWrapper()
        b = DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b)
        assert wrapper.model_card.supports_energies is True

    def test_supports_stresses_is_all(self):
        """supports_stresses: True only when all sub-models support it."""
        stress = _StressModel()  # supports_stresses=True
        no_stress = DemoModelWrapper()  # supports_stresses=False
        wrapper = ComposableModelWrapper(stress, no_stress)
        assert wrapper.model_card.supports_stresses is False

    def test_supports_stresses_true_when_all_support(self):
        a = _StressModel()
        b = _StressModel()
        wrapper = ComposableModelWrapper(a, b)
        assert wrapper.model_card.supports_stresses is True

    def test_needs_pbc_is_any(self):
        """needs_pbc: True if any sub-model needs PBC."""
        pbc = _PbcModel()  # needs_pbc=True
        no_pbc = DemoModelWrapper()  # needs_pbc=False
        wrapper = ComposableModelWrapper(pbc, no_pbc)
        assert wrapper.model_card.needs_pbc is True

    def test_needs_pbc_false_when_none_need_it(self):
        a = DemoModelWrapper()
        b = DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b)
        assert wrapper.model_card.needs_pbc is False

    def test_neighbor_config_is_none_when_no_sub_models_have_it(self):
        a = DemoModelWrapper()  # neighbor_config=None
        b = DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b)
        assert wrapper.model_card.neighbor_config is None

    def test_neighbor_config_cutoff_is_max(self):
        coo = _CooNeighborModel()  # cutoff=3.0
        matrix = _MatrixNeighborModel()  # cutoff=5.0
        wrapper = ComposableModelWrapper(coo, matrix)
        nc = wrapper.model_card.neighbor_config
        assert nc is not None
        assert nc.cutoff == 5.0

    def test_neighbor_config_format_is_matrix_if_any_matrix(self):
        """MATRIX wins if any sub-model uses MATRIX format."""
        coo = _CooNeighborModel()  # COO
        matrix = _MatrixNeighborModel()  # MATRIX
        wrapper = ComposableModelWrapper(coo, matrix)
        nc = wrapper.model_card.neighbor_config
        assert nc is not None
        assert nc.format == NeighborListFormat.MATRIX

    def test_neighbor_config_format_is_coo_when_all_coo(self):
        a = _CooNeighborModel()
        b = _CooNeighborModel()
        wrapper = ComposableModelWrapper(a, b)
        nc = wrapper.model_card.neighbor_config
        assert nc is not None
        assert nc.format == NeighborListFormat.COO

    def test_half_list_sub_model_raises_value_error(self):
        """Composing a sub-model with half_list=True must raise ValueError.

        The composite always builds a full neighbor list. Converting full→half
        is not supported, so half_list=True sub-models cannot be composed.
        """
        a = _HalfListCooNeighborModel()  # half_list=True
        b = _CooNeighborModel()  # half_list=False
        with pytest.raises(ValueError, match="half_list"):
            ComposableModelWrapper(a, b)

    def test_neighbor_config_max_neighbors_is_max(self):
        """max_neighbors of composite = max over sub-model configs that set it."""
        a = _MatrixNeighborModel()  # max_neighbors=64
        b = _CooNeighborModel()  # max_neighbors=None
        wrapper = ComposableModelWrapper(a, b)
        nc = wrapper.model_card.neighbor_config
        assert nc is not None
        assert nc.max_neighbors == 64

    def test_neighbor_config_none_when_only_no_neighbor_models(self):
        a = DemoModelWrapper()
        b = _StressModel()
        wrapper = ComposableModelWrapper(a, b)
        assert wrapper.model_card.neighbor_config is None

    def test_model_card_is_fresh_per_call(self):
        """model_card property must not cache stale data between calls."""
        a = DemoModelWrapper()
        b = DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b)
        card1 = wrapper.model_card
        card2 = wrapper.model_card
        # Both calls should yield equivalent results
        assert card1.forces_via_autograd == card2.forces_via_autograd
        assert card1.supports_energies == card2.supports_energies


# ---------------------------------------------------------------------------
# TestForwardPass
# ---------------------------------------------------------------------------


class TestForwardPass:
    """Forward-pass behaviour: summation and output ordering."""

    def test_energies_are_summed(self, simple_batch):
        """Energies from both models must be summed element-wise."""
        a = DemoModelWrapper()
        b = DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b)

        # Collect individual outputs
        result_a = a(simple_batch)
        result_b = b(simple_batch)
        result_wrapper = wrapper(simple_batch)

        expected_energies = result_a["energy"] + result_b["energy"]
        torch.testing.assert_close(result_wrapper["energy"], expected_energies)

    def test_forces_are_summed(self, simple_batch):
        """Forces from both models must be summed element-wise."""
        a = DemoModelWrapper()
        b = DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b)

        result_a = a(simple_batch)
        result_b = b(simple_batch)
        result_wrapper = wrapper(simple_batch)

        expected_forces = result_a["forces"] + result_b["forces"]
        torch.testing.assert_close(result_wrapper["forces"], expected_forces)

    def test_stress_is_summed_when_both_models_produce_it(self, simple_batch):
        """Stress from both stress-supporting models must be summed."""
        a = _StressModel()
        b = _StressModel()
        wrapper = ComposableModelWrapper(a, b)
        wrapper.model_config = ModelConfig(compute_stresses=True)

        result_a = a(simple_batch)
        result_b = b(simple_batch)
        result_wrapper = wrapper(simple_batch)

        assert "stress" in result_wrapper
        expected_stress = result_a["stress"] + result_b["stress"]
        torch.testing.assert_close(result_wrapper["stress"], expected_stress)

    def test_output_has_canonical_key_order_energies_forces(self, simple_batch):
        """Keys must appear in canonical order: energy → forces."""
        a = DemoModelWrapper()
        b = DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b)
        result = wrapper(simple_batch)

        keys = list(result.keys())
        assert keys.index("energy") < keys.index("forces")

    def test_output_has_canonical_key_order_energies_forces_stress(self, simple_batch):
        """Keys must appear in canonical order: energy → forces → stress."""
        a = _StressModel()
        b = _StressModel()
        wrapper = ComposableModelWrapper(a, b)
        wrapper.model_config = ModelConfig(compute_stresses=True)
        result = wrapper(simple_batch)

        keys = list(result.keys())
        assert "energy" in keys
        assert "forces" in keys
        assert "stress" in keys
        assert keys.index("energy") < keys.index("forces") < keys.index("stress")

    def test_output_is_ordered_dict(self, simple_batch):
        a = DemoModelWrapper()
        b = DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b)
        result = wrapper(simple_batch)
        assert isinstance(result, OrderedDict)

    def test_stress_absent_when_no_sub_model_produces_it(self, simple_batch):
        """stress key must not appear if neither model produces it."""
        a = DemoModelWrapper()  # does not produce stress
        b = DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b)
        result = wrapper(simple_batch)
        assert "stress" not in result

    def test_energies_shape_is_batch_size_by_one(self, simple_batch):
        a = DemoModelWrapper()
        b = DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b)
        result = wrapper(simple_batch)
        M = simple_batch.num_graphs
        assert result["energy"].shape == (M, 1)

    def test_forces_shape_matches_n_atoms(self, simple_batch):
        a = DemoModelWrapper()
        b = DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b)
        result = wrapper(simple_batch)
        N = simple_batch.positions.shape[0]
        assert result["forces"].shape == (N, 3)

    def test_stress_shape_is_batch_size_by_3_by_3(self, simple_batch):
        a = _StressModel()
        b = _StressModel()
        wrapper = ComposableModelWrapper(a, b)
        wrapper.model_config = ModelConfig(compute_stresses=True)
        result = wrapper(simple_batch)
        M = simple_batch.num_graphs
        assert result["stress"].shape == (M, 3, 3)

    def test_missing_key_in_one_sub_model_does_not_raise(self, simple_batch):
        """A key present in only one sub-model should not cause KeyError."""
        # model_a produces energy+forces; model_b produces only forces
        # (simulated via _NoEnergyModel producing only "forces")
        a = DemoModelWrapper()

        # Build a model that returns only forces
        class _ForcesOnlyModel(DemoModelWrapper):
            def forward(self, data, **kwargs):
                out = super().forward(data, **kwargs)
                return OrderedDict([("forces", out["forces"])])

        b = _ForcesOnlyModel()
        wrapper = ComposableModelWrapper(a, b)
        # Should not raise
        result = wrapper(simple_batch)
        # energy only come from model_a
        assert "energy" in result
        # forces come from both models
        assert "forces" in result

    def test_three_model_energies_summed(self, simple_batch):
        """Energies from three models must all be summed."""
        a = DemoModelWrapper()
        b = DemoModelWrapper()
        c = DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b, c)

        ra = a(simple_batch)
        rb = b(simple_batch)
        rc = c(simple_batch)
        result = wrapper(simple_batch)

        expected = ra["energy"] + rb["energy"] + rc["energy"]
        torch.testing.assert_close(result["energy"], expected)

    def test_non_additive_output_written_back_to_batch(self, simple_batch):
        """Non-composable outputs should be written to the batch object."""

        class _ExtraOutputModel(DemoModelWrapper):
            def forward(self, data, **kwargs):
                out = super().forward(data, **kwargs)
                # Inject a non-additive key
                return OrderedDict(
                    list(out.items()) + [("custom_key", torch.tensor(42.0))]
                )

        a = DemoModelWrapper()
        b = _ExtraOutputModel()
        wrapper = ComposableModelWrapper(a, b)
        wrapper(simple_batch)
        # The non-additive "custom_key" should have been written back to the batch
        assert hasattr(simple_batch, "custom_key")

    def test_non_additive_output_last_write_wins(self, simple_batch):
        """For non-additive keys, the last sub-model's value wins."""

        class _ModelWithCustomVal(DemoModelWrapper):
            def __init__(self, value: float):
                super().__init__()
                self._custom_value = value

            def forward(self, data, **kwargs):
                out = super().forward(data, **kwargs)
                return OrderedDict(
                    list(out.items()) + [("tag", torch.tensor(self._custom_value))]
                )

        a = _ModelWithCustomVal(1.0)
        b = _ModelWithCustomVal(2.0)
        wrapper = ComposableModelWrapper(a, b)
        wrapper(simple_batch)
        # b runs last, so its value (2.0) should win
        assert float(simple_batch.tag) == pytest.approx(2.0)

    def test_forward_with_single_system_batch(self):
        """Forward pass works correctly with a single-system batch."""
        a = DemoModelWrapper()
        b = DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b)
        batch = _make_batch(n_systems=1, n_atoms_each=5)
        result = wrapper(batch)
        assert result["energy"].shape == (1, 1)
        assert result["forces"].shape == (5, 3)

    def test_forward_with_multi_system_batch(self):
        """Forward pass works correctly with a multi-system batch."""
        a = DemoModelWrapper()
        b = DemoModelWrapper()
        wrapper = ComposableModelWrapper(a, b)
        M = 4
        batch = _make_batch(n_systems=M, n_atoms_each=3)
        result = wrapper(batch)
        assert result["energy"].shape == (M, 1)


# ---------------------------------------------------------------------------
# TestNotImplementedMethods
# ---------------------------------------------------------------------------


class TestNotImplementedMethods:
    """Methods that are not meaningful for composite models."""

    def test_compute_embeddings_raises(self, demo_model_a, demo_model_b, simple_batch):
        wrapper = ComposableModelWrapper(demo_model_a, demo_model_b)
        with pytest.raises(NotImplementedError):
            wrapper.compute_embeddings(simple_batch)

    def test_compute_embeddings_error_message_mentions_sub_models(
        self, demo_model_a, demo_model_b, simple_batch
    ):
        wrapper = ComposableModelWrapper(demo_model_a, demo_model_b)
        with pytest.raises(NotImplementedError, match="sub-model"):
            wrapper.compute_embeddings(simple_batch)

    def test_export_model_raises(self, demo_model_a, demo_model_b, tmp_path):
        wrapper = ComposableModelWrapper(demo_model_a, demo_model_b)
        with pytest.raises(NotImplementedError):
            wrapper.export_model(tmp_path / "model.pt")

    def test_export_model_error_message_mentions_sub_models(
        self, demo_model_a, demo_model_b, tmp_path
    ):
        wrapper = ComposableModelWrapper(demo_model_a, demo_model_b)
        with pytest.raises(NotImplementedError, match="sub-model"):
            wrapper.export_model(tmp_path / "model.pt")

    def test_embedding_shapes_returns_empty_dict(self, demo_model_a, demo_model_b):
        wrapper = ComposableModelWrapper(demo_model_a, demo_model_b)
        assert wrapper.embedding_shapes == {}
