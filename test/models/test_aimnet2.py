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
"""Tests for AIMNet2Wrapper.

Since aimnet is an optional dependency that may not be installed, these
tests use a mock AIMNet2Calculator to validate the wrapper logic without
requiring the actual model.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.base import ModelConfig

# ---------------------------------------------------------------------------
# Mock AIMNet2Calculator that mimics the real interface
# ---------------------------------------------------------------------------


class _MockAIMNet2Model(nn.Module):
    """Minimal mock of AIMNet2's internal model."""

    def __init__(self, num_charge_channels: int = 1):
        super().__init__()
        self.num_charge_channels = num_charge_channels
        self.aev = MagicMock()
        self.aev.rc_s = 5.2
        self.aev.rc_v = 5.0
        self.aev.output_size = 256
        self.linear = nn.Linear(3, 1)  # Dummy parameter for device tracking

    def forward(self, model_input: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        coord = model_input["coord"]
        n_atoms = coord.shape[0]
        # Simple energy: sum of squared positions (differentiable)
        energy = (coord**2).sum().unsqueeze(0)
        return {
            "energy": energy,
            "charges": torch.ones(n_atoms, dtype=coord.dtype, device=coord.device)
            * 0.1,
            "aim": torch.randn(n_atoms, 256, dtype=coord.dtype, device=coord.device),
        }


class _MockAIMNet2Calculator:
    """Minimal mock of aimnet.calculators.AIMNet2Calculator."""

    def __init__(self, model: Any = None, device: str = "cpu", **kwargs):
        self.model = model if model is not None else _MockAIMNet2Model()
        self.device = device
        self.keys_out = ["energy", "charges"]

    def mol_flatten(self, data: dict) -> dict:
        """Pass through — already flat for single-system batches."""
        return dict(data)

    def make_nbmat(self, data: dict) -> dict:
        """Add a dummy neighbor matrix."""
        n = data["coord"].shape[0]
        data["nbmat"] = torch.zeros(n, 10, dtype=torch.long)
        return data

    def pad_input(self, data: dict) -> dict:
        """Add one padding atom."""
        n = data["coord"].shape[0]
        data["coord"] = torch.cat([data["coord"], torch.zeros(1, 3)])
        data["numbers"] = torch.cat([data["numbers"], torch.zeros(1, dtype=torch.long)])
        data["nbmat"] = torch.zeros(n + 1, 10, dtype=torch.long)
        return data

    def unpad_output(self, output: dict) -> dict:
        """Strip padding from standard keys."""
        return output


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model():
    return _MockAIMNet2Model()


@pytest.fixture
def simple_batch():
    data = AtomicData(
        positions=torch.randn(5, 3),
        atomic_numbers=torch.tensor([6, 6, 8, 1, 1]),
        forces=torch.zeros(5, 3),
        energy=torch.zeros(1, 1),
    )
    return Batch.from_data_list([data])


def _make_wrapper(model: _MockAIMNet2Model) -> Any:
    """Construct an AIMNet2Wrapper with mock AIMNet2Calculator."""
    import sys

    from nvalchemi._optional import OptionalDependency

    dep = OptionalDependency.AIMNET
    orig_available = dep._available

    # Mock the aimnet.calculators module so the import inside __init__ works
    mock_calculators = MagicMock()
    mock_calculators.AIMNet2Calculator = _MockAIMNet2Calculator
    orig_mod = sys.modules.get("aimnet.calculators")
    orig_aimnet = sys.modules.get("aimnet")
    sys.modules["aimnet"] = MagicMock()
    sys.modules["aimnet.calculators"] = mock_calculators

    dep._available = True
    try:
        from nvalchemi.models.aimnet2 import AIMNet2Wrapper

        wrapper = AIMNet2Wrapper(model)
    finally:
        dep._available = orig_available
        # Restore original module state
        if orig_mod is None:
            sys.modules.pop("aimnet.calculators", None)
        else:
            sys.modules["aimnet.calculators"] = orig_mod
        if orig_aimnet is None:
            sys.modules.pop("aimnet", None)
        else:
            sys.modules["aimnet"] = orig_aimnet
    return wrapper


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAIMNet2WrapperInit:
    """Tests for AIMNet2Wrapper construction."""

    def test_import_guard(self, mock_model):
        """Should raise ImportError when aimnet is not installed."""
        from nvalchemi._optional import OptionalDependency
        from nvalchemi.models.aimnet2 import AIMNet2Wrapper

        dep = OptionalDependency.AIMNET
        orig_available = dep._available
        dep._available = False
        try:
            with pytest.raises(ImportError, match="aimnet.*not installed"):
                AIMNet2Wrapper(mock_model)
        finally:
            dep._available = orig_available

    def test_construction_with_mock(self, mock_model):
        """Wrapper constructs successfully with mock model."""
        wrapper = _make_wrapper(mock_model)
        assert wrapper.model is mock_model
        assert isinstance(wrapper.model_config, ModelConfig)

    def test_nse_detection_standard(self, mock_model):
        """Standard model (1 charge channel) is not NSE."""
        wrapper = _make_wrapper(mock_model)
        assert not wrapper._is_nse

    def test_nse_detection_nse_model(self):
        """NSE model (2 charge channels) is detected."""
        nse_model = _MockAIMNet2Model(num_charge_channels=2)
        wrapper = _make_wrapper(nse_model)
        assert wrapper._is_nse
        assert "spin_charges" in wrapper.model_config.outputs


class TestAIMNet2WrapperModelConfig:
    """Tests for model card correctness."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_model):
        self.wrapper = _make_wrapper(mock_model)

    def test_outputs(self):
        cfg = self.wrapper.model_config
        assert "energy" in cfg.outputs
        assert "forces" in cfg.outputs
        assert "charges" in cfg.outputs

    def test_autograd_outputs(self):
        cfg = self.wrapper.model_config
        assert "forces" in cfg.autograd_outputs
        assert "stress" in cfg.autograd_outputs

    def test_inputs(self):
        cfg = self.wrapper.model_config
        assert "charge" in cfg.required_inputs

    def test_supports_pbc(self):
        assert self.wrapper.model_config.supports_pbc is True
        assert self.wrapper.model_config.needs_pbc is False


class TestAIMNet2WrapperCutoff:
    """Tests for cutoff extraction."""

    def test_cutoff_from_aev(self, mock_model):
        wrapper = _make_wrapper(mock_model)
        assert wrapper._cutoff == 5.2  # max(rc_s=5.2, rc_v=5.0)

    def test_cutoff_default_without_aev(self):
        model = _MockAIMNet2Model()
        model.aev = None
        wrapper = _make_wrapper(model)
        assert wrapper._cutoff == 5.0  # default


class TestAIMNet2WrapperEmbeddings:
    """Tests for embedding shapes and compute_embeddings."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_model):
        self.wrapper = _make_wrapper(mock_model)

    def test_embedding_shapes(self):
        shapes = self.wrapper.embedding_shapes
        assert "node_embeddings" in shapes
        assert shapes["node_embeddings"] == (256,)


class TestAIMNet2WrapperExport:
    """Tests for export_model."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_model):
        self.wrapper = _make_wrapper(mock_model)

    def test_export_state_dict(self, tmp_path):
        path = tmp_path / "aimnet2.pt"
        self.wrapper.export_model(path, as_state_dict=True)
        assert path.exists()

    def test_export_full_model_requires_real_model(self):
        """Full model export requires a real (picklable) model; mock raises."""
        # This test validates that export_model is callable; full model
        # export with a real checkpoint is tested in integration tests.
        with pytest.raises(Exception):
            self.wrapper.export_model(Path("/dev/null"), as_state_dict=False)
