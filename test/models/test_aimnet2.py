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

from contextlib import contextmanager
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
        # Include trainable weights so force losses can update model parameters.
        energy = ((coord**2).sum() + self.linear(coord).sum()).unsqueeze(0)
        return {
            "energy": energy,
            "charges": torch.ones(n_atoms, dtype=coord.dtype, device=coord.device)
            * 0.1,
            "aim": torch.randn(n_atoms, 256, dtype=coord.dtype, device=coord.device),
        }


class _MockAIMNet2Calculator:
    """Minimal mock of aimnet.calculators.AIMNet2Calculator."""

    calls: list[dict[str, Any]] = []

    def __init__(self, model: Any = None, device: str = "cpu", **kwargs):
        self.model = model if isinstance(model, nn.Module) else _MockAIMNet2Model()
        self.device = device
        self.kwargs = kwargs
        self.keys_out = ["energy", "charges"]
        self.atom_feature_keys = ["charges"]
        self.calls.append({"model": model, "device": device, **kwargs})

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


@contextmanager
def _mock_aimnet_dependency():
    """Temporarily install the mock AIMNet dependency."""
    import sys

    from nvalchemi._optional import OptionalDependency

    dep = OptionalDependency.AIMNET
    orig_available = dep._available
    orig_mod = sys.modules.get("aimnet.calculators")
    orig_aimnet = sys.modules.get("aimnet")

    # Mock the aimnet.calculators module so imports inside the wrapper work.
    mock_calculators = MagicMock()
    mock_calculators.AIMNet2Calculator = _MockAIMNet2Calculator
    sys.modules["aimnet"] = MagicMock()
    sys.modules["aimnet.calculators"] = mock_calculators
    dep._available = True
    _MockAIMNet2Calculator.calls = []
    try:
        yield
    finally:
        dep._available = orig_available
        if orig_mod is None:
            sys.modules.pop("aimnet.calculators", None)
        else:
            sys.modules["aimnet.calculators"] = orig_mod
        if orig_aimnet is None:
            sys.modules.pop("aimnet", None)
        else:
            sys.modules["aimnet"] = orig_aimnet


def _make_wrapper(model: _MockAIMNet2Model, train: bool | None = None) -> Any:
    """Construct an AIMNet2Wrapper with mock AIMNet2Calculator."""
    with _mock_aimnet_dependency():
        from nvalchemi.models.aimnet2 import AIMNet2Wrapper

        return AIMNet2Wrapper(model, train=train)


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

    def test_construction_uses_model_training_mode_by_default(self, mock_model):
        """Default calculator train flag follows the model training mode."""
        mock_model.eval()

        _make_wrapper(mock_model)

        assert _MockAIMNet2Calculator.calls[-1]["train"] is False

    def test_construction_allows_explicit_train_flag(self, mock_model):
        """Explicit train flag overrides the model training mode."""
        mock_model.eval()

        _make_wrapper(mock_model, train=True)

        assert _MockAIMNet2Calculator.calls[-1]["train"] is True

    def test_from_checkpoint_trainable_when_not_compiled(self):
        """Checkpoint loading keeps parameters trainable without compilation."""
        with _mock_aimnet_dependency():
            from nvalchemi.models.aimnet2 import AIMNet2Wrapper

            wrapper = AIMNet2Wrapper.from_checkpoint("mock", compile_model=False)

        assert wrapper.model.training
        assert _MockAIMNet2Calculator.calls[0]["train"] is True
        assert _MockAIMNet2Calculator.calls[1]["train"] is True

    def test_from_checkpoint_frozen_when_compiled(self):
        """Compiled checkpoint loading keeps inference-only calculator mode."""
        with _mock_aimnet_dependency():
            from nvalchemi.models.aimnet2 import AIMNet2Wrapper

            AIMNet2Wrapper.from_checkpoint("mock", compile_model=True)

        assert _MockAIMNet2Calculator.calls[0]["train"] is False
        assert _MockAIMNet2Calculator.calls[1]["train"] is False

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
        with pytest.raises(Exception):
            self.wrapper.export_model(Path("/dev/null"), as_state_dict=False)


# ===========================================================================
# Integration tests (require aimnet + real checkpoint)
# ===========================================================================


def _make_water_batch(device="cpu", pbc=True):
    """Build a single H2O molecule batch for integration tests."""
    data = AtomicData(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
            dtype=torch.float32,
        ),
        atomic_numbers=torch.tensor([8, 1, 1], dtype=torch.long),
        forces=torch.zeros(3, 3),
        energy=torch.zeros(1, 1),
        charge=torch.zeros(1, 1),
    )
    if pbc:
        data = AtomicData(
            positions=data.positions,
            atomic_numbers=data.atomic_numbers,
            forces=data.forces,
            energy=data.energy,
            charge=data.charge,
            cell=torch.eye(3).unsqueeze(0) * 15.0,
            pbc=torch.tensor([[True, True, True]]),
        )
    return Batch.from_data_list([data], device=device)


def _build_nl(batch, model):
    """Build a real neighbor list for integration tests."""
    from nvalchemi.neighbors import compute_neighbors

    compute_neighbors(batch, model.model_config.neighbor_config.cutoff)


class TestAIMNet2WrapperMockForward:
    """CPU forward tests using the mock AIMNet2 calculator."""

    @pytest.fixture(autouse=True)
    def setup(self, mock_model):
        self.wrapper = _make_wrapper(mock_model)

    def test_forward_forces_and_stress_use_merged_helper(self, monkeypatch):
        """Forces and stress requested together use the merged autograd helper."""
        import nvalchemi.models.aimnet2 as aimnet2_module

        real_helper = aimnet2_module.autograd_forces_and_stresses
        calls = []

        def wrapped_helper(*args, **kwargs):
            calls.append((args, kwargs))
            return real_helper(*args, **kwargs)

        monkeypatch.setattr(
            aimnet2_module, "autograd_forces_and_stresses", wrapped_helper
        )

        batch = _make_water_batch(device="cpu", pbc=True)
        self.wrapper.model_config.active_outputs = {"energy", "forces", "stress"}

        out = self.wrapper(batch)

        assert len(calls) == 1
        assert calls[0][1]["training"] is True
        assert out["forces"].shape == (3, 3)
        assert out["stress"].shape == (1, 3, 3)

    def test_adapt_input_normalizes_single_system_cell(self):
        """Single-system cells are promoted to AIMNet2 batch shape."""
        data = MagicMock()
        data.positions = torch.zeros(3, 3)
        data.atomic_numbers = torch.tensor([8, 1, 1], dtype=torch.long)
        data.batch_idx = torch.zeros(3, dtype=torch.long)
        data.charge = torch.zeros(1, 1)
        data.cell = torch.eye(3) * 15.0
        data.num_nodes = 3
        data.num_graphs = 1
        data.neighbor_matrix = None

        inp = self.wrapper.adapt_input(data)

        assert inp["cell"].shape == (1, 3, 3)

    def test_force_loss_updates_weights_in_train_mode(self, simple_batch):
        """Force-only losses keep a graph back to trainable parameters."""
        self.wrapper.train()
        self.wrapper.model_config.active_outputs = {"energy", "forces"}
        optimizer = torch.optim.SGD(self.wrapper.model.parameters(), lr=0.1)
        before = self.wrapper.model.linear.weight.detach().clone()

        out = self.wrapper(simple_batch)
        loss = out["forces"].square().sum()
        loss.backward()
        optimizer.step()

        after = self.wrapper.model.linear.weight.detach()
        assert self.wrapper.model.linear.weight.grad is not None
        assert not torch.allclose(after, before)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for AIMNet2 integration"
)
class TestAIMNet2Integration:
    """Integration tests using a real AIMNet2 checkpoint."""

    @pytest.fixture(autouse=True)
    def _require_aimnet(self):
        pytest.importorskip("aimnet")

    @pytest.fixture
    def wrapper(self):
        from nvalchemi.models.aimnet2 import AIMNet2Wrapper

        return AIMNet2Wrapper.from_checkpoint("aimnet2_wb97m_d3_3", device="cuda")

    @pytest.fixture
    def batch(self):
        return _make_water_batch(device="cuda", pbc=True)

    # -- from_checkpoint --

    def test_from_checkpoint_loads(self, wrapper):
        assert wrapper.model is not None
        assert wrapper._cutoff > 0

    def test_from_checkpoint_model_config(self, wrapper):
        cfg = wrapper.model_config
        assert "energy" in cfg.outputs
        assert "charges" in cfg.outputs
        assert cfg.neighbor_config is not None
        assert cfg.neighbor_config.cutoff == wrapper._cutoff

    # -- adapt_input --

    def test_adapt_input_builds_flat_dict(self, wrapper, batch):
        _build_nl(batch, wrapper)
        inp = wrapper.adapt_input(batch)
        assert "coord" in inp
        assert "numbers" in inp
        assert "nbmat" in inp
        assert "charge" in inp

    def test_adapt_input_nbmat_has_padding_row(self, wrapper, batch):
        _build_nl(batch, wrapper)
        inp = wrapper.adapt_input(batch)
        N = batch.num_nodes
        # nbmat has N+1 rows (padding row) after mol_flatten + pad_input
        assert inp["nbmat"].shape[0] == N + 1

    def test_adapt_input_enables_grad_for_forces(self, wrapper, batch):
        _build_nl(batch, wrapper)
        wrapper.model_config.active_outputs = {"energy", "forces"}
        wrapper.adapt_input(batch)
        assert batch.positions.requires_grad

    def test_adapt_input_no_grad_energy_only(self, wrapper, batch):
        _build_nl(batch, wrapper)
        wrapper.model_config.active_outputs = {"energy", "charges"}
        wrapper.adapt_input(batch)
        assert not batch.positions.requires_grad

    def test_adapt_input_includes_cell_for_pbc(self, wrapper, batch):
        _build_nl(batch, wrapper)
        inp = wrapper.adapt_input(batch)
        assert "cell" in inp

    def test_adapt_input_includes_shifts_for_pbc(self, wrapper, batch):
        _build_nl(batch, wrapper)
        inp = wrapper.adapt_input(batch)
        assert "shifts" in inp

    # -- adapt_output --

    def test_adapt_output_maps_energy(self, wrapper):
        raw = {"energy": torch.tensor([[-50.0]])}
        wrapper.model_config.active_outputs = {"energy"}
        out = wrapper.adapt_output(raw, None)
        assert "energy" in out
        assert out["energy"].shape == (1, 1)

    def test_adapt_output_includes_charges(self, wrapper):
        raw = {"energy": torch.tensor([[-50.0]]), "charges": torch.ones(3)}
        wrapper.model_config.active_outputs = {"energy", "charges"}
        out = wrapper.adapt_output(raw, None)
        assert "charges" in out

    def test_adapt_output_no_forces_when_not_active(self, wrapper):
        raw = {"energy": torch.tensor([[-50.0]]), "forces": torch.zeros(3, 3)}
        wrapper.model_config.active_outputs = {"energy"}
        out = wrapper.adapt_output(raw, None)
        assert "forces" not in out

    # -- forward --

    def test_forward_energy_finite(self, wrapper, batch):
        _build_nl(batch, wrapper)
        wrapper.model_config.active_outputs = {"energy"}
        out = wrapper(batch)
        assert torch.isfinite(out["energy"]).all()

    def test_forward_energy_shape(self, wrapper, batch):
        _build_nl(batch, wrapper)
        wrapper.model_config.active_outputs = {"energy"}
        out = wrapper(batch)
        assert out["energy"].shape == (1, 1)

    def test_forward_forces_finite(self, wrapper, batch):
        _build_nl(batch, wrapper)
        wrapper.model_config.active_outputs = {"energy", "forces"}
        out = wrapper(batch)
        assert torch.isfinite(out["forces"]).all()

    def test_forward_forces_shape(self, wrapper, batch):
        _build_nl(batch, wrapper)
        wrapper.model_config.active_outputs = {"energy", "forces"}
        out = wrapper(batch)
        assert out["forces"].shape == (3, 3)

    def test_forward_charges_finite(self, wrapper, batch):
        _build_nl(batch, wrapper)
        wrapper.model_config.active_outputs = {"energy", "charges"}
        out = wrapper(batch)
        assert out["charges"] is not None
        assert torch.isfinite(out["charges"]).all()

    def test_forward_charges_shape(self, wrapper, batch):
        _build_nl(batch, wrapper)
        wrapper.model_config.active_outputs = {"energy", "charges"}
        out = wrapper(batch)
        assert out["charges"].shape == (3,)

    def test_forward_stresses_finite(self, wrapper, batch):
        _build_nl(batch, wrapper)
        wrapper.model_config.active_outputs = {"energy", "forces", "stress"}
        out = wrapper(batch)
        assert "stress" in out
        assert torch.isfinite(out["stress"]).all()

    def test_forward_stresses_shape(self, wrapper, batch):
        _build_nl(batch, wrapper)
        wrapper.model_config.active_outputs = {"energy", "forces", "stress"}
        out = wrapper(batch)
        assert out["stress"].shape == (1, 3, 3)

    def test_forward_water_energy_reasonable(self, wrapper, batch):
        """H2O energy should be around -2075 eV for this model."""
        _build_nl(batch, wrapper)
        wrapper.model_config.active_outputs = {"energy"}
        out = wrapper(batch)
        e = out["energy"].item()
        assert -2200 < e < -1900, f"H2O energy {e:.1f} eV outside expected range"

    def test_forward_water_charges_physical(self, wrapper, batch):
        """O should be negative, H should be positive for water."""
        _build_nl(batch, wrapper)
        wrapper.model_config.active_outputs = {"energy", "charges"}
        out = wrapper(batch)
        charges = out["charges"]
        # O is index 0, H are indices 1 and 2
        assert charges[0] < 0, f"O charge {charges[0]:.4f} should be negative"
        assert charges[1] > 0, f"H charge {charges[1]:.4f} should be positive"
        assert charges[2] > 0, f"H charge {charges[2]:.4f} should be positive"
        # Charges should sum to ~0 (neutral molecule)
        assert abs(charges.sum().item()) < 0.01

    def test_forward_two_consecutive_calls(self, wrapper, batch):
        """Two forward calls should both succeed (no stale graph)."""
        _build_nl(batch, wrapper)
        wrapper.model_config.active_outputs = {"energy", "forces"}
        wrapper(batch)
        _build_nl(batch, wrapper)
        out2 = wrapper(batch)
        assert torch.isfinite(out2["energy"]).all()
        assert torch.isfinite(out2["forces"]).all()

    def test_force_loss_updates_real_checkpoint_weights(self, wrapper, batch):
        """A force loss can update parameters from a real AIMNet2 checkpoint."""
        wrapper.train()
        wrapper.model_config.active_outputs = {"energy", "forces"}
        params = [
            param
            for param in wrapper.model.parameters()
            if param.requires_grad and param.is_floating_point()
        ]
        assert params
        before = [param.detach().clone() for param in params]
        optimizer = torch.optim.SGD(params, lr=1e-4)

        _build_nl(batch, wrapper)
        optimizer.zero_grad()
        out = wrapper(batch)
        loss = out["forces"].square().mean()
        loss.backward()
        grads = [param.grad for param in params]
        assert any(
            grad is not None and torch.count_nonzero(grad).item() for grad in grads
        )

        optimizer.step()

        assert any(
            grad is not None and not torch.equal(param.detach(), old_param)
            for param, old_param, grad in zip(params, before, grads, strict=True)
        )

    # -- compute_embeddings --

    def test_compute_embeddings(self, wrapper, batch):
        _build_nl(batch, wrapper)
        result = wrapper.compute_embeddings(batch)
        assert hasattr(result, "node_embeddings")

    # -- embedding_shapes --

    def test_embedding_shapes_from_real_model(self, wrapper):
        shapes = wrapper.embedding_shapes
        assert "node_embeddings" in shapes
        dim = shapes["node_embeddings"][0]
        assert dim > 0

    # -- export --

    def test_export_state_dict_real(self, wrapper, tmp_path):
        path = tmp_path / "aimnet2_state.pt"
        wrapper.export_model(path, as_state_dict=True)
        assert path.exists()
        state = torch.load(path, weights_only=True)
        assert isinstance(state, dict)
