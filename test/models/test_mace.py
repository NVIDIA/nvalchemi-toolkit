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
"""Tests for MACEWrapper.

All tests in this module require ``mace-torch`` and are automatically skipped
when it is not installed.  Install with::

    pip install 'nvalchemi-toolkit[mace]'
"""

from __future__ import annotations

import pytest
import torch

# Skip the entire module when mace-torch is not installed.
pytest.importorskip("mace", reason="mace-torch not installed; skipping MACE tests")

from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.models.base import ModelConfig, NeighborListFormat  # noqa: E402
from nvalchemi.models.mace import MACEWrapper  # noqa: E402

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_ATOMIC_NUMBERS = [1, 6, 8]  # H, C, O
_CUTOFF = 5.0
_HIDDEN_DIM = 32


# ---------------------------------------------------------------------------
# Mock MACE model
# ---------------------------------------------------------------------------


# Module-level classes so torch.save (pickle) can locate them by name.
class _MockIrrepsOut:
    dim = _HIDDEN_DIM


class _MockLinear:
    irreps_out = _MockIrrepsOut()


class _MockProduct:
    linear = _MockLinear()


class MockMACEModel(torch.nn.Module):
    """Minimal MACE-like model for unit tests.

    Replicates the attribute structure that ``MACEWrapper.__init__`` and
    ``embedding_shapes`` probe, and implements a differentiable
    ``forward`` so conservative-force tests work without a real checkpoint.

    Energy is defined as the sum of per-atom position L2 norms, which gives
    an analytic gradient: ``force_i = -pos_i / |pos_i|``.
    """

    def __init__(
        self,
        numbers: list[int] = _ATOMIC_NUMBERS,
        r_max: float = _CUTOFF,
        hidden_dim: int = _HIDDEN_DIM,
    ) -> None:
        super().__init__()
        self.atomic_numbers = torch.tensor(numbers, dtype=torch.long)
        self.r_max = torch.tensor(r_max)

        # Replicate the attribute path MACEWrapper.embedding_shapes probes:
        #   model.products[0].linear.irreps_out.dim
        self.products = [_MockProduct()]

        # Real parameter so _model_dtype works (next(model.parameters()).dtype).
        self._param = torch.nn.Linear(1, hidden_dim, bias=False)
        self._hidden_dim = hidden_dim

    def forward(
        self,
        data_dict: dict,
        *,
        compute_force: bool = True,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        training: bool = False,
    ) -> dict:
        positions = data_dict["positions"]  # [N, 3]
        batch = data_dict["batch"].long()  # [N]
        N = positions.shape[0]
        B = int(batch.max().item()) + 1 if N > 0 else 1

        # Energy = sum of per-atom position norms, grouped by graph.
        # Avoids zero-norm issues by clamping from below.
        norms = positions.pow(2).sum(dim=-1).clamp(min=1e-8).sqrt()  # [N]
        energy = torch.zeros(B, dtype=positions.dtype, device=positions.device)
        energy.scatter_add_(0, batch, norms)

        node_feats = torch.zeros(
            N, self._hidden_dim, dtype=positions.dtype, device=positions.device
        )

        result: dict = {"energy": energy, "node_feats": node_feats}

        if compute_force:
            (grad,) = torch.autograd.grad(
                energy.sum(),
                positions,
                create_graph=training,
                retain_graph=True,
                allow_unused=True,
            )
            # MACE convention: forces = -dE/dr (negative gradient).
            result["forces"] = (
                -grad if grad is not None else torch.zeros_like(positions)
            )

        if compute_stress:
            result["stress"] = torch.zeros(
                B, 3, 3, dtype=positions.dtype, device=positions.device
            )

        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_water(device: str = "cpu") -> AtomicData:
    """Single H2O molecule with a pre-computed full edge list (no PBC)."""
    # O at origin, H1 along x, H2 along y — all pairs within 5 Å cutoff.
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    numbers = torch.tensor([8, 1, 1], dtype=torch.long, device=device)
    neighbor_list = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
        dtype=torch.long,
        device=device,
    )
    return AtomicData(
        positions=positions, atomic_numbers=numbers, neighbor_list=neighbor_list
    )


def _make_single_atom(device: str = "cpu") -> AtomicData:
    """Single H atom at (0.5, 0, 0) with no edges — used for analytic force check."""
    positions = torch.tensor([[0.5, 0.0, 0.0]], dtype=torch.float32, device=device)
    numbers = torch.tensor([1], dtype=torch.long, device=device)
    neighbor_list = torch.zeros(0, 2, dtype=torch.long, device=device)
    return AtomicData(
        positions=positions, atomic_numbers=numbers, neighbor_list=neighbor_list
    )


def _make_pbc_water(device: str = "cpu") -> AtomicData:
    """H2O in a periodic cubic box with integer unit_shifts on edges."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]],
        dtype=torch.float32,
        device=device,
    )
    numbers = torch.tensor([8, 1, 1], dtype=torch.long, device=device)
    neighbor_list = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
        dtype=torch.long,
        device=device,
    )
    # Cubic 10 Å cell; edges are all within the same image, so unit_shifts are zero.
    # AtomicData expects cell as [B, 3, 3] and pbc as [B, 3].
    cell = (torch.eye(3, dtype=torch.float32, device=device) * 10.0).unsqueeze(
        0
    )  # [1, 3, 3]
    unit_shifts = torch.zeros(6, 3, dtype=torch.float32, device=device)
    pbc = torch.tensor([[True, True, True]], device=device)  # [1, 3]
    return AtomicData(
        positions=positions,
        atomic_numbers=numbers,
        neighbor_list=neighbor_list,
        cell=cell,
        unit_shifts=unit_shifts,
        pbc=pbc,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model() -> MockMACEModel:
    return MockMACEModel()


@pytest.fixture
def wrapper(mock_model) -> MACEWrapper:
    return MACEWrapper(mock_model)


@pytest.fixture
def single_batch() -> Batch:
    return Batch.from_data_list([_make_water()])


@pytest.fixture
def multi_batch() -> Batch:
    """Two H2O molecules as a batched system (B=2, N=6)."""
    return Batch.from_data_list([_make_water(), _make_water()])


@pytest.fixture
def pbc_batch() -> Batch:
    return Batch.from_data_list([_make_pbc_water()])


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


class TestInstantiation:
    def test_wraps_model(self, mock_model):
        w = MACEWrapper(mock_model)
        assert w.model is mock_model

    def test_training_state_mirrored_eval(self, mock_model):
        mock_model.eval()
        w = MACEWrapper(mock_model)
        assert not w.training

    def test_training_state_mirrored_train(self, mock_model):
        mock_model.train()
        w = MACEWrapper(mock_model)
        assert w.training

    def test_default_model_config(self, wrapper):
        assert wrapper.model_config.compute_forces is True
        assert wrapper.model_config.compute_stresses is False

    def test_node_emb_buffer_shape(self, wrapper):
        # [max_z + 1, num_elements] = [9, 3] for atomic_numbers=[1, 6, 8]
        assert wrapper._node_emb.shape == (9, 3)

    def test_node_emb_not_in_state_dict(self, wrapper):
        assert "_node_emb" not in wrapper.state_dict()

    def test_import_error_without_mace(self, mock_model):
        from nvalchemi import OptionalDependency

        dep = OptionalDependency.MACE
        original_available, original_error = dep._available, dep._import_error
        try:
            dep._available = False
            dep._import_error = ImportError("No module named 'mace'")
            with pytest.raises(ImportError):
                MACEWrapper(mock_model)
        finally:
            dep._available = original_available
            dep._import_error = original_error


# ---------------------------------------------------------------------------
# ModelCard
# ---------------------------------------------------------------------------


class TestModelCard:
    def test_forces_via_autograd(self, wrapper):
        assert wrapper.model_card.forces_via_autograd is True

    def test_supports_energies_forces_stresses(self, wrapper):
        card = wrapper.model_card
        assert card.supports_energies
        assert card.supports_forces
        assert card.supports_stresses

    def test_supports_pbc(self, wrapper):
        assert wrapper.model_card.supports_pbc is True

    def test_supports_non_batch(self, wrapper):
        assert wrapper.model_card.supports_non_batch is True

    def test_supports_embeddings(self, wrapper):
        card = wrapper.model_card
        assert card.supports_node_embeddings
        assert card.supports_graph_embeddings

    def test_neighbor_config_coo(self, wrapper):
        nc = wrapper.model_card.neighbor_config
        assert nc is not None
        assert nc.format == NeighborListFormat.COO
        assert nc.cutoff == pytest.approx(_CUTOFF)

    def test_needs_pbc_false(self, wrapper):
        assert wrapper.model_card.needs_pbc is False


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_cutoff_from_tensor_r_max(self, wrapper):
        assert wrapper.cutoff == pytest.approx(_CUTOFF)
        assert isinstance(wrapper.cutoff, float)

    def test_cutoff_from_scalar_r_max(self, mock_model):
        mock_model.r_max = 6.0
        w = MACEWrapper(mock_model)
        assert w.cutoff == pytest.approx(6.0)

    def test_embedding_shapes(self, wrapper):
        shapes = wrapper.embedding_shapes
        assert shapes["node_embeddings"] == (_HIDDEN_DIM,)
        assert shapes["graph_embeddings"] == (_HIDDEN_DIM,)

    def test_model_dtype(self, wrapper):
        assert wrapper._model_dtype == torch.float32


# ---------------------------------------------------------------------------
# Node attribute encoding
# ---------------------------------------------------------------------------


class TestNodeAttrs:
    def test_one_hot_correctness(self, wrapper, single_batch):
        # Atomic numbers for H2O are [8, 1, 1].
        # z_table = [1, 6, 8] → indices [2, 0, 0]
        node_attrs = wrapper._node_attrs(single_batch)
        assert node_attrs.shape == (3, 3)  # 3 atoms, 3 element types
        # O (atomic_number=8) → index 2 in z_table → one-hot [0, 0, 1]
        assert node_attrs[0].tolist() == pytest.approx([0.0, 0.0, 1.0])
        # H (atomic_number=1) → index 0 in z_table → one-hot [1, 0, 0]
        assert node_attrs[1].tolist() == pytest.approx([1.0, 0.0, 0.0])
        assert node_attrs[2].tolist() == pytest.approx([1.0, 0.0, 0.0])

    def test_dtype_matches_model(self, wrapper, single_batch):
        node_attrs = wrapper._node_attrs(single_batch)
        assert node_attrs.dtype == wrapper._model_dtype

    def test_device_matches_batch(self, wrapper, single_batch):
        node_attrs = wrapper._node_attrs(single_batch)
        assert node_attrs.device == single_batch.positions.device


# ---------------------------------------------------------------------------
# adapt_input
# ---------------------------------------------------------------------------


class TestAdaptInput:
    def test_required_keys_present(self, wrapper, single_batch):
        inp = wrapper.adapt_input(single_batch)
        for key in (
            "positions",
            "node_attrs",
            "batch",
            "ptr",
            "edge_index",
            "unit_shifts",
            "shifts",
            "cell",
        ):
            assert key in inp, f"Missing key: {key}"

    def test_positions_dtype(self, wrapper, single_batch):
        inp = wrapper.adapt_input(single_batch)
        assert inp["positions"].dtype == wrapper._model_dtype

    def test_topology_tensors_are_long(self, wrapper, single_batch):
        inp = wrapper.adapt_input(single_batch)
        assert inp["edge_index"].dtype == torch.long
        assert inp["batch"].dtype == torch.long
        assert inp["ptr"].dtype == torch.long
        # adapt_input transposes neighbor_list from nvalchemi [E, 2] to MACE [2, E]
        E = single_batch.neighbor_list.shape[0]
        assert inp["edge_index"].shape == (2, E)

    def test_positions_requires_grad_when_forces_requested(self, wrapper, single_batch):
        wrapper.model_config.compute_forces = True
        inp = wrapper.adapt_input(single_batch)
        assert inp["positions"].requires_grad

    def test_positions_no_requires_grad_energy_only(self, wrapper, single_batch):
        wrapper.model_config = ModelConfig(compute_forces=False, compute_stresses=False)
        inp = wrapper.adapt_input(single_batch)
        assert not inp["positions"].requires_grad

    def test_atomic_data_promoted_to_batch(self, wrapper):
        data = _make_water()
        inp = wrapper.adapt_input(data)
        # batch tensor should exist and be all zeros (single system)
        assert inp["batch"].shape[0] == 3  # 3 atoms
        assert inp["batch"].max().item() == 0

    def test_no_pbc_zero_unit_shifts(self, wrapper, single_batch):
        # single_batch has no unit_shifts → adapt_input fills zeros
        inp = wrapper.adapt_input(single_batch)
        # nvalchemi neighbor_list is [E, 2]; adapt_input transposes to [2, E].
        E = single_batch.neighbor_list.shape[0]
        assert inp["unit_shifts"].shape == (E, 3)
        assert inp["unit_shifts"].abs().max().item() == pytest.approx(0.0)

    def test_no_pbc_identity_cell(self, wrapper, single_batch):
        # single_batch has no cell → adapt_input fills identity [B, 3, 3]
        inp = wrapper.adapt_input(single_batch)
        B = single_batch.num_graphs
        assert inp["cell"].shape == (B, 3, 3)
        expected = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
        assert torch.allclose(inp["cell"], expected)

    def test_pbc_unit_shifts_passed_through(self, wrapper, pbc_batch):
        inp = wrapper.adapt_input(pbc_batch)
        # unit_shifts were all zeros in _make_pbc_water; should be preserved
        assert inp["unit_shifts"].shape[1] == 3
        assert inp["unit_shifts"].abs().max().item() == pytest.approx(0.0)

    def test_pbc_cell_passed_through(self, wrapper, pbc_batch):
        inp = wrapper.adapt_input(pbc_batch)
        B = pbc_batch.num_graphs
        assert inp["cell"].shape == (B, 3, 3)
        # 10 Å cubic cell
        assert torch.allclose(inp["cell"][0], torch.eye(3) * 10.0, atol=1e-5)

    def test_multi_batch_batch_indices(self, wrapper, multi_batch):
        # B=2, N=6: first 3 atoms → graph 0, last 3 → graph 1
        inp = wrapper.adapt_input(multi_batch)
        assert inp["batch"].tolist() == [0, 0, 0, 1, 1, 1]


# ---------------------------------------------------------------------------
# adapt_output
# ---------------------------------------------------------------------------


class TestAdaptOutput:
    def _raw(self, B: int = 1, N: int = 3) -> dict:
        return {
            "energy": torch.randn(B),
            "forces": torch.randn(N, 3),
            "node_feats": torch.randn(N, _HIDDEN_DIM),
        }

    def test_energy_key_in_output(self, wrapper, single_batch):
        raw = self._raw()
        out = wrapper.adapt_output(raw, single_batch)
        assert "energy" in out

    def test_energies_shape(self, wrapper, single_batch):
        raw = self._raw(B=1, N=3)
        out = wrapper.adapt_output(raw, single_batch)
        assert out["energy"].shape == (1, 1)

    def test_energies_already_2d(self, wrapper, single_batch):
        raw = self._raw()
        raw["energy"] = raw["energy"].unsqueeze(-1)  # already [B, 1]
        out = wrapper.adapt_output(raw, single_batch)
        assert out["energy"].shape == (1, 1)

    def test_forces_passed_through(self, wrapper, single_batch):
        raw = self._raw()
        wrapper.model_config.compute_forces = True
        out = wrapper.adapt_output(raw, single_batch)
        assert "forces" in out
        assert out["forces"].shape == (3, 3)

    def test_stress_key_in_output(self, wrapper, single_batch):
        raw = self._raw()
        raw["stress"] = torch.randn(1, 3, 3)
        wrapper.model_config.compute_stresses = True
        out = wrapper.adapt_output(raw, single_batch)
        assert "stress" in out

    def test_missing_optional_outputs_absent(self, wrapper, single_batch):
        # No stress or hessian in raw output → not in result
        raw = {"energy": torch.randn(1), "forces": torch.randn(3, 3)}
        out = wrapper.adapt_output(raw, single_batch)
        assert "stress" not in out or out.get("stress") is None
        assert "hessians" not in out or out.get("hessians") is None


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------


class TestForward:
    def test_energies_shape_single(self, wrapper, single_batch):
        out = wrapper.forward(single_batch)
        assert out["energy"].shape == (1, 1)

    def test_energies_shape_multi(self, wrapper, multi_batch):
        out = wrapper.forward(multi_batch)
        assert out["energy"].shape == (2, 1)

    def test_energies_dtype(self, wrapper, single_batch):
        out = wrapper.forward(single_batch)
        assert out["energy"].dtype == wrapper._model_dtype

    def test_forces_shape(self, wrapper, single_batch):
        out = wrapper.forward(single_batch)
        assert out["forces"].shape == (3, 3)  # 3 atoms, 3 coords

    def test_forces_shape_multi(self, wrapper, multi_batch):
        out = wrapper.forward(multi_batch)
        assert out["forces"].shape == (6, 3)  # 6 atoms total

    def test_forces_conservative(self, wrapper):
        """Verify forces = -dE/dpos using the mock model's analytic energy.

        For energy = |pos|, the analytic force is -pos / |pos|.
        For a single H atom at (0.5, 0, 0): force should be (-1, 0, 0).
        """
        data = _make_single_atom()
        batch = Batch.from_data_list([data])
        out = wrapper.forward(batch)
        forces = out["forces"]
        # Analytic: pos = [0.5, 0, 0], |pos| = 0.5, force = -[0.5,0,0]/0.5 = [-1,0,0]
        assert forces.shape == (1, 3)
        assert torch.allclose(forces[0], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-5)

    def test_no_forces_when_disabled(self, wrapper, single_batch):
        wrapper.model_config = ModelConfig(compute_forces=False, compute_stresses=False)
        out = wrapper.forward(single_batch)
        # forces key may be absent or None
        assert out.get("forces") is None

    def test_stresses_shape(self, wrapper, single_batch):
        wrapper.model_config = ModelConfig(compute_stresses=True)
        out = wrapper.forward(single_batch)
        assert out["stress"].shape == (1, 3, 3)

    def test_atomic_data_input(self, wrapper):
        data = _make_water()
        out = wrapper.forward(data)
        assert out["energy"].shape == (1, 1)

    def test_pbc_batch_runs(self, wrapper, pbc_batch):
        out = wrapper.forward(pbc_batch)
        assert out["energy"].shape == (1, 1)


# ---------------------------------------------------------------------------
# compute_embeddings
# ---------------------------------------------------------------------------


class TestComputeEmbeddings:
    def test_node_embeddings_shape(self, wrapper, single_batch):
        result = wrapper.compute_embeddings(single_batch)
        assert result.node_embeddings.shape == (3, _HIDDEN_DIM)

    def test_graph_embeddings_shape(self, wrapper, single_batch):
        result = wrapper.compute_embeddings(single_batch)
        assert result.graph_embeddings.shape == (1, _HIDDEN_DIM)

    def test_graph_embeddings_shape_multi(self, wrapper, multi_batch):
        result = wrapper.compute_embeddings(multi_batch)
        assert result.graph_embeddings.shape == (2, _HIDDEN_DIM)

    def test_graph_embeddings_is_sum_of_node_embeddings(self, wrapper, single_batch):
        # graph_embeddings = scatter_add of node_embeddings — for B=1 they should be equal.
        result = wrapper.compute_embeddings(single_batch)
        expected_graph = result.node_embeddings.sum(dim=0)
        assert torch.allclose(result.graph_embeddings[0], expected_graph)

    def test_does_not_mutate_model_config(self, wrapper, single_batch):
        wrapper.model_config = ModelConfig(compute_forces=True, compute_stresses=True)
        wrapper.compute_embeddings(single_batch)
        # model_config must be unchanged after the call
        assert wrapper.model_config.compute_forces is True
        assert wrapper.model_config.compute_stresses is True

    def test_atomic_data_input(self, wrapper):
        data = _make_water()
        result = wrapper.compute_embeddings(data)
        assert result.node_embeddings.shape == (3, _HIDDEN_DIM)

    def test_no_grad_on_positions_after_embeddings(self, wrapper, single_batch):
        # compute_embeddings passes compute_force=False; positions should not
        # require grad inside the call (they have no external requires_grad set).
        # The batch positions themselves should not gain requires_grad.
        wrapper.compute_embeddings(single_batch)
        assert not single_batch.positions.requires_grad


# ---------------------------------------------------------------------------
# export_model
# ---------------------------------------------------------------------------


class TestExportModel:
    def test_export_full_model(self, wrapper, tmp_path):
        path = tmp_path / "mace.pt"
        wrapper.export_model(path)
        assert path.exists()
        loaded = torch.load(path, weights_only=False)
        assert isinstance(loaded, torch.nn.Module)

    def test_export_state_dict(self, wrapper, tmp_path):
        path = tmp_path / "mace_sd.pt"
        wrapper.export_model(path, as_state_dict=True)
        assert path.exists()
        sd = torch.load(path, weights_only=True)
        assert isinstance(sd, dict)

    def test_exported_model_matches_wrapper(self, wrapper, tmp_path):
        path = tmp_path / "mace.pt"
        wrapper.export_model(path)
        loaded = torch.load(path, weights_only=False)
        # State dicts of the exported model should match the wrapped model.
        for key in loaded.state_dict():
            assert torch.allclose(
                loaded.state_dict()[key], wrapper.model.state_dict()[key]
            )


# ---------------------------------------------------------------------------
# from_checkpoint error path (no network required)
# ---------------------------------------------------------------------------


class TestFromCheckpointErrors:
    def test_raises_import_error_when_mace_unavailable(self):
        from nvalchemi import OptionalDependency

        dep = OptionalDependency.MACE
        original_available, original_error = dep._available, dep._import_error
        try:
            dep._available = False
            dep._import_error = ImportError("No module named 'mace'")
            with pytest.raises(ImportError):
                MACEWrapper.from_checkpoint("medium")
        finally:
            dep._available = original_available
            dep._import_error = original_error

    def test_raises_import_error_for_cueq_when_unavailable(
        self, monkeypatch, mock_model
    ):
        """cuEq ImportError should reference the [mace] extra."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "cuequivariance":
                raise ImportError("no module named cuequivariance")
            return real_import(name, *args, **kwargs)

        # Patch the checkpoint loader to return mock_model without network access.
        monkeypatch.setattr(
            "mace.calculators.foundations_models.download_mace_mp_checkpoint",
            lambda _: "unused",
        )
        monkeypatch.setattr("torch.load", lambda *a, **kw: mock_model)
        monkeypatch.setattr(builtins, "__import__", mock_import)

        with pytest.raises(ImportError, match="nvalchemi-toolkit\\[mace\\]"):
            MACEWrapper.from_checkpoint("medium", enable_cueq=True)


# ---------------------------------------------------------------------------
# Integration tests — real MACE checkpoint (requires network, marked slow)
# ---------------------------------------------------------------------------

# H2O molecule: O at origin, two H at ~0.96 Å.  All pairs within any
# reasonable MACE cutoff (≥ 5 Å), so a full symmetric edge list is valid.
_WATER_POSITIONS = torch.tensor(
    [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.0, 0.96, 0.0]], dtype=torch.float64
)
_WATER_ATOMIC_NUMBERS = torch.tensor([8, 1, 1], dtype=torch.long)
_WATER_EDGE_INDEX = torch.tensor(
    [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]], dtype=torch.long
)


def _water_batch(dtype: torch.dtype = torch.float64, device: str = "cpu") -> Batch:
    data = AtomicData(
        positions=_WATER_POSITIONS.to(dtype=dtype, device=device),
        atomic_numbers=_WATER_ATOMIC_NUMBERS.to(device=device),
        neighbor_list=_WATER_EDGE_INDEX.to(device=device),
    )
    return Batch.from_data_list([data])


@pytest.fixture(scope="session")
def real_wrapper_cpu():
    """Load the MACE-MP small checkpoint once per session (requires network).

    The fixture calls ``pytest.skip`` if the download fails (e.g. no internet),
    so dependent tests are cleanly skipped rather than failing.

    We use ``small-0b`` — the smallest foundation model — to keep download
    time and memory usage low.
    """
    try:
        return MACEWrapper.from_checkpoint(
            "small-0b", device=torch.device("cpu"), dtype=torch.float32
        )
    except Exception as e:
        pytest.skip(f"Could not load MACE checkpoint (network unavailable?): {e}")


@pytest.mark.slow
class TestRealCheckpoint:
    """Integration tests against a real MACE-MP checkpoint.

    All tests in this class are marked ``slow`` and skipped when
    ``real_wrapper_cpu`` cannot download the checkpoint.
    """

    def test_is_mace_wrapper(self, real_wrapper_cpu):
        assert isinstance(real_wrapper_cpu, MACEWrapper)

    def test_underlying_model_is_mace(self, real_wrapper_cpu):
        from mace.modules import ScaleShiftMACE

        assert isinstance(real_wrapper_cpu.model, ScaleShiftMACE)

    def test_model_card_matches_wrapper(self, real_wrapper_cpu):
        card = real_wrapper_cpu.model_card
        assert card.forces_via_autograd
        assert card.supports_energies
        assert card.supports_forces
        assert card.neighbor_config is not None
        assert card.neighbor_config.format == NeighborListFormat.COO

    def test_cutoff_positive(self, real_wrapper_cpu):
        assert real_wrapper_cpu.cutoff > 0.0

    def test_inference_energies_shape(self, real_wrapper_cpu):
        batch = _water_batch(dtype=torch.float32)
        out = real_wrapper_cpu.forward(batch)
        assert out["energy"].shape == (1, 1)

    def test_inference_forces_shape(self, real_wrapper_cpu):
        batch = _water_batch(dtype=torch.float32)
        out = real_wrapper_cpu.forward(batch)
        assert out["forces"].shape == (3, 3)

    def test_inference_float64(self, real_wrapper_cpu):
        """Wrapper handles float64 input even when model weights are float32."""
        batch = _water_batch(dtype=torch.float64)
        # float64 input is cast to model dtype (float32) inside adapt_input
        out = real_wrapper_cpu.forward(batch)
        assert out["energy"].dtype == torch.float32

    def test_dtype_float32_conversion(self):
        """Loading with dtype=float32 produces float32 weights."""
        try:
            w = MACEWrapper.from_checkpoint(
                "small-0b", device=torch.device("cpu"), dtype=torch.float32
            )
        except Exception as e:
            pytest.skip(f"Checkpoint unavailable: {e}")
        assert w._model_dtype == torch.float32

    def test_dtype_conversion_uniform(self):
        """All weights including atomic energy are converted to the target dtype."""
        try:
            w = MACEWrapper.from_checkpoint(
                "small-0b", device=torch.device("cpu"), dtype=torch.float32
            )
        except Exception as e:
            pytest.skip(f"Checkpoint unavailable: {e}")
        ae = w.model.atomic_energies_fn.atomic_energies
        # atomic_energies must match the model dtype so matmul with node_attrs works
        assert ae.dtype == torch.float32

    def test_export_and_reload(self, real_wrapper_cpu, tmp_path):
        path = tmp_path / "small_ob.pt"
        real_wrapper_cpu.export_model(path)
        reloaded = torch.load(path, weights_only=False)
        from mace.modules import ScaleShiftMACE

        assert isinstance(reloaded, ScaleShiftMACE)

    def test_compute_embeddings_run(self, real_wrapper_cpu):
        batch = _water_batch(dtype=torch.float32)
        result = real_wrapper_cpu.compute_embeddings(batch)
        assert result.node_embeddings.shape[0] == 3
        assert result.graph_embeddings.shape == (1, result.node_embeddings.shape[1])

    def test_compile_inference(self):
        """torch.compile produces a working inference-only model.

        Requires MACE >= the patch in mace-org/mace@6a32999 that fixes
        e3nn.Irreps.__reduce__ incompatibility with torch._dynamo guards.
        The test is skipped automatically when the known NotImplementedError
        from SEQUENCE_LENGTH guard creation is detected.
        """
        try:
            w = MACEWrapper.from_checkpoint(
                "small-0b",
                device=torch.device("cpu"),
                dtype=torch.float32,
                compile_model=True,
            )
        except Exception as e:
            pytest.skip(f"Checkpoint unavailable or compile failed: {e}")

        batch = _water_batch(dtype=torch.float32)
        # compiled model is inference-only — disable force grad to match eval state
        w.model_config = ModelConfig(compute_forces=False, compute_stresses=False)
        try:
            out = w.forward(batch)
        except Exception as e:
            if "NotImplementedError" in str(e) or "SEQUENCE_LENGTH" in str(e):
                pytest.skip(
                    "torch.compile + MACE failed (e3nn Irreps guard issue); "
                    "needs MACE patch from mace-org/mace@6a32999"
                )
            raise e
        assert out["energy"].shape == (1, 1)

    def test_cueq_conversion(self):
        """cuEquivariance conversion produces a valid model (GPU + package required)."""
        pytest.importorskip(
            "cuequivariance", reason="cuequivariance not installed; skipping cuEq test"
        )
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for cuEquivariance conversion test")
        device = torch.device("cuda")
        try:
            w = MACEWrapper.from_checkpoint(
                "small-0b",
                device=device,
                dtype=torch.float32,
                enable_cueq=True,
            )
        except Exception as e:
            pytest.skip(f"Checkpoint unavailable or cuEq failed: {e}")

        batch = _water_batch(dtype=torch.float32, device="cuda")
        out = w.forward(batch)
        assert out["energy"].shape == (1, 1)
        assert out["forces"].shape == (3, 3)

    def test_energy_and_forces_match_ase_calculator(self, real_wrapper_cpu, tmp_path):
        """MACEWrapper E+F must agree with the MACE ASE MACECalculator.

        The ASE MACECalculator is taken as ground truth.  Both operate on the
        same H2O geometry (non-PBC) so there are no unit-shift complications.
        Tolerance is 1e-4 eV for energy and 1e-4 eV/Å for forces.
        """
        try:
            from ase import Atoms
            from mace.calculators import MACECalculator
        except ImportError:
            pytest.skip("ase or mace.calculators not available")

        # Export the underlying MACE model so MACECalculator can load it.
        ckpt_path = tmp_path / "small_0b_export.pt"
        real_wrapper_cpu.export_model(ckpt_path)

        # ASE reference: single H2O, no PBC.
        atoms = Atoms(
            atomic_numbers=[8, 1, 1],
            positions=_WATER_POSITIONS.numpy(),
        )
        ase_calc = MACECalculator(
            model_paths=[str(ckpt_path)],
            device="cpu",
            default_dtype="float32",
        )
        atoms.calc = ase_calc
        ase_energy = float(atoms.get_potential_energy())  # eV
        ase_forces = torch.tensor(
            atoms.get_forces(), dtype=torch.float32
        )  # (3, 3) eV/Å

        # nvalchemi path: AtomicData → Batch → NeighborListHook → MACEWrapper.
        from nvalchemi.dynamics.hooks import NeighborListHook

        data = AtomicData.from_atoms(atoms)
        batch = Batch.from_data_list([data])

        from nvalchemi.dynamics.base import DynamicsStage
        from nvalchemi.hooks import HookContext

        nl_hook = NeighborListHook(
            real_wrapper_cpu.model_card.neighbor_config,
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
        ctx = HookContext(batch=batch, step_count=0)
        nl_hook(ctx, DynamicsStage.BEFORE_COMPUTE)

        real_wrapper_cpu.model_config.compute_forces = True
        out = real_wrapper_cpu.forward(batch)

        nv_energy = out["energy"].item()  # eV
        nv_forces = out["forces"].detach()  # (3, 3) eV/Å

        assert abs(nv_energy - ase_energy) < 1e-4, (
            f"Energy mismatch: MACEWrapper={nv_energy:.6f} eV, "
            f"ASE={ase_energy:.6f} eV, diff={abs(nv_energy - ase_energy):.2e} eV"
        )
        assert torch.allclose(nv_forces, ase_forces, atol=1e-4), (
            f"Force mismatch:\n"
            f"  MACEWrapper:        {nv_forces.tolist()}\n"
            f"  ASE MACECalculator: {ase_forces.tolist()}"
        )

    def test_cueq_then_compile(self):
        """cuEq + torch.compile pipeline works end-to-end (GPU required).

        Requires MACE >= the patch in mace-org/mace@6a32999 that fixes
        e3nn.Irreps.__reduce__ incompatibility with torch._dynamo guards.
        """
        pytest.importorskip("cuequivariance")
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        device = torch.device("cuda")
        try:
            w = MACEWrapper.from_checkpoint(
                "small-0b",
                device=device,
                dtype=torch.float32,
                enable_cueq=True,
                compile_model=True,
            )
        except Exception as e:
            pytest.skip(f"Could not build cueq+compiled model: {e}")

        batch = _water_batch(dtype=torch.float32, device="cuda")
        w.model_config = ModelConfig(compute_forces=False, compute_stresses=False)
        try:
            out = w.forward(batch)
        except Exception as e:
            if "NotImplementedError" in str(e) or "SEQUENCE_LENGTH" in str(e):
                pytest.skip(
                    "torch.compile + MACE failed (e3nn Irreps guard issue); "
                    "needs MACE patch from mace-org/mace@6a32999"
                )
            raise
        assert out["energy"].shape == (1, 1)


# ---------------------------------------------------------------------------
# dtype parametrization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_forward_dtype_consistency(dtype):
    """Model and input dtype are always in sync; outputs match."""
    model = MockMACEModel()
    model._param = torch.nn.Linear(1, _HIDDEN_DIM, bias=False).to(dtype)
    # Force r_max and numbers to stay as-is (they don't affect dtype).
    wrapper = MACEWrapper(model)

    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=dtype
    )
    numbers = torch.tensor([8, 1, 1], dtype=torch.long)
    neighbor_list = torch.tensor(
        [[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]], dtype=torch.long
    )
    data = AtomicData(
        positions=positions, atomic_numbers=numbers, neighbor_list=neighbor_list
    )
    batch = Batch.from_data_list([data])

    out = wrapper.forward(batch)
    assert out["energy"].dtype == dtype
    assert out["forces"].dtype == dtype
