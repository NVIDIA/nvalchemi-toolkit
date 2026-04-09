# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import sys
import types
from hashlib import md5
from pathlib import Path
from typing import cast

import pytest
import torch

import nvalchemi.models.dsf as dsf_module
import nvalchemi.models.ewald as ewald_module
import nvalchemi.models.neighbors as neighbors_module
import nvalchemi.models.pme as pme_module
from nvalchemi.models.aimnet2 import AIMNet2Wrapper as AIMNet2Model
from nvalchemi.models.demo import DemoModelWrapper as DemoModel
from nvalchemi.models.dftd3 import (
    DFTD3Config,
    DFTD3ParametersProcessor,
    download_dftd3_parameters,
)
from nvalchemi.models.dftd3 import (
    DFTD3ModelWrapper as DFTD3Model,
)
from nvalchemi.models.dsf import DSFModelWrapper as DSFCoulombModel
from nvalchemi.models.ewald import (
    EwaldCoulombConfig,
)
from nvalchemi.models.ewald import (
    EwaldModelWrapper as EwaldCoulombModel,
)
from nvalchemi.models.mace import MACEWrapper as MACEModel
from nvalchemi.models.neighbors import NeighborListBuilder
from nvalchemi.models.pme import PMEModelWrapper as PMEModel
from nvalchemi.models.utils import ANGSTROM_TO_BOHR


class _MinimalMACEModel(torch.nn.Module):
    """Small serializable MACE-like model for loader tests."""

    def __init__(self) -> None:
        super().__init__()
        self.atomic_numbers = torch.tensor([1], dtype=torch.long)
        self.r_max = torch.tensor(5.0)
        self._param = torch.nn.Linear(1, 1, bias=False)

    def forward(
        self, data_dict: dict[str, torch.Tensor], **kwargs: object
    ) -> dict[str, torch.Tensor]:
        del kwargs
        batch = data_dict["batch"].long()
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        return {
            "energy": torch.zeros(
                num_graphs,
                dtype=data_dict["positions"].dtype,
                device=data_dict["positions"].device,
            )
        }


class _SixAngstromMACEModel(_MinimalMACEModel):
    """Serializable MACE-like model with a 6.0 A cutoff."""

    def __init__(self) -> None:
        super().__init__()
        self.r_max = torch.tensor(6.0)


class _NoCutoffMACEModel(torch.nn.Module):
    """Serializable MACE-like model without an ``r_max`` cutoff."""

    def __init__(self) -> None:
        super().__init__()
        self.atomic_numbers = torch.tensor([1], dtype=torch.long)
        self._param = torch.nn.Linear(1, 1, bias=False)

    def forward(
        self, data_dict: dict[str, torch.Tensor], **kwargs: object
    ) -> dict[str, torch.Tensor]:
        del kwargs
        batch = data_dict["batch"].long()
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 1
        return {
            "energy": torch.zeros(
                num_graphs,
                dtype=data_dict["positions"].dtype,
                device=data_dict["positions"].device,
            )
        }


class _CueqBlock(torch.nn.Module):
    """Small block carrying a cueq-enabled marker."""

    def __init__(self, *, enabled: bool) -> None:
        super().__init__()
        self.cueq_config = types.SimpleNamespace(enabled=enabled)


class _CueqFlagMACEModel(_MinimalMACEModel):
    """MACE-like model with block-level cueq markers."""

    def __init__(self, *, enabled: bool) -> None:
        super().__init__()
        block = _CueqBlock(enabled=enabled)
        self.interactions = torch.nn.ModuleList([block])
        self.products = torch.nn.ModuleList([block])


class _AtomicEnergyFn(torch.nn.Module):
    """Minimal atomic-energy table holder for MACE dtype tests."""

    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.atomic_energies = torch.tensor([[0.1]], dtype=dtype)


class _AtomicEnergyMACEModel(_MinimalMACEModel):
    """MACE-like model exposing an atomic-energy table."""

    def __init__(self, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.atomic_energies_fn = _AtomicEnergyFn(dtype)


def _fake_d3_state() -> dict[str, torch.Tensor]:
    """Return a minimal DFT-D3 state dict."""

    return {
        "rcov": torch.zeros(95, dtype=torch.float32),
        "r4r2": torch.zeros(95, dtype=torch.float32),
        "c6ab": torch.zeros((95, 95, 5, 5), dtype=torch.float32),
        "cn_ref": torch.zeros((95, 95, 5, 5), dtype=torch.float32),
    }


def _install_fake_aimnet(
    monkeypatch: pytest.MonkeyPatch,
    calculator_cls: type[object],
) -> None:
    """Install one fake AIMNet calculator module."""

    calculator_module = types.ModuleType("aimnet.calculators")
    calculator_module.AIMNet2Calculator = calculator_cls
    monkeypatch.setitem(sys.modules, "aimnet", types.ModuleType("aimnet"))
    monkeypatch.setitem(sys.modules, "aimnet.calculators", calculator_module)


def _install_fake_mace_mp(
    monkeypatch: pytest.MonkeyPatch,
    loader: object,
) -> None:
    """Install one fake upstream ``mace_mp`` entrypoint."""

    calculators_module = types.ModuleType("mace.calculators")
    calculators_module.mace_mp = loader
    monkeypatch.setitem(sys.modules, "mace", types.ModuleType("mace"))
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators_module)


def test_mace_model_spec_constant_has_declared_default_cutoff() -> None:
    """The exported default MACE spec should declare the published cutoff."""

    assert MACEModel.spec.neighbor_config.cutoff == 6.0
    assert "batch" in MACEModel.spec.optional_inputs


def test_charge_and_coulomb_specs_publish_optional_batch_support() -> None:
    """Charge and Coulomb wrapper specs should declare optional batch input."""

    assert "batch" in AIMNet2Model.spec.optional_inputs
    assert "batch" in EwaldCoulombModel.spec.optional_inputs
    assert "batch" in PMEModel.spec.optional_inputs
    assert "batch" in DSFCoulombModel.spec.optional_inputs


def test_dsf_spec_marks_unit_shifts_optional() -> None:
    """DSF should only require unit shifts when periodic images are present."""

    assert "unit_shifts" not in DSFCoulombModel.spec.required_inputs
    assert "unit_shifts" in DSFCoulombModel.spec.optional_inputs


def test_neighbor_list_builder_retries_when_matrix_counts_exceed_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Adaptive matrix builders should retry on silently inconsistent widths."""

    calls: list[int | None] = []

    def _fake_neighbor_list(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        max_neighbors = kwargs.get("max_neighbors")
        calls.append(cast(int | None, max_neighbors))
        positions = cast(torch.Tensor, kwargs["positions"])
        if len(calls) == 1:
            return (
                torch.full((2, 16), -1, dtype=torch.int32, device=positions.device),
                torch.tensor([20, 0], dtype=torch.int32, device=positions.device),
            )
        matrix = torch.full((2, 32), -1, dtype=torch.int32, device=positions.device)
        matrix[0, :20] = 1
        return (
            matrix,
            torch.tensor([20, 0], dtype=torch.int32, device=positions.device),
        )

    monkeypatch.setattr(neighbors_module, "neighbor_list", _fake_neighbor_list)

    builder = NeighborListBuilder(cutoff=5.0, format="matrix")
    result = builder(
        positions=torch.zeros((2, 3), dtype=torch.float32),
        batch_idx=torch.zeros(2, dtype=torch.long),
        batch_ptr=torch.tensor([0, 2], dtype=torch.long),
    )

    assert len(calls) == 2
    assert calls[0] is not None
    assert calls[1] is not None
    assert calls[1] > calls[0]
    assert result["neighbor_matrix"].shape == (2, 20)
    assert int(result["num_neighbors"].max().item()) == 20


def test_neighbor_list_builder_fixed_matrix_capacity_raises_on_inconsistent_width(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fixed matrix builders should raise on silently inconsistent widths."""

    def _fake_neighbor_list(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        positions = cast(torch.Tensor, kwargs["positions"])
        return (
            torch.full((2, 16), -1, dtype=torch.int32, device=positions.device),
            torch.tensor([20, 0], dtype=torch.int32, device=positions.device),
        )

    monkeypatch.setattr(neighbors_module, "neighbor_list", _fake_neighbor_list)

    builder = NeighborListBuilder(cutoff=5.0, format="matrix", max_neighbors=16)

    with pytest.raises(
        ValueError,
        match="Neighbor matrix width is smaller than the reported valid neighbor count",
    ):
        builder(
            positions=torch.zeros((2, 3), dtype=torch.float32),
            batch_idx=torch.zeros(2, dtype=torch.long),
            batch_ptr=torch.tensor([0, 2], dtype=torch.long),
        )


def test_mace_named_resolution_uses_upstream_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Named MACE models should resolve through the upstream loader."""

    captured: dict[str, object] = {}

    def _fake_mace_mp(*args: object, **kwargs: object) -> torch.nn.Module:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _MinimalMACEModel()

    _install_fake_mace_mp(monkeypatch, _fake_mace_mp)

    model = MACEModel("medium", enable_cueq=False, compile_model=False, device="cpu")

    assert isinstance(model._model, _MinimalMACEModel)
    assert captured["kwargs"]["model"] == "medium"
    assert captured["kwargs"]["device"] == "cpu"


def test_mace_named_resolution_accepts_calculator_style_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Named MACE loading should unwrap calculator-style results."""

    class _CalculatorResult:
        def __init__(self) -> None:
            self.model = _MinimalMACEModel()

    def _fake_mace_mp(*args: object, **kwargs: object) -> _CalculatorResult:
        del args, kwargs
        return _CalculatorResult()

    _install_fake_mace_mp(monkeypatch, _fake_mace_mp)

    model = MACEModel("medium", enable_cueq=False, compile_model=False, device="cpu")

    assert isinstance(model._model, _MinimalMACEModel)


def test_mace_named_resolution_accepts_single_model_list_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Named MACE loading should unwrap calculator-style `.models` payloads."""

    class _CalculatorResult:
        def __init__(self) -> None:
            self.models = [_MinimalMACEModel()]

    def _fake_mace_mp(*args: object, **kwargs: object) -> _CalculatorResult:
        del args, kwargs
        return _CalculatorResult()

    _install_fake_mace_mp(monkeypatch, _fake_mace_mp)

    model = MACEModel("medium", enable_cueq=False, compile_model=False, device="cpu")

    assert isinstance(model._model, _MinimalMACEModel)


def test_mace_model_uses_checkpoint_cutoff_without_warning_when_matching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Matching declared and checkpoint cutoffs should not emit a warning."""

    checkpoint_path = tmp_path / "mace-six.pt"
    checkpoint_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        MACEModel,
        "load_checkpoint",
        classmethod(lambda cls, path: _SixAngstromMACEModel()),
    )

    model = MACEModel(checkpoint_path, enable_cueq=False)

    assert model.spec.neighbor_config.cutoff == 6.0


def test_mace_model_uses_checkpoint_cutoff_even_when_default_differs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Instance MACE specs should always use the checkpoint cutoff."""

    checkpoint_path = tmp_path / "mace-five.pt"
    torch.save(_MinimalMACEModel(), checkpoint_path)
    monkeypatch.setattr(
        MACEModel,
        "load_checkpoint",
        classmethod(lambda cls, path: _MinimalMACEModel()),
    )

    model = MACEModel(checkpoint_path, enable_cueq=False)

    assert model.spec.neighbor_config.cutoff == 5.0


def test_mace_typed_cutoff_and_pbc_mode_overrides_replace_instance_spec(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Explicit typed overrides should replace the derived instance spec fields."""

    checkpoint_path = tmp_path / "mace-five.pt"
    torch.save(_MinimalMACEModel(), checkpoint_path)
    monkeypatch.setattr(
        MACEModel,
        "load_checkpoint",
        classmethod(lambda cls, path: _MinimalMACEModel()),
    )

    model = MACEModel(
        checkpoint_path,
        enable_cueq=False,
        cutoff=7.0,
        pbc_mode="pbc",
    )

    assert model.spec.neighbor_config.cutoff == 7.0
    assert model.spec.pbc_mode == "pbc"


def test_mace_exposes_public_dtype_property(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MACE should publish its compute dtype through the public dtype property."""

    checkpoint_path = tmp_path / "mace-five.pt"
    torch.save(_MinimalMACEModel(), checkpoint_path)
    monkeypatch.setattr(
        MACEModel,
        "load_checkpoint",
        classmethod(lambda cls, path: _MinimalMACEModel()),
    )

    model = MACEModel(checkpoint_path, enable_cueq=False)

    assert model.dtype == torch.float32


def test_mace_preserves_atomic_energy_dtype_during_conversion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MACE should preserve the backend atomic-energy dtype during conversion."""

    checkpoint_path = tmp_path / "mace-atomic-energy.pt"
    checkpoint_path.write_bytes(b"placeholder")
    monkeypatch.setattr(
        MACEModel,
        "load_checkpoint",
        classmethod(lambda cls, path: _AtomicEnergyMACEModel(dtype=torch.float32)),
    )

    model = MACEModel(
        checkpoint_path,
        enable_cueq=False,
        dtype=torch.float32,
    )

    assert model._model.atomic_energies_fn.atomic_energies.dtype == torch.float32


def test_mace_apply_syncs_device_and_dtype_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """MACE should keep public device/dtype metadata in sync after module moves."""

    checkpoint_path = tmp_path / "mace-five.pt"
    torch.save(_MinimalMACEModel(), checkpoint_path)
    monkeypatch.setattr(
        MACEModel,
        "load_checkpoint",
        classmethod(lambda cls, path: _MinimalMACEModel()),
    )

    model = MACEModel(
        checkpoint_path, enable_cueq=False, compile_model=False, device="cpu"
    )
    result = model._apply(lambda tensor: tensor.to(device="meta", dtype=torch.float64))

    assert result is model
    assert model.device.type == "meta"
    assert model.dtype == torch.float64
    assert model._node_emb.device.type == "meta"


def test_mace_missing_rmax_warns_and_keeps_declared_cutoff(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing checkpoint cutoff should warn and keep the declared cutoff."""

    checkpoint_path = tmp_path / "mace-no-rmax.pt"
    torch.save(_NoCutoffMACEModel(), checkpoint_path)
    monkeypatch.setattr(
        MACEModel,
        "load_checkpoint",
        classmethod(lambda cls, path: _NoCutoffMACEModel()),
    )

    with pytest.warns(UserWarning, match="does not expose an 'r_max' cutoff attribute"):
        model = MACEModel(checkpoint_path, enable_cueq=False)

    assert model.spec.neighbor_config.cutoff == 6.0


def test_mace_convert_to_cueq_warns_and_falls_back_on_conversion_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed cueq conversion should warn and keep the original model."""

    fake_pkg = types.ModuleType("mace")
    fake_cli = types.ModuleType("mace.cli")
    fake_module = types.ModuleType("mace.cli.convert_e3nn_cueq")

    def _raise(_model: torch.nn.Module) -> torch.nn.Module:
        raise RuntimeError("boom")

    fake_module.run = _raise
    fake_pkg.cli = fake_cli
    fake_cli.convert_e3nn_cueq = fake_module
    monkeypatch.setitem(sys.modules, "mace", fake_pkg)
    monkeypatch.setitem(sys.modules, "mace.cli", fake_cli)
    monkeypatch.setitem(sys.modules, "mace.cli.convert_e3nn_cueq", fake_module)

    model = _MinimalMACEModel()
    with pytest.warns(
        UserWarning,
        match="Failed to convert the MACE model to cuEquivariance",
    ):
        converted = MACEModel._convert_to_cueq(model)

    assert converted is model


def test_mace_cueq_detection_uses_block_level_cueq_config() -> None:
    """Cueq detection should rely on block config markers."""

    assert MACEModel._is_cueq_model(_CueqFlagMACEModel(enabled=True))
    assert not MACEModel._is_cueq_model(_CueqFlagMACEModel(enabled=False))
    assert not MACEModel._is_cueq_model(_MinimalMACEModel())


def test_mace_prepare_loaded_model_warns_when_conversion_does_not_enable_cueq(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Returned non-cueq models should warn after a conversion attempt."""

    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(
        MACEModel,
        "_convert_to_cueq",
        staticmethod(lambda model: _CueqFlagMACEModel(enabled=False)),
    )

    with pytest.warns(
        UserWarning,
        match="returned a model without cueq acceleration enabled",
    ):
        prepared = MACEModel._prepare_loaded_model(
            _MinimalMACEModel(),
            device="cuda",
            dtype=None,
            enable_cueq=True,
            compile_model=False,
        )

    assert not MACEModel._is_cueq_model(prepared["model"])


def test_aimnet_named_resolution_uses_upstream_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Named AIMNet2 models should resolve through AIMNetCentral."""

    captured: dict[str, object] = {}

    class _FakeCalculator:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.device = "cpu"
            self.model = torch.nn.Identity()

        def mol_flatten(
            self, calc_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return calc_input

        def make_nbmat(
            self, model_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return model_input

        def pad_input(
            self, model_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return model_input

        def unpad_output(
            self, raw_output: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return raw_output

    _install_fake_aimnet(monkeypatch, _FakeCalculator)

    model = AIMNet2Model("aimnet2", device="cpu")

    assert captured["model"] == "aimnet2"
    assert "graph_spins" not in model.spec.optional_inputs


def test_aimnet_path_resolution_passes_local_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Local AIMNet2 checkpoint paths should be passed through unchanged."""

    captured: dict[str, object] = {}
    checkpoint = tmp_path / "aimnet.pt"
    checkpoint.write_bytes(b"aimnet")

    class _FakeCalculator:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)
            self.device = "cpu"
            self.model = torch.nn.Identity()

        def mol_flatten(
            self, calc_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return calc_input

        def make_nbmat(
            self, model_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return model_input

        def pad_input(
            self, model_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return model_input

        def unpad_output(
            self, raw_output: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return raw_output

    _install_fake_aimnet(monkeypatch, _FakeCalculator)

    AIMNet2Model(checkpoint, device="cpu")

    assert captured["model"] == str(checkpoint)


def test_aimnet_spec_includes_graph_spins_for_two_channel_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two-channel AIMNet checkpoints should advertise graph_spins input."""

    class _TwoChannelModel(torch.nn.Identity):
        num_charge_channels = 2

    class _FakeCalculator:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.device = "cpu"
            self.model = _TwoChannelModel()

        def mol_flatten(
            self, calc_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return calc_input

        def make_nbmat(
            self, model_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return model_input

        def pad_input(
            self, model_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return model_input

        def unpad_output(
            self, raw_output: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return raw_output

    _install_fake_aimnet(monkeypatch, _FakeCalculator)

    model = AIMNet2Model(torch.nn.Identity(), device="cpu")

    assert "graph_spins" in model.spec.optional_inputs


def test_aimnet_nse_defaults_missing_graph_inputs_to_neutral_singlets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two-channel AIMNet should default missing graph inputs to neutral singlets."""

    captured: dict[str, torch.Tensor] = {}

    class _TwoChannelModel(torch.nn.Module):
        num_charge_channels = 2

        def forward(
            self, model_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            captured["charge"] = model_input["charge"]
            captured["mult"] = model_input["mult"]
            n_atoms = model_input["coord"].shape[0]
            return {
                "energy": torch.zeros(1, dtype=model_input["coord"].dtype),
                "charges": torch.zeros(n_atoms, dtype=model_input["coord"].dtype),
            }

    class _FakeCalculator:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.device = "cpu"
            self.model = _TwoChannelModel()

        def mol_flatten(
            self, calc_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return calc_input

        def make_nbmat(
            self, model_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return model_input

        def pad_input(
            self, model_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return model_input

        def unpad_output(
            self, raw_output: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return raw_output

    _install_fake_aimnet(monkeypatch, _FakeCalculator)

    model = AIMNet2Model(torch.nn.Identity(), device="cpu")
    outputs = model(
        {
            "positions": torch.zeros(2, 3),
            "atomic_numbers": torch.tensor([1, 1]),
        }
    )

    assert outputs["node_charges"].shape == (2, 1)
    assert torch.equal(captured["charge"], torch.zeros(1))
    assert torch.equal(captured["mult"], torch.ones(1))


def test_aimnet_apply_syncs_wrapper_and_calculator_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AIMNet should keep wrapper and calculator device metadata aligned."""

    class _FakeCalculator:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.device = "cpu"
            self.model = torch.nn.Identity()

        def mol_flatten(
            self, calc_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return calc_input

        def make_nbmat(
            self, model_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return model_input

        def pad_input(
            self, model_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return model_input

        def unpad_output(
            self, raw_output: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return raw_output

    _install_fake_aimnet(monkeypatch, _FakeCalculator)

    model = AIMNet2Model(torch.nn.Identity(), device="cpu")
    result = model._apply(lambda tensor: tensor.to(device="meta"))

    assert result is model
    assert model.device.type == "meta"
    assert model._calculator.device == "meta"
    assert model._calculator.model is model._model


def test_aimnet_requires_charges_from_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AIMNet should fail clearly when the backend omits charges."""

    class _ModelWithoutCharges(torch.nn.Module):
        def forward(
            self, model_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return {"energy": torch.zeros(1, dtype=model_input["coord"].dtype)}

    class _FakeCalculator:
        def __init__(self, **kwargs: object) -> None:
            del kwargs
            self.device = "cpu"
            self.model = _ModelWithoutCharges()

        def mol_flatten(
            self, calc_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return calc_input

        def make_nbmat(
            self, model_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return model_input

        def pad_input(
            self, model_input: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return model_input

        def unpad_output(
            self, raw_output: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
            return raw_output

    _install_fake_aimnet(monkeypatch, _FakeCalculator)

    model = AIMNet2Model(torch.nn.Identity(), device="cpu")

    with pytest.raises(ValueError, match="expected 'charges'"):
        model(
            {
                "positions": torch.zeros(2, 3),
                "atomic_numbers": torch.tensor([1, 1]),
            }
        )


def test_dftd3_download_helper_caches_converted_params(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The standalone DFT-D3 helper should reuse the converted cache file."""

    calls = {"download": 0, "extract": 0}

    def _fake_download_cached_file(**kwargs: object) -> Path:
        calls["download"] += 1
        destination = Path(kwargs["cache_dir"]) / str(kwargs["filename"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"archive")
        return destination

    monkeypatch.setattr(
        "nvalchemi.models.dftd3.download_cached_file",
        _fake_download_cached_file,
    )
    monkeypatch.setattr(
        "nvalchemi.models.dftd3.DFTD3_ARCHIVE_MD5",
        md5(b"archive", usedforsecurity=False).hexdigest(),
    )
    monkeypatch.setattr(
        DFTD3ParametersProcessor,
        "extract_parameters_from_archive",
        classmethod(
            lambda cls, archive_path: (
                calls.__setitem__("extract", calls["extract"] + 1) or _fake_d3_state()
            )
        ),
    )

    first = download_dftd3_parameters(cache_dir=tmp_path)
    second = download_dftd3_parameters(cache_dir=tmp_path)

    assert first == second
    assert first.exists()
    assert calls["download"] == 1
    assert calls["extract"] == 1


def test_dftd3_model_downloads_default_params_when_param_path_omitted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DFT-D3Model should auto-materialize cached parameters by default."""

    def _fake_download_cached_file(**kwargs: object) -> Path:
        destination = Path(kwargs["cache_dir"]) / str(kwargs["filename"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"archive")
        return destination

    monkeypatch.setattr(
        "nvalchemi.models.dftd3.download_cached_file",
        _fake_download_cached_file,
    )
    monkeypatch.setattr(
        "nvalchemi.models.dftd3.DFTD3_ARCHIVE_MD5",
        md5(b"archive", usedforsecurity=False).hexdigest(),
    )
    monkeypatch.setattr(
        DFTD3ParametersProcessor,
        "extract_parameters_from_archive",
        classmethod(lambda cls, archive_path: _fake_d3_state()),
    )

    model = DFTD3Model(functional="pbe", device="cpu")

    assert model.device.type == "cpu"
    assert model._d3_params.device.type == "cpu"


def test_dftd3_model_missing_explicit_param_path_raises(tmp_path: Path) -> None:
    """Explicit missing parameter paths should fail clearly."""

    with pytest.raises(FileNotFoundError, match="DFT-D3 parameter file not found"):
        DFTD3Model(param_path=tmp_path / "missing_dftd3.pt")


def test_dftd3_download_helper_verifies_archive_checksum(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The standalone DFT-D3 helper should fail on archive checksum mismatch."""

    def _fake_download_cached_file(**kwargs: object) -> Path:
        destination = Path(kwargs["cache_dir"]) / str(kwargs["filename"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"corrupt-archive")
        return destination

    monkeypatch.setattr(
        "nvalchemi.models.dftd3.download_cached_file",
        _fake_download_cached_file,
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_dftd3_parameters(cache_dir=tmp_path, force=True)


def test_dftd3_typed_pbc_mode_override_replaces_instance_spec(tmp_path: Path) -> None:
    """Explicit pbc_mode should replace DFT-D3 instance spec metadata."""

    param_path = tmp_path / "dftd3.pt"
    torch.save(_fake_d3_state(), param_path)

    model = DFTD3Model(param_path=param_path, pbc_mode="pbc", device="cpu")

    assert model.spec.pbc_mode == "pbc"


def test_dftd3_model_passes_bohr_converted_smoothing_to_kernel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Forward should pass Bohr-converted smoothing distances to the kernel."""

    param_path = tmp_path / "dftd3.pt"
    torch.save(_fake_d3_state(), param_path)
    captured: dict[str, float] = {}

    def _fake_dftd3(
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        captured["s5_smoothing_on"] = float(kwargs["s5_smoothing_on"])
        captured["s5_smoothing_off"] = float(kwargs["s5_smoothing_off"])
        positions = kwargs["positions"]
        assert isinstance(positions, torch.Tensor)
        n_atoms = positions.shape[0]
        return (
            torch.zeros(1, dtype=positions.dtype, device=positions.device),
            torch.zeros(n_atoms, 3, dtype=positions.dtype, device=positions.device),
            torch.zeros(n_atoms, dtype=positions.dtype, device=positions.device),
        )

    monkeypatch.setattr("nvalchemi.models.dftd3.dftd3", _fake_dftd3)

    model = DFTD3Model(
        param_path=param_path,
        smoothing_fraction=0.2,
        cutoff=20.0,
        device="cpu",
    )
    model(
        {
            "positions": torch.zeros(2, 3),
            "atomic_numbers": torch.tensor([1, 1]),
            "neighbor_matrix": torch.full((2, 1), -1, dtype=torch.int32),
        }
    )

    assert captured["s5_smoothing_on"] == pytest.approx(16.0 * ANGSTROM_TO_BOHR)
    assert captured["s5_smoothing_off"] == pytest.approx(20.0 * ANGSTROM_TO_BOHR)


def test_dftd3_device_moves_cached_parameters(tmp_path: Path) -> None:
    """DFT-D3 should place cached parameters on the requested execution device."""

    param_path = tmp_path / "dftd3.pt"
    torch.save(_fake_d3_state(), param_path)

    model = DFTD3Model(param_path=param_path, device="cpu")

    assert model.device.type == "cpu"
    assert model._d3_params.device.type == "cpu"


def test_dftd3_apply_syncs_device_and_cached_parameters(tmp_path: Path) -> None:
    """DFT-D3 should move cached parameters together with wrapper metadata."""

    class _FakeD3Parameters:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.dtype = torch.float32

        def to(
            self,
            device: str | torch.device | None = None,
            dtype: torch.dtype | None = None,
        ) -> _FakeD3Parameters:
            if device is not None:
                self.device = torch.device(device)
            if dtype is not None:
                self.dtype = dtype
            return self

    param_path = tmp_path / "dftd3.pt"
    torch.save(_fake_d3_state(), param_path)

    model = DFTD3Model(param_path=param_path, device="cpu")
    fake_params = _FakeD3Parameters()
    model._d3_params = fake_params

    result = model._apply(lambda tensor: tensor.to(device="meta"))

    assert result is model
    assert model.device.type == "meta"
    assert fake_params.device.type == "meta"


def test_dftd3_repr_flattens_config_and_tracks_device(tmp_path: Path) -> None:
    """DFT-D3 repr should flatten config fields and reflect device moves."""

    param_path = tmp_path / "dftd3.pt"
    torch.save(_fake_d3_state(), param_path)
    config = DFTD3Config(functional="pbe0", k3=-3.0, param_path=param_path)
    model = DFTD3Model(config=config)
    model._apply(lambda tensor: tensor.to(device="meta"))
    rendered = repr(model)

    assert rendered.startswith("DFTD3ModelWrapper(")
    assert "functional='pbe0'" in rendered
    assert "k3=-3.0" in rendered
    assert f"param_path={param_path!r}" in rendered
    assert "device='meta'" in rendered
    assert "config=" not in rendered


def test_dftd3_and_coulomb_default_specs_publish_fifteen_angstrom_cutoffs() -> None:
    """Default DFT-D3 and Coulomb specs should declare the shared 15 A cutoff."""

    assert DFTD3Model.spec.neighbor_config.cutoff == 15.0
    assert PMEModel.spec.neighbor_config.cutoff == 15.0
    assert EwaldCoulombModel.spec.neighbor_config.cutoff == 15.0
    assert DSFCoulombModel.spec.neighbor_config.cutoff == 15.0


def test_dsf_forward_allows_missing_unit_shifts_for_non_pbc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-periodic DSF should pass missing unit shifts through as None."""

    captured: dict[str, object] = {}

    def _fake_dsf_coulomb(**kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        captured.update(kwargs)
        positions = kwargs["positions"]
        assert isinstance(positions, torch.Tensor)
        num_systems = kwargs["num_systems"]
        assert isinstance(num_systems, int)
        return (
            torch.zeros(num_systems, dtype=positions.dtype, device=positions.device),
            torch.zeros_like(positions),
        )

    monkeypatch.setattr(dsf_module, "dsf_coulomb", _fake_dsf_coulomb)

    model = DSFCoulombModel(cutoff=6.0, alpha=0.0)
    outputs = model(
        {
            "positions": torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]),
            "node_charges": torch.tensor([[0.2], [-0.2]]),
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
            "neighbor_ptr": torch.tensor([0, 1, 2], dtype=torch.int64),
            "batch": torch.tensor([0, 0], dtype=torch.int64),
        }
    )

    assert captured["unit_shifts"] is None
    assert cast(torch.Tensor, captured["neighbor_list"]).dtype == torch.int32
    assert cast(torch.Tensor, captured["neighbor_ptr"]).dtype == torch.int32
    assert cast(torch.Tensor, captured["batch_idx"]).dtype == torch.int32
    assert set(outputs) == {"energies", "forces"}


def test_dsf_forward_preserves_periodic_shift_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Periodic DSF should still pass cell and unit shifts to the kernel."""

    captured: dict[str, object] = {}

    def _fake_dsf_coulomb(
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        captured.update(kwargs)
        positions = kwargs["positions"]
        assert isinstance(positions, torch.Tensor)
        num_systems = kwargs["num_systems"]
        assert isinstance(num_systems, int)
        return (
            torch.zeros(num_systems, dtype=positions.dtype, device=positions.device),
            torch.zeros_like(positions),
            torch.zeros(
                (num_systems, 3, 3),
                dtype=positions.dtype,
                device=positions.device,
            ),
        )

    monkeypatch.setattr(dsf_module, "dsf_coulomb", _fake_dsf_coulomb)

    model = DSFCoulombModel(cutoff=6.0, alpha=0.0)
    outputs = model(
        {
            "positions": torch.tensor([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]]),
            "node_charges": torch.tensor([[0.2], [-0.2]]),
            "edge_index": torch.tensor([[0, 1], [1, 0]], dtype=torch.int64),
            "neighbor_ptr": torch.tensor([0, 1, 2], dtype=torch.int64),
            "unit_shifts": torch.tensor([[0, 0, 0], [1, 0, 0]], dtype=torch.int64),
            "batch": torch.tensor([0, 0], dtype=torch.int64),
            "cell": torch.eye(3).unsqueeze(0),
            "pbc": torch.tensor([[True, True, True]], dtype=torch.bool),
        }
    )

    assert isinstance(captured["cell"], torch.Tensor)
    assert cast(torch.Tensor, captured["unit_shifts"]).dtype == torch.int32
    assert "stresses" in outputs


def test_ewald_accepts_pme_style_config_and_overrides() -> None:
    """Ewald should accept a config plus typed overrides like PME."""

    config = EwaldCoulombConfig(cutoff=5.0, accuracy=1e-5)

    model = EwaldCoulombModel(config=config, cutoff=7.0)

    assert model._config.cutoff == pytest.approx(7.0)
    assert model._config.accuracy == pytest.approx(1e-5)
    assert model.spec.neighbor_config.cutoff == pytest.approx(7.0)
    assert model.spec.use_autograd is True
    assert model.spec.outputs == frozenset({"energies"})


def test_ewald_estimation_uses_int32_batch_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ewald should pass int32 batch indices to backend parameter estimation."""

    captured: dict[str, torch.dtype] = {}

    class _Params:
        def __init__(self, alpha: torch.Tensor) -> None:
            self.alpha = alpha
            self.reciprocal_space_cutoff = alpha

    def _fake_estimate(
        positions: torch.Tensor,
        cell: torch.Tensor,
        *,
        batch_idx: torch.Tensor | None = None,
        accuracy: float = 1e-6,
    ) -> _Params:
        del positions, accuracy
        assert batch_idx is not None
        captured["dtype"] = batch_idx.dtype
        return _Params(
            alpha=torch.ones(cell.shape[0], dtype=cell.dtype, device=cell.device)
        )

    def _fake_generate_k_vectors(
        cell: torch.Tensor,
        k_cutoff: float | torch.Tensor,
    ) -> torch.Tensor:
        del cell, k_cutoff
        return torch.zeros((1, 3), dtype=torch.float32)

    def _fake_real_space(
        **kwargs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = kwargs["positions"]
        cell = kwargs["cell"]
        return (
            torch.zeros(
                positions.shape[0], dtype=positions.dtype, device=positions.device
            ),
            torch.zeros_like(positions),
            torch.zeros(
                cell.shape[0],
                3,
                3,
                dtype=positions.dtype,
                device=positions.device,
            ),
        )

    def _fake_reciprocal_space(
        *,
        positions: torch.Tensor,
        charges: torch.Tensor,
        cell: torch.Tensor,
        k_vectors: torch.Tensor,
        alpha: torch.Tensor,
        batch_idx: torch.Tensor,
        compute_forces: bool,
        compute_virial: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        del charges, cell, k_vectors, alpha, batch_idx, compute_forces, compute_virial
        return (
            torch.zeros(
                positions.shape[0], dtype=positions.dtype, device=positions.device
            ),
            torch.zeros_like(positions),
            torch.zeros(
                1,
                3,
                3,
                dtype=positions.dtype,
                device=positions.device,
            ),
        )

    monkeypatch.setattr(ewald_module, "estimate_ewald_parameters", _fake_estimate)
    monkeypatch.setattr(
        ewald_module,
        "generate_k_vectors_ewald_summation",
        _fake_generate_k_vectors,
    )
    monkeypatch.setattr(ewald_module, "ewald_real_space", _fake_real_space)
    monkeypatch.setattr(
        ewald_module,
        "ewald_reciprocal_space",
        _fake_reciprocal_space,
    )

    model = EwaldCoulombModel(cutoff=10.0)
    outputs = model(
        {
            "positions": torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            "node_charges": torch.tensor([[1.0], [-1.0]], dtype=torch.float32),
            "cell": torch.eye(3, dtype=torch.float32).unsqueeze(0),
            "pbc": torch.tensor([[True, True, True]], dtype=torch.bool),
            "neighbor_matrix": torch.tensor([[1, -1], [0, -1]], dtype=torch.int32),
            "num_neighbors": torch.tensor([1, 1], dtype=torch.int32),
            "neighbor_shifts": torch.zeros((2, 2, 3), dtype=torch.int32),
            "batch": torch.tensor([0, 0], dtype=torch.int64),
        }
    )

    assert captured["dtype"] == torch.int32
    assert outputs["energies"].shape == (1, 1)


def test_pme_estimation_uses_int32_batch_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PME should pass int32 batch indices to backend parameter estimation."""

    captured: dict[str, torch.dtype] = {}

    class _Params:
        def __init__(self, alpha: torch.Tensor) -> None:
            self.alpha = alpha
            self.mesh_dimensions = (8, 8, 8)

    def _fake_estimate(
        positions: torch.Tensor,
        cell: torch.Tensor,
        *,
        batch_idx: torch.Tensor | None = None,
        accuracy: float = 1e-6,
    ) -> _Params:
        del positions, accuracy
        assert batch_idx is not None
        captured["dtype"] = batch_idx.dtype
        return _Params(
            alpha=torch.ones(cell.shape[0], dtype=cell.dtype, device=cell.device)
        )

    def _fake_generate_k_vectors(
        cell: torch.Tensor,
        mesh_dimensions: tuple[int, int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cell, mesh_dimensions
        zeros = torch.zeros((1, 3), dtype=torch.float32)
        return zeros, zeros

    def _fake_particle_mesh_ewald(
        **kwargs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        positions = kwargs["positions"]
        return (
            torch.zeros(
                positions.shape[0], dtype=positions.dtype, device=positions.device
            ),
            torch.zeros_like(positions),
            torch.zeros(1, 3, 3, dtype=positions.dtype, device=positions.device),
        )

    monkeypatch.setattr(pme_module, "estimate_pme_parameters", _fake_estimate)
    monkeypatch.setattr(pme_module, "generate_k_vectors_pme", _fake_generate_k_vectors)
    monkeypatch.setattr(pme_module, "particle_mesh_ewald", _fake_particle_mesh_ewald)

    model = PMEModel(cutoff=10.0)
    outputs = model(
        {
            "positions": torch.tensor(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                dtype=torch.float32,
            ),
            "node_charges": torch.tensor([[1.0], [-1.0]], dtype=torch.float32),
            "cell": torch.eye(3, dtype=torch.float32).unsqueeze(0),
            "pbc": torch.tensor([[True, True, True]], dtype=torch.bool),
            "neighbor_matrix": torch.tensor([[1, -1], [0, -1]], dtype=torch.int32),
            "num_neighbors": torch.tensor([1, 1], dtype=torch.int32),
            "neighbor_shifts": torch.zeros((2, 2, 3), dtype=torch.int32),
            "batch": torch.tensor([0, 0], dtype=torch.int64),
        }
    )

    assert captured["dtype"] == torch.int32
    assert outputs["energies"].shape == (1, 1)


def test_demo_model_repr_shows_only_non_default_fields() -> None:
    """Demo repr should stay compact and omit default kwargs."""

    model = DemoModel(hidden_dim=128, name="demo")

    assert repr(model) == "DemoModelWrapper(hidden_dim=128, name='demo')"
