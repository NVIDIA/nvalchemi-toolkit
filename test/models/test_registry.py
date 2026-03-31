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
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import torch

from nvalchemi.models.aimnet2 import AIMNet2Potential
from nvalchemi.models.dftd3 import DFTD3Config, DFTD3Potential, load_dftd3_params
from nvalchemi.models.mace import MACEPotential
from nvalchemi.models.metadata import CheckpointInfo
from nvalchemi.models.registry import (
    KnownArtifactEntry,
    ResolvedArtifact,
    get_known_artifact,
    list_known_artifacts,
    register_known_artifact,
    resolve_known_artifact,
)


class _CopyProcessor:
    """Simple processor used to test registry post-processing."""

    def materialize(
        self,
        *,
        downloaded_path: Path,
        entry: KnownArtifactEntry,
        output_path: Path,
    ) -> Path:
        del entry
        output_path.write_bytes(downloaded_path.read_bytes().upper())
        return output_path


class _MinimalMACEModel(torch.nn.Module):
    """Small serializable MACE-like model for registry resolution tests."""

    def __init__(self) -> None:
        super().__init__()
        self.atomic_numbers = torch.tensor([1], dtype=torch.long)
        self.r_max = torch.tensor(5.0)
        self._param = torch.nn.Linear(1, 1, bias=False)

    def forward(self, data_dict: dict[str, torch.Tensor], **kwargs: object) -> dict[str, torch.Tensor]:
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


def _fake_d3_state() -> dict[str, torch.Tensor]:
    """Return a minimal DFT-D3 state dict."""

    return {
        "rcov": torch.zeros(95, dtype=torch.float32),
        "r4r2": torch.zeros(95, dtype=torch.float32),
        "c6ab": torch.zeros((95, 95, 5, 5), dtype=torch.float32),
        "cn_ref": torch.zeros((95, 95, 5, 5), dtype=torch.float32),
    }


def test_registry_downloads_and_processes_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Registry should cache processed artifacts after one verified download."""

    raw_payload = b"registry payload"
    sha256 = hashlib.sha256(raw_payload).hexdigest()
    calls = {"count": 0}

    def _fake_urlretrieve(url: str, filename: str | Path) -> tuple[str, None]:
        del url
        calls["count"] += 1
        Path(filename).write_bytes(raw_payload)
        return str(filename), None

    monkeypatch.setattr("urllib.request.urlretrieve", _fake_urlretrieve)
    register_known_artifact(
        KnownArtifactEntry(
            name="test-processed-artifact",
            family="test",
            url="https://example.invalid/raw.bin",
            sha256=sha256,
            filename="raw.bin",
            materialized_filename="processed.bin",
            processor=_CopyProcessor(),
        )
    )

    resolved = resolve_known_artifact(
        "test-processed-artifact",
        family="test",
        cache_dir=tmp_path,
    )
    again = resolve_known_artifact(
        "test-processed-artifact",
        family="test",
        cache_dir=tmp_path,
    )

    assert resolved.local_path.read_bytes() == raw_payload.upper()
    assert again.local_path == resolved.local_path
    assert calls["count"] == 1


def test_registry_verifies_md5_when_provided(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Registry should verify MD5 checksums for downloaded artifacts."""

    raw_payload = b"dftd3 archive payload"
    md5_digest = hashlib.md5(raw_payload, usedforsecurity=False).hexdigest()

    def _fake_urlretrieve(url: str, filename: str | Path) -> tuple[str, None]:
        del url
        Path(filename).write_bytes(raw_payload)
        return str(filename), None

    monkeypatch.setattr("urllib.request.urlretrieve", _fake_urlretrieve)
    register_known_artifact(
        KnownArtifactEntry(
            name="test-md5-artifact",
            family="test-md5",
            url="https://example.invalid/raw.tgz",
            md5=md5_digest,
            filename="raw.tgz",
        )
    )

    resolved = resolve_known_artifact(
        "test-md5-artifact",
        family="test-md5",
        cache_dir=tmp_path,
    )

    assert resolved.local_path.read_bytes() == raw_payload


def test_mace_registry_resolution_populates_checkpoint_info(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Known MACE names should resolve through the rewrite registry first."""

    checkpoint_path = tmp_path / "mock_mace.pt"
    torch.save(_MinimalMACEModel(), checkpoint_path)

    resolved = ResolvedArtifact(
        entry=KnownArtifactEntry(
            name="known-mace",
            family="mace",
            metadata={"model_name": "known-mace"},
        ),
        local_path=checkpoint_path,
        checkpoint=CheckpointInfo(
            identifier="known-mace",
            url="https://example.invalid/known-mace.model",
            sha256="deadbeef",
            source="registry",
        ),
    )
    monkeypatch.setattr(
        "nvalchemi.models.mace.resolve_known_artifact",
        lambda name, family: resolved,
    )

    potential = MACEPotential("known-mace", enable_cueq=False)

    assert potential.model_card.model_name == "known-mace"
    assert potential.model_card.checkpoint == resolved.checkpoint


def test_dftd3_registry_resolution_populates_checkpoint_info(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Default DFT-D3 parameters should be resolved through the rewrite registry."""

    param_path = tmp_path / "registry_dftd3.pt"
    torch.save(_fake_d3_state(), param_path)

    resolved = ResolvedArtifact(
        entry=KnownArtifactEntry(
            name="dftd3_parameters",
            family="dftd3",
            metadata={"model_name": "dftd3_parameters"},
        ),
        local_path=param_path,
        checkpoint=CheckpointInfo(
            identifier="dftd3_parameters",
            url="https://example.invalid/dftd3.tgz",
            source="registry",
        ),
    )
    monkeypatch.setattr(
        "nvalchemi.models.dftd3.resolve_known_artifact",
        lambda *args, **kwargs: resolved,
    )

    params = load_dftd3_params(auto_download=False)
    potential = DFTD3Potential(DFTD3Config(functional="pbe", auto_download=False))

    assert tuple(params.rcov.shape) == (95,)
    assert potential.model_card.model_name == "dftd3_parameters"
    assert potential.model_card.checkpoint == resolved.checkpoint


def test_builtin_aimnet_registry_is_explicit_and_local() -> None:
    """Built-in AIMNet registry entries should be stable and locally defined."""

    assert list_known_artifacts("aimnet2") == ["aimnet2"]

    entry = get_known_artifact("aimnet2", "aimnet2")

    assert entry.name == "aimnet2"
    assert entry.filename == "aimnet2_wb97m_d3_0.pt"
    assert (
        entry.url
        == "https://storage.googleapis.com/aimnetcentral/aimnet2v2/AIMNet2/"
        "aimnet2_wb97m_d3_0.pt"
    )
    assert entry.metadata["reference_xc_functional"] == "wb97m"


def test_aimnet_model_card_uses_explicit_registry_metadata() -> None:
    """AIMNet2 model metadata should come from the explicit registry entry."""

    entry = get_known_artifact("aimnet2", "aimnet2")
    resolved = ResolvedArtifact(
        entry=entry,
        local_path=Path("/tmp/aimnet2_wb97m_d3_0.pt"),
        checkpoint=CheckpointInfo(
            identifier="aimnet2",
            url=entry.url,
            source="registry",
        ),
    )

    card = AIMNet2Potential._default_model_card(
        "aimnet2",
        resolved_artifact=resolved,
    )

    assert card.model_name == "aimnet2"
    assert card.reference_xc_functional == "wb97m"
    assert card.checkpoint == resolved.checkpoint


def test_builtin_mace_registry_is_reduced_to_two_models() -> None:
    """Built-in MACE registry entries should expose only the chosen MP and MPA models."""

    assert list_known_artifacts("mace") == [
        "mace-mp-0b3-medium",
        "mace-mpa-0-medium",
    ]

    mp = get_known_artifact("mace-mp-0b3", "mace")
    mp_canonical = get_known_artifact("mace-mp-0b3-medium", "mace")
    mpa = get_known_artifact("mace-mpa-0", "mace")
    mpa_canonical = get_known_artifact("mace-mpa-0-medium", "mace")

    assert mp is mp_canonical
    assert mp.metadata["upstream_name"] == "medium-0b3"
    assert (
        mp.url
        == "https://github.com/ACEsuit/mace-mp/releases/download/"
        "mace_mp_0b3/mace-mp-0b3-medium.model"
    )

    assert mpa is mpa_canonical
    assert mpa.metadata["upstream_name"] == "medium-mpa-0"
    assert (
        mpa.url
        == "https://github.com/ACEsuit/mace-mp/releases/download/"
        "mace_mpa_0/mace-mpa-0-medium.model"
    )

    for removed_name in (
        "medium-0b2",
        "small-0b2",
        "large-0b2",
        "mace-mp",
        "mace-mp-medium",
        "mace-mp-0b2",
        "mace-mp-0b2-medium",
        "mace-mp-0b2-small",
        "mace-mp-0b2-large",
        "mace-mpa-0b3",
        "mace-mpa-0b3-medium",
    ):
        with pytest.raises(KeyError):
            get_known_artifact(removed_name, "mace")
