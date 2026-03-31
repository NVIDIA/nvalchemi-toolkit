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

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import (
    DFTD3Potential,
    EwaldCoulombPotential,
    KnownArtifactEntry,
    PMEPotential,
    ResolvedArtifact,
    resolve_known_artifact,
)
from nvalchemi.models.metadata import CheckpointInfo
from nvalchemi.models.neighbors import neighbor_result_key
from nvalchemi.models.results import CalculatorResults


def _make_nonperiodic_batch() -> Batch:
    """Build a small non-periodic batch with charges."""

    data = [
        AtomicData(
            positions=torch.tensor(
                [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float32
            ),
            atomic_numbers=torch.tensor([8, 1], dtype=torch.long),
            node_charges=torch.tensor([[0.2], [-0.2]], dtype=torch.float32),
        )
    ]
    return Batch.from_data_list(data)


def _make_periodic_batch() -> Batch:
    """Build a small periodic batch with charges."""

    data = [
        AtomicData(
            positions=torch.tensor(
                [[0.1, 0.2, 0.0], [1.1, 0.2, 0.0]], dtype=torch.float32
            ),
            atomic_numbers=torch.tensor([8, 1], dtype=torch.long),
            node_charges=torch.tensor([[0.3], [-0.3]], dtype=torch.float32),
            cell=(8.0 * torch.eye(3, dtype=torch.float32)).unsqueeze(0),
            pbc=torch.tensor([[True, True, True]]),
        )
    ]
    return Batch.from_data_list(data)


def _matrix_neighbor_results(
    batch: Batch, *, name: str = "default"
) -> CalculatorResults:
    """Return a minimal matrix-neighbor result set."""

    num_nodes = batch.num_nodes
    neighbor_matrix = torch.full((num_nodes, 2), -1, dtype=torch.int32)
    if num_nodes >= 2:
        neighbor_matrix[0, 0] = 1
        neighbor_matrix[1, 0] = 0
    num_neighbors = torch.tensor([1] * num_nodes, dtype=torch.int32)
    neighbor_shifts = torch.zeros((num_nodes, 2, 3), dtype=torch.int32)
    return CalculatorResults(
        {
            neighbor_result_key(name, "neighbor_matrix"): neighbor_matrix,
            neighbor_result_key(name, "num_neighbors"): num_neighbors,
            neighbor_result_key(name, "neighbor_shifts"): neighbor_shifts,
        }
    )


def _fake_d3_state() -> dict[str, torch.Tensor]:
    """Return a minimal DFT-D3 parameter state dict."""

    return {
        "rcov": torch.zeros(95, dtype=torch.float32),
        "r4r2": torch.zeros(95, dtype=torch.float32),
        "c6ab": torch.zeros((95, 95, 5, 5), dtype=torch.float32),
        "cn_ref": torch.zeros((95, 95, 5, 5), dtype=torch.float32),
    }


def test_dftd3_root_api_smoke_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """DFT-D3 should expose direct energies/forces through the root package."""

    batch = _make_nonperiodic_batch()
    results = _matrix_neighbor_results(batch)
    param_path = tmp_path / "d3.pt"
    torch.save(_fake_d3_state(), param_path)

    def _fake_dftd3(**kwargs):  # type: ignore[no-untyped-def]
        num_systems = kwargs["num_systems"]
        positions = kwargs["positions"]
        return (
            torch.ones(num_systems, dtype=positions.dtype, device=positions.device),
            torch.zeros_like(positions),
            torch.zeros(
                positions.shape[0], dtype=positions.dtype, device=positions.device
            ),
        )

    monkeypatch.setattr("nvalchemi.models.dftd3.dftd3", _fake_dftd3)
    potential = DFTD3Potential(
        functional="pbe", param_path=param_path, auto_download=False
    )
    outputs = potential(batch, results=results, outputs={"energies", "forces"})

    assert tuple(outputs["energies"].shape) == (batch.num_graphs, 1)
    assert tuple(outputs["forces"].shape) == tuple(batch.positions.shape)
    assert potential.model_card.reference_xc_functional == "pbe"


def test_pme_direct_mode_rejects_differentiable_charges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct-mode PME should reject charge-coupled force paths."""

    batch = _make_periodic_batch()
    batch.node_charges = batch.node_charges.clone().requires_grad_(True)
    results = _matrix_neighbor_results(batch)

    monkeypatch.setattr(
        "nvalchemi.models.pme.generate_k_vectors_pme",
        lambda **kwargs: (torch.zeros((1, 3)), torch.ones(1)),
    )
    monkeypatch.setattr(
        "nvalchemi.models.pme.particle_mesh_ewald",
        lambda **kwargs: (
            torch.zeros(kwargs["cell"].shape[0], dtype=kwargs["positions"].dtype),
            torch.zeros_like(kwargs["positions"]),
            torch.zeros(kwargs["cell"].shape[0], 3, 3, dtype=kwargs["positions"].dtype),
        ),
    )

    potential = PMEPotential(cutoff=6.0, alpha=0.5, derivative_mode="direct")
    with pytest.raises(ValueError, match="fixed charges"):
        potential(batch, results=results, outputs={"energies", "forces"})


def test_ewald_autograd_mode_participates_in_gradients() -> None:
    """Autograd-mode Ewald should advertise energy participation only."""

    potential = EwaldCoulombPotential(cutoff=6.0, alpha=0.5, derivative_mode="autograd")

    assert "forces" not in potential.profile.result_keys
    assert "stresses" not in potential.profile.result_keys
    assert potential.profile.gradient_setup_targets == frozenset(
        {"positions", "cell_scaling"}
    )


def test_registry_helpers_are_available_from_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The root package should expose registry-backed artifact resolution."""

    raw_payload = b"registry payload"
    sha256 = hashlib.sha256(raw_payload).hexdigest()

    def _fake_urlretrieve(url: str, filename: str | Path) -> tuple[str, None]:
        del url
        Path(filename).write_bytes(raw_payload)
        return str(filename), None

    monkeypatch.setattr("urllib.request.urlretrieve", _fake_urlretrieve)
    entry = KnownArtifactEntry(
        name="root-registry-artifact",
        family="test",
        url="https://example.invalid/raw.bin",
        sha256=sha256,
        filename="raw.bin",
    )

    from nvalchemi.models import register_known_artifact

    register_known_artifact(entry)
    resolved = resolve_known_artifact(
        "root-registry-artifact",
        family="test",
        cache_dir=tmp_path,
    )

    assert isinstance(resolved, ResolvedArtifact)
    assert resolved.local_path.read_bytes() == raw_payload
    assert resolved.checkpoint == CheckpointInfo(
        identifier="root-registry-artifact",
        url="https://example.invalid/raw.bin",
        sha256=sha256,
        source="registry",
    )
