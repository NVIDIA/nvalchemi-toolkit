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

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models import (
    CompositeCalculator,
    EnergyDerivativesStep,
    Potential,
    PotentialCard,
)
from nvalchemi.models.base import ForwardContext
from nvalchemi.models.dftd3 import (
    DFTD3Config,
    DFTD3Potential,
    load_dftd3_params,
)
from nvalchemi.models.ewald import EwaldCoulombConfig, EwaldCoulombPotential
from nvalchemi.models.lj import LennardJonesConfig, LennardJonesPotential
from nvalchemi.models.metadata import (
    ATOMIC_CHARGES,
    DISPERSION,
    ELECTROSTATICS,
    PAIRWISE,
    REPULSION,
    SHORT_RANGE,
    CheckpointInfo,
    PhysicalTerm,
)
from nvalchemi.models.neighbors import (
    NeighborListBuilder,
    NeighborListBuilderConfig,
    neighbor_result_key,
)
from nvalchemi.models.pme import PMEConfig, PMEPotential
from nvalchemi.models.registry import KnownArtifactEntry, ResolvedArtifact
from nvalchemi.models.results import CalculatorResults
from nvalchemi.models.utils import aggregate_per_system_energy

_AUTOGRAD_ENERGY_CARD = PotentialCard(
    required_inputs=frozenset({"positions"}),
    result_keys=frozenset({"energies"}),
    default_result_keys=frozenset({"energies"}),
    additive_result_keys=frozenset({"energies"}),
    gradient_setup_targets=frozenset({"positions"}),
)


class _AutogradEnergyPotential(Potential):
    """Tiny energy-only potential used to seed autograd force tests."""

    card = _AUTOGRAD_ENERGY_CARD

    def __init__(self) -> None:
        super().__init__(name="autograd_energy")

    def compute(self, batch: Batch, ctx: ForwardContext) -> CalculatorResults:
        """Return a simple position-dependent energy."""

        positions = self.require_input(batch, "positions", ctx)
        per_atom_energies = positions.pow(2).sum(dim=-1)
        energies = aggregate_per_system_energy(
            per_atom_energies,
            batch.batch,
            batch.num_graphs,
        )
        return self.build_results(ctx, energies=energies)


def _make_nonperiodic_batch() -> Batch:
    """Build a small non-periodic batch with charges."""

    data = [
        AtomicData(
            positions=torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0]], dtype=torch.float32),
            atomic_numbers=torch.tensor([8, 1], dtype=torch.long),
            node_charges=torch.tensor([[0.2], [-0.2]], dtype=torch.float32),
        )
    ]
    return Batch.from_data_list(data)


def _make_periodic_batch() -> Batch:
    """Build a small periodic batch with charges."""

    data = [
        AtomicData(
            positions=torch.tensor([[0.1, 0.2, 0.0], [1.1, 0.2, 0.0]], dtype=torch.float32),
            atomic_numbers=torch.tensor([8, 1], dtype=torch.long),
            node_charges=torch.tensor([[0.3], [-0.3]], dtype=torch.float32),
            cell=(8.0 * torch.eye(3, dtype=torch.float32)).unsqueeze(0),
            pbc=torch.tensor([[True, True, True]]),
        )
    ]
    return Batch.from_data_list(data)


def _matrix_neighbor_results(batch: Batch, *, name: str = "default") -> CalculatorResults:
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


def test_neighbor_builder_reuses_existing_results(monkeypatch: pytest.MonkeyPatch) -> None:
    """NeighborListBuilder should reuse matching results when configured."""

    batch = _make_nonperiodic_batch()
    builder = NeighborListBuilder(
        NeighborListBuilderConfig(
            cutoff=4.0,
            format="matrix",
            reuse_if_available=True,
        )
    )
    existing = _matrix_neighbor_results(batch)

    def _should_not_build(**kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("neighbor_list backend should not run when reuse succeeds")

    monkeypatch.setattr("nvalchemi.models.neighbors.neighbor_list", _should_not_build)
    outputs = builder(batch, results=existing, outputs=builder.profile.result_keys)

    assert torch.equal(
        outputs[neighbor_result_key("default", "neighbor_matrix")],
        existing[neighbor_result_key("default", "neighbor_matrix")],
    )


def test_lennard_jones_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """LennardJonesPotential should expose direct energies and forces."""

    batch = _make_nonperiodic_batch()
    results = _matrix_neighbor_results(batch)

    def _fill_lj(**kwargs):  # type: ignore[no-untyped-def]
        kwargs["atomic_energies"].fill_(1.0)
        kwargs["forces"].zero_()

    monkeypatch.setattr("nvalchemi.models.lj.lj_energy_forces_batch_into", _fill_lj)
    potential = LennardJonesPotential(
        LennardJonesConfig(epsilon=0.2, sigma=3.5, cutoff=6.0)
    )
    outputs = potential(batch, results=results, outputs={"energies", "forces"})

    assert tuple(outputs["energies"].shape) == (batch.num_graphs, 1)
    assert tuple(outputs["forces"].shape) == tuple(batch.positions.shape)
    assert potential.model_card.provided_terms == (
        PhysicalTerm(kind=SHORT_RANGE, variant=REPULSION),
    )


def test_dftd3_smoke_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """DFTD3Potential should expose direct energies/forces and functional metadata."""

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
            torch.zeros(positions.shape[0], dtype=positions.dtype, device=positions.device),
        )

    monkeypatch.setattr("nvalchemi.models.dftd3.dftd3", _fake_dftd3)
    potential = DFTD3Potential(
        DFTD3Config(functional="pbe", param_path=param_path, auto_download=False)
    )
    outputs = potential(batch, results=results, outputs={"energies", "forces"})

    assert tuple(outputs["energies"].shape) == (batch.num_graphs, 1)
    assert tuple(outputs["forces"].shape) == tuple(batch.positions.shape)
    assert potential.model_card.reference_xc_functional == "pbe"
    assert potential.model_card.provided_terms == (
        PhysicalTerm(kind=DISPERSION, variant=PAIRWISE),
    )


def test_dftd3_loader_downloads_and_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pytest.TempPathFactory,
) -> None:
    """DFT-D3 parameter loader should copy the registry artifact to the requested path."""

    cache_path = tmp_path / "downloaded_d3.pt"
    resolved = ResolvedArtifact(
        entry=KnownArtifactEntry(
            name="dftd3_parameters",
            family="dftd3",
            metadata={"model_name": "dftd3_parameters"},
        ),
        local_path=tmp_path / "registry_d3.pt",
        checkpoint=CheckpointInfo(
            identifier="dftd3_parameters",
            source="registry",
        ),
    )
    torch.save(_fake_d3_state(), resolved.local_path)
    monkeypatch.setattr(
        "nvalchemi.models.dftd3.resolve_known_artifact",
        lambda *args, **kwargs: resolved,
    )

    params = load_dftd3_params(cache_path, auto_download=True)

    assert cache_path.exists()
    assert tuple(params.rcov.shape) == (95,)


def test_dftd3_neighbor_list_builder_config_matches_contract(
    tmp_path: pytest.TempPathFactory,
) -> None:
    """DFT-D3 should emit a ready-to-use matrix neighbor builder config."""

    param_path = tmp_path / "d3.pt"
    torch.save(_fake_d3_state(), param_path)
    potential = DFTD3Potential(
        DFTD3Config(
            functional="pbe",
            param_path=param_path,
            auto_download=False,
            neighbor_list_name="dispersion",
        )
    )

    config = potential.neighbor_list_builder_config()

    assert config is not None
    assert config.neighbor_list_name == "dispersion"
    assert config.format == "matrix"
    assert config.cutoff == pytest.approx(potential.profile.neighbor_requirement.cutoff)
    assert config.reuse_if_available is True


def test_ewald_reuses_k_vectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ewald should reuse cached k_vectors when configured."""

    batch = _make_periodic_batch()
    results = _matrix_neighbor_results(batch)
    cached_k = torch.ones((2, 3), dtype=torch.float32)
    results["k_vectors"] = cached_k

    monkeypatch.setattr(
        "nvalchemi.models.ewald.generate_k_vectors_ewald_summation",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should reuse k_vectors")),
    )
    monkeypatch.setattr(
        "nvalchemi.models.ewald.ewald_real_space",
        lambda **kwargs: torch.zeros(batch.num_nodes, dtype=batch.positions.dtype),
    )
    monkeypatch.setattr(
        "nvalchemi.models.ewald.ewald_reciprocal_space",
        lambda **kwargs: torch.zeros(batch.num_nodes, dtype=batch.positions.dtype),
    )

    potential = EwaldCoulombPotential(
        EwaldCoulombConfig(cutoff=6.0, alpha=0.5, reuse_if_available=True)
    )
    outputs = potential(batch, results=results, outputs={"energies", "k_vectors"})

    assert tuple(outputs["energies"].shape) == (batch.num_graphs, 1)
    assert outputs["k_vectors"] is cached_k
    assert potential.model_card.provided_terms == (
        PhysicalTerm(kind=ELECTROSTATICS, variant=ATOMIC_CHARGES),
    )


def test_pme_reuses_reciprocal_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """PME should reuse cached reciprocal-space tensors when configured."""

    batch = _make_periodic_batch()
    results = _matrix_neighbor_results(batch)
    cached_k = torch.ones((2, 3), dtype=torch.float32)
    cached_k2 = torch.ones((2,), dtype=torch.float32)
    results["k_vectors"] = cached_k
    results["k_squared"] = cached_k2

    monkeypatch.setattr(
        "nvalchemi.models.pme.generate_k_vectors_pme",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("should reuse reciprocal tensors")),
    )
    monkeypatch.setattr(
        "nvalchemi.models.pme.particle_mesh_ewald",
        lambda **kwargs: torch.zeros(batch.num_nodes, dtype=batch.positions.dtype),
    )

    potential = PMEPotential(
        PMEConfig(
            cutoff=6.0,
            alpha=0.5,
            mesh_dimensions=(4, 4, 4),
            reuse_if_available=True,
        )
    )
    outputs = potential(
        batch,
        results=results,
        outputs={"energies", "k_vectors", "k_squared"},
    )

    assert tuple(outputs["energies"].shape) == (batch.num_graphs, 1)
    assert outputs["k_vectors"] is cached_k
    assert outputs["k_squared"] is cached_k2


def test_pme_requires_periodic_inputs() -> None:
    """PME should fail early when periodic inputs are missing."""

    batch = _make_nonperiodic_batch()
    results = _matrix_neighbor_results(batch)
    potential = PMEPotential(PMEConfig(cutoff=6.0, mesh_dimensions=(4, 4, 4), alpha=0.5))

    with pytest.raises(ValueError, match="requires periodic inputs"):
        potential(batch, results=results, outputs={"energies"})


def test_pme_direct_rejects_differentiable_charges(monkeypatch: pytest.MonkeyPatch) -> None:
    """PME direct mode should reject charge-coupled usage."""

    batch = _make_periodic_batch()
    batch.node_charges.requires_grad_(True)
    results = _matrix_neighbor_results(batch)
    monkeypatch.setattr(
        "nvalchemi.models.pme.generate_k_vectors_pme",
        lambda *args, **kwargs: (
            torch.ones((2, 3), dtype=batch.positions.dtype),
            torch.ones((2,), dtype=batch.positions.dtype),
        ),
    )
    monkeypatch.setattr(
        "nvalchemi.models.pme.particle_mesh_ewald",
        lambda **kwargs: torch.zeros(batch.num_nodes, dtype=batch.positions.dtype),
    )

    potential = PMEPotential(
        PMEConfig(
            cutoff=6.0,
            alpha=0.5,
            mesh_dimensions=(4, 4, 4),
            derivative_mode="direct",
        )
    )

    with pytest.raises(ValueError, match="fixed charges"):
        potential(batch, results=results, outputs={"energies"})


def test_ewald_direct_rejects_differentiable_charges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ewald direct mode should reject charge-coupled usage."""

    batch = _make_periodic_batch()
    batch.node_charges.requires_grad_(True)
    results = _matrix_neighbor_results(batch)
    monkeypatch.setattr(
        "nvalchemi.models.ewald.generate_k_vectors_ewald_summation",
        lambda *args, **kwargs: torch.ones((2, 3), dtype=batch.positions.dtype),
    )
    monkeypatch.setattr(
        "nvalchemi.models.ewald.ewald_real_space",
        lambda **kwargs: torch.zeros(batch.num_nodes, dtype=batch.positions.dtype),
    )
    monkeypatch.setattr(
        "nvalchemi.models.ewald.ewald_reciprocal_space",
        lambda **kwargs: torch.zeros(batch.num_nodes, dtype=batch.positions.dtype),
    )

    potential = EwaldCoulombPotential(
        EwaldCoulombConfig(cutoff=6.0, alpha=0.5, derivative_mode="direct")
    )

    with pytest.raises(ValueError, match="fixed charges"):
        potential(batch, results=results, outputs={"energies"})


def test_pme_autograd_profile_is_energy_only() -> None:
    """PME autograd mode should participate through energy only."""

    potential = PMEPotential(
        PMEConfig(
            cutoff=6.0,
            alpha=0.5,
            mesh_dimensions=(4, 4, 4),
            derivative_mode="autograd",
        )
    )

    assert potential.profile.result_keys == frozenset({"energies", "k_vectors", "k_squared"})
    assert potential.profile.additive_result_keys == frozenset({"energies"})
    assert potential.profile.gradient_setup_targets == frozenset({"positions", "cell_scaling"})


def test_ewald_autograd_profile_is_energy_only() -> None:
    """Ewald autograd mode should participate through energy only."""

    potential = EwaldCoulombPotential(
        EwaldCoulombConfig(cutoff=6.0, alpha=0.5, derivative_mode="autograd")
    )

    assert potential.profile.result_keys == frozenset({"energies", "k_vectors"})
    assert potential.profile.additive_result_keys == frozenset({"energies"})
    assert potential.profile.gradient_setup_targets == frozenset({"positions", "cell_scaling"})


def test_pme_ignores_cached_reciprocal_state_for_stress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PME should regenerate reciprocal state for stress paths."""

    batch = _make_periodic_batch()
    results = _matrix_neighbor_results(batch)
    cached_k = torch.ones((2, 3), dtype=torch.float32)
    cached_k2 = torch.ones((2,), dtype=torch.float32)
    new_k = torch.full((3, 3), 7.0, dtype=torch.float32)
    new_k2 = torch.full((3,), 9.0, dtype=torch.float32)
    results["k_vectors"] = cached_k
    results["k_squared"] = cached_k2

    monkeypatch.setattr(
        "nvalchemi.models.pme.generate_k_vectors_pme",
        lambda *args, **kwargs: (new_k, new_k2),
    )
    monkeypatch.setattr(
        "nvalchemi.models.pme.particle_mesh_ewald",
        lambda **kwargs: (
            kwargs["positions"].sum(dim=-1),
            torch.zeros_like(kwargs["positions"]),
            torch.zeros((batch.num_graphs, 3, 3), dtype=kwargs["positions"].dtype),
        ),
    )

    potential = PMEPotential(
        PMEConfig(
            cutoff=6.0,
            alpha=0.5,
            mesh_dimensions=(4, 4, 4),
            reuse_if_available=True,
            derivative_mode="direct",
        )
    )

    with pytest.warns(UserWarning, match="ignores cached reciprocal tensors"):
        outputs = potential(
            batch,
            results=results,
            outputs={"energies", "stresses", "k_vectors", "k_squared"},
        )

    assert outputs["k_vectors"] is new_k
    assert outputs["k_squared"] is new_k2


def test_ewald_ignores_cached_k_vectors_for_stress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ewald should regenerate k-vectors for stress paths."""

    batch = _make_periodic_batch()
    results = _matrix_neighbor_results(batch)
    cached_k = torch.ones((2, 3), dtype=torch.float32)
    new_k = torch.full((3, 3), 5.0, dtype=torch.float32)
    results["k_vectors"] = cached_k

    monkeypatch.setattr(
        "nvalchemi.models.ewald.generate_k_vectors_ewald_summation",
        lambda *args, **kwargs: new_k,
    )
    monkeypatch.setattr(
        "nvalchemi.models.ewald.ewald_real_space",
        lambda **kwargs: (
            kwargs["positions"].sum(dim=-1),
            torch.zeros_like(kwargs["positions"]),
            torch.zeros((batch.num_graphs, 3, 3), dtype=kwargs["positions"].dtype),
        ),
    )
    monkeypatch.setattr(
        "nvalchemi.models.ewald.ewald_reciprocal_space",
        lambda **kwargs: (
            torch.zeros(batch.num_nodes, dtype=kwargs["positions"].dtype),
            torch.zeros_like(kwargs["positions"]),
            torch.zeros((batch.num_graphs, 3, 3), dtype=kwargs["positions"].dtype),
        ),
    )

    potential = EwaldCoulombPotential(
        EwaldCoulombConfig(
            cutoff=6.0,
            alpha=0.5,
            reuse_if_available=True,
            derivative_mode="direct",
        )
    )

    with pytest.warns(UserWarning, match="ignores cached k_vectors"):
        outputs = potential(
            batch,
            results=results,
            outputs={"energies", "stresses", "k_vectors"},
        )

    assert outputs["k_vectors"] is new_k


def test_pme_direct_does_not_double_count_with_energy_derivatives(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct-mode PME should not leak extra autograd force into EDS."""

    batch = _make_periodic_batch()
    results = _matrix_neighbor_results(batch)
    monkeypatch.setattr(
        "nvalchemi.models.pme.generate_k_vectors_pme",
        lambda *args, **kwargs: (
            torch.ones((2, 3), dtype=batch.positions.dtype),
            torch.ones((2,), dtype=batch.positions.dtype),
        ),
    )
    monkeypatch.setattr(
        "nvalchemi.models.pme.particle_mesh_ewald",
        lambda **kwargs: (
            kwargs["positions"].sum(dim=-1),
            torch.full_like(kwargs["positions"], 3.0),
        ),
    )

    calculator = CompositeCalculator(
        _AutogradEnergyPotential(),
        PMEPotential(
            PMEConfig(
                cutoff=6.0,
                alpha=0.5,
                mesh_dimensions=(4, 4, 4),
                derivative_mode="direct",
            )
        ),
        EnergyDerivativesStep(),
        outputs={"forces"},
    )

    outputs = calculator(batch, results=results)

    expected_autograd = -2.0 * batch.positions
    expected = expected_autograd + torch.full_like(
        batch.positions,
        3.0 * 14.3996,
    )
    assert torch.allclose(outputs["forces"], expected)


# ---------------------------------------------------------------------------
# Hybrid config / kwargs construction tests
# ---------------------------------------------------------------------------


class TestHybridKwargsConstruction:
    """Verify that potentials and builders accept config, kwargs, or both."""

    def test_lj_kwargs_only(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """LJ constructed from flat kwargs should match config-based construction."""

        from_kwargs = LennardJonesPotential(epsilon=0.2, sigma=3.5, cutoff=6.0)
        from_config = LennardJonesPotential(
            LennardJonesConfig(epsilon=0.2, sigma=3.5, cutoff=6.0)
        )
        assert from_kwargs.config == from_config.config

    def test_lj_config_with_override(self) -> None:
        """Passing config + kwargs should overlay the kwargs on the config."""

        base = LennardJonesConfig(epsilon=0.2, sigma=3.5, cutoff=6.0)
        potential = LennardJonesPotential(config=base, cutoff=8.0)
        assert potential.config.cutoff == 8.0
        assert potential.config.epsilon == 0.2

    def test_ewald_kwargs_only(self) -> None:
        """Ewald from flat kwargs should produce equivalent config."""

        from_kwargs = EwaldCoulombPotential(cutoff=6.0, alpha=0.5)
        from_config = EwaldCoulombPotential(
            EwaldCoulombConfig(cutoff=6.0, alpha=0.5)
        )
        assert from_kwargs.config == from_config.config

    def test_pme_kwargs_only(self) -> None:
        """PME from flat kwargs should produce equivalent config."""

        from_kwargs = PMEPotential(cutoff=6.0, alpha=0.5, mesh_dimensions=(4, 4, 4))
        from_config = PMEPotential(
            PMEConfig(cutoff=6.0, alpha=0.5, mesh_dimensions=(4, 4, 4))
        )
        assert from_kwargs.config == from_config.config

    def test_pme_config_with_override(self) -> None:
        """PME config + override should produce the merged result."""

        base = PMEConfig(cutoff=6.0, alpha=0.5)
        potential = PMEPotential(config=base, cutoff=12.0, derivative_mode="autograd")
        assert potential.config.cutoff == 12.0
        assert potential.config.derivative_mode == "autograd"
        assert potential.config.alpha == 0.5

    def test_dftd3_kwargs_only(self, tmp_path: pytest.TempPathFactory) -> None:
        """DFT-D3 from flat kwargs should produce equivalent config."""

        param_path = tmp_path / "d3.pt"
        torch.save(_fake_d3_state(), param_path)
        from_kwargs = DFTD3Potential(
            functional="pbe", param_path=param_path, auto_download=False
        )
        from_config = DFTD3Potential(
            DFTD3Config(functional="pbe", param_path=param_path, auto_download=False)
        )
        assert from_kwargs.config == from_config.config

    def test_neighbor_builder_kwargs_only(self) -> None:
        """NeighborListBuilder from flat kwargs should work."""

        from_kwargs = NeighborListBuilder(cutoff=4.0, format="matrix")
        from_config = NeighborListBuilder(
            NeighborListBuilderConfig(cutoff=4.0, format="matrix")
        )
        assert from_kwargs.config == from_config.config

    def test_neighbor_builder_config_with_override(self) -> None:
        """NeighborListBuilder config + override should merge."""

        base = NeighborListBuilderConfig(cutoff=4.0, format="coo")
        builder = NeighborListBuilder(config=base, cutoff=8.0)
        assert builder.config.cutoff == 8.0
        assert builder.config.format == "coo"

    def test_config_passthrough_unchanged(self) -> None:
        """Passing only config (no kwargs) should use it as-is."""

        config = LennardJonesConfig(epsilon=1.0, sigma=2.0, cutoff=5.0, half_list=True)
        potential = LennardJonesPotential(config=config)
        assert potential.config is config
