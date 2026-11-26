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
"""Unit tests for variance-constrained semi-grand-canonical Monte Carlo."""

from __future__ import annotations

import math

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.mc import SGC, VCSGC
from nvalchemi.models.base import BaseModelMixin, ModelConfig
from nvalchemi.models.demo import DemoModel, DemoModelWrapper


class _SiteEnergyModel(torch.nn.Module, BaseModelMixin):
    """Non-interacting sites: E = sum_i eps[Z_i], so energy depends only on composition."""

    def __init__(self, site_energies: dict[int, float]) -> None:
        """Store one energy (eV) per atomic number."""
        super().__init__()
        self.site_energies = site_energies
        self.model_config = ModelConfig(outputs=frozenset({"energy"}))

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        """Return no embedding outputs."""
        return {}

    def compute_embeddings(self, data: Batch, **kwargs: object) -> Batch:
        """Return the input batch unchanged."""
        del kwargs
        return data

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        """Sum per-species site energies over each graph."""
        numbers = batch.atomic_numbers.reshape(-1)
        per_atom = torch.zeros(
            numbers.shape, dtype=batch.positions.dtype, device=numbers.device
        )
        for number, energy in self.site_energies.items():
            per_atom = torch.where(
                numbers == number, torch.full_like(per_atom, energy), per_atom
            )
        energy = torch.zeros(
            batch.num_graphs, dtype=per_atom.dtype, device=per_atom.device
        )
        energy.index_add_(0, batch.batch_idx.to(torch.long), per_atom)
        return {"energy": energy.unsqueeze(-1)}


def _batch(numbers: list[list[int]]) -> Batch:
    """Create one small graph per supplied sequence of atomic numbers."""
    data = [
        AtomicData(
            atomic_numbers=torch.tensor(values, dtype=torch.long),
            positions=torch.zeros(len(values), 3),
        )
        for values in numbers
    ]
    batch = Batch.from_data_list(data)
    batch.energy = torch.zeros(batch.num_graphs, 1)
    return batch


def _demo() -> DemoModelWrapper:
    """A fresh demo model for tests where the energy value is irrelevant."""
    return DemoModelWrapper(DemoModel())


def test_zero_kappa_reproduces_sgc_trajectory() -> None:
    """kappa = 0 is SGC with mu_B - mu_A = -phi, move for move under one seed."""
    model = _demo()
    start = [[1, 2, 1, 1, 2, 1, 2, 2], [2, 2, 1, 1, 1, 2, 1, 1]]
    vc_batch, sgc_batch = _batch(start), _batch(start)
    phi = 0.05
    vc = VCSGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        kappa=0.0,
        phi=phi,
        random_seed=11,
    )
    sgc = SGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: -phi},
        random_seed=11,
    )

    vc.run(vc_batch, n_steps=40)
    sgc.run(sgc_batch, n_steps=40)

    assert vc_batch.atomic_numbers.tolist() == sgc_batch.atomic_numbers.tolist()
    assert vc.stats.accepted == sgc.stats.accepted
    assert vc.stats.attempted == sgc.stats.attempted == 80


def test_constraint_term_is_the_exact_midpoint_form() -> None:
    """After a proposal, the term is dn * (phi + 2 kappa * (c_old + c_new) / 2)."""
    batch = _batch([[1, 1, 2, 2, 2]])
    sampler = VCSGC(
        model=_demo(),
        temperature=1000.0,
        species=[1, 2],
        kappa=0.7,
        phi=-0.3,
        random_seed=3,
    )

    sampler.pre_update(batch)

    dn = 1.0 if int(sampler._proposal_new[0]) == 2 else -1.0
    c_old, c_new = 3 / 5, (3 + dn) / 5
    expected = dn * (-0.3 + 2 * 0.7 * 0.5 * (c_old + c_new))
    assert sampler._chemical_delta(batch).item() == pytest.approx(expected, rel=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("eps_b", "reference"),
    [
        (0.05, 0.0),
        # An ML-potential-like per-element offset, cancelled by its reference: the net
        # per-flip cost equals the case above, so the sampled distribution must too.
        (-2.85, -2.9),
    ],
    ids=["no-offset", "offset-with-reference"],
)
def test_samples_the_exact_vcsgc_distribution(eps_b: float, reference: float) -> None:
    """For a composition-only energy the solute-count histogram is known exactly."""
    n_sites, temperature, kappa, phi = 10, 700.0, 0.4, -0.35
    eps = {1: 0.0, 2: eps_b}
    batch = _batch([[1] * n_sites])
    sampler = VCSGC(
        model=_SiteEnergyModel(eps),
        temperature=temperature,
        species=[1, 2],
        kappa=kappa,
        phi=phi,
        reference_exchange_potential=reference,
        random_seed=5,
    )

    beta = 1.0 / (KB_EV * temperature)
    log_w = []
    for n in range(n_sites + 1):
        c = n / n_sites
        log_w.append(
            math.lgamma(n_sites + 1)
            - math.lgamma(n + 1)
            - math.lgamma(n_sites - n + 1)
            - beta
            * (
                n * (eps[2] - eps[1])
                + n_sites * ((phi - reference) * c + kappa * c * c)
            )
        )
    top = max(log_w)
    exact = torch.tensor([math.exp(w - top) for w in log_w], dtype=torch.float64)
    exact /= exact.sum()

    sampler.run(batch, n_steps=500)  # burn-in
    counts = torch.zeros(n_sites + 1, dtype=torch.float64)
    for _ in range(6000):
        sampler.run(batch, n_steps=1)
        counts[int((batch.atomic_numbers == 2).sum())] += 1
    sampled = counts / counts.sum()

    total_variation = 0.5 * (sampled - exact).abs().sum().item()
    mean_exact = float((torch.arange(n_sites + 1) * exact).sum())
    mean_sampled = float((torch.arange(n_sites + 1) * sampled).sum())
    assert total_variation < 0.06
    assert mean_sampled == pytest.approx(mean_exact, abs=0.25)


def test_without_a_reference_an_energy_offset_drives_the_walker_to_one_end() -> None:
    """The trap the reference exists for: an eV-scale per-element offset beats the constraint."""
    batch = _batch([[1] * 10])
    sampler = VCSGC(
        model=_SiteEnergyModel({1: 0.0, 2: -2.85}),
        temperature=700.0,
        species=[1, 2],
        kappa=1.0,
        target_concentration=0.5,
        random_seed=5,
    )

    sampler.run(
        batch, n_steps=300
    )  # every site is proposed at least once with overwhelming probability

    assert batch.atomic_numbers.tolist() == [2] * 10


def test_reference_enters_the_constraint_term() -> None:
    """With a reference the term is dn * (phi - ref + 2 kappa * c_mid)."""
    batch = _batch([[1, 1, 2, 2, 2]])
    sampler = VCSGC(
        model=_demo(),
        temperature=1000.0,
        species=[1, 2],
        kappa=0.7,
        phi=-0.3,
        reference_exchange_potential=-2.87,
        random_seed=3,
    )

    sampler.pre_update(batch)

    dn = 1.0 if int(sampler._proposal_new[0]) == 2 else -1.0
    c_mid = 0.5 * (3 / 5 + (3 + dn) / 5)
    expected = dn * (-0.3 + 2.87 + 2 * 0.7 * c_mid)
    assert sampler._chemical_delta(batch).item() == pytest.approx(expected, rel=1e-6)


def test_zero_kappa_with_reference_reproduces_sgc_trajectory() -> None:
    """kappa = 0 with a reference is SGC with mu_B - mu_A = ref - phi."""
    model = _demo()
    start = [[1, 2, 1, 1, 2, 1, 2, 2], [2, 2, 1, 1, 1, 2, 1, 1]]
    vc_batch, sgc_batch = _batch(start), _batch(start)
    phi, reference = 0.05, -0.3
    vc = VCSGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        kappa=0.0,
        phi=phi,
        reference_exchange_potential=reference,
        random_seed=11,
    )
    sgc = SGC(
        model=model,
        temperature=1000.0,
        species=[1, 2],
        chemical_potentials={1: 0.0, 2: reference - phi},
        random_seed=11,
    )

    vc.run(vc_batch, n_steps=40)
    sgc.run(sgc_batch, n_steps=40)

    assert vc_batch.atomic_numbers.tolist() == sgc_batch.atomic_numbers.tolist()
    assert vc.stats.accepted == sgc.stats.accepted


def test_reference_centres_the_exchange_potential() -> None:
    """In target mode mu_B - mu_A = ref + 2 kappa (c0 - cbar): the reference at the target."""
    sampler = VCSGC(
        model=_demo(),
        temperature=700.0,
        species=[1, 2],
        kappa=2.0,
        target_concentration=0.25,
        reference_exchange_potential=-2.8727,
    )

    assert float(sampler.exchange_chemical_potential(0.25)) == pytest.approx(-2.8727)
    assert float(sampler.exchange_chemical_potential(0.2)) == pytest.approx(
        -2.8727 + 2 * 2.0 * 0.05
    )
    assert float(sampler.phi) == pytest.approx(
        -1.0
    )  # phi stays the excess over the reference


def test_mismatched_per_graph_parameter_lengths_are_rejected() -> None:
    """Per-graph phi and reference must describe the same graphs."""
    with pytest.raises(ValueError, match="equal lengths"):
        VCSGC(
            model=_demo(),
            temperature=1000.0,
            species=[1, 2],
            kappa=1.0,
            phi=torch.tensor([0.0, 0.1]),
            reference_exchange_potential=torch.tensor([-2.9, -2.9, -2.9]),
        )


def test_target_concentration_sets_phi_and_exchange_potential() -> None:
    """c0 sets phi = -2 kappa c0, and mu_B - mu_A = 2 kappa (c0 - cbar)."""
    sampler = VCSGC(
        model=_demo(),
        temperature=1000.0,
        species=[1, 2],
        kappa=2.0,
        target_concentration=0.25,
    )

    assert float(sampler.phi) == pytest.approx(-1.0)
    assert float(sampler.exchange_chemical_potential(0.2)) == pytest.approx(
        2 * 2.0 * (0.25 - 0.2)
    )


def test_per_graph_targets_drive_different_graphs() -> None:
    """A strong constraint pulls each single-site graph towards its own target."""
    batch = _batch([[1], [1]])
    sampler = VCSGC(
        model=_demo(),
        temperature=1000.0,
        species=[1, 2],
        kappa=1.0e6,
        target_concentration=torch.tensor([1.0, 0.0]),
        random_seed=7,
    )

    sampler.run(batch, n_steps=1)

    assert batch.atomic_numbers.tolist() == [2, 1]
    assert batch.mc_accepted.tolist() == [[True], [False]]


def test_concentration_counts_the_configured_species() -> None:
    """concentration() is per graph and follows concentration_species."""
    batch = _batch([[1, 2, 2, 2], [1, 1, 1, 2]])
    default = VCSGC(
        model=_demo(), temperature=1000.0, species=[1, 2], kappa=1.0, phi=0.0
    )
    flipped = VCSGC(
        model=_demo(),
        temperature=1000.0,
        species=[1, 2],
        kappa=1.0,
        phi=0.0,
        concentration_species=1,
    )

    assert default.concentration(batch).tolist() == pytest.approx([0.75, 0.25])
    assert flipped.concentration(batch).tolist() == pytest.approx([0.25, 0.75])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"species": [1, 2, 3], "kappa": 1.0, "phi": 0.0}, "binary"),
        ({"species": [1, 2], "kappa": 1.0}, "exactly one"),
        (
            {"species": [1, 2], "kappa": 1.0, "phi": 0.0, "target_concentration": 0.5},
            "exactly one",
        ),
        ({"species": [1, 2], "kappa": -1.0, "phi": 0.0}, "non-negative"),
        ({"species": [1, 2], "kappa": 0.0, "target_concentration": 0.5}, "positive"),
        ({"species": [1, 2], "kappa": 1.0, "target_concentration": 1.5}, r"\[0, 1\]"),
        (
            {"species": [1, 2], "kappa": 1.0, "phi": 0.0, "concentration_species": 3},
            "concentration_species",
        ),
    ],
)
def test_invalid_configuration_is_rejected(kwargs: dict, match: str) -> None:
    """Configuration errors surface at construction, not mid-run."""
    with pytest.raises(ValueError, match=match):
        VCSGC(model=_demo(), temperature=1000.0, **kwargs)


def test_per_graph_parameter_shape_is_checked() -> None:
    """A per-graph parameter must match the number of graphs exactly."""
    batch = _batch([[1], [1]])
    sampler = VCSGC(
        model=_demo(),
        temperature=1000.0,
        species=[1, 2],
        kappa=torch.tensor([1.0]),
        phi=0.0,
    )

    with pytest.raises(ValueError, match="kappa tensors"):
        sampler.run(batch, n_steps=1)
