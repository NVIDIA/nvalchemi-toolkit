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
"""Shared fixtures and builders for ``test/training/``.

Fixtures are pure-value — they return built objects, not callables.
Tests that need non-default variants either import the ``_build_*``
helpers directly or construct their objects inline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.training import EnergyMSELoss, ForceMSELoss
from nvalchemi.training.optimizers import OptimizerConfig
from nvalchemi.training.strategy import TrainingStrategy

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin


@pytest.fixture(autouse=True)
def _seed_torch() -> None:
    """Seed ``torch`` (and CUDA, when visible) to ``0`` before every test."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)


def _build_atomic_data(n_atoms: int = 3, seed: int = 0) -> AtomicData:
    g = torch.Generator().manual_seed(seed)
    positions = torch.randn(n_atoms, 3, generator=g)
    atomic_numbers = torch.randint(1, 10, (n_atoms,), dtype=torch.long, generator=g)
    energy = torch.randn(1, 1, generator=g)
    forces = torch.randn(n_atoms, 3, generator=g)
    return AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        atomic_masses=torch.ones(n_atoms),
        energy=energy,
        forces=forces,
    )


def _build_batch(n_systems: int = 2, n_atoms_each: int = 3, seed: int = 0) -> Batch:
    data_list = [
        _build_atomic_data(n_atoms_each, seed=seed + i) for i in range(n_systems)
    ]
    return Batch.from_data_list(data_list)


def _build_dataset(
    n_batches: int = 3,
    n_systems: int = 2,
    n_atoms_each: int = 3,
    base_seed: int = 100,
) -> list[Batch]:
    return [
        _build_batch(
            n_systems=n_systems,
            n_atoms_each=n_atoms_each,
            seed=base_seed + i * 10,
        )
        for i in range(n_batches)
    ]


def _build_demo_model() -> Any:
    from nvalchemi.models.demo import DemoModel, DemoModelWrapper

    torch.manual_seed(0)
    return DemoModelWrapper(DemoModel(num_atom_types=20, hidden_dim=8))


def _build_adam_optimizer_configs(
    lr: float = 1e-3,
) -> dict[str, list[OptimizerConfig]]:
    return {
        "main": [
            OptimizerConfig(
                optimizer_cls=torch.optim.Adam,
                optimizer_kwargs={"lr": lr},
            )
        ]
    }


def _build_baseline_strategy_kwargs(
    models: BaseModelMixin | dict[str, BaseModelMixin] | None = None,
) -> dict[str, Any]:
    # Import locally so identity is preserved for spec round-trip tests.
    from test.training.test_strategy import demo_training_fn

    if models is None:
        models = _build_demo_model()
    return {
        "models": models,
        "optimizer_configs": OptimizerConfig(optimizer_cls=torch.optim.Adam),
        "num_epochs": 1,
        "training_fn": demo_training_fn,
        "loss_fn": EnergyMSELoss() + ForceMSELoss(normalize_by_atom_count=True),
    }


@pytest.fixture
def atomic_data() -> AtomicData:
    """Return a default :class:`AtomicData` — 3 atoms, ``seed=0``."""
    return _build_atomic_data()


@pytest.fixture
def batch() -> Batch:
    """Return a default :class:`Batch` — 2 systems, 3 atoms each, ``seed=0``."""
    return _build_batch()


@pytest.fixture
def dataset() -> list[Batch]:
    """Return a default dataset of 3 batches (``base_seed=100``)."""
    return _build_dataset()


@pytest.fixture
def demo_model() -> Any:
    """Return a freshly-seeded :class:`DemoModelWrapper`."""
    return _build_demo_model()


@pytest.fixture
def adam_optimizer_configs() -> dict[str, list[OptimizerConfig]]:
    """Return a default Adam :class:`OptimizerConfig` mapping keyed by ``main``."""
    return _build_adam_optimizer_configs()


@pytest.fixture
def baseline_strategy_kwargs(demo_model: Any) -> dict[str, Any]:
    """Return default kwargs suitable for ``TrainingStrategy(**kwargs)``."""
    return _build_baseline_strategy_kwargs(models=demo_model)


@pytest.fixture
def strategy(baseline_strategy_kwargs: dict[str, Any]) -> TrainingStrategy:
    """Return a default :class:`TrainingStrategy` built from ``baseline_strategy_kwargs``."""
    return TrainingStrategy(**baseline_strategy_kwargs)
