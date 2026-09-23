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
"""Demo generative models for testing and debugging.

The generative counterpart to :mod:`nvalchemi.models.demo`: minimal,
self-contained placeholders that satisfy the
:class:`~nvalchemi.models.gen.base.GenerativeModelMixin` contract and run
through the :class:`~nvalchemi.gen.generator.AtomisticGenerator` with no external
weights or optional dependencies. Sampling procedures that own these models
live with the workflows that use them — see the test suite and the examples.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict
from torch import nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.gen.base import GenerativeModelConfig, GenerativeModelMixin

__all__ = [
    "DemoDiffusionModel",
    "DemoGANModel",
]


class DemoGANModel(nn.Module, GenerativeModelMixin):
    """Minimal GAN-side demo: a latent draw decoded to a point cloud.

    The generative analogue of :class:`~nvalchemi.models.demo.DemoModel` — a
    placeholder for testing and debugging generative workflows. ``forward``
    follows the mixin convention (``forward(data, *, x)``) and decodes the
    latent ``x`` to flat positions.
    """

    def __init__(
        self, num_atoms: int = 3, latent_dim: int = 4, hidden: int = 32
    ) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.latent_dim = latent_dim
        self.hidden = hidden
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_atoms * 3),
        )
        self.model_config = GenerativeModelConfig(
            supports_variable_atoms=False,
            required_inputs=frozenset(),
            outputs=frozenset({"positions", "atomic_numbers"}),
        )

    def forward(self, data: Any, *, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Decode a latent draw ``x`` of shape ``(B, latent_dim)``."""
        del data, kwargs
        return self.decoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent ``z`` to positions of shape ``(B, num_atoms, 3)``."""
        return self.decoder(z).reshape(-1, self.num_atoms, 3)

    def to_batch(self, sample: TensorDict, cond_batch: Batch | None = None) -> Batch:
        """Materialize the sample: one point-cloud graph per draw."""
        del cond_batch
        positions = sample["x1"].reshape(-1, self.num_atoms, 3)
        numbers = positions.new_full((self.num_atoms,), 6, dtype=torch.long)
        return Batch.from_data_list(
            [AtomicData(positions=p, atomic_numbers=numbers) for p in positions]
        )


class DemoDiffusionModel(nn.Module, GenerativeModelMixin):
    """Minimal diffusion-side demo: an x0-predictor over point clouds.

    ``forward`` follows the PhysicsNeMo calling convention —
    ``forward(x, sigma)`` predicts clean positions from noisy ones — so the
    model slots directly into ``physicsnemo.diffusion`` preconditioners and
    samplers (see the generative user guide).
    """

    def __init__(self, num_atoms: int = 3, hidden: int = 32) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.hidden = hidden
        self.net = nn.Sequential(
            nn.Linear(num_atoms * 3 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_atoms * 3),
        )
        self.model_config = GenerativeModelConfig(
            supports_variable_atoms=False,
            required_inputs=frozenset(),
            outputs=frozenset({"positions", "atomic_numbers"}),
        )

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        class_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict clean positions from noisy ones.

        Flattens ``x`` from ``(B, N, 3)``, appends the noise level ``sigma``
        as a per-draw feature, and maps back to ``(B, N, 3)`` through the
        MLP. ``class_labels`` is accepted for the PhysicsNeMo calling
        convention and unused here.
        """
        del class_labels
        b = x.shape[0]
        s = sigma.reshape(b, 1)
        return self.net(torch.cat([x.reshape(b, -1), s], dim=-1)).reshape_as(x)

    def to_batch(self, sample: TensorDict, cond_batch: Batch | None = None) -> Batch:
        """Materialize the sample: one point-cloud graph per draw."""
        del cond_batch
        positions = sample["x1"].reshape(-1, self.num_atoms, 3)
        numbers = positions.new_full((self.num_atoms,), 6, dtype=torch.long)
        return Batch.from_data_list(
            [AtomicData(positions=p, atomic_numbers=numbers) for p in positions]
        )
