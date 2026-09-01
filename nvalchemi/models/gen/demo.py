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
through the :class:`~nvalchemi.gen.generator.AtomGenerator` with no external
weights or optional dependencies. Each demo model provides both fallback
entry points (``generate`` and ``to_batch``), so a bare
``AtomGenerator(model=...)`` works end to end.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict
from torch import nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen.enums import GenerativeIntent, Modality
from nvalchemi.models.gen.base import GenerativeModelConfig, GenerativeModelMixin

__all__ = [
    "DemoDiffusionModel",
    "DemoGANModel",
    "demo_nonparametric_generation",
]


def _demo_config() -> GenerativeModelConfig:
    """Build the shared demo config: unconditional point-cloud generation.

    Returns
    -------
    GenerativeModelConfig
        Create/sample intents over point clouds; consumes nothing (the demos
        are unconditional), produces positions and atomic numbers.
    """
    return GenerativeModelConfig(
        intents={GenerativeIntent.CREATE, GenerativeIntent.SAMPLE},
        supports_variable_atoms=False,
        output_artifact=Modality.POINT_CLOUD,
        intent_modality_map={
            GenerativeIntent.CREATE: frozenset({Modality.POINT_CLOUD}),
            GenerativeIntent.SAMPLE: frozenset({Modality.POINT_CLOUD}),
        },
        consumes_fields=frozenset(),
        produces_fields=frozenset({"positions", "atomic_numbers"}),
    )


def _sample_to_batch(sample: TensorDict, num_atoms: int) -> Batch:
    """Materialize a demo sample: one point-cloud graph per draw.

    Parameters
    ----------
    sample
        Sample TensorDict with flat positions under ``"x1"``.
    num_atoms
        Number of atoms per graph; the sample's entries are reshaped to
        ``(-1, num_atoms, 3)``.

    Returns
    -------
    Batch
        One carbon point cloud per draw.
    """
    positions = sample["x1"].reshape(-1, num_atoms, 3)
    numbers = positions.new_full((num_atoms,), 6, dtype=torch.long)
    return Batch.from_data_list(
        [AtomicData(positions=p, atomic_numbers=numbers) for p in positions]
    )


class DemoGANModel(nn.Module, GenerativeModelMixin):
    """Minimal GAN-side demo: a latent draw decoded to a point cloud.

    The generative analogue of :class:`~nvalchemi.models.demo.DemoModel` — a
    placeholder for testing and debugging generative workflows. ``forward``
    follows the mixin convention (``forward(data, *, x)``) and decodes the
    latent ``x`` to flat positions; ``generate`` is the generation-source
    fallback (draw a latent, decode it), so the model runs through an
    :class:`~nvalchemi.gen.generator.AtomGenerator` with no
    ``generator_func``.
    """

    def __init__(
        self, num_atoms: int = 3, latent_dim: int = 4, hidden: int = 32
    ) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_atoms * 3),
        )
        self.model_config = _demo_config()

    def forward(self, data: Any, *, x: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Decode a latent draw ``x`` of shape ``(B, latent_dim)``."""
        del data, kwargs
        return self.decoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent ``z`` to positions of shape ``(B, num_atoms, 3)``."""
        return self.decoder(z).reshape(-1, self.num_atoms, 3)

    def generate(
        self,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        cond: Batch | None = None,
        **kwargs: Any,
    ) -> TensorDict:
        """Draw latents from the prior and decode them (one pass).

        Parameters
        ----------
        num_samples
            Number of draws (used only when ``cond`` is ``None``).
        rng
            Optional generator for reproducible draws.
        cond
            Conditioning batch, if any; one draw per conditioning graph.
        **kwargs
            Family-specific options (ignored).

        Returns
        -------
        TensorDict
            Decoded positions under ``"x1"``, one entry per draw.
        """
        del kwargs
        n = cond.num_graphs if isinstance(cond, Batch) else num_samples
        device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, generator=rng, device=device)
        return TensorDict({"x1": self.decode(z)}, batch_size=[n])

    def to_batch(self, sample: TensorDict, cond_batch: Batch | None) -> Batch:
        """Materialize the sample: one point-cloud graph per draw."""
        del cond_batch
        return _sample_to_batch(sample, self.num_atoms)


class DemoDiffusionModel(nn.Module, GenerativeModelMixin):
    """Minimal diffusion-side demo: an x0-predictor over point clouds.

    ``forward`` follows the PhysicsNeMo calling convention —
    ``forward(x, sigma)`` predicts clean positions from noisy ones — so the
    model slots directly into ``physicsnemo.diffusion`` preconditioners and
    samplers (see the generative user guide). ``generate`` runs a small
    self-contained EDM Euler loop, so the model also works with no
    ``generator_func`` and no PhysicsNeMo involvement at all.
    """

    def __init__(self, num_atoms: int = 3, hidden: int = 32) -> None:
        super().__init__()
        self.num_atoms = num_atoms
        self.net = nn.Sequential(
            nn.Linear(num_atoms * 3 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_atoms * 3),
        )
        self.model_config = _demo_config()

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

    def generate(
        self,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        cond: Batch | None = None,
        num_steps: int = 4,
        sigma_max: float = 2.0,
        sigma_min: float = 0.01,
        **kwargs: Any,
    ) -> TensorDict:
        """Sample with a small EDM Euler loop (first-order, deterministic).

        Starts from Gaussian noise at ``sigma_max`` and integrates
        ``dx/dσ = (x − D(x, σ))/σ`` down to ``sigma_min``, where ``D`` is
        this model's x0-prediction. With all randomness in the initial
        noise, a seeded ``rng`` reproduces draws exactly.

        Parameters
        ----------
        num_samples
            Number of draws (used only when ``cond`` is ``None``).
        rng
            Optional generator for reproducible initial noise.
        cond
            Conditioning batch, if any; one draw per conditioning graph.
        num_steps
            Number of Euler steps.
        sigma_max, sigma_min
            The noise-level endpoints.
        **kwargs
            Family-specific options (ignored).

        Returns
        -------
        TensorDict
            Denoised positions under ``"x1"``, one entry per draw.
        """
        del kwargs
        n = cond.num_graphs if isinstance(cond, Batch) else num_samples
        device = next(self.parameters()).device
        sigmas = torch.linspace(sigma_max, sigma_min, num_steps + 1, device=device)
        x = torch.randn(n, self.num_atoms, 3, generator=rng, device=device)
        x = x * sigmas[0]
        for i in range(num_steps):
            s_cur, s_next = sigmas[i], sigmas[i + 1]
            drift = (x - self.forward(x, s_cur.expand(n))) / s_cur
            x = x + (s_next - s_cur) * drift
        return TensorDict({"x1": x}, batch_size=[n])

    def to_batch(self, sample: TensorDict, cond_batch: Batch | None) -> Batch:
        """Materialize the sample: one point-cloud graph per draw."""
        del cond_batch
        return _sample_to_batch(sample, self.num_atoms)


def demo_nonparametric_generation(
    cond: Any = None,
    *,
    num_samples: int = 1,
    rng: torch.Generator | None = None,
    num_atoms: int = 3,
    box: float = 5.0,
    **kwargs: Any,
) -> Batch:
    """Emit a batch of synthetic structures — no model, no learned anything.

    Positions are uniform in a cube of side ``box``; atomic numbers are
    sampled from H/C/N/O. If ``cond`` is a :class:`~nvalchemi.data.Batch`,
    one synthetic graph is emitted per conditioning graph, so the function
    can serve as a source stage in a
    :class:`~nvalchemi.gen.pipeline.GenerationPipeline`; otherwise
    ``num_samples`` graphs are emitted.

    Parameters
    ----------
    cond
        Conditioning batch, if any; only its graph count is read.
    num_samples
        Number of structures to emit when ``cond`` is not a batch.
    rng
        Optional generator for reproducible structures.
    num_atoms
        Number of atoms per structure.
    box
        Side length of the cube positions are drawn in.
    **kwargs
        Ignored; kept for call-site compatibility.

    Returns
    -------
    Batch
        The synthetic structures.
    """
    del kwargs
    n = cond.num_graphs if isinstance(cond, Batch) else num_samples
    positions = torch.rand(n, num_atoms, 3, generator=rng) * box
    choices = torch.tensor([1, 6, 7, 8], dtype=torch.long)
    picks = torch.randint(0, len(choices), (n, num_atoms), generator=rng)
    return Batch.from_data_list(
        [
            AtomicData(positions=positions[i], atomic_numbers=choices[picks[i]])
            for i in range(n)
        ]
    )
