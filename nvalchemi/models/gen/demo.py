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
weights or optional dependencies. The ``_DemoGANGenerate`` and
``_DemoDiffusionGenerate`` callables are model-owning generating functions for
the driver: they carry ``device`` (from the model's parameters), the config's
field declarations and a ``condition`` tiling helper, so they slot into the
driver's defaults chain.
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
    latent ``x`` to flat positions. Sampling lives in the companion
    ``_DemoGANGenerate`` callable (draw a latent, decode it), which owns the
    model for the :class:`~nvalchemi.gen.generator.AtomisticGenerator`.
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


class _DemoGANGenerate:
    """Model-owning GAN sampler.
    Carries the attributes the :class:`~nvalchemi.gen.generator.AtomisticGenerator`
    reads as defaults: ``device`` (the model's parameter device) and the
    model config's field declarations — plus ``condition`` (the driver's
    optional pre-generation step, tiling a conditioning batch by the draw
    count).
    """

    def __init__(self, model: DemoGANModel) -> None:
        self.model = model
        self.required_inputs = model.model_config.required_inputs
        self.outputs = model.model_config.outputs

    @property
    def device(self) -> torch.device:
        """The model's parameter device."""
        return next(self.model.parameters()).device

    def condition(
        self,
        inputs: Any,
        *,
        num_samples: int | None = None,
        rng: torch.Generator | None = None,
    ) -> Any:
        """Tile a conditioning batch by the draw count.

        The driver's optional condition step (see
        :class:`~nvalchemi.gen.generator.GeneratingFunction`): a
        :class:`~nvalchemi.data.Batch` input comes out with each graph
        repeated ``num_samples`` times and one draw is emitted per
        conditioned graph. ``rng`` is accepted for the condition-callable
        signature and unused.

        Parameters
        ----------
        inputs
            The call's raw inputs.
        num_samples
            The resolved draw count for the call; ``None`` (standalone use)
            falls back to a single draw per conditioning graph.
        rng
            The resolved RNG (unused).

        Returns
        -------
        Any
            The conditioned inputs for the generating call.
        """
        del rng
        n = 1 if num_samples is None else num_samples
        if inputs is None:
            return None
        if isinstance(inputs, Batch):
            idx = torch.arange(inputs.num_graphs).repeat_interleave(n)
            return inputs[idx.to(inputs.device)]
        if isinstance(inputs, AtomicData):
            return Batch.from_data_list([inputs] * n, device=inputs.device)
        return inputs

    def __call__(
        self,
        inputs: Batch | None = None,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        **kwargs: Any,
    ) -> Batch:
        """Draw latents from the prior and decode them (one pass).

        Parameters
        ----------
        inputs
            Conditioning batch, if any; one draw per conditioning graph.
        num_samples
            Number of draws (used only when ``inputs`` is not a batch).
        rng
            Optional generator for reproducible draws.
        **kwargs
            Family-specific options (ignored).

        Returns
        -------
        Batch
            One point-cloud graph per draw — already a ``Batch``, so the
            driver takes the ``Batch`` path.
        """
        del kwargs
        n = inputs.num_graphs if isinstance(inputs, Batch) else num_samples
        z = torch.randn(n, self.model.latent_dim, generator=rng, device=self.device)
        sample = TensorDict({"x1": self.model.decode(z)}, batch_size=[n])
        return self.model.to_batch(sample, inputs)


class DemoDiffusionModel(nn.Module, GenerativeModelMixin):
    """Minimal diffusion-side demo: an x0-predictor over point clouds.

    ``forward`` follows the PhysicsNeMo calling convention —
    ``forward(x, sigma)`` predicts clean positions from noisy ones — so the
    model slots directly into ``physicsnemo.diffusion`` preconditioners and
    samplers (see the generative user guide). Sampling lives in the companion
    ``_DemoDiffusionGenerate`` callable (a small self-contained EDM Euler
    loop), which owns the model for the
    :class:`~nvalchemi.gen.generator.AtomisticGenerator`.
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


class _DemoDiffusionGenerate:
    """Model-owning diffusion sampler.

    Carries the attributes the :class:`~nvalchemi.gen.generator.AtomisticGenerator`
    reads as defaults: ``device`` (the model's parameter device) and the
    model config's field declarations — plus ``condition`` (the driver's
    optional pre-generation step, tiling a conditioning batch by the draw
    count).
    The sampler hyperparameters are constructor-bound; per-call kwargs of the
    same names override them.
    """

    def __init__(
        self,
        model: DemoDiffusionModel,
        *,
        num_steps: int = 4,
        sigma_max: float = 2.0,
        sigma_min: float = 0.01,
    ) -> None:
        self.model = model
        self.num_steps = num_steps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.required_inputs = model.model_config.required_inputs
        self.outputs = model.model_config.outputs

    @property
    def device(self) -> torch.device:
        """The model's parameter device."""
        return next(self.model.parameters()).device

    def condition(
        self,
        inputs: Any,
        *,
        num_samples: int | None = None,
        rng: torch.Generator | None = None,
    ) -> Any:
        """Tile a conditioning batch by the draw count.

        The driver's optional condition step (see
        :class:`~nvalchemi.gen.generator.GeneratingFunction`): a
        :class:`~nvalchemi.data.Batch` input comes out with each graph
        repeated ``num_samples`` times and the EDM loop emits one draw per
        conditioned graph. ``rng`` is accepted for the condition-callable
        signature and unused.

        Parameters
        ----------
        inputs
            The call's raw inputs.
        num_samples
            The resolved draw count for the call; ``None`` (standalone use)
            falls back to a single draw per conditioning graph.
        rng
            The resolved RNG (unused).

        Returns
        -------
        Any
            The conditioned inputs for the generating call.
        """
        del rng
        n = 1 if num_samples is None else num_samples
        if inputs is None:
            return None
        if isinstance(inputs, Batch):
            idx = torch.arange(inputs.num_graphs).repeat_interleave(n)
            return inputs[idx.to(inputs.device)]
        if isinstance(inputs, AtomicData):
            return Batch.from_data_list([inputs] * n, device=inputs.device)
        return inputs

    def __call__(
        self,
        inputs: Batch | None = None,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        **kwargs: Any,
    ) -> Batch:
        """Sample with a small EDM Euler loop (first-order, deterministic).

        Starts from Gaussian noise at ``sigma_max`` and integrates
        ``dx/dσ = (x − D(x, σ))/σ`` down to ``sigma_min``, where ``D`` is
        the model's x0-prediction. With all randomness in the initial
        noise, a seeded ``rng`` reproduces draws exactly.

        Parameters
        ----------
        inputs
            Conditioning batch, if any; one draw per conditioning graph.
        num_samples
            Number of draws (used only when ``inputs`` is not a batch).
        rng
            Optional generator for reproducible initial noise.
        **kwargs
            ``num_steps``, ``sigma_max``, and ``sigma_min`` override the
            constructor-bound sampler settings for this call; any other options
            are ignored.

        Returns
        -------
        Batch
            One point-cloud graph per draw — already a ``Batch``, so the
            driver takes the ``Batch`` path.
        """
        num_steps = kwargs.pop("num_steps", self.num_steps)
        sigma_max = kwargs.pop("sigma_max", self.sigma_max)
        sigma_min = kwargs.pop("sigma_min", self.sigma_min)
        n = inputs.num_graphs if isinstance(inputs, Batch) else num_samples
        device = self.device
        sigmas = torch.linspace(sigma_max, sigma_min, num_steps + 1, device=device)
        x = torch.randn(n, self.model.num_atoms, 3, generator=rng, device=device)
        x = x * sigmas[0]
        for i in range(num_steps):
            s_cur, s_next = sigmas[i], sigmas[i + 1]
            drift = (x - self.model.forward(x, s_cur.expand(n))) / s_cur
            x = x + (s_next - s_cur) * drift
        sample = TensorDict({"x1": x}, batch_size=[n])
        return self.model.to_batch(sample, inputs)
