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
weights or optional dependencies. The module-level factories
:func:`make_demo_gan_generate` and :func:`make_demo_diffusion_generate` build
model-owning generating functions for the driver; the callable objects they
return carry ``device`` (from the model's parameters), the config's field
declarations, a ``condition`` tiling helper, and ``to_spec`` (factory kwargs
captured via :func:`~nvalchemi.training.create_model_spec`), so they slot
into the driver's defaults chain and the spec machinery.
:func:`demo_nonparametric_generation` is a plain function returning a
:class:`~nvalchemi.data.Batch` directly, taking the driver's ``Batch``
path (hooks, device and field checks).
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict
from torch import nn

from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen.generator import GeneratingFunction
from nvalchemi.models.gen.base import GenerativeModelConfig, GenerativeModelMixin

__all__ = [
    "DemoDiffusionModel",
    "DemoGANModel",
    "demo_nonparametric_generation",
    "make_demo_diffusion_generate",
    "make_demo_gan_generate",
]


def _demo_config() -> GenerativeModelConfig:
    """Build the shared demo config: unconditional point-cloud generation.

    Returns
    -------
    GenerativeModelConfig
        Consumes nothing (the demos are unconditional), produces positions
        and atomic numbers.
    """
    return GenerativeModelConfig(
        supports_variable_atoms=False,
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


def _tile_condition(inputs: Any, num_samples: int) -> Any:
    """Tile a conditioning batch so each graph gets ``num_samples`` draws.

    Parameters
    ----------
    inputs
        A :class:`~nvalchemi.data.Batch` or :class:`~nvalchemi.data.AtomicData`,
        or any other container (passed through unchanged).
    num_samples
        Draws per conditioning graph.

    Returns
    -------
    Any
        The tiled batch, or the inputs unchanged.
    """
    if inputs is None:
        return None
    if isinstance(inputs, Batch):
        idx = torch.arange(inputs.num_graphs).repeat_interleave(num_samples)
        return inputs[idx.to(inputs.device)]
    if isinstance(inputs, AtomicData):
        return Batch.from_data_list([inputs] * num_samples, device=inputs.device)
    return inputs


class DemoGANModel(nn.Module, GenerativeModelMixin):
    """Minimal GAN-side demo: a latent draw decoded to a point cloud.

    The generative analogue of :class:`~nvalchemi.models.demo.DemoModel` — a
    placeholder for testing and debugging generative workflows. ``forward``
    follows the mixin convention (``forward(data, *, x)``) and decodes the
    latent ``x`` to flat positions. Sampling lives in the callable built by
    :func:`make_demo_gan_generate` (draw a latent, decode it), which owns the
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
        self.model_config = _demo_config()

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
        return _sample_to_batch(sample, self.num_atoms)


class _DemoGANGenerate:
    """Model-owning GAN sampler, built by :func:`make_demo_gan_generate`.

    Carries the attributes the :class:`~nvalchemi.gen.generator.AtomisticGenerator`
    reads as defaults: ``device`` (the model's parameter device) and the
    model config's field declarations — plus ``condition`` (the driver's
    optional pre-generation step, tiling a conditioning batch by the draw
    count) and ``to_spec`` for spec round-trips.
    """

    def __init__(self, model: DemoGANModel) -> None:
        self.model = model
        self.consumes_fields = model.model_config.consumes_fields
        self.produces_fields = model.model_config.produces_fields

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
        return _tile_condition(inputs, 1 if num_samples is None else num_samples)

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


def make_demo_gan_generate(model: DemoGANModel) -> GeneratingFunction:
    """Build a :class:`~nvalchemi.gen.generator.GeneratingFunction` for a GAN.

    Parameters
    ----------
    model
        The :class:`DemoGANModel` the generating function owns.

    Returns
    -------
    GeneratingFunction
        A callable object that draws latents and decodes them, carrying
        ``device``, field declarations, a ``condition`` tiling helper, and
        ``to_spec``.
    """
    return _DemoGANGenerate(model)


class DemoDiffusionModel(nn.Module, GenerativeModelMixin):
    """Minimal diffusion-side demo: an x0-predictor over point clouds.

    ``forward`` follows the PhysicsNeMo calling convention —
    ``forward(x, sigma)`` predicts clean positions from noisy ones — so the
    model slots directly into ``physicsnemo.diffusion`` preconditioners and
    samplers (see the generative user guide). Sampling lives in the callable
    built by :func:`make_demo_diffusion_generate` (a small self-contained EDM
    Euler loop), which owns the model for the
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

    def to_batch(self, sample: TensorDict, cond_batch: Batch | None = None) -> Batch:
        """Materialize the sample: one point-cloud graph per draw."""
        del cond_batch
        return _sample_to_batch(sample, self.num_atoms)


class _DemoDiffusionGenerate:
    """Model-owning diffusion sampler, built by :func:`make_demo_diffusion_generate`.

    Carries the attributes the :class:`~nvalchemi.gen.generator.AtomisticGenerator`
    reads as defaults: ``device`` (the model's parameter device) and the
    model config's field declarations — plus ``condition`` (the driver's
    optional pre-generation step, tiling a conditioning batch by the draw
    count) and ``to_spec`` for spec round-trips.
    The sampler hyperparameters are factory-bound; per-call kwargs of the
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
        self.consumes_fields = model.model_config.consumes_fields
        self.produces_fields = model.model_config.produces_fields

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
        return _tile_condition(inputs, 1 if num_samples is None else num_samples)

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
            factory-bound sampler settings for this call; any other options
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


def make_demo_diffusion_generate(
    model: DemoDiffusionModel,
    *,
    num_steps: int = 4,
    sigma_max: float = 2.0,
    sigma_min: float = 0.01,
) -> GeneratingFunction:
    """Build a :class:`~nvalchemi.gen.generator.GeneratingFunction` for diffusion.

    Parameters
    ----------
    model
        The :class:`DemoDiffusionModel` the generating function owns.
    num_steps
        Number of Euler steps in the EDM loop.
    sigma_max, sigma_min
        The noise-level endpoints.

    Returns
    -------
    GeneratingFunction
        A callable object running the EDM Euler loop, carrying ``device``,
        field declarations, a ``condition`` tiling helper, and ``to_spec``.
    """
    return _DemoDiffusionGenerate(
        model, num_steps=num_steps, sigma_max=sigma_max, sigma_min=sigma_min
    )


def demo_nonparametric_generation(
    inputs: Any = None,
    *,
    num_samples: int = 1,
    rng: torch.Generator | None = None,
    num_atoms: int = 3,
    box: float = 5.0,
    **kwargs: Any,
) -> Batch:
    """Emit a batch of synthetic structures — no model, no learned anything.

    Positions are uniform in a cube of side ``box``; atomic numbers are
    sampled from H/C/N/O. If ``inputs`` is a :class:`~nvalchemi.data.Batch`,
    one synthetic graph is emitted per input graph, so the function
    can serve as a source stage in a
    :class:`~nvalchemi.gen.pipeline.GenerationPipeline`; otherwise
    ``num_samples`` graphs are emitted.

    Parameters
    ----------
    inputs
        Conditioning batch, if any; only its graph count is read.
    num_samples
        Number of structures to emit when ``inputs`` is not a batch.
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
    n = inputs.num_graphs if isinstance(inputs, Batch) else num_samples
    positions = torch.rand(n, num_atoms, 3, generator=rng) * box
    choices = torch.tensor([1, 6, 7, 8], dtype=torch.long)
    picks = torch.randint(0, len(choices), (n, num_atoms), generator=rng)
    return Batch.from_data_list(
        [
            AtomicData(positions=positions[i], atomic_numbers=choices[picks[i]])
            for i in range(n)
        ]
    )
