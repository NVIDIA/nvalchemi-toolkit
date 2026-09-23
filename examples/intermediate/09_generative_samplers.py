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
"""
Generative workflows: model-owning samplers
============================================

The :class:`~nvalchemi.gen.generator.AtomisticGenerator` driver runs any
callable with the generating-function signature
``(inputs=None, *, num_samples=1, rng=None, **kwargs)``. The pattern this
example walks through is the *model-owning sampler*: a callable object whose
constructor owns the model, carrying the attributes the driver reads as
defaults — ``device``, ``required_inputs`` / ``outputs``, and an optional
``condition`` step.

Two samplers are built on the demo models
(:class:`~nvalchemi.models.gen.demo.DemoGANModel` and
:class:`~nvalchemi.models.gen.demo.DemoDiffusionModel`): a one-pass GAN
decoder and a small EDM Euler loop. Both return a
:class:`~nvalchemi.data.Batch`, so the driver takes its contract path
(hooks, device and field checks).
"""

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen import AtomisticGenerator
from nvalchemi.models.gen import DemoDiffusionModel, DemoGANModel

# %%
# The GAN sampler: one forward pass from a latent draw. The constructor binds
# the model; ``required_inputs`` / ``outputs`` mirror the model's config so
# the driver and pipelines can validate the field contract.


class DemoGANGenerate:
    """Model-owning GAN sampler: draw a latent, decode it."""

    def __init__(self, model: DemoGANModel) -> None:
        self.model = model
        self.required_inputs = model.model_config.required_inputs
        self.outputs = model.model_config.outputs

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def __call__(
        self,
        inputs: Batch | None = None,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> Batch:
        del kwargs
        n = inputs.num_graphs if isinstance(inputs, Batch) else num_samples
        z = torch.randn(n, self.model.latent_dim, generator=rng, device=self.device)
        positions = self.model.decode(z).reshape(n, self.model.num_atoms, 3)
        numbers = torch.full((self.model.num_atoms,), 6, dtype=torch.long)
        return Batch.from_data_list(
            [AtomicData(positions=p, atomic_numbers=numbers) for p in positions]
        )


# %%
# The diffusion sampler: a small EDM Euler loop. The sampler settings are
# constructor-bound; a call may override them through kwargs of the same
# names.


class DemoDiffusionGenerate:
    """Model-owning diffusion sampler: a small EDM Euler loop."""

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
        return next(self.model.parameters()).device

    def __call__(
        self,
        inputs: Batch | None = None,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> Batch:
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
        numbers = torch.full((self.model.num_atoms,), 6, dtype=torch.long)
        return Batch.from_data_list(
            [AtomicData(positions=p, atomic_numbers=numbers) for p in x]
        )


# %%
# Driving the GAN sampler inside a session: the ``with gen:`` block owns the
# seeded RNG (and a dedicated CUDA stream when the resolved device is CUDA),
# so a ``seed`` reproduces draws exactly.

gan = AtomisticGenerator(generator_func=DemoGANGenerate(DemoGANModel()), seed=42)

with gan:
    first = gan.sample(num_samples=4)
    second = gan.sample(num_samples=4)

print(
    f"GAN batch: {first.num_graphs} graphs, "
    f"{first.num_nodes_per_graph.tolist()} atoms each"
)

# %%
# The diffusion sampler composes the same way — the family lives entirely in
# the callable, so only the constructor changes.

diffusion = AtomisticGenerator(
    generator_func=DemoDiffusionGenerate(DemoDiffusionModel(), num_steps=4),
    seed=42,
)

with diffusion:
    batch = diffusion.sample(num_samples=2, num_steps=8)  # per-call override

print(f"Diffusion batch: {batch.num_graphs} graphs at sigma_max=2.0")

# %%
# A conditional call tiles one draw per conditioning graph; the sampler reads
# the conditioning batch's graph count when ``inputs`` is a ``Batch``.

conditioning = Batch.from_data_list(
    [
        AtomicData(
            positions=torch.rand(5, 3),
            atomic_numbers=torch.full((5,), 6, dtype=torch.long),
        )
        for _ in range(3)
    ]
)

with gan:
    conditioned = gan.sample(conditioning)

print(
    f"Conditional call: {conditioned.num_graphs} draws from {conditioning.num_graphs} inputs"
)
