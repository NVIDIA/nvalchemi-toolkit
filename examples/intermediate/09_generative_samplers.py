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

The generative driver runs any callable that matches the
:class:`~nvalchemi.gen.generator.GeneratingFunction` protocol:
``(inputs=None, *, num_samples=1, rng=None, **kwargs) -> Batch``.

In the model-owning sampler pattern, the callable owns the trained model
and exposes metadata attributes like ``device``, ``required_inputs``,
``outputs``, and an optional ``condition`` hook. The driver reads these
attributes as defaults instead of managing model internals directly.

We walk through two sampling approaches:
1. Single-pass latent decoding with a GAN.
2. Iterative Euler integration with a diffusion model.

Both samplers return a :class:`~nvalchemi.data.Batch`, matching the output
contract needed for driver hooks, device checks, and pipeline chaining.
"""

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen import AtomisticGenerator
from nvalchemi.models.gen import DemoDiffusionModel, DemoGANModel

# %%
# Model-owning GAN sampler: single-pass decode
# ---------------------------------------------
# Single-pass decoders implement the
# :class:`~nvalchemi.gen.generator.GeneratingFunction` protocol directly.
#
# We hold the model inside the sampler rather than handing it to the driver.
# Exposing ``required_inputs`` and ``outputs`` lets the driver and pipeline
# stages validate tensor fields without inspecting the model itself.
#
# Packaging generated atoms into a :class:`~nvalchemi.data.Batch` satisfies the
# output contract. Downstream stages, hooks, and dynamics can consume the
# output immediately.


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
# Model-owning diffusion sampler: iterative Euler loop
# -----------------------------------------------------
# Multi-step generation uses the same callable interface as single-pass
# decoding. From the driver's perspective, the sampling algorithm is an
# internal detail.
#
# Defaults for noise schedules and step counts live on the instance. Callers
# can override them on any sample call by passing keyword arguments.


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
# Driving the sampler: sessions and reproduction
# -----------------------------------------------
# :class:`~nvalchemi.gen.generator.AtomisticGenerator` coordinates sampling,
# device transfers, and hooks.
#
# Entering a ``with gan:`` session sets up the RNG state, creates a CUDA
# stream on GPU, and manages compiled model lifetimes. Repeated draws inside
# the session advance the RNG deterministically.

gan = AtomisticGenerator(generator_func=DemoGANGenerate(DemoGANModel()), seed=42)

with gan:
    first = gan.sample(num_samples=4)
    second = gan.sample(num_samples=4)

print(
    f"GAN batch: {first.num_graphs} graphs, "
    f"{first.num_nodes_per_graph.tolist()} atoms each"
)

# %%
# Driving the diffusion sampler: per-call overrides
# --------------------------------------------------
# The driver setup looks identical across model families. We wrap the diffusion
# sampler just like the GAN.
#
# Extra keyword arguments passed to
# :meth:`~nvalchemi.gen.generator.AtomisticGenerator.sample` forward directly to
# the sampler. Here ``num_steps=8`` overrides the default 4-step schedule for a
# single call.

diffusion = AtomisticGenerator(
    generator_func=DemoDiffusionGenerate(DemoDiffusionModel(), num_steps=4),
    seed=42,
)

with diffusion:
    batch = diffusion.sample(num_samples=2, num_steps=8)  # per-call override

print(f"Diffusion batch: {batch.num_graphs} graphs at sigma_max=2.0")

# %%
# Conditional generation: batch input handling
# ---------------------------------------------
# Passing a batch as the first argument runs conditional generation.
#
# If the sampler defines a ``condition`` attribute or the driver has a
# ``condition_func``, the driver runs that transform first. Without one, the
# input batch goes straight to the callable, which reads ``inputs.num_graphs``
# and generates matching structures.

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
