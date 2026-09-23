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
"""Shared helpers for the generative API test suite.

Mirrors the dynamics/training convention: dummy-data builders and trivial
generating functions live in a per-suite ``conftest.py`` and
are imported by the test modules (``from test.gen.conftest import
make_batch``) instead of being redefined per module.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from nvalchemi.data import AtomicData, Batch
from nvalchemi.models.gen import DemoDiffusionModel, DemoGANModel


def make_atomic_data(num_atoms: int = 3) -> AtomicData:
    """Build a minimal :class:`AtomicData` for tests.

    Parameters
    ----------
    num_atoms
        Number of atoms in the dummy structure.

    Returns
    -------
    AtomicData
        A small structure with random positions and carbon atomic numbers.
    """
    return AtomicData(
        positions=torch.randn(num_atoms, 3),
        atomic_numbers=torch.full((num_atoms,), 6, dtype=torch.long),
    )


def make_batch(num_graphs: int = 2) -> Batch:
    """Build a small :class:`Batch` for tests.

    Parameters
    ----------
    num_graphs
        Number of graphs to batch.

    Returns
    -------
    Batch
        A batch of dummy structures.
    """
    return Batch.from_data_list([make_atomic_data() for _ in range(num_graphs)])


def trivial_generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
    """A minimal :class:`~nvalchemi.gen.GeneratingFunction` for tests.

    Parameters
    ----------
    inputs
        Conditioning batch, if any; the sample's leading size matches it.
    num_samples
        Number of draws (used only when ``inputs`` is not a batch).
    rng
        Optional generator (ignored).
    **kwargs
        Family-specific options (ignored).

    Returns
    -------
    TensorDict
        Zeros under the ``"x1"`` key, aligned with ``inputs``.
    """
    del rng, kwargs
    n = inputs.num_graphs if isinstance(inputs, Batch) else num_samples
    return TensorDict({"x1": torch.zeros(n, 1, 3)}, batch_size=[n])


def batch_generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
    """A minimal generating function returning a :class:`Batch` directly.

    Exercises the driver's Batch path: the function returns a ``Batch``
    directly, so ``AFTER_GENERATE`` hooks and the device/field checks apply.

    Parameters
    ----------
    inputs
        Conditioning batch, if any; the batch's graph count matches it.
    num_samples
        Number of draws (used only when ``inputs`` is not a batch).
    rng
        Optional generator (ignored).
    **kwargs
        Family-specific options (ignored).

    Returns
    -------
    Batch
        ``num_samples`` (or ``inputs.num_graphs``) dummy graphs.
    """
    del rng, kwargs
    n = inputs.num_graphs if isinstance(inputs, Batch) else num_samples
    return make_batch(n)


def tile_condition(inputs, *, num_samples=None, rng=None):
    """Trivial condition callable: pass ``inputs`` through unchanged.

    Parameters
    ----------
    inputs
        The call's raw inputs.
    num_samples
        Resolved draw count (accepted for the conditioning signature; unused).
    rng
        Resolved RNG (accepted for the conditioning signature; unused).

    Returns
    -------
    Any
        ``inputs``, unchanged.
    """
    del num_samples, rng
    return inputs


class DeviceAwareGenerate:
    """Generating function object carrying ``device`` and building batches there.

    Exercises the driver's device defaults chain, session stream creation,
    and the device-residency check on any host: the object declares a device
    (readable via the chain) and returns batches built on it.
    """

    def __init__(self, device: str | torch.device) -> None:
        self.device = torch.device(device)

    def __call__(self, inputs=None, *, num_samples=1, rng=None, **kwargs) -> Batch:
        """Return a batch of dummy graphs resident on ``self.device``.

        Parameters
        ----------
        inputs
            Conditioning batch, if any; the batch's graph count matches it.
        num_samples
            Number of draws (used only when ``inputs`` is not a batch).
        rng
            Optional generator (ignored).
        **kwargs
            Family-specific options (ignored).

        Returns
        -------
        Batch
            Dummy graphs on ``self.device``.
        """
        del rng, kwargs
        n = inputs.num_graphs if isinstance(inputs, Batch) else num_samples
        return make_batch(n).to(self.device)


class DemoGANGenerate:
    """Model-owning GAN sampler: draw a latent, decode it.

    Test utility carrying the attributes the
    :class:`~nvalchemi.gen.generator.AtomisticGenerator` reads as defaults:
    ``device`` (the model's parameter device) and the model config's field
    declarations — plus ``condition``, the driver's optional pre-generation
    step, tiling a conditioning batch by the draw count.

    Parameters
    ----------
    model
        The :class:`~nvalchemi.models.gen.demo.DemoGANModel` this sampler owns.
    """

    def __init__(self, model: "DemoGANModel") -> None:
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
        """Draw latents from the prior and decode them (one forward pass).

        Parameters
        ----------
        inputs
            Conditioning batch, if any; one draw per conditioning graph.
        num_samples
            Number of draws (used only when ``inputs`` is not a batch).
        rng
            Optional generator for reproducible latents.
        **kwargs
            Ignored.

        Returns
        -------
        Batch
            One point-cloud graph per draw.
        """
        del kwargs
        n = inputs.num_graphs if isinstance(inputs, Batch) else num_samples
        z = torch.randn(n, self.model.latent_dim, generator=rng, device=self.device)
        sample = TensorDict({"x1": self.model.decode(z)}, batch_size=[n])
        return self.model.to_batch(sample, inputs)


class DemoDiffusionGenerate:
    """Model-owning diffusion sampler: a small self-contained EDM Euler loop.

    Test utility carrying the attributes the
    :class:`~nvalchemi.gen.generator.AtomisticGenerator` reads as defaults:
    ``device`` (the model's parameter device) and the model config's field
    declarations — plus ``condition``, the driver's optional pre-generation
    step, tiling a conditioning batch by the draw count. The sampler
    hyperparameters are constructor-bound; per-call kwargs of the same names
    override them.

    Parameters
    ----------
    model
        The :class:`~nvalchemi.models.gen.demo.DemoDiffusionModel` this
        sampler owns.
    num_steps
        Number of Euler steps in the EDM loop.
    sigma_max, sigma_min
        The noise-level endpoints.
    """

    def __init__(
        self,
        model: "DemoDiffusionModel",
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
            constructor-bound sampler settings for this call; any other
            options are ignored.

        Returns
        -------
        Batch
            One point-cloud graph per draw.
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
