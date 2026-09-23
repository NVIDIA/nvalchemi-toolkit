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
"""Tests for the demo generative models (:mod:`nvalchemi.models.gen.demo`).

Covers mixin conformance and config validity, the model-owning sampler
callables the :class:`~nvalchemi.gen.generator.AtomisticGenerator` consumes,
seeding reproducibility, and PhysicsNeMo interop for the diffusion demo.
CPU-only, GPU-free.
"""

from __future__ import annotations

import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen import AtomisticGenerator
from nvalchemi.models.gen import (
    DemoDiffusionModel,
    DemoGANModel,
    GenerativeModelMixin,
)
from nvalchemi.models.gen.demo import _DemoDiffusionGenerate, _DemoGANGenerate
from test.gen.conftest import make_batch


class TestDemoGANModel:
    """``DemoGANModel``: mixin surface, config, and the sampler path."""

    def test_mixin_conformance_and_config(self) -> None:
        """The demo satisfies the mixin contract and declares its config."""
        model = DemoGANModel()
        assert isinstance(model, GenerativeModelMixin)
        assert model.model_config.supports_variable_atoms is False
        assert model.model_config.required_inputs == frozenset()
        assert model.model_config.outputs == frozenset({"positions", "atomic_numbers"})
        assert model.model_config.prediction_outputs is None

    def test_sampler_generator_runs(self) -> None:
        """The model-owning sampler drives a bare ``AtomisticGenerator``."""
        gen = AtomisticGenerator(
            generator_func=_DemoGANGenerate(DemoGANModel(num_atoms=4))
        )
        out = gen(num_samples=3)
        assert isinstance(out, Batch)
        assert out.num_graphs == 3
        assert out["positions"].shape == (12, 3)

    def test_condition_attribute_tiles_batch_input(self) -> None:
        """The sampler's ``condition`` tiles a Batch by ``num_samples``."""
        fn = _DemoGANGenerate(DemoGANModel(num_atoms=4))
        source = make_batch(num_graphs=2)
        tiled = fn.condition(source, num_samples=3)
        assert isinstance(tiled, Batch)
        assert tiled.num_graphs == 6

    def test_driver_tiles_conditioning_batch_via_attribute(self) -> None:
        """Through the driver, a Batch input yields ``num_samples`` per graph."""
        gen = AtomisticGenerator(
            generator_func=_DemoGANGenerate(DemoGANModel(num_atoms=4))
        )
        source = make_batch(num_graphs=2)
        out = gen(source, num_samples=3)
        assert out.num_graphs == 6

    def test_sampler_object_carries_device_and_fields(self) -> None:
        """The sampler object exposes the attributes the driver reads."""
        fn = _DemoGANGenerate(DemoGANModel())
        assert fn.device == torch.device("cpu")
        assert callable(fn.condition)
        assert fn.required_inputs == frozenset()
        assert fn.outputs == frozenset({"positions", "atomic_numbers"})
        gen = AtomisticGenerator(generator_func=fn)
        # The defaults chain picked the declarations up at construction.
        assert gen.required_inputs == frozenset()
        assert gen.outputs == frozenset({"positions", "atomic_numbers"})

    def test_seeded_sessions_reproduce(self) -> None:
        """Same model + same seed across sessions gives identical draws."""
        model = DemoGANModel()
        gen = AtomisticGenerator(generator_func=_DemoGANGenerate(model), seed=7)
        with gen:
            first = gen.sample(num_samples=2)
        with gen:
            second = gen.sample(num_samples=2)
        assert torch.equal(first["positions"], second["positions"])


class TestDemoDiffusionModel:
    """``DemoDiffusionModel``: the physicsnemo-convention forward and sampler."""

    def test_sampler_generator_runs(self) -> None:
        """The built-in EDM Euler loop drives a bare ``AtomisticGenerator``."""
        gen = AtomisticGenerator(
            generator_func=_DemoDiffusionGenerate(DemoDiffusionModel(num_atoms=5))
        )
        out = gen(num_samples=2, num_steps=2)
        assert out.num_graphs == 2
        assert out["positions"].shape == (10, 3)

    def test_driver_tiles_conditioning_batch_via_attribute(self) -> None:
        """The diffusion sampler's ``condition`` tiles through the driver."""
        gen = AtomisticGenerator(
            generator_func=_DemoDiffusionGenerate(DemoDiffusionModel(num_atoms=5))
        )
        source = make_batch(num_graphs=2)
        out = gen(source, num_samples=3, num_steps=2)
        assert out.num_graphs == 6

    def test_forward_physicsnemo_compatible(self) -> None:
        """The demo wraps in ``EDMPreconditioner`` and runs ``sample`` — the
        integration pattern from the generative user guide."""
        from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
        from physicsnemo.diffusion.preconditioners import EDMPreconditioner
        from physicsnemo.diffusion.samplers import sample as pn_sample

        model = DemoDiffusionModel(num_atoms=3)
        scheduler = EDMNoiseScheduler(sigma_max=5.0)
        denoiser = scheduler.get_denoiser(x0_predictor=EDMPreconditioner(model))
        xN = torch.randn(2, 3, 3, generator=torch.Generator().manual_seed(0)) * 5.0
        out = pn_sample(denoiser, xN, scheduler, num_steps=2, solver="heun")
        assert out.shape == (2, 3, 3)
        assert torch.isfinite(out).all()

    def test_seeded_sessions_reproduce(self) -> None:
        """All randomness is the initial noise, so seeds reproduce draws."""
        model = DemoDiffusionModel()
        gen = AtomisticGenerator(generator_func=_DemoDiffusionGenerate(model), seed=3)
        with gen:
            first = gen.sample(num_samples=2)
        with gen:
            second = gen.sample(num_samples=2)
        assert torch.equal(first["positions"], second["positions"])


class TestDemoHelperLegs:
    """The demo samplers' less-traveled legs (users copy these)."""

    def test_condition_wraps_a_single_structure(self) -> None:
        """An AtomicData condition tiles into a fresh Batch of ``num_samples``."""
        fn = _DemoGANGenerate(DemoGANModel(num_atoms=3))
        data = AtomicData(
            positions=torch.randn(3, 3),
            atomic_numbers=torch.full((3,), 6, dtype=torch.long),
        )
        tiled = fn.condition(data, num_samples=4)
        assert isinstance(tiled, Batch)
        assert tiled.num_graphs == 4
        assert torch.equal(tiled.get_data(0).positions, data.positions)

    def test_diffusion_per_call_sigma_overrides(self) -> None:
        """Per-call ``sigma_max``/``sigma_min`` override the constructor-bound values."""
        fn = _DemoDiffusionGenerate(DemoDiffusionModel(num_atoms=4), num_steps=2)
        default = fn(num_samples=2, rng=torch.Generator().manual_seed(3))
        wider = fn(num_samples=2, rng=torch.Generator().manual_seed(3), sigma_max=20.0)
        assert not torch.allclose(default.positions, wider.positions)
