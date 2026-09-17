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

Covers mixin conformance and config validity, the factory pattern
(``make_demo_*_generate`` builds the model-owning generating function the
:class:`~nvalchemi.gen.generator.AtomisticGenerator` consumes), seeding
reproducibility, spec capture via the factory objects' own ``to_spec``,
PhysicsNeMo interop for the diffusion demo, and the nonparametric
synthetic-structure source. CPU-only, GPU-free.
"""

from __future__ import annotations

import torch

from nvalchemi.data import Batch
from nvalchemi.gen import AtomisticGenerator
from nvalchemi.models.gen import (
    DemoDiffusionModel,
    DemoGANModel,
    GenerativeModelMixin,
    demo_nonparametric_generation,
    make_demo_diffusion_generate,
    make_demo_gan_generate,
)


class TestDemoGANModel:
    """``DemoGANModel``: mixin surface, config, and the factory path."""

    def test_mixin_conformance_and_config(self) -> None:
        """The demo satisfies the mixin contract and declares its config."""
        model = DemoGANModel()
        assert isinstance(model, GenerativeModelMixin)
        assert model.model_config.supports_variable_atoms is False
        assert model.model_config.consumes_fields == frozenset()
        assert model.model_config.produces_fields == frozenset(
            {"positions", "atomic_numbers"}
        )
        assert model.model_config.prediction_outputs is None

    def test_factory_generator_runs(self) -> None:
        """The factory-built generating function drives a bare ``AtomisticGenerator``."""
        gen = AtomisticGenerator(
            generator_func=make_demo_gan_generate(DemoGANModel(num_atoms=4))
        )
        out = gen(num_samples=3)
        assert isinstance(out, Batch)
        assert out.num_graphs == 3
        assert out["positions"].shape == (12, 3)

    def test_condition_attribute_tiles_batch_input(self) -> None:
        """The factory object's ``condition`` tiles a Batch by ``num_samples``."""
        fn = make_demo_gan_generate(DemoGANModel(num_atoms=4))
        source = demo_nonparametric_generation(num_samples=2, num_atoms=4)
        tiled = fn.condition(source, num_samples=3)
        assert isinstance(tiled, Batch)
        assert tiled.num_graphs == 6

    def test_driver_tiles_conditioning_batch_via_attribute(self) -> None:
        """Through the driver, a Batch input yields ``num_samples`` per graph."""
        gen = AtomisticGenerator(
            generator_func=make_demo_gan_generate(DemoGANModel(num_atoms=4))
        )
        source = demo_nonparametric_generation(num_samples=2, num_atoms=4)
        out = gen(source, num_samples=3)
        assert out.num_graphs == 6

    def test_factory_object_carries_device_and_fields(self) -> None:
        """The factory object exposes the attributes the driver reads."""
        fn = make_demo_gan_generate(DemoGANModel())
        assert fn.device == torch.device("cpu")
        assert callable(fn.condition)
        assert fn.consumes_fields == frozenset()
        assert fn.produces_fields == frozenset({"positions", "atomic_numbers"})
        gen = AtomisticGenerator(generator_func=fn)
        # The defaults chain picked the declarations up at construction.
        assert gen.consumes_fields == frozenset()
        assert gen.produces_fields == frozenset({"positions", "atomic_numbers"})

    def test_seeded_sessions_reproduce(self) -> None:
        """Same model + same seed across sessions gives identical draws."""
        model = DemoGANModel()
        gen = AtomisticGenerator(generator_func=make_demo_gan_generate(model), seed=7)
        with gen:
            first = gen.sample(num_samples=2)
        with gen:
            second = gen.sample(num_samples=2)
        assert torch.equal(first["positions"], second["positions"])


class TestDemoDiffusionModel:
    """``DemoDiffusionModel``: the physicsnemo-convention forward and sampler."""

    def test_factory_generator_runs(self) -> None:
        """The built-in EDM Euler loop drives a bare ``AtomisticGenerator``."""
        gen = AtomisticGenerator(
            generator_func=make_demo_diffusion_generate(DemoDiffusionModel(num_atoms=5))
        )
        out = gen(num_samples=2, num_steps=2)
        assert out.num_graphs == 2
        assert out["positions"].shape == (10, 3)

    def test_driver_tiles_conditioning_batch_via_attribute(self) -> None:
        """The diffusion factory's ``condition`` tiles through the driver."""
        gen = AtomisticGenerator(
            generator_func=make_demo_diffusion_generate(DemoDiffusionModel(num_atoms=5))
        )
        source = demo_nonparametric_generation(num_samples=2, num_atoms=5)
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
        gen = AtomisticGenerator(
            generator_func=make_demo_diffusion_generate(model), seed=3
        )
        with gen:
            first = gen.sample(num_samples=2)
        with gen:
            second = gen.sample(num_samples=2)
        assert torch.equal(first["positions"], second["positions"])


class TestDemoNonparametricGeneration:
    """``demo_nonparametric_generation``: synthetic structures, no model."""

    def test_emits_requested_count(self) -> None:
        """``num_samples`` controls the graph count; positions stay in the box."""
        out = demo_nonparametric_generation(
            num_samples=4, num_atoms=6, box=3.0, rng=torch.Generator().manual_seed(0)
        )
        assert isinstance(out, Batch)
        assert out.num_graphs == 4
        assert out["positions"].shape == (24, 3)
        assert (out["positions"] >= 0.0).all()
        assert (out["positions"] < 3.0).all()

    def test_sizes_by_conditioning_batch(self) -> None:
        """A ``Batch`` conditioning input sets the emitted graph count."""
        source = demo_nonparametric_generation(num_samples=3)
        out = demo_nonparametric_generation(source)
        assert out.num_graphs == 3

    def test_reproducible_with_rng(self) -> None:
        """The same seeded generator gives identical structures."""
        a = demo_nonparametric_generation(
            num_samples=2, rng=torch.Generator().manual_seed(1)
        )
        b = demo_nonparametric_generation(
            num_samples=2, rng=torch.Generator().manual_seed(1)
        )
        assert torch.equal(a["positions"], b["positions"])
        assert torch.equal(a["atomic_numbers"], b["atomic_numbers"])

    def test_batch_passthrough_source(self) -> None:
        """As a ``generator_func`` it returns a ``Batch`` — no mapping needed."""
        gen = AtomisticGenerator(generator_func=demo_nonparametric_generation)
        source = demo_nonparametric_generation(num_samples=2)
        out = gen(source)
        assert isinstance(out, Batch)
        assert out.num_graphs == 2

    def test_pipeline_source_stage(self) -> None:
        """The function folds into a pipeline as a plain Batch -> Batch stage."""
        pipe = (
            AtomisticGenerator(generator_func=make_demo_gan_generate(DemoGANModel()))
            | demo_nonparametric_generation
        )
        out = pipe(None)
        assert isinstance(out, Batch)
        assert out.num_graphs == 1
