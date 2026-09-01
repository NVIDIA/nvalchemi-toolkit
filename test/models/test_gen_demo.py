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

Covers mixin conformance and config validity, the defaults-everywhere path
(a bare ``AtomGenerator(model=...)`` runs via the model's ``generate`` /
``to_batch`` fallbacks), seeding reproducibility, PhysicsNeMo interop for
the diffusion demo, and the nonparametric synthetic-structure source.
CPU-only, GPU-free.
"""

from __future__ import annotations

import torch

from nvalchemi.data import Batch
from nvalchemi.gen import AtomGenerator, GenerativeIntent, Modality
from nvalchemi.models.gen import (
    DemoDiffusionModel,
    DemoGANModel,
    GenerativeModelMixin,
    demo_nonparametric_generation,
)


class TestDemoGANModel:
    """``DemoGANModel``: mixin surface, config, and the bare-generator path."""

    def test_mixin_conformance_and_config(self) -> None:
        """The demo satisfies the mixin contract and declares its config."""
        model = DemoGANModel()
        assert isinstance(model, GenerativeModelMixin)
        assert model.model_config.output_artifact is Modality.POINT_CLOUD
        assert model.model_config.intents == {
            GenerativeIntent.CREATE,
            GenerativeIntent.SAMPLE,
        }
        assert model.model_config.consumes_fields == frozenset()
        assert model.model_config.produces_fields == frozenset(
            {"positions", "atomic_numbers"}
        )

    def test_bare_generator_runs_via_model_fallbacks(self) -> None:
        """``AtomGenerator(model=...)`` needs neither generating function nor
        materialization argument — the model's ``generate``/``to_batch`` serve.
        """
        gen = AtomGenerator(model=DemoGANModel(num_atoms=4))
        out = gen(num_samples_per_batch=3)
        assert isinstance(out, Batch)
        assert out.num_graphs == 3
        assert out["positions"].shape == (12, 3)

    def test_seeded_sessions_reproduce(self) -> None:
        """Same model + same seed across sessions gives identical draws."""
        model = DemoGANModel()
        gen = AtomGenerator(model=model, seed=7)
        with gen:
            first = gen.sample(num_samples_per_batch=2)
        with gen:
            second = gen.sample(num_samples_per_batch=2)
        assert torch.equal(first["positions"], second["positions"])


class TestDemoDiffusionModel:
    """``DemoDiffusionModel``: the physicsnemo-convention forward and sampler."""

    def test_bare_generator_runs_via_model_fallbacks(self) -> None:
        """The built-in EDM Euler loop drives a bare ``AtomGenerator``."""
        gen = AtomGenerator(model=DemoDiffusionModel(num_atoms=5))
        out = gen(num_samples_per_batch=2, num_steps=2)
        assert out.num_graphs == 2
        assert out["positions"].shape == (10, 3)

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
        gen = AtomGenerator(model=model, seed=3)
        with gen:
            first = gen.sample(num_samples_per_batch=2)
        with gen:
            second = gen.sample(num_samples_per_batch=2)
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

    def test_pipeline_source_stage(self) -> None:
        """The function folds into a pipeline as a plain Batch -> Batch stage."""
        pipe = AtomGenerator(model=DemoGANModel()) | demo_nonparametric_generation
        out = pipe(None)
        assert isinstance(out, Batch)
        assert out.num_graphs == 1
