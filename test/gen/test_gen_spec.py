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
"""Structural tests for spec-based construction of generators and pipelines.

Covers :class:`~nvalchemi.gen.spec.AtomGeneratorSpec` and
:class:`~nvalchemi.gen.spec.GenerationPipelineSpec`: JSON round-trips,
factory-based spec-ability (importable factories, not
:func:`functools.partial`), hook specs with enum-stage rehydration, and
runtime model injection at ``build`` time. CPU-only, GPU-free, no optional
deps.
"""

from __future__ import annotations

import functools
import json

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.gen.generator import AtomGenerator
from nvalchemi.gen.spec import AtomGeneratorSpec, GenerationPipelineSpec
from nvalchemi.gen.stages import GenerationStage
from nvalchemi.models.gen.demo import DemoGANModel, demo_nonparametric_generation
from nvalchemi.training._spec import create_model_spec
from test.gen.conftest import make_batch, trivial_generate, zeros_to_batch


def make_trivial_generate():
    """Factory returning the trivial generating function (spec-able).

    Returns
    -------
    Callable
        ``trivial_generate``.
    """
    return trivial_generate


def make_zeros_recon():
    """Factory returning the zeros reconstruction (spec-able).

    Returns
    -------
    Callable
        ``zeros_to_batch``.
    """
    return zeros_to_batch


def make_passthrough_stage():
    """Factory returning a Batch -> Batch identity stage (spec-able).

    Returns
    -------
    Callable
        A pass-through pipeline stage.
    """

    def stage(batch: Batch) -> Batch:
        """Pass the batch through unchanged."""
        return batch

    return stage


class ScaleSampleHook:
    """Spec-able demo hook: scale the generated positions at AFTER_GENERATE."""

    def __init__(
        self,
        factor: float = 2.0,
        frequency: int = 1,
        stage: GenerationStage | None = None,
    ) -> None:
        self.factor = factor
        self.frequency = frequency
        self.stage = stage if stage is not None else GenerationStage.AFTER_GENERATE

    def __call__(self, ctx, stage) -> None:
        """Scale the materialized batch's positions by ``self.factor``."""
        ctx.batch.positions = ctx.batch.positions * self.factor


class TestGeneratorSpec:
    """``AtomGeneratorSpec`` construction, JSON round-trip, and build."""

    def _spec(self) -> AtomGeneratorSpec:
        """Build a representative spec with a hook and field declarations.

        Returns
        -------
        AtomGeneratorSpec
            A fully populated spec.
        """
        return AtomGeneratorSpec(
            generator_func=create_model_spec(make_trivial_generate),
            output_to_batch_func=create_model_spec(make_zeros_recon),
            hooks=[
                create_model_spec(
                    ScaleSampleHook, factor=3.0, stage=GenerationStage.AFTER_GENERATE
                )
            ],
            consumes_fields=["positions"],
            produces_fields=["positions", "atomic_numbers"],
            num_samples_per_batch=2,
            seed=11,
            compile_generate=True,
            compile_kwargs={"backend": "eager"},
        )

    def test_spec_builds_working_generator(self) -> None:
        """A spec builds a generator whose pieces match the spec."""
        gen = self._spec().build(model=torch.nn.Module())
        assert gen.num_samples_per_batch == 2
        assert gen.seed == 11
        assert gen.consumes_fields == frozenset({"positions"})
        assert gen.produces_fields == frozenset({"positions", "atomic_numbers"})
        assert len(gen.hooks) == 1
        hook = gen.hooks[0]
        assert isinstance(hook, ScaleSampleHook)
        assert hook.factor == 3.0
        assert gen.compile_generate is True
        assert gen.compile_kwargs == {"backend": "eager"}
        out = gen(make_batch(num_graphs=1))
        assert out.num_graphs == 2

    def test_spec_json_round_trip(self) -> None:
        """``model_dump_json`` -> ``model_validate_json`` preserves the spec."""
        spec = self._spec()
        blob = spec.model_dump_json()
        raw = json.loads(blob)  # genuinely plain JSON
        assert raw["generator_func"]["cls_path"].endswith("make_trivial_generate")
        restored = AtomGeneratorSpec.model_validate_json(blob)
        assert restored.compile_generate is True
        assert restored.compile_kwargs == {"backend": "eager"}
        gen = restored.build(model=torch.nn.Module())
        assert gen.num_samples_per_batch == 2
        hook = gen.hooks[0]
        assert isinstance(hook, ScaleSampleHook)
        assert hook.factor == 3.0
        # The enum stage survived JSON (as its int value) and was coerced back.
        assert hook.stage is GenerationStage.AFTER_GENERATE
        out = gen(make_batch(num_graphs=1))
        assert out.num_graphs == 2

    def test_build_requires_model(self) -> None:
        """``build`` without a model raises ``TypeError`` (weights are external)."""
        with pytest.raises(TypeError, match="requires a model"):
            self._spec().build()

    def test_build_overrides(self) -> None:
        """``build(**overrides)`` forwards runtime overrides to the constructor."""
        gen = self._spec().build(model=torch.nn.Module(), num_samples_per_batch=5)
        assert gen.num_samples_per_batch == 5

    def test_spec_without_funcs_relies_on_model_methods(self) -> None:
        """A spec with no callables needs ``model.generate``/``to_batch``."""
        spec = AtomGeneratorSpec()
        with pytest.raises(TypeError, match="generation source"):
            spec.build(model=torch.nn.Module())


class TestGenerationPipelineSpec:
    """``GenerationPipelineSpec`` round-trip and model injection."""

    def _stage_spec(self, **kwargs) -> AtomGeneratorSpec:
        """Build a minimal generator stage spec.

        Parameters
        ----------
        **kwargs
            Extra :class:`AtomGeneratorSpec` fields.

        Returns
        -------
        AtomGeneratorSpec
            The stage spec.
        """
        return AtomGeneratorSpec(
            generator_func=create_model_spec(make_trivial_generate),
            output_to_batch_func=create_model_spec(make_zeros_recon),
            consumes_fields=kwargs.pop("consumes_fields", []),
            produces_fields=kwargs.pop("produces_fields", ["positions"]),
            **kwargs,
        )

    def test_pipeline_spec_builds_pipeline(self) -> None:
        """Specs build into a working pipeline; models inject in stage order."""
        spec = GenerationPipelineSpec(
            stages=[self._stage_spec(), self._stage_spec(consumes_fields=["positions"])]
        )
        pipe = spec.build(models=[torch.nn.Module(), torch.nn.Module()])
        out = pipe(make_batch(num_graphs=1))
        assert out.num_graphs == 1

    def test_pipeline_spec_json_round_trip(self) -> None:
        """Pipeline specs survive JSON, including non-generator stage specs."""
        spec = GenerationPipelineSpec(
            stages=[
                self._stage_spec(),
                create_model_spec(make_passthrough_stage),
                self._stage_spec(consumes_fields=["positions"]),
            ]
        )
        blob = spec.model_dump_json()
        restored = GenerationPipelineSpec.model_validate_json(blob)
        models = [torch.nn.Module(), torch.nn.Module()]
        pipe = restored.build(models=models)
        assert len(pipe.stages) == 3
        out = pipe(make_batch(num_graphs=2))
        assert out.num_graphs == 2

    def test_pipeline_spec_missing_model_raises(self) -> None:
        """Too few models for the generator stages raises ``TypeError``."""
        spec = GenerationPipelineSpec(
            stages=[self._stage_spec(), self._stage_spec(consumes_fields=["positions"])]
        )
        with pytest.raises(TypeError, match="one model per"):
            spec.build(models=[torch.nn.Module()])


class TestToSpec:
    """``to_spec`` — the reverse direction of spec construction."""

    def _generator(self, **kwargs) -> AtomGenerator:
        """Build a demo generator with spec-able wiring plus overrides.

        Parameters
        ----------
        **kwargs
            Constructor overrides.

        Returns
        -------
        AtomGenerator
            A DemoGANModel-backed generator.
        """
        defaults: dict = {
            "model": DemoGANModel(),
            "generator_func": trivial_generate,
            "output_to_batch_func": zeros_to_batch,
        }
        defaults.update(kwargs)
        return AtomGenerator(**defaults)

    def test_round_trip_full(self) -> None:
        """A fully wired generator round-trips through JSON and rebuilds."""
        gen = self._generator(
            hooks=[ScaleSampleHook(factor=3.0)],
            consumes_fields=frozenset({"positions"}),
            produces_fields=frozenset({"positions", "atomic_numbers"}),
            num_samples_per_batch=2,
            seed=11,
            compile_kwargs={"backend": "eager"},
        )
        blob = gen.to_spec().model_dump_json()
        rebuilt = AtomGeneratorSpec.model_validate_json(blob).build(
            model=DemoGANModel()
        )
        assert rebuilt.generator_func is trivial_generate
        assert rebuilt.output_to_batch_func is zeros_to_batch
        assert rebuilt.num_samples_per_batch == 2
        assert rebuilt.seed == 11
        assert rebuilt.compile_kwargs == {"backend": "eager"}
        assert rebuilt.consumes_fields == frozenset({"positions"})
        assert rebuilt.produces_fields == frozenset({"positions", "atomic_numbers"})
        hook = rebuilt.hooks[0]
        assert isinstance(hook, ScaleSampleHook)
        assert hook.factor == 3.0
        assert hook.stage is GenerationStage.AFTER_GENERATE
        assert rebuilt(make_batch(num_graphs=1)).num_graphs == 2

    def test_round_trip_model_fallbacks(self) -> None:
        """A generator on model ``generate``/``to_batch`` specs no callables."""
        gen = AtomGenerator(model=DemoGANModel(), seed=3)
        spec = gen.to_spec()
        assert spec.generator_func is None
        assert spec.output_to_batch_func is None
        rebuilt = AtomGeneratorSpec.model_validate_json(spec.model_dump_json()).build(
            model=DemoGANModel()
        )
        assert rebuilt(make_batch(num_graphs=2)).num_graphs == 2

    def test_lambda_and_closure_rejected(self) -> None:
        """Callables without a dotted path raise ``TypeError``."""
        gen = self._generator(generator_func=lambda model, **kw: None)
        with pytest.raises(TypeError, match="module-level"):
            gen.to_spec()

        def _closure(model, **kw):
            """A closure is not importable."""

        with pytest.raises(TypeError, match="module-level"):
            self._generator(generator_func=_closure).to_spec()

    def test_partial_rejected(self) -> None:
        """``functools.partial`` has no import path — rejected."""
        gen = self._generator(generator_func=functools.partial(trivial_generate))
        with pytest.raises(TypeError, match="functools.partial"):
            gen.to_spec()

    def test_hook_not_attribute_faithful_rejected(self) -> None:
        """A hook that does not store an ``__init__`` param raises."""

        class _UnfaithfulHook:
            def __init__(self, threshold: float = 1.0) -> None:
                self.stage = GenerationStage.AFTER_GENERATE
                self.frequency = 1

            def __call__(self, ctx, stage) -> None:
                """No-op."""

        gen = self._generator(hooks=[_UnfaithfulHook(threshold=0.5)])
        with pytest.raises(TypeError, match="same-named attribute"):
            gen.to_spec()

    def test_pipeline_round_trip(self) -> None:
        """Pipelines spec their generator stages and callable stages."""
        pipe = AtomGenerator(model=DemoGANModel()) | demo_nonparametric_generation
        blob = pipe.to_spec().model_dump_json()
        rebuilt = GenerationPipelineSpec.model_validate_json(blob).build(
            models=[DemoGANModel()]
        )
        assert len(rebuilt.stages) == 2
        assert isinstance(rebuilt.stages[0], AtomGenerator)
        assert rebuilt.stages[1] is demo_nonparametric_generation
        assert rebuilt(None).num_graphs == 1
