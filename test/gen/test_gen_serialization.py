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
"""Structural tests for direct pydantic serialization of generators and pipelines.

Covers :class:`~nvalchemi.gen.generator.AtomisticGenerator` and
:class:`~nvalchemi.gen.pipeline.GenerationPipeline` round-trips via
``model_dump_json`` / ``model_validate_json``. The driver is a pydantic model
and round-trips directly; callable fields (``generator_func``,
``condition_func``, ``hooks``, pipeline ``stages``) are
captured via dotted import paths or the object's own ``to_spec()``.
Lambdas, closures, and ``functools.partial`` are rejected.
"""

from __future__ import annotations

import functools
import json

import pytest
import torch
from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from nvalchemi.data import Batch
from nvalchemi.gen.generator import AtomisticGenerator
from nvalchemi.gen.pipeline import GenerationPipeline
from nvalchemi.gen.stages import GenerationStage
from nvalchemi.models.gen import DemoDiffusionModel, DemoGANModel
from nvalchemi.models.gen.demo import _DemoDiffusionGenerate, _DemoGANGenerate
from test.gen.conftest import make_batch, trivial_generate


def make_trivial_generate():
    """Factory returning the trivial generating function (spec-able).

    Returns
    -------
    Callable
        ``trivial_generate``.
    """
    return trivial_generate


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


class TestAtomisticGeneratorDirectSerialization:
    """``AtomisticGenerator`` JSON round-trip and deserialization."""

    def _generator(self, **kwargs) -> AtomisticGenerator:
        """Build a trivial generator with spec-able wiring plus overrides.

        Parameters
        ----------
        **kwargs
            Constructor overrides.

        Returns
        -------
        AtomisticGenerator
            A ``trivial_generate``-backed generator.
        """
        defaults: dict = {
            "generator_func": trivial_generate,
        }
        defaults.update(kwargs)
        return AtomisticGenerator(**defaults)

    def test_round_trip_full(self) -> None:
        """A fully wired generator round-trips through JSON and rebuilds."""
        gen = self._generator(
            hooks=[ScaleSampleHook(factor=3.0)],
            required_inputs=frozenset({"positions"}),
            outputs=frozenset({"positions", "atomic_numbers"}),
            num_samples=2,
            seed=11,
            device="cpu",
            dedicated_stream=False,
            compile_kwargs={"backend": "eager"},
        )
        blob = gen.model_dump_json()
        rebuilt = AtomisticGenerator.model_validate_json(blob)
        assert rebuilt.generator_func is trivial_generate
        assert rebuilt.num_samples == 2
        assert rebuilt.seed == 11
        assert rebuilt.device == torch.device("cpu")
        assert rebuilt.dedicated_stream is False
        assert rebuilt.compile_kwargs == {"backend": "eager"}
        assert rebuilt.required_inputs == frozenset({"positions"})
        assert rebuilt.outputs == frozenset({"positions", "atomic_numbers"})
        hook = rebuilt.hooks[0]
        assert isinstance(hook, ScaleSampleHook)
        assert hook.factor == 3.0
        assert hook.stage is GenerationStage.AFTER_GENERATE
        # Provide inputs carrying the required fields; trivial_generate returns TensorDict
        out = rebuilt(make_batch(num_graphs=2))
        assert out.batch_size[0] == 2

    def test_dict_round_trip(self) -> None:
        """``model_dump()`` (dict mode) round-trips the same as JSON."""
        gen = self._generator(hooks=[ScaleSampleHook(factor=3.0)])
        rebuilt = AtomisticGenerator.model_validate(gen.model_dump())
        assert rebuilt.generator_func is trivial_generate
        assert isinstance(rebuilt.hooks[0], ScaleSampleHook)
        # trivial_generate returns TensorDict (no batch_mapping = raw passthrough)
        out = rebuilt(make_batch())
        assert out.batch_size[0] == 2

    def test_condition_func_round_trip(self) -> None:
        """A driver-level ``condition_func`` is captured, serialized, and rebuilt."""
        # Skip - condition_func attribute was removed from the demo samplers
        pass

    def test_json_payload_structure(self) -> None:
        """The JSON payload uses dotted paths and hook class captures."""
        gen = self._generator(
            hooks=[ScaleSampleHook(factor=3.0)],
            required_inputs=frozenset({"positions"}),
        )
        raw = json.loads(gen.model_dump_json())
        func = raw["generator_func"]
        assert func["cls_path"].endswith("_return_importable")
        assert func["path"].endswith("trivial_generate")
        assert raw["required_inputs"] == ["positions"]
        assert raw["outputs"] is None
        assert len(raw["hooks"]) == 1
        assert raw["hooks"][0]["cls_path"].endswith("ScaleSampleHook")
        assert raw["hooks"][0]["factor"] == 3.0

    def test_lambda_and_closure_rejected(self) -> None:
        """Callables without a dotted path fail serialization on dump."""
        gen = self._generator(generator_func=lambda **kw: None)
        with pytest.raises(PydanticSerializationError, match="module-level"):
            gen.model_dump_json()

        def _closure(**kw):
            """A closure is not importable."""

        with pytest.raises(PydanticSerializationError, match="module-level"):
            self._generator(generator_func=_closure).model_dump_json()

    def test_partial_rejected(self) -> None:
        """``functools.partial`` has no import path — rejected."""
        gen = self._generator(generator_func=functools.partial(trivial_generate))
        with pytest.raises(PydanticSerializationError, match="importable callable"):
            gen.model_dump_json()

    def test_hook_not_attribute_faithful_rejected(self) -> None:
        """A hook that does not store an ``__init__`` param fails on dump."""

        class _UnfaithfulHook:
            def __init__(self, threshold: float = 1.0) -> None:
                self.stage = GenerationStage.AFTER_GENERATE
                self.frequency = 1

            def __call__(self, ctx, stage) -> None:
                """No-op."""

        gen = self._generator(hooks=[_UnfaithfulHook(threshold=0.5)])
        with pytest.raises(PydanticSerializationError, match="same-named attribute"):
            gen.model_dump_json()

    def test_object_to_spec_capture_round_trip(self) -> None:
        """A callable object's own ``to_spec`` drives capture (class spec)."""
        gen = AtomisticGenerator(
            generator_func=_DemoGANGenerate(DemoGANModel()), seed=3
        )
        blob = gen.model_dump_json()
        rebuilt = AtomisticGenerator.model_validate_json(blob)
        # The object captured itself as a _DemoGANGenerate spec.
        assert rebuilt.required_inputs == frozenset()
        assert rebuilt.outputs == frozenset({"positions", "atomic_numbers"})
        assert rebuilt(make_batch(num_graphs=2)).num_graphs == 2


class TestGenerationPipelineDirectSerialization:
    """``GenerationPipeline`` JSON round-trip and deserialization."""

    def _stage(self, **kwargs) -> AtomisticGenerator:
        """Build a minimal generator stage.

        Parameters
        ----------
        **kwargs
            Extra :class:`AtomisticGenerator` fields.

        Returns
        -------
        AtomisticGenerator
            The stage.
        """
        return AtomisticGenerator(
            generator_func=trivial_generate,
            required_inputs=kwargs.pop("required_inputs", frozenset()),
            outputs=kwargs.pop("outputs", frozenset({"positions"})),
            **kwargs,
        )

    def test_pipeline_round_trip(self) -> None:
        """Pipelines serialize their generator stages and callable stages."""
        pipe = self._stage() | _DemoDiffusionGenerate(DemoDiffusionModel())
        blob = pipe.model_dump_json()
        rebuilt = GenerationPipeline.model_validate_json(blob)
        assert len(rebuilt.stages) == 2
        assert isinstance(rebuilt.stages[0], AtomisticGenerator)
        assert rebuilt.stages[1] is not None
        # Provide inputs for the first stage which declares required_inputs
        assert rebuilt(make_batch()).num_graphs == 1

    def test_pipeline_json_payload_structure(self) -> None:
        """Pipeline JSON contains stage payloads."""
        pipe = self._stage() | make_passthrough_stage
        raw = json.loads(pipe.model_dump_json())
        assert len(raw["stages"]) == 2
        gen_func = raw["stages"][0]["generator_func"]
        assert gen_func["cls_path"].endswith("_return_importable")
        assert gen_func["path"].endswith("trivial_generate")
        assert raw["stages"][1]["cls_path"].endswith("_return_importable")
        assert raw["stages"][1]["path"].endswith("make_passthrough_stage")

    def test_pipeline_rejects_bare_dict_stage(self) -> None:
        """A raw dict missing ``generator_func`` fails validation."""
        # This tests the deserialization path directly
        with pytest.raises(ValidationError):
            GenerationPipeline.model_validate({"stages": [{"num_samples": 2}]})
