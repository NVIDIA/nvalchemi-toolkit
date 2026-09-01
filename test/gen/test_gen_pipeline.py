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
"""Structural tests for :class:`~nvalchemi.gen.pipeline.GenerationPipeline`.

Covers the ``|`` composition sugar, fold/stream semantics, empty-batch
short-circuiting, and construction-time field-contract validation
(``consumes_fields`` / ``produces_fields``). CPU-only, GPU-free, no optional
deps.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import Batch
from nvalchemi.gen.generator import AtomGenerator
from nvalchemi.gen.pipeline import GenerationPipeline
from nvalchemi.gen.stages import GenerationStage
from nvalchemi.models.gen.demo import DemoGANModel
from test.gen.conftest import make_batch, trivial_generate, zeros_to_batch


def _generator(
    *,
    consumes: frozenset[str] | None = None,
    produces: frozenset[str] | None = None,
    hooks: list | None = None,
    device: str = "cpu",
) -> AtomGenerator:
    """Build a minimal pipeline-ready generator with field declarations.

    Parameters
    ----------
    consumes, produces
        Field declarations (default to empty frozensets, i.e. declared).
    hooks
        Optional generation hooks.

    Returns
    -------
    AtomGenerator
        A declared, ``DemoGANModel``-backed generator.
    """
    return AtomGenerator(
        model=DemoGANModel().to(device),
        consumes_fields=frozenset() if consumes is None else consumes,
        produces_fields=frozenset() if produces is None else produces,
        hooks=hooks or [],
    )


class _KeepFirst:
    """Hook that keeps only the first graph at AFTER_GENERATE."""

    stage = GenerationStage.AFTER_GENERATE
    frequency = 1

    def __call__(self, ctx, stage) -> None:
        """Subset the batch to its first graph."""
        ctx.batch = ctx.batch[[0]]


class TestCompositionSugar:
    """``|`` operator behavior."""

    def test_generator_or_generator_builds_pipeline(self) -> None:
        """``gen_a | gen_b`` is a two-stage pipeline."""
        gen_a, gen_b = _generator(), _generator()
        pipe = gen_a | gen_b
        assert isinstance(pipe, GenerationPipeline)
        assert pipe.stages == [gen_a, gen_b]

    def test_generator_or_pipeline_prepends(self) -> None:
        """``gen | pipe`` prepends the generator."""
        gen_a, gen_b, gen_c = _generator(), _generator(), _generator()
        pipe = gen_a | (gen_b | gen_c)
        assert pipe.stages == [gen_a, gen_b, gen_c]

    def test_pipeline_or_stage_appends(self) -> None:
        """``pipe | stage`` appends."""
        gen_a, gen_b = _generator(), _generator()
        tagger_calls = []

        def tagger(batch: Batch) -> Batch:
            """Record a call and pass the batch through."""
            tagger_calls.append(batch.num_graphs)
            return batch

        pipe = (gen_a | gen_b) | tagger
        assert len(pipe.stages) == 3
        pipe(make_batch(num_graphs=1))
        assert tagger_calls == [1]


class TestFoldAndStream:
    """Fold and stream semantics."""

    def test_call_folds_stages(self, device: str) -> None:
        """Stage 2 receives stage 1's output batch."""
        gen_a = _generator(produces=frozenset({"positions"}), device=device)
        gen_b = _generator(consumes=frozenset({"positions"}), device=device)
        pipe = GenerationPipeline(stages=[gen_a, gen_b])
        out = pipe(make_batch(num_graphs=2).to(device))
        assert isinstance(out, Batch)
        assert out.num_graphs == 2
        assert out["positions"].device.type == device

    def test_stream_mirrors_generator_stream(self) -> None:
        """``stream`` folds lazily, one pipeline call per cond item."""
        pipe = _generator() | _generator()
        conds = [make_batch(num_graphs=1), make_batch(num_graphs=2)]
        out = list(pipe.stream(conds))
        assert [b.num_graphs for b in out] == [1, 2]

    def test_stream_caps_with_max_batches(self) -> None:
        """``max_batches`` bounds a cond stream."""
        pipe = _generator() | _generator()
        conds = [make_batch(num_graphs=1)] * 3
        out = list(pipe.stream(conds, max_batches=2))
        assert len(out) == 2

    def test_subset_filter_flows_through_pipeline(self) -> None:
        """A mid-pipeline filter shrinks the batch downstream stages see."""
        calls: list = []

        class _Mark:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Record the batch size this stage produced."""
                calls.append(ctx.batch.num_graphs)

        gen_filter = _generator(hooks=[_KeepFirst()])
        gen_downstream = _generator(hooks=[_Mark()])
        pipe = GenerationPipeline(stages=[gen_filter, gen_downstream])
        out = pipe(make_batch(num_graphs=3))
        assert out.num_graphs == 1
        assert calls == [1]

    def test_filter_to_empty_raises(self) -> None:
        """Filtering to zero graphs raises ``IndexError`` (data-layer limit)."""

        class _RejectAll:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Reject every graph."""
                ctx.batch = ctx.batch[
                    torch.zeros(ctx.batch.num_graphs, dtype=torch.bool)
                ]

        pipe = GenerationPipeline(
            stages=[_generator(hooks=[_RejectAll()]), _generator()]
        )
        with pytest.raises(IndexError, match="Index is empty"):
            pipe(make_batch(num_graphs=2))

    def test_per_stage_hooks_are_isolated(self) -> None:
        """Each stage's hooks see that stage's own context."""
        seen: list = []

        class _Mark:
            def __init__(self, name: str) -> None:
                self.name = name
                self.stage = GenerationStage.AFTER_GENERATE
                self.frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Record the stage's workflow identity."""
                seen.append((self.name, ctx.workflow))

        gen_a = _generator(hooks=[_Mark("a")])
        gen_b = _generator(hooks=[_Mark("b")])
        pipe = GenerationPipeline(stages=[gen_a, gen_b])
        pipe(make_batch(num_graphs=1))
        assert seen[0][1] is gen_a
        assert seen[1][1] is gen_b


class TestFieldContractValidation:
    """Construction-time validation of AtomGenerator stage links."""

    def test_undeclared_generator_stage_raises(self) -> None:
        """A AtomGenerator with no declaration source is rejected in a pipeline."""
        undeclared = AtomGenerator(
            model=torch.nn.Module(),
            generator_func=trivial_generate,
            output_to_batch_func=zeros_to_batch,
        )
        with pytest.raises(ValueError, match="declares neither"):
            GenerationPipeline(stages=[_generator(), undeclared])

    def test_missing_upstream_field_raises(self) -> None:
        """``consumes ⊄ upstream produces`` fails fast at construction."""
        producer = _generator(produces=frozenset({"positions"}))
        consumer = _generator(consumes=frozenset({"charges"}))
        with pytest.raises(ValueError, match="charges"):
            GenerationPipeline(stages=[producer, consumer])

    def test_satisfied_link_passes(self) -> None:
        """A declared, covered link constructs cleanly."""
        producer = _generator(produces=frozenset({"charges", "positions"}))
        consumer = _generator(consumes=frozenset({"charges"}))
        pipe = GenerationPipeline(stages=[producer, consumer])
        assert isinstance(pipe, GenerationPipeline)

    def test_first_stage_consumes_unvalidated(self) -> None:
        """The first stage reads ``cond``; its consumes_fields are not checked."""
        first = _generator(consumes=frozenset({"anything"}))
        pipe = GenerationPipeline(stages=[first, _generator()])
        assert isinstance(pipe, GenerationPipeline)

    def test_non_generator_stage_unvalidated(self) -> None:
        """A Batch -> Batch callable between Generators carries no contract."""
        producer = _generator(produces=frozenset({"positions"}))
        consumer = _generator(consumes=frozenset({"charges"}))

        def passthrough(batch: Batch) -> Batch:
            """Identity stage with no field declarations."""
            return batch

        # The non-AtomGenerator stage breaks adjacency, so the link is not checked.
        pipe = GenerationPipeline(stages=[producer, passthrough, consumer])
        assert isinstance(pipe, GenerationPipeline)


class TestPipelineSessionAndCompile:
    """Pipeline-level compile orchestration and shared-stream sessions."""

    def test_compile_compiles_generator_stages(self) -> None:
        """``pipe.compile()`` compiles each AtomGenerator stage; skips others."""
        gen_a, gen_b = _generator(), _generator()

        def passthrough(batch: Batch) -> Batch:
            """Identity stage (not compilable by the pipeline)."""
            return batch

        pipe = GenerationPipeline(stages=[gen_a, passthrough, gen_b])
        out = pipe.compile(backend="eager")
        assert out is pipe
        assert gen_a._compiled_generate is not None
        assert gen_b._compiled_generate is not None

    def test_session_enters_generator_stages(self) -> None:
        """``with pipe:`` opens/closes each AtomGenerator stage's session."""
        log: list = []

        class _CMHook:
            def __init__(self, name: str) -> None:
                self.name = name
                self.stage = GenerationStage.AFTER_GENERATE
                self.frequency = 1

            def __enter__(self) -> None:
                """Record entry."""
                log.append(f"enter-{self.name}")

            def __exit__(self, *args) -> None:
                """Record exit."""
                log.append(f"exit-{self.name}")

            def __call__(self, ctx, stage) -> None:
                """No-op."""

        gen_a = _generator(hooks=[_CMHook("a")])
        gen_b = _generator(hooks=[_CMHook("b")])
        pipe = GenerationPipeline(stages=[gen_a, gen_b])
        with pipe:
            out = pipe(make_batch(num_graphs=1))
            assert out.num_graphs == 1
            # CPU: no stream anywhere.
            assert pipe._stream is None and gen_a._stream is None
        assert log == ["enter-a", "enter-b", "exit-a", "exit-b"]

    def test_session_lazy_compiles_marked_stages(self) -> None:
        """A stage with ``compile_generate=True`` compiles at pipeline entry."""
        gen_a = _generator()
        gen_b = _generator()
        gen_b.compile_generate = True
        gen_b.compile_kwargs = {"backend": "eager"}
        pipe = GenerationPipeline(stages=[gen_a, gen_b])
        assert gen_b._compiled_generate is None
        with pipe:
            assert gen_a._compiled_generate is None
            assert gen_b._compiled_generate is not None

    def test_session_stream_sharing_matches_device(self, device: str) -> None:
        """Stages share the pipeline's one stream on CUDA; no streams on CPU."""
        gen_a = _generator(device=device)
        gen_b = _generator(device=device)
        pipe = GenerationPipeline(stages=[gen_a, gen_b])
        with pipe:
            out = pipe(make_batch(num_graphs=1).to(device))
            assert out.num_graphs == 1
            if device == "cuda":
                assert pipe._stream is not None
                assert gen_a._stream is pipe._stream
                assert gen_b._stream is pipe._stream
            else:
                assert pipe._stream is None and gen_a._stream is None
        assert pipe._stream is None and gen_a._stream is None
