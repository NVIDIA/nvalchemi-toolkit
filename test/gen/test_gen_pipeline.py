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
(``required_inputs`` / ``outputs``). CPU-only, GPU-free, no optional
deps.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen.generator import AtomisticGenerator
from nvalchemi.gen.pipeline import GenerationPipeline
from nvalchemi.gen.stages import GenerationStage
from nvalchemi.models.gen import DemoGANModel
from nvalchemi.models.gen.demo import _DemoGANGenerate
from test.gen.conftest import (
    make_batch,
    trivial_generate,
)


def _generator(
    *,
    consumes: frozenset[str] | None = None,
    produces: frozenset[str] | None = None,
    hooks: list | None = None,
    device: str = "cpu",
    dedicated_stream: bool = True,
) -> AtomisticGenerator:
    """Build a minimal pipeline-ready generator with field declarations.

    Parameters
    ----------
    consumes, produces
        Field declarations (default to empty frozensets, i.e. declared).
    hooks
        Optional generation hooks.
    device
        Device the demo model (and hence the generating function) lives on.
    dedicated_stream
        Forwarded to the constructor (stream opt-out).

    Returns
    -------
    AtomisticGenerator
        A declared generator backed by a demo sampler.
    """
    return AtomisticGenerator(
        generator_func=_DemoGANGenerate(DemoGANModel().to(device)),
        required_inputs=frozenset() if consumes is None else consumes,
        outputs=frozenset() if produces is None else produces,
        hooks=hooks or [],
        dedicated_stream=dedicated_stream,
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
        """``stream`` folds lazily, one pipeline call per input item."""
        pipe = _generator() | _generator()
        inputs = [make_batch(num_graphs=1), make_batch(num_graphs=2)]
        out = list(pipe.stream(inputs))
        assert [b.num_graphs for b in out] == [1, 2]

    def test_stream_caps_with_max_batches(self) -> None:
        """``max_batches`` bounds an input stream."""
        pipe = _generator() | _generator()
        inputs = [make_batch(num_graphs=1)] * 3
        out = list(pipe.stream(inputs, max_batches=2))
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

    def test_zero_graph_materialization_short_circuits(self) -> None:
        """A stage returning zero graphs skips the remaining stages."""
        calls: list = []

        def _empty_generate(inputs=None, *, num_samples=1, rng=None, **kwargs) -> Batch:
            """Return an explicitly empty batch (total rejection)."""
            del inputs, num_samples, rng, kwargs
            return Batch.empty(num_systems=0, num_nodes=0, num_edges=0)

        class _Mark:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Record that this stage ran."""
                calls.append(True)

        gen_empty = AtomisticGenerator(
            generator_func=_empty_generate,
            required_inputs=frozenset(),
            outputs=frozenset(),
        )
        gen_downstream = _generator(hooks=[_Mark()])
        pipe = GenerationPipeline(stages=[gen_empty, gen_downstream])
        out = pipe(make_batch(num_graphs=2))
        assert out.num_graphs == 0
        assert calls == []

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


class TestMappinglessStages:
    """A mapping-less generator yields its raw sample — terminal stages only."""

    def test_mappingless_terminal_stage_returns_raw_sample(self) -> None:
        """A terminal AtomisticGenerator without ``batch_mapping`` ends the fold with
        its raw sample, not a ``Batch``."""
        terminal = AtomisticGenerator(
            generator_func=trivial_generate,
            required_inputs=frozenset(),
            outputs=frozenset(),
        )
        pipe = GenerationPipeline(stages=[_generator(), terminal])
        out = pipe(make_batch(num_graphs=2))
        assert isinstance(out, TensorDict)
        assert out.batch_size[0] == 2

    def test_mappingless_mid_pipeline_feeds_raw_sample_downstream(self) -> None:
        """A mid-pipeline mapping-less stage hands its raw sample to the next
        stage untouched (no validation, no materialization)."""
        seen: list = []

        def _spy(inputs=None, *, num_samples=1, rng=None, **kwargs):
            """Record the inputs received and emit a batch."""
            del num_samples, rng, kwargs
            seen.append(inputs)
            return make_batch(1)

        mappingless = AtomisticGenerator(
            generator_func=trivial_generate,
            required_inputs=frozenset(),
            outputs=frozenset(),
        )
        downstream = AtomisticGenerator(
            generator_func=_spy,
            required_inputs=frozenset(),
            outputs=frozenset(),
        )
        pipe = GenerationPipeline(stages=[mappingless, downstream])
        out = pipe(make_batch(num_graphs=3))
        assert isinstance(out, Batch)
        assert isinstance(seen[0], TensorDict)
        assert seen[0].batch_size[0] == 3


class TestFieldContractValidation:
    """Construction-time validation of AtomisticGenerator stage links."""

    def test_undeclared_generator_stage_raises(self) -> None:
        """A AtomisticGenerator with no declaration source is rejected in a pipeline."""
        undeclared = AtomisticGenerator(
            generator_func=trivial_generate,
        )
        with pytest.raises(
            Exception, match="missing declarations: required_inputs, outputs"
        ):
            GenerationPipeline(stages=[_generator(), undeclared])

    def test_partially_declared_stage_names_the_missing_one(self) -> None:
        """A stage declaring only one direction is told which one is missing."""
        half = AtomisticGenerator(
            generator_func=trivial_generate,
            required_inputs=frozenset(),
        )
        with pytest.raises(Exception, match="missing declarations: outputs"):
            GenerationPipeline(stages=[_generator(), half])

    def test_function_attributes_default_into_pipeline_validation(self) -> None:
        """Declarations carried by the generating function satisfy the contract."""
        declared = AtomisticGenerator(generator_func=_DemoGANGenerate(DemoGANModel()))
        assert declared.required_inputs == frozenset()
        pipe = GenerationPipeline(stages=[declared, _generator()])
        assert isinstance(pipe, GenerationPipeline)

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
        """The first stage reads ``inputs``; its required_inputs are not checked."""
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

        # The non-AtomisticGenerator stage breaks adjacency, so the link is not checked.
        pipe = GenerationPipeline(stages=[producer, passthrough, consumer])
        assert isinstance(pipe, GenerationPipeline)


class TestPipelineSessionAndCompile:
    """Pipeline-level compile orchestration and shared-stream sessions."""

    def test_compile_compiles_generator_stages(self) -> None:
        """``pipe.compile()`` compiles each AtomisticGenerator stage; skips others."""
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
        """``with pipe:`` opens/closes each AtomisticGenerator stage's session."""
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
        # ExitStack unwinds in reverse entry order (context-manager convention)
        assert log == ["enter-a", "enter-b", "exit-b", "exit-a"]

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

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="No CUDA device available."
    )
    def test_session_stream_sharing_respects_dedicated_stream_false(self) -> None:
        """A stage with ``dedicated_stream=False`` stays off the shared stream."""
        gen_a = _generator(device="cuda")
        gen_b = _generator(device="cuda", dedicated_stream=False)
        pipe = GenerationPipeline(stages=[gen_a, gen_b])
        with pipe:
            assert pipe._stream is not None
            assert gen_a._stream is pipe._stream
            assert gen_b._stream is None
            out = pipe(make_batch(num_graphs=1).to("cuda"))
            assert out.num_graphs == 1


# ---------------------------------------------------------------------------
# Run-having stages (dynamics engines, fused stages) and per-stage kwargs
# ---------------------------------------------------------------------------


def _dynamics_batch(num_graphs: int = 2) -> Batch:
    """A minimal integrable batch (forces/energies pre-allocated)."""
    from nvalchemi.data import AtomicData

    batch = Batch.from_data_list(
        [
            AtomicData(
                atomic_numbers=torch.tensor([6, 6], dtype=torch.long),
                positions=torch.randn(2, 3),
            )
            for _ in range(num_graphs)
        ]
    )
    batch.forces = torch.zeros(batch.num_nodes, 3)
    batch.energies = torch.zeros(batch.num_graphs, 1)
    return batch


def _synthetic_structures(num_samples: int, rng=None) -> Batch:
    """Small synthetic point-cloud batch with leaf tensors (dynamics-ready)."""
    positions = torch.rand(num_samples, 3, 3, generator=rng)
    numbers = torch.full((3,), 6, dtype=torch.long)
    return Batch.from_data_list(
        [AtomicData(positions=p, atomic_numbers=numbers) for p in positions]
    )


def _synthetic_source(inputs=None, *, num_samples=1, rng=None, **kwargs) -> Batch:
    """Synthetic-structure source stage for session-less folds."""
    del inputs, kwargs
    return _synthetic_structures(num_samples, rng)


def _nonparametric_cuda(inputs=None, *, num_samples=1, rng=None, **kwargs) -> Batch:
    """A synthetic-structure source with the function-owned device move."""
    del inputs, kwargs
    return _synthetic_structures(num_samples, rng).to("cuda")


class _RunRecorder:
    """Duck-typed run-having stage that records the kwargs it receives."""

    def __init__(self) -> None:
        """Record nothing yet."""
        self.calls: list[dict] = []

    def run(self, batch: Batch, **kwargs) -> Batch:
        """Record the call and pass the batch through."""
        self.calls.append(kwargs)
        return batch


class TestDynamicsStages:
    """Stages with a ``run`` method are driven by it (engines, fused stages)."""

    def test_run_takes_precedence_over_call(self) -> None:
        """A stage with both ``__call__`` and ``run`` is driven by ``run()``."""

        class _Both(_RunRecorder):
            def __call__(self, batch):
                raise AssertionError("__call__ must not fire on a run-having stage")

        engine = _Both()
        pipe = _generator() | engine
        out = pipe(make_batch(num_graphs=1))
        assert len(engine.calls) == 1
        assert isinstance(out, Batch)

    def test_optimizer_stage_runs_to_completion(self) -> None:
        """``gen | optimizer``: the fold drives the engine's own loop."""
        from nvalchemi.dynamics.demo import DemoDynamics
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        gen = AtomisticGenerator(
            generator_func=_synthetic_source,
            required_inputs=frozenset(),
            outputs=frozenset({"positions", "atomic_numbers"}),
        )
        engine = DemoDynamics(model=DemoModelWrapper(DemoModel()), n_steps=3, dt=0.5)
        pipe = gen | engine
        out = pipe(None)
        assert isinstance(out, Batch)
        assert out.num_graphs == 1
        # run() integrated the trajectory; a bare one-step __call__ would leave
        # velocities untouched
        assert not torch.allclose(out.velocities, torch.zeros_like(out.velocities))

    def test_non_batch_before_dynamics_raises(self) -> None:
        """A non-Batch output feeding a dynamics stage raises TypeError."""
        from nvalchemi.dynamics.demo import DemoDynamics
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        gen = AtomisticGenerator(
            generator_func=trivial_generate,  # returns a TensorDict, not a Batch
            required_inputs=frozenset(),
            outputs=frozenset(),
        )
        engine = DemoDynamics(model=DemoModelWrapper(DemoModel()), n_steps=1, dt=0.5)
        pipe = gen | engine
        with pytest.raises(TypeError, match="must return a Batch"):
            pipe(None)


class TestStageKwargs:
    """Per-call options addressed to stages: broadcast or per-stage."""

    def test_single_mapping_stretches(self) -> None:
        """One mapping applies to every stage."""
        seen: list[dict] = []

        class _Probe:
            def __call__(self, batch: Batch, **kwargs) -> Batch:
                seen.append(kwargs)
                return batch

        pipe = _generator() | _Probe()
        out = pipe(None, stage_kwargs={"num_samples": 3})
        assert out.num_graphs == 3  # the generator received num_samples
        assert seen == [{"num_samples": 3}]  # and so did the callable stage

    def test_per_stage_list(self) -> None:
        """A list addresses kwargs per stage; dynamics get their own channel."""
        engine = _RunRecorder()
        pipe = _generator() | engine
        out = pipe(None, stage_kwargs=[{"num_samples": 2}, {"n_steps": 7}])
        assert out.num_graphs == 2
        assert engine.calls == [{"n_steps": 7}]

    def test_none_entries_mean_no_kwargs(self) -> None:
        """``None`` entries pass no kwargs to that stage."""
        engine = _RunRecorder()
        pipe = _generator() | engine
        pipe(None, stage_kwargs=[None, None])
        assert engine.calls == [{}]

    def test_length_mismatch_raises(self) -> None:
        """A per-stage list must match the stage count."""
        pipe = _generator() | _RunRecorder()
        with pytest.raises(ValueError, match="one entry per stage"):
            pipe(None, stage_kwargs=[{}])

    def test_broadcast_into_run_stage_warns(self) -> None:
        """A kwarg a run-stage cannot accept is dropped with a UserWarning."""

        class _StrictRun:
            def run(self, batch: Batch) -> Batch:
                return batch

        pipe = _generator() | _StrictRun()
        with pytest.warns(UserWarning, match="ignores unsupported stage_kwargs"):
            out = pipe(None, stage_kwargs={"num_samples": 2})
        assert out.num_graphs == 2  # the generator still received num_samples


class TestDuckTypedSessions:
    """Context-manager stages are entered inside a pipeline session."""

    def test_context_manager_stage_entered(self) -> None:
        """A stage with ``__enter__``/``__exit__`` is entered and exited."""
        entered: list[bool] = []
        exited: list[bool] = []

        class _Managed:
            _stream = None

            def __enter__(self):
                entered.append(True)
                return self

            def __exit__(self, *args: object) -> None:
                exited.append(True)

            def __call__(self, batch: Batch) -> Batch:
                return batch

        stage = _Managed()
        pipe = _generator() | stage
        with pipe:
            assert entered == [True]
        assert exited == [True]

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="No CUDA device available."
    )
    def test_shared_stream_passes_to_duck_stage(self) -> None:
        """A stage with a ``_stream`` slot runs on the pipeline's stream."""

        class _Managed:
            _stream = None

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                pass

            def __call__(self, batch: Batch) -> Batch:
                return batch

        stage = _Managed()
        pipe = _generator(device="cuda") | stage
        with pipe:
            assert pipe._stream is not None
            assert stage._stream is pipe._stream
            out = pipe(make_batch(num_graphs=1).to("cuda"))
            assert out.num_graphs == 1

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="No CUDA device available."
    )
    def test_dynamics_engine_shares_the_pipeline_stream(self) -> None:
        """``BaseDynamics.__enter__`` honors the pre-set shared stream."""
        from nvalchemi.dynamics.demo import DemoDynamics
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        gen = AtomisticGenerator(
            generator_func=_nonparametric_cuda,
            required_inputs=frozenset(),
            outputs=frozenset({"positions", "atomic_numbers"}),
            device="cuda",
        )
        engine = DemoDynamics(
            model=DemoModelWrapper(DemoModel().to("cuda")), n_steps=1, dt=0.5
        )
        pipe = gen | engine
        with pipe:
            assert engine._stream is pipe._stream
            assert engine._stream is not None
            out = pipe(None)
            assert isinstance(out, Batch)
            assert out.positions.device.type == "cuda"


class TestBoundaryGuardCompileSkip:
    """The boundary guard stands down under ``torch.compile``."""

    def test_guard_skipped_under_compile(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With is_compiling True, a non-Batch into a dynamics stage is not the guard's error."""
        from nvalchemi.dynamics.demo import DemoDynamics
        from nvalchemi.models.demo import DemoModel, DemoModelWrapper

        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        gen = AtomisticGenerator(
            generator_func=trivial_generate,  # TensorDict, not a Batch
            required_inputs=frozenset(),
            outputs=frozenset(),
        )
        engine = DemoDynamics(model=DemoModelWrapper(DemoModel()), n_steps=1, dt=0.5)
        pipe = gen | engine
        with pytest.raises(Exception) as exc_info:
            pipe(None)
        assert "must return a Batch" not in str(exc_info.value)


class TestStreamStageKwargsForwarding:
    """``stream()`` forwards ``stage_kwargs`` into every fold."""

    def test_stream_forwards_stage_kwargs(self) -> None:
        engine = _RunRecorder()
        pipe = _generator() | engine
        outs = list(
            pipe.stream([None, None], stage_kwargs=[{"num_samples": 2}, {"n_steps": 7}])
        )
        assert all(out.num_graphs == 2 for out in outs)
        assert engine.calls == [{"n_steps": 7}, {"n_steps": 7}]

    def test_stretched_copy_survives_a_popping_stage(self) -> None:
        """A stage popping ``rng`` must not starve the next stage's copy."""

        class _Popper:
            def __call__(self, batch: Batch, rng=None) -> Batch:
                assert rng is not None
                return batch

        engine = _RunRecorder()
        pipe = _generator() | _Popper() | engine
        rng = torch.Generator().manual_seed(0)
        # the generator stage pops rng from its own copy; the engine must still see it
        out = pipe(None, stage_kwargs=[{"rng": rng}, {"rng": rng}, {"rng": rng}])
        assert engine.calls == [{"rng": rng}]
        assert isinstance(out, Batch)
