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
"""Structural tests for the generative API.

Covers the abstract :class:`~nvalchemi.gen.generator.AtomGenerator`
with its fixed two-step core and
:class:`~nvalchemi.gen.stages.GenerationStage` hooks: the defaults-everywhere
model contract, the :class:`~tensordict.TensorDict` sample contract, hook
firing order / frequency gating /
mutation-by-replacement / filter-by-subsetting, ``stream()`` semantics, the
``sample()``/``__call__`` sugar split, the ``torch.compile`` surface, and
session (context-manager) lifecycle. CPU-only, GPU-free, no optional deps.
"""

from __future__ import annotations

import itertools
from enum import Enum, auto

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from nvalchemi.data import Batch
from nvalchemi.gen.generator import AtomGenerator
from nvalchemi.gen.stages import GenerationStage
from nvalchemi.models.gen.demo import DemoDiffusionModel, DemoGANModel
from test.gen.conftest import make_batch, trivial_generate, zeros_to_batch


class _PlainModel(nn.Module):
    """A bare ``nn.Module`` — no mixin, no ``model_config``, no methods.

    Exercises the defaults-everywhere contract: the :class:`AtomGenerator`
    falls back to the module-level default condition and never reads
    ``forward``.
    """


def _rng_generate(model, *, num_samples=1, rng=None, cond=None, **kwargs):
    """A generating function that draws from ``rng`` (seed-path tests).

    Parameters
    ----------
    model
        Generative model (ignored).
    num_samples
        Number of draws (used only when ``cond`` is ``None``).
    rng
        :class:`torch.Generator` to draw from.
    cond
        Conditioning batch, if any.
    **kwargs
        Family-specific options (ignored).

    Returns
    -------
    TensorDict
        Random values under the ``"x1"`` key.
    """
    del model, kwargs
    n = cond.num_graphs if isinstance(cond, Batch) else num_samples
    return TensorDict({"x1": torch.randn(n, 1, 3, generator=rng)}, batch_size=[n])


class _CaptureRecon:
    """Materialization that records the raw sample TensorDict it receives."""

    def __init__(self) -> None:
        self.samples: list[TensorDict] = []

    def __call__(self, sample: TensorDict, batch) -> Batch:
        """Record ``sample`` and return a batch sized like it.

        Parameters
        ----------
        sample
            Sample TensorDict to record.
        batch
            Conditioning batch (ignored).

        Returns
        -------
        Batch
            Dummy graphs matching the sample's leading size.
        """
        self.samples.append(sample)
        return make_batch(sample.batch_size[0])


class TestBaseGenerator:
    """Core :class:`AtomGenerator` tests."""

    def test_unconditional_generate_returns_batch(self) -> None:
        """A free generating function + ``output_to_batch_func`` yields a batch."""
        sentinel = make_batch(num_graphs=1)

        def recon(sample: TensorDict, batch) -> Batch:
            """Reconstruction override returning a sentinel batch.

            Parameters
            ----------
            sample
                Sample TensorDict (ignored).
            batch
                Conditioning batch (ignored).

            Returns
            -------
            Batch
                The sentinel batch.
            """
            del sample, batch
            return sentinel

        gen = AtomGenerator(
            model=_PlainModel(),
            generator_func=trivial_generate,
            output_to_batch_func=recon,
        )
        out = gen(num_samples_per_batch=2)
        assert out is sentinel

    def test_conditional_generate_via_model_generate(self, device: str) -> None:
        """A ``model.generate`` fallback returns a batch sized by conditioning."""
        gen = AtomGenerator(model=DemoGANModel().to(device), num_samples_per_batch=3)
        out = gen(make_batch(num_graphs=2).to(device))
        assert isinstance(out, Batch)
        assert out.num_graphs == 6
        assert out["positions"].device.type == device

    def test_construct_raises_without_generation_source(self) -> None:
        """``TypeError`` at construction with no ``generator_func`` or ``generate``."""
        with pytest.raises(TypeError, match="generation source"):
            AtomGenerator(model=_PlainModel())

    def test_construct_raises_without_materialization(self) -> None:
        """``TypeError`` at construction with no ``output_to_batch_func``/``to_batch``."""
        with pytest.raises(TypeError, match="materialization target"):
            AtomGenerator(model=_PlainModel(), generator_func=trivial_generate)

    def test_generate_must_return_tensordict(self) -> None:
        """A generating function returning a bare tensor raises ``TypeError``."""

        def _bad_generate(model, *, num_samples=1, rng=None, cond=None, **kwargs):
            """Return a bare tensor, violating the TensorDict contract."""
            del model, num_samples, rng, cond, kwargs
            return torch.zeros(1, 1, 3)

        gen = AtomGenerator(
            model=_PlainModel(),
            generator_func=_bad_generate,
            output_to_batch_func=zeros_to_batch,
        )
        with pytest.raises(TypeError, match="TensorDict"):
            gen()

    def test_materialization_must_return_batch(self) -> None:
        """A materialization callable returning a non-Batch raises ``TypeError``."""
        gen = AtomGenerator(
            model=_PlainModel(),
            generator_func=trivial_generate,
            output_to_batch_func=lambda sample, batch: sample,
        )
        with pytest.raises(TypeError, match="must return a Batch"):
            gen()

    def test_condition_default_tiles_for_plain_model(self) -> None:
        """A model without ``condition`` gets the module-level passthrough+tile."""
        gen = AtomGenerator(
            model=_PlainModel(),
            generator_func=trivial_generate,
            output_to_batch_func=lambda sample, batch: batch,
            num_samples_per_batch=3,
        )
        out = gen(make_batch(num_graphs=2))
        assert out.num_graphs == 6

    def test_field_declarations_default_from_config(self) -> None:
        """``consumes_fields``/``produces_fields`` default from ``model_config``."""
        gen = AtomGenerator(model=DemoGANModel())
        assert gen.consumes_fields == frozenset()
        assert gen.produces_fields == frozenset({"positions", "atomic_numbers"})

    def test_field_declarations_explicit_override(self) -> None:
        """Explicit declarations win over ``model_config`` defaults."""
        gen = AtomGenerator(
            model=DemoGANModel(),
            consumes_fields=frozenset({"charges"}),
        )
        assert gen.consumes_fields == frozenset({"charges"})
        assert gen.produces_fields == frozenset({"positions", "atomic_numbers"})

    def test_field_declarations_none_without_config(self) -> None:
        """A model with no ``model_config`` leaves declarations undeclared."""
        gen = AtomGenerator(
            model=_PlainModel(),
            generator_func=trivial_generate,
            output_to_batch_func=zeros_to_batch,
        )
        assert gen.consumes_fields is None
        assert gen.produces_fields is None


class TestGenerationHooks:
    """Hook dispatch, mutation, and filtering on the three generation stages."""

    def _generator(self, hooks: list) -> AtomGenerator:
        """Build a demo generator carrying ``hooks``.

        Parameters
        ----------
        hooks
            Hooks to register.

        Returns
        -------
        AtomGenerator
            A ``DemoGANModel``-backed generator.
        """
        return AtomGenerator(model=DemoGANModel(), hooks=hooks)

    def test_hook_firing_order(self) -> None:
        """Hooks fire once per call, in pipeline order."""

        class _Recorder:
            def __init__(self, stage: GenerationStage, log: list) -> None:
                self.stage = stage
                self.frequency = 1
                self._log = log

            def __call__(self, ctx, stage) -> None:
                """Record the dispatched stage."""
                self._log.append(stage)

        log: list = []
        gen = self._generator([_Recorder(stage, log) for stage in GenerationStage])
        gen()
        assert log == [
            GenerationStage.BEFORE_CONDITION,
            GenerationStage.AFTER_CONDITION,
            GenerationStage.AFTER_GENERATE,
        ]

    def test_hook_frequency_gating_across_stream(self) -> None:
        """A ``frequency=2`` hook fires every other generation call."""

        class _EveryOther:
            def __init__(self, log: list) -> None:
                self.stage = GenerationStage.AFTER_GENERATE
                self.frequency = 2
                self._log = log

            def __call__(self, ctx, stage) -> None:
                """Record the step count seen."""
                self._log.append(ctx.step_count)

        log: list = []
        gen = self._generator([_EveryOther(log)])
        list(gen.stream([None] * 4))
        assert log == [0, 2]

    def test_hook_mutation_replaces_batch(self) -> None:
        """An ``AFTER_GENERATE`` hook replaces the materialized batch."""

        sentinel = make_batch(num_graphs=1)

        class _Swap:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Swap the generated batch for a sentinel."""
                ctx.batch = sentinel

        gen = self._generator([_Swap()])
        out = gen(make_batch(num_graphs=3))
        assert out is sentinel

    def test_intermediates_scratch_between_stages(self) -> None:
        """``ctx.intermediates`` carries hook-to-hook state within one call."""
        seen: list = []

        class _Producer:
            stage = GenerationStage.AFTER_CONDITION
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Stash a value for a later stage."""
                ctx.intermediates["tag"] = "from-condition"

        class _Consumer:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Read the stashed value."""
                seen.append(ctx.intermediates.get("tag"))

        gen = self._generator([_Producer(), _Consumer()])
        gen()
        assert seen == ["from-condition"]

    def test_cond_rewrite_before_condition(self) -> None:
        """A BEFORE_CONDITION hook replaces ``ctx.cond`` before conditioning."""

        class _SwapCond:
            stage = GenerationStage.BEFORE_CONDITION
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Replace a 3-graph cond with a 1-graph cond."""
                ctx.cond = make_batch(num_graphs=1)

        gen = AtomGenerator(
            model=_PlainModel(),
            generator_func=trivial_generate,
            output_to_batch_func=lambda sample, batch: batch,
            hooks=[_SwapCond()],
        )
        out = gen(make_batch(num_graphs=3))
        assert out.num_graphs == 1

    def test_filter_hook_subsets_batch(self, device: str) -> None:
        """Filtering is graph-level subsetting at AFTER_GENERATE."""

        class _KeepFirst:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Keep only the first graph."""
                ctx.batch = ctx.batch[[0]]

        gen = self._generator([_KeepFirst()])
        gen.model.to(device)
        out = gen(make_batch(num_graphs=3).to(device))
        assert out.num_graphs == 1
        assert out["positions"].device.type == device

    def test_filter_to_empty_raises(self) -> None:
        """A filter rejecting every graph raises ``IndexError`` from ``Batch``.

        Current data-layer behavior: zero-graph selections are not supported.
        If empty-batch semantics land in ``Batch``, this test flips to assert
        the empty batch is yielded.
        """

        class _RejectAll:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Reject every graph with an all-False mask."""
                mask = torch.zeros(ctx.batch.num_graphs, dtype=torch.bool)
                ctx.batch = ctx.batch[mask]

        gen = self._generator([_RejectAll()])
        with pytest.raises(IndexError, match="Index is empty"):
            gen(make_batch(num_graphs=3))

    def test_wrong_stage_enum_rejected(self) -> None:
        """A hook with a non-GenerationStage stage is rejected at construction."""

        class _OtherStage(Enum):
            AFTER_STEP = auto()

        class _WrongStage:
            stage = _OtherStage.AFTER_STEP
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """No-op."""

        with pytest.raises(TypeError, match="only accepts"):
            self._generator([_WrongStage()])

    def test_missing_stage_rejected(self) -> None:
        """A hook with ``stage=None`` is rejected at construction."""

        class _NoStage:
            stage = None
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """No-op."""

        with pytest.raises(TypeError, match="no stage"):
            self._generator([_NoStage()])

    def test_string_stage_coerced(self) -> None:
        """A string stage (e.g. spec-rehydrated) is coerced to GenerationStage."""

        class _StringStage:
            def __init__(self) -> None:
                self.stage = "AFTER_GENERATE"
                self.frequency = 1
                self.calls = 0

            def __call__(self, ctx, stage) -> None:
                """Count firings."""
                self.calls += 1

        hook = _StringStage()
        gen = self._generator([hook])
        assert hook.stage is GenerationStage.AFTER_GENERATE
        gen()
        assert hook.calls == 1


class TestStreaming:
    """``stream()`` semantics: cond iteration, caps, filtered batches, seeds."""

    def _generator(self, **kwargs) -> AtomGenerator:
        """Build a minimal streaming generator.

        Parameters
        ----------
        **kwargs
            Extra constructor arguments.

        Returns
        -------
        AtomGenerator
            A ``DemoGANModel``-backed generator (extra kwargs forwarded).
        """
        return AtomGenerator(model=DemoGANModel(), **kwargs)

    def test_stream_caps_with_max_batches(self) -> None:
        """``stream(None, max_batches=3)`` yields exactly 3 unconditional draws."""
        gen = self._generator()
        batches = list(gen.stream(max_batches=3))
        assert len(batches) == 3
        assert gen.step_count == 3

    def test_stream_iterates_conds(self) -> None:
        """Each cond item drives exactly one call."""
        gen = self._generator()
        conds = [make_batch(num_graphs=1), make_batch(num_graphs=2)]
        out = list(gen.stream(conds))
        assert [b.num_graphs for b in out] == [1, 2]
        assert gen.step_count == 2

    def test_stream_yields_filtered_batches_as_produced(self) -> None:
        """A subsetting filter shrinks each streamed batch; nothing is dropped."""

        class _KeepFirst:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Keep only the first graph."""
                ctx.batch = ctx.batch[[0]]

        gen = self._generator(hooks=[_KeepFirst()])
        out = list(gen.stream([make_batch(num_graphs=2), make_batch(num_graphs=3)]))
        assert [b.num_graphs for b in out] == [1, 1]

    def test_iter_is_stream_sugar(self) -> None:
        """``__iter__`` streams unconditional draws; callers bound externally."""
        gen = self._generator()
        batches = list(itertools.islice(gen, 2))
        assert len(batches) == 2

    def test_seed_makes_streams_reproducible(self) -> None:
        """Two generators with the same ``seed`` produce identical streams."""
        recon_a, recon_b = _CaptureRecon(), _CaptureRecon()
        gen_a = self._generator(
            generator_func=_rng_generate, output_to_batch_func=recon_a, seed=7
        )
        gen_b = self._generator(
            generator_func=_rng_generate, output_to_batch_func=recon_b, seed=7
        )
        list(gen_a.stream(max_batches=2))
        list(gen_b.stream(max_batches=2))
        for sample_a, sample_b in zip(recon_a.samples, recon_b.samples):
            assert torch.equal(sample_a["x1"], sample_b["x1"])
        # Per-draw seeds (seed + step_count) make consecutive draws differ.
        assert not torch.equal(recon_a.samples[0]["x1"], recon_a.samples[1]["x1"])


class TestSampleSugar:
    """``sample()`` is the real method; ``__call__`` delegates to it."""

    def test_call_matches_sample(self) -> None:
        """``__call__`` and ``sample`` produce identical behavior."""
        capture_a, capture_b = _CaptureRecon(), _CaptureRecon()
        gen_a = AtomGenerator(
            model=DemoGANModel(),
            generator_func=_rng_generate,
            output_to_batch_func=capture_a,
            seed=5,
        )
        gen_b = AtomGenerator(
            model=DemoGANModel(),
            generator_func=_rng_generate,
            output_to_batch_func=capture_b,
            seed=5,
        )
        out_a = gen_a(make_batch(num_graphs=2))
        out_b = gen_b.sample(make_batch(num_graphs=2))
        assert out_a.num_graphs == out_b.num_graphs
        assert torch.equal(capture_a.samples[0]["x1"], capture_b.samples[0]["x1"])

    def test_sample_runs_full_dispatch(self) -> None:
        """``sample()`` fires all three stages in pipeline order."""

        class _Recorder:
            def __init__(self, stage: GenerationStage, log: list) -> None:
                self.stage = stage
                self.frequency = 1
                self._log = log

            def __call__(self, ctx, stage) -> None:
                """Record the dispatched stage."""
                self._log.append(stage)

        log: list = []
        gen = AtomGenerator(
            model=DemoGANModel(),
            hooks=[_Recorder(stage, log) for stage in GenerationStage],
        )
        gen.sample()
        assert log == list(GenerationStage)

    def test_stream_calls_sample(self) -> None:
        """``stream()`` drives ``sample()`` once per cond item."""
        gen = AtomGenerator(model=DemoGANModel())
        calls = 0
        original = gen.sample

        def _spy(*args, **kwargs):
            """Count calls and delegate."""
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        object.__setattr__(gen, "sample", _spy)
        list(gen.stream([make_batch(num_graphs=1)] * 3))
        assert calls == 3


class TestCompile:
    """The ``torch.compile`` surface (``backend="eager"`` keeps CPU tests fast)."""

    def _generator(self, **kwargs) -> AtomGenerator:
        """Build a trivial generator with compile-related kwargs.

        Parameters
        ----------
        **kwargs
            Extra constructor arguments.

        Returns
        -------
        AtomGenerator
            A ``DemoGANModel``-backed generator.
        """
        return AtomGenerator(model=DemoGANModel(), **kwargs)

    def test_compile_wraps_generator_func(self) -> None:
        """``compile()`` wraps the generating function and sets the flag."""
        gen = self._generator()
        assert gen._compiled_generate is None
        out = gen.compile(backend="eager")
        assert out is gen
        assert gen.compile_generate is True
        assert gen._compiled_generate is not None
        assert gen(make_batch(num_graphs=1)).num_graphs == 1

    def test_compile_merges_kwargs(self) -> None:
        """Call-time kwargs merge over constructor ``compile_kwargs``."""
        gen = self._generator(compile_kwargs={"backend": "eager", "dynamic": True})
        gen.compile(dynamic=False)
        assert gen.compile_kwargs == {"backend": "eager", "dynamic": False}

    def test_compile_model_generate_fallback(self) -> None:
        """With no ``generator_func``, ``compile()`` wraps ``model.generate``."""
        gen = AtomGenerator(model=DemoDiffusionModel())
        gen.compile(backend="eager")
        assert gen(make_batch(num_graphs=2)).num_graphs == 2

    def test_lazy_compile_at_session_entry(self) -> None:
        """``compile_generate=True`` defers compilation to session entry."""
        gen = self._generator(
            compile_generate=True, compile_kwargs={"backend": "eager"}
        )
        assert gen._compiled_generate is None
        with gen:
            assert gen._compiled_generate is not None
            assert gen(make_batch(num_graphs=1)).num_graphs == 1

    def test_compile_kwargs_validated_against_torch_compile(self) -> None:
        """Unknown compile kwargs raise ``ValueError`` at construction."""
        with pytest.raises(ValueError, match="not keyword arguments of"):
            self._generator(compile_kwargs={"bogus_kwarg": True})

    def test_compile_kwargs_valid_keys_accepted(self) -> None:
        """Real torch.compile kwargs (e.g. ``fullgraph``/``backend``) pass."""
        gen = self._generator(compile_kwargs={"fullgraph": False, "backend": "eager"})
        assert gen.compile_kwargs == {"fullgraph": False, "backend": "eager"}

    def test_compile_kwargs_rejects_model_key(self) -> None:
        """``model`` is the compile target, not a user kwarg."""
        with pytest.raises(ValueError, match="'model'"):
            self._generator(compile_kwargs={"model": nn.Linear(1, 1)})

    def test_compile_method_validates_merged_kwargs(self) -> None:
        """``compile(**kwargs)`` applies the same validation after merging."""
        gen = self._generator()
        with pytest.raises(ValueError, match="not keyword arguments of"):
            gen.compile(bogus_kwarg=True)


class TestSession:
    """``with gen:`` — stream, session RNG, and hook lifecycle."""

    def _generator(self, **kwargs) -> AtomGenerator:
        """Build a trivial generator with session-related kwargs.

        Parameters
        ----------
        **kwargs
            Extra constructor arguments.

        Returns
        -------
        AtomGenerator
            RNG-drawing generator with a capture recon.
        """
        kwargs.setdefault("generator_func", _rng_generate)
        kwargs.setdefault("output_to_batch_func", _CaptureRecon())
        return AtomGenerator(model=DemoGANModel(), **kwargs)

    def test_session_stream_matches_model_device(self, device: str) -> None:
        """A CUDA-resident model gets a dedicated session stream; CPU does not."""
        gen = self._generator()
        gen.model.to(device)
        with gen:
            if device == "cuda":
                assert gen._stream is not None
                assert gen._stream_ctx is not None
            else:
                assert gen._stream is None
                assert gen._stream_ctx is None
            assert gen._session_rng is None  # no seed set
        assert gen._stream is None

    def test_context_manager_hooks_open_and_close(self) -> None:
        """Hooks with ``__enter__``/``__exit__`` wrap the session."""
        log: list = []

        class _CMHook:
            def __init__(self) -> None:
                self.stage = GenerationStage.AFTER_GENERATE
                self.frequency = 1

            def __enter__(self) -> None:
                """Record session entry."""
                log.append("enter")

            def __exit__(self, *args) -> None:
                """Record session exit."""
                log.append("exit")

            def __call__(self, ctx, stage) -> None:
                """No-op."""

        gen = self._generator(hooks=[_CMHook()])
        with gen:
            gen.sample()
        assert log == ["enter", "exit"]

    def test_session_rng_reproducible_and_advancing(self, device: str) -> None:
        """Same seed → identical sessions; draws advance within a session."""
        recon_a, recon_b = _CaptureRecon(), _CaptureRecon()
        gen_a = self._generator(output_to_batch_func=recon_a, seed=11)
        gen_b = self._generator(output_to_batch_func=recon_b, seed=11)
        gen_a.model.to(device)
        gen_b.model.to(device)
        with gen_a:
            gen_a.sample(make_batch(num_graphs=1).to(device))
            gen_a.sample(make_batch(num_graphs=1).to(device))
        with gen_b:
            gen_b.sample(make_batch(num_graphs=1).to(device))
            gen_b.sample(make_batch(num_graphs=1).to(device))
        for sa, sb in zip(recon_a.samples, recon_b.samples):
            assert torch.equal(sa["x1"], sb["x1"])
        assert not torch.equal(recon_a.samples[0]["x1"], recon_a.samples[1]["x1"])
        # Session RNG is dropped on exit.
        assert gen_a._session_rng is None

    def test_rng_kwarg_overrides_session_rng(self) -> None:
        """A per-call ``rng=`` wins over the session generator."""
        recon_a, recon_b = _CaptureRecon(), _CaptureRecon()
        gen_a = self._generator(output_to_batch_func=recon_a, seed=11)
        gen_b = self._generator(output_to_batch_func=recon_b)  # no seed, no session
        with gen_a:
            gen_a.sample(
                make_batch(num_graphs=1), rng=torch.Generator().manual_seed(99)
            )
        gen_b.sample(make_batch(num_graphs=1), rng=torch.Generator().manual_seed(99))
        assert torch.equal(recon_a.samples[0]["x1"], recon_b.samples[0]["x1"])
