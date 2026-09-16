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

Covers the abstract :class:`~nvalchemi.gen.generator.AtomisticGenerator`
with its fixed (optional condition →) generate → materialize core and
:class:`~nvalchemi.gen.stages.GenerationStage` hooks: the function-owns-model
contract, the optional condition step (resolution order, stage firing
policy, pass-through without a provider), the defaults chain (driver
argument > function attribute > module default), ``batch_mapping``
materialization vs raw-sample passthrough, device validation / dedicated
streams / the device-residency check, hook firing order / frequency gating /
mutation-by-replacement / filter-by-subsetting, ``stream()`` semantics, the
``sample()``/``__call__`` sugar split, the ``torch.compile`` surface, and
session (context-manager) lifecycle. CPU-only, GPU-free, no optional deps.
"""

from __future__ import annotations

import itertools
import warnings
from enum import Enum, auto

import pytest
import torch
from pydantic import ValidationError
from tensordict import TensorDict
from torch import nn

from nvalchemi.data import Batch
from nvalchemi.gen.generator import AtomisticGenerator
from nvalchemi.gen.stages import GenerationStage
from nvalchemi.models.gen import DemoGANModel, make_demo_gan_generate
from test.gen.conftest import (
    DeviceAwareGenerate,
    batch_generate,
    make_batch,
    passthrough_mapping,
    trivial_generate,
    zeros_to_batch,
)


def _rng_generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
    """A generating function that draws from ``rng`` (seed-path tests).

    Parameters
    ----------
    inputs
        Conditioning batch, if any.
    num_samples
        Number of draws (used only when ``inputs`` is not a batch).
    rng
        :class:`torch.Generator` to draw from.
    **kwargs
        Family-specific options (ignored).

    Returns
    -------
    TensorDict
        Random values under the ``"x1"`` key.
    """
    del kwargs
    n = inputs.num_graphs if isinstance(inputs, Batch) else num_samples
    return TensorDict(
        {"x1": torch.randn(n, 1, 3, generator=rng, device=rng.device if rng else None)},
        batch_size=[n],
    )


class _CaptureRecon:
    """Materialization that records the raw sample TensorDict it receives."""

    def __init__(self, device: str | torch.device | None = None) -> None:
        self.samples: list[TensorDict] = []
        self.device = device

    def __call__(self, sample: TensorDict) -> Batch:
        """Record ``sample`` and return a batch sized like it.

        Parameters
        ----------
        sample
            Sample TensorDict to record.

        Returns
        -------
        Batch
            Dummy graphs matching the sample's leading size, moved to
            ``self.device`` when one was supplied (so device-resolved
            generators satisfy the driver's residency check).
        """
        self.samples.append(sample)
        out = make_batch(sample.batch_size[0])
        return out.to(self.device) if self.device is not None else out


class TestBaseGenerator:
    """Core :class:`AtomisticGenerator` tests."""

    def test_unconditional_generate_returns_batch(self) -> None:
        """A free generating function + ``batch_mapping`` yields a batch."""
        sentinel = make_batch(num_graphs=1)

        def recon(sample: TensorDict) -> Batch:
            """Materialization override returning a sentinel batch.

            Parameters
            ----------
            sample
                Ignored.

            Returns
            -------
            Batch
                ``sentinel``.
            """
            del sample
            return sentinel

        gen = AtomisticGenerator(
            generator_func=trivial_generate,
            batch_mapping=recon,
        )
        assert gen() is sentinel

    def test_conditional_generate_via_factory(self, device: str) -> None:
        """A factory-built, model-owning function runs and lands on its device."""
        gen = AtomisticGenerator(
            generator_func=make_demo_gan_generate(DemoGANModel().to(device))
        )
        out = gen(make_batch(num_graphs=3).to(device))
        assert isinstance(out, Batch)
        assert out.num_graphs == 3
        assert out["positions"].device.type == device

    def test_construct_requires_generator_func(self) -> None:
        """``generator_func`` is required: there is no model-level fallback."""
        with pytest.raises(ValidationError):
            AtomisticGenerator()
        with pytest.raises(ValidationError):
            AtomisticGenerator(generator_func=None)

    def test_batch_returned_raw_without_mapping(self) -> None:
        """A function returning a ``Batch`` needs no mapping: raw passthrough."""
        gen = AtomisticGenerator(generator_func=batch_generate)
        out = gen(num_samples=3)
        assert isinstance(out, Batch)
        assert out.num_graphs == 3

    def test_batch_mapping_runs_after_generate(self) -> None:
        """``batch_mapping`` is applied after the generating function."""
        calls: list[str] = []

        def _generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
            """Record generation, then emit a zeros sample."""
            calls.append("generate")
            return trivial_generate(inputs, num_samples=num_samples, rng=rng, **kwargs)

        def _mapping(sample) -> Batch:
            """Record materialization."""
            calls.append("mapping")
            return zeros_to_batch(sample)

        gen = AtomisticGenerator(generator_func=_generate, batch_mapping=_mapping)
        gen()
        assert calls == ["generate", "mapping"]

    def test_no_mapping_returns_raw_sample(self) -> None:
        """Without ``batch_mapping``, ``sample()`` returns the raw output as-is."""
        gen = AtomisticGenerator(generator_func=trivial_generate)
        out = gen(num_samples=2)
        assert isinstance(out, TensorDict)
        assert out.batch_size[0] == 2

    def test_no_mapping_fires_only_before_mapping(self) -> None:
        """Without ``batch_mapping``: ``BEFORE_MAPPING`` fires, ``AFTER_GENERATE``
        does not, and no construction warning is raised for BEFORE_MAPPING-only
        hooks."""

        class _Recorder:
            def __init__(self, stage: GenerationStage, log: list) -> None:
                self.stage = stage
                self.frequency = 1
                self._log = log

            def __call__(self, ctx, stage) -> None:
                """Record the dispatched stage."""
                self._log.append(stage)

        log: list = []
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning without AFTER_GENERATE hooks
            gen = AtomisticGenerator(
                generator_func=trivial_generate,
                hooks=[_Recorder(GenerationStage.BEFORE_MAPPING, log)],
            )
        out = gen(num_samples=2)
        assert isinstance(out, TensorDict)
        assert log == [GenerationStage.BEFORE_MAPPING]

    def test_after_generate_hook_without_mapping_warns_at_construction(self) -> None:
        """``batch_mapping=None`` + an ``AFTER_GENERATE`` hook warns once, at
        construction; the hook then never fires and no per-call warning is
        raised."""

        class _Hook:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __init__(self) -> None:
                self.fired = False

            def __call__(self, ctx, stage) -> None:
                """Record firing."""
                self.fired = True

        hook = _Hook()
        with pytest.warns(UserWarning, match="batch_mapping"):
            gen = AtomisticGenerator(generator_func=trivial_generate, hooks=[hook])
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # the warning is construction-time only
            out = gen(num_samples=1)
        assert isinstance(out, TensorDict)
        assert not hook.fired

    def test_before_mapping_hook_can_replace_sample_without_mapping(self) -> None:
        """``BEFORE_MAPPING`` fires even without a mapping; its replacement is
        what ``sample()`` returns."""
        replacement = TensorDict({"x1": torch.ones(2, 1, 3)}, batch_size=[2])

        class _SwapSample:
            stage = GenerationStage.BEFORE_MAPPING
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Replace the raw sample outright."""
                ctx.sample = replacement

        gen = AtomisticGenerator(generator_func=trivial_generate, hooks=[_SwapSample()])
        out = gen(num_samples=5)
        assert out is replacement

    def test_num_samples_passthrough_and_per_call_override(self) -> None:
        """The driver's ``num_samples`` reaches the function; the per-call
        keyword overrides it."""
        seen: list[int] = []

        def _probe_generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
            """Record the requested draw count and emit that many graphs."""
            del inputs, rng, kwargs
            seen.append(num_samples)
            return make_batch(num_samples)

        gen = AtomisticGenerator(generator_func=_probe_generate, num_samples=4)
        assert gen().num_graphs == 4
        assert gen(num_samples=2).num_graphs == 2
        assert seen == [4, 2]

    def test_materialization_must_return_batch(self) -> None:
        """A ``batch_mapping`` returning a non-Batch raises ``TypeError``."""
        gen = AtomisticGenerator(
            generator_func=trivial_generate,
            batch_mapping=lambda sample: sample,
        )
        with pytest.raises(TypeError, match="not a Batch"):
            gen()

    def test_non_tensordict_sample_flows_through(self) -> None:
        """A generating function may return any container the mapping understands."""

        def _compact_generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
            """Return a plain dict as a compact stand-in sample container."""
            del inputs, rng, kwargs
            return {"rows": torch.zeros(num_samples, 2)}

        def _compact_recon(sample) -> Batch:
            """Materialize a dict sample: one graph per row."""
            return make_batch(sample["rows"].shape[0])

        gen = AtomisticGenerator(
            generator_func=_compact_generate,
            batch_mapping=_compact_recon,
        )
        out = gen(num_samples=3)
        assert isinstance(out, Batch)
        assert out.num_graphs == 3

    def test_field_declarations_default_from_function(self) -> None:
        """``consumes_fields``/``produces_fields`` default from the function."""
        gen = AtomisticGenerator(generator_func=make_demo_gan_generate(DemoGANModel()))
        assert gen.consumes_fields == frozenset()
        assert gen.produces_fields == frozenset({"positions", "atomic_numbers"})

    def test_field_declarations_explicit_override(self) -> None:
        """Explicit declarations win over the function's attributes."""
        gen = AtomisticGenerator(
            generator_func=make_demo_gan_generate(DemoGANModel()),
            consumes_fields=frozenset({"charges"}),
        )
        assert gen.consumes_fields == frozenset({"charges"})
        assert gen.produces_fields == frozenset({"positions", "atomic_numbers"})

    def test_field_declarations_none_without_attributes(self) -> None:
        """A function with no declarations leaves them undeclared (``None``)."""
        gen = AtomisticGenerator(
            generator_func=trivial_generate,
            batch_mapping=zeros_to_batch,
        )
        assert gen.consumes_fields is None
        assert gen.produces_fields is None

    def test_ctx_has_no_model_and_workflow_carries_procedure(self) -> None:
        """``ctx.model`` is ``None``; hooks reach the procedure via the workflow."""
        seen: list = []

        class _Probe:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Record the context fields."""
                seen.append((ctx.model, ctx.workflow))

        gen = AtomisticGenerator(
            generator_func=trivial_generate,
            batch_mapping=zeros_to_batch,
            hooks=[_Probe()],
        )
        gen()
        model, workflow = seen[0]
        assert model is None
        assert workflow is gen
        assert workflow.generator_func is trivial_generate


class TestDefaultsChain:
    """The defaults chain: driver argument > function attribute > module default."""

    def test_device_explicit_beats_function_attribute(self) -> None:
        """The ``device`` field wins over the function's ``device`` attribute."""
        func = DeviceAwareGenerate("meta")
        gen = AtomisticGenerator(generator_func=func, device="cpu")
        assert gen._infer_device() == torch.device("cpu")

    def test_device_function_attribute_used(self) -> None:
        """Without a ``device`` field, the function's ``device`` attribute resolves."""
        gen = AtomisticGenerator(generator_func=DeviceAwareGenerate("cpu"))
        assert gen._infer_device() == torch.device("cpu")

    def test_device_unresolved_without_sources(self) -> None:
        """No ``device`` field and no function attribute resolves to ``None``."""
        gen = AtomisticGenerator(
            generator_func=trivial_generate, batch_mapping=zeros_to_batch
        )
        assert gen._infer_device() is None


class TestDeviceValidation:
    """Construction-time validation of the ``device`` field."""

    def test_cpu_always_valid(self) -> None:
        """``"cpu"`` (string or ``torch.device``) always validates."""
        gen = AtomisticGenerator(generator_func=batch_generate, device="cpu")
        assert gen.device == torch.device("cpu")
        gen = AtomisticGenerator(
            generator_func=batch_generate, device=torch.device("cpu")
        )
        assert gen.device == torch.device("cpu")

    def test_invalid_string_raises(self) -> None:
        """An unparseable device string raises at construction."""
        with pytest.raises(ValidationError, match="Invalid device string"):
            AtomisticGenerator(generator_func=batch_generate, device="not-a-device")

    def test_wrong_type_raises(self) -> None:
        """A non-string, non-torch.device ``device`` raises at construction."""
        with pytest.raises(ValidationError, match="string or torch.device"):
            AtomisticGenerator(generator_func=batch_generate, device=42)

    @pytest.mark.skipif(torch.cuda.is_available(), reason="Requires a CUDA-less host.")
    def test_cuda_unavailable_raises(self) -> None:
        """``cuda`` on a CUDA-less host raises at construction."""
        with pytest.raises(ValidationError, match="CUDA"):
            AtomisticGenerator(generator_func=batch_generate, device="cuda")

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="No CUDA device available."
    )
    def test_cuda_index_out_of_range_raises(self) -> None:
        """A ``cuda:i`` index beyond the host's device count raises."""
        with pytest.raises(ValidationError, match="out of range"):
            AtomisticGenerator(
                generator_func=batch_generate,
                device=f"cuda:{torch.cuda.device_count()}",
            )


class _OffDeviceGenerate:
    """Claims CUDA via the attribute; produces CPU-resident output."""

    device = torch.device("cuda:0")

    def __call__(self, inputs=None, *, num_samples=1, rng=None, **kwargs):
        """Return a CPU batch regardless of the declared device."""
        del inputs, rng, kwargs
        return make_batch(num_samples)


class TestDeviceResidency:
    """The post-materialization device-residency check (mapping path only)."""

    def test_residency_check_fires_on_mismatch(self) -> None:
        """A mapping materializing off the resolved device raises ``ValueError``."""
        gen = AtomisticGenerator(
            generator_func=_OffDeviceGenerate(), batch_mapping=passthrough_mapping
        )
        with pytest.raises(ValueError, match="materialized the batch on device"):
            gen()

    def test_residency_check_passes_on_match(self) -> None:
        """A mapping materializing on the resolved device passes."""
        gen = AtomisticGenerator(
            generator_func=DeviceAwareGenerate("cpu"),
            batch_mapping=passthrough_mapping,
        )
        out = gen(num_samples=2)
        assert out.num_graphs == 2
        assert out["positions"].device.type == "cpu"

    def test_residency_check_skipped_without_mapping(self) -> None:
        """No ``batch_mapping``, no residency check: off-device output passes
        through as-is."""
        gen = AtomisticGenerator(generator_func=_OffDeviceGenerate())
        out = gen()
        assert out["positions"].device.type == "cpu"

    def test_residency_check_skipped_under_compile(self, monkeypatch) -> None:
        """The check is skipped while ``torch.compiler.is_compiling()`` is true."""
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        gen = AtomisticGenerator(
            generator_func=_OffDeviceGenerate(), batch_mapping=passthrough_mapping
        )
        assert gen().num_graphs == 1


class TestGenerationHooks:
    """Hook dispatch, mutation, and filtering on the generation stages."""

    def _generator(self, hooks: list) -> AtomisticGenerator:
        """Build a trivial generator carrying ``hooks``.

        Parameters
        ----------
        hooks
            Hooks to register.

        Returns
        -------
        AtomisticGenerator
            A zeros-generating generator.
        """
        return AtomisticGenerator(
            generator_func=trivial_generate,
            batch_mapping=zeros_to_batch,
            hooks=hooks,
        )

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
            GenerationStage.BEFORE_MAPPING,
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
            stage = GenerationStage.BEFORE_MAPPING
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Stash a value for a later stage."""
                ctx.intermediates["tag"] = "from-before-mapping"

        class _Consumer:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Read the stashed value."""
                seen.append(ctx.intermediates.get("tag"))

        gen = self._generator([_Producer(), _Consumer()])
        gen()
        assert seen == ["from-before-mapping"]

    def test_filter_hook_subsets_batch(self, device: str) -> None:
        """Filtering is graph-level subsetting at AFTER_GENERATE."""

        class _KeepFirst:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Keep only the first graph."""
                ctx.batch = ctx.batch[[0]]

        gen = AtomisticGenerator(
            generator_func=DeviceAwareGenerate(device),
            batch_mapping=passthrough_mapping,
            hooks=[_KeepFirst()],
        )
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

    def test_before_mapping_hook_sees_pre_materialization_state(self) -> None:
        """At BEFORE_MAPPING, ``ctx.sample`` holds the raw sample, ``ctx.batch``
        the inputs (when they were a batch)."""

        class _Observe:
            stage = GenerationStage.BEFORE_MAPPING
            frequency = 1

            def __init__(self) -> None:
                self.sample = None
                self.batch = None
                self.inputs = None

            def __call__(self, ctx, stage) -> None:
                """Record the context fields visible at this stage."""
                self.sample = ctx.sample
                self.batch = ctx.batch
                self.inputs = ctx.inputs

        probe = _Observe()
        capture = _CaptureRecon()
        gen = AtomisticGenerator(
            generator_func=trivial_generate,
            batch_mapping=capture,
            hooks=[probe],
        )
        cond = make_batch(num_graphs=3)
        gen(cond)
        assert probe.inputs is cond
        assert probe.batch is cond  # inputs were a Batch
        assert probe.sample is capture.samples[0]  # materialization saw it as-is

    def test_before_mapping_hook_replaces_sample(self) -> None:
        """A BEFORE_MAPPING hook's replacement is what materialization receives."""
        replacement = TensorDict({"x1": torch.ones(2, 1, 3)}, batch_size=[2])

        class _SwapSample:
            stage = GenerationStage.BEFORE_MAPPING
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Replace the raw sample outright."""
                ctx.sample = replacement

        capture = _CaptureRecon()
        gen = AtomisticGenerator(
            generator_func=trivial_generate,
            batch_mapping=capture,
            hooks=[_SwapSample()],
        )
        out = gen(make_batch(num_graphs=3))
        assert capture.samples[0] is replacement
        assert out.num_graphs == 2

    def test_before_mapping_string_stage_coerced(self) -> None:
        """A string ``"BEFORE_MAPPING"`` stage is coerced at construction."""

        class _StringStage:
            frequency = 1

            def __init__(self) -> None:
                self.stage = "BEFORE_MAPPING"
                self.fired = False

            def __call__(self, ctx, stage) -> None:
                """Record firing."""
                self.fired = True

        hook = _StringStage()
        gen = self._generator([hook])
        assert hook.stage is GenerationStage.BEFORE_MAPPING
        gen()
        assert hook.fired

    def test_zero_graph_materialization_returned(self) -> None:
        """A mapping returning a zero-graph ``Batch`` signals total rejection."""

        def _empty_recon(sample) -> Batch:
            """Materialize to an explicitly empty batch."""
            del sample
            return Batch.empty(num_systems=0, num_nodes=0, num_edges=0)

        gen = AtomisticGenerator(
            generator_func=trivial_generate,
            batch_mapping=_empty_recon,
        )
        out = gen(make_batch(num_graphs=3))
        assert isinstance(out, Batch)
        assert out.num_graphs == 0

    def test_accepted_mask_recorded_and_visible_later(self) -> None:
        """``accepted_mask`` written at BEFORE_MAPPING is readable at AFTER_GENERATE."""

        class _Accept:
            stage = GenerationStage.BEFORE_MAPPING
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Accept every draw."""
                ctx.accepted_mask = torch.ones(
                    ctx.sample.batch_size[0], dtype=torch.bool
                )

        seen: list = []

        class _Read:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Read the recorded mask."""
                seen.append(ctx.accepted_mask)

        gen = self._generator([_Accept(), _Read()])
        out = gen(make_batch(num_graphs=2))
        assert len(seen) == 1
        assert seen[0] is not None
        assert int(seen[0].sum()) == out.num_graphs == 2

    def test_before_mapping_dispatch_inside_session_stream(self, device: str) -> None:
        """BEFORE_MAPPING hooks dispatch on the session CUDA stream, when any."""

        class _Probe:
            stage = GenerationStage.BEFORE_MAPPING
            frequency = 1

            def __init__(self) -> None:
                self.cuda_stream = None

            def __call__(self, ctx, stage) -> None:
                """Record the active CUDA stream pointer, when any."""
                if torch.cuda.is_available():
                    self.cuda_stream = torch.cuda.current_stream().cuda_stream

        probe = _Probe()
        gen = AtomisticGenerator(
            generator_func=DeviceAwareGenerate(device), hooks=[probe]
        )
        with gen:
            session_stream = gen._stream
            gen(make_batch(num_graphs=1).to(device))
        if device == "cuda":
            assert session_stream is not None
            assert probe.cuda_stream == session_stream.cuda_stream
        else:
            assert session_stream is None

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
        """A string stage (e.g. from a spec payload) is coerced to GenerationStage."""

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


class _TiledGenerate:
    """Generating function object carrying a tiling ``condition`` attribute."""

    def __init__(self) -> None:
        self.condition_calls: list[dict] = []

    def condition(self, inputs=None, *, num_samples=None, rng=None):
        """Record the resolved call arguments, then tile the inputs.

        Parameters
        ----------
        inputs
            The call's raw inputs.
        num_samples
            The resolved draw count; ``None`` tiles by one.
        rng
            The resolved RNG (recorded, unused).

        Returns
        -------
        The inputs tiled by ``num_samples`` (a user-written condition).
        """
        self.condition_calls.append({"num_samples": num_samples, "rng": rng})
        if inputs is None:
            return None
        n = 1 if num_samples is None else num_samples
        if isinstance(inputs, Batch):
            idx = torch.arange(inputs.num_graphs).repeat_interleave(n)
            return inputs[idx.to(inputs.device)]
        return inputs

    def __call__(self, inputs=None, *, num_samples=1, rng=None, **kwargs):
        """Echo a :class:`Batch` input, else emit ``num_samples`` graphs."""
        del rng, kwargs
        return inputs if isinstance(inputs, Batch) else make_batch(num_samples)


class TestConditioning:
    """The optional condition step: resolution, firing policy, pass-through."""

    def test_no_provider_passes_inputs_through_verbatim(self) -> None:
        """With no condition provider the raw inputs reach the function
        unchanged and the condition stages do not fire."""
        received: list = []
        fired: list = []

        class _Recorder:
            def __init__(self, stage: GenerationStage) -> None:
                self.stage = stage
                self.frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Record the dispatched stage."""
                fired.append(stage)

        def _capture_generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
            """Record the received inputs; echo a Batch, else emit draws."""
            received.append(inputs)
            return inputs if isinstance(inputs, Batch) else make_batch(num_samples)

        gen = AtomisticGenerator(
            generator_func=_capture_generate,
            batch_mapping=passthrough_mapping,
            hooks=[_Recorder(stage) for stage in GenerationStage],
        )
        source = make_batch(num_graphs=2)
        out = gen(source)
        assert received[0] is source
        assert out is source
        assert fired == [
            GenerationStage.BEFORE_MAPPING,
            GenerationStage.AFTER_GENERATE,
        ]

    def test_function_condition_attribute_runs_before_generation(self) -> None:
        """A function object's ``condition`` runs between the condition
        stages, ahead of generation and mapping."""
        fired: list = []

        class _Recorder:
            def __init__(self, stage: GenerationStage) -> None:
                self.stage = stage
                self.frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Record the dispatched stage."""
                fired.append(stage)

        gen = AtomisticGenerator(
            generator_func=_TiledGenerate(),
            batch_mapping=passthrough_mapping,
            hooks=[_Recorder(stage) for stage in GenerationStage],
        )
        out = gen(make_batch(num_graphs=2), num_samples=3)
        assert out.num_graphs == 6
        assert fired == [
            GenerationStage.BEFORE_CONDITION,
            GenerationStage.AFTER_CONDITION,
            GenerationStage.BEFORE_MAPPING,
            GenerationStage.AFTER_GENERATE,
        ]

    def test_driver_condition_func_overrides_function_attribute(self) -> None:
        """An explicit ``condition_func`` wins over the function's attribute."""
        calls: list[str] = []

        def _driver_condition(inputs, *, num_samples=None, rng=None):
            """Record the driver-level call; pass the inputs through."""
            del num_samples, rng
            calls.append("driver")
            return inputs

        func = _TiledGenerate()
        gen = AtomisticGenerator(generator_func=func, condition_func=_driver_condition)
        out = gen(make_batch(num_graphs=2))
        assert calls == ["driver"]
        assert func.condition_calls == []
        assert out.num_graphs == 2

    def test_condition_receives_resolved_num_samples_and_rng(self) -> None:
        """The resolved draw count and RNG reach the condition callable."""
        func = _TiledGenerate()
        gen = AtomisticGenerator(generator_func=func, num_samples=4)
        gen()
        assert func.condition_calls[-1]["num_samples"] == 4
        assert func.condition_calls[-1]["rng"] is None

        rng = torch.Generator().manual_seed(0)
        gen(make_batch(num_graphs=1), num_samples=2, rng=rng)
        assert func.condition_calls[-1]["num_samples"] == 2
        assert func.condition_calls[-1]["rng"] is rng

    def test_before_condition_hook_replaces_inputs(self) -> None:
        """A ``BEFORE_CONDITION`` hook's ``ctx.inputs`` replacement is what
        the condition callable receives."""

        class _SwapInputs:
            stage = GenerationStage.BEFORE_CONDITION
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Replace a 3-graph input with a 1-graph one."""
                ctx.inputs = make_batch(num_graphs=1)

        gen = AtomisticGenerator(
            generator_func=_TiledGenerate(),
            batch_mapping=passthrough_mapping,
            hooks=[_SwapInputs()],
        )
        out = gen(make_batch(num_graphs=3))
        assert out.num_graphs == 1

    def test_after_condition_hook_sees_conditioned_inputs(self) -> None:
        """At ``AFTER_CONDITION`` ``ctx.inputs`` holds the conditioned value;
        ``ctx.batch`` still holds the raw call input."""
        seen: list = []

        class _Probe:
            stage = GenerationStage.AFTER_CONDITION
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Record graph counts at the condition boundary."""
                seen.append((ctx.inputs.num_graphs, ctx.batch.num_graphs))

        gen = AtomisticGenerator(
            generator_func=_TiledGenerate(),
            batch_mapping=passthrough_mapping,
            hooks=[_Probe()],
        )
        gen(make_batch(num_graphs=2), num_samples=3)
        assert seen == [(6, 2)]


class TestStreaming:
    """``stream()`` semantics: input iteration, caps, filtered batches, seeds."""

    def _generator(self, **kwargs) -> AtomisticGenerator:
        """Build a minimal streaming generator.

        Parameters
        ----------
        **kwargs
            Extra constructor arguments.

        Returns
        -------
        AtomisticGenerator
            A factory-backed demo generator (extra kwargs forwarded).
        """
        kwargs.setdefault("generator_func", make_demo_gan_generate(DemoGANModel()))
        kwargs.setdefault("batch_mapping", passthrough_mapping)
        return AtomisticGenerator(**kwargs)

    def test_stream_caps_with_max_batches(self) -> None:
        """``stream(None, max_batches=3)`` yields exactly 3 unconditional draws."""
        gen = self._generator()
        batches = list(gen.stream(max_batches=3))
        assert len(batches) == 3
        assert gen.step_count == 3

    def test_stream_iterates_inputs(self) -> None:
        """Each input item drives exactly one call."""
        gen = self._generator()
        inputs = [make_batch(num_graphs=1), make_batch(num_graphs=2)]
        out = list(gen.stream(inputs))
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
            generator_func=_rng_generate, batch_mapping=recon_a, seed=7
        )
        gen_b = self._generator(
            generator_func=_rng_generate, batch_mapping=recon_b, seed=7
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
        gen_a = AtomisticGenerator(
            generator_func=_rng_generate,
            batch_mapping=capture_a,
            seed=5,
        )
        gen_b = AtomisticGenerator(
            generator_func=_rng_generate,
            batch_mapping=capture_b,
            seed=5,
        )
        out_a = gen_a(make_batch(num_graphs=2))
        out_b = gen_b.sample(make_batch(num_graphs=2))
        assert out_a.num_graphs == out_b.num_graphs
        assert torch.equal(capture_a.samples[0]["x1"], capture_b.samples[0]["x1"])

    def test_sample_runs_full_dispatch(self) -> None:
        """With a condition provider set, ``sample()`` fires every stage in order."""

        class _Recorder:
            def __init__(self, stage: GenerationStage, log: list) -> None:
                self.stage = stage
                self.frequency = 1
                self._log = log

            def __call__(self, ctx, stage) -> None:
                """Record the dispatched stage."""
                self._log.append(stage)

        log: list = []
        gen = AtomisticGenerator(
            generator_func=_TiledGenerate(),
            batch_mapping=passthrough_mapping,
            hooks=[_Recorder(stage, log) for stage in GenerationStage],
        )
        gen.sample()
        assert log == list(GenerationStage)

    def test_stream_calls_sample(self) -> None:
        """``stream()`` drives ``sample()`` once per input item."""
        gen = AtomisticGenerator(generator_func=batch_generate)
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

    def _generator(self, **kwargs) -> AtomisticGenerator:
        """Build a factory-backed generator with compile-related kwargs.

        Parameters
        ----------
        **kwargs
            Extra constructor arguments.

        Returns
        -------
        AtomisticGenerator
            A demo-GAN-backed generator.
        """
        return AtomisticGenerator(
            generator_func=make_demo_gan_generate(DemoGANModel()), **kwargs
        )

    def test_compile_wraps_generator_func(self) -> None:
        """``compile()`` wraps the generating function and sets the flag."""
        gen = self._generator()
        assert gen._compiled_generate is None
        out = gen.compile(backend="eager")
        assert out is gen
        assert gen.compile_generate is True
        assert gen._compiled_generate is not None
        assert gen(make_batch(num_graphs=1)).num_graphs == 1

    def test_compile_wraps_plain_function(self) -> None:
        """``compile()`` targets a plain module-level function as-is."""
        gen = AtomisticGenerator(
            generator_func=trivial_generate, batch_mapping=zeros_to_batch
        )
        gen.compile(backend="eager")
        assert gen._compiled_generate is not None
        assert gen(make_batch(num_graphs=2)).num_graphs == 2

    def test_compile_merges_kwargs(self) -> None:
        """Call-time kwargs merge over constructor ``compile_kwargs``."""
        gen = self._generator(compile_kwargs={"backend": "eager", "dynamic": True})
        gen.compile(dynamic=False)
        assert gen.compile_kwargs == {"backend": "eager", "dynamic": False}

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

    def _generator(self, **kwargs) -> AtomisticGenerator:
        """Build a trivial generator with session-related kwargs.

        Parameters
        ----------
        **kwargs
            Extra constructor arguments.

        Returns
        -------
        AtomisticGenerator
            RNG-drawing generator with a capture recon.
        """
        kwargs.setdefault("generator_func", _rng_generate)
        kwargs.setdefault("batch_mapping", _CaptureRecon())
        return AtomisticGenerator(**kwargs)

    def test_session_stream_matches_resolved_device(self, device: str) -> None:
        """A CUDA-resolved generator gets a dedicated session stream; CPU does not."""
        gen = self._generator(device=device)
        with gen:
            if device == "cuda":
                assert gen._stream is not None
                assert gen._stream_ctx is not None
            else:
                assert gen._stream is None
                assert gen._stream_ctx is None
            assert gen._session_rng is None  # no seed set
        assert gen._stream is None

    def test_session_stream_from_function_device(self, device: str) -> None:
        """The function's ``device`` attribute drives stream creation too."""
        gen = AtomisticGenerator(generator_func=DeviceAwareGenerate(device))
        with gen:
            if device == "cuda":
                assert gen._stream is not None
            else:
                assert gen._stream is None
            gen(make_batch(num_graphs=1).to(device))
        assert gen._stream is None

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="No CUDA device available."
    )
    def test_dedicated_stream_false_suppresses_stream(self) -> None:
        """``dedicated_stream=False``: no stream even with a CUDA device."""
        gen = AtomisticGenerator(
            generator_func=DeviceAwareGenerate("cuda"), dedicated_stream=False
        )
        with gen:
            assert gen._stream is None
            assert gen._stream_ctx is None
            out = gen(make_batch(num_graphs=1).to("cuda"))
            assert out["positions"].device.type == "cuda"
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
        recon_a = _CaptureRecon(device=device)
        recon_b = _CaptureRecon(device=device)
        gen_a = self._generator(batch_mapping=recon_a, seed=11, device=device)
        gen_b = self._generator(batch_mapping=recon_b, seed=11, device=device)
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
        gen_a = self._generator(batch_mapping=recon_a, seed=11)
        gen_b = self._generator(batch_mapping=recon_b)  # no seed, no session
        with gen_a:
            gen_a.sample(
                make_batch(num_graphs=1), rng=torch.Generator().manual_seed(99)
            )
        gen_b.sample(make_batch(num_graphs=1), rng=torch.Generator().manual_seed(99))
        assert torch.equal(recon_a.samples[0]["x1"], recon_b.samples[0]["x1"])
