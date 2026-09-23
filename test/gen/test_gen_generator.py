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
with its fixed (optional condition →) generate core and
:class:`~nvalchemi.gen.stages.GenerationStage` hooks: the function-owns-model
contract, the optional condition step (resolution order, stage firing
policy, pass-through without a provider), the defaults chain (driver
argument > function attribute > module default), the ``Batch`` path vs
raw-sample passthrough, device validation / dedicated
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

from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen.generator import AtomisticGenerator
from nvalchemi.gen.stages import GenerationStage
from nvalchemi.models.gen import DemoGANModel
from nvalchemi.models.gen.demo import _DemoGANGenerate
from test.gen.conftest import (
    DeviceAwareGenerate,
    batch_generate,
    make_batch,
    trivial_generate,
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


class TestBaseGenerator:
    """Core :class:`AtomisticGenerator` tests."""

    def test_unconditional_generate_returns_batch(self) -> None:
        """A function returning a ``Batch`` yields it directly."""
        sentinel = make_batch(num_graphs=1)

        def _generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
            """Return the sentinel batch directly."""
            del inputs, num_samples, rng, kwargs
            return sentinel

        gen = AtomisticGenerator(generator_func=_generate)
        assert gen() is sentinel

    def test_conditional_generate_via_model_sampler(self, device: str) -> None:
        """A model-owning sampler runs and lands on its device."""
        gen = AtomisticGenerator(
            generator_func=_DemoGANGenerate(DemoGANModel().to(device))
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

    def test_batch_returned_on_batch_path(self) -> None:
        """A function returning a ``Batch`` takes the Batch path: it comes
        back as produced."""
        gen = AtomisticGenerator(generator_func=batch_generate)
        out = gen(num_samples=3)
        assert isinstance(out, Batch)
        assert out.num_graphs == 3

    def test_batch_output_fires_after_generate(self) -> None:
        """A ``Batch`` returned by the function fires ``AFTER_GENERATE``."""
        seen: list = []

        class _Recorder:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Record the batch at this stage."""
                seen.append(ctx.batch)

        gen = AtomisticGenerator(generator_func=batch_generate, hooks=[_Recorder()])
        out = gen(num_samples=2)
        assert len(seen) == 1
        assert out is seen[0]

    def test_raw_sample_passthrough(self) -> None:
        """A non-``Batch`` output comes back untouched (raw passthrough)."""
        gen = AtomisticGenerator(generator_func=trivial_generate)
        out = gen(num_samples=2)
        assert isinstance(out, TensorDict)
        assert out.batch_size[0] == 2

    def test_raw_sample_fires_no_hooks(self) -> None:
        """A raw (non-``Batch``) sample fires no hooks: dispatch happens only
        on the Batch path."""

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
            generator_func=trivial_generate,
            hooks=[_Recorder(stage, log) for stage in GenerationStage],
        )
        out = gen(num_samples=2)
        assert isinstance(out, TensorDict)
        assert log == []

    def test_after_generate_hook_skipped_for_raw_samples(self) -> None:
        """An ``AFTER_GENERATE`` hook on a raw-sample generator never fires;
        no construction warning is raised either."""

        class _Hook:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __init__(self) -> None:
                self.fired = False

            def __call__(self, ctx, stage) -> None:
                """Record firing."""
                self.fired = True

        hook = _Hook()
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # no warning at all now
            gen = AtomisticGenerator(generator_func=trivial_generate, hooks=[hook])
            out = gen(num_samples=1)
        assert isinstance(out, TensorDict)
        assert not hook.fired

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

    def test_non_tensordict_sample_flows_through(self) -> None:
        """A generating function may return any container; non-``Batch``
        outputs pass through untouched."""

        def _compact_generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
            """Return a plain dict as a compact stand-in sample container."""
            del inputs, rng, kwargs
            return {"rows": torch.zeros(num_samples, 2)}

        gen = AtomisticGenerator(generator_func=_compact_generate)
        out = gen(num_samples=3)
        assert out["rows"].shape == (3, 2)

    def test_field_declarations_default_from_function(self) -> None:
        """``required_inputs``/``outputs`` default from the function."""
        gen = AtomisticGenerator(generator_func=_DemoGANGenerate(DemoGANModel()))
        assert gen.required_inputs == frozenset()
        assert gen.outputs == frozenset({"positions", "atomic_numbers"})

    def test_field_declarations_explicit_override(self) -> None:
        """Explicit declarations win over the function's attributes."""
        gen = AtomisticGenerator(
            generator_func=_DemoGANGenerate(DemoGANModel()),
            required_inputs=frozenset({"charges"}),
        )
        assert gen.required_inputs == frozenset({"charges"})
        assert gen.outputs == frozenset({"positions", "atomic_numbers"})

    def test_field_declarations_none_without_attributes(self) -> None:
        """A function with no declarations leaves them undeclared (``None``)."""
        gen = AtomisticGenerator(generator_func=batch_generate)
        assert gen.required_inputs is None
        assert gen.outputs is None

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
            generator_func=batch_generate,
            hooks=[_Probe()],
        )
        gen()
        model, workflow = seen[0]
        assert model is None
        assert workflow is gen
        assert workflow.generator_func is batch_generate


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
        gen = AtomisticGenerator(generator_func=batch_generate)
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


class _OffDeviceRawGenerate:
    """Claims CUDA via the attribute; produces a CPU-resident TensorDict."""

    device = torch.device("cuda:0")

    def __call__(self, inputs=None, *, num_samples=1, rng=None, **kwargs):
        """Return a CPU TensorDict regardless of the declared device."""
        del inputs, rng, kwargs
        return TensorDict(
            {"x1": torch.zeros(num_samples, 1, 3)}, batch_size=[num_samples]
        )


class TestDeviceResidency:
    """The device-residency check on returned ``Batch`` outputs."""

    def test_residency_check_fires_on_mismatch(self) -> None:
        """A function returning a ``Batch`` on the wrong device raises."""
        gen = AtomisticGenerator(generator_func=_OffDeviceGenerate())
        with pytest.raises(ValueError, match="lives on device"):
            gen()

    def test_residency_check_passes_on_match(self) -> None:
        """A function returning a ``Batch`` on the resolved device passes."""
        gen = AtomisticGenerator(generator_func=DeviceAwareGenerate("cpu"))
        out = gen(num_samples=2)
        assert out.num_graphs == 2
        assert out["positions"].device.type == "cpu"

    def test_residency_check_skipped_for_non_batch(self) -> None:
        """A non-``Batch`` output skips the residency check, even with a
        device pinned via the function's attribute."""
        gen = AtomisticGenerator(generator_func=_OffDeviceRawGenerate())
        out = gen()
        assert out["x1"].device.type == "cpu"

    def test_residency_check_skipped_under_compile(self, monkeypatch) -> None:
        """The check is skipped while ``torch.compiler.is_compiling()`` is true."""
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        gen = AtomisticGenerator(generator_func=_OffDeviceGenerate())
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
            A batch-generating generator.
        """
        return AtomisticGenerator(
            generator_func=batch_generate,
            hooks=hooks,
        )

    def test_hook_firing_order(self) -> None:
        """Hooks fire once per call, in pipeline order; with no condition step
        only ``AFTER_GENERATE`` fires on the Batch path."""

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
        assert log == [GenerationStage.AFTER_GENERATE]

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
        """An ``AFTER_GENERATE`` hook replaces the generated batch."""
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
            stage = GenerationStage.BEFORE_CONDITION
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Stash a value for a later stage."""
                ctx.intermediates["tag"] = "from-before-condition"

        class _Consumer:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Read the stashed value."""
                seen.append(ctx.intermediates.get("tag"))

        gen = AtomisticGenerator(
            generator_func=_TiledGenerate(),
            hooks=[_Producer(), _Consumer()],
        )
        gen()
        assert seen == ["from-before-condition"]

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

    def test_int_stage_coerced(self) -> None:
        """An int stage (e.g. from a spec payload) is coerced at construction."""

        class _IntStage:
            frequency = 1

            def __init__(self) -> None:
                self.stage = GenerationStage.AFTER_GENERATE.value
                self.fired = False

            def __call__(self, ctx, stage) -> None:
                """Record firing."""
                self.fired = True

        hook = _IntStage()
        gen = self._generator([hook])
        assert hook.stage is GenerationStage.AFTER_GENERATE
        gen()
        assert hook.fired

    def test_zero_graph_batch_returned(self) -> None:
        """A function returning a zero-graph ``Batch`` signals total rejection."""

        def _empty_generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
            """Return an explicitly empty batch."""
            del inputs, num_samples, rng, kwargs
            return Batch.empty(num_systems=0, num_nodes=0, num_edges=0)

        gen = AtomisticGenerator(generator_func=_empty_generate)
        out = gen(make_batch(num_graphs=3))
        assert isinstance(out, Batch)
        assert out.num_graphs == 0

    def test_accepted_mask_recorded_and_visible_later(self) -> None:
        """``accepted_mask`` written at ``AFTER_CONDITION`` is readable at
        ``AFTER_GENERATE``."""

        class _Accept:
            stage = GenerationStage.AFTER_CONDITION
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Accept every draw."""
                ctx.accepted_mask = torch.ones(ctx.inputs.num_graphs, dtype=torch.bool)

        seen: list = []

        class _Read:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                """Read the recorded mask."""
                seen.append(ctx.accepted_mask)

        gen = AtomisticGenerator(
            generator_func=_TiledGenerate(),
            hooks=[_Accept(), _Read()],
        )
        out = gen(make_batch(num_graphs=2))
        assert len(seen) == 1
        assert seen[0] is not None
        assert int(seen[0].sum()) == out.num_graphs == 2

    def test_after_generate_dispatch_inside_session_stream(self, device: str) -> None:
        """AFTER_GENERATE hooks dispatch on the session CUDA stream, when any."""

        class _Probe:
            stage = GenerationStage.AFTER_GENERATE
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
            hooks=[_Recorder(stage) for stage in GenerationStage],
        )
        source = make_batch(num_graphs=2)
        out = gen(source)
        assert received[0] is source
        assert out is source
        assert fired == [GenerationStage.AFTER_GENERATE]

    def test_function_condition_attribute_runs_before_generation(self) -> None:
        """A function object's ``condition`` runs between the condition
        stages, ahead of generation."""
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
            hooks=[_Recorder(stage) for stage in GenerationStage],
        )
        out = gen(make_batch(num_graphs=2), num_samples=3)
        assert out.num_graphs == 6
        assert fired == [
            GenerationStage.BEFORE_CONDITION,
            GenerationStage.AFTER_CONDITION,
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
            A demo-sampler-backed generator (extra kwargs forwarded).
        """
        kwargs.setdefault("generator_func", _DemoGANGenerate(DemoGANModel()))
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
        gen_a = self._generator(generator_func=_rng_generate, seed=7)
        gen_b = self._generator(generator_func=_rng_generate, seed=7)
        out_a = list(gen_a.stream(max_batches=2))
        out_b = list(gen_b.stream(max_batches=2))
        for sample_a, sample_b in zip(out_a, out_b):
            assert torch.equal(sample_a["x1"], sample_b["x1"])
        # Per-draw seeds (seed + step_count) make consecutive draws differ.
        assert not torch.equal(out_a[0]["x1"], out_a[1]["x1"])


class TestSampleSugar:
    """``sample()`` is the real method; ``__call__`` delegates to it."""

    def test_call_matches_sample(self) -> None:
        """``__call__`` and ``sample`` produce identical behavior."""
        gen_a = AtomisticGenerator(generator_func=_rng_generate, seed=5)
        gen_b = AtomisticGenerator(generator_func=_rng_generate, seed=5)
        out_a = gen_a(make_batch(num_graphs=2))
        out_b = gen_b.sample(make_batch(num_graphs=2))
        assert out_a.batch_size == out_b.batch_size
        assert torch.equal(out_a["x1"], out_b["x1"])

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
        """Build a demo-sampler-backed generator with compile-related kwargs.

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
            generator_func=_DemoGANGenerate(DemoGANModel()), **kwargs
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
        gen = AtomisticGenerator(generator_func=batch_generate)
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
            RNG-drawing generator.
        """
        kwargs.setdefault("generator_func", _rng_generate)
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
        gen_a = self._generator(seed=11, device=device)
        gen_b = self._generator(seed=11, device=device)
        with gen_a:
            out_a = [
                gen_a.sample(make_batch(num_graphs=1).to(device)),
                gen_a.sample(make_batch(num_graphs=1).to(device)),
            ]
        with gen_b:
            out_b = [
                gen_b.sample(make_batch(num_graphs=1).to(device)),
                gen_b.sample(make_batch(num_graphs=1).to(device)),
            ]
        for sample_a, sample_b in zip(out_a, out_b):
            assert torch.equal(sample_a["x1"], sample_b["x1"])
        assert not torch.equal(out_a[0]["x1"], out_a[1]["x1"])
        # Session RNG is dropped on exit.
        assert gen_a._session_rng is None

    def test_rng_kwarg_overrides_session_rng(self) -> None:
        """A per-call ``rng=`` wins over the session generator."""
        gen_a = self._generator(seed=11)
        gen_b = self._generator()  # no seed, no session
        with gen_a:
            out_a = gen_a.sample(
                make_batch(num_graphs=1), rng=torch.Generator().manual_seed(99)
            )
        out_b = gen_b.sample(
            make_batch(num_graphs=1), rng=torch.Generator().manual_seed(99)
        )
        assert torch.equal(out_a["x1"], out_b["x1"])


class TestCallTimeFieldValidation:
    """``required_inputs`` / ``outputs`` are enforced inside ``sample``."""

    def test_required_inputs_missing_raises(self) -> None:
        """Inputs lacking a declared field fail before the function runs."""
        gen = AtomisticGenerator(
            generator_func=trivial_generate,
            required_inputs=frozenset({"positions", "cell"}),
        )
        with pytest.raises(ValueError, match="cell"):
            gen(make_batch())  # carries positions and atomic_numbers, no cell

    def test_required_inputs_present_passes(self) -> None:
        gen = AtomisticGenerator(
            generator_func=trivial_generate,
            required_inputs=frozenset({"positions", "atomic_numbers"}),
        )
        out = gen(make_batch())
        assert out["x1"].shape[0] == 2

    def test_required_inputs_checked_after_conditioning(self) -> None:
        """A condition step that adds the field satisfies the check."""

        def add_cell(inputs: Batch, *, num_samples: int, rng=None) -> Batch:
            inputs.cell = torch.zeros(inputs.num_graphs, 3, 3)
            return inputs

        gen = AtomisticGenerator(
            generator_func=trivial_generate,
            condition_func=add_cell,
            required_inputs=frozenset({"cell"}),
        )
        out = gen(make_batch())
        assert out["x1"].shape[0] == 2

    def test_required_inputs_none_inputs_raise_type_error(self) -> None:
        """Declared consumers cannot run on empty or non-container inputs."""
        gen = AtomisticGenerator(
            generator_func=trivial_generate,
            required_inputs=frozenset({"positions"}),
        )
        with pytest.raises(TypeError, match="field-addressable"):
            gen(None)

    def test_undeclared_consumes_skips_check(self) -> None:
        """required_inputs=None never inspects the inputs."""
        gen = AtomisticGenerator(generator_func=trivial_generate)
        out = gen("a composition string")  # not a container at all
        assert out["x1"].shape[0] == 1

    def test_outputs_missing_raises(self) -> None:
        """A returned batch lacking a declared field fails at return."""
        gen = AtomisticGenerator(
            generator_func=batch_generate,
            outputs=frozenset({"positions", "cell"}),
        )
        with pytest.raises(ValueError, match="cell"):
            gen(make_batch())

    def test_outputs_checked_after_hooks(self) -> None:
        """A hook that drops a declared field trips the return-time check."""

        class _FieldDropper:
            def __init__(self) -> None:
                self.stage = GenerationStage.AFTER_GENERATE
                self.frequency = 1

            def __call__(self, ctx, stage) -> None:
                # positions-only batch: atomic_numbers dropped
                ctx.batch = Batch.from_data_list(
                    [AtomicData(positions=torch.zeros(2, 3))]
                )

        gen = AtomisticGenerator(
            generator_func=batch_generate,
            outputs=frozenset({"atomic_numbers"}),
            hooks=[_FieldDropper()],
        )
        with pytest.raises(ValueError, match="atomic_numbers"):
            gen(make_batch())

    def test_raw_passthrough_skips_produces_check(self) -> None:
        """A non-``Batch`` (raw) sample skips the ``outputs`` check."""
        gen = AtomisticGenerator(
            generator_func=trivial_generate,
            outputs=frozenset({"cell"}),
        )
        out = gen(make_batch())
        assert out["x1"].shape[0] == 2


class TestReviewPins:
    """Pins added from the second-round review: error paths and orderings."""

    @pytest.mark.parametrize("bad", [0, -1])
    def test_per_call_num_samples_must_be_positive(self, bad: int) -> None:
        """The per-call override validates like the constructor's ge=1."""
        gen = AtomisticGenerator(generator_func=batch_generate)
        with pytest.raises(ValueError, match="num_samples must be positive"):
            gen(num_samples=bad)

    def test_step_count_increments_on_failed_calls(self) -> None:
        """A raising function still advances the counter (finally semantics)."""

        def _boom(inputs=None, *, num_samples=1, rng=None, **kwargs):
            raise RuntimeError("boom")

        gen = AtomisticGenerator(generator_func=_boom)
        with pytest.raises(RuntimeError, match="boom"):
            gen()
        assert gen.step_count == 1

    def test_hook_close_fallback_without_exit(self) -> None:
        """A hook with ``close`` but no ``__exit__`` is closed at session exit."""
        calls: list[str] = []

        class _CloseOnly:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                pass

            def close(self) -> None:
                calls.append("closed")

        gen = AtomisticGenerator(generator_func=batch_generate, hooks=[_CloseOnly()])
        with gen:
            gen()
        assert calls == ["closed"]

    def test_hook_can_add_a_missing_output_field(self) -> None:
        """The outputs check runs after hooks: a hook may satisfy it."""

        class _AddForces:
            stage = GenerationStage.AFTER_GENERATE
            frequency = 1

            def __call__(self, ctx, stage) -> None:
                ctx.batch.forces = torch.zeros(ctx.batch.num_nodes, 3)

        gen = AtomisticGenerator(
            generator_func=batch_generate,
            outputs=frozenset({"positions", "forces"}),
            hooks=[_AddForces()],
        )
        out = gen(make_batch())
        assert out.forces is not None

    def test_zero_graph_batch_exempt_from_field_check(self) -> None:
        """A total-rejection batch declares intent, not content: no field check."""
        gen = AtomisticGenerator(
            generator_func=lambda inputs=None, **kw: Batch.empty(
                num_systems=0, num_nodes=0, num_edges=0
            ),
            outputs=frozenset({"forces"}),  # outside the Batch.empty template
        )
        out = gen()
        assert out.num_graphs == 0

    def test_zero_graph_batch_exempt_from_residency_check(self) -> None:
        """The residency exemption: a CPU zero-graph batch under a CUDA pin passes."""
        if torch.cuda.is_available():
            pytest.skip("the exemption needs no CUDA device to be host-checkable")

        class _Pinned:
            device = torch.device("cuda:0")  # attribute-declared, no host check

            def __call__(self, inputs=None, **kw):
                return Batch.empty(num_systems=0, num_nodes=0, num_edges=0)

        gen = AtomisticGenerator(generator_func=_Pinned())
        out = gen()  # without the num_graphs exemption this raises ValueError
        assert out.num_graphs == 0

    @pytest.mark.parametrize(
        "stage_value",
        ["AFTER_GENERATE", GenerationStage.AFTER_GENERATE.value],
        ids=["name", "int"],
    )
    def test_stage_coercion_accepts_names_and_ints(self, stage_value) -> None:
        """Hook stage coerces from both name strings and ints at construction."""

        class _Hook:
            frequency = 1

            def __init__(self) -> None:
                self.stage = stage_value

            def __call__(self, ctx, stage) -> None:
                pass

        gen = AtomisticGenerator(generator_func=batch_generate, hooks=[_Hook()])
        assert gen.hooks[0].stage is GenerationStage.AFTER_GENERATE
