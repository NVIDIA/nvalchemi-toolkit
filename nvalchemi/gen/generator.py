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
"""Batched generative inference driven by a user-supplied callable.

``AtomisticGenerator`` (the *driver* throughout this module) wraps your generation
code — a function, or a callable object holding the trained model — and runs
a fixed pipeline per call: optionally condition the inputs, generate,
optionally map the raw sample to a :class:`~nvalchemi.data.Batch`. Any
sampling procedure works (diffusion, flow matching, GANs, VAEs, search
loops); the callable owns the model and does the sampling.

The pipeline per :meth:`AtomisticGenerator.sample` looks like::

    # condition step — only when a condition callable is provided:
    BEFORE_CONDITION   hooks  (edit or replace ctx.inputs)
    ctx.inputs = condition(ctx.inputs, num_samples=..., rng=...)
    AFTER_CONDITION    hooks  (replace what the function is called with)
    # generation and mapping:
    ctx.sample = generator_func(ctx.inputs, num_samples=..., rng=...)
    BEFORE_MAPPING     hooks  (filtering = replacing ctx.sample)
    ctx.batch = batch_mapping(ctx.sample)      # only when batch_mapping set
    AFTER_GENERATE     hooks  (filtering = subsetting ctx.batch)
    return ctx.batch      # or ctx.sample as-is when no batch_mapping

Conditioning is optional: pass ``condition_func``, or set a ``condition``
attribute on the generating function, to transform inputs before generation.
It prepares the request — what the procedure is asked to do — it does not
constrain what comes out (output constraints live in guidance, rejection
hooks, or pipeline stages). With neither, the inputs reach the function
untouched and the ``BEFORE_CONDITION``/``AFTER_CONDITION`` stages do not
fire.

Hooks registered at the
:class:`~nvalchemi.gen.stages.GenerationStage` points receive one shared
:class:`~nvalchemi.hooks.GenerationContext` per call and mutate it by
replacing its fields; the driver re-reads the context after each dispatch.
Hooks run in list order at each stage, and a raising hook aborts the call.
See the stage enum for what may change where and when each stage fires.
Hooks reach the procedure via ``ctx.workflow.generator_func``;
``ctx.model`` is always ``None`` — the driver has no model field.

The defaults chain
------------------
Every cross-cutting knob resolves the same way: an explicit driver argument
wins, then the generating function's attribute, then the module default.

* conditioning: ``condition_func`` > ``generator_func.condition`` > ``None``
  (no condition step; the inputs pass through untouched).
* field declarations: constructor ``consumes_fields`` / ``produces_fields`` >
  ``generator_func``'s attributes > ``None`` (undeclared;
  :class:`~nvalchemi.gen.pipeline.GenerationPipeline` validation fails fast).
* device: ``device`` > ``generator_func.device`` > ``None`` (no stream, no
  device-residency check, CPU session RNG).

Materialization is optional: supply ``batch_mapping`` to map the raw sample
(a :class:`~tensordict.TensorDict` for tensor-native families, or any other
container) to a :class:`~nvalchemi.data.Batch` — a non-``Batch`` result
raises ``TypeError``. With no ``batch_mapping``, ``sample()`` returns the raw
sample, and ``AFTER_GENERATE`` hooks do not fire.

Examples
--------
A GAN, one forward pass, with a complete mapping::

    def gan_generate(inputs=None, *, num_samples=1, rng=None, **kwargs):
        z = torch.randn(num_samples, latent_dim, generator=rng)
        return TensorDict({"x1": decode(z)}, batch_size=[num_samples])

    def to_batch(sample):
        numbers = torch.full((num_atoms,), 6)
        return Batch.from_data_list(
            [
                AtomicData(positions=positions, atomic_numbers=numbers)
                for positions in sample["x1"]
            ]
        )

    gan = AtomisticGenerator(generator_func=gan_generate, batch_mapping=to_batch)

Model-owning procedures — a callable object carries the model (plus
``device`` and field declarations the driver reads as defaults); module-level
factories such as
:func:`~nvalchemi.models.gen.demo.make_demo_gan_generate` build such objects
so they can be captured for serialization
(``AtomisticGenerator.model_dump_json()``)::

    gen = AtomisticGenerator(generator_func=make_demo_gan_generate(DemoGANModel()))

Streaming — one ``sample()`` call per input item; ``None`` means repeated
unconditional draws (an infinite stream unless capped)::

    for batch in gen.stream(None, max_batches=10):
        ...

Composition — sequential pipelines mirror the dynamics ``|`` sugar
(:class:`~nvalchemi.gen.pipeline.GenerationPipeline`)::

    pipe = gen_a | gen_b
    out = pipe(inputs)

Sessions: streams, RNG, and compile — the driver is a context manager.
Entering a session (``with gen:``) creates a dedicated CUDA stream when the
resolved device is CUDA (unless ``dedicated_stream=False``), seeds a
session-scoped :class:`torch.Generator` (advanced per draw), compiles the
generating function when ``compile_generate`` is set, and opens
context-manager hooks; exiting unwinds all of it::

    with gen.compile(fullgraph=True):
        for batch in gen.stream(inputs):
            ...
"""

from __future__ import annotations

import inspect
import itertools
import warnings
from collections.abc import Iterator
from contextlib import nullcontext
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen.stages import GenerationStage
from nvalchemi.hooks import GenerationContext, Hook, HookRegistryMixin

if TYPE_CHECKING:
    from nvalchemi.gen.pipeline import GenerationPipeline

__all__ = [
    "GeneratingFunction",
    "ConditionFunction",
    "AtomisticGenerator",
    "MaterializationFunction",
]

InputT = TypeVar("InputT")
SampleT = TypeVar("SampleT")

MaterializationFunction: TypeAlias = Callable[[SampleT], Batch]
"""Map a raw sample of type ``SampleT`` to a :class:`~nvalchemi.data.Batch`.

The mapping receives the sample alone: conditioning has already happened
on the inputs, so no conditioning batch is passed in. The sample must be
self-describing: anything the mapping needs (cell parameters, conditioning
metadata, a per-candidate condition index for tracing through filtering)
has to be part of the sample the generating function emitted. Hooks at
``BEFORE_MAPPING`` see the sample (and the inputs via ``ctx.batch`` when
they were a :class:`~nvalchemi.data.Batch`), and ``ctx.accepted_mask``
records which candidates survived filtering.
"""


@runtime_checkable
class GeneratingFunction(Protocol[InputT, SampleT]):
    """Callable that encapsulates a family-specific generation strategy.

    A :class:`GeneratingFunction` owns everything generation needs —
    including the model, when there is one (held as a closure or an
    attribute of a callable object) — samples from it, and returns the raw
    sample.

    Conditioning may live inside the function, or as a ``condition``
    attribute the driver calls before generation (see below); with neither,
    the inputs pass through untouched as the first positional argument.

    ``num_samples`` semantics are the function's own (per-conditioning-entry
    draws vs an unconditional total); the driver passes its ``num_samples``
    straight through.

    ``rng`` is ``None`` unless the driver has a ``seed``, a session, or a
    per-call ``rng=`` — handle ``None`` (torch sampling ops accept it).
    Outside a session the resolved RNG is CPU-resident even when generating
    on CUDA; inside a session it lives on the resolved device. When you draw
    on a device, make sure the generator matches it (e.g.
    ``torch.randn(..., generator=rng, device=...)``).

    For tensor outputs, return a :class:`~tensordict.TensorDict` with a
    leading sample dimension (e.g. the denoised endpoint under ``"x1"``,
    per-sample log-probabilities under ``"logp"``). TensorDict is the
    recommended container: it survives ``torch.compile``, where arbitrary
    containers graph-break under ``compile_generate``. Returning a
    :class:`~nvalchemi.data.Batch` directly is equally supported — with no
    ``batch_mapping`` set the driver returns it untouched. Library-native
    sampling loops (e.g. PhysicsNeMo diffusion samplers) plug in through a
    thin adapter with this signature.

    The returned sample is mapped to a :class:`~nvalchemi.data.Batch` by the
    driver when it has a ``batch_mapping`` set, and returned as-is otherwise.

    **Optional attributes.** A callable object (rather than a plain function)
    may carry attributes the :class:`AtomisticGenerator` reads as defaults when
    the driver does not set them explicitly:

    - ``condition`` — a :class:`ConditionFunction` mapping the call's raw
      inputs to the conditioned value the function is then called with
      (e.g. tiling a conditioning :class:`~nvalchemi.data.Batch` so each
      graph gets ``num_samples`` draws). The driver runs it between the
      ``BEFORE_CONDITION`` and ``AFTER_CONDITION`` dispatches, which fire
      only when a condition step is provided; an explicit driver
      ``condition_func`` takes precedence over this attribute. There is no
      default condition: functions implement their own per use case, binding
      any extra context in their closure.
    - ``device`` — a :class:`torch.device` (or parseable device string),
      used when the driver's ``device`` is unset.
    - ``consumes_fields`` / ``produces_fields`` — batch-field declarations,
      used when the driver does not pass them at construction.
    - ``to_spec()`` — a zero-argument method returning a
      :class:`~nvalchemi.training.BaseSpec` that captures the object's
      construction (typically via a module-level factory and
      :func:`~nvalchemi.training.create_model_spec`); used by the driver's
      serialization (``model_dump()`` / ``model_dump_json()``) in
      preference to dotted-path capture.
    """

    def __call__(
        self,
        inputs: InputT | AtomicData | Batch | None = None,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        **kwargs: Any,
    ) -> SampleT: ...


@runtime_checkable
class ConditionFunction(Protocol):
    """Translate a call's raw request into the generating function's working input.

    Receives the raw inputs, the resolved ``num_samples``, and the resolved
    ``rng``; returns the conditioned value the generating function is then
    called with. The return type is the procedure's own — conditioning may
    change the type (e.g. a SMILES string becomes a
    :class:`~nvalchemi.data.Batch`). There is no default: procedures
    implement their own per use case.

    The job is preparation, not enforcement: conditioning decides what the
    procedure is *asked* to do, not what it must emit. Constraints on the
    output (e.g. "structures with a carboxyl group") live elsewhere —
    guidance inside the generating function, rejection via hooks at
    :class:`~nvalchemi.gen.stages.GenerationStage.BEFORE_MAPPING` or a
    wrapper generating function, or an explicit stage in a
    :class:`~nvalchemi.gen.pipeline.GenerationPipeline`.
    """

    def __call__(
        self, inputs: Any, *, num_samples: int, rng: torch.Generator | None
    ) -> Any: ...


class AtomisticGenerator(BaseModel, HookRegistryMixin):
    """Generative inference driver: optionally condition the inputs, generate,
    optionally materialize to a :class:`~nvalchemi.data.Batch`.

    Attributes
    ----------
    generator_func
        The :class:`GeneratingFunction` driving generation. Required; the
        function owns the model when there is one. Serialized as a
        :class:`~nvalchemi.training.BaseSpec` payload — the object's own
        ``to_spec()`` when it provides one, else dotted import path — and
        rebuilt at validation.
    condition_func
        Optional :class:`ConditionFunction` run before generation: maps the
        call's raw inputs to the conditioned value the generating function is
        then called with. It prepares the request — it does not constrain
        what comes out. Takes precedence over the generating function's own
        ``condition`` attribute. When neither is provided, no conditioning
        happens — the inputs pass through untouched and the
        ``BEFORE_CONDITION``/``AFTER_CONDITION`` stages do not fire. Serialized
        like ``generator_func`` (its own ``to_spec()``, else dotted import
        path).
    batch_mapping
        Optional :data:`MaterializationFunction` ``batch_mapping(sample) ->
        Batch`` mapping the raw sample to a :class:`~nvalchemi.data.Batch`
        after generation. When unset, :meth:`sample` returns the raw sample
        as-is and ``AFTER_GENERATE`` hooks do not fire. The sample is whatever
        container the generating function produced;
        :class:`~tensordict.TensorDict` is the recommended container for
        tensor outputs.
    device
        Optional device pin (``"cpu"``, ``"cuda"``, ``"cuda:0"``, or a
        :class:`torch.device`). Validated at construction: ``cpu`` always
        passes; ``cuda[:i]`` requires :func:`torch.cuda.is_available` and an
        in-range index. When unset, the generating function's ``device``
        attribute is used; when neither is present, no device resolves and no
        stream or device-residency check applies.
    dedicated_stream
        When ``True`` (default), entering a session creates a dedicated CUDA
        stream if the resolved device is CUDA. When ``False``, no stream is
        created even then.
    hooks
        Generation hooks, each with a :class:`GenerationStage` ``stage``.
        Validated at registration; see :class:`~nvalchemi.hooks.HookRegistryMixin`.
        Serialized as attribute-faithful class specs (each ``__init__``
        parameter read back from the same-named attribute).
    consumes_fields
        Batch fields this generator's input reads (empty means
        unconditional). Defaults from the generating function's
        ``consumes_fields`` attribute; required by
        :class:`~nvalchemi.gen.pipeline.GenerationPipeline` for link
        validation.
    produces_fields
        Batch fields this generator's output carries (written or forwarded).
        Defaults from the generating function's ``produces_fields`` attribute.
    num_samples
        Independent draws requested per call, passed straight through to the
        generating function (per-entry vs total-draw semantics are the
        function's own). Per-call override via :meth:`sample`'s
        ``num_samples`` keyword.
    seed
        Optional base seed. Inside a session (``with gen:``) one
        :class:`torch.Generator` is seeded at entry and advanced per draw;
        outside a session each call derives
        ``torch.Generator().manual_seed(seed + step_count)``. A per-call
        ``rng=`` kwarg overrides both.
    step_count
        Runtime counter of generation calls — incremented in a ``finally``,
        so failed calls count too. Drives hook frequency gating. Excluded from
        serialization.
    compile_generate
        Compile the generating function with ``torch.compile`` — immediately
        via :meth:`compile`, or lazily at session entry. Default ``False``.
    compile_kwargs
        Keyword arguments forwarded to ``torch.compile``.

    Notes
    -----
    ``generator_func`` and the other callables are held as arbitrary types
    (``arbitrary_types_allowed=True``); they serialize to
    :class:`~nvalchemi.training.BaseSpec` payloads and back through the
    training spec machinery (:mod:`nvalchemi.training`).

    ``AtomisticGenerator`` is a context manager (``with gen:``): a session owns a
    dedicated CUDA stream (when the resolved device is CUDA and
    ``dedicated_stream`` is set), a session-scoped RNG, lazy compilation, and
    context-manager hooks. See :meth:`__enter__`.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    generator_func: GeneratingFunction
    batch_mapping: MaterializationFunction | None = None
    condition_func: ConditionFunction | None = None
    device: torch.device | None = Field(
        default=None,
        description=(
            "Device pin for generated batches: 'cpu' always valid, 'cuda[:i]' "
            "validated against host availability at construction. Defaults from "
            "generator_func.device when unset."
        ),
    )
    dedicated_stream: bool = Field(
        default=True,
        description=(
            "Create a dedicated CUDA stream at session entry when the resolved "
            "device is CUDA. Set False to suppress stream creation."
        ),
    )
    hooks: list[Hook] = Field(
        default_factory=list,
        description=(
            "Generation hooks, fired at GenerationStage points with a shared "
            "GenerationContext per call."
        ),
    )
    consumes_fields: frozenset[str] | None = Field(
        default=None,
        description=(
            "Batch fields the input carries (empty = unconditional). "
            "Defaults from generator_func.consumes_fields when available. "
            "Checked against the (conditioned) inputs on every call."
        ),
    )
    produces_fields: frozenset[str] | None = Field(
        default=None,
        description=(
            "Batch fields the output carries (written or forwarded). "
            "Defaults from generator_func.produces_fields when available. "
            "Checked against the materialized batch before each return."
        ),
    )
    num_samples: int = Field(
        default=1,
        ge=1,
        description="Independent draws per call, passed straight through to the generating function.",
    )
    seed: int | None = Field(
        default=None,
        description="Base seed for per-draw RNGs (seed + step_count per call).",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        exclude=True,
        description="Runtime counter of generation calls (hook frequency).",
    )
    compile_generate: bool = Field(
        default=False,
        description=(
            "Compile the generating function with torch.compile (immediate via "
            "compile(), or lazily at session entry)."
        ),
    )
    compile_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Keyword arguments forwarded to torch.compile; validated against "
            "its signature at construction and in compile()."
        ),
    )

    @property
    def _stage_type(self) -> type[GenerationStage]:
        """Hook stage enum accepted by this engine (see :class:`HookRegistryMixin`).

        A property rather than a class attribute: pydantic collects the
        mixin's ``_stage_type`` annotation as a private attribute initialised
        to ``None``, and a plain class assignment does not override that on
        instances.
        """
        return GenerationStage

    def model_post_init(self, __context: Any) -> None:
        """Register hooks (validating stages) and initialize run state.

        Hook ``stage`` values arriving as raw ints or name strings (e.g.
        deserialized from a spec payload) are coerced to :class:`GenerationStage`
        before registration.
        """
        for hook in self.hooks:
            stage = getattr(hook, "stage", None)
            if isinstance(stage, str):
                hook.stage = GenerationStage[stage]
            elif isinstance(stage, int):
                hook.stage = GenerationStage(stage)
        self._init_hooks(list(self.hooks))
        if self.batch_mapping is None and any(
            hook.stage is GenerationStage.AFTER_GENERATE for hook in self.hooks
        ):
            warnings.warn(
                "AtomisticGenerator has no batch_mapping but hooks target "
                "GenerationStage.AFTER_GENERATE: sample() returns the raw "
                "sample as-is and AFTER_GENERATE hooks never fire. Set "
                "batch_mapping, or retarget the hooks to BEFORE_MAPPING.",
                UserWarning,
                stacklevel=2,
            )
        self._ctx: GenerationContext | None = None
        self._stream: torch.cuda.Stream | None = None
        self._stream_ctx: Any = None
        self._session_rng: torch.Generator | None = None
        self._compiled_generate: Any = None

    @field_validator("device", mode="before")
    @classmethod
    def _validate_device(cls, value: Any) -> torch.device | None:
        """Normalize ``device`` to a :class:`torch.device` and validate availability.

        ``cpu`` is always valid. ``cuda[:i]`` requires
        :func:`torch.cuda.is_available` and an index within
        :func:`torch.cuda.device_count` when one is given. Other device types
        (e.g. ``mps``) are accepted as parsed.

        Parameters
        ----------
        value
            The raw ``device`` input: ``None``, a device string, or a
            :class:`torch.device`.

        Returns
        -------
        torch.device | None
            The normalized device.

        Raises
        ------
        ValueError
            If the string cannot be parsed, the input is neither a string nor
            a :class:`torch.device`, or a CUDA device is requested that the
            host cannot provide.
        """
        if value is None:
            return None
        if isinstance(value, torch.device):
            device = value
        elif isinstance(value, str):
            try:
                device = torch.device(value)
            except RuntimeError as e:
                raise ValueError(f"Invalid device string {value!r}: {e}") from e
        else:
            raise ValueError(
                f"device must be a string or torch.device, got {type(value).__name__}."
            )
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise ValueError(
                    f"device={device} requests CUDA, but "
                    "torch.cuda.is_available() is False on this host."
                )
            count = torch.cuda.device_count()
            if device.index is not None and device.index >= count:
                raise ValueError(
                    f"device={device} is out of range: {count} CUDA device(s) "
                    "available on this host."
                )
        return device

    @model_validator(mode="after")
    def _default_field_declarations(self) -> AtomisticGenerator:
        """Default field declarations from ``generator_func`` attributes when present.

        Explicit constructor values always win; declarations stay ``None``
        (undeclared) when the generating function carries no such attributes —
        :class:`~nvalchemi.gen.pipeline.GenerationPipeline` validation then
        fails fast on the undeclared stage.

        Returns
        -------
        AtomisticGenerator
            The generator with declarations defaulted.
        """
        consumes = self.consumes_fields
        if consumes is None:
            consumes = getattr(self.generator_func, "consumes_fields", None)
        if consumes is not None:
            self.consumes_fields = frozenset(consumes)
        produces = self.produces_fields
        if produces is None:
            produces = getattr(self.generator_func, "produces_fields", None)
        if produces is not None:
            self.produces_fields = frozenset(produces)
        return self

    @staticmethod
    def _check_compile_kwargs(kwargs: dict[str, Any]) -> None:
        """Raise ``ValueError`` for kwargs that are not ``torch.compile`` keywords.

        Parameters
        ----------
        kwargs
            The kwargs to check against the installed ``torch.compile``
            signature.

        Raises
        ------
        ValueError
            If a key is not a keyword argument of ``torch.compile``, or is
            ``model`` (the compile target, filled by :meth:`compile` with the
            generating function).
        """
        if not kwargs:
            return
        if "model" in kwargs:
            raise ValueError(
                "compile_kwargs must not contain 'model': compile() passes the "
                "generating function as the torch.compile target."
            )
        params = inspect.signature(torch.compile).parameters
        valid = {
            name
            for name, param in params.items()
            if param.kind is inspect.Parameter.KEYWORD_ONLY
        }
        invalid = set(kwargs) - valid
        if invalid:
            raise ValueError(
                f"compile_kwargs {sorted(invalid)} are not keyword arguments of "
                f"torch.compile; valid options: {sorted(valid)}."
            )

    @model_validator(mode="after")
    def _validate_compile_kwargs(self) -> AtomisticGenerator:
        """Check ``compile_kwargs`` against the installed ``torch.compile`` signature.

        Returns
        -------
        AtomisticGenerator
            The validated generator.
        """
        self._check_compile_kwargs(self.compile_kwargs)
        return self

    def compile(self, **kwargs: Any) -> AtomisticGenerator:
        """Compile the generating function with ``torch.compile``.

        Merges *kwargs* with :attr:`compile_kwargs` (values passed here win),
        sets :attr:`compile_generate`, and wraps ``generator_func``.
        Calling again re-compiles with the new kwargs.

        Only the generating function is compiled; ``batch_mapping`` and hook
        dispatch run eagerly because they build data structures and would
        graph-break anyway. Non-tensor-pure generating functions will
        graph-break under ``torch.compile``; compile the model inside such
        functions directly instead.

        Parameters
        ----------
        **kwargs
            Forwarded to ``torch.compile``.

        Returns
        -------
        AtomisticGenerator
            This instance, for fluent chaining.
        """
        merged = {**self.compile_kwargs, **kwargs}
        self._check_compile_kwargs(merged)
        self.compile_kwargs = merged
        self.compile_generate = True
        self._compiled_generate = torch.compile(self.generator_func, **merged)
        return self

    def _infer_device(self) -> torch.device | None:
        """Resolve the device: the ``device`` field, then ``generator_func.device``.

        A string attribute on the generating function is parsed via
        :class:`torch.device`; attribute values of any other type are ignored
        (treated as no device declared).

        Returns
        -------
        torch.device | None
            The resolved device, or ``None`` when neither source provides one.

        Raises
        ------
        ValueError
            If the generating function's ``device`` attribute is a string that
            cannot be parsed as a device.
        """
        device = self.device
        if device is None:
            device = getattr(self.generator_func, "device", None)
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except RuntimeError as e:
                raise ValueError(
                    f"generator_func.device attribute {device!r} is not a valid "
                    f"device string: {e}"
                ) from e
        return device if isinstance(device, torch.device) else None

    def __enter__(self) -> AtomisticGenerator:
        """Enter a generation session.

        Creates and enters a dedicated CUDA stream when the resolved device
        is CUDA and :attr:`dedicated_stream` is set (skipped when a pipeline
        has already supplied a stream), seeds the session RNG from
        :attr:`seed`, compiles the generating function when
        :attr:`compile_generate` is set, and opens any context-manager hooks.
        When no device resolves — the driver has no ``device`` and the
        generating function declares none — no stream is created, and the
        session owns only the RNG (when :attr:`seed` is set) and hook
        lifecycles.

        Returns
        -------
        AtomisticGenerator
            This instance.
        """
        if self._stream is None:
            device = self._infer_device()
            if device is not None and device.type == "cuda" and self.dedicated_stream:
                self._stream = torch.cuda.Stream(device=device)
                self._stream_ctx = torch.cuda.stream(self._stream)
                self._stream_ctx.__enter__()
        if self.seed is not None and self._session_rng is None:
            device = self._infer_device()
            rng_device = (
                device if device is not None and device.type == "cuda" else "cpu"
            )
            self._session_rng = torch.Generator(device=rng_device).manual_seed(
                self.seed
            )
        if self.compile_generate and self._compiled_generate is None:
            self.compile()
        for hook in self.hooks:
            enter = getattr(hook, "__enter__", None)
            if enter is not None:
                enter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the session: close hooks, exit the stream, drop session RNG.

        Parameters
        ----------
        exc_type, exc_val, exc_tb
            The active exception, if any.
        """
        for hook in self.hooks:
            exit_ = getattr(hook, "__exit__", None)
            if exit_ is not None:
                exit_(None, None, None)
            else:
                close = getattr(hook, "close", None)
                if close is not None:
                    close()
        if self._stream_ctx is not None:
            self._stream_ctx.__exit__(exc_type, exc_val, exc_tb)
        self._stream = None
        self._stream_ctx = None
        self._session_rng = None

    def _build_context(self, batch: Batch | None) -> GenerationContext:
        """Return the live per-call context built by :meth:`sample`.

        Hooks always see the same context object within a call, so mutations
        made at one stage are visible to later stages and to the core steps
        in between. Outside a call, a minimal context is built on demand.

        Parameters
        ----------
        batch : Batch | None
            Current batch; used only when no per-call context is live.

        Returns
        -------
        GenerationContext
            The live or freshly built context.
        """
        if self._ctx is not None:
            return self._ctx
        return GenerationContext(batch=batch, workflow=self)
        ctx = getattr(self, "_ctx", None)
        if ctx is not None:
            return ctx
        from nvalchemi.training.distributed import get_rank

        return GenerationContext(
            batch=batch,
            model=None,
            global_rank=get_rank(None),
            workflow=self,
            step_count=self.step_count,
        )

    def sample(
        self,
        inputs: Any = None,
        *,
        num_samples: int | None = None,
        **kwargs: Any,
    ) -> Any:
        """Generate samples for the given inputs.

        Runs the per-call pipeline shown in the module docstring: condition
        (only if a condition callable resolved for the call), generate on the
        session's CUDA stream, materialize (only if ``batch_mapping`` is
        set). Hooks fire at the
        :class:`~nvalchemi.gen.stages.GenerationStage` points — each stage
        only when the step it brackets ran — and share one
        :class:`~nvalchemi.hooks.GenerationContext` for the whole call.

        Parameters
        ----------
        inputs
            The input for this call — a :class:`~nvalchemi.data.Batch`,
            another tensor container, or ``None`` for unconditional
            generation. When a condition step resolved for the call it runs on
            this value first and the generating function receives the
            conditioned result; otherwise the input is forwarded untouched.
        num_samples
            Per-call override for :attr:`num_samples`, passed straight
            through to the generating function.
        **kwargs
            Per-call options forwarded to the generating function (e.g.
            ``mask`` for inpainting). One reserved kwarg, ``rng=``, is
            consumed by ``sample`` itself to override the session/seeded RNG
            for this call.

        Returns
        -------
        Any
            The generated :class:`~nvalchemi.data.Batch` when
            ``batch_mapping`` is set — post-filter, so possibly with fewer
            graphs than were sampled. A mapping may return a zero-graph
            ``Batch`` (built via :meth:`~nvalchemi.data.Batch.empty`) to
            signal that nothing was accepted; :meth:`sample` returns it
            as-is (subsetting a batch to zero graphs still raises
            ``IndexError``). When ``batch_mapping`` is unset, the generating
            function's raw output is returned as-is and ``AFTER_GENERATE``
            hooks do not fire.

        Raises
        ------
        TypeError
            If ``batch_mapping`` is set but its result is not a
            :class:`~nvalchemi.data.Batch` — the mapping must return a
            ``Batch``.
        ValueError
            If ``num_samples`` resolves to less than 1, or a device resolved
            (via ``device`` or ``generator_func.device``) but the
            materialized batch lives on a different device.
        """
        from nvalchemi.training.distributed import get_rank

        ctx = GenerationContext(
            batch=inputs if isinstance(inputs, Batch) else None,
            global_rank=get_rank(None),
            workflow=self,
            inputs=inputs,
            step_count=self.step_count,
        )
        self._ctx = ctx
        try:
            n_draws = num_samples if num_samples is not None else self.num_samples
            if n_draws < 1:
                raise ValueError(f"num_samples must be positive, got {n_draws}")
            # Resolve RNG: per-call override, then session, then seed-derived
            rng = kwargs.pop("rng", None)
            if rng is None:
                if self._session_rng is not None:
                    rng = self._session_rng
                elif self.seed is not None:
                    rng = torch.Generator().manual_seed(self.seed + ctx.step_count)
            # Resolve condition: driver's condition_func > generator_func.condition
            condition = self.condition_func
            if condition is None:
                condition = getattr(self.generator_func, "condition", None)
            if condition is not None and callable(condition):
                self._call_hooks(GenerationStage.BEFORE_CONDITION, None)
                ctx.inputs = condition(ctx.inputs, num_samples=n_draws, rng=rng)
                self._call_hooks(GenerationStage.AFTER_CONDITION, None)
            if self.consumes_fields:
                if ctx.inputs is None or not hasattr(ctx.inputs, "__contains__"):
                    raise TypeError(
                        "AtomisticGenerator declares consumes_fields="
                        f"{sorted(self.consumes_fields)}, but the call's inputs "
                        f"({type(ctx.inputs).__name__}) are not a "
                        "field-addressable container (Batch or TensorDict). Pass "
                        "inputs carrying those fields, or fix the declaration."
                    )
                missing = [
                    f for f in sorted(self.consumes_fields) if f not in ctx.inputs
                ]
                if missing:
                    raise ValueError(
                        "AtomisticGenerator declares consumes_fields but the "
                        f"call's inputs lack {missing}. Provide the fields in the "
                        "inputs or in the condition step, or fix the declaration."
                    )
            with (
                torch.cuda.stream(self._stream)
                if self._stream is not None
                else nullcontext()
            ):
                gen_fn = self._compiled_generate or self.generator_func
                ctx.sample = gen_fn(
                    ctx.inputs,
                    num_samples=n_draws,
                    rng=rng,
                    **kwargs,
                )
                self._call_hooks(GenerationStage.BEFORE_MAPPING, None)
                if self.batch_mapping is None:
                    return ctx.sample
                ctx.batch = self.batch_mapping(ctx.sample)
            if not isinstance(ctx.batch, Batch):
                raise TypeError(
                    "AtomisticGenerator batch_mapping produced "
                    f"{type(ctx.batch).__name__}, not a Batch: the mapping "
                    "must return a Batch. Leave batch_mapping unset to return "
                    "the raw sample as-is."
                )
            device = self._infer_device()
            if (
                device is not None
                and not torch.compiler.is_compiling()
                # a zero-graph batch (total rejection) carries no device state
                and ctx.batch.num_graphs > 0
            ):
                batch_device = ctx.batch.device
                if device.type != batch_device.type or (
                    device.index is not None
                    and batch_device.index is not None
                    and device.index != batch_device.index
                ):
                    raise ValueError(
                        f"batch_mapping materialized the batch on device "
                        f"'{batch_device}', but the resolved device is "
                        f"'{device}'. Move the batch onto the resolved device "
                        "inside the generating function or batch_mapping, or "
                        "fix the device chain (AtomisticGenerator.device / "
                        "generator_func.device)."
                    )
            self._call_hooks(GenerationStage.AFTER_GENERATE, None)
            batch = ctx.batch
            if not isinstance(batch, Batch):
                raise TypeError(
                    "AFTER_GENERATE hooks must leave ctx.batch a Batch, got "
                    f"{type(batch).__name__}."
                )
            if self.produces_fields:
                missing = [f for f in sorted(self.produces_fields) if f not in batch]
                if missing:
                    raise ValueError(
                        "AtomisticGenerator declares produces_fields but the "
                        f"returned batch lacks {missing}. Add the fields in the "
                        "generating function or batch_mapping, or fix the "
                        "declaration."
                    )
            return batch
        finally:
            self._ctx = None
            self.step_count += 1

    def __call__(self, inputs: Any = None, **kwargs: Any) -> Any:
        """Syntactic sugar for :meth:`sample`.

        Parameters
        ----------
        inputs
            The input for this call; forwarded to :meth:`sample`.
        **kwargs
            Forwarded to :meth:`sample`.

        Returns
        -------
        Any
            The generated batch, or the raw sample when no ``batch_mapping``
            is set.
        """
        return self.sample(inputs, **kwargs)

    def stream(
        self,
        inputs: Any = None,
        *,
        max_batches: int | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Iterate over inputs, calling :meth:`sample` once per item.

        Pass any iterable of inputs, or ``None`` for repeated unconditional
        draws. ``stream(None)`` without ``max_batches`` is an infinite
        iterator.

        Parameters
        ----------
        inputs
            Iterable of inputs, or ``None`` for unconditional draws.
        max_batches
            Stop after this many batches; ``None`` follows ``inputs``. When
            ``inputs`` is ``None`` and ``max_batches`` is unset the stream is
            unbounded.
        **kwargs
            Per-call options forwarded to :meth:`sample`.

        Yields
        ------
        Any
            Each :meth:`sample` result exactly as produced: a
            :class:`~nvalchemi.data.Batch` when ``batch_mapping`` is set,
            otherwise the raw sample. A mapping may signal total rejection
            with :meth:`~nvalchemi.data.Batch.empty` — that zero-graph batch
            is yielded, so rejection-aware consumers should check
            ``num_graphs == 0``. Retrying rejected draws is the consumer's job.
        """
        if inputs is None:
            inputs = itertools.repeat(None)
        for index, item in enumerate(inputs):
            if max_batches is not None and index >= max_batches:
                return
            yield self.sample(item, **kwargs)

    def __iter__(self) -> Iterator[Any]:
        """Thin sugar for :meth:`stream` with default arguments.

        Returns
        -------
        Iterator
            ``self.stream()`` — an unbounded stream of unconditional draws.
        """
        return self.stream()

    def __or__(self, other: Any) -> GenerationPipeline:
        """Compose sequentially into a :class:`GenerationPipeline`.

        Mirrors :meth:`nvalchemi.dynamics.base.BaseDynamics.__or__`. (``+``
        stays reserved for concurrent/fused composition, as in dynamics.)

        Parameters
        ----------
        other
            A :class:`AtomisticGenerator`, a dynamics engine, a ``Batch -> Batch``
            callable, or an existing pipeline.

        Returns
        -------
        GenerationPipeline
            ``self`` followed by ``other`` (prepended when ``other`` is
            already a pipeline).
        """
        from nvalchemi.gen.pipeline import GenerationPipeline

        if isinstance(other, GenerationPipeline):
            return GenerationPipeline(stages=[self, *other.stages])
        return GenerationPipeline(stages=[self, other])
