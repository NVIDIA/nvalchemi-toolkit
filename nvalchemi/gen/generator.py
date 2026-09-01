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
"""The :class:`AtomGenerator`: the toolkit's abstract interface for generative inference.

Conceptually, :class:`AtomGenerator` is a high-level wrapper that ensures
generative _workflows_ can be readily pipelined into the general dynamics
interface, as well as the ability to generate samples. It is intentionally
kept abstract so as to support a wide range of generative modeling
approaches — GANs, VAEs, flow matching and diffusion, genetic algorithms, and so on.

It wires a generative ``model`` to a user-supplied :class:`GeneratingFunction`
— the callable that owns the family-specific generation procedure — and maps
the result to a :class:`~nvalchemi.data.Batch`. The fixed pipeline per
:meth:`AtomGenerator.sample` is::

    BEFORE_CONDITION    hooks
                        ctx.batch = condition(ctx.cond, num_samples_per_batch)
    AFTER_CONDITION     hooks
                        sample = generator_func(model, ..., cond=ctx.batch)
                        ctx.batch = to_batch(sample, ctx.batch)  # materialize
    AFTER_GENERATE      hooks  (filtering = subsetting ctx.batch)
    return ctx.batch

Hooks registered at the three
:class:`~nvalchemi.gen.stages.GenerationStage` points all receive one shared
:class:`~nvalchemi.hooks.GenerationContext` per call and mutate it by
replacing its fields; the ``AtomGenerator`` re-reads the context after each
dispatch. See the stage enum for what may change where.

Model contract — defaults everywhere
------------------------------------
``model`` is typed ``Any``: no protocol accurately describes a fully
defaulted contract. What the pipeline reads from the model:

* ``condition(cond, num_samples)`` — optional; falls back to
  :func:`~nvalchemi.gen.default_condition` (passthrough + tile).
* ``generate(*, num_samples, rng, cond, **kwargs)`` — required only when no
  ``generator_func`` is supplied (checked by a construction validator).
* ``to_batch(sample, batch)`` — required only when no ``output_to_batch_func``
  is supplied (checked by a construction validator).

``forward`` / ``adapt_output`` are never read by the ``AtomGenerator``; they
belong to the model-side contract
(:class:`~nvalchemi.models.gen.base.GenerativeModelMixin`) that typed
generating functions rely on.

Composition examples
--------------------
**GAN** (noise → net, one forward pass)::

    def gan_generate(model, *, num_samples=1, rng=None, cond=None, **kwargs):
        z = torch.randn(num_samples, model.latent_dim, generator=rng)
        return TensorDict({"x1": model.decode(z)}, batch_size=[num_samples])

    gan = AtomGenerator(
        model=GANModel(), generator_func=gan_generate,
        output_to_batch_func=to_batch,
    )

**Streaming** — one ``sample()`` call per conditioning item; ``None`` means
repeated unconditional draws::

    for batch in gan.stream(conds=None, max_batches=10):
        ...

**Composition** — sequential pipelines mirror the dynamics ``|`` sugar
(:class:`~nvalchemi.gen.pipeline.GenerationPipeline`)::

    pipe = gen_a | gen_b
    out = pipe(cond)

**Sessions: streams, RNG, and compile** — an ``AtomGenerator`` is a context
manager. Entering a session (``with gen:``) creates a dedicated CUDA stream
when the model is CUDA-resident, seeds a session-scoped
:class:`torch.Generator` (advanced per draw), compiles the generating
function when ``compile_generate`` is set, and opens context-manager hooks;
exiting unwinds all of it::

    with gen.compile(fullgraph=True):
        for batch in gen.stream(conds):
            ...
"""

from __future__ import annotations

import inspect
import itertools
from collections.abc import Iterator
from contextlib import nullcontext
from typing import TYPE_CHECKING, Any, Callable, Protocol, runtime_checkable

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tensordict import TensorDict, TensorDictBase

from nvalchemi.data import AtomicData, Batch
from nvalchemi.gen.stages import GenerationStage
from nvalchemi.hooks import GenerationContext, Hook, HookRegistryMixin

if TYPE_CHECKING:
    from nvalchemi.gen.pipeline import GenerationPipeline

__all__ = ["GeneratingFunction", "AtomGenerator", "default_condition"]


@runtime_checkable
class GeneratingFunction(Protocol):
    """Callable that encapsulates a family-specific generation strategy.

    A :class:`GeneratingFunction` samples from a generative model and emits a
    sample :class:`~tensordict.TensorDict` — a collection of named tensors
    (positions, atom types, lattice, ...), so multi-tensor families are
    first-class. It is the single extension point that lets the
    :class:`AtomGenerator` support diffusion, flow matching, GANs, VAEs,
    normalizing flows, and population-based methods without the toolkit
    hardcoding any one workflow.

    The contract is deliberately minimal — one calling
    convention for every family:

    * ``model`` — the generative model (the only required argument).
    * ``num_samples`` — number of independent draws requested for this call.
    * ``rng`` — optional :class:`torch.Generator` for reproducibility.
    * ``cond`` — the conditioning batch built by ``model.condition`` (or the
      module-level default), ``None`` for unconditional generation.
    * ``**kwargs`` — family-specific per-call options (e.g. ``mask`` for
      inpainting). Family-specific *config* a generating function needs
      (e.g. a diffusion schedule/sampler) is bound in the closure via an
      importable factory, not threaded by the :class:`AtomGenerator`.

    There is exactly one calling convention — no signature inspection or
    dispatch in the core. External sampler ecosystems (e.g. physicsnemo
    diffusion samplers) plug in through a thin user-side adapter function
    with the same signature.

    The returned :class:`~tensordict.TensorDict` is mapped to a
    :class:`~nvalchemi.data.Batch` by the :class:`AtomGenerator` via
    ``output_to_batch_func`` or ``model.to_batch``.

    A model may alternatively provide the same contract as a ``generate``
    method, minus the leading ``model`` argument; the :class:`AtomGenerator`
    calls that fallback when no ``generator_func`` is supplied.
    """

    def __call__(
        self,
        model: Any,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        cond: Any = None,
        **kwargs: Any,
    ) -> TensorDict: ...


def default_condition(cond: Any, num_samples: int = 1) -> Any:
    """Default conditioning: pass through an already-built batch, tiled.

    Mirrors the default
    :meth:`~nvalchemi.models.gen.base.GenerativeModelMixin.condition`; the
    :class:`AtomGenerator` falls back to this function when the model defines
    no ``condition`` of its own.

    Parameters
    ----------
    cond
        An already-built :class:`~nvalchemi.data.Batch` or
        :class:`~nvalchemi.data.AtomicData`, another tensor container (e.g.
        :class:`~tensordict.TensorDict`), or ``None`` for unconditional
        generation.
    num_samples
        Number of independent draws per conditioning graph.

    Returns
    -------
    Any
        For a :class:`~nvalchemi.data.Batch` /
        :class:`~nvalchemi.data.AtomicData`, a ``Batch`` with each
        conditioning graph repeated ``num_samples`` times; ``None`` passes
        through as ``None``; any other container passes through unchanged
        (replication semantics for non-batch containers belong to the model
        or generating function).

    Raises
    ------
    ValueError
        If ``num_samples`` is not positive.
    """
    if cond is None:
        return None
    if num_samples < 1:
        raise ValueError(f"num_samples must be positive, got {num_samples}")
    if isinstance(cond, Batch):
        graphs = [
            cond.get_data(i) for i in range(cond.num_graphs) for _ in range(num_samples)
        ]
        return Batch.from_data_list(graphs, device=cond.device)
    if isinstance(cond, AtomicData):
        return Batch.from_data_list([cond] * num_samples, device=cond.device)
    return cond


class AtomGenerator(BaseModel, HookRegistryMixin):
    """Abstract generative inference pipeline.

    Attributes
    ----------
    model
        The generative model. Typed ``Any`` with a documented contract (see
        the module docstring): ``condition``/``generate``/``to_batch`` are
        all optional-with-default or conditionally required;
        ``model.model_config`` may supply default field declarations.
        Non-serializable (weights live in the checkpoint machinery).
    generator_func
        User-supplied :class:`GeneratingFunction`; takes precedence over
        ``model.generate``. Non-serializable.
    output_to_batch_func
        Optional callable ``output_to_batch_func(sample, batch) -> Batch``
        overriding ``model.to_batch`` for mapping a sample
        :class:`~tensordict.TensorDict` to a :class:`~nvalchemi.data.Batch`.
        Swapping it lets the same model emit a ``Batch`` for dynamics or a
        different artifact for file-writing.
    hooks
        Generation hooks, each with a :class:`GenerationStage` ``stage``.
        Validated at registration; see :class:`~nvalchemi.hooks.HookRegistryMixin`.
    consumes_fields
        Batch fields this generator's conditioning reads (empty means
        unconditional). Defaults from ``model.model_config`` when the model
        provides one; required by
        :class:`~nvalchemi.gen.pipeline.GenerationPipeline` for link
        validation.
    produces_fields
        Batch fields this generator's output carries (written or forwarded).
        Defaults from ``model.model_config`` when present.
    num_samples_per_batch
        Independent draws **per conditioning entry**, fed to
        ``model.condition``. The model emits a batch of
        ``len(cond) * num_samples_per_batch`` samples per call.
    seed
        Optional base seed. Inside a session (``with gen:``) one
        :class:`torch.Generator` is seeded at entry and advanced per draw;
        outside a session each call derives
        ``torch.Generator().manual_seed(seed + step_count)``. A per-call
        ``rng=`` kwarg overrides both.
    step_count
        Runtime counter of completed generation calls; drives hook frequency
        gating. Excluded from serialization.
    compile_generate
        Compile the generating function with ``torch.compile`` — immediately
        via :meth:`compile`, or lazily at session entry. Default ``False``.
    compile_kwargs
        Keyword arguments forwarded to ``torch.compile``.

    Notes
    -----
    ``model`` and ``generator_func`` are held as arbitrary types
    (``arbitrary_types_allowed=True``); they are the non-serializable fields.

    ``AtomGenerator`` is a context manager (``with gen:``): a session owns a
    dedicated CUDA stream (when the model is CUDA-resident), a
    session-scoped RNG, lazy compilation, and context-manager hooks. See
    :meth:`__enter__`.

    At least one generation source (``generator_func`` or ``model.generate``)
    and one materialization target (``output_to_batch_func`` or
    ``model.to_batch``) are required; construction validators
    enforce both.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    model: Any
    generator_func: GeneratingFunction | None = None
    output_to_batch_func: Callable[[TensorDict, Any], Batch] | None = None
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
            "Batch fields conditioning reads (empty = unconditional). "
            "Defaults from model.model_config when available."
        ),
    )
    produces_fields: frozenset[str] | None = Field(
        default=None,
        description=(
            "Batch fields the output carries (written or forwarded). "
            "Defaults from model.model_config when available."
        ),
    )
    num_samples_per_batch: int = Field(
        default=1,
        ge=1,
        description="Independent draws per conditioning entry, fed to model.condition.",
    )
    seed: int | None = Field(
        default=None,
        description="Base seed for per-draw RNGs (seed + step_count per call).",
    )
    step_count: int = Field(
        default=0,
        ge=0,
        exclude=True,
        description="Runtime counter of completed generation calls (hook frequency).",
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
        rehydrated from a spec) are coerced to :class:`GenerationStage`
        before registration.
        """
        for hook in self.hooks:
            stage = getattr(hook, "stage", None)
            if isinstance(stage, str):
                hook.stage = GenerationStage[stage]
            elif isinstance(stage, int):
                hook.stage = GenerationStage(stage)
        self._init_hooks(list(self.hooks))
        self._ctx: GenerationContext | None = None
        self._stream: torch.cuda.Stream | None = None
        self._stream_ctx: Any = None
        self._session_rng: torch.Generator | None = None
        self._compiled_generate: Any = None

    @model_validator(mode="after")
    def _require_generation_source(self) -> AtomGenerator:
        """Ensure a generation source is available.

        Returns
        -------
        AtomGenerator
            The validated generator.

        Raises
        ------
        TypeError
            If neither ``generator_func`` nor a callable ``model.generate``
            is present.
        """
        if self.generator_func is None and not callable(
            getattr(self.model, "generate", None)
        ):
            raise TypeError(
                "AtomGenerator requires a generation source: pass generator_func= "
                "or implement model.generate(*, num_samples, rng, cond, **kwargs)."
            )
        return self

    @model_validator(mode="after")
    def _require_materialization(self) -> AtomGenerator:
        """Ensure a materialization target is available.

        Returns
        -------
        AtomGenerator
            The validated generator.

        Raises
        ------
        TypeError
            If neither ``output_to_batch_func`` nor a callable
            ``model.to_batch`` is present.
        """
        if self.output_to_batch_func is None and not callable(
            getattr(self.model, "to_batch", None)
        ):
            raise TypeError(
                "AtomGenerator requires a materialization target: pass "
                "output_to_batch_func= or implement model.to_batch(sample, batch)."
            )
        return self

    @model_validator(mode="after")
    def _default_field_declarations(self) -> AtomGenerator:
        """Default field declarations from ``model.model_config`` when present.

        Explicit constructor values always win; declarations stay ``None``
        (undeclared) when the model carries no config or its config has no
        such attributes.

        Returns
        -------
        AtomGenerator
            The generator with declarations defaulted.
        """
        config = getattr(self.model, "model_config", None)
        if config is not None:
            consumes = getattr(config, "consumes_fields", None)
            if self.consumes_fields is None and consumes is not None:
                self.consumes_fields = frozenset(consumes)
            produces = getattr(config, "produces_fields", None)
            if self.produces_fields is None and produces is not None:
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
    def _validate_compile_kwargs(self) -> AtomGenerator:
        """Check ``compile_kwargs`` against the installed ``torch.compile`` signature.

        Returns
        -------
        AtomGenerator
            The validated generator.
        """
        self._check_compile_kwargs(self.compile_kwargs)
        return self

    def compile(self, **kwargs: Any) -> AtomGenerator:
        """Compile the generating function with ``torch.compile``.

        Merges *kwargs* with :attr:`compile_kwargs` (values passed here win),
        sets :attr:`compile_generate`, and wraps the resolved generating
        function (``generator_func``, or ``model.generate`` when no
        ``generator_func`` is set). Idempotent in intent but **will**
        re-compile if called again (e.g. with different kwargs).

        The compiled unit is deliberately the generating function only:
        conditioning, materialization, and hook dispatch stay eager (they
        build data structures — graph breaks anyway). ``torch.compile`` on an
        arbitrary user callable is best-effort; non-tensor-pure code will
        graph-break. Users with exotic generating functions can instead
        compile their model themselves.

        Parameters
        ----------
        **kwargs
            Forwarded to ``torch.compile``.

        Returns
        -------
        AtomGenerator
            This instance, for fluent chaining.
        """
        merged = {**self.compile_kwargs, **kwargs}
        self._check_compile_kwargs(merged)
        self.compile_kwargs = merged
        self.compile_generate = True
        target = (
            self.generator_func
            if self.generator_func is not None
            else self.model.generate
        )
        self._compiled_generate = torch.compile(target, **merged)
        return self

    def _infer_device(self) -> torch.device | None:
        """Best-effort device inference from the model (parameters first).

        Returns
        -------
        torch.device | None
            The model's device, or ``None`` when it cannot be inferred.
        """
        model = self.model
        if isinstance(model, torch.nn.Module):
            try:
                return next(model.parameters()).device
            except StopIteration:
                pass
        device = getattr(model, "device", None)
        return device if isinstance(device, torch.device) else None

    def __enter__(self) -> AtomGenerator:
        """Enter a generation session.

        Creates and enters a dedicated CUDA stream when the model is
        CUDA-resident (skipped when a pipeline has already supplied one),
        seeds the session RNG from :attr:`seed`, compiles the generating
        function when :attr:`compile_generate` is set, and opens any
        context-manager hooks.

        Returns
        -------
        AtomGenerator
            This instance.
        """
        if self._stream is None:
            device = self._infer_device()
            if device is not None and device.type == "cuda":
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
        ctx = getattr(self, "_ctx", None)
        if ctx is not None:
            return ctx
        from nvalchemi.training.distributed import get_rank

        return GenerationContext(
            batch=batch,
            model=self.model,
            global_rank=get_rank(None),
            workflow=self,
            step_count=self.step_count,
        )

    def _generate(
        self, ctx: GenerationContext, num_samples: int, **kwargs: Any
    ) -> TensorDict:
        """Invoke the generating function and enforce the TensorDict contract.

        Parameters
        ----------
        ctx
            The live generation context (``ctx.batch`` is the conditioning
            batch).
        num_samples
            Number of independent draws requested for this call.
        **kwargs
            Per-call options forwarded to the generating function; ``rng``
            is consumed here.

        Returns
        -------
        TensorDict
            The sample as a collection of named tensors.

        Raises
        ------
        TypeError
            If the generating function does not return a TensorDict.
        """
        rng = kwargs.pop("rng", None)
        if rng is None:
            if self._session_rng is not None:
                rng = self._session_rng
            elif self.seed is not None:
                rng = torch.Generator().manual_seed(self.seed + ctx.step_count)
        if self.generator_func is not None:
            gen_fn = self._compiled_generate or self.generator_func
            sample = gen_fn(
                self.model,
                num_samples=num_samples,
                rng=rng,
                cond=ctx.batch,
                **kwargs,
            )
        else:
            gen_fn = self._compiled_generate or self.model.generate
            sample = gen_fn(
                num_samples=num_samples,
                rng=rng,
                cond=ctx.batch,
                **kwargs,
            )
        if not isinstance(sample, TensorDictBase):
            raise TypeError(
                "Generating functions must return a TensorDict of named sample "
                f"tensors, got {type(sample).__name__}. Wrap the output, e.g. "
                "TensorDict({'x1': x1}, batch_size=[x1.shape[0]])."
            )
        return sample

    def sample(
        self,
        cond: Any = None,
        *,
        num_samples_per_batch: int | None = None,
        **kwargs: Any,
    ) -> Batch:
        """Generate samples for a conditioning spec.

        Runs the fixed pipeline (condition → generate, with the raw sample
        materialized into a :class:`~nvalchemi.data.Batch` as part of the
        generate step) with hook dispatch at the three
        :class:`~nvalchemi.gen.stages.GenerationStage` points, sharing one
        :class:`~nvalchemi.hooks.GenerationContext` across the call.

        Parameters
        ----------
        cond
            Conditioning input — a :class:`~nvalchemi.data.Batch`, another
            tensor container, or ``None`` for unconditional generation.
            Passed to ``model.condition`` (or the module-level default) to
            build the conditioning batch.
        num_samples_per_batch
            Per-call override for :attr:`num_samples_per_batch` (draws per
            conditioning entry).
        **kwargs
            Per-call options forwarded to the generating function (e.g.
            ``mask`` for inpainting); ``rng`` is consumed here.

        Returns
        -------
        Batch
            The generated batch — post-filter, so possibly with fewer graphs
            than were sampled. (A filter that rejects every graph currently
            raises ``IndexError``: :class:`~nvalchemi.data.Batch` does not
            support zero-graph selections.)
        """
        from nvalchemi.training.distributed import get_rank

        ctx = GenerationContext(
            batch=None,
            model=self.model,
            global_rank=get_rank(None),
            workflow=self,
            cond=cond,
            step_count=self.step_count,
        )
        self._ctx = ctx
        try:
            self._call_hooks(GenerationStage.BEFORE_CONDITION, None)

            condition = getattr(self.model, "condition", None)
            if not callable(condition):
                condition = default_condition
            n_draws = (
                num_samples_per_batch
                if num_samples_per_batch is not None
                else self.num_samples_per_batch
            )
            if n_draws < 1:
                raise ValueError(
                    f"num_samples_per_batch must be positive, got {n_draws}"
                )
            ctx.batch = condition(ctx.cond, n_draws)
            self._call_hooks(GenerationStage.AFTER_CONDITION, None)

            with (
                torch.cuda.stream(self._stream)
                if self._stream is not None
                else nullcontext()
            ):
                sample = self._generate(ctx, num_samples=n_draws, **kwargs)
                recon = self.output_to_batch_func or getattr(
                    self.model, "to_batch", None
                )
                if (
                    recon is None
                ):  # pragma: no cover - guarded by a construction validator
                    raise TypeError(
                        "AtomGenerator needs output_to_batch_func or model.to_batch "
                        "to materialize generation outputs into a Batch."
                    )
                ctx.batch = recon(sample, ctx.batch)
            if not isinstance(ctx.batch, Batch):
                raise TypeError(
                    "Materialization must return a Batch; got "
                    f"{type(ctx.batch).__name__}."
                )
            self._call_hooks(GenerationStage.AFTER_GENERATE, None)
            batch = ctx.batch
            if not isinstance(batch, Batch):
                raise TypeError(
                    "AFTER_GENERATE hooks must leave ctx.batch a Batch, got "
                    f"{type(batch).__name__}."
                )
            return batch
        finally:
            self._ctx = None
            self.step_count += 1

    def __call__(self, cond: Any = None, **kwargs: Any) -> Batch:
        """Syntactic sugar for :meth:`sample`.

        Parameters
        ----------
        cond
            Conditioning input; forwarded to :meth:`sample`.
        **kwargs
            Forwarded to :meth:`sample`.

        Returns
        -------
        Batch
            The generated batch.
        """
        return self.sample(cond, **kwargs)

    def stream(
        self,
        conds: Any = None,
        *,
        max_batches: int | None = None,
        **kwargs: Any,
    ) -> Iterator[Batch]:
        """Stream generated batches as a dynamics data source.

        One :meth:`sample` call per conditioning item. ``conds`` *is* the data
        source: pass any iterable of conditioning inputs, or ``None`` for
        repeated unconditional draws.

        Parameters
        ----------
        conds
            Iterable of conditioning inputs, or ``None`` for unconditional
            draws.
        max_batches
            Cap on batches yielded (``None`` means unbounded — follow
            ``conds``).
        **kwargs
            Per-call options forwarded to :meth:`sample`.

        Yields
        ------
        Batch
            One batch per call, exactly as produced, so consumers count
            graphs themselves. Retries/resampling belong to the consuming
            loop, not the stream. Note a filter that empties a batch raises
            ``IndexError`` from :class:`~nvalchemi.data.Batch` (no zero-graph
            selections); rejection-rate-aware streaming needs that data-layer
            change first.
        """
        if conds is None:
            conds = itertools.repeat(None)
        for index, cond in enumerate(conds):
            if max_batches is not None and index >= max_batches:
                return
            yield self.sample(cond, **kwargs)

    def __iter__(self) -> Iterator[Batch]:
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
            A :class:`AtomGenerator`, a dynamics engine, a ``Batch -> Batch``
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
