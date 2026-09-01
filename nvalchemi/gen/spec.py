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
"""Spec-based construction for generators and pipelines.

Mirrors the training spec machinery
(:class:`~nvalchemi.training._spec.BaseSpec`,
:func:`~nvalchemi.training._spec.create_model_spec` — dotted import paths +
JSON-safe kwargs, no pickle) so generation workflows can be driven from
config files and, eventually, a CLI.

This module is deliberately **not** re-exported from
:mod:`nvalchemi.gen` — importing it pulls in the training spec machinery, so
it stays an opt-in concrete-module import (``from nvalchemi.gen.spec import
AtomGeneratorSpec``).

The unit of spec-ability is the **importable factory**. Family config (latent
dimensions, step counts, sampler paths) is bound by a module-level factory
function, which is then specced with
:func:`~nvalchemi.training._spec.create_model_spec` — never a raw
:class:`functools.partial`, which is not importable::

    # my_project/generation.py  (dotted-path importable)
    def make_gan_generate(latent_dim: int = 128):
        def _generate(model, *, num_samples=1, rng=None, cond=None, **kwargs):
            z = torch.randn(num_samples, latent_dim, generator=rng)
            return TensorDict({"x1": model.decode(z)}, batch_size=[num_samples])

        return _generate

    # config layer
    func_spec = create_model_spec(make_gan_generate, latent_dim=256)
    spec = AtomGeneratorSpec(generator_func=func_spec, num_samples_per_batch=4)
    blob = spec.model_dump_json()                 # plain JSON; safe to store
    spec2 = AtomGeneratorSpec.model_validate_json(blob)
    gen = spec2.build(model=model)                # model comes from checkpoints

Weights are never part of a spec — the model is supplied at :meth:`build`
time from the checkpoint machinery (``torch.load(..., weights_only=True)``).

The reverse direction — capturing a live generator or pipeline — is
:meth:`~nvalchemi.gen.generator.AtomGenerator.to_spec` /
:meth:`~nvalchemi.gen.pipeline.GenerationPipeline.to_spec`: callables are
captured by dotted import path, hooks by attribute-faithful class
construction (each ``__init__`` parameter read back from the same-named
attribute).
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializeAsAny,
    field_validator,
)

from nvalchemi._serialization import _import_callable
from nvalchemi.gen.generator import AtomGenerator
from nvalchemi.gen.pipeline import GenerationPipeline
from nvalchemi.training._spec import (
    BaseSpec,
    create_model_spec,
    create_model_spec_from_json,
)

__all__ = ["GenerationPipelineSpec", "AtomGeneratorSpec"]


def _rehydrate_spec(value: Any) -> Any:
    """Rehydrate a spec-dict to its dynamic :class:`BaseSpec` subclass."""
    if isinstance(value, dict):
        return create_model_spec_from_json(value)
    return value


def _return_importable(path: str) -> Callable[..., Any]:
    """Identity factory: return the callable at ``path``.

    Exists so a bare module-level callable (not a factory) can live in a
    spec: ``create_model_spec(_return_importable, path=...)`` builds back to
    the callable itself. Referenced by dotted path inside stored specs — do
    not rename or move.

    Parameters
    ----------
    path
        Dotted import path of the callable to return.

    Returns
    -------
    Callable
        The imported callable.
    """
    return _import_callable(path)


def _spec_from_callable(fn: Callable[..., Any], *, name: str) -> BaseSpec:
    """Capture a live callable as a spec via its dotted import path.

    The callable must be module-level and importable; lambdas, closures,
    ``functools.partial`` objects, and callable instances carry no import
    path and raise ``TypeError``. The spec builds back to the callable
    itself (through the :func:`_return_importable` identity factory), so it
    round-trips whether the callable was factory-made or not.

    Parameters
    ----------
    fn
        The live callable to capture.
    name
        The field/role name used in error messages.

    Returns
    -------
    BaseSpec
        A spec whose ``build()`` returns ``fn`` (re-imported).

    Raises
    ------
    TypeError
        If ``fn`` has no importable dotted path.
    """
    if isinstance(fn, functools.partial):
        raise TypeError(
            f"{name} cannot be captured in a spec: functools.partial objects "
            "have no import path. Bind arguments in a module-level factory "
            "function instead."
        )
    module = getattr(fn, "__module__", None)
    qualname = getattr(fn, "__qualname__", None)
    if not module or not qualname or "<locals>" in qualname or qualname == "<lambda>":
        raise TypeError(
            f"{name} cannot be captured in a spec: {fn!r} is not a "
            "module-level importable callable (lambdas, closures, and "
            "callable instances have no dotted path). Define it at module "
            "level, or bind config in a module-level factory."
        )
    return create_model_spec(_return_importable, path=f"{module}.{qualname}")


def _spec_from_hook(hook: Any) -> BaseSpec:
    """Capture a live hook as a spec of its class plus ``__init__`` kwargs.

    Each ``__init__`` parameter is read back from the same-named attribute
    on the instance — the convention spec-able hooks already follow
    (``self.x = x``). Parameters that are positional-only, variadic, or not
    stored as a same-named attribute raise ``TypeError``. Attribute values
    must be JSON-safe (enums serialize by value).

    Parameters
    ----------
    hook
        The live hook instance.

    Returns
    -------
    BaseSpec
        A spec whose ``build()`` reconstructs the hook.

    Raises
    ------
    TypeError
        If the hook's constructor is not attribute-faithful.
    """
    cls = type(hook)
    if cls.__init__ is object.__init__:
        return create_model_spec(cls)
    sig = inspect.signature(cls.__init__)
    kwargs: dict[str, Any] = {}
    for pname, param in sig.parameters.items():
        if pname == "self":
            continue
        if param.kind not in (
            inspect.Parameter.KEYWORD_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError(
                f"{cls.__qualname__} is not spec-able: __init__ parameter "
                f"{pname!r} is positional-only or variadic; spec-able hooks "
                "use keyword-able parameters only."
            )
        if not hasattr(hook, pname):
            raise TypeError(
                f"{cls.__qualname__} is not spec-able: __init__ parameter "
                f"{pname!r} is not stored as a same-named attribute on the "
                "instance (the self.x = x convention)."
            )
        kwargs[pname] = getattr(hook, pname)
    return create_model_spec(cls, **kwargs)


class AtomGeneratorSpec(BaseModel):
    """JSON-serializable construction spec for a :class:`AtomGenerator`.

    Attributes
    ----------
    generator_func
        Spec for a dotted-path factory returning the
        :class:`~nvalchemi.gen.generator.GeneratingFunction` (e.g. from
        :func:`~nvalchemi.training._spec.create_model_spec`). ``None`` when
        the model's own ``generate`` is the generation source.
    output_to_batch_func
        Spec for a dotted-path materialization callable/factory. ``None``
        when the model's own ``to_batch`` is the materialization target.
    hooks
        Specs for the generation hooks (hook classes must have
        keyword-only, JSON-safe constructors — the
        :func:`~nvalchemi.training._spec.create_model_spec` constraint).
    consumes_fields, produces_fields
        Field declarations forwarded to the :class:`AtomGenerator` (``None``
        leaves the AtomGenerator-side defaulting from ``model.model_config``
        in place).
    num_samples_per_batch
        Draws per conditioning entry.
    seed
        Base seed for per-draw RNGs.

    Notes
    -----
    ``revalidate_instances="never"`` matches
    :class:`~nvalchemi.training._spec.BaseSpec`: specs are immutable records
    of construction config.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, revalidate_instances="never"
    )

    spec_kind: Literal["generator"] = "generator"
    generator_func: SerializeAsAny[BaseSpec] | None = None
    output_to_batch_func: SerializeAsAny[BaseSpec] | None = None
    hooks: list[SerializeAsAny[BaseSpec]] = Field(default_factory=list)
    consumes_fields: list[str] | None = None
    produces_fields: list[str] | None = None
    num_samples_per_batch: int = Field(default=1, ge=1)
    seed: int | None = None
    compile_generate: bool = False
    compile_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("generator_func", "output_to_batch_func", mode="before")
    @classmethod
    def _rehydrate_callable(cls, value: Any) -> Any:
        """Rehydrate JSON spec-dicts for the callable fields."""
        return _rehydrate_spec(value)

    @field_validator("hooks", mode="before")
    @classmethod
    def _rehydrate_hooks(cls, value: Any) -> Any:
        """Rehydrate JSON spec-dicts for the hook list."""
        return [_rehydrate_spec(item) for item in value]

    def build(self, *, model: Any = None, **overrides: Any) -> AtomGenerator:
        """Build the :class:`AtomGenerator`, injecting the runtime model.

        Parameters
        ----------
        model
            The generative model (weights from the checkpoint machinery).
            Required — specs cover construction, not parameters.
        **overrides
            Extra keyword arguments forwarded to the
            :class:`~nvalchemi.gen.generator.AtomGenerator` constructor,
            overriding spec-stored values of the same name.

        Returns
        -------
        AtomGenerator
            The constructed generator.

        Raises
        ------
        TypeError
            If no model is supplied.
        """
        if model is None and "model" not in overrides:
            raise TypeError(
                "AtomGeneratorSpec.build requires a model; weights come from the "
                "checkpoint machinery, not the spec."
            )
        kwargs: dict[str, Any] = {
            "generator_func": (
                self.generator_func.build() if self.generator_func is not None else None
            ),
            "output_to_batch_func": (
                self.output_to_batch_func.build()
                if self.output_to_batch_func is not None
                else None
            ),
            "hooks": [hook.build() for hook in self.hooks],
            "consumes_fields": (
                frozenset(self.consumes_fields)
                if self.consumes_fields is not None
                else None
            ),
            "produces_fields": (
                frozenset(self.produces_fields)
                if self.produces_fields is not None
                else None
            ),
            "num_samples_per_batch": self.num_samples_per_batch,
            "seed": self.seed,
            "compile_generate": self.compile_generate,
            "compile_kwargs": dict(self.compile_kwargs),
        }
        kwargs.update(overrides)
        return AtomGenerator(model=model, **kwargs)


class GenerationPipelineSpec(BaseModel):
    """JSON-serializable construction spec for a :class:`GenerationPipeline`.

    Attributes
    ----------
    stages
        Ordered stage specs: :class:`AtomGeneratorSpec` entries and/or
        :class:`~nvalchemi.training._spec.BaseSpec` entries that build
        ``Batch -> Batch`` stages (dynamics engines, callables).

    Notes
    -----
    JSON rehydration distinguishes entries by shape: dicts with a
    ``cls_path`` key are callable specs (rebuilt via
    :func:`~nvalchemi.training._spec.create_model_spec_from_json`); any
    other dict is a :class:`AtomGeneratorSpec`.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True, revalidate_instances="never"
    )

    spec_kind: Literal["generation_pipeline"] = "generation_pipeline"
    stages: list[SerializeAsAny[BaseSpec] | AtomGeneratorSpec] = Field(min_length=1)

    @field_validator("stages", mode="before")
    @classmethod
    def _rehydrate_stages(cls, value: Any) -> Any:
        """Rehydrate JSON dicts to stage specs (see class Notes)."""
        out: list[Any] = []
        for item in value:
            if isinstance(item, dict):
                if "cls_path" in item:
                    out.append(create_model_spec_from_json(item))
                else:
                    out.append(AtomGeneratorSpec.model_validate(item))
            else:
                out.append(item)
        return out

    def build(self, *, models: Any = None) -> GenerationPipeline:
        """Build the pipeline, injecting runtime models for generator stages.

        Parameters
        ----------
        models
            One model per :class:`AtomGeneratorSpec` stage, in stage order.

        Returns
        -------
        GenerationPipeline
            The constructed pipeline.

        Raises
        ------
        TypeError
            If fewer models than generator stages are supplied.
        """
        model_iter = iter(models or [])
        stages: list[Any] = []
        for spec in self.stages:
            if isinstance(spec, AtomGeneratorSpec):
                try:
                    model = next(model_iter)
                except StopIteration:
                    raise TypeError(
                        "GenerationPipelineSpec.build needs one model per "
                        "AtomGeneratorSpec stage, in stage order."
                    ) from None
                stages.append(spec.build(model=model))
            else:
                stages.append(spec.build())
        return GenerationPipeline(stages=stages)
