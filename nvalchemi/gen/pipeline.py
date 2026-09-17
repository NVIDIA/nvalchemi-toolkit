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
"""Sequential composition of generators and other batch-processing stages.

A :class:`GenerationPipeline` is a thin orchestrator: it folds a
conditioning input through an ordered list of stages — generators, dynamics
engines, or any ``Batch -> Batch`` callable — mirroring the dynamics
``|`` sugar (:meth:`nvalchemi.dynamics.base.BaseDynamics.__or__` builds a
``DistributedPipeline``; here ``AtomisticGenerator.__or__`` builds a
``GenerationPipeline``).

Example
-------
::

    pipe = gen_a | gen_b | optimizer
    out = pipe(inputs)                    # one fold through the stages
    for batch in pipe.stream(inputs):     # lazy per-item fold
        ...

Semantics:

* **Stage 1 consumes the user's ``inputs``** (its generating function owns
  conditioning); every later stage maps Batch → Batch.
* **1→1 cardinality** per stage: filters may shrink a batch; nothing fans
  out. (A filter may not shrink a batch to *empty* today —
  :class:`~nvalchemi.data.Batch` raises ``IndexError`` on zero-graph
  selections; empty-batch support is a separate data-layer decision.)
* **Empty batches short-circuit** (defensive contract): should a stage ever
  yield a zero-graph batch, remaining stages are skipped for that item and
  the empty batch is returned as-is. No shipped path currently produces one.
* **Mapping-less generators are terminal-only**: an AtomisticGenerator without a
  ``batch_mapping`` yields its raw sample (not necessarily a
  :class:`~nvalchemi.data.Batch`), so it can only be the last stage — every
  upstream stage must produce a ``Batch``.
* **Per-stage hooks**: each :class:`~nvalchemi.gen.generator.AtomisticGenerator`
  stage keeps its own hooks and
  :class:`~nvalchemi.hooks.GenerationContext`; the pipeline passes only
  the batch between stages.
* **Sessions and compile**: ``with pipe:`` creates one dedicated CUDA
  stream (when the first AtomisticGenerator stage's resolved device is CUDA)
  shared by every AtomisticGenerator stage that opts in — sequential stages
  serialize on it with no cross-stream sync. :meth:`compile` compiles each
  AtomisticGenerator stage's generating function (non-AtomisticGenerator stages are
  skipped); there is no whole-fold compile, since the Batch
  plumbing between stages would graph-break for no real capture.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator
from typing import Any

import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from nvalchemi.data import Batch
from nvalchemi.gen.generator import AtomisticGenerator

__all__ = ["GenerationPipeline"]


class GenerationPipeline(BaseModel):
    """Sequential composition of generation and batch-processing stages.

    Attributes
    ----------
    stages
        Ordered pipeline stages: :class:`~nvalchemi.gen.generator.AtomisticGenerator`
        instances, dynamics engines, or ``Batch -> Batch`` callables.

    Notes
    -----
    **Field-contract validation.** Every ``AtomisticGenerator`` stage must declare
    ``consumes_fields`` / ``produces_fields`` (set on the AtomisticGenerator directly
    or defaulted from the generating function's attributes); construction
    raises otherwise. For each adjacent AtomisticGenerator → AtomisticGenerator link,
    the downstream stage's ``consumes_fields`` must be covered by the upstream
    stage's
    (AIMNet2 ``charges`` → Ewald) applied to generation. Authors of custom
    ``batch_mapping`` callables own keeping their stage's declaration in sync
    with what the callable actually writes. Non-AtomisticGenerator stages carry no
    declarations and are not validated (their outputs are unknown at
    construction).

    The first stage's ``consumes_fields`` describe its *conditioning* input
    and are not validated (the pipeline cannot know what a user's ``inputs``
    carries).

    **Sessions and compile.** ``GenerationPipeline`` is a context manager:
    entry creates one dedicated CUDA stream (when the first
    :class:`~nvalchemi.gen.generator.AtomisticGenerator` stage's resolved device
    is CUDA) and shares it with every AtomisticGenerator stage that has
    ``dedicated_stream`` set, then enters each AtomisticGenerator stage's own
    session (session RNG, lazy compile, context-manager hooks).
    Non-AtomisticGenerator stages manage their own contexts.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stages: list[Any] = Field(
        min_length=1,
        description=(
            "Ordered stages: Generators, dynamics engines, or Batch -> Batch callables."
        ),
    )

    @model_validator(mode="after")
    def _validate_links(self) -> GenerationPipeline:
        """Validate declarations and adjacent AtomisticGenerator→AtomisticGenerator links.

        Returns
        -------
        GenerationPipeline
            The validated pipeline.

        Raises
        ------
        ValueError
            If a AtomisticGenerator stage lacks field declarations, or a stage's
            ``consumes_fields`` are not covered by the immediately
            upstream AtomisticGenerator's ``produces_fields``.
        """
        for index, stage in enumerate(self.stages):
            if not isinstance(stage, AtomisticGenerator):
                continue
            consumes = stage.consumes_fields
            produces = stage.produces_fields
            if consumes is None or produces is None:
                raise ValueError(
                    f"Pipeline stage {index} ({type(stage.generator_func).__name__} "
                    "generator) declares neither consumes_fields nor "
                    "produces_fields: set them on the AtomisticGenerator or on the "
                    "generating function."
                )
            prev = self.stages[index - 1] if index > 0 else None
            if isinstance(prev, AtomisticGenerator):
                # Validated non-None on the previous iteration.
                produced = prev.produces_fields or frozenset()
                missing = set(consumes) - set(produced)
                if missing:
                    raise ValueError(
                        f"Pipeline stage {index} consumes fields "
                        f"{sorted(missing)} that the upstream stage does not "
                        "produce (produces_fields="
                        f"{sorted(produced)}). Fix the "
                        "declarations or insert a stage that writes them."
                    )
        return self

    def model_post_init(self, __context: Any) -> None:
        """Initialize session state."""
        self._stream: torch.cuda.Stream | None = None
        self._stream_ctx: Any = None

    def compile(self, **kwargs: Any) -> GenerationPipeline:
        """Compile every AtomisticGenerator stage's generating function.

        Per-stage compilation (see :meth:`AtomisticGenerator.compile`); non-AtomisticGenerator
        stages are skipped. There is no whole-fold compile: the
        Batch plumbing and hook dispatch between stages would graph-break for
        no real capture. (Cross-stage tensor fusion is a separate research
        item.)

        Parameters
        ----------
        **kwargs
            Forwarded to each stage's :meth:`AtomisticGenerator.compile`.

        Returns
        -------
        GenerationPipeline
            This instance, for fluent chaining.
        """
        for stage in self.stages:
            if isinstance(stage, AtomisticGenerator):
                stage.compile(**kwargs)
        return self

    def _infer_device(self) -> torch.device | None:
        """Infer the session device from the first AtomisticGenerator stage.

        Resolves the stage's device chain (``device`` field, then the
        generating function's ``device`` attribute).

        Returns
        -------
        torch.device | None
            The device, or ``None`` when no AtomisticGenerator stage can provide one.
        """
        for stage in self.stages:
            if isinstance(stage, AtomisticGenerator):
                return stage._infer_device()
        return None

    def __enter__(self) -> GenerationPipeline:
        """Enter a pipeline session: one CUDA stream shared across stages.

        Creates one dedicated CUDA stream (when the first AtomisticGenerator
        stage's resolved device is CUDA), points every AtomisticGenerator stage
        with ``dedicated_stream`` set at it, and enters each AtomisticGenerator
        stage's own session (session RNG, lazy compile, context-manager
        hooks — stream creation is skipped because ``stage._stream`` is
        already set). Non-AtomisticGenerator stages manage their own contexts.

        Returns
        -------
        GenerationPipeline
            This instance.
        """
        device = self._infer_device()
        if device is not None and device.type == "cuda":
            self._stream = torch.cuda.Stream(device=device)
            self._stream_ctx = torch.cuda.stream(self._stream)
            self._stream_ctx.__enter__()
        for stage in self.stages:
            if isinstance(stage, AtomisticGenerator):
                if stage.dedicated_stream:
                    stage._stream = self._stream
                stage.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the session: exit each AtomisticGenerator stage, then the stream.

        Parameters
        ----------
        exc_type, exc_val, exc_tb
            The active exception, if any.
        """
        for stage in self.stages:
            if isinstance(stage, AtomisticGenerator):
                stage.__exit__(exc_type, exc_val, exc_tb)
        if self._stream_ctx is not None:
            self._stream_ctx.__exit__(exc_type, exc_val, exc_tb)
        self._stream = None
        self._stream_ctx = None

    def __call__(self, inputs: Any = None, **kwargs: Any) -> Any:
        """Fold ``inputs`` through the stages.

        Parameters
        ----------
        inputs
            Input for the first stage (a
            :class:`~nvalchemi.data.Batch`, another tensor container, or
            ``None``).
        **kwargs
            Per-call options forwarded to every stage (e.g. generator
            function options).

        Returns
        -------
        Any
            The final stage's output — a :class:`~nvalchemi.data.Batch`,
            unless the terminal stage is a mapping-less generator (raw
            sample). Should a stage ever yield a zero-graph batch, remaining
            stages are skipped and it is returned as-is (defensive; no
            current :class:`~nvalchemi.data.Batch` path produces one).
        """
        result: Any = inputs
        for stage in self.stages:
            if isinstance(result, Batch) and result.num_graphs == 0:
                break
            result = stage(result, **kwargs)
        return result

    def stream(
        self,
        inputs: Any = None,
        *,
        max_batches: int | None = None,
        **kwargs: Any,
    ) -> Iterator[Any]:
        """Stream pipeline outputs, mirroring :meth:`AtomisticGenerator.stream`.

        One fold per input item; ``inputs`` is the data source.

        Parameters
        ----------
        inputs
            Iterable of inputs, or ``None`` for repeated unconditional draws.
        max_batches
            Cap on batches yielded (``None`` means unbounded).
        **kwargs
            Per-call options forwarded to :meth:`__call__`.

        Yields
        ------
        Any
            One output per fold, exactly as produced.
        """
        if inputs is None:
            inputs = itertools.repeat(None)
        for index, item in enumerate(inputs):
            if max_batches is not None and index >= max_batches:
                return
            yield self(item, **kwargs)

    def __or__(self, other: Any) -> GenerationPipeline:
        """Append a stage, returning a new pipeline.

        Parameters
        ----------
        other
            A stage to append (AtomisticGenerator, dynamics engine, or callable).

        Returns
        -------
        GenerationPipeline
            A pipeline of ``self.stages`` followed by ``other``.
        """
        return GenerationPipeline(stages=[*self.stages, other])
