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
"""Lifecycle stages of the generative pipeline at which hooks can fire."""

from __future__ import annotations

from enum import Enum, auto

__all__ = ["GenerationStage"]


class GenerationStage(Enum):
    """Stages of the :class:`~nvalchemi.gen.generator.AtomisticGenerator` pipeline.

    One stage per distinct point of the fixed pipeline (optionally condition
    the inputs, generate, then materialize the raw sample into a
    :class:`~nvalchemi.data.Batch` — the raw sample is exposed to hooks
    between generation and materialization). Hooks mutate the
    :class:`~nvalchemi.hooks.GenerationContext` by replacing its fields, and
    the driver re-reads the context after each dispatch. Hooks at one stage
    run in list order, and a raising hook aborts the call (nothing catches
    it).

    Firing policy: ``BEFORE_CONDITION``/``AFTER_CONDITION`` fire only when a
    condition step was provided for the call (the driver's ``condition_func``
    or the generating function's ``condition`` attribute); ``BEFORE_MAPPING``
    always fires, after generation with ``ctx.sample`` set; ``AFTER_GENERATE``
    fires only when the sample is a :class:`~nvalchemi.data.Batch`.

    Attributes
    ----------
    BEFORE_CONDITION
        Fired before the condition step runs; ``ctx.inputs`` holds the call's
        raw input (and ``ctx.batch`` holds it too when it is a
        :class:`~nvalchemi.data.Batch`, ``None`` otherwise). Edit or replace
        ``ctx.inputs`` here — the condition callable receives whatever
        ``ctx.inputs`` holds after this dispatch. Fires only when a condition
        step was provided for the call.
    AFTER_CONDITION
        Fired after the condition step; ``ctx.inputs`` holds the conditioned
        value the generating function is about to be called with (e.g. a
        conditioning :class:`~nvalchemi.data.Batch` tiled by ``num_samples``).
        Attach conditioning metadata (e.g. text embeddings for classifier-free
        guidance) or replace the conditioned input here. Fires only when a
        condition step was provided for the call.
    BEFORE_MAPPING
        Fired after the generating function returns; ``ctx.sample`` holds
        whatever the function produced (a :class:`~nvalchemi.data.Batch` on
        the contract path), and ``ctx.batch`` holds the call's inputs when they
        were a :class:`~nvalchemi.data.Batch` (``None`` otherwise). Filter or
        replace ``ctx.sample`` here — workflows with compact internal
        representations can drop rejected candidates before any downstream
        ``Batch`` work. The driver re-reads ``ctx.sample`` after dispatch and
        returns it: through the ``Batch`` path when it is a ``Batch``, as-is
        otherwise. This stage fires either way.
    AFTER_GENERATE
        Fired after generation when the sample is a
        :class:`~nvalchemi.data.Batch`; ``ctx.batch`` holds it. A function
        returning a non-``Batch`` container skips this stage. Filter or
        mutate the generated batch here — filtering is graph-level subsetting
        (``ctx.batch = ctx.batch[keep]``). The batch may already be zero-graph
        (a function may signal total rejection via
        :meth:`~nvalchemi.data.Batch.empty`), so filters should tolerate
        ``num_graphs == 0``. Zero-graph *selections* still raise
        ``IndexError`` — a hook signalling total rejection replaces
        ``ctx.batch`` with an explicitly built empty batch rather than
        subsetting to nothing.
    """

    BEFORE_CONDITION = auto()
    AFTER_CONDITION = auto()
    BEFORE_MAPPING = auto()
    AFTER_GENERATE = auto()
