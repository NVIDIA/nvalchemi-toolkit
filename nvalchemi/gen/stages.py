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
    """Stages of the :class:`~nvalchemi.gen.generator.AtomGenerator` pipeline.

    One stage per distinct point of the fixed pipeline (condition →
    generate; the raw sample is materialized into a
    :class:`~nvalchemi.data.Batch` inline, as part of the generate step).
    Before/after pairs collapse into single stages because hooks mutate the
    :class:`~nvalchemi.hooks.GenerationContext` by replacing its fields,
    and the next step re-reads the context — so a "before generate" hook and
    an "after condition" hook are the same point.

    Attributes
    ----------
    BEFORE_CONDITION
        Fired before the conditioning batch is built; ``ctx.batch`` is
        ``None``. Edit or replace ``ctx.cond`` here. (Text encodings are
        expected to happen before the ``AtomGenerator`` is entered, so ``cond``
        already holds tensor data.)
    AFTER_CONDITION
        Fired after conditioning; ``ctx.batch`` holds the conditioning
        batch, tiled by ``num_samples_per_batch``. Attach conditioning
        metadata (e.g. text embeddings for classifier-free guidance) or
        replace the conditioning batch here.
    AFTER_GENERATE
        Fired after the raw sample has been materialized into the generated
        :class:`~nvalchemi.data.Batch`; ``ctx.batch`` holds it. Filter or
        mutate the generated batch here — filtering is graph-level
        subsetting (``ctx.batch = ctx.batch[keep]``).
        :class:`~nvalchemi.data.Batch` does not support zero-graph
        selections (``IndexError``), so a filter must keep at least one
        graph; empty-batch semantics are a separate data-layer decision.
    """

    BEFORE_CONDITION = auto()
    AFTER_CONDITION = auto()
    AFTER_GENERATE = auto()
