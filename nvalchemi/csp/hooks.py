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
"""Generation hooks for crystal-structure generation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch
from torch import Tensor

from nvalchemi.csp._validation import (
    finite_nonnegative,
    finite_positive,
    positive_integer,
)
from nvalchemi.csp.comparison import RadialComparisonIndex
from nvalchemi.data import Batch
from nvalchemi.gen.stages import GenerationStage
from nvalchemi.hooks._context import GenerationContext

__all__ = ["DeduplicationEngine", "DeduplicateHook"]


class DeduplicationEngine(Protocol):
    """Structural interface for generation-time Batch deduplication.

    Implementations return a one-dimensional ``torch.bool`` keep mask of
    length ``batch.num_graphs`` on exactly ``batch.device`` and must not mutate
    the supplied Batch in place. Engines may keep their own state across calls;
    the bundled radial engine compares only within each input Batch.
    """

    def deduplicate(self, batch: Batch) -> Tensor:
        """Return a ``torch.bool [batch.num_graphs]`` mask on ``batch.device``."""
        ...


class _RadialDeduplicationEngine:
    """Build and use a fresh typed radial index for one input Batch."""

    def __init__(
        self,
        *,
        cutoff: float,
        threshold: float,
        confirm: Callable[[Tensor], Tensor] | None,
    ) -> None:
        self.cutoff = cutoff
        self.threshold = threshold
        self.confirm = confirm

    def deduplicate(self, batch: Batch) -> Tensor:
        """Return retained input rows using element-sensitive radial screening."""
        index = RadialComparisonIndex.build(
            batch,
            cutoff=self.cutoff,
            atom_types=batch.atomic_numbers,
            typed_neighbors=True,
        )
        result = index.deduplicate(threshold=self.threshold, confirm=self.confirm)
        keep = torch.zeros(batch.num_graphs, dtype=torch.bool, device=batch.device)
        keep[result.retained_indices.to(dtype=torch.long)] = True
        return keep


class DeduplicateHook:
    """Filter generated Batches through a caller-selected deduplication engine.

    The hook runs at ``GenerationStage.AFTER_GENERATE``. Its engine receives
    the Batch entering this hook and returns a boolean keep mask for that
    Batch. The hook replaces ``ctx.batch`` with the selected rows and writes
    the same input-aligned mask to ``ctx.accepted_mask``. A previous reporting
    mask is replaced; ``ctx.sample`` remains the original generated sample. An
    all-false mask produces ``Batch.empty_like(input_batch)``, retaining its
    materialized schema and capacities. Engine, validation, and selection
    errors leave hook-owned context fields unchanged when the engine obeys the
    no-in-place-mutation contract; a mutating engine cannot be rolled back.

    Parameters
    ----------
    engine : DeduplicationEngine
        Object with a callable ``deduplicate(batch)`` method returning a
        boolean mask on the input Batch's device. It must not mutate the input
        Batch in place.
    frequency : int, default=1
        Run the engine on every ``frequency``-th eligible generation call.
        Frequency gating is performed by the hook registry.

    Notes
    -----
    :meth:`radial` constructs a stateless, within-Batch approximate radial
    engine. Without confirmation, its proposed matches can remove distinct
    structures; atomic-number typing and typed neighbors make screening
    element-sensitive but do not establish structural identity.
    """

    stage = GenerationStage.AFTER_GENERATE

    def __init__(self, engine: DeduplicationEngine, *, frequency: int = 1) -> None:
        """Create a hook around a structural deduplication engine.

        Parameters
        ----------
        engine : DeduplicationEngine
            Custom implementation of the boolean-mask engine protocol.
        frequency : int, default=1
            Positive number of generation calls between eligible dispatches.

        Raises
        ------
        TypeError
            If ``engine`` lacks a callable ``deduplicate`` method, or if
            ``frequency`` is a boolean or noninteger.
        ValueError
            If ``frequency`` is not positive.
        """
        if not callable(getattr(engine, "deduplicate", None)):
            raise TypeError("engine must provide a callable deduplicate(batch) method")
        if isinstance(frequency, bool) or not isinstance(frequency, int):
            raise TypeError("frequency must be a positive integer")
        try:
            positive_integer(frequency, name="frequency")
        except ValueError:
            raise ValueError("frequency must be a positive integer") from None
        self.engine = engine
        self.frequency = frequency

    @classmethod
    def radial(
        cls,
        *,
        cutoff: float,
        threshold: float,
        confirm: Callable[[Tensor], Tensor] | None = None,
        frequency: int = 1,
    ) -> DeduplicateHook:
        """Create a within-Batch element-sensitive radial deduplication hook.

        Parameters
        ----------
        cutoff : float
            Positive radial cutoff in angstroms.
        threshold : float
            Finite nonnegative fractional mismatch limit for approximate
            radial matches.
        confirm : callable, optional
            Optional comparison confirmation callback. It receives proposed
            ``int32 [K, 2]`` input-row pairs on the Batch device, with the
            candidate in column zero and its earlier retained representative
            in column one. It returns an ordered subset on that device.
            Without it, radial proposals are accepted as matches.
        frequency : int, default=1
            Run the engine on every ``frequency``-th eligible generation call.

        Returns
        -------
        DeduplicateHook
            Hook using a fresh radial comparison index for each nonempty Batch.

        Raises
        ------
        TypeError
            If ``confirm`` is not callable or ``frequency`` is a boolean or
            noninteger.
        ValueError
            If ``cutoff`` is not finite and positive, ``threshold`` is not
            finite and nonnegative, or ``frequency`` is not positive.
        """
        cutoff_value = finite_positive(cutoff, "cutoff")
        try:
            threshold_value = finite_nonnegative(threshold, "threshold")
        except TypeError:
            raise ValueError("threshold must be finite and nonnegative") from None
        if confirm is not None and not callable(confirm):
            raise TypeError("confirm must be callable or None")
        return cls(
            _RadialDeduplicationEngine(
                cutoff=cutoff_value, threshold=threshold_value, confirm=confirm
            ),
            frequency=frequency,
        )

    def __call__(self, ctx: GenerationContext, stage: GenerationStage) -> None:
        """Validate engine output, then replace the generated Batch locally."""
        if stage is not GenerationStage.AFTER_GENERATE:
            return
        batch = ctx.batch
        if not isinstance(batch, Batch):
            raise TypeError("DeduplicateHook requires ctx.batch to be a Batch")

        if batch.num_graphs == 0:
            keep = torch.empty((0,), dtype=torch.bool, device=batch.device)
            selected = batch
        else:
            keep = self.engine.deduplicate(batch)
            if not isinstance(keep, Tensor):
                raise TypeError("deduplication engine must return a torch.Tensor")
            if keep.dtype is not torch.bool:
                raise TypeError("deduplication engine must return a torch.bool mask")
            if keep.ndim != 1:
                raise ValueError("deduplication engine mask must be one-dimensional")
            if keep.numel() != batch.num_graphs:
                raise ValueError(
                    "deduplication engine mask length must match batch.num_graphs"
                )
            if keep.device != batch.device:
                raise ValueError(
                    "deduplication engine mask must be on the input Batch device"
                )
            if torch.any(keep):
                selected = batch[keep]
            else:
                selected = Batch.empty_like(batch)

        ctx.batch = selected
        ctx.accepted_mask = keep
