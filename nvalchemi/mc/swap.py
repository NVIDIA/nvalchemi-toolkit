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
"""Canonical (fixed-composition) Monte Carlo with non-local unlike-species swaps."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.data import Batch
from nvalchemi.mc.base import _draw_from_mask, _PairSwapMonteCarlo

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["NonlocalSwap"]


class NonlocalSwap(_PairSwapMonteCarlo):
    r"""Batched canonical MC that swaps the species of two unlike atoms anywhere in a graph.

    Composition is fixed. A proposal draws one atom :math:`i` uniformly from
    the graph's :math:`N` atoms, then one atom :math:`j` uniformly from the
    atoms of any *other* species, and exchanges their identities. Unlike
    :class:`~nvalchemi.mc.Kawasaki` there is no neighbour graph: the pair can
    be any distance apart, so one accepted move transports a solute across the
    whole system (e.g. from a nanoparticle's core to its surface) instead of
    diffusing it one bond at a time.

    For the unordered pair :math:`\{i, j\}` with species :math:`a \ne b`,

    .. math::

        q(\{i, j\}) = \frac{1}{N}\left[\frac{1}{N - N_a} +
                      \frac{1}{N - N_b}\right],

    which depends only on the species counts, and a swap leaves those counts
    unchanged. The proposal is therefore symmetric, every proposal changes the
    state, and plain Metropolis acceptance is detailed-balance exact with no
    Hastings correction. A single-species graph has no legal move and is
    dropped from the step's proposals and statistics.

    The draw needs no geometry, so nothing is cached between steps and no
    :meth:`synchronize` after an MD block is required beyond the energy
    re-baseline every sampler needs.
    """

    def __init__(
        self,
        model: BaseModelMixin,
        temperature: float | torch.Tensor,
        **kwargs: Any,
    ) -> None:
        """Initialize a non-local canonical swap sampler.

        Parameters
        ----------
        model
            Model that returns one energy per graph.
        temperature
            Positive scalar temperature in K, or one temperature per graph.
        **kwargs
            Forwarded to :class:`~nvalchemi.mc.base.BaseMonteCarlo`.
        """
        super().__init__(model=model, temperature=temperature, **kwargs)

    def _propose(
        self,
        batch: Batch,
        generator: torch.Generator,
        active: torch.Tensor,
    ) -> None:
        """Swap one random unlike-species atom pair per active graph.

        Device-only: no op here has a data-dependent output size, so no
        device-to-host sync stalls the next model call.
        """
        numbers = batch.atomic_numbers
        ptr = batch.batch_ptr.to(torch.long)
        starts = ptr[:-1]
        draws = torch.rand(
            2, batch.num_graphs, device=batch.device, generator=generator
        )
        # First atom: uniform over the graph's atoms.
        first = starts + torch.floor(draws[0] * (ptr[1:] - starts)).to(torch.long)
        first = torch.minimum(first, ptr[1:] - 1)
        # Second atom: uniform over the graph's atoms of another species.
        graph_of_atom = batch.batch_idx.to(torch.long)
        other = (numbers != numbers[first][graph_of_atom]).to(torch.long)
        counts = torch.zeros(batch.num_graphs, dtype=torch.long, device=batch.device)
        counts.index_add_(0, graph_of_atom, other)
        # A single-species graph has no legal move: drop it from this step
        # (``active`` is the sampler's own mask tensor, narrowed in place).
        active &= counts > 0
        second = _draw_from_mask(other, starts, counts, draws[1])
        # Inactive graphs draw no pair; keep their indices in range anyway.
        first = torch.where(active, first, starts)
        self._apply_swap(batch, first, second, active)
