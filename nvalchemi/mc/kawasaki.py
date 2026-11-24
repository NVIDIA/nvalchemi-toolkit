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
"""Canonical (fixed-composition) nearest-neighbour Kawasaki Monte Carlo."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

import torch
from nvalchemiops.neighbors.neighbor_utils import estimate_max_neighbors
from nvalchemiops.torch.neighbors import neighbor_list
from nvalchemiops.torch.neighbors.neighbor_utils import (
    get_neighbor_list_from_neighbor_matrix,
)

from nvalchemi.data import Batch
from nvalchemi.dynamics.hooks._utils import KB_EV
from nvalchemi.mc.base import _draw_from_mask, _is_batch, _PairSwapMonteCarlo

if TYPE_CHECKING:
    from nvalchemi.models.base import BaseModelMixin

__all__ = ["Kawasaki"]


def _nearest_neighbor_edges(
    batch: Batch, cutoff: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a deduplicated, undirected nearest-neighbour pair list.

    Uses the same GPU cutoff neighbor-search kernel as
    :func:`nvalchemi.neighbors.compute_neighbors`, kept off the batch so it
    never collides with the energy model's own (typically longer-range)
    interaction neighbor list. The search is a plain radius cutoff over the
    current positions with no lattice-site assumption, so it is robust to
    thermally disordered geometry from a finite-temperature MD block.

    Parameters
    ----------
    batch : Batch
        Batch whose current positions define the proposal geometry.
    cutoff : float
        Nearest-neighbour proposal cutoff radius, in Angstrom.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(edges, counts)`` where ``edges`` has shape ``(E, 2)`` int64 (each
        undirected pair listed once) and ``counts`` has shape
        ``(num_graphs,)`` int64 with the per-graph edge count.
    """
    N = batch.num_nodes
    device = batch.device
    pbc = getattr(batch, "pbc", None)
    cell = getattr(batch, "cell", None)
    if pbc is not None and not bool(pbc.any()):
        pbc = None
        cell = None

    max_neighbors = estimate_max_neighbors(cutoff=cutoff)
    if pbc is None and batch.max_num_nodes > 0:
        cap = ((batch.max_num_nodes + 15) // 16) * 16
        max_neighbors = min(max_neighbors, cap)

    batch_ptr = batch.batch_ptr.to(torch.int32)
    batch_idx = batch.batch_idx.to(torch.int32)

    while True:
        nb_matrix = torch.full((N, max_neighbors), N, dtype=torch.int32, device=device)
        nb_counts = torch.zeros(N, dtype=torch.int32, device=device)
        neighbor_list(
            positions=batch.positions,
            cutoff=cutoff,
            cell=cell,
            pbc=pbc,
            max_neighbors=max_neighbors,
            half_fill=True,
            batch_ptr=batch_ptr,
            batch_idx=batch_idx,
            neighbor_matrix=nb_matrix,
            num_neighbors=nb_counts,
            neighbor_matrix_shifts=None,
            rebuild_flags=None,
        )
        actual_max = int(nb_counts.max())
        if actual_max < max_neighbors:
            break
        max_neighbors = int(actual_max * 1.5) + 1

    neighbor_list_coo = get_neighbor_list_from_neighbor_matrix(
        neighbor_matrix=nb_matrix,
        num_neighbors=nb_counts,
        neighbor_shift_matrix=None,
        fill_value=N,
    )
    edges = neighbor_list_coo[0].T.contiguous().to(torch.long)  # (E, 2)
    graph_per_edge = batch.batch_idx[edges[:, 0]]
    counts = torch.bincount(graph_per_edge, minlength=batch.num_graphs).to(torch.long)
    return edges, counts


class Kawasaki(_PairSwapMonteCarlo):
    r"""Batched canonical MC with nearest-neighbour site-swap moves.

    Composition is fixed. A proposal draws one nearest-neighbour edge per
    active graph from a fixed-geometry proposal graph and swaps the pair's
    identities.

    With ``unlike_pairs_only=True`` (the default) the draw is uniform over
    only that graph's *unlike-species* edges, so every proposal changes the
    state and every model evaluation does useful work. That proposal
    distribution depends on the configuration -- :math:`q(x \to x') =
    1 / n(x)` for :math:`n(x)` unlike pairs -- so acceptance carries the
    Metropolis-Hastings ratio

    .. math::

        A(x \to x') = \min\left[1, \frac{n(x)}{n(x')}
                        e^{-\Delta E / k_B T}\right],

    applied through :meth:`_chemical_delta` as the equivalent energy term
    :math:`k_B T \ln[n(x') / n(x)]`. The swapped edge is still unlike after
    the swap, so the reverse move always exists and :math:`n(x') > 0`. A
    graph whose neighbour pairs are all same-species has no legal move and is
    dropped from the step's proposals and statistics.

    With ``unlike_pairs_only=False`` the draw is uniform over all edges
    regardless of species and a same-species draw leaves the state unchanged
    (still evaluating the model, which is the waste the default avoids). The
    edge is then chosen independently of species, so every ordered
    configuration has exactly one reverse proposal of equal probability and
    plain Metropolis acceptance is detailed-balance exact with no correction.

    The proposal graph is a short-range nearest-neighbour list, independent
    of any interaction cutoff the energy model uses for its own neighbor
    list and never written to the batch. It is built once per accepted
    geometry and cached; call :meth:`synchronize` after positions change
    outside this sampler (e.g. an interleaved MD block) to rebuild it.
    """

    def __init__(
        self,
        model: BaseModelMixin,
        temperature: float | torch.Tensor,
        cutoff: float,
        unlike_pairs_only: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize a canonical Kawasaki sampler.

        Parameters
        ----------
        model
            Model that returns one energy per graph.
        temperature
            Positive scalar temperature in K, or one temperature per graph.
        cutoff
            Nearest-neighbour proposal cutoff radius in Angstrom, e.g. the
            system's first RDF minimum. Independent of the model's own
            interaction cutoff.
        unlike_pairs_only
            Draw proposals only from unlike-species pairs (default), so no
            model evaluation is spent on a same-species no-op. ``False``
            restores the older species-blind draw -- kept for reproducing
            runs recorded before this became the default.
        **kwargs
            Forwarded to :class:`~nvalchemi.mc.base.BaseMonteCarlo`.
        """
        super().__init__(model=model, temperature=temperature, **kwargs)
        if cutoff <= 0.0:
            raise ValueError("Kawasaki proposal cutoff must be positive")
        self.cutoff = cutoff
        self.unlike_pairs_only = unlike_pairs_only
        self._unlike_before: torch.Tensor | None = None
        self._edges: torch.Tensor | None = None
        self._edge_graph: torch.Tensor | None = None
        self._edge_offsets: torch.Tensor | None = None
        self._graph_batch: weakref.ref[Batch] | None = None

    def _build_proposal_graph(self, batch: Batch) -> None:
        """(Re)build the fixed nearest-neighbour proposal graph in-place."""
        edges, counts = _nearest_neighbor_edges(batch, self.cutoff)
        if bool((counts == 0).any()):
            raise ValueError(
                f"Kawasaki proposal cutoff {self.cutoff:g} produced a graph with no "
                "neighbour pairs for at least one system"
            )
        offsets = torch.zeros(counts.numel() + 1, dtype=torch.long, device=batch.device)
        offsets[1:] = torch.cumsum(counts, dim=0)
        self._edges = edges
        self._edge_graph = batch.batch_idx[edges[:, 0]].to(torch.long)
        self._edge_offsets = offsets
        self._graph_batch = weakref.ref(batch)

    def _ensure_proposal_graph(self, batch: Batch) -> None:
        """Build the proposal graph on first use for a given batch object."""
        if self._edges is None or not _is_batch(self._graph_batch, batch):
            self._build_proposal_graph(batch)

    def synchronize(self, batch: Batch) -> None:
        """Adopt a trusted current energy and rebuild the proposal graph.

        Call this after positions change outside this sampler (e.g. an MD
        block), since the fixed proposal graph must reflect the new geometry
        before further Kawasaki proposals.
        """
        super().synchronize(batch)
        self._build_proposal_graph(batch)

    def _unlike_counts(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return per-edge unlike-species flags and their per-graph counts.

        Runs twice per MC step, so it avoids ops whose output size depends on
        the data (``torch.bincount``, boolean-mask indexing): on CUDA those
        force a device-to-host sync that stalls the next model call.
        """
        edges = self._edges
        numbers = batch.atomic_numbers
        unlike = (numbers[edges[:, 0]] != numbers[edges[:, 1]]).to(torch.long)
        counts = torch.zeros(batch.num_graphs, dtype=torch.long, device=numbers.device)
        counts.index_add_(0, self._edge_graph, unlike)
        return unlike, counts

    def _propose(
        self,
        batch: Batch,
        generator: torch.Generator,
        active: torch.Tensor,
    ) -> None:
        """Swap one random nearest-neighbour pair per active graph."""
        self._ensure_proposal_graph(batch)
        starts = self._edge_offsets[:-1]
        draws = torch.rand(batch.num_graphs, device=batch.device, generator=generator)
        if self.unlike_pairs_only:
            unlike, counts = self._unlike_counts(batch)
            self._unlike_before = counts
            # All-same-species neighbours mean no legal move: drop the graph
            # from this step's proposals and statistics (``active`` is the
            # sampler's own mask tensor, so this narrows it in place).
            active &= counts > 0
            rows = _draw_from_mask(unlike, starts, counts, draws)
        else:
            self._unlike_before = None
            rows = starts + torch.floor(draws * (self._edge_offsets[1:] - starts)).to(
                torch.long
            )
        self._apply_swap(batch, self._edges[rows, 0], self._edges[rows, 1], active)

    def _chemical_delta(self, batch: Batch) -> torch.Tensor:
        r"""Return the Metropolis-Hastings correction for unlike-pair proposals.

        Drawing uniformly from the *current* unlike-species pairs gives
        :math:`q(x \to x') = 1 / n(x)`, which changes with the configuration,
        so acceptance needs the factor :math:`n(x) / n(x')`. The base class
        adds this return value to :math:`\Delta E` before dividing by
        :math:`k_B T`, so the equivalent energy term is
        :math:`k_B T \ln[n(x') / n(x)]`. Called after the trial swap, so
        ``batch`` already holds :math:`x'`.
        """
        if not self.unlike_pairs_only or self._unlike_before is None:
            return super()._chemical_delta(batch)
        _, after = self._unlike_counts(batch)
        before = self._unlike_before
        dtype = batch.positions.dtype
        # A swapped unlike pair is still unlike, so n(x') > 0 whenever the
        # graph moved; the guard covers graphs that proposed nothing.
        movable = (before > 0) & (after > 0)
        ratio = torch.where(
            movable,
            torch.log(after.to(dtype).clamp(min=1.0) / before.to(dtype).clamp(min=1.0)),
            torch.zeros(batch.num_graphs, dtype=dtype, device=batch.device),
        )
        return KB_EV * self._temperature_for(batch) * ratio
