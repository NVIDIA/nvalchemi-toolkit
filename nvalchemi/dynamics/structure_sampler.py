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
"""Ordered structure sampling for a propagator that admits structures over time.

A propagator run with in-flight batching reads structures at two points. It
reads them once to build the batch its first step propagates from. It reads
them again whenever a structure graduates and a fresh structure backfills the
room it frees. This module serves both reads from a single position over the
rows one rank owns. Each structure is therefore propagated once, and a restart
resumes where the run stopped. Whether a structure fits a batch is decided by
one :class:`FitPolicy` predicate rather than by a fixed set of budget
arguments.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import torch

from nvalchemi.data.datapipes.samplers import distributed_shard
from nvalchemi.dynamics.base import BaseDynamics

if TYPE_CHECKING:
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol

__all__ = ["FitPolicy", "OrderedStructureSampler", "StructureSource", "WithinBudget"]


class FitPolicy(Protocol):
    """Decide whether the batch being drawn still fits once a candidate joins it.

    :meth:`OrderedStructureSampler.draw` calls the policy with the atom and
    edge totals the drawn structures would hold with the candidate included.
    A policy is therefore a stateless predicate over running totals.
    :class:`WithinBudget` bounds the totals. A policy that bounds a memory
    estimate, or any other quantity, is another class with the same call
    signature.
    """

    def __call__(self, num_atoms: int, num_edges: int) -> bool:
        """Return whether a drawn batch totaling *num_atoms* and *num_edges* fits."""
        ...


@dataclasses.dataclass(frozen=True)
class WithinBudget:
    """Fit policy that admits a batch while its totals stay within the given bounds.

    Parameters
    ----------
    atoms : int | None, optional
        Total atoms the drawn batch may hold. Default ``None`` (unbounded).
    edges : int | None, optional
        Total stored edges the drawn batch may hold. A dataset reports the edge
        count it stored, not the size of the neighbor list a propagator
        rebuilds every step, so set this bound only when the stored count is
        the one that matters. Default ``None`` (unbounded).

    Examples
    --------
    >>> from nvalchemi.dynamics import WithinBudget
    >>> WithinBudget(atoms=10)(num_atoms=8, num_edges=0)
    True
    >>> WithinBudget(atoms=10)(num_atoms=12, num_edges=0)
    False
    """

    atoms: int | None = None
    edges: int | None = None

    def __call__(self, num_atoms: int, num_edges: int) -> bool:
        """Return whether *num_atoms* and *num_edges* both stay within the bounds."""
        return (self.atoms is None or num_atoms <= self.atoms) and (
            self.edges is None or num_edges <= self.edges
        )


@runtime_checkable
class StructureSource(Protocol):
    """Structures a propagator starts from, served in order from one position.

    The protocol lists the members an in-flight batching driver reads, so any
    object that provides them can stand in for a dataset-backed sampler.
    :meth:`probe` returns one row for checks that run at construction.
    :meth:`shard` narrows the source to the rows one rank owns and resets the
    position. :meth:`initial_batch` builds the batch the first step
    propagates from. :meth:`draw` serves the fresh structures a backfill
    adds. :attr:`exhausted` reports whether any structure is left.
    :meth:`state_dict` and :meth:`load_state_dict` carry the position through
    a restart. :class:`OrderedStructureSampler` is the reference
    implementation, backed by a dataset.

    Examples
    --------
    >>> from nvalchemi.dynamics import OrderedStructureSampler, StructureSource
    >>> isinstance(OrderedStructureSampler(dataset), StructureSource)  # doctest: +SKIP
    True
    """

    @property
    def exhausted(self) -> bool:
        """Whether the source has no structure left to hand out."""
        ...

    def shard(self, rank: int, world_size: int) -> None:
        """Narrow the source to the rows *rank* of *world_size* owns and reset it."""
        ...

    def probe(self) -> Batch:
        """Return one row, collated as the one-graph batch a propagator receives."""
        ...

    def initial_batch(self) -> Batch:
        """Return the batch the first step propagates from, advancing the position.

        A source that feeds a trajectory lifecycle stamps every structure with a
        ``status`` of ``0`` and its own ``system_id``, as
        :class:`OrderedStructureSampler` does.
        """
        ...

    def draw(
        self,
        *,
        limit: int | None = None,
        fits: FitPolicy | None = None,
        on_miss: Literal["stop", "skip"] = "stop",
    ) -> list[AtomicData]:
        """Serve the next structures while they pass *fits*."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Return the position a restart resumes this source from."""
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Resume this source at the position *state* recorded."""
        ...


class OrderedStructureSampler:
    """Structures served in dataset order from one position, for in-flight batching.

    The reference :class:`StructureSource`, backed by a dataset. A run reads
    the sampler once to build the batch the first step propagates from. A
    driver layered on top then draws from it again to backfill each structure
    it graduates. Both reads advance the same position, so no structure is
    propagated twice within one pass over the rows. An *unbudgeted* sampler
    serves every row it owns as one batch, so the run has one trajectory per
    row. A *budgeted* sampler packs the initial batch while structures fit
    and leaves the remaining rows, in order, for :meth:`draw`. Either way, the
    initial batch is packed as one :meth:`draw` call with a
    :class:`WithinBudget` policy and ``on_miss="stop"`` would pack it. A
    backfill that fills the room a graduation freed passes its own policy
    with ``on_miss="skip"``.

    :meth:`shard` narrows the sampler to the rows one rank owns. The rows are
    dealt strided and unpadded, so the shards are disjoint and no structure is
    propagated twice. :attr:`next_row` counts positions in :attr:`rows`. A
    ``system_id`` is not a position: ids count only the structures the run
    has started, not the rows a policy passed over. :attr:`next_system_id` is
    therefore tracked separately from :attr:`next_row`.

    A sampler built with ``recycle=True`` never reports itself exhausted.
    When its position reaches the end of the shard, it wraps to the front, and
    :attr:`wraps` counts how often that happened. The ``system_id`` numbers
    keep increasing across a wrap. One :meth:`draw` reaches every row at most
    once, so a single call never serves two copies of one structure. Across
    calls there is no such guarantee: a trajectory that outlives a full pass
    over the shard shares the batch with a second copy of its structure.

    Parameters
    ----------
    dataset : BatchDatasetProtocol
        Structures, indexed in the order they are served.
    max_atoms : int | None, optional
        Total atoms the initial batch may hold. Default ``None`` puts no atom
        bound on it.
    max_edges : int | None, optional
        Total stored edges the initial batch may hold. Default ``None`` puts no
        edge bound on it.
    max_batch_size : int | None, optional
        Total structures the initial batch may hold. Default ``None`` puts no
        bound on the count. With all three ``None``, the sampler is unbudgeted
        and serves every row it owns as one batch.
    recycle : bool, optional
        Whether the position wraps to the front of the shard when it reaches
        the end, instead of the sampler reporting itself exhausted. Default
        ``False``.

    Raises
    ------
    ValueError
        If a budget is set and not positive.

    Examples
    --------
    >>> from nvalchemi.dynamics import OrderedStructureSampler, WithinBudget
    >>> sampler = OrderedStructureSampler(dataset, max_atoms=10_000)  # doctest: +SKIP
    >>> batch = sampler.initial_batch()  # doctest: +SKIP
    >>> fresh = sampler.draw(limit=2, fits=WithinBudget(atoms=64))  # doctest: +SKIP
    >>> sampler.next_row  # doctest: +SKIP
    6
    """

    def __init__(
        self,
        dataset: BatchDatasetProtocol,
        *,
        max_atoms: int | None = None,
        max_edges: int | None = None,
        max_batch_size: int | None = None,
        recycle: bool = False,
    ) -> None:
        """Open the sampler at the first row of *dataset*."""
        declared = {
            "max_atoms": max_atoms,
            "max_edges": max_edges,
            "max_batch_size": max_batch_size,
        }
        for name, value in declared.items():
            if value is not None and value <= 0:
                raise ValueError(
                    f"{type(self).__name__} {name} bounds the initial batch, so "
                    f"it must be positive when set; got {name}={value!r}. Leave "
                    "it None to lift that bound."
                )
        self.dataset = dataset
        self.max_atoms = max_atoms
        self.max_edges = max_edges
        self.max_batch_size = max_batch_size
        self.recycle = recycle
        self._rows: tuple[int, ...] = tuple(range(len(dataset)))
        self._next_row = 0
        self._wraps = 0
        self._next_system_id = 0
        self._rank = 0
        self._world_size = 1

    def __len__(self) -> int:
        """Return the number of rows this sampler owns."""
        return len(self._rows)

    @property
    def rows(self) -> tuple[int, ...]:
        """Dataset rows this sampler serves, in the order it serves them."""
        return self._rows

    @property
    def next_row(self) -> int:
        """Position in :attr:`rows` of the next structure served."""
        return self._next_row

    @property
    def next_system_id(self) -> int:
        """``system_id`` stamped on the next structure handed out."""
        return self._next_system_id

    @property
    def wraps(self) -> int:
        """How often a recycling position has wrapped to the front of the shard."""
        return self._wraps

    @property
    def exhausted(self) -> bool:
        """Whether the shard has no structure left to hand out."""
        return not self.recycle and self._next_row >= len(self._rows)

    def shard(self, rank: int, world_size: int) -> None:
        """Narrow this sampler to the rows that rank *rank* of *world_size* owns.

        :func:`~nvalchemi.data.datapipes.distributed_shard` deals the rows
        strided: rank ``r`` takes every ``world_size``-th structure, starting
        at offset ``r``. The shards are disjoint, cover the dataset, and differ
        by at most one structure. The deal balances the number of structures,
        not the work, so sort the dataset by atom count when structure sizes
        differ widely. The shards are not padded, because a padded structure
        would be propagated twice. Sharding resets the position, its wrap
        count, and the next ``system_id``, so a sampler that has already run
        restarts from its first row instead of resuming.

        Parameters
        ----------
        rank : int
            Global rank claiming a shard.
        world_size : int
            Number of ranks the dataset is dealt across. A single-rank run
            keeps the whole dataset, unchanged.

        Raises
        ------
        ValueError
            If *world_size* is not positive or *rank* falls outside it.
        """
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(
                "A shard needs a positive world_size and a rank within it, "
                f"0 <= rank < world_size; got rank={rank!r} of "
                f"world_size={world_size!r}. Pass the launcher's global rank "
                "and world size."
            )
        self._rank = rank
        self._world_size = world_size
        self._rows = tuple(
            distributed_shard(
                list(range(len(self.dataset))),
                num_replicas=world_size,
                rank=rank,
                drop_last=False,
                pad=False,
            )
        )
        self._next_row = 0
        self._wraps = 0
        self._next_system_id = 0

    def probe(self) -> Batch:
        """Return the first row of the shard as a one-graph batch.

        The row is loaded through the dataset's own collation rather than read
        as an :class:`~nvalchemi.data.AtomicData`. The collation fills in the
        ``velocities`` and ``atomic_masses`` that a propagator reads but a
        store may not have kept.

        Returns
        -------
        Batch
            One-graph batch, for checks that must run before any compute is
            spent on the run.

        Raises
        ------
        ValueError
            If this sampler owns no rows at all.
        """
        if not self._rows:
            raise ValueError(
                f"{type(self).__name__} owns no row to probe: a "
                f"{type(self.dataset).__name__} of length {len(self.dataset)!r} "
                f"dealt across world_size={self._world_size!r} leaves "
                f"rank={self._rank!r} empty. Provide at least as many structures "
                "as ranks."
            )
        return self.dataset.load_batches([[self._rows[0]]])[0]

    def initial_batch(self) -> Batch:
        """Return the batch the first step propagates from, advancing the position.

        Any propagator bookkeeping the loaded structures carry is dropped, and
        this sampler installs fresh bookkeeping. A structure loaded from a
        store that a dynamics sink filled arrives with the ``status`` it
        graduated with. :meth:`~nvalchemi.dynamics.base.BaseDynamics.step`
        would keep such a structure frozen at ``exit_status``, so its steps
        would move nothing.

        Returns
        -------
        Batch
            Initial batch, stamped with clean bookkeeping and numbered from
            :attr:`next_system_id`.

        Raises
        ------
        ValueError
            If the sampler has nothing left to serve, or if the structure at
            :attr:`next_row` is larger than the declared budget.
        """
        budget = WithinBudget(atoms=self.max_atoms, edges=self.max_edges)
        rows = self._scan_rows(
            limit=self.max_batch_size,
            fits=None if budget == WithinBudget() else budget,
            on_miss="stop",
        )
        if not rows:
            raise ValueError(
                "A run has to propagate something; got no structure at row "
                f"{self._next_row!r} of {len(self._rows)!r} fitting "
                f"max_atoms={self.max_atoms!r}, max_edges={self.max_edges!r}, "
                f"and max_batch_size={self.max_batch_size!r}. Widen the budget, "
                "or pass a dataset holding a structure that fits it."
            )
        state = self.dataset.load_batches([rows])[0]
        for key in BaseDynamics._bookkeeping_keys:
            if key in state:
                del state[key]
        self._stamp_bookkeeping(state)
        return state

    def draw(
        self,
        *,
        limit: int | None = None,
        fits: FitPolicy | None = None,
        on_miss: Literal["stop", "skip"] = "stop",
    ) -> list[AtomicData]:
        """Serve the next structures while they pass *fits*.

        Parameters
        ----------
        limit : int | None, optional
            Maximum number of structures to serve. Default ``None`` (the rest
            of the shard).
        fits : FitPolicy | None, optional
            Policy called with the atom and edge totals the drawn structures
            would hold with each candidate included. Default ``None`` (every
            structure fits).
        on_miss : {"stop", "skip"}, optional
            How the scan treats a candidate that does not fit. ``"stop"`` ends
            the draw and leaves :attr:`next_row` on the candidate; an initial
            batch is packed this way. ``"skip"`` passes over the candidate and
            continues. A backfill uses ``"skip"`` to fill the room a graduation
            freed, so one oversized structure cannot block every refill behind
            it. Default ``"stop"``.

        Returns
        -------
        list[AtomicData]
            Structures in row order, each stamped with its own ``system_id``.
            Empty once the shard is exhausted, or once the first candidate
            misses under ``on_miss="stop"``. A recycling sampler wraps to the
            front of the shard instead of running out. One call still reaches
            every row at most once.
        """
        drawn: list[AtomicData] = []
        for index in self._scan_rows(limit=limit, fits=fits, on_miss=on_miss):
            data, _ = self.dataset[index]
            data.add_system_property(
                "system_id",
                torch.tensor([[self._next_system_id]], dtype=torch.long),
            )
            self._next_system_id += 1
            drawn.append(data)
        return drawn

    def state_dict(self) -> dict[str, int]:
        """Return the position a restart resumes this sampler from.

        Returns
        -------
        dict[str, int]
            ``next_row``, ``wraps`` (how often the position has wrapped),
            ``next_system_id``, and the ``rank`` and ``world_size`` of the
            shard they count in. The dataset, the declared budgets, and
            ``recycle`` are configuration, not state, so they are left out.
        """
        return {
            "next_row": self._next_row,
            "wraps": self._wraps,
            "next_system_id": self._next_system_id,
            "rank": self._rank,
            "world_size": self._world_size,
        }

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        """Resume this sampler at the position *state* recorded.

        Parameters
        ----------
        state : Mapping[str, int]
            Bundle written by :meth:`state_dict` for the shard this sampler is
            already narrowed to.

        Raises
        ------
        KeyError
            If *state* lacks any of ``next_row``, ``wraps``, ``next_system_id``,
            ``rank``, or ``world_size``. One error names every missing key. A
            bundle written under the former ``cursor`` key is not read, so it
            raises too.
        ValueError
            If *state* was written for another rank or another world size. Its
            position counts rows in a different shard.
        """
        required = ("next_row", "wraps", "next_system_id", "rank", "world_size")
        missing = [key for key in required if key not in state]
        if missing:
            raise KeyError(
                f"{type(self).__name__} state is resumed from the keys "
                f"{list(required)!r}; got {sorted(state)!r}, missing {missing!r}. "
                "Pass a bundle written by state_dict()."
            )
        rank = int(state["rank"])
        world_size = int(state["world_size"])
        if (rank, world_size) != (self._rank, self._world_size):
            raise ValueError(
                "The restart bundle's position was written for rank "
                f"{rank!r} of {world_size!r}; this rank is {self._rank!r} of "
                f"{self._world_size!r}. Restart on the world that wrote it, or "
                "start over from the first row."
            )
        self._next_row = int(state["next_row"])
        self._wraps = int(state["wraps"])
        self._next_system_id = int(state["next_system_id"])

    def _scan_rows(
        self,
        *,
        limit: int | None,
        fits: FitPolicy | None,
        on_miss: Literal["stop", "skip"],
    ) -> list[int]:
        """Advance the position and return the rows the policy admitted.

        The scan reaches every row of the shard at most once. A recycling
        position that wraps during the scan therefore never serves a structure
        twice in the same call.
        """
        rows: list[int] = []
        atoms = edges = 0
        scanned = 0
        while scanned < len(self._rows) and (limit is None or len(rows) < limit):
            if self._next_row >= len(self._rows):
                if not self.recycle:
                    break
                self._next_row = 0
                self._wraps += 1
            index = self._rows[self._next_row]
            scanned += 1
            if fits is not None:
                num_atoms, num_edges = self.dataset.get_metadata(index)
                if not fits(atoms + num_atoms, edges + num_edges):
                    if on_miss == "stop":
                        break
                    self._next_row += 1
                    continue
                atoms += num_atoms
                edges += num_edges
            rows.append(index)
            self._next_row += 1
        return rows

    def _stamp_bookkeeping(self, state: Batch) -> None:
        """Give *state* the graph-level fields an in-flight run maintains.

        ``status`` is the field a status-migrating
        :class:`~nvalchemi.dynamics.base.ConvergenceHook` writes and a driver
        graduates structures on. ``system_id`` numbers the structures in the
        same sequence a backfill continues.
        """
        state["status"] = torch.zeros(
            state.num_graphs, 1, dtype=torch.long, device=state.device
        )
        state["system_id"] = torch.arange(
            self._next_system_id,
            self._next_system_id + state.num_graphs,
            dtype=torch.long,
            device=state.device,
        ).unsqueeze(-1)
        self._next_system_id += state.num_graphs
