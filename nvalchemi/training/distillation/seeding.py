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
"""Seed source of an on-policy segment loop, a dataset behind one cursor.

A segment loop reads its seed structures twice: once to build the batch the
first segment propagates from, and again whenever a trajectory finishes and
:meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` backfills a fresh
one. This module holds both behind a single cursor over the rows one rank owns,
so a structure is propagated once, a restart resumes where it stopped, and the
sampler surface ``refill_check`` requires is answered by the same object that
seeded the run.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import torch
from pydantic import BaseModel, ConfigDict, Field

from nvalchemi.dynamics.base import BaseDynamics

if TYPE_CHECKING:
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
    from nvalchemi.dynamics.base import ConvergenceHook
    from nvalchemi.dynamics.sampler import SizeAwareSampler

__all__ = ["SeedSource"]


def _dataset_spec_dict(dataset: BatchDatasetProtocol, field: str) -> dict[str, Any]:
    """Return the store reference a path-backed dataset round-trips as.

    Parameters
    ----------
    dataset : BatchDatasetProtocol
        Dataset to reference. Only a dataset reading a filesystem or URI store
        can be named in a recipe; one holding its samples in memory cannot.
    field : str
        Name of the recipe field being serialized, quoted in the error.

    Returns
    -------
    dict[str, Any]
        ``{"path": ..., "device": ...}`` reference the rebuild reopens.

    Raises
    ------
    ValueError
        If *dataset* is not backed by a store a path names.
    """
    store = getattr(getattr(dataset, "reader", None), "_store", None)
    if not isinstance(store, (str, Path)):
        raise ValueError(
            f"{field} is a {type(dataset).__name__} holding its samples in "
            "memory, which no recipe can name: a spec references a dataset by "
            "the store it reads. Write the samples to a store with "
            "nvalchemi.training.distillation.label_dataset (or an "
            "AtomicDataZarrWriter) and point the recipe at that path, or "
            f"re-supply {field} at construction."
        )
    return {"path": str(store), "device": str(getattr(dataset, "target_device", "cpu"))}


def _dataset_from_spec_dict(spec: Mapping[str, Any]) -> BatchDatasetProtocol:
    """Reopen the dataset :func:`_dataset_spec_dict` referenced.

    Parameters
    ----------
    spec : Mapping[str, Any]
        Reference produced by :func:`_dataset_spec_dict`.

    Returns
    -------
    BatchDatasetProtocol
        Dataset over the referenced store. The reader it opens stays open for
        the caller to close.

    Raises
    ------
    pydantic.ValidationError
        If *spec* names no store to read, or carries a key that is not part of
        a store reference.
    """
    from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset

    reference = _DatasetRef.model_validate(spec)
    return Dataset(AtomicDataZarrReader(reference.path), device=reference.device)


class _DatasetRef(BaseModel):
    """Store reference a recipe names one dataset by."""

    path: Annotated[
        str,
        Field(description="Filesystem path or URI of the store to read."),
    ]
    device: Annotated[
        str,
        Field(
            default="cpu",
            description="Device the dataset collates the rows it serves onto.",
        ),
    ] = "cpu"

    model_config = ConfigDict(extra="forbid")


class _SeedSourceSpec(BaseModel):
    """Recipe block a :class:`SeedSource` is rebuilt from.

    Validating the block before anything is opened refuses a budget that is
    not a positive count, a ``recycle`` flag nothing reads as a boolean, and a
    misspelled knob where a recipe is read rather than inside the run it
    describes — a misspelling in particular, since a source is unbudgeted by
    default and one that never reached a field silently generates under no
    budget at all.
    """

    dataset: Annotated[
        _DatasetRef,
        Field(description="Store the seed structures are read from."),
    ]
    max_atoms: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Total atoms a seeded or refilled batch may hold.",
        ),
    ] = None
    max_edges: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Total stored edges a seeded or refilled batch may hold.",
        ),
    ] = None
    max_batch_size: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Total structures a seeded or refilled batch may hold.",
        ),
    ] = None
    recycle: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Whether a cursor at the end of the shard wraps to its front "
                "instead of reporting the source exhausted."
            ),
        ),
    ] = False

    model_config = ConfigDict(extra="forbid")


def _propagator_tree(dynamics: BaseDynamics) -> Iterator[BaseDynamics]:
    """Yield *dynamics* and every propagator it composes, each exactly once.

    A composition holds none of the state that drives a step: a
    :class:`~nvalchemi.dynamics.FusedStage` keeps its integrators in
    ``sub_stages`` and a pipeline keeps its stages in ``stages``, so anything
    read off the root alone misses the propagator actually running. Nodes are
    compared by identity rather than by equality, because one integrator object
    reached through two sub-stages is a single propagator holding a single
    seed.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator at the root of the composition.

    Yields
    ------
    BaseDynamics
        Every propagator in the tree, the root first.
    """
    seen: list[BaseDynamics] = []
    pending: list[BaseDynamics] = [dynamics]
    while pending:
        node = pending.pop()
        if any(node is visited for visited in seen):
            continue
        seen.append(node)
        yield node
        pending.extend(sub for _, sub in getattr(node, "sub_stages", ()))
        stages = getattr(node, "stages", ())
        pending.extend(stages.values() if isinstance(stages, Mapping) else stages)


def _seed_field_requirements(dynamics: BaseDynamics) -> tuple[str, ...]:
    """Return the batch fields *dynamics* reads before its first force evaluation.

    A propagator opens its step with ``pre_update``, which runs on the outputs
    of the *previous* step: the fields its ``__needs_keys__`` model outputs
    populate have to be on the seed batch already, zero-filled if nothing has
    computed them yet. It also reads whatever it updates in place, which is its
    ``__provides_keys__`` state other than ``positions`` — ``velocities`` for
    the integrators and the fixed-cell optimizers, and ``cell`` on top of that
    for the variable-cell ones, which invert it before the first force
    evaluation. A propagator that carries momentum divides forces by masses, so
    it reads ``atomic_masses`` too.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator the seed structures are propagated by.

    Returns
    -------
    tuple[str, ...]
        Sorted batch field names the seed structures have to carry.
    """
    fields = {
        dynamics._OUTPUT_KEY_TO_BATCH_ATTR.get(key, key)
        for key in dynamics.__needs_keys__
    }
    fields |= dynamics.__provides_keys__ - {"positions"}
    if "velocities" in fields:
        fields.add("atomic_masses")
    return tuple(sorted(fields))


def _check_seed_fields(state: Batch, dynamics: BaseDynamics) -> None:
    """Reject a seed batch the propagator cannot take its first step from.

    Parameters
    ----------
    state : Batch
        Batch the first segment would propagate from, or the one-row probe
        standing in for it at construction.
    dynamics : BaseDynamics
        Propagator the batch is seeded for.

    Raises
    ------
    ValueError
        If *state* is missing a field *dynamics* reads before its first force
        evaluation.
    """
    missing = [
        field for field in _seed_field_requirements(dynamics) if field not in state
    ]
    if not missing:
        return
    raise ValueError(
        f"Seed structures must carry the fields {type(dynamics).__name__} "
        f"propagates from; got missing {missing!r}. It reads the batch fields of "
        f"__needs_keys__={sorted(dynamics.__needs_keys__)!r} before evaluating "
        f"the model for the first time, and updates "
        f"__provides_keys__={sorted(dynamics.__provides_keys__)!r} in place from "
        "them, so a seed structure has to arrive with all of them — zeros are "
        "enough for the model outputs, AtomicData fills velocities and "
        "atomic_masses in itself unless a store dropped them, and a cell has to "
        "be carried because nothing fills that in for an aperiodic structure."
    )


def _check_seed_status(state: Batch, criterion: ConvergenceHook) -> None:
    """Reject a criterion that migrates off a status no seed graph holds.

    :meth:`~nvalchemi.dynamics.base.ConvergenceHook.__call__` migrates only the
    graphs sitting on its ``source_status``, so a criterion aimed at another one
    leaves the lifecycle inert in the worst way: nothing freezes, nothing
    graduates, and the same criterion installed as the detector keeps cutting
    segments short over structures that are still being propagated and
    re-captured. Nothing warns, because there is no exhaustion to warn about.

    Parameters
    ----------
    state : Batch
        Seed batch, already stamped with the run's own bookkeeping.
    criterion : ConvergenceHook
        Criterion driving the trajectory lifecycle.

    Raises
    ------
    ValueError
        If no seed graph carries the criterion's ``source_status``.
    """
    statuses = sorted({int(value) for value in state["status"].view(-1).tolist()})
    if criterion.source_status in statuses:
        return
    raise ValueError(
        "A converged graph migrates off the status its seed carries, and the "
        "run stamps that status itself rather than reading it from the seed "
        f"structures; got source_status={criterion.source_status!r} against "
        f"seed statuses {statuses!r}, so nothing would ever freeze or "
        "graduate. Pass source_status=0, or pass the fmax threshold itself and "
        "let the shorthand wire it up."
    )


class SeedSource:
    """Seed structures of a segment loop, served in order from one cursor.

    A run reads its seeds twice — once to build the batch the first segment
    propagates from, and again for every trajectory a relaxation lifecycle
    graduates and backfills — and this class is both, so the two never disagree
    about what has been served. It answers the five members
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` reads off
    ``dynamics.sampler`` (``max_atoms``, ``max_edges``, ``max_batch_size``,
    ``request_replacements_budget``, and ``exhausted``), and structures are
    handed out sequentially from the position the initial batch left behind, so
    no structure is propagated twice within one pass.

    An *unbudgeted* source — the 90% case, and what a bare dataset is coerced
    into — seeds every row it owns as one batch, which keeps the trajectory
    count explicit: it *is* the set of systems the run generates from, so size
    it to the device. It opens exhausted, and the batch then narrows one
    trajectory per graduation unless ``recycle`` wraps the cursor back to the
    beginning. A *budgeted* source packs the initial batch from the cursor
    while structures fit and stops at the first that does not, leaving the
    remainder in cursor order for the backfill to draw on.

    The size envelope of an unbudgeted source is the seeded batch itself:
    ``max_batch_size`` is the number of trajectories the run started with, so a
    backfill never widens the frame past it, and ``max_atoms`` is the atom
    count it started with, so a backfill never grows it beyond the footprint
    the device already held. It is measured once, off the rows
    :meth:`initial_batch` packed, and carried across a restart by
    :meth:`state_dict`, because the batch a restart resumes has already
    narrowed away every trajectory the run graduated and a source that
    re-derived its envelope from that batch would ratchet the run's footprint
    down a little further at every restart. ``max_edges`` stays ``None`` unless
    the caller set it, deliberately: the edges of a live frame are the neighbor
    list a propagator rebuilds every step, while the edge count a dataset
    reports is whatever it stored, and budgeting the first against the second
    would reject every replacement of a run whose neighbor list is denser than
    its store.

    :meth:`shard` narrows the source to the rows one rank of a data-parallel
    run owns. These rows are the whole of what that rank may propagate, and
    anything refilling or backfilling the trajectory batch has to draw from
    them alone. The deal is strided, unpadded, and unshuffled, so the shards
    are disjoint, and a structure served to a rank that does not own it is
    propagated twice and billed to the teacher twice. The cursor is therefore
    shard-local — what it has consumed, its length, where it wraps, and when it
    reports itself exhausted all count positions in :attr:`rows` rather than
    rows of the dataset.

    A ``system_id`` is not a position. Ids number the trajectories the run has
    started, so under ``recycle`` they keep climbing past the shard's length
    while the cursor wraps back through it, and each rank hands them out from
    its own base rather than from a dataset row. That is why
    :attr:`next_system_id` is tracked separately from :attr:`cursor`: a restart
    that derives one from the other rewinds a recycled run to the first
    structure instead of resuming where it stopped.

    Parameters
    ----------
    dataset : BatchDatasetProtocol
        Seed structures, indexed in the order they are served.
    max_atoms : int | None, optional
        Total atoms a seeded or refilled batch may hold. Default ``None``,
        which seeds every row this source owns and then holds the backfill to
        the envelope that batch established.
    max_edges : int | None, optional
        Total stored edges a seeded or refilled batch may hold. Default
        ``None``, which budgets no edges at all.
    max_batch_size : int | None, optional
        Total structures a seeded or refilled batch may hold. Default
        ``None``, resolved like ``max_atoms``.
    recycle : bool, optional
        Whether a cursor at the end of the shard wraps to the beginning
        instead of reporting the source exhausted. Default ``False``.

    Raises
    ------
    ValueError
        If a budget is set and not positive.

    Examples
    --------
    >>> from nvalchemi.training.distillation import SeedSource
    >>> seeds = SeedSource(seed_dataset, recycle=True)  # doctest: +SKIP
    >>> state = seeds.initial_batch()  # doctest: +SKIP

    Notes
    -----
    ``max_edges`` is honored on a refill only when the caller set it, so an
    unbudgeted source that recorded its envelope from the seeded batch still
    passes every edge budget it is handed.
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
        """Open a cursor at the first row of *dataset*."""
        declared = {
            "max_atoms": max_atoms,
            "max_edges": max_edges,
            "max_batch_size": max_batch_size,
        }
        for name, value in declared.items():
            if value is not None and value <= 0:
                raise ValueError(
                    f"SeedSource {name} bounds a batch and must be positive "
                    f"when set; got {value!r}. Leave it None to budget on the "
                    "seeded batch instead."
                )
        self.dataset = dataset
        self.recycle = recycle
        self.max_atoms = max_atoms
        self.max_edges = max_edges
        self.max_batch_size = max_batch_size
        self._declared = declared
        self._rows: tuple[int, ...] = tuple(range(len(dataset)))
        self._cursor = 0
        self._wraps = 0
        self._next_system_id = 0
        self._rank = 0
        self._world_size = 1

    def __len__(self) -> int:
        """Return the number of rows this source owns."""
        return len(self._rows)

    @property
    def rows(self) -> tuple[int, ...]:
        """Dataset rows this source serves, in the order it serves them."""
        return self._rows

    @property
    def cursor(self) -> int:
        """Position in :attr:`rows` the next structure is served from."""
        return self._cursor

    @property
    def wraps(self) -> int:
        """Times a recycling cursor has restarted at the front of the shard."""
        return self._wraps

    @property
    def next_system_id(self) -> int:
        """``system_id`` the next structure handed out is stamped with."""
        return self._next_system_id

    @property
    def budgeted(self) -> bool:
        """Whether the caller declared a size budget of its own."""
        return any(value is not None for value in self._declared.values())

    @property
    def exhausted(self) -> bool:
        """Whether the shard has no structure left to hand out."""
        return not self.recycle and self._cursor >= len(self._rows)

    def shard(self, rank: int, world_size: int) -> None:
        """Narrow this source to the rows rank *rank* of *world_size* owns.

        Seeds are dealt out strided — rank ``r`` takes every
        ``world_size``-th structure from offset ``r`` — so the shards are
        disjoint, cover the dataset, and differ by at most one *structure*: the
        deal balances the count, not the work, because it strides by index and
        never reads how big a structure is. An ordering whose period shares a
        factor with the world therefore hands one rank a many-fold heavier
        shard; sorting the seed dataset by atom count makes the strided deal
        balance by construction. The deal is unpadded and unshuffled, which is
        where it parts company with
        :class:`~torch.utils.data.DistributedSampler`: that one pads its index
        list up to a whole multiple of the world, handing a structure to two
        ranks, and here that structure would be propagated twice and billed to
        the teacher twice.

        The cursor, the wrap count, and the next ``system_id`` are reset, and a
        recorded envelope is dropped back to whatever the caller declared, so
        installing a shard on a source that has already run reseeds it rather
        than resuming it.

        Parameters
        ----------
        rank : int
            Global rank claiming a shard.
        world_size : int
            Ranks the seed dataset is dealt across. A single-rank run gets the
            whole dataset, unchanged.

        Raises
        ------
        ValueError
            If *world_size* is not positive or *rank* falls outside it.
        """
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(
                "A seed shard is dealt to one rank of a world, so the rank has "
                f"to name a position in it; got rank={rank!r} of "
                f"world_size={world_size!r}."
            )
        self._rank = rank
        self._world_size = world_size
        self._rows = tuple(range(rank, len(self.dataset), world_size))
        self._cursor = 0
        self._wraps = 0
        self._next_system_id = 0
        self.max_atoms = self._declared["max_atoms"]
        self.max_edges = self._declared["max_edges"]
        self.max_batch_size = self._declared["max_batch_size"]

    def probe(self) -> Batch:
        """Return the first row of the shard, as the one-graph batch it loads as.

        The row is loaded through the dataset's own collation rather than read
        as an :class:`~nvalchemi.data.AtomicData`, because that is what fills
        in the ``velocities`` and ``atomic_masses`` a store need not have kept
        and a propagator still reads.

        Returns
        -------
        Batch
            One graph, for a check that has to run before a run is paid for.

        Raises
        ------
        ValueError
            If this source owns no rows at all.
        """
        if not self._rows:
            raise ValueError(
                "A seed source has to hold at least one structure; got a "
                f"{type(self.dataset).__name__} of length "
                f"{len(self.dataset)!r} sharded to no rows."
            )
        return self.dataset.load_batches([[self._rows[0]]])[0]

    def initial_batch(self) -> Batch:
        """Return the batch the first segment propagates from, advancing the cursor.

        The batch enters the run carrying none of the propagator's
        bookkeeping, so this source installs its own. ``status`` and
        ``system_id`` describe the run that wrote them, and a seed loaded from
        a store a dynamics sink filled — the obvious provenance for "relax
        these structures, then generate from the minima" — arrives holding
        whatever it graduated with.
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.step` freezes every graph
        whose ``status`` has reached ``exit_status``, so a stale one would run
        a segment that moves nothing and fills the buffer with copies of the
        seeds, reported as a normal run.

        An unbudgeted source records its envelope here, from the sizes the
        dataset reports for the rows it packed rather than from the batch they
        loaded as. The two agree, since an unbudgeted pack takes every row left
        at the cursor, but only the first is a figure the source owns: an
        envelope read off a live batch is whatever batch the caller happens to
        hand over.

        Returns
        -------
        Batch
            Seed batch, stamped with clean bookkeeping and numbered from
            :attr:`next_system_id`.

        Raises
        ------
        ValueError
            If the cursor has nothing left to seed from, or if the first
            structure at the cursor is larger than the declared budget.
        """
        rows = self._pack_initial_rows()
        if not rows:
            raise ValueError(
                "A segment loop has to propagate something; got no seed "
                f"structure at cursor {self._cursor!r} of {len(self._rows)!r} "
                f"rows fitting max_atoms={self.max_atoms!r}, "
                f"max_edges={self.max_edges!r}, and "
                f"max_batch_size={self.max_batch_size!r}. Widen the budget, or "
                "pass a seed dataset holding a structure that fits it."
            )
        state = self.dataset.load_batches([rows])[0]
        for key in BaseDynamics._bookkeeping_keys:
            if key in state:
                del state[key]
        self._stamp_bookkeeping(state)
        if not self.budgeted:
            self.max_atoms = sum(self.dataset.get_metadata(row)[0] for row in rows)
            self.max_batch_size = len(rows)
        return state

    def record_envelope(self, state: Batch) -> None:
        """Adopt *state*'s own size as the envelope a backfill refills under.

        A source the caller gave a budget keeps that budget, and one that has
        already recorded an envelope keeps that too: this is the fallback for a
        run restored from a bundle written before :meth:`state_dict` carried
        the figure, not a way to reset it. The batch such a run resumes has
        already narrowed away every trajectory it graduated, so a source that
        adopted it every time would ratchet its envelope down one restart at a
        time; a source holding no envelope at all is still better off with that
        batch than backfilling under none.

        Parameters
        ----------
        state : Batch
            Batch the run is propagating, whose size is the envelope.
        """
        if self.budgeted or self.max_atoms is not None:
            return
        self.max_atoms = int(state.num_nodes)
        self.max_batch_size = int(state.num_graphs)

    def request_replacements_budget(
        self,
        atom_budget: int | None = None,
        edge_budget: int | None = None,
        max_count: int | None = None,
    ) -> list[AtomicData]:
        """Return the next structures that fit the freed slot and atom budget.

        A structure too large for the budget is skipped rather than allowed to
        block the queue, the way
        :meth:`~nvalchemi.dynamics.sampler.SizeAwareSampler.request_replacements_budget`
        passes over a candidate that does not fit — the budget after a
        graduation is exactly what graduated, so on a heterogeneous seed set a
        large structure at the cursor would otherwise starve every refill
        behind it. The scan gives up after one pass over the shard, counting
        every structure it reaches rather than only the ones it skipped: a
        recycling cursor that wrapped mid-scan would otherwise serve a
        structure it had already served in the same call, and two copies of one
        seed entering the batch together relax in lockstep into duplicate
        frames.

        Parameters
        ----------
        atom_budget : int | None, optional
            Atoms the graduated structures freed. Default ``None``
            (unconstrained).
        edge_budget : int | None, optional
            Edges the graduated structures freed. Default ``None``, and
            ignored unless the caller declared ``max_edges``, because the
            stored edge count a dataset reports is not the neighbor list a
            propagator rebuilds every step.
        max_count : int | None, optional
            Slots the graduated structures freed. Default ``None``, which caps
            the request at one pass over the shard.

        Returns
        -------
        list[AtomicData]
            Structures to append to the active batch, oldest cursor position
            first, each stamped with its own ``system_id``. Empty once the
            source is exhausted, or once nothing a pass over it reaches fits
            the atom budget.
        """
        replacements: list[AtomicData] = []
        length = len(self._rows)
        atoms = atom_budget
        edges = edge_budget if self.max_edges is not None else None
        wanted = length if max_count is None else max_count
        scanned = 0
        while len(replacements) < wanted and scanned < length:
            if self._cursor >= length:
                if not self.recycle:
                    break
                self._cursor = 0
                self._wraps += 1
            index = self._rows[self._cursor]
            self._cursor += 1
            scanned += 1
            num_atoms, num_edges = self.dataset.get_metadata(index)
            if atoms is not None and num_atoms > atoms:
                continue
            if edges is not None and num_edges > edges:
                continue
            data, _ = self.dataset[index]
            data.add_system_property(
                "system_id",
                torch.tensor([[self._next_system_id]], dtype=torch.long),
            )
            self._next_system_id += 1
            replacements.append(data)
            if atoms is not None:
                atoms -= num_atoms
            if edges is not None:
                edges -= num_edges
        return replacements

    def state_dict(self) -> dict[str, int | None]:
        """Return the position and envelope a restart resumes this source from.

        Returns
        -------
        dict[str, int | None]
            The cursor, its wrap count, the next ``system_id``, and the shard
            the three were counted in. A source the caller gave no budget also
            writes ``max_atoms`` and ``max_batch_size``, the envelope it
            measured off the rows it seeded, which is state for the same
            reason the cursor is: nothing a restart holds can re-derive it. A
            budgeted source writes neither, so a bundle can never talk a run
            out of the budget its recipe declares. The dataset, the declared
            budgets and ``recycle`` are configuration a recipe carries, not
            state, and are left out.
        """
        state: dict[str, int | None] = {
            "cursor": self._cursor,
            "wraps": self._wraps,
            "next_system_id": self._next_system_id,
            "rank": self._rank,
            "world_size": self._world_size,
        }
        if not self.budgeted:
            state["max_atoms"] = self.max_atoms
            state["max_batch_size"] = self.max_batch_size
        return state

    def load_state_dict(self, state: Mapping[str, int | None]) -> None:
        """Resume this source at the cursor and envelope *state* recorded.

        An envelope in *state* is adopted only by a source the caller gave no
        budget of its own, so a bundle written before a recipe declared one
        cannot override it. A bundle carrying none — one a budgeted source
        wrote, or one written before this pair recorded the envelope at all —
        leaves the envelope to :meth:`record_envelope`.

        Parameters
        ----------
        state : Mapping[str, int | None]
            Bundle written by :meth:`state_dict`, on the shard this source is
            already narrowed to.

        Raises
        ------
        ValueError
            If *state* was written for another rank or another world size,
            whose cursor counts positions in a different set of rows.
        """
        rank = int(state["rank"])
        world_size = int(state["world_size"])
        if (rank, world_size) != (self._rank, self._world_size):
            raise ValueError(
                "The restart bundle's seed cursor was written for rank "
                f"{rank!r} of {world_size!r}; this rank is {self._rank!r} of "
                f"{self._world_size!r}. Restart on the world that wrote it, or "
                "reseed with a cold buffer."
            )
        self._cursor = int(state["cursor"])
        self._wraps = int(state["wraps"])
        self._next_system_id = int(state["next_system_id"])
        if not self.budgeted and "max_atoms" in state:
            self.max_atoms = state["max_atoms"]
            self.max_batch_size = state["max_batch_size"]

    def to_spec_dict(self) -> dict[str, Any]:
        """Return the JSON-ready reference a recipe names this source by.

        Returns
        -------
        dict[str, Any]
            The store the seeds are read from and the budgets the caller
            declared. The cursor is state and belongs to a restart bundle
            instead, and the rank shard is a launcher fact that belongs to
            neither.

        Raises
        ------
        ValueError
            If the seed dataset holds its samples in memory, which no recipe
            can name.
        """
        return {
            "dataset": _dataset_spec_dict(self.dataset, "OnPolicyConfig.seeds"),
            **self._declared,
            "recycle": self.recycle,
        }

    @classmethod
    def from_spec_dict(cls, spec: Mapping[str, Any]) -> SeedSource:
        """Rebuild the source :meth:`to_spec_dict` described.

        Parameters
        ----------
        spec : Mapping[str, Any]
            Reference produced by :meth:`to_spec_dict`.

        Returns
        -------
        SeedSource
            Source over the referenced store, with a cursor at its first row.

        Raises
        ------
        pydantic.ValidationError
            If *spec* carries a key no source takes, names no store to read
            the seeds from, or gives a budget that is not a positive count. It
            derives from :class:`ValueError`, so a caller that already reports
            a bad recipe reports this one the same way.
        """
        validated = _SeedSourceSpec.model_validate(spec)
        return cls(
            _dataset_from_spec_dict(validated.dataset.model_dump()),
            max_atoms=validated.max_atoms,
            max_edges=validated.max_edges,
            max_batch_size=validated.max_batch_size,
            recycle=validated.recycle,
        )

    @classmethod
    def from_sampler(cls, sampler: SizeAwareSampler) -> SeedSource:
        """Return the source equivalent to *sampler*, which it replaces.

        The sampler is read as an *input* rather than kept as a live delegate:
        its dataset and its three budgets describe a source exactly, while its
        largest-bin-first packing is a throughput heuristic for inflight
        batching over a whole store and its ``_consumed`` set carries no order,
        no index subset, and no state dict. On-policy seeding needs
        determinism, shard-locality and restart exactness instead, so the
        initial batch a converted source packs differs — first-fit in row order
        rather than largest-bin-first — while the contract does not: the budget
        is respected and the refill comes from the same dataset.

        Parameters
        ----------
        sampler : SizeAwareSampler
            Sampler that used to seed the run.

        Returns
        -------
        SeedSource
            Source over the sampler's dataset, under the sampler's budgets.

        Warns
        -----
        DeprecationWarning
            Always: a segment loop is seeded by a ``SeedSource`` now.
        """
        warnings.warn(
            "OnPolicyConfig takes a SeedSource under seeds= rather than a "
            "SizeAwareSampler, because the segment loop needs a cursor it can "
            "shard, restart and recycle; converting the sampler's dataset and "
            "budgets. Build the source directly to keep this quiet.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls(
            sampler._dataset,
            max_atoms=sampler.max_atoms,
            max_edges=sampler.max_edges,
            max_batch_size=sampler.max_batch_size,
        )

    def _pack_initial_rows(self) -> list[int]:
        """Return the rows the initial batch is built from, advancing the cursor."""
        if not self.budgeted:
            rows = list(self._rows[self._cursor :])
            self._cursor = len(self._rows)
            return rows
        rows = []
        atoms = 0
        edges = 0
        while self._cursor < len(self._rows):
            if self.max_batch_size is not None and len(rows) >= self.max_batch_size:
                break
            index = self._rows[self._cursor]
            num_atoms, num_edges = self.dataset.get_metadata(index)
            if self.max_atoms is not None and atoms + num_atoms > self.max_atoms:
                break
            if self.max_edges is not None and edges + num_edges > self.max_edges:
                break
            rows.append(index)
            atoms += num_atoms
            edges += num_edges
            self._cursor += 1
        return rows

    def _stamp_bookkeeping(self, state: Batch) -> None:
        """Give *state* the graph-level fields the refill cycle maintains.

        ``status`` is what a status-migrating
        :class:`~nvalchemi.dynamics.base.ConvergenceHook` writes and what
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.refill_check` graduates
        on, and ``system_id`` numbers the structures the way a backfill
        continues numbering them.
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
