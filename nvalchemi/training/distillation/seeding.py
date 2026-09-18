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
"""Initial structures of an on-policy segment loop, a dataset behind one cursor.

A segment loop reads its initial structures to build the batch the first segment
propagates from, and a trajectory lifecycle layered on top draws from them again
whenever a trajectory finishes and a fresh one is backfilled. This module serves
both from a single cursor over the rows one rank owns, so a structure is
propagated once and a restart resumes where it stopped, and it decides what fits
a batch through one :class:`FitPolicy` predicate rather than a fixed set of
budget arguments.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    Protocol,
    runtime_checkable,
)

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nvalchemi.dynamics.base import BaseDynamics

if TYPE_CHECKING:
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol
    from nvalchemi.dynamics.base import ConvergenceHook

__all__ = ["FitPolicy", "InitialStructures", "InitialStructuresSource", "WithinBudget"]


def _dataset_spec_dict(dataset: BatchDatasetProtocol, field: str) -> dict[str, Any]:
    """Return the store reference a path-backed dataset round-trips as.

    Parameters
    ----------
    dataset : BatchDatasetProtocol
        Dataset to reference. Only a dataset reading a filesystem or URI store
        can be named in a recipe; one holding its samples in memory cannot. A
        :class:`~nvalchemi.data.datapipes.multidataset.MultiDataset` is named
        by the stores it concatenates, in order.
    field : str
        Name of the recipe field being serialized, quoted in the error.

    Returns
    -------
    dict[str, Any]
        ``{"path": ..., "device": ...}`` for one store, or
        ``{"paths": [...], "device": ...}`` for a composition, which the
        rebuild reopens.

    Raises
    ------
    ValueError
        If *dataset*, or a dataset it composes, is not backed by a store a
        path names, or if a composition collates onto more than one device.
    """
    children = getattr(dataset, "datasets", None)
    if children is not None:
        references = [_dataset_spec_dict(child, field) for child in children]
        devices = sorted({reference["device"] for reference in references})
        if len(devices) != 1:
            raise ValueError(
                f"{field} composes stores collating onto different devices, "
                f"which one recipe reference cannot name; got {devices!r}. Open "
                "every store on one device."
            )
        paths = [
            path
            for reference in references
            for path in reference.get("paths") or [reference["path"]]
        ]
        return {"paths": paths, "device": devices[0]}
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
        Dataset over the referenced store, or a
        :class:`~nvalchemi.data.datapipes.multidataset.MultiDataset` over the
        referenced stores. The readers it opens stay open for the caller to
        close.

    Raises
    ------
    pydantic.ValidationError
        If *spec* names no store to read, names both one store and a list of
        them, or carries a key that is not part of a store reference.
    """
    from nvalchemi.data.datapipes import AtomicDataZarrReader, Dataset, MultiDataset

    reference = _DatasetRef.model_validate(spec)
    datasets = [
        Dataset(AtomicDataZarrReader(path), device=reference.device)
        for path in reference.paths or [reference.path]
    ]
    return datasets[0] if reference.paths is None else MultiDataset(*datasets)


class _DatasetRef(BaseModel):
    """Store reference a recipe names one dataset by: one path, or a composition's."""

    path: Annotated[
        str | None,
        Field(default=None, description="Filesystem path or URI of the store to read."),
    ] = None
    paths: Annotated[
        list[str] | None,
        Field(
            default=None,
            min_length=1,
            description="Stores a MultiDataset concatenates, in global index order.",
        ),
    ] = None
    device: Annotated[
        str,
        Field(
            default="cpu",
            description="Device the dataset collates the rows it serves onto.",
        ),
    ] = "cpu"

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_one_reference(self) -> _DatasetRef:
        """Require exactly one of ``path`` and ``paths``."""
        if (self.path is None) == (self.paths is None):
            raise ValueError(
                "A dataset reference names either one store under path or the "
                f"stores of a composition under paths; got path={self.path!r}, "
                f"paths={self.paths!r}."
            )
        return self


class _InitialStructuresSpec(BaseModel):
    """Recipe block a :class:`InitialStructures` is rebuilt from.

    Validating the block before anything is opened refuses a budget that is
    not a positive count and a misspelled setting where a recipe is read rather
    than inside the run it describes — a misspelling in particular, since a
    source is unbudgeted by default and one that never reached a field silently
    generates under no budget at all.
    """

    dataset: Annotated[
        _DatasetRef,
        Field(description="Store the initial structures are read from."),
    ]
    max_atoms: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Total atoms the initial batch may hold.",
        ),
    ] = None
    max_edges: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Total stored edges the initial batch may hold.",
        ),
    ] = None
    max_batch_size: Annotated[
        int | None,
        Field(
            default=None,
            gt=0,
            description="Total structures the initial batch may hold.",
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

    A :class:`~nvalchemi.dynamics.FusedStage` keeps its integrators in
    ``sub_stages`` and a pipeline keeps its stages in ``stages``, so anything
    read off the root alone misses the propagator actually running. Nodes are
    compared by identity, because one integrator reached through two sub-stages
    is a single propagator.

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


def _required_structure_fields(dynamics: BaseDynamics) -> tuple[str, ...]:
    """Return the batch fields *dynamics* updates in place from its first step.

    A propagator primes its model outputs — ``BEFORE_COMPUTE``, ``compute``,
    ``AFTER_COMPUTE`` — before its first ``pre_update``, so the fields its
    ``__needs_keys__`` outputs land in need not be on the initial batch.
    Whatever it updates in place has to be — its ``__provides_keys__`` other
    than ``positions`` — plus ``atomic_masses`` for a propagator carrying
    momentum.

    Parameters
    ----------
    dynamics : BaseDynamics
        Propagator the initial structures are propagated by.

    Returns
    -------
    tuple[str, ...]
        Sorted batch field names the initial structures have to carry.
    """
    fields = dynamics.__provides_keys__ - {"positions"}
    if "velocities" in fields:
        fields.add("atomic_masses")
    return tuple(sorted(fields))


def _check_structure_fields(state: Batch, dynamics: BaseDynamics) -> None:
    """Reject an initial batch the propagator cannot take its first step from.

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
        If *state* is missing a field *dynamics* updates in place from its
        first step.
    """
    missing = [
        field for field in _required_structure_fields(dynamics) if field not in state
    ]
    if not missing:
        return
    raise ValueError(
        f"Initial structures must carry the fields {type(dynamics).__name__} "
        f"propagates from; got missing {missing!r}. It primes the model outputs "
        f"of __needs_keys__={sorted(dynamics.__needs_keys__)!r} itself before "
        f"its first step, but updates "
        f"__provides_keys__={sorted(dynamics.__provides_keys__)!r} in place from "
        "what the structures carry, so an initial structure has to arrive with "
        "all of those — AtomicData fills velocities and atomic_masses in itself "
        "unless a store dropped them, and a cell has to be carried because "
        "nothing fills that in for an aperiodic structure."
    )


def _check_structure_status(state: Batch, criterion: ConvergenceHook) -> None:
    """Reject a criterion that migrates off a status no initial structure holds.

    :meth:`~nvalchemi.dynamics.base.ConvergenceHook.__call__` migrates only the
    graphs sitting on its ``source_status``, so a criterion aimed at another one
    leaves the lifecycle inert: nothing freezes, nothing graduates, and nothing
    warns, because there is no exhaustion to warn about.

    Parameters
    ----------
    state : Batch
        Initial batch, already stamped with the run's own bookkeeping.
    criterion : ConvergenceHook
        Criterion driving the trajectory lifecycle.

    Raises
    ------
    ValueError
        If *state* carries no ``status`` column, or if no graph of it carries
        the criterion's ``source_status``.
    """
    if "status" not in state:
        raise ValueError(
            "A relaxation lifecycle graduates structures on the status column the "
            "initial batch carries, and this one carries none; an "
            "InitialStructuresSource driving a lifecycle stamps status zeros and "
            "system_ids on the batch initial_batch returns, as InitialStructures "
            "does."
        )
    statuses = sorted({int(value) for value in state["status"].view(-1).tolist()})
    if criterion.source_status in statuses:
        return
    raise ValueError(
        "A converged graph migrates off the status its initial structure "
        "carries, and the run stamps that status itself rather than reading it "
        f"from the structures; got source_status={criterion.source_status!r} "
        f"against initial statuses {statuses!r}, so nothing would ever freeze or "
        "graduate. Pass source_status=0, or pass the threshold itself as fmax "
        "and let the shorthand wire it up."
    )


class FitPolicy(Protocol):
    """Decide whether the batch being drawn still fits once a candidate joins it.

    Called by :meth:`InitialStructures.draw` with the atom and edge totals the drawn
    structures would hold with the candidate included, so a policy is a
    stateless predicate over running totals: :class:`WithinBudget` bounds them,
    and a memory estimate or any other axis is one more class of this shape.
    """

    def __call__(self, num_atoms: int, num_edges: int) -> bool:
        """Return whether a drawn batch totaling *num_atoms* and *num_edges* fits."""
        ...


@dataclasses.dataclass(frozen=True)
class WithinBudget:
    """Fit policy admitting a batch while its totals stay within the given bounds.

    Parameters
    ----------
    atoms : int | None, optional
        Total atoms the drawn batch may hold. Default ``None`` (unbounded).
    edges : int | None, optional
        Total stored edges the drawn batch may hold. Default ``None``
        (unbounded). The edge count a dataset reports is whatever it stored,
        not the neighbor list a propagator rebuilds every step, so bound it only
        when the stored count is the one that matters.

    Examples
    --------
    >>> from nvalchemi.training.distillation import WithinBudget
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
class InitialStructuresSource(Protocol):
    """Structures a segment loop starts its trajectories from, behind one cursor.

    These are the members the loop reads, so an object providing them drives
    the loop directly: :meth:`probe` hands the construction-time checks one
    row; :meth:`shard` narrows the source to the rows one rank owns and reopens
    the cursor; :meth:`initial_batch` builds the batch the first segment
    propagates from; :meth:`draw` serves the structures a backfill starts fresh
    trajectories from; :attr:`exhausted` reports a cursor with nothing left;
    and :meth:`state_dict` / :meth:`load_state_dict` carry the cursor through a
    restart. :class:`InitialStructures` is the reference implementation, over a
    dataset. ``to_spec_dict`` / ``from_spec_dict`` are not part of the
    protocol: a recipe names a source through them, and a streaming source
    with no stable cursor position to serialize leaves them out and stays
    runtime-only.

    Examples
    --------
    >>> from nvalchemi.training.distillation import InitialStructuresSource
    >>> isinstance(InitialStructures(dataset), InitialStructuresSource)  # doctest: +SKIP
    True
    """

    @property
    def exhausted(self) -> bool:
        """Whether the cursor has no structure left to hand out."""
        ...

    def shard(self, rank: int, world_size: int) -> None:
        """Narrow the source to the rows *rank* of *world_size* owns and reopen the cursor."""
        ...

    def probe(self) -> Batch:
        """Return one row as the one-graph batch the loop would propagate it as."""
        ...

    def initial_batch(self) -> Batch:
        """Return the batch the first segment propagates from, advancing the cursor.

        A source driving a relaxation lifecycle stamps the batch with ``status``
        zeros and ``system_id`` numbers, as :class:`InitialStructures` does.
        """
        ...

    def draw(
        self,
        *,
        limit: int | None = None,
        fits: FitPolicy | None = None,
        on_miss: Literal["stop", "skip"] = "stop",
    ) -> list[AtomicData]:
        """Serve the next structures from the cursor while they pass *fits*."""
        ...

    def state_dict(self) -> dict[str, Any]:
        """Return the position a restart resumes this source from."""
        ...

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Resume this source at the position *state* recorded."""
        ...


class InitialStructures:
    """Initial structures of a segment loop, served in order from one cursor.

    The reference :class:`InitialStructuresSource`, over a dataset. A run reads
    its initial structures to build the batch the first segment
    propagates from, and a trajectory lifecycle layered on top draws from them
    again for every trajectory it graduates and backfills; both go through the
    one cursor here, so no structure is propagated twice within one pass over
    the rows. An *unbudgeted* source — what a bare dataset is coerced into —
    seeds every row it owns as one batch, so the trajectory count is the
    dataset's; a *budgeted* one packs the initial batch while structures fit
    and leaves the remainder in cursor order for :meth:`draw`. Both are one
    :meth:`draw` call under a :class:`WithinBudget` policy with
    ``on_miss="stop"``, while a backfill filling the room a graduation freed
    passes its own policy with ``on_miss="skip"``.

    :meth:`shard` narrows the source to the rows one rank owns, dealt strided
    and unpadded so the shards are disjoint and no structure is propagated or
    billed to the teacher twice; the cursor counts positions in :attr:`rows`.
    A ``system_id`` is not a position — ids number the trajectories the run
    has started, past any structure a policy passed over — so
    :attr:`next_system_id` is tracked separately from :attr:`cursor`. Under
    ``recycle`` the cursor wraps to the front of the shard instead of reporting
    the source exhausted, ids keep climbing, and one :meth:`draw` reaches every
    row at most once, so two copies of one structure never enter a batch
    together and relax into duplicate frames.

    Parameters
    ----------
    dataset : BatchDatasetProtocol
        Structures, indexed in the order they are served.
    max_atoms : int | None, optional
        Total atoms the initial batch may hold. Default ``None``, which seeds
        every row this source owns.
    max_edges : int | None, optional
        Total stored edges the initial batch may hold. Default ``None``.
    max_batch_size : int | None, optional
        Total structures the initial batch may hold. Default ``None``.
    recycle : bool, optional
        Whether a cursor at the end of the shard wraps to its front instead of
        reporting the source exhausted. Default ``False``.

    Raises
    ------
    ValueError
        If a budget is set and not positive.

    Examples
    --------
    >>> from nvalchemi.training.distillation import InitialStructures, WithinBudget
    >>> structures = InitialStructures(dataset, max_atoms=10_000)  # doctest: +SKIP
    >>> state = structures.initial_batch()  # doctest: +SKIP
    >>> fresh = structures.draw(limit=2, fits=WithinBudget(atoms=64))  # doctest: +SKIP
    >>> endless = InitialStructures(dataset, recycle=True)  # doctest: +SKIP
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
                    f"InitialStructures {name} bounds a batch and must be positive "
                    f"when set; got {value!r}. Leave it None to seed every row."
                )
        self.dataset = dataset
        self.max_atoms = max_atoms
        self.max_edges = max_edges
        self.max_batch_size = max_batch_size
        self.recycle = recycle
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
    def next_system_id(self) -> int:
        """``system_id`` the next structure handed out is stamped with."""
        return self._next_system_id

    @property
    def wraps(self) -> int:
        """Times a recycling cursor has wrapped to the front of the shard."""
        return self._wraps

    @property
    def exhausted(self) -> bool:
        """Whether the shard has no structure left to hand out."""
        return not self.recycle and self._cursor >= len(self._rows)

    def shard(self, rank: int, world_size: int) -> None:
        """Narrow this source to the rows rank *rank* of *world_size* owns.

        Rows are dealt out strided — rank ``r`` takes every ``world_size``-th
        structure from offset ``r`` — so the shards are disjoint, cover the
        dataset, and differ by at most one structure. The deal balances the count,
        not the work, so sort the dataset by atom count when structures differ
        widely in size. It is unpadded, since a padded structure would be
        propagated twice and billed to the teacher twice. The cursor and the next
        ``system_id`` are reset, so installing a shard on a source that has
        already run reseeds it rather than resuming it.

        Parameters
        ----------
        rank : int
            Global rank claiming a shard.
        world_size : int
            Ranks the dataset is dealt across. A single-rank run gets the whole
            dataset, unchanged.

        Raises
        ------
        ValueError
            If *world_size* is not positive or *rank* falls outside it.
        """
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(
                "A shard is dealt to one rank of a world, so the rank has "
                f"to name a position in it; got rank={rank!r} of "
                f"world_size={world_size!r}."
            )
        self._rank = rank
        self._world_size = world_size
        self._rows = tuple(range(rank, len(self.dataset), world_size))
        self._cursor = 0
        self._wraps = 0
        self._next_system_id = 0

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
                "InitialStructures has to hold at least one structure; got a "
                f"{type(self.dataset).__name__} of length "
                f"{len(self.dataset)!r} sharded to no rows."
            )
        return self.dataset.load_batches([[self._rows[0]]])[0]

    def initial_batch(self) -> Batch:
        """Return the batch the first segment propagates from, advancing the cursor.

        The batch enters the run carrying none of the propagator's bookkeeping, so
        this source installs its own: a structure loaded from a store a dynamics
        sink filled arrives holding the ``status`` it graduated with, which
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.step` would freeze at
        ``exit_status`` for a segment that moves nothing.

        Returns
        -------
        Batch
            Initial batch, stamped with clean bookkeeping and numbered from
            :attr:`next_system_id`.

        Raises
        ------
        ValueError
            If the cursor has nothing left to seed from, or if the first structure
            at the cursor is larger than the declared budget.
        """
        budget = WithinBudget(atoms=self.max_atoms, edges=self.max_edges)
        rows = self._scan_rows(
            limit=self.max_batch_size,
            fits=None if budget == WithinBudget() else budget,
            on_miss="stop",
        )
        if not rows:
            raise ValueError(
                "A segment loop has to propagate something; got no "
                f"structure at cursor {self._cursor!r} of {len(self._rows)!r} "
                f"rows fitting max_atoms={self.max_atoms!r}, "
                f"max_edges={self.max_edges!r}, and "
                f"max_batch_size={self.max_batch_size!r}. Widen the budget, or "
                "pass a dataset holding a structure that fits it."
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
        """Serve the next structures from the cursor while they pass *fits*.

        Parameters
        ----------
        limit : int | None, optional
            Most structures to serve. Default ``None`` (the rest of the shard).
        fits : FitPolicy | None, optional
            Policy called with the atom and edge totals the drawn structures
            would hold with each candidate included. Default ``None`` (every
            structure fits).
        on_miss : {"stop", "skip"}, optional
            What a candidate that does not fit does to the scan. ``"stop"``
            ends the draw and leaves the cursor on it, which is how an initial
            batch is packed; ``"skip"`` passes over it and goes on, which is how
            a backfill fills the room a graduation freed without one oversized
            structure starving every refill behind it. Default ``"stop"``.

        Returns
        -------
        list[AtomicData]
            Structures in cursor order, each stamped with its own
            ``system_id``. Empty once the shard is exhausted, or once the first
            candidate misses under ``on_miss="stop"``. A recycling cursor wraps
            to the front of the shard instead, and one call reaches every row
            at most once.
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
        """Return the position a restart resumes this source from.

        Returns
        -------
        dict[str, int]
            The cursor, the wraps behind it, the next ``system_id``, and the
            shard all three were counted in. The dataset, the declared budgets,
            and ``recycle`` are configuration a recipe carries, not state, and
            are left out.
        """
        return {
            "cursor": self._cursor,
            "wraps": self._wraps,
            "next_system_id": self._next_system_id,
            "rank": self._rank,
            "world_size": self._world_size,
        }

    def load_state_dict(self, state: Mapping[str, int]) -> None:
        """Resume this source at the cursor *state* recorded.

        Parameters
        ----------
        state : Mapping[str, int]
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
                "The restart bundle's cursor was written for rank "
                f"{rank!r} of {world_size!r}; this rank is {self._rank!r} of "
                f"{self._world_size!r}. Restart on the world that wrote it, or "
                "reseed with a cold buffer."
            )
        self._cursor = int(state["cursor"])
        self._wraps = int(state["wraps"])
        self._next_system_id = int(state["next_system_id"])

    def to_spec_dict(self) -> dict[str, Any]:
        """Return the JSON-ready reference a recipe names this source by.

        Returns
        -------
        dict[str, Any]
            The store the structures are read from, the budgets the caller
            declared, and ``recycle``. The cursor is state and belongs to a
            restart bundle instead, and the rank shard is a launcher fact that
            belongs to neither.

        Raises
        ------
        ValueError
            If the dataset holds its samples in memory, which no recipe
            can name.
        """
        return {
            "dataset": _dataset_spec_dict(
                self.dataset, "OnPolicyConfig.initial_structures"
            ),
            "max_atoms": self.max_atoms,
            "max_edges": self.max_edges,
            "max_batch_size": self.max_batch_size,
            "recycle": self.recycle,
        }

    @classmethod
    def from_spec_dict(cls, spec: Mapping[str, Any]) -> InitialStructures:
        """Rebuild the source :meth:`to_spec_dict` described.

        Parameters
        ----------
        spec : Mapping[str, Any]
            Reference produced by :meth:`to_spec_dict`.

        Returns
        -------
        InitialStructures
            Source over the referenced store, with a cursor at its first row.

        Raises
        ------
        pydantic.ValidationError
            If *spec* carries a key no source takes, names no store to read
            the structures from, or gives a budget that is not a positive count. It
            derives from :class:`ValueError`, so a caller that already reports
            a bad recipe reports this one the same way.
        """
        validated = _InitialStructuresSpec.model_validate(spec)
        return cls(
            _dataset_from_spec_dict(validated.dataset.model_dump()),
            max_atoms=validated.max_atoms,
            max_edges=validated.max_edges,
            max_batch_size=validated.max_batch_size,
            recycle=validated.recycle,
        )

    def _scan_rows(
        self,
        *,
        limit: int | None,
        fits: FitPolicy | None,
        on_miss: Literal["stop", "skip"],
    ) -> list[int]:
        """Advance the cursor and return the rows the policy admitted.

        The scan reaches every row of the shard at most once, so a recycling
        cursor that wrapped mid-scan never serves a structure it already served
        in the same call.
        """
        rows: list[int] = []
        atoms = edges = 0
        scanned = 0
        while scanned < len(self._rows) and (limit is None or len(rows) < limit):
            if self._cursor >= len(self._rows):
                if not self.recycle:
                    break
                self._cursor = 0
                self._wraps += 1
            index = self._rows[self._cursor]
            scanned += 1
            if fits is not None:
                num_atoms, num_edges = self.dataset.get_metadata(index)
                if not fits(atoms + num_atoms, edges + num_edges):
                    if on_miss == "stop":
                        break
                    self._cursor += 1
                    continue
                atoms += num_atoms
                edges += num_edges
            rows.append(index)
            self._cursor += 1
        return rows

    def _stamp_bookkeeping(self, state: Batch) -> None:
        """Give *state* the graph-level fields a trajectory lifecycle maintains.

        ``status`` is what a status-migrating
        :class:`~nvalchemi.dynamics.base.ConvergenceHook` writes and a
        lifecycle graduates on, and ``system_id`` numbers the structures the way
        a backfill continues numbering them.
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
