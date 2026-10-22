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
"""Initial structures of an on-policy segment loop, in a form a recipe can name.

The segment loop seeds its trajectories from an
:class:`~nvalchemi.dynamics.OrderedStructureSampler`. This module adds the one
thing a recipe needs from that sampler: a spec round-trip through the store
the structures are read from. It also keeps the loop's historical names
importable.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from nvalchemi.dynamics.structure_sampler import (
    FitPolicy,
    OrderedStructureSampler,
    StructureSource,
    WithinBudget,
)
from nvalchemi.training._spec_utils import (
    DatasetRef,
    dataset_from_spec_dict,
    dataset_spec_dict,
)

if TYPE_CHECKING:
    from nvalchemi.data import AtomicData
    from nvalchemi.data.datapipes.dataset import BatchDatasetProtocol

__all__ = ["FitPolicy", "InitialStructures", "InitialStructuresSource", "WithinBudget"]

InitialStructuresSource = StructureSource
"""The loop's historical name for :class:`~nvalchemi.dynamics.StructureSource`."""

_IN_MEMORY_REMEDY = (
    "Write the samples to a store with "
    "nvalchemi.training.distillation.label_dataset (or an AtomicDataZarrWriter) "
    "and point the recipe at that path, or re-supply "
    "OnPolicyConfig.initial_structures at construction."
)
"""Final sentence of the in-memory error, naming the distillation writer."""


class _InitialStructuresSpec(BaseModel):
    """Recipe block that an :class:`InitialStructures` is rebuilt from.

    The block is validated before anything is opened. A budget that is not a
    positive count, or a misspelled setting, is therefore rejected when the
    recipe is read rather than inside the run it describes. This matters most
    for a misspelling: a source is unbudgeted by default, so a misspelled
    budget never reaches its field and the run silently generates with no
    budget at all.
    """

    dataset: Annotated[
        DatasetRef,
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
                "Whether the source starts again from its first row once the "
                "last one has been served, instead of reporting itself exhausted."
            ),
        ),
    ] = False

    model_config = ConfigDict(extra="forbid")


class InitialStructures(OrderedStructureSampler):
    """An :class:`~nvalchemi.dynamics.OrderedStructureSampler` that a recipe can name.

    The sampler is the loop's reference :class:`InitialStructuresSource`. This
    subclass adds :meth:`to_spec_dict` and :meth:`from_spec_dict`. They
    describe the sampler by the store its dataset reads, the budgets declared
    on it, and whether it recycles its rows. An
    :class:`~nvalchemi.training.distillation.OnPolicyConfig` holding it can
    therefore be written to a recipe and rebuilt from one. A streaming source
    has no stable position to serialize, so it omits both methods and stays
    runtime-only. Writing a recipe from a config that holds such a source
    raises an error that names it.

    Examples
    --------
    >>> from nvalchemi.training.distillation import InitialStructures, WithinBudget
    >>> structures = InitialStructures(dataset, max_atoms=10_000)  # doctest: +SKIP
    >>> state = structures.initial_batch()  # doctest: +SKIP
    >>> fresh = structures.draw(limit=2, fits=WithinBudget(atoms=64))  # doctest: +SKIP
    >>> rebuilt = InitialStructures.from_spec_dict(structures.to_spec_dict())  # doctest: +SKIP

    A :meth:`draw` with ``on_miss="stop"`` — the backfill under a trajectory
    lifetime — stops at a row that does not fit the room left, since the room
    grows back to the declared budget as trajectories retire. A row larger
    than the declared ``max_atoms`` or ``max_edges`` on its own never fits
    any room, so it is passed over with a warning instead of holding the
    position forever; :attr:`oversized_rows` records it.
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
        """Open the sampler at the first row of *dataset*, with no row passed over."""
        super().__init__(
            dataset,
            max_atoms=max_atoms,
            max_edges=max_edges,
            max_batch_size=max_batch_size,
            recycle=recycle,
        )
        self._oversized_rows: set[int] = set()

    @property
    def oversized_rows(self) -> frozenset[int]:
        """Dataset rows a stopping :meth:`draw` passed over as larger than the budget."""
        return frozenset(self._oversized_rows)

    def draw(
        self,
        *,
        limit: int | None = None,
        fits: FitPolicy | None = None,
        on_miss: Literal["stop", "skip"] = "stop",
    ) -> list[AtomicData]:
        """Serve the next structures while they pass *fits*, past oversized rows.

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
            How the scan treats a candidate that does not fit, as in
            :meth:`~nvalchemi.dynamics.OrderedStructureSampler.draw`. Under
            ``"stop"`` a row larger than the declared budget on its own is
            passed over and warned about once, and never ends the draw.
            Default ``"stop"``.

        Returns
        -------
        list[AtomicData]
            Structures in row order, each stamped with its own ``system_id``.
        """
        if on_miss == "skip" or (self.max_atoms is None and self.max_edges is None):
            return super().draw(limit=limit, fits=fits, on_miss=on_miss)
        drawn: list[AtomicData] = []
        for index in self._scan_past_oversized(limit=limit, fits=fits):
            data, _ = self.dataset[index]
            data.add_system_property(
                "system_id",
                torch.tensor([[self._next_system_id]], dtype=torch.long),
            )
            self._next_system_id += 1
            drawn.append(data)
        return drawn

    def _scan_past_oversized(
        self, *, limit: int | None, fits: FitPolicy | None
    ) -> list[int]:
        """Advance the position like a stopping scan, passing over oversized rows."""
        budget = WithinBudget(atoms=self.max_atoms, edges=self.max_edges)
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
            num_atoms, num_edges = self.dataset.get_metadata(index)
            if not budget(num_atoms, num_edges):
                if index not in self._oversized_rows:
                    self._oversized_rows.add(index)
                    warnings.warn(
                        f"Initial structure at row {index!r} holds {num_atoms!r} "
                        f"atoms and {num_edges!r} edges, more than max_atoms="
                        f"{self.max_atoms!r} and max_edges={self.max_edges!r} "
                        "allow on their own, so no backfill can ever fit it. It "
                        "is passed over. Widen the budget to propagate it.",
                        UserWarning,
                        stacklevel=3,
                    )
                self._next_row += 1
                continue
            if fits is not None and not fits(atoms + num_atoms, edges + num_edges):
                break
            atoms += num_atoms
            edges += num_edges
            rows.append(index)
            self._next_row += 1
        return rows

    def to_spec_dict(self) -> dict[str, Any]:
        """Return the JSON-ready reference by which a recipe names this source.

        Returns
        -------
        dict[str, Any]
            The store the structures are read from, the budgets the caller
            declared, and ``recycle``. The position is run state and belongs
            in a restart bundle instead. The rank shard is set by the launcher
            and belongs in neither.

        Raises
        ------
        ValueError
            If the dataset holds its samples in memory, which no recipe
            can name.
        """
        return {
            "dataset": dataset_spec_dict(
                self.dataset,
                field="OnPolicyConfig.initial_structures",
                remedy=_IN_MEMORY_REMEDY,
            ),
            "max_atoms": self.max_atoms,
            "max_edges": self.max_edges,
            "max_batch_size": self.max_batch_size,
            "recycle": self.recycle,
        }

    @classmethod
    def from_spec_dict(cls, spec: Mapping[str, Any]) -> InitialStructures:
        """Rebuild the source that :meth:`to_spec_dict` described.

        Parameters
        ----------
        spec : Mapping[str, Any]
            Reference produced by :meth:`to_spec_dict`.

        Returns
        -------
        InitialStructures
            Source over the referenced store, opened at its first row.

        Raises
        ------
        pydantic.ValidationError
            If *spec* carries a key no source accepts, names no store to read
            the structures from, or gives a budget that is not a positive
            count. It derives from :class:`ValueError`, so a caller that
            already reports a bad recipe reports this one the same way.
        """
        validated = _InitialStructuresSpec.model_validate(spec)
        return cls(
            dataset_from_spec_dict(
                validated.dataset.model_dump(),
                field="OnPolicyConfig.initial_structures",
            ),
            max_atoms=validated.max_atoms,
            max_edges=validated.max_edges,
            max_batch_size=validated.max_batch_size,
            recycle=validated.recycle,
        )
