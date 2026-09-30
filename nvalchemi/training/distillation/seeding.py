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

from collections.abc import Mapping
from typing import Annotated, Any

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

    model_config = ConfigDict(extra="forbid")


class InitialStructures(OrderedStructureSampler):
    """An :class:`~nvalchemi.dynamics.OrderedStructureSampler` that a recipe can name.

    The sampler is the loop's reference :class:`InitialStructuresSource`. This
    subclass adds :meth:`to_spec_dict` and :meth:`from_spec_dict`. They name
    the sampler by the store its dataset reads and by the budgets declared on
    it, so :class:`~nvalchemi.training.distillation.OnPolicyConfig` can be
    written to a recipe and rebuilt from one. A streaming source has no stable
    position to serialize, so it omits both methods and stays runtime-only.
    Writing a recipe from a config that holds such a source raises an error
    that names it.

    Examples
    --------
    >>> from nvalchemi.training.distillation import InitialStructures, WithinBudget
    >>> structures = InitialStructures(dataset, max_atoms=10_000)  # doctest: +SKIP
    >>> state = structures.initial_batch()  # doctest: +SKIP
    >>> fresh = structures.draw(limit=2, fits=WithinBudget(atoms=64))  # doctest: +SKIP
    >>> rebuilt = InitialStructures.from_spec_dict(structures.to_spec_dict())  # doctest: +SKIP
    """

    def to_spec_dict(self) -> dict[str, Any]:
        """Return the JSON-ready reference by which a recipe names this source.

        Returns
        -------
        dict[str, Any]
            The store the structures are read from and the budgets the caller
            declared. The position is run state and belongs in a restart
            bundle instead. The rank shard is set by the launcher and belongs
            in neither.

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
        )
