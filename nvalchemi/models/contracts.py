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
from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class NeighborRequirement(BaseModel):
    """Semantic neighbor requirement advertised by a potential profile.

    Describes *how* a potential needs neighbor-list data.  ``source``
    controls who is responsible for building neighbors:

    - ``"none"`` -- the potential does not use neighbors.
    - ``"internal"`` -- the potential builds its own neighbors; only
      an optional ``cutoff`` may be advertised.
    - ``"external"`` -- the composite pipeline must supply an explicit
      neighbor-list step that satisfies the declared ``format``,
      ``cutoff``, and ``name``.

    Attributes
    ----------
    source : {"none", "internal", "external"}, default "none"
        Who is responsible for building the neighbor list.
    cutoff : float or None, default None
        Interaction cutoff radius (assumed Angstrom).
    format : {"coo", "matrix"} or None, default None
        Required neighbor-list storage layout.
    half_list : bool or None, default None
        Whether the neighbor list should be half (``True``) or full.
    name : str or None, default None
        Logical name of the neighbor list (e.g. ``"default"``).
    """

    source: Annotated[
        Literal["none", "internal", "external"],
        Field(description="Who is responsible for building the neighbor list."),
    ] = "none"
    cutoff: Annotated[
        float | None,
        Field(description="Interaction cutoff radius (assumed Angstrom)."),
    ] = None
    format: Annotated[
        Literal["coo", "matrix"] | None,
        Field(description="Required neighbor-list storage layout."),
    ] = None
    half_list: Annotated[
        bool | None,
        Field(description="Whether the neighbor list should be half (True) or full."),
    ] = None
    name: Annotated[
        str | None,
        Field(description="Logical name of the neighbor list."),
    ] = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _validate_requirement(self) -> Self:
        """Validate the semantic consistency of the neighbor requirement."""

        if self.source == "none":
            return self
        if self.source == "internal":
            if self.format is not None or self.half_list is not None or self.name is not None:
                raise ValueError(
                    "Internal neighbor requirements may only advertise an optional cutoff."
                )
            return self
        if self.format is None:
            raise ValueError("External neighbor requirements must declare a format.")
        if self.name is None:
            raise ValueError("External neighbor requirements must declare a name.")
        return self


class StepProfile(BaseModel):
    """Resolved immutable runtime contract for one configured step instance.

    A profile is the frozen, instance-specific snapshot of what a step
    can consume and produce.  It is constructed once during ``__init__``
    and remains constant for the lifetime of the step.

    Attributes
    ----------
    required_inputs : frozenset[str]
        Batch or result keys the step *must* receive.
    optional_inputs : frozenset[str]
        Batch or result keys the step *may* use when available.
    result_keys : frozenset[str]
        All output keys the step is capable of producing.
    default_result_keys : frozenset[str]
        Subset of ``result_keys`` produced when no explicit outputs
        are requested.  Must be a subset of ``result_keys``.
    additive_result_keys : frozenset[str]
        Subset of ``result_keys`` whose values are *summed* when
        multiple steps produce the same key.  Must be a subset of
        ``result_keys``.
    """

    required_inputs: Annotated[
        frozenset[str],
        Field(description="Batch or result keys the step must receive."),
    ] = frozenset()
    optional_inputs: Annotated[
        frozenset[str],
        Field(description="Batch or result keys the step may use when available."),
    ] = frozenset()
    result_keys: Annotated[
        frozenset[str],
        Field(description="All output keys the step is capable of producing."),
    ] = frozenset()
    default_result_keys: Annotated[
        frozenset[str],
        Field(description="Output keys produced when no explicit outputs are requested."),
    ] = frozenset()
    additive_result_keys: Annotated[
        frozenset[str],
        Field(description="Output keys whose values are summed across multiple producers."),
    ] = frozenset()

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _validate_result_keys(self) -> Self:
        """Validate default and additive result-key subsets."""

        if not (self.default_result_keys <= self.result_keys):
            raise ValueError(
                f"default_result_keys {sorted(self.default_result_keys)} must be a "
                f"subset of result_keys {sorted(self.result_keys)}."
            )
        if not (self.additive_result_keys <= self.result_keys):
            raise ValueError(
                f"additive_result_keys {sorted(self.additive_result_keys)} must be a "
                f"subset of result_keys {sorted(self.result_keys)}."
            )
        return self


class StepCard(StepProfile):
    """Class-level default declaration for one step type.

    A card captures the *class-level* I/O contract.  Individual instances
    may customise the contract via ``to_profile(**updates)``.

    Attributes
    ----------
    parameterized_by : frozenset[str]
        Config field names that may change the resolved profile
        (e.g. ``{"neighbor_list_name"}``).
    """

    parameterized_by: Annotated[
        frozenset[str],
        Field(description="Config field names that may change the resolved profile."),
    ] = frozenset()

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_profile(self, **updates: object) -> StepProfile:
        """Resolve this card into an instance profile.

        Parameters
        ----------
        **updates
            Field overrides applied on top of the card defaults.

        Returns
        -------
        StepProfile
            Frozen instance-level contract.
        """

        data = self.model_dump(exclude={"parameterized_by"})
        data.update(updates)
        return StepProfile(**data)


class PotentialProfile(StepProfile):
    """Resolved immutable runtime contract for one configured potential.

    Extends :class:`StepProfile` with potential-specific fields for
    boundary-mode support, neighbor requirements, and gradient tracking.

    Attributes
    ----------
    boundary_modes : frozenset[{"non_pbc", "pbc"}]
        Supported boundary conditions.  Must contain at least one mode.
    neighbor_requirement : NeighborRequirement
        Describes how this potential consumes neighbor-list data.
    gradient_setup_targets : frozenset[str]
        Derivative target names (e.g. ``{"positions", "cell_scaling"}``)
        this potential sets up during the ``prepare()`` phase.
    """

    boundary_modes: Annotated[
        frozenset[Literal["non_pbc", "pbc"]],
        Field(description="Supported boundary conditions."),
    ] = frozenset({"non_pbc", "pbc"})
    neighbor_requirement: Annotated[
        NeighborRequirement,
        Field(description="How this potential consumes neighbor-list data."),
    ] = Field(default_factory=NeighborRequirement)
    gradient_setup_targets: Annotated[
        frozenset[str],
        Field(description="Derivative targets set up during prepare()."),
    ] = frozenset()

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _validate_potential_profile(self) -> Self:
        """Validate the potential-specific contract fields."""

        if not self.boundary_modes:
            raise ValueError("boundary_modes must contain at least one mode.")
        return self


class PotentialCard(PotentialProfile):
    """Class-level default declaration for one potential type.

    Extends :class:`PotentialProfile` with the ``parameterized_by``
    metadata used by the card → profile resolution step.

    Attributes
    ----------
    parameterized_by : frozenset[str]
        Config field names that may change the resolved profile.
    """

    parameterized_by: Annotated[
        frozenset[str],
        Field(description="Config field names that may change the resolved profile."),
    ] = frozenset()

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_profile(self, **updates: object) -> PotentialProfile:
        """Resolve this card into an instance potential profile.

        Parameters
        ----------
        **updates
            Field overrides applied on top of the card defaults.

        Returns
        -------
        PotentialProfile
            Frozen instance-level potential contract.
        """

        data = self.model_dump(exclude={"parameterized_by"})
        data.update(updates)
        return PotentialProfile(**data)


class MLIPPotentialCard(PotentialCard):
    """Card defaults for MLIPs that support autograd-derived forces and stresses.

    Pre-sets ``gradient_setup_targets`` to ``{"positions", "cell_scaling"}``
    and limits ``default_result_keys`` / ``additive_result_keys`` to
    ``{"energies"}``.  Not every MLIP must use this subclass; MLIPs that
    do not participate in the split-gradient architecture should use
    :class:`PotentialCard` directly.

    Attributes
    ----------
    gradient_setup_targets : frozenset[str]
        Defaults to ``{"positions", "cell_scaling"}``.
    default_result_keys : frozenset[str]
        Defaults to ``{"energies"}``.
    additive_result_keys : frozenset[str]
        Defaults to ``{"energies"}``.
    """

    gradient_setup_targets: frozenset[str] = frozenset({"positions", "cell_scaling"})
    default_result_keys: frozenset[str] = frozenset({"energies"})
    additive_result_keys: frozenset[str] = frozenset({"energies"})

    model_config = ConfigDict(extra="forbid", frozen=True)


class NeighborListProfile(StepProfile):
    """Resolved immutable runtime contract for one configured neighbor builder.

    Extends :class:`StepProfile` with the neighbor-list-specific
    parameters that describe the output format and cutoff.

    Attributes
    ----------
    neighbor_list_name : str, default "default"
        Logical name used to namespace result keys
        (e.g. ``"neighbor_lists.default.neighbor_matrix"``).
    cutoff : float or None, default None
        Interaction cutoff radius (assumed Angstrom).
    format : {"coo", "matrix"} or None, default None
        Output storage layout.
    half_list : bool or None, default None
        Whether the builder produces a half-list or full-list.
    """

    neighbor_list_name: Annotated[
        str,
        Field(description="Logical name used to namespace result keys."),
    ] = "default"
    cutoff: Annotated[
        float | None,
        Field(description="Interaction cutoff radius (assumed Angstrom)."),
    ] = None
    format: Annotated[
        Literal["coo", "matrix"] | None,
        Field(description="Output storage layout."),
    ] = None
    half_list: Annotated[
        bool | None,
        Field(description="Whether the builder produces a half-list or full-list."),
    ] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class NeighborListCard(NeighborListProfile):
    """Class-level default declaration for one neighbor-builder type.

    Attributes
    ----------
    parameterized_by : frozenset[str]
        Config field names that may change the resolved profile.
    """

    parameterized_by: Annotated[
        frozenset[str],
        Field(description="Config field names that may change the resolved profile."),
    ] = frozenset()

    model_config = ConfigDict(extra="forbid", frozen=True)

    def to_profile(self, **updates: object) -> NeighborListProfile:
        """Resolve this card into an instance neighbor-list profile.

        Parameters
        ----------
        **updates
            Field overrides applied on top of the card defaults.

        Returns
        -------
        NeighborListProfile
            Frozen instance-level neighbor-list contract.
        """

        data = self.model_dump(exclude={"parameterized_by"})
        data.update(updates)
        return NeighborListProfile(**data)


class PipelineContract(BaseModel):
    """Frozen contract describing an assembled explicit pipeline.

    Built automatically by :class:`CompositeCalculator` from its ordered
    sequence of steps.

    Attributes
    ----------
    steps : tuple[str, ...]
        Ordered step names in insertion order.
    result_keys : frozenset[str]
        Union of all result keys across all steps.
    additive_result_keys : frozenset[str]
        Union of all additive result keys across all steps.
    """

    steps: Annotated[
        tuple[str, ...],
        Field(description="Ordered step names in insertion order."),
    ]
    result_keys: Annotated[
        frozenset[str],
        Field(description="Union of all result keys across all steps."),
    ] = frozenset()
    additive_result_keys: Annotated[
        frozenset[str],
        Field(description="Union of all additive result keys across all steps."),
    ] = frozenset()

    model_config = ConfigDict(extra="forbid", frozen=True)
