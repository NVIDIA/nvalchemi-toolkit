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

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SHORT_RANGE: str = "short_range"
ELECTROSTATICS: str = "electrostatics"
DISPERSION: str = "dispersion"

MLIP: str = "mlip"
REPULSION: str = "repulsion"
ATOMIC_CHARGES: str = "atomic_charges"
PAIRWISE: str = "pairwise"
IMPLICIT: str = "implicit"


class PhysicalTerm(BaseModel):
    """A physics term advertised by one checkpoint or potential.

    ``kind`` is a broad physical category and ``variant`` is a recognised
    subtype (physical representation, phenomenology, or model family).
    Both are free-form strings — the constants exported from this module
    document the values recognised by the library, but custom values are
    accepted.

    When ``variant`` is ``None`` the term matches any variant of the
    given kind.

    Recognised kind → variant mapping:

    - ``short_range``     → ``mlip``, ``repulsion``
    - ``electrostatics``  → ``atomic_charges``, ``implicit``
    - ``dispersion``      → ``pairwise``, ``implicit``

    Attributes
    ----------
    kind : str
        Broad physical category (e.g. ``"short_range"``).
    variant : str or None, default None
        Recognised subtype within the category.
    """

    kind: Annotated[
        str,
        Field(description="Broad physical category (e.g. 'short_range')."),
    ]
    variant: Annotated[
        str | None,
        Field(description="Recognised subtype within the category."),
    ] = None

    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="after")
    def _validate_term(self) -> Self:
        """Reject empty or whitespace-only term names.

        Raises
        ------
        ValueError
            If ``kind`` is empty or whitespace-only, or if ``variant``
            is provided but empty or whitespace-only.
        """

        if not self.kind.strip():
            raise ValueError("PhysicalTerm.kind must not be empty.")
        if self.variant is not None and not self.variant.strip():
            raise ValueError("PhysicalTerm.variant must not be empty when provided.")
        return self


class CheckpointInfo(BaseModel):
    """Provenance metadata for the resolved checkpoint artifact.

    Attributes
    ----------
    identifier : str or None
        Logical name or path that was used to resolve the checkpoint.
    url : str or None
        Remote URL the checkpoint was downloaded from, if applicable.
    sha256 : str or None
        SHA-256 hex digest of the checkpoint file, if known.
    source : str or None
        How the checkpoint was obtained: ``"registry"``, ``"local_path"``,
        or ``"in_memory"``.
    """

    identifier: Annotated[
        str | None,
        Field(description="Logical name or path used to resolve the checkpoint."),
    ] = None
    url: Annotated[
        str | None,
        Field(description="Remote URL the checkpoint was downloaded from."),
    ] = None
    sha256: Annotated[
        str | None,
        Field(description="SHA-256 hex digest of the checkpoint file."),
    ] = None
    source: Annotated[
        str | None,
        Field(description="How the checkpoint was obtained (e.g. 'registry', 'local_path')."),
    ] = None

    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelCard(BaseModel):
    """Checkpoint-level scientific and provenance metadata for one configured model.

    Attributes
    ----------
    model_family : str or None
        Broad model family name (e.g. ``"mace"``, ``"aimnet2"``).
    model_name : str or None
        Specific checkpoint or model identifier.
    reference_xc_functional : str or None
        Reference exchange-correlation functional, if applicable.
    provided_terms : tuple[PhysicalTerm, ...]
        Physics terms that this checkpoint or potential already covers.
    required_external_terms : tuple[PhysicalTerm, ...]
        Physics terms that must be added externally for intended use.
    optional_external_terms : tuple[PhysicalTerm, ...]
        Physics terms that are commonly added externally but not strictly
        required.
    checkpoint : CheckpointInfo or None
        Provenance metadata for the resolved checkpoint artifact.
    """

    model_family: Annotated[
        str | None,
        Field(description="Broad model family name (e.g. 'mace', 'aimnet2')."),
    ] = None
    model_name: Annotated[
        str | None,
        Field(description="Specific checkpoint or model identifier."),
    ] = None
    reference_xc_functional: Annotated[
        str | None,
        Field(description="Reference exchange-correlation functional."),
    ] = None
    provided_terms: Annotated[
        tuple[PhysicalTerm, ...],
        Field(description="Physics terms this checkpoint already covers."),
    ] = ()
    required_external_terms: Annotated[
        tuple[PhysicalTerm, ...],
        Field(description="Physics terms that must be added externally."),
    ] = ()
    optional_external_terms: Annotated[
        tuple[PhysicalTerm, ...],
        Field(description="Physics terms commonly added externally but not required."),
    ] = ()
    checkpoint: Annotated[
        CheckpointInfo | None,
        Field(description="Provenance metadata for the resolved checkpoint artifact."),
    ] = None

    model_config = ConfigDict(extra="forbid", frozen=True)
