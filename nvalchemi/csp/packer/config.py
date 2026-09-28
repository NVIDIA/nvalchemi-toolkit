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
"""Validated configuration for tensor-based crystal packing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from nvalchemi.csp.symmetry import (
    CrystalSystem,
    csd_space_group_probabilities,
    get_space_group_candidates,
    get_space_group_operation_count,
    is_sohncke_space_group,
)


class PackingConfig(BaseModel):
    """Choose how many trial structures to sample, which crystal symmetries and cell
    sizes to use, and when a trial is accepted.

    ``batch_size`` is the maximum number of candidates active at once.
    ``max_candidates`` optionally limits total candidates generated; it may be
    smaller than a call's requested output target, in which case that call can
    return a short result. The output target and random generator belong to the
    generating-function call.

    Exactly one cell volume mode is required. ``cell_volume_range`` gives an
    absolute conventional-cell volume interval in cubic angstroms, while
    ``cell_volume_scale_range`` scales the formula-unit estimate by ``z``.
    Space-group settings either select a fixed group or filter sampled groups.
    The default relative sampling weights are derived from the Cambridge
    Structural Database (CSD). After compatible groups are filtered, the
    remaining positive weights set their relative probabilities. A supplied
    probability mapping replaces these default weights.

    Parameters
    ----------
    z : int
        Number of formula units in the conventional cell.
    z_prime : int
        Number of formula units in the asymmetric unit; must divide ``z``.
    batch_size : int
        Maximum number of candidates active at once.
    max_candidates : int, optional
        Maximum number of trial structures initialized during a call; ``None``
        leaves this budget unlimited.
    max_steps_per_candidate : int, default=500
        Maximum relaxation updates for one candidate before it expires.
    convergence_check_interval : int, default=10
        Number of relaxation iterations between regular acceptance checks.
        Checks also occur at iteration zero and at candidate expiry.
    overlap_tolerance : float, default=0.05
        Largest positive overlap allowed between atoms in different molecular
        copies, in Å.
    max_step : float, default=0.3
        Upper bound in Å on the small-rotation estimate of per-atom movement
        used to scale one rigid update. A finite rotation can move an atom
        slightly farther.
    step_scale : float, default=0.12
        Scale applied to numerical clash-removal forces and torques; these are
        not physical model forces.
    cell_step_scale : float, default=0.01
        Scale for cell-shape and excess-volume updates; zero disables them.
    max_cell_strain : float, default=0.003
        Limit on a cell update's strain and compression components.
    volume_compression_scale : float, default=1.0e-5
        Strength of compression when a cell exceeds its sampled reference
        volume.
    min_cell_height : float, default=3.8
        Minimum perpendicular cell height for initial sampling, in Å.
        Relaxation separately rejects proposed cells below 2.5 Å.
    max_axis_ratio : float, default=10.0
        Largest cell-axis length ratio allowed during initial sampling.
    cell_oversample_factor : float, default=1.5
        Multiplier for the number of initial cell trials.
    cell_volume_range : tuple[float, float], optional
        Inclusive initial conventional-cell volume range, in Å³.
    cell_volume_scale_range : tuple[float, float], optional
        Initial volume multipliers applied to formula-unit volume times ``z``.
        Supply exactly one of the two volume ranges.
    fixed_space_group : int, optional
        International space-group number to use for every candidate.
    space_group_probabilities : Mapping[int, float], optional
        Unnormalized, nonnegative sampling weights keyed by International
        number. Compatible groups are filtered; the remaining positive weights
        set their relative probabilities. Replaces the default relative
        sampling weights derived from the Cambridge Structural Database (CSD).
    crystal_system : CrystalSystem, optional
        Restrict sampled space groups to this crystal system.
    sohncke_only : bool, default=False
        Restrict sampling to groups whose symmetry operations preserve
        molecular handedness. Use this when symmetry-generated copies must
        preserve a chiral molecule's handedness.

    Examples
    --------
    Use a scale range with a finite candidate budget::

        config = PackingConfig(
            z=4,
            z_prime=1,
            batch_size=16,
            max_candidates=64,
            cell_volume_scale_range=(0.8, 1.2),
            crystal_system=CrystalSystem.MONOCLINIC,
            sohncke_only=True,
        )

    Derive a separately validated configuration for one call::

        shorter_run = config.effective(max_steps_per_candidate=300)
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    z: int = Field(strict=True)
    z_prime: int = Field(strict=True)
    batch_size: int = Field(strict=True)
    max_candidates: int | None = Field(default=None, strict=True)
    max_steps_per_candidate: int = Field(default=500, strict=True)
    convergence_check_interval: int = Field(default=10, strict=True)
    overlap_tolerance: float = 0.05
    max_step: float = 0.3
    step_scale: float = 0.12
    cell_step_scale: float = 0.01
    max_cell_strain: float = 0.003
    volume_compression_scale: float = 1.0e-5
    min_cell_height: float = 3.8
    max_axis_ratio: float = 10.0
    cell_oversample_factor: float = 1.5
    cell_volume_range: tuple[float, float] | None = None
    cell_volume_scale_range: tuple[float, float] | None = None
    fixed_space_group: int | None = Field(default=None, strict=True)
    space_group_probabilities: Mapping[int, float] | None = Field(
        default=None, repr=False
    )
    crystal_system: CrystalSystem | None = None
    sohncke_only: bool = Field(default=False, strict=True)

    @field_validator("space_group_probabilities", mode="before")
    @classmethod
    def _copy_probability_mapping(cls, value: Any) -> Any:
        """Copy the input mapping before probability validation and freezing."""
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise TypeError("space_group_probabilities must be a mapping or None")
        return dict(value)

    @field_validator("space_group_probabilities")
    @classmethod
    def _freeze_probability_mapping(
        cls, value: Mapping[int, float] | None
    ) -> Mapping[int, float] | None:
        """Validate space-group weights and expose an immutable mapping."""
        if value is None:
            return None
        clean: dict[int, float] = {}
        for key, weight in value.items():
            if isinstance(key, bool) or not isinstance(key, int) or not 1 <= key <= 230:
                raise ValueError(
                    "space-group probability keys must be integers in [1, 230]"
                )
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError(
                    "space-group probabilities must be finite nonnegative numbers"
                )
            numeric_weight = float(weight)
            if not math.isfinite(numeric_weight) or numeric_weight < 0:
                raise ValueError(
                    "space-group probabilities must be finite nonnegative numbers"
                )
            clean[key] = numeric_weight
        return MappingProxyType(clean)

    @model_validator(mode="after")
    def _validate_configuration(self) -> PackingConfig:
        """Validate coupled packing, sampling, and relaxation settings."""
        for name in (
            "z",
            "z_prime",
            "batch_size",
            "max_steps_per_candidate",
            "convergence_check_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.z % self.z_prime:
            raise ValueError("z_prime must divide z")
        if self.max_candidates is not None and (
            isinstance(self.max_candidates, bool) or self.max_candidates <= 0
        ):
            raise ValueError("max_candidates must be a positive integer or None")

        positive_controls = ("max_step", "min_cell_height")
        for name in positive_controls:
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        nonnegative_controls = (
            "step_scale",
            "cell_step_scale",
            "max_cell_strain",
            "volume_compression_scale",
            "overlap_tolerance",
        )
        for name in nonnegative_controls:
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(self.max_axis_ratio) or self.max_axis_ratio < 1:
            raise ValueError("max_axis_ratio must be finite and at least 1")
        if (
            not math.isfinite(self.cell_oversample_factor)
            or self.cell_oversample_factor < 1
        ):
            raise ValueError("cell_oversample_factor must be finite and at least 1")

        if (self.cell_volume_range is None) == (self.cell_volume_scale_range is None):
            raise ValueError(
                "exactly one of cell_volume_range and cell_volume_scale_range is required"
            )
        for name in ("cell_volume_range", "cell_volume_scale_range"):
            interval = getattr(self, name)
            if interval is not None:
                low, high = interval
                if (
                    not math.isfinite(low)
                    or not math.isfinite(high)
                    or low <= 0
                    or high <= 0
                    or low > high
                ):
                    raise ValueError(
                        f"{name} must contain finite positive values in ascending order"
                    )

        if self.fixed_space_group is not None:
            group = self.fixed_space_group
            if (
                isinstance(group, bool)
                or not isinstance(group, int)
                or not 1 <= group <= 230
            ):
                raise ValueError(
                    "fixed_space_group must be an International number in [1, 230]"
                )
            if get_space_group_operation_count(group) != self.z // self.z_prime:
                raise ValueError(
                    "fixed_space_group operation count must equal z / z_prime"
                )
            if (
                self.space_group_probabilities is not None
                or self.crystal_system is not None
            ):
                raise ValueError(
                    "fixed_space_group cannot be combined with sampled-group filters"
                )
            if self.sohncke_only and not is_sohncke_space_group(group):
                raise ValueError(
                    "fixed_space_group must be a Sohncke group when sohncke_only=True"
                )
        elif self.crystal_system is not None and not isinstance(
            self.crystal_system, CrystalSystem
        ):
            raise TypeError("crystal_system must be a CrystalSystem or None")
        if not isinstance(self.sohncke_only, bool):
            raise TypeError("sohncke_only must be a bool")

        if self.space_group_probabilities is not None:
            candidates = get_space_group_candidates(
                self.z // self.z_prime,
                crystal_system=self.crystal_system,
                sohncke_only=self.sohncke_only,
            ).tolist()
            if not any(
                self.space_group_probabilities.get(int(group), 0.0) > 0.0
                for group in candidates
            ):
                raise ValueError(
                    "space_group_probabilities must have positive weight among compatible groups"
                )
        elif self.fixed_space_group is None:
            # Confirm the bundled prior retains positive weight after filtering.
            candidates = get_space_group_candidates(
                self.z // self.z_prime,
                crystal_system=self.crystal_system,
                sohncke_only=self.sohncke_only,
            ).tolist()
            default_prior = csd_space_group_probabilities()
            if not any(
                default_prior.get(int(group), 0.0) > 0.0 for group in candidates
            ):
                raise ValueError(
                    "bundled CSD space-group prior has no positive weight among compatible groups"
                )
        return self

    def effective(self, **overrides: Any) -> PackingConfig:
        """Return a new config with overrides fully validated.

        Unknown fields are rejected by the same ``extra='forbid'`` policy as
        direct construction. Rebuilding also revalidates the retained prior
        mapping instead of bypassing Pydantic validation.
        """
        values = {name: getattr(self, name) for name in type(self).model_fields}
        values.update(overrides)
        return type(self).model_validate(values)
