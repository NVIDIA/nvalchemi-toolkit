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
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nvalchemi.csp.symmetry import (
    SpaceGroupPolicy,
    _resolve_space_group_distribution,
    get_space_group_operation_count,
)


class PackingConfig(BaseModel):
    """Choose how many trial structures to sample, which crystal symmetries and cell
    sizes to use, and when a trial is accepted.

    ``batch_size`` is the maximum number of candidates active at once. By
    default, ``max_candidates="auto"`` limits initialized trials to
    ``1000 * num_samples`` for each call. A positive integer sets a fixed cap;
    explicit ``None`` leaves the budget unlimited. A finite budget may return
    fewer structures than requested. The output target and random generator
    belong to the generating-function call.

    Exactly one cell volume mode is required. ``cell_volume_range`` gives an
    absolute conventional-cell volume interval in cubic angstroms, while
    ``cell_volume_scale_range`` scales the formula-unit estimate by ``z``.
    ``space_groups`` is a fixed or sampled :class:`SpaceGroupPolicy`. The
    default sampled policy uses relative weights derived from the Cambridge
    Structural Database (CSD). A custom prior replaces those weights.

    Parameters
    ----------
    z : int
        Number of formula units in the conventional cell.
    z_prime : int
        Number of formula units in the asymmetric unit; must divide ``z``.
    batch_size : int
        Maximum number of candidates active at once.
    max_candidates : int or {"auto"} or None, default="auto"
        Maximum number of trial structures initialized during a call.
        ``"auto"`` resolves to ``1000 * num_samples``; a positive integer sets
        an absolute cap, and ``None`` leaves the budget unlimited.
        An explicit unlimited budget assumes reachable acceptance criteria and
        can keep the search running without useful progress. Before a large or
        unlimited run, use a short pre-flight with a finite candidate budget
        and the intended starting-volume range, overlap tolerance, and
        per-candidate relaxation limit. Inspect accepted versus generated
        counts; low or zero acceptance can indicate an overly small starting
        volume. Revisit the volume settings and repeat the pre-flight before
        scaling up.
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
    space_groups : SpaceGroupPolicy, default=SpaceGroupPolicy.sampled()
        Fixed or sampled space-group selection. Its groups must have exactly
        ``z / z_prime`` symmetry operations. A sampled policy needs positive
        compatible weight after filtering.

    Examples
    --------
    Use a scale range with a finite candidate budget::

        from nvalchemi.csp.symmetry import CrystalSystem, SpaceGroupPolicy

        config = PackingConfig(
            z=4,
            z_prime=1,
            batch_size=16,
            max_candidates=64,
            cell_volume_scale_range=(0.8, 1.2),
            space_groups=SpaceGroupPolicy.sampled(
                crystal_system=CrystalSystem.MONOCLINIC,
                sohncke_only=True,
            ),
        )

    Derive a separately validated configuration for one call::

        shorter_run = config.effective(max_steps_per_candidate=300)
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    z: int = Field(strict=True)
    z_prime: int = Field(strict=True)
    batch_size: int = Field(strict=True)
    max_candidates: Annotated[int, Field(strict=True)] | Literal["auto"] | None = "auto"
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
    space_groups: SpaceGroupPolicy = Field(default_factory=SpaceGroupPolicy.sampled)

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
        if isinstance(self.max_candidates, int) and (
            isinstance(self.max_candidates, bool) or self.max_candidates <= 0
        ):
            raise ValueError(
                "max_candidates must be a positive integer, 'auto', or None"
            )

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

        required_operations = self.z // self.z_prime
        if self.space_groups.mode == "fixed":
            if self.space_groups.group is None:
                raise RuntimeError("fixed space-group policy has no group")
            group = self.space_groups.group
            actual_operations = get_space_group_operation_count(group)
            if actual_operations != required_operations:
                raise ValueError(
                    f"fixed space group {group} has {actual_operations} operations; "
                    f"z / z_prime requires {required_operations} "
                    f"(z={self.z}, z_prime={self.z_prime})"
                )
        else:
            _resolve_space_group_distribution(self.space_groups, required_operations)
        return self

    def effective(self, **overrides: Any) -> PackingConfig:
        """Return a new config with overrides fully validated.

        Unknown fields are rejected by the same ``extra='forbid'`` policy as
        direct construction. The retained policy is checked against the
        effective ``z / z_prime`` without changing its weights.
        """
        values = {name: getattr(self, name) for name in type(self).model_fields}
        values.update(overrides)
        return type(self).model_validate(values)
