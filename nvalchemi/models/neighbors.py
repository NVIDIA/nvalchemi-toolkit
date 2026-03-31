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

import math
from typing import Annotated, Any, Literal

import torch
from pydantic import BaseModel, ConfigDict, Field

from nvalchemi.data import Batch
from nvalchemi.models.base import ForwardContext, _NeighborListStep, _UNSET, _resolve_config
from nvalchemi.models.contracts import NeighborListCard
from nvalchemi.models.results import CalculatorResults
from nvalchemiops.torch.neighbors import neighbor_list

__all__ = [
    "AdaptiveNeighborListBuilder",
    "AdaptiveNeighborListConfig",
    "NeighborListBuilder",
    "NeighborListBuilderConfig",
    "neighbor_result_key",
]


def neighbor_result_key(neighbor_list_name: str, name: str) -> str:
    """Return the canonical result key for a named neighbor-list output."""

    return f"neighbor_lists.{neighbor_list_name}.{name}"


def _round_to_16(value: int) -> int:
    """Round *value* up to the next multiple of 16."""

    return ((value + 15) // 16) * 16


class NeighborListBuilderConfig(BaseModel):
    """Configuration for :class:`NeighborListBuilder`.

    Attributes
    ----------
    neighbor_list_name : str, default "default"
        Logical name used to namespace result keys
        (e.g. ``"neighbor_lists.default.neighbor_matrix"``).
    cutoff : float
        Interaction cutoff radius (assumed Angstrom).
    format : {"coo", "matrix"}, default "coo"
        Output storage layout.
    half_list : bool, default False
        Build a half-list (each pair counted once) instead of a full list.
    trim_matrix_to_fit : bool, default False
        Trim the matrix output to the actual maximum neighbor count.
    reuse_if_available : bool, default True
        Skip computation when matching outputs are already present in
        the batch or accumulated results.
    """

    neighbor_list_name: Annotated[
        str, Field(description="Logical neighbor-list name for result-key namespacing.")
    ] = "default"
    cutoff: Annotated[float, Field(description="Interaction cutoff radius (assumed Angstrom).")]
    format: Annotated[
        Literal["coo", "matrix"], Field(description="Output storage layout.")
    ] = "coo"
    half_list: Annotated[
        bool, Field(description="Build a half-list instead of a full list.")
    ] = False
    trim_matrix_to_fit: Annotated[
        bool, Field(description="Trim matrix output to actual maximum neighbor count.")
    ] = False
    reuse_if_available: Annotated[
        bool, Field(description="Skip computation when matching outputs already exist.")
    ] = True

    model_config = ConfigDict(extra="forbid")


class AdaptiveNeighborListConfig(NeighborListBuilderConfig):
    """Configuration for :class:`AdaptiveNeighborListBuilder`.

    Extends :class:`NeighborListBuilderConfig` with parameters that
    control the adaptive capacity sizing.

    Attributes
    ----------
    density : float, default 0.2
        Estimated atomic number density (assumed atoms / Angstrom^3) used to
        initialise the internal buffer capacity.
    target_utilization : float, default 0.75
        Desired buffer utilisation fraction.  The builder shrinks
        capacity when actual utilisation drops well below this value.
    """

    density: Annotated[
        float, Field(description="Estimated atomic density (assumed atoms / Angstrom^3).")
    ] = 0.2
    target_utilization: Annotated[
        float, Field(description="Desired buffer utilisation fraction.")
    ] = 0.75

    model_config = ConfigDict(extra="forbid")


DefaultNeighborListBuilderCard = NeighborListCard(
    neighbor_list_name="default",
    cutoff=None,
    format="coo",
    half_list=False,
    required_inputs=frozenset({"positions"}),
    optional_inputs=frozenset({"cell", "pbc"}),
    result_keys=frozenset(
        {
            neighbor_result_key("default", "neighbor_list"),
            neighbor_result_key("default", "neighbor_ptr"),
            neighbor_result_key("default", "unit_shifts"),
        }
    ),
    default_result_keys=frozenset(
        {
            neighbor_result_key("default", "neighbor_list"),
            neighbor_result_key("default", "neighbor_ptr"),
            neighbor_result_key("default", "unit_shifts"),
        }
    ),
    parameterized_by=frozenset(
        {"neighbor_list_name", "cutoff", "format", "half_list"}
    ),
)


class NeighborListBuilder(_NeighborListStep):
    """Concrete explicit neighbor-list builder for COO or matrix output."""

    card = DefaultNeighborListBuilderCard

    def __init__(
        self,
        config: NeighborListBuilderConfig | None = None,
        *,
        neighbor_list_name: str = _UNSET,
        cutoff: float = _UNSET,
        format: Literal["coo", "matrix"] = _UNSET,
        half_list: bool = _UNSET,
        trim_matrix_to_fit: bool = _UNSET,
        reuse_if_available: bool = _UNSET,
        name: str | None = None,
    ) -> None:
        """Initialise a neighbor-list builder.

        Accepts either a :class:`NeighborListBuilderConfig` object,
        individual keyword arguments matching the config fields, or
        both (keyword arguments override corresponding config fields).

        Parameters
        ----------
        config
            Pre-built configuration object.
        neighbor_list_name
            Logical name used to namespace result keys.
        cutoff
            Interaction cutoff radius (assumed Angstrom).
        format
            Output storage layout (``"coo"`` or ``"matrix"``).
        half_list
            Build a half-list instead of a full list.
        trim_matrix_to_fit
            Trim matrix output to the actual maximum neighbor count.
        reuse_if_available
            Skip computation when matching outputs already exist.
        name
            Human-readable step name.
        """

        config = _resolve_config(
            NeighborListBuilderConfig,
            config,
            {
                "neighbor_list_name": neighbor_list_name,
                "cutoff": cutoff,
                "format": format,
                "half_list": half_list,
                "trim_matrix_to_fit": trim_matrix_to_fit,
                "reuse_if_available": reuse_if_available,
            },
        )
        result_keys = self._supported_result_keys(config)
        super().__init__(
            name=name,
            neighbor_list_name=config.neighbor_list_name,
            cutoff=config.cutoff,
            format=config.format,
            half_list=config.half_list,
            result_keys=result_keys,
            default_result_keys=result_keys,
        )
        self.config = config

    @staticmethod
    def _supported_result_keys(
        config: NeighborListBuilderConfig,
    ) -> frozenset[str]:
        """Return the stable result schema for one configured builder."""

        if config.format == "coo":
            keys = {
                neighbor_result_key(config.neighbor_list_name, "neighbor_list"),
                neighbor_result_key(config.neighbor_list_name, "neighbor_ptr"),
                neighbor_result_key(config.neighbor_list_name, "unit_shifts"),
            }
        else:
            keys = {
                neighbor_result_key(config.neighbor_list_name, "neighbor_matrix"),
                neighbor_result_key(config.neighbor_list_name, "num_neighbors"),
                neighbor_result_key(config.neighbor_list_name, "neighbor_shifts"),
            }
        return frozenset(keys)

    def _reusable_outputs(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> CalculatorResults | None:
        """Return requested outputs from prior results when reuse is enabled."""

        if not self.config.reuse_if_available:
            return None

        requested = tuple(sorted(ctx.outputs))
        reused = CalculatorResults()
        for key in requested:
            if ctx.results is not None and key in ctx.results:
                reused[key] = ctx.results[key]
                continue
            if self.config.neighbor_list_name == "default":
                batch_key = key.rsplit(".", maxsplit=1)[-1]
                if hasattr(batch, batch_key):
                    reused[key] = getattr(batch, batch_key)
                    continue
            return None
        return reused

    def _backend_kwargs(
        self,
        *,
        positions: torch.Tensor,
        batch: Batch,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
        max_neighbors: int | None = None,
    ) -> dict[str, Any]:
        """Build backend kwargs shared by standard and adaptive builders."""

        kwargs: dict[str, Any] = {
            "positions": positions,
            "cutoff": self.config.cutoff,
            "batch_idx": batch.batch.to(torch.int32),
            "batch_ptr": batch.ptr.to(torch.int32),
            "half_fill": self.config.half_list,
            "return_neighbor_list": self.config.format == "coo",
        }
        if max_neighbors is not None:
            kwargs["max_neighbors"] = max_neighbors
        if cell is not None and pbc is not None:
            kwargs["cell"] = cell
            kwargs["pbc"] = pbc
        return kwargs

    def _build_neighbor_list(
        self,
        *,
        positions: torch.Tensor,
        batch: Batch,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Call the low-level neighbor-list backend for this builder."""

        return neighbor_list(
            **self._backend_kwargs(
                positions=positions,
                batch=batch,
                cell=cell,
                pbc=pbc,
            )
        )

    def _trim_matrix_outputs(
        self,
        neighbor_matrix: torch.Tensor,
        num_neighbors: torch.Tensor,
        neighbor_shifts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trim matrix outputs down to the actual maximum neighbor count."""

        if not self.config.trim_matrix_to_fit:
            return neighbor_matrix, num_neighbors, neighbor_shifts

        actual_nnb = max(1, int(num_neighbors.max().item()))
        return (
            neighbor_matrix[:, :actual_nnb],
            num_neighbors,
            neighbor_shifts[:, :actual_nnb],
        )

    def _materialize_values(
        self,
        built: tuple[torch.Tensor, ...],
    ) -> dict[str, torch.Tensor]:
        """Convert backend output into the stable result schema."""

        if self.config.format == "coo":
            neighbor_list_tensor, neighbor_ptr = built[:2]
            unit_shifts = (
                built[2]
                if len(built) > 2
                else torch.zeros(
                    (neighbor_list_tensor.shape[1], 3),
                    dtype=torch.int32,
                    device=neighbor_list_tensor.device,
                )
            )
            return {
                "neighbor_list": neighbor_list_tensor,
                "neighbor_ptr": neighbor_ptr,
                "unit_shifts": unit_shifts,
            }

        neighbor_matrix, num_neighbors = built[:2]
        neighbor_shifts = (
            built[2]
            if len(built) > 2
            else torch.zeros(
                (*neighbor_matrix.shape, 3),
                dtype=torch.int32,
                device=neighbor_matrix.device,
            )
        )
        neighbor_matrix, num_neighbors, neighbor_shifts = self._trim_matrix_outputs(
            neighbor_matrix,
            num_neighbors,
            neighbor_shifts,
        )
        return {
            "neighbor_matrix": neighbor_matrix,
            "num_neighbors": num_neighbors,
            "neighbor_shifts": neighbor_shifts,
        }

    def compute(
        self,
        batch: Batch,
        ctx: ForwardContext,
    ) -> CalculatorResults:
        """Build a neighbor list in the configured format."""

        reused = self._reusable_outputs(batch, ctx)
        if reused is not None:
            return reused

        positions = self.require_input(batch, "positions", ctx)
        cell, pbc = self.resolve_periodic_inputs(batch, ctx)
        built = self._build_neighbor_list(
            positions=positions,
            batch=batch,
            cell=cell,
            pbc=pbc,
        )
        values = self._materialize_values(built)
        out = CalculatorResults()
        for name, value in values.items():
            key = neighbor_result_key(self.config.neighbor_list_name, name)
            if key in ctx.outputs:
                out[key] = value
        return out


class AdaptiveNeighborListBuilder(NeighborListBuilder):
    """Neighbor-list builder with adaptive internal neighbor capacity."""

    config: AdaptiveNeighborListConfig

    def __init__(
        self,
        config: AdaptiveNeighborListConfig | None = None,
        *,
        neighbor_list_name: str = _UNSET,
        cutoff: float = _UNSET,
        format: Literal["coo", "matrix"] = _UNSET,
        half_list: bool = _UNSET,
        trim_matrix_to_fit: bool = _UNSET,
        reuse_if_available: bool = _UNSET,
        density: float = _UNSET,
        target_utilization: float = _UNSET,
        name: str | None = None,
    ) -> None:
        """Initialise an adaptive-capacity neighbor-list builder.

        Accepts either an :class:`AdaptiveNeighborListConfig` object,
        individual keyword arguments matching the config fields, or
        both (keyword arguments override corresponding config fields).

        Parameters
        ----------
        config
            Pre-built configuration object.
        neighbor_list_name
            Logical name used to namespace result keys.
        cutoff
            Interaction cutoff radius (assumed Angstrom).
        format
            Output storage layout (``"coo"`` or ``"matrix"``).
        half_list
            Build a half-list instead of a full list.
        trim_matrix_to_fit
            Trim matrix output to the actual maximum neighbor count.
        reuse_if_available
            Skip computation when matching outputs already exist.
        density
            Estimated atomic density (assumed atoms / Angstrom^3) for initial
            buffer sizing.
        target_utilization
            Desired buffer utilisation fraction.
        name
            Human-readable step name.
        """

        config = _resolve_config(
            AdaptiveNeighborListConfig,
            config,
            {
                "neighbor_list_name": neighbor_list_name,
                "cutoff": cutoff,
                "format": format,
                "half_list": half_list,
                "trim_matrix_to_fit": trim_matrix_to_fit,
                "reuse_if_available": reuse_if_available,
                "density": density,
                "target_utilization": target_utilization,
            },
        )
        super().__init__(config=config, name=name)
        sphere_volume = 4.0 / 3.0 * math.pi * config.cutoff**3
        estimated_neighbors = max(16, int(config.density * sphere_volume))
        self._max_neighbors = _round_to_16(estimated_neighbors)

    @staticmethod
    def _is_overflow_error(exc: Exception) -> bool:
        """Return whether *exc* reports a neighbor-list buffer overflow."""

        return (
            "overflow" in str(exc).lower()
            or type(exc).__name__ == "NeighborOverflowError"
        )

    def _actual_max_neighbors(self, built: tuple[torch.Tensor, ...]) -> int:
        """Return the actual maximum neighbors represented in one backend result."""

        if self.config.format == "coo":
            neighbor_ptr = built[1]
            return int((neighbor_ptr[1:] - neighbor_ptr[:-1]).max().item())
        return int(built[1].max().item())

    def _maybe_shrink_capacity(self, built: tuple[torch.Tensor, ...]) -> None:
        """Shrink capacity when utilization drops below the target."""

        actual_max = self._actual_max_neighbors(built)
        threshold = (2.0 / 3.0) * self.config.target_utilization * self._max_neighbors
        if actual_max < threshold:
            target = max(16, int(actual_max / self.config.target_utilization))
            self._max_neighbors = _round_to_16(target)

    def _build_neighbor_list(
        self,
        *,
        positions: torch.Tensor,
        batch: Batch,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Call the low-level neighbor-list backend with adaptive sizing."""

        kwargs = self._backend_kwargs(
            positions=positions,
            batch=batch,
            cell=cell,
            pbc=pbc,
            max_neighbors=self._max_neighbors,
        )

        while True:
            try:
                result = neighbor_list(**kwargs)
                break
            except Exception as exc:  # pragma: no cover - backend-specific path
                if not self._is_overflow_error(exc):
                    raise
                self._max_neighbors = _round_to_16(int(self._max_neighbors * 1.5))
                kwargs["max_neighbors"] = self._max_neighbors

        self._maybe_shrink_capacity(result)
        return result
