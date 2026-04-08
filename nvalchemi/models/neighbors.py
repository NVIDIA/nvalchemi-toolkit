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
"""Neighbor-list builders and runtime neighbor payloads.

This module contains the flat neighbor-list runtime used by composable
models.  :class:`NeighborListBuilder` constructs COO or matrix-format
neighbor data, while :class:`NeighborList` stores one built payload and can
adapt it to downstream consumer requirements.

The public config types remain in :mod:`nvalchemi.models.base` so the
top-level structure stays close to the historical layout.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

import torch
from nvalchemiops.torch.neighbors import neighbor_list
from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, PositiveInt

from nvalchemi.models.base import (
    NeighborConfig,
    NeighborListFormat,
    _UNSET,
    _resolve_config,
)
from nvalchemi.models.utils import (
    build_model_repr,
    collect_nondefault_repr_kwargs,
    initialize_model_repr,
)

__all__ = [
    "NeighborConfig",
    "NeighborList",
    "NeighborListBuilder",
    "NeighborListBuilderConfig",
    "neighbor_list",
    "unify_neighbor_requirements",
]


def unify_neighbor_requirements(reqs: list[NeighborConfig]) -> NeighborConfig:
    """Unify multiple external neighbor requirements into one requirement.

    Parameters
    ----------
    reqs
        External neighbor requirements collected from composed models.

    Returns
    -------
    NeighborConfig
        One requirement large enough to satisfy all consumers.

    Raises
    ------
    ValueError
        If no external requirements are provided or none declares a cutoff.
    """

    if not reqs:
        raise ValueError("No external neighbor requirements to merge")
    cutoffs = [req.cutoff for req in reqs if req.cutoff is not None]
    if not cutoffs:
        raise ValueError("External neighbor requirements must declare a cutoff")
    cutoff = max(cutoffs)
    use_matrix = any(req.format == "matrix" for req in reqs)
    half_preferences = {req.half_list for req in reqs if req.half_list is not None}
    if False in half_preferences:
        resolved_half_list = False
    elif half_preferences == {True}:
        resolved_half_list = True
    else:
        resolved_half_list = None
    max_neighbors = None
    if use_matrix:
        declared_max = [req.max_neighbors for req in reqs if req.max_neighbors is not None]
        max_neighbors = max(declared_max) if declared_max else None
    return NeighborConfig(
        source="external",
        cutoff=cutoff,
        format="matrix" if use_matrix else "coo",
        half_list=resolved_half_list,
        max_neighbors=max_neighbors,
    )


class NeighborList:
    """Runtime neighbor payload for one composable execution.

    Parameters
    ----------
    neighbor_matrix
        Dense neighbor matrix or matrix-like backing storage.
    num_neighbors
        Per-atom valid neighbor counts.
    neighbor_shifts
        Optional integer image shifts associated with each neighbor entry.
    batch_idx
        Batch index per atom.
    fill_value
        Sentinel value used for invalid matrix entries.
    cutoff
        Cutoff radius used when the payload was built.
    format
        Payload layout, either ``"coo"`` or ``"matrix"``.
    half_list
        Whether the payload represents a half-list.
    """

    def __init__(
        self,
        neighbor_matrix: torch.Tensor,
        num_neighbors: torch.Tensor,
        neighbor_shifts: torch.Tensor | None,
        batch_idx: torch.Tensor,
        fill_value: int,
        cutoff: float,
        format: Literal["coo", "matrix"],
        half_list: bool,
    ) -> None:
        self.neighbor_matrix = neighbor_matrix
        self.num_neighbors = num_neighbors
        self.neighbor_shifts = neighbor_shifts
        self.batch_idx = batch_idx
        self.fill_value = fill_value
        self.cutoff = cutoff
        self.format = format
        self.half_list = half_list

    def _compute_distances_sq(
        self,
        positions: torch.Tensor,
        cell: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute squared pairwise distances for valid neighbor pairs."""

        n_atoms, max_neighbors = self.neighbor_matrix.shape
        valid_mask = self.neighbor_matrix < self.fill_value
        flat_valid = valid_mask.reshape(-1)
        valid_idx = flat_valid.nonzero(as_tuple=True)[0]

        flat_neighbors = self.neighbor_matrix.reshape(-1).long()
        center_idx = (
            torch.arange(n_atoms, device=positions.device)
            .unsqueeze(1)
            .expand(n_atoms, max_neighbors)
            .reshape(-1)
        )
        valid_center = center_idx[valid_idx]
        valid_neighbor = flat_neighbors[valid_idx].clamp(0, n_atoms - 1)
        diff = positions[valid_neighbor] - positions[valid_center]

        if self.neighbor_shifts is not None and cell is not None:
            cell_tensor = cell if cell.ndim == 3 else cell.unsqueeze(0)
            flat_shifts = self.neighbor_shifts.reshape(-1, 3)
            valid_shifts = flat_shifts[valid_idx].to(positions.dtype)
            if cell_tensor.shape[0] == 1:
                shift_real = valid_shifts @ cell_tensor.squeeze(0)
            else:
                center_batch = self.batch_idx[valid_center].long()
                shift_real = torch.bmm(
                    valid_shifts.unsqueeze(1),
                    cell_tensor[center_batch].to(positions.dtype),
                ).squeeze(1)
            diff = diff + shift_real

        valid_dist_sq = (diff * diff).sum(dim=-1)
        dist_sq = torch.full(
            (n_atoms * max_neighbors,),
            float("inf"),
            dtype=positions.dtype,
            device=positions.device,
        )
        dist_sq[valid_idx] = valid_dist_sq
        return dist_sq.reshape(n_atoms, max_neighbors)

    def _rebuild_neighbor_ptr(self, edge_index: torch.Tensor) -> torch.Tensor:
        """Return a CSR pointer consistent with one COO edge list."""

        counts = torch.bincount(
            edge_index[0].to(dtype=torch.int64),
            minlength=self.neighbor_matrix.shape[0],
        ).to(dtype=torch.int32, device=edge_index.device)
        neighbor_ptr = torch.zeros(
            self.neighbor_matrix.shape[0] + 1,
            dtype=torch.int32,
            device=edge_index.device,
        )
        neighbor_ptr[1:] = counts.cumsum(dim=0)
        return neighbor_ptr

    def adapt(
        self,
        *,
        positions: torch.Tensor,
        cell: torch.Tensor | None = None,
        cutoff: float | None = None,
        format: Literal["coo", "matrix"] | None = None,
        half_list: bool | None = None,
        max_neighbors: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Return neighbor data adapted to one consumer requirement.

        Parameters
        ----------
        positions
            Current positions used for cutoff narrowing and shift handling.
        cell
            Optional cell tensor used for periodic shift application.
        cutoff
            Optional smaller cutoff to apply to the stored payload.
        format
            Requested output layout.
        half_list
            Requested half-list setting.
        max_neighbors
            Optional target matrix width for matrix-format outputs.

        Returns
        -------
        dict[str, torch.Tensor]
            Neighbor tensors in the schema expected by the downstream model.
        """

        target_format = format or self.format
        target_half = half_list if half_list is not None else self.half_list

        if target_format == "matrix" and self.format == "coo":
            raise ValueError("Cannot convert coo to matrix format")
        if not target_half and self.half_list:
            raise ValueError("Cannot expand half list to full list")
        if cutoff is not None and cutoff > self.cutoff:
            raise ValueError(f"Requested cutoff {cutoff} > built cutoff {self.cutoff}")
        if max_neighbors is not None and target_format != "matrix":
            raise ValueError("max_neighbors adaptation is only supported for matrix format")

        neighbor_matrix = self.neighbor_matrix
        num_neighbors = self.num_neighbors
        neighbor_shifts = self.neighbor_shifts

        if cutoff is not None and cutoff < self.cutoff:
            dist_sq = self._compute_distances_sq(positions, cell)
            within_cutoff = dist_sq <= cutoff * cutoff
            valid_within_cutoff = within_cutoff & (neighbor_matrix < self.fill_value)
            fill_values = torch.full_like(neighbor_matrix, self.fill_value)
            neighbor_matrix = torch.where(valid_within_cutoff, neighbor_matrix, fill_values)
            num_neighbors = valid_within_cutoff.sum(dim=-1).to(num_neighbors.dtype)
            if neighbor_shifts is not None:
                neighbor_shifts = torch.where(
                    valid_within_cutoff.unsqueeze(-1),
                    neighbor_shifts,
                    torch.zeros_like(neighbor_shifts),
                )

        if target_format == "coo" and self.format == "matrix":
            from nvalchemiops.torch.neighbors.neighbor_utils import (
                get_neighbor_list_from_neighbor_matrix,
            )

            convert_result = get_neighbor_list_from_neighbor_matrix(
                neighbor_matrix.to(torch.int32),
                num_neighbors.to(torch.int32),
                neighbor_shift_matrix=(
                    neighbor_shifts.to(torch.int32)
                    if neighbor_shifts is not None
                    else None
                ),
                fill_value=self.fill_value,
            )
            if neighbor_shifts is not None:
                edge_index, neighbor_ptr, shifts_coo = convert_result
            else:
                edge_index, neighbor_ptr = convert_result
                shifts_coo = None
            if target_half and not self.half_list:
                keep = edge_index[0] < edge_index[1]
                edge_index = edge_index[:, keep]
                neighbor_ptr = self._rebuild_neighbor_ptr(edge_index)
                if shifts_coo is not None:
                    shifts_coo = shifts_coo[keep]
            result: dict[str, torch.Tensor] = {
                "edge_index": edge_index,
                "neighbor_ptr": neighbor_ptr,
                "fill_value": torch.tensor(self.fill_value, device=neighbor_matrix.device),
            }
            if shifts_coo is not None:
                result["unit_shifts"] = shifts_coo
            return result

        if target_half and not self.half_list and self.format == "matrix":
            raise NotImplementedError(
                "Half-list narrowing on matrix format not yet implemented. Convert to coo first."
            )

        if max_neighbors is not None:
            actual_max = int(num_neighbors.max().item()) if num_neighbors.numel() else 0
            if max_neighbors < actual_max:
                raise ValueError(
                    f"Requested max_neighbors {max_neighbors} is smaller than the actual valid neighbor count {actual_max}."
                )
            target_width = min(max_neighbors, neighbor_matrix.shape[1])
            if target_width < neighbor_matrix.shape[1]:
                neighbor_matrix = neighbor_matrix[:, :target_width]
                if neighbor_shifts is not None:
                    neighbor_shifts = neighbor_shifts[:, :target_width]

        result = {
            "neighbor_matrix": neighbor_matrix,
            "num_neighbors": num_neighbors,
            "fill_value": torch.tensor(self.fill_value, device=neighbor_matrix.device),
        }
        if neighbor_shifts is not None:
            result["neighbor_shifts"] = neighbor_shifts
        return result


def _round_to_16(value: int) -> int:
    """Round one integer up to the next multiple of sixteen."""

    return ((value + 15) // 16) * 16


class NeighborListBuilderConfig(BaseModel):
    """Configuration for :class:`NeighborListBuilder`.

    Attributes
    ----------
    cutoff : float
        Interaction cutoff radius.
    format : {"coo", "matrix"}
        Output storage layout produced by the builder.
    half_list : bool
        Whether to build a half-list instead of a full list.
    trim_matrix : bool
        Whether matrix outputs should be trimmed to the actual maximum
        neighbor count.
    max_neighbors : int | None
        Fixed matrix width used for stable-shape matrix outputs.
    adaptive_capacity : bool
        Whether matrix capacity should grow and shrink automatically.
    initial_density : float
        Density estimate used for initial adaptive matrix sizing.
    target_utilization : float
        Target utilization used for shrink decisions in matrix mode.
    """

    cutoff: Annotated[
        PositiveFloat, Field(description="Interaction cutoff radius (assumed Angstrom).")
    ]
    format: Annotated[
        Literal["coo", "matrix"], Field(description="Output storage layout.")
    ] = NeighborListFormat.COO.value
    half_list: Annotated[
        bool, Field(description="Build a half-list instead of a full list.")
    ] = False
    trim_matrix: Annotated[
        bool, Field(description="Trim matrix output to actual maximum neighbor count.")
    ] = True
    max_neighbors: Annotated[
        PositiveInt | None, Field(description="Fixed matrix width for stable shapes.")
    ] = None
    adaptive_capacity: Annotated[
        bool, Field(description="Whether to adapt matrix capacity automatically.")
    ] = True
    initial_density: Annotated[
        PositiveFloat, Field(description="Initial density estimate for adaptive sizing.")
    ] = 0.2
    target_utilization: Annotated[
        float,
        Field(gt=0.0, lt=1.0, description="Target matrix utilization for shrink decisions."),
    ] = 0.75

    model_config = ConfigDict(extra="forbid")


class NeighborListBuilder:
    """Concrete neighbor-list builder for COO or matrix output.

    Parameters
    ----------
    config
        Optional prebuilt builder configuration.
    cutoff, format, half_list, trim_matrix, max_neighbors, adaptive_capacity, initial_density, target_utilization
        Keyword overrides applied on top of ``config``.
    """

    def __init__(
        self,
        config: NeighborListBuilderConfig | None = None,
        *,
        cutoff: float = _UNSET,
        format: Literal["coo", "matrix"] = _UNSET,
        half_list: bool = _UNSET,
        trim_matrix: bool = _UNSET,
        max_neighbors: int | None = _UNSET,
        adaptive_capacity: bool = _UNSET,
        initial_density: float = _UNSET,
        target_utilization: float = _UNSET,
    ) -> None:
        config = _resolve_config(
            NeighborListBuilderConfig,
            config,
            {
                "cutoff": cutoff,
                "format": format,
                "half_list": half_list,
                "trim_matrix": trim_matrix,
                "max_neighbors": max_neighbors,
                "adaptive_capacity": adaptive_capacity,
                "initial_density": initial_density,
                "target_utilization": target_utilization,
            },
        )
        self.config = config
        if self.config.format == "matrix":
            self._matrix_capacity = self._initial_matrix_capacity()
        else:
            self._matrix_capacity = None
        initialize_model_repr(
            self,
            static_kwargs=collect_nondefault_repr_kwargs(
                explicit_values={
                    "cutoff": self.config.cutoff,
                    "format": self.config.format,
                    "half_list": self.config.half_list,
                    "trim_matrix": self.config.trim_matrix,
                    "max_neighbors": self.config.max_neighbors,
                    "adaptive_capacity": self.config.adaptive_capacity,
                    "initial_density": self.config.initial_density,
                    "target_utilization": self.config.target_utilization,
                },
                defaults={
                    "format": "coo",
                    "half_list": False,
                    "trim_matrix": True,
                    "max_neighbors": None,
                    "adaptive_capacity": True,
                    "initial_density": 0.2,
                    "target_utilization": 0.75,
                },
                order=(
                    "cutoff",
                    "format",
                    "half_list",
                    "trim_matrix",
                    "max_neighbors",
                    "adaptive_capacity",
                    "initial_density",
                    "target_utilization",
                ),
            ),
            kwarg_order=(
                "cutoff",
                "format",
                "half_list",
                "trim_matrix",
                "max_neighbors",
                "adaptive_capacity",
                "initial_density",
                "target_utilization",
            ),
        )

    def __repr__(self) -> str:
        """Return one compact constructor-style representation."""

        return build_model_repr(self)

    def _initial_matrix_capacity(self) -> int | None:
        """Return the initial matrix width used for backend calls."""

        if self.config.max_neighbors is not None:
            return self.config.max_neighbors
        if not self.config.adaptive_capacity:
            return None
        sphere_volume = 4.0 / 3.0 * math.pi * self.config.cutoff**3
        estimated_neighbors = max(16, int(self.config.initial_density * sphere_volume))
        return _round_to_16(estimated_neighbors)

    @staticmethod
    def _is_overflow_error(exc: Exception) -> bool:
        """Return whether one backend error reports buffer overflow."""

        return "overflow" in str(exc).lower() or type(exc).__name__ == "NeighborOverflowError"

    def _actual_max_neighbors(self, built: tuple[torch.Tensor, ...]) -> int:
        """Return the actual maximum neighbor count represented in one result."""

        if self.config.format == "coo":
            neighbor_ptr = built[1]
            return int((neighbor_ptr[1:] - neighbor_ptr[:-1]).max().item())
        return int(built[1].max().item())

    def _maybe_shrink_capacity(self, built: tuple[torch.Tensor, ...]) -> None:
        """Shrink the adaptive matrix capacity when utilization is low."""

        if (
            self.config.format != "matrix"
            or self.config.max_neighbors is not None
            or not self.config.adaptive_capacity
            or self._matrix_capacity is None
        ):
            return
        actual_max = self._actual_max_neighbors(built)
        threshold = (2.0 / 3.0) * self.config.target_utilization * self._matrix_capacity
        if actual_max < threshold:
            target = max(16, math.ceil(actual_max / self.config.target_utilization))
            self._matrix_capacity = _round_to_16(target)

    def _backend_kwargs(
        self,
        *,
        positions: torch.Tensor,
        batch_idx: torch.Tensor,
        batch_ptr: torch.Tensor,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
        max_neighbors: int | None = None,
    ) -> dict[str, Any]:
        """Build backend keyword arguments for the neighbor-list call."""

        kwargs: dict[str, Any] = {
            "positions": positions,
            "cutoff": self.config.cutoff,
            "batch_idx": batch_idx.to(torch.int32),
            "batch_ptr": batch_ptr.to(torch.int32),
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
        batch_idx: torch.Tensor,
        batch_ptr: torch.Tensor,
        cell: torch.Tensor | None,
        pbc: torch.Tensor | None,
    ) -> tuple[torch.Tensor, ...]:
        """Call the low-level backend for this builder."""

        backend_kwargs = self._backend_kwargs(
            positions=positions,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cell=cell,
            pbc=pbc,
            max_neighbors=self._matrix_capacity,
        )
        while True:
            try:
                built = neighbor_list(**backend_kwargs)
                break
            except Exception as exc:
                if (
                    self.config.format != "matrix"
                    or self.config.max_neighbors is not None
                    or not self.config.adaptive_capacity
                    or not self._is_overflow_error(exc)
                ):
                    raise
                if self._matrix_capacity is None:
                    self._matrix_capacity = 16
                self._matrix_capacity = _round_to_16(int(self._matrix_capacity * 1.5))
                backend_kwargs["max_neighbors"] = self._matrix_capacity
        self._maybe_shrink_capacity(built)
        return built

    def _trim_matrix_outputs(
        self,
        neighbor_matrix: torch.Tensor,
        num_neighbors: torch.Tensor,
        neighbor_shifts: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Trim matrix outputs down to the actual maximum neighbor count."""

        if not self.config.trim_matrix:
            return neighbor_matrix, num_neighbors, neighbor_shifts
        actual_nnb = max(1, int(num_neighbors.max().item()))
        trimmed_shifts = None
        if neighbor_shifts is not None:
            trimmed_shifts = neighbor_shifts[:, :actual_nnb]
        return neighbor_matrix[:, :actual_nnb], num_neighbors, trimmed_shifts

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
                "edge_index": neighbor_list_tensor,
                "neighbor_ptr": neighbor_ptr,
                "unit_shifts": unit_shifts,
            }

        neighbor_matrix, num_neighbors = built[:2]
        neighbor_shifts = built[2] if len(built) > 2 else None
        neighbor_matrix, num_neighbors, neighbor_shifts = self._trim_matrix_outputs(
            neighbor_matrix,
            num_neighbors,
            neighbor_shifts,
        )
        result = {
            "neighbor_matrix": neighbor_matrix,
            "num_neighbors": num_neighbors,
        }
        if neighbor_shifts is not None:
            result["neighbor_shifts"] = neighbor_shifts
        return result

    def __call__(
        self,
        *,
        positions: torch.Tensor,
        batch_idx: torch.Tensor,
        batch_ptr: torch.Tensor,
        cell: torch.Tensor | None = None,
        pbc: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Build a neighbor list in the configured format.

        Parameters
        ----------
        positions
            Atomic positions.
        batch_idx
            Batch index per atom.
        batch_ptr
            CSR-style graph pointer.
        cell
            Optional cell tensor for periodic runs.
        pbc
            Optional periodic-boundary flags.

        Returns
        -------
        dict[str, torch.Tensor]
            Neighbor-list tensors in either COO or matrix schema.
        """

        built = self._build_neighbor_list(
            positions=positions,
            batch_idx=batch_idx,
            batch_ptr=batch_ptr,
            cell=cell,
            pbc=pbc,
        )
        return self._materialize_values(built)
