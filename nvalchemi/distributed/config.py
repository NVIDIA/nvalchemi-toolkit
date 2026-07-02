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

"""Configuration types for spatial domain decomposition.

:class:`DomainConfig` is a flat Pydantic model bundling the three concerns a
distributed scope needs: process-mesh topology, halo/skin geometry, and the
spatial-partition grid.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class StrategyKind(str, Enum):
    """Which parallelization strategy a distributed scope runs under.

    Selected on :class:`DomainConfig` (config-driven, not an env var). The model's
    ``distribution_spec(strategy)`` returns the ``(policy, adapters, shard_fields,
    consolidation)`` bundle for the chosen strategy; the framework builds the live
    :class:`~nvalchemi.distributed.strategy.ParallelizationStrategy` from the
    resulting storage policy.

    Attributes
    ----------
    HALO : str
        Spatial domain decomposition (owned atoms + ghost halo). Default.
    GRAPH_PARTITION : str
        Node-partition graph parallel (owned node slice per rank).
    """

    HALO = "halo"
    GRAPH_PARTITION = "graph_partition"


class HookScope(Enum):
    """Determines which ranks execute a hook callback.

    Attributes
    ----------
    LOCAL : str
        Hook runs on every rank with its local subdomain batch.
    GLOBAL : str
        Hook runs on every rank after an all-gather produces the full batch.
    RANK_ZERO : str
        Hook runs only on rank 0 after gathering.
    """

    LOCAL = "local"
    GLOBAL = "global"
    RANK_ZERO = "rank_zero"


class DomainConfig(BaseModel):
    """Configuration for one spatial domain-decomposition scope.

    Parameters
    ----------
    cutoff : float
        Interaction cutoff radius used by the model.
    skin : float
        Additional skin distance for neighbor-list buffering. Default 0.
    ghost_width : float | None
        Width of the ghost (halo) region. When ``None``, the effective
        width defaults to ``cutoff + skin`` via :meth:`effective_ghost_width`.
    mesh : DeviceMesh | None
        Optional ``torch.distributed.device_mesh.DeviceMesh`` describing the
        rank topology. ``None`` for single-rank runs.
    mesh_dim : str
        Name of the mesh dimension used for domain parallelism. Default
        ``"domain"``.
    grid_dims : tuple[int, int, int] | None
        Explicit grid dimensions for the spatial decomposition. When ``None``,
        the partitioner chooses cells-per-dim from the cell matrix and cutoff.
    scripted_marshal : {"auto", "declared", "off"}
        Controls marshalling of ``@torch.jit.script`` ops across the ShardTensor
        boundary (a scripted kernel reading a ShardTensor's storage-less
        ``data_ptr`` triggers a CUDA illegal memory access). ``"auto"`` (default):
        auto-discover scripted submodules and wrap them, plus install the spec's
        declared ``JitAdapter`` marshallers. ``"declared"``: install only the
        spec's declared adapters, no auto-discovery. ``"off"``: no marshalling at
        all. Overridable via ``NVALCHEMI_SCRIPTED_MARSHAL``.
    scripted_marshal_exclude : tuple[str, ...]
        Submodule-name substrings to skip during ``"auto"`` discovery — for a
        scripted op that genuinely needs cross-rank data (where marshalling to
        local would silently give wrong numbers) or is handled via ``custom_ops``.
    """

    model_config = {"arbitrary_types_allowed": True}

    cutoff: float = Field(gt=0)
    skin: float = Field(default=0.0, ge=0)
    ghost_width: float | None = Field(default=None, gt=0)
    strategy: StrategyKind = StrategyKind.HALO
    # Compile intent for the DD forward. When True the framework owns the compiled
    # forward (fixed-shape caps + compiled energy-autograd), so the per-rank atom /
    # edge counts are padded to stable shapes and the compiled graph is reused
    # across MD steps. Without it, a model compiled by its own loader recompiles
    # every step as the owned+ghost count drifts (atoms migrating across the domain
    # boundary) — the padder is gated on this flag.
    compile: bool = False
    # DeviceMesh | None at runtime; typed ``Any`` so pydantic doesn't reject the
    # ducktyped test-harness mock meshes the collectives (mesh_group) also accept.
    mesh: Any = None
    mesh_dim: str = "domain"
    grid_dims: tuple[int, int, int] | None = None
    scripted_marshal: Literal["auto", "declared", "off"] = "auto"
    scripted_marshal_exclude: tuple[str, ...] = ()
    migration_hysteresis: float | None = None

    @field_validator("grid_dims")
    @classmethod
    def _grid_dims_positive(
        cls, v: tuple[int, int, int] | None
    ) -> tuple[int, int, int] | None:
        if v is not None and any(d < 1 for d in v):
            raise ValueError(f"grid_dims entries must be >= 1, got {v}")
        return v

    def effective_migration_hysteresis(self) -> float:
        """Migration-hysteresis margin (angstrom). Defaults to ``skin/2``.

        An atom keeps its current owner until it is this far past a domain
        boundary, preventing per-step migration thrashing of boundary atoms.
        Must be ``< skin`` so a deferred atom stays within the owner's halo
        (ghost_width = cutoff + skin).
        """
        h = (
            self.migration_hysteresis
            if self.migration_hysteresis is not None
            else self.skin / 2.0
        )
        if self.skin > 0.0 and h >= self.skin:
            raise ValueError(
                f"migration_hysteresis ({h}) must be < skin ({self.skin}) for "
                "halo-coverage correctness (a deferred atom must remain within "
                "the owner's ghost region)."
            )
        return max(0.0, float(h))

    def effective_ghost_width(self) -> float:
        """Return the ghost region width, defaulting to ``cutoff + skin``."""
        return (
            self.ghost_width
            if self.ghost_width is not None
            else self.cutoff + self.skin
        )


__all__ = [
    "HookScope",
    "DomainConfig",
    "StrategyKind",
]
