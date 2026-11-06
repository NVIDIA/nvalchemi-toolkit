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
"""Structural protocols for crystal packers."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from nvalchemi.csp.data import MolecularPackingInput
    from nvalchemi.csp.packer.result import PackingResult

__all__ = ["CandidateBudgetPacker", "CrystalPacker", "PackingContext"]


@dataclass(frozen=True)
class PackingContext:
    """Identity and rank assignment for one locally executed packing call.

    The run ID is a nonnegative 63-bit identifier shared by every rank in a
    logical run. Rank and world size describe the caller's rank assignment;
    this value object does not initialize or communicate with a process group.

    Parameters
    ----------
    run_id : int
        Shared packing-run identity in [0, 2**63).
    rank : int, default=0
        Zero-based caller-assigned rank in [0, world_size).
    world_size : int, default=1
        Positive number of caller-assigned ranks.
    """

    run_id: int
    rank: int = 0
    world_size: int = 1

    def __post_init__(self) -> None:
        """Validate integral run and rank values, excluding booleans."""
        for name in ("run_id", "rank", "world_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if not 0 <= self.run_id < 2**63:
            raise ValueError("run_id must satisfy 0 <= run_id < 2**63")
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")

    def structure_ids(self, count: int, *, device: torch.device | str) -> torch.Tensor:
        """Return rank-strided int64 identities.

        Parameters
        ----------
        count : int
            Nonnegative number of accepted structures.
        device : torch.device or str
            Device for the returned tensor.

        Returns
        -------
        torch.Tensor
            Shape [count, 2] with [run_id, rank + world_size * ordinal] rows.

        Raises
        ------
        OverflowError
            If an assigned ordinal does not fit in signed int64.
        """
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise TypeError("count must be an integer")
        count = int(count)
        if count < 0:
            raise ValueError("count must be nonnegative")
        if count and self.rank + self.world_size * (count - 1) >= 2**63:
            raise OverflowError("rank-strided structure IDs exceed int64 capacity")
        target = torch.device(device)
        if count == 0:
            return torch.empty((0, 2), dtype=torch.int64, device=target)
        run_ids = torch.full((count,), self.run_id, dtype=torch.int64, device=target)
        if count == 1:
            local_ids = torch.full((1,), self.rank, dtype=torch.int64, device=target)
        else:
            ordinals = torch.arange(count, dtype=torch.int64, device=target)
            local_ids = ordinals * self.world_size + self.rank
        return torch.stack((run_ids, local_ids), dim=1)


@runtime_checkable
class CrystalPacker(Protocol):
    """Structural interface for a local crystal packer.

    An implementation owns its device and completes one local packing call.
    A caller can use PackingContext to assign shared run identity and
    rank-strided structure IDs; this protocol does not perform communication.
    Implementations accept a zero output target and return a typed empty result.
    """

    device: torch.device

    def pack(
        self,
        inputs: MolecularPackingInput,
        *,
        num_samples: int = 1,
        rng: torch.Generator | None = None,
        context: PackingContext | None = None,
        **options: Any,
    ) -> PackingResult:
        """Generate packed structures for one formula-unit input.

        Parameters
        ----------
        inputs : MolecularPackingInput
            One validated formula-unit input.
        num_samples : int, default=1
            Nonnegative number of accepted structures to request.
        rng : torch.Generator, optional
            Random source for candidate generation.
        context : PackingContext, optional
            Caller-assigned run identity and rank information.
        **options
            Implementation-specific packing options.

        Returns
        -------
        PackingResult
            Local typed structures and one rank-local packing report.
        """
        ...


@runtime_checkable
class CandidateBudgetPacker(Protocol):
    """Optional interface for packers with an observable candidate budget."""

    def resolve_candidate_budget(
        self, *, num_samples: int, **pack_options: Any
    ) -> int | None:
        """Return the candidate cap that a call would use without sampling.

        Parameters
        ----------
        num_samples : int
            Requested number of accepted structures.
        **pack_options
            Options that affect the cap or are accepted by the pack operation.

        Returns
        -------
        int or None
            Nonnegative candidate cap, or None for an unlimited search.
        """
        ...
