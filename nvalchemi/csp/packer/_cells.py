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

"""Space-group-conditioned cell sampling for the private Packer engine."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from nvalchemi.csp._space_group_tables import SG_OPS_IDX, SG_OPS_PTR, SYMM_OPS
from nvalchemi.csp.packer._kernels.sampling import generate_cells
from nvalchemi.csp.symmetry import (
    csd_space_group_probabilities,
    get_space_group_candidates,
)


def sampling_tables(
    *,
    config: object,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Prepare compatible groups, sampling weights, and standard operations."""
    operation_count = int(config.z) // int(config.z_prime)
    if config.fixed_space_group is None:
        groups = get_space_group_candidates(
            operation_count,
            crystal_system=config.crystal_system,
            sohncke_only=config.sohncke_only,
        )
        prior = (
            csd_space_group_probabilities()
            if config.space_group_probabilities is None
            else config.space_group_probabilities
        )
        raw_weights = [float(prior.get(int(group), 0.0)) for group in groups.tolist()]
        max_weight = max(raw_weights, default=0.0)
        if max_weight <= 0.0:
            raise ValueError("no compatible space group has positive sampling weight")
        scaled_weights = [weight / max_weight for weight in raw_weights]
        scaled_total = math.fsum(scaled_weights)
        weights = torch.tensor(
            [scaled / scaled_total for scaled in scaled_weights],
            dtype=torch.float32,
        )
        positive_float32 = weights > 0.0
        groups = groups[positive_float32]
        weights = weights[positive_float32]
    else:
        groups = torch.tensor([config.fixed_space_group], dtype=torch.int32)
        weights = torch.ones((1,), dtype=torch.float32)
    groups = groups.to(device=device, dtype=torch.int32).contiguous()
    weights = weights.to(device=device, dtype=torch.float32).contiguous()
    operations = torch.as_tensor(
        SYMM_OPS.copy(), dtype=torch.float32, device=device
    ).contiguous()
    indices = torch.as_tensor(
        SG_OPS_IDX.copy(), dtype=torch.int32, device=device
    ).contiguous()
    pointers = torch.as_tensor(
        SG_OPS_PTR.copy(), dtype=torch.int32, device=device
    ).contiguous()
    return groups, weights, operations, indices, pointers, torch.cumsum(weights, 0)


def sample_valid_cells(
    *,
    count: int,
    config: object,
    groups: Tensor,
    probabilities: Tensor,
    formula_unit_volume: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    """Return the first ``count`` valid cells in Warp trial order."""
    volume_range = config.cell_volume_range
    if volume_range is None:
        low_scale, high_scale = config.cell_volume_scale_range
        base = float(formula_unit_volume) * int(config.z)
        volume_range = (base * low_scale, base * high_scale)
    return generate_cells(
        count=count,
        candidate_groups=groups,
        probabilities=probabilities,
        volume_range=(float(volume_range[0]), float(volume_range[1])),
        min_cell_height=float(config.min_cell_height),
        max_axis_ratio=float(config.max_axis_ratio),
        oversample_factor=float(config.cell_oversample_factor),
        seed=seed,
    )
