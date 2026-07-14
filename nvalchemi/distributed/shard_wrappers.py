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

"""nvalchemi MD/hook op registry for ShardTensor dispatch.

Names the concrete ``nvalchemi::`` / ``nvalchemi_hooks::`` Warp custom ops that
should route over :class:`ShardTensor` inputs, and classifies each as a
*passthrough* (per-atom, embarrassingly parallel) or a *reduction* (per-atom →
per-system). The generic registration + wrapper machinery lives in the
domain-neutral :mod:`nvalchemi.distributed._core.shard_wrappers`.

Call :func:`register_shard_wrappers` once at startup (e.g. in
``DomainParallel.__init__``).
"""

from __future__ import annotations

import torch

from nvalchemi.distributed._core.shard_wrappers import register_op_wrappers

__all__ = ["PASSTHROUGH_OPS", "REDUCTION_OPS", "register_shard_wrappers"]

# Category 1: Per-atom, embarrassingly parallel (passthrough)
PASSTHROUGH_OPS: list[str] = [
    # Velocity Verlet
    "nvalchemi::vv_position_update",
    "nvalchemi::vv_velocity_finalize",
    # Langevin
    "nvalchemi::langevin_half_step",
    "nvalchemi::langevin_finalize",
    # Nose-Hoover
    "nvalchemi::nhc_velocity_half_step",
    "nvalchemi::nhc_position_update",
    "nvalchemi::nhc_chain_update",
    # NPT/NPH
    "nvalchemi::npt_position_update",
    "nvalchemi::npt_cell_update",
    "nvalchemi::nph_barostat_half_step",
    "nvalchemi::nph_velocity_half_step",
    "nvalchemi::npt_barostat_half_step",
    "nvalchemi::npt_thermostat_half_step",
    "nvalchemi::npt_velocity_half_step",
    "nvalchemi::stress_to_cell_force",
    # FIRE
    "nvalchemi::_fire_step_op",
    "nvalchemi::_fire_update_op",
    # Thermostat utilities
    "nvalchemi::initialize_velocities",
    "nvalchemi::remove_com_motion",
    "nvalchemi::velocity_rescale",
    # Hooks
    "nvalchemi_hooks::wrap_positions",
    # NL rebuild detection
    "nvalchemi::_batch_neighbor_list_rebuild_inplace",
]

# Category 2: Reductions (per-atom → per-system)
REDUCTION_OPS: dict[str, torch.distributed.ReduceOp] = {
    "nvalchemi::compute_kinetic_energy": torch.distributed.ReduceOp.SUM,
    "nvalchemi::compute_temperature": torch.distributed.ReduceOp.SUM,
    "nvalchemi::compute_pressure_tensor": torch.distributed.ReduceOp.SUM,
    "nvalchemi::compute_scalar_pressure": torch.distributed.ReduceOp.SUM,
    "nvalchemi_hooks::compute_kinetic_energy": torch.distributed.ReduceOp.SUM,
    "nvalchemi_hooks::segmented_sum": torch.distributed.ReduceOp.SUM,
    "nvalchemi_hooks::segmented_max": torch.distributed.ReduceOp.MAX,
    "nvalchemi_hooks::segmented_min": torch.distributed.ReduceOp.MIN,
}

_registered = False


def register_shard_wrappers() -> None:
    """Register ShardTensor dispatch handlers for all nvalchemi MD/hook ops.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _registered
    if _registered:
        return
    if register_op_wrappers(PASSTHROUGH_OPS, REDUCTION_OPS):
        _registered = True
