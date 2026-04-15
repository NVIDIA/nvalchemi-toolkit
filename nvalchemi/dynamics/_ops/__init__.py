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
"""
PyTorch bindings for nvalchemi-toolkit-ops Warp kernels.

Each sub-module wraps a family of Warp GPU kernels as callable Python
functions that accept and return ``torch.Tensor`` objects via zero-copy
``wp.from_torch`` conversion.

Modules
-------
_bridge
    Shared utilities: state-batch construction, per-system tensor
    broadcasting, and Warp dtype helpers.
velocity_verlet
    NVE velocity Verlet: ``vv_position_update``, ``vv_velocity_finalize``.
langevin
    BAOAB Langevin NVT: ``langevin_half_step``, ``langevin_finalize``.
nose_hoover
    Nosé-Hoover chain NVT: ``nhc_compute_masses``, ``nhc_chain_update``,
    ``nhc_velocity_half_step``, ``nhc_position_update``.
npt_nph
    NPT/NPH barostat and pressure ops.
fire
    FIRE and FIRE2 optimizer step functions.
cell_align
    Cell alignment to upper-triangular form for variable-cell optimization.
thermostat_utils
    Velocity initialization, COM removal, kinetic temperature.
"""
