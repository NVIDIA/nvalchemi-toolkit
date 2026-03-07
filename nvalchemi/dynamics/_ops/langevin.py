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
PyTorch bindings for BAOAB Langevin NVT integrator kernels.

Wraps :mod:`nvalchemiops.dynamics.integrators.langevin` as
``torch.library.custom_op`` operations, enabling correct behaviour
under ``torch.compile`` and PyTorch's autograd infrastructure.

The BAOAB splitting scheme applies:

* **B** — half velocity kick from forces
* **A** — half-step position drift
* **O** — Ornstein-Uhlenbeck stochastic velocity update (thermostat)
* **A** — half-step position drift
* **B** — half velocity kick (deferred to post-force evaluation)

Functions
---------
langevin_half_step
    Pre-force half: B-A-O-A sequence.
langevin_finalize
    Post-force half: final B step.
"""

from __future__ import annotations

import torch
import torch.library
import warp as wp
from nvalchemiops.dynamics.integrators import (
    langevin_baoab_half_step as _lang_half,
    langevin_baoab_finalize as _lang_finalize,
)

from nvalchemi.dynamics._ops._bridge import _scalar_type, _vec_type

__all__ = ["langevin_half_step", "langevin_finalize"]


@torch.library.custom_op(
    "nvalchemi::langevin_half_step",
    mutates_args={"positions", "velocities"},
)
def langevin_half_step(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    temperature: torch.Tensor,
    friction: torch.Tensor,
    random_seed: int,
    batch_idx: torch.Tensor,
) -> None:
    """BAOAB Langevin pre-force half-step (B-A-O-A).

    Applies half velocity kick, half-step drift, Ornstein-Uhlenbeck
    thermostat, and another half-step drift. Modifies *positions* and
    *velocities* in-place.  The RNG state is managed internally by the
    kernel using *random_seed* combined with atom indices.

    Parameters
    ----------
    positions : torch.Tensor
        Atomic positions ``[N, 3]``, float32 or float64.
    velocities : torch.Tensor
        Atomic velocities ``[N, 3]``, same dtype.
    forces : torch.Tensor
        Atomic forces ``[N, 3]``, same dtype.
    masses : torch.Tensor
        Atomic masses ``[N]``, same dtype.
    dt : torch.Tensor
        Per-system timestep ``[M]``, same dtype.
    temperature : torch.Tensor
        Per-system temperature in Kelvin ``[M]``, same dtype.
    friction : torch.Tensor
        Per-system Langevin friction coefficient γ in 1/time ``[M]``,
        same dtype.
    random_seed : int
        Global RNG seed.  The kernel combines this with atom/step indices
        to produce unique random numbers per atom per step.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    """
    dtype = positions.dtype
    vec_t = _vec_type(dtype)
    scl_t = _scalar_type(dtype)
    _lang_half(
        wp.from_torch(positions, dtype=vec_t),
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(forces, dtype=vec_t),
        wp.from_torch(masses, dtype=scl_t),
        wp.from_torch(dt, dtype=scl_t),
        wp.from_torch(temperature, dtype=scl_t),
        wp.from_torch(friction, dtype=scl_t),
        random_seed,
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
    )


@langevin_half_step.register_fake
def _langevin_half_step_fake(
    positions, velocities, forces, masses, dt, temperature, friction, random_seed, batch_idx
) -> None:
    pass


@torch.library.custom_op(
    "nvalchemi::langevin_finalize",
    mutates_args={"velocities"},
)
def langevin_finalize(
    velocities: torch.Tensor,
    forces_new: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """BAOAB Langevin post-force final step (B).

    Applies the final half velocity kick using new forces.
    Modifies *velocities* in-place.

    Parameters
    ----------
    velocities : torch.Tensor
        Half-step velocities ``[N, 3]``, float32 or float64.
    forces_new : torch.Tensor
        Forces at updated positions ``[N, 3]``, same dtype.
    masses : torch.Tensor
        Atomic masses ``[N]``, same dtype.
    dt : torch.Tensor
        Per-system timestep ``[M]``, same dtype.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    """
    dtype = velocities.dtype
    vec_t = _vec_type(dtype)
    scl_t = _scalar_type(dtype)
    _lang_finalize(
        wp.from_torch(velocities, dtype=vec_t),
        wp.from_torch(forces_new, dtype=vec_t),
        wp.from_torch(masses, dtype=scl_t),
        wp.from_torch(dt, dtype=scl_t),
        batch_idx=wp.from_torch(batch_idx, dtype=wp.int32),
    )


@langevin_finalize.register_fake
def _langevin_finalize_fake(velocities, forces_new, masses, dt, batch_idx) -> None:
    pass
