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
    Pre-force half: B-A-O-A sequence.  The random seed is a Python ``int``
    launch argument.
langevin_half_step_device_seed
    The same half step with the seed held in a one-element device tensor
    that the op advances by one after every launch, so a replayed CUDA graph
    draws fresh noise on every replay (a Python ``int`` seed is a constant of
    the captured kernel node and repeats the same noise each replay).
langevin_finalize
    Post-force half: final B step.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.library
import warp as wp
from nvalchemiops.dynamics.integrators import (
    langevin_baoab_finalize as _lang_finalize,
)
from nvalchemiops.dynamics.integrators import (
    langevin_baoab_half_step as _lang_half,
)

from nvalchemi.dynamics._ops._bridge import _scalar_type, _vec_type

__all__ = [
    "langevin_half_step",
    "langevin_half_step_device_seed",
    "langevin_finalize",
]


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
    r"""BAOAB Langevin pre-force half-step (B-A-O-A).

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
        Per-system Langevin friction coefficient :math:`\gamma` in 1/time ``[M]``,
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
    positions,
    velocities,
    forces,
    masses,
    dt,
    temperature,
    friction,
    random_seed,
    batch_idx,
) -> None:
    pass


@wp.kernel
def _langevin_baoab_half_step_device_seed_kernel(
    positions: wp.array(dtype=Any),
    velocities: wp.array(dtype=Any),
    forces: wp.array(dtype=Any),
    masses: wp.array(dtype=Any),
    batch_idx: wp.array(dtype=wp.int32),
    dt: wp.array(dtype=Any),
    temperature: wp.array(dtype=Any),
    friction: wp.array(dtype=Any),
    seed: wp.array(dtype=wp.int32),
):
    """Batched BAOAB half step (B-A-O-A) whose RNG seed is read from ``seed[0]``.

    Same arithmetic and RNG stream as ``nvalchemiops``'
    ``_batch_langevin_baoab_half_step_kernel`` (``rand_init(seed, atom)``), so
    ``seed[0] == s`` reproduces the ``random_seed=s`` result bit for bit.
    """
    atom_idx = wp.tid()
    system_id = batch_idx[atom_idx]

    pos = positions[atom_idx]
    vel = velocities[atom_idx]
    force = forces[atom_idx]
    mass = masses[atom_idx]

    dt_val = dt[system_id]
    kT = temperature[system_id]
    gamma = friction[system_id]

    if mass > type(mass)(0.0):
        inv_mass = type(mass)(1.0) / mass
    else:
        inv_mass = type(mass)(0.0)
    half_dt = type(dt_val)(0.5) * dt_val

    vel_step = vel + half_dt * force * inv_mass
    pos_step = pos + half_dt * vel_step

    gamma_dt = gamma * dt_val
    c1 = wp.exp(-gamma_dt)
    c2_sq = kT * (type(kT)(1.0) - c1 * c1) * inv_mass
    c2 = wp.sqrt(c2_sq)

    rng_state = wp.rand_init(seed[0], atom_idx)

    xi = type(vel)(
        type(kT)(wp.randn(rng_state)),
        type(kT)(wp.randn(rng_state)),
        type(kT)(wp.randn(rng_state)),
    )
    vel_step = c1 * vel_step + c2 * xi

    pos_step2 = pos_step + half_dt * vel_step

    positions[atom_idx] = pos_step2
    velocities[atom_idx] = vel_step


@wp.kernel
def _advance_seed_kernel(seed: wp.array(dtype=wp.int32)):
    """Increment the device seed by one (single thread, runs after the half step)."""
    seed[0] = seed[0] + 1


@torch.library.custom_op(
    "nvalchemi::langevin_half_step_device_seed",
    mutates_args={"positions", "velocities", "seed"},
)
def langevin_half_step_device_seed(
    positions: torch.Tensor,
    velocities: torch.Tensor,
    forces: torch.Tensor,
    masses: torch.Tensor,
    dt: torch.Tensor,
    temperature: torch.Tensor,
    friction: torch.Tensor,
    seed: torch.Tensor,
    batch_idx: torch.Tensor,
) -> None:
    """BAOAB Langevin pre-force half-step (B-A-O-A) with a device-side seed.

    Identical to :func:`langevin_half_step` except that the RNG seed is
    ``seed[0]`` (read by the kernel at run time) and ``seed[0]`` is advanced
    by one after the launch.  Because nothing about the seed is baked into the
    launch, capturing the op in a CUDA graph and replaying it draws a new
    noise sample per replay, exactly like calling the eager op with
    ``random_seed + step``.  Modifies *positions*, *velocities* and *seed*
    in-place.

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
        Per-system kT in energy units ``[M]``, same dtype.
    friction : torch.Tensor
        Per-system friction ``[M]``, same dtype.
    seed : torch.Tensor
        One-element ``int32`` tensor on the same device: the seed of THIS
        call, advanced by one afterwards.
    batch_idx : torch.Tensor
        Per-atom system index ``[N]``, int32, non-decreasing.
    """
    dtype = positions.dtype
    vec_t = _vec_type(dtype)
    scl_t = _scalar_type(dtype)
    wp_seed = wp.from_torch(seed, dtype=wp.int32)
    device = wp_seed.device
    wp.launch(
        _langevin_baoab_half_step_device_seed_kernel,
        dim=positions.shape[0],
        inputs=[
            wp.from_torch(positions, dtype=vec_t),
            wp.from_torch(velocities, dtype=vec_t),
            wp.from_torch(forces, dtype=vec_t),
            wp.from_torch(masses, dtype=scl_t),
            wp.from_torch(batch_idx, dtype=wp.int32),
            wp.from_torch(dt, dtype=scl_t),
            wp.from_torch(temperature, dtype=scl_t),
            wp.from_torch(friction, dtype=scl_t),
            wp_seed,
        ],
        device=device,
    )
    wp.launch(_advance_seed_kernel, dim=1, inputs=[wp_seed], device=device)


@langevin_half_step_device_seed.register_fake
def _langevin_half_step_device_seed_fake(
    positions,
    velocities,
    forces,
    masses,
    dt,
    temperature,
    friction,
    seed,
    batch_idx,
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
