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
"""PyTorch custom ops wrapping the Warp Lennard-Jones interaction kernels.

Exposes two ``torch.library`` operators:

``nvalchemi::lj_energy_forces_batch``
    Compute per-atom LJ energies and forces for a batched collection of
    atomic systems using the dense neighbor-matrix format.

``nvalchemi::lj_energy_forces_virial_batch``
    Same as above, additionally returning the per-system virial tensor
    needed for NPT/NPH pressure coupling.

Both operators are registered with fake implementations so they work with
``torch.compile``.  They do **not** register autograd formulas — forces and
virials are computed analytically inside the Warp kernels.

Sign Convention
---------------
The LJ kernels accumulate the virial with the convention ``W = -Σ r_ij ⊗ F_ij``
(negative).  The MTK NPT/NPH integrator expects the *positive* convention
``+Σ r_ij ⊗ F_ij``; :class:`~nvalchemi.models.lj.LennardJonesModelWrapper`
negates the output before writing to ``batch.stresses``.

Variable-cell optimizers (:class:`~nvalchemi.dynamics.optimizers.FIRE2VariableCell`,
:class:`~nvalchemi.dynamics.optimizers.FIREVariableCell`) require the mechanical
stress tensor ``σ = W_phys / V``; they divide ``batch.stresses`` by the cell volume
internally before calling ``stress_to_cell_force``.

Notes
-----
* Internal math is performed in float64 for numerical stability; inputs and
  outputs match the dtype of ``positions`` (float32 or float64).
* The neighbor matrix must use **global** atom indices (0 … N_total−1).
* ``fill_value`` is the sentinel used to pad short rows in the neighbor
  matrix; pass ``batch.num_nodes`` (total atoms across all systems).
"""

from __future__ import annotations

import torch
import warp as wp
from torch import Tensor

# ---------------------------------------------------------------------------
# Dtype helpers (mirrors nvalchemi.dynamics._ops._bridge)
# ---------------------------------------------------------------------------


def _vec_type(dtype: torch.dtype) -> type:
    return wp.vec3d if dtype == torch.float64 else wp.vec3f


def _mat_type(dtype: torch.dtype) -> type:
    return wp.mat33d if dtype == torch.float64 else wp.mat33f


def _scalar_type(dtype: torch.dtype) -> type:
    return wp.float64 if dtype == torch.float64 else wp.float32


# ---------------------------------------------------------------------------
# Warp scalar-parameter cache
# ---------------------------------------------------------------------------

_WP_PARAM_CACHE: dict = {}
"""Module-level cache for single-element Warp arrays (epsilon, sigma, cutoff,
switch_width).  Keyed by (epsilon, sigma, cutoff, switch_width, scl_t, wp_dev)
so the same parameters are never reallocated."""


def _get_cached_wp_params(
    epsilon: float,
    sigma: float,
    cutoff: float,
    switch_width: float,
    scl_t: type,
    wp_dev: str,
) -> dict:
    """Return (or create) cached single-element wp.array objects for LJ scalars."""
    key = (epsilon, sigma, cutoff, switch_width, scl_t, wp_dev)
    if key not in _WP_PARAM_CACHE:
        import warp as wp  # noqa: PLC0415

        _WP_PARAM_CACHE[key] = {
            "epsilon": wp.array([epsilon], dtype=scl_t, device=wp_dev),
            "sigma": wp.array([sigma], dtype=scl_t, device=wp_dev),
            "cutoff": wp.array([cutoff], dtype=scl_t, device=wp_dev),
            "switch": wp.array([switch_width], dtype=scl_t, device=wp_dev),
        }
    return _WP_PARAM_CACHE[key]


# ---------------------------------------------------------------------------
# lj_energy_forces_batch
# ---------------------------------------------------------------------------


@torch.library.custom_op("nvalchemi::lj_energy_forces_batch", mutates_args={})
def lj_energy_forces_batch(
    positions: Tensor,
    cells: Tensor,
    neighbor_matrix: Tensor,
    neighbor_shifts: Tensor,
    num_neighbors: Tensor,
    batch_idx: Tensor,
    fill_value: int,
    epsilon: float,
    sigma: float,
    cutoff: float,
    switch_width: float,
    half_list: bool,
) -> tuple[Tensor, Tensor]:
    """Compute LJ energies and forces for a batch of systems.

    Parameters
    ----------
    positions : Tensor, shape (N, 3), float32 or float64
        Concatenated atom positions for all systems.
    cells : Tensor, shape (B, 3, 3), same dtype as positions
        Unit-cell matrices, one per system.  Pass identity matrices for
        non-periodic systems (shifts will be zero so the cell is unused).
    neighbor_matrix : Tensor, shape (N, max_neighbors), int32
        Global atom indices of neighbors, padded with ``fill_value``.
    neighbor_shifts : Tensor, shape (N, max_neighbors, 3), int32
        Integer lattice-shift vectors for each neighbor entry.  Pass zeros
        for non-periodic systems.
    num_neighbors : Tensor, shape (N,), int32
        Number of valid neighbors per atom.
    batch_idx : Tensor, shape (N,), int32
        System index (0 … B−1) for each atom.
    fill_value : int
        Padding sentinel used in ``neighbor_matrix`` rows; typically
        ``batch.num_nodes`` (total atoms).
    epsilon : float
        LJ well-depth parameter (eV or chosen energy unit).
    sigma : float
        LJ zero-crossing distance (Å or chosen length unit).
    cutoff : float
        Interaction cutoff radius (same length unit as positions).
    switch_width : float
        Width of the C2-continuous switching region applied between
        ``cutoff - switch_width`` and ``cutoff``.  Use ``0.0`` to disable.
    half_list : bool
        ``True`` when the neighbor matrix is a half list (each pair once,
        Newton's third law applied); ``False`` for full lists.

    Returns
    -------
    atomic_energies : Tensor, shape (N,)
        Per-atom LJ energies.  Sum over atoms in each system for total energy.
    forces : Tensor, shape (N, 3)
        Per-atom forces.
    """
    from nvalchemiops.interactions.lj import (
        _batch_lj_energy_forces_matrix_kernel_overload,
    )

    N = positions.shape[0]
    dtype = positions.dtype
    vec_t = _vec_type(dtype)
    mat_t = _mat_type(dtype)
    scl_t = _scalar_type(dtype)

    dev = positions.device
    wp_dev = f"cuda:{dev.index}" if dev.type == "cuda" else "cpu"

    atomic_energies = torch.zeros(N, dtype=dtype, device=dev)
    forces = torch.zeros(N, 3, dtype=dtype, device=dev)

    wp_params = _get_cached_wp_params(
        epsilon, sigma, cutoff, switch_width, scl_t, wp_dev
    )

    wp.launch(
        _batch_lj_energy_forces_matrix_kernel_overload[scl_t],
        dim=N,
        inputs=[
            wp.from_torch(positions.contiguous(), vec_t),
            wp.from_torch(cells.contiguous(), mat_t),
            # neighbor_matrix: (N, max_neighbors) int32 → wp.array2d(int32)
            wp.from_torch(neighbor_matrix.contiguous(), wp.int32),
            # neighbor_shifts: (N, max_neighbors, 3) int32 → wp.array2d(vec3i)
            wp.from_torch(neighbor_shifts.contiguous(), wp.vec3i),
            wp.from_torch(num_neighbors.contiguous(), wp.int32),
            wp.from_torch(batch_idx.contiguous(), wp.int32),
            wp_params["epsilon"],
            wp_params["sigma"],
            wp_params["cutoff"],
            wp_params["switch"],
            wp.bool(half_list),
            wp.int32(fill_value),
            wp.from_torch(atomic_energies, scl_t),
            wp.from_torch(forces.contiguous(), vec_t),
        ],
        device=wp_dev,
    )

    return atomic_energies, forces


@lj_energy_forces_batch.register_fake
def _lj_energy_forces_batch_fake(
    positions: Tensor,
    cells: Tensor,
    neighbor_matrix: Tensor,
    neighbor_shifts: Tensor,
    num_neighbors: Tensor,
    batch_idx: Tensor,
    fill_value: int,
    epsilon: float,
    sigma: float,
    cutoff: float,
    switch_width: float,
    half_list: bool,
) -> tuple[Tensor, Tensor]:
    N = positions.shape[0]
    return (
        torch.empty(N, dtype=positions.dtype, device=positions.device),
        torch.empty(N, 3, dtype=positions.dtype, device=positions.device),
    )


# ---------------------------------------------------------------------------
# lj_energy_forces_virial_batch
# ---------------------------------------------------------------------------


@torch.library.custom_op("nvalchemi::lj_energy_forces_virial_batch", mutates_args={})
def lj_energy_forces_virial_batch(
    positions: Tensor,
    cells: Tensor,
    neighbor_matrix: Tensor,
    neighbor_shifts: Tensor,
    num_neighbors: Tensor,
    batch_idx: Tensor,
    fill_value: int,
    epsilon: float,
    sigma: float,
    cutoff: float,
    switch_width: float,
    half_list: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute LJ energies, forces, and per-system virials.

    Parameters and first two return values are identical to
    :func:`lj_energy_forces_batch`.

    Returns
    -------
    atomic_energies : Tensor, shape (N,)
    forces : Tensor, shape (N, 3)
    virials : Tensor, shape (B, 9)
        Flattened per-system virial tensors in row-major order
        ``[xx, xy, xz, yx, yy, yz, zx, zy, zz]`` with the sign convention
        ``W = -Σ r_ij ⊗ F_ij``.
    """
    from nvalchemiops.interactions.lj import (
        _batch_lj_energy_forces_virial_matrix_kernel_overload,
    )

    N = positions.shape[0]
    B = cells.shape[0]
    dtype = positions.dtype
    vec_t = _vec_type(dtype)
    mat_t = _mat_type(dtype)
    scl_t = _scalar_type(dtype)

    dev = positions.device
    wp_dev = f"cuda:{dev.index}" if dev.type == "cuda" else "cpu"

    atomic_energies = torch.zeros(N, dtype=dtype, device=dev)
    forces = torch.zeros(N, 3, dtype=dtype, device=dev)
    virials = torch.zeros(B, 9, dtype=dtype, device=dev)

    wp_params = _get_cached_wp_params(
        epsilon, sigma, cutoff, switch_width, scl_t, wp_dev
    )

    wp.launch(
        _batch_lj_energy_forces_virial_matrix_kernel_overload[scl_t],
        dim=N,
        inputs=[
            wp.from_torch(positions.contiguous(), vec_t),
            wp.from_torch(cells.contiguous(), mat_t),
            wp.from_torch(neighbor_matrix.contiguous(), wp.int32),
            wp.from_torch(neighbor_shifts.contiguous(), wp.vec3i),
            wp.from_torch(num_neighbors.contiguous(), wp.int32),
            wp.from_torch(batch_idx.contiguous(), wp.int32),
            wp_params["epsilon"],
            wp_params["sigma"],
            wp_params["cutoff"],
            wp_params["switch"],
            wp.bool(half_list),
            wp.int32(fill_value),
            wp.from_torch(atomic_energies, scl_t),
            wp.from_torch(forces.contiguous(), vec_t),
            # virials: (B, 9) → wp.array2d(dtype=scl_t)
            wp.from_torch(virials.contiguous(), scl_t),
        ],
        device=wp_dev,
    )

    return atomic_energies, forces, virials


@lj_energy_forces_virial_batch.register_fake
def _lj_energy_forces_virial_batch_fake(
    positions: Tensor,
    cells: Tensor,
    neighbor_matrix: Tensor,
    neighbor_shifts: Tensor,
    num_neighbors: Tensor,
    batch_idx: Tensor,
    fill_value: int,
    epsilon: float,
    sigma: float,
    cutoff: float,
    switch_width: float,
    half_list: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    N = positions.shape[0]
    B = cells.shape[0]
    return (
        torch.empty(N, dtype=positions.dtype, device=positions.device),
        torch.empty(N, 3, dtype=positions.dtype, device=positions.device),
        torch.empty(B, 9, dtype=positions.dtype, device=positions.device),
    )


# ---------------------------------------------------------------------------
# _into variants: accept pre-allocated mutable output buffers
# ---------------------------------------------------------------------------


@torch.library.custom_op(
    "nvalchemi::lj_energy_forces_batch_into",
    mutates_args={"atomic_energies", "forces"},
)
def lj_energy_forces_batch_into(
    positions: Tensor,
    cells: Tensor,
    neighbor_matrix: Tensor,
    neighbor_shifts: Tensor,
    num_neighbors: Tensor,
    batch_idx: Tensor,
    fill_value: int,
    epsilon: float,
    sigma: float,
    cutoff: float,
    switch_width: float,
    half_list: bool,
    atomic_energies: Tensor,
    forces: Tensor,
) -> None:
    """In-place LJ energy+force kernel writing into pre-allocated output buffers.

    ``atomic_energies`` and ``forces`` are zeroed then filled by the Warp kernel.
    The caller is responsible for allocating correctly-shaped tensors.
    """
    from nvalchemiops.interactions.lj import (  # noqa: PLC0415
        _batch_lj_energy_forces_matrix_kernel_overload,
    )

    N = positions.shape[0]
    dtype = positions.dtype
    vec_t = _vec_type(dtype)
    mat_t = _mat_type(dtype)
    scl_t = _scalar_type(dtype)

    dev = positions.device
    wp_dev = f"cuda:{dev.index}" if dev.type == "cuda" else "cpu"

    atomic_energies.zero_()
    forces.zero_()

    wp_params = _get_cached_wp_params(
        epsilon, sigma, cutoff, switch_width, scl_t, wp_dev
    )

    wp.launch(
        _batch_lj_energy_forces_matrix_kernel_overload[scl_t],
        dim=N,
        inputs=[
            wp.from_torch(positions.contiguous(), vec_t),
            wp.from_torch(cells.contiguous(), mat_t),
            wp.from_torch(neighbor_matrix.contiguous(), wp.int32),
            wp.from_torch(neighbor_shifts.contiguous(), wp.vec3i),
            wp.from_torch(num_neighbors.contiguous(), wp.int32),
            wp.from_torch(batch_idx.contiguous(), wp.int32),
            wp_params["epsilon"],
            wp_params["sigma"],
            wp_params["cutoff"],
            wp_params["switch"],
            wp.bool(half_list),
            wp.int32(fill_value),
            wp.from_torch(atomic_energies, scl_t),
            wp.from_torch(forces.contiguous(), vec_t),
        ],
        device=wp_dev,
    )


@lj_energy_forces_batch_into.register_fake
def _lj_energy_forces_batch_into_fake(
    positions: Tensor,
    cells: Tensor,
    neighbor_matrix: Tensor,
    neighbor_shifts: Tensor,
    num_neighbors: Tensor,
    batch_idx: Tensor,
    fill_value: int,
    epsilon: float,
    sigma: float,
    cutoff: float,
    switch_width: float,
    half_list: bool,
    atomic_energies: Tensor,
    forces: Tensor,
) -> None:
    return None


@torch.library.custom_op(
    "nvalchemi::lj_energy_forces_virial_batch_into",
    mutates_args={"atomic_energies", "forces", "virials"},
)
def lj_energy_forces_virial_batch_into(
    positions: Tensor,
    cells: Tensor,
    neighbor_matrix: Tensor,
    neighbor_shifts: Tensor,
    num_neighbors: Tensor,
    batch_idx: Tensor,
    fill_value: int,
    epsilon: float,
    sigma: float,
    cutoff: float,
    switch_width: float,
    half_list: bool,
    atomic_energies: Tensor,
    forces: Tensor,
    virials: Tensor,
) -> None:
    """In-place LJ energy+force+virial kernel writing into pre-allocated buffers.

    ``atomic_energies``, ``forces``, and ``virials`` are zeroed then filled.
    ``virials`` must have shape ``(B, 9)``.
    """
    from nvalchemiops.interactions.lj import (  # noqa: PLC0415
        _batch_lj_energy_forces_virial_matrix_kernel_overload,
    )

    N = positions.shape[0]
    dtype = positions.dtype
    vec_t = _vec_type(dtype)
    mat_t = _mat_type(dtype)
    scl_t = _scalar_type(dtype)

    dev = positions.device
    wp_dev = f"cuda:{dev.index}" if dev.type == "cuda" else "cpu"

    atomic_energies.zero_()
    forces.zero_()
    virials.zero_()

    wp_params = _get_cached_wp_params(
        epsilon, sigma, cutoff, switch_width, scl_t, wp_dev
    )

    wp.launch(
        _batch_lj_energy_forces_virial_matrix_kernel_overload[scl_t],
        dim=N,
        inputs=[
            wp.from_torch(positions.contiguous(), vec_t),
            wp.from_torch(cells.contiguous(), mat_t),
            wp.from_torch(neighbor_matrix.contiguous(), wp.int32),
            wp.from_torch(neighbor_shifts.contiguous(), wp.vec3i),
            wp.from_torch(num_neighbors.contiguous(), wp.int32),
            wp.from_torch(batch_idx.contiguous(), wp.int32),
            wp_params["epsilon"],
            wp_params["sigma"],
            wp_params["cutoff"],
            wp_params["switch"],
            wp.bool(half_list),
            wp.int32(fill_value),
            wp.from_torch(atomic_energies, scl_t),
            wp.from_torch(forces.contiguous(), vec_t),
            wp.from_torch(virials.contiguous(), scl_t),
        ],
        device=wp_dev,
    )


@lj_energy_forces_virial_batch_into.register_fake
def _lj_energy_forces_virial_batch_into_fake(
    positions: Tensor,
    cells: Tensor,
    neighbor_matrix: Tensor,
    neighbor_shifts: Tensor,
    num_neighbors: Tensor,
    batch_idx: Tensor,
    fill_value: int,
    epsilon: float,
    sigma: float,
    cutoff: float,
    switch_width: float,
    half_list: bool,
    atomic_energies: Tensor,
    forces: Tensor,
    virials: Tensor,
) -> None:
    return None
