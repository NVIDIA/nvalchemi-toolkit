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

r"""Domain-decomposition-aware Yeh-Berkowitz slab correction.

The slab correction (Yeh-Berkowitz 1999 + Ballenegger 2009 Eq. 29 for
non-neutral systems) removes the spurious dipole interaction between periodic
images along the non-periodic axis of a 2D-periodic (slab) cell. For atom
``i`` in a slab system with non-periodic unit normal :math:`\mathbf{n}`,
projected coordinate :math:`z_i = \mathbf{r}_i \cdot \mathbf{n}`, cell volume
:math:`V`, and non-periodic cell-vector projection :math:`L`:

.. math::

    E_{\text{slab},i} = \frac{2\pi}{V} q_i
        \left[ z_i M - \tfrac{1}{2}(M_2 + Q z_i^2) - \tfrac{Q}{12} L^2 \right]

with the three **global per-system moments**

.. math::

    M = \sum_j q_j z_j, \quad M_2 = \sum_j q_j z_j^2, \quad Q = \sum_j q_j.

Per-atom force, charge gradient, and virial follow analytically (see
``nvalchemiops`` ``slab_kernels``):

.. math::

    \mathbf{F}_{\text{slab},i} = -\frac{4\pi}{V} q_i (M - Q z_i)\,\mathbf{n},
    \quad
    \frac{\partial E_{\text{slab}}}{\partial q_i} = \frac{4\pi}{V}
        \left[ z_i M - \tfrac{1}{2}(M_2 + Q z_i^2) - \tfrac{Q}{12} L^2 \right],
    \quad
    \mathbf{W}_{\text{slab},i} =
        E_{\text{slab},i}(\mathbf{I} - 2\mathbf{n}\mathbf{n}^{T}).

Domain decomposition
--------------------
The only non-local quantities are the three moments ``(M, M_2, Q)``, which are
global per-system sums. Under halo storage each rank holds owned + ghost atoms,
so summing over the padded batch would double-count ghosts. This module mirrors
the PME ``total_charge`` pattern:

* :func:`slab_compute_partial_moments` is a registered custom op
  (``alchemiops::_slab_compute_partial_moments`` / its batched variant) so the
  distributed layer can attach an owned-slice + all-reduce handler. The wrapper
  ``distribution_spec`` lists it in ``custom_ops``.
* On the compiled DD path (where those handlers do not fire) the same
  owned-mask + all-reduce is applied directly via :func:`current_dd_context`.
* :func:`compute_slab_correction_from_moments` consumes the globally-reduced
  moments and runs the per-atom correction in pure Torch (differentiable,
  compile-traceable, machine-precision-equivalent to the warp kernel).
"""

from __future__ import annotations

import math
from typing import Any

import torch

PI = math.pi

__all__ = [
    "slab_normals_and_axis",
    "slab_compute_partial_moments",
    "compute_slab_correction_from_moments",
]


def _prepare_cell(cell: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Coerce ``cell`` to ``(B, 3, 3)`` and return ``(cell, num_systems)``."""
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    return cell, cell.shape[0]


def _prepare_pbc(pbc: torch.Tensor, num_systems: int) -> torch.Tensor:
    """Validate / normalize ``pbc`` to ``(B, 3)`` bool (mirrors nvalchemiops)."""
    if pbc.dtype != torch.bool:
        raise ValueError(f"`pbc` must be a bool tensor, got dtype={pbc.dtype}.")
    if pbc.dim() == 1:
        if pbc.shape[0] != 3:
            raise ValueError(
                f"`pbc` of shape (3,) expected, got shape {tuple(pbc.shape)}."
            )
        if num_systems != 1:
            raise ValueError(
                "Batched slab correction requires `pbc` shape (B, 3); got (3,)."
            )
        pbc = pbc.unsqueeze(0)
    elif pbc.dim() == 2:
        if pbc.shape != (num_systems, 3):
            raise ValueError(
                f"`pbc` of shape ({num_systems}, 3) expected, "
                f"got shape {tuple(pbc.shape)}."
            )
    else:
        raise ValueError(f"`pbc` must be 1D (3,) or 2D (B, 3), got {pbc.dim()}D.")
    return pbc.contiguous()


def slab_normals_and_axis(
    cell: torch.Tensor, pbc: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Per-system slab normal, mask, axis index, and projected height.

    Reproduces the ``nvalchemiops`` slab-kernel geometry in pure Torch.

    A system is a slab when its ``pbc`` row has exactly one ``False`` entry; the
    non-periodic axis is that index. The unit normal follows the cyclic
    convention ``axis 0`` -> :math:`\mathbf{h}_1 \times \mathbf{h}_2`,
    ``axis 1`` -> :math:`\mathbf{h}_2 \times \mathbf{h}_0`,
    ``axis 2`` -> :math:`\mathbf{h}_0 \times \mathbf{h}_1` (so axis-aligned
    right-handed cells give +x/+y/+z).
    :math:`L = |\mathbf{h}_{axis} \cdot \mathbf{n}|` is the non-periodic
    cell-vector projection.

    Returns
    -------
    normal : torch.Tensor ``(B, 3)`` float64
        Per-system unit normal (zeros for non-slab systems).
    is_slab : torch.Tensor ``(B,)`` bool
    height_sq : torch.Tensor ``(B,)`` float64
        :math:`L^2` (zero for non-slab systems).
    inv_vol : torch.Tensor ``(B,)`` float64
        :math:`1/V` (zero for non-slab systems; never used there).
    """
    cell64 = cell.to(torch.float64)

    n_false = (~pbc).sum(dim=1)  # (B,)
    is_slab = n_false == 1
    # Axis index = position of the single False entry (0 for non-slab; masked).
    false_pos = torch.argmax((~pbc).to(torch.int64), dim=1)  # (B,)
    axis = torch.where(is_slab, false_pos, torch.zeros_like(false_pos))

    h0, h1, h2 = cell64[:, 0], cell64[:, 1], cell64[:, 2]  # each (B, 3)
    # Periodic vector pairs per axis (cyclic), and the non-periodic vector.
    cross0 = torch.cross(h1, h2, dim=1)
    cross1 = torch.cross(h2, h0, dim=1)
    cross2 = torch.cross(h0, h1, dim=1)
    axis_e = axis.reshape(-1, 1)
    normal_raw = torch.where(
        axis_e == 0, cross0, torch.where(axis_e == 1, cross1, cross2)
    )
    nonperiodic = torch.where(axis_e == 0, h0, torch.where(axis_e == 1, h1, h2))

    norm = torch.linalg.norm(normal_raw, dim=1, keepdim=True).clamp_min(1e-300)
    normal = normal_raw / norm
    c_dot_n = (nonperiodic * normal).sum(dim=1)  # (B,)
    height_sq = c_dot_n * c_dot_n

    vol = torch.abs(torch.linalg.det(cell64))  # (B,)
    inv_vol = torch.where(is_slab, 1.0 / vol.clamp_min(1e-300), torch.zeros_like(vol))

    slab_f = is_slab.to(torch.float64).reshape(-1, 1)
    normal = normal * slab_f
    height_sq = torch.where(is_slab, height_sq, torch.zeros_like(height_sq))
    return normal, is_slab, height_sq, inv_vol


# ======================================================================
# Stage 1 — partial per-system moments (DD-adaptable custom ops)
# ======================================================================
#
# Output ordering is (mz, mz2, qtotal); each is summed over the atoms this rank
# sees. Registered so the distributed layer can slice the per-atom inputs to
# owned and all-reduce the partials into the true global moments.


@torch.library.custom_op("alchemiops::_slab_compute_partial_moments", mutates_args=())
def _slab_compute_partial_moments(
    z: torch.Tensor, charges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Internal: single-system partial moments :math:`(\sum q z, \sum q z^{2}, \sum q)`."""
    q = charges.to(torch.float64)
    z64 = z.to(torch.float64)
    mz = (q * z64).sum().reshape(1)
    mz2 = (q * z64 * z64).sum().reshape(1)
    qtot = q.sum().reshape(1)
    return mz, mz2, qtot


@_slab_compute_partial_moments.register_fake
def _fake_slab_compute_partial_moments(
    z: torch.Tensor, charges: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e = charges.new_empty((1,), dtype=torch.float64)
    return e, e.clone(), e.clone()


def _setup_ctx_single(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    z, charges = inputs
    ctx.save_for_backward(z, charges)


def _backward_single(
    ctx: Any, g_mz: torch.Tensor, g_mz2: torch.Tensor, g_qtot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    z, charges = ctx.saved_tensors
    q = charges.to(torch.float64)
    z64 = z.to(torch.float64)
    # d(mz)/dz = q ; d(mz2)/dz = 2 q z ; qtot independent of z.
    grad_z = (g_mz * q + g_mz2 * 2.0 * q * z64).to(z.dtype)
    # d(mz)/dq = z ; d(mz2)/dq = z^2 ; d(qtot)/dq = 1.
    grad_q = (g_mz * z64 + g_mz2 * z64 * z64 + g_qtot).to(charges.dtype)
    return grad_z, grad_q


_slab_compute_partial_moments.register_autograd(
    _backward_single, setup_context=_setup_ctx_single
)


@torch.library.custom_op(
    "alchemiops::_batch_slab_compute_partial_moments", mutates_args=()
)
def _batch_slab_compute_partial_moments(
    z: torch.Tensor, charges: torch.Tensor, batch_idx: torch.Tensor, num_systems: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: per-system partial moments via scatter_add."""
    q = charges.to(torch.float64)
    z64 = z.to(torch.float64)
    idx = batch_idx.to(torch.int64)
    mz = torch.zeros(num_systems, dtype=torch.float64, device=charges.device)
    mz2 = torch.zeros_like(mz)
    qtot = torch.zeros_like(mz)
    mz.scatter_add_(0, idx, q * z64)
    mz2.scatter_add_(0, idx, q * z64 * z64)
    qtot.scatter_add_(0, idx, q)
    return mz, mz2, qtot


@_batch_slab_compute_partial_moments.register_fake
def _fake_batch_slab_compute_partial_moments(
    z: torch.Tensor, charges: torch.Tensor, batch_idx: torch.Tensor, num_systems: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    e = charges.new_empty((num_systems,), dtype=torch.float64)
    return e, e.clone(), e.clone()


def _setup_ctx_batch(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    z, charges, batch_idx, _ = inputs
    ctx.save_for_backward(z, charges, batch_idx)


def _backward_batch(
    ctx: Any, g_mz: torch.Tensor, g_mz2: torch.Tensor, g_qtot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, None, None]:
    z, charges, batch_idx = ctx.saved_tensors
    idx = batch_idx.to(torch.int64)
    q = charges.to(torch.float64)
    z64 = z.to(torch.float64)
    g_mz_a = g_mz.index_select(0, idx)
    g_mz2_a = g_mz2.index_select(0, idx)
    g_qtot_a = g_qtot.index_select(0, idx)
    grad_z = (g_mz_a * q + g_mz2_a * 2.0 * q * z64).to(z.dtype)
    grad_q = (g_mz_a * z64 + g_mz2_a * z64 * z64 + g_qtot_a).to(charges.dtype)
    return grad_z, grad_q, None, None


_batch_slab_compute_partial_moments.register_autograd(
    _backward_batch, setup_context=_setup_ctx_batch
)


def _plain_compile_dd(*tensors: Any) -> "tuple[Any, Any] | None":
    """``(n_owned, halo_config)`` when running on plain tensors inside a halo DD
    forward (so the custom-op handlers do not fire), else ``None``.

    Mirrors the PME helper of the same name. Returns ``None`` single-GPU, in
    eager distributed runs (handlers still apply), and for ShardTensor inputs.
    """
    from nvalchemi.distributed._core.compile_routing import (  # noqa: PLC0415
        get_compile_routing,
    )
    from nvalchemi.distributed._core.context import current_dd_context  # noqa: PLC0415
    from nvalchemi.distributed._core.shard_tensor import ShardTensor  # noqa: PLC0415

    ctx = current_dd_context()
    if ctx is None or not getattr(ctx, "is_halo", False):
        return None
    if any(isinstance(t, ShardTensor) for t in tensors):
        return None
    meta = getattr(ctx, "halo_meta", None)
    config = getattr(ctx, "halo_config", None)
    if meta is None or config is None:
        return None
    routing = get_compile_routing()
    n_owned = routing[3] if routing is not None else int(meta.n_owned)
    return n_owned, config


def _owned_mask(values: torch.Tensor, n_owned: Any) -> torch.Tensor:
    """Zero ghost rows (``>= n_owned``) via a fixed-shape mask (compile-stable)."""
    rowidx = torch.arange(values.shape[0], device=values.device)
    return values * (rowidx < n_owned).to(values.dtype)


def slab_compute_partial_moments(
    z: torch.Tensor,
    charges: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    num_systems: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Globally-reduced slab moments ``(M, M_2, Q)`` per system.

    Under halo storage the custom op is intercepted by the distributed layer:
    ``z`` / ``charges`` are sliced to their owned prefix and the partials are
    all-reduced across the mesh. On the compiled path that interception is
    absent, so the same owned-mask + all-reduce is applied here. Single-GPU,
    both are pass-throughs. Returns ``float64`` tensors of shape ``(1,)``
    (single-system) or ``(num_systems,)`` (batched).
    """
    if num_systems is None and batch_idx is not None:
        num_systems = int(batch_idx.max().item()) + 1

    dd = _plain_compile_dd(z, charges)
    if dd is not None:
        from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
            distributed_all_reduce,
        )

        n_owned, config = dd
        owned_q = _owned_mask(charges, n_owned)
        if batch_idx is None:
            mz, mz2, qtot = _slab_compute_partial_moments(z, owned_q)
        else:
            mz, mz2, qtot = _batch_slab_compute_partial_moments(
                z, owned_q, batch_idx, num_systems
            )
        return (
            distributed_all_reduce(mz, config),
            distributed_all_reduce(mz2, config),
            distributed_all_reduce(qtot, config),
        )

    if batch_idx is None:
        return _slab_compute_partial_moments(z, charges)
    return _batch_slab_compute_partial_moments(z, charges, batch_idx, num_systems)


# ======================================================================
# Stage 2 — per-atom correction from globally-reduced moments
# ======================================================================


def compute_slab_correction_from_moments(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    pbc: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""Per-atom slab correction with the moments computed DD-aware internally.

    Computes the projected coordinates :math:`z_i`, reduces the three global moments
    via :func:`slab_compute_partial_moments` (owned-only + all-reduced under
    distribution), then evaluates the per-atom energy/force/charge-grad/virial
    in pure Torch — matching the ``nvalchemiops`` slab kernel formulas. The
    returned tuple ordering is ``(energies, [forces], [charge_grads],
    [virial])`` exactly like ``nvalchemiops.compute_slab_correction``.

    ``energies`` and ``charge_grads`` are ``float64``; ``forces`` and ``virial``
    match ``positions.dtype``. Non-slab systems contribute exactly zero.
    """
    cell, num_systems = _prepare_cell(cell)
    pbc = _prepare_pbc(pbc.to(positions.device), num_systems)

    batched = batch_idx is not None
    if batched:
        bidx = batch_idx.to(torch.int64)
    else:
        bidx = None

    normal, is_slab, height_sq, inv_vol = slab_normals_and_axis(cell, pbc)
    # Per-atom projected coordinate z_i = r_i . n_{system(i)}.
    if batched:
        normal_atom = normal.index_select(0, bidx)  # (N, 3)
        inv_vol_atom = inv_vol.index_select(0, bidx)  # (N,)
        height_sq_atom = height_sq.index_select(0, bidx)  # (N,)
    else:
        normal_atom = normal.expand(positions.shape[0], 3)
        inv_vol_atom = inv_vol.expand(positions.shape[0])
        height_sq_atom = height_sq.expand(positions.shape[0])

    z = (positions.to(torch.float64) * normal_atom).sum(dim=1)  # (N,)

    bidx_arg = batch_idx if batched else None
    mz, mz2, qtot = slab_compute_partial_moments(
        z, charges, batch_idx=bidx_arg, num_systems=num_systems if batched else None
    )

    if batched:
        M = mz.index_select(0, bidx)
        M2 = mz2.index_select(0, bidx)
        Q = qtot.index_select(0, bidx)
    else:
        M = mz.expand(positions.shape[0])
        M2 = mz2.expand(positions.shape[0])
        Q = qtot.expand(positions.shape[0])

    q64 = charges.to(torch.float64)
    bracket = z * M - 0.5 * (M2 + Q * z * z) - (Q / 12.0) * height_sq_atom
    twopi_invV = (2.0 * PI) * inv_vol_atom
    fourpi_invV = (4.0 * PI) * inv_vol_atom

    energies = twopi_invV * q64 * bracket  # (N,) float64

    result: list[torch.Tensor] = [energies]

    if compute_forces:
        # F_i = -(4pi/V) q_i (M - Q z_i) n.
        f_mag = -fourpi_invV * q64 * (M - Q * z)  # (N,) float64
        forces = (f_mag.unsqueeze(1) * normal_atom).to(positions.dtype)
        result.append(forces)

    if compute_charge_gradients:
        charge_grads = fourpi_invV * bracket  # (N,) float64
        result.append(charge_grads)

    if compute_virial:
        # W_i = E_i (I - 2 n n^T), summed per system.
        eye = torch.eye(3, dtype=torch.float64, device=positions.device)
        nnt = normal_atom.unsqueeze(2) * normal_atom.unsqueeze(1)  # (N, 3, 3)
        per_atom_w = energies.reshape(-1, 1, 1) * (eye - 2.0 * nnt)  # (N, 3, 3)
        virial = torch.zeros(
            num_systems, 3, 3, dtype=torch.float64, device=positions.device
        )
        if batched:
            virial.index_add_(0, bidx, per_atom_w)
        else:
            virial[0] = per_atom_w.sum(dim=0)
        result.append(virial.to(positions.dtype))

    if len(result) == 1:
        return result[0]
    return tuple(result)
