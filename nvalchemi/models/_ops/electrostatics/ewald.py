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

r"""Ewald reciprocal-space staged bindings.

Splits the monolithic ``nvalchemiops`` ``ewald_reciprocal_space`` into two
stages so distributed callers can insert a cross-rank reduction between them:

- :func:`ewald_compute_partial_structure_factors` — per-rank partial
  green-weighted structure factors ``S̃(k) = G(k)·Σᵢ qᵢ exp(i k·rᵢ)`` and total
  charge ``Q = Σᵢ qᵢ`` (runs only the reciprocal ``fill`` warp kernel).
  Registered as a ``torch.library.custom_op`` so the distributed layer can
  intercept it and run *owned-slice × fill × all-reduce* (each atom is owned by
  one rank, and ``G(k)`` depends only on globally-replicated ``k², V, α`` so it
  is identical on every rank — the reduce commutes with the green weight).
- :func:`ewald_reciprocal_space_from_structure_factors` — per-atom reciprocal
  energy + optional forces / charge gradients / virial from the globally-reduced
  structure factors. Runs ``fill`` to recover ``cos(k·r)`` / ``sin(k·r)`` of the
  local atoms, then ``compute`` / ``virial`` consuming the externally supplied
  ``S̃(k)``, plus the public ``ewald_energy_corrections`` (which accepts an
  explicit ``total_charge``).
"""

from __future__ import annotations

import math
from typing import Any

import torch
import warp as wp
from nvalchemiops.interactions.electrostatics._factory_common import _DerivState
from nvalchemiops.interactions.electrostatics.ewald_kernels import (
    BATCH_BLOCK_SIZE,
    EIGHTPI,
)
from nvalchemiops.interactions.electrostatics.ewald_recip_factory import (
    alloc_ewald_recip_sentinels,
    get_ewald_recip_kernel,
)
from nvalchemiops.torch.interactions.electrostatics._util import _InjectChargeGrad
from nvalchemiops.torch.interactions.electrostatics.ewald import (
    ensure_electrostatics_ops_registered,
    ewald_energy_corrections,
    ewald_energy_corrections_batch,
)
from nvalchemiops.torch.types import (
    get_wp_dtype,
    get_wp_mat_dtype,
    get_wp_vec_dtype,
)

_PI = math.pi

__all__ = [
    "ewald_compute_partial_structure_factors",
    "ewald_reciprocal_space_from_structure_factors",
    "ewald_reciprocal_contribution",
]


# ======================================================================
# Warp interop helpers
# ======================================================================


def _wp(tensor: torch.Tensor, dtype):
    """``wp.from_torch`` on a detached, contiguous view (this module owns autograd)."""
    return wp.from_torch(tensor.detach().contiguous(), dtype=dtype, requires_grad=False)


def _scoped_stream(device: torch.device):
    """Bind Warp's stream to PyTorch's current CUDA stream (graph-capture safe)."""
    if device.type != "cuda":
        from contextlib import nullcontext

        return nullcontext()
    return wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream(device)))


def _atom_ranges(
    batch_idx: torch.Tensor, num_systems: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-system ``(atom_start, atom_end)`` int32 prefix sums."""
    counts = torch.bincount(batch_idx.to(torch.int64), minlength=num_systems)
    atom_end = torch.cumsum(counts, dim=0).to(torch.int32)
    atom_start = torch.cat(
        [torch.zeros(1, device=batch_idx.device, dtype=torch.int32), atom_end[:-1]]
    )
    return atom_start, atom_end


def _normalize(
    cell: torch.Tensor, k_vectors: torch.Tensor, num_systems: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Coerce ``cell`` to ``(S, 3, 3)`` and ``k_vectors`` to ``(S, K, 3)``."""
    if cell.dim() == 2:
        cell = cell.reshape(1, 3, 3)
    if k_vectors.dim() == 2:
        k_vectors = k_vectors.reshape(1, -1, 3)
    if k_vectors.shape[0] == 1 and num_systems > 1:
        k_vectors = k_vectors.expand(num_systems, -1, 3).contiguous()
    return cell, k_vectors


def _run_fill(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell_3d: torch.Tensor,
    k_vectors_2d: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Launch the reciprocal ``fill`` kernel over the *local* atoms.

    Returns Torch tensors ``(cos_kr, sin_kr, real_sf, imag_sf, total_charge)``
    where ``real_sf`` / ``imag_sf`` are the green-weighted partial structure
    factors ``S̃(k)`` with shape ``(S, K)``, ``total_charge`` is ``(S,)``, and
    ``cos_kr`` / ``sin_kr`` are ``(K, N)``. ``cell`` enters only the Green's
    ``1/V`` factor (detached); the differentiable cell path is owned by the
    corrections / virial kernels, not the fill.
    """
    num_k = k_vectors_2d.shape[-2]
    num_atoms = positions.shape[0]
    num_systems = cell_3d.shape[0]
    input_dtype = positions.dtype
    device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    batched = batch_idx is not None

    cos_kr = torch.empty(num_k, num_atoms, device=positions.device, dtype=torch.float64)
    sin_kr = torch.empty_like(cos_kr)
    real_sf = torch.zeros(
        num_systems, num_k, device=positions.device, dtype=torch.float64
    )
    imag_sf = torch.zeros_like(real_sf)
    total_charge = torch.zeros(
        num_systems, device=positions.device, dtype=torch.float64
    )
    if num_atoms == 0 or num_k == 0:
        return cos_kr, sin_kr, real_sf, imag_sf, total_charge

    bundle = get_ewald_recip_kernel(
        wp_scalar, batched=batched, deriv_state=_DerivState.E, order="forward"
    )
    wp_pos = _wp(positions, wp_vec)
    wp_chg = _wp(charges, wp_scalar)
    wp_cell = _wp(cell_3d, wp_mat)
    wp_alpha = _wp(alpha, wp_scalar)
    wp_cos = _wp(cos_kr, wp.float64)
    wp_sin = _wp(sin_kr, wp.float64)
    with _scoped_stream(positions.device):
        if batched:
            atom_start, atom_end = _atom_ranges(batch_idx, num_systems)
            max_atoms = int((atom_end - atom_start).max().item()) if num_atoms else 0
            max_blocks = max((max_atoms + BATCH_BLOCK_SIZE - 1) // BATCH_BLOCK_SIZE, 1)
            wp.launch(
                bundle.fill,
                dim=(num_k, num_systems, max_blocks),
                inputs=[
                    wp_pos,
                    wp_chg,
                    _wp(k_vectors_2d, wp_vec),
                    wp_cell,
                    wp_alpha,
                    _wp(atom_start, wp.int32),
                    _wp(atom_end, wp.int32),
                    _wp(total_charge, wp.float64),
                    wp_cos,
                    wp_sin,
                    _wp(real_sf, wp.float64),
                    _wp(imag_sf, wp.float64),
                ],
                device=device,
            )
        else:
            kv_1d = _wp(k_vectors_2d.reshape(num_k, 3), wp_vec)
            wp.launch(
                bundle.fill,
                dim=num_k,
                inputs=[
                    wp_pos,
                    wp_chg,
                    kv_1d,
                    wp_cell,
                    wp_alpha,
                    _wp(total_charge.reshape(1), wp.float64),
                    wp_cos,
                    wp_sin,
                    _wp(real_sf.reshape(num_k), wp.float64),
                    _wp(imag_sf.reshape(num_k), wp.float64),
                ],
                device=device,
            )
    return cos_kr, sin_kr, real_sf, imag_sf, total_charge


# ======================================================================
# Stage 1 — partial green-weighted structure factors (autograd custom ops)
# ======================================================================
#
# Forward runs the reciprocal ``fill`` kernel; backward is the fill adjoint
# dL/dS̃(k) -> dL/d{positions, charges}. ``cell`` / ``k_vectors`` / ``alpha``
# enter S̃(k) only through the detached Green's function, so they are
# non-differentiable here; the cell first order is handled by the corrections
# and virial kernels in stage 2.


def _green(
    k_vectors_2d: torch.Tensor, volume: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """``G[s,k] = 8π/V_s · exp(-k²/4α_s²) / k²`` (masked ``k² < 1e-10``), shape ``(S, K)``."""
    ksq = (k_vectors_2d * k_vectors_2d).sum(-1)  # (S, K)
    a = alpha.reshape(-1, 1).to(torch.float64)  # (S, 1)
    g = (
        (EIGHTPI / volume.reshape(-1, 1).to(torch.float64))
        * torch.exp(-ksq * (0.25 / (a * a)))
        / ksq
    )
    return torch.where(ksq < 1e-10, torch.zeros_like(g), g)


def _stage1_backward(
    g_re: torch.Tensor,
    g_im: torch.Tensor,
    g_tot: torch.Tensor,
    positions: torch.Tensor,
    charges: torch.Tensor,
    k_vectors_2d: torch.Tensor,
    volume: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None,
    num_systems: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill adjoint: gradients on ``(real_sf, imag_sf, total_charge)`` -> ``(positions, charges)``.

    ``dL/dq_j = Σ_k G_k (g_re_k cos_jk + g_im_k sin_jk) + g_tot[s]``;
    ``dL/dr_j = q_j Σ_k k_vec_k · G_k (−g_re_k sin_jk + g_im_k cos_jk)``.
    """
    green = _green(k_vectors_2d, volume, alpha)  # (S, K)
    g_re = g_re.reshape(num_systems, -1).to(torch.float64)
    g_im = g_im.reshape(num_systems, -1).to(torch.float64)
    g_tot = g_tot.reshape(-1).to(torch.float64)
    grad_pos = torch.zeros_like(positions, dtype=torch.float64)
    grad_chg = torch.zeros_like(charges, dtype=torch.float64)
    for s in range(num_systems):
        sel = slice(None) if batch_idx is None else (batch_idx == s)
        p = positions[sel].to(torch.float64)
        q = charges[sel].to(torch.float64)
        k = k_vectors_2d[s].to(torch.float64)  # (K, 3)
        G = green[s]  # (K,)
        gre, gim = g_re[s], g_im[s]  # (K,)
        kr = p @ k.transpose(0, 1)  # (n, K)
        cos, sin = torch.cos(kr), torch.sin(kr)
        gq = cos @ (G * gre) + sin @ (G * gim) + g_tot[s]
        w = (G * (-gre)).unsqueeze(0) * sin + (G * gim).unsqueeze(0) * cos  # (n, K)
        gp = q.unsqueeze(1) * (w @ k)  # (n, 3)
        if batch_idx is None:
            grad_chg = gq
            grad_pos = gp
        else:
            idx = sel.nonzero(as_tuple=True)[0]
            grad_chg = grad_chg.index_copy(0, idx, gq)
            grad_pos = grad_pos.index_copy(0, idx, gp)
    return grad_pos.to(positions.dtype), grad_chg.to(charges.dtype)


@torch.library.custom_op(
    "alchemiops::_ewald_compute_partial_structure_factors", mutates_args=()
)
def _ewald_compute_partial_structure_factors(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: single-system partial green-weighted structure factors."""
    cell_3d, k_vectors_2d = _normalize(cell, k_vectors, 1)
    _cos, _sin, real_sf, imag_sf, total_charge = _run_fill(
        positions, charges, cell_3d, k_vectors_2d, alpha, None
    )
    return real_sf.reshape(-1), imag_sf.reshape(-1), total_charge.reshape(1)


@_ewald_compute_partial_structure_factors.register_fake
def _fake_ewald_compute_partial_structure_factors(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_k = k_vectors.shape[-2]
    return (
        positions.new_empty(num_k, dtype=torch.float64),
        positions.new_empty(num_k, dtype=torch.float64),
        positions.new_empty(1, dtype=torch.float64),
    )


def _setup_ctx_single(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    positions, charges, cell, k_vectors, alpha = inputs
    cell_3d, k_vectors_2d = _normalize(cell, k_vectors, 1)
    volume = torch.abs(torch.det(cell_3d)).reshape(1).to(torch.float64)
    ctx.save_for_backward(positions, charges, k_vectors_2d, volume, alpha)


def _backward_single(
    ctx: Any, g_re: torch.Tensor, g_im: torch.Tensor, g_tot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, None, None, None]:
    positions, charges, k_vectors_2d, volume, alpha = ctx.saved_tensors
    grad_pos, grad_chg = _stage1_backward(
        g_re, g_im, g_tot, positions, charges, k_vectors_2d, volume, alpha, None, 1
    )
    return grad_pos, grad_chg, None, None, None


_ewald_compute_partial_structure_factors.register_autograd(
    _backward_single, setup_context=_setup_ctx_single
)


@torch.library.custom_op(
    "alchemiops::_batch_ewald_compute_partial_structure_factors", mutates_args=()
)
def _batch_ewald_compute_partial_structure_factors(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Internal: batched partial green-weighted structure factors."""
    num_systems = cell.shape[0]
    cell_3d, k_vectors_2d = _normalize(cell, k_vectors, num_systems)
    _cos, _sin, real_sf, imag_sf, total_charge = _run_fill(
        positions, charges, cell_3d, k_vectors_2d, alpha, batch_idx
    )
    return real_sf, imag_sf, total_charge


@_batch_ewald_compute_partial_structure_factors.register_fake
def _fake_batch_ewald_compute_partial_structure_factors(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_systems = cell.shape[0]
    num_k = k_vectors.shape[-2]
    return (
        positions.new_empty(num_systems, num_k, dtype=torch.float64),
        positions.new_empty(num_systems, num_k, dtype=torch.float64),
        positions.new_empty(num_systems, dtype=torch.float64),
    )


def _setup_ctx_batch(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    positions, charges, cell, k_vectors, alpha, batch_idx = inputs
    num_systems = cell.shape[0]
    cell_3d, k_vectors_2d = _normalize(cell, k_vectors, num_systems)
    volume = torch.abs(torch.linalg.det(cell_3d)).to(torch.float64)
    ctx.num_systems = num_systems
    ctx.save_for_backward(positions, charges, k_vectors_2d, volume, alpha, batch_idx)


def _backward_batch(
    ctx: Any, g_re: torch.Tensor, g_im: torch.Tensor, g_tot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, None, None, None, None]:
    positions, charges, k_vectors_2d, volume, alpha, batch_idx = ctx.saved_tensors
    grad_pos, grad_chg = _stage1_backward(
        g_re,
        g_im,
        g_tot,
        positions,
        charges,
        k_vectors_2d,
        volume,
        alpha,
        batch_idx,
        ctx.num_systems,
    )
    return grad_pos, grad_chg, None, None, None, None


_batch_ewald_compute_partial_structure_factors.register_autograd(
    _backward_batch, setup_context=_setup_ctx_batch
)


# ======================================================================
# Stage 2 — per-atom E / F / dE/dq / virial from globally-reduced S̃(k)
# ======================================================================
#
# Consumes externally-supplied (already cross-rank reduced) structure factors
# and total charge instead of re-running the fill's S(k) and re-summing the
# local charges. The local ``fill`` still runs to recover the per-atom
# ``cos(k·r)`` / ``sin(k·r)``; its structure-factor outputs are discarded in
# favour of the reduced inputs.


def _recip_direct_from_sf(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell_3d: torch.Tensor,
    k_vectors_2d: torch.Tensor,
    alpha: torch.Tensor,
    real_sf: torch.Tensor,
    imag_sf: torch.Tensor,
    total_charge: torch.Tensor,
    batch_idx: torch.Tensor | None,
    want_charge_grad: bool,
    want_virial: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Direct k-space ``(e_ksum, forces, charge_grads, virial)`` from external ``S̃(k)``.

    ``forces`` are ``-dE/dR`` from the k-sum, ``charge_grads`` is the full
    reciprocal ``dE/dq`` (k-sum potential minus the self / background
    derivatives, the latter using the *global* ``total_charge``), and
    ``virial`` is the k-major virial minus the background ``-E_bg I`` term.
    The energy is the k-sum only; callers apply the self/background energy
    corrections via :func:`ewald_energy_corrections`.
    """
    num_atoms = positions.shape[0]
    num_k = k_vectors_2d.shape[-2]
    num_systems = cell_3d.shape[0]
    input_dtype = positions.dtype
    device = wp.device_from_torch(positions.device)
    wp_scalar = get_wp_dtype(input_dtype)
    wp_vec = get_wp_vec_dtype(input_dtype)
    wp_mat = get_wp_mat_dtype(input_dtype)
    batched = batch_idx is not None

    energies = torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
    forces = torch.zeros(num_atoms, 3, device=positions.device, dtype=input_dtype)
    charge_grads = (
        torch.zeros(num_atoms, device=positions.device, dtype=torch.float64)
        if want_charge_grad
        else None
    )
    virial = (
        torch.zeros(num_systems, 3, 3, device=positions.device, dtype=input_dtype)
        if want_virial
        else None
    )
    if num_atoms == 0 or num_k == 0:
        return energies, forces, charge_grads, virial

    real_sf_2d = real_sf.reshape(num_systems, num_k).to(torch.float64)
    imag_sf_2d = imag_sf.reshape(num_systems, num_k).to(torch.float64)

    deriv_state = _DerivState.E_F_dQ if want_charge_grad else _DerivState.E_F
    bundle = get_ewald_recip_kernel(
        wp_scalar,
        batched=batched,
        deriv_state=deriv_state,
        cell_grad=want_virial,
        order="forward",
    )
    s = alloc_ewald_recip_sentinels(wp_scalar, device)

    cos_kr, sin_kr, _rsf, _isf, _tc = _run_fill(
        positions, charges, cell_3d, k_vectors_2d, alpha, batch_idx
    )
    wp_cos = _wp(cos_kr, wp.float64)
    wp_sin = _wp(sin_kr, wp.float64)
    wp_real = _wp(real_sf_2d, wp.float64)
    wp_imag = _wp(imag_sf_2d, wp.float64)
    batch_id = (
        _wp(batch_idx, wp.int32)
        if batched
        else wp.empty((0,), dtype=wp.int32, device=device)
    )
    wp_cg = _wp(charge_grads, wp.float64) if want_charge_grad else s["charge_gradients"]
    with _scoped_stream(positions.device):
        wp.launch(
            bundle.compute,
            dim=num_atoms,
            inputs=[
                _wp(charges, wp_scalar),
                batch_id,
                _wp(k_vectors_2d, wp_vec),
                wp_cos,
                wp_sin,
                wp_real,
                wp_imag,
                s["grad_energy"],
                _wp(energies, wp.float64),
                _wp(forces, wp_vec),
                wp_cg,
            ],
            device=device,
        )
        if want_virial:
            volume = torch.abs(torch.det(cell_3d.to(torch.float64))).reshape(
                num_systems
            )
            wp_vol = _wp(volume, wp.float64)
            wp_virial = _wp(virial, wp_mat)
            if batched:
                wp.launch(
                    bundle.virial,
                    dim=(num_k, num_systems),
                    inputs=[
                        _wp(k_vectors_2d, wp_vec),
                        _wp(alpha, wp_scalar),
                        wp_vol,
                        wp_real,
                        wp_imag,
                        wp_virial,
                    ],
                    device=device,
                )
            else:
                kv_1d = _wp(k_vectors_2d.reshape(num_k, 3), wp_vec)
                wp.launch(
                    bundle.virial,
                    dim=num_k,
                    inputs=[
                        kv_1d,
                        _wp(alpha, wp_scalar),
                        wp_vol,
                        _wp(real_sf_2d.reshape(num_k), wp.float64),
                        _wp(imag_sf_2d.reshape(num_k), wp.float64),
                        wp_virial,
                    ],
                    device=device,
                )

    # Charge-gradient self + background corrections (Torch; the global
    # total_charge feeds the background term):
    #   dE_self/dq_i = 2 α q_i / sqrt(π);  dE_bg/dq_i = π (Q_tot / V) / α².
    if want_charge_grad:
        charges64 = charges.to(torch.float64)
        q_tot = total_charge.to(torch.float64).reshape(-1)
        if batched:
            alpha_atom = alpha.to(torch.float64).index_select(0, batch_idx)
            vol = torch.abs(torch.linalg.det(cell_3d.to(torch.float64)))  # (S,)
            q_over_v_atom = (q_tot / vol).index_select(0, batch_idx)
        else:
            alpha_atom = alpha.to(torch.float64).reshape(-1)[0]
            vol = torch.abs(torch.det(cell_3d[0].to(torch.float64)))
            q_over_v_atom = q_tot[0] / vol
        self_grad = 2.0 * alpha_atom / math.sqrt(_PI) * charges64
        bg_grad = _PI / (alpha_atom * alpha_atom) * q_over_v_atom
        charge_grads.sub_(self_grad + bg_grad)

    # Virial background correction: W_bg = -E_bg I, with the global Q_tot.
    if want_virial:
        q_tot = total_charge.to(input_dtype).reshape(-1)
        eye = torch.eye(3, device=positions.device, dtype=input_dtype)
        if batched:
            vol = torch.abs(torch.linalg.det(cell_3d)).to(input_dtype)
            alpha_b = alpha.to(input_dtype)
            e_bg = _PI * q_tot**2 / (2.0 * alpha_b**2 * vol)
            virial.sub_(e_bg[:, None, None] * eye)
        else:
            vol = torch.abs(torch.det(cell_3d[0].to(input_dtype)))
            alpha_v = alpha.to(input_dtype).reshape(-1)[0]
            e_bg = _PI * q_tot[0] ** 2 / (2.0 * alpha_v**2 * vol)
            virial.sub_(e_bg * eye)

    return energies, forces, charge_grads, virial


def _apply_corrections(
    e_ksum: torch.Tensor,
    charges: torch.Tensor,
    cell_3d: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
    batch_idx: torch.Tensor | None,
) -> torch.Tensor:
    """Add the reciprocal self + background energy with an external ``total_charge``.

    Wraps the public ``ewald_energy_corrections{,_batch}`` (which accepts an
    explicit ``total_charge``); ``alpha`` is setup-only (detached).
    """
    ensure_electrostatics_ops_registered()
    alpha = alpha.detach()
    if batch_idx is None:
        volume = torch.abs(torch.det(cell_3d[0])).reshape(1).to(torch.float64)
        return ewald_energy_corrections(
            e_ksum, charges, volume, alpha, total_charge.reshape(1)
        )
    volume = torch.abs(torch.linalg.det(cell_3d)).to(torch.float64)
    return ewald_energy_corrections_batch(
        e_ksum,
        charges,
        batch_idx.to(torch.int32),
        volume,
        alpha,
        total_charge,
    )


# ======================================================================
# Public staged API
# ======================================================================


def ewald_compute_partial_structure_factors(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Compute per-rank partial green-weighted ``S̃(k)`` and total charge.

    Runs only the reciprocal ``fill`` warp kernel and returns its per-rank
    partials; distributed callers all-reduce these across ranks before passing
    them to :func:`ewald_reciprocal_space_from_structure_factors`.

    Returns ``(real_sf, imag_sf, total_charge)`` with shapes ``(K,)`` /
    ``(B, K)``, same, and ``(1,)`` / ``(B,)`` for single vs batched. Always
    ``float64``.
    """
    is_batch = batch_idx is not None
    if is_batch and k_vectors.dim() == 2:
        k_vectors = k_vectors.unsqueeze(0)
    elif not is_batch and k_vectors.dim() == 3 and k_vectors.shape[0] == 1:
        k_vectors = k_vectors.squeeze(0)

    if is_batch:
        return _batch_ewald_compute_partial_structure_factors(
            positions, charges, cell, k_vectors, alpha, batch_idx
        )
    return _ewald_compute_partial_structure_factors(
        positions, charges, cell, k_vectors, alpha
    )


def ewald_reciprocal_space_from_structure_factors(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    real_sf: torch.Tensor,
    imag_sf: torch.Tensor,
    total_charge: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    hybrid_forces: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""Compute per-atom reciprocal-space quantities from pre-reduced ``S̃(k)``.

    Flags and return-tuple layout mirror the monolithic
    ``ewald_reciprocal_space`` — same ``(energies, [forces], [charge_grads],
    [virial])`` ordering — so callers can swap between the two without changing
    surrounding code. The only difference is that the structure factors + total
    charge are provided as inputs (from stage 1 or a cross-rank all-reduce).

    Forces / charge gradients / virial are computed from the globally-reduced
    ``S̃(k)``; the energy adds the self / background corrections via the public
    ``ewald_energy_corrections`` with the external ``total_charge``.
    ``hybrid_forces=True`` detaches positions / cell and wires ``dE/dq`` into
    the energy via ``_InjectChargeGrad`` (forward-only forces / virial);
    otherwise the direct forces are returned and the energy stays connected to
    charges through the corrections.
    """
    # Stage 2 launches raw Warp kernels that read plain storage, so any
    # ShardTensor input must be localized to its owned+ghost block first —
    # otherwise Warp reads a global-shaped extent off a local buffer.
    # ``to_local`` is a no-op on plain (single-GPU) tensors.
    from nvalchemi.distributed.helpers import to_local  # noqa: PLC0415

    positions = to_local(positions)
    charges = to_local(charges)
    cell = to_local(cell)
    k_vectors = to_local(k_vectors)
    alpha = to_local(alpha)
    real_sf = to_local(real_sf)
    imag_sf = to_local(imag_sf)
    total_charge = to_local(total_charge)
    batch_idx = to_local(batch_idx)

    is_batch = batch_idx is not None
    if is_batch and k_vectors.dim() == 2:
        k_vectors = k_vectors.unsqueeze(0)
    elif not is_batch and k_vectors.dim() == 3 and k_vectors.shape[0] == 1:
        k_vectors = k_vectors.squeeze(0)

    num_systems = cell.shape[0] if cell.dim() == 3 else 1
    cell_3d, k_vectors_2d = _normalize(cell, k_vectors, num_systems)

    def _build_result(energies, forces=None, charge_grads=None, virial=None):
        match (
            compute_forces and forces is not None,
            compute_charge_gradients and charge_grads is not None,
            compute_virial and virial is not None,
        ):
            case (True, True, True):
                return energies, forces, charge_grads, virial
            case (True, True, False):
                return energies, forces, charge_grads
            case (True, False, True):
                return energies, forces, virial
            case (True, False, False):
                return energies, forces
            case (False, True, True):
                return energies, charge_grads, virial
            case (False, True, False):
                return energies, charge_grads
            case (False, False, True):
                return energies, virial
            case _:
                return energies

    want_charge_grad = compute_charge_gradients or hybrid_forces
    want_virial = compute_virial

    if hybrid_forces:
        pos_d = positions.detach()
        chg_d = charges.detach()
        cell_d = cell_3d.detach()
        alpha_d = alpha.detach()
        e_ksum, forces, charge_grads, virial = _recip_direct_from_sf(
            pos_d,
            chg_d,
            cell_d,
            k_vectors_2d.detach(),
            alpha_d,
            real_sf.detach(),
            imag_sf.detach(),
            total_charge.detach(),
            batch_idx,
            want_charge_grad=True,
            want_virial=want_virial,
        )
        energies = _apply_corrections(
            e_ksum, chg_d, cell_d, alpha_d, total_charge.detach(), batch_idx
        )
        if charges.requires_grad:
            energies = _InjectChargeGrad.apply(
                energies, charges, charge_grads, batch_idx
            )
        cg_out = charge_grads if compute_charge_gradients else None
        return _build_result(energies, forces, cg_out, virial)

    e_ksum, forces, charge_grads, virial = _recip_direct_from_sf(
        positions,
        charges,
        cell_3d,
        k_vectors_2d,
        alpha,
        real_sf,
        imag_sf,
        total_charge,
        batch_idx,
        want_charge_grad=want_charge_grad,
        want_virial=want_virial,
    )
    energies = _apply_corrections(
        e_ksum, charges, cell_3d, alpha, total_charge, batch_idx
    )
    cg_out = charge_grads if compute_charge_gradients else None
    return _build_result(energies, forces, cg_out, virial)


# ======================================================================
# Reciprocal-contribution dispatcher (backend selection, incl. DD)
# ======================================================================


def _owned_charge_mask(charges: torch.Tensor, n_owned: Any) -> torch.Tensor:
    """Zero ghost/dead rows (>= n_owned) of a charge vector via a fixed-shape
    mask rather than a ``[:n_owned]`` slice, keeping the shape stable under
    compile. ``n_owned`` may be a runtime tensor (compiled) or a Python int."""
    rowidx = torch.arange(charges.shape[0], device=charges.device)
    return charges * (rowidx < n_owned).to(charges.dtype)


def _reciprocal_torch_dd(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None,
    ctx: Any,
) -> torch.Tensor:
    """Autograd-native reciprocal per-atom energy for the compiled DD path.

    Owned-only partial green-weighted ``S̃(k)`` -> cross-rank all-reduce -> per-atom
    energy from the global ``S̃``. Pure Torch so it is compile-traceable and autograd
    yields the exact force.
    """
    from nvalchemi.distributed._core.compile_routing import (  # noqa: PLC0415
        get_compile_routing,
    )
    from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
        distributed_all_reduce,
    )
    from nvalchemi.models._ops.electrostatics.ewald_recip_torch import (  # noqa: PLC0415
        ewald_energy_from_structure_factors,
        ewald_partial_structure_factors,
    )

    config = ctx.halo_config
    routing = get_compile_routing()
    n_owned = routing[3] if routing is not None else int(ctx.halo_meta.n_owned)
    owned_charges = _owned_charge_mask(charges, n_owned)
    real_sf, imag_sf, total_charge = ewald_partial_structure_factors(
        positions, owned_charges, cell, k_vectors, alpha, batch_idx=batch_idx
    )
    real_sf = distributed_all_reduce(real_sf, config)
    imag_sf = distributed_all_reduce(imag_sf, config)
    total_charge = distributed_all_reduce(total_charge, config)
    return ewald_energy_from_structure_factors(
        positions,
        charges,
        cell,
        k_vectors,
        alpha,
        real_sf,
        imag_sf,
        total_charge,
        batch_idx=batch_idx,
    )


def ewald_reciprocal_contribution(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor,
    num_systems: int,
    compute_forces: bool,
    compute_virial: bool,
    hybrid_forces: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Per-atom reciprocal energy (+ kernel forces/virial), backend chosen here.

    The warp reciprocal writes its forces directly and is not differentiable, so
    an energy-only forward (forces wanted via autograd) needs a differentiable
    reciprocal instead. Picks, transparently to the caller:

    * energy-only under domain decomposition -> Torch staged ``S̃`` (compile-safe,
      cross-rank all-reduced);
    * energy-only single-GPU -> the differentiable monolithic ``ewald_reciprocal_space``;
    * forces/virial requested (eager) -> warp staged structure factors (the spec's
      halo handlers reduce ``S̃`` across ranks; single-GPU fires nothing).

    Returns ``(e_recip, f_recip|None, v_recip|None)``.
    """
    from nvalchemi.distributed._core.context import current_dd_context  # noqa: PLC0415

    # Single-system batches pass batch_idx=None to the differentiable paths to
    # avoid a data-dependent ``nonzero`` (a graph break under compile).
    bidx = batch_idx if num_systems > 1 else None
    autograd_recip = not compute_forces and not compute_virial
    ctx = current_dd_context()

    if autograd_recip and ctx is not None and getattr(ctx, "is_halo", False):
        return (
            _reciprocal_torch_dd(positions, charges, cell, k_vectors, alpha, bidx, ctx),
            None,
            None,
        )
    if autograd_recip:
        from nvalchemiops.torch.interactions.electrostatics.ewald import (  # noqa: PLC0415
            ewald_reciprocal_space,
        )

        e_recip = ewald_reciprocal_space(
            positions,
            charges,
            cell,
            k_vectors,
            alpha,
            batch_idx=bidx,
            compute_forces=False,
            hybrid_forces=False,
        )
        return e_recip, None, None

    real_sf, imag_sf, total_charge = ewald_compute_partial_structure_factors(
        positions, charges, cell, k_vectors, alpha, batch_idx=batch_idx
    )
    recip = ewald_reciprocal_space_from_structure_factors(
        positions,
        charges,
        cell,
        k_vectors,
        alpha,
        real_sf,
        imag_sf,
        total_charge,
        batch_idx=batch_idx,
        compute_forces=compute_forces,
        compute_virial=compute_virial,
        hybrid_forces=hybrid_forces,
    )
    if isinstance(recip, torch.Tensor):
        return recip, None, None
    recip = list(recip)
    f = recip[1] if compute_forces else None
    v = recip[-1] if compute_virial else None
    return recip[0], f, v
