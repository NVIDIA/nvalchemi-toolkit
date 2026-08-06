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

r"""Autograd-native staged Ewald reciprocal space — the DD-*compile* path only.

The reciprocal energy is bilinear in the structure factor :math:`S(k)`
(:math:`E = \tfrac{1}{2} \sum_k G(k)|S(k)|^{2}`), so domain decomposition must assemble the global
:math:`S(k)` (cross-rank all-reduce) *between* "compute partial :math:`\tilde{S}(k)`" and
"compute energy from :math:`\tilde{S}(k)`". Crucially the framework's compiled energy-autograd
path consolidates energy **owned-only**, so the cotangent into the reciprocal
energy is a non-uniform (owned-mask) vector; the correct weighted force then needs
both the direct :math:`\cos(\mathbf{k}\cdot\mathbf{r}_i)` ("field") term **and** the :math:`\partial E/\partial \tilde{S}` ("source")
term routed back through the partial-:math:`\tilde{S}` stage and the all-reduce to the other
ranks' atoms. nvalchemiops' warp reciprocal exposes neither cleanly, and its
cached-full-force backward is only correct for a *uniform* cotangent over the
whole system — verified wrong on a non-trivial halo. So this path is pure Torch:
autograd produces the exact weighted VJP for any cotangent and it is
``torch.compile``-traceable (no warp launches → no graph break).

This is used **only** for compiled DD. Non-DD inference and eager DD ride the
warp kernels (``EwaldModelWrapper`` branches on the execution mode). The energy
math matches nvalchemiops' in-tree ``_recip_ksum_energy_torch`` reference, just
with :math:`\tilde{S}(k)` (and :math:`Q`) supplied externally so the all-reduce can sit in the
seam. The self/background corrections reuse the upstream
``ewald_energy_corrections`` (already accept an explicit ``total_charge``).
"""

from __future__ import annotations

import torch

__all__ = [
    "ewald_partial_structure_factors",
    "ewald_energy_from_structure_factors",
]


def _normalize(cell, k_vectors, alpha, num_systems):
    """Coerce ``cell`` -> ``(S,3,3)``, ``k_vectors`` -> ``(S,K,3)``, ``alpha`` -> ``(S,)`` f64."""
    if cell.dim() == 2:
        cell = cell.reshape(1, 3, 3)
    if k_vectors.dim() == 2:
        k_vectors = k_vectors.reshape(1, -1, 3)
    if k_vectors.shape[0] == 1 and num_systems > 1:
        k_vectors = k_vectors.expand(num_systems, -1, 3)
    alpha = alpha.reshape(-1).to(torch.float64)
    if alpha.numel() == 1 and num_systems > 1:
        alpha = alpha.expand(num_systems)
    return cell, k_vectors, alpha


def _green(k_vectors_s, volume_s, alpha_s):
    r""":math:`G[k] = (8\pi/V)\,\exp(-k^{2}/4\alpha^{2})/k^{2}` on the half-space k-vectors (:math:`k^{2}<10^{-10}` -> 0)."""
    from nvalchemiops.interactions.electrostatics.ewald_kernels import (  # noqa: PLC0415
        EIGHTPI,
    )

    ksq = (k_vectors_s * k_vectors_s).sum(-1)
    g = (EIGHTPI / volume_s) * torch.exp(-ksq * (0.25 / (alpha_s * alpha_s))) / ksq
    return torch.where(ksq < 1e-10, torch.zeros_like(g), g)


def ewald_partial_structure_factors(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Green-weighted partial structure factors :math:`\tilde{S}(k)` and total charge :math:`Q`.

    :math:`\mathrm{Re}\,\tilde{S}[s,k] = G(k) \sum_j q_j \cos(\mathbf{k}\cdot\mathbf{r}_j)`,
    :math:`\mathrm{Im}\,\tilde{S}[s,k] = G(k) \sum_j q_j \sin(\mathbf{k}\cdot\mathbf{r}_j)`,
    :math:`Q[s] = \sum_j q_j` over the atoms supplied. Pure Torch, autograd-connected to
    ``positions`` / ``charges`` (required so the :math:`\partial E/\partial \tilde{S}` source term reaches the
    producing atoms, incl. cross-rank via the caller's all-reduce). To restrict to
    a rank's owned atoms under DD, pass owned-masked ``charges`` (ghost charges
    zeroed). Shapes ``(K,)/(1,)`` single, ``(B,K)/(B,)`` batched; ``float64``.
    """
    pos = positions.to(torch.float64)
    q = charges.to(torch.float64)
    is_batch = batch_idx is not None
    num_systems = cell.shape[0] if (is_batch or cell.dim() == 3) else 1
    cell_3d, k_2d, alpha_s = _normalize(cell, k_vectors, alpha, num_systems)
    volume = torch.abs(torch.linalg.det(cell_3d)).to(torch.float64)
    num_k = k_2d.shape[-2]

    real = pos.new_zeros(num_systems, num_k)
    imag = pos.new_zeros(num_systems, num_k)
    qtot = pos.new_zeros(num_systems)
    for s in range(num_systems):
        k = k_2d[s].to(torch.float64)
        green = _green(k, volume[s], alpha_s[s])
        if batch_idx is None:
            p_s, q_s = pos, q
        else:
            sel = batch_idx == s
            p_s, q_s = pos[sel], q[sel]
        kr = p_s @ k.transpose(0, 1)
        real[s] = (q_s.unsqueeze(1) * torch.cos(kr)).sum(0) * green
        imag[s] = (q_s.unsqueeze(1) * torch.sin(kr)).sum(0) * green
        qtot[s] = q_s.sum()

    if batch_idx is None:
        return real.reshape(num_k), imag.reshape(num_k), qtot.reshape(1)
    return real, imag, qtot


def ewald_energy_from_structure_factors(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    k_vectors: torch.Tensor,
    alpha: torch.Tensor,
    real_sf: torch.Tensor,
    imag_sf: torch.Tensor,
    total_charge: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Per-atom reciprocal energy from externally supplied (reduced) :math:`\tilde{S}(k)` / :math:`Q`.

    :math:`E_i = \tfrac{1}{2} q_i \sum_k [\cos(\mathbf{k}\cdot\mathbf{r}_i)\,\mathrm{Re}\,\tilde{S}[k] + \sin(\mathbf{k}\cdot\mathbf{r}_i)\,\mathrm{Im}\,\tilde{S}[k]]` + self/background
    corrections (with the global ``total_charge``). Autograd flows to ``positions``
    (field) and ``real_sf``/``imag_sf`` (source) — so with :math:`\tilde{S}` from
    :func:`ewald_partial_structure_factors` + a cross-rank all-reduce the weighted
    VJP is exact for any cotangent (incl. the owned-mask DD consolidation
    produces). Returns per-atom energy ``(N,)`` float64.
    """
    from nvalchemiops.torch.interactions.electrostatics.ewald import (  # noqa: PLC0415
        ewald_energy_corrections,
        ewald_energy_corrections_batch,
    )

    pos = positions.to(torch.float64)
    q = charges.to(torch.float64)
    is_batch = batch_idx is not None
    num_systems = cell.shape[0] if (is_batch or cell.dim() == 3) else 1
    cell_3d, k_2d, alpha_s = _normalize(cell, k_vectors, alpha, num_systems)
    num_k = k_2d.shape[-2]
    re_2d = real_sf.reshape(num_systems, num_k).to(torch.float64)
    im_2d = imag_sf.reshape(num_systems, num_k).to(torch.float64)

    e_ksum = pos.new_zeros(pos.shape[0])
    for s in range(num_systems):
        k = k_2d[s].to(torch.float64)
        if batch_idx is None:
            p_s, q_s, idx = pos, q, None
        else:
            sel = batch_idx == s
            p_s, q_s = pos[sel], q[sel]
            idx = sel.nonzero(as_tuple=True)[0]
        kr = p_s @ k.transpose(0, 1)
        e_s = 0.5 * q_s * (torch.cos(kr) @ re_2d[s] + torch.sin(kr) @ im_2d[s])
        if batch_idx is None:
            e_ksum = e_s
        else:
            e_ksum = e_ksum.index_copy(0, idx, e_s)

    volume = torch.abs(torch.linalg.det(cell_3d)).to(torch.float64)
    if batch_idx is None:
        return ewald_energy_corrections(
            e_ksum, charges, volume.reshape(1), alpha_s, total_charge.reshape(1)
        )
    return ewald_energy_corrections_batch(
        e_ksum, charges, batch_idx.to(torch.int32), volume, alpha_s, total_charge
    )
