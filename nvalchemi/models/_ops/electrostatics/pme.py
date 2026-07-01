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


r"""
PME reciprocal-space, split into stages for distributed use.

Splits the monolithic ``particle_mesh_ewald`` so distributed callers can
insert a cross-rank reduction between the per-rank partial total charge
(``Q = Σᵢ qᵢ`` over owned atoms) and the per-atom background correction
``(π/(2α²V))·qᵢ·Q_total``.

Stages:

- :func:`pme_compute_partial_total_charge` — per-rank partial ``Σᵢ qᵢ``
  (per-system for batched inputs). A custom op so the distributed layer
  can intercept it under halo storage and apply owned-slice + all-reduce.
- :func:`particle_mesh_ewald_from_total_charge` — full PME pipeline (real
  + reciprocal + corrections) that takes the globally-reduced total charge
  as an explicit input.

Motivation: under halo storage each rank sees its padded (owned + halo)
charges, and the upstream correction sums over every row it sees — so
ranks disagree on the background term whenever the halo is not
charge-symmetric (e.g. 2-rank NaCl where one rank's padded charges sum to
0 and the other to +2·q₀). Threading a globally-reduced total charge
through the pipeline fixes this.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from nvalchemiops.torch.interactions.electrostatics._util import _InjectChargeGrad
from nvalchemiops.torch.interactions.electrostatics.ewald import ewald_real_space
from nvalchemiops.torch.interactions.electrostatics.k_vectors import (
    generate_k_vectors_pme,
)
from nvalchemiops.torch.interactions.electrostatics.parameters import (
    estimate_pme_mesh_dimensions,
    estimate_pme_parameters,
    mesh_spacing_to_dimensions,
)
from nvalchemiops.torch.interactions.electrostatics.pme import (
    _batch_pme_energy_corrections,
    _batch_pme_energy_corrections_with_charge_grad,
    _pme_energy_corrections,
    _pme_energy_corrections_with_charge_grad,
    _prepare_alpha,
    _prepare_cell,
    compute_bspline_moduli_1d,
    register_pme_ops,
)
from nvalchemiops.torch.spline import (
    spline_gather,
    spline_gather_with_force,
    spline_spread,
)

PI = math.pi

__all__ = [
    "pme_compute_partial_total_charge",
    "pme_energy_corrections_from_total_charge",
    "pme_energy_corrections_with_charge_grad_from_total_charge",
    "pme_reciprocal_space_from_total_charge",
    "particle_mesh_ewald_from_total_charge",
]


# Stage 1 — partial total charge.
# Registered as custom ops (no kernel needed — it's just a reduction) so the
# distributed layer can attach an owned-slice + all-reduce handler under halo
# storage. Autograd is the trivial d(Σqᵢ)/dqⱼ = 1 gradient.


@torch.library.custom_op(
    "alchemiops::_pme_compute_partial_total_charge", mutates_args=()
)
def _pme_compute_partial_total_charge(charges: torch.Tensor) -> torch.Tensor:
    """Internal: single-system partial total charge ``Σᵢ qᵢ``.

    Output is shape ``(1,)`` and always ``float64`` for accumulation
    stability.
    """
    return charges.to(torch.float64).sum().reshape(1)


@_pme_compute_partial_total_charge.register_fake
def _fake_pme_compute_partial_total_charge(
    charges: torch.Tensor,
) -> torch.Tensor:
    return charges.new_empty((1,), dtype=torch.float64)


def _setup_ctx_single(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    (charges,) = inputs
    ctx.charges_shape = charges.shape
    ctx.charges_dtype = charges.dtype


def _backward_single(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
    # d(Σqᵢ)/dqⱼ = 1 ⇒ grad_charges = grad_output.expand(charges.shape)
    return grad_output.to(ctx.charges_dtype).expand(ctx.charges_shape).contiguous()


_pme_compute_partial_total_charge.register_autograd(
    _backward_single, setup_context=_setup_ctx_single
)


@torch.library.custom_op(
    "alchemiops::_batch_pme_compute_partial_total_charge", mutates_args=()
)
def _batch_pme_compute_partial_total_charge(
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    num_systems: int,
) -> torch.Tensor:
    """Internal: per-system partial total charge via scatter_add."""
    out = torch.zeros(num_systems, dtype=torch.float64, device=charges.device)
    out.scatter_add_(0, batch_idx.to(torch.int64), charges.to(torch.float64))
    return out


@_batch_pme_compute_partial_total_charge.register_fake
def _fake_batch_pme_compute_partial_total_charge(
    charges: torch.Tensor,
    batch_idx: torch.Tensor,
    num_systems: int,
) -> torch.Tensor:
    return charges.new_empty((num_systems,), dtype=torch.float64)


def _setup_ctx_batch(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    charges, batch_idx, _ = inputs
    ctx.charges_shape = charges.shape
    ctx.charges_dtype = charges.dtype
    ctx.save_for_backward(batch_idx)


def _backward_batch(
    ctx: Any, grad_output: torch.Tensor
) -> tuple[torch.Tensor, None, None]:
    # d(out[b]) / d(charges[i]) = δ(batch_idx[i], b) — i.e.
    # grad_charges[i] = grad_output[batch_idx[i]].
    (batch_idx,) = ctx.saved_tensors
    grad_charges = grad_output.to(ctx.charges_dtype).index_select(
        0, batch_idx.to(torch.int64)
    )
    return grad_charges, None, None


_batch_pme_compute_partial_total_charge.register_autograd(
    _backward_batch, setup_context=_setup_ctx_batch
)


def _plain_compile_dd(*tensors: Any) -> "tuple[Any, Any] | None":
    """``(n_owned, halo_config)`` when the reciprocal path runs on plain
    tensors inside a distributed forward, else ``None``.

    The owned-slice + all-reduce corrections are normally applied by the
    distributed layer's handlers, but the compiled energy/autograd path runs on
    plain tensors where those handlers don't fire — so the same corrections must
    be applied directly here. Returns the owned count + mesh config only in that
    case; returns ``None`` single-GPU, in eager distributed runs (where the
    handlers still apply), and when no halo metadata is active. Keeps the
    correction firing exactly once.

    ``n_owned`` is a runtime tensor when one is published (so the owned/ghost
    split is a fixed-shape mask ``rowidx < n_owned`` and the graph does not
    recompile as the partition boundary drifts across MD steps), falling back to
    a Python int from the halo metadata otherwise.
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


def _owned_charge_mask(
    charges: torch.Tensor, n_owned: Any
) -> torch.Tensor:
    """Zero the ghost rows of a per-atom charge vector, fixed-shape.

    Returns ``charges`` with rows ``>= n_owned`` set to zero via an element-wise
    mask (``rowidx < n_owned``) rather than a ``[:n_owned]`` slice, so the tensor
    keeps its length under compile. Spreading or summing the masked charges is
    identical to operating on the owned slice (ghost atoms contribute zero
    charge) but the shape is stable, so the graph does not recompile as
    ``n_owned`` drifts. ``n_owned`` may be a runtime tensor or a Python int.
    """
    rowidx = torch.arange(charges.shape[0], device=charges.device)
    mask = (rowidx < n_owned).to(charges.dtype)
    return charges * mask


def pme_compute_partial_total_charge(
    charges: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
    num_systems: int | None = None,
) -> torch.Tensor:
    r"""Compute the total charge ``Q = Σᵢ qᵢ`` (globally reduced under distribution).

    Under halo storage the custom op is intercepted by the distributed layer:
    ``charges`` is sliced to its owned prefix before the kernel, and the partial
    is all-reduced across the mesh. On the compiled path that interception is
    absent, so the same owned-slice + all-reduce is applied here. Single-GPU,
    both are pass-throughs.

    Returns a ``float64`` tensor of shape ``(1,)`` for single-system inputs
    (``batch_idx is None``), or ``(num_systems,)`` for batched inputs.
    """
    if num_systems is None and batch_idx is not None:
        num_systems = int(batch_idx.max().item()) + 1

    dd = _plain_compile_dd(charges)
    if dd is not None:
        from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
            distributed_all_reduce,
        )

        n_owned, config = dd
        # Zero the ghost charges (fixed-shape mask) so the partial sum counts
        # owned only while keeping a stable shape, then all-reduce to the global
        # total. Full batch_idx is harmless — masked ghost rows scatter zero.
        owned_charges = _owned_charge_mask(charges, n_owned)
        if batch_idx is None:
            partial = _pme_compute_partial_total_charge(owned_charges)
        else:
            partial = _batch_pme_compute_partial_total_charge(
                owned_charges, batch_idx, num_systems
            )
        return distributed_all_reduce(partial, config)

    if batch_idx is None:
        return _pme_compute_partial_total_charge(charges)
    return _batch_pme_compute_partial_total_charge(charges, batch_idx, num_systems)


# Stage 2 — energy corrections with externally provided total_charge.
# Like upstream pme_energy_corrections but skips the internal charges.sum() and
# passes the caller-supplied total_charge (already reduced across ranks) to the
# kernels.


def pme_energy_corrections_from_total_charge(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Apply self-energy and background corrections with pre-reduced ``Q``.

    Same contract as upstream ``pme_energy_corrections``, except the caller
    provides the globally-correct ``total_charge`` (shape ``(1,)`` for
    single-system, ``(B,)`` for batched) rather than computing
    ``charges.sum()`` from the inputs it sees — which is incorrect under halo
    storage.
    """
    input_dtype = raw_energies.dtype

    if batch_idx is None:
        volume = torch.abs(torch.det(cell)).reshape(1)
        return _pme_energy_corrections(
            raw_energies,
            charges.to(input_dtype),
            volume.to(input_dtype),
            alpha.to(input_dtype),
            total_charge.to(input_dtype).reshape(1),
        )

    volumes = torch.abs(torch.linalg.det(cell)).to(input_dtype)
    return _batch_pme_energy_corrections(
        raw_energies,
        charges.to(input_dtype),
        batch_idx,
        volumes,
        alpha.to(input_dtype),
        total_charge.to(input_dtype),
    )


def pme_energy_corrections_with_charge_grad_from_total_charge(
    raw_energies: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    total_charge: torch.Tensor,
    batch_idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Charge-gradient variant of :func:`pme_energy_corrections_from_total_charge`.

    Same as upstream ``pme_energy_corrections_with_charge_grad`` with an extra
    ``total_charge`` argument threaded through.
    """
    input_dtype = raw_energies.dtype

    if batch_idx is None:
        volume = torch.abs(torch.det(cell)).reshape(1)
        return _pme_energy_corrections_with_charge_grad(
            raw_energies,
            charges.to(input_dtype),
            volume.to(input_dtype),
            alpha.to(input_dtype),
            total_charge.to(input_dtype).reshape(1),
        )

    volumes = torch.abs(torch.linalg.det(cell)).to(input_dtype)
    return _batch_pme_energy_corrections_with_charge_grad(
        raw_energies,
        charges.to(input_dtype),
        batch_idx,
        volumes,
        alpha.to(input_dtype),
        total_charge.to(input_dtype),
    )


# Stage 2 — PME reciprocal-space impl with pre-reduced total_charge.
# Like upstream _pme_reciprocal_space_impl with two changes:
#   1. The corrections call uses total_charges instead of charges.sum().
#   2. The virial background term (e_bg = π·Q²/(2α²V)) uses total_charges.


def _pme_reciprocal_space_impl_from_total_charge(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: torch.Tensor,
    total_charges: torch.Tensor,
    mesh_dimensions: tuple[int, int, int],
    spline_order: int,
    batch_idx: torch.Tensor | None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    k_vectors: torch.Tensor | None = None,
    k_squared: torch.Tensor | None = None,
    hybrid_forces: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """Internal PME reciprocal-space with externally provided total_charges.

    Same contract as upstream ``_pme_reciprocal_space_impl`` plus the new
    positional ``total_charges`` (shape ``(1,)`` or ``(B,)``, ``float64``).
    Caller ensures ``total_charges`` is globally correct under distribution.
    """
    from nvalchemiops.torch.interactions.electrostatics.pme import (
        _compute_pme_reciprocal_virial,
    )

    device = positions.device
    input_dtype = positions.dtype
    num_atoms = positions.shape[0]
    is_batch = batch_idx is not None
    fft_dims = (1, 2, 3) if is_batch else (0, 1, 2)

    if hybrid_forces:
        compute_charge_gradients = True

    if num_atoms == 0:
        energies = torch.zeros(num_atoms, device=device, dtype=input_dtype)
        forces = (
            torch.zeros(num_atoms, 3, device=device, dtype=input_dtype)
            if compute_forces
            else None
        )
        charge_grads = (
            torch.zeros(num_atoms, device=device, dtype=input_dtype)
            if compute_charge_gradients
            else None
        )
        num_systems_zero = cell.shape[0] if is_batch else 1
        virial = (
            torch.zeros(num_systems_zero, 3, 3, device=device, dtype=input_dtype)
            if compute_virial
            else None
        )
        return energies, forces, charge_grads, virial

    mesh_nx, mesh_ny, mesh_nz = mesh_dimensions

    pos_spline = positions.detach() if hybrid_forces else positions
    chg_spline = charges.detach() if hybrid_forces else charges
    cell_spline = cell.detach() if hybrid_forces else cell

    cell_inv = torch.linalg.inv_ex(cell_spline)[0]
    cell_inv_t = cell_inv.transpose(-1, -2).contiguous()
    reciprocal_cell = 2.0 * PI * cell_inv

    # Under compiled distribution: spread only the owned charges, then all-reduce
    # the partial mesh (the correction the eager handlers apply but the compiled
    # path bypasses). Zero the ghost charges with a fixed-shape mask so the
    # partial mesh is owned-only, while the full pos_spline / chg_spline stay
    # intact for the downstream per-atom force gather (every atom feels the
    # global field) and the spread shape stays stable across MD steps.
    dd = _plain_compile_dd(chg_spline, pos_spline)
    spread_chg = chg_spline
    if dd is not None:
        spread_chg = _owned_charge_mask(chg_spline, dd[0])

    mesh_grid = spline_spread(
        pos_spline,
        spread_chg,
        cell_spline,
        mesh_dims=(mesh_nx, mesh_ny, mesh_nz),
        spline_order=spline_order,
        batch_idx=batch_idx,
        cell_inv_t=cell_inv_t,
    )
    if dd is not None:
        from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
            distributed_all_reduce,
        )

        mesh_grid = distributed_all_reduce(mesh_grid, dd[1])

    if k_vectors is None or k_squared is None:
        k_vectors, k_squared = generate_k_vectors_pme(
            cell_spline,
            mesh_dimensions=mesh_dimensions,
            reciprocal_cell=reciprocal_cell,
        )

    alpha_gsf = alpha.detach() if hybrid_forces else alpha

    # Precomputed 1D B-spline modulus tables feed the fused convolve op, which
    # combines the spline-moduli deconvolution, Green's-function multiply, and
    # convolution into a single traceable custom op.
    mesh_nx_m, mesh_ny_m, mesh_nz_m = mesh_dimensions
    miller_x = torch.fft.fftfreq(
        mesh_nx_m, d=1.0 / mesh_nx_m, device=device, dtype=input_dtype
    )
    miller_y = torch.fft.fftfreq(
        mesh_ny_m, d=1.0 / mesh_ny_m, device=device, dtype=input_dtype
    )
    miller_z = torch.fft.rfftfreq(
        mesh_nz_m, d=1.0 / mesh_nz_m, device=device, dtype=input_dtype
    )
    moduli_x = compute_bspline_moduli_1d(miller_x, mesh_nx_m, spline_order)
    moduli_y = compute_bspline_moduli_1d(miller_y, mesh_ny_m, spline_order)
    moduli_z = compute_bspline_moduli_1d(miller_z, mesh_nz_m, spline_order)

    cell_for_vol = cell_spline if cell_spline.dim() == 3 else cell_spline.unsqueeze(0)
    volume = torch.abs(torch.linalg.det(cell_for_vol)).to(input_dtype)

    mesh_fft = torch.fft.rfftn(mesh_grid, norm="backward", dim=fft_dims)
    if torch.compiler.is_compiling():
        # cuFFT emits non-contiguous output; the convolve op requires contiguous
        # input under compile.
        mesh_fft = mesh_fft.contiguous()
    mesh_fft_raw = mesh_fft if compute_virial else None
    register_pme_ops()
    convolved_mesh = torch.ops.nvalchemiops.pme_fused_convolve(
        mesh_fft,
        k_squared,
        moduli_x,
        moduli_y,
        moduli_z,
        alpha_gsf,
        volume,
        is_batch,
    )
    potential_mesh = torch.fft.irfftn(
        convolved_mesh, norm="forward", s=mesh_dimensions, dim=fft_dims
    ).to(input_dtype)

    # The fused gather-with-force kernel writes potential energy and force in one
    # pass over the mesh.
    if compute_forces:
        raw_energies, gathered_force = spline_gather_with_force(
            pos_spline,
            chg_spline,
            potential_mesh,
            cell_spline,
            spline_order=spline_order,
            batch_idx=batch_idx,
            cell_inv_t=cell_inv_t,
        )
    else:
        raw_energies = spline_gather(
            pos_spline,
            potential_mesh,
            cell_spline,
            spline_order=spline_order,
            batch_idx=batch_idx,
            cell_inv_t=cell_inv_t,
        )
        gathered_force = None

    # Corrections with externally provided total_charge.
    charge_grads = None
    if compute_charge_gradients:
        reciprocal_energies, charge_grads = (
            pme_energy_corrections_with_charge_grad_from_total_charge(
                raw_energies, chg_spline, cell_spline, alpha, total_charges, batch_idx
            )
        )
    else:
        reciprocal_energies = pme_energy_corrections_from_total_charge(
            raw_energies, chg_spline, cell_spline, alpha, total_charges, batch_idx
        )

    virial = None
    if compute_virial:
        virial = _compute_pme_reciprocal_virial(
            mesh_fft_raw=mesh_fft_raw,
            convolved_mesh=convolved_mesh,
            k_vectors=k_vectors,
            k_squared=k_squared,
            alpha=alpha,
            mesh_dimensions=mesh_dimensions,
            is_batch=is_batch,
            device=device,
            dtype=input_dtype,
        )
        del mesh_fft_raw

        # Virial background correction uses total_charges. Detach under
        # hybrid_forces (like upstream) so stress doesn't pick up a gradient
        # edge through the background term back to charges.
        total_charges_virial = (
            total_charges.detach() if hybrid_forces else total_charges
        )
        eye = torch.eye(3, device=device, dtype=input_dtype)
        if is_batch:
            volumes = torch.abs(torch.linalg.det(cell_spline)).to(input_dtype)
            alpha_batch = alpha.to(input_dtype)
            q_b = total_charges_virial.to(input_dtype)
            e_bg = PI * q_b**2 / (2.0 * alpha_batch**2 * volumes)
            virial = virial - e_bg[:, None, None] * eye
        else:
            volume = torch.abs(torch.det(cell_spline)).to(input_dtype)
            alpha_val = alpha.to(input_dtype)
            q_scalar = total_charges_virial.to(input_dtype).reshape(-1)[0]
            e_bg = PI * q_scalar**2 / (2.0 * alpha_val**2 * volume)
            virial = virial - e_bg * eye

    forces = None
    if compute_forces:
        # gathered_force is -q*∇Φ in Cartesian coords; the 2× absorbs the 1/2
        # pair-counting factor baked into the Green's function (G = 2π/(V k²)).
        forces = 2.0 * gathered_force

    if hybrid_forces and charges.requires_grad:
        reciprocal_energies = _InjectChargeGrad.apply(
            reciprocal_energies, charges, charge_grads, batch_idx
        )

    return reciprocal_energies, forces, charge_grads, virial


def pme_reciprocal_space_from_total_charge(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    alpha: float | torch.Tensor,
    total_charges: torch.Tensor,
    mesh_dimensions: tuple[int, int, int] | None = None,
    mesh_spacing: float | None = None,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    k_vectors: torch.Tensor | None = None,
    k_squared: torch.Tensor | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    hybrid_forces: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""Reciprocal-space PME with externally-provided ``total_charges``.

    Same signature as upstream ``pme_reciprocal_space`` plus ``total_charges``
    (shape ``(1,)`` or ``(B,)``, ``float64``) threaded through to
    :func:`_pme_reciprocal_space_impl_from_total_charge`.
    """
    cell, num_systems = _prepare_cell(cell)
    alpha_tensor = _prepare_alpha(alpha, num_systems, torch.float64, positions.device)

    if mesh_dimensions is None:
        if mesh_spacing is None:
            raise ValueError("Either mesh_dimensions or mesh_spacing must be provided")
        cell_lengths = torch.norm(cell[0], dim=1)
        mesh_dimensions = tuple(
            int(torch.ceil(length / mesh_spacing).item()) for length in cell_lengths
        )

    energies, forces, charge_grads, virial = (
        _pme_reciprocal_space_impl_from_total_charge(
            positions,
            charges,
            cell,
            alpha_tensor,
            total_charges,
            mesh_dimensions,
            spline_order,
            batch_idx,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
            k_vectors=k_vectors,
            k_squared=k_squared,
            hybrid_forces=hybrid_forces,
        )
    )

    match (compute_forces, compute_charge_gradients, compute_virial):
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


# Stage 2 — top-level PME with pre-reduced total_charges.


def particle_mesh_ewald_from_total_charge(
    positions: torch.Tensor,
    charges: torch.Tensor,
    cell: torch.Tensor,
    total_charges: torch.Tensor,
    alpha: float | torch.Tensor | None = None,
    mesh_spacing: float | None = None,
    mesh_dimensions: tuple[int, int, int] | None = None,
    spline_order: int = 4,
    batch_idx: torch.Tensor | None = None,
    k_vectors: torch.Tensor | None = None,
    k_squared: torch.Tensor | None = None,
    neighbor_list: torch.Tensor | None = None,
    neighbor_ptr: torch.Tensor | None = None,
    neighbor_shifts: torch.Tensor | None = None,
    neighbor_matrix: torch.Tensor | None = None,
    neighbor_matrix_shifts: torch.Tensor | None = None,
    mask_value: int | None = None,
    compute_forces: bool = False,
    compute_charge_gradients: bool = False,
    compute_virial: bool = False,
    accuracy: float = 1e-6,
    hybrid_forces: bool = False,
    pbc: torch.Tensor | None = None,
    slab_correction: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, ...]:
    r"""Top-level PME with externally-provided ``total_charges``.

    Drop-in replacement for upstream ``particle_mesh_ewald`` that accepts
    pre-reduced ``total_charges`` (shape ``(1,)`` or ``(B,)``, ``float64``).
    Everything else — real-space pair sum, spline spread, FFT, Green's-function
    multiply, IFFT, spline gather — is unchanged. Real space uses the neighbor
    list directly (pair contributions, no global sum), so it is halo-correct
    without distribution-specific plumbing.

    When ``slab_correction=True`` the Yeh-Berkowitz slab term (with the
    Ballenegger non-neutral extension) is added element-wise into the result
    tuple. Its global per-system moments are computed owned-only and all-reduced
    across ranks via
    :func:`~nvalchemi.models._ops.electrostatics.slab.slab_compute_partial_moments`,
    so the correction is halo-correct (``pbc`` is the per-system ``(B, 3)`` bool
    mask; rows with exactly one ``False`` are slab systems).
    """
    num_atoms = positions.shape[0]

    cell, num_systems = _prepare_cell(cell)

    # Estimate parameters if not provided. Under distribution the caller must
    # ensure alpha / mesh_dimensions are computed from the global atom count,
    # since estimate_pme_parameters derives these from shape only.
    if alpha is None:
        params = estimate_pme_parameters(positions, cell, batch_idx, accuracy)
        alpha = params.alpha
        if mesh_dimensions is None and mesh_spacing is None:
            mesh_dimensions = tuple(params.mesh_dimensions)

    alpha = _prepare_alpha(alpha, num_systems, positions.dtype, positions.device)

    if mask_value is None:
        mask_value = num_atoms

    if mesh_dimensions is None:
        if mesh_spacing is not None:
            mesh_dimensions = mesh_spacing_to_dimensions(cell, mesh_spacing)
        else:
            mesh_dimensions = estimate_pme_mesh_dimensions(cell, alpha, accuracy)

    rs = ewald_real_space(
        positions=positions,
        charges=charges,
        cell=cell,
        alpha=alpha,
        neighbor_list=neighbor_list,
        neighbor_ptr=neighbor_ptr,
        neighbor_shifts=neighbor_shifts,
        neighbor_matrix=neighbor_matrix,
        neighbor_matrix_shifts=neighbor_matrix_shifts,
        mask_value=mask_value,
        batch_idx=batch_idx,
        compute_forces=compute_forces,
        compute_charge_gradients=compute_charge_gradients,
        compute_virial=compute_virial,
        hybrid_forces=hybrid_forces,
    )

    rec = pme_reciprocal_space_from_total_charge(
        positions=positions,
        charges=charges,
        cell=cell,
        alpha=alpha,
        total_charges=total_charges,
        mesh_dimensions=mesh_dimensions,
        spline_order=spline_order,
        batch_idx=batch_idx,
        compute_forces=compute_forces,
        compute_charge_gradients=compute_charge_gradients,
        compute_virial=compute_virial,
        k_vectors=k_vectors,
        k_squared=k_squared,
        hybrid_forces=hybrid_forces,
    )

    rs_tuple = rs if isinstance(rs, tuple) else (rs,)
    rec_tuple = rec if isinstance(rec, tuple) else (rec,)

    results = [r + s for r, s in zip(rs_tuple, rec_tuple)]

    if slab_correction:
        from nvalchemi.models._ops.electrostatics.slab import (  # noqa: PLC0415
            compute_slab_correction_from_moments,
        )

        # The slab tuple follows the same (energy, [forces], [charge_grads],
        # [virial]) ordering as the PME result, so it adds element-wise. Its
        # per-atom energy is float64 while rs/rec energies are too, so the sum
        # stays float64.
        slab = compute_slab_correction_from_moments(
            positions=positions,
            charges=charges,
            cell=cell,
            pbc=pbc,
            batch_idx=batch_idx,
            compute_forces=compute_forces,
            compute_charge_gradients=compute_charge_gradients,
            compute_virial=compute_virial,
        )
        slab_tuple = slab if isinstance(slab, tuple) else (slab,)
        results = [
            r + s.to(r.dtype) for r, s in zip(results, slab_tuple)
        ]

    if len(results) == 1:
        return results[0]
    return tuple(results)
