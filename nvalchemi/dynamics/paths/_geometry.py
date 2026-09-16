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
r"""Exact minimum-image convention (MIC) geometry for reaction-path implementations.

This module implements the minimum-image convention (MIC) used by reaction-path
algorithms such as interpolation, IDPP, and NEB. For a Cartesian displacement
:math:`\mathbf d`, it solves

.. math::

    \operatorname{MIC}(\mathbf d)
    = \underset{\mathbf n\in\mathbb Z^r}{\arg\min}
      \left\|\mathbf d-\mathbf nB\right\|,

where :math:`B\in\mathbb R^{r\times 3}` contains the :math:`r` periodic cell
vectors as rows, :math:`\mathbf n\in\mathbb Z^r` selects a periodic image, and
:math:`r\in\{0,1,2,3\}` is the number of periodic directions.

Independently rounding fractional coordinates is exact only for orthogonal
cells. For skew cells, finding the shortest Cartesian image is a closest-vector
problem on a lattice.

The implementation separates the work into two phases:
:func:`prepare_mic` performs relatively expensive CPU/float64 lattice analysis
during path setup, while :func:`minimum_image_displacement` performs cheap,
batched, GPU-friendly runtime evaluation. This separation is particularly
useful for NEB, which repeatedly evaluates changing positions with fixed cell
geometry.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from nvalchemi.data import Batch

_MIC_NONE = 0
_MIC_ORTHOGONAL = 1
_MIC_SEARCH = 2
_MAX_REDUCTION_STEPS = 256
_MAX_CLOSEST_VECTOR_CANDIDATES = 4096
_MAX_CANDIDATE_SHIFTS = 26


# =============================================================================
# Prepared MIC data
# =============================================================================


@dataclass(frozen=True, slots=True)
class PreparedMIC:
    """Cell-dependent buffers for minimum-image convention operations.

    Parameters
    ----------
    execution_mode : int
        Precomputed batch-wide runtime selector: zero for all nonperiodic paths,
        one when only orthogonal/nonperiodic paths occur, two when search is
        needed.
    max_candidate_count : int
        Preparation-time search bound across all paths (zero, eight, or 26).
    mode : Tensor, shape (num_paths,), dtype int32
        Per-path algorithm selector: ``0`` for nonperiodic passthrough,
        ``1`` for orthogonal-basis wrapping, and ``2`` for general
        reduced-basis candidate search.
    rank : Tensor, shape (num_paths,), dtype int32
        Number of periodic lattice vectors for each path, from zero to three.
    periodic_basis : Tensor, shape (num_paths, 3, 3)
        Reduced basis representing the same periodic translations as the cell.
        Active vectors occupy the first ``rank`` rows; unused rows are zero.
    cartesian_to_fractional : Tensor, shape (num_paths, 3, 3)
        Maps Cartesian row vectors to coordinates in ``periodic_basis``.
        Active coordinates occupy the first ``rank`` columns; unused columns
        are zero.
    candidate_count : Tensor, shape (num_paths,), dtype int32
        Number of active nonzero candidate translations for each path.
    candidate_shifts : Tensor, shape (num_paths, 26, 3)
        Fixed-capacity nonzero Cartesian translations. Unused rows are zero.

    Notes
    -----
    Floating-point buffers use the cell's dtype and device. Preparation must
    be repeated if the cell, periodicity, path ordering, dtype, or device
    changes.
    """

    execution_mode: int
    max_candidate_count: int
    mode: Tensor
    rank: Tensor
    periodic_basis: Tensor
    cartesian_to_fractional: Tensor
    candidate_count: Tensor
    candidate_shifts: Tensor


# =============================================================================
# Lattice geometry and reduction
# =============================================================================
# Replace the periodic cell rows with shorter vectors that generate the same
# lattice. Rank one needs no reduction, rank two uses Gauss reduction, and
# rank three uses Minkowski reduction with Gauss-reduced two-vector subproblems.
# The reduced basis enables an exact fixed-neighbor MIC search at runtime.


def _gram_schmidt_rows(basis: Tensor) -> tuple[Tensor, Tensor]:
    """Orthogonalize lattice-basis rows for Minkowski reduction.

    Parameters
    ----------
    basis : Tensor, shape (rank, 3)
        Cartesian lattice-basis vectors for one path.

    Returns
    -------
    orthogonal : Tensor, shape (rank, 3)
        Gram--Schmidt orthogonalized rows used to analyze the basis.
    mu : Tensor, shape (rank, rank)
        Projection coefficients; only entries below the diagonal are used.

    The outputs guide lattice reduction but do not replace the lattice basis.
    """
    rank = basis.shape[0]
    orthogonal = torch.zeros_like(basis)
    mu = torch.zeros((rank, rank), dtype=basis.dtype)
    for row in range(rank):
        orthogonal[row] = basis[row]
        for previous in range(row):
            denominator = torch.dot(orthogonal[previous], orthogonal[previous])
            mu[row, previous] = (
                torch.dot(basis[row], orthogonal[previous]) / denominator
            )
            orthogonal[row] -= mu[row, previous] * orthogonal[previous]
    return orthogonal, mu


def _cartesian_to_fractional(basis: Tensor) -> Tensor:
    """Construct a Cartesian-to-periodic coordinate map for one path.

    Parameters
    ----------
    basis : Tensor, shape (rank, 3)
        Reduced periodic lattice vectors stored as rows.

    Returns
    -------
    Tensor, shape (3, rank)
        Right inverse ``C`` satisfying ``basis @ C = I``. For a Cartesian
        row vector ``x``, ``x @ C`` gives its periodic coordinates and
        ``(x @ C) @ basis`` reconstructs its periodic-space projection.

    QR decomposition provides a stable map for partial periodicity, where the
    basis is rectangular and cannot be inverted directly.
    """
    q, r = torch.linalg.qr(basis.T, mode="reduced")
    identity = torch.eye(basis.shape[0], dtype=basis.dtype)
    inverse_r_transpose = torch.linalg.solve_triangular(r.T, identity, upper=False)
    result = q @ inverse_r_transpose
    if not torch.allclose(basis @ result, identity, rtol=1.0e-11, atol=1.0e-12):
        raise ValueError("Periodic cell vectors are numerically dependent")
    return result


def _gauss_reduce(basis: Tensor) -> tuple[Tensor, Tensor]:
    """Gauss-reduce a two-vector basis and return its unimodular transform."""
    reduced = basis.clone()
    transform = torch.eye(2, dtype=torch.int64)
    for _ in range(_MAX_REDUCTION_STEPS):
        norm_sq = torch.sum(reduced.square(), dim=1)
        if bool(norm_sq[1] < norm_sq[0]):
            reduced[[0, 1]] = reduced[[1, 0]]
            transform[[0, 1]] = transform[[1, 0]]
        coefficient = int(
            torch.round(
                torch.dot(reduced[0], reduced[1]) / torch.dot(reduced[0], reduced[0])
            ).item()
        )
        if coefficient == 0:
            return reduced, transform
        reduced[1] -= coefficient * reduced[0]
        transform[1] -= coefficient * transform[0]
    raise ValueError("MIC Gauss reduction did not converge")


def _closest_plane_coefficients(target: Tensor, basis: Tensor) -> Tensor:
    """Return the closest two-dimensional lattice coefficients to ``target``."""
    coordinate_map = _cartesian_to_fractional(basis)
    fractional = target @ coordinate_map
    nearest = torch.round(fractional).to(torch.int64)
    radius = torch.linalg.vector_norm(target - nearest.to(basis.dtype) @ basis)
    margins = radius * torch.linalg.vector_norm(coordinate_map, dim=0)
    tolerance = (
        32
        * torch.finfo(basis.dtype).eps
        * torch.maximum(torch.ones_like(margins), torch.abs(fractional))
    )
    lower = torch.ceil(fractional - margins - tolerance).to(torch.int64)
    upper = torch.floor(fractional + margins + tolerance).to(torch.int64)
    counts = upper - lower + 1
    candidate_count = int(torch.prod(counts).item())
    if candidate_count > _MAX_CLOSEST_VECTOR_CANDIDATES:
        raise ValueError(
            "Minkowski reduction requires too many closest-vector candidates"
        )

    best = nearest
    best_sq = torch.sum((target - nearest.to(basis.dtype) @ basis).square())
    ranges = [range(int(lower[index]), int(upper[index]) + 1) for index in range(2)]
    for values in product(*ranges):
        coefficients = torch.tensor(values, dtype=torch.int64)
        candidate = target - coefficients.to(basis.dtype) @ basis
        candidate_sq = torch.sum(candidate.square())
        if bool(candidate_sq < best_sq):
            best = coefficients
            best_sq = candidate_sq
    return best


def _is_minkowski_reduced(basis: Tensor) -> bool:
    """Return whether a rank-zero to rank-three basis is Minkowski-reduced."""
    if basis.shape[0] <= 1:
        return True
    if basis.shape[0] == 2:
        # Check vector ordering and whether b1 +/- b2 can shorten b2.
        coefficients = torch.tensor([[0, 1], [1, -1], [1, 1]], dtype=basis.dtype)
        reference = torch.tensor([0, 1, 1])
    else:
        # Check vector ordering and the pairwise and triple signed combinations
        # that characterize a Minkowski-reduced basis in three dimensions.
        coefficients = torch.tensor(
            [
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, -1, 0],
                [1, 0, -1],
                [0, 1, -1],
                [1, 1, 1],
                [1, -1, 1],
                [1, 1, -1],
                [1, -1, -1],
            ],
            dtype=basis.dtype,
        )
        reference = torch.tensor([0, 1, 1, 2, 2, 1, 2, 2, 2, 2, 2, 2])
    norm_sq = torch.sum(basis.square(), dim=1)
    # Each coefficient row forms a candidate lattice vector; ``reference``
    # selects the basis vector that the candidate must not be shorter than.
    lhs = torch.sum((coefficients @ basis).square(), dim=1)
    rhs = norm_sq.index_select(0, reference)
    # Conservative allowance for accumulated floating-point roundoff.
    tolerance = (
        64 * torch.finfo(basis.dtype).eps * torch.maximum(torch.ones_like(rhs), rhs)
    )
    return bool(torch.all(lhs + tolerance >= rhs))


def _validate_reduction(basis: Tensor, reduced: Tensor, transform: Tensor) -> None:
    """Validate lattice preservation and Minkowski-reduced output."""
    determinant = round(torch.linalg.det(transform.to(torch.float64)).item())
    if abs(determinant) != 1 or not torch.allclose(
        reduced,
        transform.to(basis.dtype) @ basis,
        rtol=1.0e-12,
        atol=1.0e-12,
    ):
        raise ValueError("MIC reduction failed its lattice-preservation check")
    if not _is_minkowski_reduced(reduced):
        raise ValueError("MIC reduction failed its Minkowski validation")


def _reduce_periodic_basis(basis: Tensor) -> Tensor:
    """Reduce a rank-one, rank-two, or rank-three periodic lattice basis.

    Reduction means replacing the input basis with shorter, less-skewed vectors
    that generate exactly the same lattice. That guarantees that the nearest image
    can be found among a small fixed neighborhood. The reduction strategy depends on
    the number of periodic vectors:

    - Rank one: return the basis unchanged.
    - Rank two: apply Gauss reduction.
    - Rank three: iteratively Gauss-reduce the two shortest vectors and shorten
      the third using the closest lattice vector in their plane.

    All transformations are unimodular integer row operations, so they preserve
    the lattice. Rank-two and rank-three results are validated before return.

    Parameters
    ----------
    basis : Tensor, shape (rank, 3)
        Linearly independent periodic lattice vectors, where ``rank`` is from
        one to three.

    Returns
    -------
    Tensor, shape (rank, 3)
        A reduced basis generating the same periodic lattice.
    """
    rank = basis.shape[0]
    if rank == 1:
        return basis.clone()
    if rank == 2:
        _, transform = _gauss_reduce(basis)
        reduced = transform.to(basis.dtype) @ basis
        _validate_reduction(basis, reduced, transform)
        return reduced

    transform = torch.eye(3, dtype=torch.int64)
    for _ in range(_MAX_REDUCTION_STEPS):
        # Step 1: order the basis vectors from shortest to longest.
        reduced = transform.to(basis.dtype) @ basis
        order = sorted(
            range(3),
            key=lambda row: (float(torch.dot(reduced[row], reduced[row])), row),
        )
        transform = transform[order]
        reduced = transform.to(basis.dtype) @ basis

        # Step 2: Gauss-reduce the plane spanned by the two shortest vectors.
        _, pair_transform = _gauss_reduce(reduced[:2])
        transform[:2] = pair_transform @ transform[:2]
        reduced = transform.to(basis.dtype) @ basis
        pair = reduced[:2]

        # Step 3: project the third vector into that plane and find its closest
        # lattice vector in the reduced two-dimensional basis.
        orthogonal, _ = _gram_schmidt_rows(reduced[:2])
        unit_x = orthogonal[0] / torch.linalg.vector_norm(orthogonal[0])
        unit_y = orthogonal[1] / torch.linalg.vector_norm(orthogonal[1])
        projected = (
            torch.dot(reduced[2], unit_x) * unit_x
            + torch.dot(reduced[2], unit_y) * unit_y
        )
        coefficients = _closest_plane_coefficients(projected, pair)

        # Step 4: subtract that plane-lattice vector to shorten the third vector.
        transform[2] -= coefficients[0] * transform[0]
        transform[2] -= coefficients[1] * transform[1]
        reduced = transform.to(basis.dtype) @ basis

        # Step 5: stop once the shortened third vector is no shorter than the
        # second; otherwise reorder and repeat with the new basis.
        norm_sq = torch.sum(reduced.square(), dim=1)
        tolerance = 64 * torch.finfo(basis.dtype).eps * max(1.0, float(norm_sq.max()))
        if bool(norm_sq[2] + tolerance >= norm_sq[1]):
            _validate_reduction(basis, reduced, transform)
            return reduced
    raise ValueError("MIC Minkowski reduction did not converge")


def _candidate_shifts(basis: Tensor) -> tuple[Tensor, int]:
    """Enumerate neighboring translations of a reduced lattice basis.

    Returns all nonzero translations ``coefficients @ basis`` for coefficient
    tuples in ``{-1, 0, 1}**rank``. The result is padded to the fixed rank-three
    capacity of 26 candidates.
    """
    coefficients = [
        values for values in product((-1, 0, 1), repeat=basis.shape[0]) if any(values)
    ]
    shifts = torch.zeros((_MAX_CANDIDATE_SHIFTS, 3), dtype=basis.dtype)
    count = len(coefficients)
    shifts[:count] = torch.tensor(coefficients, dtype=basis.dtype) @ basis
    return shifts, count


# =============================================================================
# MIC preparation and caching
# =============================================================================


def prepare_mic(cell: Tensor, pbc: Tensor) -> PreparedMIC:
    """Precompute minimum-image geometry for a batch of periodic cells.

    For each system, this function extracts the cell vectors selected by
    ``pbc``, reduces the resulting periodic lattice basis, and constructs the
    coordinate maps and neighboring lattice translations needed by
    :func:`minimum_image_displacement`.

    Each ``pbc[..., i]`` flag enables row ``i`` of ``cell`` as a periodic
    lattice vector.

    Preparation is performed on the CPU in ``float64`` for numerical
    robustness. The returned floating-point buffers are converted to the
    original cell's dtype and device.

    Parameters
    ----------
    cell : Tensor, shape (..., 3, 3)
        Cell vectors stored as rows. The dtype must be ``float32`` or
        ``float64``.
    pbc : Tensor, shape (..., 3), dtype bool
        Flags selecting the periodic cell-vector rows. Its batch dimensions
        must contain the same number of systems as ``cell``.

    Returns
    -------
    PreparedMIC
        Reusable minimum-image data for the prepared cells. The result may be
        passed to :func:`minimum_image_displacement` and reused while the
        cells, periodicity, system ordering, dtype, and device remain
        unchanged.

    Raises
    ------
    TypeError
        If ``cell`` is not ``float32`` or ``float64``, or if ``pbc`` is not
        Boolean.
    ValueError
        If ``cell`` and ``pbc`` describe different numbers of systems, are on
        different devices, or require more than 4096 candidates during lattice
        reduction.

    Notes
    -----
    Highly anisotropic but linearly independent cells can exceed the bounded
    candidate search used during reduction.
    """
    cells = cell.reshape(-1, 3, 3)
    pbcs = pbc.reshape(-1, 3)
    if cells.shape[0] != pbcs.shape[0]:
        raise ValueError("cell and pbc must contain the same number of systems")
    if cells.dtype not in {torch.float32, torch.float64}:
        raise TypeError("cell must have dtype float32 or float64")
    if pbcs.dtype != torch.bool:
        raise TypeError("pbc must have dtype bool")
    if cells.device != pbcs.device:
        raise ValueError("cell and pbc must be on the same device")

    if not bool(torch.any(pbcs)):
        num_systems = cells.shape[0]
        return PreparedMIC(
            execution_mode=_MIC_NONE,
            max_candidate_count=0,
            mode=torch.zeros(num_systems, dtype=torch.int32, device=cells.device),
            rank=torch.zeros(num_systems, dtype=torch.int32, device=cells.device),
            periodic_basis=torch.zeros_like(cells),
            cartesian_to_fractional=torch.zeros_like(cells),
            candidate_count=torch.zeros(
                num_systems, dtype=torch.int32, device=cells.device
            ),
            candidate_shifts=torch.zeros(
                (num_systems, _MAX_CANDIDATE_SHIFTS, 3),
                dtype=cells.dtype,
                device=cells.device,
            ),
        )

    host_cells = cells.detach().to(device="cpu", dtype=torch.float64)
    host_pbcs = pbcs.detach().to(device="cpu", dtype=torch.bool)
    modes: list[int] = []
    ranks: list[int] = []
    bases: list[Tensor] = []
    maps: list[Tensor] = []
    counts: list[int] = []
    all_shifts: list[Tensor] = []
    for index in range(host_cells.shape[0]):
        periodic_basis = host_cells[index, host_pbcs[index]]
        rank = periodic_basis.shape[0]
        padded_basis = torch.zeros((3, 3), dtype=torch.float64)
        padded_map = torch.zeros((3, 3), dtype=torch.float64)
        shifts = torch.zeros((_MAX_CANDIDATE_SHIFTS, 3), dtype=torch.float64)
        count = 0
        if rank == 0:
            mode = _MIC_NONE
        else:
            # Periodic cell vectors must be linearly independent
            reduced = _reduce_periodic_basis(periodic_basis)
            cartesian_to_fractional = _cartesian_to_fractional(reduced)
            padded_basis[:rank] = reduced
            padded_map[:, :rank] = cartesian_to_fractional
            gram = reduced @ reduced.T
            off_diagonal = gram - torch.diag(torch.diagonal(gram))
            if bool(torch.count_nonzero(off_diagonal) == 0):
                mode = _MIC_ORTHOGONAL
            else:
                mode = _MIC_SEARCH
                shifts, count = _candidate_shifts(reduced)
        modes.append(mode)
        ranks.append(rank)
        bases.append(padded_basis)
        maps.append(padded_map)
        counts.append(count)
        all_shifts.append(shifts)

    device, dtype = cells.device, cells.dtype
    return PreparedMIC(
        execution_mode=max(modes),
        max_candidate_count=max(counts),
        mode=torch.tensor(modes, dtype=torch.int32, device=device),
        rank=torch.tensor(ranks, dtype=torch.int32, device=device),
        periodic_basis=torch.stack(bases).to(device=device, dtype=dtype).contiguous(),
        cartesian_to_fractional=torch.stack(maps)
        .to(device=device, dtype=dtype)
        .contiguous(),
        candidate_count=torch.tensor(counts, dtype=torch.int32, device=device),
        candidate_shifts=torch.stack(all_shifts)
        .to(device=device, dtype=dtype)
        .contiguous(),
    )


def prepare_batch_mic(batch: Batch, cell: Tensor, pbc: Tensor) -> PreparedMIC:
    """Prepare or reuse path-level MIC buffers during batch setup.

    Parameters
    ----------
    batch : Batch
        Validated grouped paths owning the private ``_mic_data`` cache.
    cell : Tensor
        Representative cells, shape ``(num_paths, 3, 3)``, in the positions'
        dtype and device.
    pbc : Tensor
        Representative periodic flags, shape ``(num_paths, 3)``.

    Returns
    -------
    PreparedMIC
        Shared geometry for IDPP and NEB. Position updates do not invalidate it.

    Notes
    -----
    Setup compares detached snapshots to detect in-place geometry and layout
    changes. Re-run setup after changing cells, PBC, path layout, dtype, or
    device. Batch cloning and device transfers discard this private cache.
    This function must not run inside the compiled force-evaluation loop.
    """
    layout = batch.group_layout
    inputs = (cell, pbc, layout.group_ptr, layout.group_idx)
    cached = getattr(batch, "_mic_data", None)
    snapshots = getattr(batch, "_mic_inputs", None)
    if cached is not None and snapshots is not None:
        if all(
            current.device == previous.device
            and current.dtype == previous.dtype
            and torch.equal(current, previous)
            for current, previous in zip(inputs, snapshots, strict=True)
        ):
            return cached

    prepared = prepare_mic(cell, pbc)
    batch._mic_data = prepared
    batch._mic_inputs = tuple(value.detach().clone() for value in inputs)
    return prepared


# =============================================================================
# Runtime MIC
# =============================================================================


def minimum_image_displacement(
    displacement: Tensor,
    graph_idx: Tensor,
    cell: Tensor | None = None,
    pbc: Tensor | None = None,
    *,
    prepared: PreparedMIC | None = None,
) -> Tensor:
    """Return exact batched minimum-image Cartesian displacements.

    The component perpendicular to the span of the selected periodic lattice
    rows is preserved. Exact ties retain the wrapped representative, followed
    by the deterministic preparation-time candidate order.

    Parameters
    ----------
    displacement : Tensor, shape (num_displacements, 3)
        Cartesian row-vector displacements to wrap.
    graph_idx : Tensor, shape (num_displacements,), dtype int32 or int64
        Path index selecting the cell geometry for each displacement.
    cell : Tensor, shape (num_paths, 3, 3), optional
        Cell vectors stored as rows. Used to prepare MIC data when ``prepared``
        is not supplied.
    pbc : Tensor, shape (num_paths, 3), dtype bool, optional
        Flags selecting periodic cell-vector rows. Used with ``cell`` when
        ``prepared`` is not supplied.
    prepared : PreparedMIC, optional
        Reusable cell-dependent MIC data. When supplied, ``cell`` and ``pbc``
        are ignored.

    Returns
    -------
    Tensor, shape (num_displacements, 3)
        Minimum-image Cartesian displacements. If MIC geometry is unavailable
        or all paths are nonperiodic, the input tensor is returned unchanged.
    """
    if prepared is None:
        if cell is None or pbc is None:
            return displacement
        prepared = prepare_mic(cell, pbc)
    # Batch-wide metadata keeps dispatch independent of device tensor values.
    if prepared.execution_mode == _MIC_NONE:
        return displacement

    # Gather the prepared path geometry associated with each displacement.
    periodic_basis = prepared.periodic_basis.index_select(0, graph_idx)
    cartesian_to_fractional = prepared.cartesian_to_fractional.index_select(
        0, graph_idx
    )
    fractional = torch.bmm(displacement.unsqueeze(1), cartesian_to_fractional).squeeze(
        1
    )
    if prepared.execution_mode == _MIC_ORTHOGONAL:
        # Orthogonal lattices minimize independently in each fractional coordinate.
        translations = torch.bmm(
            torch.floor(fractional + 0.5).unsqueeze(1), periodic_basis
        ).squeeze(1)
        return displacement - translations

    # Use nearest wrapping for orthogonal paths and a fundamental-cell seed for
    # skew paths before searching their reduced-basis neighbors.
    mode = prepared.mode.index_select(0, graph_idx)
    orthogonal = mode == _MIC_ORTHOGONAL
    wrapped_nearest = fractional - torch.floor(
        fractional + 0.5
    )  # wraps into [-0.5, 0.5)
    wrapped_general = fractional - torch.floor(fractional)  # wraps into [0, 1)
    wrapped = torch.where(orthogonal[:, None], wrapped_nearest, wrapped_general)

    # Split each displacement into periodic and perpendicular components. Periodic
    # translations affect only the former, so compare candidates without the common
    # residual; this also avoids a large residual hiding differences in float32.
    projection = torch.bmm(fractional.unsqueeze(1), periodic_basis).squeeze(1)
    residual = displacement - projection
    wrapped_periodic = torch.bmm(wrapped.unsqueeze(1), periodic_basis).squeeze(1)

    best = wrapped_periodic
    best_sq = torch.sum(best.square(), dim=-1)
    candidate_count = prepared.candidate_count.index_select(0, graph_idx)

    # Search all neighboring cells; for a Minkowski-reduced rank-2 or rank-3
    # basis, these 8 or 26 shifts are guaranteed to include the minimum image.
    for shift_index in range(prepared.max_candidate_count):
        shift = prepared.candidate_shifts[:, shift_index].index_select(0, graph_idx)
        candidate = wrapped_periodic + shift
        candidate_sq = torch.sum(candidate.square(), dim=-1)
        improve = (shift_index < candidate_count) & (candidate_sq < best_sq)
        best = torch.where(improve[:, None], candidate, best)
        best_sq = torch.where(improve, candidate_sq, best_sq)
    return torch.where((mode != _MIC_NONE)[:, None], residual + best, displacement)


__all__ = [
    "PreparedMIC",
    "minimum_image_displacement",
    "prepare_batch_mic",
    "prepare_mic",
]
