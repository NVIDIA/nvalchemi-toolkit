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

"""Particle-based halo (ghost) exchange primitive.

Provides ``particle_halo_padding`` and ``particle_halo_unpadding``, the
particle analogue of physicsnemo's grid-based ``halo_padding`` /
``unhalo_padding`` in ``physicsnemo.domain_parallel.shard_utils.halo``.

Same contract:

- Forward: exchange ghost atoms, return a padded plain tensor.
- Backward: exchange ghost gradients back, accumulate into owned gradients.

Ghosts are identified by fractional-coordinate proximity to domain
boundaries (PBC-aware). The padded tensor is NOT a ShardTensor because
ghosts violate the ``sum(shard_sizes) == global_size`` invariant; ghost
metadata is ephemeral.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from nvalchemi.distributed._core.gather_primitives import (
    _halo_p2p_enabled,
    funcol_all_to_all_v_rows,
    halo_exchange_fixed,
    mesh_group,
)
from nvalchemi.distributed._core.halo_types import (
    GNNHaloMarkers,
    ParticleHaloConfig,
    ParticleHaloMetadata,
)

if TYPE_CHECKING:
    from nvalchemi.distributed.partitioner import SpatialPartitioner

logger = logging.getLogger(__name__)


# ======================================================================
# Ghost identification
# ======================================================================


def _rank_fractional_bounds(
    partitioner: SpatialPartitioner, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fractional bounds ``(frac_lo, frac_hi)`` for *rank*, each ``(3,)``."""
    lo_cell, hi_cell = partitioner.rank_to_cell_bounds(rank)
    cells_per_dim = partitioner.cells_per_dim
    device = partitioner.cell_matrix.device
    dtype = partitioner.cell_matrix.dtype
    frac_lo = torch.tensor(
        [lo_cell[d] / cells_per_dim[d] for d in range(3)], device=device, dtype=dtype
    )
    frac_hi = torch.tensor(
        [hi_cell[d] / cells_per_dim[d] for d in range(3)], device=device, dtype=dtype
    )
    return frac_lo, frac_hi


def _ghost_width_fractional(
    partitioner: SpatialPartitioner, ghost_width: float
) -> torch.Tensor:
    """Ghost width in fractional coordinates per dimension, shape ``(3,)``.

    With a row-vector convention where ``cell_matrix`` has rows ``(a, b, c)``
    (so ``cart = frac @ cell_matrix``), the reciprocal lattice vectors are the
    columns of ``inv(cell_matrix)`` — i.e. the rows of ``inv(cell_matrix).T``.
    ``||reciprocal_row||`` is the inverse interplanar spacing along that axis,
    so the fractional width of a Cartesian ``ghost_width`` shell is
    ``ghost_width * ||reciprocal_row||``.
    """
    cell = partitioner.cell_matrix
    # Reuse the partitioner's cached inverse; cell is fixed in NVT/NVE, so
    # recomputing inv_cell every step is pure overhead.
    inv_cell_rows_T = partitioner._inv_cell.T  # rows are reciprocal vectors
    norms = torch.linalg.norm(inv_cell_rows_T, dim=1)
    return (ghost_width * norms).to(dtype=cell.dtype)


def _check_halo_region(
    frac_pos: torch.Tensor,
    frac_lo: torch.Tensor,
    frac_hi: torch.Tensor,
    gw_frac: torch.Tensor,
) -> torch.Tensor:
    """Return ``(N,)`` bool mask for atoms in the halo of a domain box.

    The extent check uses inclusive inequalities so atoms whose (possibly
    PBC-shifted) position sits exactly on the receiver's domain boundary are
    still counted as halo — otherwise lattice atoms at integer multiples of
    the cell (e.g. FCC basis at Z=0 shifted to Z=box) fall through the gap.

    The mask is computed from sender-owned atoms needed by a receiver domain.
    ``particle_halo_padding`` receives only atoms owned by the sender. During
    migration hysteresis, one of those atoms may already be geometrically
    inside the receiver's core while ownership still belongs to the sender.
    Include the full receiver extent, core plus ghost width, so the receiver
    gets that atom as a ghost until ownership migrates.
    """
    expanded_lo = frac_lo - gw_frac
    expanded_hi = frac_hi + gw_frac

    inside = (frac_pos >= expanded_lo) & (frac_pos <= expanded_hi)
    return inside.all(dim=1)


def _identify_ghosts_split(
    positions: torch.Tensor,
    neighbor_rank: int,
    config: ParticleHaloConfig,
) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    """Return ``(direct_mask, [(pbc_mask, cart_shift), ...])`` for one neighbor."""
    partitioner = config.partitioner
    # Reuse the cached inverse; cell is fixed in NVT/NVE (see
    # ``_ghost_width_fractional``).
    inv_cell = partitioner._inv_cell.to(device=positions.device, dtype=positions.dtype)

    # cart = frac @ cell_matrix (rows of cell_matrix = lattice vectors a, b, c),
    # so frac = cart @ inv(cell_matrix). Using inv(cell).T instead is only
    # correct for orthorhombic (diagonal) cells; for skew cells it yields wrong
    # fractional coordinates that miss PBC halo atoms at cell boundaries and
    # under-count neighbors on per-rank neighbor lists.
    frac_pos = positions @ inv_cell
    gw_frac = _ghost_width_fractional(partitioner, config.ghost_width).to(
        device=positions.device
    )
    frac_lo, frac_hi = _rank_fractional_bounds(partitioner, neighbor_rank)
    frac_lo = frac_lo.to(device=positions.device)
    frac_hi = frac_hi.to(device=positions.device)

    direct_mask = _check_halo_region(frac_pos, frac_lo, frac_hi, gw_frac)

    # Exclude atoms from each PBC-variant mask that have already been selected
    # in a previous (direct or PBC) mask. Otherwise a single atom can be sent
    # to the same neighbor rank multiple times — once at its raw position and
    # once shifted — which produces duplicate halo entries and, in turn,
    # duplicate edges on the receiver's side.
    already_selected = direct_mask.clone()
    pbc_list: list[tuple[torch.Tensor, torch.Tensor]] = []
    shift_key = (config.rank, neighbor_rank)
    if shift_key in config.pbc_shifts:
        for cart_shift in config.pbc_shifts[shift_key]:
            cart_shift = cart_shift.to(device=positions.device, dtype=positions.dtype)
            # Same transpose fix as above: ``frac = cart @ inv(cell)``.
            frac_shift = cart_shift @ inv_cell
            frac_pos_shifted = frac_pos + frac_shift
            mask = _check_halo_region(frac_pos_shifted, frac_lo, frac_hi, gw_frac)
            mask = mask & ~already_selected
            if mask.any():
                pbc_list.append((mask, cart_shift))
                already_selected = already_selected | mask

    return direct_mask, pbc_list


def _compute_ghost_masks_batched(
    positions: torch.Tensor, config: ParticleHaloConfig
) -> dict[int, tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]]:
    """Compute split ghost masks for all neighbors."""
    masks: dict[int, tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]] = {}
    for nr in config.neighbor_ranks:
        masks[nr] = _identify_ghosts_split(positions, nr, config)
    return masks


# ======================================================================
# Routing: build send indices + extended tensor
# ======================================================================


def _build_send_data(
    positions: torch.Tensor,
    ghost_masks: dict[
        int, tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]
    ],
    config: ParticleHaloConfig,
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    list[list[int]],
    list[torch.Tensor],
]:
    """Build the extended position tensor and per-rank send indices.

    Returns ``(extended_positions, send_indices, sizes, send_indices_owned)`` where:

    - extended_positions: ``[owned | PBC-shifted copies]``
    - send_indices[r]: indices into extended_positions to send to rank r
    - sizes: all_gathered sizes matrix ``sizes[i][j]`` = count rank i sends to rank j
    - send_indices_owned[r]: parallel to send_indices[r] but using the owned-tensor
      index of each row (PBC copies collapsed to their source). Used by
      autograd-aware feature exchange where PBC shifts are identity.
    """
    device = positions.device
    group = mesh_group(config.mesh)
    world_size = dist.get_world_size(group=group)

    # Build extended tensor with PBC-shifted copies appended, tracking each
    # PBC row's owned source so features can reuse the owned index.
    pbc_parts: list[torch.Tensor] = []
    pbc_index_maps: dict[int, list[torch.Tensor]] = {}
    pbc_owned_source_maps: dict[int, list[torch.Tensor]] = {}
    pbc_offset = positions.shape[0]

    for nr in config.neighbor_ranks:
        _direct_mask, pbc_list = ghost_masks[nr]
        nr_idx_parts: list[torch.Tensor] = []
        nr_owned_parts: list[torch.Tensor] = []
        for mask, cart_shift in pbc_list:
            pbc_pos = positions[mask].clone()
            pbc_pos = pbc_pos + cart_shift.to(device=device, dtype=pbc_pos.dtype)
            n_pbc = pbc_pos.shape[0]
            nr_idx_parts.append(
                torch.arange(
                    pbc_offset, pbc_offset + n_pbc, device=device, dtype=torch.int64
                )
            )
            nr_owned_parts.append(torch.where(mask)[0].to(torch.int64))
            pbc_offset += n_pbc
            pbc_parts.append(pbc_pos)
        if nr_idx_parts:
            pbc_index_maps[nr] = nr_idx_parts
            pbc_owned_source_maps[nr] = nr_owned_parts

    if pbc_parts:
        extended_positions = torch.cat([positions, *pbc_parts], dim=0)
    else:
        extended_positions = positions

    # Build per-rank send indices, both extended and owned-only variants.
    send_indices: list[torch.Tensor] = []
    send_indices_owned: list[torch.Tensor] = []
    for r in range(world_size):
        if r not in ghost_masks:
            send_indices.append(torch.empty(0, dtype=torch.int64, device=device))
            send_indices_owned.append(torch.empty(0, dtype=torch.int64, device=device))
            continue
        direct_mask, _pbc_list = ghost_masks[r]
        parts_ext: list[torch.Tensor] = []
        parts_own: list[torch.Tensor] = []

        if direct_mask.any():
            direct_idx = torch.where(direct_mask)[0].to(torch.int64)
            parts_ext.append(direct_idx)
            parts_own.append(direct_idx)
        if r in pbc_index_maps:
            parts_ext.extend(pbc_index_maps[r])
            parts_own.extend(pbc_owned_source_maps[r])

        if parts_ext:
            idx_ext = torch.cat(parts_ext, dim=0)
            idx_own = torch.cat(parts_own, dim=0)
        else:
            idx_ext = torch.empty(0, dtype=torch.int64, device=device)
            idx_own = torch.empty(0, dtype=torch.int64, device=device)
        send_indices.append(idx_ext)
        send_indices_owned.append(idx_own)

    # All-gather send counts → sizes matrix.
    local_counts = torch.zeros(world_size, dtype=torch.int64, device=device)
    for r in range(world_size):
        local_counts[r] = send_indices[r].shape[0]

    all_counts_list = [torch.zeros_like(local_counts) for _ in range(world_size)]
    dist.all_gather(all_counts_list, local_counts, group=group)
    # The halo all-to-all-v split counts must be host ints, but materialize them
    # with a *single* device→host sync (one ``.tolist()`` on the stacked matrix)
    # rather than one blocking sync per rank on the hot per-step halo path.
    sizes = torch.stack(all_counts_list).tolist()

    return extended_positions, send_indices, sizes, send_indices_owned


# ======================================================================
# Public API
# ======================================================================


def particle_halo_padding(
    positions: torch.Tensor,
    config: ParticleHaloConfig,
) -> tuple[torch.Tensor, ParticleHaloMetadata]:
    """Exchange ghost atoms with domain neighbors.

    This is the particle analogue of physicsnemo's ``halo_padding()``:
    identifies atoms near domain boundaries, sends copies to neighbors,
    returns a padded plain tensor.

    Parameters
    ----------
    positions : torch.Tensor
        ``(N_owned, 3)`` owned atom positions (plain tensor).
    config : ParticleHaloConfig
        Halo exchange configuration.

    Returns
    -------
    tuple[torch.Tensor, ParticleHaloMetadata]
        ``(padded_positions, metadata)`` where padded_positions is
        ``(N_owned + N_ghost, 3)`` and metadata enables stripping.
    """
    # Single-process is a no-op (no peers to borrow ghost rows from). Gate on
    # the default group's world size so we return before touching ``config.mesh``,
    # which stays robust even when an ambient 1-rank group is up.
    if not dist.is_initialized() or dist.get_world_size() == 1:
        meta = ParticleHaloMetadata(
            n_owned=positions.shape[0],
            n_padded=positions.shape[0],
            send_indices=[],
            send_sizes=[],
            recv_sizes=[],
            gnn_markers=GNNHaloMarkers(send_indices_owned=[]),
        )
        return positions, meta

    n_owned = positions.shape[0]

    # 1. Identify ghosts per neighbor.
    ghost_masks = _compute_ghost_masks_batched(positions, config)

    # 2. Build extended tensor + send indices + sizes.
    extended_positions, send_indices, sizes, send_indices_owned = _build_send_data(
        positions, ghost_masks, config
    )

    # Exchange ghost rows through the neighbor point-to-point path when enabled,
    # else physicsnemo's indexed ``all_to_all_v``.
    group = mesh_group(config.mesh)
    world_size = dist.get_world_size(group=group)
    if _halo_p2p_enabled():
        received = _funcol_indexed_all_to_all_v_rows(
            extended_positions,
            send_indices,
            sizes,
            config.mesh,
            config.rank,
            world_size,
        )
    else:
        from physicsnemo.distributed.utils import indexed_all_to_all_v_wrapper

        received = indexed_all_to_all_v_wrapper(
            tensor=extended_positions,
            indices=send_indices,
            sizes=sizes,
            dim=0,
            group=group,
        )

    # 4. Padded output: [owned | ghosts].
    if received.shape[0] > 0:
        padded = torch.cat([positions, received], dim=0)
    else:
        padded = positions

    meta = ParticleHaloMetadata(
        n_owned=n_owned,
        n_padded=padded.shape[0],
        send_indices=send_indices,
        send_sizes=sizes,
        recv_sizes=sizes,  # symmetric — recv_sizes[j][i] = send_sizes[i][j]
        gnn_markers=GNNHaloMarkers(send_indices_owned=send_indices_owned),
    )

    logger.info(
        "[rank %d] particle_halo_padding: %d owned + %d ghosts = %d total",
        config.rank,
        n_owned,
        padded.shape[0] - n_owned,
        padded.shape[0],
    )

    return padded, meta


def particle_halo_padding_multi(
    positions: torch.Tensor,
    other_fields: dict[str, torch.Tensor],
    config: ParticleHaloConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], ParticleHaloMetadata]:
    """Exchange ghost atoms for positions and associated fields.

    Ghost routing is computed from *positions* once, then the same
    routing is applied to all other fields.

    Parameters
    ----------
    positions : torch.Tensor
        ``(N_owned, 3)`` owned positions.
    other_fields : dict[str, torch.Tensor]
        Other per-atom fields to exchange (e.g. velocities, atomic_numbers).
        Ghost copies get **zero** values for these fields.
    config : ParticleHaloConfig
        Halo exchange configuration.

    Returns
    -------
    tuple[torch.Tensor, dict[str, torch.Tensor], ParticleHaloMetadata]
        ``(padded_positions, padded_fields, metadata)``
    """
    padded_pos, meta = particle_halo_padding(positions, config)
    n_ghosts = meta.n_padded - meta.n_owned
    device = positions.device

    padded_fields: dict[str, torch.Tensor] = {}
    for name, tensor in other_fields.items():
        if n_ghosts == 0:
            padded_fields[name] = tensor
            continue

        # Build ghost values: zeros matching tensor shape.
        if tensor.ndim == 1:
            ghost_vals = torch.zeros(n_ghosts, dtype=tensor.dtype, device=device)
        else:
            ghost_shape = (n_ghosts,) + tensor.shape[1:]
            ghost_vals = torch.zeros(ghost_shape, dtype=tensor.dtype, device=device)

        padded_fields[name] = torch.cat([tensor, ghost_vals], dim=0)

    return padded_pos, padded_fields, meta


def particle_halo_unpadding(
    padded: torch.Tensor,
    meta: ParticleHaloMetadata,
) -> torch.Tensor:
    """Strip ghost atoms, returning only the owned portion.

    Parameters
    ----------
    padded : torch.Tensor
        ``(N_padded, ...)`` tensor with owned + ghost atoms.
    meta : ParticleHaloMetadata
        Metadata from ``particle_halo_padding``.

    Returns
    -------
    torch.Tensor
        ``(N_owned, ...)`` owned-only tensor.
    """
    return padded[: meta.n_owned]


# ======================================================================
# Autograd-aware feature exchange
# ======================================================================


def _require_markers(meta: ParticleHaloMetadata) -> GNNHaloMarkers:
    if meta.gnn_markers is None:
        raise ValueError(
            "ParticleHaloMetadata is missing gnn_markers; call "
            "particle_halo_padding() to populate them."
        )
    return meta.gnn_markers


def _funcol_indexed_all_to_all_v_rows(
    tensor: torch.Tensor,
    indices: list[torch.Tensor],
    sizes: list[list[int]],
    mesh: Any,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """AOT-traceable funcol analogue of physicsnemo's
    ``indexed_all_to_all_v_wrapper`` for ``dim=0``.

    For each destination rank ``j`` gather ``tensor[indices[j]]`` (``sizes[rank][j]``
    rows) into a dest-ordered send buffer; receive ``sizes[i][rank]`` rows from
    each source ``i``. Returns the received rows concatenated in source-rank
    order — same contract as the physicsnemo wrapper. ``sizes`` is precomputed
    (halo metadata), so the split sizes are graph constants under compile.
    """
    send_rows = torch.cat(
        [tensor.index_select(0, indices[j]) for j in range(world_size)], dim=0
    )
    send_counts = [int(sizes[rank][j]) for j in range(world_size)]
    recv_counts = [int(sizes[i][rank]) for i in range(world_size)]
    return funcol_all_to_all_v_rows(send_rows, send_counts, recv_counts, mesh)


def _halo_accumulate_to_owners(
    halo: torch.Tensor,
    meta: ParticleHaloMetadata,
    config: ParticleHaloConfig,
) -> torch.Tensor:
    """Transpose of the forward halo gather.

    Takes a halo-region tensor ``(n_halo, *F)`` whose rows are laid out as
    ``[from_rank_0 | from_rank_1 | ...]`` and returns an owned-region tensor
    ``(n_owned, *F)`` where each owned row accumulates contributions from every
    rank that had borrowed it. Pure communication + index_add, no autograd.
    """
    markers = _require_markers(meta)
    world_size = len(meta.send_sizes)
    rank = config.rank
    device = halo.device
    dtype = halo.dtype

    # Partition halo by source rank; build reverse send indices.
    rev_indices: list[torch.Tensor] = []
    offset = 0
    for r in range(world_size):
        n_from_r = meta.send_sizes[r][rank]
        rev_indices.append(
            torch.arange(offset, offset + n_from_r, device=device, dtype=torch.int64)
        )
        offset += n_from_r

    rev_sizes = [
        [meta.send_sizes[j][i] for j in range(world_size)] for i in range(world_size)
    ]

    received_back = _funcol_indexed_all_to_all_v_rows(
        halo, rev_indices, rev_sizes, config.mesh, rank, world_size
    )

    # fp64 accumulation when inputs are fp32 (boundary atoms fold many cross-rank
    # contributions; atomic-add order is GPU-nondeterministic) -> downcast at end.
    _acc_dt = torch.float64 if dtype == torch.float32 else dtype
    accumulated = torch.zeros(
        (meta.n_owned,) + halo.shape[1:], dtype=_acc_dt, device=device
    )
    offset = 0
    for j in range(world_size):
        n_to_j = meta.send_sizes[rank][j]
        if n_to_j == 0:
            continue
        chunk = received_back[offset : offset + n_to_j]
        accumulated.index_add_(0, markers.send_indices_owned[j], chunk.to(_acc_dt))
        offset += n_to_j

    return accumulated.to(dtype) if _acc_dt != dtype else accumulated


def _halo_gather_from_owners(
    owned: torch.Tensor,
    meta: ParticleHaloMetadata,
    config: ParticleHaloConfig,
) -> torch.Tensor:
    """Forward halo gather: fetch neighbors' owned rows into local halo region.

    Returns the halo-only tensor ``(n_halo, *F)`` without the owned prefix.
    """
    markers = _require_markers(meta)
    world_size = len(meta.send_sizes)
    return _funcol_indexed_all_to_all_v_rows(
        owned,
        markers.send_indices_owned,
        meta.send_sizes,
        config.mesh,
        config.rank,
        world_size,
    )


class _HaloForwardExchange(torch.autograd.Function):
    """Owned rows → padded ``[owned | halo]`` with autograd-correct backward."""

    @staticmethod
    def forward(  # type: ignore[override]
        features: torch.Tensor,
        meta: ParticleHaloMetadata,
        config: ParticleHaloConfig,
    ) -> torch.Tensor:
        received = _halo_gather_from_owners(features, meta, config)
        return torch.cat([features, received], dim=0)

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        features, meta, config = inputs
        ctx.meta = meta
        ctx.config = config
        ctx.n_owned = features.shape[0]

    @staticmethod
    def backward(ctx: Any, grad_padded: torch.Tensor) -> tuple:  # type: ignore[override]
        meta = ctx.meta
        config = ctx.config
        n_owned = ctx.n_owned
        grad_direct = grad_padded[:n_owned]
        grad_halo = grad_padded[n_owned:]
        grad_from_halo = _halo_accumulate_to_owners(
            grad_halo.contiguous(), meta, config
        )
        return grad_direct + grad_from_halo, None, None


class _HaloReverseExchange(torch.autograd.Function):
    """Padded ``[owned | halo]`` → owned with halo contributions accumulated."""

    @staticmethod
    def forward(  # type: ignore[override]
        padded: torch.Tensor,
        meta: ParticleHaloMetadata,
        config: ParticleHaloConfig,
    ) -> torch.Tensor:
        n_owned = meta.n_owned
        owned_direct = padded[:n_owned]
        halo = padded[n_owned:]
        accumulated = _halo_accumulate_to_owners(halo.contiguous(), meta, config)
        return owned_direct + accumulated

    @staticmethod
    def setup_context(ctx: Any, inputs: tuple, output: torch.Tensor) -> None:
        _padded, meta, config = inputs
        ctx.meta = meta
        ctx.config = config
        ctx.n_owned = meta.n_owned

    @staticmethod
    def backward(ctx: Any, grad_owned: torch.Tensor) -> tuple:  # type: ignore[override]
        meta = ctx.meta
        config = ctx.config
        received = _halo_gather_from_owners(grad_owned.contiguous(), meta, config)
        grad_padded = torch.cat([grad_owned, received], dim=0)
        return grad_padded, None, None


def halo_forward_exchange(
    features: torch.Tensor,
    meta: ParticleHaloMetadata,
    config: ParticleHaloConfig,
) -> torch.Tensor:
    """Exchange owned feature rows and return a padded ``[owned | halo]`` tensor.

    Autograd-aware: the backward pass accumulates halo-row gradients back
    into the ranks that own the source atoms. This is the primitive used to
    refresh a GNN's node features at the start of each message-passing layer.

    Parameters
    ----------
    features : torch.Tensor
        ``(N_owned, *F)`` owned atom features on the local rank.
    meta : ParticleHaloMetadata
        Metadata from :func:`particle_halo_padding` (must have ``gnn_markers``).
    config : ParticleHaloConfig
        The halo config used to produce ``meta``.

    Returns
    -------
    torch.Tensor
        ``(N_padded, *F)`` tensor with halo rows filled from neighbors.
    """
    return _HaloForwardExchange.apply(features, meta, config)


def particle_halo_padding_autograd(
    positions: torch.Tensor,
    config: ParticleHaloConfig,
) -> tuple[torch.Tensor, ParticleHaloMetadata]:
    """Autograd-aware equivalent of :func:`particle_halo_padding`.

    Exchanges halo positions via :class:`_HaloForwardExchange` so that the
    returned padded tensor is differentiable w.r.t. the input positions.
    PBC shifts are applied as a detached additive vector — they carry no
    gradient (shifts are determined by discrete rank topology, not atomic
    coordinates, so treating them as constants is correct).

    Use this when forces are to be computed via
    ``torch.autograd.grad(energy, positions)``.
    """
    with torch.no_grad():
        padded_ref, meta = particle_halo_padding(positions.detach(), config)
        halo_unshifted = _halo_gather_from_owners(positions.detach(), meta, config)
        halo_shift = padded_ref[meta.n_owned :] - halo_unshifted

    padded_unshifted = halo_forward_exchange(positions, meta, config)
    shift_vec = torch.zeros_like(padded_unshifted)
    if halo_shift.numel() > 0:
        shift_vec[meta.n_owned :] = halo_shift
    return padded_unshifted + shift_vec, meta


def pad_field(
    shard_or_tensor: Any,
    meta: ParticleHaloMetadata,
    config: ParticleHaloConfig,
) -> torch.Tensor:
    """Gather the halo slice of a per-row field (accepts either a
    ShardTensor-like object exposing ``.to_local()`` or a plain tensor)
    and concat onto the owned rows."""
    local = (
        shard_or_tensor.to_local()
        if hasattr(shard_or_tensor, "to_local")
        else shard_or_tensor
    )
    halo = _halo_gather_from_owners(local, meta, config)
    return torch.cat([local, halo], dim=0)


# ======================================================================
# Compile-path halo correction (custom op).
# ======================================================================


def _halo_a2a_v_default_group(
    tensor: torch.Tensor,
    indices: list[torch.Tensor],
    sizes: list[list[int]],
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """:func:`_funcol_indexed_all_to_all_v_rows` over the DEFAULT process group.

    Used inside the halo-correction custom op, which runs eagerly at runtime and
    cannot take a ``DeviceMesh`` arg. Valid for single-domain-mesh-dim topology
    (the domain group IS the world group); ``funcol`` mesh=None resolves the
    default group.
    """
    send_rows = torch.cat(
        [tensor.index_select(0, indices[j]) for j in range(world_size)], dim=0
    )
    send_counts = [int(sizes[rank][j]) for j in range(world_size)]
    recv_counts = [int(sizes[i][rank]) for i in range(world_size)]
    return funcol_all_to_all_v_rows(send_rows, send_counts, recv_counts, None)


def _halo_scatter_correct_dense(
    padded: torch.Tensor,
    send_indices: list[torch.Tensor],
    send_sizes: list[list[int]],
    n_owned: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """``halo_forward_exchange(halo_reverse_exchange(padded))`` as pure
    compute + collective (no autograd.Function, default group)."""
    # reverse: fold borrowed halo rows back into their owners.
    halo = padded[n_owned:].contiguous()
    rev_indices: list[torch.Tensor] = []
    off = 0
    for r in range(world_size):
        n = int(send_sizes[r][rank])
        rev_indices.append(
            torch.arange(off, off + n, device=padded.device, dtype=torch.int64)
        )
        off += n
    rev_sizes = [
        [int(send_sizes[j][i]) for j in range(world_size)] for i in range(world_size)
    ]
    received_back = _halo_a2a_v_default_group(
        halo, rev_indices, rev_sizes, rank, world_size
    )
    _acc_dt = torch.float64 if padded.dtype == torch.float32 else padded.dtype
    owned = padded[:n_owned].to(_acc_dt)
    off = 0
    for j in range(world_size):
        n = int(send_sizes[rank][j])
        if n == 0:
            continue
        owned = owned.index_add(
            0, send_indices[j], received_back[off : off + n].to(_acc_dt)
        )
        off += n
    owned = owned.to(padded.dtype)  # downcast before the move-only forward broadcast
    # forward: refresh halo rows from the (now corrected) owners.
    halo_new = _halo_a2a_v_default_group(
        owned, send_indices, send_sizes, rank, world_size
    )
    return torch.cat([owned, halo_new], dim=0)


@torch.library.custom_op("nvalchemi::halo_scatter_correct", mutates_args=())
def halo_scatter_correct_op(
    padded: torch.Tensor,
    send_idx_flat: list[int],
    send_idx_lens: list[int],
    send_sizes_flat: list[int],
    n_owned: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Dispatcher-visible halo scatter-correction for the compiled path.

    Transpose of :func:`halo_forward_op`: each halo (ghost) row's contribution
    is scattered back and summed into its owning rank's owned row, yielding the
    halo-corrected owned block. Runs eagerly at runtime (real tensors + default
    group); the trace sees only :func:`_halo_scatter_correct_fake`. The marker
    arrays ride as ``int[]`` (not tensors) so inductor lowering does not see a
    real-Tensor constant alongside the fake ``padded`` input.

    Parameters
    ----------
    padded : torch.Tensor
        ``(n_owned + n_halo, *F)`` tensor laid out as ``[owned | halo]`` (halo
        rows grouped by source rank).
    send_idx_flat : list[int]
        Concatenation of the per-destination-rank send-index lists (which owned
        rows this rank sent to each peer), flattened for the op boundary.
    send_idx_lens : list[int]
        Length of each per-destination slice in ``send_idx_flat`` — splits it
        back into ``world_size`` index tensors.
    send_sizes_flat : list[int]
        Row-major flattening of the ``world_size × world_size`` send-counts
        matrix; ``send_sizes[i][j]`` = rows rank ``i`` sent to rank ``j``.
    n_owned : int
        Number of owned rows = length of the returned block.
    rank : int
        This rank's index in the mesh group.
    world_size : int
        Number of ranks in the mesh group.

    Returns
    -------
    torch.Tensor
        ``(n_owned, *F)`` owned block with every borrowed ghost contribution
        summed back into its owner row.
    """
    _flat_t = torch.tensor(send_idx_flat, dtype=torch.int64, device=padded.device)
    send_indices = list(torch.split(_flat_t, send_idx_lens)) if send_idx_lens else []
    send_sizes = [
        [send_sizes_flat[i * world_size + j] for j in range(world_size)]
        for i in range(world_size)
    ]
    return _halo_scatter_correct_dense(
        padded, send_indices, send_sizes, n_owned, rank, world_size
    )


@halo_scatter_correct_op.register_fake
def _halo_scatter_correct_fake(
    padded, send_idx_flat, send_idx_lens, send_sizes_flat, n_owned, rank, world_size
):
    return torch.empty_like(padded)


def _halo_correct_setup_context(ctx, inputs, output):  # type: ignore[no-untyped-def]
    (_padded, send_idx_flat, send_idx_lens, send_sizes_flat, n_owned, rank, ws) = inputs
    ctx.send_idx_flat = send_idx_flat
    ctx.send_idx_lens = send_idx_lens
    ctx.send_sizes_flat = send_sizes_flat
    ctx.n_owned = n_owned
    ctx.rank = rank
    ctx.world_size = ws


def _halo_correct_backward(ctx, grad):  # type: ignore[no-untyped-def]
    # forward(reverse(.)) is self-adjoint -> the VJP is the op applied to grad.
    grad_in = halo_scatter_correct_op(
        grad.contiguous(),
        ctx.send_idx_flat,
        ctx.send_idx_lens,
        ctx.send_sizes_flat,
        ctx.n_owned,
        ctx.rank,
        ctx.world_size,
    )
    return grad_in, None, None, None, None, None, None


halo_scatter_correct_op.register_autograd(
    _halo_correct_backward, setup_context=_halo_correct_setup_context
)


@torch.library.custom_op("nvalchemi::halo_forward", mutates_args=())
def halo_forward_op(
    owned: torch.Tensor,
    send_idx_flat: list[int],
    send_idx_lens: list[int],
    send_sizes_flat: list[int],
    n_padded: int,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """Owned rows → padded ``[owned | halo]``: gather neighbours' owned rows into
    this rank's halo (ghost) region.

    Compile-safe counterpart of :func:`halo_forward_exchange`; runs eagerly at
    runtime (default group), while the trace sees only the registered fake. Its
    adjoint (backward) is :func:`halo_scatter_correct_op`. The marker arrays ride
    as ``int[]`` (not tensors) so inductor lowering does not see a real-Tensor
    constant alongside the fake ``owned`` input.

    Parameters
    ----------
    owned : torch.Tensor
        ``(n_owned, *F)`` this rank's owned rows.
    send_idx_flat : list[int]
        Concatenated per-destination-rank send-index lists (which owned rows go
        to each peer), flattened for the op boundary.
    send_idx_lens : list[int]
        Length of each per-destination slice in ``send_idx_flat``.
    send_sizes_flat : list[int]
        Row-major ``world_size × world_size`` send-counts matrix;
        ``send_sizes[i][j]`` = rows rank ``i`` sends to rank ``j``.
    n_padded : int
        Expected total rows of the result (``n_owned + n_halo``); the registered
        fake uses it to shape the traced output.
    rank : int
        This rank's index in the mesh group.
    world_size : int
        Number of ranks in the mesh group.

    Returns
    -------
    torch.Tensor
        ``(n_padded, *F)`` = ``[owned | halo]``, the halo region filled from
        peers' owned rows (ordered by source rank).
    """
    _flat_t = torch.tensor(send_idx_flat, dtype=torch.int64, device=owned.device)
    send_indices = list(torch.split(_flat_t, send_idx_lens)) if send_idx_lens else []
    send_sizes = [
        [send_sizes_flat[i * world_size + j] for j in range(world_size)]
        for i in range(world_size)
    ]
    halo = _halo_a2a_v_default_group(owned, send_indices, send_sizes, rank, world_size)
    return torch.cat([owned, halo], dim=0)


@halo_forward_op.register_fake
def _halo_forward_fake(
    owned, send_idx_flat, send_idx_lens, send_sizes_flat, n_padded, rank, world_size
):
    return owned.new_empty((n_padded,) + tuple(owned.shape[1:]))


def _halo_forward_setup_context(ctx, inputs, output):  # type: ignore[no-untyped-def]
    owned, send_idx_flat, send_idx_lens, send_sizes_flat, _n_padded, rank, ws = inputs
    ctx.send_idx_flat = send_idx_flat
    ctx.send_idx_lens = send_idx_lens
    ctx.send_sizes_flat = send_sizes_flat
    ctx.n_owned = owned.shape[0]
    ctx.rank = rank
    ctx.world_size = ws


def _halo_forward_backward(ctx, grad_padded):  # type: ignore[no-untyped-def]
    # Adjoint of the forward gather is the reverse accumulate: owned-row grad +
    # the halo-row grads folded back into their owners.
    ws = ctx.world_size
    _flat_t = torch.tensor(
        ctx.send_idx_flat, dtype=torch.int64, device=grad_padded.device
    )
    send_indices = (
        list(torch.split(_flat_t, ctx.send_idx_lens)) if ctx.send_idx_lens else []
    )
    send_sizes = [
        [ctx.send_sizes_flat[i * ws + j] for j in range(ws)] for i in range(ws)
    ]
    n_owned = ctx.n_owned
    grad_owned_direct = grad_padded[:n_owned]
    halo = grad_padded[n_owned:].contiguous()
    # reverse all_to_all of the halo grads back to owners, then index_add.
    rev_indices: list[torch.Tensor] = []
    off = 0
    for r in range(ws):
        n = int(send_sizes[r][ctx.rank])
        rev_indices.append(
            torch.arange(off, off + n, device=grad_padded.device, dtype=torch.int64)
        )
        off += n
    rev_sizes = [[int(send_sizes[j][i]) for j in range(ws)] for i in range(ws)]
    received_back = _halo_a2a_v_default_group(
        halo, rev_indices, rev_sizes, ctx.rank, ws
    )
    _acc_dt = (
        torch.float64
        if grad_owned_direct.dtype == torch.float32
        else grad_owned_direct.dtype
    )
    grad_owned = grad_owned_direct.to(_acc_dt)
    off = 0
    for j in range(ws):
        n = int(send_sizes[ctx.rank][j])
        if n == 0:
            continue
        grad_owned = grad_owned.index_add(
            0, send_indices[j], received_back[off : off + n].to(_acc_dt)
        )
        off += n
    return grad_owned.to(grad_owned_direct.dtype), None, None, None, None, None, None


halo_forward_op.register_autograd(
    _halo_forward_backward, setup_context=_halo_forward_setup_context
)


def halo_forward_compiled(
    owned: torch.Tensor,
    meta: "ParticleHaloMetadata",
    config: "ParticleHaloConfig",
) -> torch.Tensor:
    """Compile-friendly :func:`halo_forward_exchange`: marshals markers into the
    :func:`halo_forward_op` custom op (marker indices + sizes as int[])."""
    markers = _require_markers(meta)
    world_size = len(meta.send_sizes)
    # Marker indices ride as int[] constants precomputed eagerly on the markers
    # (GNNHaloMarkers.__post_init__) -- no tensor read under fake mode, no
    # real-Tensor graph constant for inductor to choke on.
    send_idx_flat = list(markers.send_idx_flat)
    send_idx_lens = list(markers.send_idx_lens)
    send_sizes_flat = [
        int(meta.send_sizes[i][j]) for i in range(world_size) for j in range(world_size)
    ]
    return halo_forward_op(
        owned,
        send_idx_flat,
        send_idx_lens,
        send_sizes_flat,
        int(meta.n_padded),
        int(config.rank),
        world_size,
    )


def halo_scatter_correct_compiled(
    padded: torch.Tensor,
    meta: "ParticleHaloMetadata",
    config: "ParticleHaloConfig",
) -> torch.Tensor:
    """Compile-friendly halo correction: marshals the marker metadata into the
    :func:`halo_scatter_correct_op` custom-op arg form (marker indices + sizes
    as int[]) and invokes it. Numerically equals
    ``halo_forward_exchange(halo_reverse_exchange(padded, ...), ...)``."""
    markers = _require_markers(meta)
    world_size = len(meta.send_sizes)
    # Marker indices ride as int[] constants precomputed eagerly on the markers
    # (GNNHaloMarkers.__post_init__) -- no tensor read under fake, no real-Tensor
    # graph constant for inductor.
    send_idx_flat = list(markers.send_idx_flat)
    send_idx_lens = list(markers.send_idx_lens)
    send_sizes_flat = [
        int(meta.send_sizes[i][j]) for i in range(world_size) for j in range(world_size)
    ]
    return halo_scatter_correct_op(
        padded,
        send_idx_flat,
        send_idx_lens,
        send_sizes_flat,
        int(meta.n_owned),
        int(config.rank),
        world_size,
    )


# ======================================================================
# Fixed-shape (compile-static) halo ops.
#
# Uniform-split all_to_all + tensor routing metadata (send_index / recv_dest /
# recv_real / n_owned), so the per-step routing rides as runtime tensor graph
# inputs (via ShardTensor._halo_meta_packed) rather than baked list[int]
# constants — the latter go stale under torch.compile because
# ShardTensor.__metadata_guard__ guards only on (spec, requires_grad).
# Layout: [ owned(n_owned) | ghost(source-rank order) | PAD -> N_pad ]; row
# N_pad-1 is the DEAD row (padding recv slots land there and are discarded).
# ======================================================================


def _row_mask_like(mask_1d: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Reshape a ``(R,)`` per-row mask to broadcast over ``ref``'s trailing dims."""
    return mask_1d.reshape((mask_1d.shape[0],) + (1,) * (ref.ndim - 1))


def build_halo_meta_tensors(
    meta: "ParticleHaloMetadata",
    rank: int,
    max_send: int,
    n_pad: int,
    device: "torch.device",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the fixed-shape halo routing tensors from halo metadata.

    Ghost rows are placed contiguously in source-rank order starting at
    ``n_owned`` (matching ``particle_halo_padding``'s ``[owned | ghost]``
    layout); ``recv_dest`` padding slots and sentinel sends point at the dead
    row ``n_pad - 1`` and owned row 0 respectively.

    Parameters
    ----------
    meta : ParticleHaloMetadata
        Halo metadata; must carry ``gnn_markers``.
    rank : int
        Local rank id.
    max_send : int
        Per-peer send/recv capacity. Each peer slice has this fixed width.
    n_pad : int
        Total padded row count; the last row is the dead row.
    device : torch.device
        Device for the returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(send_index, recv_dest, recv_real, n_owned_t)`` where ``send_index``
        and ``recv_dest`` are int64 ``[world_size * max_send]``, ``recv_real``
        is bool ``[world_size * max_send]``, and ``n_owned_t`` is a 0-dim int64
        scalar.

    Raises
    ------
    ValueError
        If any per-peer count exceeds ``max_send`` or the ghost region exceeds
        ``n_pad`` (the caller grows the cap and retries).
    """
    markers = _require_markers(meta)
    world_size = len(meta.send_sizes)
    sizes = meta.send_sizes
    wm = world_size * max_send
    send_index = torch.zeros(wm, dtype=torch.int64, device=device)
    recv_dest = torch.full((wm,), n_pad - 1, dtype=torch.int64, device=device)
    recv_real = torch.zeros(wm, dtype=torch.bool, device=device)

    for j in range(world_size):
        cnt = int(sizes[rank][j])
        if cnt > max_send:
            raise ValueError(
                f"halo send count {cnt} (rank {rank}->{j}) exceeds max_send={max_send}"
            )
        if cnt:
            send_index[j * max_send : j * max_send + cnt] = markers.send_indices_owned[
                j
            ].to(device=device, dtype=torch.int64)

    off = meta.n_owned
    for i in range(world_size):
        cnt = int(sizes[i][rank])
        if cnt > max_send:
            raise ValueError(
                f"halo recv count {cnt} (rank {i}->{rank}) exceeds max_send={max_send}"
            )
        if cnt:
            recv_dest[i * max_send : i * max_send + cnt] = torch.arange(
                off, off + cnt, device=device, dtype=torch.int64
            )
            recv_real[i * max_send : i * max_send + cnt] = True
            off += cnt
    if off > n_pad:
        raise ValueError(f"halo ghost region {off} exceeds n_pad={n_pad}")

    n_owned_t = torch.tensor(meta.n_owned, dtype=torch.int64, device=device)
    return send_index, recv_dest, recv_real, n_owned_t


def pack_halo_meta(
    send_index: torch.Tensor,
    recv_dest: torch.Tensor,
    recv_real: torch.Tensor,
    n_owned_t: torch.Tensor,
) -> torch.Tensor:
    """Pack the four routing tensors into one int64 ``[3*W*M + 1]`` buffer so the
    ShardTensor carries a single extra inner tensor (``_halo_meta_packed``)."""
    return torch.cat(
        [
            send_index.to(torch.int64),
            recv_dest.to(torch.int64),
            recv_real.to(torch.int64),
            n_owned_t.reshape(1).to(torch.int64),
        ]
    )


def unpack_halo_meta(
    packed: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`pack_halo_meta`. The packed length is a static shape
    under compile, so ``wm`` is a graph constant and the slices are static."""
    wm = (packed.shape[0] - 1) // 3
    send_index = packed[:wm]
    recv_dest = packed[wm : 2 * wm]
    recv_real = packed[2 * wm : 3 * wm].to(torch.bool)
    n_owned_t = packed[3 * wm]
    return send_index, recv_dest, recv_real, n_owned_t


@torch.library.custom_op("nvalchemi::halo_forward_static", mutates_args=())
def halo_forward_static_op(
    padded_in: torch.Tensor,
    send_index: torch.Tensor,
    recv_dest: torch.Tensor,
    recv_real: torch.Tensor,
    n_owned: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Fixed-shape owned->[owned|ghost] refresh. Runs eagerly at runtime
    (default group); the trace sees only the fake. Owned rows pass through;
    ghost rows are gathered from neighbors via a uniform-split all_to_all."""
    send_rows = padded_in.index_select(0, send_index)
    recv = halo_exchange_fixed(send_rows, world_size, None)
    recv = recv * _row_mask_like(recv_real, recv).to(recv.dtype)
    ghost_acc = torch.zeros_like(padded_in).index_add(0, recv_dest, recv)
    rowidx = torch.arange(padded_in.shape[0], device=padded_in.device)
    ghostmask = _row_mask_like(rowidx >= n_owned, padded_in)
    return torch.where(ghostmask, ghost_acc, padded_in)


@halo_forward_static_op.register_fake
def _halo_forward_static_fake(
    padded_in, send_index, recv_dest, recv_real, n_owned, world_size
):
    return torch.empty_like(padded_in)


def _hfs_setup(ctx, inputs, output):  # type: ignore[no-untyped-def]
    _padded_in, send_index, recv_dest, recv_real, n_owned, world_size = inputs
    ctx.save_for_backward(send_index, recv_dest, recv_real, n_owned)
    ctx.world_size = world_size


def _hfs_backward(ctx, grad_out):  # type: ignore[no-untyped-def]
    send_index, recv_dest, recv_real, n_owned = ctx.saved_tensors
    ws = ctx.world_size
    rowidx = torch.arange(grad_out.shape[0], device=grad_out.device)
    ghostmask = _row_mask_like(rowidx >= n_owned, grad_out)
    grad_ghost = grad_out * ghostmask.to(grad_out.dtype)
    grad_recv = grad_ghost.index_select(0, recv_dest) * _row_mask_like(
        recv_real, grad_out
    ).to(grad_out.dtype)
    grad_send = halo_exchange_fixed(grad_recv, ws, None)
    grad_in = grad_out * (~ghostmask).to(grad_out.dtype)
    grad_in = grad_in.index_add(0, send_index, grad_send)
    return grad_in, None, None, None, None, None


halo_forward_static_op.register_autograd(_hfs_backward, setup_context=_hfs_setup)


@torch.library.custom_op("nvalchemi::halo_scatter_correct_static", mutates_args=())
def halo_scatter_correct_static_op(
    padded_in: torch.Tensor,
    send_index: torch.Tensor,
    recv_dest: torch.Tensor,
    recv_real: torch.Tensor,
    n_owned: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    """Fixed-shape ``halo_forward_exchange(halo_reverse_exchange(.))``: fold ghost
    rows back into their owners (reverse all_to_all + index_add), then
    re-broadcast the corrected owners to the ghost region. fp64 accumulation when
    inputs are fp32 (a boundary owner folds many cross-rank contributions;
    atomic-add order is GPU-nondeterministic), matching ``_halo_scatter_correct_dense``."""
    rowidx = torch.arange(padded_in.shape[0], device=padded_in.device)
    ghostmask = _row_mask_like(rowidx >= n_owned, padded_in)
    recv_real_f = _row_mask_like(recv_real, padded_in).to(padded_in.dtype)
    acc_dt = torch.float64 if padded_in.dtype == torch.float32 else padded_in.dtype

    # reverse: ghost rows (recv-slot order) -> owning rank -> index_add into owners.
    ghost_rows = padded_in.index_select(0, recv_dest) * recv_real_f
    back = halo_exchange_fixed(ghost_rows, world_size, None)
    owned_only = (padded_in * (~ghostmask).to(padded_in.dtype)).to(acc_dt)
    owned_acc = owned_only.index_add(0, send_index, back.to(acc_dt)).to(padded_in.dtype)

    # forward: re-broadcast corrected owners to ghosts.
    send_rows = owned_acc.index_select(0, send_index)
    recv = halo_exchange_fixed(send_rows, world_size, None) * recv_real_f
    ghost_acc = torch.zeros_like(padded_in).index_add(0, recv_dest, recv)
    return torch.where(ghostmask, ghost_acc, owned_acc)


@halo_scatter_correct_static_op.register_fake
def _halo_scatter_correct_static_fake(
    padded_in, send_index, recv_dest, recv_real, n_owned, world_size
):
    return torch.empty_like(padded_in)


def _hscs_setup(ctx, inputs, output):  # type: ignore[no-untyped-def]
    _padded_in, send_index, recv_dest, recv_real, n_owned, world_size = inputs
    ctx.save_for_backward(send_index, recv_dest, recv_real, n_owned)
    ctx.world_size = world_size


def _hscs_backward(ctx, grad_out):  # type: ignore[no-untyped-def]
    # forward(reverse(.)) is self-adjoint -> the VJP is the op applied to grad.
    send_index, recv_dest, recv_real, n_owned = ctx.saved_tensors
    grad_in = halo_scatter_correct_static_op(
        grad_out.contiguous(), send_index, recv_dest, recv_real, n_owned, ctx.world_size
    )
    return grad_in, None, None, None, None, None


halo_scatter_correct_static_op.register_autograd(
    _hscs_backward, setup_context=_hscs_setup
)


def halo_forward_static_from_meta(
    padded_in: torch.Tensor,
    meta: "ParticleHaloMetadata",
    rank: int,
    max_send: int,
) -> torch.Tensor:
    """Convenience: build routing tensors from ``meta`` and call the static
    forward op. (The compiled dispatch path instead unpacks the ShardTensor's
    ``_halo_meta_packed`` so the metadata rides as a graph input.)"""
    si, rd, rr, no = build_halo_meta_tensors(
        meta, rank, max_send, padded_in.shape[0], padded_in.device
    )
    return halo_forward_static_op(padded_in, si, rd, rr, no, len(meta.send_sizes))


def halo_scatter_correct_static_from_meta(
    padded_in: torch.Tensor,
    meta: "ParticleHaloMetadata",
    rank: int,
    max_send: int,
) -> torch.Tensor:
    """Convenience counterpart of :func:`halo_forward_static_from_meta`."""
    si, rd, rr, no = build_halo_meta_tensors(
        meta, rank, max_send, padded_in.shape[0], padded_in.device
    )
    return halo_scatter_correct_static_op(
        padded_in, si, rd, rr, no, len(meta.send_sizes)
    )


def halo_reverse_exchange(
    padded: torch.Tensor,
    meta: ParticleHaloMetadata,
    config: ParticleHaloConfig,
) -> torch.Tensor:
    """Accumulate halo-row contributions back into owners.

    Autograd-aware inverse-in-role of :func:`halo_forward_exchange`. Use after
    a per-layer ``scatter_add_`` to fold the partial contributions written into
    halo rows back into the owning ranks' owned rows. The backward pass
    forward-exchanges owned-row gradients into halo positions.

    Parameters
    ----------
    padded : torch.Tensor
        ``(N_padded, *F)`` tensor ``[owned | halo_partials]``.
    meta : ParticleHaloMetadata
        Metadata from :func:`particle_halo_padding` (must have ``gnn_markers``).
    config : ParticleHaloConfig
        The halo config used to produce ``meta``.

    Returns
    -------
    torch.Tensor
        ``(N_owned, *F)`` owned tensor with accumulated halo contributions.
    """
    return _HaloReverseExchange.apply(padded, meta, config)
