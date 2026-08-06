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

"""Chemistry orchestration over the generic particle-halo primitives.

:func:`halo_exchange` materializes a :class:`ShardedBatch`'s local working
view — owned + halo atoms — as a standard nvalchemi
:class:`~nvalchemi.data.batch.Batch`. It is the MLIP-layer glue on top of the
domain-neutral halo primitives in
:mod:`nvalchemi.distributed._core.particle_halo` (``particle_halo_padding``,
``particle_halo_padding_autograd``, ``pad_field``), which know nothing about
``Batch`` / ``AtomicData`` / ``ShardedBatch``.

Keeping this glue out of ``_core`` is what lets the core halo layer stay an
upstream-candidate for physicsnemo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.distributed._core.particle_halo import (
    halo_forward_exchange,
    pad_field,
    particle_halo_padding,
    particle_halo_padding_autograd,
)

if TYPE_CHECKING:
    from nvalchemi.distributed._core.halo_types import ParticleHaloConfig

__all__ = ["halo_exchange"]


def _apply_shared_strain(sharded: Any) -> None:
    """Apply ``sharded.grad_strain`` to the padded view, if one is pinned.

    The strain must land on the padded geometry: a ghost row holds an absolute
    position, so straining before the exchange leaves its periodic offset
    unscaled. The cell is taken from the owned batch because the padded one
    survives a refresh and would compound the strain each time.
    """
    strain = getattr(sharded, "grad_strain", None)
    if strain is None:
        return
    from nvalchemi.data.batch import set_transient  # noqa: PLC0415
    from nvalchemi.models._utils import apply_strain  # noqa: PLC0415

    strain_pos, strain_cell = strain
    padded = sharded.padded_batch
    atoms = padded._atoms_group
    bidx = padded.batch_idx
    bidx = (bidx.to_local() if hasattr(bidx, "to_local") else bidx).long()
    cell = sharded.cell
    cell = cell.to_local() if hasattr(cell, "to_local") else cell

    scaled_pos, scaled_cell = apply_strain(
        atoms["positions"], cell, bidx, strain_pos, strain_cell
    )
    atoms["positions"] = scaled_pos
    set_transient(padded, "cell", scaled_cell)


def halo_exchange(
    sharded: Any,
    config: ParticleHaloConfig,
    compute_forces: bool = False,
) -> None:
    """Populate / refresh ``sharded.padded_batch`` with owned + halo atoms.

    The distributed-system analog of "materialize the local working view
    for the model call." Halo-pads per-atom fields on ``sharded``
    (positions via the autograd-aware primitive when
    ``compute_forces=True``, others via plain gather) and attaches a
    standard :class:`~nvalchemi.data.batch.Batch` at
    ``sharded.padded_batch`` plus the routing metadata at
    ``sharded.halo_meta``.

    **Idempotent in-place update.** If ``sharded.padded_batch`` already
    exists with a compatible shape, fields are updated *in place* on the
    existing Batch object — any attributes attached by downstream
    callers (notably the neighbor list written by
    ``NeighborListHook`` / ``compute_neighbors``) survive. That's the
    single-system analogy: ``compute_neighbors(batch, cfg)`` stashes NL
    onto a ``batch``, and updating ``batch.positions`` later leaves the
    NL in place until an explicit rebuild. ``NeighborListHook``'s skin
    check decides when to rebuild.

    If shapes change (atom migration has altered ``n_owned`` or the
    halo routing), the Batch is rebuilt from scratch and any cached NL
    is lost — which is correct, because migration invalidates NL.

    Parameters
    ----------
    sharded
        :class:`~nvalchemi.distributed.sharded_batch.ShardedBatch`. Per-atom
        ShardTensors are read via ``.to_local()`` to extract this rank's
        owned rows; halo rows are gathered from peer ranks.
    config
        Shared :class:`ParticleHaloConfig` (ghost width, partitioner, mesh).
    compute_forces
        If True, build the padded positions with
        :func:`particle_halo_padding_autograd` so ``autograd.grad`` can
        flow back through the halo exchange. If False, build plain
        no-grad positions.
    """
    from nvalchemi.data.atomic_data import AtomicData
    from nvalchemi.data.batch import Batch as BatchCls

    # A caller that builds several halos over one owned set (the wired pipeline
    # group) pins its autograd leaves here so every halo shares one graph;
    # otherwise each exchange mints a fresh leaf and the caller's differentiation
    # target is orphaned the moment the model refreshes its own halo.
    pinned: dict[str, torch.Tensor] = (
        (getattr(sharded, "grad_fields", None) or {}) if compute_forces else {}
    )
    local_pos = pinned.get("positions")
    if local_pos is None:
        local_pos = sharded.positions.to_local()
        if compute_forces and not local_pos.requires_grad:
            local_pos = local_pos.clone().requires_grad_(True)

    if compute_forces:
        padded_pos, meta = particle_halo_padding_autograd(local_pos, config)
    else:
        with torch.no_grad():
            padded_pos, meta = particle_halo_padding(local_pos, config)

    device = padded_pos.device
    n_padded = padded_pos.shape[0]
    cell = sharded.cell
    pbc = sharded.pbc

    # Every per-atom field scattered onto the ShardedBatch rides through.
    # positions has already been padded (autograd-aware); forces is
    # reset to zero because the model writes into it in place; every
    # other field — atomic_numbers, atomic_masses, velocities, charges,
    # momenta, anything the producer attached via add_node_property —
    # halo-gathers through ``pad_field``.
    atom_fields = sharded.atom_fields()

    def _build_padded_field(name: str, shard: Any) -> torch.Tensor:
        if name == "positions":
            return padded_pos
        local = shard.to_local() if hasattr(shard, "to_local") else shard
        if name == "forces":
            return torch.zeros(
                (n_padded,) + tuple(local.shape[1:]),
                dtype=local.dtype,
                device=device,
            )
        leaf = pinned.get(name)
        if leaf is not None:
            # Grad-bearing field: gather it through the autograd-aware
            # primitive, so a ghost row's gradient reaches the rank that owns
            # the atom. ``pad_field``'s plain collective would drop it.
            return halo_forward_exchange(leaf, meta, config)
        return pad_field(shard, meta, config)

    # Try in-place update on existing padded_batch if shape-compatible.
    # Write per-atom fields directly to ``_atoms_group`` — dict-style
    # assignment uniformly handles every field and crucially leaves
    # attrs attached by downstream callers (``NeighborListHook``'s
    # ``neighbor_matrix`` / ``num_neighbors`` / ``neighbor_list`` etc.)
    # untouched — that's the whole point of the in-place path.
    existing = sharded.padded_batch
    if (
        existing is not None
        and sharded.halo_meta is not None
        and sharded.halo_meta.n_owned == meta.n_owned
        and sharded.halo_meta.n_padded == meta.n_padded
    ):
        system = existing._system_group
        with torch.no_grad():
            system["cell"].copy_(cell)
            system["pbc"].copy_(pbc)

        atoms = existing._atoms_group
        for name, shard in atom_fields.items():
            if name == "forces" and "forces" in atoms:
                # Zero in place so the pre-allocated buffer that kernels
                # may write into keeps its identity.
                atoms["forces"].zero_()
                continue
            atoms[name] = _build_padded_field(name, shard)
        for name, tensor in getattr(sharded, "system_fields", {}).items():
            system[name] = tensor
        sharded.halo_meta = meta
        _apply_shared_strain(sharded)
        return

    # Fresh build — AtomicData's ctor only accepts its declared fields;
    # positions / atomic_numbers / atomic_masses go there, everything
    # else rides through ``add_node_property``.
    padded_kwargs: dict[str, torch.Tensor] = {}
    for required in ("positions", "atomic_numbers", "atomic_masses"):
        if required in atom_fields:
            padded_kwargs[required] = _build_padded_field(
                required, atom_fields[required]
            )
    padded_kwargs["cell"] = cell
    padded_kwargs["pbc"] = pbc

    # Replicated per-system inputs (total ``charge``, ``spin``, custom system
    # properties) ride through too. A wrapper reads them to decide WHICH
    # physical system it is computing, and a model that finds them missing
    # falls back to a neutral default rather than failing — so dropping them
    # here silently changes the physics.
    known_fields = set(AtomicData.model_fields)
    system_extras: dict[str, torch.Tensor] = {}
    for name, tensor in getattr(sharded, "system_fields", {}).items():
        if name in known_fields:
            padded_kwargs[name] = tensor
        else:
            system_extras[name] = tensor

    padded_data = AtomicData(**padded_kwargs)
    for name, tensor in system_extras.items():
        padded_data.add_system_property(name, tensor)
    for name, shard in atom_fields.items():
        if name in ("positions", "atomic_numbers", "atomic_masses"):
            continue
        padded_data.add_node_property(name, _build_padded_field(name, shard))

    sharded.padded_batch = BatchCls.from_data_list([padded_data], device=device)
    sharded.halo_meta = meta
    _apply_shared_strain(sharded)
