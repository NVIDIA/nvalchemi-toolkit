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

"""ShardedCollection: a set of named tensors distributed over a device mesh,
where **each field declares how it is distributed** via a
:class:`~nvalchemi.distributed._core.storage_policy.StoragePolicy`.

This is the domain-agnostic, multi-field counterpart to a single ShardTensor.
Distribution is *declared* (a field->policy map), not *inferred* from what a
field means — so the scatter/gather bookkeeping lives here once, and
domain-specific containers (e.g. an atomic-data batch) subclass it and supply
only their field->policy map + any extra padding logic.

The container carries no chemistry: it knows ``mesh``, ``fields``, and
``policies`` and nothing about atoms, energies, or forces.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from nvalchemi.distributed._core.storage_policy import StoragePolicy

__all__ = ["ShardedCollection"]


def _world_size(mesh: Any) -> int:
    if hasattr(mesh, "size"):
        try:
            return int(mesh.size(0))
        except TypeError:
            return int(mesh.size())
    return dist.get_world_size()


def _mesh_group(mesh: Any) -> Any:
    """The mesh's process group, or ``None`` if it isn't group-capable.

    All scatter broadcasts run on *this* group (the domain sub-mesh's group when
    the mesh is a sliced sub-mesh), not the world group — so a scatter confined
    to one sub-mesh of a larger (e.g. pipeline × domain) mesh doesn't stall the
    ranks outside it.
    """
    get_group = getattr(mesh, "get_group", None)
    return get_group() if get_group is not None else None


def _global_src(group: Any, src: int) -> int:
    """Map a **group-local** ``src`` rank to the global rank ``dist`` collectives
    expect for their ``src=`` argument (global regardless of ``group``).

    Identity when there's no group or distribution isn't initialized (the 1-D
    whole-mesh case, where group-local == global).
    """
    if group is None or not dist.is_initialized():
        return src
    return dist.get_global_rank(group, src)


def _broadcast_object(obj: Any, src: int, group: Any = None) -> Any:
    if not dist.is_initialized():
        return obj
    holder = [obj]
    dist.broadcast_object_list(holder, src=_global_src(group, src), group=group)
    return holder[0]


def _broadcast_sizes(
    sizes: list[int] | None,
    *,
    world_size: int,
    device: torch.device,
    src: int,
    local_rank: int,
    group: Any = None,
) -> list[int]:
    src_sizes = sizes if (local_rank == src and sizes is not None) else [0] * world_size
    sizes_t = torch.tensor(src_sizes, dtype=torch.int64, device=device)
    if dist.is_initialized():
        dist.broadcast(sizes_t, src=_global_src(group, src), group=group)
    return [int(x) for x in sizes_t.tolist()]


def _broadcast_full(
    src_tensor: torch.Tensor | None,
    *,
    n_global: int,
    dtype: torch.dtype,
    trailing: tuple[int, ...],
    device: torch.device,
    src: int,
    local_rank: int,
    group: Any = None,
) -> torch.Tensor:
    """Broadcast a full ``(n_global, *trailing)`` tensor from *src* to all ranks.

    Broadcast is collective on every backend (NCCL, gloo) and needs no per-rank
    P2P; for a one-time setup scatter the ``n_global x world_size`` bandwidth vs
    a perfect scatter is negligible. Each rank then slices its own rows per its
    field policy.
    """
    full_shape = (n_global,) + trailing
    if local_rank == src:
        if src_tensor is None:
            raise ValueError("source rank must provide src_tensor to broadcast")
        full_t = src_tensor.to(dtype=dtype).contiguous()
    else:
        full_t = torch.empty(full_shape, dtype=dtype, device=device)
    if dist.is_initialized():
        dist.broadcast(full_t, src=_global_src(group, src), group=group)
    return full_t


class ShardedCollection:
    """A set of named tensors distributed over a 1-D ``DeviceMesh`` by explicit
    per-field policy.

    Attributes
    ----------
    mesh
        The device mesh the fields are distributed over.
    fields
        ``name -> stored field`` (a ShardTensor for sharded/halo policies, a
        plain tensor for replicated).
    policies
        ``name -> StoragePolicy`` — the declared distribution for each field.
    """

    def __init__(
        self,
        mesh: Any,
        fields: dict[str, Any],
        policies: dict[str, StoragePolicy],
    ) -> None:
        self.mesh = mesh
        self.fields = fields
        self.policies = policies

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def scatter(
        cls,
        source: dict[str, torch.Tensor] | None,
        *,
        mesh: Any,
        policies: dict[str, StoragePolicy],
        sizes: list[int] | None,
        device: torch.device,
        src: int = 0,
    ) -> "ShardedCollection":
        """Distribute each field from *src* across *mesh*, honoring its policy.

        Parameters
        ----------
        source
            ``name -> full tensor`` on *src* (rows already ordered to match the
            ``sizes`` partition); ``None`` on non-source ranks.
        policies
            ``name -> StoragePolicy``. Sharded policies slice the broadcast
            tensor by ``sizes``; replicated policies keep the full tensor.
        sizes
            Per-rank owned-row counts (length ``world_size``, sum ==
            ``n_global``). Valid on *src*; broadcast to every rank here.
        device
            Device to allocate received tensors on.
        src
            The rank holding ``source``.
        """
        local_rank = mesh.get_local_rank()
        world_size = _world_size(mesh)
        # Every broadcast below runs on the mesh's own group with ``src`` mapped
        # group-local -> global, so a scatter over a sliced sub-mesh doesn't
        # broadcast on (and stall) the world group.
        group = _mesh_group(mesh)

        sizes = _broadcast_sizes(
            sizes,
            world_size=world_size,
            device=device,
            src=src,
            local_rank=local_rank,
            group=group,
        )
        n_global = int(sum(sizes))

        names = list(policies.keys())
        schema: list[dict[str, Any]] | None = None
        if local_rank == src:
            if source is None:
                raise ValueError("source rank must provide the source field dict")
            schema = [
                {
                    "name": n,
                    "dtype": source[n].dtype,
                    "trailing": tuple(source[n].shape[1:]),
                }
                for n in names
            ]
        schema = _broadcast_object(schema, src, group)
        if schema is None:
            raise RuntimeError("schema broadcast returned None on a non-source rank")

        fields: dict[str, Any] = {}
        for entry in schema:
            name = entry["name"]
            full_t = _broadcast_full(
                source.get(name) if source is not None else None,
                n_global=n_global,
                dtype=entry["dtype"],
                trailing=tuple(entry["trailing"]),
                device=device,
                src=src,
                local_rank=local_rank,
                group=group,
            )
            fields[name] = policies[name].place_from_full(
                full_t, mesh=mesh, sizes=sizes, local_rank=local_rank
            )
        return cls(mesh, fields, policies)

    @classmethod
    def from_local(
        cls,
        local_fields: dict[str, torch.Tensor],
        *,
        mesh: Any,
        policies: dict[str, StoragePolicy],
        sizes: list[int] | None = None,
    ) -> "ShardedCollection":
        """Build a collection where each rank already holds its own rows.

        ``sizes`` (per-rank owned-row counts) is all-gathered from the local row
        count of the first sharded field when not supplied. Replicated fields'
        local rows are the full tensor.
        """
        if sizes is None:
            sizes = cls._all_gather_sizes(local_fields, mesh, policies)
        fields = {
            name: policies[name].place_from_local(t, mesh=mesh, sizes=sizes)
            for name, t in local_fields.items()
        }
        return cls(mesh, fields, policies)

    @staticmethod
    def _all_gather_sizes(
        local_fields: dict[str, torch.Tensor],
        mesh: Any,
        policies: dict[str, StoragePolicy],
    ) -> list[int]:
        world_size = _world_size(mesh)
        # Use the first sharded field's local row count; fall back to any field.
        sample = next(iter(local_fields.values()))
        my_n = int(sample.shape[0])
        if not dist.is_initialized() or world_size == 1:
            return [my_n]
        device = sample.device
        my_t = torch.tensor([my_n], dtype=torch.int64, device=device)
        out = torch.empty(world_size, dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(out, my_t, group=mesh.get_group())
        return [int(x) for x in out.tolist()]

    # ------------------------------------------------------------------
    # Views & gathering
    # ------------------------------------------------------------------

    def local(self) -> dict[str, torch.Tensor]:
        """Local-rank view of each field (per its policy's ``to_local``)."""
        return {
            name: self.policies[name].to_local(stored)
            for name, stored in self.fields.items()
        }

    def gather(self, *, dst: int | None = 0) -> dict[str, torch.Tensor] | None:
        """Reconstruct full tensors. With ``dst=int`` returns the dict on *dst*
        (``None`` elsewhere); with ``dst=None`` returns it on every rank.

        All ranks must call this — the underlying transport is collective.
        """
        local_rank = self.mesh.get_local_rank()
        if dst is None:
            world_size = _world_size(self.mesh)
            out: dict[str, torch.Tensor] = {}
            for name, stored in self.fields.items():
                policy = self.policies[name]
                parts = [
                    policy.full_tensor(stored, mesh=self.mesh, dst=r)
                    for r in range(world_size)
                ]
                out[name] = parts[local_rank]
            return out

        gathered = {
            name: self.policies[name].full_tensor(stored, mesh=self.mesh, dst=dst)
            for name, stored in self.fields.items()
        }
        if local_rank != dst:
            return None
        return gathered

    def redistribute(
        self, policies: dict[str, StoragePolicy], *, src: int = 0
    ) -> "ShardedCollection":
        """Return a new collection with each field re-placed under *policies*.

        Implemented as gather-to-all then re-scatter; only fields named in
        *policies* change policy (others keep their current one). Generic and
        unused on the hot path — present for completeness of the protocol.
        """
        new_policies = {**self.policies, **policies}
        full = self.gather(dst=None)
        if full is None:  # dst=None populates every rank
            raise RuntimeError("gather(dst=None) must populate every rank")
        device = next(iter(full.values())).device
        sizes = self._all_gather_sizes(self.local(), self.mesh, new_policies)
        return type(self).scatter(
            full,
            mesh=self.mesh,
            policies=new_policies,
            sizes=sizes,
            device=device,
            src=src,
        )
