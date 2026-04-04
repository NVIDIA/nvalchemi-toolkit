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
"""Graph-aware Pydantic batch backed by :class:`MultiLevelStorage`.

This module provides a :class:`Batch` class that combines the Pydantic-model
interface of ``nvalchemi.data.batch.Batch`` with the performant tensor storage
of :class:`~nvalchemi.data.level_storage.MultiLevelStorage`.

Performance advantages over the Pydantic-based ``nvalchemi.data.batch.Batch``:

* **index_select** operates directly on concatenated tensors via segment
  selection -- no per-graph object reconstruction and re-batching.
* **to / clone** move / copy tensors in a single pass -- no
  ``model_dump`` / ``map_structure`` / ``model_validate`` round-trip.
* **batch / ptr** are lazily derived from ``segment_lengths`` -- never
  eagerly built or manually maintained.
* **No slices / cumsum bookkeeping** -- edge-index offsets are recovered
  from ``atoms.batch_ptr`` at unbatching time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from torch import Tensor
from torch import distributed as dist
from torch.distributed import ProcessGroup, Work

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.data import DataMixin
from nvalchemi.data.level_storage import (
    LevelSchema,
    MultiLevelStorage,
    SegmentedLevelStorage,
    UniformLevelStorage,
)

# Edge-level keys whose values are node indices and therefore need
# cumulative node-offset correction when batching (from_data_list) and
# the reverse correction when extracting a single graph (get_data).
# This set does NOT control concatenation dimension or shape semantics;
# all edge tensors are stored as (E, ...) and concatenated on dim 0.
_INDEX_KEYS = frozenset({"edge_index"})
_EXCLUDED_KEYS = frozenset({"batch", "ptr", "device", "dtype", "info"})


_OWN_ATTRS = frozenset({"device", "keys", "_storage", "_data_class"})


class Batch(DataMixin):
    """Graph-aware batch built on :class:`MultiLevelStorage`.

    Internally stores three attribute groups via an :class:`MultiLevelStorage`:

    * ``"atoms"`` (:class:`SegmentedLevelStorage`) -- node-level tensors
    * ``"edges"`` (:class:`SegmentedLevelStorage`) -- edge-level tensors
    * ``"system"`` (:class:`UniformLevelStorage`) -- graph-level tensors

    ``batch``, ``ptr``, ``num_nodes_list``, and ``num_edges_list`` are
    derived lazily from the segmented groups.

    Attributes
    ----------
    device : torch.device
        Device of the underlying storage.
    keys : dict[str, set[str]] | None
        Level categorisation: ``{"node": ..., "edge": ..., "system": ...}``.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        storage: MultiLevelStorage | None = None,
        keys: dict[str, set[str]] | None = None,
    ) -> None:
        object.__setattr__(
            self, "_storage", storage if storage is not None else MultiLevelStorage()
        )
        object.__setattr__(self, "_data_class", AtomicData)
        object.__setattr__(
            self,
            "device",
            torch.device(device) if isinstance(device, str) else device,
        )
        object.__setattr__(self, "keys", keys)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _OWN_ATTRS:
            object.__setattr__(self, name, value)
        elif isinstance(value, torch.Tensor):
            self._storage[name] = value
        else:
            object.__setattr__(self, name, value)

    @classmethod
    def _construct(
        cls,
        *,
        device: torch.device | str,
        keys: dict[str, set[str]] | None,
        storage: MultiLevelStorage,
        data_class: type = AtomicData,
    ) -> Batch:
        """Fast constructor that bypasses __init__."""
        batch = cls.__new__(cls)
        object.__setattr__(batch, "_storage", storage)
        object.__setattr__(batch, "_data_class", data_class)
        object.__setattr__(
            batch,
            "device",
            torch.device(device) if isinstance(device, str) else device,
        )
        object.__setattr__(batch, "keys", keys)
        return batch

    # ------------------------------------------------------------------
    # Properties derived from storage
    # ------------------------------------------------------------------

    @property
    def num_graphs(self) -> int:
        """Number of graphs in the batch."""
        return len(self._storage)

    @property
    def batch_size(self) -> int:
        """Alias for :attr:`num_graphs`."""
        return self.num_graphs

    @property
    def num_nodes(self) -> int:
        """Total number of nodes across all graphs."""
        atoms = self._atoms_group
        return atoms.num_elements() if atoms is not None else 0

    @property
    def num_edges(self) -> int:
        """Total number of edges across all graphs."""
        edges = self._edges_group
        return edges.num_elements() if edges is not None else 0

    @property
    def system_capacity(self) -> int:
        """Maximum number of systems (graphs) this buffer can hold (e.g. from :meth:`empty`)."""
        system = self._system_group
        if system is None:
            return 0
        return system._data.shape[0]

    @property
    def batch(self) -> Tensor:
        """Per-node graph assignment tensor (lazily computed)."""
        atoms = self._atoms_group
        if atoms is None:
            return torch.tensor([], dtype=torch.long, device=self.device)
        return atoms.batch_idx

    @property
    def ptr(self) -> Tensor:
        """Cumulative node count per graph (lazily computed)."""
        atoms = self._atoms_group
        if atoms is None:
            return torch.zeros(1, dtype=torch.int32, device=self.device)
        return atoms.batch_ptr

    @property
    def edge_ptr(self) -> Tensor:
        """Per-atom CSR pointer into the edge list (N+1,), int32.

        Returns a tensor where ``edge_ptr[i] : edge_ptr[i+1]`` is the slice of
        edge rows in ``edge_index`` that belong to atom ``i`` (i.e. where atom
        ``i`` is the sender).  Valid only after a COO-format
        :class:`~nvalchemi.dynamics.hooks.NeighborListHook` has populated the
        edges group.

        An all-zeros pointer of length ``num_nodes + 1`` is returned when the
        edges group is absent or empty.
        """
        edges = self._edges_group
        if edges is None or edges.num_elements() == 0:
            N = self.num_nodes
            return torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        ei = edges["edge_index"]  # (E, 2)
        N = self.num_nodes
        src = ei[:, 0]  # (E,)
        counts = torch.zeros(N, dtype=torch.int32, device=self.device)
        counts.scatter_add_(
            0, src, torch.ones(src.shape[0], dtype=torch.int32, device=self.device)
        )
        ptr = torch.zeros(N + 1, dtype=torch.int32, device=self.device)
        ptr[1:] = counts.cumsum(0)
        return ptr

    @property
    def num_nodes_list(self) -> list[int]:
        """Per-graph node counts as a Python list."""
        atoms = self._atoms_group
        if atoms is None:
            return []
        return atoms.segment_lengths[: len(atoms)].tolist()

    @property
    def num_edges_list(self) -> list[int]:
        """Per-graph edge counts as a Python list."""
        edges = self._edges_group
        if edges is None:
            return []
        return edges.segment_lengths[: len(edges)].tolist()

    @property
    def num_nodes_per_graph(self) -> Tensor:
        """Per-graph node counts as a tensor."""
        atoms = self._atoms_group
        if atoms is None:
            return torch.tensor([], dtype=torch.long, device=self.device)
        return atoms.segment_lengths[: len(atoms)]

    @property
    def num_edges_per_graph(self) -> Tensor:
        """Per-graph edge counts as a tensor."""
        edges = self._edges_group
        if edges is None:
            return torch.tensor([], dtype=torch.long, device=self.device)
        return edges.segment_lengths[: len(edges)]

    @property
    def max_num_nodes(self) -> int:
        """Maximum node count in any graph."""
        nodes = self.num_nodes_list
        return max(nodes) if nodes else 0

    # ------------------------------------------------------------------
    # Internal group accessors
    # ------------------------------------------------------------------

    @property
    def _atoms_group(self) -> SegmentedLevelStorage | None:
        g = self._storage.groups.get("atoms")
        return g if isinstance(g, SegmentedLevelStorage) else None

    @property
    def _edges_group(self) -> SegmentedLevelStorage | None:
        g = self._storage.groups.get("edges")
        return g if isinstance(g, SegmentedLevelStorage) else None

    @property
    def _system_group(self) -> UniformLevelStorage | None:
        return self._storage.groups.get("system")

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_data_list(
        cls,
        data_list: list[AtomicData],
        device: torch.device | str | None = None,
        skip_validation: bool = False,
        attr_map: LevelSchema | None = None,
        exclude_keys: list[str] | None = None,
    ) -> Batch:
        """Construct a batch from a list of :class:`AtomicData` objects.

        Parameters
        ----------
        data_list : list[AtomicData]
            Individual graphs to batch.
        device : torch.device | str, optional
            Target device.  Inferred from *data_list* if ``None``.
        skip_validation : bool
            If ``True``, skip shape validation for speed.
        attr_map : LevelSchema, optional
            Attribute registry.  Defaults to ``LevelSchema()``.
        exclude_keys : list[str], optional
            Keys to exclude from batching.

        Returns
        -------
        Batch
        """
        if not data_list:
            raise ValueError("Cannot create batch from empty data list")

        if device is None:
            device = data_list[0].device
        device = torch.device(device) if isinstance(device, str) else device

        if attr_map is None:
            attr_map = LevelSchema()

        representative = data_list[0]
        data_cls = representative.__class__
        node_keys = representative.__node_keys__
        edge_keys = representative.__edge_keys__
        system_keys = representative.__system_keys__

        excluded = _EXCLUDED_KEYS | set(exclude_keys or [])
        actual_keys = set(data_list[0].model_dump(exclude_none=True).keys()) - excluded

        node_tensors: dict[str, list[Tensor]] = defaultdict(list)
        edge_tensors: dict[str, list[Tensor]] = defaultdict(list)
        system_tensors: dict[str, list[Tensor]] = defaultdict(list)
        node_counts: list[int] = []
        edge_counts: list[int] = []

        node_offset = 0
        for data in data_list:
            n_nodes = data.num_nodes
            n_edges = data.num_edges
            node_counts.append(n_nodes)
            edge_counts.append(n_edges)

            for key in actual_keys:
                value = getattr(data, key, None)
                if not isinstance(value, Tensor):
                    continue
                value = value.to(device)

                if key in node_keys:
                    node_tensors[key].append(value)
                elif key in edge_keys:
                    if key in _INDEX_KEYS:
                        value = value + node_offset
                    edge_tensors[key].append(value)
                elif key in system_keys:
                    system_tensors[key].append(value)

            node_offset += n_nodes

        atoms_data = {k: torch.cat(v, dim=0) for k, v in node_tensors.items()}
        edges_data: dict[str, Tensor] = {}
        for k, v in edge_tensors.items():
            edges_data[k] = torch.cat(v, dim=0)
        system_data = {k: torch.cat(v, dim=0) for k, v in system_tensors.items()}

        validate = not skip_validation
        groups: dict[str, UniformLevelStorage | SegmentedLevelStorage] = {}
        if atoms_data:
            groups["atoms"] = SegmentedLevelStorage(
                data=atoms_data,
                device=device,
                segment_lengths=node_counts,
                validate=validate,
                attr_map=attr_map,
            )
        if edges_data:
            groups["edges"] = SegmentedLevelStorage(
                data=edges_data,
                device=device,
                segment_lengths=edge_counts,
                validate=validate,
                attr_map=attr_map,
            )
        if system_data:
            groups["system"] = UniformLevelStorage(
                data=system_data,
                device=device,
                validate=validate,
                attr_map=attr_map,
            )

        storage = MultiLevelStorage(groups=groups, attr_map=attr_map, validate=validate)

        tracked_keys = {
            "node": set(node_tensors.keys()),
            "edge": set(edge_tensors.keys()),
            "system": set(system_tensors.keys()),
        }
        batch = cls._construct(
            device=device,
            keys=tracked_keys,
            storage=storage,
            data_class=data_cls,
        )
        return batch._make_contiguous()

    @classmethod
    def empty(
        cls,
        *,
        num_systems: int,
        num_nodes: int,
        num_edges: int,
        template: AtomicData | Batch | None = None,
        device: torch.device | str = "cpu",
        attr_map: LevelSchema | None = None,
    ) -> Batch:
        """Construct an empty batch with pre-allocated capacity (zero graphs, fixed storage).

        Storage tensors are allocated with the given capacities; no graphs are
        stored initially (``num_graphs == 0``). Use :meth:`put` to copy graphs
        into the buffer; pass ``dest_mask`` of shape ``(num_systems,)`` with
        ``False`` for empty slots.

        Parameters
        ----------
        num_systems : int
            Maximum number of systems (graphs) the buffer can hold.
        num_nodes : int
            Total node (atom) capacity across all graphs.
        num_edges : int
            Total edge capacity across all graphs.
        template : AtomicData or Batch, optional
            Template for attribute keys and per-key shapes/dtypes. If ``None``,
            a minimal :class:`AtomicData` with ``positions``, ``atomic_numbers``,
            and ``energies`` is used.
        device : torch.device or str, optional
            Device for allocated tensors.
        attr_map : LevelSchema, optional
            Attribute registry; used when template is provided.

        Returns
        -------
        Batch
            Batch with ``num_graphs == 0`` and capacity for the given sizes.
        """
        if num_systems < 0 or num_nodes < 0 or num_edges < 0:
            raise ValueError(
                "num_systems, num_nodes, and num_edges must be non-negative"
            )
        device = torch.device(device) if isinstance(device, str) else device
        if attr_map is None:
            attr_map = LevelSchema()

        if template is None:
            template = AtomicData(
                positions=torch.zeros(1, 3),
                atomic_numbers=torch.zeros(1, dtype=torch.long),
                energies=torch.tensor([[0.0]]),
            )
        if isinstance(template, AtomicData):
            ref = cls.from_data_list([template], device=device, attr_map=attr_map)
        else:
            ref = template

        groups: dict[str, UniformLevelStorage | SegmentedLevelStorage] = {}
        for name, group in ref._storage.groups.items():
            keys = list(group.keys())
            if not keys:
                continue
            if name == "system":
                data = {
                    k: torch.zeros(
                        (num_systems,) + group[k].shape[1:],
                        device=device,
                        dtype=group[k].dtype,
                    )
                    for k in keys
                }
                storage = UniformLevelStorage(
                    data=data, device=device, validate=False, attr_map=attr_map
                )
                object.__setattr__(storage, "_num_kept", 0)
                groups[name] = storage
            elif name == "atoms":
                data = {
                    k: torch.zeros(
                        (num_nodes,) + group[k].shape[1:],
                        device=device,
                        dtype=group[k].dtype,
                    )
                    for k in keys
                }
                groups[name] = SegmentedLevelStorage(
                    data=data,
                    segment_lengths=torch.tensor([], device=device, dtype=torch.int32),
                    device=device,
                    batch_ptr_capacity=max(num_systems + 2, 2),
                    validate=False,
                    attr_map=attr_map,
                )
            else:
                data = {
                    k: torch.zeros(
                        (num_edges,) + group[k].shape[1:],
                        device=device,
                        dtype=group[k].dtype,
                    )
                    for k in keys
                }
                groups[name] = SegmentedLevelStorage(
                    data=data,
                    segment_lengths=torch.tensor([], device=device, dtype=torch.int32),
                    device=device,
                    batch_ptr_capacity=max(num_systems + 2, 2),
                    validate=False,
                    attr_map=attr_map,
                )

        storage = MultiLevelStorage(groups=groups, attr_map=attr_map, validate=False)
        return cls._construct(
            device=device,
            keys=ref.keys,
            storage=storage,
            data_class=ref._data_class,
        )

    def zero(self) -> None:
        """Reset this batch to an empty-but-allocated state.

        Zeros all leaf data tensors while preserving the allocated storage
        capacity.  After calling ``zero()``, ``num_graphs`` returns 0 but
        ``system_capacity`` remains unchanged.

        This method is used to reset pre-allocated communication buffers
        (created via :meth:`empty`) between pipeline steps without
        reallocating memory.

        Notes
        -----
        Modeled after :meth:`GPUBuffer.zero` in ``nvalchemi.dynamics.sinks``.
        Resets bookkeeping for both :class:`UniformLevelStorage` (``_num_kept``)
        and :class:`SegmentedLevelStorage` (``segment_lengths``, ``_batch_ptr``).

        Examples
        --------
        >>> batch = Batch.empty(num_systems=10, num_nodes=100, num_edges=200)
        >>> batch.zero()
        >>> batch.num_graphs
        0
        >>> batch.system_capacity
        10
        """
        for group in self._storage.groups.values():
            group._data.apply_(lambda x: x.zero_())

            if hasattr(group, "_num_kept"):
                object.__setattr__(group, "_num_kept", 0)

            if hasattr(group, "segment_lengths"):
                group.segment_lengths = torch.empty(
                    0,
                    dtype=group.segment_lengths.dtype,
                    device=group.segment_lengths.device,
                )
                if group._batch_ptr is not None:
                    batch_ptr_capacity = group._batch_ptr.shape[0]
                    group._batch_ptr = torch.zeros(
                        batch_ptr_capacity,
                        dtype=torch.int32,
                        device=group.device,
                    )
                if hasattr(group, "_batch_idx"):
                    group._batch_idx = None
                group._batch_ptr_np = None

    # ------------------------------------------------------------------
    # Per-graph reconstruction
    # ------------------------------------------------------------------

    def get_data(self, idx: int) -> AtomicData:
        """Reconstruct the :class:`AtomicData` object at position *idx*.

        Edge-index offsets applied during batching are automatically undone.

        Parameters
        ----------
        idx : int
            Graph index (supports negative indexing).

        Returns
        -------
        AtomicData
        """
        if idx < 0:
            idx = self.num_graphs + idx

        data: dict[str, Any] = {}

        atoms = self._atoms_group
        if atoms is not None:
            atoms._lazy_init_batch_ptr()
            node_start = atoms._batch_ptr[idx].item()
            node_end = atoms._batch_ptr[idx + 1].item()
            for key, tensor in atoms.items():
                data[key] = tensor[node_start:node_end]

        edges = self._edges_group
        if edges is not None and edges.num_elements() > 0:
            edges._lazy_init_batch_ptr()
            edge_start = edges._batch_ptr[idx].item()
            edge_end = edges._batch_ptr[idx + 1].item()
            node_offset = atoms._batch_ptr[idx] if atoms is not None else 0
            for key, tensor in edges.items():
                if key in _INDEX_KEYS:
                    data[key] = tensor[edge_start:edge_end] - node_offset
                else:
                    data[key] = tensor[edge_start:edge_end]

        system = self._system_group
        if system is not None:
            for key, tensor in system.items():
                data[key] = tensor[idx].unsqueeze(0)

        return self._data_class(**data)

    def to_data_list(self) -> list[AtomicData]:
        """Reconstruct all individual :class:`AtomicData` objects.

        Returns
        -------
        list[AtomicData]
        """
        return [self.get_data(i) for i in range(self.num_graphs)]

    # ------------------------------------------------------------------
    # Selection / indexing
    # ------------------------------------------------------------------

    def index_select(
        self,
        idx: int | slice | Tensor | list[int] | np.ndarray | Sequence[int],
    ) -> Batch:
        """Select a subset of graphs by index.

        Operates directly on concatenated tensors via segment selection --
        no per-graph :class:`AtomicData` reconstruction.

        Parameters
        ----------
        idx : int, slice, Tensor, list[int], np.ndarray, or Sequence[int]
            Graph-level index specification.

        Returns
        -------
        Batch
        """
        idx_list = self._normalize_index(idx)
        idx_tensor = torch.tensor(idx_list, dtype=torch.int32, device=self.device)

        new_groups: dict[str, UniformLevelStorage | SegmentedLevelStorage] = {}

        atoms = self._atoms_group
        offset_diff: Tensor | None = None
        if atoms is not None:
            new_atoms = atoms.select(idx_tensor)
            old_offsets = atoms.batch_ptr[idx_tensor]
            new_atoms._lazy_init_batch_ptr()
            new_offsets = new_atoms._batch_ptr[:-1]
            offset_diff = old_offsets - new_offsets
            new_groups["atoms"] = new_atoms

        edges = self._edges_group
        if edges is not None:
            new_edges = edges.select(idx_tensor)
            if "edge_index" in new_edges and offset_diff is not None:
                new_edges._lazy_init_batch_ptr()
                ei = new_edges["edge_index"]
                edge_batch_idx = new_edges.batch_idx
                correction = offset_diff[edge_batch_idx]
                new_edges._data["edge_index"] = ei - correction.unsqueeze(1)
            new_groups["edges"] = new_edges

        system = self._system_group
        if system is not None:
            new_groups["system"] = system.select(idx_tensor)

        new_storage = MultiLevelStorage(
            groups=new_groups,
            attr_map=self._storage.attr_map,
            validate=False,
        )
        return Batch._construct(
            device=self.device,
            keys={k: v.copy() for k, v in self.keys.items()} if self.keys else None,
            storage=new_storage,
            data_class=self._data_class,
        )

    def put(
        self,
        src_batch: Batch,
        mask: Tensor,
        *,
        copied_mask: Tensor | None = None,
        dest_mask: Tensor | None = None,
    ) -> None:
        """Put graphs where mask[i] is True from src_batch into this batch (buffer).

        Computes per-level fit masks (system/atoms/edges), takes their logical_and
        as the copy mask, then puts with that mask so all levels only copy systems
        that fit in every level. Uses Warp buffer kernels; only float32 attributes
        copied. If copied_mask is provided, it is updated with the copy mask for
        :meth:`defrag`.

        Parameters
        ----------
        src_batch : Batch
            Source batch; must have same groups (atoms/edges/system).
        mask : Tensor
            (num_graphs,) bool, True = consider copying this graph.
        copied_mask : Tensor, optional
            (num_graphs,) bool; if provided, modified in place with the actual
            copy mask (fit in all levels). If None, stored on *src_batch*.
        dest_mask : Tensor, optional
            For uniform (system) level: (len(self),) bool, True = slot occupied.
            If None, system level treats all slots as empty.
        """
        device = self.device
        n = src_batch.num_graphs
        if mask.shape[0] != n:
            raise ValueError(f"mask shape {mask.shape[0]} != num_graphs {n}")
        mask = mask.to(device=device, dtype=torch.bool)
        if copied_mask is not None:
            if copied_mask.shape[0] != n:
                raise ValueError(f"copied_mask shape {copied_mask.shape[0]} != {n}")
            copy_mask = copied_mask.to(device=device, dtype=torch.bool)
        else:
            copy_mask = torch.zeros(n, device=device, dtype=torch.bool)
            object.__setattr__(src_batch, "_copied_mask", copy_mask)

        fit_mask = torch.ones(n, device=device, dtype=torch.bool)
        system = self._system_group
        src_system = src_batch._system_group
        if system is not None and src_system is not None:
            level_fit = torch.empty(n, device=device, dtype=torch.bool)
            system.compute_put_per_system_fit_mask(
                src_system, mask, dest_mask, level_fit
            )
            fit_mask.logical_and_(level_fit)
        atoms = self._atoms_group
        src_atoms = src_batch._atoms_group
        if atoms is not None and src_atoms is not None:
            level_fit = torch.empty(n, device=device, dtype=torch.bool)
            atoms.compute_put_per_system_fit_mask(src_atoms, mask, None, level_fit)
            fit_mask.logical_and_(level_fit)
        edges = self._edges_group
        src_edges = src_batch._edges_group
        if edges is not None and src_edges is not None:
            level_fit = torch.empty(n, device=device, dtype=torch.bool)
            edges.compute_put_per_system_fit_mask(src_edges, mask, None, level_fit)
            fit_mask.logical_and_(level_fit)
        copy_mask.copy_(fit_mask)

        if system is not None and src_system is not None:
            system.put(
                src_system, copy_mask, copied_mask=copy_mask, dest_mask=dest_mask
            )
        if atoms is not None and src_atoms is not None:
            atoms.put(src_atoms, copy_mask, copied_mask=copy_mask)
        if edges is not None and src_edges is not None:
            edges.put(src_edges, copy_mask, copied_mask=copy_mask)

    def defrag(
        self,
        copied_mask: Tensor | None = None,
    ) -> Batch:
        """Defrag this batch in-place by removing graphs that were put.

        Drops graphs where copied_mask[i] is True (e.g. from a prior
        :meth:`put`). Uses Warp buffer kernels; one host sync per group to
        trim. Only float32 attributes are compacted.

        Parameters
        ----------
        copied_mask : Tensor, optional
            (num_graphs,) bool; if None, uses stored value from last :meth:`put`.

        Returns
        -------
        Self
            For method chaining.
        """
        if copied_mask is None:
            copied_mask = getattr(self, "_copied_mask", None)
            if copied_mask is None:
                raise ValueError("defrag requires copied_mask or a prior put")
        system = self._system_group
        if system is not None:
            system.defrag(copied_mask=copied_mask)
        atoms = self._atoms_group
        if atoms is not None:
            atoms.defrag(copied_mask=copied_mask)
        edges = self._edges_group
        if edges is not None:
            edges.defrag(copied_mask=copied_mask)
        if hasattr(self, "_copied_mask"):
            object.__delattr__(self, "_copied_mask")
        return self

    def trim(
        self,
        copied_mask: Tensor | None = None,
    ) -> Batch | None:
        """Remove marked graphs and return a new :class:`Batch` with tight storage.

        Unlike :meth:`defrag`, which compacts data to the front of
        pre-allocated buffers while preserving their capacity (ideal for
        fixed-size GPU buffers that will be reused with :meth:`put`),
        ``trim`` produces a brand-new :class:`Batch` whose underlying
        storage tensors are sized to exactly fit the remaining graphs —
        no padding, no unused trailing slots.

        Use :meth:`defrag` when you need to keep the buffer alive for
        further :meth:`put` / :meth:`defrag` cycles (e.g. communication
        buffers).  Use ``trim`` when the batch will be consumed directly
        by a model or integrator and must have self-consistent tensor
        shapes across all storage groups.

        Parameters
        ----------
        copied_mask : Tensor, optional
            ``(num_graphs,)`` boolean tensor where ``True`` marks graphs
            to remove.  If *None*, uses the ``_copied_mask`` stored by
            the most recent :meth:`put`.

        Returns
        -------
        Batch or None
            A new :class:`Batch` containing only the kept graphs with
            all tensors sized to exactly fit, or *None* if every graph
            was removed.

        Raises
        ------
        ValueError
            If no *copied_mask* is provided and no prior :meth:`put`
            has stored one.

        See Also
        --------
        defrag : In-place compaction that preserves buffer capacity.
        """
        if copied_mask is None:
            copied_mask = getattr(self, "_copied_mask", None)
            if copied_mask is None:
                raise ValueError("trim requires copied_mask or a prior put")
        keep_mask = ~copied_mask
        if not keep_mask.any():
            return None
        keep_indices = torch.where(keep_mask)[0]
        return self.index_select(keep_indices)

    def _normalize_index(
        self,
        idx: int | slice | Tensor | list[int] | np.ndarray | Sequence[int],
    ) -> list[int]:
        """Convert various index types to a flat list of integer indices."""
        match idx:
            case int():
                result = [idx]
            case slice():
                result = list(range(self.num_graphs)[idx])
            case Tensor():
                if idx.dtype == torch.bool:
                    result = idx.flatten().nonzero(as_tuple=False).flatten().tolist()
                elif idx.dtype.is_floating_point:
                    raise IndexError(
                        f"Tensor index must be integer or bool, got {idx.dtype}"
                    )
                else:
                    result = idx.flatten().tolist()
            case np.ndarray():
                if idx.dtype == np.bool_:
                    result = idx.flatten().nonzero()[0].flatten().tolist()
                else:
                    result = idx.flatten().tolist()
            case list():
                result = idx
            case _ if isinstance(idx, Sequence) and not isinstance(idx, str):
                result = list(idx)
            case _:
                raise IndexError(f"Unsupported index type: {type(idx).__name__}")
        if not result:
            raise IndexError("Index is empty")
        return [self.num_graphs + i if i < 0 else i for i in result]

    def __getitem__(self, key: str | int | slice | Tensor | list) -> Any:
        """Access an attribute by name, or select graphs by index.

        Parameters
        ----------
        key : str or index
            Attribute name (returns tensor) or graph index (returns
            :class:`AtomicData` for int, :class:`Batch` for slice/tensor).
        """
        match key:
            case str():
                return self._get_attr(key)
            case int():
                return self.get_data(key)
            case _:
                return self.index_select(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Set an attribute, routing to the correct group."""
        self._storage[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._storage

    def __len__(self) -> int:
        return self.num_graphs

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        yield from self._storage.items()

    def __repr__(self) -> str:
        return (
            f"Batch(num_graphs={self.num_graphs}, "
            f"num_nodes={self.num_nodes}, "
            f"num_edges={self.num_edges}, "
            f"device={self.device})"
        )

    def _get_attr(self, key: str) -> Tensor:
        """Look up *key* across all groups."""
        for group in self._storage.groups.values():
            if key in group:
                return group[key]
        raise KeyError(f"Attribute '{key}' not found in batch")

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attribute access to the storage groups."""
        if name.startswith("_") or name in {"device", "keys"}:
            raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")
        try:
            return self._get_attr(name)
        except KeyError:
            raise AttributeError(
                f"'{type(self).__name__}' has no attribute '{name}'"
            ) from None

    def __delitem__(self, key: str) -> None:
        """Delete an attribute from the underlying storage."""
        del self._storage[key]

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def append(self, other: Batch) -> None:
        """Append another batch (in-place via concatenation).

        If *other* is missing a group that this batch has (e.g. system-level
        data), this batch's tensors in that group are extended with zeros so
        that the first dimension (num graphs) stays aligned.

        Parameters
        ----------
        other : Batch
            Batch to append.  Must not share storage with *self* —
            use ``batch.append(batch.clone())`` to double a batch.
        """
        if other is self or other._storage is self._storage:
            raise ValueError(
                "Cannot append a Batch that shares storage with the "
                "receiver (would corrupt both).  Use "
                "batch.append(batch.clone()) instead."
            )

        atoms = self._atoms_group
        other_atoms = other._atoms_group
        saved_ei = None
        if atoms is not None and other_atoms is not None:
            total_nodes = atoms.num_elements()
            other_edges = other._edges_group
            if other_edges is not None and "edge_index" in other_edges:
                saved_ei = other_edges._data["edge_index"]
                other_edges._data["edge_index"] = saved_ei + total_nodes

        n_other = other.num_graphs
        for group_name, group in self._storage.groups.items():
            other_group = other._storage.groups.get(group_name)
            if other_group is not None:
                group.concatenate(other_group)
            else:
                group.extend_for_appended_graphs(n_other)

        # Restore other's edge_index to avoid mutating the input batch.
        if saved_ei is not None:
            other_edges._data["edge_index"] = saved_ei

    def append_data(
        self,
        data_list: list[AtomicData],
        exclude_keys: list[str] | None = None,
    ) -> None:
        """Append individual :class:`AtomicData` objects to this batch.

        Parameters
        ----------
        data_list : list[AtomicData]
            Data objects to append.
        exclude_keys : list[str], optional
            Keys to exclude.

        Raises
        ------
        ValueError
            If *data_list* is empty.
        """
        if not data_list:
            raise ValueError("No data provided to append.")
        other = Batch.from_data_list(
            data_list,
            device=self.device,
            exclude_keys=exclude_keys,
        )
        self.append(other)

    def add_key(
        self,
        key: str,
        values: list[Tensor],
        level: str = "node",
        overwrite: bool = False,
    ) -> None:
        """Add a new key-value pair to the batch.

        Parameters
        ----------
        key : str
            Name of the new attribute.
        values : list[Tensor]
            One value per graph.
        level : str
            One of ``"node"``, ``"edge"``, ``"system"``.
        overwrite : bool
            If ``True``, overwrite existing keys.

        Raises
        ------
        ValueError
            If key exists and *overwrite* is ``False``, or if the number
            of values does not match the batch size.
        """
        if key in self._storage and not overwrite:
            raise ValueError(
                f"Key '{key}' already exists in batch. "
                "Set overwrite=True to replace existing values."
            )
        if len(values) != self.num_graphs:
            raise ValueError(
                f"Number of values ({len(values)}) must match "
                f"number of graphs in batch ({self.num_graphs})"
            )

        device = self.device
        values = [v.to(device) if isinstance(v, Tensor) else v for v in values]

        group_name = {"node": "atoms", "edge": "edges", "system": "system"}.get(
            level, "atoms"
        )
        group = self._storage.groups.get(group_name)
        if group is None:
            raise ValueError(f"Group '{group_name}' not found in batch")

        if level == "system":
            # squeeze (1, *trailing) per-graph to (num_graphs, *trailing)
            squeezed = [
                v.squeeze(0) if v.dim() >= 1 and v.shape[0] == 1 else v for v in values
            ]
            group._data[key] = torch.stack(squeezed, dim=0)
        else:
            group._data[key] = torch.cat(values, dim=0)

        if self.keys is not None:
            self.keys[level].add(key)

    # ------------------------------------------------------------------
    # DataMixin overrides (performance-critical)
    # ------------------------------------------------------------------

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> Batch:
        """Move all tensors to *device*.

        Overrides :meth:`DataMixin.to` for performance: delegates to
        :meth:`MultiLevelStorage.to_device` instead of the
        ``model_dump`` / ``map_structure`` / ``model_validate`` round-trip.

        Parameters
        ----------
        device : torch.device | str
            Target device.
        dtype : torch.dtype, optional
            Ignored (present for API compatibility).
        non_blocking : bool
            Ignored (present for API compatibility).

        Returns
        -------
        Batch
        """
        new = self.clone()
        new._storage.to_device(device)
        new.device = torch.device(device) if isinstance(device, str) else device
        return new

    def clone(self) -> Batch:
        """Return a deep copy.

        Overrides :meth:`DataMixin.clone` for performance.

        Returns
        -------
        Batch
        """
        return Batch._construct(
            device=self.device,
            keys={k: v.copy() for k, v in self.keys.items()} if self.keys else None,
            storage=self._storage.clone(),
            data_class=self._data_class,
        )

    def cpu(self) -> Batch:
        """Return a copy on CPU."""
        return self.to("cpu")

    def cuda(self, device: int | None = None, non_blocking: bool = False) -> Batch:
        """Return a copy on CUDA."""
        dev = f"cuda:{device}" if device is not None else "cuda"
        return self.to(dev)

    def contiguous(self) -> Batch:
        """Ensure contiguous memory layout for all tensors.

        Returns
        -------
        Self
            For method chaining.
        """
        self._make_contiguous()
        return self

    def pin_memory(self) -> Batch:
        """Pin all tensors to page-locked memory.

        Returns
        -------
        Self
            For method chaining.
        """
        for group in self._storage.groups.values():
            for key, tensor in list(group.items()):
                group._data[key] = tensor.pin_memory()
        return self

    def _make_contiguous(self) -> Batch:
        """Ensure all tensors are contiguous. Returns self for chaining."""
        for group in self._storage.groups.values():
            for key, tensor in list(group.items()):
                if not tensor.is_contiguous():
                    group._data[key] = tensor.contiguous()
        return self

    # ------------------------------------------------------------------
    # Custom serialization
    # ------------------------------------------------------------------

    def model_dump(self, **kwargs: Any) -> dict[str, Any]:
        """Serialize the batch into a flat dictionary.

        Collects all tensors from the underlying :class:`MultiLevelStorage`
        groups, plus metadata fields (``device``, ``keys``, ``batch``,
        ``ptr``, ``num_nodes_list``, ``num_edges_list``, ``num_graphs``).

        Returns
        -------
        dict[str, Any]
        """
        result: dict[str, Any] = {
            "device": self.device,
            "keys": self.keys,
            "batch": self.batch,
            "ptr": self.ptr,
            "num_graphs": self.num_graphs,
            "num_nodes_list": self.num_nodes_list,
            "num_edges_list": self.num_edges_list,
        }
        result.update(
            {
                key: tensor
                for group in self._storage.groups.values()
                for key, tensor in group.items()
            }
        )
        exclude_none = kwargs.get("exclude_none", False)
        if exclude_none:
            result = {k: v for k, v in result.items() if v is not None}
        return result

    # ------------------------------------------------------------------
    # Distributed communication
    # ------------------------------------------------------------------

    def isend(
        self,
        dst: int,
        *,
        tag: int = 0,
        group: ProcessGroup | None = None,
    ) -> _BatchSendHandle:
        """Non-blocking send of this batch to *dst*.

        Transmits a 3-int metadata header (``num_graphs``, ``num_nodes``,
        ``num_edges``), per-group segment lengths for segmented groups,
        and the bulk tensor data via ``TensorDict.isend()``.

        Parameters
        ----------
        dst : int
            Destination rank.
        tag : int
            Base message tag.  Incremented deterministically per group.
        group : ProcessGroup, optional
            Process group.  ``None`` uses the default group.

        Returns
        -------
        _BatchSendHandle
            Handle whose ``.wait()`` blocks until all sends complete.
        """
        handles: list[Work | list[Work] | int | None] = []

        meta = torch.tensor(
            [self.num_graphs, self.num_nodes, self.num_edges],
            dtype=torch.int64,
            device=self.device,
        )
        handles.append(dist.isend(meta, dst=dst, tag=tag, group=group))
        tag_offset = 1

        if self.num_graphs == 0:
            return _BatchSendHandle(handles)

        for name in ("atoms", "edges"):
            grp = self._storage.groups.get(name)
            if grp is not None and isinstance(grp, SegmentedLevelStorage):
                seg_len = grp.segment_lengths[: self.num_graphs].contiguous()
                handles.append(
                    dist.isend(seg_len, dst=dst, tag=tag + tag_offset, group=group)
                )
            tag_offset += 1

        for name in ("atoms", "edges", "system"):
            grp = self._storage.groups.get(name)
            if grp is None:
                tag_offset += 1
                continue
            if isinstance(grp, SegmentedLevelStorage):
                n = grp.num_elements()
            else:
                n = self.num_graphs
            occupied_td = grp._data[:n]
            result = occupied_td.isend(
                dst=dst,
                init_tag=tag + tag_offset,
                group=group,
                return_early=True,
            )
            if isinstance(result, list):
                handles.extend(result)
            else:
                handles.append(result)
            tag_offset += len(list(grp.keys())) + 1

        return _BatchSendHandle(handles)

    @classmethod
    def irecv(
        cls,
        src: int,
        device: torch.device | str,
        *,
        template: Batch | None = None,
        tag: int = 0,
        group: ProcessGroup | None = None,
    ) -> _BatchRecvHandle:
        """Non-blocking receive of a batch from *src*.

        Posts non-blocking receives for the metadata header, then returns
        a :class:`_BatchRecvHandle` whose ``.wait()`` blocks until all
        data arrives and reconstructs a :class:`Batch`.

        Parameters
        ----------
        src : int
            Source rank.
        device : torch.device | str
            Device to receive tensors onto.
        template : Batch, optional
            Template batch providing attribute keys, dtypes, and group
            structure.  Required for the first receive; may be cached
            by the caller for subsequent calls.
        tag : int
            Base message tag.
        group : ProcessGroup, optional
            Process group.

        Returns
        -------
        _BatchRecvHandle
            Handle whose ``.wait()`` returns the received :class:`Batch`.
        """
        device = torch.device(device) if isinstance(device, str) else device

        meta = torch.empty(3, dtype=torch.int64, device=device)
        meta_handle = dist.irecv(meta, src=src, tag=tag, group=group)

        return _BatchRecvHandle(
            meta=meta,
            meta_handle=meta_handle,
            src=src,
            device=device,
            template=template,
            base_tag=tag,
            group=group,
        )

    def send(
        self,
        dst: int,
        *,
        tag: int = 0,
        group: ProcessGroup | None = None,
    ) -> None:
        """Blocking send to *dst*.

        Equivalent to ``self.isend(dst, tag=tag, group=group).wait()``.

        Parameters
        ----------
        dst : int
            Destination rank.
        tag : int
            Base message tag.
        group : ProcessGroup, optional
            Process group.
        """
        self.isend(dst=dst, tag=tag, group=group).wait()

    @classmethod
    def recv(
        cls,
        src: int,
        device: torch.device | str,
        *,
        template: Batch | None = None,
        tag: int = 0,
        group: ProcessGroup | None = None,
    ) -> Batch:
        """Blocking receive from *src*.

        Equivalent to ``cls.irecv(src, device, ...).wait()``.

        Parameters
        ----------
        src : int
            Source rank.
        device : torch.device | str
            Device to receive tensors onto.
        template : Batch, optional
            Template batch.
        tag : int
            Base message tag.
        group : ProcessGroup, optional
            Process group.

        Returns
        -------
        Batch
        """
        return cls.irecv(
            src=src,
            device=device,
            template=template,
            tag=tag,
            group=group,
        ).wait()

    @classmethod
    def empty_like(
        cls,
        batch: Batch,
        *,
        device: torch.device | str | None = None,
    ) -> Batch:
        """Create an empty batch (0 graphs) with the same schema as *batch*.

        Parameters
        ----------
        batch : Batch
            Template batch for attribute keys and dtypes.
        device : torch.device | str, optional
            Device for the new batch.  Defaults to ``batch.device``.

        Returns
        -------
        Batch
            A batch with ``num_graphs == 0``.
        """
        dev = device if device is not None else batch.device
        return cls.empty(
            num_systems=0,
            num_nodes=0,
            num_edges=0,
            template=batch,
            device=dev,
        )


# ======================================================================
# Distributed communication handle classes
# ======================================================================


class _BatchSendHandle:
    """Aggregates multiple async distributed send handles.

    Calling ``.wait()`` blocks until all underlying sends have completed.

    Parameters
    ----------
    handles : list
        A list of ``torch.distributed.Work`` objects (or ``int`` /
        ``None`` values which are silently skipped).
    """

    def __init__(self, handles: list) -> None:
        self._handles = handles

    def wait(self) -> None:
        """Block until all sends complete."""
        for h in self._handles:
            if h is not None and hasattr(h, "wait"):
                h.wait()


class _BatchRecvHandle:
    """Deferred receive that reconstructs a :class:`Batch` on ``.wait()``.

    Created by :meth:`Batch.irecv`.  The metadata header receive is
    already posted; ``.wait()`` blocks on it, then posts and completes
    the segment-length and bulk-data receives.

    Parameters
    ----------
    meta : Tensor
        Pre-allocated ``(3,)`` int64 tensor for the metadata header.
    meta_handle : Work
        Async receive handle for *meta*.
    src : int
        Source rank.
    device : torch.device
        Device to receive tensors onto.
    template : Batch | None
        Template batch for attribute keys and dtypes.
    base_tag : int
        Base message tag (must match sender's *tag*).
    group : ProcessGroup | None
        Process group.
    """

    def __init__(
        self,
        *,
        meta: Tensor,
        meta_handle: Work,
        src: int,
        device: torch.device,
        template: Batch | None,
        base_tag: int,
        group: ProcessGroup | None,
    ) -> None:
        self._meta = meta
        self._meta_handle = meta_handle
        self._src = src
        self._device = device
        self._template = template
        self._base_tag = base_tag
        self._group = group

    def wait(self) -> Batch:
        """Block until all data arrives and return the received :class:`Batch`.

        Returns
        -------
        Batch
            The reconstructed batch.  If the sender sent a sentinel
            (0-graph batch), returns ``Batch.empty(...)`` with 0 capacity.
        """
        self._meta_handle.wait()
        num_graphs, num_nodes, num_edges = self._meta.tolist()
        num_graphs = int(num_graphs)
        num_nodes = int(num_nodes)
        num_edges = int(num_edges)

        tag_offset = 1

        if num_graphs == 0:
            if self._template is not None:
                return Batch.empty(
                    num_systems=0,
                    num_nodes=0,
                    num_edges=0,
                    template=self._template,
                    device=self._device,
                )
            return Batch(device=self._device)

        handles: list = []

        atoms_seg: Tensor | None = None
        edges_seg: Tensor | None = None

        if self._template is not None:
            atoms_grp = self._template._storage.groups.get("atoms")
            if atoms_grp is not None and isinstance(atoms_grp, SegmentedLevelStorage):
                atoms_seg = torch.empty(
                    num_graphs, dtype=torch.int32, device=self._device
                )
                handles.append(
                    dist.irecv(
                        atoms_seg,
                        src=self._src,
                        tag=self._base_tag + tag_offset,
                        group=self._group,
                    )
                )
        tag_offset += 1

        if self._template is not None:
            edges_grp = self._template._storage.groups.get("edges")
            if edges_grp is not None and isinstance(edges_grp, SegmentedLevelStorage):
                edges_seg = torch.empty(
                    num_graphs, dtype=torch.int32, device=self._device
                )
                handles.append(
                    dist.irecv(
                        edges_seg,
                        src=self._src,
                        tag=self._base_tag + tag_offset,
                        group=self._group,
                    )
                )
        tag_offset += 1

        groups: dict[str, UniformLevelStorage | SegmentedLevelStorage] = {}
        attr_map = (
            self._template._storage.attr_map
            if self._template is not None
            else LevelSchema()
        )

        for name, capacity, seg_lens in [
            ("atoms", num_nodes, atoms_seg),
            ("edges", num_edges, edges_seg),
            ("system", num_graphs, None),
        ]:
            template_grp = (
                self._template._storage.groups.get(name)
                if self._template is not None
                else None
            )
            if template_grp is None:
                tag_offset += (
                    (len(list(template_grp.keys())) + 1)
                    if template_grp is not None
                    else 1
                )
                continue

            keys = list(template_grp.keys())
            if not keys:
                tag_offset += 1
                continue

            recv_data = {}
            for k in keys:
                ref_tensor = template_grp[k]
                trailing_shape = ref_tensor.shape[1:]
                recv_data[k] = torch.empty(
                    (capacity,) + trailing_shape,
                    dtype=ref_tensor.dtype,
                    device=self._device,
                )

            recv_td = TensorDict(recv_data, batch_size=[capacity], device=self._device)
            td_handles = recv_td.irecv(
                src=self._src,
                init_tag=self._base_tag + tag_offset,
                group=self._group,
                return_premature=True,
            )
            if isinstance(td_handles, list):
                handles.extend(td_handles)
            else:
                handles.append(td_handles)
            tag_offset += len(keys) + 1

            if name == "system":
                storage = UniformLevelStorage(
                    data={k: recv_td[k] for k in keys},
                    device=self._device,
                    validate=False,
                    attr_map=attr_map,
                )
                groups[name] = storage
            else:
                if seg_lens is None:
                    continue
                storage = SegmentedLevelStorage(
                    data={k: recv_td[k] for k in keys},
                    segment_lengths=seg_lens,
                    device=self._device,
                    validate=False,
                    attr_map=attr_map,
                )
                groups[name] = storage

        for h in handles:
            if h is not None and hasattr(h, "wait"):
                h.wait()

        mls = MultiLevelStorage(groups=groups, attr_map=attr_map, validate=False)
        return Batch._construct(
            device=self._device,
            keys=(
                {k: v.copy() for k, v in self._template.keys.items()}
                if self._template is not None and self._template.keys is not None
                else None
            ),
            storage=mls,
            data_class=(
                self._template._data_class if self._template is not None else AtomicData
            ),
        )
