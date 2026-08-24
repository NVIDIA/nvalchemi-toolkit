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
"""Derived group-related metadata for graph batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch


_INTEGRAL_DTYPES = frozenset(
    {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }
)


def _normalize_and_validate_group_idx(
    group_idx: Tensor,
    *,
    num_graphs: int,
    device: torch.device,
) -> Tensor:
    """Validate group labels and normalize their contiguous runs to dense IDs."""
    if not isinstance(group_idx, Tensor):
        raise TypeError("group_idx must be a torch.Tensor")
    if group_idx.ndim != 1 or group_idx.shape[0] != num_graphs:
        raise ValueError(
            f"group_idx must have shape ({num_graphs},), got {tuple(group_idx.shape)}"
        )
    if group_idx.dtype not in _INTEGRAL_DTYPES:
        raise TypeError(f"group_idx must have an integer dtype, got {group_idx.dtype}")

    group_idx = group_idx.to(device=device, dtype=torch.long)
    if num_graphs == 0:
        return group_idx

    group_start = torch.ones(num_graphs, dtype=torch.bool, device=device)
    group_start[1:] = group_idx[1:] != group_idx[:-1]

    # Check to reject inputs like [4, 4, 1, 1, 4]
    group_labels = group_idx[group_start]
    if torch.unique(group_labels).numel() != group_labels.numel():
        raise ValueError("Graphs belonging to the same group must be contiguous")

    return group_start.cumsum(0) - 1


@dataclass(frozen=True)
class GroupLayout:
    """Derived mappings between node, graph, and group.

    A group is a contiguous collection of one or more graphs that are treated
    as a single logical unit for operations such as selection, updates, or
    convergence checks. Examples include the images along one NEB path and
    members of an ensemble that are processed together.
    This grouping is independent of the internal node-, edge-, and
    system-level storage groups used by :class:`Batch`.

    Group membership is read exclusively from the system-level ``group_idx``
    tensor stored on a :class:`~nvalchemi.data.batch.Batch`. The tensor has
    shape ``[B]``, and ``group_idx[b]`` identifies the group containing graph
    ``b``. Group labels are dense and zero-based, and all graphs in a group
    occupy contiguous positions in the batch. Here, ``G`` denotes the number
    of groups in the batch.

    Parameters
    ----------
    group_idx : torch.Tensor
        Dense group index for each graph, shape ``[B]``.
    graph_rank : torch.Tensor
        Position of each graph within its group, shape ``[B]``.
    node_to_group : torch.Tensor
        Dense group index for each node, shape ``[V]``.
    group_ptr : torch.Tensor
        Offsets delimiting each group's graph range, shape ``[G + 1]``.
        Graphs in group ``g`` occupy ``[group_ptr[g], group_ptr[g + 1])``.
    num_graphs_per_group : torch.Tensor
        Number of graphs in each group, shape ``[G]``.

    Examples
    --------
    For ``group_idx = [0, 0, 1, 1, 1]`` and per-graph node counts
    ``[2, 1, 1, 2, 1]``, the derived mappings are::

        group_idx = [0, 0, 1, 1, 1]
        graph_rank = [0, 1, 0, 1, 2]
        node_to_group = [0, 0, 0, 1, 1, 1, 1]
        group_ptr = [0, 2, 5]
        num_graphs_per_group = [2, 3]
    """

    group_idx: Tensor
    graph_rank: Tensor
    node_to_group: Tensor
    group_ptr: Tensor
    num_graphs_per_group: Tensor

    @classmethod
    def from_batch(cls, batch: Batch) -> GroupLayout:
        """Build a layout from a batch's system-level ``group_idx`` tensor.

        Parameters
        ----------
        batch : Batch
            Batch carrying a dense, contiguous ``group_idx`` tensor.

        Returns
        -------
        GroupLayout

        Raises
        ------
        ValueError
            If grouping metadata is missing or malformed.
        TypeError
            If ``group_idx`` does not use an integer dtype.
        """
        if "group_idx" not in batch:
            raise ValueError("Batch has no group_idx; call set_group_layout() first")

        raw_group_idx = batch.group_idx
        if raw_group_idx.ndim != 1 or raw_group_idx.shape[0] != batch.num_graphs:
            raise ValueError(
                "group_idx must have shape "
                f"({batch.num_graphs},), got {tuple(raw_group_idx.shape)}"
            )
        if raw_group_idx.dtype not in _INTEGRAL_DTYPES:
            raise TypeError(
                f"group_idx must have an integer dtype, got {raw_group_idx.dtype}"
            )

        group_idx = raw_group_idx.to(device=batch.device, dtype=torch.long)
        if batch.num_graphs == 0:
            num_graphs_per_group = torch.empty(0, dtype=torch.long, device=batch.device)
        else:
            # Check that group_idx is valid first
            step = group_idx[1:] - group_idx[:-1]
            if group_idx[0].item() != 0 or torch.any((step < 0) | (step > 1)):
                raise ValueError(
                    "group_idx must contain dense, contiguous groups starting at zero"
                )
            num_graphs_per_group = torch.bincount(group_idx)

        group_ptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=batch.device),
                num_graphs_per_group.cumsum(0),
            ]
        )
        graph_rank = (
            torch.arange(batch.num_graphs, dtype=torch.long, device=batch.device)
            - group_ptr[group_idx]
        )
        node_to_group = group_idx[batch.batch_idx.to(dtype=torch.long)]

        return cls(
            group_idx=group_idx,
            graph_rank=graph_rank,
            node_to_group=node_to_group,
            group_ptr=group_ptr,
            num_graphs_per_group=num_graphs_per_group,
        )

    @property
    def num_groups(self) -> int:
        """Number of groups in the layout.

        Returns
        -------
        int
            Size of the group cardinality ``G``.
        """
        return len(self.num_graphs_per_group)

    @property
    def device(self) -> torch.device:
        """Device containing the layout tensors.

        Returns
        -------
        torch.device
            Device inherited from the grouped batch.
        """
        return self.group_idx.device

    def reduce_all(self, graph_mask: Tensor) -> Tensor:
        """Reduce a boolean graph mask to groups using logical all.

        Parameters
        ----------
        graph_mask : torch.Tensor
            Boolean mask over graphs, shape ``[B]``.

        Returns
        -------
        torch.Tensor
            Boolean mask over groups, shape ``[G]``. An entry is true only
            when every graph in that group is selected.

        Raises
        ------
        TypeError
            If ``graph_mask`` is not a boolean tensor.
        ValueError
            If ``graph_mask`` does not have shape ``[B]``.

        Examples
        --------
        With ``group_idx = [0, 0, 1, 1, 1]``::

            graph_mask = [True, True, True, False, True]
            reduce_all = [True, False]
        """
        graph_mask = self._validate_mask(graph_mask, self.group_idx.numel(), "graph")
        counts = torch.zeros(self.num_groups, dtype=torch.long, device=self.device)
        counts.scatter_add_(0, self.group_idx, graph_mask.to(dtype=torch.long))
        return counts == self.num_graphs_per_group

    def reduce_any(self, graph_mask: Tensor) -> Tensor:
        """Reduce a boolean graph mask to groups using logical any.

        Parameters
        ----------
        graph_mask : torch.Tensor
            Boolean mask over graphs, shape ``[B]``.

        Returns
        -------
        torch.Tensor
            Boolean mask over groups, shape ``[G]``. An entry is true when
            at least one graph in that group is selected.

        Raises
        ------
        TypeError
            If ``graph_mask`` is not a boolean tensor.
        ValueError
            If ``graph_mask`` does not have shape ``[B]``.

        Examples
        --------
        With ``group_idx = [0, 0, 1, 1, 1]``::

            graph_mask = [True, True, True, False, True]
            reduce_any = [True, True]
        """
        graph_mask = self._validate_mask(graph_mask, self.group_idx.numel(), "graph")
        counts = torch.zeros(self.num_groups, dtype=torch.long, device=self.device)
        counts.scatter_add_(0, self.group_idx, graph_mask.to(dtype=torch.long))
        return counts > 0

    def broadcast(self, group_values: Tensor) -> Tensor:
        """Broadcast group values to their member graphs.

        Parameters
        ----------
        group_values : torch.Tensor
            Values with leading group cardinality, shape ``[G, ...]``.

        Returns
        -------
        torch.Tensor
            Values repeated at graph cardinality, shape ``[B, ...]``, on the
            layout device.

        Raises
        ------
        ValueError
            If ``group_values`` is scalar or its leading dimension is not
            ``G``.
        """
        if group_values.ndim == 0 or group_values.shape[0] != self.num_groups:
            raise ValueError(
                "group_values must have leading shape "
                f"({self.num_groups},), got {tuple(group_values.shape)}"
            )
        return group_values.to(device=self.device)[self.group_idx]

    def graph_mask(self, group_mask: Tensor) -> Tensor:
        """Broadcast a boolean group mask to its member graphs.

        Parameters
        ----------
        group_mask : torch.Tensor
            Boolean mask over groups, shape ``[G]``.

        Returns
        -------
        torch.Tensor
            Boolean mask over graphs, shape ``[B]``.

        Raises
        ------
        TypeError
            If ``group_mask`` is not a boolean tensor.
        ValueError
            If ``group_mask`` does not have shape ``[G]``.
        """
        group_mask = self._validate_mask(group_mask, self.num_groups, "group")
        return group_mask[self.group_idx]

    def selected_group_idx(self, group_mask: Tensor) -> Tensor:
        """Build local group indices for graphs in selected groups.

        Parameters
        ----------
        group_mask : torch.Tensor
            Boolean mask over groups, shape ``[G]``.

        Returns
        -------
        torch.Tensor
            Dense, zero-based group indices for the selected graphs, shape
            ``[B_selected]``. Selected groups retain their original order.

        Raises
        ------
        TypeError
            If ``group_mask`` is not a boolean tensor.
        ValueError
            If ``group_mask`` does not have shape ``[G]``.
        """
        group_mask = self._validate_mask(group_mask, self.num_groups, "group")
        dense_group_idx = group_mask.to(dtype=torch.long).cumsum(0) - 1
        selected_graphs = self.graph_mask(group_mask)
        return dense_group_idx[self.group_idx[selected_graphs]]

    def _validate_mask(self, mask: Tensor, size: int, cardinality: str) -> Tensor:
        if not isinstance(mask, Tensor):
            raise TypeError(f"{cardinality}_mask must be a torch.Tensor")
        if mask.dtype != torch.bool:
            raise TypeError(f"{cardinality}_mask must have dtype torch.bool")
        if mask.ndim != 1 or mask.shape[0] != size:
            raise ValueError(
                f"{cardinality}_mask must have shape ({size},), got {tuple(mask.shape)}"
            )
        return mask.to(device=self.device)


def _select_groups(
    batch: Batch,
    group_mask: Tensor,
) -> Batch:
    """Select complete groups from a batch into a new batch and rebase their local indices.

    Parameters
    ----------
    batch : Batch
        Grouped batch to select from.
    group_mask : torch.Tensor
        Boolean mask over groups, shape ``[G]``.

    Returns
    -------
    Batch
        Tight batch containing the selected complete groups. An all-false mask
        returns a zero-graph batch with the same schema.
    """
    layout = batch.group_layout
    graph_mask = layout.graph_mask(group_mask)
    selected_group_idx = layout.selected_group_idx(group_mask)

    if not torch.any(graph_mask):
        selected = type(batch).empty_like(batch)
        _ = selected.group_layout
        return selected

    selected = batch.index_select(graph_mask)
    selected.set_group_layout(selected_group_idx)
    return selected
