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
# type: ignore
from __future__ import annotations

import collections
import copy
import re
from abc import abstractmethod
from collections.abc import Callable, Iterator
from typing import Any

import torch
from pydantic import BaseModel
from tree import map_structure


def _move_obj_to_device(
    obj: Any,
    device: torch.device,
    fp_dtype: torch.dtype | None = None,
    non_blocking: bool = False,
) -> Any:
    """Move an object to a specified device if it supports device movement.

    Optionally, floating point tensors can also be re-cast to a different
    floating point dtype.

    Parameters
    ----------
    obj : Any
        The object to move to the device. Can be any type.
    device : torch.device
        The target device to move the object to.
    fp_dtype : torch.dtype, default torch.float32
        The floating point dtype to move the object to.
    non_blocking : bool, default False
        Whether to perform the operation asynchronously.

    Returns
    -------
    Any
        The object moved to the specified device if it supports device movement,
        otherwise returns the original object unchanged.

    Notes
    -----
    This function checks if the object is a PyTorch tensor or has a `.to()` method
    before attempting to move it to the device. Objects without device movement
    support are returned as-is.
    """
    if isinstance(obj, torch.Tensor) or hasattr(obj, "to"):
        # only cast floating point tensors
        if hasattr(obj, "dtype") and obj.dtype.is_floating_point:
            obj = obj.to(device=device, dtype=fp_dtype, non_blocking=non_blocking)
        else:
            obj = obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, list):
        return [_move_obj_to_device(o, device, fp_dtype, non_blocking) for o in obj]
    if isinstance(obj, tuple):
        return tuple(
            _move_obj_to_device(o, device, fp_dtype, non_blocking) for o in obj
        )
    if isinstance(obj, dict):
        return {
            k: _move_obj_to_device(v, device, fp_dtype, non_blocking)
            for k, v in obj.items()
        }
    if isinstance(obj, torch.device):
        return device
    return obj


def size_repr(key: str, item: Any, indent: int = 0) -> str:
    """
    Returns a string representation of the size of the item.
    """
    indent_str = " " * indent
    if torch.is_tensor(item) and item.dim() == 0:
        out = item.item()
    elif torch.is_tensor(item):
        out = str(list(item.size()))
    elif isinstance(item, list) or isinstance(item, tuple):
        out = str([len(item)])
    elif isinstance(item, dict):
        lines = [indent_str + size_repr(k, v, 2) for k, v in item.items()]
        out = "{\n" + ",\n".join(lines) + "\n" + indent_str + "}"
    elif isinstance(item, str):
        out = f'"{item}"'
    else:
        out = str(item)

    return f"{indent_str}{key}={out}"


class DataMixin:
    r"""A mixin class providing graph data functionality for Pydantic models.

    This mixin provides all the graph data manipulation methods and properties
    that can be used with Pydantic models. It includes methods for accessing,
    iterating, and manipulating graph attributes.

    The mixin is designed to work with Pydantic models that have graph attributes
    as fields. It provides the same interface as the original Data class but
    works with Pydantic's attribute access patterns.

    Example::

        class MyGraphData(BaseModel, DataMixin):
            x: Optional[torch.Tensor] = None
            neighbor_list: Optional[torch.Tensor] = None
            y: Optional[torch.Tensor] = None
    """

    def __iter__(self) -> Iterator[tuple[str, Any]]:
        r"""Iterates over all present attributes in the data, yielding their
        attribute names and content."""
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys: str) -> Iterator[tuple[str, Any]]:
        r"""Iterates over all attributes :obj:`*keys` in the data, yielding
        their attribute names and content.
        If :obj:`*keys` is not given this method will iterative over all
        present attributes."""
        for key in self.__class__.model_fields:
            if key in self:
                yield key, self[key]

    def __inc__(self, key: str, value: Any) -> int:
        r"""Returns the incremental count to cumulatively increase the value
        of the next attribute of :obj:`key` when creating batches.

        .. note::

            This method is for internal use only, and should only be overridden
            if the batch concatenation process is corrupted for a specific data
            attribute.
        """
        # Only `*index*` and `*face*` attributes should be cumulatively summed
        # up when creating batches.
        return (
            self.num_nodes if bool(re.search("(index|face|neighbor_list)", key)) else 0
        )

    @property
    @abstractmethod
    def num_nodes(self) -> torch.LongTensor:
        """Return the number of nodes in a single graph, or a batch of graphs."""
        raise NotImplementedError

    @property
    def num_edges(self) -> int | None:
        """
        Returns the number of edges in the graph.
        For undirected graphs, this will return the number of bi-directional
        edges, which is double the amount of unique edges.
        """
        if self.neighbor_list is not None:
            return self.neighbor_list.size(0)
        if self.edge_attr is not None:
            return self.edge_attr.size(0)
        if (adj := self.__dict__.get("adj")) is not None:
            return adj.nnz()
        if (adj_t := self.__dict__.get("adj_t")) is not None:
            return adj_t.nnz()
        return None

    def __apply__(self, item: Any, func: Callable[[Any], Any]) -> Any:
        """
        Apply the function :obj:`func` to the item.
        """
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func: Callable[[Any], Any], *keys: str) -> DataMixin:
        r"""Applies the function :obj:`func` to all tensor attributes
        :obj:`*keys`. If :obj:`*keys` is not given, :obj:`func` is applied to
        all present attributes.
        """
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def contiguous(self, *keys: str) -> DataMixin:
        r"""Ensures a contiguous memory layout for all attributes :obj:`*keys`.
        If :obj:`*keys` is not given, all present attributes are ensured to
        have a contiguous memory layout."""
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(
        self,
        device: torch.device,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> Any:
        """
        Move all tensor data to a specified device.

        Parameters
        ----------
        device : torch.device
            The target device to move the data to.
        non_blocking : bool, default False
            Whether to perform the operation asynchronously.

        Returns
        -------
        Any
            Returns a new instance of the data object with all tensor data
            moved to the specified device.
        """
        if not issubclass(self.__class__, BaseModel):
            raise ValueError("The `to` method can only be used with pydantic models.")
        # we use model_construct to avoid validation, because we assume
        # that validation has already been performed
        return self.__class__.model_validate(
            map_structure(
                lambda x: _move_obj_to_device(x, device, dtype, non_blocking),
                self.model_dump(exclude_none=True, exclude={"__data_class__"}),
            )
        )

    def cpu(self, *keys: str) -> DataMixin:
        r"""Copies all attributes :obj:`*keys` to CPU memory.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.cpu(), *keys)

    def cuda(self, device=None, non_blocking=False, *keys: str) -> DataMixin:
        r"""Copies all attributes :obj:`*keys` to CUDA memory.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(
            lambda x: x.cuda(device=device, non_blocking=non_blocking), *keys
        )

    def clone(self) -> DataMixin:
        r"""
        Duplicates the data contents, assuming that the Mixin
        is used with a Pydantic model.

        Uses model_construct to avoid re-validation of data to
        minimize overhead.
        """
        if not hasattr(self, "model_construct"):
            raise ValueError(
                "The `clone` method can only be used with pydantic models."
            )
        return self.model_construct(
            **{
                k: v.clone() if torch.is_tensor(v) else copy.deepcopy(v)
                for k, v in self.model_dump(exclude_none=True).items()
            }
        )

    def pin_memory(self, *keys: str) -> DataMixin:
        r"""Copies all attributes :obj:`*keys` to pinned memory.
        If :obj:`*keys` is not given, the conversion is applied to all present
        attributes."""
        return self.apply(lambda x: x.pin_memory(), *keys)

    @classmethod
    def from_dict(cls, dictionary):
        r"""Creates a data object from a python dictionary."""
        data = cls()

        for key, item in dictionary.items():
            data[key] = item

        return data

    def to_dict(self) -> dict:
        """
        Returns a dictionary of the data object.
        """
        return {key: item for key, item in self}

    def to_namedtuple(self) -> Any:
        """
        Returns a namedtuple of the data object.
        """
        keys = self.keys
        DataTuple = collections.namedtuple("DataTuple", keys)
        return DataTuple(*[self[key] for key in keys])

    def debug(self) -> None:
        """
        Debug the data object.
        """
        if self.neighbor_list is not None:
            if self.neighbor_list.dtype not in [torch.int32, torch.int64]:
                raise RuntimeError(
                    ("Expected neighbor_list of dtype {}, but found dtype  {}").format(
                        torch.int32, self.neighbor_list.dtype
                    )
                )

        if self.face is not None:
            if self.face.dtype not in [torch.int32, torch.int64]:
                raise RuntimeError(
                    ("Expected face indices of dtype {}, but found dtype  {}").format(
                        torch.int32, self.face.dtype
                    )
                )

        if self.neighbor_list is not None:
            if self.neighbor_list.dim() != 2 or self.neighbor_list.size(1) != 2:
                raise RuntimeError(
                    (
                        "Neighbor list should have shape [num_edges, 2] but found"
                        " shape {}"
                    ).format(self.neighbor_list.size())
                )

        if self.neighbor_list is not None and self.num_nodes is not None:
            if self.neighbor_list.numel() > 0:
                min_index = self.neighbor_list.min()
                max_index = self.neighbor_list.max()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    (
                        "Neighbor list indices must lay in the interval [0, {}]"
                        " but found them in the interval [{}, {}]"
                    ).format(self.num_nodes - 1, min_index, max_index)
                )

        if self.face is not None:
            if self.face.dim() != 2 or self.face.size(0) != 3:
                raise RuntimeError(
                    (
                        "Face indices should have shape [3, num_faces] but found"
                        " shape {}"
                    ).format(self.face.size())
                )

        if self.face is not None and self.num_nodes is not None:
            if self.face.numel() > 0:
                min_index = self.face.min()
                max_index = self.face.max()
            else:
                min_index = max_index = 0
            if min_index < 0 or max_index > self.num_nodes - 1:
                raise RuntimeError(
                    (
                        "Face indices must lay in the interval [0, {}]"
                        " but found them in the interval [{}, {}]"
                    ).format(self.num_nodes - 1, min_index, max_index)
                )

        if self.neighbor_list is not None and self.edge_attr is not None:
            if self.neighbor_list.size(0) != self.edge_attr.size(0):
                raise RuntimeError(
                    (
                        "Neighbor list and edge attributes hold a differing "
                        "number of edges, found {} and {}"
                    ).format(self.neighbor_list.size(), self.edge_attr.size())
                )

        if self.x is not None and self.num_nodes is not None:
            if self.x.size(0) != self.num_nodes:
                raise RuntimeError(
                    (
                        "Node features should hold {} elements in the first "
                        "dimension but found {}"
                    ).format(self.num_nodes, self.x.size(0))
                )

        if self.pos is not None and self.num_nodes is not None:
            if self.pos.size(0) != self.num_nodes:
                raise RuntimeError(
                    (
                        "Node positions should hold {} elements in the first "
                        "dimension but found {}"
                    ).format(self.num_nodes, self.pos.size(0))
                )

        if self.normal is not None and self.num_nodes is not None:
            if self.normal.size(0) != self.num_nodes:
                raise RuntimeError(
                    (
                        "Node normals should hold {} elements in the first "
                        "dimension but found {}"
                    ).format(self.num_nodes, self.normal.size(0))
                )

    def __repr__(self) -> str:
        """
        Returns a string representation of the data object.
        """
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))
