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
"""Datapipes package for AtomicData serialization and loading.

The datapipes package provides a composable pipeline for persisting and
loading :class:`~nvalchemi.data.atomic_data.AtomicData` objects, with
CUDA-stream prefetching support for high-throughput training workflows.

Pipeline overview
-----------------

::

    Writer                          Reader
    (AtomicData/Batch -> Zarr)      (Zarr -> dict[str, Tensor])
                                        |
                                    Dataset
                                    (dict -> AtomicData, device transfer, prefetch)
                                        |
                                    DataLoader
                                    (AtomicData -> Batch, batching, iteration)

**Writer** (:class:`AtomicDataZarrWriter`) serializes ``AtomicData`` or
``Batch`` objects into a structured Zarr store with CSR-style pointer
arrays for variable-size graph data.

**Reader** (:class:`AtomicDataZarrReader`, or any
:class:`~nvalchemi.data.datapipes.backends.base.Reader` subclass)
provides random access to individual samples as ``dict[str, Tensor]``.

**Dataset** wraps a Reader and constructs ``AtomicData`` objects,
handling device transfers and optional CUDA-stream prefetching.

**DataLoader** iterates over a Dataset in batches, collating
``AtomicData`` samples into ``Batch`` objects via
:meth:`~nvalchemi.data.batch.Batch.from_data_list`.
"""

from __future__ import annotations

from nvalchemi.data.datapipes.backends.base import Reader
from nvalchemi.data.datapipes.backends.zarr import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
)
from nvalchemi.data.datapipes.collection import (
    ConcatReader,
    DataCollection,
    RandomSplit,
    Split,
    SplitStrategy,
    SubsetReader,
)
from nvalchemi.data.datapipes.dataloader import DataLoader
from nvalchemi.data.datapipes.dataset import Dataset

__all__ = [
    "ConcatReader",
    "Reader",
    "AtomicDataZarrReader",
    "AtomicDataZarrWriter",
    "DataCollection",
    "DataLoader",
    "Dataset",
    "RandomSplit",
    "Split",
    "SplitStrategy",
    "SubsetReader",
]
