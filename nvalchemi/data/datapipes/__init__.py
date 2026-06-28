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
                                    (dict -> AtomicData, device transfer,
                                     [optional] per-sample transforms, prefetch)
                                        |
                                    DataLoader
                                    (Dataset.load_batches -> Batch,
                                     [optional] per-batch transforms, iteration)

    MultiDataset can wrap several Dataset instances behind one global
    index space while preserving the same batch-loading contract.

**Writer** (:class:`AtomicDataZarrWriter`) serializes ``AtomicData`` or
``Batch`` objects into a structured Zarr store with CSR-style pointer
arrays for variable-size graph data.

**Reader** (:class:`AtomicDataZarrReader`, or any
:class:`~nvalchemi.data.datapipes.backends.base.Reader` subclass)
provides random access to individual samples as ``dict[str, Tensor]``.

**Dataset** wraps a Reader and constructs ``AtomicData`` objects,
handling device transfers and optional CUDA-stream prefetching. It also
applies optional per-sample transforms after device transfer; see
:class:`~nvalchemi.data.transforms.Compose`, passed via the
``transforms=`` kwarg. Its canonical explicit batch API is
:meth:`~nvalchemi.data.datapipes.dataset.Dataset.load_batches`, which
uses fused ``read_many`` requests and returns one ``Batch`` per requested
batch-index list.

**DataLoader** iterates over a Dataset in batches, collating
``AtomicData`` samples into ``Batch`` objects through the Dataset batch
loader. Positive ``prefetch_factor`` values fuse several emitted batches
into one background read window. Optional per-batch transforms run on the
collated batch; see :class:`~nvalchemi.data.transforms.Compose`, passed
via the ``batch_transforms=`` kwarg.

**MultiDataset** composes multiple Dataset instances and routes
``load_batches`` requests to the owning child datasets before restoring
the requested global sample order.
"""

from __future__ import annotations

from nvalchemi.data.datapipes.backends.base import Reader
from nvalchemi.data.datapipes.backends.zarr import (
    AtomicDataZarrReader,
    AtomicDataZarrWriter,
    ZarrArrayConfig,
    ZarrWriteConfig,
)
from nvalchemi.data.datapipes.dataloader import DataLoader
from nvalchemi.data.datapipes.dataset import Dataset
from nvalchemi.data.datapipes.multidataset import MultiDataset
from nvalchemi.data.datapipes.samplers import (
    DistributedSamplerProtocol,
    MultiDatasetBatchSampler,
    MultiDatasetSampler,
)

__all__ = [
    # Backends
    "Reader",
    "AtomicDataZarrReader",
    "AtomicDataZarrWriter",
    "ZarrArrayConfig",
    "ZarrWriteConfig",
    # Pipeline
    "Dataset",
    "MultiDataset",
    "DistributedSamplerProtocol",
    "MultiDatasetSampler",
    "MultiDatasetBatchSampler",
    "DataLoader",
]
