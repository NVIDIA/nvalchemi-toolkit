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

"""Low-level distributed communication primitives.

The distributed framework is organized into three import namespaces:

* **Declare + run** (:mod:`nvalchemi.distributed`): the spec types and run
  classes. Name what and where, not how.
* **Intent vocabulary** (:mod:`nvalchemi.distributed.helpers`): the context-aware
  helpers a model author calls inside a wrapper (``refresh_neighbors`` /
  ``system_sum`` / ``to_local`` / …).
* **Mechanism** (this module): the communication primitives promoted out of the
  private ``_core`` package so a power user can call a halo exchange or
  per-system reduce by hand, or write a novel ``StoragePolicy``.

This module changes exposure, not behavior.
"""

from __future__ import annotations

# Each symbol is a ``_core`` primitive promoted to this public path.
from nvalchemi.distributed._core.context import (
    NOT_DISTRIBUTED,
    DistributedContext,
    activate_dd_context,
    current_dd_context,
)
from nvalchemi.distributed._core.gather_primitives import (
    distributed_all_reduce,
    mesh_group,
)
from nvalchemi.distributed._core.halo_types import (
    GNNHaloMarkers,
    ParticleHaloConfig,
    ParticleHaloMetadata,
)
from nvalchemi.distributed._core.op_transforms import (
    AllReduceSum,
    GatherInputs,
    GatherInputsFull,
    ScatterOutputs,
    SliceOutputsOwned,
    SliceOwned,
)
from nvalchemi.distributed._core.particle_halo import (
    build_halo_meta_tensors,
    halo_forward_exchange,
    halo_forward_static_from_meta,
    halo_forward_static_op,
    halo_reverse_exchange,
    halo_scatter_correct_static_from_meta,
    halo_scatter_correct_static_op,
    pack_halo_meta,
    pad_field,
    particle_halo_padding_autograd,
    unpack_halo_meta,
)
from nvalchemi.distributed._core.per_system import (
    per_system_reduce,
    per_system_reduce_op,
)
from nvalchemi.distributed._core.shard_tensor import ShardTensor
from nvalchemi.distributed._core.storage_policy import (
    HaloStoragePolicy,
    StoragePolicy,
)

__all__ = [
    # halo exchange — eager
    "halo_forward_exchange",
    "halo_reverse_exchange",
    "particle_halo_padding_autograd",
    "pad_field",
    # halo — compile / fixed-shape static ops
    "halo_forward_static_op",
    "halo_scatter_correct_static_op",
    "halo_forward_static_from_meta",
    "halo_scatter_correct_static_from_meta",
    "build_halo_meta_tensors",
    "pack_halo_meta",
    "unpack_halo_meta",
    # DD context accessor + object (read by the intent helpers)
    "current_dd_context",
    "activate_dd_context",
    "NOT_DISTRIBUTED",
    "DistributedContext",
    # per-system reduce + collectives
    "per_system_reduce",
    "per_system_reduce_op",
    "distributed_all_reduce",
    "mesh_group",
    # low-level op transforms
    "GatherInputs",
    "GatherInputsFull",
    "SliceOwned",
    "ScatterOutputs",
    "AllReduceSum",
    "SliceOutputsOwned",
    # storage policies (write a novel one here; declare it on a spec)
    "StoragePolicy",
    "HaloStoragePolicy",
    # routing / metadata
    "ParticleHaloConfig",
    "ParticleHaloMetadata",
    "GNNHaloMarkers",
    # distributed tensor
    "ShardTensor",
]
