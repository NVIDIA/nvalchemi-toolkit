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

"""Construction helpers for :class:`ShardTensor`.

Every ``ShardTensor`` construction routes through the base
``_make_wrapper_subclass`` pattern, which requires a ``ShardTensorSpec`` at
construction time. The base's high-level ``from_local`` builds one via
``_infer_shard_tensor_spec_from_local_chunks``, which has hard CUDA
dependencies; this module bypasses that with manual construction.

It is the single entry point for "build a ShardTensorSpec from a local tensor +
a mesh" — used by ``ShardTensor.wrap`` and by handler output sites in
``_core/shard_tensor.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh


__all__ = ["make_local_shard_tensor_spec"]


def make_local_shard_tensor_spec(
    local_tensor: torch.Tensor,
    mesh: "DeviceMesh",
    placements: tuple | None = None,
) -> object:
    """Build a ``ShardTensorSpec`` from a local tensor + mesh.

    Bypasses the base's CUDA-only spec inference path.

    Parameters
    ----------
    local_tensor : torch.Tensor
        The per-rank local tensor whose shape/stride/dtype seed the spec's
        ``tensor_meta``. For the halo-padded case this is the ``[owned | halo]``
        block, whose shape differs per rank; cross-rank coherence is the
        dispatch handlers' responsibility, not the spec's.
    mesh : DeviceMesh
        Carries the process group for cross-rank ops.
    placements : tuple, optional
        Base-spec placement — plumbing, not the semantic truth. Defaults to
        ``(Replicate(),)`` per mesh dim: an honest ``Shard(0)`` would need every
        rank's padded row count (an all-gather on every construction, and this
        helper runs on every op-result re-wrap). The honest semantics live on
        the field's :class:`HaloStoragePolicy`; cross-rank routing flows through
        the registered dispatch handlers, not this placeholder.

    Returns
    -------
    ShardTensorSpec
        A spec suitable for ``ShardTensor.__new__(cls, local_tensor=t,
        spec=this, requires_grad=...)``.
    """
    # Lazy imports — these modules pull in DTensor machinery only available
    # when physicsnemo is installed.
    from torch.distributed.tensor._dtensor_spec import TensorMeta
    from torch.distributed.tensor.placement_types import Replicate

    from nvalchemi.distributed._core._st_backend import ShardTensorSpec

    if placements is None:
        placements = (Replicate(),) * mesh.ndim

    return ShardTensorSpec(
        mesh=mesh,
        placements=placements,
        tensor_meta=TensorMeta(
            shape=local_tensor.shape,
            stride=local_tensor.stride(),
            dtype=local_tensor.dtype,
        ),
        _local_shape=local_tensor.shape,
        _sharding_shapes=None,
    )
