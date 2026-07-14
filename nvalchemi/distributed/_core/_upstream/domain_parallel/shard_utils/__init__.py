# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from physicsnemo.core.version_check import check_version_spec

# Prevent importing this module if the minimum version of pytorch is not met.
ST_AVAILABLE = check_version_spec("torch", "2.6.0a0", hard_fail=False)

if ST_AVAILABLE:
    from ..shard_tensor import ShardTensor

    def register_shard_wrappers():
        """Import and register all shard-aware operation wrappers with ShardTensor.

        Each imported module registers its wrapper via
        :meth:`ShardTensor.register_op` at import time.
        """
        # VENDOR-EDIT (nvalchemi): grid/CFD wrappers are not vendored —
        # attention_patches, conv_patches, knn, mesh_ops, natten_patches,
        # padding, point_cloud_ops, pooling_patches, unpooling_patches. Only the
        # MLIP-relevant wrappers are registered. See
        # proposal-distributed-compile-vendoring.md §4.
        from .index_ops import (  # noqa: F401
            index_select_wrapper,
            sharded_select_backward_helper,
            sharded_select_helper,
        )
        from .normalization_patches import group_norm_wrapper  # noqa: F401
        from .unary_ops import unsqueeze_wrapper  # noqa: F401
        from .view_ops import reshape_wrapper, view_wrapper  # noqa: F401
