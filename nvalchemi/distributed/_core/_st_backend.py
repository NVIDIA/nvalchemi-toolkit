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

"""Single import seam for the ShardTensor backend.

Every nvalchemi module that needs ``ShardTensor`` / ``ShardTensorSpec`` /
``scatter_tensor`` imports them from here, so the backend can be swapped in
one place. Currently re-exports the vendored copy under ``_upstream/`` (see
that package's ``README.md``).

Once physicsnemo ships the merged version in a release, replace the three
imports below with::

    from physicsnemo.domain_parallel import ShardTensor, scatter_tensor
    from physicsnemo.domain_parallel._shard_tensor_spec import ShardTensorSpec

and delete ``nvalchemi/distributed/_core/_upstream/``. Nothing else changes.
"""

from __future__ import annotations

from nvalchemi.distributed._core._upstream import (
    ShardTensor,
    ShardTensorSpec,
    scatter_tensor,
)

__all__ = ["ShardTensor", "ShardTensorSpec", "scatter_tensor"]
