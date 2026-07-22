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

"""Vendored copy of physicsnemo's ``domain_parallel`` ShardTensor backend.

This is a **temporary internal copy** of the ``torch.compile``-enabled
``ShardTensor`` from physicsnemo PRs #1556 (``sharded_view_backwards``) +
#1682 (``shard_tensor_compile``), which are unmerged and deferred past the
26.05 release on an FSDP1/StormScope blocker unrelated to MLIP inference.

Provenance, the kept/dropped file list, and the re-sync recipe live in
``README.md``. Import rewrites are applied by ``resync.sh``. The vendored
files are otherwise byte-for-byte upstream; the only hand-edits carry a
``VENDOR-EDIT`` marker (the cuda-gate in ``domain_parallel/__init__.py`` and
the trimmed registration in ``shard_utils/__init__.py``).

Retirement: when physicsnemo ships the merged version in a release, repoint
``nvalchemi.distributed._core._st_backend`` at ``physicsnemo.domain_parallel``
and delete this package. Nothing else in the tree imports it directly.
"""

from __future__ import annotations

from .domain_parallel import (
    FSDPOutputTensorAdapter,
    ShardTensor,
    ShardTensorSpec,
    distribute_over_domain_for_fsdp,
    scatter_tensor,
    wrap_for_fsdp,
)

__all__ = [
    "ShardTensor",
    "ShardTensorSpec",
    "scatter_tensor",
    "FSDPOutputTensorAdapter",
    "wrap_for_fsdp",
    "distribute_over_domain_for_fsdp",
]
