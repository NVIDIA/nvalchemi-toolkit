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

"""Contract tests for the gloo-harness ShardTensor stand-ins in
``conftest.py``.

``DistributedModel._call_sharded_storage`` reads
``positions._spec.sharding_shapes()[0]`` to derive the per-rank sizes it
feeds ``_all_gather_v_rows``. The :class:`_LocalShardTensor` double must
honour that exact contract so the sharded path can be exercised under
gloo without a real ShardTensor (and without an installed model).
"""

from __future__ import annotations

import torch
from _helpers import _LocalShardTensor


def test_local_shard_tensor_spec_matches_sharded_read_contract() -> None:
    # Rank holds 3 rows of a (N, 3) field; the global tensor is split 3/4.
    sizes = [3, 4]
    local = torch.zeros(3, 3)
    st = _LocalShardTensor(local, sizes=sizes)

    shapes = st._spec.sharding_shapes()
    # Mirror DistributedModel._call_sharded_storage's exact read.
    rank_sizes = [int(s[0]) for s in shapes[0]]
    assert rank_sizes == sizes
    # Trailing (feature) dims are preserved per rank.
    assert tuple(shapes[0][0][1:]) == (3,)
    assert len(shapes[0]) == len(sizes)


def test_local_shard_tensor_to_local_is_the_owned_slice() -> None:
    local = torch.arange(6).reshape(3, 2)
    st = _LocalShardTensor(local, sizes=[3, 3])
    assert torch.equal(st.to_local(), local)
