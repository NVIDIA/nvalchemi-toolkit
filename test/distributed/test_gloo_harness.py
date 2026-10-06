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

"""Regression tests for the reusable Gloo process-group harness."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch
import torch.distributed as dist
from _gloo_harness import run_gloo


def _all_reduce_worker(
    rank: int, world_size: int, queue: Any, group_value: int
) -> None:
    assert dist.get_world_size() == world_size
    value = torch.tensor([group_value + rank], dtype=torch.int64)
    dist.all_reduce(value)
    queue.put((rank, int(value.item())))


def test_concurrent_gloo_groups_keep_collectives_separate() -> None:
    with ThreadPoolExecutor(max_workers=2) as executor:
        group_a = executor.submit(
            run_gloo, world_size=2, fn=_all_reduce_worker, args=(100,)
        )
        group_b = executor.submit(
            run_gloo, world_size=2, fn=_all_reduce_worker, args=(1000,)
        )
        results_a = group_a.result()
        results_b = group_b.result()

    assert sorted(results_a) == [(0, 201), (1, 201)]
    assert sorted(results_b) == [(0, 2001), (1, 2001)]
