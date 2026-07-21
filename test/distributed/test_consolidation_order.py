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

"""Regression: output consolidation must issue its per-key collectives in a
rank-independent order (NVBUGS 6472536).

Consolidation issues one collective per output key in dict-iteration order —
``distributed_all_reduce`` for per-graph outputs (energy/stress) and
``halo_reverse_exchange`` (all-to-all) for per-atom autograd outputs (forces).
If the output dict is seeded from a ``set`` (e.g. ``active_outputs``), its key
order is randomized per process, so ranks issue those collectives in different
orders → mismatched NCCL ops at the same seq → deadlock. Consolidation must
therefore iterate a sorted key order regardless of the input dict's order.

All keys below take the passthrough branch, so no process group is needed —
the test asserts the *iteration/emit order*, which is what pins the schedule.
"""

from __future__ import annotations

from collections import OrderedDict
from types import SimpleNamespace

import torch

from nvalchemi.distributed.output_consolidation import consolidate_sharded_outputs


def test_consolidate_sharded_emits_sorted_key_order() -> None:
    # Deliberately-unsorted insertion order, mimicking set-hash randomization.
    output = OrderedDict()
    for key in ("stress", "energy", "forces", "charges"):
        output[key] = torch.zeros(1)

    # Only ``autograd_outputs`` is read; empty owned_only/all_reduce sets mean
    # every key is passthrough (no collective fires).
    model_config = SimpleNamespace(autograd_outputs=frozenset())

    reduced = consolidate_sharded_outputs(output, model_config, world_size=1)

    assert list(reduced.keys()) == sorted(output.keys())
