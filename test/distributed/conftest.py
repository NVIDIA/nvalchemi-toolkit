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
"""Session-scoped gloo fixture for distributed tests.

Importable helper classes (``_MockMesh`` / ``_LocalShardTensor`` /
``make_gloo_sharded_batch``) live in ``_helpers.py`` so subfolder test
packages can import them; this file holds only the pytest fixture."""

from __future__ import annotations

import pytest
import torch.distributed as dist

# ----------------------------------------------------------------------
# Function-scoped 1-rank gloo init for tests that construct a ShardTensor
# without an explicit dist setup.
#
# ShardTensor construction requires a real DeviceMesh, which in turn
# requires a process group. Tests that wrap plain tensors for dispatch
# unit-testing (test_halo_tensor.py, test_registry_and_contexts.py,
# test_compile_smoke.py, test_escape_hatches.py) get a default 1-rank
# gloo group via this fixture.
#
# mp.spawn-based tests (test_distributed_all_reduce.py,
# test_dispatch_trace_gloo.py, etc.) fork their own workers that set
# MASTER_ADDR/MASTER_PORT and call init_process_group independently —
# those workers don't inherit this state, so they don't collide with
# our 1-rank default.
# ----------------------------------------------------------------------


@pytest.fixture
def _session_gloo_pg():
    """Opt-in 1-rank gloo process group + default DeviceMesh for tests
    that construct :class:`ShardTensor` instances without an explicit
    distributed setup.

    Function-scoped (not session-scoped): a session-scoped group stays
    initialized for the rest of the run once any test pulls it, which
    leaks ``dist.is_initialized()`` into unrelated later tests (e.g. the
    pipeline-composition guards under ``test/dynamics``). Per-test
    init/teardown keeps each test isolated. Not autouse: empirically an
    autouse group interferes with ``mp.spawn``-based tests under
    ``test_validate_cuda.py`` (the parent's gloo group + spawned NCCL
    workers conflict in PyTorch's distributed state). Tests that need the
    default mesh pull this fixture by name; tests that do their own dist
    setup (mp.spawn / torchrun harnesses) ignore it.

    Uses ``init_method`` directly rather than ``MASTER_ADDR`` /
    ``MASTER_PORT`` env vars — env-var-based init pollutes the
    process-wide environment, which child processes inherit.

    Yields the constructed :class:`DeviceMesh` so tests that want it
    explicitly can pull it; tests that just need ``ShardTensor.wrap``
    to find a current mesh can pull the fixture for its side effect
    (constructed mesh registers with ``_mesh_resources``).
    """
    we_initialized = False
    if not dist.is_initialized():
        dist.init_process_group(
            backend="gloo",
            init_method="tcp://127.0.0.1:29612",
            rank=0,
            world_size=1,
        )
        we_initialized = True
    from torch.distributed.device_mesh import DeviceMesh

    mesh = DeviceMesh("cpu", [0], mesh_dim_names=("dom",))
    # Enter the mesh as a context so ``_mesh_resources.get_current_mesh()``
    # returns it — that's what ``ShardTensor.wrap()`` consults when no
    # explicit mesh is provided.
    with mesh:
        yield mesh
    if we_initialized and dist.is_initialized():
        dist.destroy_process_group()
