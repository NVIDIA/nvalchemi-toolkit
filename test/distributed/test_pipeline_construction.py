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
"""``DistributedPipelineModel`` builds without a process group.

Every other pipeline test needs 2+ GPUs, so nothing exercised group planning on
a workstation — and planning is where each sub-model's persistent
``DistributedModel`` is constructed. A missing import there raises ``NameError``
at construction, which lint cannot catch because ``F821`` is in this repo's ruff
ignore list, and which no local test would reach.

These run on CPU in milliseconds and cover the import graph of the construction
path, not its numerics.
"""

from __future__ import annotations

import pytest

from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.distributed_pipeline import DistributedPipelineModel
from nvalchemi.models.lj import LennardJonesModelWrapper


def _pipeline():
    """Two cheap kernel-force models, so the group plans per-step."""
    outer = LennardJonesModelWrapper(epsilon=0.0103, sigma=3.4, cutoff=6.0)
    inner = LennardJonesModelWrapper(epsilon=0.0103, sigma=3.4, cutoff=4.0)
    return outer + inner


class TestPipelineConstruction:
    def test_plans_its_groups_without_a_mesh(self) -> None:
        with DistributedPipelineModel(_pipeline(), DomainConfig(cutoff=6.0)) as dpm:
            assert dpm._group_plans
            assert all(p["kind"] in ("per_step", "wired") for p in dpm._group_plans)

    def test_per_step_group_holds_a_persistent_model(self) -> None:
        with DistributedPipelineModel(_pipeline(), DomainConfig(cutoff=6.0)) as dpm:
            for plan in dpm._group_plans:
                for item in plan.get("steps", ()):
                    assert item["dm"] is not None

    def test_close_is_idempotent(self) -> None:
        dpm = DistributedPipelineModel(_pipeline(), DomainConfig(cutoff=6.0))
        dpm.close()
        dpm.close()

    def test_rejects_a_bare_model(self) -> None:
        with pytest.raises(TypeError, match="PipelineModelWrapper"):
            DistributedPipelineModel(
                LennardJonesModelWrapper(epsilon=0.0103, sigma=3.4, cutoff=6.0),
                DomainConfig(cutoff=6.0),
            )
