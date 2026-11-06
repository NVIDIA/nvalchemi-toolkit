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
"""A loader-compiled wrapper whose spec has no ``CompilePolicy`` is rejected clearly."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.distributed_model import (
    DistributedModel,
    DistributionError,
)
from nvalchemi.distributed.ops import HaloStoragePolicy
from nvalchemi.distributed.spec import (
    DistributionSpec,
    MLIPSpec,
    OutputKind,
    OutputSpec,
    Reduce,
)


class _PolicylessWrapper(nn.Module):
    """A wrapper whose distribution spec declares no ``CompilePolicy``."""

    def __init__(self, *, precompiled: bool) -> None:
        super().__init__()
        inner = nn.Linear(3, 1)
        self.model = torch.compile(inner) if precompiled else inner

    def distribution_spec(self, strategy=None) -> MLIPSpec:
        return MLIPSpec(
            distribution=DistributionSpec(policy=HaloStoragePolicy(), shard_fields=()),
            outputs={"energy": OutputSpec(OutputKind.PER_GRAPH, Reduce.ALL_REDUCE)},
        )


def _config() -> DomainConfig:
    return DomainConfig(cutoff=5.0)


def test_precompiled_wrapper_without_policy_is_rejected_with_usable_advice():
    with pytest.raises(DistributionError) as excinfo:
        DistributedModel(_PolicylessWrapper(precompiled=True), _config())
    message = str(excinfo.value)
    assert "already torch.compiled" in message
    assert "WITHOUT compiling it" in message
    # ``compile=True`` is not offered as a way out: it needs the same policy.
    assert "also requires a CompilePolicy" in message


def test_uncompiled_wrapper_without_policy_is_accepted_eagerly():
    """The advised route, building the wrapper without compiling it, works."""
    DistributedModel(_PolicylessWrapper(precompiled=False), _config())


def test_compile_true_without_policy_raises_the_same_requirement():
    """Why ``compile=True`` cannot be the suggested alternative."""
    with pytest.raises(
        DistributionError, match="requires the model's distribution_spec"
    ):
        DistributedModel(_PolicylessWrapper(precompiled=False), _config(), compile=True)
