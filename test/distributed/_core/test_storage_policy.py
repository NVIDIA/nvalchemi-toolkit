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

"""Unit tests for the storage-policy spec model.

These mock the behaviors we expect in practice for the paths that can't be
validated end-to-end without a multi-rank machine — especially the UMA
``scatter="local"`` override (``replace_policy``) and the policy
serialization round-trips.
"""

from __future__ import annotations

import pytest

from nvalchemi.distributed._core.spec import DistributionSpec
from nvalchemi.distributed._core.storage_policy import (
    HaloStoragePolicy,
    PlainShard,
    policy_from_dict,
    policy_to_dict,
)
from nvalchemi.distributed.spec import (
    SPEC_MPNN_HALO,
    MLIPSpec,
    replace_policy,
)

# --------------------------------------------------------------------------
# Policy serialization round-trips (None = local).
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "policy",
    [
        None,
        HaloStoragePolicy(),
        HaloStoragePolicy(scatter_mode="local", gather_mode="local"),
        PlainShard(),
    ],
)
def test_policy_dict_round_trip(policy):
    assert policy_from_dict(policy_to_dict(policy)) == policy


def test_policy_to_dict_halo_carries_modes():
    d = policy_to_dict(HaloStoragePolicy(scatter_mode="local", gather_mode="halo_read"))
    assert d == {"kind": "halo", "scatter_mode": "local", "gather_mode": "halo_read"}


# --------------------------------------------------------------------------
# Preset specs carry the expected policy + system_reductions field.
# --------------------------------------------------------------------------


def test_preset_specs_carry_expected_policy():
    halo = SPEC_MPNN_HALO.distribution.policy
    assert isinstance(halo, HaloStoragePolicy)
    assert halo.scatter_mode == "halo_correction"
    assert halo.gather_mode == "halo_read"
    # system_reductions is a first-class MLIPSpec field.
    assert SPEC_MPNN_HALO.system_reductions is True


# --------------------------------------------------------------------------
# replace_policy: the UMA wrapper-level scatter="local" override.
# --------------------------------------------------------------------------


def test_replace_policy_maps_scatter_to_scatter_mode():
    # UMA's wrapper does ``replace_policy(spec, scatter="local")`` to skip
    # halo correction on its halo-unaware backbone.
    overridden = replace_policy(SPEC_MPNN_HALO, scatter="local")
    assert isinstance(overridden.distribution.policy, HaloStoragePolicy)
    assert overridden.distribution.policy.scatter_mode == "local"
    # The original spec is unchanged (frozen dataclasses, functional replace).
    assert SPEC_MPNN_HALO.distribution.policy.scatter_mode == "halo_correction"


def test_replace_policy_maps_gather_to_gather_mode():
    overridden = replace_policy(SPEC_MPNN_HALO, gather="local")
    assert overridden.distribution.policy.gather_mode == "local"


def test_replace_policy_on_local_policy_raises():
    local_spec = MLIPSpec(distribution=DistributionSpec(policy=None))
    with pytest.raises(ValueError, match="no storage policy"):
        replace_policy(local_spec, scatter="local")


# --------------------------------------------------------------------------
# Spec merge over policies.
# --------------------------------------------------------------------------


def test_merge_two_halo_specs_takes_more_permissive_modes():
    local_scatter = replace_policy(SPEC_MPNN_HALO, scatter="local")
    merged = local_scatter.merge(SPEC_MPNN_HALO)
    # halo_correction is more permissive than local.
    assert merged.distribution.policy.scatter_mode == "halo_correction"




def test_merge_system_reductions_is_or():
    a = MLIPSpec(distribution=DistributionSpec(policy=HaloStoragePolicy()), system_reductions=False)
    b = MLIPSpec(distribution=DistributionSpec(policy=HaloStoragePolicy()), system_reductions=True)
    assert a.merge(b).system_reductions is True
