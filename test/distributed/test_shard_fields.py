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

"""``DistributionSpec.shard_fields`` — the eager-DD ShardTensor-promotion set —
plus the declaration-time spec validator.

``shard_fields`` is always a concrete tuple (defaulting to
:data:`DEFAULT_SHARD_FIELDS`); there is no ``None`` sentinel, so ``()`` ("promote
nothing", e.g. UMA's plain interior) can never collapse to the default under
truthiness. The validator rejects structurally-broken specs at construction.
"""

from __future__ import annotations

import json

import pytest

from nvalchemi.distributed._core.spec import DEFAULT_SHARD_FIELDS, DistributionSpec
from nvalchemi.distributed._core.storage_policy import PlainShard
from nvalchemi.distributed.ops import HaloStoragePolicy
from nvalchemi.distributed.spec import MLIPSpec


def test_validator_rejects_non_policy_default():
    with pytest.raises(TypeError, match="StoragePolicy"):
        DistributionSpec(policy="not-a-policy")


def test_validator_accepts_the_shipped_policies():
    # Both shipped policies pass the structural check.
    DistributionSpec(policy=HaloStoragePolicy())
    DistributionSpec(policy=PlainShard())
    DistributionSpec(policy=None)  # local / single-process


def test_shard_fields_defaults_to_standard_set_and_round_trips():
    # Default: a spec declares nothing -> the standard MLIP promotion set; the
    # serialized form gains no new key (back-compatible with older on-disk specs).
    d = DistributionSpec(policy=HaloStoragePolicy())
    assert d.shard_fields == DEFAULT_SHARD_FIELDS
    assert "shard_fields" not in d.to_dict()
    # Declared: narrows the promotion set and survives a JSON round-trip.
    nd = DistributionSpec(policy=HaloStoragePolicy(), shard_fields=("positions",))
    assert nd.to_dict()["shard_fields"] == ["positions"]
    restored = DistributionSpec.from_dict(json.loads(json.dumps(nd.to_dict())))
    assert restored.shard_fields == ("positions",)


def test_shard_fields_preserved_through_with_adapters():
    # with_adapters reconstructs DistributionSpec; shard_fields must carry through
    # so a narrowed model keeps it after attaching adapters.
    base = MLIPSpec(
        distribution=DistributionSpec(
            policy=HaloStoragePolicy(), shard_fields=("positions",)
        )
    )
    assert base.with_adapters().distribution.shard_fields == ("positions",)


def test_shard_fields_empty_tuple_promotes_nothing():
    # The empty tuple means "promote NOTHING" (UMA, plain-interior) and is a
    # distinct, first-class value — not the default. Because the field is always a
    # concrete tuple (never None), there is no falsy ``or default`` trap that could
    # collapse () to the default and wrongly promote everything.
    d = DistributionSpec(policy=HaloStoragePolicy(), shard_fields=())
    assert d.shard_fields == ()
    assert d.shard_fields != DEFAULT_SHARD_FIELDS
    assert d.to_dict()["shard_fields"] == []
    restored = DistributionSpec.from_dict(json.loads(json.dumps(d.to_dict())))
    assert restored.shard_fields == ()


def test_to_local_and_localize_are_identity_on_plain():
    import torch

    from nvalchemi.distributed.helpers import localize, to_local

    t = torch.zeros(3, 2)
    assert to_local(t) is t  # plain tensor: unchanged
    assert to_local(None) is None
    assert to_local(5) == 5  # non-tensor: unchanged
    out = localize({"a": t, "n": 7, "z": None})
    assert out["a"] is t and out["n"] == 7 and out["z"] is None
