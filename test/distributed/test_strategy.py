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

"""CPU unit tests for the parallelization-strategy layer (S0/S1).

The full multi-GPU step (halo exchange + migration reshard) is gated on the box;
these single-process checks lock the strategy protocol: the policy→strategy
factory, capability flags, graph-parallel no-op cell/migration semantics, and the
per-strategy reduce semantics (in particular that node-replicate reduction is the
identity, not an over-counting all_reduce).
"""

from __future__ import annotations

import torch

from nvalchemi.distributed._core.storage_policy import (
    GraphParallelPolicy,
    GraphReplicatePolicy,
    HaloStoragePolicy,
    RefreshOnlyHaloPolicy,
)
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.strategy import (
    GraphPartitionStrategy,
    GraphReplicateStrategy,
    HaloStrategy,
    MigrationPlan,
    Reduce,
    strategy_for_policy,
)

CFG = DomainConfig(cutoff=5.0)


def test_factory_maps_policy_to_strategy():
    assert isinstance(strategy_for_policy(HaloStoragePolicy(), CFG, 0), HaloStrategy)
    assert isinstance(
        strategy_for_policy(RefreshOnlyHaloPolicy(), CFG, 0), HaloStrategy
    )
    assert isinstance(
        strategy_for_policy(GraphParallelPolicy(), CFG, 0), GraphPartitionStrategy
    )
    assert isinstance(
        strategy_for_policy(GraphReplicatePolicy(), CFG, 0), GraphReplicateStrategy
    )


def test_factory_rejects_local_and_unknown():
    import pytest

    with pytest.raises(ValueError):
        strategy_for_policy(None, CFG, 0)
    with pytest.raises(ValueError):
        strategy_for_policy(object(), CFG, 0)


def test_capability_flags():
    halo = strategy_for_policy(HaloStoragePolicy(), CFG, 0)
    gpp = strategy_for_policy(GraphParallelPolicy(), CFG, 0)
    gpr = strategy_for_policy(GraphReplicatePolicy(), CFG, 0)
    assert halo.evolves_partition and halo.uses_cell_for_partition
    assert not gpp.evolves_partition and not gpp.uses_cell_for_partition
    assert not gpr.evolves_partition and not gpr.uses_cell_for_partition


def test_graph_parallel_migration_and_cell_are_noops():
    sentinel = object()
    for policy in (GraphParallelPolicy(), GraphReplicatePolicy()):
        s = strategy_for_policy(policy, CFG, 0)
        plan = s.plan_migration(None, None)
        assert not plan.is_pending
        # apply is identity (returns the same object) for a non-pending plan.
        assert s.apply_migration(None, sentinel, plan) is sentinel
        # on_cell_change never raises and does nothing observable.
        s.on_cell_change(None, torch.eye(3))


def test_halo_no_partitioner_or_dist_is_noop():
    halo = strategy_for_policy(HaloStoragePolicy(), CFG, 0)
    state = type("S", (), {"partitioner": None})()
    assert not halo.plan_migration(state, None).is_pending
    sentinel = object()
    assert halo.apply_migration(state, sentinel, MigrationPlan.none()) is sentinel


def test_reduce_semantics_single_process():
    halo = strategy_for_policy(HaloStoragePolicy(), CFG, 0)
    gpp = strategy_for_policy(GraphParallelPolicy(), CFG, 0)
    gpr = strategy_for_policy(GraphReplicatePolicy(), CFG, 0)
    # No process group initialized -> owned-shard reductions are local identity.
    assert halo.reduce_system(torch.tensor([3.0]), Reduce.SUM).item() == 3.0
    assert gpp.reduce_system(torch.tensor([3.0]), Reduce.SUM).item() == 3.0
    # Node-replicate is ALWAYS the identity (already global) — never all_reduce.
    assert gpr.reduce_system(torch.tensor([5.0]), Reduce.SUM).item() == 5.0


def test_global_atom_count():
    dev = torch.device("cpu")
    halo = strategy_for_policy(HaloStoragePolicy(), CFG, 0)
    gpr = strategy_for_policy(GraphReplicatePolicy(), CFG, 0)
    # Single-process: halo/partition sum is local; replicate is already global.
    assert halo.global_atom_count(7, dev).item() == 7
    assert gpr.global_atom_count(7, dev).item() == 7


def test_reduce_enum_ops_map():
    assert Reduce.SUM.to_op() is torch.distributed.ReduceOp.SUM
    assert Reduce.MAX.to_op() is torch.distributed.ReduceOp.MAX
    assert Reduce.MIN.to_op() is torch.distributed.ReduceOp.MIN


# ----------------------------------------------------------------------
# S4a: config-driven strategy selection (no env vars)
# ----------------------------------------------------------------------


def test_domain_config_strategy_default_and_set():
    from nvalchemi.distributed.config import DomainConfig, StrategyKind

    assert DomainConfig(cutoff=5.0).strategy is StrategyKind.HALO
    cfg = DomainConfig(cutoff=5.0, strategy=StrategyKind.GRAPH_REPLICATE)
    assert cfg.strategy is StrategyKind.GRAPH_REPLICATE
    # Accepts the string form too (str-enum), for config files.
    cfg2 = DomainConfig(cutoff=5.0, strategy="graph_partition")
    assert cfg2.strategy is StrategyKind.GRAPH_PARTITION


def test_base_distribution_spec_is_strategy_parameterized_method():
    from nvalchemi.distributed.config import StrategyKind
    from nvalchemi.models.base import BaseModelMixin

    # The base declaration is now a method taking a strategy (default None →
    # halo), returning None for a model that declares no DD support.
    m = BaseModelMixin.distribution_spec
    assert callable(m)
    # A bare object exposing the base method returns None for any strategy.
    class _Bare(BaseModelMixin):
        pass

    # Cannot instantiate the abstract mixin fully; assert the method arity via
    # the unbound function accepting a strategy kwarg without error on None self
    # is not meaningful — instead check the signature accepts `strategy`.
    import inspect

    params = inspect.signature(BaseModelMixin.distribution_spec).parameters
    assert "strategy" in params
    assert list(StrategyKind) == [
        StrategyKind.HALO,
        StrategyKind.GRAPH_REPLICATE,
        StrategyKind.GRAPH_PARTITION,
    ]
