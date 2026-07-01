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

"""``nvalchemi.distributed``'s public API surface stays stable.

The names here are documented user-facing entry points. Users construct
:class:`DomainConfig` instances, wrap models with :class:`DomainParallel`,
and reach for :class:`ShardedBatch` / :class:`SpatialPartitioner` when
plumbing custom dynamics integrators. Hiding any of these behind an
internal namespace silently breaks downstream code; deleting one without
a deprecation cycle does too.

This test asserts the canonical names resolve to objects (lazy or
eager). It is *the* stability boundary — if you intentionally rename
or remove a name, update :data:`EXPECTED_PUBLIC_NAMES`.
"""

from __future__ import annotations

import importlib

import pytest

EXPECTED_PUBLIC_NAMES: frozenset[str] = frozenset(
    {
        # Runtime entry points.
        "DomainConfig",
        "DomainParallel",
        "HookScope",
        "ParticleHaloConfig",
        "ShardedBatch",
        "SpatialPartitioner",
        "autograd_target",
        "reshard_by_destination",
        # Top layer — declarative spec types a model author names in a wrapper's
        # ``distribution_spec``.
        "AdapterRegistry",
        "AdapterStatus",
        "CompilePolicy",
        "ForceStrategy",
        "DistributionSpec",
        "FunctionAdapter",
        "GraphPadder",
        "COOPadder",
        "DensePadder",
        "DenseBatchPadder",
        "resolve_cap",
        "JitAdapter",
        "MLIPSpec",
        "MethodAdapter",
        "OpAdapter",
        "OutputKind",
        "OutputSpec",
        "PythonAdapter",
        "Reduce",
        "trace_and_validate",
        # Middle layer — intent vocabulary, re-exported here for convenience
        # (canonical home: ``nvalchemi.distributed.helpers``).
        "Scope",
        "current_dd_context",
        "neighbor_refresh_adapters",
        "refresh_neighbors",
        "scatter_to_owners",
        "system_sum",
        "to_local",
        "localize",
        "distributed_method",
        # DDP training-runtime helpers (recommended manager + rank/world/device
        # resolvers), folded in from the former top-level ``distributed.py``.
        "DistributedManager",
        "PhysicsNeMoUninitializedDistributedManagerWarning",
        "collective_device",
        "resolve_global_rank",
        "resolve_world_size",
    }
)

# The bottom layer (``nvalchemi.distributed.ops``) — communication mechanism
# only. The intent vocabulary (``refresh_neighbors`` / ``system_sum`` / …) lives
# one layer up in ``nvalchemi.distributed.helpers``; the declarative spec types
# (adapters / ``DistributionSpec`` / ``GraphPadder`` family) at the top in
# ``nvalchemi.distributed``. This module never re-exports those upward layers —
# it is the stability boundary for the raw primitives.
EXPECTED_OPS_NAMES: frozenset[str] = frozenset(
    {
        # halo exchange — eager
        "halo_forward_exchange",
        "halo_reverse_exchange",
        "particle_halo_padding_autograd",
        "pad_field",
        # halo — compile / fixed-shape static ops
        "halo_forward_static_op",
        "halo_scatter_correct_static_op",
        "halo_forward_static_from_meta",
        "halo_scatter_correct_static_from_meta",
        "build_halo_meta_tensors",
        "pack_halo_meta",
        "unpack_halo_meta",
        # DD context accessor + object
        "current_dd_context",
        "activate_dd_context",
        "NOT_DISTRIBUTED",
        "DistributedContext",
        # per-system reduce + collectives
        "per_system_reduce",
        "per_system_reduce_op",
        "distributed_all_reduce",
        "mesh_group",
        # low-level op transforms (explicit form behind OpAdapter's role kwargs)
        "GatherInputs",
        "GatherInputsFull",
        "SliceOwned",
        "ScatterOutputs",
        "AllReduceSum",
        "SliceOutputsOwned",
        # storage policies
        "StoragePolicy",
        "HaloStoragePolicy",
        # routing / metadata
        "ParticleHaloConfig",
        "ParticleHaloMetadata",
        "GNNHaloMarkers",
        # distributed tensor
        "ShardTensor",
    }
)


def test_public_api_resolves() -> None:
    """Every name in :data:`EXPECTED_PUBLIC_NAMES` resolves via
    ``nvalchemi.distributed``'s lazy ``__getattr__``."""
    mod = importlib.import_module("nvalchemi.distributed")
    for name in EXPECTED_PUBLIC_NAMES:
        obj = getattr(mod, name, None)
        assert obj is not None, (
            f"nvalchemi.distributed.{name} did not resolve. The lazy "
            f"``__getattr__`` may be missing this entry, or the underlying "
            f"module path was refactored without updating the import map."
        )


def test_public_api_dunder_all_matches() -> None:
    """``__all__`` is the documented public surface; it must match
    :data:`EXPECTED_PUBLIC_NAMES` exactly. Drift in either direction is
    a regression."""
    mod = importlib.import_module("nvalchemi.distributed")
    declared = set(mod.__all__)
    assert declared == set(EXPECTED_PUBLIC_NAMES), (
        f"nvalchemi.distributed.__all__ = {sorted(declared)} "
        f"diverges from EXPECTED_PUBLIC_NAMES = "
        f"{sorted(EXPECTED_PUBLIC_NAMES)}. Update both together if "
        f"renaming the public surface."
    )


@pytest.mark.parametrize("name", sorted(EXPECTED_PUBLIC_NAMES))
def test_attribute_error_on_unknown_name(name: str) -> None:
    """Sanity: known names resolve; unknown names raise AttributeError
    (verifies the lazy __getattr__'s error path)."""
    mod = importlib.import_module("nvalchemi.distributed")
    # The known name resolves.
    assert getattr(mod, name) is not None
    # An obvious typo raises.
    with pytest.raises(AttributeError):
        _ = getattr(mod, "NonExistent_NoSuchSymbol")


# ----------------------------------------------------------------------
# Power-user toolbox: nvalchemi.distributed.ops.
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED_OPS_NAMES))
def test_ops_symbol_imports(name: str) -> None:
    """Every promoted primitive imports from ``nvalchemi.distributed.ops``.

    The load-bearing distributed primitives are reachable on a public
    path, so an external author never has to import
    ``nvalchemi.distributed._core``."""
    ops = importlib.import_module("nvalchemi.distributed.ops")
    obj = getattr(ops, name, None)
    assert obj is not None, (
        f"nvalchemi.distributed.ops.{name} did not import. The re-export "
        f"may be missing, or the underlying ``_core`` symbol was renamed "
        f"without updating ops.py."
    )


def test_ops_dunder_all_matches() -> None:
    """``ops.__all__`` is the documented ops surface; it must match
    :data:`EXPECTED_OPS_NAMES` exactly."""
    ops = importlib.import_module("nvalchemi.distributed.ops")
    declared = set(ops.__all__)
    assert declared == set(EXPECTED_OPS_NAMES), (
        f"nvalchemi.distributed.ops.__all__ = {sorted(declared)} "
        f"diverges from EXPECTED_OPS_NAMES = {sorted(EXPECTED_OPS_NAMES)}. "
        f"Update both together if changing the ops toolbox."
    )


def test_ops_star_import_is_clean() -> None:
    """``from nvalchemi.distributed.ops import *`` exposes exactly
    ``__all__`` — no private leakage, no missing name."""
    ns: dict[str, object] = {}
    exec("from nvalchemi.distributed.ops import *", ns)  # noqa: S102
    exported = {k for k in ns if not k.startswith("__")}
    assert exported == set(EXPECTED_OPS_NAMES)


def test_no_core_import_needed_for_byo() -> None:
    """An author can pull adapters, the spec types, halo primitives, and a
    storage policy from the *public* surface alone — zero ``_core``
    imports."""
    from nvalchemi.distributed import (  # noqa: F401  — public
        DistributionSpec,
        MLIPSpec,
        OpAdapter,
    )
    from nvalchemi.distributed.ops import (  # noqa: F401  — public
        HaloStoragePolicy,
        ShardTensor,
        halo_forward_exchange,
        per_system_reduce,
    )
