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

"""Spatial domain decomposition for distributed molecular dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch


def _register_dynamo_subclass() -> None:
    """Register :class:`ShardTensor` so Dynamo recognises it as a tensor.

    Adds our subclass to :data:`torch._dynamo.config.traceable_tensor_subclasses`
    so :func:`torch._dynamo.utils.istensor` returns True for ShardTensor
    instances. Required for ``torch.compile`` to trace through models
    that receive ShardTensor inputs (the ``_promote_positions_to_shardtensor``
    path in :class:`DistributedModel`).

    Done at module import — eager-only callers pay the cost of importing
    the dynamo config module (cheap; it's already loaded in any
    torch-using process).
    """
    try:
        import torch._dynamo.config as _dynamo_config
    except ImportError:  # pragma: no cover — torch without dynamo is rare
        return
    from nvalchemi.distributed._core.shard_tensor import ShardTensor as _ShardTensor

    _dynamo_config.traceable_tensor_subclasses.add(_ShardTensor)


_register_dynamo_subclass()


if TYPE_CHECKING:
    from nvalchemi.distributed._core.particle_halo import (
        ParticleHaloConfig as ParticleHaloConfig,
    )
    from nvalchemi.distributed._core.reshard import (
        reshard_by_destination as reshard_by_destination,
    )
    from nvalchemi.distributed.config import (
        DomainConfig as DomainConfig,
    )
    from nvalchemi.distributed.config import (
        HookScope as HookScope,
    )
    from nvalchemi.distributed.domain_parallel import DomainParallel as DomainParallel
    from nvalchemi.distributed.partitioner import (
        SpatialPartitioner as SpatialPartitioner,
    )
    from nvalchemi.distributed.sharded_batch import ShardedBatch as ShardedBatch


def autograd_target(t: torch.Tensor) -> torch.Tensor:
    """Return the tensor to pass as :func:`torch.autograd.grad`'s ``inputs=``.

    Under domain decomposition the framework wraps ``data.positions`` (and
    ``data.charges``) as a :class:`ShardTensor` view of a halo-padded leaf via
    :meth:`Tensor.as_subclass`. The view is *not* itself in the autograd graph —
    only the underlying tensor is — so passing the view directly to
    :func:`torch.autograd.grad` raises "differentiated Tensors appears to not
    have been used in the graph". This helper returns the in-graph leaf instead.

    Call this once where the autograd target is set up (e.g. in ``adapt_input``);
    the wrapper stays distribution-unaware otherwise.

    Parameters
    ----------
    t : torch.Tensor
        The tensor to differentiate against — a plain tensor, or a ShardTensor
        view of a halo-padded leaf.

    Returns
    -------
    torch.Tensor
        The underlying ``(n_padded, *F)`` leaf when ``t`` is a ShardTensor with a
        captured autograd source (the halo-padded positions case); ``t`` itself
        otherwise (single-process, where ``t`` is already the right target).
    """
    target_method = getattr(t, "autograd_target", None)
    if callable(target_method):
        return target_method()
    return t


def __getattr__(name: str):  # noqa: ANN201
    """Lazy-import public symbols on first access."""
    _imports = {
        "DomainConfig": ("nvalchemi.distributed.config", "DomainConfig"),
        "HookScope": ("nvalchemi.distributed.config", "HookScope"),
        "SpatialPartitioner": (
            "nvalchemi.distributed.partitioner",
            "SpatialPartitioner",
        ),
        "DomainParallel": ("nvalchemi.distributed.domain_parallel", "DomainParallel"),
        "ShardedBatch": ("nvalchemi.distributed.sharded_batch", "ShardedBatch"),
        "ParticleHaloConfig": (
            "nvalchemi.distributed._core.particle_halo",
            "ParticleHaloConfig",
        ),
        "reshard_by_destination": (
            "nvalchemi.distributed._core.reshard",
            "reshard_by_destination",
        ),
        # Declarative spec types named in a wrapper's ``distribution_spec``. The
        # intent vocabulary an adapter body calls lives in
        # ``nvalchemi.distributed.helpers``; the communication primitives in
        # ``nvalchemi.distributed.ops``.
        "MLIPSpec": ("nvalchemi.distributed.spec", "MLIPSpec"),
        "DistributionSpec": ("nvalchemi.distributed.spec", "DistributionSpec"),
        "OpAdapter": ("nvalchemi.distributed.spec", "OpAdapter"),
        "MethodAdapter": ("nvalchemi.distributed.spec", "MethodAdapter"),
        "FunctionAdapter": ("nvalchemi.distributed.spec", "FunctionAdapter"),
        "PythonAdapter": ("nvalchemi.distributed.spec", "PythonAdapter"),
        "JitAdapter": ("nvalchemi.distributed.spec", "JitAdapter"),
        "AdapterRegistry": ("nvalchemi.distributed.spec", "AdapterRegistry"),
        "AdapterStatus": ("nvalchemi.distributed._core.adapter", "AdapterStatus"),
        "OutputKind": ("nvalchemi.distributed.output_kinds", "OutputKind"),
        "OutputSpec": ("nvalchemi.distributed.output_kinds", "OutputSpec"),
        "Reduce": ("nvalchemi.distributed.output_kinds", "Reduce"),
        "CompilePolicy": ("nvalchemi.distributed.spec", "CompilePolicy"),
        "ForceStrategy": ("nvalchemi.distributed.spec", "ForceStrategy"),
        "GraphPadder": ("nvalchemi.distributed.graph_padder", "GraphPadder"),
        "COOPadder": ("nvalchemi.distributed.graph_padder", "COOPadder"),
        "DensePadder": ("nvalchemi.distributed.graph_padder", "DensePadder"),
        "DenseBatchPadder": (
            "nvalchemi.distributed.graph_padder",
            "DenseBatchPadder",
        ),
        "resolve_cap": ("nvalchemi.distributed.graph_padder", "resolve_cap"),
        "trace_and_validate": (
            "nvalchemi.distributed.validate",
            "trace_and_validate",
        ),
        # Intent vocabulary an adapter body / wrapper calls, re-exported here for
        # convenience (canonical home: ``nvalchemi.distributed.helpers``;
        # mechanism in ``nvalchemi.distributed.ops``).
        "current_dd_context": (
            "nvalchemi.distributed._core.context",
            "current_dd_context",
        ),
        "neighbor_refresh_adapters": (
            "nvalchemi.distributed.helpers",
            "neighbor_refresh_adapters",
        ),
        "refresh_neighbors": ("nvalchemi.distributed.helpers", "refresh_neighbors"),
        "scatter_to_owners": ("nvalchemi.distributed.helpers", "scatter_to_owners"),
        "system_sum": ("nvalchemi.distributed.helpers", "system_sum"),
        "to_local": ("nvalchemi.distributed.helpers", "to_local"),
        "localize": ("nvalchemi.distributed.helpers", "localize"),
        "distributed_method": ("nvalchemi.distributed.helpers", "distributed_method"),
        "Scope": ("nvalchemi.distributed._core.enums", "Scope"),
        # DDP training-runtime helpers (recommended manager + rank/world/device
        # resolvers). Folded in from the former top-level distributed.py module.
        "DistributedManager": (
            "nvalchemi.distributed._runtime",
            "DistributedManager",
        ),
        "PhysicsNeMoUninitializedDistributedManagerWarning": (
            "nvalchemi.distributed._runtime",
            "PhysicsNeMoUninitializedDistributedManagerWarning",
        ),
        "resolve_world_size": (
            "nvalchemi.distributed._runtime",
            "resolve_world_size",
        ),
        "resolve_global_rank": (
            "nvalchemi.distributed._runtime",
            "resolve_global_rank",
        ),
        "collective_device": (
            "nvalchemi.distributed._runtime",
            "collective_device",
        ),
    }
    if name in _imports:
        module_path, attr = _imports[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AdapterRegistry",
    "AdapterStatus",
    "CompilePolicy",
    "ForceStrategy",
    "DistributionSpec",
    "DomainConfig",
    "DomainParallel",
    "HookScope",
    "JitAdapter",
    "FunctionAdapter",
    "GraphPadder",
    "COOPadder",
    "DensePadder",
    "DenseBatchPadder",
    "resolve_cap",
    "MLIPSpec",
    "MethodAdapter",
    "OpAdapter",
    "OutputKind",
    "OutputSpec",
    "ParticleHaloConfig",
    "PythonAdapter",
    "Reduce",
    "Scope",
    "ShardedBatch",
    "SpatialPartitioner",
    "autograd_target",
    "collective_device",
    "current_dd_context",
    "DistributedManager",
    "PhysicsNeMoUninitializedDistributedManagerWarning",
    "resolve_global_rank",
    "resolve_world_size",
    "neighbor_refresh_adapters",
    "distributed_method",
    "localize",
    "refresh_neighbors",
    "reshard_by_destination",
    "scatter_to_owners",
    "system_sum",
    "to_local",
    "trace_and_validate",
]
