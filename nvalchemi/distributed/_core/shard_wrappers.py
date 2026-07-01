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
"""Generic ShardTensor dispatch wrappers for opaque custom ops.

Provides the domain-neutral machinery for routing ``@torch.library.custom_op``
kernels over :class:`ShardTensor` inputs: the kernel sees plain local tensors
(via ``.to_local()``), and the result is re-wrapped as a ``ShardTensor`` with
the appropriate placement.

Two wrapper shapes:

- **Passthrough** (:func:`make_passthrough_wrapper`): per-row ops where each
  rank's shard is processed independently. In-place mutations persist because
  ``to_local()`` returns the backing ``_local_tensor`` directly (not a copy).

- **Reduction** (:func:`make_reduction_wrapper`): per-row → per-group
  reductions. Output gets ``Partial(reduce_op)`` placement so the all-reduce
  happens lazily when the result is consumed via ``.full_tensor()`` /
  ``.redistribute()``.

Ops that need cross-rank neighbor data (the halo category) are NOT handled
here — those use ``particle_halo_padding`` and operate on plain tensors.

The set of op *names* to register is a caller (domain-layer) concern: pass
them to :func:`register_op_wrappers`. This module names no specific op.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


def _is_shard_tensor(x: Any) -> bool:
    """Check if x is a ShardTensor without importing at module level."""
    return type(x).__name__ == "ShardTensor"


def _to_local_if_shard(x: Any) -> Any:
    """Call .to_local() if x is a ShardTensor, else return as-is."""
    if _is_shard_tensor(x):
        return x.to_local()
    return x


def _first_shard_tensor(*args: Any) -> Any | None:
    """Find the first ShardTensor in args."""
    for a in args:
        if _is_shard_tensor(a):
            return a
    return None


def make_passthrough_wrapper(op_name: str) -> Callable:
    """Generate a ShardTensor handler that unwraps all args to local tensors.

    Works for any op where each rank's shard is processed independently.
    In-place mutations persist because ``to_local()`` returns the backing
    ``_local_tensor`` directly (not a copy).

    Parameters
    ----------
    op_name : str
        The custom op name (for logging/debugging).

    Returns
    -------
    Callable
        A handler suitable for ``ShardTensor.register_named_function_handler``.
    """

    def wrapper(func: Callable, types: Any, args: tuple, kwargs: dict) -> Any:
        local_args = tuple(_to_local_if_shard(a) for a in args)
        local_kwargs = {k: _to_local_if_shard(v) for k, v in kwargs.items()}
        result = func(*local_args, **local_kwargs)

        if result is None:
            # mutates_args ops return None; ShardTensor storage updated in-place
            return None

        # For ops returning new tensors: match sharding of first ShardTensor input
        ref = _first_shard_tensor(*args)
        if ref is not None and isinstance(result, torch.Tensor):
            from nvalchemi.distributed._core._st_backend import ShardTensor

            return ShardTensor.from_local(
                result,
                ref._spec.mesh,
                ref._spec.placements,
                sharding_shapes=ref._spec.sharding_shapes(),
            )

        # For tuple results: wrap each tensor element
        if ref is not None and isinstance(result, tuple):
            from nvalchemi.distributed._core._st_backend import ShardTensor

            wrapped = []
            for r in result:
                if isinstance(r, torch.Tensor):
                    wrapped.append(
                        ShardTensor.from_local(
                            r,
                            ref._spec.mesh,
                            ref._spec.placements,
                            sharding_shapes=ref._spec.sharding_shapes(),
                        )
                    )
                else:
                    wrapped.append(r)
            return tuple(wrapped)

        return result

    return wrapper


def make_reduction_wrapper(reduce_op: Any) -> Callable:
    """Generate a ShardTensor handler for per-atom → per-system reductions.

    The output gets ``Partial(reduce_op)`` placement.  The actual
    all-reduce happens lazily when ``.full_tensor()`` or
    ``.redistribute(..., [Replicate()])`` is called.

    Parameters
    ----------
    reduce_op : torch.distributed.ReduceOp
        The reduction operation (SUM, MAX, MIN).

    Returns
    -------
    Callable
        A handler suitable for ``ShardTensor.register_named_function_handler``.
    """

    def wrapper(func: Callable, types: Any, args: tuple, kwargs: dict) -> Any:
        local_args = tuple(_to_local_if_shard(a) for a in args)
        local_kwargs = {k: _to_local_if_shard(v) for k, v in kwargs.items()}
        result = func(*local_args, **local_kwargs)

        ref = _first_shard_tensor(*args)
        if ref is not None and isinstance(result, torch.Tensor):
            from torch.distributed.tensor import Partial

            from nvalchemi.distributed._core._st_backend import ShardTensor

            return ShardTensor.from_local(
                result,
                ref._spec.mesh,
                (Partial(reduce_op),),
            )

        return result

    return wrapper


def register_op_wrappers(
    passthrough_ops: Iterable[str],
    reduction_ops: Mapping[str, Any],
) -> bool:
    """Register ShardTensor dispatch handlers for the given op names.

    Op names follow the ``torch.library`` convention ``"namespace::op_name"``;
    the ``".default"`` overload suffix is appended automatically to match how
    ShardTensor dispatch resolves the call.

    Parameters
    ----------
    passthrough_ops : Iterable[str]
        Per-row ops whose ShardTensor inputs are unwrapped to local tensors and
        whose tensor results are re-wrapped matching the first ShardTensor
        input. See :func:`make_passthrough_wrapper`.
    reduction_ops : Mapping[str, torch.distributed.ReduceOp]
        ``{op_name: reduce_op}`` for per-row → per-group reductions; the output
        gets ``Partial(reduce_op)`` placement. See :func:`make_reduction_wrapper`.

    Returns
    -------
    bool
        ``True`` if ShardTensor was available and handlers were registered,
        ``False`` if registration was skipped (no ShardTensor backend).

    Idempotency is the caller's concern: re-registering simply overwrites the
    handler for a name. Domain layers typically guard with a module-level flag.
    """
    try:
        from nvalchemi.distributed._core._st_backend import ShardTensor
    except ImportError:
        logger.debug(
            "physicsnemo.domain_parallel not available; skipping shard wrapper registration"
        )
        return False

    if ShardTensor is None:
        logger.debug(
            "ShardTensor is None (PyTorch < 2.6); skipping shard wrapper registration"
        )
        return False

    n_passthrough = 0
    for op_name in passthrough_ops:
        full_name = f"{op_name}.default"
        try:
            ShardTensor.register_named_function_handler(
                full_name, make_passthrough_wrapper(op_name)
            )
            n_passthrough += 1
        except Exception:
            logger.debug("Failed to register passthrough wrapper for %s", full_name)

    n_reduction = 0
    for op_name, reduce_op in reduction_ops.items():
        full_name = f"{op_name}.default"
        try:
            ShardTensor.register_named_function_handler(
                full_name, make_reduction_wrapper(reduce_op)
            )
            n_reduction += 1
        except Exception:
            logger.debug("Failed to register reduction wrapper for %s", full_name)

    logger.info(
        "Registered %d passthrough + %d reduction ShardTensor wrappers",
        n_passthrough,
        n_reduction,
    )
    return True
