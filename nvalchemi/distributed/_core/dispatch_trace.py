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

"""Lightweight dispatch-trace mechanism for the distributed handlers.

When a test or debug session opens a :func:`dispatch_trace` context,
every :class:`~nvalchemi.distributed._core.shard_tensor.ShardTensor` handler
that fires inside the context appends a record to a list. The records
capture which handler ran, on which op, with what shapes, and the
branch it took — enough to assert dispatch correctness in tests
*without* running the underlying kernels.

Why this exists
---------------
Multi-rank distributed bugs manifest as wrong numbers on the cluster,
with no local way to verify whether the right handler fired on the right
op. Env-gated debug prints (``NVALCHEMI_REDUCE_DEBUG`` etc.) live only in
print logs and can't be asserted on. A structured trace buffer serves
both: tests assert on it, debug sessions ``json.dumps`` it.

Usage
-----
::

    from nvalchemi.distributed._core.dispatch_trace import dispatch_trace

    with dispatch_trace() as records:
        out = model(input_batch)

    # Each record is a dict with at minimum {"handler": str, "rank": int}.
    handlers_fired = [r["handler"] for r in records]
    assert handlers_fired.count("per_system_reduce") == 1

Recursion
---------
``dispatch_trace`` uses a single module-level slot, so nested
``with`` blocks would shadow the outer trace until the inner exits.
Don't nest. (Tests should ``yield`` once; debug should open one.)

Thread safety
-------------
None — distributed runs are one process per rank, single-threaded
inside each. If we ever go thread-pool, this becomes a
``ContextVar``.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

import torch

__all__ = [
    "dispatch_trace",
    "is_tracing",
    "record_dispatch",
]


_TRACE_SINK: list[dict[str, Any]] | None = None


@contextlib.contextmanager
def dispatch_trace() -> Iterator[list[dict[str, Any]]]:
    """Open a dispatch-trace scope. Yields a list that handlers append to."""
    global _TRACE_SINK
    records: list[dict[str, Any]] = []
    prev = _TRACE_SINK
    _TRACE_SINK = records
    try:
        yield records
    finally:
        _TRACE_SINK = prev


def is_tracing() -> bool:
    """``True`` while a :func:`dispatch_trace` context is active. Handlers
    short-circuit their record-building when this is ``False`` — the
    common case in production has near-zero overhead."""
    return _TRACE_SINK is not None


def record_dispatch(handler: str, **fields: Any) -> None:
    """Append a record into the active trace, if any. No-op otherwise.

    Convention for ``fields``:
      - ``op``           — string name of the op being dispatched
                           (``"scatter_add_"``, ``"index_add_"``, etc.).
      - ``branch``       — string indicating which sub-path inside the
                           handler ran (``"halo_correction"``,
                           ``"slice_only"``, ``"all_reduce_only"``, ...).
      - ``shapes``       — dict mapping arg-name → tuple shape.
      - ``meta``         — dict for any handler-specific extras
                           (``n_owned``, ``n_padded``, ``n_systems``,
                           pre/post sums on demand).

    Records also auto-tag with the rank (or ``-1`` if no process group
    is initialised) so single-process debug sessions remain readable.
    """
    if _TRACE_SINK is None:
        return
    import torch.distributed as _td  # noqa: PLC0415

    rank = _td.get_rank() if _td.is_initialized() else -1
    record: dict[str, Any] = {"handler": handler, "rank": rank}
    record.update(fields)
    _TRACE_SINK.append(record)


def _shape_of(t: Any) -> tuple[int, ...] | None:
    """Helper for handlers building ``shapes`` dicts — returns ``None``
    for non-tensors so the trace is JSON-friendly."""
    if isinstance(t, torch.Tensor):
        return tuple(t.shape)
    return None
