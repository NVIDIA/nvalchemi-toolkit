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

"""Bridge for running a halo-distributed model under ``torch.compile``.

Under ``torch.compile`` a ShardTensor is traced away, so a domain-decomposed
model must run the compiled region on plain tensors with the halo routing
surfaced as graph inputs (not Python constants), and re-create the per-layer
ghost refresh that eager dispatch does for free. :class:`HaloCompileBridge`
provides that scaffolding once so each model wrapper does not hand-roll it:

* **plain-ify + route.** The caller hands a ``dict`` of inputs (some entries
  ShardTensors); the bridge ``to_local``\\ s them and threads the four halo
  routing tensors ``(send_index, recv_dest, recv_real, n_owned)`` as graph
  inputs, anchored live on an entry tensor via
  :func:`~nvalchemi.distributed.compile_refresh.keep_routing_live` (otherwise
  Dynamo prunes unused inputs). The dict preserves input names so the
  compile-refresh pass can match ``edge_index``.
* **pluggable refresh.** ``refresh="pass"`` uses the compile-refresh graph
  pass, which auto-inserts the halo correction at each ``edge_index``-keyed
  scatter — the zero-config path for a clean message-passing model.
  ``refresh="self"`` uses a plain backend for models that carry their own
  refresh inside the traced region (MACE, AIMNet2), which the pass can't reach
  (e.g. MACE's e3nn graph breaks); the bridge then only provides the plain-ify
  + routing scaffolding.

The compiled callable is cached on first call and reused across MD steps;
fixed-shape input caps keep the graph stable so there are no steady-state
recompiles.
"""

from __future__ import annotations

import contextlib
from typing import Any, Callable, Iterator

__all__ = [
    "HaloCompileBridge",
    "force_compile_static",
]


@contextlib.contextmanager
def force_compile_static() -> Iterator[None]:
    """Force ``dynamic=False`` on every ``torch.compile`` call within the block.

    Some libraries hard-code ``torch.compile(model, dynamic=True)``. Under
    domain decomposition the joint (forward+backward) graph trips an inductor
    assertion with dynamic shapes, while static shapes compile cleanly and
    ~2x faster. Pair with fixed-shape caps so compiled MD stays compiled."""
    import torch  # noqa: PLC0415

    orig = torch.compile

    def _static(*args: Any, **kwargs: Any) -> Any:
        kwargs["dynamic"] = False
        return orig(*args, **kwargs)

    torch.compile = _static  # type: ignore[assignment]
    try:
        yield
    finally:
        torch.compile = orig  # type: ignore[assignment]


def _to_local(x: Any) -> Any:
    """ShardTensor -> its plain local tensor; pass-through otherwise. Runs
    eagerly so autograd flows from the plain local back to the ShardTensor."""
    return x.to_local() if hasattr(x, "to_local") else x


class HaloCompileBridge:
    """Run a per-atom model forward under ``torch.compile`` + domain
    decomposition with no wrapper-side bridge code.

    Parameters
    ----------
    forward
        ``forward(inputs: dict) -> output`` — the compilable region. The caller
        adapts its model's signature here (e.g. ``lambda mi: model(mi["positions"],
        mi["edge_index"])``), keeping the bridge signature-agnostic.
    world_size
        Static mesh size — baked into the inserted halo op by the pass.
    refresh
        ``"pass"`` (auto-insert the halo correction; clean models) or
        ``"self"`` (model carries its own refresh; plain backend).
    inner_backend
        Backend the pass defers to / the plain backend (default ``inductor``;
        ``aot_eager`` for codegen-free dev).
    anchor_key
        Input the routing tensors are anchored live on (default
        ``"positions"``); must be a tensor present in every ``inputs`` dict.
    routing_keys
        Names the four routing tensors are threaded under (must match the pass).
    """

    def __init__(
        self,
        forward: Callable[[dict], Any],
        *,
        world_size: int,
        refresh: str = "pass",
        inner_backend: str = "inductor",
        anchor_key: str = "positions",
        routing_keys: tuple[str, str, str, str] = (
            "_halo_si",
            "_halo_rd",
            "_halo_rr",
            "_halo_no",
        ),
        compile_kwargs: dict | None = None,
    ) -> None:
        if refresh not in ("pass", "self"):
            raise ValueError(f"refresh must be 'pass' or 'self', got {refresh!r}")
        self._forward = forward
        self._world_size = int(world_size)
        self._refresh = refresh
        self._inner_backend = inner_backend
        self._anchor_key = anchor_key
        self._routing_keys = routing_keys
        self._compile_kwargs = dict(compile_kwargs or {})
        self._compiled: Callable[..., Any] | None = None

    def _build(self) -> None:
        import torch  # noqa: PLC0415

        from nvalchemi.distributed._core.compile_routing import (  # noqa: PLC0415
            set_compile_routing,
        )
        from nvalchemi.distributed.compile_refresh import (  # noqa: PLC0415
            keep_routing_live,
            make_dd_halo_backend,
        )

        forward = self._forward
        anchor_key = self._anchor_key

        if self._refresh == "pass":
            # Anchor routing live on the entry tensor, then run the user's
            # forward on the plain input dict. Routing is passed as positional
            # tensors so they become graph inputs, not constants. The parameter
            # names must contain the pass's routing names (``_halo_*``): the
            # pass matches routing placeholders by name, and Dynamo derives
            # names from these params.
            def _compiled_region(
                inputs: dict,
                _halo_si: Any,
                _halo_rd: Any,
                _halo_rr: Any,
                _halo_no: Any,
            ) -> Any:
                inputs = dict(inputs)
                inputs[anchor_key] = keep_routing_live(
                    inputs[anchor_key], _halo_si, _halo_rd, _halo_rr, _halo_no
                )
                return forward(inputs)

            backend: Any = make_dd_halo_backend(self._world_size, self._inner_backend)
        else:
            # Self-refresh: the model carries its own halo correction inside the
            # traced region, reading routing through the compile-routing holder.
            # The bridge publishes that routing here, inside the region, from the
            # graph-input ``_halo_*`` tensors. Used where the pass can't reach
            # the scatter (MACE's e3nn graph breaks; AIMNet2's Warp conv).
            si_k, rd_k, rr_k, no_k = self._routing_keys
            default_ws = self._world_size

            def _compiled_region(inputs: dict) -> Any:  # type: ignore[misc]
                si = inputs.get(si_k)
                if si is not None:
                    # Prefer the threaded ``_halo_ws`` (the real mesh size — the
                    # bridge is built before the mesh exists); fall back to
                    # self._world_size.
                    set_compile_routing(
                        si,
                        inputs[rd_k],
                        inputs[rr_k],
                        inputs[no_k],
                        int(inputs.get("_halo_ws", default_ws)),
                    )
                return forward(inputs)

            backend = self._inner_backend

        self._compiled = torch.compile(
            _compiled_region, backend=backend, **self._compile_kwargs
        )

    def __call__(
        self, inputs: dict, routing: tuple[Any, Any, Any, Any] | None = None
    ) -> Any:
        """Run the bridged forward.

        ``inputs`` is the model's input dict (ShardTensor entries are plain-ified
        here, eagerly). For ``refresh="pass"``, ``routing`` is required —
        ``(send_index, recv_dest, recv_real, n_owned)`` from
        :func:`~nvalchemi.distributed._core.particle_halo.build_halo_meta_tensors`
        — and is anchored as graph inputs for the pass. For ``refresh="self"``,
        ``routing`` is ignored (the model reads whatever routing the caller placed
        in ``inputs``).
        """
        if self._compiled is None:
            self._build()
        local = {k: _to_local(v) for k, v in inputs.items()}
        if self._refresh == "pass":
            if routing is None:
                raise ValueError("refresh='pass' requires routing tensors")
            si, rd, rr, no = routing
            return self._compiled(local, si, rd, rr, no)
        # refresh="self": the compiled region publishes routing to the holder
        # for its in-region refresh helpers. Clear it afterward so a later eager
        # refresh never reads stale trace-time routing.
        from nvalchemi.distributed._core.compile_routing import (  # noqa: PLC0415
            clear_compile_routing,
        )

        try:
            return self._compiled(local)
        finally:
            clear_compile_routing()


def _consolidate_node_energy(node_e: Any, batch: Any, n_graphs: int) -> Any:
    """Per-node energy -> per-system energy (eager, outside compile).

    Halo path: owned-only per-system sum plus cross-rank all-reduce (via
    ``system_sum``). Single-process: a plain per-system ``index_add``. Runs
    eagerly so ``to_local``'s backward is eager (in-compile ``to_local`` trips a
    meta-conversion assert on an op-result ShardTensor)."""
    import torch  # noqa: PLC0415

    from nvalchemi.distributed._core.context import current_dd_context  # noqa: PLC0415
    from nvalchemi.distributed._core.enums import Scope  # noqa: PLC0415
    from nvalchemi.distributed.helpers import system_sum  # noqa: PLC0415

    node_e_local = node_e.to_local() if hasattr(node_e, "to_local") else node_e
    ctx = current_dd_context()
    if ctx.is_halo and node_e_local.shape[0] > ctx.n_owned:
        return system_sum(node_e_local, batch, n_graphs, scope=Scope.OWNED)
    energy = torch.zeros(n_graphs, dtype=node_e_local.dtype, device=node_e_local.device)
    return energy.index_add_(0, batch, node_e_local)
