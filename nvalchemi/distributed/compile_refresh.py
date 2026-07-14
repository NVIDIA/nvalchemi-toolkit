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

"""The compile-refresh graph pass.

In eager domain-decomposed execution, ``ShardTensor`` dispatch halo-corrects each
message-passing layer's node aggregation: after a scatter into the per-node
accumulator, ghost rows hold this rank's *partial* sums, so the framework
reverse-exchanges partials to owners and refreshes ghosts. Under ``torch.compile``
the subclass is traced away, so that correction disappears and ghosts go stale ->
silently wrong forces. This pass auto-places the correction so a standard
message-passing model needs no hand-written refresh in eager *or* compile.

It inserts ``torch.ops.nvalchemi.halo_scatter_correct_static`` (a fixed-shape
custom op with a registered autograd backward), finding the sites by
match-and-replace over the FX graph.

Detection is keyed on the neighbor list. The framework owns the graph boundary
(it builds the padded batch + threads routing), so the ``edge_index`` and
halo-routing tensors are known graph inputs. A ``scatter_add`` / ``index_add``
whose *index* argument traces back to an ``edge_index`` input is a message-passing
site. Keying on a known input is more robust than generic shape analysis (a
per-graph energy scatter, indexed by ``batch_idx``, never traces to ``edge_index``
and is correctly left alone).

The pass runs on the Dynamo FX graph inside a ``torch.compile`` backend, before
autograd is traced — so the inserted op's registered backward is picked up for
free. Matching by index provenance (not op target) sidesteps variation in op form
at this stage.

Safety: the backend fails safe — if it is active under DD + compile but finds
zero sites (or the routing inputs are absent), it raises rather than running with
stale ghosts. Per-model validation against an eager reference remains the
correctness net; this pass need only be validated, not provably complete.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import torch

if TYPE_CHECKING:
    import torch.fx

__all__ = [
    "RefreshReport",
    "insert_halo_refresh",
    "keep_routing_live",
    "make_dd_halo_backend",
]

logger = logging.getLogger(__name__)


# Routing keep-alive op, registered at import (not lazily inside a traced call —
# fullgraph=True cannot trace op registration). An identity on ``x`` that consumes
# the four routing tensors: nothing in a zero-config model uses routing and Dynamo
# prunes unused args, so the bridge calls this on an entry tensor to anchor the
# routing as live graph inputs the pass can wire to. ``world_size`` is static
# (baked by the pass), so it is not anchored here. Explicit schema is needed
# because ``from __future__ import annotations`` stringizes the hints and breaks
# torch's infer_schema.
if not hasattr(torch.ops.nvalchemi, "halo_keepalive"):

    @torch.library.custom_op(
        "nvalchemi::halo_keepalive",
        mutates_args=(),
        schema="(Tensor x, Tensor si, Tensor rd, Tensor rr, Tensor no) -> Tensor",
    )
    def _halo_keepalive(x, si, rd, rr, no):  # noqa: ANN001, ANN202
        return x.clone()

    @_halo_keepalive.register_fake
    def _(x, si, rd, rr, no):  # noqa: ANN001, ANN202
        return torch.empty_like(x)

    _halo_keepalive.register_autograd(
        lambda ctx, grad: (grad, None, None, None, None),
        setup_context=lambda ctx, inputs, output: None,
    )


def keep_routing_live(x: Any, si: Any, rd: Any, rr: Any, no: Any) -> Any:
    """Anchor the halo routing tensors as live graph inputs (identity on ``x``).

    The bridge calls this on an entry tensor (e.g. the plain positions) before
    running the compiled model, so the four routing tensors the pass needs survive
    Dynamo's dead-input pruning.
    """
    return torch.ops.nvalchemi.halo_keepalive.default(x, si, rd, rr, no)


# The four halo-routing tensors the inserted op consumes, in argument order after
# ``node_feats``. The bridge threads + anchors these as live graph inputs.
# ``world_size`` is baked into the inserted call as a literal, not a routing input.
DEFAULT_ROUTING_NAMES: tuple[str, ...] = (
    "_halo_si",
    "_halo_rd",
    "_halo_rr",
    "_halo_no",
)
DEFAULT_EDGE_INDEX_NAMES: tuple[str, ...] = ("edge_index", "nbmat", "neighbor_list")

# FX targets recognized as a node-aggregation scatter; both aggregate ``src`` rows
# into ``self`` at rows given by ``index`` (arg 2).
_SCATTER_METHODS = frozenset({"scatter_add", "scatter_add_", "index_add", "index_add_"})


@dataclass
class RefreshReport:
    """What :func:`insert_halo_refresh` did, for logging + the safety net."""

    n_sites: int = 0
    site_names: list[str] = field(default_factory=list)
    edge_index_inputs: list[str] = field(default_factory=list)
    routing_present: bool = False
    routing_missing: list[str] = field(default_factory=list)
    # Diagnostics (graph-break models fragment into many graphs; these explain
    # why a fragment did/did not get a correction).
    scatters_seen: int = 0
    scatters_edge_keyed: int = 0
    n_placeholders: int = 0


def _placeholders_matching(
    gm: "torch.fx.GraphModule", names: tuple[str, ...]
) -> dict[str, "torch.fx.Node"]:
    """Map each wanted name to the placeholder whose mangled name contains it.

    Dynamo mangles inputs (``edge_index`` -> ``l_edge_index_``, dict entries ->
    ``l_mi_halo_si_`` etc.), so we match by substring rather than exact name.
    """
    out: dict[str, Any] = {}
    placeholders = [n for n in gm.graph.nodes if n.op == "placeholder"]
    for want in names:
        for n in placeholders:
            if want in n.name or want in str(n.target):
                out[want] = n
                break
    return out


def _traces_to(node: Any, targets: set, _seen: set | None = None) -> bool:
    """Whether ``node``'s data-flow reaches any node in ``targets`` (DFS over
    ``all_input_nodes``)."""
    if _seen is None:
        _seen = set()
    if node in targets:
        return True
    if not hasattr(node, "all_input_nodes") or node in _seen:
        return False
    _seen.add(node)
    return any(_traces_to(a, targets, _seen) for a in node.all_input_nodes)


def _is_node_scatter(node: Any) -> bool:
    """A node-aggregation scatter (``scatter_add`` / ``index_add``), in either
    ``call_method`` or ``call_function`` (aten) form."""
    if node.op == "call_method" and node.target in _SCATTER_METHODS:
        return True
    if node.op == "call_function":
        name = getattr(node.target, "_opname", None) or getattr(
            node.target, "__name__", ""
        )
        return any(name.startswith(m.rstrip("_")) for m in _SCATTER_METHODS)
    return False


def insert_halo_refresh(
    gm: "torch.fx.GraphModule",
    *,
    correction_op: Callable[..., Any],
    world_size: int,
    edge_index_names: tuple[str, ...] = DEFAULT_EDGE_INDEX_NAMES,
    routing_names: tuple[str, ...] = DEFAULT_ROUTING_NAMES,
    require_routing: bool = True,
) -> RefreshReport:
    """Insert ``correction_op(out, si, rd, rr, no, world_size)`` after every
    message-passing node-scatter in ``gm``.

    A site is a ``scatter_add``/``index_add`` whose *index* argument (arg 2)
    traces to an ``edge_index`` graph input. Mutates ``gm`` in place and returns a
    :class:`RefreshReport`. Does *not* raise on zero sites — the caller
    (:func:`make_dd_halo_backend`) owns the fail-safe.

    ``correction_op`` is injected (not hard-wired) so the pass is unit-testable
    with a stand-in op; production passes
    ``torch.ops.nvalchemi.halo_scatter_correct_static``.
    """
    report = RefreshReport()
    g = gm.graph
    report.n_placeholders = sum(1 for n in g.nodes if n.op == "placeholder")
    report.scatters_seen = sum(1 for n in g.nodes if _is_node_scatter(n))

    edge_inputs = _placeholders_matching(gm, edge_index_names)
    report.edge_index_inputs = [n.name for n in edge_inputs.values()]
    if not edge_inputs:
        return report  # not a message-passing graph we own
    ei_nodes = set(edge_inputs.values())

    routing = _placeholders_matching(gm, routing_names)
    report.routing_missing = [r for r in routing_names if r not in routing]
    report.routing_present = not report.routing_missing
    # Count edge-keyed scatters regardless of routing, so the diagnostic can show
    # "found the site but routing was absent in this fragment".
    for node in g.nodes:
        if not _is_node_scatter(node):
            continue
        index_arg = node.args[2] if len(node.args) > 2 else node.kwargs.get("index")
        if index_arg is not None and _traces_to(index_arg, ei_nodes):
            report.scatters_edge_keyed += 1
    if require_routing and not report.routing_present:
        return report  # caller decides if this is a hard error
    routing_args = tuple(routing[r] for r in routing_names if r in routing)

    for node in list(g.nodes):
        if not _is_node_scatter(node):
            continue
        index_arg = node.args[2] if len(node.args) > 2 else node.kwargs.get("index")
        if index_arg is None or not _traces_to(index_arg, ei_nodes):
            continue
        with g.inserting_after(node):
            new = g.call_function(correction_op, args=(node, *routing_args, world_size))
        node.replace_all_uses_with(new)
        new.update_arg(0, node)  # undo the self-reference replace_all just made
        report.n_sites += 1
        report.site_names.append(node.name)

    if report.n_sites:
        g.lint()
        gm.recompile()
    return report


def make_dd_halo_backend(
    world_size: int,
    inner_backend: str = "inductor",
    *,
    correction_op: Callable[..., Any] | None = None,
    edge_index_names: tuple[str, ...] = DEFAULT_EDGE_INDEX_NAMES,
    routing_names: tuple[str, ...] = DEFAULT_ROUTING_NAMES,
    strict: bool = True,
    log_fragments: bool = False,
) -> Callable[..., Any]:
    """A ``torch.compile`` backend that auto-inserts halo refresh then defers to
    ``inner_backend``.

    Use when a domain-decomposed model is compiled and its message passing is
    expressed in framework-visible ops (the bridge has threaded the routing
    inputs). Fails safe: if routing is threaded (so this *is* a DD-compile run)
    but no message-passing site is found, it raises rather than silently running
    with stale ghosts — the author then declares an explicit refresh.
    """
    if correction_op is None:
        import torch  # noqa: PLC0415

        correction_op = torch.ops.nvalchemi.halo_scatter_correct_static.default

    from torch._dynamo import lookup_backend  # noqa: PLC0415

    inner = lookup_backend(inner_backend)

    fragment_counter = [0]

    def backend(gm: "torch.fx.GraphModule", example_inputs: Any) -> Any:
        report = insert_halo_refresh(
            gm,
            correction_op=correction_op,
            world_size=world_size,
            edge_index_names=edge_index_names,
            routing_names=routing_names,
            require_routing=True,
        )
        if log_fragments:
            fragment_counter[0] += 1
            logger.info(
                "compile-refresh fragment #%d: placeholders=%d scatters_seen=%d "
                "scatters_edge_keyed=%d edge_inputs=%s routing_present=%s "
                "routing_missing=%s -> n_sites=%d %s",
                fragment_counter[0],
                report.n_placeholders,
                report.scatters_seen,
                report.scatters_edge_keyed,
                report.edge_index_inputs or "none",
                report.routing_present,
                report.routing_missing or "none",
                report.n_sites,
                report.site_names,
            )
        if strict and report.routing_present and report.n_sites == 0:
            raise RuntimeError(
                "compile-refresh graph pass: routing inputs were threaded "
                f"({routing_names}) so this is a domain-decomposed compiled run, "
                "but no message-passing node-scatter keyed on an edge_index input "
                f"({report.edge_index_inputs or 'none found'}) was located. Running "
                "would leave ghost rows stale -> silently wrong forces. Declare an "
                "explicit refresh for the message-passing block instead."
            )
        if report.n_sites:
            logger.info(
                "compile-refresh: inserted halo correction at %d site(s): %s",
                report.n_sites,
                report.site_names,
            )
        return inner(gm, example_inputs)

    return backend
