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
"""``ShardTensor`` — a policy-driven distributed tensor subclass.

A single :class:`torch.Tensor` wrapper-subclass that carries a
:class:`~nvalchemi.distributed._core.storage_policy.StoragePolicy` and routes
global-index ops (``scatter_add_`` / ``index_select`` / …) across ranks per
that policy. The policy is the per-field declaration of how local storage
relates to the tensor's placement and how those ops behave; this module's
dispatch handlers read ``_storage_policy`` and delegate to it.

The two shipped policies (in ``storage_policy.py``):

- **Halo** (:class:`HaloStoragePolicy`). Each rank physically stores
  ``owned + halo`` rows. Scatter does a local scatter + halo correction
  (``halo_reverse_exchange + halo_forward_exchange``); gather reads from the
  refreshed local halo rows. Fits spatially-local stencils where a row's
  neighbors live on adjacent ranks (e.g. MACE / NequIP / LJ / UMA gp-off).

- **Sharded** (:class:`PlainShard`). Each rank stores only ``n_owned`` rows;
  scatter / gather route through :func:`distributed_scatter_add` /
  :func:`distributed_index_select` (``all_to_all_v`` by global id). Fits
  arbitrary global-index gather/scatter with no locality assumption
  (e.g. AIMNet2).

A per-segment scatter (accumulator whose leading dim equals the segment count)
routes through :func:`per_system_reduce` when the spec's ``system_reductions``
flag is set — independent of the storage policy.

Metadata is instance-level — no ambient context. ``__torch_function__``
propagates the policy + metadata onto op outputs so a wrapped tensor carries
its routing info through arbitrary elementwise chains. The attached
``_distribution_spec`` is treated as an opaque config object; ``_core`` reads
only domain-neutral fields off it.
"""

from __future__ import annotations

import logging
import os as _os
from typing import TYPE_CHECKING, Any, Callable

import torch

from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy, PlainShard

if TYPE_CHECKING:
    from nvalchemi.distributed._core.gather_primitives import ShardRouting
    from nvalchemi.distributed._core.halo_types import (
        ParticleHaloConfig,
        ParticleHaloMetadata,
    )
    # ``_distribution_spec`` is annotated ``Any`` and treated as an opaque
    # distribution-config object, keeping ``_core/`` free of a back-edge to the
    # high-level ``nvalchemi.distributed.spec`` module. Dispatch reads one
    # domain-neutral field, ``.system_reductions`` (bool); the storage policy is
    # sourced from ``.distribution.policy`` once in ``wrap()`` and carried as
    # ``_storage_policy``.

logger = logging.getLogger(__name__)


__all__ = [
    "ShardTensor",
    "register_handler",
    "clear_handlers",
    "list_handlers",
]


# ======================================================================
# Op-handler registry — single extension point. First matching entry wins.
# ======================================================================


# Handlers live in the base's ``_function_registry`` / ``_named_function_registry``
# and are dispatched by our ``__torch_function__`` override. ``_OUR_HANDLERS``
# tracks which entries are ours so ``clear_handlers`` / ``list_handlers`` never
# disturb the base's own handlers (e.g. unbind / mean). Two registration paths
# share the base registry: ``register_handler`` (public escape hatch, adapts a
# user ``(*args, **kwargs)`` handler that always handles the op) and
# ``_register_function`` (internal base-signature dispatchers that self-classify
# and fall through to default dispatch when an op isn't MLIP-routed).

_OUR_HANDLERS: dict[Any, str] = {}  # registry key -> display name


def _register_function(key: Any, handler: Callable[..., Any], name: str) -> None:
    """Register a base-signature ``(func, types, args, kwargs)`` handler in the
    base function registry and record it as ours."""
    ShardTensor.register_function_handler(key, handler)
    _OUR_HANDLERS[key] = name


def register_handler(
    func: Callable[..., Any],
    handler: Callable[..., Any] | None = None,
    *,
    name: str = "",
) -> Callable[..., Any] | None:
    """Register a handler to fire when an op is dispatched with a ShardTensor.

    The handler receives the op's ``(*args, **kwargs)`` directly and is expected
    to fully handle the call (branch internally if behavior is conditional). It
    is stored in the base ShardTensor function registry via an adapter to the
    base's ``(func, types, args, kwargs)`` contract.

    Parameters
    ----------
    func : Callable
        The op to intercept when a :class:`ShardTensor` appears in its args.
    handler : Callable, optional
        The handler, called as ``handler(*args, **kwargs)``. If ``None``, this
        returns a decorator that registers the decorated function.
    name : str, optional
        Display name recorded in the registry; defaults to the handler name.

    Returns
    -------
    Callable or None
        The registered handler, or a decorator when ``handler`` is ``None``.
    """

    def _register(h: Callable[..., Any]) -> Callable[..., Any]:
        # Register at the __torch_dispatch__ (aten / dispatcher-op) level: the
        # custom __torch_function__ is the C sentinel (Dynamo native handling),
        # so handlers fire under both eager and compile and must be keyed by the
        # dispatcher op that appears in __torch_dispatch__, not the public
        # callable. The handler is invoked as h(*args, **kwargs).
        ShardTensor.register_dispatch_handler(func, h)
        _OUR_HANDLERS[func] = name or getattr(h, "__name__", repr(h))
        return h

    if handler is None:
        return _register
    return _register(handler)


def clear_handlers(func: Callable[..., Any] | None = None) -> None:
    """Remove nvalchemi-registered handlers — all, or for one op.

    Never touches the base's own (non-nvalchemi) handlers.

    Parameters
    ----------
    func : Callable, optional
        Remove only the handler for this op. If ``None``, remove all
        nvalchemi-registered handlers.
    """
    keys = list(_OUR_HANDLERS) if func is None else [func]
    for key in keys:
        ShardTensor._function_registry.pop(key, None)
        ShardTensor._named_function_registry.pop(str(key), None)
        ShardTensor._dispatch_registry.pop(key, None)
        ShardTensor._dispatch_registry_by_name.pop(str(key), None)
        _OUR_HANDLERS.pop(key, None)


def list_handlers() -> list[tuple[str, str]]:
    """Return ``(op_name, handler_name)`` pairs for nvalchemi registrations.

    Returns
    -------
    list of tuple of str
        One ``(op_name, handler_name)`` entry per nvalchemi-registered handler.
    """
    return [(getattr(f, "__qualname__", str(f)), n) for f, n in _OUR_HANDLERS.items()]


# ======================================================================
# Source discovery and metadata propagation.
# ======================================================================


def _find_source(args: Any) -> "ShardTensor | None":
    """Return the first :class:`ShardTensor` in ``args`` that carries a
    spec. Recurses into tuples/lists so handlers work on ops like
    :func:`torch.cat` / :func:`torch.stack` whose first argument is a
    sequence of tensors.
    """
    if isinstance(args, ShardTensor) and getattr(args, "_distribution_spec", None) is not None:
        return args
    if isinstance(args, (tuple, list)):
        for a in args:
            found = _find_source(a)
            if found is not None:
                return found
    return None


def _find_any_shard(args: Any) -> "ShardTensor | None":
    """Like :func:`_find_source` but matches ShardTensors even when
    they don't carry a spec — used for metadata propagation in the
    default fall-through path.
    """
    if isinstance(args, ShardTensor):
        return args
    if isinstance(args, (tuple, list)):
        for a in args:
            found = _find_any_shard(a)
            if found is not None:
                return found
    return None


def _prefer_source(args: Any, kwargs: dict | None = None) -> "ShardTensor | None":
    """Prefer a ShardTensor with a spec; fall back to any ShardTensor.

    Collapses the paired ``source = _find_source(args); if source is
    None: source = _find_any_shard(args)`` pattern that was repeated
    in every dispatch predicate / handler. When ``kwargs`` is supplied,
    the values are searched after ``args`` only if ``args`` had nothing.
    """
    source = _find_source(args)
    if source is not None:
        return source
    source = _find_any_shard(args)
    if source is not None:
        return source
    if kwargs:
        vals = tuple(kwargs.values())
        source = _find_source(vals)
        if source is not None:
            return source
        source = _find_any_shard(vals)
    return source


_PROPAGATED_ATTRS = (
    "_distribution_spec",
    "_config",
    "_meta",
    "_gather_meta",
    "_n_systems",
    "_system_index",
    "_extra_suffix_padding",
    "_storage_policy",
)


def _propagate_attrs(result: Any, source: "ShardTensor") -> None:
    """Copy routing metadata from ``source`` onto any ShardTensor in
    ``result`` that doesn't already have its own. Walks tuples / lists.

    Also coerces upstream-base ``ShardTensor`` instances back to our
    subclass: upstream's ``__torch_function__`` autowrap path constructs
    via ``ShardTensor(...)`` hardcoded to the base
    class, losing our subclass type. Reassigning ``__class__`` is safe
    here because our subclass and upstream share the same instance
    layout (we don't add slots).
    """
    from nvalchemi.distributed._core._st_backend import (
        ShardTensor as _UpstreamShardTensor,
    )

    def _apply(t: Any) -> None:
        if isinstance(t, _UpstreamShardTensor) and not isinstance(t, ShardTensor):
            t.__class__ = ShardTensor
        if isinstance(t, ShardTensor) and getattr(t, "_distribution_spec", None) is None:
            for attr in _PROPAGATED_ATTRS:
                setattr(t, attr, getattr(source, attr))
            # _halo_meta_packed propagates separately (flatten inner tensor, not
            # an mlip_ctx constant). Copy the BACKING value (None or real) so an
            # unset source stays unset (the target's own property lazily mints a
            # sentinel in ITS context); copying the property's sentinel would
            # freeze a possibly-cross-context tensor.
            hmp = getattr(source, "_halo_meta_packed_v", None)
            if hmp is not None:
                t._halo_meta_packed_v = hmp

    if isinstance(result, torch.Tensor):
        _apply(result)
    elif isinstance(result, (tuple, list)):
        for r in result:
            _apply(r)


def _strip_to_local(arg: Any) -> Any:
    """Return ``arg._local_tensor`` if it's a ShardTensor, else ``arg``.

    Unwraps recursively into tuples / lists. Used by the extract-local /
    run-plain / rewrap path so the op sees plain tensors regardless of whether
    some args are ShardTensor and some aren't.
    """
    from nvalchemi.distributed._core._st_backend import (
        ShardTensor as _UpstreamShardTensor,
    )

    if isinstance(arg, _UpstreamShardTensor):
        return arg._local_tensor
    if isinstance(arg, tuple):
        return tuple(_strip_to_local(x) for x in arg)
    if isinstance(arg, list):
        return [_strip_to_local(x) for x in arg]
    return arg


class _AutogradPreservingWrap(torch.autograd.Function):
    """Construct a :class:`ShardTensor` around ``local_tensor`` while
    threading autograd through the wrap step.

    The wrapper-subclass carries autograd state independent of
    ``_local_tensor`` — ``wrapper.grad_fn`` is ``None`` even when
    ``_local_tensor.grad_fn`` is set — so ``torch.autograd.grad`` against a
    leaf reachable through the inner can't find it. This Function (mirroring
    DTensor's ``_FromTorchTensor``) attaches a ``grad_fn`` to the wrapper so
    gradients terminate on ``local_tensor``; backward extracts ``_local_tensor``
    from the incoming wrapper-shaped gradient and returns it as the gradient
    w.r.t. ``local_tensor``.
    """

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        local_tensor: torch.Tensor,
        spec: Any,
    ) -> "ShardTensor":
        # ``Function.apply`` attaches a ``grad_fn`` to the wrapper output,
        # threading the wrap step into the autograd graph so gradients
        # terminate on ``local_tensor``. We route grad exclusively through the
        # outer wrapper (no inner ``view_as`` as DTensor uses) because a view
        # inner triggers ``_automatic_dynamic`` recursion on ``e._base`` under
        # compile.
        return ShardTensor(local_tensor=local_tensor,
            spec=spec,
            requires_grad=local_tensor.requires_grad,
        )

    @staticmethod
    def backward(ctx: Any, grad_out: Any) -> tuple:  # type: ignore[override]
        # grad_out arrives as a ShardTensor (autograd preserves type).
        # Extract its inner; the spec arg gets None (no grad).
        from nvalchemi.distributed._core._st_backend import (  # noqa: PLC0415
            ShardTensor as _Upstream,
        )

        if isinstance(grad_out, _Upstream):
            return grad_out._local_tensor, None
        return grad_out, None


class _UnwrapSource:
    """Grad-free snapshot of a ShardTensor's routing metadata.

    Stored on the autograd ``ctx`` in place of the source ShardTensor itself.
    Storing the tensor subclass (``ctx.source = wrapper``) forms a reference
    cycle — the backward node holds ``ctx`` -> ``ctx.source`` (the wrapper) ->
    ``wrapper.grad_fn`` (back into the graph) — that Python refcounting cannot
    break; only the cyclic GC can. Across multi-step inference (MD / repeated
    forward+autograd-forces) that leaks ~one full autograd graph per iteration
    until OOM. This surrogate carries only ``_spec`` + the propagated routing
    attrs (no ``_local_tensor``, no ``grad_fn``), so ``ctx`` no longer closes a
    cycle and the graph is freed by refcount as soon as the step's outputs drop.
    ``_make_handler_output`` reads exactly these fields, so backward is
    unchanged. Mirrors upstream DTensor's ``_ToTorchTensor`` keeping
    placements/mesh metadata rather than the DTensor.
    """

    __slots__ = ("_spec", *_PROPAGATED_ATTRS, "_halo_meta_packed_v")

    def __init__(self, source: "ShardTensor") -> None:
        self._spec = source._spec
        for _attr in _PROPAGATED_ATTRS:
            setattr(self, _attr, getattr(source, _attr, None))
        self._halo_meta_packed_v = getattr(source, "_halo_meta_packed_v", None)


class _AutogradPreservingUnwrap(torch.autograd.Function):
    """Extract a ShardTensor's ``_local_tensor`` while threading autograd
    through the unwrap step — the inverse of :class:`_AutogradPreservingWrap`.

    The wrapper-subclass tracks its autograd on the WRAPPER (``__torch_dispatch__``
    + ``return_and_correct_aliasing``), so for an op-result ShardTensor the
    ``_local_tensor`` is autograd-DETACHED (``requires_grad=False`` / no
    ``grad_fn``) even though the wrapper itself requires grad. A dispatch
    handler that computes on the plain ``_local_tensor`` and re-wraps the
    result therefore SEVERS the graph: the model-side autograd (positions →
    energy) terminates at the detached local tensor.

    This Function fixes that: forward returns ``wrapper._local_tensor`` but
    ``Function.apply`` tags the output with ``grad_fn=_AutogradPreservingUnwrapBackward``,
    so gradients flowing back from the handler's computation re-enter the
    wrapper's graph. Backward re-wraps the incoming local-shaped gradient as a
    ShardTensor (the type the wrapper's grad_fn expects).
    """

    @staticmethod
    def forward(ctx: Any, wrapper: "ShardTensor") -> torch.Tensor:  # type: ignore[override]
        # Store grad-free metadata, NOT the ShardTensor: ``ctx.source =
        # wrapper`` would close a grad_fn<->ctx<->wrapper cycle that only the
        # cyclic GC can break, leaking the autograd graph across inference
        # steps (OOM). See :class:`_UnwrapSource`.
        ctx.source = _UnwrapSource(wrapper)
        return wrapper._local_tensor

    @staticmethod
    def backward(ctx: Any, grad_local: torch.Tensor) -> Any:  # type: ignore[override]
        # The wrapper's grad_fn expects a ShardTensor gradient; wrap the
        # plain local-shaped gradient with the source's routing metadata.
        return _make_handler_output(grad_local, ctx.source)


def _unwrap_grad_aware(t: Any) -> Any:
    """Like :func:`_unwrap` but preserves the wrapper-subclass autograd graph.

    Handlers that compute on the plain local tensor and re-wrap the result
    must route the grad-carrying inputs through this — a plain ``_unwrap``
    returns the autograd-detached ``_local_tensor`` and severs the graph (see
    :class:`_AutogradPreservingUnwrap`). Falls back to plain ``_unwrap`` when
    ``t`` isn't a grad-requiring ShardTensor or grad is disabled (the no-grad
    reference run / inference), where the extra Function is pure overhead.
    """
    from nvalchemi.distributed._core._st_backend import (
        ShardTensor as _UpstreamShardTensor,
    )

    if (
        isinstance(t, _UpstreamShardTensor)
        and t.requires_grad
        and torch.is_grad_enabled()
    ):
        return _AutogradPreservingUnwrap.apply(t)
    return _unwrap(t)


def _wrap_back_to_shardtensor(result: Any, source: "ShardTensor") -> Any:
    """Wrap a plain tensor result from ``func(*local_args, ...)`` back
    into our :class:`ShardTensor` subclass with ``source``'s routing
    metadata. Walks tuples / lists; non-tensor results pass through
    unchanged.
    """
    from nvalchemi.distributed._core.shard_tensor_construction import (  # noqa: PLC0415
        make_local_shard_tensor_spec,
    )

    def _apply(t: Any) -> Any:
        if not isinstance(t, torch.Tensor):
            return t
        if isinstance(t, ShardTensor):
            return t
        out_spec = make_local_shard_tensor_spec(
            t, source._spec.mesh, placements=source._spec.placements
        )
        if t.requires_grad:
            # Route through autograd.Function so grad_fn flows from ``t``
            # through the wrap step into the wrapper output — otherwise
            # ``torch.autograd.grad(wrapper, leaf)`` can't find ``leaf``.
            out = _AutogradPreservingWrap.apply(t, out_spec)
        else:
            out = ShardTensor(local_tensor=t,
                spec=out_spec,
                requires_grad=False,
            )
        for attr in _PROPAGATED_ATTRS:
            setattr(out, attr, getattr(source, attr))
        hmp = getattr(source, "_halo_meta_packed_v", None)
        if hmp is not None:
            out._halo_meta_packed_v = hmp
        return out

    if isinstance(result, torch.Tensor):
        return _apply(result)
    if isinstance(result, tuple):
        converted = [_apply(r) for r in result]
        # Preserve named-tuple-like types (e.g. ``torch.return_types.max``
        # from ``a.max(dim=...)``) so the returned ``.values`` /
        # ``.indices`` access pattern keeps working.
        try:
            return type(result)(converted)
        except TypeError:
            try:
                return type(result)(*converted)
            except TypeError:
                return tuple(converted)
    if isinstance(result, list):
        return [_apply(r) for r in result]
    return result


def _make_handler_output(result: torch.Tensor, source: "ShardTensor") -> "ShardTensor":
    """Wrap a plain handler-result tensor back into a :class:`ShardTensor`,
    copying routing metadata from ``source``.

    Single point of construction for handler outputs. ``source._spec`` supplies
    the mesh + placements for the synthesized output spec; the routing metadata
    (``_distribution_spec``, ``_meta``, ``_config``, …) propagates onto the
    wrapped output.
    """
    from nvalchemi.distributed._core.shard_tensor_construction import (  # noqa: PLC0415
        make_local_shard_tensor_spec,
    )

    out_spec = make_local_shard_tensor_spec(
        result, source._spec.mesh, placements=source._spec.placements
    )
    if result.requires_grad:
        out = _AutogradPreservingWrap.apply(result, out_spec)
    else:
        out = ShardTensor(local_tensor=result,
            spec=out_spec,
            requires_grad=False,
        )
    _propagate_attrs(out, source)
    return out


# ======================================================================
# Predicates — consult ``source._distribution_spec`` to decide routing.
# ======================================================================


def _dim0(args: tuple, kwargs: dict) -> bool:
    """Return True when the scatter / index op targets the leading axis.

    Normalizes negative ``dim`` values against the accumulator's rank so
    callers may pass ``dim=-1`` on a 1-D target — semantically identical
    to ``dim=0`` and commonly emitted by torch-scatter-style helpers
    (e.g. MACE's ``scatter_sum(..., dim=-1, dim_size=num_graphs)`` on a
    1-D ``node_es``). Without this normalization per-system reductions
    silently miss the final per-graph scatter and each rank returns its
    rank-local padded sum.
    """
    dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
    if dim < 0:
        target = args[0] if args else None
        if isinstance(target, torch.Tensor):
            dim = dim + target.ndim
    return dim == 0


def _classify_scatter(args: tuple, kwargs: dict) -> str | None:
    """Classify a dim-0 ``scatter_add_`` / ``index_add_`` / ``index_copy_`` on a
    ShardTensor into the MLIP branch that should handle it, or ``None`` if it is
    not MLIP-routed (falls through to the default dispatch).

    The branch order here *is* the dispatch priority (per-system reduction, then
    halo correction, then distributed scatter). Each branch reads the field's
    declared
    ``_storage_policy`` (+ the spec's ``system_reductions`` flag) and the
    accumulator's role — the shape disambiguation is intrinsic: a per-graph
    reduction and a per-atom halo scatter are both dim-0 ``scatter_add`` and
    differ only in the accumulator's row count.
    """
    if not args or not _dim0(args, kwargs):
        return None
    accumulator = args[0]

    # 1. Per-system reduction: accumulator rows == n_systems, when the spec
    #    declares system reductions.
    source = _prefer_source(args)
    if (
        source is not None
        and source._distribution_spec is not None
        and source._distribution_spec.system_reductions
        and source._n_systems is not None
        and accumulator.shape[0] == source._n_systems
    ):
        return "per_system"

    # 2. Halo correction: accumulator rows == padded halo shape, halo policy.
    halo_source = _find_source(args)
    if halo_source is not None:
        policy = halo_source._storage_policy
        if (
            isinstance(policy, HaloStoragePolicy)
            and policy.scatter_mode == "halo_correction"
            and halo_source._meta is not None
            and accumulator.shape[0]
            == halo_source._meta.n_padded + halo_source._extra_suffix_padding
        ):
            return "halo"

    # 3. Distributed scatter: accumulator is a sharded ShardTensor.
    if isinstance(accumulator, ShardTensor) and isinstance(
        accumulator._storage_policy, PlainShard
    ):
        if accumulator._gather_meta is not None:
            return "distributed"

    return None


def _debug_log_unrouted_scatter(func: Any, args: tuple, kwargs: dict) -> None:
    """Debug breadcrumb for a scatter on a halo-correction ShardTensor that the
    classifier did not route (typically ``dim != 0``).

    A ``dim != 0`` scatter operates within the feature axis — it is local,
    needs no halo synchronization, and is correct via the default dispatch — so
    this is *not* a warning. It is a quiet, opt-in diagnostic (visible only at
    DEBUG) for the rare case where the breadcrumb helps explain why a particular
    scatter didn't take the halo-correction path.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    source = _find_source(args)
    if source is None:
        return
    policy = source._storage_policy
    if not isinstance(policy, HaloStoragePolicy) or policy.scatter_mode != "halo_correction":
        return
    dim = args[1] if len(args) > 1 and isinstance(args[1], int) else kwargs.get("dim")
    logger.debug(
        "scatter %s on a halo-correction ShardTensor not routed (dim=%s); "
        "running default (local) dispatch — no halo synchronization.",
        getattr(func, "__qualname__", str(func)),
        dim,
    )


def _shard_gather_branch(input_t: Any) -> str | None:
    """Classify the STORAGE of a dim-0 gather input into the MLIP branch that
    handles it (``"halo"`` halo-read refresh, ``"distributed"`` cross-rank
    gather), or ``None`` if it is not MLIP-routed. Branch order is the dispatch
    priority (halo before distributed).

    Shared by :func:`_classify_index_select` (``aten.index_select`` dispatch)
    and :meth:`ShardTensor.__getitem__` (which rewrites a halo-refresh
    ``node_feats[sender]`` advanced index to ``index_select``) so both gather
    forms route the borrowed-row backward through the same reverse-exchange."""
    if not isinstance(input_t, ShardTensor):
        return None
    policy = input_t._storage_policy

    # Halo-read gather: input is a halo ShardTensor at the padded shape.
    if (
        isinstance(policy, HaloStoragePolicy)
        and policy.gather_mode == "halo_read"
        and input_t._meta is not None
        and input_t.shape[0]
        == input_t._meta.n_padded + input_t._extra_suffix_padding
    ):
        return "halo"

    # Distributed gather: input is a sharded ShardTensor.
    if isinstance(policy, PlainShard) and input_t._gather_meta is not None:
        return "distributed"

    return None


def _classify_index_select(args: tuple, kwargs: dict) -> str | None:
    """Classify a dim-0 ``index_select`` on a ShardTensor into the MLIP branch
    that handles it, or ``None`` if it is not MLIP-routed."""
    if not args or not _dim0(args, kwargs):
        return None
    return _shard_gather_branch(args[0])


# ======================================================================
# Handlers.
# ======================================================================


def _halo_scatter_correction(
    self_t: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    *,
    reduce: str | None = None,
) -> "ShardTensor":
    """Halo correction on a per-atom scatter_add_ / index_add_ / index_copy_.

    Drops in-place semantics — returns a fresh tensor via the functional
    form matching the caller's op (``scatter_add`` uses a full-shape
    ``index``, ``index_add`` uses a 1-D ``index``) so the autograd tape
    chains cleanly into :func:`halo_reverse_exchange` +
    :func:`halo_forward_exchange`.
    """
    from nvalchemi.distributed._core.particle_halo import (
        halo_forward_exchange,
        halo_reverse_exchange,
    )

    source = _find_source((self_t, index, src))

    self_plain = _unwrap(self_t)
    # ``src`` carries the model's autograd; unwrap grad-aware so scatter +
    # halo-correction stay connected to the wrapper graph (positions→energy).
    src_plain = _unwrap_grad_aware(src)
    index_plain = _unwrap(index)

    # scatter_add requires ``index.shape == src.shape``; index_add /
    # index_copy supply a 1-D index of length ``src.shape[dim]``. The
    # registered-ops tuple (_SCATTER_OP_NAMES) covers both — discriminate
    # by index rank so UMA's edge→node ``index_add_`` doesn't get
    # shoehorned into scatter_add's shape contract.
    if index_plain.ndim == src_plain.ndim:
        result = torch.scatter_add(self_plain, dim, index_plain, src_plain)
    elif index_plain.ndim == 1:
        result = torch.index_add(self_plain, dim, index_plain, src_plain)
    else:
        raise RuntimeError(
            "halo-correction scatter: index has ndim="
            f"{index_plain.ndim}, incompatible with src ndim="
            f"{src_plain.ndim} (expected equal or 1-D index)"
        )

    halo_corrected = dim == 0 and result.shape[0] == source._meta.n_padded
    if halo_corrected:
        owned = halo_reverse_exchange(result, source._meta, source._config)
        result = halo_forward_exchange(owned, source._meta, source._config)

    from nvalchemi.distributed._core.dispatch_trace import (  # noqa: PLC0415
        is_tracing,
        record_dispatch,
    )

    if is_tracing():
        record_dispatch(
            "halo_scatter_correction",
            branch="halo_reverse+halo_forward" if halo_corrected else "scatter_only",
            shapes={
                "self": tuple(self_plain.shape),
                "index": tuple(index_plain.shape),
                "src": tuple(src_plain.shape),
            },
            meta={
                "dim": dim,
                "n_owned": source._meta.n_owned,
                "n_padded": source._meta.n_padded,
            },
        )

    return _make_handler_output(result, source)


def _per_system_reduce_handler(
    self_t: torch.Tensor,
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    *,
    reduce: str | None = None,
) -> torch.Tensor:
    """Route a per-system scatter_add_ / index_add_ through
    :func:`per_system_reduce`.

    In-place semantics: the MLIP idiom is ``res.scatter_add_(...); return
    res`` with the return value of scatter_add_ discarded. We mutate
    ``self_t`` in place with the computed global sum so the bound variable
    on the caller's side ends up with the right value.
    """
    from nvalchemi.distributed._core.per_system import per_system_reduce

    source = _prefer_source((self_t, index, src))
    if source is None or source._n_systems is None:
        raise RuntimeError(
            "per_system_reduce handler invoked without a ShardTensor "
            "carrying n_systems; this should be predicate-gated."
        )

    self_plain = _unwrap(self_t)
    # ``src`` carries the model's autograd; unwrap grad-aware so the reduction
    # stays connected to the wrapper graph (e.g. node energies → energy).
    src_plain = _unwrap_grad_aware(src)
    index_plain = _unwrap(index)

    if not torch.equal(self_plain, torch.zeros_like(self_plain)):
        raise RuntimeError(
            "per_system_reduce handler requires the scatter accumulator to "
            "be zero-initialized. Detected a non-zero initial value on "
            f"shape={tuple(self_plain.shape)}."
        )

    index_1d = index_plain[:, 0] if index_plain.ndim > 1 else index_plain

    # Halo-mode: slice off halo rows; they are contributed by their owner,
    # not by a borrower.
    if source._meta is not None:
        n_owned = source._meta.n_owned
        if src_plain.shape[0] > n_owned:
            src_plain = src_plain[:n_owned]
            index_1d = index_1d[:n_owned]

    if _os.environ.get("NVALCHEMI_REDUCE_DEBUG"):
        import torch.distributed as _td

        rank = _td.get_rank() if _td.is_initialized() else 0
        local_pre = src_plain.detach().to(torch.float64).sum().item()
        print(
            f"[reduce-debug rank {rank}] _per_system_reduce_handler FIRED  "
            f"src.shape={tuple(src_plain.shape)} accum.shape={tuple(self_plain.shape)}  "
            f"src.sum(rank-local, owned-only)={local_pre:+.6e}",
            flush=True,
        )

    from nvalchemi.distributed._core.dispatch_trace import (  # noqa: PLC0415
        is_tracing,
        record_dispatch,
    )

    if is_tracing():
        record_dispatch(
            "per_system_reduce",
            branch="owned_slice+all_reduce",
            shapes={
                "accumulator": tuple(self_plain.shape),
                "src_post_slice": tuple(src_plain.shape),
                "index_post_slice": tuple(index_1d.shape),
            },
            meta={
                "n_systems": source._n_systems,
                "n_owned": source._meta.n_owned if source._meta is not None else None,
            },
        )

    result = per_system_reduce(src_plain, index_1d, source._n_systems, source._config)
    # In-place semantics: callers that do ``acc.scatter_add_(...)`` and
    # discard the return expect ``acc`` to hold the reduced values.
    # ``copy_`` writes through into ``self_t._local_tensor``'s storage.
    self_t.copy_(result)
    # Return a fresh wrapper around ``result``: ``self_t`` is a wrapper-subclass
    # leaf whose ``grad_fn`` is None even after ``copy_`` from an
    # autograd-connected source. Wrapping ``result`` directly preserves
    # ``_local_tensor.grad_fn`` so callers that USE the return
    # (``y = acc.scatter_add_(...)``) can run ``torch.autograd.grad(y, leaf)``
    # through per_system_reduce's autograd.Function back to the source.
    if not isinstance(self_t, ShardTensor):
        return self_t
    return _make_handler_output(result, self_t)


def _distributed_scatter_add_handler(
    self_t: "ShardTensor",
    dim: int,
    index: torch.Tensor,
    src: torch.Tensor,
    *,
    reduce: str | None = None,
) -> "ShardTensor":
    """Route ``self_t.scatter_add_(0, global_indices, src)`` through
    :func:`distributed_scatter_add`. The MLIP idiom uses ``index =
    system_index.unsqueeze(-1).expand(-1, F)`` (2-D); flatten to 1-D.
    """
    from nvalchemi.distributed._core.gather_primitives import distributed_scatter_add

    # Grad-aware on the accumulator + source (both can carry the model's
    # autograd); index is integer. Plain ``_unwrap`` would sever the graph and
    # break autograd-derived forces.
    self_plain = _unwrap_grad_aware(self_t)
    src_plain = _unwrap_grad_aware(src)
    index_plain = _unwrap(index)
    if index_plain.ndim > 1:
        index_plain = index_plain[:, 0]

    out = distributed_scatter_add(
        self_plain, index_plain, src_plain, self_t._gather_meta, self_t._config
    )

    from nvalchemi.distributed._core.dispatch_trace import (  # noqa: PLC0415
        is_tracing,
        record_dispatch,
    )

    if is_tracing():
        record_dispatch(
            "distributed_scatter_add",
            branch="all_to_all",
            shapes={
                "self": tuple(self_plain.shape),
                "index": tuple(index_plain.shape),
                "src": tuple(src_plain.shape),
            },
            meta={"dim": dim},
        )

    return _make_handler_output(out, self_t)


def _halo_forward_sync_before_index_select(
    input_t: "ShardTensor",
    dim: int,
    index: torch.Tensor,
) -> "ShardTensor":
    """Refresh halo rows before a gather. For halo-storage models whose
    per-layer update isn't a scatter (so halo rows drift stale between
    layers). Reads metadata directly from ``input_t``.
    """
    from nvalchemi.distributed._core.particle_halo import halo_forward_exchange

    # Grad-aware: a gather feeding the model's autograd path (e.g. node
    # features) must keep the wrapper graph connected through the index_select.
    input_plain = _unwrap_grad_aware(input_t)

    n_owned = input_t._meta.n_owned
    n_halo_padded = input_t._meta.n_padded

    # Short-circuit when there is no cross-rank halo.
    if n_halo_padded == n_owned:
        return _make_handler_output(
            torch.index_select(input_plain, dim, index), input_t
        )

    # Coord is halo-exchanged at setup with PBC shifts; re-fetching would
    # drop the shift correction. Skip the (N, 3) signature.
    if input_plain.ndim == 2 and input_plain.shape[-1] == 3:
        return _make_handler_output(
            torch.index_select(input_plain, dim, index), input_t
        )

    owned = input_plain[:n_owned].contiguous()
    refreshed_halo_padded = halo_forward_exchange(owned, input_t._meta, input_t._config)
    if input_t._extra_suffix_padding > 0:
        refreshed = torch.cat(
            [refreshed_halo_padded, input_plain[n_halo_padded:]], dim=0
        )
    else:
        refreshed = refreshed_halo_padded

    return _make_handler_output(torch.index_select(refreshed, dim, index), input_t)


def _distributed_index_select_handler(
    input_t: "ShardTensor",
    dim: int,
    index: torch.Tensor,
) -> "ShardTensor":
    """Route a dim-0 index_select on a sharded ShardTensor through
    :func:`distributed_index_select`. The index tensor contains GLOBAL
    atom IDs.
    """
    from nvalchemi.distributed._core.gather_primitives import distributed_index_select

    # Grad-aware: the gathered features feed the model's autograd path
    # (conservative forces = autograd.grad(energy, positions)); a plain
    # ``_unwrap`` returns the detached local and severs the graph.
    input_plain = _unwrap_grad_aware(input_t)
    out = distributed_index_select(
        input_plain, index, input_t._gather_meta, input_t._config
    )

    from nvalchemi.distributed._core.dispatch_trace import (  # noqa: PLC0415
        is_tracing,
        record_dispatch,
    )

    if is_tracing():
        record_dispatch(
            "distributed_index_select",
            branch="all_to_all",
            shapes={
                "input": tuple(input_plain.shape),
                "index": tuple(index.shape),
                "out": tuple(out.shape),
            },
            meta={"dim": dim},
        )

    return _make_handler_output(out, input_t)


# Aten scatter overloads that, on a halo-correction ShardTensor, need the
# cross-rank halo reverse+forward applied to the local result under compile (the
# eager equivalent, _halo_scatter_correction, is bypassed under compile because
# Dynamo cannot trace its manual ShardTensor construction).
_HALO_SCATTER_OVERLOADS = frozenset(
    {
        torch.ops.aten.scatter_add_.default,
        torch.ops.aten.scatter_add.default,
        torch.ops.aten.index_add_.default,
        torch.ops.aten.index_add.default,
    }
)


def _dispatch_halo_scatter_correct(
    func: Any, args: tuple, kwargs: dict, local_result: Any, source: Any
) -> Any:
    """Return the halo-corrected scatter result, or ``None`` if not applicable.

    Mirrors the eager :func:`_halo_scatter_correction` at the
    ``__torch_dispatch__`` level for the compile path: applies
    :func:`halo_reverse_exchange` + :func:`halo_forward_exchange` (funcol-backed,
    AOT-traceable) to the plain local scatter ``local_result`` so contributions
    written into borrowed halo rows fold back into their owners. Returns ``None``
    when the op isn't a halo-correction scatter (caller keeps the default path).
    World=1 is a structural no-op (no halo rows), but the classifier still gates
    it so the fast path is unchanged.
    """
    if (
        source is None
        or getattr(source, "_meta", None) is None
        or func not in _HALO_SCATTER_OVERLOADS
        or not isinstance(local_result, torch.Tensor)
        or local_result.shape[0] != source._meta.n_padded
        or _classify_scatter(args, kwargs) != "halo"
    ):
        return None
    hmp = getattr(source, "_halo_meta_packed", None)
    if hmp is not None and hmp.numel() > 0:
        from nvalchemi.distributed._core.particle_halo import (  # noqa: PLC0415
            halo_scatter_correct_static_op,
            unpack_halo_meta,
        )

        si, rd, rr, no = unpack_halo_meta(hmp)
        return halo_scatter_correct_static_op(
            local_result, si, rd, rr, no, len(source._meta.send_sizes)
        )
    from nvalchemi.distributed._core.particle_halo import (  # noqa: PLC0415
        halo_scatter_correct_compiled,
    )

    # Single dispatcher-visible custom op (opaque to fake mode; marker indices
    # ride as a flat tensor arg) — equals halo_forward(halo_reverse(local)).
    return halo_scatter_correct_compiled(local_result, source._meta, source._config)


def _under_compile_trace(args: Any) -> bool:
    """True when an op is being traced/decomposed for ``torch.compile``.

    ``torch.compiler.is_compiling()`` only covers Dynamo's bytecode trace; the
    subclass ops are decomposed later, during AOTAutograd's fake-tensor
    propagation, where it returns False. Detect an active ``FakeTensorMode``
    (and fake operands) so the compile gate fires in BOTH phases.
    """
    import torch  # noqa: PLC0415

    if torch.compiler.is_compiling():
        return True
    try:
        from torch._guards import detect_fake_mode  # noqa: PLC0415

        if detect_fake_mode(args) is not None:
            return True
    except Exception:  # noqa: BLE001
        pass
    from torch._subclasses.fake_tensor import FakeTensor  # noqa: PLC0415

    stack = list(args)
    while stack:
        a = stack.pop()
        if isinstance(a, FakeTensor):
            return True
        local = getattr(a, "_local_tensor", None)
        if isinstance(local, FakeTensor):
            return True
        if isinstance(a, (list, tuple)):
            stack.extend(a)
    return False


_INDEX_SELECT_OVERLOADS = frozenset({torch.ops.aten.index_select.default})
# Plain advanced indexing ``x[idx]`` (``node_feats[sender]``) lowers to
# ``aten.index.Tensor``, whose adjoint (``index_put``) drops the upstream
# halo-gather custom-op backward under AOTAutograd. Rather than route it here,
# ``ShardTensor.__getitem__`` rewrites a dim-0 integer halo-refresh gather to
# ``index_select`` at the Dynamo graph-build level (the only layer where the
# autograd structure can still be fixed). See ``ShardTensor.__getitem__``.


import contextlib as _contextlib  # noqa: E402


@_contextlib.contextmanager
def _allow_real_constants(ref: Any):
    """Admit real constant tensors (routing metadata never lifted by Dynamo)
    into the trace for the duration of a custom-op call. No-op outside compile.
    Mirrors the halo marker marshalling."""
    fm = None
    try:
        from torch._guards import detect_fake_mode  # noqa: PLC0415

        fm = detect_fake_mode((ref,))
    except Exception:  # noqa: BLE001
        fm = None
    if fm is None:
        yield
        return
    prev = fm.allow_non_fake_inputs
    fm.allow_non_fake_inputs = True
    try:
        yield
    finally:
        fm.allow_non_fake_inputs = prev


def _dispatch_distributed_scatter(func: Any, args: tuple, kwargs: dict, source: Any) -> Any:
    """Compile-path distributed scatter-add via the custom op, or ``None``."""
    if (
        source is None
        or getattr(source, "_gather_meta", None) is None
        or func not in _HALO_SCATTER_OVERLOADS
        or _classify_scatter(args, kwargs) != "distributed"
    ):
        return None
    self_t, _dim, index, src = args[0], args[1], args[2], args[3]
    self_plain = _strip_to_local(self_t)
    src_plain = _strip_to_local(src)
    index_plain = _strip_to_local(index)
    if index_plain.ndim > 1:
        index_plain = index_plain[:, 0]
    import torch.distributed as _dist  # noqa: PLC0415

    world_size = _dist.get_world_size() if _dist.is_initialized() else 1
    _routing = source._halo_meta_packed
    _ng = _routing.shape[0] // 2
    owner_rank, local_index = _routing[:_ng], _routing[_ng:]
    from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
        distributed_scatter_add_op,
    )

    with _allow_real_constants(self_plain):
        return distributed_scatter_add_op(
            self_plain, index_plain, src_plain, owner_rank, local_index, world_size
        )


def _dispatch_distributed_gather(func: Any, args: tuple, kwargs: dict, source: Any) -> Any:
    """Compile-path distributed index_select via the custom op, or ``None``."""
    if (
        source is None
        or getattr(source, "_gather_meta", None) is None
        or func not in _INDEX_SELECT_OVERLOADS
        or _classify_index_select(args, kwargs) != "distributed"
    ):
        return None
    input_t, _dim, index = args[0], args[1], args[2]
    input_plain = _strip_to_local(input_t)
    index_plain = _strip_to_local(index)
    import torch.distributed as _dist  # noqa: PLC0415

    world_size = _dist.get_world_size() if _dist.is_initialized() else 1
    # owner_rank / local_index ride as the _halo_meta_packed inner tensor
    # (fakified graph input under compile) rather than gm.* python-attr
    # tensors (baked _tensor_constants Inductor's lowering rejects). The slot
    # holds owner_rank || local_index, each (n_global,); split at the half.
    _routing = source._halo_meta_packed
    _ng = _routing.shape[0] // 2
    owner_rank, local_index = _routing[:_ng], _routing[_ng:]
    from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
        distributed_index_select_op,
    )

    with _allow_real_constants(input_plain):
        return distributed_index_select_op(
            input_plain, index_plain, owner_rank, local_index, world_size
        )


def _dispatch_halo_gather(func: Any, args: tuple, kwargs: dict, source: Any) -> Any:
    """Return the halo-refreshed index_select result via the
    ``nvalchemi::halo_forward`` custom op, or ``None`` if not applicable.

    Compile-path analogue of :func:`_halo_forward_sync_before_index_select`:
    refresh the borrowed halo rows (cross-rank, via the custom op) then gather by
    index. Short-circuits when there is no halo or for the PBC-shifted ``(N, 3)``
    coordinate field (re-fetching would drop the shift), matching the eager path.
    """
    if (
        source is None
        or getattr(source, "_meta", None) is None
        or func not in _INDEX_SELECT_OVERLOADS
        or _classify_index_select(args, kwargs) != "halo"
    ):
        return None
    input_t, dim, index = args[0], args[1], args[2]
    input_plain = _strip_to_local(input_t)
    index_plain = _strip_to_local(index)
    n_owned = source._meta.n_owned
    n_padded = source._meta.n_padded
    if n_padded == n_owned or (input_plain.ndim == 2 and input_plain.shape[-1] == 3):
        return torch.index_select(input_plain, dim, index_plain)
    hmp = getattr(source, "_halo_meta_packed", None)
    if hmp is not None and hmp.numel() > 0 and source._extra_suffix_padding == 0:
        # Fixed-shape path: routing rides as a graph-input tensor
        # (_halo_meta_packed), so it can't go stale / force recompiles the way
        # the baked list[int] markers do under torch.compile.
        from nvalchemi.distributed._core.particle_halo import (  # noqa: PLC0415
            halo_forward_static_op,
            unpack_halo_meta,
        )

        si, rd, rr, no = unpack_halo_meta(hmp)
        refreshed = halo_forward_static_op(
            input_plain, si, rd, rr, no, len(source._meta.send_sizes)
        )
        return torch.index_select(refreshed, dim, index_plain)
    from nvalchemi.distributed._core.particle_halo import (  # noqa: PLC0415
        halo_forward_compiled,
    )

    owned = input_plain[:n_owned].contiguous()
    refreshed = halo_forward_compiled(owned, source._meta, source._config)
    if source._extra_suffix_padding > 0:
        refreshed = torch.cat([refreshed, input_plain[n_padded:]], dim=0)
    return torch.index_select(refreshed, dim, index_plain)


def _dispatch_per_system_reduce(
    func: Any, args: tuple, kwargs: dict, source: Any
) -> Any:
    """Return the per-system reduced scatter result via the
    ``nvalchemi::per_system_reduce`` custom op, or ``None`` if not applicable.

    The compile-path analogue of :func:`_per_system_reduce_handler`: opaque to
    fake mode, autograd-correct below autograd (the custom op carries the
    cross-rank ``all_reduce`` adjoint), so it works under ``torch.compile`` where
    the eager ``__torch_function__`` reduce handler is bypassed. Marshals the
    owned-only per-atom values + system index into the custom op (``n_systems``
    as an int constant).
    """
    spec = getattr(source, "_distribution_spec", None) if source is not None else None
    if (
        spec is None
        or not getattr(spec, "system_reductions", False)
        or getattr(source, "_n_systems", None) is None
        or func not in _HALO_SCATTER_OVERLOADS
        or _classify_scatter(args, kwargs) != "per_system"
    ):
        return None
    _self_t, _dim, index, src = args[0], args[1], args[2], args[3]
    src_plain = _strip_to_local(src)
    index_plain = _strip_to_local(index)
    index_1d = index_plain[:, 0] if index_plain.ndim > 1 else index_plain
    # Halo mode: drop borrowed halo rows (the owner contributes them).
    if source._meta is not None and src_plain.shape[0] > source._meta.n_owned:
        n_owned = source._meta.n_owned
        src_plain = src_plain[:n_owned]
        index_1d = index_1d[:n_owned]
    from nvalchemi.distributed._core.per_system import (  # noqa: PLC0415
        per_system_reduce_op,
    )

    return per_system_reduce_op(
        src_plain.contiguous(), index_1d.contiguous(), int(source._n_systems)
    )


def _scatter_add_dispatch(func: Any, types: Any, args: tuple, kwargs: dict) -> Any:
    """Single scatter-family handler (one per op, branching on placement/kind).

    A base-signature function handler: it self-classifies and, when the scatter
    isn't MLIP-routed, falls through to the default dispatch (disabling
    torch_function routes the op to our ``__torch_dispatch__`` extract-local
    path).
    """
    if _under_compile_trace(args):
        # Under torch.compile the __torch_function__ scatter handlers manually
        # construct ShardTensors (untraceable by dynamo) and the op carries
        # mixed plain+ShardTensor operands that cannot be fake-traced here. Fall
        # through to __torch_dispatch__ — dynamo's traceable wrapper-subclass
        # path, where halo correction runs via functional collectives; at
        # world=1 the extract-local fallback is already numerically exact.
        return torch._C._disabled_torch_function_impl(func, types, args, kwargs)

    branch = _classify_scatter(args, kwargs)
    if branch == "per_system":
        return _per_system_reduce_handler(*args, **kwargs)
    if branch == "halo":
        # Route through the field's declared storage policy: HaloStoragePolicy
        # carries the overlay-aware scatter (halo correction).
        source = _find_source(args)
        policy = getattr(source, "_storage_policy", None)
        if isinstance(policy, HaloStoragePolicy):
            return policy.scatter(args[0], args[1], args[2], args[3])
        return _halo_scatter_correction(*args, **kwargs)
    if branch == "distributed":
        source = args[0]
        policy = getattr(source, "_storage_policy", None)
        if isinstance(policy, PlainShard):
            return policy.scatter(args[0], args[1], args[2], args[3])
        return _distributed_scatter_add_handler(*args, **kwargs)
    _debug_log_unrouted_scatter(func, args, kwargs)
    return torch._C._disabled_torch_function_impl(func, types, args, kwargs)


def _index_select_dispatch(func: Any, types: Any, args: tuple, kwargs: dict) -> Any:
    """Single index_select handler (one per op, branching on placement/kind).

    Base-signature; self-classifies and falls through to default dispatch when
    the gather isn't MLIP-routed.
    """
    if _under_compile_trace(args):
        # Under compile the eager gather handlers manually construct
        # ShardTensors / use marker-based autograd.Functions (fake-prop faults).
        # Fall through to __torch_dispatch__ — the halo gather is handled there
        # by the nvalchemi::halo_forward custom op.
        return torch._C._disabled_torch_function_impl(func, types, args, kwargs)
    branch = _classify_index_select(args, kwargs)
    if branch == "halo":
        # Route through the declared storage policy: HaloStoragePolicy refreshes
        # the borrowed halo rows, then gathers by index.
        input_t = args[0]
        policy = getattr(input_t, "_storage_policy", None)
        if isinstance(policy, HaloStoragePolicy):
            return policy.gather(args[0], args[1], args[2])
        return _halo_forward_sync_before_index_select(*args, **kwargs)
    if branch == "distributed":
        input_t = args[0]
        policy = getattr(input_t, "_storage_policy", None)
        if isinstance(policy, PlainShard):
            return policy.gather(args[0], args[1], args[2])
        return _distributed_index_select_handler(*args, **kwargs)
    return torch._C._disabled_torch_function_impl(func, types, args, kwargs)


def _unwrap(t: Any) -> Any:
    """Return the plain ``torch.Tensor`` storage of a ShardTensor — used
    by handlers that pass tensors to primitives that don't tolerate
    subclasses (Warp kernels, autograd.Function subclasses).

    Returns ``t._local_tensor`` since our class is a wrapper-subclass.
    """
    from nvalchemi.distributed._core._st_backend import (
        ShardTensor as _UpstreamShardTensor,
    )

    if isinstance(t, _UpstreamShardTensor):
        return t._local_tensor
    return t


# ======================================================================
# The class.
# ======================================================================


from nvalchemi.distributed._core._st_backend import (  # noqa: E402  # mid-file: helpers above reference it
    ShardTensor as _UpstreamShardTensor,
)


class ShardTensor(_UpstreamShardTensor):
    """Spec-driven distributed tensor.

    Carries routing metadata as instance attributes; dispatch handlers read
    directly from the tensor. Created via :meth:`wrap`; propagates metadata
    to outputs of ops via ``__torch_function__``.

    Halo usage::

        padded = particle_halo_padding_autograd(local_feats, cfg)
        x = ShardTensor.wrap(padded, spec=SPEC_MPNN_HALO, meta=meta, config=cfg)
        out = model(x, ...)   # scatter_add_ fires halo-correction handler

    Sharded usage::

        gather_meta = ShardRouting.from_assignment(assignment, rank)
        x = ShardTensor.wrap(
            owned_feats, spec=SPEC_AIMNET2_SHARDED,
            gather_meta=gather_meta, config=cfg,
        )
        neighbors = x.index_select(0, global_nbmat.flatten())  # auto-routed

    For MLIP-style system reductions (aimnet's ``mol_sum``), additionally
    pass ``n_systems=`` and ``system_index=`` to ``wrap`` so the per-system
    scatter predicate fires.
    """

    # Class-level defaults so metadata-less views don't AttributeError on
    # lookup. ``_distribution_spec`` is duck-typed and opaque — dispatch reads
    # only ``.system_reductions``, and ``wrap()`` reads ``.distribution.policy``.
    # Named ``_distribution_spec`` (not ``_spec``) to avoid collision with the
    # base ``ShardTensor._spec``.
    _distribution_spec: Any = None
    _config: "ParticleHaloConfig | None" = None
    _meta: "ParticleHaloMetadata | None" = None
    _gather_meta: "ShardRouting | None" = None
    _n_systems: int | None = None
    _system_index: torch.Tensor | None = None
    _extra_suffix_padding: int = 0
    # Declared storage policy, set at wrap time from the field's halo / sharded
    # nature. Carries the honest semantics (e.g. ``HaloStoragePolicy`` =
    # Shard(0) owned rows + a borrowed overlay) and the overlay-aware ops.
    # ``None`` for plain metadata-propagating views.
    _storage_policy: Any = None
    # The pre-wrap tensor, preserved so :func:`torch.autograd.grad` against the
    # ShardTensor view can be routed to the underlying in-graph tensor. ``None``
    # for ShardTensors not produced via ``wrap``.
    _autograd_source: torch.Tensor | None = None
    # Fixed-shape inner-halo routing, packed into one int64 tensor. Rides as an
    # UNCONDITIONAL 2nd flatten inner tensor (graph input under compile, not a
    # baked mlip_ctx constant) so the subclass attr count is STABLE across the
    # forward trace AND every backward tangent — avoiding AOT's
    # ``len(meta.attrs) == len(runtime_subclass_keys)`` assert. The property
    # returns the stored routing, or a per-instance empty sentinel created in
    # ``_local_tensor``'s fake/real context (a shared real sentinel would mix
    # fake+real inner tensors under tracing). Backing store: _halo_meta_packed_v.
    _halo_meta_packed_v: "torch.Tensor | None" = None
    _halo_meta_packed_c: "torch.Tensor | None" = None

    @property
    def _halo_meta_packed(self) -> "torch.Tensor":
        v = self._halo_meta_packed_v
        if v is not None:
            return v
        # CACHE the sentinel per-instance: AOT flattens an input subclass several
        # times and asserts the inner tensors are identical across calls, so the
        # getter must return the SAME object (not a fresh one each call).
        c = self._halo_meta_packed_c
        if c is None:
            c = torch.zeros(0, dtype=torch.int64, device=self._local_tensor.device)
            self._halo_meta_packed_c = c
        return c

    @_halo_meta_packed.setter
    def _halo_meta_packed(self, value: "torch.Tensor | None") -> None:
        self._halo_meta_packed_v = value
        self._halo_meta_packed_c = None

    # ``__new__`` is inherited from the vendored base: its ``torch.Tensor``-based
    # ``_make_wrapper_subclass`` + C-level ``requires_grad`` setter passes our
    # fwd+bwd fullgraph smoke. Signature:
    # ``__new__(cls, local_tensor, spec, *, requires_grad)``.

    def __tensor_flatten__(self) -> tuple:
        # Delegate the base context (spec, requires_grad) and bundle MLIP routing
        # metadata alongside it so it survives the Dynamo flatten/unflatten
        # round-trip. Only ``_local_tensor`` is a graph-traced inner tensor; the
        # MLIP metadata are per-compile graph-constants (set at sim init, read by
        # dispatch handlers before AOT lowering), so they go opaquely in context.
        # Context shape: ``(base_ctx, mlip_ctx)``.
        inner_names, base_ctx = super().__tensor_flatten__()
        # _halo_meta_packed is ALWAYS a 2nd inner tensor (graph input under
        # compile); the property guarantees a tensor (sentinel when unset) so the
        # attr count is uniform across all ShardTensors / tangents.
        inner_names = list(inner_names) + ["_halo_meta_packed"]
        mlip_ctx = tuple(getattr(self, attr) for attr in _PROPAGATED_ATTRS)
        return inner_names, (base_ctx, mlip_ctx)

    @classmethod
    def __metadata_guard__(cls, orig: Any, other: Any) -> bool:
        # Dynamo's tensor-subclass metadata guard. ``orig`` / ``other`` are the
        # context our ``__tensor_flatten__`` emits: ``((spec, requires_grad),
        # mlip_ctx)``. The default guard (``==`` against a deepcopy) fails here
        # because ``mlip_ctx`` holds tensors / objects without value-equality
        # (``_meta`` / ``_gather_meta`` routing tensors), so the context never
        # equals its own deepcopy and the guard fails on the frame it was
        # created — blocking ``torch.compile`` on the distributed path. The MLIP
        # metadata are per-compile graph-constants (set at sim init, read by
        # dispatch handlers before AOT lowering); genuine shape / placement
        # changes are caught by the base ``ShardTensorSpec`` (its
        # ``_sharding_shapes`` + mesh + placements) and dynamo's own size
        # guards. So guard only on ``(spec, requires_grad)`` — mirrors DTensor's
        # ``__metadata_guard__`` (guards the DTensorSpec).
        try:
            (orig_base, _orig_mlip), (other_base, _other_mlip) = orig, other
            orig_spec, orig_rg = orig_base
            other_spec, other_rg = other_base
        except (TypeError, ValueError):
            return bool(orig == other)
        return bool(orig_rg == other_rg) and bool(orig_spec == other_spec)

    @staticmethod
    def __tensor_unflatten__(
        inner_tensors: dict,
        flatten_spec: Any,
        outer_size: Any,
        outer_stride: Any,
    ) -> "ShardTensor":
        # Delegate spec reconstruction to the vendored base — it carries the
        # ``_sharding_shapes`` tuple-normalization / chunk-derivation that
        # keeps AOT tracing off ``PendingUnbackedSymbolNotFound``. Then coerce
        # the base instance back to OUR subclass (``__class__`` reassignment is
        # safe — same instance layout, no added slots; see ``_propagate_attrs``)
        # and reattach MLIP metadata.
        from nvalchemi.distributed._core._st_backend import (  # noqa: PLC0415
            ShardTensor as _Upstream,
        )

        # Accept our nested ``(base_ctx, mlip_ctx)`` or the base's bare
        # ``(spec, requires_grad)`` (Dynamo may flatten via the base if our
        # override wasn't bound). Our nesting has a tuple at [0]; the base's
        # has a ShardTensorSpec there.
        if isinstance(flatten_spec[0], tuple):
            base_ctx, mlip_ctx = flatten_spec
        else:
            base_ctx, mlip_ctx = flatten_spec, None
        out = _Upstream.__tensor_unflatten__(
            inner_tensors, base_ctx, outer_size, outer_stride
        )
        out.__class__ = ShardTensor
        if mlip_ctx is not None:
            for attr, value in zip(_PROPAGATED_ATTRS, mlip_ctx, strict=True):
                if value is not None:
                    setattr(out, attr, value)
        hmp = inner_tensors.get("_halo_meta_packed")
        if hmp is not None:
            out._halo_meta_packed = hmp
        return out

    def __coerce_same_metadata_as_tangent__(
        self, flatten_spec: Any, expected_type: Any = None
    ) -> "ShardTensor | None":
        # AOTAutograd calls this when the runtime backward tangent's metadata
        # doesn't match what was traced. ``flatten_spec`` is whatever OUR
        # ``__tensor_flatten__`` emitted, i.e. the nested ``(base_ctx,
        # mlip_ctx)``. Unwrap it and delegate the redistribute to the vendored
        # base (it threads per-tensor-dim shard sizes through and preserves
        # uneven layouts), then coerce the result back to our subclass and
        # reattach MLIP metadata.
        from nvalchemi.distributed._core._st_backend import (  # noqa: PLC0415
            ShardTensor as _Upstream,
        )

        # AOT expected a PLAIN tensor tangent (``PlainTensorMeta`` -> ``flatten_spec``
        # is None and/or ``expected_type`` is the base ``torch.Tensor``, not a
        # ShardTensor) but the RUNTIME tangent is this ShardTensor: coerce DOWN to
        # the local plain tensor. Happens when the compiled fn's ShardTensor output
        # is consumed by an eager ``to_local`` whose backward feeds a ShardTensor
        # cotangent into the AOT backward, which traced a plain output tangent.
        if flatten_spec is None or (
            expected_type is not None
            and not (
                isinstance(expected_type, type)
                and issubclass(expected_type, _Upstream)
            )
        ):
            return self._local_tensor

        if isinstance(flatten_spec[0], tuple):
            base_ctx, mlip_ctx = flatten_spec
        else:
            base_ctx, mlip_ctx = flatten_spec, None
        out = _Upstream.__coerce_same_metadata_as_tangent__(
            self, base_ctx, expected_type
        )
        if out is None:
            return None
        if not isinstance(out, ShardTensor):
            out.__class__ = ShardTensor
        if mlip_ctx is not None and getattr(out, "_distribution_spec", None) is None:
            for attr, value in zip(_PROPAGATED_ATTRS, mlip_ctx, strict=True):
                if value is not None:
                    setattr(out, attr, value)
        return out

    @staticmethod
    def wrap(
        t: torch.Tensor,
        *,
        mesh: Any = None,
        spec: Any = None,
        config: "ParticleHaloConfig | None" = None,
        meta: "ParticleHaloMetadata | None" = None,
        gather_meta: "ShardRouting | None" = None,
        n_systems: int | None = None,
        system_index: torch.Tensor | None = None,
        extra_suffix_padding: int = 0,
        halo_meta_packed: "torch.Tensor | None" = None,
    ) -> "ShardTensor":
        """Construct a ShardTensor wrapping ``t``, attaching spec + metadata.

        Builds a ``ShardTensorSpec`` for the base storage layer from the local
        tensor + mesh, then attaches the MLIP routing metadata
        (``_distribution_spec``, ``_meta``, …) as instance attributes.

        Parameters
        ----------
        t : torch.Tensor
            Input tensor (the per-rank "local" view).
        mesh : DeviceMesh, optional
            Device mesh for the synthesized spec. If ``None``, looked up from
            ``config.mesh``, then ``_mesh_resources.get_current_mesh()``, then a
            default 1-D mesh from the world process group.
        spec : optional
            The spec governing cross-rank dispatch routing for this tensor. When
            ``None``, the tensor flows through ``__torch_function__`` as a
            metadata-propagating view with no special dispatch.
        config : ParticleHaloConfig, optional
            Process-group config; both halo and gather modes need it to find the
            torch ``ProcessGroup``.
        meta : ParticleHaloMetadata, optional
            Halo topology (owned / halo / padded sizes, routing index).
            Required for halo-storage specs with cross-rank halo.
        gather_meta : ShardRouting, optional
            Global-ID routing table. Required for sharded-storage specs.
        n_systems : int, optional
            Number of systems on this rank; when set, per-system scatters at
            this shape dispatch to :func:`per_system_reduce`.
        system_index : torch.Tensor, optional
            Per-owned-atom system id; used by per-system reductions.
        extra_suffix_padding : int, optional
            Trailing non-halo rows (e.g. a single padding atom) that shape
            predicates must account for.
        halo_meta_packed : torch.Tensor, optional
            Pre-packed fixed-shape halo routing tensor to attach directly.

        Returns
        -------
        ShardTensor
            The wrapped tensor carrying the supplied spec and metadata.
        """
        from nvalchemi.distributed._core.shard_tensor_construction import (  # noqa: PLC0415
            make_local_shard_tensor_spec,
        )

        if isinstance(t, ShardTensor):
            # Idempotent: if already a ShardTensor, just attach any new
            # MLIP metadata fields. Useful for callers that re-wrap
            # something that's already in our subclass.
            out = t
        else:
            import torch.distributed as _dist  # noqa: PLC0415
            from torch.distributed.device_mesh import (  # noqa: PLC0415
                DeviceMesh,
                _mesh_resources,
            )

            if mesh is None and config is not None:
                mesh = getattr(config, "mesh", None)
            # ``config.mesh`` may be a non-DeviceMesh stub (e.g. tests
            # that mock just ``get_group()``). Treat anything that
            # isn't a real DeviceMesh as "no mesh provided" and fall
            # back to the resolution chain.
            if not isinstance(mesh, DeviceMesh):
                mesh = None
            if mesh is None:
                try:
                    mesh = _mesh_resources.get_current_mesh()
                except Exception:
                    mesh = None
            if mesh is None and _dist.is_initialized():
                # Default 1-D mesh from the world process group. The mesh must
                # declare the device of the tensor being wrapped (not "cuda
                # whenever cuda exists"): a mesh device disagreeing with
                # ``_local_tensor`` makes ops allocate results on different
                # devices ("found two devices, cuda:0 and cpu").
                world = _dist.get_world_size()
                device_type = t.device.type
                mesh = DeviceMesh(
                    device_type, list(range(world)), mesh_dim_names=("dom",)
                )
            if mesh is None:
                raise RuntimeError(
                    "ShardTensor.wrap could not resolve a DeviceMesh. "
                    "Pass mesh= explicitly, supply a config carrying a "
                    "mesh, or ensure a DeviceMesh is constructed in scope."
                )
            # Device-aware mesh: if the resolved cpu mesh doesn't match a cuda
            # tensor, synthesize a matching one. AOTAutograd reads the mesh's
            # device_type during fake-tensor propagation, so a cpu mesh wrapping
            # a cuda tensor causes "found two different devices" at compile time.
            # A user-supplied cuda mesh wrapping cpu data is left alone so the
            # genuine misconfiguration surfaces.
            if t.device.type != mesh.device_type and mesh.device_type == "cpu":
                world = mesh.size()
                ranks = (
                    mesh.mesh.tolist()
                    if hasattr(mesh.mesh, "tolist")
                    else list(range(world))
                )
                mesh = DeviceMesh(
                    t.device.type,
                    ranks,
                    mesh_dim_names=mesh.mesh_dim_names or ("dom",),
                    _init_backend=False,
                )
            shard_spec = make_local_shard_tensor_spec(t, mesh)
            if t.requires_grad:
                # Mirror DTensor's ``_FromTorchTensor.apply``: the wrap step
                # goes through an autograd.Function so gradients from ops on the
                # wrapper terminate on ``t`` (the user's leaf). Combined with
                # AOTAutograd attaching a grad_fn to compile outputs (via
                # ``return_and_correct_aliasing`` in ``__torch_dispatch__``),
                # ``torch.autograd.grad(out, t)`` works across the boundary.
                out = _AutogradPreservingWrap.apply(t, shard_spec)
            else:
                out = ShardTensor(local_tensor=t,
                    spec=shard_spec,
                    requires_grad=False,
                )
            # Preserve the pre-wrap tensor so :func:`torch.autograd.grad`
            # against this wrapper can find a concrete in-graph leaf —
            # see :func:`autograd_target` for the user-facing helper.
            out._autograd_source = t
        if spec is not None:
            out._distribution_spec = spec
            # Source the field's storage policy from the spec — the authoritative
            # declaration (with halo scatter/gather modes). Falls back to
            # meta/gather_meta-derived defaults below when the spec carries none.
            spec_policy = getattr(getattr(spec, "distribution", None), "policy", None)
            if spec_policy is not None:
                out._storage_policy = spec_policy
        if config is not None:
            out._config = config
        if meta is not None:
            out._meta = meta
            # Halo field: declare its storage policy. HaloStoragePolicy carries
            # the Shard(0)+overlay semantics + overlay-aware scatter/gather ops.
            if out._storage_policy is None:
                out._storage_policy = HaloStoragePolicy()
        if gather_meta is not None:
            out._gather_meta = gather_meta
            # Carry the gather routing (owner_rank || local_index, each
            # (n_global,)) in the _halo_meta_packed inner-tensor slot so it rides
            # as a fakified GRAPH INPUT under compile (not a baked _tensor_constant
            # Inductor lowering would reject). Sharded and halo tensors are
            # mutually exclusive (this one has no _meta), so the slot is
            # unambiguous; the distributed-gather dispatch unpacks the halves.
            out._halo_meta_packed = torch.cat(
                (
                    gather_meta.owner_rank.to(torch.int64).reshape(-1),
                    gather_meta.local_index.to(torch.int64).reshape(-1),
                )
            )
            # Sharded field: declare its storage policy. PlainShard carries the
            # cross-rank distributed scatter/gather.
            if out._storage_policy is None:
                out._storage_policy = PlainShard()
        if n_systems is not None:
            out._n_systems = n_systems
        if system_index is not None:
            out._system_index = system_index
        if extra_suffix_padding:
            out._extra_suffix_padding = extra_suffix_padding
        if halo_meta_packed is not None:
            out._halo_meta_packed = halo_meta_packed
        return out

    def autograd_target(self) -> torch.Tensor:
        """Return the tensor to pass as ``inputs=`` to :func:`torch.autograd.grad`.

        ``torch.autograd.grad`` matches inputs by graph identity, not Python
        object, so passing the wrapper directly can raise "differentiated
        Tensors appears to not have been used in the graph". This returns the
        ``_autograd_source`` captured at wrap time, which IS in the graph and
        has the right shape (e.g. ``[n_padded, *F]`` for halo-padded positions).

        Returns
        -------
        torch.Tensor
            The captured pre-wrap source tensor, or ``self`` when the
            ShardTensor was produced without a wrap step.
        """
        return self._autograd_source if self._autograd_source is not None else self

    def __getitem__(self, key: Any) -> Any:
        """Normalize a dim-0 integer advanced-index to ``index_select``.

        A dim-0 integer ``self[idx]`` is rewritten to ``index_select(self, 0,
        idx)`` so the gather takes the autograd-correct routed path (halo
        borrowed-row reverse-exchange) under ``torch.compile``.

        Notes
        -----
        ``self[idx]`` lowers to ``aten.index.Tensor`` whose adjoint
        (``index_put``) DROPS the halo-gather custom op's reverse-exchange
        backward under AOTAutograd, giving wrong boundary forces;
        ``index_select`` (adjoint ``index_add``) keeps it and computes the same
        value for a 1-D integer index. The rewrite must land at graph-build time
        — a ``__torch_function__`` handler is bypassed by Dynamo and a
        ``__torch_dispatch__`` rewrite runs too late. Only routed (halo /
        sharded) ShardTensors are rewritten; slices, boolean masks, multi-dim /
        tuple keys, and non-routed tensors use the default indexing.
        """
        if (
            isinstance(key, torch.Tensor)
            and key.dim() == 1
            and key.dtype in (torch.int32, torch.int64)
        ):
            branch = _shard_gather_branch(self)
            if branch == "distributed":
                return torch.index_select(self, 0, key)
            if branch == "halo":
                # Only the halo-REFRESH gather (stale per-layer features) goes
                # through the ``halo_forward`` custom op, which re-establishes the
                # borrowed-row autograd — there ``index_select`` is required for a
                # correct backward. The PBC-shifted ``(N, 3)`` coordinate field
                # and the no-halo case take the plain short-circuit gather on the
                # detached local; rewriting THEM to ``index_select`` severs the
                # gradient (no custom op to re-establish it), whereas the default
                # ``index.Tensor`` fallback preserves it via
                # ``return_and_correct_aliasing``. So skip those.
                meta = self._meta
                refresh = (
                    meta is not None
                    and meta.n_padded != meta.n_owned
                    and not (self.ndim == 2 and self.shape[-1] == 3)
                )
                if refresh:
                    return torch.index_select(self, 0, key)
        return torch.Tensor.__getitem__(self, key)

    def unwrap(self) -> torch.Tensor:
        """Return this rank's local plain ``torch.Tensor`` storage.

        Returns
        -------
        torch.Tensor
            ``self._local_tensor`` — the wrapper-subclass holds no own data.
        """
        return self._local_tensor

    def requires_grad_(self, requires_grad: bool = True) -> "ShardTensor":
        """Set ``requires_grad`` with native no-op-when-unchanged semantics.

        The vendored base unconditionally re-sets the flag, raising "you can
        only change requires_grad flags of leaf variables" on a *non-leaf*
        ShardTensor even when the flag already matches. Native PyTorch no-ops in
        that case — and model code calls ``positions.requires_grad_(True)``
        defensively. Honor it as a no-op when nothing changes; only delegate to
        the base when the flag actually flips.

        Parameters
        ----------
        requires_grad : bool, optional
            Target value for the autograd flag. Default ``True``.

        Returns
        -------
        ShardTensor
            ``self``.
        """
        if bool(self.requires_grad) == bool(requires_grad):
            return self
        return super().requires_grad_(requires_grad)

    def full_tensor(self, *, dst: int | None = None) -> torch.Tensor:
        """Reconstruct the semantic global tensor across ranks.

        Policy-aware: when this tensor carries a :class:`HaloStoragePolicy` the
        borrowed halo overlay is dropped — only the OWNED rows are gathered,
        giving the honest global tensor. Plain tensors (no policy) fall back to
        the base reconstruction.

        Parameters
        ----------
        dst : int, optional
            Destination rank for the gather; if ``None``, the result is
            replicated on every rank.

        Returns
        -------
        torch.Tensor
            The reconstructed global tensor.
        """
        if self._storage_policy is not None:
            return self._storage_policy.full_tensor(
                self, mesh=self._spec.mesh, dst=dst
            )
        return super().full_tensor()

    def local_sum(self) -> torch.Tensor:
        """Sum over this rank's OWNED rows only — drops halo duplicates.

        Autograd-friendly: the sum is differentiable and cross-rank
        contributions flow through the distributed scatter's halo-correction
        backward, so ``autograd.grad(local_sum, local_pos)`` gives the correct
        per-rank force. Halo-mode tensors slice the first ``meta.n_owned`` rows
        then sum; sharded-mode tensors hold only owned rows, so a plain sum is
        already local.

        Returns
        -------
        torch.Tensor
            Scalar sum over this rank's owned rows.
        """
        t = self.as_subclass(torch.Tensor)
        if self._meta is not None:
            return t[: self._meta.n_owned].sum()
        return t.sum()

    def global_sum(self) -> torch.Tensor:
        """Globally-reduced scalar: :meth:`local_sum` then all-reduce.

        Intended for reporting (total energy, logging a loss). For gradients the
        DDP-replicated-loss pattern applies: ``autograd.grad(global_sum,
        local_pos)`` gives ``world_size`` times the per-rank physical force, so
        use :meth:`local_sum` for autograd-targeted scalars.

        Returns
        -------
        torch.Tensor
            World-wide total, replicated on every rank.
        """
        import torch.distributed as dist

        from nvalchemi.distributed._core.gather_primitives import mesh_group

        total = self.local_sum()
        if dist.is_initialized() and self._config is not None:
            group = mesh_group(getattr(self._config, "mesh", None))
            # all_reduce in-place; autograd sees ``total`` unchanged as an
            # autograd.Function input (no backward sync). This matches the
            # "replicated forward, local backward" DDP pattern.
            total_contig = total.contiguous()
            dist.all_reduce(total_contig, op=dist.ReduceOp.SUM, group=group)
            if total_contig.data_ptr() != total.data_ptr():
                total = total_contig
        return total

    @property
    def spec(self) -> Any:
        # Public name is ``.spec``; internal storage is ``_distribution_spec`` to
        # avoid collision with the base ``_spec`` (``ShardTensorSpec``).
        return self._distribution_spec

    @property
    def config(self) -> "ParticleHaloConfig | None":
        return self._config

    @property
    def meta(self) -> "ParticleHaloMetadata | None":
        return self._meta

    @property
    def gather_meta(self) -> "ShardRouting | None":
        return self._gather_meta

    @property
    def n_systems(self) -> int | None:
        return self._n_systems

    @property
    def system_index(self) -> torch.Tensor | None:
        return self._system_index

    @property
    def extra_suffix_padding(self) -> int:
        return self._extra_suffix_padding

    # Under compile this returns the C sentinel
    # ``torch._C._disabled_torch_function_impl`` so Dynamo treats the subclass as
    # having NO torch-function override and uses NATIVE traceable wrapper-subclass
    # handling (runs ``__torch_dispatch__`` during fake-tensor propagation and
    # reconstructs outputs via ``__tensor_unflatten__``, never tracing our Python
    # construction). All ops then reach ``__torch_dispatch__``, where the
    # cross-rank ones route through the ``_dispatch_*`` helpers via
    # dispatcher-visible custom ops. Mirrors DTensor /
    # ``torch.testing._internal.two_tensor``.
    @classmethod
    def __torch_function__(
        cls,
        func: Any,
        types: Any,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        # Hybrid eager/compile routing. Under ``torch.compile`` (Dynamo bytecode
        # trace OR AOTAutograd fake-tensor decomposition) we behave as a NATIVE
        # traceable wrapper subclass: return the C disabled-impl so EVERY op
        # routes through ``__torch_dispatch__`` (the cross-rank ops are handled by
        # the ``_dispatch_*`` custom-op helpers there). Our Python handlers build
        # ShardTensors imperatively, which Dynamo cannot trace — hence the guard.
        #
        # In EAGER we route the MLIP-owned function ops (index_select / scatter /
        # the ``register_handler`` escape hatch) to their cross-rank handlers in
        # ``_function_registry`` (tracked in ``_OUR_HANDLERS``): the compile-path
        # ``_dispatch_*`` custom ops issue collectives that are not symmetric
        # under eager per-op dispatch, so eager keeps the handler path.
        if kwargs is None:
            kwargs = {}
        if _under_compile_trace(args) or torch.compiler.is_compiling():
            return torch._C._disabled_torch_function_impl(func, types, args, kwargs)
        if func in _OUR_HANDLERS:
            # ``_OUR_HANDLERS`` tracks BOTH function-level (``_register_function``:
            # index_select / scatter, called as ``(func, types, args, kwargs)``)
            # and dispatch-level (``register_handler`` / ``wrap_custom_op``,
            # called as ``(*args, **kwargs)``) registrations. At
            # ``__torch_function__`` ``func`` is the PUBLIC callable both
            # registries key on, so a core op like ``torch.sigmoid`` registered
            # via ``register_handler`` only fires HERE. Gating on ``_OUR_HANDLERS``
            # keeps base aten-keyed view/reshape handlers untouched.
            fn = cls._function_registry.get(func)
            if fn is not None:
                return fn(func, types, args, kwargs)
            disp = cls._dispatch_registry.get(func)
            if disp is not None:
                return disp(*args, **kwargs)
        return torch._C._disabled_torch_function_impl(func, types, args, kwargs)

    @classmethod
    def __torch_dispatch__(
        cls,
        func: Any,
        types: Any,
        args: tuple = (),
        kwargs: dict | None = None,
    ) -> Any:
        # Preserve upstream's view/reshape dispatch handlers (unbind, select,
        # select_backward, unsqueeze, view, reshape) — they redistribute spec
        # metadata correctly for those shape ops.
        if kwargs is None:
            kwargs = {}
        handler = cls._dispatch_registry.get(func)
        if handler is None:
            handler = cls._dispatch_registry_by_name.get(str(func))
        if handler is not None:
            result = handler(*args, **kwargs)
            # The base handlers (sharded_select_helper, _unbind_dispatch, etc.)
            # construct base-class ``ShardTensor`` instances. Coerce those back
            # to our subclass via ``__class__`` reassignment so MLIP methods
            # (``unwrap``, ``local_sum``, etc.) remain available on the result.
            source = _prefer_source(args)
            if source is not None:
                _propagate_attrs(result, source)
            return result
        # Fallback: extract-local / run-plain / re-wrap. We deliberately do NOT
        # delegate to the vendored ``_dispatch_fallback_via_dtensor`` here — its
        # DTensor cast-down/cast-up diverges from the MLIP semantics for ops
        # like ``index_add_`` / ``index_copy_`` / ``mol_sum`` and the owned-slice
        # custom-op path (verified: delegating breaks test_per_system_dispatch /
        # test_wrap_custom_op_extensions / test_registry_and_contexts). The
        # extract-local path also avoids NotImplementedError for ops without a
        # DTensor sharding strategy (e.g. ``aten.set_.source_Storage_storage_offset``
        # from Dynamo's MetaConverter during fake conversion).
        from torch.utils._python_dispatch import (  # noqa: PLC0415
            return_and_correct_aliasing,
        )

        source = _prefer_source(args)
        local_args = tuple(_strip_to_local(a) for a in args)
        local_kwargs = {k: _strip_to_local(v) for k, v in kwargs.items()}
        result = func(*local_args, **local_kwargs)
        # Halo correction for the compile path. A halo scatter is out-of-place
        # (cross-rank exchange yields a fresh tensor), so it must NOT go through
        # ``return_and_correct_aliasing`` below — that returns the original
        # (uncorrected) accumulator for an in-place op. Return the freshly-wrapped
        # result directly, matching the eager handler.
        gathered = _dispatch_halo_gather(func, args, kwargs, source)
        if gathered is not None:
            return _make_handler_output(gathered, source)
        dist_gathered = _dispatch_distributed_gather(func, args, kwargs, source)
        if dist_gathered is not None:
            return _make_handler_output(dist_gathered, source)
        dist_scattered = _dispatch_distributed_scatter(func, args, kwargs, source)
        if dist_scattered is not None:
            return _make_handler_output(dist_scattered, source)
        reduced = _dispatch_per_system_reduce(func, args, kwargs, source)
        if reduced is not None:
            return _make_handler_output(reduced, source)
        corrected = _dispatch_halo_scatter_correct(func, args, kwargs, result, source)
        if corrected is not None:
            # A MUTATING scatter (index_add_/scatter_add_) under compile: AOT
            # functionalization propagates the in-place INPUT's mutated storage,
            # not a fresh return value, so returning the corrected tensor alone
            # is silently dropped (downstream reads the uncorrected local
            # scatter). Write the corrected (halo reverse+forward) values back
            # into the in-place local and alias-correct so the correction is
            # what propagates; the copy_ keeps the halo_scatter_correct backward
            # in the traced graph. Out-of-place returns the fresh corrected.
            _mutable = getattr(getattr(func, "_schema", None), "is_mutable", False)
            if _mutable and _under_compile_trace(args):
                result.copy_(corrected)
                return return_and_correct_aliasing(
                    func, args, kwargs, _make_handler_output(result, source)
                )
            return _make_handler_output(corrected, source)
        if source is not None:
            result = _wrap_back_to_shardtensor(result, source)
        # ``return_and_correct_aliasing`` is REQUIRED for AOTAutograd
        # correctness on wrapper subclasses: it patches up storage
        # aliasing for view ops and ensures in-place ops return the
        # correct input. Without this, AOTAutograd's compile-output
        # autograd chain is severed (no ``grad_fn`` attached to the
        # wrapper), reproducible via ``test_compile_smoke_world1_backward``.
        # The reference pattern is in ``torch/testing/_internal/custom_tensor.py``.
        return return_and_correct_aliasing(func, args, kwargs, result)


# Default handler registrations: ONE base-signature dispatcher per intercepted
# op, registered in the base function registry. Each dispatcher self-classifies
# (per-system reduce → halo correction → distributed scatter for scatters;
# halo forward-sync → distributed gather for index_select) and falls through to
# default dispatch when the op isn't MLIP-routed.


for _scatter_op in (
    torch.Tensor.scatter_add_,
    torch.Tensor.index_add_,
    torch.Tensor.index_copy_,
):
    _register_function(
        _scatter_op, _scatter_add_dispatch, f"mlip_scatter[{_scatter_op.__name__}]"
    )

# index_select can be invoked either as ``t.index_select(...)``
# (``torch.Tensor.index_select``) or as ``torch.index_select(t, ...)``
# (the functional form). They dispatch as distinct function objects, so
# the registry holds an entry for both.
for _index_select_op in (torch.Tensor.index_select, torch.index_select):
    _register_function(
        _index_select_op,
        _index_select_dispatch,
        f"mlip_index_select[{_index_select_op.__name__}]",
    )

# Advanced indexing ``x[idx]`` (``node_feats[sender]``) is handled by the Python
# ``ShardTensor.__getitem__`` method (Dynamo inline-traces it), NOT a
# ``__torch_function__`` handler — Dynamo bypasses ``__torch_function__`` for
# ``__getitem__``, so the rewrite to ``index_select`` must land as a real method
# at the graph-build level. See ``ShardTensor.__getitem__``.


# Opaque ``@torch.library.custom_op`` kernels (Warp / Triton) bypass
# ``__torch_function__`` via ``wp.from_torch``. Models integrate them by
# declaring a *functional* op (returns its outputs) plus an
# :class:`~nvalchemi.distributed._core.adapter.OpAdapter` on the spec —
# the adapter's installed handler unwraps the ShardTensor args, runs the
# kernel on the locals, and wraps the returned tensors back into
# ShardTensors (applying any declared output transforms). See LJ /
# MACE-cueq specs and ``examples/distributed/05_byo_graph_transformer.py``.
# No framework-side in-place buffer promotion exists: a plain output
# buffer cannot be reclassed into a wrapper-subclass ShardTensor in place
# (CPython rejects the layout change), so the functional + OpAdapter path
# is the single supported integration for opaque kernels.


# ----------------------------------------------------------------------
# AOTAutograd shim: rebuild a ShardTensor from a PLAIN runtime tangent
# ----------------------------------------------------------------------
def _install_aot_plain_tangent_coercion() -> None:
    """Monkeypatch ``AOTDispatchAutograd.process_runtime_tangent`` so a PLAIN
    runtime backward tangent is rebuilt into a :class:`ShardTensor` when the
    compiled graph traced a ShardTensor tangent at that position.

    PyTorch's ``process_runtime_tangent`` can coerce a runtime *subclass* tangent
    to its traced metadata (via ``x.__coerce_same_metadata_as_tangent__`` — see
    :meth:`ShardTensor.__coerce_same_metadata_as_tangent__`, which covers the
    subclass-runtime cases incl. coerce-DOWN to a plain tensor). A runtime
    *plain* tensor has no such hook, so AOT raises ``...guessed its metadata
    incorrectly``. This is the MIRROR case: when a ShardTensor boundary tensor
    crosses a Dynamo graph break (e.g. MACE-cueq's SphericalHarmonics-marshal
    split), AOT materializes that boundary's cotangent as a PLAIN tensor while
    the upstream subgraph traced a ShardTensor tangent. The plain cotangent is
    value-correct (the boundary placement is ``Replicate`` — local == global),
    so rebuilding the subclass from it + the traced ``SubclassCreationMeta`` via
    ``__tensor_unflatten__`` is lossless and restores correct conservative
    forces.

    A general PyTorch AOT expressiveness gap (no plain->subclass tangent hook),
    not MACE/cueq-specific. Scoped to OUR ShardTensor: other subclasses and
    plain-where-plain tangents fall through untouched. Idempotent; failures to
    import the torch internals or rebuild the subclass degrade gracefully to the
    stock behaviour (the original informative error).
    """
    try:
        from torch._functorch._aot_autograd.runtime_wrappers import (  # noqa: PLC0415
            AOTDispatchAutograd,
        )
        from torch._functorch._aot_autograd.schemas import (  # noqa: PLC0415
            SubclassCreationMeta,
        )
        from torch.utils._python_dispatch import (  # noqa: PLC0415
            is_traceable_wrapper_subclass,
        )
    except Exception:  # pragma: no cover - torch internals moved/unavailable
        logger.debug(
            "AOT plain-tangent coercion shim not installed (import failed)",
            exc_info=True,
        )
        return

    _orig = AOTDispatchAutograd.process_runtime_tangent
    if getattr(_orig, "_mlip_plain_tangent_shim", False):
        return

    def _process_runtime_tangent(x: Any, meta: Any) -> Any:
        if (
            isinstance(x, torch.Tensor)
            and not is_traceable_wrapper_subclass(x)
            and isinstance(meta, SubclassCreationMeta)
            and isinstance(getattr(meta, "original_subclass_type", None), type)
            and issubclass(meta.original_subclass_type, ShardTensor)
            and 1 <= len(meta.attrs) <= 2
        ):
            try:
                inner = {
                    a: (
                        torch.zeros(0, dtype=torch.int64, device=x.device)
                        if a == "_halo_meta_packed"
                        else x
                    )
                    for a in meta.attrs
                }
                x = meta.original_subclass_type.__tensor_unflatten__(
                    inner, meta.meta, meta.outer_size, meta.outer_stride
                )
            except Exception:  # pragma: no cover - fall through to stock error
                logger.debug(
                    "plain->ShardTensor runtime-tangent rebuild failed",
                    exc_info=True,
                )
        return _orig(x, meta)

    _process_runtime_tangent._mlip_plain_tangent_shim = True  # type: ignore[attr-defined]
    AOTDispatchAutograd.process_runtime_tangent = staticmethod(
        _process_runtime_tangent
    )


_install_aot_plain_tangent_coercion()
