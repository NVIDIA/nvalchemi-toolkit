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

"""Unified third-party adapter API.

The adapter classes share one mental model: declare on a
:class:`DistributionSpec`, and the framework owns install/restore via
:class:`AdapterRegistry`. :class:`OpAdapter` delegates to
:func:`~nvalchemi.distributed._core.escape_hatches.wrap_custom_op`;
:class:`JitAdapter` / :class:`PythonAdapter` swap a module-level callable,
with install + restore owned by the registry rather than a hand-managed
handle.

**Which adapter? — decide by what you are adapting:**

==========================================  ==========================
What you are adapting                       Use
==========================================  ==========================
a custom / Triton / Warp kernel registered  :class:`OpAdapter` — declare
via ``@torch.library.custom_op`` /          its cross-rank I/O roles
``@torch.library.triton_op``                (``gather_inputs`` /
                                            ``scatter_outputs`` / …)
a module-level ``@torch.jit.script``        :class:`JitAdapter` — marshal
helper a ShardTensor must cross             the scripted op, or swap a
                                            plain-Python copy
a plain module-level Python function        :class:`PythonAdapter`, or
whose layout assumptions break under        :func:`FunctionAdapter` to name
partition (e.g. ``aimnet.nbops.mol_sum``)   it by the function object, not
                                            ``module``/``attr`` strings
a class *method* you must wrap              :class:`MethodAdapter` — wrap
(transform an arg, then call the            the call, transform, then
original) — e.g. ``ConvSV.forward``         invoke ``original``
==========================================  ==========================

The rule of thumb: ``OpAdapter`` / ``JitAdapter`` / ``PythonAdapter`` /
``FunctionAdapter`` *replace* a callable outright; :class:`MethodAdapter`
*wraps* one (it hands the replacement the original as its first argument).

* :class:`OpAdapter` — wrap a ``@torch.library.custom_op`` /
  ``@torch.library.triton_op`` kernel. Routes through ShardTensor
  dispatch when called with a ShardTensor argument.
* :class:`JitAdapter` — replace a ``@torch.jit.script`` helper with a
  plain-Python equivalent so ShardTensor's ``__torch_function__`` can
  fire inside.
* :class:`PythonAdapter` — replace a plain-Python module-level helper
  whose tensor-layout assumptions break under partition (e.g. AIMNet2's
  ``aimnet.nbops.mol_sum``). :func:`FunctionAdapter` is the same thing
  named by the function object instead of module/attr strings.
* :class:`MethodAdapter` — wrap a class method: intercept the call,
  transform an argument, then invoke the original.

All are picklable, frozen dataclasses. Lifecycle (install /
restore / introspection) is owned by :class:`AdapterRegistry`, which
:class:`DistributedModel` instantiates per scope.

Worked example::

    from nvalchemi.distributed._core.adapter import (
        OpAdapter, JitAdapter, PythonAdapter,
    )
    from nvalchemi.distributed._core.op_transforms import (
        GatherInputsFull, SliceOutputsOwned,
    )
    from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy
    from nvalchemi.distributed._core.spec import DistributionSpec
    from nvalchemi.distributed.spec import MLIPSpec

    # Plain-Python equivalent of the model's @torch.jit.script helper.
    # Must be byte-for-byte identical except for the ``@torch.jit.script``
    # decorator, so ShardTensor's __torch_function__ can fire inside.
    def _envelopes_plain(r, cutoff):
        return ((r < cutoff).float() * (1 - r / cutoff) ** 2)

    def distribution_spec(self, strategy=None):
        return MLIPSpec(
            distribution=DistributionSpec(
                policy=HaloStoragePolicy(),
                custom_ops=(
                    OpAdapter(
                        mymace._kernel_radial_basis,
                        arg_transforms={0: GatherInputsFull()},
                        output_transforms={0: SliceOutputsOwned()},
                    ),
                ),
                third_party_helpers=(
                    JitAdapter(
                        "mymace.scripts", "envelopes",
                        replacement=_envelopes_plain,
                    ),
                    PythonAdapter(
                        "mymace.utils", "build_neighbor_mask",
                        replacement=self._distributed_neighbor_mask,
                    ),
                ),
            ),
        )

    # ``DistributedModel.__enter__`` builds an ``AdapterRegistry`` from
    # the spec's adapters and calls ``install()``; ``__exit__`` calls
    # ``restore()``. Adapters with ``replacement=None`` are pure
    # declarations — the wrapper installs the actual swap elsewhere
    # (e.g. ``distributed_setup`` when the replacement closes over
    # runtime metadata).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Sequence

from nvalchemi.distributed._core.op_transforms import (
    AllReduceSum,
    ArgTransform,
    GatherInputs,
    GatherInputsFull,
    OutputTransform,
    ScatterOutputs,
    SliceOutputsOwned,
    SliceOwned,
)

logger = logging.getLogger(__name__)


__all__ = [
    "AdapterStatus",
    "OpAdapter",
    "JitAdapter",
    "PythonAdapter",
    "FunctionAdapter",
    "MethodAdapter",
    "ModuleForwardAdapter",
    "AdapterRegistry",
    "ThirdPartyHelper",
    "register_adapter_kind",
    "_op_qualname",
    "_resolve_op",
]


# ----------------------------------------------------------------------
# Introspection.
# ----------------------------------------------------------------------


AdapterKind = Literal["op", "jit", "python", "method"]
AdapterState = Literal["pending", "installed", "restored", "failed"]


@dataclass(frozen=True)
class AdapterStatus:
    """Introspectable record of one adapter's lifecycle state.

    Returned by :meth:`AdapterRegistry.list_active`. Surfaces what's
    been swapped in this process — useful when debugging "why is
    ``fairchem`` behaving weirdly outside the distributed scope" kinds
    of questions.

    Attributes
    ----------
    kind
        ``"op"`` for :class:`OpAdapter`, ``"jit"`` / ``"python"``.
    target
        Human-readable identifier of the adapted callable
        (``"torch.ops.fairchem._kernel_xyz"`` /
        ``"aimnet.nbops.mol_sum"``).
    state
        ``"pending"`` (registered but not installed), ``"installed"``,
        ``"restored"`` (installed then cleaned up), ``"failed"``
        (install raised — see ``error``).
    install_site
        ``"filename:lineno"`` capturing where the adapter was
        constructed. Empty when not auto-captured.
    error
        ``str(exception)`` when state == ``"failed"``; else ``None``.
    """

    kind: AdapterKind
    target: str
    state: AdapterState
    install_site: str = ""
    error: str | None = None


def _capture_call_site() -> str:
    """Best-effort ``filename:lineno`` of the user code that constructed this
    adapter.

    Walks up the frame chain skipping any frame whose filename is ``<string>``
    (the dataclass-synthesised init) or this module itself, so the result is
    always the caller in real source.
    """
    try:
        # Start two up (skip ourselves + __post_init__).
        depth = 2
        while True:
            frame = sys._getframe(depth)
            filename = frame.f_code.co_filename
            if filename == "<string>" or filename.endswith("/_core/adapter.py"):
                depth += 1
                continue
            return f"{filename}:{frame.f_lineno}"
    except (ValueError, AttributeError):
        return ""


# ----------------------------------------------------------------------
# OpAdapter — torch.library.custom_op wrapper.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class OpAdapter:
    """Adapt a ``@torch.library.custom_op`` / ``@torch.library.triton_op``
    kernel to ShardTensor-aware dispatch.

    Two dicts declare per-position pre/post transformations:

    ``arg_transforms`` (input position → :data:`ArgTransform`)
        :class:`~nvalchemi.distributed._core.op_transforms.GatherInputs` —
        halo-pad owned input to ``(n_padded, *F)`` before kernel.
        :class:`~nvalchemi.distributed._core.op_transforms.GatherInputsFull` —
        sharded analogue: full-gather to ``(n_global + 1, *F)``.
        :class:`~nvalchemi.distributed._core.op_transforms.SliceOwned` —
        slice halo-padded input to ``(n_owned, *F)``.

    ``output_transforms`` (output position → :data:`OutputTransform`)
        :class:`~nvalchemi.distributed._core.op_transforms.ScatterOutputs` —
        ``halo_reverse_exchange + halo_forward_exchange`` after kernel.
        :class:`~nvalchemi.distributed._core.op_transforms.AllReduceSum` —
        cross-rank SUM (autograd-symmetric).
        :class:`~nvalchemi.distributed._core.op_transforms.SliceOutputsOwned` —
        slice ``(n_global + 1, *F)`` back to ``(n_owned + 1, *F)``.

    The **positional-role** form names which I/O plays which cross-rank role by
    position, with no transform objects to import::

        OpAdapter(torch.ops.ns.fused_op, scatter_outputs=[0])      # node scatter
        OpAdapter(op, gather_inputs=[0], scatter_outputs=[0])      # neighbor read + scatter

    The role keywords lower onto the same two transform dicts, so the dict form
    above stays valid and serialization is unchanged.

    See the module docstring for the full worked example.
    """

    op: Any
    arg_transforms: dict[int, ArgTransform] = field(default_factory=dict)
    output_transforms: dict[int, OutputTransform] = field(default_factory=dict)
    install_site: str = field(default="", compare=False, hash=False)

    def __init__(
        self,
        op: Any,
        arg_transforms: dict[int, ArgTransform] | None = None,
        output_transforms: dict[int, OutputTransform] | None = None,
        *,
        gather_inputs: tuple[int, ...] = (),
        neighbors_inputs: tuple[int, ...] = (),
        gather_inputs_full: tuple[int, ...] = (),
        owned_slice_inputs: tuple[int, ...] = (),
        scatter_outputs: tuple[int, ...] = (),
        all_reduce_outputs: tuple[int, ...] = (),
        slice_outputs_owned: tuple[int, ...] = (),
        install_site: str = "",
    ) -> None:
        # Positional-role keywords lower onto the transform dicts. An explicit
        # dict entry for a position wins over a role keyword for that position.
        # ``neighbors_inputs`` is an alias for ``gather_inputs``: an opaque kernel
        # that reads each atom's NEIGHBOR rows — the framework refreshes those
        # rows' ghosts before the kernel (halo forward-exchange in eager dispatch;
        # the static op under compile).
        at: dict[int, ArgTransform] = dict(arg_transforms or {})
        ot: dict[int, OutputTransform] = dict(output_transforms or {})
        for _p in (*gather_inputs, *neighbors_inputs):
            at.setdefault(_p, GatherInputs())
        for _p in gather_inputs_full:
            at.setdefault(_p, GatherInputsFull())
        for _p in owned_slice_inputs:
            at.setdefault(_p, SliceOwned())
        for _p in scatter_outputs:
            ot.setdefault(_p, ScatterOutputs())
        for _p in all_reduce_outputs:
            ot.setdefault(_p, AllReduceSum())
        for _p in slice_outputs_owned:
            ot.setdefault(_p, SliceOutputsOwned())
        # Accept the op PACKET (``torch.ops.ns.name``) and resolve ``.default``
        # ourselves so callers never type ``.default``. An explicit overload
        # (``...name.default``) is used as-is.
        if type(op).__name__ == "OpOverloadPacket":
            op = op.default
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "arg_transforms", at)
        object.__setattr__(self, "output_transforms", ot)
        object.__setattr__(self, "install_site", install_site or _capture_call_site())

    # -- Per-transform-kind position views (used by the dispatch path) --

    @property
    def gather_inputs(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                p for p, t in self.arg_transforms.items() if isinstance(t, GatherInputs)
            )
        )

    @property
    def gather_inputs_full(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                p
                for p, t in self.arg_transforms.items()
                if isinstance(t, GatherInputsFull)
            )
        )

    @property
    def owned_slice_inputs(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                p for p, t in self.arg_transforms.items() if isinstance(t, SliceOwned)
            )
        )

    @property
    def scatter_outputs(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                p
                for p, t in self.output_transforms.items()
                if isinstance(t, ScatterOutputs)
            )
        )

    @property
    def all_reduce_outputs(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                p
                for p, t in self.output_transforms.items()
                if isinstance(t, AllReduceSum)
            )
        )

    @property
    def slice_outputs_owned(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                p
                for p, t in self.output_transforms.items()
                if isinstance(t, SliceOutputsOwned)
            )
        )

    # -- Lifecycle --

    def _target_str(self) -> str:
        op = self.op
        schema = getattr(op, "_schema", None)
        if schema is not None and getattr(schema, "name", None):
            return schema.name
        return str(op)

    def install(self) -> dict[str, Any]:
        """Register a ShardTensor-aware dispatch handler on the op (and
        its overload packet, if any). Returns a memento that
        :meth:`restore` consumes to clear the registration.
        """
        # See ``escape_hatches.wrap_custom_op`` for full semantics.
        from nvalchemi.distributed._core.escape_hatches import (
            wrap_custom_op,  # noqa: PLC0415
        )

        wrap_custom_op(
            self.op,
            gather_inputs=self.gather_inputs,
            scatter_outputs=self.scatter_outputs,
            owned_slice_inputs=self.owned_slice_inputs,
            all_reduce_outputs=self.all_reduce_outputs,
            gather_inputs_full=self.gather_inputs_full,
            slice_outputs_owned=self.slice_outputs_owned,
        )
        # Memento captures the op + packet for clear_handlers.
        packet = getattr(self.op, "_overloadpacket", None)
        return {"op": self.op, "packet": packet}

    def restore(self, memento: dict[str, Any]) -> None:
        """Clear the handler registered by :meth:`install`."""
        from nvalchemi.distributed._core.shard_tensor import (
            clear_handlers,  # noqa: PLC0415
        )

        clear_handlers(memento["op"])
        if memento.get("packet") is not None and memento["packet"] is not memento["op"]:
            clear_handlers(memento["packet"])

    def describe(
        self, state: AdapterState = "pending", error: str | None = None
    ) -> AdapterStatus:
        """Return an :class:`AdapterStatus` snapshot of this adapter."""
        return AdapterStatus(
            kind="op",
            target=self._target_str(),
            state=state,
            install_site=self.install_site,
            error=error,
        )

    # -- JSON serialization --

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-roundtrippable dict."""
        return {
            "op": _op_qualname(self.op),
            "arg_transforms": {
                str(pos): _arg_transform_to_dict(t)
                for pos, t in sorted(self.arg_transforms.items())
            },
            "output_transforms": {
                str(pos): _output_transform_to_dict(t)
                for pos, t in sorted(self.output_transforms.items())
            },
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OpAdapter":
        """Reconstruct an :class:`OpAdapter` from :meth:`to_dict` output."""
        return cls(
            op=_resolve_op(d["op"]),
            arg_transforms={
                int(p): _arg_transform_from_dict(td)
                for p, td in d.get("arg_transforms", {}).items()
            },
            output_transforms={
                int(p): _output_transform_from_dict(td)
                for p, td in d.get("output_transforms", {}).items()
            },
        )


# ----------------------------------------------------------------------
# Op-handle <-> qualname round-trip + transform JSON helpers.
# Used by OpAdapter and re-exported from spec.py.
# ----------------------------------------------------------------------


def _op_qualname(op: Any) -> str:
    """Return the schema-qualified ``"<namespace>::<name>"`` string for
    a torch op overload (or overload-packet). Falls back to ``str(op)``
    when no schema is exposed."""
    schema = getattr(op, "_schema", None)
    if schema is not None and getattr(schema, "name", None):
        return schema.name
    overloads = getattr(op, "overloads", None)
    if callable(overloads):
        try:
            for overload_name in op.overloads():
                child = getattr(op, overload_name)
                child_schema = getattr(child, "_schema", None)
                if child_schema is not None and getattr(child_schema, "name", None):
                    return child_schema.name
        except Exception:  # noqa: S110, BLE001
            # Best-effort introspection — we fall through to ``str(op)``
            # below for any op whose overload list raises.
            pass
    return str(op)


def _resolve_op(qualname: str) -> Any:
    """Inverse of :func:`_op_qualname`. Resolves
    ``"<namespace>::<name>"`` to ``torch.ops.<namespace>.<name>.default``.
    The op-registering module must already be imported."""
    import torch  # noqa: PLC0415

    if "::" not in qualname:
        raise ValueError(
            f"_resolve_op: expected '<namespace>::<name>' qualified form, "
            f"got {qualname!r}"
        )
    ns, name = qualname.split("::", 1)
    namespace = getattr(torch.ops, ns, None)
    if namespace is None:
        raise RuntimeError(
            f"_resolve_op: torch.ops.{ns} not registered. The module that "
            f"declares {qualname!r} must be imported before loading the spec."
        )
    overload_packet = getattr(namespace, name, None)
    if overload_packet is None:
        raise RuntimeError(
            f"_resolve_op: torch.ops.{ns}.{name} not registered in namespace {ns!r}."
        )
    return overload_packet.default


_ARG_TRANSFORM_REGISTRY: dict[str, type] = {
    "gather_inputs": GatherInputs,
    "gather_inputs_full": GatherInputsFull,
    "slice_owned": SliceOwned,
}
_OUTPUT_TRANSFORM_REGISTRY: dict[str, type] = {
    "scatter_outputs": ScatterOutputs,
    "all_reduce_sum": AllReduceSum,
    "slice_outputs_owned": SliceOutputsOwned,
}


def _arg_transform_to_dict(t: ArgTransform) -> dict[str, Any]:
    for kind, cls in _ARG_TRANSFORM_REGISTRY.items():
        if isinstance(t, cls):
            return {"type": kind}
    raise TypeError(f"unknown ArgTransform: {type(t).__name__}")


def _arg_transform_from_dict(d: dict[str, Any]) -> ArgTransform:
    cls = _ARG_TRANSFORM_REGISTRY.get(d.get("type"))
    if cls is None:
        raise ValueError(
            f"unknown ArgTransform type {d.get('type')!r}; expected one of "
            f"{list(_ARG_TRANSFORM_REGISTRY)}."
        )
    return cls()


def _output_transform_to_dict(t: OutputTransform) -> dict[str, Any]:
    for kind, cls in _OUTPUT_TRANSFORM_REGISTRY.items():
        if isinstance(t, cls):
            return {"type": kind}
    raise TypeError(f"unknown OutputTransform: {type(t).__name__}")


def _output_transform_from_dict(d: dict[str, Any]) -> OutputTransform:
    cls = _OUTPUT_TRANSFORM_REGISTRY.get(d.get("type"))
    if cls is None:
        raise ValueError(
            f"unknown OutputTransform type {d.get('type')!r}; expected one of "
            f"{list(_OUTPUT_TRANSFORM_REGISTRY)}."
        )
    return cls()


# ----------------------------------------------------------------------
# JitAdapter — replace @torch.jit.script helper with plain-Python.
# ----------------------------------------------------------------------


def make_marshaller(original: Callable[..., Any]) -> Callable[..., Any]:
    """Build a marshaller around a scripted callable.

    A ``@torch.jit.script`` op called with a ShardTensor goes straight into the
    JIT executor — ``__torch_function__`` does NOT fire for scripted calls — so a
    TensorExpr-fused kernel reads the wrapper's raw ``data_ptr`` (``≈0`` for the
    storage-less ShardTensor) → CUDA illegal memory access. The marshaller
    unwraps ShardTensor args to their real-storage local tensors, runs the
    (still-scripted, still-fused) op, and re-wraps the output as a ShardTensor.
    The unwrap/rewrap go through ``_unwrap_grad_aware`` /
    ``_wrap_back_to_shardtensor`` so the autograd graph (positions → energy)
    stays intact for ``F = -dE/dx`` — the dominant MLIP force convention.

    Only correct for node/edge-LOCAL scripted ops (no cross-rank dependency
    inside the scripted region); the equivalence check is the correctness
    backstop, and ``DomainConfig.scripted_marshal`` / a denylist let a
    cross-rank op be excluded.
    """

    def _marshalled(*args: Any, **kwargs: Any) -> Any:
        from nvalchemi.distributed._core.shard_tensor import (  # noqa: PLC0415
            ShardTensor,
            _prefer_source,
            _unwrap_grad_aware,
            _wrap_back_to_shardtensor,
        )

        source = _prefer_source(args, kwargs)
        if source is None:
            # No ShardTensor in the call — nothing to marshal; run as-is.
            return original(*args, **kwargs)

        def _unwrap(t: Any) -> Any:
            if isinstance(t, ShardTensor):
                return _unwrap_grad_aware(t)
            if isinstance(t, (list, tuple)):
                return type(t)(_unwrap(x) for x in t)
            return t

        local_args = tuple(_unwrap(a) for a in args)
        local_kwargs = {k: _unwrap(v) for k, v in kwargs.items()}
        out = original(*local_args, **local_kwargs)
        return _wrap_back_to_shardtensor(out, source)

    # Tag for diagnostics / dedup (auto-discovery skips already-marshalled attrs).
    _marshalled._nvalchemi_marshaller = True  # type: ignore[attr-defined]

    # ``torch.compile`` cannot trace a scripted (``RecursiveScriptModule``) op —
    # dynamo raises ``UnspecializedNNModuleVariable ... ScriptModules
    # unsupported`` under ``fullgraph=True``. ``torch.compiler.disable`` makes
    # dynamo graph-break here and run the marshaller EAGERLY — unwrapping the
    # ShardTensor to its real-storage local (no TorchScript-fusion IMA), running
    # the still-scripted op, and re-wrapping via ``_wrap_back_to_shardtensor``
    # (its ``_AutogradPreservingWrap`` keeps
    # ``wrapper.requires_grad == _local_tensor.requires_grad``, so the surrounding
    # compiled regions' fake-tensorization of the result does not trip the
    # inner/outer requires_grad assertion). A no-op in eager.
    import torch  # noqa: PLC0415

    disabled = torch.compiler.disable(_marshalled)
    disabled._nvalchemi_marshaller = True  # type: ignore[attr-defined]
    return disabled


_MARSHAL_WRAPPER_CLS: Any = None


def _marshalling_module_cls() -> Any:
    """Lazily build the marshalling wrapper ``nn.Module`` (this module avoids a
    load-time torch import)."""
    global _MARSHAL_WRAPPER_CLS
    if _MARSHAL_WRAPPER_CLS is None:
        import torch  # noqa: PLC0415

        class _MarshallingModule(torch.nn.Module):
            _nvalchemi_marshal_wrap = True

            def __init__(self, inner: Any) -> None:
                super().__init__()
                self.inner = inner
                self._marshalled = make_marshaller(inner)

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return self._marshalled(*args, **kwargs)

        _MARSHAL_WRAPPER_CLS = _MarshallingModule
    return _MARSHAL_WRAPPER_CLS


def auto_marshal_scripted_submodules(
    model: Any, *, exclude: Sequence[str] = (), declared_targets: Sequence[str] = ()
) -> list[tuple[Any, str, Any]]:
    """Auto-discover and wrap each scripted submodule's ``forward`` with a
    marshaller so a ShardTensor can cross it (scripted/fused kernels can't read
    the storage-less wrapper). Returns
    ``[(parent, child_name, original_submodule), ...]`` for
    :func:`restore_auto_marshalled`.

    Skips submodules whose qualified name contains an ``exclude`` substring, or
    is already covered by a declared ``JitAdapter`` (``declared_targets``), or is
    already wrapped (idempotent). The wrapper intercepts at the Python
    ``__call__`` boundary — before the JIT executor — so the marshalled inputs
    reach the scripted graph.
    """
    import torch  # noqa: PLC0415

    wrap_cls = _marshalling_module_cls()
    mementos: list[tuple[Any, str, Any]] = []
    scripted = [
        (name, mod)
        for name, mod in model.named_modules()
        if name and isinstance(mod, torch.jit.ScriptModule)
    ]
    for name, mod in scripted:
        if any(pat in name for pat in exclude):
            continue
        if any(name in t or t in name for t in declared_targets):
            continue
        parent_name, _, child = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        if getattr(getattr(parent, child, None), "_nvalchemi_marshal_wrap", False):
            continue
        setattr(parent, child, wrap_cls(mod))
        mementos.append((parent, child, mod))
        logger.warning(
            "auto-marshalled scripted submodule %r for the distributed path "
            "(ShardTensor inputs unwrapped to local). If a result diverges, "
            "exclude it via DomainConfig.scripted_marshal_exclude or declare a "
            "JitAdapter; disable auto-discovery with scripted_marshal='declared'.",
            name,
        )
    return mementos


def restore_auto_marshalled(mementos: list[tuple[Any, str, Any]]) -> None:
    """Undo :func:`auto_marshal_scripted_submodules` (reverse order)."""
    for parent, child, original in reversed(mementos):
        setattr(parent, child, original)


@dataclass(frozen=True)
class JitAdapter:
    """Replace a ``@torch.jit.script``-decorated module-level helper so a
    ShardTensor can cross it safely on the distributed path.

    Two modes:

    * ``mode="marshal"``: wrap the *original* scripted op with
      :func:`make_marshaller` at install time — unwrap ShardTensor→local, run
      the scripted op, rewrap. Keeps the op scripted/fused; no hand-written copy.
    * ``mode="eager"``: swap in ``replacement`` (a hand-written plain-Python
      equivalent so ShardTensor ``__torch_function__`` fires inside it). The
      author keeps that copy in sync with upstream.
    """

    module_path: str
    attr_name: str
    replacement: Callable[..., Any] | None = None
    mode: Literal["eager", "marshal"] = "eager"
    install_site: str = field(default="", compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.install_site:
            object.__setattr__(self, "install_site", _capture_call_site())

    def _target_str(self) -> str:
        return f"{self.module_path}.{self.attr_name}"

    def install(self) -> dict[str, Any]:
        """Swap in the replacement at ``module.attr``.

        ``mode="marshal"``: build a :func:`make_marshaller` around the *current*
        attribute (the original scripted op) — no hand-written copy needed.

        ``mode="eager"``: swap in ``self.replacement``. ``replacement=None`` is
        the declaration-only form: the entry is in the spec for diagnostics, but
        the wrapper's ``distributed_setup`` hook swaps the attribute (it closes
        over per-run partition metadata). The registry no-ops here.
        """
        import importlib  # noqa: PLC0415

        if self.mode == "marshal":
            module = importlib.import_module(self.module_path)
            original = getattr(module, self.attr_name)
            logger.info(
                "JitAdapter.install: marshalling %s (%s)",
                self._target_str(),
                type(original).__name__,
            )
            setattr(module, self.attr_name, make_marshaller(original))
            return {"module": module, "original": original}

        if self.replacement is None:
            return {"deferred": True}

        module = importlib.import_module(self.module_path)
        original = getattr(module, self.attr_name)
        logger.info(
            "JitAdapter.install: replacing %s (%s) with %s",
            self._target_str(),
            type(original).__name__,
            getattr(self.replacement, "__qualname__", str(self.replacement)),
        )
        setattr(module, self.attr_name, self.replacement)
        return {"module": module, "original": original}

    def restore(self, memento: dict[str, Any]) -> None:
        """Reverse :meth:`install`: put the original attribute back."""
        if memento.get("deferred"):
            return
        setattr(memento["module"], self.attr_name, memento["original"])

    def describe(
        self, state: AdapterState = "pending", error: str | None = None
    ) -> AdapterStatus:
        """Return an :class:`AdapterStatus` snapshot of this adapter."""
        return AdapterStatus(
            kind="jit",
            target=self._target_str(),
            state=state,
            install_site=self.install_site,
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-roundtrippable dict."""
        return {
            "kind": "jit",
            "module_path": self.module_path,
            "attr_name": self.attr_name,
            "replacement": _replacement_qualname(self.replacement),
            "mode": self.mode,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JitAdapter":
        """Reconstruct a :class:`JitAdapter` from :meth:`to_dict` output."""
        return cls(
            module_path=d["module_path"],
            attr_name=d["attr_name"],
            replacement=_resolve_replacement(d.get("replacement")),
            mode=d.get("mode", "eager"),
        )


# ----------------------------------------------------------------------
# PythonAdapter — replace a plain-Python helper.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class PythonAdapter:
    """Replace a plain-Python module-level helper with a
    distributed-aware version.

    Unlike :class:`JitAdapter`, the helper isn't ``@torch.jit.script``
    — it's a normal Python function whose single-process tensor-layout
    assumptions break under partition. Canonical case:
    ``aimnet.nbops.mol_sum`` reading ``mol_idx[-1] + 1`` for its output
    size.

    ``replacement`` may be ``None`` if it must be built at install time
    by the wrapper (e.g. it closes over per-run partition metadata
    that's only available in ``DistributedModel.__enter__``). Pass a
    factory via the wrapper's ``distributed_setup`` hook or override
    :meth:`install` in a subclass.
    """

    module_path: str
    attr_name: str
    replacement: Callable[..., Any] | None = None
    install_site: str = field(default="", compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.install_site:
            object.__setattr__(self, "install_site", _capture_call_site())

    def _target_str(self) -> str:
        return f"{self.module_path}.{self.attr_name}"

    def install(self) -> dict[str, Any]:
        """Swap in the plain-Python replacement at ``module.attr``.

        ``replacement=None`` = declaration-only: the wrapper's
        ``distributed_setup`` is responsible for swapping the attr.
        See note on :meth:`JitAdapter.install` for the rationale.
        """
        if self.replacement is None:
            return {"deferred": True}
        import importlib  # noqa: PLC0415

        module = importlib.import_module(self.module_path)
        original = getattr(module, self.attr_name)
        logger.info(
            "PythonAdapter.install: replacing %s with %s",
            self._target_str(),
            getattr(self.replacement, "__qualname__", str(self.replacement)),
        )
        setattr(module, self.attr_name, self.replacement)
        return {"module": module, "original": original}

    def restore(self, memento: dict[str, Any]) -> None:
        """Reverse :meth:`install`: put the original attribute back."""
        if memento.get("deferred"):
            return
        setattr(memento["module"], self.attr_name, memento["original"])

    def describe(
        self, state: AdapterState = "pending", error: str | None = None
    ) -> AdapterStatus:
        """Return an :class:`AdapterStatus` snapshot of this adapter."""
        return AdapterStatus(
            kind="python",
            target=self._target_str(),
            state=state,
            install_site=self.install_site,
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-roundtrippable dict."""
        return {
            "kind": "python",
            "module_path": self.module_path,
            "attr_name": self.attr_name,
            "replacement": _replacement_qualname(self.replacement),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PythonAdapter":
        """Reconstruct a :class:`PythonAdapter` from :meth:`to_dict` output."""
        return cls(
            module_path=d["module_path"],
            attr_name=d["attr_name"],
            replacement=_resolve_replacement(d.get("replacement")),
        )


def FunctionAdapter(  # noqa: N802 — constructor-style factory
    func: Any,
    replacement: Callable[..., Any] | None = None,
) -> PythonAdapter:
    """Adapt a module-level function named by the **real function object**.

    Ergonomic constructor for :class:`PythonAdapter`: derives the import path
    from ``func`` itself, so callers write
    ``FunctionAdapter(mol_sum, _owned_mol_sum)`` instead of spelling the module
    path and attribute name as strings. Returns a :class:`PythonAdapter`, so
    install / restore / serialization are unchanged.

    Safe by construction: the target is resolved as ``func.__module__`` +
    ``func.__name__`` and **verified** to be that exact object. A re-exported
    name (the same function bound under more than one module — e.g. fairchem's
    ``reduce_node_to_system``) can't be distinguished from the resolved object,
    so it raises and asks for an explicit
    ``PythonAdapter(module_path=..., attr_name=...)`` naming the binding.
    """
    import importlib  # noqa: PLC0415

    module_path = getattr(func, "__module__", None)
    attr_name = getattr(func, "__name__", None)
    if not module_path or not attr_name:
        raise TypeError(
            "FunctionAdapter expects a module-level function object with "
            "__module__ and __name__."
        )
    resolved = getattr(importlib.import_module(module_path), attr_name, None)
    if resolved is not func:
        raise ValueError(
            f"FunctionAdapter cannot bind {attr_name!r}: it is not the same "
            f"object as {module_path}.{attr_name} (likely re-exported under "
            f"another module). Use PythonAdapter(module_path=..., attr_name=...) "
            f"to name the exact module binding to patch."
        )
    return PythonAdapter(
        module_path=module_path, attr_name=attr_name, replacement=replacement
    )


# ----------------------------------------------------------------------
# MethodAdapter — wrap a class method (call-original), not replace it.
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class MethodAdapter:
    """Wrap a class method: intercept the call, transform an argument, then
    invoke the original — as opposed to :class:`PythonAdapter` /
    :class:`JitAdapter`, which *replace* a module-level function outright.

    Some third-party models need a *method-internal* arg wrap that can't be
    expressed by replacing a module-level helper. Canonical case: AIMNet2's
    ``aimnet.modules.aev.ConvSV.forward`` — its ``conv_q`` path
    (``d2features=False``) indexes a charges-derived arg that has lost
    ShardTensor metadata through the MLP, so it must be re-wrapped as a
    sharded ShardTensor before the stock ``a.index_select(0, nbmat.flatten())``,
    or it reads global ``nbmat`` indices off a rank-local tensor.

    ``replacement`` is a *wrapping* function ``(original, *args, **kwargs)``:
    :meth:`install` captures the original (unbound) method and binds it as the
    first argument, so the replacement can transform args and call through.
    Reading per-step routing off the method's own arguments (e.g. ConvSV's
    ``data`` dict, which :meth:`AIMNet2Wrapper.adapt_input` populates) keeps it
    free of ambient context — the same discipline the PythonAdapter
    replacements follow. Declared on a spec's ``third_party_helpers`` so the
    framework's :class:`AdapterRegistry` installs + restores it; no wrapper
    ``distributed_setup`` hook needed.

    Normally name the **real imported class**, not a string path::

        MethodAdapter(ConvSV, "forward", _rewrap_conv_q)   # real class + method

    The string form ``MethodAdapter("module", "Class", "method", ...)`` is also
    accepted (disambiguated by the first argument's type), so serialized dicts
    round-trip unchanged.
    """

    module_path: str
    class_name: str
    method_name: str
    replacement: Callable[..., Any] | None = None
    mode: Literal["wrap", "marshal"] = "wrap"
    install_site: str = field(default="", compare=False, hash=False)

    def __init__(
        self,
        target: type | str | None = None,
        method_or_class: str | None = None,
        replacement_or_method: Any = None,
        replacement: Callable[..., Any] | None = None,
        *,
        module_path: str | None = None,
        class_name: str | None = None,
        method_name: str | None = None,
        mode: Literal["wrap", "marshal"] = "wrap",
        install_site: str = "",
    ) -> None:
        if isinstance(target, type):
            # Class form: MethodAdapter(RealClass, "method", replacement?)
            module_path = target.__module__
            class_name = target.__qualname__
            method_name = method_or_class
            if replacement is None:
                replacement = replacement_or_method
        elif target is not None:
            # String positional form:
            # MethodAdapter("module", "Class", "method", replacement?)
            module_path = target
            class_name = method_or_class
            method_name = replacement_or_method
        # else: fully-keyword form (module_path=/class_name=/method_name=)
        if module_path is None or class_name is None or method_name is None:
            raise TypeError(
                "MethodAdapter requires a class + method: either "
                "MethodAdapter(RealClass, 'method', fn) or "
                "MethodAdapter('module.path', 'Class', 'method')."
            )
        object.__setattr__(self, "module_path", module_path)
        object.__setattr__(self, "class_name", class_name)
        object.__setattr__(self, "method_name", method_name)
        object.__setattr__(self, "replacement", replacement)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "install_site", install_site or _capture_call_site())

    def _target_str(self) -> str:
        return f"{self.module_path}.{self.class_name}.{self.method_name}"

    def install(self) -> dict[str, Any]:
        """Wrap ``class.method`` so calls route through ``replacement``.

        ``replacement=None`` is the declaration-only form (registry no-ops);
        present for parity with the other adapters.

        ``mode="marshal"``: wrap the *whole method* (e.g. e3nn
        ``SphericalHarmonics.forward``) with :func:`make_marshaller` — the
        smallest region that resolves BOTH (A) a scripted call inside the method
        AND (B) a subsequent in-place mutation of a ShardTensor (e.g.
        ``sh.mul_(cat)``): the marshaller unwraps the ShardTensor input to its
        local ONCE, so the scripted op and the in-place op both run on a plain
        local tensor (eager, ``torch.compiler.disable`` graph-break), then the
        output is re-wrapped. Subsumes a separate ``JitAdapter`` on the inner
        scripted function. (B is a PyTorch AOT limitation — in-place mutation of
        a subclass that is a graph input across a graph break — reproduced on
        stock ``TwoTensor``; keeping the region eager sidesteps it.)
        """
        import functools  # noqa: PLC0415
        import importlib  # noqa: PLC0415

        if self.mode == "marshal":
            cls = getattr(importlib.import_module(self.module_path), self.class_name)
            original = getattr(cls, self.method_name)
            marshalled = make_marshaller(original)

            @functools.wraps(original)
            def _marshalled_method(*args: Any, **kwargs: Any) -> Any:
                return marshalled(*args, **kwargs)

            logger.info("MethodAdapter.install: marshalling %s", self._target_str())
            setattr(cls, self.method_name, _marshalled_method)
            return {"cls": cls, "original": original}

        if self.replacement is None:
            return {"deferred": True}

        cls = getattr(importlib.import_module(self.module_path), self.class_name)
        original = getattr(cls, self.method_name)
        replacement = self.replacement

        @functools.wraps(original)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            return replacement(original, *args, **kwargs)

        logger.info(
            "MethodAdapter.install: wrapping %s with %s",
            self._target_str(),
            getattr(replacement, "__qualname__", str(replacement)),
        )
        setattr(cls, self.method_name, _wrapped)
        return {"cls": cls, "original": original}

    def restore(self, memento: dict[str, Any]) -> None:
        """Reverse :meth:`install`: put the original method back."""
        if memento.get("deferred"):
            return
        setattr(memento["cls"], self.method_name, memento["original"])

    def describe(
        self, state: AdapterState = "pending", error: str | None = None
    ) -> AdapterStatus:
        """Return an :class:`AdapterStatus` snapshot of this adapter."""
        return AdapterStatus(
            kind="method",
            target=self._target_str(),
            state=state,
            install_site=self.install_site,
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-roundtrippable dict."""
        return {
            "kind": "method",
            "module_path": self.module_path,
            "class_name": self.class_name,
            "method_name": self.method_name,
            "replacement": _replacement_qualname(self.replacement),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MethodAdapter":
        """Reconstruct a :class:`MethodAdapter` from :meth:`to_dict` output."""
        return cls(
            module_path=d["module_path"],
            class_name=d["class_name"],
            method_name=d["method_name"],
            replacement=_resolve_replacement(d.get("replacement")),
        )


# ----------------------------------------------------------------------
# ModuleForwardAdapter — swap a specific module INSTANCE's forward.
# ----------------------------------------------------------------------


@dataclass
class ModuleForwardAdapter:
    """Swap one ``nn.Module`` *instance*'s ``forward`` for the DD scope, restore
    on exit — as opposed to :class:`MethodAdapter`, which swaps a method on the
    *class* (all instances).

    Use when the target forward is bound per-instance (a closure monkeypatched
    onto the object), so a class-level swap can't reach it. Canonical case: a
    cuequivariance ``conv_tp`` whose fused message-pass forward is set on the
    instance by ``mace.modules.wrapper_ops.with_cueq_conv_fusion``; under DD that
    fused kernel hides the gather/scatter from halo correction, so the spec
    swaps in an external gather + scatter forward (built model-bound, like
    :func:`neighbor_refresh_adapters`).

    Because the framework installs spec adapters only inside the distributed
    scope, ``replacement`` carries no DD branch of its own — single-process keeps
    the original (fused) forward untouched. Built with a live module instance, so
    it is rebuilt per-process from the wrapper's ``distribution_spec`` rather than
    round-tripped through :meth:`to_dict` (the instance can't serialize).
    """

    module: Any
    replacement: Callable[..., Any]
    label: str = "module_forward"
    install_site: str = field(default="", compare=False, hash=False)

    def __post_init__(self) -> None:
        if not self.install_site:
            object.__setattr__(self, "install_site", _capture_call_site())

    def _target_str(self) -> str:
        return f"{type(self.module).__module__}.{type(self.module).__qualname__}.forward"

    def install(self) -> dict[str, Any]:
        """Set ``module.forward = replacement``, capturing the prior binding.

        ``forward`` is read off the instance ``__dict__`` so restore can tell a
        per-instance override (put the old callable back) from the inherited
        class method (drop the instance attribute).
        """
        had = "forward" in self.module.__dict__
        prev = self.module.__dict__.get("forward")
        logger.info("ModuleForwardAdapter.install: swapping %s", self._target_str())
        self.module.forward = self.replacement
        return {"had": had, "prev": prev}

    def restore(self, memento: dict[str, Any]) -> None:
        """Reverse :meth:`install`: restore the per-instance forward or, if there
        was none, drop the instance attribute to fall back to the class method."""
        if memento["had"]:
            self.module.forward = memento["prev"]
        else:
            self.module.__dict__.pop("forward", None)

    def describe(
        self, state: AdapterState = "pending", error: str | None = None
    ) -> AdapterStatus:
        """Return an :class:`AdapterStatus` snapshot of this adapter."""
        return AdapterStatus(
            kind="method",
            target=self._target_str(),
            state=state,
            install_site=self.install_site,
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        """Best-effort serialization. The bound instance + closure don't
        round-trip; the wrapper rebuilds this from ``distribution_spec`` per
        process. Emits a marker so :meth:`DistributionSpec.to_dict` doesn't fail.
        """
        return {"kind": "module_forward", "target": self._target_str(), "label": self.label}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ModuleForwardAdapter":
        """Declaration-only reconstruction (install/restore no-op). The real,
        model-bound adapter is rebuilt by the wrapper's ``distribution_spec``."""
        return _DeclaredModuleForwardAdapter(label=d.get("label", "module_forward"))


class _DeclaredModuleForwardAdapter(ModuleForwardAdapter):
    """Deserialized placeholder: no live module, so install/restore no-op."""

    def __init__(self, label: str = "module_forward") -> None:
        object.__setattr__(self, "module", None)
        object.__setattr__(self, "replacement", lambda *a, **k: None)
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "install_site", "")

    def _target_str(self) -> str:
        return f"<declared module_forward:{self.label}>"

    def install(self) -> dict[str, Any]:
        return {"deferred": True}

    def restore(self, memento: dict[str, Any]) -> None:
        return None


# Discriminated union of helper-style adapters that go in
# ``DistributionSpec.third_party_helpers``. (OpAdapter lives in
# ``DistributionSpec.custom_ops`` — different slot, same lifecycle protocol.)
ThirdPartyHelper = "JitAdapter | PythonAdapter | MethodAdapter | ModuleForwardAdapter"


def _replacement_qualname(fn: Callable | None) -> str | None:
    """Encode a function reference as ``"<module>:<qualname>"`` for
    serialization. Returns ``None`` for unresolvable references
    (closures, lambdas) — caller must rebuild at install time."""
    if fn is None:
        return None
    mod = getattr(fn, "__module__", None)
    name = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", None)
    return f"{mod}:{name}" if (mod and name) else None


def _resolve_replacement(qualname: str | None) -> Callable | None:
    """Inverse of :func:`_replacement_qualname`. Best-effort: returns
    ``None`` if the qualname doesn't resolve."""
    if not qualname or ":" not in qualname:
        return None
    import importlib  # noqa: PLC0415

    mod_path, qual = qualname.split(":", 1)
    try:
        mod = importlib.import_module(mod_path)
    except Exception:
        return None
    obj: Any = mod
    for part in qual.split("."):
        if part.startswith("<"):
            return None
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


# Registry mapping serialized "kind" → adapter class. Subclasses that
# need to survive ``to_dict``/``from_dict`` round-trips (e.g. the
# validator's spawn boundary) register themselves here via
# :func:`register_adapter_kind`.
_ADAPTER_KIND_REGISTRY: dict[str, type] = {}


def register_adapter_kind(kind: str, cls: type) -> None:
    """Register *cls* as the adapter type for serialized ``"kind": <kind>``.

    Use this when a custom :class:`PythonAdapter` / :class:`JitAdapter`
    subclass needs to round-trip through :meth:`MLIPSpec.to_dict` /
    :meth:`MLIPSpec.from_dict` (e.g. when the validator harness ships
    the spec across an ``mp.spawn`` boundary). The subclass must:

    * Override :meth:`to_dict` to emit a unique ``"kind"`` value.
    * Provide a :meth:`from_dict` ``@classmethod`` that reconstructs
      the same fields its ``to_dict`` emitted.

    Re-registration with the same ``kind`` name overrides the prior
    binding (allows test fixtures to swap implementations cleanly).
    """
    _ADAPTER_KIND_REGISTRY[kind] = cls


# Register the built-in kinds. A model that needs a bespoke adapter kind can
# subclass one of these and register it via ``register_adapter_kind`` at import.
register_adapter_kind("jit", JitAdapter)
register_adapter_kind("python", PythonAdapter)
register_adapter_kind("method", MethodAdapter)
register_adapter_kind("module_forward", ModuleForwardAdapter)


def _adapter_from_dict(d: dict[str, Any]) -> "JitAdapter | PythonAdapter":
    """Discriminate a third-party helper dict by its ``"kind"``,
    dispatching through :data:`_ADAPTER_KIND_REGISTRY`."""
    kind = d.get("kind")
    cls = _ADAPTER_KIND_REGISTRY.get(kind)
    if cls is None:
        raise ValueError(
            f"unknown third-party helper kind {kind!r}; expected one of "
            f"{sorted(_ADAPTER_KIND_REGISTRY)}. Subclasses of PythonAdapter "
            f"/ JitAdapter that introduce new kinds must call "
            f"``register_adapter_kind(kind, cls)`` at import time."
        )
    return cls.from_dict(d)


# ----------------------------------------------------------------------
# AdapterRegistry — owns lifecycle of a set of adapters.
# ----------------------------------------------------------------------


# A registered adapter together with its install state.
@dataclass
class _Handle:
    adapter: Any  # OpAdapter | JitAdapter | PythonAdapter
    state: AdapterState = "pending"
    memento: dict[str, Any] | None = None
    error: str | None = None


class AdapterRegistry:
    """Owns the install / restore lifecycle for a set of adapters.

    :class:`DistributedModel` instantiates a registry on
    ``__enter__``, calls :meth:`install` with the adapters declared on
    the spec's :class:`DistributionSpec`, and calls :meth:`restore` on
    ``__exit__``.

    ``install`` is fail-fast: if any adapter raises, all
    previously-installed adapters are rolled back before the exception
    propagates. ``restore`` is best-effort: failures are logged but
    don't raise (so a single broken adapter doesn't block teardown of
    the others).
    """

    def __init__(self) -> None:
        self._handles: list[_Handle] = []

    def install(self, adapters: Sequence[Any]) -> None:
        """Install each adapter in order. Rolls back partial state on
        failure and re-raises."""
        for adapter in adapters:
            handle = _Handle(adapter=adapter)
            self._handles.append(handle)
            try:
                handle.memento = adapter.install()
                handle.state = "installed"
            except Exception as e:
                handle.state = "failed"
                handle.error = repr(e)
                logger.error(
                    "AdapterRegistry.install failed for %s: %s",
                    adapter._target_str(),
                    e,
                )
                # Roll back any earlier successful installs and re-raise.
                self.restore()
                raise

    def restore(self) -> None:
        """Restore all installed adapters in reverse order. Failures
        are logged; never raises."""
        for handle in reversed(self._handles):
            if handle.state != "installed":
                continue
            try:
                handle.adapter.restore(handle.memento)
                handle.state = "restored"
            except Exception as e:  # noqa: BLE001
                handle.error = repr(e)
                logger.warning(
                    "AdapterRegistry.restore failed for %s: %s",
                    handle.adapter._target_str(),
                    e,
                )

    def list_active(self) -> list[AdapterStatus]:
        """Return the introspectable lifecycle status of each adapter
        registered in this registry."""
        return [h.adapter.describe(state=h.state, error=h.error) for h in self._handles]
