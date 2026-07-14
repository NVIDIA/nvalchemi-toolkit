# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import threading
from collections.abc import Iterable, Mapping
from contextlib import contextmanager
from typing import Callable, Sequence, cast

import torch
try:  # torch>=2.5 exposes the traceable-subclass constructor marker
    from torch._export.wrappers import (
        mark_subclass_constructor_exportable_experimental,
    )
except Exception:  # pragma: no cover - older torch
    def mark_subclass_constructor_exportable_experimental(fn):  # type: ignore[misc]
        return fn
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, _mesh_resources
from torch.distributed.tensor import DTensor, distribute_module
from torch.distributed.tensor._dtensor_spec import (
    TensorMeta,
)
from torch.distributed.tensor.placement_types import (
    Partial,
    Placement,
    Replicate,
    Shard,
)

from physicsnemo.distributed import DistributedManager
from ._shard_redistribute import (
    ShardRedistribute,
    redistribute_local_shard_tensor,
)
from ._shard_tensor_spec import (
    ShardTensorSpec,
    _infer_shard_tensor_spec_from_local_chunks,
    _stride_from_contiguous_shape_C_style,
    compute_sharding_shapes_from_chunking_global_shape,
)

aten = torch.ops.aten


# ======================================================================

# ============================================================================
# Layer 1 -- Semi-private conversions (no autograd, no spec inference)
# ============================================================================


def _shard_tensor_to_dtensor(st: "ShardTensor") -> DTensor:
    r"""Convert a ShardTensor to a plain DTensor (no autograd).

    Creates a DTensor sharing the same ``_local_tensor`` and ``_spec``.
    Use for dispatch or inside backward when building a DTensor gradient.
    """
    if hasattr(torch.Tensor, "_dtensor__new__"):
        dtensor = torch.Tensor._dtensor__new__(
            DTensor, st._local_tensor, st._spec, requires_grad=st.requires_grad
        )
    else:
        dtensor = torch.Tensor._make_wrapper_subclass(
            DTensor,
            st._spec.tensor_meta.shape,
            strides=st._spec.tensor_meta.stride,
            dtype=st.dtype,
            device=st.device,
            layout=st.layout,
            requires_grad=st.requires_grad,
        )
    dtensor._local_tensor = st._local_tensor
    dtensor._spec = st._spec
    return dtensor


def _dtensor_to_shard_tensor(dtensor: DTensor, spec: ShardTensorSpec) -> "ShardTensor":
    r"""Promote a DTensor to a ShardTensor (no autograd).

    Callers must supply a resolved ``spec``.  Use inside backward (with spec
    from ctx) or after resolving a spec via :func:`_resolve_spec_for_dtensor`.
    """
    if isinstance(dtensor, ShardTensor):
        # Shortcut if we're already a ShardTensor:
        return dtensor
    st = ShardTensor(local_tensor=dtensor._local_tensor,
        spec=spec,
        requires_grad=dtensor.requires_grad,
    )
    return st


# ============================================================================
# Layer 2 -- Autograd Functions (use Layer 1 inside fwd / bwd)
# ============================================================================


class _DTensorToShardTensor(torch.autograd.Function):
    r"""Differentiable promotion: DTensor -> ShardTensor.

    This is to always connect the graphs for the backward pass
    when we have to use a fallback option.

    Forward: :func:`_dtensor_to_shard_tensor`.
    Backward: :func:`_shard_tensor_to_dtensor`.
    """

    @staticmethod
    def forward(dtensor: DTensor, spec: ShardTensorSpec) -> "ShardTensor":
        return _dtensor_to_shard_tensor(dtensor, spec)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        # Nothing to save; backward only needs grad_output.
        pass

    @staticmethod
    def backward(ctx, grad_output: "ShardTensor"):
        return _shard_tensor_to_dtensor(grad_output), None


class _ShardTensorToDTensor(torch.autograd.Function):
    r"""Differentiable conversion: ShardTensor -> DTensor.

    This is to always connect the graphs for the backward pass
    when we have to use a fallback option.

    Forward: :func:`_shard_tensor_to_dtensor` (caches spec).
    Backward: :func:`_dtensor_to_shard_tensor` (reuses cached spec).
    """

    @staticmethod
    def forward(st: "ShardTensor") -> DTensor:
        return _shard_tensor_to_dtensor(st)

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        (st,) = inputs
        ctx.shard_tensor_spec = st._spec

    @staticmethod
    def backward(ctx, grad_output: DTensor):
        return (_dtensor_to_shard_tensor(grad_output, ctx.shard_tensor_spec),)


# ============================================================================
# Layer 3 -- Smart single-tensor converters (auto-diff when grad_fn present)
# ============================================================================


def _resolve_spec_for_dtensor(
    dtensor: DTensor, input_args: tuple = ()
) -> ShardTensorSpec:
    r"""Resolve a ShardTensorSpec for *dtensor*.

    Tries to reuse a spec from a ShardTensor in *input_args* whose
    ``tensor_meta`` and ``placements`` match.  Falls back to chunk-based
    inference (no communication).
    """
    for arg in input_args:
        if (
            isinstance(arg, ShardTensor)
            and dtensor._spec.tensor_meta == arg._spec.tensor_meta
            and dtensor._spec.placements == arg._spec.placements
        ):
            return arg._spec
    return _infer_shard_tensor_spec_from_local_chunks(
        dtensor._local_tensor,
        dtensor._spec.mesh,
        dtensor._spec.placements,
        sharding_shapes="chunk",
        global_shape=dtensor.shape,
    )


# This is a thread-safe reentry guard.
# Goal is to prevent recursion into the fallback conversion paths.
_conversion_guard = threading.local()


def _conversion_active() -> bool:
    r"""Return whether ShardTensor<->DTensor conversion is currently active."""
    return getattr(_conversion_guard, "depth", 0) > 0


@contextmanager
def _conversion_scope():
    r"""Re-entrant conversion guard for cast-down/cast-up paths."""
    previous_depth = getattr(_conversion_guard, "depth", 0)
    _conversion_guard.depth = previous_depth + 1
    try:
        yield
    finally:
        if previous_depth == 0:
            delattr(_conversion_guard, "depth")
        else:
            _conversion_guard.depth = previous_depth


def _dispatch_fallback_via_dtensor(
    func: torch._ops.OpOverload,
    args: tuple[object, ...],
    kwargs: dict[str, object] | None = None,
) -> object:
    r"""Execute an ATen op through DTensor fallback using PURE data conversion.

    Native Autograd wraps this hook, so we must NOT build an internal graph
    using .apply(). We just do the math and let PyTorch track the outer graph.
    """
    with _conversion_scope():
        converted_args = tuple(
            _convert_args_to_dtensor(arg, use_autograd=False) for arg in args
        )
        converted_kwargs = {
            k: _convert_args_to_dtensor(v, use_autograd=False)
            for k, v in (kwargs or {}).items()
        }

    dispatch_res = func(*converted_args, **(converted_kwargs or {}))

    with _conversion_scope():
        return _convert_results_to_shard_tensor(dispatch_res, args, use_autograd=False)


def _torch_function_fallback_via_dtensor(
    func: Callable,
    args: tuple[object, ...],
    kwargs: dict[str, object] | None = None,
) -> object:
    r"""Execute a __torch_function__ fallback through DTensor safely.

    Because this executes at the Python API level (above Autograd), we MUST
    use autograd functions (.apply) to bridge the tracking manually.
    """
    with _conversion_scope():
        converted_args = tuple(
            _convert_args_to_dtensor(arg, use_autograd=True) for arg in args
        )
        converted_kwargs = {
            k: _convert_args_to_dtensor(v, use_autograd=True)
            for k, v in (kwargs or {}).items()
        }

    with torch._C.DisableTorchFunctionSubclass():
        result = func(*converted_args, **converted_kwargs)

    with _conversion_scope():
        return _convert_results_to_shard_tensor(result, args, use_autograd=True)


# ============================================================================
# Layer 4 -- Recurse utilities (walk args / kwargs / results)
# ============================================================================


def _convert_args_to_dtensor(arg: object, use_autograd: bool = False) -> object:
    r"""Recursively replace ShardTensors with DTensors.

    If use_autograd is True, uses Layer 2 to preserve the graph connection.
    """
    match arg:
        case ShardTensor():
            if use_autograd and arg.requires_grad and torch.is_grad_enabled():
                return _ShardTensorToDTensor.apply(arg)
            return _shard_tensor_to_dtensor(arg)
        case DTensor():
            # DTensor can be iterable; exit early deliberately
            return arg
        case Mapping():
            return type(arg)(
                {k: _convert_args_to_dtensor(v, use_autograd) for k, v in arg.items()}
            )
        case tuple():
            return tuple(_convert_args_to_dtensor(a, use_autograd) for a in arg)
        case list():
            return [_convert_args_to_dtensor(a, use_autograd) for a in arg]
        case _:
            return arg


def _convert_results_to_shard_tensor(
    result: object, input_args: tuple, use_autograd: bool = False
) -> object:
    r"""Recursively replace DTensors with ShardTensors in an op result.

    If use_autograd is True, uses Layer 2 to preserve the graph connection.
    Handles None returns gracefully for inplace ATen operations.
    """
    if result is None:
        return None

    if isinstance(result, DTensor):
        spec = _resolve_spec_for_dtensor(result, input_args)

        # If autograd graph connection is requested AND the DTensor actually
        # requires tracking (it has a grad_fn or requires_grad is active)
        if (
            use_autograd
            and torch.is_grad_enabled()
            and (result.grad_fn is not None or result.requires_grad)
        ):
            return _DTensorToShardTensor.apply(result, spec)

        return _dtensor_to_shard_tensor(result, spec)

    if isinstance(result, Mapping):
        return type(result)(
            {
                k: _convert_results_to_shard_tensor(v, input_args, use_autograd)
                for k, v in result.items()
            }
        )

    # Explicit allowlist mirroring _convert_args_to_dtensor: only walk into
    # plain tuple / list containers. A generic Iterable check would crash on
    # things like torch.UntypedStorage (iterable over bytes) or torch.Tensor
    # because their constructors don't accept a generator. Note: namedtuples
    # degrade to plain tuple here, same as in the args walker.
    if isinstance(result, tuple):
        return tuple(
            _convert_results_to_shard_tensor(d, input_args, use_autograd)
            for d in result
        )

    if isinstance(result, list):
        return [
            _convert_results_to_shard_tensor(d, input_args, use_autograd)
            for d in result
        ]

    return result


class _ToTorchTensor(torch.autograd.Function):
    r"""Autograd function to convert a ShardTensor to a regular PyTorch tensor.

    This class handles the conversion from ShardTensor to ``torch.Tensor`` in both
    forward and backward passes, maintaining proper gradient flow. Slices the
    ShardTensor to the local component only on the current rank.
    """

    @staticmethod
    def forward(
        input: "ShardTensor",
        grad_placements: Sequence[Placement] | None = None,
    ) -> torch.Tensor:
        r"""Convert ShardTensor to torch.Tensor in forward pass.

        Parameters
        ----------
        input : ShardTensor
            ShardTensor to convert.
        grad_placements : Sequence[Placement], optional
            Sequence of placements to use for gradients.

        Returns
        -------
        torch.Tensor
            Local tensor representation of the ShardTensor.
        """
        # # JUST LIKE DTENSOR:
        # # We need to return a fresh Tensor object there as autograd metadata
        # # will be inplaced into it. So we don't want to pollute the Tensor
        # # object stored in the _local_tensor of this ShardTensor.
        # return local_tensor.view_as(local_tensor)

        # Force the local view to inherit the requires_grad state of the ShardTensor
        local_tensor = input._local_tensor
        res = local_tensor.view_as(local_tensor)
        res.requires_grad_(input.requires_grad)
        return res

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        r"""Save the source ShardTensorSpec and optional grad_placements."""
        input, grad_placements = inputs
        ctx.shard_tensor_spec = input._spec
        ctx.grad_placements = grad_placements

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple["ShardTensor", None]:
        r"""Convert gradient torch.Tensor back to ShardTensor in backward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved tensors/variables from forward.
        grad_output : torch.Tensor
            Gradient tensor to convert back to ShardTensor.

        Returns
        -------
        Tuple[ShardTensor, None]
            Tuple containing the ShardTensor gradient and None for
            grad_placements gradient (not differentiable).
        """
        shard_tensor_spec = ctx.shard_tensor_spec
        mesh = shard_tensor_spec.mesh
        if ctx.grad_placements is not None:
            if ctx.grad_placements != shard_tensor_spec.placements:
                grad_placements = ctx.grad_placements
                grad_sharding_shapes = "infer"
            else:
                # If the placements are the same as the input placements,
                # we reuse the sharding sizes from the input placements.
                grad_placements = ctx.grad_placements
                grad_sharding_shapes = shard_tensor_spec._sharding_shapes
        else:
            grad_placements = shard_tensor_spec.placements
            grad_sharding_shapes = shard_tensor_spec._sharding_shapes
        if grad_sharding_shapes is None:
            grad_sharding_shapes = "infer"
        # Generate a spec based on grad outputs and the expected placements:
        grad_tensor_spec = _infer_shard_tensor_spec_from_local_chunks(
            grad_output, mesh, grad_placements, grad_sharding_shapes
        )

        return (
            ShardTensor(
                grad_output, grad_tensor_spec, requires_grad=grad_output.requires_grad
            ),
            None,
        )


class _FromTorchTensor(torch.autograd.Function):
    r"""Autograd function for converting a torch.Tensor to a ShardTensor.

    This class handles the forward and backward passes for converting between
    ``torch.Tensor`` and ShardTensor types, maintaining gradient information.

    Global shape information is inferred using collective communication on
    the specified device mesh.

    """

    @staticmethod
    def forward(
        local_input: torch.Tensor,
        device_mesh: DeviceMesh,
        placements: tuple[Placement, ...],
        sharding_shapes: str | dict[int, list[tuple[int, ...]]] = "chunk",
    ) -> "ShardTensor":
        r"""Convert a local torch.Tensor to a ShardTensor in forward pass.

        Parameters
        ----------
        local_input : torch.Tensor
            Local tensor to convert to ShardTensor.
        device_mesh : DeviceMesh
            Device mesh specifying process groups.
        placements : Tuple[Placement, ...]
            Tuple of placement rules for sharding.
        sharding_shapes : Union[str, Dict[int, List[Tuple[int, ...]]]], default="chunk"
            Controls how shard tensor spec is generated:

            - ``"chunk"``: Use ``torch.chunk`` shapes to infer shapes from
              global shape (no communication).
            - ``"infer"``: Use collective communication to infer shapes from
              mesh neighbors.
            - Manual dict mapping mesh dim to list of shard shapes: Use
              provided shapes. Must pass on each rank.

        Returns
        -------
        ShardTensor
            ShardTensor constructed from the local input tensor.
        """
        # This function is simpler than the corresponding DTensor implementation on the surface
        # because under the hood, we have some logic here to infer the sharding shapes.
        shard_tensor_spec = _infer_shard_tensor_spec_from_local_chunks(
            local_input, device_mesh, placements, sharding_shapes
        )

        shard_tensor = ShardTensor(
            local_input,
            shard_tensor_spec,
            requires_grad=local_input.requires_grad,
        )

        return shard_tensor

    @staticmethod
    def setup_context(ctx, inputs, output) -> None:
        r"""Save the source mesh and placements for the backward redistribute."""
        _local_input, device_mesh, placements, _sharding_shapes = inputs
        ctx.previous_placement = placements
        ctx.previous_mesh = device_mesh

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: "ShardTensor",
    ) -> tuple[torch.Tensor, None, None, None]:
        r"""Convert gradient ShardTensor back to torch.Tensor in backward pass.

        Parameters
        ----------
        ctx : torch.autograd.function.FunctionCtx
            Autograd context containing saved tensors/variables from forward.
        grad_output : ShardTensor
            Gradient ShardTensor to convert back to torch.Tensor.

        Returns
        -------
        Tuple[torch.Tensor, None, None, None]
            Tuple containing the local tensor gradient, and None for
            device_mesh, placements, and sharding_shapes gradients
            (not differentiable).

        Raises
        ------
        RuntimeError
            If gradient tensor has different placement than original and
            the original placement contains partial placements.
        """
        previous_placement = ctx.previous_placement
        if grad_output.placements != previous_placement:
            # Automatically redistribute to the previous placement as long as it's not a partial.
            if not any(p.is_partial() for p in previous_placement):
                grad_output = grad_output.redistribute(
                    grad_output._spec.mesh, previous_placement
                )
            else:
                raise RuntimeError(
                    "Resharding gradients with partial placements not implemented"
                )

        return grad_output.to_local(), None, None, None


class ShardTensor(torch.Tensor):
    r"""A distributed tensor class with support for uneven data sharding.

    Similar to PyTorch's native ``DTensor`` but with more flexibility for
    uneven data sharding. Leverages a very similar API to ``DTensor``
    (identical where possible) but deliberately tweaks routines to avoid
    implicit assumptions about tensor sharding.

    The key differences from ``DTensor`` are:

    - Supports uneven sharding where different ranks can have different
      local tensor sizes
    - Tracks and propagates shard size information across operations
    - Handles redistribution of unevenly sharded tensors
    - Provides custom collective operations optimized for uneven sharding

    Like ``DTensor``, operations are dispatched through PyTorch's dispatcher
    system. Most operations work by:

    1. Converting inputs to local tensors
    2. Performing the operation locally
    3. Constructing a new ShardTensor with appropriate sharding spec
    4. Handling any needed communication between ranks

    The class provides methods for:

    - Converting to/from local tensors
    - Redistributing between different sharding schemes
    - Performing collective operations like all_gather and reduce_scatter
    - Basic tensor operations that maintain sharding information

    Attributes
    ----------
    _local_tensor : torch.Tensor
        The local tensor data on this rank.
    _spec : ShardTensorSpec
        The specification defining sharding scheme and metadata.
    """

    _local_tensor: torch.Tensor
    _spec: ShardTensorSpec
    __slots__ = ["_local_tensor", "_spec"]

    # For torch.ops.aten operators (low-level dispatch)
    _dispatch_registry: dict[torch._ops.OpOverload, Callable] = {}
    # Fallback by op name (e.g. "aten.neg.default") when the OpOverload
    # passed to __torch_dispatch__ is not the same object as the one used to register.
    _dispatch_registry_by_name: dict[str, Callable] = {}

    # For Python-level functions (torch.mean, tensor.mean, etc.)
    _function_registry: dict[Callable, Callable] = {}

    # For custom functions registered with PyTorch,
    # it is sometimes necessary to match by name.
    # For instance, if you declare an op with
    #
    # @torch.library.custom_op(
    #    "module::function_name", mutates_args=()
    # )
    # def function_external_to_torch(
    #
    # Then, you likely want to register the handler with
    #
    # ShardTensor.register_named_function_handler("module.function_name.default", handler)
    _named_function_registry: dict[str, Callable] = {}

    # Upon construction of any ShardTensor objects, this will be set to true.
    # Wrappers are triggered dynamically, so the wrapping will be pass-through
    # exclusively until true.
    _enable_shard_patches: bool = False

    @classmethod
    def patches_enabled(cls) -> bool:
        r"""Check whether patches are enabled for this class.

        Returns
        -------
        bool
            ``True`` if shard patches are enabled, ``False`` otherwise.
            Default is ``False`` until a ShardTensor is constructed.
        """
        return cls._enable_shard_patches

    @classmethod
    def register_dispatch_handler(
        cls, op: torch._ops.OpOverload, handler: Callable
    ) -> None:
        r"""Register a handler for a specific PyTorch operator in the dispatch system.

        Parameters
        ----------
        op : torch._ops.OpOverload
            The PyTorch operator to register a handler for.
        handler : Callable
            The handler function to call when the operator is invoked.
        """
        cls._dispatch_registry[op] = handler
        cls._dispatch_registry_by_name[str(op)] = handler

    @classmethod
    def register_function_handler(cls, func: Callable, handler: Callable) -> None:
        r"""Register a handler for a Python-level function or method.

        Parameters
        ----------
        func : Callable
            The Python function to register a handler for.
        handler : Callable
            The handler function to call when the function is invoked.
        """
        cls._function_registry[func] = handler

    @classmethod
    def register_named_function_handler(cls, func_name: str, handler: Callable) -> None:
        r"""Register a named function registered via ``torch.library.custom_op``.

        Parameters
        ----------
        func_name : str
            The string name of the custom op (e.g., ``"module.function_name.default"``).
        handler : Callable
            The handler function to call when the function is invoked.
        """
        cls._named_function_registry[func_name] = handler

    @staticmethod
    @torch._disable_dynamo
    def __new__(
        cls,
        local_tensor: torch.Tensor,
        spec: ShardTensorSpec,
        *,
        requires_grad: bool,
    ) -> "ShardTensor":
        # Bare wrapper construction ONLY — no attribute assignment. The
        # ``_spec`` / ``_local_tensor`` assignment lives in ``__init__`` so that
        # ``torch.compile`` can trace ``ShardTensor(local, spec, ...)`` via
        # Dynamo's native subclass-constructor handling. (A bare ``__new__``
        # that set attributes is traced as Python; Dynamo realizes the
        # half-built wrapper at the ``_spec`` STORE_ATTR and
        # ``is_fake`` -> ``__tensor_flatten__`` reads an unset ``_spec``.)
        # Mirrors ``torch.testing._internal.two_tensor.TwoTensor``.
        ret = torch.Tensor._make_wrapper_subclass(
            cls,
            spec.tensor_meta.shape,
            strides=spec.tensor_meta.stride,
            dtype=local_tensor.dtype,
            device=local_tensor.device,
            layout=local_tensor.layout,
            requires_grad=requires_grad,
        )
        cls._enable_shard_patches = True
        return ret

    @torch._disable_dynamo
    @mark_subclass_constructor_exportable_experimental
    def __init__(
        self,
        local_tensor: torch.Tensor,
        spec: ShardTensorSpec,
        *,
        requires_grad: bool,
    ) -> None:
        # Attribute assignment kept OUT of the Dynamo trace (constructor
        # protocol): ``@torch._disable_dynamo`` + the exportable-constructor
        # marker let ``ShardTensor(local, spec, ...)`` be traced natively
        # without a graph break. ``requires_grad`` is set in ``__new__`` via
        # ``_make_wrapper_subclass`` (the wrapper is a fresh leaf there).
        self._spec = spec
        self._local_tensor = local_tensor

    def __repr__(self) -> str:
        return (
            "ShardTensor("
            f"local_tensor={repr(self._local_tensor)}, "
            f"device_mesh={repr(self._spec.mesh)}, "
            f"placements={repr(self._spec.placements)}"
            ")"
        )

    def __str__(self) -> str:
        # Avoid Tensor/DTensor string formatting paths that can re-enter dispatch.
        return self.__repr__()

    def __format__(self, format_spec: str) -> str:
        # Format as plain Python string to bypass tensor formatting internals.
        return format(str(self), format_spec)

    @property
    def device_mesh(self) -> DeviceMesh:
        """Return the :class:`DeviceMesh` that this tensor is distributed over."""
        return self._spec.mesh

    @property
    def placements(self) -> tuple[Placement, ...]:
        """Return the placement strategy for each mesh dimension."""
        return self._spec.placements

    def __tensor_flatten__(self):
        return ["_local_tensor"], (self._spec, self.requires_grad)

    @staticmethod
    def __tensor_unflatten__(inner_tensors, flatten_spec, outer_size, outer_stride):
        spec, requires_grad = flatten_spec
        local_tensor = inner_tensors["_local_tensor"]
        unflatten_meta = TensorMeta(
            shape=outer_size,
            stride=outer_stride,
            dtype=spec.tensor_meta.dtype,
        )

        # Normalize ``_sharding_shapes`` to plain ``tuple[int, ...]`` entries
        # (never ``torch.Size``). Under dynamo fakeification, ``torch.Size``
        # special-casing converts the contained ints into unbacked SymInts
        # that orphan whenever an op's output drops shard tracking
        # (Partial / Replicate / None), producing
        # ``PendingUnbackedSymbolNotFound`` during AOT tracing.
        #
        # If the incoming spec has no ``_sharding_shapes``, derive them from
        # chunk semantics against the outer global shape -- pure arithmetic,
        # no collectives. This avoids leaving the field ``None``, which would
        # force the next ``sharding_shapes()`` call to ``_all_gather_shard_shapes``
        # (a blocking collective that is not AOT-traceable).
        if spec._sharding_shapes is not None:
            sharding_shapes = {
                mesh_dim: tuple(tuple(s) for s in shapes)
                for mesh_dim, shapes in spec._sharding_shapes.items()
            }
        else:
            chunk_shapes = compute_sharding_shapes_from_chunking_global_shape(
                spec.mesh, spec.placements, tuple(outer_size)
            )
            sharding_shapes = {
                mesh_dim: tuple(tuple(s) for s in shapes)
                for mesh_dim, shapes in chunk_shapes.items()
            }

        unflatten_spec = ShardTensorSpec(
            mesh=spec.mesh,
            placements=spec.placements,
            tensor_meta=unflatten_meta,
            _local_shape=local_tensor.shape,
            _sharding_shapes=sharding_shapes,
        )
        # VENDOR-EDIT: do NOT force ``local_tensor.requires_grad_(requires_grad)``.
        # The wrapper's ``requires_grad`` is already set below via ``__new__`` ->
        # ``_make_wrapper_subclass(requires_grad=...)``. Forcing the inner makes
        # the reconstructed local's ``requires_grad`` disagree with the real
        # tensor's for the (normal) detached-local op-result case
        # (``wrapper.requires_grad=True`` from a ``grad_fn`` but
        # ``_local_tensor.requires_grad=False``): under dynamo re-faking across a
        # graph break, ``assert_metadata_eq`` then trips on the inner
        # (``False != True``). Stock DTensor never forces the inner — it keeps the
        # local's own flag and sets ``requires_grad`` on the wrapper only — which
        # is why DTensor survives the identical op-result/to_local-after-graph-
        # break that MACE hits. Match DTensor: pass ``local_tensor`` unchanged.
        return ShardTensor(local_tensor=local_tensor,
            spec=unflatten_spec,
            requires_grad=requires_grad,
        )

    # -- AOTAutograd tangent coercion hooks ------------------------------------
    # AOTAutograd records the expected tangent metadata at trace time and
    # validates it at backward runtime. When a forward output has a
    # ``Partial`` placement (typical right after a reduction like sum/mean),
    # the tangent flowing back from ``.backward()`` is materialized as
    # ``Replicate`` and AOT raises:
    #     "During the backward, we encountered a tensor subclass where we
    #      guessed its metadata incorrectly."
    # These two hooks mirror DTensor's implementation in
    # ``torch.distributed.tensor._api`` and reconcile the two ends:
    # (1) at trace time, rewrite the expected metadata so any Partial
    #     placement becomes Replicate (so the recorded tangent metadata
    #     matches what runtime will actually produce);
    # (2) at runtime, redistribute the incoming tangent to whatever
    #     placement the expected spec demands.

    def __coerce_tangent_metadata__(self) -> "ShardTensor":
        """Trace-time hook: coerce this tensor so its metadata matches a tangent.

        Returns ``self`` if no Partial placement is present (no work needed).
        Otherwise redistributes Partial placements to Replicate, which is the
        layout the autograd engine produces for tangents.
        """
        if not any(isinstance(p, Partial) for p in self.placements):
            return self
        new_placements = [
            Replicate() if isinstance(p, Partial) else p for p in self.placements
        ]
        return self.redistribute(
            device_mesh=self.device_mesh, placements=new_placements
        )

    def __coerce_same_metadata_as_tangent__(
        self,
        flatten_spec: tuple,
        expected_type: type | None = None,
    ) -> "ShardTensor | None":
        """Runtime hook: redistribute ``self`` to match the recorded tangent's
        placements and ``_sharding_shapes`` (preserves uneven layouts).

        VENDOR-EDIT: the upstream version returned ``None`` whenever
        ``expected_type`` was set (DTensor convention: refuse cross-type). Our
        MLIP ``ShardTensor`` subclass produces forward OUTPUTS of the subclass
        type, but many backward TANGENTS are constructed as this BASE type
        (e.g. ``_ToTorchTensor.backward``). AOT records
        ``expected_type=<subclass>`` from the output, then asks the base-typed
        tangent to coerce UP to it — refusing made *every* op's backward raise
        "guessed its metadata incorrectly". Instead: (1) accept the subclass's
        NESTED ``(base_ctx, mlip_ctx)`` flatten context as well as the base's
        flat ``(spec, requires_grad)``; (2) treat ``{}`` and ``None``
        ``_sharding_shapes`` as equal; (3) reconcile placements / sharding as
        before; (4) when a differing ``expected_type`` is requested, rebuild via
        THAT type's own ``__tensor_unflatten__`` (which reclasses + reattaches
        its metadata) rather than returning ``None``.
        """
        # Accept the subclass's nested ``(base_ctx, mlip_ctx)`` or the base's
        # flat ``(spec, requires_grad)`` context.
        base_ctx = flatten_spec[0] if isinstance(flatten_spec[0], tuple) else flatten_spec
        (spec, _requires_grad) = base_ctx

        def _norm_ss(ss: object) -> object:
            # ``{}`` (op-result re-wrap) and ``None`` (fresh spec) both mean
            # "no explicit per-rank sharding shapes" — don't redistribute over it.
            return ss or None

        if (
            self._spec.placements == spec.placements
            and _norm_ss(self._spec._sharding_shapes) == _norm_ss(spec._sharding_shapes)
        ):
            coerced: "ShardTensor" = self
        else:
            # Bypass ``self.redistribute()`` so we can thread the recorded
            # per-tensor-dim shard sizes through to the local redistribute (the
            # public API drops them).
            target_spec = ShardTensorSpec(
                mesh=self.device_mesh,
                placements=spec.placements,
                tensor_meta=self._spec.tensor_meta,
                _sharding_shapes=spec._sharding_shapes,
            )

            target_sharding_shapes_by_tensor_dim: dict[int, list[int]] = {}
            if spec._sharding_shapes is not None:
                for mesh_dim, placement in enumerate(spec.placements):
                    if isinstance(placement, Shard) and mesh_dim in spec._sharding_shapes:
                        shard_shapes = spec._sharding_shapes[mesh_dim]
                        target_sharding_shapes_by_tensor_dim[placement.dim] = [
                            s[placement.dim] for s in shard_shapes
                        ]

            new_local = redistribute_local_shard_tensor(
                self._local_tensor,
                self._spec,
                target_spec,
                async_op=False,
                target_sharding_shapes=target_sharding_shapes_by_tensor_dim,
            )
            target_spec._local_shape = new_local.shape
            coerced = ShardTensor(
                new_local.contiguous(),
                target_spec,
                requires_grad=self.requires_grad,
            )

        # Cross-type: rebuild as the expected subclass via ITS own unflatten
        # (handles ``__class__`` reclass + reattaching subclass metadata from the
        # nested ``mlip_ctx``). Same-type tangents return ``coerced`` unchanged.
        if expected_type is not None and expected_type is not type(coerced):
            return expected_type.__tensor_unflatten__(
                {"_local_tensor": coerced._local_tensor},
                flatten_spec,
                tuple(coerced.shape),
                coerced.stride(),
            )
        return coerced

    # -- Autograd property overrides -------------------------------------------
    # The C-level requires_grad is authoritative for autograd engine
    # decisions; we read it first and fall back to _local_tensor for the
    # case where _make_wrapper_subclass didn't propagate it correctly.
    # For grad, the autograd engine accumulates at the C level, so we
    # check there first then fall back to _local_tensor.grad.

    @property  # type: ignore[override]
    def requires_grad(self) -> bool:  # type: ignore[override]
        """Whether this tensor requires gradient computation.

        Returns ``True`` if either the wrapper tensor or the underlying local
        tensor has ``requires_grad`` set.
        """
        with torch._C.DisableTorchFunctionSubclass():
            if torch.Tensor.requires_grad.__get__(self):
                return True
        return self._local_tensor.requires_grad

    @requires_grad.setter
    def requires_grad(self, value: bool) -> None:
        """Set ``requires_grad`` on both the wrapper and the local tensor.

        Only mutates the flag on LEAF tensors. A non-leaf's ``requires_grad`` is
        implied by its ``grad_fn`` and setting it raises in C++ ("you can only
        change requires_grad flags of leaf variables"). This is required for
        ``torch.compile`` on the distributed path: dynamo's FakeTensor
        construction replicates the requires_grad state of a non-leaf wrapper
        (positions carries a grad_fn from the autograd-preserving halo wrap) by
        calling this setter — unconditionally setting the flag broke compile.
        For non-leaves the flag is already correct, so skipping is a no-op that
        matches eager behavior.
        """
        with torch._C.DisableTorchFunctionSubclass():
            if torch.Tensor.is_leaf.__get__(self):
                torch.Tensor.requires_grad.__set__(self, value)
        if self._local_tensor.is_leaf:
            self._local_tensor.requires_grad = value

    def requires_grad_(self, requires_grad: bool = True) -> "ShardTensor":
        """Set ``requires_grad`` in-place on both the wrapper and local tensor.

        Parameters
        ----------
        requires_grad : bool, optional
            Whether to enable gradient tracking. Default is ``True``.

        Returns
        -------
        ShardTensor
            ``self``, for method chaining.
        """
        with torch._C.DisableTorchFunctionSubclass():
            if torch.Tensor.is_leaf.__get__(self):
                torch.Tensor.requires_grad.__set__(self, requires_grad)
        if self._local_tensor.is_leaf:
            self._local_tensor.requires_grad_(requires_grad)
        return self

    @property  # type: ignore[override]
    def is_leaf(self) -> bool:  # type: ignore[override]
        """Whether this tensor is a leaf in the autograd graph."""
        with torch._C.DisableTorchFunctionSubclass():
            return torch.Tensor.is_leaf.__get__(self)

    @property  # type: ignore[override]
    def grad_fn(self):  # type: ignore[override]
        """Return the stored grad_fn without re-entering ``__torch_function__``.

        Without this override, ``.grad_fn`` (a C-level getset_descriptor on
        ``torch.Tensor``) re-enters ``ShardTensor.__torch_function__``
        whenever someone reads it, falls back via
        :func:`_torch_function_fallback_via_dtensor`, and the fallback
        constructs a *new* temporary DTensor via
        ``_ShardTensorToDTensor.apply(self)`` -- whose ``.grad_fn`` (a
        ``_ShardTensorToDTensorBackward`` ``BackwardCFunction`` instance)
        is what the caller actually receives. On newer PyTorch that
        node's ``.next_functions`` accessor raises a "legacy access
        pattern" error, which is exactly what makes
        ``AOTAutograd.setup_stacktrace_preservation_hooks`` (and our
        own diagnostic ``dump_grad_fn_chain``) fail when they try to
        walk the autograd graph of a ShardTensor output.

        Mirrors the same shielding pattern already used by ``.is_leaf``
        and ``.grad``.
        """
        with torch._C.DisableTorchFunctionSubclass():
            return torch.Tensor.grad_fn.__get__(self)

    @property  # type: ignore[override]
    def grad(self) -> "ShardTensor | None":  # type: ignore[override]
        """Return the accumulated gradient, wrapped as a :class:`ShardTensor`.

        If no gradient has been accumulated yet, returns ``None``.
        """
        with torch._C.DisableTorchFunctionSubclass():
            c_grad = torch.Tensor.grad.__get__(self)
        if c_grad is not None:
            if isinstance(c_grad, ShardTensor):
                return c_grad
            return ShardTensor(local_tensor=c_grad._local_tensor
                if isinstance(c_grad, DTensor)
                else c_grad,
                spec=self._spec,
                requires_grad=False,
            )
        local_grad = self._local_tensor.grad
        if local_grad is None:
            return None
        return ShardTensor(local_tensor=local_grad,
            spec=self._spec,
            requires_grad=False,
        )

    @grad.setter
    def grad(self, value: "ShardTensor | torch.Tensor | None") -> None:
        """Set or clear the gradient on both the wrapper and local tensor."""
        if value is None:
            with torch._C.DisableTorchFunctionSubclass():
                torch.Tensor.grad.__set__(self, None)
            self._local_tensor.grad = None
        elif isinstance(value, ShardTensor):
            with torch._C.DisableTorchFunctionSubclass():
                torch.Tensor.grad.__set__(self, value)
            self._local_tensor.grad = value._local_tensor
        else:
            with torch._C.DisableTorchFunctionSubclass():
                torch.Tensor.grad.__set__(self, value)
            self._local_tensor.grad = value

    @classmethod
    def from_dtensor(cls, dtensor: DTensor) -> "ShardTensor":
        r"""Convert a DTensor to a ShardTensor.

        Differentiable when *dtensor* is non-leaf (has a ``grad_fn``).
        Spec is inferred from the DTensor (chunk-based, no communication).

        Parameters
        ----------
        dtensor : DTensor
            DTensor to convert.

        Returns
        -------
        ShardTensor
            Equivalent ShardTensor with the same local tensor and inferred spec.
        """
        if isinstance(dtensor, ShardTensor):
            return dtensor
        spec = _resolve_spec_for_dtensor(dtensor)
        if dtensor.grad_fn is not None:
            return _DTensorToShardTensor.apply(dtensor, spec)
        return _dtensor_to_shard_tensor(dtensor, spec)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if _conversion_active():
            # When converting shard tensor to dtensor, or dtensor to shard tensor,
            # we just run the function without ShardTensor dispatch.
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)
        if func in cls._function_registry and cls._enable_shard_patches:
            return cls._function_registry[func](func, types, args, kwargs)
        if str(func) in cls._named_function_registry and cls._enable_shard_patches:
            return cls._named_function_registry[str(func)](func, types, args, kwargs)
        res = _torch_function_fallback_via_dtensor(func, args, kwargs)
        return res

    @classmethod
    def __torch_dispatch__(
        cls,
        func: torch._ops.OpOverload,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> "ShardTensor" | Iterable["ShardTensor"] | object:
        # Use a handler, if we have one:
        handler = cls._dispatch_registry.get(func)
        if handler is None:
            handler = cls._dispatch_registry_by_name.get(str(func))
        if handler is not None:
            return handler(*args, **kwargs)
        # Otherwise, try the dtensor route:
        return _dispatch_fallback_via_dtensor(func, args, kwargs)

    @staticmethod
    def from_local(
        local_tensor: torch.Tensor,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        sharding_shapes: str | dict[int, list[tuple[int, ...]]] = "infer",
    ) -> "ShardTensor":
        r"""Generate a new ShardTensor from local torch tensors.

        Uses device mesh and placements to infer global tensor properties.
        No restriction is made on forcing tensors to have equal shapes locally.
        Instead, the requirement is that tensor shapes could be concatenated
        into a single tensor according to the placements.

        Parameters
        ----------
        local_tensor : torch.Tensor
            Local chunk of tensor. All participating tensors must be of the
            same rank and concatenatable across the mesh dimensions.
        device_mesh : Optional[DeviceMesh], optional
            Target device mesh. If not specified, will use the current mesh.
        placements : Optional[Sequence[Placement]], optional
            Target placements. Must have same number of elements as
            ``device_mesh.ndim``.
        sharding_shapes : Union[str, Dict[int, List[Tuple[int, ...]]]], default="infer"
            Controls how shard tensor spec is generated:

            - ``"chunk"``: Use ``torch.chunk`` shapes to infer shapes from
              global shape (no communication).
            - ``"infer"``: Use collective communication to infer shapes from
              mesh neighbors.
            - Manual dict mapping mesh dim to list of shard shapes: Use
              provided shapes. Must pass on each rank.

        Returns
        -------
        ShardTensor
            A new ShardTensor instance.
        """

        # This implementation follows the pytorch DTensor Implementation Closely.
        device_mesh = device_mesh or _mesh_resources.get_current_mesh()
        device_type = device_mesh.device_type

        # convert the local tensor to desired device base on device mesh's device_type
        if device_type != local_tensor.device.type and not local_tensor.is_meta:
            local_tensor = local_tensor.to(device_type)

        # set default placements to replicated if not specified
        if placements is None:
            placements = [Replicate() for _ in range(device_mesh.ndim)]
        else:
            placements = list(placements)
            for idx, placement in enumerate(placements):
                # normalize shard dim to be positive
                if placement.is_shard():
                    placement = cast(Shard, placement)
                    if placement.dim < 0:
                        placements[idx] = Shard(placement.dim + local_tensor.ndim)

        # `from_local` is differentiable, and the gradient of the dist tensor this function
        # created should flow back the gradients to the local_tensor, so we call an autograd
        # function to construct the dist tensor instead.
        return _FromTorchTensor.apply(  # pyre-ignore[16]: autograd func
            local_tensor,
            device_mesh,
            tuple(placements),
            sharding_shapes,
        )

    def offsets(self, mesh_dim: int | None = None) -> list[int] | int:
        r"""Get offsets of shards along a mesh dimension.

        Parameters
        ----------
        mesh_dim : Optional[int], optional
            Mesh dimension to get offsets for. If ``None``, returns all offsets.

        Returns
        -------
        Union[List[int], int]
            List of offsets for shards along all dimensions, or single offset
            if ``mesh_dim`` is specified.
        """
        return self._spec.offsets(mesh_dim)

    def redistribute(
        self,
        device_mesh: DeviceMesh | None = None,
        placements: Sequence[Placement] | None = None,
        *,
        async_op: bool = False,
    ) -> "ShardTensor":
        r"""Redistribute tensor across device mesh with new placement scheme.

        Like ``DTensor.redistribute`` but uses custom layer for shard
        redistribution that supports uneven sharding.

        Parameters
        ----------
        device_mesh : Optional[DeviceMesh], optional
            Target device mesh. Uses current mesh if ``None``.
        placements : Optional[Sequence[Placement]], optional
            Target placement scheme. Required.
        async_op : bool, default=False
            Whether to run asynchronously.

        Returns
        -------
        ShardTensor
            Redistributed ShardTensor with new placement scheme.

        Raises
        ------
        RuntimeError
            If placements is not specified or contains invalid placements
            (e.g., ``Partial`` placements or negative shard dimensions).
        """

        # if device_mesh is not specified, use the current device_mesh
        device_mesh = device_mesh or self.device_mesh
        # raise error if new placements not specified
        if placements is None:
            raise RuntimeError("placements is needed for redistribute!")

        placements = list(placements)
        for i, placement in enumerate(placements):
            if placement.is_partial():
                raise RuntimeError(
                    "Can not redistribute to Partial, redistributing to Partial is for internal use only!"
                )
            elif isinstance(placement, Shard) and placement.dim < 0:
                # normalize shard dim to be positive
                placements[i] = Shard(placement.dim + self.ndim)
        placements = tuple(placements)

        return ShardRedistribute.apply(self, device_mesh, placements, async_op)

    def to_local(
        self, *, grad_placements: Sequence[Placement] | None = None
    ) -> torch.Tensor:
        r"""Get local tensor from this ShardTensor.

        Parameters
        ----------
        grad_placements : Optional[Sequence[Placement]], optional
            Future layout of gradients. If provided, gradients will be
            constructed with this placement scheme during backward pass.

        Returns
        -------
        torch.Tensor
            Local tensor. Shape may vary between ranks for sharded tensors.
        """

        if not torch.is_grad_enabled():
            return self._local_tensor

        if grad_placements is not None:
            grad_placements = tuple(grad_placements)

        return _ToTorchTensor.apply(self, grad_placements)

    def full_tensor(
        self, *, grad_placements: Sequence[Placement] | None = None
    ) -> torch.Tensor:
        r"""Gather the full tensor from all ranks.

        Redistributes to ``Replicate`` placement on all mesh dimensions and
        returns the local tensor.

        Parameters
        ----------
        grad_placements : Optional[Sequence[Placement]], optional
            Future layout of gradients. If provided, gradients will be
            constructed with this placement scheme during backward pass.

        Returns
        -------
        torch.Tensor
            The full gathered tensor, identical on all ranks.
        """

        redist_res = self.redistribute(
            placements=[Replicate()] * self.device_mesh.ndim, async_op=False
        )
        if grad_placements is not None:
            grad_placements = tuple(grad_placements)
        return _ToTorchTensor.apply(redist_res, grad_placements)

    def backward(self, *args, **kwargs):
        r"""Perform backward pass for ShardTensor.

        Handles the redistribution of the tensor to resolve any partial
        placements before calling backward on the local tensor.

        Parameters
        ----------
        *args
            Positional arguments passed to ``torch.Tensor.backward``.
        **kwargs
            Keyword arguments passed to ``torch.Tensor.backward``.
        """

        # Before calling backward, we need to resolve any partial placements.
        new_placements = []
        needs_redistribute = False
        for placement in self._spec.placements:
            if placement.is_partial():
                new_placements.append(Replicate())
                needs_redistribute = True
            else:
                new_placements.append(placement)

        if needs_redistribute:
            self = self.redistribute(placements=new_placements)

        if self.grad_fn is not None:
            return torch.Tensor.backward(self, *args, **kwargs)

        return self.to_local().backward(*args, **kwargs)


### TODO
### Do we still need this?
### I think we do not - CJA


class FSDPOutputTensorAdapter(nn.Module):
    """Wrap a module and convert ShardTensor outputs to torch.Tensor."""

    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        out = self.module(*args, **kwargs)
        return out.to_local() if isinstance(out, ShardTensor) else out


def wrap_for_fsdp(module: nn.Module) -> nn.Module:
    """Return a module wrapper that exposes tensor outputs for FSDP hooks."""
    return FSDPOutputTensorAdapter(module)


def distribute_over_domain_for_fsdp(
    module: nn.Module,
    device_mesh: DeviceMesh,
    partition_fn: (Callable[[str, nn.Module, DeviceMesh], None] | None) = None,
) -> nn.Module:
    """Distribute a module over a domain mesh and adapt outputs for FSDP."""
    distributed_module = distribute_module(
        module,
        device_mesh=device_mesh,
        partition_fn=partition_fn,
    )
    return wrap_for_fsdp(distributed_module)


def scatter_tensor(
    tensor: torch.Tensor,
    global_src: int,
    mesh: DeviceMesh,
    placements: tuple[Placement, ...],
    global_shape: torch.Size | None = None,
    dtype: torch.dtype | None = None,
    requires_grad: bool = False,
) -> "ShardTensor":
    r"""Distribute a tensor from source rank across devices on the mesh.

    Takes a tensor that exists on a single source rank and distributes it
    across a device mesh according to the specified placement scheme. For
    multi-dimensional meshes, it performs a flattened scatter operation
    before constructing the sharded tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to distribute. Must exist on source rank; can be ``None``
        on other ranks.
    global_src : int
        Global rank ID of the source process.
    mesh : DeviceMesh
        Device mesh defining the process topology.
    placements : Tuple[Placement, ...]
        Tuple of placement specifications defining how to distribute the tensor.
    global_shape : Optional[torch.Size], optional
        Global shape of the tensor. If ``None``, will be broadcast from source.
    dtype : Optional[torch.dtype], optional
        Data type of the tensor. If ``None``, will be broadcast from source.
    requires_grad : bool, default=False
        Whether the resulting ShardTensor requires gradients.

    Returns
    -------
    ShardTensor
        The distributed tensor with specified placements.

    Raises
    ------
    ValueError
        If ``global_src`` is not an integer or not in the mesh.
    """
    dm = DistributedManager()

    if not isinstance(global_src, int):
        raise ValueError("Global source must be an integer rank")
    if global_src not in mesh.mesh:
        raise ValueError("Please specify a tensor source in this mesh")

    is_src = dm.rank == global_src

    # For multi-dimensional meshes, we use a flattened process group
    mesh_group = dm.get_mesh_group(mesh)

    # Broadcast tensor metadata from source
    if global_shape is None or dtype is None:
        if dm.rank == global_src:
            meta = [TensorMeta(tensor.shape, tensor.stride(), tensor.dtype)]
        else:
            meta = [None]

        dist.broadcast_object_list(meta, src=global_src, group=mesh_group)

        local_meta = meta[0]
    else:
        stride = _stride_from_contiguous_shape_C_style(global_shape)
        local_meta = TensorMeta(global_shape, stride, dtype)

    # This needs to be optimized, but I want to get the whole pipeline optimized first.
    # This only gets done when scatter_tensor is called and it should be relatively small
    # in full applications.

    # What isn't optimized?  Broadcasting the full tensor when placement is likely
    # Shard on at least one mesh dimension.  It would be more efficient to iteratively
    # scatter along Shard dimensions.  BUT, the focus is on performance of full applications
    # and this is a once-per-iteration cost.

    # Broadcast the tensor to all ranks.
    # scatter_tensor is an input-boundary utility; keep internal collectives/layout
    # transforms out of autograd and construct the requested leaf explicitly.
    if tensor is None and not is_src:
        # Tensor is allowed to be none if not on the root rank
        tensor = torch.empty(local_meta.shape, dtype=local_meta.dtype, device=dm.device)

    with torch.no_grad():
        dist.broadcast(tensor, src=global_src, group=mesh_group)

    # Create a fully-replicated spec:
    spec = ShardTensorSpec(
        mesh=mesh,
        placements=[Replicate() for _ in range(mesh.ndim)],
        tensor_meta=local_meta,
        _sharding_shapes={},
    )

    with torch.no_grad():
        # Build a replicated ShardTensor and redistribute to the requested
        # placements without recording autograd history.
        st = ShardTensor(local_tensor=tensor,
            spec=spec,
            requires_grad=False,
        )
        st = st.redistribute(mesh, placements, async_op=False)

    if requires_grad:
        # 1. Ensure the local data is a clean leaf
        local_leaf = st._local_tensor.detach().requires_grad_(True)

        # 2. Create the ShardTensor wrapper
        st = ShardTensor(local_tensor=local_leaf,
            spec=st._spec,
            requires_grad=True,
        )

        # 3. CRITICAL: Force the wrapper itself to be a leaf in the autograd graph
        st = st.detach().requires_grad_(True)

    return st
