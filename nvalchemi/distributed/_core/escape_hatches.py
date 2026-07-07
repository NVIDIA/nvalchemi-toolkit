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
"""Escape hatch for opaque ops that :meth:`ShardTensor.__torch_function__`
can't reach directly.

Custom CUDA / Warp / Triton ops registered via
``@torch.library.custom_op`` bypass ``__torch_function__`` — the kernel
sees plain tensors and reads its own per-rank slice. :func:`wrap_custom_op`
installs a distribution-aware handler that materializes the kernel's inputs
(halo gather / sharded full-gather) and corrects its outputs.

Per-model wrappers register the handler in their ``distributed_setup()``
hook. Metadata flows through the wrapper via the ShardTensor instance
attributes of the args — no ambient context lookup.

Helpers that aren't opaque custom ops — e.g. ``@torch.jit.script`` or
plain-Python model internals that bake in single-process layout — are
handled at the chemistry layer via the :class:`JitAdapter` /
:class:`PythonAdapter` declared on a spec's ``third_party_helpers``.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Sequence

import torch

from nvalchemi.distributed._core.shard_tensor import ShardTensor, register_handler

logger = logging.getLogger(__name__)


def _handle_sharded(
    args: tuple,
    kwargs: dict,
    source: ShardTensor,
    op: Any,
    op_name: str,
    gather_inputs_full: Sequence[int],
    slice_outputs_owned: Sequence[int],
    _unwrap: Callable[[Any], Any],
    _wrap: Callable[..., Any],
    _call_op_on_device: Callable[[Any, dict], Any],
    is_tracing: Callable[[], bool],
    record_dispatch: Callable[..., None],
) -> Any:
    """Sharded-storage path of :func:`wrap_custom_op`.

    Full-gathers the indicated inputs from per-rank ``[n_owned + 1, *F]``
    to global ``[n_global + 1, *F]`` (via :func:`distributed_index_select`
    over ``[0, n_global)`` plus the caller's local padding row), invokes
    the kernel, then slices the indicated outputs back to
    ``[n_owned + 1, *F']`` for this rank. The per-rank block is
    contiguous in the gathered ordering because ranks' atoms are stored
    in rank-block order.
    """
    import torch.distributed as dist  # noqa: PLC0415

    from nvalchemi.distributed._core.gather_primitives import (  # noqa: PLC0415
        distributed_index_select,
        mesh_group,
    )

    gather_meta = source._gather_meta
    halo_cfg = source._config
    n_owned = gather_meta.n_owned
    n_global = gather_meta.n_global

    # Block offset for this rank's owned atoms in the global rank-sorted
    # ordering: count atoms owned by lower-numbered ranks. Cheap on the
    # replicated owner_rank tensor; no comm.
    group = mesh_group(halo_cfg.mesh) if halo_cfg is not None else None
    rank = dist.get_rank(group=group) if dist.is_initialized() else 0
    owner_rank = gather_meta.owner_rank
    block_offset = int((owner_rank < rank).sum().item())

    new_args = [_unwrap(a) for a in args]

    if gather_inputs_full:
        from nvalchemi.distributed._core.shard_tensor import (  # noqa: PLC0415
            _unwrap_grad_aware,
        )

        all_global_ids = torch.arange(n_global, dtype=torch.long, device=source.device)
        for idx in gather_inputs_full:
            # Grad-aware unwrap so gathered features stay connected to the
            # autograd graph: distributed_index_select's backward scatters each
            # rank's force-gradient back to the owning ranks. A plain _unwrap
            # would detach _local_tensor and sever conservative forces.
            raw = args[idx]
            t = (
                _unwrap_grad_aware(raw)
                if isinstance(raw, ShardTensor)
                else new_args[idx]
            )
            if not isinstance(t, torch.Tensor):
                continue
            # Convention: caller passes ``(n_owned + 1, *F)`` — last row
            # is the local padding atom. Owned rows are ``[:-1]``.
            owned = t[:n_owned].contiguous()
            pad_row = t[n_owned:].contiguous()  # (1, *F) — preserved verbatim.
            global_owned = distributed_index_select(
                owned, all_global_ids, gather_meta, halo_cfg
            )
            new_args[idx] = torch.cat([global_owned, pad_row], dim=0)

    plain_kw = {k: _unwrap(v) for k, v in kwargs.items()}
    result = _call_op_on_device(new_args, plain_kw)

    was_single = isinstance(result, torch.Tensor)
    result_list = [result] if was_single else list(result)

    if slice_outputs_owned:
        for idx in slice_outputs_owned:
            full = result_list[idx]
            if not isinstance(full, torch.Tensor):
                raise TypeError(
                    f"wrap_custom_op slice_output_owned at index {idx} "
                    f"is {type(full).__name__}, expected Tensor."
                )
            # Per-rank owned block + the trailing padding row (which the
            # kernel propagated through unchanged when ``B-1`` excluded
            # it from the launch). ``.contiguous()`` because downstream
            # cuBLAS calls (e.g. AIMNet2's ``einsum`` on the convolution
            # output) require contiguous strides on a fresh storage.
            owned_block = full[block_offset : block_offset + n_owned]
            pad_row = full[n_global:]  # one row at the end
            result_list[idx] = torch.cat([owned_block, pad_row], dim=0).contiguous()

    if is_tracing():
        branches = []
        if gather_inputs_full:
            branches.append(f"gather_inputs_full={tuple(gather_inputs_full)}")
        if slice_outputs_owned:
            branches.append(f"slice_outputs_owned={tuple(slice_outputs_owned)}")
        record_dispatch(
            "wrap_custom_op",
            op=op_name,
            branch="sharded:" + ("+".join(branches) if branches else "passthrough"),
            shapes={
                f"arg{i}": tuple(a.shape) if isinstance(a, torch.Tensor) else None
                for i, a in enumerate(new_args)
            },
            meta={"n_owned": n_owned, "n_global": n_global, "rank": rank},
        )

    wrapped = [_wrap(r, source) for r in result_list]
    return wrapped[0] if was_single else tuple(wrapped)


def wrap_custom_op(
    op: Any,
    *,
    gather_inputs: Sequence[int] = (),
    scatter_outputs: Sequence[int] = (),
    owned_slice_inputs: Sequence[int] = (),
    all_reduce_outputs: Sequence[int] = (),
    gather_inputs_full: Sequence[int] = (),
    slice_outputs_owned: Sequence[int] = (),
) -> None:
    """Install a distribution-aware handler on a ``@torch.library.custom_op``.

    Handles two storage classes — halo and sharded — selected at call
    time by the metadata on the source ShardTensor argument.

    Halo storage (source has ``_meta``):

    1. ``gather_inputs``: halo-materialize each owned-shape arg into a
       padded tensor via :func:`halo_forward_exchange` before the call.
    2. ``owned_slice_inputs``: slice each padded-shape arg to its
       ``[:n_owned]`` prefix (inverse of ``gather_inputs``) so the
       kernel iterates over each atom once globally; the cross-rank sum
       happens on the outputs.
    3. Call the underlying ``op`` on the materialized inputs.
    4. ``scatter_outputs``: halo-correct each output
       (``halo_reverse_exchange`` + ``halo_forward_exchange``) so halo
       rows hold owner values.
    5. ``all_reduce_outputs``: all-reduce each per-rank partial across
       the domain mesh into a globally-summed tensor replicated on every
       rank. Backward is symmetric.

    Sharded storage (source has ``_gather_meta``):

    1. ``gather_inputs_full``: full-gather each arg from per-rank
       ``[n_owned + 1, *F]`` to the global ``[n_global + 1, *F]`` view.
       The trailing row is the caller's local padding atom, preserved so
       kernels that treat the last row as padding still see one.
    2. Call the kernel with the global-view inputs.
    3. ``slice_outputs_owned``: slice each ``[n_global + 1, *F']`` output
       back to ``[n_owned + 1, *F']`` so downstream per-rank model code
       keeps its layout.

    When no ShardTensor argument carries metadata, the wrapper is a
    transparent pass-through (subclass args are unwrapped so the kernel
    never sees subclass tensors).

    Parameters
    ----------
    op
        The op to wrap. Typically an ``OpOverload`` from
        ``torch.ops.<ns>.<name>.default``, or a Python function.
    gather_inputs
        Halo storage. Positional indices of owned-shape args to
        halo-materialize before the kernel. Usually empty — the standard
        flow passes padded inputs to the model directly.
    scatter_outputs
        Halo storage. Positional indices of return values whose halo
        rows need correction after the kernel. Use ``[0]`` for a single
        tensor return; list each position for a tuple return.
    owned_slice_inputs
        Halo storage. Positional indices of padded-shape args to slice
        to the owned prefix before the kernel, so each atom's
        contribution is accumulated exactly once across ranks.
    all_reduce_outputs
        Halo storage. Positional indices of return values that are
        per-rank partials needing a global SUM all-reduce across the
        domain mesh.
    gather_inputs_full
        Sharded storage. Positional indices of ``[n_owned + 1, *F]``
        ShardTensor args whose contents must be made globally available
        before the kernel. Without this the kernel reads only its own
        rank's rows and a global-index neighbor matrix reads
        out-of-bounds rows.
    slice_outputs_owned
        Sharded storage. Positional indices of ``[n_global + 1, *F']``
        return values to slice back to ``[n_owned + 1, *F']`` for this
        rank.

    Returns
    -------
    None
        Registers the handler as a side effect.

    Notes
    -----
    Register once per op (typically in a model wrapper's
    ``distributed_setup``). After registration, every call to the op
    goes through the wrapper. Un-register via
    :func:`nvalchemi.distributed._core.shard_tensor.clear_handlers`.

    If the op never receives a ``ShardTensor`` argument, the wrapper
    never fires — ``__torch_function__`` only dispatches on tensor
    subclass types.
    """
    from nvalchemi.distributed._core.particle_halo import (
        halo_forward_exchange,
        halo_reverse_exchange,
        halo_scatter_correct_compiled,
    )
    from nvalchemi.distributed._core.shard_tensor import (
        _find_source,
        _propagate_attrs,
        _unwrap_grad_aware,
    )

    def _unwrap(t: Any) -> Any:
        """Strip the ShardTensor subclass from *t*.

        Descends into lists, tuples, and dicts because custom ops often
        take ``List[Tensor]`` as a single positional arg (e.g. cueq's
        ``uniform_1d(name, ..., tensors)`` where ``tensors`` is a list
        of ShardTensors). Leaving those subclasses in the list triggers
        ``__torch_function__`` when the op re-dispatches internally →
        the wrapper re-enters itself forever.

        ShardTensor is a wrapper-subclass over ``_local_tensor``;
        ``as_subclass(torch.Tensor)`` would produce a tensor with no
        real storage, so we return the underlying ``_local_tensor``
        directly.
        """
        if isinstance(t, ShardTensor):
            return t._local_tensor
        if isinstance(t, list):
            return [_unwrap(x) for x in t]
        if isinstance(t, tuple):
            return tuple(_unwrap(x) for x in t)
        if isinstance(t, dict):
            return {k: _unwrap(v) for k, v in t.items()}
        return t

    def _wrap(t: Any, source: Any = None) -> Any:
        """Re-promote plain Tensors to ShardTensor, recursively.

        Mirrors :func:`_unwrap`. When *source* is provided, propagates
        ``_meta`` / ``_config`` / ``_n_systems`` onto each wrapped
        tensor so downstream ops continue to see halo metadata.
        Construction routes through :meth:`ShardTensor.wrap` (the
        wrapper-subclass ``__new__`` pattern).
        """
        if isinstance(t, torch.Tensor) and not isinstance(t, ShardTensor):
            mesh = (
                source._spec.mesh
                if source is not None and getattr(source, "_spec", None) is not None
                else None
            )
            w = ShardTensor.wrap(t, mesh=mesh)
            if source is not None:
                _propagate_attrs(w, source)
            return w
        if isinstance(t, list):
            return [_wrap(x, source) for x in t]
        if isinstance(t, tuple):
            return tuple(_wrap(x, source) for x in t)
        if isinstance(t, dict):
            return {k: _wrap(v, source) for k, v in t.items()}
        return t

    def _unwrap_ga(t: Any) -> Any:
        """Grad-aware :func:`_unwrap`: identical container descent, but bridges
        each grad-requiring ShardTensor leaf through
        :func:`_unwrap_grad_aware` (an autograd.Function) instead of returning
        the autograd-detached ``_local_tensor``.

        The halo branch computes on plain locals and re-wraps the result; a
        plain ``_unwrap`` here severs the graph from the kernel's differentiable
        inputs back to ``positions`` — the forward (energy) is unaffected, but
        ``torch.autograd.grad(E, positions)`` returns zero (conservative forces
        vanish). cueq passes ``List[Tensor]`` as a single positional arg, so the
        bridge must descend into containers — which bare ``_unwrap_grad_aware``
        does not. Falls back to plain unwrap under no-grad (inference)."""
        if isinstance(t, ShardTensor):
            return _unwrap_grad_aware(t)
        if isinstance(t, list):
            return [_unwrap_ga(x) for x in t]
        if isinstance(t, tuple):
            return tuple(_unwrap_ga(x) for x in t)
        if isinstance(t, dict):
            return {k: _unwrap_ga(v) for k, v in t.items()}
        return t

    def _find_cuda_device(x: Any) -> "torch.device | None":
        """First CUDA device found in a nested structure. Used to pin
        the current CUDA device to the input tensor's device before
        invoking an opaque custom op.

        Defensive: most custom-op bindings resolve the launch stream
        via ``torch.cuda.current_stream()`` inside their C++ path, and
        rely on the caller having ``cudaSetDevice``'d to the tensor's
        device beforehand. Wrapping the op call in
        ``torch.cuda.device(dev)`` guarantees that invariant even when
        the surrounding Python code didn't explicitly call
        ``set_device`` (e.g. a dispatcher handler triggered deep inside
        a model forward).
        """
        if isinstance(x, torch.Tensor):
            return x.device if x.is_cuda else None
        if isinstance(x, (list, tuple)):
            for y in x:
                d = _find_cuda_device(y)
                if d is not None:
                    return d
        if isinstance(x, dict):
            for y in x.values():
                d = _find_cuda_device(y)
                if d is not None:
                    return d
        return None

    def _call_op_on_device(args_seq: Any, kw: dict) -> Any:
        """Invoke *op* under the correct CUDA device context. Pure
        pass-through for CPU ops (no cuda tensor in args)."""
        dev = _find_cuda_device(args_seq)
        if dev is None:
            return op(*args_seq, **kw)
        with torch.cuda.device(dev):
            return op(*args_seq, **kw)

    def _handler(*args: Any, **kwargs: Any) -> Any:
        from nvalchemi.distributed._core.dispatch_trace import (  # noqa: PLC0415
            is_tracing,
            record_dispatch,
        )

        op_name = str(op)

        # Find the ShardTensor carrying distribution metadata. Either
        # ``_meta`` (halo storage) or ``_gather_meta`` (sharded storage)
        # qualifies as "active"; without one the op runs as a passthrough.
        # Search both args and kwargs: kernels invoked entirely by keyword
        # would otherwise miss the source and silently sever the distribution
        # chain (no downstream per-system reduce / halo correction).
        source = _find_source(args)
        if source is None:
            source = _find_source(tuple(kwargs.values()))
        active = source is not None and (
            source._meta is not None or source._gather_meta is not None
        )
        if not active:
            # No metadata — pass-through. Unwrap subclass args so the kernel
            # sees plain tensors (nested subclasses in List[Tensor] would
            # re-fire the dispatcher → infinite recursion), then re-wrap
            # outputs so subclass identity propagates across the opaque kernel
            # boundary and downstream halo/per-system handlers still dispatch.
            plain_args = tuple(_unwrap(a) for a in args)
            plain_kw = {k: _unwrap(v) for k, v in kwargs.items()}
            result = _call_op_on_device(plain_args, plain_kw)
            if is_tracing():
                record_dispatch(
                    "wrap_custom_op",
                    op=op_name,
                    branch="passthrough_no_meta",
                    shapes={
                        f"arg{i}": tuple(a.shape)
                        if isinstance(a, torch.Tensor)
                        else None
                        for i, a in enumerate(args)
                    },
                )
            if source is None:
                # No ShardTensor was even in the inputs — the op was
                # called with plain tensors; return the plain result.
                return result
            return _wrap(result, source)

        # Sharded-storage branch. Source has ``_gather_meta`` (and
        # ``_meta`` is None — a single ShardTensor only ever carries
        # one of the two).  Full-gather inputs to global, call the
        # kernel, slice outputs back to per-rank.
        if source._meta is None and source._gather_meta is not None:
            return _handle_sharded(
                args,
                kwargs,
                source,
                op,
                op_name,
                gather_inputs_full,
                slice_outputs_owned,
                _unwrap,
                _wrap,
                _call_op_on_device,
                is_tracing,
                record_dispatch,
            )

        meta = source._meta
        config = source._config

        # 1. Halo-materialize gather_inputs. Grad-aware unwrap so the kernel's
        # differentiable inputs stay connected to the wrapper autograd graph —
        # a plain ``_unwrap`` detaches ``_local_tensor`` and severs the path
        # back to ``positions`` (correct energy, but zero conservative forces).
        # Mirrors the sharded branch (``_handle_sharded`` uses
        # ``_unwrap_grad_aware`` on its gather inputs).
        new_args = [_unwrap_ga(a) for a in args]
        for idx in gather_inputs:
            owned = new_args[idx]
            new_args[idx] = halo_forward_exchange(owned, meta, config)

        # 1b. Owned-slice inputs. Inverse of gather_inputs — a padded
        # ShardTensor carries owned + halo rows; slice to the owned
        # prefix so the kernel iterates over each global atom exactly
        # once. Plain (non-ShardTensor) inputs pass through unchanged.
        if owned_slice_inputs:
            n_owned = meta.n_owned
            for idx in owned_slice_inputs:
                t = new_args[idx]
                if isinstance(t, torch.Tensor) and t.shape[0] > n_owned:
                    new_args[idx] = t[:n_owned].contiguous()

        # 2. Call the kernel.
        plain_kw = {k: _unwrap(v) for k, v in kwargs.items()}
        result = _call_op_on_device(new_args, plain_kw)

        # 3. Halo-correct scatter_outputs + all-reduce partial outputs.
        if not scatter_outputs and not all_reduce_outputs:
            return _wrap(result, source)

        was_single = isinstance(result, torch.Tensor)
        result_list = [result] if was_single else list(result)

        for idx in scatter_outputs:
            padded = result_list[idx]
            if not isinstance(padded, torch.Tensor):
                raise TypeError(
                    f"wrap_custom_op scatter_output at index {idx} is "
                    f"{type(padded).__name__}, expected Tensor."
                )
            from nvalchemi.distributed._core.shard_tensor import (  # noqa: PLC0415
                _under_compile_trace,
            )

            if _under_compile_trace((padded,)):
                # halo_forward(halo_reverse(.)) via the dispatcher-visible,
                # fake-mode-opaque custom op so the cueq fused-kernel output
                # is halo-corrected under torch.compile.
                result_list[idx] = halo_scatter_correct_compiled(padded, meta, config)
            else:
                owned = halo_reverse_exchange(padded, meta, config)
                result_list[idx] = halo_forward_exchange(owned, meta, config)

        # 3b. All-reduce partial outputs across the domain mesh. The
        # autograd-aware primitive handles both forward and backward
        # all_reduce, so gradients flowing back through the staged op
        # propagate correctly to owned_slice_inputs.
        if all_reduce_outputs:
            from nvalchemi.distributed._core.gather_primitives import (
                distributed_all_reduce,
            )

            for idx in all_reduce_outputs:
                partial = result_list[idx]
                if not isinstance(partial, torch.Tensor):
                    raise TypeError(
                        f"wrap_custom_op all_reduce_output at index {idx} is "
                        f"{type(partial).__name__}, expected Tensor."
                    )
                result_list[idx] = distributed_all_reduce(partial, config)

        if is_tracing():
            branches = []
            if gather_inputs:
                branches.append(f"gather_inputs={tuple(gather_inputs)}")
            if owned_slice_inputs:
                branches.append(f"owned_slice_inputs={tuple(owned_slice_inputs)}")
            if scatter_outputs:
                branches.append(f"scatter_outputs={tuple(scatter_outputs)}")
            if all_reduce_outputs:
                branches.append(f"all_reduce_outputs={tuple(all_reduce_outputs)}")
            record_dispatch(
                "wrap_custom_op",
                op=op_name,
                branch="+".join(branches) if branches else "subclass_only",
                shapes={
                    f"arg{i}": tuple(a.shape) if isinstance(a, torch.Tensor) else None
                    for i, a in enumerate(new_args)
                },
                meta={"n_owned": meta.n_owned, "n_padded": meta.n_padded},
            )

        # Carry halo metadata onto the wrapped outputs so downstream ops
        # keep dispatching correctly.
        wrapped_outputs = [_wrap(r, source) for r in result_list]
        return wrapped_outputs[0] if was_single else tuple(wrapped_outputs)

    register_handler(op, handler=_handler, name=f"wrap_custom_op[{op}]")

    # When an OpOverload (``torch.ops.ns.name.default``) is passed, also
    # register on the OpOverloadPacket (``torch.ops.ns.name``) because
    # ``torch.ops.ns.name(args)`` — the common call form — dispatches
    # through the packet, not the overload. ``is`` comparison on either
    # works; we need both bindings.
    packet = getattr(op, "_overloadpacket", None)
    if packet is not None and packet is not op:
        register_handler(packet, handler=_handler, name=f"wrap_custom_op[{packet}]")


__all__ = ["wrap_custom_op"]
