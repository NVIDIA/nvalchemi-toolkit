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

"""Third-party helper trace.

When a wrapper composes a third-party model that has Python helpers
encoding single-process tensor-layout assumptions
(``aimnet.nbops.mol_sum`` reading ``mol_idx[-1] + 1`` for its output
size, ``calc_masks`` building its sentinel from
``numbers.shape[0] - 1``), distributed correctness depends on the
wrapper *replacing* those helpers via a
:class:`~nvalchemi.distributed._core.adapter.PythonAdapter` declared on the
spec's ``third_party_helpers``. When
it forgets to wrap one — or wraps it incorrectly — the model produces
wrong numbers silently. The validator's output diff catches the
wrongness, but offers no breadcrumb to *which* helper is the culprit.

This module solves the discovery half. Open a :func:`helper_trace`
context with a list of fully-qualified package paths
(``["aimnet.nbops"]``) and every module-level ``def`` in those
packages is monkey-patched with a logging proxy for the duration. Each
proxy records a :class:`HelperCall` carrying argument shapes / dtypes,
output shape / dtype / ``sum`` / ``max_abs``, the rank, and a
per-(module, function) call index — enough metadata for
:mod:`~nvalchemi.distributed._core.helper_diagnosis` to classify the
helper's pattern (per-system reduction, gather, scatter, mask, ...)
and check whether the per-rank outputs combine into the reference
output the way that pattern predicts. Originals are restored on
context exit.

Why a separate module from :mod:`dispatch_trace`
------------------------------------------------
``dispatch_trace`` records ``__torch_function__``-routed dispatch on
ShardTensor inputs — i.e. things the framework already intercepts.
``helper_trace`` records *opaque* third-party Python calls that the
framework does *not* route through dispatch. They're complementary:
together they account for every place a distributed mismatch could
slip in.

Recursion
---------
A re-entrancy guard prevents nested proxy firings. If
``aimnet.nbops.mol_sum`` internally calls ``aimnet.nbops.calc_masks``,
only ``mol_sum`` is recorded for that user-visible call site. (The
inner ``calc_masks`` would still be recorded *separately* when called
directly from the model.)

Import-time vs. call-time lookups
---------------------------------
This module patches ``setattr(module, attr, proxy)``. If a downstream
package binds the helper at import time
(``from aimnet.nbops import mol_sum``), the cached binding is *not*
intercepted. AIMNet2's source uses the call-time form
(``nbops.mol_sum(x, data)``), so the trace works for it; same
constraint as :class:`~nvalchemi.distributed._core.adapter.PythonAdapter`.
Document this for users bringing their own model packages.
"""

from __future__ import annotations

import contextlib
import importlib
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Sequence

import torch

logger = logging.getLogger(__name__)

__all__ = ["HelperCall", "helper_trace", "is_helper_tracing"]


# Module-level slots, mirroring dispatch_trace's design. Single-process
# per rank, single-threaded inside each — ContextVar would be needed
# only if we go thread-pool.
_HELPER_SINK: "list[HelperCall] | None" = None
_IN_PROXY: bool = False


@dataclass(slots=True)
class HelperCall:
    """One invocation of a watched helper function.

    Attributes
    ----------
    module
        Fully-qualified module path (``"aimnet.nbops"``).
    function
        Attribute name on the module (``"mol_sum"``).
    rank
        ``torch.distributed`` rank, or ``-1`` if no process group.
    call_index
        Monotonically increasing per ``(module, function)`` within the
        current trace context. Used by the diagnosis pass to align
        per-rank records with the corresponding reference call.
    input_summary
        Mapping ``arg_name -> {"shape", "dtype"}`` (or
        ``{"type": "..."}`` for non-tensor args). Positional args use
        keys ``arg0``, ``arg1``, .... Dict args (e.g. AIMNet's ``data``
        bag) recurse one level deep — keys whose values are tensors
        appear as ``argN.<key>``.
    output_summary
        ``{"shape", "dtype", "sum", "max_abs"}`` for tensor outputs.
        ``sum`` / ``max_abs`` are ``None`` for non-float outputs (bool
        masks, integer indices). Non-tensor outputs record
        ``{"type": "..."}`` only.
    """

    module: str
    function: str
    rank: int
    call_index: int
    input_summary: dict[str, Any]
    output_summary: dict[str, Any]


def is_helper_tracing() -> bool:
    """``True`` while a :func:`helper_trace` context is active."""
    return _HELPER_SINK is not None


def _summarize_tensor(t: torch.Tensor) -> dict[str, Any]:
    """Capture shape / dtype / sum / max_abs of a single tensor.

    ``sum`` and ``max_abs`` force a GPU sync, so callers that worry
    about overhead should use ``helper_trace``'s ``sample_every`` knob
    rather than calling this on every tensor.
    """
    summary: dict[str, Any] = {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype),
    }
    if t.is_floating_point() and t.numel() > 0:
        # Promote to fp64 for sum stability — avoids spurious overflow on
        # extreme magnitudes.
        x64 = t.detach().to(torch.float64)
        summary["sum"] = float(x64.sum().item())
        summary["max_abs"] = float(x64.abs().max().item())
    else:
        summary["sum"] = None
        summary["max_abs"] = None
    return summary


def _summarize_value(v: Any) -> dict[str, Any]:
    """Generic value summarizer for input args. Recurses one level into
    dicts (which is how aimnet's ``data`` bag is shaped) and lists/
    tuples (first 4 elements). Non-tensor leaves get ``{"type": ...}``.
    """
    if isinstance(v, torch.Tensor):
        return _summarize_tensor(v)
    if isinstance(v, dict):
        # AIMNet's ``data`` is a flat dict of tensors keyed by name.
        # Recurse one level so the diagnosis can see e.g.
        # ``data.mol_idx`` shape.
        out: dict[str, Any] = {"type": "dict"}
        for k, val in v.items():
            if isinstance(val, torch.Tensor):
                out[str(k)] = _summarize_tensor(val)
        return out
    if isinstance(v, (list, tuple)):
        out = {"type": type(v).__name__, "len": len(v)}
        for i, val in enumerate(v[:4]):
            if isinstance(val, torch.Tensor):
                out[f"[{i}]"] = _summarize_tensor(val)
        return out
    return {"type": type(v).__name__}


def _make_proxy(
    module_name: str,
    fn_name: str,
    original: Callable,
    call_counts: dict[tuple[str, str], int],
    sample_every: int,
) -> Callable:
    """Build the per-function proxy closure. Captured in a separate
    function (rather than inline lambda) so each proxy gets its own
    closure cell and a useful ``__qualname__`` for tracebacks."""

    def proxy(*args: Any, **kwargs: Any) -> Any:
        global _IN_PROXY
        # Pass-through if not tracing or already inside a proxy
        # (recursion-from-inside-watched-call).
        if _HELPER_SINK is None or _IN_PROXY:
            return original(*args, **kwargs)

        key = (module_name, fn_name)
        idx = call_counts.get(key, 0)
        call_counts[key] = idx + 1

        # Sample: record the first call always, then every Nth.
        record_this_call = (idx == 0) or (idx % sample_every == 0)

        _IN_PROXY = True
        try:
            if record_this_call:
                input_summary: dict[str, Any] = {}
                for i, a in enumerate(args):
                    input_summary[f"arg{i}"] = _summarize_value(a)
                for k, v in kwargs.items():
                    input_summary[f"kw.{k}"] = _summarize_value(v)
                result = original(*args, **kwargs)
                output_summary = _summarize_value(result)
                import torch.distributed as _td  # noqa: PLC0415

                rank = _td.get_rank() if _td.is_initialized() else -1
                _HELPER_SINK.append(
                    HelperCall(
                        module=module_name,
                        function=fn_name,
                        rank=rank,
                        call_index=idx,
                        input_summary=input_summary,
                        output_summary=output_summary,
                    )
                )
                return result
            return original(*args, **kwargs)
        finally:
            _IN_PROXY = False

    proxy.__name__ = f"helper_trace_proxy[{module_name}.{fn_name}]"
    proxy.__qualname__ = proxy.__name__
    proxy.__wrapped__ = original  # type: ignore[attr-defined]
    return proxy


def _is_local_function(obj: Any, module: Any) -> bool:
    """``True`` if ``obj`` is a function *defined in* ``module`` (not
    a re-export, not a class, not a builtin). Filters by
    ``__module__`` to avoid patching things like ``torch.tensor``
    that happen to be imported at the top of ``aimnet.nbops``."""
    if not inspect.isfunction(obj):
        return False
    return getattr(obj, "__module__", None) == getattr(module, "__name__", None)


@contextlib.contextmanager
def helper_trace(
    packages: Sequence[str],
    *,
    sample_every: int = 8,
) -> Iterator[list[HelperCall]]:
    """Open a helper-trace scope.

    For the duration of the ``with`` block, every module-level function
    defined in any module listed in ``packages`` is monkey-patched
    with a recording proxy. Yields a list to which proxies append
    :class:`HelperCall` records. Originals are restored on exit even
    when the wrapped block raises.

    Parameters
    ----------
    packages
        Fully-qualified module paths to watch
        (``["aimnet.nbops"]``). Modules that aren't importable are
        skipped silently with a debug log — wrapping into try/except
        ``ModuleNotFoundError`` keeps the validator usable when the
        watched package isn't installed (e.g. AIMNet2 not present
        but MACE is).
    sample_every
        Record the *first* call to each ``(module, function)`` always,
        then every Nth call after that. Default 8 keeps runtime
        overhead bounded for hot helpers (``mol_sum`` is called
        multiple times per layer × multiple layers per forward) while
        still capturing enough samples for the consistency check.
        Set to 1 for exhaustive recording (debug only).

    Yields
    ------
    records : list[HelperCall]
        Append-only list. Iterate or filter after the ``with`` block
        exits.

    Notes
    -----
    Single-process per rank, single-threaded inside each — uses
    module-level state, not ``ContextVar``. Don't nest.
    """
    global _HELPER_SINK
    records: list[HelperCall] = []
    prev_sink = _HELPER_SINK
    _HELPER_SINK = records

    # (module, attr) -> original-callable, plus a parallel dict of
    # the live module objects so we can restore via setattr.
    originals: dict[tuple[str, str], Callable] = {}
    live_modules: dict[str, Any] = {}
    call_counts: dict[tuple[str, str], int] = {}

    try:
        for pkg in packages:
            try:
                module = importlib.import_module(pkg)
            except ModuleNotFoundError:
                logger.debug("helper_trace: skipping unimportable package %s", pkg)
                continue
            live_modules[pkg] = module
            for attr_name in dir(module):
                if attr_name.startswith("_"):
                    continue
                obj = getattr(module, attr_name, None)
                if not _is_local_function(obj, module):
                    continue
                originals[(pkg, attr_name)] = obj
                setattr(
                    module,
                    attr_name,
                    _make_proxy(pkg, attr_name, obj, call_counts, sample_every),
                )

        yield records

    finally:
        # Restore everything, even if the body raised. Iterate over the
        # captured originals dict — guaranteed to match what we patched.
        for (pkg, attr_name), orig in originals.items():
            mod = live_modules.get(pkg)
            if mod is None:
                continue
            try:
                setattr(mod, attr_name, orig)
            except Exception as e:
                logger.warning(
                    "helper_trace: failed to restore %s.%s: %s",
                    pkg,
                    attr_name,
                    e,
                )
        _HELPER_SINK = prev_sink
