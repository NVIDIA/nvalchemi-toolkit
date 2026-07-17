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

"""Module-level layer-by-layer divergence diagnostic.

When ``trace_and_validate`` rejects a spec on per-output diff alone,
the helper-trace classifier can flag a few suspect ``aimnet.nbops``
calls but rarely points at the actual root cause when the bug is two
or three layers upstream. This module installs forward hooks on every
named sub-module of ``wrapper.model`` to record per-module owned-row
output sums on both the single-process reference and each spawned
rank, then in the launcher compares ref vs ``sum(ranks)`` per module
and surfaces the first divergent module name.

Per-rank tensors carry one trailing rank-local padding row by
convention (e.g. AIMNet2's nb_mode=1 layout); single-process carries
exactly one. To compare apples-to-apples we drop the trailing row
before summing on both sides — the partition arithmetic
``sum(rank) == ref`` then holds when the spec is correct.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "LayerRecord",
    "attach_layer_hooks",
    "diff_layer_records",
]


LayerRecord = tuple[str, tuple[int, ...], float, float]
"""(qualified_module_name, output_shape, owned_sum, owned_max_abs)."""


def _walk_modules(
    obj: Any,
    prefix: str = "",
    seen: set[int] | None = None,
    depth: int = 0,
    max_depth: int = 6,
) -> Any:
    """Yield ``(qualified_name, module)`` for every ``nn.Module``
    reachable from ``obj`` via attribute traversal — including modules
    held by plain Python objects that aren't themselves nn.Modules.

    Wrappers like UMA's ``UMAWrapper`` keep the actual model under
    ``self.predict_unit.model.module`` where ``predict_unit`` is a
    plain object. ``Module.named_modules()`` only recurses through
    *registered* submodules, missing this layer entirely. This helper
    walks all attributes (``__dict__``, plus ``_modules`` for proper
    submodules) bounded by ``max_depth`` to keep nested torch internals
    from blowing up.
    """
    if seen is None:
        seen = set()
    if id(obj) in seen or depth > max_depth:
        return
    seen.add(id(obj))

    if isinstance(obj, torch.nn.Module):
        if prefix:
            yield prefix, obj
        # ``named_modules`` recurses through ``_modules`` registry only.
        for name, sub in obj.named_modules():
            if name == "":
                continue
            full = f"{prefix}.{name}" if prefix else name
            if id(sub) not in seen:
                seen.add(id(sub))
                yield full, sub

    # Walk plain-Python attributes that may hold off-registry modules
    # (e.g. UMA's ``predict_unit``, which isn't itself an nn.Module but
    # holds the actual model under its torchtnt-style ``_modules`` dict).
    # Skip dunders, callables, and primitive values.
    if hasattr(obj, "__dict__"):
        for attr, val in obj.__dict__.items():
            if attr.startswith("__"):
                continue
            if callable(val) and not isinstance(val, torch.nn.Module):
                continue
            if isinstance(val, (str, int, float, bool, bytes, type(None))):
                continue
            child_prefix = f"{prefix}.{attr}" if prefix else attr
            yield from _walk_modules(val, child_prefix, seen, depth + 1, max_depth)

    # Handle container types that may hold modules: dict values, list /
    # tuple elements. Dict-of-modules is the torchtnt pattern UMA's
    # ``MLIPPredictUnit._modules`` uses; list-of-modules covers
    # ``nn.ModuleList``-equivalent plain Python lists.
    if isinstance(obj, dict):
        for k, v in obj.items():
            child_prefix = f"{prefix}[{k!r}]" if prefix else f"[{k!r}]"
            yield from _walk_modules(v, child_prefix, seen, depth + 1, max_depth)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            child_prefix = f"{prefix}[{i}]" if prefix else f"[{i}]"
            yield from _walk_modules(v, child_prefix, seen, depth + 1, max_depth)


def attach_layer_hooks(model: torch.nn.Module, records: list[LayerRecord]) -> list[Any]:
    """Attach forward hooks to every sub-module reachable from ``model``,
    recording per-call ``(name, shape, owned_sum, owned_max_abs)``.

    Records the OWNED-rows sum (``output[:-1]`` when ``shape[0] > 1``)
    so partition arithmetic compares cleanly against ref. Tuple/list
    outputs and dict outputs are skipped — the diagnostic only needs
    Tensor returns to localize the first divergent submodule.

    Walks beyond ``Module.named_modules()`` via :func:`_walk_modules` so
    wrappers that hold their actual model under a plain-Python attribute
    (UMA's ``predict_unit.model.module`` is the canonical case) still
    get every layer hooked.

    Important — hooks must NOT sync the GPU. Big models (UMA: 218
    submodules × many MP iterations) sync the device per hook call,
    turning a 1-second forward into a 5-minute timeout. Hooks instead
    accumulate tiny on-device tensors into ``records``; the caller
    materializes via :func:`finalize_layer_records` at the end of the
    forward, paying a single bulk sync.

    Returns the list of hook handles; the caller is responsible for
    ``handle.remove()`` after the forward pass.
    """
    handles: list[Any] = []

    def _record(name: str, out: Any) -> None:
        if not isinstance(out, torch.Tensor):
            return
        try:
            if out.ndim >= 1 and out.shape[0] > 1:
                owned = out[:-1]
            else:
                owned = out
            # Store on-device 0-d tensors; no .item(), no host copy, no
            # dtype upcast. ``finalize_layer_records`` later converts to
            # floats with one bulk sync.
            o = owned.detach()
            records.append(
                (
                    name,
                    tuple(out.shape),
                    o.sum(),
                    o.abs().max(),
                )
            )
        except Exception:  # noqa: S110, BLE001
            # Hook failure is purely informational — never let it mask
            # the actual model error or kill the validator run.
            pass

    def _make_hook(name: str):
        def _hook(_mod: Any, _inp: Any, out: Any) -> None:
            _record(name, out)

        return _hook

    for name, mod in _walk_modules(model):
        if name == "":
            continue
        try:
            handles.append(mod.register_forward_hook(_make_hook(name)))
        except (RuntimeError, AttributeError):
            # TorchScript submodules (``RecursiveScriptModule``, e.g. MACE's
            # scripted blocks) reject ``register_forward_hook`` — and are
            # opaque to Python-level hooks anyway. Skip them so the layer
            # diagnostic degrades gracefully instead of aborting the whole
            # validation run.
            continue
    return handles


def finalize_layer_records(records: list[Any]) -> list[LayerRecord]:
    """Materialize the on-device sums recorded by :func:`attach_layer_hooks`
    into Python floats. Single bulk sync replaces N per-hook syncs —
    critical for big models like UMA where the hooks fire 200+ times.

    Mutates and returns ``records`` in place; safe to call multiple
    times (idempotent on already-materialized records).
    """
    out: list[LayerRecord] = []
    for rec in records:
        if len(rec) != 4:
            out.append(rec)  # already flat or malformed — pass through
            continue
        name, shape, s_t, m_t = rec
        try:
            s = float(s_t.item()) if isinstance(s_t, torch.Tensor) else float(s_t)
            m = float(m_t.item()) if isinstance(m_t, torch.Tensor) else float(m_t)
        except Exception:  # noqa: BLE001
            s, m = float("nan"), float("nan")
        out.append((name, shape, s, m))
    records[:] = out
    return out


def diff_layer_records(
    ref_records: list[LayerRecord],
    per_rank_records: dict[int, list[LayerRecord]],
    *,
    tolerance: float = 1e-3,
) -> dict[str, Any]:
    """Walk ref + per-rank module records in execution order, return
    the first module whose sum-of-ranks doesn't match ref's sum.

    Returns a dict::

        {
            "first_divergent": {"module": str, "ref_sum": float,
                                "ranks_sum": float, "rel_diff": float}
                                or None,
            "checked": int,             # modules compared
            "max_rel_diff": float,      # worst rel diff seen
        }

    Empty per-rank dict or empty ref records → ``first_divergent=None``,
    ``checked=0``. The caller surfaces ``first_divergent.module`` in
    the report's ``next_action``.
    """
    if not ref_records or not per_rank_records:
        return {
            "first_divergent": None,
            "checked": 0,
            "max_rel_diff": 0.0,
            "top_divergent": [],
        }

    rank_lists = [per_rank_records[r] for r in sorted(per_rank_records)]
    n_min = min(len(ref_records), *(len(rl) for rl in rank_lists))

    first_divergent: dict[str, Any] | None = None
    max_rel = 0.0
    all_divergent: list[dict[str, Any]] = []

    for i in range(n_min):
        name = ref_records[i][0]
        ref_sum = ref_records[i][2]
        rank_sums = [rl[i][2] for rl in rank_lists]
        # Detect replicated outputs vs partition-distributed outputs.
        # Replicated (e.g. lookup tables, embeddings of replicated
        # inputs, parameter buffers): every rank's sum equals the ref —
        # comparing ``sum(rank_sums)`` vs ``ref_sum`` would falsely
        # report ``world_size× ref`` as 100% divergence. Use any-rank
        # vs ref instead. Tolerance: 1e-6 of the max rank value covers
        # fp32 noise without false negatives.
        biggest = max(abs(s) for s in rank_sums) or 1.0
        rank_spread = max(rank_sums) - min(rank_sums)
        replicated = rank_spread / biggest < 1e-6
        if replicated:
            cmp_value = rank_sums[0]
        else:
            cmp_value = sum(rank_sums)
        denom = max(abs(ref_sum), 1.0)
        rel = abs(cmp_value - ref_sum) / denom
        max_rel = max(max_rel, rel)
        if rel > tolerance:
            entry = {
                "module": name,
                "ref_sum": ref_sum,
                "ranks_sum": cmp_value,
                "rel_diff": rel,
                "kind": "replicated" if replicated else "partition",
                "exec_index": i,
            }
            all_divergent.append(entry)
            if first_divergent is None:
                first_divergent = entry

    # Top-N by rel_diff so the user sees both "first" (root cause
    # localization) and "biggest" (impact). Cap at 8 to keep next_action
    # bounded even when the model has hundreds of divergent layers.
    top_divergent = sorted(all_divergent, key=lambda d: d["rel_diff"], reverse=True)[:8]

    return {
        "first_divergent": first_divergent,
        "checked": n_min,
        "max_rel_diff": max_rel,
        "top_divergent": top_divergent,
    }
