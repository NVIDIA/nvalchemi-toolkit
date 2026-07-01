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

"""Pre-flight detection of ``@torch.jit.script`` ops that need ShardTensor
marshalling — turn an opaque CUDA illegal-memory-access into an up-front hint.

A scripted op that receives a requires-grad ShardTensor on the distributed
halo path builds a TensorExpr-fused CUDA kernel that reads the storage-less
wrapper's near-null ``data_ptr`` → CUDA illegal memory access (IMA). The fix is
to *marshal* the op across the boundary (Route C — unwrap ShardTensor→local,
run the still-scripted op, re-wrap).

Two scripted-op shapes exist, and they are covered differently:

* **Scripted submodules** (``torch.jit.ScriptModule`` instances, e.g. e3nn's
  ``TensorProduct._compiled_main_*``) — caught by the **auto-discovery** safety
  net (``DomainConfig.scripted_marshal="auto"``, the default), which wraps each
  one's ``forward`` at ``DistributedModel`` setup.
* **Module-level scripted functions** (``torch.jit.ScriptFunction`` globals
  called from a plain ``nn.Module.forward``, e.g. e3nn's
  ``_spherical_harmonics``) — **NOT** caught by auto-discovery (it only walks
  ``named_modules`` for submodules), so they must be declared as a marshalling
  ``JitAdapter`` on the spec. This is the real IMA vector that auto-discovery
  cannot self-heal — and the one this pre-flight check exists to surface.

The check is static (no GPU, no forward), so it runs cheaply before the
multi-process validation spawn.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

import torch

from nvalchemi.distributed.validate.layer_diagnostics import _walk_modules

__all__ = [
    "ScriptedOpReport",
    "detect_scripted_ops",
    "apply_marshal_adapters",
]


def _is_script_module(obj: Any) -> bool:
    return isinstance(obj, torch.jit.ScriptModule)


def _is_script_function(obj: Any) -> bool:
    sf = getattr(torch.jit, "ScriptFunction", None)
    return sf is not None and isinstance(obj, sf)


def _declared_jit_targets(spec: Any) -> set[tuple[str, str]]:
    """``(module_path, attr_name)`` for every marshalling ``JitAdapter`` already
    declared on ``spec``'s ``third_party_helpers``."""
    targets: set[tuple[str, str]] = set()
    dist = getattr(spec, "distribution", None)
    helpers = getattr(dist, "third_party_helpers", ()) if dist is not None else ()
    for h in helpers:
        if type(h).__name__ == "JitAdapter":
            targets.add((h.module_path, h.attr_name))
    return targets


@dataclass
class ScriptedOpReport:
    """Static inventory of a model's scripted ops vs. the marshalling coverage.

    ``undeclared_functions`` is the actionable set: module-level scripted
    functions reachable from the model that are NOT covered by a declared
    ``JitAdapter`` (and not excluded). Each is a near-certain IMA on the
    distributed halo path unless marshalled.
    """

    scripted_submodules: list[str] = field(default_factory=list)
    """Qualified names of ``ScriptModule`` submodules — auto-marshalled by the
    default ``scripted_marshal="auto"`` safety net (no user action needed)."""

    module_level_functions: list[tuple[str, str]] = field(default_factory=list)
    """All ``(module_path, attr)`` module-level ``ScriptFunction``s found on the
    defining modules of the model's submodules."""

    declared_functions: list[tuple[str, str]] = field(default_factory=list)
    """Subset of :attr:`module_level_functions` already covered by a declared
    marshalling ``JitAdapter`` (or matched by ``exclude``)."""

    undeclared_functions: list[tuple[str, str]] = field(default_factory=list)
    """Subset NOT covered — the IMA risk. Each needs a declared ``JitAdapter``
    (auto-discovery cannot catch module-level scripted functions)."""

    @property
    def has_risk(self) -> bool:
        return bool(self.undeclared_functions)

    def suggested_adapters_src(self) -> str:
        """Paste-able ``JitAdapter`` declarations for the undeclared functions."""
        lines = [
            f'        JitAdapter("{mp}", "{attr}", mode="marshal"),'
            for mp, attr in self.undeclared_functions
        ]
        return "\n".join(lines)

    def format_hint(self) -> str:
        """Human-readable pre-flight hint, or ``""`` when there's no risk."""
        if not self.undeclared_functions:
            return ""
        names = ", ".join(f"{mp}.{attr}" for mp, attr in self.undeclared_functions)
        return (
            "Scripted-op marshalling pre-flight: the model calls module-level "
            "``@torch.jit.script`` function(s) that auto-discovery cannot wrap "
            f"(it only catches ScriptModule submodules): {names}. On the "
            "distributed halo path a scripted op that receives a requires-grad "
            "ShardTensor builds a fused CUDA kernel that reads the storage-less "
            "wrapper's null data_ptr → CUDA illegal memory access. Declare a "
            "marshalling JitAdapter for each on the spec's "
            "``distribution.third_party_helpers`` (Route C):\n"
            "    from nvalchemi.distributed._core.adapter import JitAdapter\n"
            "    spec = dataclasses.replace(spec, distribution=dataclasses.replace(\n"
            "        spec.distribution,\n"
            "        third_party_helpers=spec.distribution.third_party_helpers + (\n"
            f"{self.suggested_adapters_src()}\n"
            "        ),\n"
            "    ))\n"
            "If any listed op is genuinely CROSS-RANK (marshalling to local "
            "would give wrong numbers), exclude it via "
            "``DomainConfig.scripted_marshal_exclude`` and handle it with a "
            "custom_op / halo-aware path instead. The equivalence check below "
            "will catch a wrongly-marshalled cross-rank op as a divergence."
        )


def detect_scripted_ops(
    model: Any,
    spec: Any = None,
    *,
    exclude: tuple[str, ...] = (),
    max_depth: int = 6,
) -> ScriptedOpReport:
    """Statically inventory ``model``'s scripted ops and cross-reference the
    marshalling coverage declared on ``spec``.

    Walks every module reachable from ``model`` (via
    :func:`~nvalchemi.distributed.validate.layer_diagnostics._walk_modules`, so
    wrappers holding their net under a plain attribute are covered). Records
    ``ScriptModule`` submodules (auto-covered) and scans the *defining module*
    of every submodule's class for module-level ``ScriptFunction`` globals — the
    vector auto-discovery misses. ``exclude`` substrings suppress a
    ``module.attr`` from the undeclared set (cross-rank ops handled elsewhere).
    """
    submodules: list[str] = []
    candidate_modules: dict[str, None] = {}  # ordered set of defining module paths

    # Seed with the model/wrapper's own defining module — its forward may call
    # module-level scripted helpers defined alongside it (``_walk_modules``
    # yields submodules, not the root, and a scripted submodule's runtime type
    # is ``torch.jit._script.RecursiveScriptModule``, not its source module).
    root_mod = type(model).__module__
    if root_mod:
        candidate_modules[root_mod] = None

    for name, mod in _walk_modules(model, max_depth=max_depth):
        if _is_script_module(mod):
            submodules.append(name)
        mod_path = type(mod).__module__
        if mod_path and mod_path not in candidate_modules:
            candidate_modules[mod_path] = None

    functions: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for mod_path in candidate_modules:
        pymod = sys.modules.get(mod_path)
        if pymod is None:
            continue
        for attr in dir(pymod):
            try:
                obj = getattr(pymod, attr)
            except Exception:  # noqa: BLE001
                continue
            if _is_script_function(obj):
                key = (mod_path, attr)
                if key not in seen:
                    seen.add(key)
                    functions.append(key)

    declared_targets = _declared_jit_targets(spec) if spec is not None else set()

    def _excluded(mp: str, attr: str) -> bool:
        target = f"{mp}.{attr}"
        return any(pat in target or pat == attr for pat in exclude)

    declared: list[tuple[str, str]] = []
    undeclared: list[tuple[str, str]] = []
    for mp, attr in functions:
        if (mp, attr) in declared_targets or _excluded(mp, attr):
            declared.append((mp, attr))
        else:
            undeclared.append((mp, attr))

    return ScriptedOpReport(
        scripted_submodules=submodules,
        module_level_functions=functions,
        declared_functions=declared,
        undeclared_functions=undeclared,
    )


def apply_marshal_adapters(spec: Any, functions: list[tuple[str, str]]) -> Any:
    """Return ``spec`` with a marshalling ``JitAdapter`` added to
    ``distribution.third_party_helpers`` for each ``(module_path, attr)`` not
    already declared. Used by ``trace_and_validate``'s auto-fix to self-heal an
    undeclared module-level scripted function before it IMAs."""
    if not functions:
        return spec

    import dataclasses

    from nvalchemi.distributed._core.adapter import JitAdapter

    existing = _declared_jit_targets(spec)
    additions = tuple(
        JitAdapter(mp, attr, mode="marshal")
        for mp, attr in functions
        if (mp, attr) not in existing
    )
    if not additions:
        return spec
    return dataclasses.replace(
        spec,
        distribution=dataclasses.replace(
            spec.distribution,
            third_party_helpers=spec.distribution.third_party_helpers + additions,
        ),
    )
