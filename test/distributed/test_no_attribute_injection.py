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

"""Static-analysis regression guard against wrapper attribute injection.

:class:`DistributedModel` passes runtime state to wrapped models via a single
:class:`DistributedContext` reference (passed to :meth:`distributed_setup` and
mutated in place per-step), not via ``setattr(self._wrapper, "_halo_meta", ...)``
style attribute injection.

This test AST-walks ``distributed_model.py`` (and a few related modules) and
asserts that no assignment writes the known-leak attribute names on
``self._wrapper`` (or any expression referring to the wrapper).

If you find yourself wanting to add an attribute injection on the
wrapper, instead add a typed field to :class:`DistributedContext` (or
its ``extras`` dict) and write to ``self._dist_ctx.<field>``.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


# Attribute names that must never be injected onto the wrapper. The list
# is closed by design — if you add a runtime field, plumb it through
# DistributedContext, not via setattr-on-wrapper.
LEAK_ATTRS: frozenset[str] = frozenset(
    {
        "_halo_meta",
        "_halo_cfg",
        "_gather_meta",
        "_n_systems_global",
    }
)

# Attribute access expressions whose write to a LEAK_ATTRS name
# constitutes injection. Each entry is the tail of a Name/Attribute
# chain that resolves to "the wrapper".
WRAPPER_ATTR_TAILS: frozenset[str] = frozenset({"_wrapper", "wrapper"})


def _files_to_check() -> list[pathlib.Path]:
    return [
        REPO_ROOT / "nvalchemi" / "distributed" / "distributed_model.py",
        REPO_ROOT / "nvalchemi" / "distributed" / "domain_parallel.py",
    ]


def _attr_writes_to_wrapper(tree: ast.AST) -> list[tuple[int, str]]:
    """Walk *tree* and return ``(lineno, attr_name)`` for every
    assignment of the form ``<expr ending in ._wrapper>.<LEAK_ATTRS>
    = ...``.
    """
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Attribute):
                continue
            if target.attr not in LEAK_ATTRS:
                continue
            # The LHS is ``<obj>.<leak_attr>``; ``<obj>`` itself must
            # be an Attribute whose own ``.attr`` is a wrapper alias.
            obj = target.value
            if isinstance(obj, ast.Attribute) and obj.attr in WRAPPER_ATTR_TAILS:
                hits.append((target.lineno, target.attr))
            elif isinstance(obj, ast.Name) and obj.id in WRAPPER_ATTR_TAILS:
                hits.append((target.lineno, target.attr))
    return hits


@pytest.mark.parametrize("path", _files_to_check(), ids=lambda p: p.name)
def test_no_setattr_on_wrapper(path: pathlib.Path) -> None:
    """Assert *path* contains no ``self._wrapper.<leak_attr> = ...``
    or ``wrapper.<leak_attr> = ...`` assignments.

    If this fires: route the value through
    :class:`~nvalchemi.distributed._core.context.DistributedContext`
    instead — write to ``self._dist_ctx.<field>`` (or
    ``self._dist_ctx.extras[<key>]`` for non-typed scratch).
    """
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    hits = _attr_writes_to_wrapper(tree)
    if hits:
        joined = "; ".join(f"line {ln}: ._wrapper.{name}" for ln, name in hits)
        pytest.fail(
            f"{path.name}: found {len(hits)} attribute injection write(s) "
            f"on the wrapper that bypass DistributedContext: {joined}. "
            f"Add a typed field on DistributedContext and write through "
            f"``self._dist_ctx.<field>`` instead."
        )


def test_distributed_context_carries_all_known_runtime_fields() -> None:
    """Sanity check: every leak-attr name has a corresponding field on
    :class:`DistributedContext` so callers have a real migration path
    when this guard fires."""
    from nvalchemi.distributed._core.context import DistributedContext

    fields = set(DistributedContext.__dataclass_fields__)

    # Mapping from old attribute → new ctx field. Update if the ctx
    # field name diverges from the leak attr (e.g. ``_halo_cfg`` →
    # ``halo_config``).
    expected = {
        "_halo_meta": "halo_meta",
        "_halo_cfg": "halo_config",
        "_gather_meta": "gather_meta",
        "_n_systems_global": "n_systems_global",
    }
    missing = {old for old, new in expected.items() if new not in fields}
    assert not missing, (
        f"DistributedContext is missing fields for migrated attrs: {missing}"
    )
