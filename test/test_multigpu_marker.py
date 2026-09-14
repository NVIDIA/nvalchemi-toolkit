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

"""Static check that multi-GPU tests carry ``@pytest.mark.multigpu``.

An unmarked one runs nowhere: the 1-GPU job skips it for lack of devices and
the 2-GPU job, which selects on the marker, never picks it up.

Multi-GPU means NCCL with rank ``r`` on ``cuda:r``. Gloo/CPU multi-rank tests,
and NCCL ranks sharing ``cuda:0``, are not multi-GPU.
"""

from __future__ import annotations

import ast
import pathlib

TEST_ROOT = pathlib.Path(__file__).parent
# This module names the idioms it forbids, so it excludes itself.
_SELF = pathlib.Path(__file__).resolve()

_HARNESS_SEEDS = {"init_nccl", "nccl_worker"}


def _is_multigpu_marker(node: ast.expr) -> bool:
    """``True`` for a ``@pytest.mark.multigpu`` decorator."""
    return isinstance(node, ast.Attribute) and node.attr == "multigpu"


def _referenced_names(node: ast.AST) -> set[str]:
    """Every bare name and attribute tail referenced anywhere under *node*."""
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name):
            names.add(child.id)
        elif isinstance(child, ast.Attribute):
            names.add(child.attr)
    return names


def _pins_rank_to_gpu(fn: ast.FunctionDef, source: str) -> bool:
    """``True`` if *fn* opens a NCCL group and pins a GPU.

    Both halves are required — NCCL alone is satisfied by ranks sharing
    ``cuda:0``, which needs one device.
    """
    segment = ast.get_source_segment(source, fn) or ""
    return "nccl" in segment and "set_device" in segment


def _imported_seed_aliases(tree: ast.Module) -> set[str]:
    """Local names bound to a harness helper, following ``as`` renames.

    Most of the suite imports ``from _dd_harness import nccl_worker as _worker``,
    so matching the original name alone sees nothing and the check passes
    vacuously.
    """
    aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom | ast.Import):
            for a in node.names:
                if a.name in _HARNESS_SEEDS:
                    aliases.add(a.asname or a.name)
    return aliases


def _multigpu_functions(tree: ast.Module, source: str) -> set[str]:
    """Module-level functions that reach a rank-pinned NCCL group.

    Closed transitively, so a test calling a local wrapper around
    ``mp.spawn(nccl_worker, ...)`` is still caught.
    """
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    }
    multigpu = {name for name in functions if name in _HARNESS_SEEDS}
    multigpu |= {
        name
        for name, node in functions.items()
        if isinstance(node, ast.FunctionDef) and _pins_rank_to_gpu(node, source)
    }
    # Imported harness helpers are referenced by name, not defined here --
    # under whatever local name the import bound them to.
    multigpu |= _HARNESS_SEEDS | _imported_seed_aliases(tree)

    # Calling a multi-GPU function makes you one.
    changed = True
    while changed:
        changed = False
        for name, node in functions.items():
            if name not in multigpu and _referenced_names(node) & multigpu:
                multigpu.add(name)
                changed = True
    return multigpu


def _unmarked_multigpu_tests(path: pathlib.Path) -> list[str]:
    """Test functions in *path* that drive a rank-pinned NCCL group unmarked."""
    source = path.read_text()
    tree = ast.parse(source)
    multigpu = _multigpu_functions(tree, source)

    offenders = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or not node.name.startswith("test_"):
            continue
        if any(_is_multigpu_marker(d) for d in node.decorator_list):
            continue
        if _referenced_names(node) & multigpu:
            offenders.append(f"{path.relative_to(TEST_ROOT)}::{node.name}")
    return offenders


def test_every_nccl_multigpu_test_carries_the_marker() -> None:
    """No test spawns rank-pinned NCCL workers without the marker."""
    offenders: list[str] = []
    for path in sorted(TEST_ROOT.rglob("test_*.py")):
        if path.resolve() != _SELF:
            offenders.extend(_unmarked_multigpu_tests(path))

    assert not offenders, (
        "These tests spawn NCCL workers pinned to distinct GPUs but are not "
        "marked, so no CI tier runs them. Add @pytest.mark.multigpu:\n  "
        + "\n  ".join(offenders)
    )


def test_no_hand_rolled_device_count_gates() -> None:
    """GPU-count gating lives in the marker, not in per-file ``skipif``s.

    Such a gate skips on the 1-GPU runner without making the test selectable
    by ``-m multigpu``, so it runs nowhere.
    """
    offenders = [
        str(path.relative_to(TEST_ROOT))
        for path in sorted(TEST_ROOT.rglob("test_*.py"))
        if path.resolve() != _SELF and "device_count()" in path.read_text()
    ]
    assert not offenders, (
        "Use @pytest.mark.multigpu instead of a hand-rolled "
        "torch.cuda.device_count() skip gate in:\n  " + "\n  ".join(offenders)
    )
