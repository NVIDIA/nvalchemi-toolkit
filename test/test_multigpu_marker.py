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

import pytest

TEST_ROOT = pathlib.Path(__file__).parent
# This module names the idioms it forbids, so it excludes itself.
_SELF = pathlib.Path(__file__).resolve()

_HARNESS_SEEDS = {"init_nccl", "nccl_worker"}


def _is_multigpu_marker(node: ast.expr) -> bool:
    """``True`` for a bare or parameterized ``pytest.mark.multigpu`` decorator."""
    if isinstance(node, ast.Call):
        node = node.func
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


def test_multigpu_marker_audit_recognizes_parameterized_form() -> None:
    """The audit recognizes the GPU-count option without matching other calls."""
    bare = ast.parse("@pytest.mark.multigpu\ndef test_bare(): pass").body[0]
    parameterized = ast.parse(
        "@pytest.mark.multigpu(min_gpus=4)\ndef test_four(): pass"
    ).body[0]
    other = ast.parse("@pytest.mark.slow(reason='slow')\ndef test_other(): pass").body[
        0
    ]

    assert _is_multigpu_marker(bare.decorator_list[0])
    assert _is_multigpu_marker(parameterized.decorator_list[0])
    assert not _is_multigpu_marker(other.decorator_list[0])


def _multigpu_item(request, min_gpus="default", *, name=None):
    """Create a real pytest item carrying the public marker metadata."""
    item = pytest.Function.from_parent(
        request.node.parent,
        name=name or f"synthetic_multigpu_{request.node.name}",
        callobj=lambda: None,
    )
    marker = (
        pytest.mark.multigpu
        if min_gpus == "default"
        else pytest.mark.multigpu(min_gpus=min_gpus)
    )
    item.add_marker(marker)
    return item


@pytest.mark.parametrize(
    ("available_gpus", "min_gpus", "force", "expected_reason"),
    [
        (0, "default", False, "requires >=2 CUDA GPUs (mark: multigpu)"),
        (1, "default", False, "requires >=2 CUDA GPUs (mark: multigpu)"),
        (2, "default", False, None),
        (4, "default", False, None),
        (
            0,
            4,
            False,
            "requires >=4 CUDA GPUs (mark: multigpu(min_gpus=4))",
        ),
        (
            1,
            4,
            False,
            "requires >=4 CUDA GPUs (mark: multigpu(min_gpus=4))",
        ),
        (
            2,
            4,
            False,
            "requires >=4 CUDA GPUs (mark: multigpu(min_gpus=4))",
        ),
        (4, 4, False, None),
        (0, 4, True, None),
    ],
    ids=[
        "default-no-cuda",
        "default-one-gpu",
        "default-two-gpus",
        "default-four-gpus",
        "four-required-no-cuda",
        "four-required-one-gpu",
        "four-required-two-gpus",
        "four-required-four-gpus",
        "force-four-required-no-cuda",
    ],
)
def test_multigpu_collection_hook_uses_marker_minimum(
    request, monkeypatch, available_gpus, min_gpus, force, expected_reason
) -> None:
    """The collection hook gates each marker by its requested GPU count."""
    import torch

    monkeypatch.delenv("NVALCHEMI_FORCE_MULTIGPU", raising=False)
    if force:
        monkeypatch.setenv("NVALCHEMI_FORCE_MULTIGPU", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available_gpus > 0)
    device_count_calls = 0

    def device_count() -> int:
        nonlocal device_count_calls
        device_count_calls += 1
        return available_gpus

    monkeypatch.setattr(torch.cuda, "device_count", device_count)
    item = _multigpu_item(request, min_gpus)
    request.config.hook.pytest_collection_modifyitems(
        session=request.session, config=request.config, items=[item]
    )

    skip = item.get_closest_marker("skip")
    if expected_reason is None:
        assert skip is None
    else:
        assert skip is not None
        assert skip.kwargs["reason"] == expected_reason
    assert device_count_calls == (1 if available_gpus else 0)


@pytest.mark.parametrize("available_gpus", [2, 4])
def test_multigpu_collection_hook_applies_each_items_minimum_once(
    request, monkeypatch, available_gpus
) -> None:
    """One collection can mix default and four-GPU marker requirements."""
    import torch

    monkeypatch.delenv("NVALCHEMI_FORCE_MULTIGPU", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    device_count_calls = 0

    def device_count() -> int:
        nonlocal device_count_calls
        device_count_calls += 1
        return available_gpus

    monkeypatch.setattr(torch.cuda, "device_count", device_count)
    default_item = _multigpu_item(request, name="synthetic_default_multigpu")
    four_gpu_item = _multigpu_item(request, 4, name="synthetic_four_gpu_multigpu")
    request.config.hook.pytest_collection_modifyitems(
        session=request.session,
        config=request.config,
        items=[default_item, four_gpu_item],
    )

    assert default_item.get_closest_marker("skip") is None
    four_gpu_skip = four_gpu_item.get_closest_marker("skip")
    if available_gpus == 2:
        assert four_gpu_skip is not None
        assert (
            four_gpu_skip.kwargs["reason"]
            == "requires >=4 CUDA GPUs (mark: multigpu(min_gpus=4))"
        )
    else:
        assert four_gpu_skip is None
    assert device_count_calls == 1


@pytest.mark.parametrize(
    "min_gpus",
    [
        pytest.param(True, id="true-is-not-an-integer-count"),
        pytest.param(False, id="false-is-not-an-integer-count"),
        pytest.param(1, id="below-minimum"),
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="float"),
        pytest.param("4", id="string"),
        pytest.param(None, id="none"),
    ],
)
def test_multigpu_collection_hook_rejects_invalid_minimum(
    request, monkeypatch, min_gpus
) -> None:
    """Invalid marker metadata fails collection, even with the force override."""
    import torch

    monkeypatch.setenv("NVALCHEMI_FORCE_MULTIGPU", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: pytest.fail("device count is unavailable without CUDA"),
    )
    item = _multigpu_item(request, min_gpus)

    with pytest.raises(
        pytest.UsageError, match="multigpu min_gpus must be an integer >= 2"
    ):
        request.config.hook.pytest_collection_modifyitems(
            session=request.session, config=request.config, items=[item]
        )
