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

"""CI gate: examples must not import framework internals.

A contributor should be able to wire domain-decomposed inference for a new
model using only the public ``nvalchemi.distributed`` surface, touching zero
framework internals. The ``*_byo_*`` examples demonstrate that; this test
fails if one of them reaches into a private module (``_core`` / ``_upstream``
/ ``_chemistry``), which would mean the public API is missing something a
real author needs.

Static AST scan only — no model dependencies, GPU, or distributed
backend required, so it runs everywhere CI does.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples" / "distributed"

# Private subpackages of nvalchemi.distributed an example author must never need.
_PRIVATE_MARKERS = ("_core", "_upstream", "_chemistry")


def _byo_example_files() -> list[Path]:
    return sorted(_EXAMPLES_DIR.glob("*_byo_*.py"))


def _private_imports(source: str) -> list[str]:
    """Return the dotted module paths imported from a private internal."""
    tree = ast.parse(source)
    offenders: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names = [node.module]
        for mod in names:
            parts = mod.split(".")
            if any(marker in parts for marker in _PRIVATE_MARKERS):
                offenders.append(mod)
    return offenders


def test_byo_examples_exist() -> None:
    files = _byo_example_files()
    assert files, f"no *_byo_*.py examples found under {_EXAMPLES_DIR}"


@pytest.mark.parametrize("example", _byo_example_files(), ids=lambda p: p.name)
def test_byo_example_imports_only_public_api(example: Path) -> None:
    offenders = _private_imports(example.read_text())
    assert not offenders, (
        f"{example.name} imports framework internals {offenders}; an example "
        "must reach only the public nvalchemi.distributed surface. Promote the "
        "needed symbol to nvalchemi/distributed/__init__.py or ops.py."
    )
