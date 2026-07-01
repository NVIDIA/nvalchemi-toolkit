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

"""Regression test: import-graph boundary between the core
upstream-candidate surface and chemistry-flavored modules.

Wraps :mod:`tools.check_core_imports` as a parameterized pytest. Each
core-surface module gets one test case; failures surface the violating
import line so the offending wrapper change is easy to revert.

Adding a new module to ``nvalchemi/distributed/_core/`` (or to the
``CORE_SURFACE_OUTSIDE_CORE_DIR`` allowlist in the linter) automatically
adds a test case here — the parametrize source is the linter's own
``_core_modules()`` helper.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

# Walk up to the repo root (the dir holding pyproject.toml) rather than a
# fixed parent count, so this survives moving the test between subfolders.
REPO_ROOT = next(
    p for p in pathlib.Path(__file__).resolve().parents if (p / "pyproject.toml").exists()
)
TOOLS_DIR = REPO_ROOT / "tools"


def _load_linter():
    """Import ``tools/check_core_imports.py`` by file path. Avoids
    requiring ``tools/`` to be a package importable on the regular path."""
    if str(TOOLS_DIR) not in sys.path:
        sys.path.insert(0, str(TOOLS_DIR))
    spec = importlib.util.spec_from_file_location(
        "_check_core_imports", TOOLS_DIR / "check_core_imports.py"
    )
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_linter = _load_linter()


def _id(path: pathlib.Path) -> str:
    return str(path.relative_to(REPO_ROOT))


@pytest.mark.parametrize("path", _linter._core_modules(), ids=_id)
def test_no_chemistry_imports_at_load_time(path: pathlib.Path) -> None:
    """Each core-surface module must not import from chemistry-flavored
    modules at module load time.

    Allowed: stdlib, third-party packages, ``_core/`` siblings, and
    other modules in the linter's ``CORE_SURFACE_OUTSIDE_CORE_DIR``
    allowlist. ``TYPE_CHECKING``-guarded imports and lazy in-function
    imports are exempt.

    If this fires: either move the dependency behind a TYPE_CHECKING
    guard (if only used in annotations) or convert it to a callsite-level
    lazy import inside the function body.
    """
    violations = _linter._find_violations(path)
    if violations:
        msg = "\n  ".join(violations)
        pytest.fail(
            f"{path.relative_to(REPO_ROOT)} has chemistry-boundary violations:\n  {msg}"
        )
