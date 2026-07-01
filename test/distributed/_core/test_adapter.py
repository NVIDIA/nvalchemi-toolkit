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

"""Lifecycle and introspection tests for ``_core/adapter.py``.

These exercise the registry's install / restore semantics and
``AdapterStatus`` shape; the integration with
:class:`DistributedModel.__enter__/__exit__` is covered by the
validator suite.
"""

from __future__ import annotations

import sys
import types

import pytest

from nvalchemi.distributed._core.adapter import (
    AdapterRegistry,
    AdapterStatus,
    JitAdapter,
    PythonAdapter,
)

# ----------------------------------------------------------------------
# A throwaway test module we can mutate via PythonAdapter / JitAdapter.
# ----------------------------------------------------------------------


def _make_test_module(name: str = "_test_adapter_target_module") -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.original_fn = lambda x: f"original({x})"
    mod.original_fn.__module__ = name
    sys.modules[name] = mod
    return mod


def _replacement(x: object) -> str:
    return f"replaced({x})"


# ----------------------------------------------------------------------
# PythonAdapter
# ----------------------------------------------------------------------


class TestPythonAdapter:
    def test_install_replaces_module_attr(self):
        mod = _make_test_module("p_test_install_replaces")
        adapter = PythonAdapter(
            module_path=mod.__name__,
            attr_name="original_fn",
            replacement=_replacement,
        )
        assert mod.original_fn(1) == "original(1)"
        memento = adapter.install()
        try:
            assert mod.original_fn(1) == "replaced(1)"
            assert memento["module"] is mod
        finally:
            adapter.restore(memento)
            assert mod.original_fn(1) == "original(1)"

    def test_no_replacement_is_deferred(self):
        # ``replacement=None`` is the declaration-only form: the registry
        # tracks the entry (so diagnostics see it) but doesn't swap the
        # attribute — the wrapper's distributed_setup is responsible.
        mod = _make_test_module("p_test_no_repl")
        adapter = PythonAdapter(
            module_path=mod.__name__,
            attr_name="original_fn",
        )
        memento = adapter.install()
        assert memento.get("deferred") is True
        # Module attribute is untouched.
        assert mod.original_fn(1) == "original(1)"
        # Restore is also a no-op for deferred mementos.
        adapter.restore(memento)
        assert mod.original_fn(1) == "original(1)"

    def test_install_site_captured(self):
        mod = _make_test_module("p_test_site")
        adapter = PythonAdapter(
            module_path=mod.__name__,
            attr_name="original_fn",
            replacement=_replacement,
        )
        # Captured in __post_init__ — ends with this file's name.
        assert "test_adapter.py" in adapter.install_site

    def test_describe_pending(self):
        adapter = PythonAdapter(
            module_path="x",
            attr_name="y",
            replacement=_replacement,
        )
        status = adapter.describe()
        assert isinstance(status, AdapterStatus)
        assert status.kind == "python"
        assert status.target == "x.y"
        assert status.state == "pending"


# ----------------------------------------------------------------------
# JitAdapter — same lifecycle, different ``kind``.
# ----------------------------------------------------------------------


class TestJitAdapter:
    def test_install_replaces_module_attr(self):
        mod = _make_test_module("j_test_install")
        adapter = JitAdapter(
            module_path=mod.__name__,
            attr_name="original_fn",
            replacement=_replacement,
        )
        memento = adapter.install()
        try:
            assert mod.original_fn(1) == "replaced(1)"
        finally:
            adapter.restore(memento)
            assert mod.original_fn(1) == "original(1)"

    def test_describe_kind_is_jit(self):
        adapter = JitAdapter(
            module_path="x",
            attr_name="y",
            replacement=_replacement,
        )
        assert adapter.describe().kind == "jit"


# ----------------------------------------------------------------------
# AdapterRegistry
# ----------------------------------------------------------------------


class TestAdapterRegistry:
    def test_install_then_restore_basic(self):
        mod = _make_test_module("r_test_basic")
        adapter = PythonAdapter(
            module_path=mod.__name__,
            attr_name="original_fn",
            replacement=_replacement,
        )
        registry = AdapterRegistry()
        registry.install([adapter])
        assert mod.original_fn(1) == "replaced(1)"
        statuses = registry.list_active()
        assert len(statuses) == 1
        assert statuses[0].state == "installed"

        registry.restore()
        assert mod.original_fn(1) == "original(1)"
        assert registry.list_active()[0].state == "restored"

    def test_install_failure_rolls_back(self):
        mod = _make_test_module("r_test_rollback")
        # First adapter installs cleanly. Second points at a module that
        # cannot be imported, so its install raises.
        good = PythonAdapter(
            module_path=mod.__name__,
            attr_name="original_fn",
            replacement=_replacement,
        )
        bad = PythonAdapter(
            module_path="this_module_does_not_exist_xyz",
            attr_name="anything",
            replacement=_replacement,
        )

        registry = AdapterRegistry()
        with pytest.raises(ModuleNotFoundError):
            registry.install([good, bad])

        # ``good`` should have been rolled back — original is back.
        assert mod.original_fn(1) == "original(1)"

    def test_restore_is_idempotent(self):
        mod = _make_test_module("r_test_idempotent")
        adapter = PythonAdapter(
            module_path=mod.__name__,
            attr_name="original_fn",
            replacement=_replacement,
        )
        registry = AdapterRegistry()
        registry.install([adapter])
        registry.restore()
        # Calling restore again does nothing (already-restored handles
        # are skipped) and doesn't raise.
        registry.restore()
        assert mod.original_fn(1) == "original(1)"

    def test_list_active_shape(self):
        mod = _make_test_module("r_test_list_active")
        adapter = PythonAdapter(
            module_path=mod.__name__,
            attr_name="original_fn",
            replacement=_replacement,
        )
        registry = AdapterRegistry()
        registry.install([adapter])
        try:
            [status] = registry.list_active()
            assert status.kind == "python"
            assert status.target == f"{mod.__name__}.original_fn"
            assert status.state == "installed"
            assert "test_adapter.py" in status.install_site
            assert status.error is None
        finally:
            registry.restore()
