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
"""Utilities for guarding optional dependencies at runtime."""

from __future__ import annotations

import importlib
import sys
from enum import Enum
from functools import wraps
from types import TracebackType
from typing import Any


class OptionalDependencyError(ImportError):
    """Raised when a required optional dependency is not installed."""

    def __init__(
        self,
        dep: OptionalDependency,
        func_name: str,
        cause: ImportError | None = None,
    ) -> None:
        from rich.console import Console
        from rich.table import Table

        console = Console(stderr=True)
        table = Table(show_header=False, show_lines=True)
        table.add_row(
            "[blue]nvalchemi Optional Dependency Error\n"
            "This error typically indicates an extra dependency group is needed.[/blue]"
        )
        escaped_target = dep.install_target.replace("[", "\\[")
        table.add_row(
            f"[yellow]This feature ('{func_name}') requires optional "
            f"dependency '{dep.import_name}'.\n\n"
            f"Install with: pip install '{escaped_target}'[/yellow]"
        )
        if cause:
            table.add_row(f"[red]{type(cause).__name__}: {cause}[/red]")
        console.print(table)
        super().__init__(
            f"'{dep.import_name}' is not installed. "
            f"Run: pip install '{dep.install_target}'"
        )


def _install_excepthook() -> None:
    """Lazily install a sys.excepthook that suppresses the traceback for OptionalDependencyError."""
    if getattr(_install_excepthook, "_installed", False):
        return
    _install_excepthook._installed = True  # type: ignore[attr-defined]

    _default_excepthook = sys.excepthook

    def _optional_dep_excepthook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        if isinstance(exc_value, OptionalDependencyError):
            return
        _default_excepthook(exc_type, exc_value, exc_tb)

    sys.excepthook = _optional_dep_excepthook


class OptionalDependency(Enum):
    """Registry of optional dependencies with install instructions.

    Each member maps to an ``(import_name, install_target)`` pair.
    Use :meth:`require` as a decorator to guard functions or classes
    that need the dependency::

        @OptionalDependency.PYMATGEN.require
        def needs_pymatgen():
            ...
    """

    ASE = ("ase", "nvalchemi-toolkit[ase]")
    PYMATGEN = ("pymatgen", "nvalchemi-toolkit[pymatgen]")
    MACE = ("mace", "nvalchemi-toolkit[mace]")

    def __init__(self, import_name: str, install_target: str) -> None:
        self.import_name = import_name
        self.install_target = install_target
        self._available: bool | None = None
        self._import_error: ImportError | None = None

    def is_available(self) -> bool:
        """Return True if the dependency can be imported (cached)."""
        if self._available is None:
            try:
                importlib.import_module(self.import_name)
                self._available = True
            except ImportError as e:
                self._available = False
                self._import_error = e
        return self._available

    def _raise_error(self, name: str) -> None:
        """Raise :class:`OptionalDependencyError` with a clean traceback."""
        _install_excepthook()
        err = OptionalDependencyError(self, name, self._import_error)
        err.__traceback__ = None
        raise err from None

    def require(self, obj: Any) -> Any:
        """Decorator that raises :class:`OptionalDependencyError` when the dependency is missing."""
        dep = self

        if isinstance(obj, type):
            original_init = obj.__init__

            @wraps(original_init)
            def wrapped_init(self_arg: Any, *args: Any, **kwargs: Any) -> None:
                if not dep.is_available():
                    dep._raise_error(obj.__qualname__)
                original_init(self_arg, *args, **kwargs)

            obj.__init__ = wrapped_init  # type: ignore[method-assign]
            return obj

        @wraps(obj)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not dep.is_available():
                dep._raise_error(obj.__qualname__)
            return obj(*args, **kwargs)

        return wrapper
