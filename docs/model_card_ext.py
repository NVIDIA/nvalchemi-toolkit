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
"""Sphinx directives for the new ``nvalchemi.models`` component surface.

Provides two directives:

``.. model-capability-table::``
    Render a table of component-level capabilities for built-in
    calculator steps in :mod:`nvalchemi.models`.

``.. foundation-model-table::``
    Render a table of built-in named-loading entry points.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)
_EM_DASH = "\u2014"

_ComponentSpec = tuple[str, str, str]

_COMPONENTS: dict[str, list[_ComponentSpec]] = {
    "ml": [
        ("MACEModel", "nvalchemi.models.mace", "MACEModel"),
        ("AIMNet2Model", "nvalchemi.models.aimnet2", "AIMNet2Model"),
        ("DemoModel", "nvalchemi.models.demo", "DemoModel"),
    ],
    "physical": [
        ("DSFCoulombModel", "nvalchemi.models.electrostatics", "DSFCoulombModel"),
        ("DFTD3Model", "nvalchemi.models.dftd3", "DFTD3Model"),
        ("LennardJonesModel", "nvalchemi.models.lj", "LennardJonesModel"),
        ("EwaldCoulombModel", "nvalchemi.models.electrostatics", "EwaldCoulombModel"),
        ("PMEModel", "nvalchemi.models.electrostatics", "PMEModel"),
    ],
}
_COMPONENTS["all"] = _COMPONENTS["ml"] + _COMPONENTS["physical"]


def _format_set(values: frozenset[str]) -> str:
    """Return a compact comma-separated representation for one set."""

    return ", ".join(sorted(values)) if values else "\u2014"


def _load_spec(module_path: str, class_name: str) -> Any | None:
    """Return the class-level spec for one component, or ``None`` on failure."""

    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return getattr(cls, "spec", None)
    except Exception as exc:
        logger.warning(
            "model_card_ext: could not import %s.%s: %s",
            module_path,
            class_name,
            exc,
        )
        return None


class ModelCapabilityTableDirective(SphinxDirective):
    """Render a table of component-level capabilities for built-in models."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec = {"category": lambda x: x.strip().lower()}

    def run(self) -> list[nodes.Node]:
        category = self.options.get("category", "all")
        specs = _COMPONENTS.get(category, _COMPONENTS["all"])

        rst_lines = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 22 16 16 14 14 18",
            "",
            "   * - Component",
            "     - Outputs",
            "     - Optional outputs",
            "     - Additive outputs",
            "     - Autograd",
            "     - Neighbor format",
            "     - PBC mode",
        ]

        for display_name, module_path, class_name in specs:
            spec = _load_spec(module_path, class_name)
            if spec is None:
                continue
            qualified = f"{module_path}.{class_name}"
            rst_lines.append(f"   * - :class:`~{qualified}`")
            rst_lines.append(f"     - {_format_set(spec.outputs)}")
            rst_lines.append(f"     - {_format_set(frozenset(spec.optional_outputs))}")
            rst_lines.append(f"     - {_format_set(spec.additive_outputs)}")
            rst_lines.append(f"     - {'yes' if spec.use_autograd else 'no'}")
            rst_lines.append(
                "     - "
                + (
                    spec.neighbor_requirement.format.upper()
                    if spec.neighbor_requirement.format is not None
                    else _EM_DASH
                )
            )
            rst_lines.append(f"     - {spec.pbc_mode or _EM_DASH}")

        node = nodes.container()
        self.state.nested_parse(StringList(rst_lines, source="model_card_ext"), 0, node)
        return node.children


class FoundationModelTableDirective(SphinxDirective):
    """Render a table of built-in named-loading entry points."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec = {"family": lambda x: x.strip().lower()}

    def run(self) -> list[nodes.Node]:
        rst_lines = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 22 22 56",
            "",
            "   * - Wrapper",
            "     - Named loading",
            "     - Notes",
            "   * - ``AIMNet2Model``",
            "     - AIMNetCentral model names",
            "     - Strings are forwarded directly to ``AIMNet2Calculator``.",
            "   * - ``MACEModel``",
            "     - Upstream MACE foundation model names",
            "     - Strings are forwarded directly to the upstream MACE loader.",
            "   * - ``DFTD3Model``",
            "     - Built-in cached parameter materialization",
            "     - ``download_dftd3_parameters()`` prewarms the cached converted parameter file.",
        ]

        node = nodes.container()
        self.state.nested_parse(StringList(rst_lines, source="model_card_ext"), 0, node)
        return node.children


def setup(app: Sphinx) -> dict[str, bool]:
    """Register the custom directives with Sphinx."""

    app.add_directive("model-capability-table", ModelCapabilityTableDirective)
    app.add_directive("foundation-model-table", FoundationModelTableDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
