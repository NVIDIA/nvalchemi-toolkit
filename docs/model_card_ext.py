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
    Render a table of canonical names from the known-artifact registry.
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

from nvalchemi.models.registry import get_known_artifact, list_known_artifacts

logger = logging.getLogger(__name__)

_ComponentSpec = tuple[str, str, str]

_COMPONENTS: dict[str, list[_ComponentSpec]] = {
    "ml": [
        ("MACEPotential", "nvalchemi.models.mace", "MACEPotential"),
        ("AIMNet2Potential", "nvalchemi.models.aimnet2", "AIMNet2Potential"),
        ("DemoPotential", "nvalchemi.models.demo", "DemoPotential"),
    ],
    "physical": [
        ("DSFCoulombPotential", "nvalchemi.models.dsf", "DSFCoulombPotential"),
        ("DFTD3Potential", "nvalchemi.models.dftd3", "DFTD3Potential"),
        ("LennardJonesPotential", "nvalchemi.models.lj", "LennardJonesPotential"),
        ("EwaldCoulombPotential", "nvalchemi.models.ewald", "EwaldCoulombPotential"),
        ("PMEPotential", "nvalchemi.models.pme", "PMEPotential"),
    ],
}
_COMPONENTS["all"] = _COMPONENTS["ml"] + _COMPONENTS["physical"]


def _format_set(values: frozenset[str]) -> str:
    """Return a compact comma-separated representation for one set."""

    return ", ".join(sorted(values)) if values else "\u2014"


def _load_card(module_path: str, class_name: str) -> Any | None:
    """Return the class-level card for one component, or ``None`` on failure."""

    try:
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return getattr(cls, "card", None)
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
            "     - Default outputs",
            "     - Neighbor source",
            "     - Neighbor format",
            "     - Boundary modes",
        ]

        for display_name, module_path, class_name in specs:
            card = _load_card(module_path, class_name)
            if card is None:
                continue
            qualified = f"{module_path}.{class_name}"
            rst_lines.append(f"   * - :class:`~{qualified}`")
            rst_lines.append(f"     - {_format_set(card.result_keys)}")
            rst_lines.append(f"     - {_format_set(card.default_result_keys)}")
            rst_lines.append(f"     - {card.neighbor_requirement.source}")
            rst_lines.append(
                "     - "
                + (
                    card.neighbor_requirement.format.upper()
                    if card.neighbor_requirement.format is not None
                    else "\u2014"
                )
            )
            rst_lines.append(f"     - {_format_set(card.boundary_modes)}")

        node = nodes.container()
        self.state.nested_parse(StringList(rst_lines, source="model_card_ext"), 0, node)
        return node.children


class FoundationModelTableDirective(SphinxDirective):
    """Render a table of canonical known-artifact registry names."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec = {"family": lambda x: x.strip().lower()}

    def run(self) -> list[nodes.Node]:
        family = self.options.get("family")
        names = list_known_artifacts(family)

        if not names:
            return [nodes.paragraph(text="No known artifacts are registered.")]

        rst_lines = [
            ".. list-table::",
            "   :header-rows: 1",
            "   :widths: 24 16 18 42",
            "",
            "   * - Name",
            "     - Family",
            "     - Aliases",
            "     - Source",
        ]

        for name in names:
            if family is None:
                entry = None
                for candidate_family in ("aimnet2", "mace", "dftd3"):
                    try:
                        entry = get_known_artifact(name, candidate_family)
                        break
                    except KeyError:
                        continue
                if entry is None:
                    continue
            else:
                entry = get_known_artifact(name, family)

            aliases = ", ".join(entry.aliases) if entry.aliases else "\u2014"
            source = entry.url or entry.metadata.get("source", "\u2014")
            rst_lines.append(f"   * - ``{entry.name}``")
            rst_lines.append(f"     - {entry.family}")
            rst_lines.append(f"     - {aliases}")
            rst_lines.append(f"     - {source}")

        node = nodes.container()
        self.state.nested_parse(StringList(rst_lines, source="model_card_ext"), 0, node)
        return node.children


def setup(app: Sphinx) -> dict[str, bool]:
    """Register the custom directives with Sphinx."""

    app.add_directive("model-capability-table", ModelCapabilityTableDirective)
    app.add_directive("foundation-model-table", FoundationModelTableDirective)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
