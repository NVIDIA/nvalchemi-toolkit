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
"""Sphinx extension that auto-generates model capability and registry tables.

Provides two directives:

``.. model-capability-table::``
    Renders an RST table of ModelCard capabilities for every wrapper that
    inherits from :class:`~nvalchemi.models.base.BaseModelMixin`.

``.. foundation-model-table::``
    Renders an RST table of registered foundation model checkpoints from
    :func:`~nvalchemi.models.registry.list_foundation_models`.
"""

from __future__ import annotations

import logging
from typing import Any

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective

from nvalchemi.models.base import ModelCard, NeighborConfig, NeighborListFormat

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Static fallback ModelCard values for wrappers that cannot be trivially
# instantiated at doc-build time (require nn.Module, download files, etc.)
# ---------------------------------------------------------------------------

_FALLBACK_CARDS: dict[str, ModelCard] = {
    "MACEWrapper": ModelCard(
        outputs={"energy", "forces", "stress", "hessian"},
        autograd_outputs={"forces", "stress"},
        autograd_inputs={"positions"},
        required_inputs=set(),
        supports_pbc=True,
        needs_pbc=False,
        neighbor_config=NeighborConfig(
            cutoff=5.0,
            format=NeighborListFormat.COO,
            half_list=False,
        ),
    ),
    "DFTD3ModelWrapper": ModelCard(
        outputs={"energy", "forces", "stress"},
        autograd_outputs=set(),
        required_inputs=set(),
        supports_pbc=True,
        needs_pbc=False,
        neighbor_config=NeighborConfig(
            cutoff=50.0,
            format=NeighborListFormat.MATRIX,
            half_list=False,
            max_neighbors=128,
        ),
    ),
}

# Wrappers to include in the capability tables, in display order.
# Each tuple: (display_name, module_path, class_name, instantiation_kwargs | None)
# If kwargs is None, use the fallback card.

_WrapperSpec = tuple[str, str, str, dict[str, Any] | None]

_ML_WRAPPER_SPECS: list[_WrapperSpec] = [
    ("MACEWrapper", "nvalchemi.models.mace", "MACEWrapper", None),
    ("DemoModelWrapper", "nvalchemi.models.demo", "DemoModelWrapper", {}),
]

_PHYSICAL_WRAPPER_SPECS: list[_WrapperSpec] = [
    (
        "LennardJonesModelWrapper",
        "nvalchemi.models.lj",
        "LennardJonesModelWrapper",
        {"epsilon": 1.0, "sigma": 1.0, "cutoff": 5.0},
    ),
    ("DFTD3ModelWrapper", "nvalchemi.models.dftd3", "DFTD3ModelWrapper", None),
    ("PMEModelWrapper", "nvalchemi.models.pme", "PMEModelWrapper", {"cutoff": 10.0}),
    (
        "EwaldModelWrapper",
        "nvalchemi.models.ewald",
        "EwaldModelWrapper",
        {"cutoff": 10.0},
    ),
]

# Map class_name -> fully-qualified module path for cross-references.
_WRAPPER_MODULE_MAP: dict[str, str] = {
    spec[2]: f"{spec[1]}.{spec[2]}"
    for spec in _ML_WRAPPER_SPECS + _PHYSICAL_WRAPPER_SPECS
}

_WRAPPER_CATEGORIES: dict[str, list[_WrapperSpec]] = {
    "ml": _ML_WRAPPER_SPECS,
    "physical": _PHYSICAL_WRAPPER_SPECS,
    "all": _ML_WRAPPER_SPECS + _PHYSICAL_WRAPPER_SPECS,
}

# Columns for the capability table: (header_label, field_name_or_callable_key)
# The new set-based ModelCard uses 'outputs' and 'autograd_outputs' sets.
_CAPABILITY_COLUMNS: list[tuple[str, str]] = [
    ("Energies", "_has_energies"),
    ("Forces", "_has_forces"),
    ("Stresses", "_has_stresses"),
    ("PBC", "supports_pbc"),
    ("Needs PBC", "needs_pbc"),
    ("Autograd Forces", "_autograd_forces"),
    ("Extra Inputs", "_extra_inputs"),
    ("Neighbor Fmt.", "_neighbor_format"),
]


def _get_model_card(
    display_name: str,
    module_path: str,
    class_name: str,
    kwargs: dict[str, Any] | None,
) -> ModelCard | None:
    """Try to obtain a ModelCard, falling back to the static dict."""
    if kwargs is None:
        card = _FALLBACK_CARDS.get(class_name)
        if card is not None:
            return card

    try:
        import importlib

        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        instance = cls(**(kwargs or {}))
        return instance.model_card
    except Exception as exc:
        logger.warning(
            "model_card_ext: could not instantiate %s.%s: %s. "
            "Using fallback card if available.",
            module_path,
            class_name,
            exc,
        )
        return _FALLBACK_CARDS.get(class_name)


def _bool_icon(val: bool) -> str:
    return "\u2713" if val else "\u2717"


def _neighbor_format(card: ModelCard) -> str:
    if card.neighbor_config is None:
        return "\u2014"
    return card.neighbor_config.format.value.upper()


def _cell_value(card: ModelCard, field: str) -> str:
    if field == "_neighbor_format":
        return _neighbor_format(card)
    if field == "_has_energies":
        return _bool_icon("energy" in card.outputs)
    if field == "_has_forces":
        return _bool_icon("forces" in card.outputs)
    if field == "_has_stresses":
        return _bool_icon("stress" in card.outputs)
    if field == "_autograd_forces":
        return _bool_icon("forces" in card.autograd_outputs)
    if field == "_extra_inputs":
        if card.required_inputs:
            return ", ".join(sorted(card.required_inputs))
        return "\u2014"
    return _bool_icon(getattr(card, field))


# ---------------------------------------------------------------------------
# Directive: model-capability-table
# ---------------------------------------------------------------------------


class ModelCapabilityTableDirective(SphinxDirective):
    """Render a table of ModelCard capabilities for a category of wrappers.

    Options
    -------
    category : str
        One of ``"ml"``, ``"physical"``, or ``"all"`` (default).
    """

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec = {"category": lambda x: x.strip().lower()}

    def run(self) -> list[nodes.Node]:
        category = self.options.get("category", "all")
        specs = _WRAPPER_CATEGORIES.get(category, _WRAPPER_CATEGORIES["all"])

        rows: list[tuple[str, str, ModelCard]] = []
        for display_name, mod_path, cls_name, kwargs in specs:
            card = _get_model_card(display_name, mod_path, cls_name, kwargs)
            if card is not None:
                rows.append((display_name, f"{mod_path}.{cls_name}", card))
            else:
                logger.warning(
                    "model_card_ext: skipping %s (no card available)", display_name
                )

        if not rows:
            para = nodes.paragraph(text="No model cards could be loaded.")
            return [para]

        # Build RST list-table lines
        headers = ["Wrapper"] + [h for h, _ in _CAPABILITY_COLUMNS]
        col_widths = [25] + [8] * len(_CAPABILITY_COLUMNS)

        rst_lines: list[str] = []
        rst_lines.append(".. list-table::")
        rst_lines.append("   :header-rows: 1")
        rst_lines.append("   :widths: " + " ".join(str(w) for w in col_widths))
        rst_lines.append("")

        # Header row
        rst_lines.append("   * - " + headers[0])
        rst_lines.extend(f"     - {h}" for h in headers[1:])

        # Data rows
        for name, qualified, card in rows:
            api_ref = f":class:`~{qualified}`"
            rst_lines.append(f"   * - {api_ref}")
            for _, field in _CAPABILITY_COLUMNS:
                rst_lines.append(f"     - {_cell_value(card, field)}")

        rst_lines.append("")

        # Parse the RST into docutils nodes
        vl = StringList(rst_lines, source="model_card_ext")
        node = nodes.container()
        self.state.nested_parse(vl, 0, node)
        return node.children


# ---------------------------------------------------------------------------
# Directive: foundation-model-table
# ---------------------------------------------------------------------------


class FoundationModelTableDirective(SphinxDirective):
    """Render a table of registered foundation model checkpoints."""

    has_content = False
    required_arguments = 0
    optional_arguments = 0

    def run(self) -> list[nodes.Node]:
        from nvalchemi.models.registry import (
            get_registry_entry,
            list_foundation_models,
        )

        model_names = list_foundation_models()
        if not model_names:
            para = nodes.paragraph(text="No foundation models registered.")
            return [para]

        rst_lines: list[str] = []
        rst_lines.append(".. list-table::")
        rst_lines.append("   :header-rows: 1")
        rst_lines.append("   :widths: 20 40 25 15")
        rst_lines.append("")
        rst_lines.append("   * - Name")
        rst_lines.append("     - Description")
        rst_lines.append("     - Aliases")
        rst_lines.append("     - Wrapper")

        for name in model_names:
            entry = get_registry_entry(name)
            aliases = (
                ", ".join(f"``{a}``" for a in entry.aliases)
                if entry.aliases
                else "\u2014"
            )
            rst_lines.append(f"   * - ``{entry.name}``")
            rst_lines.append(f"     - {entry.description}")
            rst_lines.append(f"     - {aliases}")
            wrapper_ref = _WRAPPER_MODULE_MAP.get(entry.model_class, entry.model_class)
            rst_lines.append(f"     - :class:`~{wrapper_ref}`")

        rst_lines.append("")

        vl = StringList(rst_lines, source="model_card_ext")
        node = nodes.container()
        self.state.nested_parse(vl, 0, node)
        return node.children


# ---------------------------------------------------------------------------
# Sphinx setup
# ---------------------------------------------------------------------------


def setup(app: Sphinx) -> dict[str, Any]:
    """Register the directives with Sphinx."""
    app.add_directive("model-capability-table", ModelCapabilityTableDirective)
    app.add_directive("foundation-model-table", FoundationModelTableDirective)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
