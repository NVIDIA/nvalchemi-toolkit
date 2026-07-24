# SPDX-FileCopyrightText: Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import dataclasses
import importlib
import inspect
import re

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList
from sphinx.util.typing import stringify_annotation


def _clean_type(annotation) -> str:
    """Stringify a pydantic field annotation without the ``Annotated[...]`` noise.

    autodoc-pydantic renders the raw source annotation, which for this codebase
    leaks ``Field(...)`` and ``PlainSerializer(...)`` metadata. ``FieldInfo``
    already exposes the resolved inner type, so we stringify that with the
    "smart" mode, which abbreviates every module path to its short name
    (e.g. ``jaxtyping.Integer[Tensor, "V"]`` -> ``Integer[Tensor, "V"]``,
    ``...optimizers.OptimizerConfig`` -> ``OptimizerConfig``) while keeping the
    cross-reference target intact.
    """
    return stringify_annotation(annotation, "smart")


def is_pydantic_model(module: str, objname: str) -> bool:
    """Return True if ``module.objname`` should render via ``autopydantic_model``.

    Used from the autosummary class template to route models to the
    ``autopydantic_model`` directive (autodoc-pydantic does not auto-claim the
    plain ``.. autoclass::`` that autosummary emits).

    Routing is gated on the class docstring having no numpy ``Parameters``/
    ``Attributes`` section. autodoc-pydantic renders each field from ``Field``
    metadata, so a leftover section would double-document every field (napoleon
    plus the pydantic field). Models that still describe their fields only in
    such a section are therefore left on ``autoclass`` (descriptions intact)
    until those descriptions move onto ``Field(description=...)``.
    """
    try:
        import pydantic

        cls = getattr(importlib.import_module(module), objname)
    except Exception:
        return False
    if not (isinstance(cls, type) and issubclass(cls, pydantic.BaseModel)):
        return False
    doc = cls.__doc__
    has_section = bool(
        _parse_numpy_section(doc, "Parameters")
        or _parse_numpy_section(doc, "Attributes")
    )
    return not has_section


def _parse_numpy_section(docstring, section="Attributes"):
    """Return {name: description} from a named section of a numpy-style docstring."""
    if not docstring:
        return {}
    # ``cls.__doc__`` keeps the source indentation, but this parser treats only
    # non-indented lines as field names. Dedent so field lines start at column 0.
    docstring = inspect.cleandoc(docstring)
    lines = docstring.splitlines()
    result = {}
    i = 0
    while i < len(lines):
        if lines[i].strip() == section:
            i += 1
            if i < len(lines) and re.fullmatch(r"-+", lines[i].strip()):
                i += 1
                break
        else:
            i += 1
    else:
        return {}

    current_name = None
    current_desc = []

    def flush():
        if current_name and current_desc:
            result[current_name] = " ".join(current_desc).strip()

    while i < len(lines):
        line = lines[i]
        i += 1
        stripped = line.strip()
        if not stripped:
            continue
        if line == stripped:
            if i < len(lines) and re.fullmatch(r"-+", lines[i].strip()):
                break
            flush()
            current_desc = []
            m = re.match(r"^(\w+)", stripped)
            current_name = m.group(1) if m else None
        elif line[0] in (" ", "\t") and current_name is not None:
            current_desc.append(stripped)

    flush()
    return result


def _extract_fields(cls):
    """Return list of (name, type_str, description) for own fields of a dataclass or Pydantic model."""
    own = set(getattr(cls, "__annotations__", {}))

    if dataclasses.is_dataclass(cls):
        descriptions = _parse_numpy_section(cls.__doc__, "Attributes")
        return [
            (
                f.name,
                f.type if isinstance(f.type, str) else str(f.type),
                descriptions.get(f.name, ""),
            )
            for f in dataclasses.fields(cls)
            if f.name in own
        ]

    try:
        import pydantic
    except ImportError:
        return None

    if isinstance(cls, type) and issubclass(cls, pydantic.BaseModel):
        # Prefer FieldInfo.description (Field(description=...)), fall back to docstring
        # Parameters section, then Attributes section (models vary in which they use).
        param_descs = _parse_numpy_section(cls.__doc__, "Parameters")
        if not param_descs:
            param_descs = _parse_numpy_section(cls.__doc__, "Attributes")
        result = []
        for name, info in cls.model_fields.items():
            if name not in own:
                continue
            type_str = str(info.annotation)
            desc = (info.description or "").strip() or param_descs.get(name, "")
            result.append((name, type_str, desc))
        return result

    return None


class DataclassTableDirective(Directive):
    """Render fields of a dataclass or Pydantic model as a list-table."""

    required_arguments = 1
    optional_arguments = 0
    has_content = False
    option_spec = {}

    def run(self):
        class_path = self.arguments[0].strip()
        module_path, _, class_name = class_path.rpartition(".")

        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
        except (ImportError, AttributeError) as exc:
            error = self.state_machine.reporter.error(
                f"dataclass-table: cannot import {class_path!r}: {exc}",
                line=self.lineno,
            )
            return [error]

        fields = _extract_fields(cls)
        if fields is None:
            error = self.state_machine.reporter.error(
                f"dataclass-table: {class_path!r} is not a dataclass or Pydantic model",
                line=self.lineno,
            )
            return [error]

        rst_lines = [
            ".. list-table::",
            "   :widths: 20 25 55",
            "   :header-rows: 1",
            "",
            "   * - Field",
            "     - Type",
            "     - Description",
        ]
        for name, type_str, desc in fields:
            rst_lines += [
                f"   * - ``{name}``",
                f"     - ``{type_str}``",
                f"     - {desc}",
            ]

        vl = StringList(rst_lines, source="<dataclass-table>")
        node = nodes.section()
        node.document = self.state.document
        self.state.nested_parse(vl, self.content_offset, node)
        return node.children


def _build_clean_field_documenter():
    """Subclass autodoc-pydantic's field documenter to emit clean field types.

    Returns ``None`` if autodoc-pydantic is unavailable, so the extension stays
    usable without it.
    """
    try:
        from sphinxcontrib.autodoc_pydantic.directives.autodocumenters import (
            PydanticFieldDocumenter,
        )
    except ImportError:
        return None

    class CleanTypePydanticFieldDocumenter(PydanticFieldDocumenter):
        """Render field types from ``FieldInfo.annotation`` instead of source text.

        autodoc (via the module analyzer) emits ``:type:`` using the verbatim
        source annotation, which here includes ``Annotated[..., Field(...),
        PlainSerializer(...)]``. We rewrite that single line using the resolved
        pydantic annotation so jaxtyping shapes render cleanly.
        """

        def add_line(self, line, source, *lineno):
            stripped = line.lstrip()
            if stripped.startswith(":type:"):
                field_name = self.objpath[-1]
                fields = getattr(self.parent, "model_fields", {})
                info = fields.get(field_name)
                if info is not None:
                    indent = line[: len(line) - len(stripped)]
                    line = f"{indent}:type: {_clean_type(info.annotation)}"
            super().add_line(line, source, *lineno)

    return CleanTypePydanticFieldDocumenter


def setup(app):
    app.add_directive("dataclass-table", DataclassTableDirective)

    # Load autodoc-pydantic first so our field documenter can override its own.
    clean_field_documenter = None
    try:
        app.setup_extension("sphinxcontrib.autodoc_pydantic")
        clean_field_documenter = _build_clean_field_documenter()
    except Exception:
        clean_field_documenter = None
    if clean_field_documenter is not None:
        app.add_autodocumenter(clean_field_documenter, override=True)

    return {"version": "0.1", "parallel_read_safe": True}


def reset_torch(gallery_conf, fname):
    """Reset PyTorch's state between examples."""
    import numpy
    import torch

    # Clear CUDA memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    # Reset random seeds
    numpy.random.seed(42)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
