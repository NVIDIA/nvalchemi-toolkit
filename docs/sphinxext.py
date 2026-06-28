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
import re

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.statemachine import StringList


def _parse_numpy_section(docstring, section="Attributes"):
    """Return {name: description} from a named section of a numpy-style docstring."""
    if not docstring:
        return {}
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


def setup(app):
    app.add_directive("dataclass-table", DataclassTableDirective)
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
