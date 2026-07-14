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
"""CPU unit tests for the scripted-op marshalling pre-flight + error
translator (``validate.scripted_diagnostics`` + the IMA worker-error
translator). No GPU / no MACE — a tiny scripted toy exercises the static
detection, declared-adapter cross-reference, exclude, auto-fix injection, and
the IMA error-string translator."""

from __future__ import annotations

import dataclasses

import torch
from torch import nn

from nvalchemi.distributed._core.adapter import JitAdapter
from nvalchemi.distributed.spec import SPEC_MPNN_HALO
from nvalchemi.distributed.validate import _detect_scripted_op_shardtensor_ima
from nvalchemi.distributed.validate.scripted_diagnostics import (
    apply_marshal_adapters,
    detect_scripted_ops,
)

_THIS_MODULE = __name__


@torch.jit.script
def _toy_scripted_fn(x: torch.Tensor) -> torch.Tensor:
    return x * x


class _ScriptedSub(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 1.0


class _ModelWithScripts(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.scripted_sub = torch.jit.script(_ScriptedSub())


def _spec_with_adapter(module_path: str, attr: str):
    d = SPEC_MPNN_HALO.distribution
    return dataclasses.replace(
        SPEC_MPNN_HALO,
        distribution=dataclasses.replace(
            d,
            third_party_helpers=d.third_party_helpers
            + (JitAdapter(module_path, attr, mode="marshal"),),
        ),
    )


def test_detect_finds_module_level_scripted_function() -> None:
    report = detect_scripted_ops(_ModelWithScripts(), SPEC_MPNN_HALO)
    # The ScriptModule submodule is found (auto-covered).
    assert any("scripted_sub" in n for n in report.scripted_submodules)
    # The module-level scripted function is found AND undeclared (the IMA risk).
    assert (_THIS_MODULE, "_toy_scripted_fn") in report.module_level_functions
    assert (_THIS_MODULE, "_toy_scripted_fn") in report.undeclared_functions
    assert report.has_risk


def test_detect_respects_declared_adapter() -> None:
    spec = _spec_with_adapter(_THIS_MODULE, "_toy_scripted_fn")
    report = detect_scripted_ops(_ModelWithScripts(), spec)
    assert (_THIS_MODULE, "_toy_scripted_fn") in report.declared_functions
    assert (_THIS_MODULE, "_toy_scripted_fn") not in report.undeclared_functions
    assert not report.has_risk


def test_detect_respects_exclude() -> None:
    report = detect_scripted_ops(
        _ModelWithScripts(), SPEC_MPNN_HALO, exclude=("_toy_scripted_fn",)
    )
    assert (_THIS_MODULE, "_toy_scripted_fn") not in report.undeclared_functions
    assert not report.has_risk


def test_apply_marshal_adapters_adds_jitadapter() -> None:
    report = detect_scripted_ops(_ModelWithScripts(), SPEC_MPNN_HALO)
    fixed = apply_marshal_adapters(SPEC_MPNN_HALO, report.undeclared_functions)
    targets = {
        (h.module_path, h.attr_name, h.mode)
        for h in fixed.distribution.third_party_helpers
        if isinstance(h, JitAdapter)
    }
    assert (_THIS_MODULE, "_toy_scripted_fn", "marshal") in targets
    # Idempotent: re-applying doesn't double-add.
    refixed = apply_marshal_adapters(fixed, report.undeclared_functions)
    assert detect_scripted_ops(_ModelWithScripts(), refixed).undeclared_functions == []


def test_format_hint_includes_paste_able_delta() -> None:
    report = detect_scripted_ops(_ModelWithScripts(), SPEC_MPNN_HALO)
    hint = report.format_hint()
    assert "JitAdapter" in hint
    assert "_toy_scripted_fn" in hint
    assert 'mode="marshal"' in hint
    # No risk -> empty hint.
    clean = detect_scripted_ops(
        _ModelWithScripts(), _spec_with_adapter(_THIS_MODULE, "_toy_scripted_fn")
    )
    assert clean.format_hint() == ""


def test_ima_translator_fires_on_scripted_signature() -> None:
    ima = (
        "RuntimeError: The following operation failed in the TorchScript "
        "interpreter.\nRuntimeError: CUDA driver error: an illegal memory "
        "access was encountered"
    )
    hint = _detect_scripted_op_shardtensor_ima(ima, {})
    assert hint is not None
    assert "marshal" in hint.lower()
    assert "JitAdapter" in hint

    warp_ima = (
        "Warp CUDA error 700: an illegal memory access was encountered "
        "(in function wp_free_device_async, warp.cu:816)"
    )
    assert _detect_scripted_op_shardtensor_ima(warp_ima, {}) is not None


def test_ima_translator_quiet_on_unrelated_errors() -> None:
    # Plain OOM (no scripted context) must NOT be mislabeled.
    assert _detect_scripted_op_shardtensor_ima("CUDA out of memory", {}) is None
    # An illegal access with no scripted/Warp context is left to other
    # translators.
    assert (
        _detect_scripted_op_shardtensor_ima(
            "an illegal memory access in a custom kernel", {}
        )
        is None
    )
