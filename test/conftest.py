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
from __future__ import annotations

import contextlib

import pytest
import torch
import torch.distributed as dist


def _cueq_ops_registered() -> bool:
    """``True`` iff the cuequivariance fused-tensor-product torch ops are
    registered. Importing ``cuequivariance``/``cuequivariance_torch`` is not
    enough — the ``torch.ops.cuequivariance`` namespace is populated only when a
    compatible build registers its custom ops, and version skews (e.g. 0.10 vs
    the 0.8-era op names) leave it empty."""
    try:
        import cuequivariance_torch  # noqa: F401  (side effect: op registration)

        return hasattr(torch.ops.cuequivariance, "fused_tensor_product")
    except Exception:
        return False


def _fairchem_installed() -> bool:
    """``True`` iff ``fairchem.core`` (the UMA backbone) is importable."""
    import importlib.util

    try:
        return importlib.util.find_spec("fairchem.core") is not None
    except ModuleNotFoundError:
        # ``find_spec`` on a dotted name imports the parent package to read its
        # ``__path__``; when ``fairchem`` itself is absent (the cu13/mace env
        # that has no UMA stack) that import raises rather than returning None.
        return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip environment-gated tests so they skip (not fail) where a
    capability is absent:

    * ``@pytest.mark.multigpu`` — needs >=2 CUDA GPUs (override with
      ``NVALCHEMI_FORCE_MULTIGPU=1`` to see the underlying error).
    * ``@pytest.mark.requires_cueq`` — needs the ``cuequivariance`` torch ops
      *registered* (not merely the package installed).
    * ``@pytest.mark.requires_uma`` — needs ``fairchem-core`` installed.
    """
    import os

    force_multigpu = os.environ.get("NVALCHEMI_FORCE_MULTIGPU") == "1"
    have_multigpu = torch.cuda.is_available() and torch.cuda.device_count() >= 2
    skip_multigpu = (
        None
        if (force_multigpu or have_multigpu)
        else pytest.mark.skip(reason="requires >=2 CUDA GPUs (mark: multigpu)")
    )
    skip_cueq = (
        None
        if _cueq_ops_registered()
        else pytest.mark.skip(reason="cuequivariance torch ops not registered")
    )
    skip_uma = (
        None
        if _fairchem_installed()
        else pytest.mark.skip(reason="fairchem-core not installed")
    )
    for item in items:
        if skip_multigpu is not None and "multigpu" in item.keywords:
            item.add_marker(skip_multigpu)
        if skip_cueq is not None and "requires_cueq" in item.keywords:
            item.add_marker(skip_cueq)
        if skip_uma is not None and "requires_uma" in item.keywords:
            item.add_marker(skip_uma)


@pytest.fixture(autouse=True)
def _dist_leak_guard():
    """Tear down a process group that a test leaves initialized.

    A test that calls ``init_process_group`` in the main process and fails
    before its own teardown (or a fixture that leaks one) otherwise poisons
    every later test that checks ``dist.is_initialized()`` — e.g. the
    pipeline-composition guards and rank-resolution helpers. Only groups this
    test newly initialized are destroyed; a group already up at test start
    (an outer-scope fixture) is left for its owner to tear down."""
    was_initialized = dist.is_available() and dist.is_initialized()
    yield
    if dist.is_available() and dist.is_initialized() and not was_initialized:
        with contextlib.suppress(Exception):
            dist.destroy_process_group()

from nvalchemi.data import atomic_data


@pytest.fixture(autouse=True)
def _reset_fp_cast_warned() -> None:
    """Clear the process-global fp-cast warning cache before each test.

    ``check_fp_dtype_consistency`` dedups its cast warning per ``(field, dtype)``
    across all constructions, so without this reset warning assertions would
    depend on test order (an earlier test could consume the only warning for a
    given field/dtype)."""
    atomic_data._FP_CAST_WARNED.clear()


@pytest.fixture(params=["cpu", "cuda"])
def device(request) -> str:
    """Return either CPU or GPU device; skips GPU if torch.cuda is unavailable."""
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("No CUDA device available.")
    return request.param


@pytest.fixture(params=["cuda"])
def gpu_device(request) -> str:
    """Used to skip GPU specific tests if device is not available."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA device available for GPU test.")
    return request.param


@pytest.fixture
def fixed_torch_seed() -> None:
    """Set a fixed PyTorch RNG seed for tests that compare random tensors."""
    torch.manual_seed(0)
