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
import gc
import pathlib
import warnings

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


_MODULE_USES_COMPILE: dict[str, bool] = {}


def _module_uses_cudagraphs(module) -> bool:
    """``True`` iff a test module compiles in a way that builds a cudagraph pool.

    Only the ``cudagraphs`` backend and ``mode="reduce-overhead"`` route through
    :mod:`torch._inductor.cudagraph_trees`; the ``eager`` / ``aot_eager`` /
    default-inductor compiles used elsewhere in the suite never allocate from
    the cudagraph pool and so cannot leak into it. Cached per file: the source
    is read once, not once per test."""
    path = getattr(module, "__file__", None)
    if path is None:
        return False
    cached = _MODULE_USES_COMPILE.get(path)
    if cached is None:
        try:
            src = pathlib.Path(path).read_text()
        except OSError:  # pragma: no cover — unreadable test module
            src = ""
        cached = "cudagraphs" in src or "reduce-overhead" in src
        _MODULE_USES_COMPILE[path] = cached
    return cached


@pytest.fixture(autouse=True)
def _isolate_torch_compile(request):
    """Reset ``torch.compile`` / CUDA-graph state around compiling tests.

    Tests that build ``torch.compile(fn, backend="cudagraphs")`` callables on
    CUDA share one global cudagraph *memory pool*, and the tree manager retains
    each graph's output tensors across invocations. A tensor left live in that
    pool by an earlier test makes the next graph capture fail
    ``check_memory_pool``'s "Detected N tensor(s) in the cudagraph pool not
    tracked as outputs" correctness check — a cross-test leak, not a defect in
    the test that reports it. The leak crosses directories (``test/distributed``
    collects before ``test/dynamics``), so keying off a class or test name is
    not enough; every module that can build a cudagraph pool is bracketed.

    Resetting dynamo and freeing the pool before and after each such test makes
    the suite hermetic under any collection order or :mod:`pytest-testmon`
    subset. Only modules that actually compile pay the cost."""
    cls = getattr(request, "cls", None)
    is_compile = _module_uses_cudagraphs(request.module) and (
        (cls is not None and cls.__name__.endswith("Compile"))
        or "compile" in request.node.name
    )

    def _reset() -> None:
        # Order matters: drain in-flight work and drop unreferenced tensors
        # *before* tearing down the cudagraph trees, otherwise a dangling
        # reference still pins the pool and the teardown cannot free it.
        if torch.cuda.is_available():
            with contextlib.suppress(Exception):
                torch.cuda.synchronize()
        gc.collect()
        reset = getattr(torch.compiler, "reset", None)
        if reset is not None:
            try:
                reset()
            except Exception as exc:  # pragma: no cover — diagnostic path
                # A failed reset is exactly what leaves a live tensor in the
                # cudagraph pool for the next capture; surface it rather than
                # letting the next test report the confusing downstream error.
                warnings.warn(
                    f"cudagraph/dynamo reset failed for {request.node.nodeid}: {exc!r}",
                    RuntimeWarning,
                    stacklevel=1,
                )
        gc.collect()

    if is_compile:
        _reset()
    yield
    if is_compile:
        _reset()


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
