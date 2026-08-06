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
"""Recommended distributed runtime manager for nvalchemi workflows."""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from collections.abc import Iterator
from typing import Any

import torch
from physicsnemo.distributed import (
    DistributedManager,
    PhysicsNeMoUninitializedDistributedManagerWarning,
)
from torch import distributed as dist

logger = logging.getLogger(__name__)

__all__ = [
    "DistributedManager",
    "PhysicsNeMoUninitializedDistributedManagerWarning",
    "collective_device",
    "resolve_global_rank",
    "resolve_world_size",
]


def resolve_world_size() -> int:
    """Resolve world size from PhysicsNeMo, torch.distributed, or environment."""
    if DistributedManager.is_initialized():
        return int(DistributedManager().world_size)
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_world_size())
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return world_size


def resolve_global_rank(global_rank: int | None = None) -> int:
    """Resolve global rank from an explicit value, distributed state, or env."""
    if global_rank is not None:
        return int(global_rank)
    if DistributedManager.is_initialized():
        return int(DistributedManager().rank)
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank())
    rank = int(os.environ.get("RANK", 0))
    return rank


def collective_device(fallback: torch.device | str = "cpu") -> torch.device:
    """Resolve the rank-local device for distributed tensor collectives."""
    if dist.is_available() and dist.is_initialized():
        try:
            backend = dist.get_backend()
        except RuntimeError:
            backend = None
        if backend != "nccl":
            return torch.device("cpu")
    if DistributedManager.is_initialized():
        device = torch.device(DistributedManager().device)
    elif torch.cuda.is_available():
        index = int(os.environ.get("LOCAL_RANK", 0))
        device = torch.device("cuda", index)
    else:
        device = torch.device(fallback)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


# Full-precision fp32 lands far below this; reduced precision far above.
_REDUCED_PRECISION_THRESHOLD = 1e-5
_warned_reduced_precision = False


def pin_fp32() -> None:
    """Force full-precision fp32 matmul and convolution.

    A distributed forward pads to different shapes than a single-process one, so
    under reduced-precision fp32 (TF32) the backend can pick a different kernel
    for each and the results separate by far more than fp32 rounding. Also sets
    ``NVIDIA_TF32_OVERRIDE``, which is what reaches ``mp.spawn`` / ``torchrun``
    workers — they inherit the environment, not the torch flags. Call before the
    process builds a CUDA context.

    Returns
    -------
    None
    """
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    for holder in (torch.backends.cuda.matmul, torch.backends.cudnn):
        if hasattr(holder, "fp32_precision"):
            try:
                holder.fp32_precision = "ieee"
            except Exception:  # pragma: no cover - varies by torch version
                # Non-fatal: the primary flags above already pin precision, and
                # this attribute only exists on some torch versions.
                logger.warning("could not set fp32_precision", exc_info=True)


@contextlib.contextmanager
def pinned_fp32() -> Iterator[None]:
    """Pin full-precision fp32 for the duration of the block, then restore.

    The scoped counterpart to :func:`pin_fp32`, for callers that need one
    comparison at full precision without changing the rest of the process.
    Prefer :func:`pin_fp32` for a whole run: it also sets the environment
    variable that ``mp.spawn`` / ``torchrun`` workers inherit, and restoring
    that on exit would unpin the workers.

    Yields
    ------
    None
    """
    saved: list[tuple[Any, str, Any]] = [
        (
            torch.backends.cuda.matmul,
            "allow_tf32",
            torch.backends.cuda.matmul.allow_tf32,
        ),
        (torch.backends.cudnn, "allow_tf32", torch.backends.cudnn.allow_tf32),
    ]
    saved.extend(
        (holder, "fp32_precision", holder.fp32_precision)
        for holder in (torch.backends.cuda.matmul, torch.backends.cudnn)
        if hasattr(holder, "fp32_precision")
    )
    # Reading the global precision raises once legacy (``allow_tf32``) and new
    # (``fp32_precision``) APIs have both been written, which ``pin_fp32`` does.
    try:
        saved_precision = torch.get_float32_matmul_precision()
    except RuntimeError:  # pragma: no cover - depends on prior calls
        saved_precision = None
    try:
        pin_fp32()
        yield
    finally:
        for holder, attr, value in saved:
            try:
                setattr(holder, attr, value)
            except Exception:  # pragma: no cover - varies by torch version
                logger.warning("could not restore %s", attr, exc_info=True)
        if saved_precision is not None:
            torch.set_float32_matmul_precision(saved_precision)


def _is_reduced_precision(device: str | torch.device | None = None) -> bool:
    """Whether fp32 matmul currently runs on the reduced-precision path.

    Measured rather than read off the backend flags: which kernel runs depends on
    torch version, backend and shape.
    """
    if not torch.cuda.is_available():
        return False
    try:
        gen = torch.Generator(device="cpu").manual_seed(0)
        a = torch.randn(512, 512, generator=gen).to(device or "cuda")
        b = torch.randn(512, 512, generator=gen).to(device or "cuda")
        ref = a.double() @ b.double()
        err = (((a @ b).double() - ref).abs().max() / ref.abs().max()).item()
    except Exception:  # pragma: no cover - a probe must not break a forward
        logger.debug("fp32 precision probe failed", exc_info=True)
        return False
    return err > _REDUCED_PRECISION_THRESHOLD


def warn_if_reduced_precision(
    device: str | torch.device | None = None,
) -> None:
    """Warn once per process if reduced-precision fp32 is in force."""
    global _warned_reduced_precision
    if _warned_reduced_precision or not torch.cuda.is_available():
        return
    _warned_reduced_precision = True
    if not _is_reduced_precision(device):
        return
    warnings.warn(
        "Reduced-precision fp32 (TF32) is enabled; distributed and "
        "single-process results can then differ by much more than fp32 rounding. "
        "Call nvalchemi.distributed.pin_fp32() before building models if this "
        "run must match a reference.",
        UserWarning,
        stacklevel=3,
    )
