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
"""Checkpoint save and load utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from nvalchemi._imports import OptionalDependencyFailure, check_optional_dependencies

try:
    from physicsnemo.utils.checkpoint import (
        load_checkpoint as _load_checkpoint,
    )
    from physicsnemo.utils.checkpoint import (
        save_checkpoint as _save_checkpoint,
    )
except ImportError:
    _save_checkpoint = None  # type: ignore[assignment]
    _load_checkpoint = None  # type: ignore[assignment]
    OptionalDependencyFailure("training")


@check_optional_dependencies()
def save_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer] | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int = 0,
    metrics: dict[str, Any] | None = None,
    dist_manager: Any | None = None,
) -> None:
    """Save a training checkpoint via ``physicsnemo.utils.checkpoint``.

    Only rank-0 (or single-process) actually writes to disk.

    Parameters
    ----------
    path : str or Path
        Directory where checkpoint files are stored.
    model : nn.Module
        Model whose state dict is saved.
    optimizer : Optimizer, list[Optimizer], or None
        Optimizer(s) whose state is saved.
    scheduler : Any or None
        LR scheduler(s) whose state is saved.
    scaler : torch.amp.GradScaler or None
        Gradient scaler state (for mixed-precision).
    epoch : int
        Current epoch index.
    metrics : dict[str, Any] or None
        Training metrics to persist alongside the checkpoint.
    dist_manager : Any or None
        ``physicsnemo.distributed.DistributedManager`` instance, or ``None``.
    """
    rank: int = getattr(dist_manager, "rank", 0) if dist_manager is not None else 0
    if rank != 0:
        return

    metadata = {"metrics": metrics or {}}

    # physicsnemo accepts a single optimizer/scheduler; when the trainer
    # passes a list we unwrap single-element lists and stash multi-optimizer
    # state dicts in the metadata so they survive the round-trip.
    opt_arg: torch.optim.Optimizer | None = None
    sched_arg: Any | None = None
    if isinstance(optimizer, list):
        if len(optimizer) == 1:
            opt_arg = optimizer[0]
        else:
            metadata["extra_optimizers"] = [o.state_dict() for o in optimizer]
    else:
        opt_arg = optimizer

    if isinstance(scheduler, list):
        if len(scheduler) == 1:
            sched_arg = scheduler[0]
        else:
            metadata["extra_schedulers"] = [s.state_dict() for s in scheduler]
    else:
        sched_arg = scheduler

    _save_checkpoint(
        path=str(path),
        models=model,
        optimizer=opt_arg,
        scheduler=sched_arg,
        scaler=scaler,
        epoch=epoch,
        metadata=metadata,
    )


@check_optional_dependencies()
def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | list[torch.optim.Optimizer] | None = None,
    scheduler: Any | None = None,
    scaler: torch.amp.GradScaler | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load a training checkpoint via ``physicsnemo.utils.checkpoint``.

    If the checkpoint *path* does not exist, returns a clean-start metadata
    dict (``epoch=0``, empty metrics) without raising an error.

    Parameters
    ----------
    path : str or Path
        Directory containing checkpoint files.
    model : nn.Module
        Model to load state into.
    optimizer : Optimizer, list[Optimizer], or None
        Optimizer(s) to restore.
    scheduler : Any or None
        LR scheduler(s) to restore.
    scaler : torch.amp.GradScaler or None
        Gradient scaler to restore.
    device : str or torch.device
        Device to map the loaded tensors to.

    Returns
    -------
    dict[str, Any]
        Metadata dictionary with at least ``"epoch"`` and ``"metrics"`` keys.
    """
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        return {"epoch": 0, "metrics": {}}

    # Unwrap single-element lists; multi-optimizer state is restored from
    # metadata after the main checkpoint load.
    opt_arg: torch.optim.Optimizer | None = None
    sched_arg: Any | None = None
    multi_opt = False
    multi_sched = False

    if isinstance(optimizer, list):
        if len(optimizer) == 1:
            opt_arg = optimizer[0]
        else:
            multi_opt = True
    else:
        opt_arg = optimizer

    if isinstance(scheduler, list):
        if len(scheduler) == 1:
            sched_arg = scheduler[0]
        else:
            multi_sched = True
    else:
        sched_arg = scheduler

    metadata: dict[str, Any] = {}
    epoch = _load_checkpoint(
        path=str(path),
        models=model,
        optimizer=opt_arg,
        scheduler=sched_arg,
        scaler=scaler,
        metadata_dict=metadata,
        device=str(device),
    )

    # Restore multi-optimizer / multi-scheduler state from metadata.
    if multi_opt and isinstance(optimizer, list):
        extra = metadata.get("extra_optimizers", [])
        for opt, sd in zip(optimizer, extra):
            opt.load_state_dict(sd)

    if multi_sched and isinstance(scheduler, list):
        extra = metadata.get("extra_schedulers", [])
        for sched, sd in zip(scheduler, extra):
            sched.load_state_dict(sd)

    return {
        "epoch": epoch,
        "metrics": metadata.get("metrics", {}),
    }
