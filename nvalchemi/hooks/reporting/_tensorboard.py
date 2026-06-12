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
"""TensorBoard reporting sink."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Protocol

from torch import distributed as dist

from nvalchemi._optional import OptionalDependency
from nvalchemi.hooks._context import HookContext
from nvalchemi.hooks.reporting._distributed import (
    normalize_rank_reduction,
    reduce_scalar_snapshot,
)
from nvalchemi.hooks.reporting._scalars import ScalarCallback, collect_scalars
from nvalchemi.hooks.reporting._state import ReportingState


class TensorBoardWriter(Protocol):
    """Subset of ``SummaryWriter`` used by :class:`TensorBoardReporter`."""

    def add_scalar(
        self,
        tag: str,
        scalar_value: float,
        global_step: int | None = None,
    ) -> None:
        """Write one scalar event.

        Parameters
        ----------
        tag : str
            TensorBoard scalar tag.
        scalar_value : float
            Scalar value to write.
        global_step : int | None, optional
            Step associated with the scalar.
        """
        ...

    def flush(self) -> None:
        """Flush pending TensorBoard events."""
        ...

    def close(self) -> None:
        """Close the writer."""
        ...


@OptionalDependency.TENSORBOARD.require
class TensorBoardReporter:
    """Write scalar reporting snapshots to TensorBoard.

    Parameters
    ----------
    log_dir : str | Path
        TensorBoard log directory.
    custom_scalars : Mapping[str, ScalarCallback] | None, optional
        Additional scalar callbacks passed to :func:`collect_scalars`.
    include_losses : bool, default True
        When ``True``, include loss scalars from the hook context.
    include_optimizer_lrs : bool, default True
        When ``True``, include optimizer learning rates from the hook context.
    rank_reduction : torch.distributed.ReduceOp | {"none", "mean", "sum", "min", "max"} | None, default None
        Optional distributed reduction applied to scalars before writing. String
        values are normalized to :class:`torch.distributed.ReduceOp`. Reduction
        requires every rank to call this reporter; only rank zero writes the
        reduced snapshot.
    tag_prefix : str | None, optional
        Optional prefix prepended to every TensorBoard tag.
    flush : bool, default True
        Flush the writer after every report event.
    rank_zero_only : bool, default True
        Request rank-zero-only dispatch from :class:`ReportingOrchestrator`.
        When ``False`` and ``rank_reduction="none"``, ``log_dir`` must contain
        ``"{rank}"`` or ``"{global_rank}"`` so every rank writes its own event
        directory.
    writer : TensorBoardWriter | None, optional
        Preconstructed writer. This is mainly useful for tests or integrations
        that own writer construction.
    """

    def __init__(
        self,
        log_dir: str | Path,
        *,
        custom_scalars: Mapping[str, ScalarCallback] | None = None,
        include_losses: bool = True,
        include_optimizer_lrs: bool = True,
        rank_reduction: dist.ReduceOp | str | None = None,
        tag_prefix: str | None = None,
        flush: bool = True,
        rank_zero_only: bool = True,
        writer: TensorBoardWriter | None = None,
    ) -> None:
        self.rank_reduction = rank_reduction
        self._rank_reduction_op, _ = normalize_rank_reduction(rank_reduction)
        self.log_dir = Path(log_dir)
        self.custom_scalars = custom_scalars
        self.include_losses = include_losses
        self.include_optimizer_lrs = include_optimizer_lrs
        self.tag_prefix = tag_prefix.strip("/") if tag_prefix is not None else None
        self.flush = flush
        self._write_rank_zero_only = (
            rank_zero_only or self._rank_reduction_op is not None
        )
        self.rank_zero_only = rank_zero_only and self._rank_reduction_op is None
        self.requires_all_ranks = self._rank_reduction_op is not None
        self._writer = writer
        self._external_writer = writer is not None
        self._open_log_dir: Path | None = None
        if not self._write_rank_zero_only and not self._has_rank_token:
            raise ValueError(
                "TensorBoardReporter log_dir must contain '{rank}' or "
                "'{global_rank}' when rank_zero_only=False and "
                "rank_reduction='none'."
            )

    def __enter__(self) -> TensorBoardReporter:
        """Return this reporter; writers are opened lazily on first write."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the TensorBoard writer."""
        self.close()

    def close(self) -> None:
        """Close the writer if it is open."""
        if self._writer is None:
            return
        self._writer.close()
        self._writer = None
        self._open_log_dir = None

    def report(self, ctx: HookContext, stage: Enum, state: ReportingState) -> None:
        """Write one scalar snapshot to TensorBoard.

        Parameters
        ----------
        ctx : HookContext
            Workflow hook context.
        stage : Enum
            Hook stage being reported.
        state : ReportingState
            Shared reporting state from the orchestrator.
        """
        snapshot = collect_scalars(
            ctx,
            stage,
            state,
            custom_scalars=self.custom_scalars,
            include_losses=self.include_losses,
            include_optimizer_lrs=self.include_optimizer_lrs,
        )
        if self._rank_reduction_op is not None:
            snapshot = reduce_scalar_snapshot(
                snapshot,
                self.rank_reduction,
                reporter_name=type(self).__name__,
            )
            if not self._is_rank_zero(ctx):
                return
        elif self._write_rank_zero_only and not self._is_rank_zero(ctx):
            return

        writer = self._open(self._resolve_log_dir(ctx.global_rank))
        step = snapshot.step_count if snapshot.step_count is not None else None
        if step is None:
            step = snapshot.event_count
        for key, value in sorted(snapshot.scalars.items()):
            writer.add_scalar(self._tag(key), value, global_step=step)
        if self.flush:
            writer.flush()

    @property
    def _has_rank_token(self) -> bool:
        path = str(self.log_dir)
        return "{rank}" in path or "{global_rank}" in path

    def _open(self, log_dir: Path) -> TensorBoardWriter:
        if self._writer is not None and self._external_writer:
            return self._writer
        if self._writer is not None and self._open_log_dir == log_dir:
            return self._writer
        if self._writer is not None:
            self.close()
        from torch.utils.tensorboard import SummaryWriter

        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._open_log_dir = log_dir
        return self._writer

    def _resolve_log_dir(self, global_rank: int) -> Path:
        path = str(self.log_dir)
        path = path.replace("{global_rank}", str(global_rank))
        path = path.replace("{rank}", str(global_rank))
        return Path(path)

    def _tag(self, key: str) -> str:
        return key if self.tag_prefix is None else f"{self.tag_prefix}/{key}"

    def _is_rank_zero(self, ctx: HookContext) -> bool:
        return ctx.global_rank == 0
