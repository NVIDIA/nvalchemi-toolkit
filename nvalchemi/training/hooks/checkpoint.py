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
"""Periodic checkpoint-saving training hook."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import TracebackType
from typing import Annotated, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from nvalchemi.hooks._context import TrainContext
from nvalchemi.training._checkpoint import (
    _create_checkpoint_snapshot,
    _write_checkpoint_snapshot,
)
from nvalchemi.training._stages import TrainingStage

__all__ = ["CheckpointHook"]


class CheckpointHook(BaseModel):
    """Periodically save restartable training strategy checkpoints.

    The hook observes completed training counters and saves
    :class:`~nvalchemi.training.strategy.TrainingStrategy` checkpoints through
    the same manifest layout as :func:`nvalchemi.training.save_checkpoint`.
    It fires either every ``step_interval`` completed optimizer steps or every
    ``epoch_interval`` completed epochs. The two cadences are mutually
    exclusive so each hook owns one clear checkpoint policy.

    With ``async_save=True`` (default), the hook first captures an immutable
    CPU snapshot of model, optimizer, scheduler, and strategy metadata on the
    training thread, then writes that snapshot on a single background thread.
    This avoids racing against live training tensors while still moving the
    filesystem work off the critical path. If a later checkpoint is due while
    the previous background write is still running, the hook waits for the
    previous write before capturing the next snapshot so manifest indices stay
    ordered.

    Parameters
    ----------
    checkpoint_dir : Path | str
        Directory where checkpoint manifests and component state files are
        written.
    step_interval : int | None, optional
        Save every N completed optimizer steps. Skipped optimizer steps do not
        advance this cadence. Exactly one of ``step_interval`` or
        ``epoch_interval`` must be provided.
    epoch_interval : int | None, optional
        Save every N completed epochs. Exactly one of ``step_interval`` or
        ``epoch_interval`` must be provided.
    async_save : bool, optional
        If ``True``, write captured snapshots on a background thread. If
        ``False``, write synchronously during hook dispatch. Default ``True``.
    rank_zero_only : bool, optional
        If ``True``, only distributed rank 0 writes checkpoints. Default
        ``True``.

    Attributes
    ----------
    last_checkpoint_index : int | None
        Most recent checkpoint index known to have been written. In async mode,
        this updates when the background future completes.

    Raises
    ------
    ValueError
        If neither interval is provided, or an interval is not positive.
    RuntimeError
        If the hook is called without a strategy workflow in ``TrainContext``.

    Examples
    --------
    >>> from nvalchemi.training import CheckpointHook, TrainingStrategy
    >>> hook = CheckpointHook("runs/example/checkpoints", step_interval=1000)
    >>> strategy = TrainingStrategy(..., hooks=[hook])  # doctest: +SKIP
    >>> strategy.run(train_loader)  # doctest: +SKIP
    """

    checkpoint_dir: Annotated[
        Path,
        Field(description="Root directory for restartable training checkpoints."),
    ]
    step_interval: Annotated[
        int | None,
        Field(default=None, gt=0, description="Completed-step save interval."),
    ] = None
    epoch_interval: Annotated[
        int | None,
        Field(default=None, gt=0, description="Completed-epoch save interval."),
    ] = None
    async_save: Annotated[
        bool,
        Field(description="Write checkpoint snapshots on a background thread."),
    ] = True
    rank_zero_only: Annotated[
        bool,
        Field(description="Restrict checkpoint writes to distributed rank 0."),
    ] = True
    last_checkpoint_index: Annotated[
        int | None,
        Field(default=None, ge=0, exclude=True),
    ] = None

    frequency: ClassVar[int] = 1
    stage: ClassVar[TrainingStage | None] = None

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    _executor: ThreadPoolExecutor | None = PrivateAttr(default=None)
    _future: Future[int] | None = PrivateAttr(default=None)

    def __init__(
        self, checkpoint_dir: Path | str | None = None, **data: object
    ) -> None:
        """Initialize the hook, accepting ``checkpoint_dir`` positionally."""
        if checkpoint_dir is not None:
            if "checkpoint_dir" in data:
                raise TypeError(
                    "CheckpointHook got checkpoint_dir both positionally and "
                    "as a keyword argument."
                )
            data["checkpoint_dir"] = checkpoint_dir
        super().__init__(**data)

    @model_validator(mode="after")
    def _validate_cadence(self) -> CheckpointHook:
        """Require exactly one save cadence."""
        if (self.epoch_interval is None) == (self.step_interval is None):
            raise ValueError(
                "CheckpointHook requires exactly one of step_interval or "
                "epoch_interval."
            )
        return self

    def _runs_on_stage(self, stage: TrainingStage) -> bool:
        """Return whether this hook observes a training stage."""
        return (
            self.step_interval is not None and stage is TrainingStage.AFTER_BATCH
        ) or (self.epoch_interval is not None and stage is TrainingStage.AFTER_EPOCH)

    def __enter__(self) -> CheckpointHook:
        """Create the background writer when async checkpointing is enabled."""
        if self.async_save and self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="nvalchemi-checkpoint",
            )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Flush any pending checkpoint write before leaving training."""
        del exc, tb
        try:
            self.close()
        except Exception:
            if exc_type is None:
                raise

    def close(self) -> None:
        """Wait for pending async writes and close the background writer."""
        try:
            self._finish_pending(block=True)
        finally:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None

    def _finish_pending(self, *, block: bool) -> None:
        """Collect a pending async result, optionally waiting for it."""
        if self._future is None:
            return
        if not block and not self._future.done():
            return
        self.last_checkpoint_index = self._future.result()
        self._future = None

    def _should_save(self, ctx: TrainContext, stage: TrainingStage) -> bool:
        """Return whether ``ctx`` reaches the configured save cadence."""
        if self.rank_zero_only and ctx.global_rank != 0:
            return False
        if (
            stage is TrainingStage.AFTER_BATCH
            and self.step_interval is not None
            and ctx.step_count > 0
        ):
            return ctx.step_count % self.step_interval == 0
        if (
            stage is TrainingStage.AFTER_EPOCH
            and self.epoch_interval is not None
            and ctx.epoch > 0
        ):
            return ctx.epoch % self.epoch_interval == 0
        return False

    def _save_checkpoint(self, ctx: TrainContext) -> None:
        """Capture and write one strategy checkpoint."""
        if ctx.workflow is None:
            raise RuntimeError(
                "CheckpointHook requires TrainContext.workflow to reference "
                "the active TrainingStrategy."
            )
        self._finish_pending(block=False)
        if self._future is not None:
            self._finish_pending(block=True)

        snapshot = _create_checkpoint_snapshot(
            self.checkpoint_dir,
            strategy=ctx.workflow,
        )
        if not self.async_save:
            self.last_checkpoint_index = _write_checkpoint_snapshot(
                self.checkpoint_dir,
                snapshot,
            )
            return

        if self._executor is None:
            raise RuntimeError(
                "CheckpointHook async writer is not initialized. Run it through "
                "TrainingStrategy so hook contexts are entered, or call "
                "__enter__() before invoking the hook directly."
            )
        self._future = self._executor.submit(
            _write_checkpoint_snapshot,
            self.checkpoint_dir,
            snapshot,
        )

    def __call__(self, ctx: TrainContext, stage: TrainingStage) -> None:
        """Save a checkpoint when the configured cadence is reached."""
        if self._should_save(ctx, stage):
            self._save_checkpoint(ctx)
