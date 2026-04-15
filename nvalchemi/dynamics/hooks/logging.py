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
"""
Logging hook for recording per-sample simulation observables.

Provides :class:`LoggingHook`, which computes and logs per-graph scalar
statistics (energy, temperatures, max forces, etc.) at a configurable
frequency.  Each graph in the batch is written as an individual row,
together with the current step and status (stage) information.

I/O is offloaded to a background thread via :class:`ThreadPoolExecutor`
so that file writes do not block the simulation loop.  When the batch
resides on CUDA, scalar computation and the D2H transfer run on a
dedicated side stream (created in :meth:`__enter__`) so that they
overlap with ongoing compute on the default stream.  The transfer uses
``non_blocking=True``; the worker thread calls
``stream.synchronize()`` before reading the resulting CPU tensor.

The hook implements the context manager protocol::

    with LoggingHook(backend="csv", log_path="log.csv") as hook:
        dynamics.register_hook(hook)
        dynamics.run(batch)
    # all pending writes flushed, files closed
"""

from __future__ import annotations

import contextlib
import csv
import io
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from typing import TYPE_CHECKING, Literal, get_args

import torch
from tensordict import TensorDict

from nvalchemi.data import Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks._utils import (
    scatter_reduce_per_graph,
    temperature_per_graph,
)
from nvalchemi.hooks._context import HookContext

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Boltzmann constant in eV/K — consistent with typical atomistic MD unit
# systems (positions in Å, masses in amu, velocities in Å/fs, energy in eV).
_KB_EV_PER_K: float = 8.617333262e-5

__all__ = ["LoggingHook"]

LogBackend = Literal["csv", "tensorboard", "custom"]


class LoggingHook:
    """Log per-sample scalar observables from the simulation.

    At each firing step, this hook computes per-graph scalars from the
    :class:`~nvalchemi.data.Batch` and writes **one row per graph** to the
    configured logging backend.  Each row includes:

    * **step** — the current ``dynamics.step_count``.
    * **graph_idx** — the graph's index within the batch.
    * **status** — the sample's status code (from ``batch.status``),
      indicating which pipeline stage it belongs to.  Always ``0`` for
      single-stage dynamics.
    * **energy** — per-graph potential energy (from ``batch.energy``).
    * **fmax** — per-graph maximum atomic force norm.
    * **temperature** — per-graph instantaneous kinetic temperature
      (from ``batch.velocities`` and ``batch.atomic_masses`` via the
      equipartition theorem), if velocities are present.

    Users can extend or replace this set by providing a ``custom_scalars``
    mapping of ``{name: callable}`` pairs, where each callable has
    signature ``(batch, dynamics) -> Tensor`` of shape ``(B,)`` (one
    value per graph) or a plain ``float`` (broadcast to all graphs).

    **Asynchronous I/O.** Scalar computation and the GPU-to-CPU transfer
    run on a dedicated CUDA side stream (set up in :meth:`__enter__`) so
    they do not stall the default compute stream.  The D2H copy uses
    ``non_blocking=True``; the single-worker
    :class:`ThreadPoolExecutor` synchronizes the stream before reading
    the CPU tensor and writing to the backend.

    **Context manager.** Use ``with`` to guarantee the CUDA stream is
    created, all pending writes are flushed, and file handles are
    closed on exit::

        with LoggingHook(backend="csv", log_path="out.csv") as hook:
            dynamics.register_hook(hook)
            dynamics.run(batch)

    This hook is non-blocking: scalar computation and the D2H transfer
    run on a dedicated CUDA side stream (when available), and all
    ``.item()`` calls and I/O happen in a background
    :class:`~concurrent.futures.ThreadPoolExecutor` worker, so the
    GPU pipeline is not stalled.

    Parameters
    ----------
    backend : {"csv", "tensorboard", "custom"}
        Logging backend to use.
    frequency : int, optional
        Log every ``frequency`` steps. Default ``1``.
    log_path : str | Path | None, optional
        File path for file-based backends (``"csv"``,
        ``"tensorboard"``).  Default ``None``.
    custom_scalars : dict[str, Callable] | None, optional
        Additional named scalars to compute and log.  Each callable
        receives ``(ctx,)`` — a :class:`HookContext` — and returns
        either a ``(B,)`` tensor (per-graph values) or a ``float``
        (broadcast to all graphs).  Name collisions override defaults.
        Default ``None``.
    writer_fn : Callable[[int, list[dict[str, float]]], None] | None, optional
        Custom writer function, required when ``backend="custom"``.
        Receives ``(step_count, rows)``.  Default ``None``.

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import LoggingHook
    >>> with LoggingHook(backend="csv", log_path="md_log.csv", frequency=100) as hook:
    ...     dynamics = DemoDynamics(model=model, n_steps=10_000, dt=0.5, hooks=[hook])
    ...     dynamics.run(batch)

    Using custom scalars:

    >>> def pressure(ctx):
    ...     return compute_pressure(ctx.batch.stress, ctx.batch.cell)
    >>> hook = LoggingHook(
    ...     backend="csv",
    ...     log_path="md_log.csv",
    ...     frequency=50,
    ...     custom_scalars={"pressure": pressure},
    ... )

    Notes
    -----
    * The default temperature calculation assumes an NVT-like system
      with ``3N`` degrees of freedom (no constraint correction).
      Override via ``custom_scalars`` if constraints remove DOFs.
    * For distributed pipelines, each rank logs independently. Use
      ``log_path`` with rank-specific filenames to avoid file
      contention.
    """

    def __init__(
        self,
        backend: LogBackend,
        frequency: int = 1,
        log_path: str | Path | None = None,
        custom_scalars: (
            dict[str, Callable[[HookContext], float | torch.Tensor]] | None
        ) = None,
        writer_fn: (Callable[[int, list[dict[str, float]]], None] | None) = None,
        stage: Enum = DynamicsStage.AFTER_STEP,
    ) -> None:
        self.frequency = frequency
        self.stage = stage

        self._csv_file: io.TextIOWrapper | None = None
        self._csv_writer: csv.DictWriter | None = None
        self._tb_writer = None
        self._stream: torch.cuda.Stream | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)

        if backend == "csv" and log_path is None:
            raise ValueError("csv backend requires log_path")
        if backend == "tensorboard" and log_path is None:
            raise ValueError("tensorboard backend requires log_path")
        if backend == "custom" and writer_fn is None:
            raise ValueError("custom backend requires writer_fn")
        if backend not in get_args(LogBackend):
            raise ValueError(
                f"LoggingHook only supports backends: "
                f"{get_args(LogBackend)}, got {backend!r}."
            )

        self.backend = backend
        self.log_path = log_path
        self.custom_scalars = custom_scalars
        self.writer_fn = writer_fn

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> LoggingHook:
        if torch.cuda.is_available():
            self._stream = torch.cuda.Stream()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # noqa: ANN001
        self.close()
        return None

    def close(self) -> None:
        """Flush pending writes and release stream / file resources."""
        self._executor.shutdown(wait=True)
        self._executor = ThreadPoolExecutor(max_workers=1)
        if self._stream is not None:
            self._stream.synchronize()
            self._stream = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
            self._csv_writer = None
        if self._tb_writer is not None:
            self._tb_writer.close()
            self._tb_writer = None

    def __del__(self) -> None:
        self._executor.shutdown(wait=True)

    # ------------------------------------------------------------------
    # Hook entry point
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def _log(
        self,
        batch: Batch,
        step_count: int,
        ctx: HookContext,
    ) -> None:
        """Compute per-graph scalars and dispatch to the logging backend.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data.
        step_count : int
            The current step number.
        ctx : HookContext
            The hook context (required for custom_scalars).
        """
        device = batch.device
        use_stream = self._stream is not None and device.type == "cuda"

        td = self._compute_columns(batch, step_count, ctx)
        td = _snapshot_tensordict(td)

        if use_stream:
            main_stream = torch.cuda.current_stream(device)
            stream_ctx = torch.cuda.stream(self._stream)
        else:
            stream_ctx = contextlib.nullcontext()

        with stream_ctx:
            if use_stream:
                self._stream.wait_stream(main_stream)
            td = td.to("cpu", non_blocking=True)

        self._executor.submit(self._dispatch, td, step_count)

    @torch.compiler.disable
    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        """Log scalar observables for the current step."""
        self._log(ctx.batch, ctx.step_count, ctx)

    # ------------------------------------------------------------------
    # Column computation (on-device)
    # ------------------------------------------------------------------

    def _compute_columns(
        self,
        batch: Batch,
        step_count: int,
        ctx: HookContext,
    ) -> TensorDict:
        """Build a ``TensorDict(batch_size=[B])`` of per-graph scalars.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data.
        step_count : int
            The current step number.
        ctx : HookContext
            The hook context (required for custom_scalars).
        """
        dev = batch.device
        num_graphs = batch.num_graphs

        td = TensorDict(
            step=torch.full((num_graphs,), step_count, device=dev, dtype=torch.int64),
            graph_idx=torch.arange(num_graphs, device=dev, dtype=torch.int64),
            batch_size=[num_graphs],
        )

        # Status (stage code) — 0 if not present
        if hasattr(batch, "status") and batch.status is not None:
            status = (
                batch.status.squeeze(-1) if batch.status.dim() == 2 else batch.status
            )
            td.set("status", status.float())
        else:
            td.set("status", torch.zeros(num_graphs, device=dev))

        if batch.energy is not None:
            td.set("energy", batch.energy.squeeze(-1))

        if batch.forces is not None:
            norms = torch.linalg.vector_norm(batch.forces, dim=-1)
            td.set(
                "fmax",
                scatter_reduce_per_graph(
                    norms, batch.batch_idx, num_graphs, reduce="amax"
                ),
            )

        if getattr(batch, "velocities", None) is not None:
            td.set(
                "temperature",
                temperature_per_graph(
                    batch.velocities,
                    batch.atomic_masses,
                    batch.batch_idx,
                    num_graphs,
                    batch.num_nodes_per_graph,
                ),
            )

        if self.custom_scalars:
            for name, fn in self.custom_scalars.items():
                # custom_scalars receive (ctx,) — a HookContext
                val = fn(ctx)
                if isinstance(val, torch.Tensor):
                    td.set(name, val)
                else:
                    td.set(name, torch.full((num_graphs,), val, device=dev))

        return td

    # ------------------------------------------------------------------
    # Dispatch (runs on ThreadPoolExecutor worker)
    # ------------------------------------------------------------------

    def _dispatch(self, td: TensorDict, step: int) -> None:
        """Synchronize stream (if needed), convert to rows, write."""
        if self._stream is not None:
            self._stream.synchronize()
        rows = _tensordict_to_rows(td)
        match self.backend:
            case "csv":
                self._write_csv(rows)
            case "tensorboard":
                self._write_tensorboard(rows, step)
            case "custom":
                self.writer_fn(step, rows)  # type: ignore[misc]

    def _write_csv(self, rows: list[dict[str, float]]) -> None:
        """Convert row data and write out CSV"""
        if self._csv_writer is None:
            self._csv_file = open(self.log_path, "w", newline="")  # noqa: SIM115
            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=list(rows[0].keys()),
            )
            self._csv_writer.writeheader()
        self._csv_writer.writerows(rows)
        self._csv_file.flush()  # type: ignore[union-attr]

    def _write_tensorboard(
        self,
        rows: list[dict[str, float]],
        step: int,
    ) -> None:
        if self._tb_writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self._tb_writer = SummaryWriter(log_dir=str(self.log_path))
        for row in rows:
            graph_idx = int(row.get("graph_idx", 0))
            for key, value in row.items():
                if key in ("step", "graph_idx"):
                    continue
                tag = f"{key}/graph_{graph_idx}" if len(rows) > 1 else key
                self._tb_writer.add_scalar(tag, value, step)


def _snapshot_tensordict(td: TensorDict) -> TensorDict:
    """Clone logged columns so async dispatch never aliases live batch tensors."""
    return TensorDict(
        {key: td[key].detach().clone() for key in td.keys()},
        batch_size=td.batch_size,
    )


def _tensordict_to_rows(td: TensorDict) -> list[dict[str, float]]:
    """Convert a ``TensorDict(batch_size=[B])`` into a list of dicts."""
    cols = {k: td[k].tolist() for k in td.keys()}
    num_graphs = td.batch_size[0]
    keys = list(cols.keys())
    return [{k: float(cols[k][i]) for k in keys} for i in range(num_graphs)]
