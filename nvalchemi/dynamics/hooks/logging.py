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
Logging hook for recording scalar simulation observables.

Provides :class:`LoggingHook`, which computes and logs summary
statistics (energies, temperatures, max forces, etc.) at a
configurable frequency.
"""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, Literal

import torch

from nvalchemi.dynamics.hooks._base import _ObserverHook

if TYPE_CHECKING:
    from collections.abc import Callable

    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import BaseDynamics

# Boltzmann constant in eV/K — consistent with typical atomistic MD unit
# systems (positions in Å, masses in amu, velocities in Å/fs, energy in eV).
_KB_EV_PER_K: float = 8.617333262e-5

__all__ = ["LoggingHook"]

LogBackend = Literal["loguru", "csv", "tensorboard", "custom"]


class LoggingHook(_ObserverHook):
    """Log scalar observables from the simulation at a specified frequency.

    At each firing step, this hook extracts per-batch scalar summaries
    from the :class:`~nvalchemi.data.Batch` and writes them to the
    configured logging backend.  It is designed for lightweight,
    human-readable monitoring — for full-state recording, use
    :class:`SnapshotHook` instead.

    The default observable set includes:

    * **step** — the current ``dynamics.step_count``.
    * **total_energy** — mean total potential energy across the batch
      (from ``batch.energies``).
    * **fmax** — maximum force component magnitude across all atoms
      (from ``batch.forces``).
    * **temperature** — instantaneous kinetic temperature
      (computed from ``batch.velocities`` and ``batch.atomic_masses``
      via the equipartition theorem), if velocities are present.

    Users can extend or replace this set by providing a ``custom_scalars``
    mapping of ``{name: callable}`` pairs, where each callable has
    signature ``(batch, dynamics) -> float``.

    Supported logging backends:

    * ``"loguru"`` (default) — structured log messages via the
      :mod:`loguru` logger at ``INFO`` level.
    * ``"csv"`` — append rows to a CSV file specified by ``log_path``.
    * ``"tensorboard"`` — write scalars via a
      ``torch.utils.tensorboard.SummaryWriter``.
    * ``"custom"`` — call a user-provided ``writer_fn(step, scalars)``
      where ``scalars`` is a ``dict[str, float]``.

    Parameters
    ----------
    frequency : int, optional
        Log every ``frequency`` steps. Default ``1``.
    backend : {"loguru", "csv", "tensorboard", "custom"}, optional
        Logging backend to use. Default ``"loguru"``.
    log_path : str | None, optional
        File path for file-based backends (``"csv"``,
        ``"tensorboard"``). Ignored for ``"loguru"`` and ``"custom"``.
        Default ``None``.
    custom_scalars : dict[str, Callable[[Batch, BaseDynamics], float]] | None, optional
        Additional named scalars to compute and log. Each callable
        receives the current batch and dynamics engine and returns a
        float. These are merged with the default scalar set; name
        collisions override the default.  Default ``None``.
    writer_fn : Callable[[int, dict[str, float]], None] | None, optional
        Custom writer function, required when ``backend="custom"``.
        Receives ``(step_count, scalars_dict)``. Default ``None``.

    Attributes
    ----------
    backend : LogBackend
        The active logging backend.
    log_path : str | None
        File path for file-based backends.
    custom_scalars : dict[str, Callable] | None
        User-defined scalar extractors.
    writer_fn : Callable | None
        Custom writer function (``"custom"`` backend only).
    frequency : int
        Logging frequency in steps.
    stage : HookStageEnum
        Fixed to ``AFTER_STEP``.

    Examples
    --------
    >>> from nvalchemi.dynamics.hooks import LoggingHook
    >>> hook = LoggingHook(frequency=100, backend="csv", log_path="md_log.csv")
    >>> dynamics = DemoDynamics(model=model, n_steps=10_000, dt=0.5, hooks=[hook])
    >>> dynamics.run(batch)

    Using custom scalars:

    >>> def pressure(batch, dynamics):
    ...     return compute_pressure(batch.stresses, batch.cell)
    >>> hook = LoggingHook(
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
        frequency: int = 1,
        backend: LogBackend = "loguru",
        log_path: str | None = None,
        custom_scalars: (
            dict[str, Callable[[Batch, BaseDynamics], float]] | None
        ) = None,
        writer_fn: Callable[[int, dict[str, float]], None] | None = None,
    ) -> None:
        super().__init__(frequency=frequency)
        self.backend = backend
        self.log_path = log_path
        self.custom_scalars = custom_scalars
        self.writer_fn = writer_fn

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Compute scalar observables and write them to the logging backend.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data.
        dynamics : BaseDynamics
            The dynamics engine instance.
        """
        step = dynamics.step_count
        scalars: dict[str, float] = {"step": float(step)}

        # Mean total potential energy across the batch
        if getattr(batch, "energies", None) is not None:
            scalars["total_energy"] = batch.energies.mean().item()

        # Max per-atom force L2-norm across the entire batch
        if getattr(batch, "forces", None) is not None:
            fmax = torch.linalg.vector_norm(batch.forces, dim=-1).max()
            scalars["fmax"] = fmax.item()

        # Instantaneous kinetic temperature via the equipartition theorem.
        # Assumes velocities in Å/fs, masses in amu, and energies in eV,
        # which is the standard unit system for atomistic MD. Override via
        # custom_scalars if a different unit convention is used.
        velocities = getattr(batch, "velocities", None)
        masses = getattr(batch, "atomic_masses", None)
        if velocities is not None and masses is not None:
            m = masses if masses.dim() == 2 else masses.unsqueeze(-1)  # (N, 1)
            ke = 0.5 * (m * velocities.pow(2)).sum()
            n_dof = 3 * velocities.shape[0]
            scalars["temperature"] = (2.0 * ke.item()) / (n_dof * _KB_EV_PER_K)

        # Merge custom scalars (may override defaults by name)
        if self.custom_scalars is not None:
            for name, fn in self.custom_scalars.items():
                scalars[name] = fn(batch, dynamics)

        self._write(step, scalars)

    def _write(self, step: int, scalars: dict[str, float]) -> None:
        """Dispatch scalar dict to the configured logging backend.

        Parameters
        ----------
        step : int
            Current simulation step count.
        scalars : dict[str, float]
            Computed scalar observables to record.
        """
        match self.backend:
            case "loguru":
                from loguru import logger

                msg = "  ".join(f"{k}={v:.6g}" for k, v in scalars.items())
                logger.info(f"[dynamics] {msg}")

            case "csv":
                if self.log_path is None:
                    raise RuntimeError(
                        "LoggingHook with backend='csv' requires log_path to be set."
                    )
                write_header = not os.path.exists(self.log_path)
                with open(self.log_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(scalars.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(scalars)

            case "tensorboard":
                from torch.utils.tensorboard import SummaryWriter

                if not hasattr(self, "_tb_writer"):
                    self._tb_writer: SummaryWriter = SummaryWriter(
                        log_dir=self.log_path
                    )
                for k, v in scalars.items():
                    if k != "step":
                        self._tb_writer.add_scalar(k, v, global_step=step)

            case "custom":
                if self.writer_fn is None:
                    raise RuntimeError(
                        "LoggingHook with backend='custom' requires writer_fn to be set."
                    )
                self.writer_fn(step, scalars)
