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

from typing import TYPE_CHECKING, Literal

from nvalchemi.dynamics.hooks._base import _ObserverHook

if TYPE_CHECKING:
    from collections.abc import Callable

    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import BaseDynamics

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

        Raises
        ------
        NotImplementedError
            This hook is not yet implemented.
        """
        raise NotImplementedError("LoggingHook is not yet implemented.")
