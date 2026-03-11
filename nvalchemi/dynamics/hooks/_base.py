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
Internal base classes for common hook categories.

These classes reduce boilerplate by pre-wiring the ``stage`` attribute and
providing a common ``__init__`` signature.  They are **not** part of the
public API — users should import the concrete hook classes from the
``nvalchemi.dynamics.hooks`` namespace instead.

Two categories are defined:

``_ObserverHook``
    Read-only hooks that record or log simulation state without modifying
    it.  Default stage: ``AFTER_STEP``.

``_PostComputeHook``
    Hooks that modify the batch **after** the model forward pass
    (e.g. clamping forces, adding bias potentials).  Default stage:
    ``AFTER_COMPUTE``.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

import torch

from nvalchemi.dynamics.base import HookStageEnum

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import BaseDynamics


class _ObserverHook:
    """Base class for hooks that observe simulation state without modifying it.

    Observer hooks fire at :attr:`~HookStageEnum.AFTER_STEP` by default,
    after all integrator updates and other hooks have completed.  Subclasses
    should override ``__call__`` to implement the observation logic (e.g.
    writing snapshots, computing summary statistics).

    Parameters
    ----------
    frequency : int
        Execute the hook every ``frequency`` steps.

    Attributes
    ----------
    frequency : int
        Execution frequency in steps.
    stage : HookStageEnum
        Fixed to ``AFTER_STEP``.
    """

    stage: HookStageEnum = HookStageEnum.AFTER_STEP

    def __init__(self, frequency: int = 1) -> None:
        self.frequency = frequency

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Execute the observer hook.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data (should **not** be modified).
        dynamics : BaseDynamics
            The dynamics engine instance.
        """
        raise NotImplementedError


class DeferredObserverHook(_ObserverHook):
    """Observer hook that runs :meth:`observe` in a background thread.

    Hooks that log or analyze simulation data often call ``.item()`` or
    ``.cpu()``, which force a GPU–CPU synchronization in the main simulation
    loop and degrade throughput.  ``DeferredObserverHook`` removes this
    penalty by splitting the hook into two methods:

    1. :meth:`extract` — runs in the **main thread** immediately after the
       step.  Should copy the tensors you need to CPU using
       ``non_blocking=True`` and return them in a ``dict``.  Must not call
       ``.item()`` or any other synchronizing operation.

    2. :meth:`observe` — runs in a **background thread** after all
       non-blocking copies from :meth:`extract` have completed.  Safe to
       call ``.item()``, ``print()``, write to files, etc.

    The framework records a CUDA event after :meth:`extract` and
    synchronizes it inside the background thread before calling
    :meth:`observe`, guaranteeing that every ``non_blocking=True`` copy is
    complete before the data is read — without blocking the main thread.

    A single background worker thread is used per hook instance so
    observations are always processed in step order.  The thread pool is
    started lazily on the first firing.

    :meth:`flush` is called automatically when
    :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` or
    :meth:`~nvalchemi.dynamics.base.FusedStage.run` returns, ensuring all
    pending observations are written before the method exits.

    Parameters
    ----------
    frequency : int, optional
        Execute the hook every ``frequency`` steps. Default ``1``.
    max_pending : int, optional
        Maximum number of submitted-but-unprocessed observations before
        the main thread waits for the oldest one to finish.  Prevents
        unbounded queue growth when logging is slower than the simulation.
        Default ``16``.

    Attributes
    ----------
    frequency : int
        Execution frequency in steps.
    stage : HookStageEnum
        Fixed to ``AFTER_STEP``.
    max_pending : int
        Maximum pending observations before backpressure kicks in.

    Examples
    --------
    Write per-step mean energy to a file without stalling the GPU:

    >>> class EnergyLogger(DeferredObserverHook):
    ...     def __init__(self, path):
    ...         super().__init__(frequency=10)
    ...         self._file = open(path, "w")
    ...
    ...     def extract(self, batch, dynamics):
    ...         return {
    ...             "step": dynamics.step_count,
    ...             "energies": batch.energies.to("cpu", non_blocking=True),
    ...         }
    ...
    ...     def observe(self, step, data):
    ...         mean_e = data["energies"].mean().item()  # safe: DMA complete
    ...         self._file.write(f"{data['step']},{mean_e:.6f}\\n")
    """

    def __init__(self, frequency: int = 1, max_pending: int = 16) -> None:
        super().__init__(frequency=frequency)
        self.max_pending = max_pending
        self._executor: ThreadPoolExecutor | None = None
        self._futures: list[Future] = []

    # ------------------------------------------------------------------
    # Subclass interface
    # ------------------------------------------------------------------

    def extract(self, batch: "Batch", dynamics: "BaseDynamics") -> dict[str, Any]:
        """Copy tensors from the batch to CPU (non-blocking).

        Called in the **main simulation thread** on every hook firing.
        Should return a ``dict`` of CPU-destined tensors and any
        plain-Python metadata needed by :meth:`observe`.  Must not call
        ``.item()`` or any GPU–CPU synchronizing operation — those belong
        in :meth:`observe`.

        Parameters
        ----------
        batch : Batch
            The current batch (GPU tensors).
        dynamics : BaseDynamics
            The dynamics engine.

        Returns
        -------
        dict[str, Any]
            Data to pass to :meth:`observe`.  Use
            ``tensor.to("cpu", non_blocking=True)`` for tensors.
        """
        raise NotImplementedError

    def observe(self, step: int, data: dict[str, Any]) -> None:
        """Process extracted data in a background thread.

        By the time this method is called, all non-blocking copies
        scheduled in :meth:`extract` are guaranteed to be complete.

        Parameters
        ----------
        step : int
            The step count at which :meth:`extract` was called.
        data : dict[str, Any]
            The ``dict`` returned by :meth:`extract`.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Internal machinery
    # ------------------------------------------------------------------

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None or self._executor._shutdown:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="nvalchemi_hook"
            )
        return self._executor

    def __call__(self, batch: "Batch", dynamics: "BaseDynamics") -> None:
        step = dynamics.step_count
        data = self.extract(batch, dynamics)

        # Record a CUDA event so the background thread can synchronize it
        # before reading any tensors copied with non_blocking=True.
        event: torch.cuda.Event | None = None
        if torch.cuda.is_available():
            event = torch.cuda.Event()
            event.record()

        # Drain completed futures, then apply backpressure if the queue is full.
        self._futures = [f for f in self._futures if not f.done()]
        if len(self._futures) >= self.max_pending:
            self._futures.pop(0).result()

        future = self._get_executor().submit(self._run_observe, step, data, event)
        self._futures.append(future)

    def _run_observe(
        self,
        step: int,
        data: dict[str, Any],
        event: "torch.cuda.Event | None",
    ) -> None:
        if event is not None:
            event.synchronize()
        self.observe(step, data)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Wait for all pending :meth:`observe` calls to complete.

        Called automatically at the end of
        :meth:`~nvalchemi.dynamics.base.BaseDynamics.run` and
        :meth:`~nvalchemi.dynamics.base.FusedStage.run`.  Call manually
        if you need all log entries written before inspecting output files
        mid-simulation.
        """
        for f in self._futures:
            f.result()
        self._futures.clear()

    def close(self) -> None:
        """Flush pending work and shut down the background thread pool."""
        self.flush()
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: S110
            pass


class _PostComputeHook:
    """Base class for hooks that modify batch state after the model forward pass.

    Post-compute hooks fire at :attr:`~HookStageEnum.AFTER_COMPUTE` by
    default, immediately after :meth:`~BaseDynamics.compute` writes forces
    and energies to the batch.  Subclasses should override ``__call__`` to
    implement the modification logic (e.g. clamping forces, adding bias
    potentials, detecting NaNs).

    Parameters
    ----------
    frequency : int
        Execute the hook every ``frequency`` steps.

    Attributes
    ----------
    frequency : int
        Execution frequency in steps.
    stage : HookStageEnum
        Fixed to ``AFTER_COMPUTE``.
    """

    stage: HookStageEnum = HookStageEnum.AFTER_COMPUTE

    def __init__(self, frequency: int = 1) -> None:
        self.frequency = frequency

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Execute the post-compute hook.

        Parameters
        ----------
        batch : Batch
            The current batch of atomic data (modified in-place).
        dynamics : BaseDynamics
            The dynamics engine instance.
        """
        raise NotImplementedError
