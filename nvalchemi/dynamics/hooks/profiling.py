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
Performance profiling and step timing hook.

Provides :class:`ProfilerHook`, which instruments dynamics steps with
NVTX ranges and wall-clock timing for performance analysis with
NVIDIA Nsight Systems and PyTorch profiler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from nvalchemi.dynamics.base import HookStageEnum

if TYPE_CHECKING:
    from nvalchemi.data import Batch
    from nvalchemi.dynamics.base import BaseDynamics

__all__ = ["ProfilerHook"]


class ProfilerHook:
    """Instrument dynamics steps with NVTX ranges and wall-clock timing.

    This hook provides two complementary profiling capabilities:

    **NVTX Annotation** (``enable_nvtx=True``)
        Wraps each dynamics step in an `NVTX range
        <https://docs.nvidia.com/nvtx/>`_ so that steps are visible
        as named regions in NVIDIA Nsight Systems timelines.  The
        range is pushed at :attr:`~HookStageEnum.BEFORE_STEP` and
        popped at :attr:`~HookStageEnum.AFTER_STEP`, covering the
        entire step including all sub-hooks.

        The range label includes the step number for easy correlation::

            "dynamics_step/42"

        NVTX ranges have negligible overhead when Nsight is not
        attached, making it safe to leave enabled in production.

    **Wall-Clock Timing** (``enable_timer=True``)
        Measures the wall-clock duration of each step using
        ``torch.cuda.Event`` for GPU-accurate timing (on CUDA
        devices) or ``time.perf_counter_ns`` as a fallback (on CPU).

        Timing results are accumulated in an internal buffer and
        can be summarized on demand.  The hook tracks:

        * Per-step wall time (seconds).
        * Rolling mean and standard deviation.
        * Throughput in steps/second and atoms*steps/second.

        Timing data is logged via :mod:`loguru` at ``DEBUG`` level
        every ``frequency`` steps, and a final summary is available
        via the ``summary()`` method (not yet implemented).

    Because profiling requires hooks at **two** stages
    (``BEFORE_STEP`` to start, ``AFTER_STEP`` to stop), this hook
    registers itself at ``BEFORE_STEP`` and internally manages the
    corresponding end-of-step logic.  The dynamics engine calls it at
    ``BEFORE_STEP``; the hook records the start event and installs a
    one-shot ``AFTER_STEP`` callback to record the end event.

    .. note::

        An alternative implementation strategy is to register **two**
        separate hook instances (one per stage) that share state via a
        common object.  The single-instance approach was chosen to keep
        the user-facing API simple (one object to construct and
        register).

    Parameters
    ----------
    frequency : int, optional
        Profile every ``frequency`` steps. Default ``1``.
    enable_nvtx : bool, optional
        Whether to push/pop NVTX ranges. Default ``True``.
    enable_timer : bool, optional
        Whether to record wall-clock step times. Default ``True``.
    timer_backend : {"cuda_event", "perf_counter"}, optional
        Timing backend. ``"cuda_event"`` uses
        ``torch.cuda.Event(enable_timing=True)`` for sub-millisecond
        GPU-synchronized accuracy; ``"perf_counter"`` uses
        ``time.perf_counter_ns`` (CPU-only, includes host overhead).
        Default ``"cuda_event"``.

    Attributes
    ----------
    enable_nvtx : bool
        Whether NVTX annotation is active.
    enable_timer : bool
        Whether step timing is active.
    timer_backend : str
        The active timing backend.
    frequency : int
        Profiling frequency in steps.
    stage : HookStageEnum
        Fixed to ``BEFORE_STEP`` (the hook manages the ``AFTER_STEP``
        counterpart internally).

    Examples
    --------
    Profile with Nsight Systems:

    >>> from nvalchemi.dynamics.hooks import ProfilerHook
    >>> hook = ProfilerHook(enable_nvtx=True, enable_timer=False)
    >>> dynamics = DemoDynamics(model=model, n_steps=100, dt=0.5, hooks=[hook])
    >>> # Run under: nsys profile python my_script.py
    >>> dynamics.run(batch)

    Benchmark throughput:

    >>> hook = ProfilerHook(enable_nvtx=False, enable_timer=True, frequency=10)
    >>> dynamics = DemoDynamics(model=model, n_steps=1000, dt=0.5, hooks=[hook])
    >>> dynamics.run(batch)

    Notes
    -----
    * NVTX ranges nest correctly with ranges from other libraries
      (e.g. ``nvtx.annotate`` in the model forward pass), providing
      a hierarchical view in Nsight.
    * ``cuda_event`` timing introduces a device synchronization
      point, which can perturb the performance characteristics of
      fully asynchronous pipelines.  Use ``frequency > 1`` to
      amortize this cost, or switch to ``"perf_counter"`` for a
      non-synchronizing (but less accurate) alternative.
    * For distributed pipelines, each rank profiles independently.
    """

    stage: HookStageEnum = HookStageEnum.BEFORE_STEP

    def __init__(
        self,
        frequency: int = 1,
        enable_nvtx: bool = True,
        enable_timer: bool = True,
        timer_backend: Literal["cuda_event", "perf_counter"] = "cuda_event",
    ) -> None:
        self.frequency = frequency
        self.enable_nvtx = enable_nvtx
        self.enable_timer = enable_timer
        self.timer_backend = timer_backend

    def __call__(self, batch: Batch, dynamics: BaseDynamics) -> None:
        """Begin profiling for the current step.

        Pushes an NVTX range and/or records the start timing event.
        The corresponding end operations are handled internally at
        ``AFTER_STEP``.

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
        raise NotImplementedError("ProfilerHook is not yet implemented.")
