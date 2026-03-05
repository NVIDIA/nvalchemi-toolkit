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

from typing import TYPE_CHECKING

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
