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
"""Base protocols and models for loss-function schedules.

This module also re-exports :class:`BaseLossFunction` and
:class:`ComposedLossFunction` from :mod:`.composition` for discoverability:
subclass authors can do ``from nvalchemi.training.losses.base import
BaseLossFunction`` without tracking the internal module layout. The
canonical home of the leaf base class, keyed composition aggregator, and
composition output type remains :mod:`.composition`.
"""

from __future__ import annotations

from typing import Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, Field


@runtime_checkable
class LossWeightSchedule(Protocol):
    """Runtime-checkable protocol for loss-weight schedules.

    Any object callable with signature ``(step: int, epoch: int) -> float``
    and exposing a ``per_epoch`` attribute satisfies this protocol, and is
    therefore accepted inside
    :class:`~nvalchemi.training.losses.ComposedLossFunction`'s ``weights``
    sequence or as the right-hand side of ``schedule * leaf``. Concrete
    Pydantic schedules live in
    :mod:`~nvalchemi.training.losses.schedules`.

    Attributes
    ----------
    per_epoch
        If ``True``, the schedule should advance by ``epoch`` instead of
        by ``step``. This aligns loss-weight updates with training loops
        that update learning-rate schedules once per epoch.

    Parameters
    ----------
    step
        Current global training step (0-indexed).
    epoch
        Current epoch number (0-indexed).

    Returns
    -------
    float
        Scalar weight to apply to the associated loss term.
    """

    per_epoch: Annotated[
        bool,
        "Whether the schedule steps per epoch; if False, schedule will update per step/batch.",
    ]

    def __call__(self, step: int, epoch: int) -> float:
        """Evaluate the schedule at ``(step, epoch)``."""
        ...


class _BaseWeightSchedule(BaseModel):
    """Base Pydantic model for serializable loss-weight schedules.

    Attributes
    ----------
    per_epoch
        If ``False``, schedule windows advance by global step. If
        ``True``, they advance by epoch.
    """

    model_config = {"frozen": True}

    per_epoch: Annotated[
        bool,
        Field(
            default=False,
            description=(
                "Whether to advance this schedule by epoch instead of by global step."
            ),
        ),
    ] = False

    def _map_schedule_index(self, step: int, epoch: int) -> int:
        """Return the counter used to advance this schedule.

        This method is only intended to be used if your schedule is mutually
        exclusive; if your schedule uses both step *and* epoch values, then
        you do not need to use this function as it's only for routing.
        """
        return epoch if self.per_epoch else step


# Re-exports for discoverability. Import at the bottom to avoid a circular
# import: ``composition`` imports ``_BaseWeightSchedule`` indirectly through
# ``schedules``, which imports this module.
from nvalchemi.training.losses.composition import (  # noqa: E402
    BaseLossFunction,
    ComposedLossFunction,
    ComposedLossOutput,
)

__all__ = [
    "BaseLossFunction",
    "ComposedLossFunction",
    "ComposedLossOutput",
    "LossWeightSchedule",
]
